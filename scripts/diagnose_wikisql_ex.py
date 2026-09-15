#!/usr/bin/env python3
"""
diagnose_wikisql_ex.py
========================
Root-cause diagnosis for the gap between WikiSQL's structural EM (fuzzy,
non-executing {agg,sel,conds} comparison) and raw execution accuracy
(actually running SQL against the real SQLite database).

Structural EM can say "correct" for a question where the predicted SQL
still fails or returns a different result when actually executed — e.g.
unquoted/misquoted column names with spaces or punctuation, a WHERE value
that doesn't literally match the stored text, or a genuinely different
result set despite matching agg/sel/cond fields. This script executes
both sides for real and classifies exactly what's going wrong.

Usage:
    python scripts/diagnose_wikisql_ex.py \
        --gold  data/raw/wikisql/dev_wiki_format.json \
        --db    data/raw/wikisql/database \
        --predict results/predictions_wikisql_full.tsv \
        --limit 147
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

WIKISQL_TABLE_NAME = "wikisql_data"


def _run_sql(db_path: str, sql: str) -> Tuple[bool, Optional[list], Optional[str]]:
    """Execute sql against db_path. Returns (ok, rows, error_message)."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return True, rows, None
    except Exception as e:
        return False, None, str(e)


def _results_match(rows_a: list, rows_b: list) -> bool:
    """Order-insensitive, type-loose result set comparison."""
    def norm_row(r):
        return tuple(
            round(v, 6) if isinstance(v, float) else
            (str(v).strip().lower() if isinstance(v, str) else v)
            for v in r
        )
    a = sorted(norm_row(r) for r in rows_a)
    b = sorted(norm_row(r) for r in rows_b)
    return a == b


def _error_cluster_key(error_msg: str) -> str:
    msg = error_msg.lower()
    for pattern, key in [
        (r"no such column", "no such column"),
        (r"no such table", "no such table"),
        (r"syntax error", "syntax error"),
        (r"unrecognized token", "unrecognized token"),
        (r"near \"", "syntax error near token"),
        (r"incomplete input", "incomplete input"),
        (r"misuse of aggregate", "misuse of aggregate function"),
        (r"datatype mismatch", "datatype mismatch"),
    ]:
        if re.search(pattern, msg):
            return key
    return "other: " + msg[:60]


def diagnose(
    gold_path: str,
    db_dir: str,
    predict_path: str,
    limit: Optional[int] = None,
    samples_per_category: int = 4,
) -> None:
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    if limit:
        gold_data = gold_data[:limit]

    with open(predict_path, "r", encoding="utf-8") as f:
        pred_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if limit:
        pred_lines = pred_lines[:limit]

    n = min(len(gold_data), len(pred_lines))
    if len(gold_data) != len(pred_lines):
        print(f"⚠ gold has {len(gold_data)} rows, predict has {len(pred_lines)} rows — "
              f"comparing only the first {n}.")

    total = 0
    both_ok_match = 0
    both_ok_mismatch = 0
    pred_error = 0
    gold_error = 0

    pred_error_clusters = Counter()
    examples = {"pred_error": [], "wrong_result": [], "gold_error": []}

    for i in range(n):
        item = gold_data[i]
        question = item.get("question", "")
        db_id = item.get("db_id", "")
        gold_sql = item.get("query", item.get("sql", ""))
        pred_sql = pred_lines[i].split("\t")[0] if "\t" in pred_lines[i] else pred_lines[i]

        if not db_id or not gold_sql:
            continue
        db_path = str(Path(db_dir) / db_id / f"{db_id}.sqlite")
        if not Path(db_path).exists():
            continue

        total += 1

        gold_ok, gold_rows, gold_err = _run_sql(db_path, gold_sql)
        pred_ok, pred_rows, pred_err = _run_sql(db_path, pred_sql)

        record = {
            "db_id": db_id, "question": question,
            "gold_sql": gold_sql, "pred_sql": pred_sql,
        }

        if not gold_ok:
            gold_error += 1
            record["error"] = gold_err
            if len(examples["gold_error"]) < samples_per_category:
                examples["gold_error"].append(record)
            continue

        if not pred_ok:
            pred_error += 1
            cluster = _error_cluster_key(pred_err)
            pred_error_clusters[cluster] += 1
            record["error"] = pred_err
            record["cluster"] = cluster
            if len(examples["pred_error"]) < samples_per_category:
                examples["pred_error"].append(record)
            continue

        if _results_match(gold_rows, pred_rows):
            both_ok_match += 1
        else:
            both_ok_mismatch += 1
            record["gold_rows_sample"] = gold_rows[:3]
            record["pred_rows_sample"] = pred_rows[:3]
            if len(examples["wrong_result"]) < samples_per_category:
                examples["wrong_result"].append(record)

    print("=" * 70)
    print("WIKISQL EXECUTION DIAGNOSIS")
    print("=" * 70)
    print(f"Total evaluated       : {total}")
    print(f"Both executed, MATCH  : {both_ok_match} ({both_ok_match/total:.1%})  <- true EX pass")
    print(f"Both executed, WRONG  : {both_ok_mismatch} ({both_ok_mismatch/total:.1%})")
    print(f"Predicted SQL errored : {pred_error} ({pred_error/total:.1%})")
    print(f"Gold SQL errored      : {gold_error} ({gold_error/total:.1%})  <- gold conversion bug, not model's fault")

    if pred_error_clusters:
        print("\n--- Predicted-SQL error clusters ---")
        for cluster, count in pred_error_clusters.most_common(10):
            print(f"  {count:>4}x  {cluster}")

    for category, label in [
        ("pred_error", "Sample: predicted SQL throws an execution error"),
        ("wrong_result", "Sample: both execute, but results differ"),
        ("gold_error", "Sample: GOLD SQL itself fails to execute (conversion bug)"),
    ]:
        exs = examples[category]
        if not exs:
            continue
        print(f"\n--- {label} ---")
        for ex in exs:
            print(f"\n  db={ex['db_id']}  Q: {ex['question']}")
            print(f"    GOLD: {ex['gold_sql']}")
            print(f"    PRED: {ex['pred_sql']}")
            if "error" in ex:
                print(f"    ERROR: {ex['error']}")
            if "gold_rows_sample" in ex:
                print(f"    GOLD ROWS (sample): {ex['gold_rows_sample']}")
                print(f"    PRED ROWS (sample): {ex['pred_rows_sample']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Diagnose WikiSQL EX gap")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples_per_category", type=int, default=4)
    args = parser.parse_args()

    diagnose(
        gold_path=args.gold,
        db_dir=args.db,
        predict_path=args.predict,
        limit=args.limit,
        samples_per_category=args.samples_per_category,
    )


if __name__ == "__main__":
    main()