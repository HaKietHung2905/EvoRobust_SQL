#!/usr/bin/env python3
"""
analyze_failures.py
=====================
Root-cause breakdown of execution-accuracy failures on a Spider predictions
TSV, reusing the project's own schema/parser/hardness/exec-match utilities
(load_schema, parse_sql, eval_hardness, eval_exec_match) so the numbers here
line up exactly with what evaluate_spider.py reports.

For every question where execution accuracy fails, this classifies it as:
  - "exec_error"    : the predicted SQL throws a SQLite error when run
                       (syntax error, no such column/table, ambiguous
                       column, etc.) — usually the cheapest category to fix,
                       often a single systemic prompt/schema issue.
  - "wrong_result"   : predicted SQL runs fine but returns a different
                       result set than gold — a real reasoning/logic gap.
  - "gold_parse_error": gold SQL itself failed to parse with the project's
                       parser (rare, but worth knowing about separately so
                       it doesn't get misread as a model failure).

Breaks failures down by:
  - hardness (easy/medium/hard/extra, via eval_hardness on the parsed gold)
  - db_id (to catch schema-loading issues isolated to specific databases)
  - clustered SQLite error message (first few words) for exec_error rows

Also prints concrete gold/predicted SQL pairs for the top failure clusters,
so you can go straight to "this pattern of prompt is wrong" instead of
guessing.

Usage:
    python scripts/analyze_failures.py \
        --gold data/raw/spider/dev.json \
        --db   data/raw/spider/database \
        --predict results/predictions_spider_full.tsv \
        --output_dir output/failure_analysis \
        --samples_per_cluster 3
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sql_schema import load_schema
from src.data.sql_parser import parse_sql
from src.evaluation.hardness import eval_hardness
from src.evaluation.exec_evaluator import eval_exec_match
from utils.eval_utils import normalize_sql_for_evaluation


def _run_sql(db_path: str, sql: str) -> Tuple[bool, Optional[str]]:
    """Execute `sql` against db_path. Returns (ok, error_message)."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        cur.fetchall()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)


def _extract_select_columns(sql: str) -> List[str]:
    """Heuristic extraction of the top-level SELECT column list, normalized
    for comparison: lowercased, table-alias prefixes stripped (t1.col ->
    col), whitespace collapsed. Not a real parser — good enough to detect
    'model selected different columns than gold asked for' as a signal,
    not as a scoring mechanism (eval_exec_match already does real scoring)."""
    m = re.search(r"\bSELECT\b\s+(DISTINCT\s+)?(.*?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    col_block = m.group(2)
    # naive top-level comma split (ignores commas inside parens, e.g. func args)
    parts, depth, current = [], 0, ""
    for ch in col_block:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)

    cols = []
    for p in parts:
        c = p.strip().lower()
        c = re.sub(r"\bas\s+\w+$", "", c).strip()          # drop "AS alias"
        c = re.sub(r"^[a-z_]\w*\.", "", c)                  # drop "t1." prefix
        c = re.sub(r"\s+", " ", c)
        if c:
            cols.append(c)
    return cols


def _has_subquery_pattern(sql: str) -> bool:
    """True if the query uses NOT IN / EXCEPT / INTERSECT / UNION, or has a
    nested SELECT — the multi-step-reasoning patterns that showed up
    repeatedly in the manual sample review (has-X-but-not-Y questions)."""
    s = sql.upper()
    if re.search(r"\bNOT\s+IN\b|\bEXCEPT\b|\bINTERSECT\b|\bUNION\b", s):
        return True
    return s.count("SELECT") > 1


def _extract_where_columns(sql: str) -> List[str]:
    """Heuristic extraction of column names referenced in comparisons inside
    the WHERE clause (col = val, col > val, col LIKE val, col IN (...)),
    normalized the same way as _extract_select_columns. Catches the
    'compared against the wrong column' failure mode (e.g. filtering on
    Model when gold filters on Make) that a SELECT-column-only comparison
    misses entirely, since the SELECT list itself can be identical while
    the WHERE clause silently targets the wrong column."""
    m = re.search(
        r"\bWHERE\b(.*?)(\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|"
        r"\bEXCEPT\b|\bINTERSECT\b|\bUNION\b|;|$)",
        sql, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []
    where_block = m.group(1)
    cols = []
    for match in re.finditer(
        r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?)\s*(?:=|>|<|>=|<=|<>|!=|\bLIKE\b|\bIN\b|\bBETWEEN\b)",
        where_block, re.IGNORECASE,
    ):
        col = match.group(1).lower()
        col = re.sub(r"^[a-z_]\w*\.", "", col)  # drop "t1." prefix
        if col not in ("and", "or", "not"):
            cols.append(col)
    return cols


def _error_cluster_key(error_msg: str) -> str:
    """Collapse a raw SQLite error message down to a stable cluster key,
    e.g. 'no such column: t1.name' and 'no such column: t2.age' both
    become 'no such column'."""
    msg = error_msg.lower()
    for pattern, key in [
        (r"no such column", "no such column"),
        (r"no such table", "no such table"),
        (r"ambiguous column name", "ambiguous column name"),
        (r"syntax error", "syntax error"),
        (r"misuse of aggregate", "misuse of aggregate function"),
        (r"unrecognized token", "unrecognized token"),
        (r"near \"", "syntax error near token"),
        (r"incomplete input", "incomplete input"),
        (r"datatype mismatch", "datatype mismatch"),
    ]:
        if re.search(pattern, msg):
            return key
    return "other: " + msg[:60]


def analyze(
    gold_path: str,
    db_dir: str,
    predict_path: str,
    output_dir: str,
    samples_per_cluster: int = 3,
    limit: Optional[int] = None,
) -> Dict:
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
        print(
            f"⚠ WARNING: gold has {len(gold_data)} rows, predictions has "
            f"{len(pred_lines)} rows — comparing only the first {n} "
            f"(they must be line-aligned to the same dev.json ordering)."
        )

    hardness_stats = defaultdict(lambda: {"total": 0, "pass": 0, "exec_error": 0, "wrong_result": 0})
    db_stats = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0})
    error_clusters = Counter()
    cluster_examples = defaultdict(list)
    gold_parse_errors = []
    total = 0
    total_pass = 0

    # wrong_result sub-classification — flags are independent, not mutually
    # exclusive (a row can be both select_mismatch AND subquery_pattern).
    wrong_result_flags = Counter()
    wrong_result_flag_examples = defaultdict(list)
    n_wrong_result_total = 0

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
        db_stats[db_id]["total"] += 1

        # ── Hardness (via project's own parser) ──
        try:
            schema = load_schema(db_path)
            g_norm = normalize_sql_for_evaluation(gold_sql) or gold_sql
            g_parsed = parse_sql(g_norm, schema)
            hardness = eval_hardness(g_parsed)
        except Exception as e:
            hardness = "unknown"
            gold_parse_errors.append({
                "index": i, "db_id": db_id, "question": question,
                "gold_sql": gold_sql, "parse_error": str(e),
            })

        hardness_stats[hardness]["total"] += 1

        # ── Execution match (same function evaluate_spider.py uses) ──
        try:
            exec_ok = eval_exec_match(
                db=db_path, p_str=pred_sql, g_str=gold_sql,
                plug_value=False, keep_distinct=False,
            )
        except Exception:
            exec_ok = 0

        if exec_ok:
            total_pass += 1
            hardness_stats[hardness]["pass"] += 1
            db_stats[db_id]["pass"] += 1
            continue

        db_stats[db_id]["fail"] += 1

        # ── Failure: classify exec_error vs wrong_result ──
        pred_ok, pred_err = _run_sql(db_path, pred_sql)
        record = {
            "index": i, "db_id": db_id, "question": question,
            "gold_sql": gold_sql, "predicted_sql": pred_sql, "hardness": hardness,
        }

        if not pred_ok:
            hardness_stats[hardness]["exec_error"] += 1
            cluster_key = _error_cluster_key(pred_err)
            error_clusters[cluster_key] += 1
            record["error"] = pred_err
            if len(cluster_examples[cluster_key]) < samples_per_cluster:
                cluster_examples[cluster_key].append(record)
        else:
            hardness_stats[hardness]["wrong_result"] += 1
            n_wrong_result_total += 1
            if len(cluster_examples["wrong_result"]) < samples_per_cluster * 3:
                cluster_examples["wrong_result"].append(record)

            # ── Sub-classify this wrong_result row ──
            flags = []
            if pred_sql.strip().upper() == "SELECT 1":
                flags.append("select1_fallback")
            else:
                gold_cols = _extract_select_columns(gold_sql)
                pred_cols = _extract_select_columns(pred_sql)
                if gold_cols and pred_cols and set(gold_cols) != set(pred_cols):
                    flags.append("select_mismatch")

                gold_where_cols = _extract_where_columns(gold_sql)
                pred_where_cols = _extract_where_columns(pred_sql)
                if gold_where_cols and pred_where_cols and set(gold_where_cols) != set(pred_where_cols):
                    flags.append("where_column_mismatch")

                if _has_subquery_pattern(gold_sql) or _has_subquery_pattern(pred_sql):
                    flags.append("subquery_pattern")

            if not flags:
                flags.append("unclassified")

            record["wrong_result_flags"] = flags
            for flag in flags:
                wrong_result_flags[flag] += 1
                if len(wrong_result_flag_examples[flag]) < samples_per_cluster:
                    wrong_result_flag_examples[flag].append(record)

    # ── Report ──
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "total": total,
        "pass": total_pass,
        "execution_accuracy": total_pass / total if total else 0.0,
        "hardness_breakdown": dict(hardness_stats),
        "db_breakdown": dict(db_stats),
        "error_clusters": dict(error_clusters.most_common()),
        "gold_parse_errors_count": len(gold_parse_errors),
        "wrong_result_total": n_wrong_result_total,
        "wrong_result_flags": dict(wrong_result_flags.most_common()),
    }

    with open(out_dir / "failure_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "report": report,
            "gold_parse_errors": gold_parse_errors[:20],
            "cluster_examples": cluster_examples,
            "wrong_result_flag_examples": wrong_result_flag_examples,
        }, f, indent=2, ensure_ascii=False)

    _print_report(report, error_clusters, cluster_examples,
                  wrong_result_flags, wrong_result_flag_examples, n_wrong_result_total)

    print(f"\n✓ Full detail (incl. all sample pairs) saved → {out_dir / 'failure_report.json'}")
    return report


def _print_report(
    report: Dict,
    error_clusters: Counter,
    cluster_examples: Dict,
    wrong_result_flags: Counter,
    wrong_result_flag_examples: Dict,
    n_wrong_result_total: int,
) -> None:
    print("=" * 70)
    print("EXECUTION-ACCURACY FAILURE ANALYSIS")
    print("=" * 70)
    print(f"Total evaluated : {report['total']}")
    print(f"Passed (EX)     : {report['pass']} ({report['execution_accuracy']:.2%})")
    print(f"Failed (EX)     : {report['total'] - report['pass']}")
    if report["gold_parse_errors_count"]:
        print(f"⚠ Gold SQL parse errors (excluded from hardness buckets): "
              f"{report['gold_parse_errors_count']}")

    print("\n--- Breakdown by hardness ---")
    print(f"{'hardness':<10} {'total':>6} {'pass':>6} {'EX%':>7} {'exec_err':>9} {'wrong_res':>10}")
    for level in ["easy", "medium", "hard", "extra", "unknown"]:
        s = report["hardness_breakdown"].get(level)
        if not s or s["total"] == 0:
            continue
        ex_pct = s["pass"] / s["total"] if s["total"] else 0.0
        print(f"{level:<10} {s['total']:>6} {s['pass']:>6} {ex_pct:>7.1%} "
              f"{s['exec_error']:>9} {s['wrong_result']:>10}")

    n_exec_err = sum(s["exec_error"] for s in report["hardness_breakdown"].values())
    n_wrong_res = sum(s["wrong_result"] for s in report["hardness_breakdown"].values())
    print(f"\nTotal failures split: {n_exec_err} exec_error (SQL threw an error) "
          f"vs {n_wrong_res} wrong_result (ran fine, wrong answer)")
    if n_exec_err + n_wrong_res > 0:
        print(f"  → {n_exec_err / (n_exec_err + n_wrong_res):.0%} of failures are "
              f"exec_error — usually the cheaper fix (schema/prompt issue),")
        print(f"    vs {n_wrong_res / (n_exec_err + n_wrong_res):.0%} wrong_result "
              f"— usually a real reasoning gap (needs better retrieval/context).")

    print("\n--- Top SQLite error clusters (exec_error rows) ---")
    for cluster, count in error_clusters.most_common(10):
        print(f"  {count:>4}x  {cluster}")

    print(f"\n--- wrong_result sub-classification ({n_wrong_result_total} rows; "
          f"flags are independent, a row can match more than one) ---")
    if n_wrong_result_total:
        for flag, count in wrong_result_flags.most_common():
            print(f"  {count:>4}x ({count / n_wrong_result_total:.0%})  {flag}")
        print("\n  Legend:")
        print("    select1_fallback      = model produced literal 'SELECT 1' — pure")
        print("                            generation failure, not a reasoning error.")
        print("    select_mismatch       = predicted SELECT column list differs from gold's.")
        print("    where_column_mismatch = predicted WHERE clause filters on a different")
        print("                            column than gold (e.g. Model vs Make) — often a")
        print("                            schema-naming ambiguity specific to one database.")
        print("    subquery_pattern      = gold or prediction uses NOT IN / EXCEPT /")
        print("                            INTERSECT / UNION / nested SELECT.")
        print("    unclassified          = wrong_result but none of the above fired;")
        print("                            needs manual read (see JSON for full list).")

    print("\n--- Databases with the worst EX rate (min 5 questions) ---")
    db_rates = [
        (db_id, s["pass"] / s["total"], s["total"])
        for db_id, s in report["db_breakdown"].items()
        if s["total"] >= 5
    ]
    db_rates.sort(key=lambda x: x[1])
    for db_id, rate, total in db_rates[:8]:
        print(f"  {db_id:<25} EX={rate:.1%}  (n={total})")

    print("\n--- Sample failures (top error cluster) ---")
    if error_clusters:
        top_cluster = error_clusters.most_common(1)[0][0]
        for ex in cluster_examples.get(top_cluster, [])[:3]:
            print(f"\n  [{ex['hardness']}] db={ex['db_id']}  Q: {ex['question']}")
            print(f"    GOLD: {ex['gold_sql']}")
            print(f"    PRED: {ex['predicted_sql']}")
            print(f"    ERROR: {ex.get('error', '')}")

    for flag in ["select1_fallback", "select_mismatch", "where_column_mismatch",
                 "subquery_pattern", "unclassified"]:
        examples = wrong_result_flag_examples.get(flag, [])
        if not examples:
            continue
        print(f"\n--- Sample: {flag} ---")
        for ex in examples[:2]:
            print(f"\n  [{ex['hardness']}] db={ex['db_id']}  Q: {ex['question']}")
            print(f"    GOLD: {ex['gold_sql']}")
            print(f"    PRED: {ex['predicted_sql']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Root-cause analysis of EX failures")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output_dir", default="output/failure_analysis")
    parser.add_argument("--samples_per_cluster", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    analyze(
        gold_path=args.gold,
        db_dir=args.db,
        predict_path=args.predict,
        output_dir=args.output_dir,
        samples_per_cluster=args.samples_per_cluster,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()