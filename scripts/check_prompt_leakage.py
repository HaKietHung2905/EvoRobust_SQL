"""
check_prompt_leakage.py
=======================
Detect Spider-dev gold queries that appear (EM-equivalent, official
evaluator) among the SQL snippets hard-coded in the Spider prompt header.

    python scripts/check_prompt_leakage.py \
        --official_dir third_party/test-suite-sql-eval \
        --dev data/spider/dev.json --tables data/spider/tables.json \
        --variant legacy_dev            # or: train

Result for the submitted paper (legacy_dev): 18 snippets, 16 dev questions
whose gold SQL is EM-equivalent to a snippet (dev idx 22, 23, 31, 32, 43, 44,
51, 52, 65, 66, 71, 72, 85, 86, 95, 96). With --variant train
the expected count is 0. Use --write_ids to save the indices so they can be
excluded in scripts/analyze_run.py (--exclude_ids).
"""
import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_dir", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--variant", default="legacy_dev", choices=["legacy_dev", "train"])
    ap.add_argument("--write_ids", default=None, help="Write leaked dev indices to this JSON file")
    args = ap.parse_args()

    os.environ["SPIDER_PROMPT_EXAMPLES"] = args.variant
    sys.path.insert(0, str(Path(args.official_dir).resolve()))
    import evaluation as ev          # official
    import process_sql as ps         # official
    from src.generation.sql_generator import SQLGenerator

    gen = object.__new__(SQLGenerator)     # no LLM client needed to build a prompt
    text = gen._construct_prompt_spider("QQQ", "")
    snippets = [m.strip() for m in re.findall(r"(?:A|GOOD|BAD):\s+(SELECT.+?)(?=\n|$)", text)]

    dev = json.load(open(args.dev, encoding="utf-8"))
    tables = {t["db_id"]: t for t in json.load(open(args.tables, encoding="utf-8"))}
    kmaps = {db: ev.build_foreign_key_map(t) for db, t in tables.items()}
    evaluator = ev.Evaluator()

    def schema(db):
        t = tables[db]
        d = {x.lower(): [] for x in t["table_names_original"]}
        for ti, c in t["column_names_original"]:
            if ti >= 0:
                d[t["table_names_original"][ti].lower()].append(c.lower())
        return ps.Schema(d)

    def prep(sql, db, sch):
        p = ps.get_sql(sch, sql)
        valid = ev.build_valid_col_units(p["from"]["table_units"], sch)
        return ev.rebuild_sql_col(valid, ev.rebuild_sql_val(p), kmaps[db])

    hits = {}
    for i, x in enumerate(dev):
        sch = schema(x["db_id"])
        try:
            g = prep(x["query"], x["db_id"], sch)
        except Exception:
            continue
        for s in snippets:
            try:
                # eval_exact_match mutates its arguments -> compare on copies
                if evaluator.eval_exact_match(prep(s, x["db_id"], sch), copy.deepcopy(g)):
                    hits.setdefault(i, s)
                    break
            except Exception:
                continue

    print(f"variant={args.variant}: {len(snippets)} SQL snippets in the Spider prompt header")
    print(f"dev questions whose gold SQL is EM-equivalent to a prompt snippet: {len(hits)}")
    for i, s in sorted(hits.items()):
        print(f"  dev[{i}] {dev[i]['db_id']}: {dev[i]['question']}")
    if args.write_ids:
        with open(args.write_ids, "w") as f:
            json.dump(sorted(hits), f)
    return 0 if args.variant == "legacy_dev" or not hits else 1


if __name__ == "__main__":
    sys.exit(main())