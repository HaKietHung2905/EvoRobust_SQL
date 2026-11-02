"""
official_spider_eval.py
=======================
Re-score Spider prediction TSVs with the OFFICIAL evaluator
(taoyds/test-suite-sql-eval), independent of the project's own evaluator.

Why: reviewers flagged Table VI (per-difficulty) as inconsistent with the
overall numbers. Root cause was a parser quirk in src/data/sql_parser.py
(orderBy=('asc', []) for queries without ORDER BY) that pushed almost every
query one difficulty level up. This script gives an independent reference:
hardness, EM and EX are all computed by the official code, with no project
normalisation applied to the predictions.

Outputs
  <out_dir>/<name>.per_item.jsonl   one line per question:
        {"idx", "db_id", "hardness", "em", "ex" (if --db), "gold", "pred"}
  <out_dir>/<name>.summary.json     counts + EM/EX per level
  stdout                            Table-VI style table

Modes
  EM only (no databases needed): schema is read from tables.json.
  EM + EX:  pass --db data/spider/database
  Test-suite EX (Zhong et al., 2020): pass --db <test-suite database dir>
        (download from the test-suite-sql-eval README); EX then checks the
        prediction on every DB variant in that folder, which removes most
        single-DB false positives.

Usage
  git clone https://github.com/taoyds/test-suite-sql-eval third_party/test-suite-sql-eval
  python scripts/official_spider_eval.py \
      --official_dir third_party/test-suite-sql-eval \
      --gold   data/spider/dev.json \
      --tables data/spider/tables.json \
      --pred   results/predictions_spider_full_v2.tsv results/predictions_zeroshot_spider.tsv \
      --db     data/spider/database \
      --out_dir results/official_eval
"""
import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

LEVELS = ["easy", "medium", "hard", "extra", "all"]
EMPTY_SQL = {
    "except": None, "from": {"conds": [], "table_units": []}, "groupBy": [],
    "having": [], "intersect": None, "limit": None, "orderBy": [],
    "select": [False, []], "union": None, "where": [],
}


def load_official(official_dir: str):
    path = Path(official_dir).resolve()
    missing = [f for f in ("evaluation.py", "process_sql.py") if not (path / f).exists()]
    if missing:
        sys.exit(
            f"ERROR: {path} does not contain {', '.join(missing)}.\n"
            f"Clone the official evaluator first (from the repo root):\n"
            f"  mkdir -p third_party && git clone https://github.com/taoyds/test-suite-sql-eval "
            f"third_party/test-suite-sql-eval\n"
            f"or pass --official_dir <path to your clone>."
        )
    official_dir = str(path)
    if official_dir not in sys.path:
        sys.path.insert(0, official_dir)
    import evaluation as ev          # noqa: E402  (official module)
    import process_sql as ps         # noqa: E402
    try:
        from exec_eval import eval_exec_match as ts_exec   # test-suite repo
    except Exception:
        ts_exec = None
    return ev, ps, ts_exec


def check_tables(tables, needed_dbs, path):
    """
    Refuse a tables.json that is not the official Spider file. The project's
    HuggingFace downloader (src/data/spider_downloader.py) writes a tables.json
    reconstructed from SQL strings: columns without table indices and no
    foreign/primary keys. With it, the official EM cannot map foreign keys.
    """
    problems = []
    missing = sorted(needed_dbs - set(tables))
    if missing:
        problems.append(f"{len(missing)} gold databases are missing (e.g. {missing[:3]})")
    for db in sorted(needed_dbs & set(tables)):
        t = tables[db]
        cols = t.get("column_names_original", [])
        if not cols or not all(isinstance(c, (list, tuple)) and len(c) == 2 and isinstance(c[0], int)
                               for c in cols):
            problems.append(f"'{db}': column_names_original is not [[table_idx:int, name], ...]")
            break
    if not problems and not any(tables[db].get("foreign_keys") for db in needed_dbs & set(tables)):
        problems.append("no database has foreign_keys (reconstructed schema?)")
    if problems:
        sys.exit(
            f"ERROR: {path} is not the official Spider tables.json:\n  - " + "\n  - ".join(problems) +
            "\nUse the official file, e.g. <test-suite-sql-eval>/tables.json or tables.json from the "
            "Spider release zip (not the one written by src/data/spider_downloader.py)."
        )


def schema_from_tables(entry):
    """Same shape as process_sql.get_schema(db): {table_lower: [col_lower, ...]}."""
    schema = {}
    for t in entry["table_names_original"]:
        schema[t.lower()] = []
    for ti, col in entry["column_names_original"]:
        if ti >= 0:
            schema[entry["table_names_original"][ti].lower()].append(col.lower())
    return schema


def load_gold(path: str):
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return [(x["query"], x["db_id"]) for x in json.load(f)]
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q, db = line.rstrip("\n").split("\t")
                out.append((q, db))
    return out


def load_pred(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            out.append(line.split("\t")[0])
    return out


def score_file(pred_path, gold, tables, db_dir, ev, ps, ts_exec, plug_value, keep_distinct):
    preds = load_pred(pred_path)
    if len(preds) != len(gold):
        raise ValueError(f"{pred_path}: {len(preds)} predictions vs {len(gold)} gold")

    evaluator = ev.Evaluator()
    kmaps = {db: ev.build_foreign_key_map(t) for db, t in tables.items()}
    items = []
    for i, (p_str, (g_str, db_id)) in enumerate(zip(preds, gold)):
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite") if db_dir else None
        if db_path and os.path.exists(db_path):
            schema = ps.Schema(ps.get_schema(db_path))
        else:
            schema = ps.Schema(schema_from_tables(tables[db_id]))

        g_sql = ps.get_sql(schema, g_str)
        hardness = evaluator.eval_hardness(g_sql)
        p_str_eval = p_str.replace("value", "1")
        try:
            p_sql = ps.get_sql(schema, p_str_eval)
            parse_ok = True
        except Exception:
            p_sql = json.loads(json.dumps(EMPTY_SQL))
            parse_ok = False

        kmap = kmaps[db_id]
        g_valid = ev.build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = ev.rebuild_sql_col(g_valid, ev.rebuild_sql_val(g_sql), kmap)
        p_valid = ev.build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = ev.rebuild_sql_col(p_valid, ev.rebuild_sql_val(p_sql), kmap)
        em = int(bool(evaluator.eval_exact_match(p_sql, g_sql)))

        rec = OrderedDict(idx=i, db_id=db_id, hardness=hardness, em=em,
                          pred_parsed=parse_ok, gold=g_str, pred=p_str)
        if db_path and os.path.exists(db_path):
            if ts_exec is not None:
                ex = ts_exec(db_path, p_str_eval, g_str, plug_value, keep_distinct, False)
            else:  # original spider repo signature
                ex = ev.eval_exec_match(db_path, p_str_eval, g_str, p_sql, g_sql)
            rec["ex"] = int(bool(ex))
        items.append(rec)
    return items


def summarize(items):
    has_ex = all("ex" in it for it in items)
    s = {lvl: {"count": 0, "em": 0, "ex": 0} for lvl in LEVELS}
    for it in items:
        for lvl in (it["hardness"], "all"):
            s[lvl]["count"] += 1
            s[lvl]["em"] += it["em"]
            if has_ex:
                s[lvl]["ex"] += it["ex"]
    for lvl in LEVELS:
        c = s[lvl]["count"] or 1
        s[lvl]["em_acc"] = s[lvl]["em"] / c
        s[lvl]["ex_acc"] = s[lvl]["ex"] / c if has_ex else None
    # consistency check that reviewers performed by hand
    weighted = sum(s[l]["em"] for l in LEVELS[:-1])
    assert weighted == s["all"]["em"], "per-level EM does not add up to overall"
    assert sum(s[l]["count"] for l in LEVELS[:-1]) == s["all"]["count"]
    return s


def print_table(name, s):
    print(f"\n{name}")
    print(f"{'level':<8}{'count':>7}{'EM':>9}{'EX':>9}")
    for lvl in LEVELS:
        r = s[lvl]
        ex = f"{100 * r['ex_acc']:8.1f}" if r["ex_acc"] is not None else "       -"
        print(f"{lvl:<8}{r['count']:>7}{100 * r['em_acc']:8.1f} {ex}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official_dir", required=True,
                    help="Path to a clone of taoyds/test-suite-sql-eval (or taoyds/spider)")
    ap.add_argument("--gold", required=True, help="Spider dev.json, or gold.txt (sql<TAB>db_id)")
    ap.add_argument("--tables", required=True, help="Spider tables.json")
    ap.add_argument("--pred", required=True, nargs="+", help="One or more prediction TSVs (sql<TAB>db_id)")
    ap.add_argument("--db", default=None, help="Database dir; enables EX. Test-suite DB dir => test-suite EX")
    ap.add_argument("--out_dir", default="results/official_eval")
    ap.add_argument("--plug_value", action="store_true")
    ap.add_argument("--keep_distinct", action="store_true")
    args = ap.parse_args()

    ev, ps, ts_exec = load_official(args.official_dir)
    gold = load_gold(args.gold)
    with open(args.tables, encoding="utf-8") as f:
        tables = {t["db_id"]: t for t in json.load(f)}
    check_tables(tables, {db for _, db in gold}, args.tables)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for pred_path in args.pred:
        name = Path(pred_path).stem
        items = score_file(pred_path, gold, tables, args.db, ev, ps, ts_exec,
                           args.plug_value, args.keep_distinct)
        with open(out_dir / f"{name}.per_item.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        s = summarize(items)
        with open(out_dir / f"{name}.summary.json", "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        all_summaries[name] = s
        print_table(name, s)

    with open(out_dir / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()