"""
analyze_ablation.py
===================
Turn the output of scripts/run_ablation.py into the tables and figures the
reviewers asked for. Reads only files on disk; no LLM calls.

    python scripts/analyze_ablation.py --ablation_dir results/ablation/spider/online
    # optional: --reference full  --window 100  --exclude_ids leaked_dev_ids.json

Writes to <ablation_dir>/analysis/:
  report.md                      all tables below, ready to adapt for the paper
  main_table.csv                 config x metric, mean ± std over orders, n_orders
  significance.csv               exact McNemar test of each config vs the reference
  per_difficulty.csv             (Spider) EM/EX per hardness level per config
  pseudo_labels.csv              self-consistency outcome vs gold EX (reliability)
  learning_curve.csv             EX by processing step + memory size
  memory_evolution.csv           one row per consolidation (lifecycle decisions)
  strategies_<cfg>_<seed>.csv    strategies in memory at the end (+ r_s, n)
  complementarity.csv            EX overlap between pairs of configurations
  cost.csv                       LLM calls / tokens / latency per question
  *.png                          learning curves (if matplotlib is installed)

Inputs expected per configuration directory (created by run_ablation.py):
  <cfg>/seed-<s>.tsv, .trace.jsonl, .manifest.json, .consolidation.jsonl
  <cfg>/eval/seed-<s>.per_item.jsonl            (and .majority_vote.per_item.jsonl)
  <cfg>/memory_seed-<s>/reasoning_bank.db
"""
import argparse
import csv
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import run_artifacts as ra  # noqa: E402

LEVELS = ["easy", "medium", "hard", "extra"]
CONFIG_ORDER = ["template_only", "semantic_layer", "semantic_rag", "reasoning_bank",
                "layer_rag", "rag_rb", "full"]


# ───────────────────────────── statistics ──────────────────────────────────
def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired 0/1 outcomes. Returns (b01, b10, p)."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return b01, b10, min(1.0, p)


def fmt(m, s=None, pct=True):
    if m is None:
        return "–"
    if pct:
        return f"{100 * m:.1f}" + (f" ± {100 * s:.1f}" if s else "")
    return f"{m:.2f}" + (f" ± {s:.2f}" if s else "")


# ───────────────────────────── loading ─────────────────────────────────────
def load_per_item(path, exclude):
    rows = ra.read_jsonl(path)
    return {int(r["idx"]): r for r in rows if int(r["idx"]) not in exclude}


def metric_keys(sample_row):
    keys = [k for k in ("em", "ex", "em_strict", "em_lenient") if k in sample_row]
    return keys


def discover(ablation_dir, exclude):
    """runs[cfg][variant][seed] = {'items': {idx: row}, 'trace':..., ...}"""
    runs = defaultdict(lambda: defaultdict(dict))
    for cdir in sorted(p for p in Path(ablation_dir).iterdir() if p.is_dir() and p.name != "analysis"):
        evald = cdir / "eval"
        if not evald.exists():
            continue
        for f in sorted(evald.glob("seed-*.per_item.jsonl")):
            stem = f.name[: -len(".per_item.jsonl")]
            variant = "majority_vote" if stem.endswith(".majority_vote") else "greedy"
            seed = stem.replace(".majority_vote", "").replace("seed-", "")
            base = cdir / f"seed-{seed}"
            runs[cdir.name][variant][seed] = {
                "items": load_per_item(f, exclude),
                "trace": ra.read_trace(str(base) + ".trace.jsonl"),
                "manifest": (json.load(open(str(base) + ".manifest.json"))
                             if Path(str(base) + ".manifest.json").exists() else {}),
                "consolidation": ra.read_jsonl(str(base) + ".consolidation.jsonl"),
                "memory_db": cdir / f"memory_seed-{seed}" / "reasoning_bank.db",
            }
    return runs


def ordered(cfgs):
    known = [c for c in CONFIG_ORDER if c in cfgs]
    return known + sorted(c for c in cfgs if c not in known)


# ───────────────────────────── analyses ────────────────────────────────────
def main_table(runs, keys):
    rows = []
    for cfg in ordered(runs):
        for variant in ("greedy", "majority_vote"):
            seeds = runs[cfg].get(variant)
            if not seeds:
                continue
            row = {"config": cfg if variant == "greedy" else f"{cfg} (majority vote)",
                   "n_orders": len(seeds), "n_questions": len(next(iter(seeds.values()))["items"])}
            for k in keys:
                accs = []
                for s in seeds.values():
                    vals = [r[k] for r in s["items"].values() if r.get(k) is not None]
                    accs.append(sum(vals) / len(vals) if vals else None)
                row[f"{k}_mean"], row[f"{k}_std"] = mean_std(accs)
            costs = [s["manifest"].get("cost", {}) for s in seeds.values() if s["manifest"].get("cost")]
            for ck in ("calls_mean", "prompt_tokens_mean", "completion_tokens_mean", "wall_s_mean"):
                vals = [c.get(ck) for c in costs if c.get(ck) is not None]
                row[ck] = statistics.mean(vals) if vals else None
            rows.append(row)
    return rows


def significance(runs, keys, reference):
    out = []
    if reference not in runs or "greedy" not in runs[reference]:
        return out
    ref_seeds = runs[reference]["greedy"]
    for cfg in ordered(runs):
        if cfg == reference or "greedy" not in runs[cfg]:
            continue
        for seed, other in runs[cfg]["greedy"].items():
            ref = ref_seeds.get(seed) or ref_seeds.get("none") or next(iter(ref_seeds.values()))
            common = sorted(set(ref["items"]) & set(other["items"]))
            for k in keys:
                a = [other["items"][i].get(k) for i in common]
                b = [ref["items"][i].get(k) for i in common]
                pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
                if not pairs:
                    continue
                b01, b10, p = mcnemar_exact([x for x, _ in pairs], [y for _, y in pairs])
                out.append({"config": cfg, "seed": seed, "metric": k, "reference": reference,
                            "acc_config": sum(x for x, _ in pairs) / len(pairs),
                            "acc_reference": sum(y for _, y in pairs) / len(pairs),
                            "only_reference_correct": b01, "only_config_correct": b10,
                            "p_value": p, "n": len(pairs)})
    # Majority vote over the self-consistency candidates vs the greedy answer
    # of the same run: does the memory add anything beyond the sampling it
    # already pays for? (reviewer 3, comment 5)
    for cfg in ordered(runs):
        for seed, mv in runs[cfg].get("majority_vote", {}).items():
            gr = runs[cfg].get("greedy", {}).get(seed)
            if not gr:
                continue
            common = sorted(set(mv["items"]) & set(gr["items"]))
            for k in keys:
                pairs = [(mv["items"][i].get(k), gr["items"][i].get(k)) for i in common]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                if not pairs:
                    continue
                b01, b10, p = mcnemar_exact([x for x, _ in pairs], [y for _, y in pairs])
                out.append({"config": f"{cfg} (majority vote)", "seed": seed, "metric": k,
                            "reference": f"{cfg} (greedy)",
                            "acc_config": sum(x for x, _ in pairs) / len(pairs),
                            "acc_reference": sum(y for _, y in pairs) / len(pairs),
                            "only_reference_correct": b01, "only_config_correct": b10,
                            "p_value": p, "n": len(pairs)})
    return out


def per_difficulty(runs, keys):
    out = []
    for cfg in ordered(runs):
        seeds = runs[cfg].get("greedy", {})
        if not seeds or "hardness" not in next(iter(next(iter(seeds.values()))["items"].values()), {}):
            continue
        for lvl in LEVELS + ["all"]:
            row = {"config": cfg, "level": lvl}
            for k in keys:
                accs = []
                count = 0
                for s in seeds.values():
                    sel = [r for r in s["items"].values() if lvl == "all" or r.get("hardness") == lvl]
                    count = len(sel)
                    accs.append(sum(r[k] for r in sel) / len(sel) if sel else None)
                row["count"] = count
                row[f"{k}_mean"], row[f"{k}_std"] = mean_std(accs)
            out.append(row)
    return out


def pseudo_labels(runs):
    """Reliability of the self-consistency signal against gold EX (post hoc)."""
    by_outcome, summary = [], []
    for cfg in ordered(runs):
        for seed, s in runs[cfg].get("greedy", {}).items():
            rows = [(t["self_consistency"], s["items"][i].get("ex"))
                    for i, t in s["trace"].items()
                    if t.get("self_consistency") and i in s["items"] and s["items"][i].get("ex") is not None]
            if not rows:
                continue
            cnt = Counter(sc["outcome"] for sc, _ in rows)
            for outcome, n in sorted(cnt.items()):
                exs = [ex for sc, ex in rows if sc["outcome"] == outcome]
                by_outcome.append({"config": cfg, "seed": seed, "outcome": outcome, "n": n,
                                   "share": n / len(rows), "gold_ex_rate": sum(exs) / len(exs)})
            pos = [ex for sc, ex in rows if sc["outcome"] == "consensus"]
            tp = sum(pos)
            all_correct = sum(ex for _, ex in rows)
            empty = [ex for sc, ex in rows if sc["outcome"] == "consensus" and sc.get("majority_is_empty")]
            nonempty = [ex for sc, ex in rows if sc["outcome"] == "consensus" and not sc.get("majority_is_empty")]
            summary.append({
                "config": cfg, "seed": seed, "n": len(rows),
                "consensus_precision": tp / len(pos) if pos else None,
                "consensus_recall": tp / all_correct if all_correct else None,
                "false_consensus_rate": (len(pos) - tp) / len(rows),
                "consensus_on_empty_n": len(empty),
                "consensus_on_empty_precision": sum(empty) / len(empty) if empty else None,
                "consensus_nonempty_precision": sum(nonempty) / len(nonempty) if nonempty else None,
            })
    return by_outcome, summary


def learning_curve(runs, window):
    rows, blocks = [], []
    for cfg in ordered(runs):
        for seed, s in runs[cfg].get("greedy", {}).items():
            steps = sorted((t["step"], i, t) for i, t in s["trace"].items() if i in s["items"])
            if not steps or "memory_size_after" not in steps[0][2]:
                continue
            exs = []
            for step, i, t in steps:
                ex = s["items"][i].get("ex")
                exs.append(ex if ex is not None else 0)
                win = exs[-window:]
                rows.append({"config": cfg, "seed": seed, "step": step, "idx": i, "ex": ex,
                             "rolling_ex": sum(win) / len(win),
                             "memory_size": t.get("memory_size_after"),
                             "strategies_used": int(bool(t.get("strategies")))})
            n = len(exs)
            for b in range(0, n, window):
                seg = exs[b:b + window]
                used = [r["strategies_used"] for r in rows[-n:][b:b + window]]
                blocks.append({"config": cfg, "seed": seed, "steps": f"{b}-{b + len(seg) - 1}",
                               "ex": sum(seg) / len(seg), "share_with_strategy": sum(used) / len(used)})
    return rows, blocks


def memory_evolution(runs):
    out = []
    for cfg in ordered(runs):
        for seed, s in runs[cfg].get("greedy", {}).items():
            for ev in s["consolidation"]:
                lc = ev.get("lifecycle") or {}
                out.append({"config": cfg, "seed": seed, "step": ev.get("step"),
                            "final": bool(ev.get("final")),
                            "trajectories": ev.get("trajectories_collected"),
                            "new_distilled": ev.get("new_strategies"),
                            "n_strategies_after": ev.get("n_strategies_after"),
                            **{k: lc.get(k, 0) for k in ("kept", "refined", "split", "deprecated",
                                                          "deleted", "merged", "new_created")}})
    return out


def dump_strategies(runs, out_dir):
    written = []
    for cfg in ordered(runs):
        for seed, s in runs[cfg].get("greedy", {}).items():
            db = s["memory_db"]
            if not Path(db).exists():
                continue
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                rows = con.execute(
                    "SELECT s.name, s.pattern, s.description, s.success_rate, s.sample_count, "
                    "s.version, s.is_active, COUNT(a.application_id) "
                    "FROM strategies s LEFT JOIN strategy_applications a ON a.strategy_id = s.strategy_id "
                    "GROUP BY s.strategy_id ORDER BY COUNT(a.application_id) DESC").fetchall()
            except sqlite3.Error:
                rows = []
            finally:
                con.close()
            path = out_dir / f"strategies_{cfg}_{seed}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["name", "pattern", "description", "success_rate_r_s", "sample_count",
                            "version", "is_active", "n_applications"])
                w.writerows(rows)
            written.append((cfg, seed, path, rows[:5]))
    return written


def complementarity(runs, key="ex"):
    out = []
    cfgs = [c for c in ordered(runs) if "greedy" in runs[c]]
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1:]:
            ra_ = runs[a]["greedy"].get("none") or next(iter(runs[a]["greedy"].values()))
            rb_ = runs[b]["greedy"].get("none") or next(iter(runs[b]["greedy"].values()))
            common = set(ra_["items"]) & set(rb_["items"])
            c = Counter()
            for q in common:
                x, y = ra_["items"][q].get(key), rb_["items"][q].get(key)
                if x is None or y is None:
                    continue
                c[(x, y)] += 1
            out.append({"a": a, "b": b, "metric": key, "both": c[(1, 1)], "only_a": c[(1, 0)],
                        "only_b": c[(0, 1)], "neither": c[(0, 0)],
                        "oracle_union": (c[(1, 1)] + c[(1, 0)] + c[(0, 1)]) / max(1, sum(c.values()))})
    return out


# ───────────────────────────── output ──────────────────────────────────────
def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def try_plot(curve, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    by = defaultdict(list)
    for r in curve:
        by[(r["config"], r["seed"])].append(r)
    if not by:
        return None
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    for (cfg, seed), rs in sorted(by.items()):
        steps = [r["step"] for r in rs]
        line, = ax1.plot(steps, [100 * r["rolling_ex"] for r in rs], lw=1.6, label=f"{cfg} / {seed}")
        ax2.plot(steps, [r["memory_size"] or 0 for r in rs], lw=1, ls="--", color=line.get_color(), alpha=.6)
    ax1.set_xlabel("questions processed")
    ax1.set_ylabel("rolling EX (%)")
    ax2.set_ylabel("strategies in memory (dashed)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=.25)
    fig.tight_layout()
    path = out_dir / "learning_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation_dir", required=True)
    ap.add_argument("--reference", default="full")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--exclude_ids", default=None,
                    help="JSON list of question indices to drop (e.g. prompt-leaked dev ids)")
    args = ap.parse_args()

    exclude = set(json.load(open(args.exclude_ids))) if args.exclude_ids else set()
    runs = discover(args.ablation_dir, exclude)
    if not runs:
        print(f"No evaluated runs found under {args.ablation_dir} (expected <cfg>/eval/seed-*.per_item.jsonl)")
        return 1
    sample = next(iter(next(iter(next(iter(runs.values())).values())).values()))["items"]
    keys = metric_keys(next(iter(sample.values())))
    out_dir = Path(args.ablation_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    mt = main_table(runs, keys)
    sig = significance(runs, keys, args.reference)
    pdiff = per_difficulty(runs, keys)
    pl_outcome, pl_summary = pseudo_labels(runs)
    curve, blocks = learning_curve(runs, args.window)
    mem = memory_evolution(runs)
    strategies = dump_strategies(runs, out_dir)
    comp = complementarity(runs, "ex" if "ex" in keys else keys[0])

    write_csv(out_dir / "main_table.csv", mt)
    write_csv(out_dir / "significance.csv", sig)
    write_csv(out_dir / "per_difficulty.csv", pdiff)
    write_csv(out_dir / "pseudo_labels.csv", pl_summary)
    write_csv(out_dir / "pseudo_labels_by_outcome.csv", pl_outcome)
    write_csv(out_dir / "learning_curve.csv", curve)
    write_csv(out_dir / "learning_curve_blocks.csv", blocks)
    write_csv(out_dir / "memory_evolution.csv", mem)
    write_csv(out_dir / "complementarity.csv", comp)
    write_csv(out_dir / "cost.csv", [{k: r[k] for k in r if k == "config" or k.endswith("_mean") and
                                      k.split("_mean")[0] in ("calls", "prompt_tokens", "completion_tokens", "wall_s")}
                                     for r in mt])
    plot = try_plot(curve, out_dir)

    # ── report.md ──
    md = [f"# Ablation analysis — `{args.ablation_dir}`", ""]
    if exclude:
        md += [f"Excluded {len(exclude)} question ids ({args.exclude_ids}).", ""]
    md += ["## Main results (mean ± std over processing orders)", "",
           md_table(["Config", "orders"] + [k.upper() for k in keys] +
                    ["LLM calls/q", "tokens in/q", "tokens out/q", "s/q"],
                    [[r["config"], r["n_orders"]] + [fmt(r[f"{k}_mean"], r[f"{k}_std"]) for k in keys] +
                     [fmt(r["calls_mean"], pct=False), fmt(r["prompt_tokens_mean"], pct=False),
                      fmt(r["completion_tokens_mean"], pct=False), fmt(r["wall_s_mean"], pct=False)]
                     for r in mt]), ""]
    if sig:
        md += [f"## Exact McNemar tests (paired, two-sided)", "",
               md_table(["Config", "vs", "order", "metric", "acc", "acc ref",
                         "only ref ✓", "only config ✓", "p"],
                        [[r["config"], r["reference"], r["seed"], r["metric"].upper(), fmt(r["acc_config"]),
                          fmt(r["acc_reference"]), r["only_reference_correct"],
                          r["only_config_correct"], f"{r['p_value']:.3g}"] for r in sig]), ""]
    if pdiff:
        md += ["## Per difficulty (official Spider hardness)", "",
               md_table(["Config", "Level", "n"] + [k.upper() for k in keys],
                        [[r["config"], r["level"], r["count"]] + [fmt(r[f"{k}_mean"], r[f"{k}_std"]) for k in keys]
                         for r in pdiff]), ""]
    if pl_summary:
        md += ["## Pseudo-label reliability (self-consistency vs gold EX, computed after the run)", "",
               md_table(["Config", "order", "n", "precision(consensus)", "recall(consensus)",
                         "false consensus", "consensus on empty: n / precision", "non-empty precision"],
                        [[r["config"], r["seed"], r["n"], fmt(r["consensus_precision"]),
                          fmt(r["consensus_recall"]), fmt(r["false_consensus_rate"]),
                          f"{r['consensus_on_empty_n']} / {fmt(r['consensus_on_empty_precision'])}",
                          fmt(r["consensus_nonempty_precision"])] for r in pl_summary]), "",
               md_table(["Config", "order", "outcome", "n", "share", "gold EX rate"],
                        [[r["config"], r["seed"], r["outcome"], r["n"], fmt(r["share"]),
                          fmt(r["gold_ex_rate"])] for r in pl_outcome]), ""]
    if blocks:
        md += [f"## Learning curve (EX per block of {args.window} questions, processing order)", "",
               md_table(["Config", "order", "steps", "EX", "share with ≥1 strategy"],
                        [[r["config"], r["seed"], r["steps"], fmt(r["ex"]), fmt(r["share_with_strategy"])]
                         for r in blocks]), ""]
        if plot:
            md += [f"![learning curve]({plot.name})", ""]
    if mem:
        md += ["## Memory evolution (one row per consolidation)", "",
               md_table(["Config", "order", "step", "new", "strategies after", "kept", "refined",
                         "split", "deprecated", "deleted", "merged"],
                        [[r["config"], r["seed"], ("final" if r["final"] else r["step"]), r["new_distilled"],
                          r["n_strategies_after"], r["kept"], r["refined"], r["split"], r["deprecated"],
                          r["deleted"], r["merged"]] for r in mem]), ""]
    if strategies:
        md += ["## Most-applied strategies at the end of each run", ""]
        for cfg, seed, path, top in strategies:
            md += [f"**{cfg} / order {seed}** ({path.name})", "",
                   md_table(["name", "pattern", "r_s", "n samples", "applications"],
                            [[t[0], t[1], fmt(t[3]) if t[3] is not None else "–", t[4], t[7]] for t in top]), ""]
    if comp:
        md += ["## Complementarity (questions solved, EX)", "",
               md_table(["A", "B", "both", "only A", "only B", "neither", "oracle(A∪B)"],
                        [[r["a"], r["b"], r["both"], r["only_a"], r["only_b"], r["neither"],
                          fmt(r["oracle_union"])] for r in comp]), ""]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:40]))
    print(f"\n✓ Analysis written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())