"""
run_ablation.py
===============
Component ablation that isolates each module (reviewers 1, 2, 3), on Spider
AND WikiSQL, with a fresh memory per run, several processing orders and a
static (frozen-memory) control. Every configuration runs from the SAME code
version and prompt, recorded in each run's manifest.

Configurations (all use the same "Enhanced" prompt template):
    template_only     Enhanced template, nothing else
    semantic_layer    + rule-based Semantic Layer
    semantic_rag      + Semantic RAG (k retrieved train examples)
    reasoning_bank    + ReasoningBank (memory + self-consistency labelling)
    layer_rag         + Semantic Layer + Semantic RAG   (= submitted "Semantic" row)
    rag_rb            + Semantic RAG + ReasoningBank
    full              + Semantic Layer + Semantic RAG + ReasoningBank

Stateless configurations (no ReasoningBank) do not depend on question order,
so they run once; ReasoningBank configurations run once per --seeds entry
("none" = original order, an integer = seeded shuffle).

Protocols
    online  (paper): memory starts empty and is updated on the evaluation set.
    static  : memory is built on --warmup_n questions sampled from
              --warmup_questions (Spider train) and then FROZEN for the
              evaluation set, i.e. a standard static benchmark evaluation.

Usage (Spider, online, 3 orders):
    python scripts/run_ablation.py --dataset spider \\
        --questions data/spider/dev.json --db data/spider/database \\
        --tables data/spider/tables.json \\
        --official_dir third_party/test-suite-sql-eval \\
        --out_dir results/ablation --seeds none 1 2

Usage (WikiSQL):
    python scripts/run_ablation.py --dataset wikisql \\
        --questions data/raw/wikisql/dev_wiki_format.json \\
        --db data/raw/wikisql/database --wikisql_gold data/raw/wikisql/dev.json \\
        --tables data/raw/wikisql/tables.json \\
        --reasoning_config configs/reasoning_config_wikisql.yaml \\
        --out_dir results/ablation --seeds none 1 2

Then:  python scripts/analyze_ablation.py --ablation_dir results/ablation/<dataset>

Re-running the same command resumes: finished runs are skipped, partial runs
continue with --resume.
"""
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import run_artifacts as ra  # noqa: E402

CONFIGS = {
    "template_only":  [],
    "semantic_layer": ["--use_semantic"],
    "semantic_rag":   ["--use_chromadb"],
    "reasoning_bank": ["--use_reasoning_bank"],
    "layer_rag":      ["--use_semantic", "--use_chromadb"],
    "rag_rb":         ["--use_chromadb", "--use_reasoning_bank"],
    "full":           ["--use_semantic", "--use_chromadb", "--use_reasoning_bank"],
}


def uses_rb(cfg):
    return "--use_reasoning_bank" in CONFIGS[cfg]


def run(cmd, env=None):
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def n_lines(path):
    if not Path(path).exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def is_complete(tsv, n_expected):
    man = ra.sidecar(str(tsv), ".manifest.json")
    if not man.exists() or n_lines(tsv) < n_expected:
        return False
    try:
        return bool(json.load(open(man)).get("complete"))
    except Exception:
        return False


def make_warmup_file(src, n, seed, out_path):
    data = json.load(open(src, encoding="utf-8"))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(data)), min(n, len(data))))
    sub = [data[i] for i in idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sub, open(out_path, "w", encoding="utf-8"), indent=1)
    return out_path


def generate(args, cfg, seed, out_tsv, n_expected, env, memory_dir=None, frozen=False,
             questions=None, run_name=None):
    if is_complete(out_tsv, n_expected):
        print(f"✓ {out_tsv} already complete — skipping")
        return
    cmd = [args.python, str(ROOT / "scripts" / "generate_predictions.py"),
           "--questions", questions or args.questions, "--db", args.db,
           "--output", str(out_tsv), "--top_k", str(args.top_k),
           "--chromadb_persist_dir", args.chromadb_persist_dir,
           "--reasoning_config", args.reasoning_config,
           "--run_name", run_name or f"{args.dataset}/{cfg}/seed-{seed}",
           *CONFIGS[cfg]]
    if args.limit and questions is None:
        cmd += ["--limit", str(args.limit)]
    if seed not in (None, "none"):
        cmd += ["--shuffle_seed", str(seed)]
    if memory_dir is not None:
        cmd += ["--rb_memory_dir", str(memory_dir)]
        if not frozen:
            cmd += ["--fresh_memory"]
    if frozen:
        cmd += ["--rb_frozen"]
    work = ra.sidecar(str(out_tsv), ".processing_order.tsv") if seed not in (None, "none") else Path(out_tsv)
    if n_lines(work) > 0:
        cmd += ["--resume"]
    run(cmd, env=env)


def evaluate(args, pred_tsv, env):
    pred_tsv = Path(pred_tsv)
    if not pred_tsv.exists():
        return
    out_dir = pred_tsv.parent / "eval"
    if args.dataset == "spider":
        if not args.official_dir:
            print("  (no --official_dir: skipping Spider evaluation)")
            return
        if (out_dir / f"{pred_tsv.stem}.summary.json").exists():
            return
        cmd = [args.python, str(ROOT / "scripts" / "official_spider_eval.py"),
               "--official_dir", args.official_dir, "--gold", args.questions,
               "--tables", args.tables, "--pred", str(pred_tsv), "--db", args.db,
               "--out_dir", str(out_dir)]
        if args.test_suite_db:
            # second pass with test-suite databases -> separate folder
            run(cmd, env=env)
            cmd = cmd[:-2] + ["--out_dir", str(pred_tsv.parent / "eval_testsuite")]
            cmd[cmd.index("--db") + 1] = args.test_suite_db
        run(cmd, env=env)
    else:
        per_item = out_dir / f"{pred_tsv.stem}.per_item.jsonl"
        if per_item.exists():
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [args.python, str(ROOT / "scripts" / "evaluate_wikisql.py"),
               "--gold", args.wikisql_gold or args.questions, "--table", args.tables,
               "--db", args.db, "--predict", str(pred_tsv), "--etype", "all",
               "--output", str(out_dir / f"{pred_tsv.stem}.json"),
               "--per_item_out", str(per_item)]
        if args.strict:
            cmd.append("--strict")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run(cmd, env=env)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["spider", "wikisql"])
    ap.add_argument("--questions", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out_dir", default="results/ablation")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", default=["none"],
                    help="'none' = original order; integers = shuffled orders (RB configs only)")
    ap.add_argument("--protocol", choices=["online", "static"], default="online")
    ap.add_argument("--warmup_questions", default="data/spider/train_spider.json")
    ap.add_argument("--warmup_n", type=int, default=1000)
    ap.add_argument("--warmup_seed", type=int, default=0)
    ap.add_argument("--prompt_examples", choices=["train", "legacy_dev"], default="train",
                    help="Spider prompt header variant (legacy_dev reproduces the submitted paper)")
    ap.add_argument("--reasoning_config", default="./configs/reasoning_config.yaml")
    ap.add_argument("--chromadb_persist_dir", default="./data/embeddings/chroma_db")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--official_dir", default=None, help="test-suite-sql-eval clone (Spider)")
    ap.add_argument("--test_suite_db", default=None, help="test-suite database dir (Spider)")
    ap.add_argument("--wikisql_gold", default=None, help="original WikiSQL dev.json")
    ap.add_argument("--strict", action="store_true", help="WikiSQL strict scoring")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    env = dict(os.environ, SPIDER_PROMPT_EXAMPLES=args.prompt_examples)
    base = Path(args.out_dir) / args.dataset / args.protocol
    n_expected = len(json.load(open(args.questions, encoding="utf-8")))
    if args.limit:
        n_expected = min(n_expected, args.limit)

    plan = {"dataset": args.dataset, "protocol": args.protocol, "configs": args.configs,
            "seeds": args.seeds, "prompt_examples": args.prompt_examples,
            "questions": args.questions, "n": n_expected,
            "questions_sha256": ra.sha256_file(args.questions),
            "git": ra.git_info(str(ROOT))}
    base.mkdir(parents=True, exist_ok=True)
    ra.write_json(base / "ablation_plan.json", plan)

    produced = []
    for cfg in args.configs:
        cdir = base / cfg
        cdir.mkdir(parents=True, exist_ok=True)
        seeds = args.seeds if uses_rb(cfg) else ["none"]
        for seed in seeds:
            tag = f"seed-{seed}"
            out_tsv = cdir / f"{tag}.tsv"
            memory_dir = None
            frozen = False
            if uses_rb(cfg):
                memory_dir = cdir / f"memory_{tag}"
                if args.protocol == "static":
                    # 1) build memory on a train sample (online), 2) evaluate frozen
                    warm_q = make_warmup_file(args.warmup_questions, args.warmup_n,
                                              args.warmup_seed, base / "warmup_questions.json")
                    warm_tsv = cdir / f"warmup_{tag}.tsv"
                    generate(args, cfg, seed, warm_tsv, args.warmup_n, env,
                             memory_dir=memory_dir, frozen=False, questions=str(warm_q),
                             run_name=f"{args.dataset}/{cfg}/warmup/{tag}")
                    frozen = True
            generate(args, cfg, seed, out_tsv, n_expected, env, memory_dir=memory_dir, frozen=frozen)
            produced.append(out_tsv)
            mv = ra.sidecar(str(out_tsv), ".majority_vote.tsv")
            if mv.exists():
                produced.append(mv)

    if not args.no_eval:
        for tsv in produced:
            evaluate(args, tsv, env)
    print(f"\n✓ Ablation finished: {len(produced)} prediction files under {base}")
    print(f"  Next: python scripts/analyze_ablation.py --ablation_dir {base}")


if __name__ == "__main__":
    main()