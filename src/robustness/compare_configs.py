"""
compare_configs.py
====================
Phase 3: Run the four-way ablation (baseline / +Semantic RAG / +ReasoningBank
/ +both) against the FIXED noisy_dev.json produced by Phase 2, by shelling
out to the project's existing scripts/generate_predictions.py and
scripts/evaluate_spider.py or scripts/evaluate_wikisql.py — unmodified, per
project convention. Produces a markdown + JSON comparison report.

Resumable by design: a config already fully evaluated (eval_{name}.json
exists) is skipped entirely; a config with a partially-generated predictions
TSV is continued via generate_predictions.py's own --resume (line-buffered,
safe to interrupt anytime). This means re-running the exact same command
after any pause — minutes or many hours — picks up exactly where it left
off, without any extra flags.

Flag mapping (confirmed against generate_predictions.py / evaluate_spider.py):
    Semantic RAG   -> --use_chromadb (+ --top_k, --chromadb_persist_dir)
    ReasoningBank  -> --use_reasoning_bank (+ --reasoning_config)
    (--use_semantic is a separate rule-based Semantic Layer component, not
    part of this ablation, so it is intentionally left off in all 4 configs.)

Dataset routing: "wikisql" substring convention (same as generate_predictions.py)
routes to scripts/evaluate_wikisql.py (--table required); otherwise Spider.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

CONFIGS = [
    {"name": "baseline",       "flags": []},
    {"name": "semantic_rag",   "flags": ["--use_chromadb"]},
    {"name": "reasoning_bank", "flags": ["--use_reasoning_bank"]},
    {"name": "semantic_rag_and_reasoning_bank",
     "flags": ["--use_chromadb", "--use_reasoning_bank"]},
]


def _run(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def _detect_wikisql(noisy_questions: str, db_dir: str, dataset: str) -> bool:
    if dataset == "wikisql":
        return True
    if dataset == "spider":
        return False
    return "wikisql" in noisy_questions.lower() or "wikisql" in db_dir.lower()


def run_comparison(
    noisy_questions: str,
    db_dir: str,
    output_dir: str,
    project_root: str = ".",
    top_k: int = 3,
    chromadb_persist_dir: str = "./data/embeddings/chroma_db",
    reasoning_config: str = "./configs/reasoning_config.yaml",
    limit: Optional[int] = None,
    python_exe: str = sys.executable,
    dataset: str = "auto",
    table_file: Optional[str] = None,
) -> Dict:
    root = Path(project_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_wikisql = _detect_wikisql(noisy_questions, db_dir, dataset)

    generate_script = str(root / "scripts" / "generate_predictions.py")
    evaluate_script = str(
        root / "scripts" / ("evaluate_wikisql.py" if is_wikisql else "evaluate_spider.py")
    )

    if is_wikisql:
        resolved_table_file = table_file or str(Path(db_dir) / "tables.json")
        if not Path(resolved_table_file).exists():
            raise RuntimeError(
                f"WikiSQL detected but tables.json not found at {resolved_table_file}. "
                "Run Phase 1/2 first (they auto-build it), or pass --table explicitly."
            )
        print(f"Dataset detected: WikiSQL (table_file={resolved_table_file})")
    else:
        resolved_table_file = None
        print("Dataset detected: Spider")

    all_results = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        pred_path = out_dir / f"predictions_{name}.tsv"
        eval_path = out_dir / f"eval_{name}.json"

        print("\n" + "=" * 70)
        print(f"CONFIG: {name}  (flags: {cfg['flags'] or '[none]'})")
        print("=" * 70)

        if eval_path.exists():
            print(f"✓ Already evaluated → {eval_path} (skipping generation + eval)")
            with open(eval_path, "r", encoding="utf-8") as f:
                all_results[name] = json.load(f)
            continue

        gen_cmd = [
            python_exe, generate_script,
            "--questions", noisy_questions,
            "--db", db_dir,
            "--output", str(pred_path),
            *cfg["flags"],
        ]
        if "--use_chromadb" in cfg["flags"]:
            gen_cmd += ["--top_k", str(top_k), "--chromadb_persist_dir", chromadb_persist_dir]
        if "--use_reasoning_bank" in cfg["flags"]:
            gen_cmd += ["--reasoning_config", reasoning_config]
        if limit:
            gen_cmd += ["--limit", str(limit)]
        if pred_path.exists():
            # Safe whether the file is partial or already complete — generate_predictions.py
            # counts already-written lines and only generates what's missing.
            print(f"↻ Found partial predictions at {pred_path} — resuming generation")
            gen_cmd += ["--resume"]
        _run(gen_cmd)

        eval_cmd = [
            python_exe, evaluate_script,
            "--gold", noisy_questions,
            "--db", db_dir,
            "--predict", str(pred_path),
            "--etype", "all",
            "--output", str(eval_path),
        ]
        if is_wikisql:
            eval_cmd += ["--table", resolved_table_file]
        if limit:
            eval_cmd += ["--limit", str(limit)]
        _run(eval_cmd)

        with open(eval_path, "r", encoding="utf-8") as f:
            all_results[name] = json.load(f)

    report = _build_report(all_results)

    report_json_path = out_dir / "comparison_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_md_path = out_dir / "comparison_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(report))

    print("\n" + "=" * 70)
    print("PHASE 3: COMPARISON COMPLETE")
    print("=" * 70)
    print(f"✓ JSON report     → {report_json_path}")
    print(f"✓ Markdown report → {report_md_path}")
    print("=" * 70)

    return report


def _build_report(all_results: Dict[str, Dict]) -> Dict:
    baseline = all_results.get("baseline", {})
    baseline_exec_err = 1 - baseline.get("execution_accuracy", 0.0)

    rows = []
    for cfg in CONFIGS:
        name = cfg["name"]
        res = all_results.get(name, {})
        exec_acc = res.get("execution_accuracy", 0.0)
        exact_acc = res.get("exact_match_accuracy", 0.0)
        error_rate = 1 - exec_acc
        reduction = (
            (baseline_exec_err - error_rate) / baseline_exec_err
            if baseline_exec_err > 0 else 0.0
        )
        # total_evaluated may live at top level (older/simpler eval scripts)
        # or nested under scores["all"]["count"] (evaluate_spider.py /
        # evaluate_wikisql.py's actual output shape) — check both.
        total = res.get("total_evaluated")
        if total is None:
            total = res.get("scores", {}).get("all", {}).get("count", 0)
        rows.append({
            "config": name,
            "flags": cfg["flags"],
            "exact_match_accuracy": exact_acc,
            "execution_accuracy": exec_acc,
            "error_rate": error_rate,
            "error_rate_reduction_vs_baseline": reduction if name != "baseline" else 0.0,
            "total_evaluated": total,
        })

    return {"baseline_error_rate": baseline_exec_err, "configs": rows}

def _render_markdown(report: Dict) -> str:
    lines = [
        "# Robustness Ablation Report",
        "",
        "Four-way comparison on the fixed, EA-discovered adversarial noisy dev set.",
        "",
        f"Baseline error rate (1 - execution accuracy): **{report['baseline_error_rate']:.2%}**",
        "",
        "| Config | Exact Match | Execution Acc. | Error Rate | Error Reduction vs Baseline | N |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["configs"]:
        lines.append(
            f"| {row['config']} | {row['exact_match_accuracy']:.2%} | "
            f"{row['execution_accuracy']:.2%} | {row['error_rate']:.2%} | "
            f"{row['error_rate_reduction_vs_baseline']:.2%} | {row['total_evaluated']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Phase 3: 4-way robustness ablation")
    parser.add_argument("--noisy_questions", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output_dir", default="output/robustness/comparison")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--chromadb_persist_dir", default="./data/embeddings/chroma_db")
    parser.add_argument("--reasoning_config", default="./configs/reasoning_config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", choices=["auto", "spider", "wikisql"], default="auto")
    parser.add_argument("--table", default=None)
    args = parser.parse_args()

    run_comparison(
        noisy_questions=args.noisy_questions,
        db_dir=args.db,
        output_dir=args.output_dir,
        project_root=args.project_root,
        top_k=args.top_k,
        chromadb_persist_dir=args.chromadb_persist_dir,
        reasoning_config=args.reasoning_config,
        limit=args.limit,
        dataset=args.dataset,
        table_file=args.table,
    )


if __name__ == "__main__":
    main()