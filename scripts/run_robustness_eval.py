#!/usr/bin/env python3
"""
run_robustness_eval.py
=======================
End-to-end runner for the three-phase robustness evaluation:
    Phase 1 (EA search)      -> winning_noise_profile.json
    Phase 2 (dataset build)  -> noisy_dev.json (fixed seed)
    Phase 3 (4-way ablation) -> comparison_report.md / .json

Designed to be interrupted and re-run with the SAME command at any point:
  - Phase 1: if winning_noise_profile.json already exists, reused as-is.
    Else if ea_checkpoint.json exists, EA search resumes from the last
    completed generation. Else it starts fresh.
  - Phase 2: if noisy_dev.json already exists, reused as-is.
  - Phase 3: compare_configs.py skips already-evaluated configs and resumes
    partially-generated ones on its own (see that file's docstring).

Pass --force_ea / --force_dataset_build to ignore existing outputs and
redo those phases from scratch.

Full run (Spider):
    python scripts/run_robustness_eval.py \
        --questions data/spider/dev.json \
        --db data/spider/database \
        --output_dir output/robustness

Full run (WikiSQL):
    python scripts/run_robustness_eval.py \
        --questions data/raw/wikisql/dev_wiki_format.json \
        --db data/raw/wikisql/database \
        --output_dir output/robustness_wikisql \
        --reasoning_config ./configs/reasoning_config_wikisql.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robustness.evolutionary_search import run_ea
from src.robustness.dataset_builder import build_noisy_dataset
from src.robustness.compare_configs import run_comparison


def main():
    parser = argparse.ArgumentParser(description="End-to-end robustness evaluation runner")
    parser.add_argument("--questions", default="data/spider/dev.json")
    parser.add_argument("--db", default="data/spider/database")
    parser.add_argument("--output_dir", default="output/robustness")

    # Phase 1 — EA search
    parser.add_argument("--skip_ea", action="store_true")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--force_ea", action="store_true",
                        help="Ignore existing winning_noise_profile.json / checkpoint, redo Phase 1")
    parser.add_argument("--sample_size", type=int, default=15)
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--elite_frac", type=float, default=0.2)
    parser.add_argument("--tournament_k", type=int, default=3)
    parser.add_argument("--mutation_sigma", type=float, default=0.15)
    parser.add_argument("--lambda_realism", type=float, default=0.5)
    parser.add_argument("--ea_seed", type=int, default=42)

    # Phase 2 — dataset build
    parser.add_argument("--skip_dataset_build", action="store_true")
    parser.add_argument("--noisy_questions", default=None)
    parser.add_argument("--force_dataset_build", action="store_true",
                        help="Ignore existing noisy_dev.json, rebuild Phase 2")
    parser.add_argument("--dataset_seed", type=int, default=123)

    # Phase 3 — 4-way ablation
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--chromadb_persist_dir", default="./data/embeddings/chroma_db")
    parser.add_argument("--reasoning_config", default="./configs/reasoning_config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", choices=["auto", "spider", "wikisql"], default="auto")
    parser.add_argument("--table", default=None)

    args = parser.parse_args()
    out_root = Path(args.output_dir)

    # ---- Phase 1 ----
    ea_dir = out_root / "ea_search"
    winner_path = ea_dir / "winning_noise_profile.json"
    checkpoint_path = ea_dir / "ea_checkpoint.json"

    if args.skip_ea:
        if not args.profile:
            parser.error("--skip_ea requires --profile")
        profile_path = args.profile
        print(f"Skipping Phase 1 — using existing profile: {profile_path}")
    elif winner_path.exists() and not args.force_ea:
        profile_path = str(winner_path)
        print(f"✓ Phase 1 already complete — reusing {profile_path}")
    else:
        should_resume = checkpoint_path.exists() and not args.force_ea
        if should_resume:
            print(f"↻ Found EA checkpoint — resuming Phase 1 from {checkpoint_path}")
        run_ea(
            questions_path=args.questions,
            db_dir=args.db,
            output_dir=str(ea_dir),
            sample_size=args.sample_size,
            population_size=args.population_size,
            generations=args.generations,
            elite_frac=args.elite_frac,
            tournament_k=args.tournament_k,
            mutation_sigma=args.mutation_sigma,
            lambda_realism=args.lambda_realism,
            seed=args.ea_seed,
            resume=should_resume,
        )
        profile_path = str(winner_path)

    # ---- Phase 2 ----
    noisy_path = out_root / "noisy_dev.json"
    if args.skip_dataset_build:
        if not args.noisy_questions:
            parser.error("--skip_dataset_build requires --noisy_questions")
        noisy_questions_path = args.noisy_questions
        print(f"Skipping Phase 2 — using existing noisy set: {noisy_questions_path}")
    elif noisy_path.exists() and not args.force_dataset_build:
        noisy_questions_path = str(noisy_path)
        print(f"✓ Phase 2 already complete — reusing {noisy_questions_path}")
    else:
        noisy_questions_path = build_noisy_dataset(
            questions_path=args.questions,
            profile_path=profile_path,
            output_path=str(noisy_path),
            seed=args.dataset_seed,
        )

    # ---- Phase 3 ----
    run_comparison(
        noisy_questions=noisy_questions_path,
        db_dir=args.db,
        output_dir=str(out_root / "comparison"),
        top_k=args.top_k,
        chromadb_persist_dir=args.chromadb_persist_dir,
        reasoning_config=args.reasoning_config,
        limit=args.limit,
        dataset=args.dataset,
        table_file=args.table,
    )


if __name__ == "__main__":
    main()