"""
dataset_builder.py
===================
Phase 2: Apply a winning NoiseProfile (from Phase 1) to the full Spider dev
set with a FIXED SEED, producing a reproducible noisy_dev.json. All four
pipeline configs in Phase 3 are evaluated against this SAME file — that's
what makes the four-way comparison a controlled, valid ablation rather than
four separately-noised runs.

Usage:
    python -m src.robustness.dataset_builder \
        --questions data/spider/dev.json \
        --profile output/robustness/ea_search/winning_noise_profile.json \
        --output output/robustness/noisy_dev.json \
        --seed 123
"""
import argparse
import json
import random
from pathlib import Path

from .genome import NoiseProfile


def build_noisy_dataset(
    questions_path: str,
    profile_path: str,
    output_path: str,
    seed: int = 123,
) -> str:
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(profile_path, "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    profile = NoiseProfile.from_dict(profile_data["profile"])

    rng = random.Random(seed)
    noisy_data = []
    n_changed = 0
    for item in data:
        new_item = dict(item)
        question = item.get("question", "")
        if question:
            noisy_question = profile.apply_to_text(question, rng)
            if noisy_question != question:
                n_changed += 1
            new_item["question"] = noisy_question
            new_item["original_question"] = question
        noisy_data.append(new_item)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(noisy_data, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("PHASE 2: NOISY DATASET GENERATION")
    print("=" * 70)
    print(f"Source questions : {questions_path} ({len(data)} items)")
    print(f"Noise profile    : {profile_path}")
    print(f"Fixed seed       : {seed}")
    print(f"Changed          : {n_changed}/{len(data)} questions")
    print(f"Output           : {out_path}")
    print("=" * 70)

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: build fixed-seed noisy Spider dev set"
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--profile", required=True,
                        help="winning_noise_profile.json from Phase 1")
    parser.add_argument("--output", default="output/robustness/noisy_dev.json")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    build_noisy_dataset(
        questions_path=args.questions,
        profile_path=args.profile,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()