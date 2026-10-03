"""
fitness.py
==========
Scores a NoiseProfile by how much it degrades the baseline SQLGenerator's
execution accuracy relative to clean questions, minus a realism penalty that
discourages gibberish-level corruption (the GA could otherwise "win" by just
maximizing intensity/ops_per_word until questions are unrecognizable, which
would make the whole benchmark meaningless).

    fitness = (noisy_error_rate - clean_error_rate) - lambda_realism * realism_penalty

Uses src.evaluation.exec_evaluator.eval_exec_match directly (execution-match
only, no SQL parsing / hardness / foreign-key-map machinery) since this
function is called population_size * generations times and the full
evaluate_spider.py pipeline would be far too slow inside the GA loop.
Phase 3 (compare_configs.py) is what uses the full evaluate_spider.py.

Progress + checkpointing: BOTH compute_clean_error_rate (one big batch run
once, before the GA loop) AND evaluate_fitness (one batch run per
individual, per generation) accept an optional checkpoint_path and print a
tqdm progress bar. Each item's result is flushed to disk immediately, so an
interruption at ANY point — mid clean-baseline scoring, or mid-way through
scoring individual 3 of generation 2 — loses at most the single in-flight
model call, never the whole batch.

This per-individual checkpoint relies on the GA being deterministic under a
fixed --seed: evolutionary_search.py always reconstructs the exact same
population (and RNG trajectory) for a generation that hasn't finished yet,
so a given (generation, individual) index always corresponds to the same
NoiseProfile and the same noisy questions across restarts — the checkpoint
for that index is safe to reuse even after a full process restart.
"""
import difflib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.evaluation.exec_evaluator import eval_exec_match
from src.generation.sql_generator import SQLGenerator
from .genome import NoiseProfile

DEFAULT_LAMBDA_REALISM = 0.5

# Beyond this average "changed fraction" of a question's characters
# (1 - difflib.SequenceMatcher ratio), extra corruption is penalized instead
# of rewarded.
REALISM_CHANGE_CAP = 0.35


@dataclass
class FitnessSample:
    question: str
    gold_sql: str
    db_id: str
    db_path: str


def load_fitness_samples(
    questions_path: str,
    db_dir: str,
    sample_size: int,
    seed: int = 42,
) -> List[FitnessSample]:
    """Fixed-seed random sample of usable (question, gold_sql, db) triples,
    used as the cost-bounded fitness set for every individual/generation."""
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    usable = [
        item for item in data
        if item.get("question") and item.get("db_id")
        and (item.get("query") or item.get("sql"))
    ]
    rng.shuffle(usable)
    picked = usable[:sample_size]

    samples = []
    for item in picked:
        db_id = item["db_id"]
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            continue
        samples.append(FitnessSample(
            question=item["question"],
            gold_sql=item.get("query", item.get("sql")),
            db_id=db_id,
            db_path=db_path,
        ))
    return samples


def _load_error_checkpoint(checkpoint_path: Optional[str]) -> List[Optional[int]]:
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_error_checkpoint(checkpoint_path: Optional[str], matches: List[Optional[int]]) -> None:
    if not checkpoint_path:
        return
    tmp_path = Path(checkpoint_path).with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(matches, f)
    tmp_path.replace(checkpoint_path)  # atomic swap so a crash mid-write can't corrupt it


def _error_rate(
    questions: List[str],
    samples: List[FitnessSample],
    sql_generator: SQLGenerator,
    desc: str = "Scoring",
    checkpoint_path: Optional[str] = None,
) -> float:
    total = len(samples)
    matches: List[Optional[int]] = _load_error_checkpoint(checkpoint_path)
    if len(matches) != total:
        matches = [None] * total  # stale/mismatched checkpoint -> start clean

    n_done = sum(1 for m in matches if m is not None)
    if n_done:
        tqdm.write(f"[{desc}] resuming from checkpoint: {n_done}/{total} already scored")

    pbar = tqdm(total=total, initial=n_done, desc=desc, unit="q")
    for i, (q, sample) in enumerate(zip(questions, samples)):
        if matches[i] is not None:
            continue
        try:
            pred_sql = sql_generator.generate(q, sample.db_path, temperature=0.0)
            match = eval_exec_match(
                db=sample.db_path,
                p_str=pred_sql,
                g_str=sample.gold_sql,
                plug_value=False,
                keep_distinct=False,
            )
        except Exception:
            match = 0
        matches[i] = int(bool(match))
        _save_error_checkpoint(checkpoint_path, matches)  # flush every item — safe to interrupt anytime
        pbar.update(1)
        errors_so_far = sum(1 for m in matches if m == 0 and m is not None)
        pbar.set_postfix(errors=errors_so_far)
    pbar.close()

    errors = sum(1 for m in matches if m == 0)
    return errors / total if total else 0.0


def compute_clean_error_rate(
    samples: List[FitnessSample],
    sql_generator: SQLGenerator,
    checkpoint_path: Optional[str] = None,
) -> float:
    """Baseline error rate on the UN-noised questions — computed once up
    front and reused across all generations."""
    return _error_rate(
        [s.question for s in samples], samples, sql_generator,
        desc="Clean baseline", checkpoint_path=checkpoint_path,
    )


def _realism_penalty(clean_questions: List[str], noisy_questions: List[str]) -> float:
    ratios = []
    for clean, noisy in zip(clean_questions, noisy_questions):
        sm = difflib.SequenceMatcher(None, clean, noisy)
        changed_fraction = 1.0 - sm.ratio()
        ratios.append(changed_fraction)
    avg_change = sum(ratios) / len(ratios) if ratios else 0.0
    return max(0.0, avg_change - REALISM_CHANGE_CAP)


def evaluate_fitness(
    profile: NoiseProfile,
    samples: List[FitnessSample],
    sql_generator: SQLGenerator,
    clean_error_rate: float,
    rng: random.Random,
    lambda_realism: float = DEFAULT_LAMBDA_REALISM,
    desc: str = "Individual",
    checkpoint_path: Optional[str] = None,
) -> Dict:
    clean_questions = [s.question for s in samples]
    # NOTE: apply_to_text runs for every sample regardless of checkpoint state,
    # so the RNG advances identically whether this individual is fresh or
    # partially resumed — this is what keeps the whole GA run deterministic
    # and checkpoint-safe across restarts.
    noisy_questions = [profile.apply_to_text(q, rng) for q in clean_questions]

    noisy_error_rate = _error_rate(
        noisy_questions, samples, sql_generator,
        desc=desc, checkpoint_path=checkpoint_path,
    )
    penalty = _realism_penalty(clean_questions, noisy_questions)

    fitness = (noisy_error_rate - clean_error_rate) - lambda_realism * penalty

    return {
        "fitness": fitness,
        "noisy_error_rate": noisy_error_rate,
        "clean_error_rate": clean_error_rate,
        "realism_penalty": penalty,
    }