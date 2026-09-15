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
"""
import difflib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

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


def _error_rate(
    questions: List[str],
    samples: List[FitnessSample],
    sql_generator: SQLGenerator,
) -> float:
    errors = 0
    total = 0
    for q, sample in zip(questions, samples):
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
        total += 1
        if not match:
            errors += 1
    return errors / total if total else 0.0


def compute_clean_error_rate(samples: List[FitnessSample], sql_generator: SQLGenerator) -> float:
    """Baseline error rate on the UN-noised questions — computed once up
    front and reused across all generations."""
    return _error_rate([s.question for s in samples], samples, sql_generator)


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
) -> Dict:
    clean_questions = [s.question for s in samples]
    noisy_questions = [profile.apply_to_text(q, rng) for q in clean_questions]

    noisy_error_rate = _error_rate(noisy_questions, samples, sql_generator)
    penalty = _realism_penalty(clean_questions, noisy_questions)

    fitness = (noisy_error_rate - clean_error_rate) - lambda_realism * penalty

    return {
        "fitness": fitness,
        "noisy_error_rate": noisy_error_rate,
        "clean_error_rate": clean_error_rate,
        "realism_penalty": penalty,
    }