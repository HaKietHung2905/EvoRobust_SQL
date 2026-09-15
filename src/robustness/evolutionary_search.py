"""
evolutionary_search.py
=======================
Phase 1: GA loop that evolves a NoiseProfile maximizing

    (noisy_error_rate - clean_error_rate) - lambda_realism * realism_penalty

on the baseline SQLGenerator, over a fixed sample of Spider dev questions.
Saves the winning profile + full per-generation history to --output_dir.

Usage:
    python -m src.robustness.evolutionary_search \
        --questions data/spider/dev.json \
        --db data/spider/database \
        --output_dir output/robustness/ea_search \
        --sample_size 15 --population_size 10 --generations 5 --seed 42
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import List

from src.generation.sql_generator import SQLGenerator
from .genome import NoiseProfile
from .fitness import (
    load_fitness_samples,
    compute_clean_error_rate,
    evaluate_fitness,
)


def _tournament_select(
    pop: List[NoiseProfile], scores: List[float], k: int, rng: random.Random
) -> NoiseProfile:
    contenders = rng.sample(range(len(pop)), k)
    best_idx = max(contenders, key=lambda i: scores[i])
    return pop[best_idx]


def run_ea(
    questions_path: str,
    db_dir: str,
    output_dir: str,
    sample_size: int = 15,
    population_size: int = 10,
    generations: int = 5,
    elite_frac: float = 0.2,
    tournament_k: int = 3,
    mutation_sigma: float = 0.15,
    lambda_realism: float = 0.5,
    seed: int = 42,
) -> NoiseProfile:
    rng = random.Random(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 1: EA NOISE PROFILE SEARCH")
    print("=" * 70)

    sql_generator = SQLGenerator()
    samples = load_fitness_samples(questions_path, db_dir, sample_size, seed=seed)
    if not samples:
        raise RuntimeError(
            "No usable fitness samples found — check --questions/--db paths."
        )
    print(f"Loaded {len(samples)} fitness-sample questions (seed={seed})")

    clean_error_rate = compute_clean_error_rate(samples, sql_generator)
    print(f"Baseline clean error rate: {clean_error_rate:.2%}")

    population = [NoiseProfile.random_init(rng) for _ in range(population_size)]
    n_elite = max(1, int(population_size * elite_frac))

    history = []
    best_profile = None
    best_fitness = float("-inf")

    for gen in range(generations):
        t0 = time.time()
        gen_results = [
            evaluate_fitness(
                profile, samples, sql_generator, clean_error_rate, rng,
                lambda_realism=lambda_realism,
            )
            for profile in population
        ]
        scores = [r["fitness"] for r in gen_results]
        ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)

        gen_best_idx = ranked[0]
        gen_best_fitness = scores[gen_best_idx]
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_profile = population[gen_best_idx]

        elapsed = time.time() - t0
        print(
            f"[gen {gen + 1}/{generations}] best_fitness={gen_best_fitness:.4f} "
            f"noisy_err={gen_results[gen_best_idx]['noisy_error_rate']:.2%} "
            f"penalty={gen_results[gen_best_idx]['realism_penalty']:.3f} "
            f"({elapsed:.1f}s)"
        )

        history.append({
            "generation": gen + 1,
            "best_fitness": gen_best_fitness,
            "mean_fitness": sum(scores) / len(scores),
            "best_profile": population[gen_best_idx].to_dict(),
            "best_result": gen_results[gen_best_idx],
        })

        # ---- next generation: elitism + tournament selection + crossover + mutation ----
        elites = [population[i] for i in ranked[:n_elite]]
        next_population = list(elites)
        while len(next_population) < population_size:
            parent_a = _tournament_select(population, scores, tournament_k, rng)
            parent_b = _tournament_select(population, scores, tournament_k, rng)
            child = parent_a.crossover(parent_b, rng)
            child = child.mutate(rng, sigma=mutation_sigma)
            next_population.append(child)
        population = next_population

    # ---- save outputs ----
    winner_path = out_dir / "winning_noise_profile.json"
    with open(winner_path, "w", encoding="utf-8") as f:
        json.dump({
            "profile": best_profile.to_dict(),
            "fitness": best_fitness,
            "clean_error_rate": clean_error_rate,
            "sample_size": sample_size,
            "seed": seed,
        }, f, indent=2)

    history_path = out_dir / "ea_search_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"✓ Winning NoiseProfile saved → {winner_path}")
    print(f"✓ Search history saved      → {history_path}")
    print(f"  Best fitness: {best_fitness:.4f}")
    print("=" * 70)

    return best_profile


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: EA search for adversarial NoiseProfile"
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output_dir", default="output/robustness/ea_search")
    parser.add_argument("--sample_size", type=int, default=15)
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--elite_frac", type=float, default=0.2)
    parser.add_argument("--tournament_k", type=int, default=3)
    parser.add_argument("--mutation_sigma", type=float, default=0.15)
    parser.add_argument("--lambda_realism", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_ea(
        questions_path=args.questions,
        db_dir=args.db,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        population_size=args.population_size,
        generations=args.generations,
        elite_frac=args.elite_frac,
        tournament_k=args.tournament_k,
        mutation_sigma=args.mutation_sigma,
        lambda_realism=args.lambda_realism,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()