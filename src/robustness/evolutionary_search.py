"""
evolutionary_search.py
=======================
Phase 1: GA loop that evolves a NoiseProfile maximizing

    (noisy_error_rate - clean_error_rate) - lambda_realism * realism_penalty

on the baseline SQLGenerator, over a fixed sample of Spider dev questions.
Saves the winning profile + full per-generation history to --output_dir.

Checkpointing (two levels):
  1. Generation-level: after EVERY generation fully completes, the current
     population, RNG state and best-so-far are written to
     <output_dir>/ea_checkpoint.json. Re-run with --resume to continue from
     the last completed generation.
  2. Individual-level (within an in-progress generation): each individual's
     per-question scoring is itself checkpointed to
     <output_dir>/gen{g}_indiv{i}_checkpoint.json (see fitness.py). This
     matters because generation-level checkpointing alone loses ALL work
     done on a generation that gets interrupted before it fully finishes —
     with population_size=10, that's up to 10x the cost of one individual.
     Individual-level checkpoints are safe to reuse across restarts because
     the GA is deterministic under a fixed --seed: a generation that hasn't
     been checkpointed yet is always reconstructed identically (same RNG
     trajectory, same population), so a given (generation, individual)
     index always corresponds to the same NoiseProfile and same noisy
     questions.

Usage:
    python -m src.robustness.evolutionary_search \
        --questions data/spider/dev.json \
        --db data/spider/database \
        --output_dir output/robustness/ea_search \
        --sample_size 15 --population_size 10 --generations 5 --seed 42

Resume after an interruption (works whether the interruption happened
mid-generation or between generations):
    (same command) --resume
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Optional

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


def _rng_state_to_json(state):
    if isinstance(state, tuple):
        return [_rng_state_to_json(x) for x in state]
    return state


def _rng_state_from_json(obj):
    version, internalstate, gauss_next = obj
    return (version, tuple(internalstate), gauss_next)


def _save_checkpoint(
    checkpoint_path: Path,
    generation_done: int,
    population: List[NoiseProfile],
    rng: random.Random,
    history: list,
    best_profile: Optional[NoiseProfile],
    best_fitness: float,
    clean_error_rate: float,
) -> None:
    payload = {
        "generation_done": generation_done,
        "population": [p.to_dict() for p in population],
        "rng_state": _rng_state_to_json(rng.getstate()),
        "history": history,
        "best_profile": best_profile.to_dict() if best_profile else None,
        "best_fitness": best_fitness,
        "clean_error_rate": clean_error_rate,
    }
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(checkpoint_path)


def _load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_individual_checkpoints(out_dir: Path, generation: int, population_size: int) -> None:
    """Remove per-individual checkpoint files for a generation that just
    fully completed — they're no longer needed once ea_checkpoint.json
    covers this generation, and leaving them around risks confusion if
    population composition ever changes for the same generation number."""
    for idx in range(1, population_size + 1):
        p = out_dir / f"gen{generation}_indiv{idx}_checkpoint.json"
        if p.exists():
            p.unlink()


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
    resume: bool = False,
) -> NoiseProfile:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "ea_checkpoint.json"

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

    n_elite = max(1, int(population_size * elite_frac))
    checkpoint = _load_checkpoint(checkpoint_path) if resume else None

    if checkpoint:
        print(f"↻ Resuming from checkpoint: {checkpoint['generation_done']}/{generations} "
              f"generations already completed")
        rng = random.Random()
        rng.setstate(_rng_state_from_json(checkpoint["rng_state"]))
        population = [NoiseProfile.from_dict(p) for p in checkpoint["population"]]
        history = checkpoint["history"]
        best_profile = (
            NoiseProfile.from_dict(checkpoint["best_profile"])
            if checkpoint["best_profile"] else None
        )
        best_fitness = checkpoint["best_fitness"]
        clean_error_rate = checkpoint["clean_error_rate"]
        start_gen = checkpoint["generation_done"]
        if start_gen >= generations:
            print("✓ Checkpoint already covers the requested number of generations — "
                  "nothing to resume, using saved winner.")
            _write_final_outputs(out_dir, best_profile, best_fitness, history,
                                  clean_error_rate, sample_size, seed)
            return best_profile
    else:
        if resume:
            print("No completed-generation checkpoint found — re-deriving generation 1 "
                  "deterministically from --seed (any per-individual progress already "
                  "made this generation will be picked up automatically).")
        rng = random.Random(seed)
        clean_error_rate = compute_clean_error_rate(
            samples, sql_generator,
            checkpoint_path=str(out_dir / "clean_error_checkpoint.json"),
        )
        print(f"Baseline clean error rate: {clean_error_rate:.2%}")
        population = [NoiseProfile.random_init(rng) for _ in range(population_size)]
        history = []
        best_profile = None
        best_fitness = float("-inf")
        start_gen = 0

    for gen in range(start_gen, generations):
        t0 = time.time()
        gen_results = [
            evaluate_fitness(
                profile, samples, sql_generator, clean_error_rate, rng,
                lambda_realism=lambda_realism,
                desc=f"Gen {gen + 1}/{generations} indiv {idx + 1}/{len(population)}",
                checkpoint_path=str(out_dir / f"gen{gen + 1}_indiv{idx + 1}_checkpoint.json"),
            )
            for idx, profile in enumerate(population)
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

        elites = [population[i] for i in ranked[:n_elite]]
        next_population = list(elites)
        while len(next_population) < population_size:
            parent_a = _tournament_select(population, scores, tournament_k, rng)
            parent_b = _tournament_select(population, scores, tournament_k, rng)
            child = parent_a.crossover(parent_b, rng)
            child = child.mutate(rng, sigma=mutation_sigma)
            next_population.append(child)
        population = next_population

        # ---- generation fully done: checkpoint it, then clean up the
        # now-redundant per-individual files for this generation ----
        _save_checkpoint(
            checkpoint_path, gen + 1, population, rng, history,
            best_profile, best_fitness, clean_error_rate,
        )
        _cleanup_individual_checkpoints(out_dir, gen + 1, len(population))

    _write_final_outputs(out_dir, best_profile, best_fitness, history,
                          clean_error_rate, sample_size, seed)
    return best_profile


def _write_final_outputs(out_dir, best_profile, best_fitness, history,
                          clean_error_rate, sample_size, seed):
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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from ea_checkpoint.json / per-individual checkpoints in --output_dir if present")
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
        resume=args.resume,
    )


if __name__ == "__main__":
    main()