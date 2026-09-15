"""
genome.py
=========
NoiseProfile chromosome for the EA-driven robustness search.

Encodes:
    intensity         - fraction of words in a question that get perturbed (0.05-0.6)
    operator_weights   - relative weight per noise operator (9 ops total, incl.
                          the two sentence-level ones: space_merge, space_split)
    ops_per_word       - max number of operators stacked on a single perturbed
                          word in one pass (1-3)

Provides encode/decode (to_dict/from_dict), random initialization, mutation,
crossover, and application of a profile to raw question text.

space_split reuses the word-level function already defined in
noise_operators.py. space_merge has no word-level equivalent (it needs to see
the *next* word), so it's implemented here as a post-pass over the tokenized
sentence — this matches the note left in noise_operators.py's docstring.
"""
import random
from dataclasses import dataclass
from typing import Dict, List

from .noise_operators import (
    ALL_OPERATOR_NAMES,
    WORD_LEVEL_OPERATORS,
    SPACE_MERGE,
    SPACE_SPLIT,
    space_split,
)

INTENSITY_MIN, INTENSITY_MAX = 0.05, 0.6
OPS_PER_WORD_MIN, OPS_PER_WORD_MAX = 1, 3


@dataclass
class NoiseProfile:
    intensity: float
    operator_weights: Dict[str, float]
    ops_per_word: int

    # ---------------------------------------------------------------- #
    # Encode / decode
    # ---------------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "intensity": self.intensity,
            "operator_weights": dict(self.operator_weights),
            "ops_per_word": self.ops_per_word,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseProfile":
        return cls(
            intensity=float(d["intensity"]),
            operator_weights={k: float(v) for k, v in d["operator_weights"].items()},
            ops_per_word=int(d["ops_per_word"]),
        )

    # ---------------------------------------------------------------- #
    # Initialization
    # ---------------------------------------------------------------- #
    @classmethod
    def random_init(cls, rng: random.Random) -> "NoiseProfile":
        weights = {op: rng.random() for op in ALL_OPERATOR_NAMES}
        weights = _normalize(weights)
        return cls(
            intensity=rng.uniform(INTENSITY_MIN, INTENSITY_MAX),
            operator_weights=weights,
            ops_per_word=rng.randint(OPS_PER_WORD_MIN, OPS_PER_WORD_MAX),
        )

    # ---------------------------------------------------------------- #
    # GA operators
    # ---------------------------------------------------------------- #
    def mutate(self, rng: random.Random, sigma: float = 0.15) -> "NoiseProfile":
        """Gaussian perturbation of intensity + operator weights, renormalized;
        ops_per_word occasionally nudged by +/-1."""
        new_intensity = _clip(
            self.intensity + rng.gauss(0, sigma * (INTENSITY_MAX - INTENSITY_MIN)),
            INTENSITY_MIN, INTENSITY_MAX,
        )

        new_weights = {}
        for op, w in self.operator_weights.items():
            nw = w + rng.gauss(0, sigma)
            new_weights[op] = max(0.0, nw)
        new_weights = _normalize(new_weights)

        new_ops_per_word = self.ops_per_word
        if rng.random() < 0.3:
            delta = rng.choice([-1, 1])
            new_ops_per_word = min(
                OPS_PER_WORD_MAX, max(OPS_PER_WORD_MIN, self.ops_per_word + delta)
            )

        return NoiseProfile(new_intensity, new_weights, new_ops_per_word)

    def crossover(self, other: "NoiseProfile", rng: random.Random) -> "NoiseProfile":
        """Blend intensity (arithmetic), uniform per-gene crossover on
        operator weights, and a coin-flip pick on ops_per_word."""
        alpha = rng.random()
        child_intensity = alpha * self.intensity + (1 - alpha) * other.intensity

        child_weights = {}
        for op in ALL_OPERATOR_NAMES:
            a = self.operator_weights.get(op, 0.0)
            b = other.operator_weights.get(op, 0.0)
            child_weights[op] = rng.choice([a, b])
        child_weights = _normalize(child_weights)

        child_ops_per_word = rng.choice([self.ops_per_word, other.ops_per_word])

        return NoiseProfile(child_intensity, child_weights, child_ops_per_word)

    # ---------------------------------------------------------------- #
    # Application
    # ---------------------------------------------------------------- #
    def apply_to_text(self, text: str, rng: random.Random) -> str:
        return apply_profile_to_text(self, text, rng)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _weighted_choice(weights: Dict[str, float], rng: random.Random) -> str:
    ops = list(weights.keys())
    ws = [weights[o] for o in ops]
    return rng.choices(ops, weights=ws, k=1)[0]


def apply_profile_to_text(profile: "NoiseProfile", text: str, rng: random.Random) -> str:
    """
    Tokenize on whitespace. For each word, with probability `intensity`,
    apply up to `ops_per_word` operators sampled by `operator_weights`.

    space_merge is flagged per-word and resolved in a second pass that joins
    a perturbed word with its right neighbor (so index bookkeeping stays
    correct even when multiple merges happen in one sentence).
    space_split calls the word-level function from noise_operators.py.
    """
    words = text.split(" ")
    if not words:
        return text

    out_words = list(words)
    merge_next = [False] * len(words)

    for i, word in enumerate(words):
        if not word or rng.random() >= profile.intensity:
            continue

        n_ops = rng.randint(1, profile.ops_per_word)
        current = word
        for _ in range(n_ops):
            op_name = _weighted_choice(profile.operator_weights, rng)

            if op_name == SPACE_MERGE:
                if i < len(words) - 1:
                    merge_next[i] = True
                # no per-character change on this word itself
                continue
            elif op_name == SPACE_SPLIT:
                current = space_split(current, rng)
                continue
            else:
                fn = WORD_LEVEL_OPERATORS.get(op_name)
                if fn:
                    current = fn(current, rng)

        out_words[i] = current

    # Resolve merges left-to-right; skip the word that got absorbed.
    result_words: List[str] = []
    skip_next = False
    for i, w in enumerate(out_words):
        if skip_next:
            skip_next = False
            continue
        if merge_next[i] and i < len(out_words) - 1:
            result_words.append(w + out_words[i + 1])
            skip_next = True
        else:
            result_words.append(w)

    return " ".join(result_words)