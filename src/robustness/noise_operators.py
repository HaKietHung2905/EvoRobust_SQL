"""
noise_operators.py
===================
Low-level, word-scoped typo generators. Each operator takes a single word
(str) plus a random.Random instance and returns a perturbed word. They are
deliberately "human-plausible" (adjacent-key slips, transpositions, doubled
letters, dropped letters) rather than random gibberish, since the goal is to
model the kind of noise real users actually produce, not adversarial garbage.

Two operators (SPACE_MERGE, SPACE_SPLIT) act at the sentence/token level and
are handled specially in `genome.apply_profile_to_text` — they are still
registered here so the GA can assign them weight like any other operator.
"""

import json
import random
import string
from pathlib import Path
from typing import Callable, Dict, List

# --- QWERTY adjacency map (lowercase). Used for keyboard-slip realism. ---
_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def _build_adjacency() -> Dict[str, str]:
    adj: Dict[str, list] = {c: [] for c in string.ascii_lowercase}
    for row_idx, row in enumerate(_ROWS):
        for i, c in enumerate(row):
            neighbors = []
            if i > 0:
                neighbors.append(row[i - 1])
            if i < len(row) - 1:
                neighbors.append(row[i + 1])
            # loose vertical adjacency to the row above/below at same index
            for other_row in (_ROWS[row_idx - 1] if row_idx > 0 else "",
                               _ROWS[row_idx + 1] if row_idx < len(_ROWS) - 1 else ""):
                if i < len(other_row):
                    neighbors.append(other_row[i])
            adj[c] = neighbors or [c]
    return adj


QWERTY_ADJACENT = _build_adjacency()

# ---------------------------------------------------------------------- #
# Common-misspelling corpus
# ---------------------------------------------------------------------- #
# Loaded from data/common_misspellings.json: {correct_word: [variant, ...]}.
# That file merges two sources:
#   1. The Wikipedia "Lists of common misspellings" corpus (~3200 correct
#      words, ~4500 variants) - broad, general-English coverage.
#   2. A small hand-curated set of SQL/Text-to-SQL-domain words (average,
#      count, customer, employee, ...) that the generic corpus doesn't
#      cover but that dominate this project's question vocabulary.
# If the data file is missing, falls back to a minimal built-in set so the
# module still works standalone.
_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "common_misspellings.json"

_FALLBACK_MISSPELLINGS: Dict[str, List[str]] = {
    "the": ["teh"],
    "receive": ["recieve"],
    "separate": ["seperate"],
    "definitely": ["definately", "definatly"],
    "which": ["wich"],
    "their": ["there", "thier"],
    "average": ["avrage", "avarage"],
    "number": ["numer"],
    "count": ["cuont"],
    "customer": ["customre"],
}


def _load_misspellings() -> Dict[str, List[str]]:
    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            return data
    except Exception:
        pass
    return _FALLBACK_MISSPELLINGS


COMMON_MISSPELLINGS: Dict[str, List[str]] = _load_misspellings()


def _reapply_case(original: str, transformed: str) -> str:
    if original.isupper():
        return transformed.upper()
    if original[:1].isupper():
        return transformed[:1].upper() + transformed[1:]
    return transformed


def keyboard_slip(word: str, rng: random.Random) -> str:
    """Replace one letter with an adjacent QWERTY key."""
    if len(word) < 1:
        return word
    idx = rng.randrange(len(word))
    ch = word[idx].lower()
    if ch not in QWERTY_ADJACENT:
        return word
    repl = rng.choice(QWERTY_ADJACENT[ch])
    out = word[:idx] + repl + word[idx + 1:]
    return _reapply_case(word, out)


def char_swap(word: str, rng: random.Random) -> str:
    """Transpose two adjacent characters (classic fat-finger typo)."""
    if len(word) < 2:
        return word
    idx = rng.randrange(len(word) - 1)
    chars = list(word)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def char_delete(word: str, rng: random.Random) -> str:
    """Drop a random character."""
    if len(word) < 2:
        return word
    idx = rng.randrange(len(word))
    return word[:idx] + word[idx + 1:]


def char_duplicate(word: str, rng: random.Random) -> str:
    """Double a random character (key held too long)."""
    if len(word) < 1:
        return word
    idx = rng.randrange(len(word))
    return word[:idx + 1] + word[idx] + word[idx + 1:]


def char_insert_random(word: str, rng: random.Random) -> str:
    """Insert a stray adjacent-key character."""
    if len(word) < 1:
        return word
    idx = rng.randrange(len(word))
    ch = word[idx].lower()
    ins = rng.choice(QWERTY_ADJACENT.get(ch, [ch]))
    return word[:idx] + ins + word[idx:]


def case_toggle(word: str, rng: random.Random) -> str:
    """Flip the case of a single character."""
    if len(word) < 1:
        return word
    idx = rng.randrange(len(word))
    ch = word[idx]
    ch = ch.lower() if ch.isupper() else ch.upper()
    return word[:idx] + ch + word[idx + 1:]


def common_misspelling(word: str, rng: random.Random) -> str:
    """
    Swap in a known common misspelling if one exists, else no-op.
    Picks randomly among the recorded real-world variants for that word
    (e.g. "definitely" -> one of "definately"/"definatly"/"definetly"/"definitly")
    rather than always producing the same fixed typo.
    """
    key = word.lower()
    variants = COMMON_MISSPELLINGS.get(key)
    if variants:
        return _reapply_case(word, rng.choice(variants))
    return word


def identity(word: str, rng: random.Random) -> str:
    """No-op operator; lets the GA learn to leave some words untouched."""
    return word


# Sentence-level operators are registered as markers; real logic lives in
# genome.apply_profile_to_text, which needs access to neighboring tokens.
SPACE_MERGE = "space_merge"
SPACE_SPLIT = "space_split"

WORD_LEVEL_OPERATORS: Dict[str, Callable[[str, random.Random], str]] = {
    "keyboard_slip": keyboard_slip,
    "char_swap": char_swap,
    "char_delete": char_delete,
    "char_duplicate": char_duplicate,
    "char_insert_random": char_insert_random,
    "case_toggle": case_toggle,
    "common_misspelling": common_misspelling,
}

# Full operator set the GA is allowed to weight, including sentence-level ones.
ALL_OPERATOR_NAMES = list(WORD_LEVEL_OPERATORS.keys()) + [SPACE_MERGE, SPACE_SPLIT]


def space_split(word: str, rng: random.Random) -> str:
    """Insert a stray space in the middle of a word (returns 'wo rd')."""
    if len(word) < 4:
        return word
    idx = rng.randrange(2, len(word) - 1)
    return word[:idx] + " " + word[idx:]
