#!/usr/bin/env python3
"""
debug_reasoning_bank.py
=========================
One-off debug wrapper: runs generate_predictions.py's normal main() flow,
but forces DEBUG-level logging on the ReasoningBank modules BEFORE main()
sets up its own logging config, so any init failure or per-question
"ReasoningBank failed: ..., falling back" message (normally logged at
`debug` level and invisible under the script's default WARNING config)
gets printed.

Does not modify generate_predictions.py — run this instead, with the same
CLI args, capped to a small --limit so you get the error fast.

Usage:
    python scripts/debug_reasoning_bank.py \
        --questions data/raw/spider/dev.json \
        --db        data/raw/spider/database \
        --output    results/debug_predictions.tsv \
        --use_reasoning_bank --use_chromadb --use_semantic \
        --limit 3
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import generate_predictions as gp  # noqa: E402

_original_main = gp.main


def main():
    # Let generate_predictions.py do its own basicConfig(WARNING) first...
    result = None
    try:
        # Patch logging AFTER gp module import (its basicConfig already ran
        # at import time via the module-level logging setup at the top of
        # generate_predictions.py), but BEFORE any pipeline objects are
        # constructed inside main(). Force DEBUG on the specific loggers
        # that generate_predictions.py otherwise silences to ERROR/WARNING.
        for name in [
            "src.reasoning.reasoning_pipeline",
            "src.reasoning.memory_retrieval",
            "src.reasoning.memory_store",
            "__main__",
        ]:
            logging.getLogger(name).setLevel(logging.DEBUG)

        # Also raise the root handler level so DEBUG records actually print
        # (basicConfig set stream=sys.stdout already; just lower the bar).
        for h in logging.root.handlers:
            h.setLevel(logging.DEBUG)

        _original_main()
    except SystemExit as e:
        result = e.code
    return result


if __name__ == "__main__":
    sys.exit(main() or 0)