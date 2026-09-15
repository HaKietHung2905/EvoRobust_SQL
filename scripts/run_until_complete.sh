#!/usr/bin/env bash
# run_until_complete.sh
# =======================
# Repeatedly runs generate_predictions.py with --resume until the output
# TSV has as many lines as the questions JSON has items. Use this instead
# of babysitting a single long-running generate call — if it dies from a
# network drop, rate-limit exhaustion, or you close the terminal, just
# re-run this script and it picks up where it left off.
#
# IMPORTANT: only safe to resume across runs of the SAME sql_generator.py
# (or any other prompt/logic file) version. If you've edited the generation
# code since the output file was last touched, delete it first:
#   rm -f results/predictions_spider_full.tsv
# otherwise you'll silently mix predictions from two different prompt
# versions in one file.
#
# Usage:
#   chmod +x scripts/run_until_complete.sh
#   ./scripts/run_until_complete.sh \
#       data/raw/spider/dev.json \
#       data/raw/spider/database \
#       results/predictions_spider_full.tsv \
#       "--use_reasoning_bank --use_chromadb --use_semantic"
#
# Args:
#   $1 = questions JSON path
#   $2 = database directory
#   $3 = output TSV path
#   $4 = extra flags for generate_predictions.py (quoted as one string)

set -uo pipefail

QUESTIONS="${1:?questions JSON path required}"
DB_DIR="${2:?database directory required}"
OUTPUT="${3:?output TSV path required}"
EXTRA_FLAGS="${4:-}"

CHECKPOINT_SIZE="${CHECKPOINT_SIZE:-200}"   # override via env var if desired
MAX_ATTEMPTS="${MAX_ATTEMPTS:-100}"         # safety cap on retry loop
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"         # seconds between attempts

TARGET=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$QUESTIONS")
echo "Target: $TARGET questions from $QUESTIONS"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))

    if [ -f "$OUTPUT" ]; then
        DONE=$(grep -c . "$OUTPUT" 2>/dev/null || echo 0)
    else
        DONE=0
    fi

    if [ "$DONE" -ge "$TARGET" ]; then
        echo "✓ Complete: $DONE/$TARGET lines in $OUTPUT"
        exit 0
    fi

    echo ""
    echo "=== Attempt $attempt: $DONE/$TARGET done, resuming ==="
    RESUME_FLAG=""
    if [ "$DONE" -gt 0 ]; then
        RESUME_FLAG="--resume"
    fi

    # shellcheck disable=SC2086
    python scripts/generate_predictions.py \
        --questions "$QUESTIONS" \
        --db        "$DB_DIR" \
        --output    "$OUTPUT" \
        --checkpoint_size "$CHECKPOINT_SIZE" \
        $RESUME_FLAG \
        $EXTRA_FLAGS

    EXIT_CODE=$?
    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "⚠ generate_predictions.py exited with code $EXIT_CODE — retrying in ${SLEEP_BETWEEN}s"
    fi
    sleep "$SLEEP_BETWEEN"
done

echo "✗ Gave up after $MAX_ATTEMPTS attempts — check logs above for a persistent error."
exit 1