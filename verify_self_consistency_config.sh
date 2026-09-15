#!/usr/bin/env bash
# ============================================================================
# verify_self_consistency_config.sh
#
# Runs a TINY real generate_predictions.py job (a handful of questions) with
# --use_reasoning_bank, and checks that the "SELF-CONSISTENCY / TEMPERATURE
# CONFIG (checked at startup)" banner it prints actually matches the values
# in configs/reasoning_config.yaml — instead of always showing the hardcoded
# defaults (4 / 0.6 / 0.85 / True), which is what the pre-fix bug produced.
#
# Run from the repo root:
#   bash verify_self_consistency_config.sh
#
# Optional overrides:
#   QUESTIONS=data/spider/dev.json DB=data/spider/database \
#     bash verify_self_consistency_config.sh
# ============================================================================

set -euo pipefail

QUESTIONS="${QUESTIONS:-data/raw/wikisql/dev_wiki_format.json}"
DB="${DB:-data/raw/wikisql/database}"
REASONING_CONFIG="${REASONING_CONFIG:-./configs/reasoning_config.yaml}"
OUTPUT="./results/_verify_self_consistency_config.tsv"
LOG="./results/_verify_self_consistency_config.log"
LIMIT="${LIMIT:-5}"

mkdir -p ./results

echo "============================================================"
echo "Step 1/3 — reading expected values from ${REASONING_CONFIG}"
echo "============================================================"

EXPECTED=$(python3 - "$REASONING_CONFIG" <<'PYEOF'
import sys, yaml
path = sys.argv[1]
with open(path) as f:
    cfg = yaml.safe_load(f) or {}
exp = cfg.get('experimental', {}) or {}
print(exp.get('enable_self_consistency_judging', True))
print(exp.get('self_consistency_n_candidates', 4))
print(exp.get('self_consistency_agreement_threshold', 0.6))
print(exp.get('self_consistency_max_confidence', 0.85))
PYEOF
)

EXP_ENABLED=$(echo "$EXPECTED" | sed -n '1p')
EXP_N=$(echo "$EXPECTED" | sed -n '2p')
EXP_THR=$(echo "$EXPECTED" | sed -n '3p')
EXP_CONF=$(echo "$EXPECTED" | sed -n '4p')

echo "  enable_self_consistency_judging : ${EXP_ENABLED}"
echo "  self_consistency_n_candidates   : ${EXP_N}"
echo "  agreement_threshold             : ${EXP_THR}"
echo "  max_confidence                  : ${EXP_CONF}"
echo

echo "============================================================"
echo "Step 2/3 — running generate_predictions.py --limit ${LIMIT}"
echo "============================================================"

python3 scripts/generate_predictions.py \
    --questions "$QUESTIONS" \
    --db "$DB" \
    --output "$OUTPUT" \
    --use_reasoning_bank \
    --reasoning_config "$REASONING_CONFIG" \
    --limit "$LIMIT" \
    2>&1 | tee "$LOG"

echo
echo "============================================================"
echo "Step 3/3 — comparing printed startup banner to YAML"
echo "============================================================"

ACTUAL_ENABLED=$(grep -m1 "enable_self_consistency_judging :" "$LOG" | awk -F': ' '{print $2}' | xargs)
ACTUAL_N=$(grep -m1 "n_additional_candidates" "$LOG" | awk -F': ' '{print $2}' | xargs)
ACTUAL_THR=$(grep -m1 "agreement_threshold" "$LOG" | awk -F': ' '{print $2}' | xargs)
ACTUAL_CONF=$(grep -m1 "max_confidence cap" "$LOG" | awk -F': ' '{print $2}' | xargs)

if [ -z "$ACTUAL_ENABLED" ]; then
    echo "❌ Could not find the SELF-CONSISTENCY startup banner in the log."
    echo "   Check ${LOG} manually — ReasoningBank may have failed to init."
    exit 1
fi

echo "  Printed at runtime:"
echo "    enable_self_consistency_judging : ${ACTUAL_ENABLED}"
echo "    n_additional_candidates         : ${ACTUAL_N}"
echo "    agreement_threshold             : ${ACTUAL_THR}"
echo "    max_confidence cap              : ${ACTUAL_CONF}"
echo

PASS=true

check() {
    local label="$1" expected="$2" actual="$3"
    if [ "$expected" != "$actual" ]; then
        echo "❌ MISMATCH — ${label}: expected '${expected}' (from YAML), got '${actual}' (printed at runtime)"
        PASS=false
    else
        echo "✓ MATCH — ${label}: ${actual}"
    fi
}

check "enable_self_consistency_judging" "$EXP_ENABLED" "$ACTUAL_ENABLED"
check "self_consistency_n_candidates"   "$EXP_N"       "$ACTUAL_N"
check "agreement_threshold"             "$EXP_THR"     "$ACTUAL_THR"
check "max_confidence"                  "$EXP_CONF"    "$ACTUAL_CONF"

echo
if [ "$PASS" = true ]; then
    echo "✅ PASS — reasoning_config.yaml values are correctly reaching the pipeline."
    echo "   (If you're on the pre-fix code, this would show n_additional_candidates=4"
    echo "   even after changing the YAML — that mismatch is exactly the bug.)"
else
    echo "🔴 FAIL — one or more values printed at runtime do NOT match reasoning_config.yaml."
    echo "   This means the config-propagation fix in reasoning_pipeline.py is either"
    echo "   missing or not applied correctly. See ${LOG} for full output."
    exit 1
fi