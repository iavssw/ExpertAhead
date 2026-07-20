#!/usr/bin/env bash
# 2-prompt final collection with scaled cgroup memory caps (5 methods × C=16,32,48,64).
#
# Usage:
#   ./run_final_best_2prompt_memlimit.sh
#   CACHE_SIZE=32 ./run_final_best_2prompt_memlimit.sh
#   DRY_RUN=1 ./run_final_best_2prompt_memlimit.sh
#
# Compare vs canonical final_10prompt when done:
#   VALIDATE=1 ./run_final_best_2prompt_memlimit.sh
#
# Optional:
#   NUM_PROMPTS=2
#   MAX_NEW_TOKENS=150
#   CACHE_SIZES="32 48"
#   OUT_ROOT=py/utils/final_results_runs/final_2prompt_memlimit
#   MEMORY_MAX_BY_CACHE="16:6G,32:8G,48:10G,64:11G"
#   CANONICAL_SUFFIX=20260709_001606
#   RTOL=0.10

set -euo pipefail
cd "$(dirname "$0")"
source utils/setup.sh

NUM_PROMPTS="${NUM_PROMPTS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"
CACHE_SIZES="${CACHE_SIZES:-${CACHE_SIZE:-16 32 48 64}}"
BATCH_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-py/utils/final_results_runs/final_2prompt_memlimit/${BATCH_TS}}"
MEMORY_MAX_BY_CACHE="${MEMORY_MAX_BY_CACHE:-16:7G,32:9G,48:11G,64:12G}"

echo "=== 2-prompt memlimit collection ==="
echo "NUM_PROMPTS=$NUM_PROMPTS  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "CACHE_SIZES=$CACHE_SIZES"
echo "MEMORY_MAX_BY_CACHE=$MEMORY_MAX_BY_CACHE"
echo "OUT_ROOT=$OUT_ROOT"
echo

NUM_PROMPTS="$NUM_PROMPTS" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
CACHE_SIZES="$CACHE_SIZES" \
MEMORY_MAX_BY_CACHE="$MEMORY_MAX_BY_CACHE" \
OUT_ROOT="$OUT_ROOT" \
RUN_TS="$BATCH_TS" \
./run_final_best_10prompt_memlimit.sh

if [[ "${VALIDATE:-0}" == "1" ]]; then
  CANONICAL_SUFFIX="${CANONICAL_SUFFIX:-20260709_001606}"
  RTOL="${RTOL:-0.10}"
  COMPARE_CSV="$OUT_ROOT/validation_compare.csv"
  python3 py/utils/compare_memlimit_validation.py \
    --validate-root "$OUT_ROOT" \
    --canonical-suffix "$CANONICAL_SUFFIX" \
    --cache-sizes $CACHE_SIZES \
    --rtol "$RTOL" \
    --out-csv "$COMPARE_CSV"
  echo "Comparison CSV: $COMPARE_CSV"
fi

echo "Done. Results: $OUT_ROOT"
