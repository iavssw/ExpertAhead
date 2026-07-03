#!/usr/bin/env bash
# Prompt comparison: cache-cond λ=1 at J=5, J=6, and J=0 — greedy T=0.
# Default: 15 prompts from the bank. Three configs → 6 model loads (WikiText + generate each).
#
# Usage:
#   sudo -E bash py/utils/demo_cc_j6_vs_strict_prompt.sh
#   CC_FORCED_TOP_NS="5 6 0" NUM_PROMPTS=15 bash py/utils/demo_cc_j6_vs_strict_prompt.sh
#
# Optional env: CACHE_SIZE, MAX_NEW_TOKENS, NUM_PROMPTS, CC_FORCED_TOP_NS,
#               OUT_DIR, PROMPTS_TXT, PYTHON

set -euo pipefail
cd "$(dirname "$0")/../.."
source utils/setup.sh

CACHE_SIZE="${CACHE_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-15}"
CC_FORCED_TOP_NS="${CC_FORCED_TOP_NS:-5 6 0}"
OUT_DIR="${OUT_DIR:-py/utils/final_results_runs/cc_prompt_compare}"
PROMPTS_TXT="${PROMPTS_TXT:-py/utils/prompts_cc_compare_bank.txt}"
PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

echo "=== CC λ=1 @ J=${CC_FORCED_TOP_NS} (T=0) ==="
echo "Cache size: $CACHE_SIZE"
echo "Prompts: $PROMPTS_TXT (first $NUM_PROMPTS)"
echo "Generations: $OUT_DIR/generations.md"
echo ""

"$PYTHON" py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question cc_prompt_compare \
  --model qwen \
  --cache-sizes "$CACHE_SIZE" \
  --cc-forced-top-ns $CC_FORCED_TOP_NS \
  --cc-temperatures 0 \
  --dataset txt \
  --prompts-txt "$PROMPTS_TXT" \
  --num-prompts "$NUM_PROMPTS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --out-dir "$OUT_DIR" \
  --csv-file "$OUT_DIR/cc_j5_j6_j0_${TS}.csv" \
  --log-file "$OUT_DIR/cc_j5_j6_j0_${TS}.log"

echo ""
echo "Done. Open $OUT_DIR/generations.md to compare responses."
