#!/usr/bin/env bash
# Compare straight LRU (cached) vs LRU + cross-layer gating prefetch (B=2,4,6,8).
#
# Usage:
#   bash py/utils/run_lru_vs_gating_sweep.sh
#   CACHE_SIZE=16 NUM_PROMPTS=5 MAX_NEW_TOKENS=32 bash py/utils/run_lru_vs_gating_sweep.sh

set -euo pipefail
cd "$(dirname "$0")/../.."
source utils/setup.sh

CACHE_SIZE="${CACHE_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
OUT_DIR="${OUT_DIR:-py/utils/final_results_runs/lru_vs_gating}"
PROMPTS_TXT="${PROMPTS_TXT:-py/utils/prompts_cc_compare_bank.txt}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

echo "=== LRU vs LRU+Gating sweep ==="
echo "Cache size:       $CACHE_SIZE"
echo "Prefetch budgets: 2 4 6 8"
echo "Prompts:          $PROMPTS_TXT (first $NUM_PROMPTS)"
echo "Max new tokens:   $MAX_NEW_TOKENS"
echo "Output CSV:       $OUT_DIR/lru_vs_gating_${TS}.csv"
echo ""

python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question lru_vs_gating \
  --model qwen \
  --cache-sizes "$CACHE_SIZE" \
  --custom-explicit-prefetch-budgets \
  --prefetch-budgets 2 4 6 8 \
  --dataset txt \
  --prompts-txt "$PROMPTS_TXT" \
  --num-prompts "$NUM_PROMPTS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0.0 \
  --out-dir "$OUT_DIR" \
  --csv-file "$OUT_DIR/lru_vs_gating_${TS}.csv" \
  --log-file "$OUT_DIR/lru_vs_gating_${TS}.log"

echo ""
echo "Done. Results: $OUT_DIR/lru_vs_gating_${TS}.csv"
