#!/bin/bash
set -euo pipefail

# Parallel async expert loads (not sequential pread)
export HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0

TIME=$(date +%Y%m%d_%H%M%S)
OUT_DIR="py/utils/final_results_runs/sec5_verify_fixed/mixtral_sweetspot_sweep_${TIME}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Starting Mixtral Sweet Spot Sweep"
echo "Output Directory: $OUT_DIR"
echo "Parallel IO:     HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0"
echo "Expert format:   packed (mixtral-8x7b-v0.1-AWQ_packed)"
echo "Baselines:       LRU + RANDOM per cache size (--prefetch-only)"
echo "Prompts:         5 wikitext chunks"
echo "============================================================"

source utils/setup.sh

python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model mixtral \
  --dataset wikitext \
  --cache-sizes 3 4 5 6 \
  --lookaheads 2 4 6 \
  --custom-explicit-prefetch-budgets \
  --prefetch-budgets 2 3 4 5 \
  --routing-bias-top-n 5 \
  --sweep-question custom_1_16_no_ppl \
  --predictor-base-dir trainingData/mixtral_transformer \
  --expert-weights-dir py/unified_llm_w4a16/model_weights/mixtral-8x7b-v0.1-AWQ_packed \
  --num-prompts 5 \
  --prompt-max-chars 4096 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --disable-measurement \
  --non-baseline-cache-policy LFRU \
  --prefetch-only \
  --drop-page-cache-between-runs \
  --out-dir "$OUT_DIR" \
  --csv-file "$OUT_DIR/sweep.csv"

echo "Sweep completed! Results saved to $OUT_DIR/sweep.csv"
