#!/bin/bash
set -euo pipefail

# Enable parallel asynchronous expert loads from the NVMe SSD
export HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0

echo "============================================================"
echo "Starting Mixtral Quick Smoke Test"
echo "============================================================"

source utils/setup.sh

OUT_DIR="py/utils/final_results_runs/sec5_verify_fixed/smoke_test_mixtral"
mkdir -p "$OUT_DIR"

python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model mixtral \
  --dataset wikitext \
  --cache-sizes 4 \
  --lookaheads 2 \
  --custom-explicit-prefetch-budgets \
  --prefetch-budgets 2 \
  --routing-bias-top-n 5 \
  --sweep-question custom_1_16_no_ppl \
  --predictor-base-dir trainingData/mixtral_transformer \
  --expert-weights-dir py/unified_llm_w4a16/model_weights/mixtral-8x7b-v0.1-AWQ_packed \
  --num-prompts 1 \
  --prompt-max-chars 256 \
  --max-new-tokens 5 \
  --temperature 0.0 \
  --disable-measurement \
  --non-baseline-cache-policy LFRU \
  --prefetch-only \
  --drop-page-cache-between-runs \
  --out-dir "$OUT_DIR" \
  --csv-file "$OUT_DIR/sweep.csv"

echo "Smoke test completed successfully!"
