#!/bin/bash
set -e

# Enable parallel asynchronous expert loads from the NVMe SSD
export HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0

TIME=$(date +%Y%m%d_%H%M%S)
OUT_DIR="py/utils/final_results_runs/sec5_verify_fixed/large_cache_sweep_${TIME}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Starting Large Cache Sweet Spot Sweep"
echo "Output Directory: $OUT_DIR"
echo "Parallel IO: Enabled"
echo "Expert Format: Packed"
echo "============================================================"

source utils/setup.sh

python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --dataset wikitext \
  --cache-sizes 56 64 \
  --lookaheads 2 4 6 \
  --cache-lookahead-slack 15 \
  --budget-fractions 0.25 0.5 0.75 1.0 \
  --routing-bias-top-n 5 \
  --sweep-question custom_1_16_no_ppl \
  --lambdas 0 1 \
  --cache-cond-forced-top-ns 5 \
  --predictor-base-dir trainingData/qwen3_30b/transformer_final_pfill_markov_emb \
  --expert-weights-dir py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed \
  --num-prompts 1 \
  --prompt-max-chars 4096 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --disable-measurement \
  --non-baseline-cache-policy LFRU \
  --drop-page-cache-between-runs \
  --out-dir "$OUT_DIR" \
  --csv-file "$OUT_DIR/sweep.csv"

echo "Sweep completed! Results saved to $OUT_DIR/sweep.csv"
