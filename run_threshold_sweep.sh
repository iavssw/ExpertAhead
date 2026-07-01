#!/bin/bash
set -e

source utils/setup.sh

CSV_OUTPUT="py/utils/final_results_runs/sec2_expert_io_ab/sweep_threshold_test_2.csv"

echo "========================================================="
echo "Starting Threshold vs Budget Sweep (Cache 32)"
echo "========================================================="

HETEROPREDICT_IO_THREADS=32 python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --dataset oracle \
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces \
  --num-prompts 3 \
  --max-new-tokens 100 \
  --lookaheads 1 \
  --cache-sizes 32 \
  --custom-explicit-prefetch-budgets \
  --prefetch-budgets 8 \
  --prefetch-thresholds 0.7 0.8 \
  --lambdas 1.0 \
  --sweep-question custom_1_16_no_ppl \
  --include-oracle-full-union \
  --csv-file "$CSV_OUTPUT"

echo "========================================================="
echo "Sweep Complete!"
