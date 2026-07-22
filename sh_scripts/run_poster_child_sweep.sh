#!/bin/bash
set -e

# Activate the virtual environment
source utils/setup.sh

CSV_OUTPUT="py/utils/final_results_runs/sec2_expert_io_ab/poster_child_cache32_sequential.csv"

echo "========================================================="
echo "Starting Poster Child Sweep (Cache 32, Parallel I/O)"
echo "========================================================="
HETEROPREDICT_IO_THREADS=1 python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --dataset oracle \
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces \
  --num-prompts 3 \
  --max-new-tokens 100 \
  --lookaheads 1 2 3 4 6 \
  --cache-sizes 32 \
  --custom-explicit-prefetch-budgets \
  --prefetch-budgets 8 16 24 \
  --lambdas 1.0 \
  --sweep-question custom_1_16_no_ppl \
  --include-oracle-full-union \
  --csv-file "$CSV_OUTPUT"

echo ""
echo "========================================================="
echo "Sweep completed! Results saved to $CSV_OUTPUT"
echo "========================================================="
