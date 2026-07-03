#!/bin/bash
set -e

# Activate the virtual environment
source utils/setup.sh

echo "========================================================="
echo "Starting Parallel Sweep (32 Threads)"
echo "========================================================="
HETEROPREDICT_IO_THREADS=32 python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --dataset oracle \
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces \
  --num-prompts 3 \
  --max-new-tokens 100 \
  --lookaheads 1 2 3 \
  --cache-sizes 24 32 40 \
  --sweep-question custom_1_16_no_ppl \
  --budget-fractions 0.25 0.5 0.75 1.0 \
  --include-oracle-full-union \
  --csv-file py/utils/final_results_runs/sec2_expert_io_ab/oracle_sec2_sweep_parallel_2.csv


echo ""
echo "========================================================="
echo "Starting Sequential Sweep (1 Thread)"
echo "========================================================="
HETEROPREDICT_IO_THREADS=1 python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --dataset oracle \
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces \
  --num-prompts 3 \
  --max-new-tokens 100 \
  --lookaheads 1 2 3 \
  --cache-sizes 24 32 40 \
  --sweep-question custom_1_16_no_ppl \
  --budget-fractions 0.25 0.5 0.75 1.0 \
  --include-oracle-full-union \
  --csv-file py/utils/final_results_runs/sec2_expert_io_ab/oracle_sec2_sweep_sequential_2.csv


echo ""
echo "========================================================="
echo "All sweeps completed successfully!"
echo "========================================================="
