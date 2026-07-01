#!/bin/bash
set -e

# We will run the thread_benchmark sweep 3 times.

echo "========================================="
echo "1. PURE INLINE SEQUENTIAL"
echo "========================================="
export HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO=1
unset HETEROPREDICT_IO_THREADS
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question thread_benchmark \
  --cache-sizes 24 \
  --dataset oracle \
  --num-prompts 2 \
  --cold-per-prompt \
  --csv-file py/utils/final_results_runs/thread_bench_inline_seq.csv

echo "========================================="
echo "2. 1-THREAD IO POOL"
echo "========================================="
unset HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO
export HETEROPREDICT_IO_THREADS=1
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question thread_benchmark \
  --cache-sizes 24 \
  --dataset oracle \
  --num-prompts 2 \
  --cold-per-prompt \
  --csv-file py/utils/final_results_runs/thread_bench_1thread.csv

echo "========================================="
echo "3. 16-THREAD IO POOL (PARALLEL)"
echo "========================================="
unset HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO
export HETEROPREDICT_IO_THREADS=16
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question thread_benchmark \
  --cache-sizes 24 \
  --dataset oracle \
  --num-prompts 2 \
  --cold-per-prompt \
  --csv-file py/utils/final_results_runs/thread_bench_16thread.csv

