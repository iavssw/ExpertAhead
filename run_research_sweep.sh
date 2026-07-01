#!/bin/bash
set -e

# =====================================================================
# 1. Gather Oracle Predictor and RANDOM baseline data
# =====================================================================

# Parallel IO Regime (32 Threads)
echo "=== Running Parallel IO Sweep (32 Threads) ==="
unset HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO
export HETEROPREDICT_IO_THREADS=32
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question thread_benchmark \
  --cache-sizes 24 32 \
  --dataset oracle \
  --num-prompts 3 \
  --cold-per-prompt \
  --csv-file py/utils/final_results_runs/sec2_expert_io_ab/research_oracle_parallel_32.csv

# Sequential IO Regime (1 Thread)
echo "=== Running Sequential IO Sweep (1 Thread) ==="
unset HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO
export HETEROPREDICT_IO_THREADS=1
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question thread_benchmark \
  --cache-sizes 24 32 \
  --dataset oracle \
  --num-prompts 3 \
  --cold-per-prompt \
  --csv-file py/utils/final_results_runs/sec2_expert_io_ab/research_oracle_sequential_1.csv


echo "Done! Run 'python3 py/plot_research_meeting.py' to generate the plots."
