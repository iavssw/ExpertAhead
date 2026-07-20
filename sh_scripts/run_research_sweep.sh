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
  --sweep-question oracle_baseline_sweep \
  --cache-sizes 8 24 \
  --dataset oracle \
  --num-prompts 2 \
  --cold-per-prompt \
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces \
  --include-oracle-full-union \
  --csv-file py/utils/final_results_runs/sec2_expert_io_ab/research_oracle_parallel_32.csv

# Sequential IO Regime (1 Thread)
echo "=== Running Sequential IO Sweep (1 Thread) ==="
unset HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO
export HETEROPREDICT_IO_THREADS=1
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question oracle_baseline_sweep \
  --cache-sizes 8 24 \
  --dataset oracle \
  --num-prompts 2 \
  --cold-per-prompt \
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces \
  --include-oracle-full-union \
  --csv-file py/utils/final_results_runs/sec2_expert_io_ab/research_oracle_sequential_1.csv


# =====================================================================
# 2. Gather Cache-Conditional and Hybrid Data (For Later Comparison)
# =====================================================================
# Uncomment the following to run the unified sweeps for cache conditional (J=6) + Predictor
# Note: make sure HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO is unset to use parallel IO!

echo "=== Running Unified (Cache-Conditional + Oracle) Sweep ==="
unset HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO
export HETEROPREDICT_IO_THREADS=32
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question unified_sweep \
  --cache-sizes 8 16 24 32 40 \
  --dataset oracle \
  --num-prompts 5 \
  --cold-per-prompt \
  --csv-file py/utils/final_results_runs/sec2_expert_io_ab/research_unified_parallel.csv

echo "Done! Run 'python3 py/plot_research_meeting.py' to generate the plots."
