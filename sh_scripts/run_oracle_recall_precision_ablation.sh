#!/bin/bash
set -euo pipefail

# Rebuild predict backend if sources changed (skip if you already built).
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cmake --build build --target unified_llm_w4a16_predict_libtorch -j"$(nproc)"
fi

source utils/setup.sh

CSV="py/utils/final_results_runs/sec2_expert_io_ab/oracle_recall_precision_ablation.csv"

# Parallel IO (32 threads) — main regime for hidden-load overlap.
echo "=== Oracle recall/precision ablation (parallel IO, C=32) ==="
HETEROPREDICT_IO_THREADS=32 python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question oracle_recall_precision_ablation \
  --model qwen \
  --dataset oracle \
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces \
  --num-prompts 3 \
  --max-new-tokens 100 \
  --cache-sizes 32 \
  --lookaheads 1 2 3 \
  --custom-explicit-prefetch-budgets \
  --prefetch-budgets 8 16 24 32 \
  --csv-file "$CSV"

# Optional sequential IO comparison (uncomment to run both regimes in one script).
# echo "=== Oracle recall/precision ablation (sequential IO, C=32) ==="
# HETEROPREDICT_IO_THREADS=1 python3 py/utils/sweep_predict_cached_cache_metrics.py \
#   --sweep-question oracle_recall_precision_ablation \
#   --model qwen \
#   --dataset oracle \
#   --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces \
#   --num-prompts 3 \
#   --max-new-tokens 100 \
#   --cache-sizes 32 \
#   --lookaheads 1 2 3 \
#   --custom-explicit-prefetch-budgets \
#   --prefetch-budgets 8 16 24 32 \
#   --csv-file "${CSV%.csv}_sequential.csv"

echo "Done. Results: $CSV"
