#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
source "$REPO_ROOT/utils/setup.sh"

export HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0
export HETEROPREDICT_IO_THREADS="${HETEROPREDICT_IO_THREADS:-32}"

ORACLE_TRACE_DIR="$REPO_ROOT/trainingData/wikitext_test_traces"
NUM_PROMPTS=5
MAX_NEW_TOKENS=80

OUT_DIR="$REPO_ROOT/py/utils/final_results_runs/oracle_la_vs_cache/20260705_222605"
CSV_FILE="$OUT_DIR/sweep.csv"
MEMORY_LOG="$OUT_DIR/memory_log.txt"

C=64
LAs="1 4 8 12 16 18"

echo "[oracle-la-vs-cache] Measuring base memory (LRU) for C=${C}..."
TIME_LOG=$(mktemp)

/usr/bin/time -v python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --dataset oracle \
  --oracle-trace-dir "$ORACLE_TRACE_DIR" \
  --sweep-question oracle_baseline_sweep \
  --lru-only \
  --cache-sizes "$C" \
  --lookaheads 1 \
  --num-prompts 1 \
  --max-new-tokens 2 \
  --cold-per-prompt \
  --disable-measurement \
  --temperature 0.0 > /dev/null 2> "$TIME_LOG"

MAX_RSS_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
rm -f "$TIME_LOG"

if [[ -z "$MAX_RSS_KB" ]]; then
  echo "  ERROR: Failed to measure Max RSS for C=$C"
  exit 1
fi

MB=$(( MAX_RSS_KB / 1024 ))
echo "  Measured LRU Peak RAM: ${MB} MB"
echo "${C},${MB}" >> "$MEMORY_LOG"

echo "[oracle-la-vs-cache] Running Oracle Full Union @ C=${C}, LA=${LAs} → $CSV_FILE"

python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --dataset oracle \
  --oracle-trace-dir "$ORACLE_TRACE_DIR" \
  --sweep-question oracle_baseline_sweep \
  --oracle-full-union-only \
  --no-actual-predictor \
  --cache-sizes "$C" \
  --lookaheads $LAs \
  --num-prompts "$NUM_PROMPTS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --cold-per-prompt \
  --drop-page-cache-between-prompts \
  --disable-measurement \
  --temperature 0.0 \
  --drop-page-cache-between-runs \
  --out-dir "$OUT_DIR" \
  --csv-file "$CSV_FILE" \
  --append

echo "Done!"
