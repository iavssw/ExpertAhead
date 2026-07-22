#!/usr/bin/env bash
# Oracle Full Union sweep for multiple cache sizes and lookaheads
#
# Usage:
#   ./run_oracle_la_vs_cache.sh
#   SWEEP_CSV=path/to/sweep.csv RETRY_FAILED=1 ./run_oracle_la_vs_cache.sh
#   DRY_RUN=1 ./run_oracle_la_vs_cache.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
source "$REPO_ROOT/utils/setup.sh"

export HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0
export HETEROPREDICT_IO_THREADS="${HETEROPREDICT_IO_THREADS:-32}"

ORACLE_TRACE_DIR="${ORACLE_TRACE_DIR:-$REPO_ROOT/trainingData/wikitext_test_traces}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"

if [[ -n "${SWEEP_CSV:-}" ]]; then
  OUT_DIR="${RUN_DIR:-$(dirname "$SWEEP_CSV")}"
  CSV_FILE="$SWEEP_CSV"
else
  TIME=$(date +%Y%m%d_%H%M%S)
  OUT_DIR="$REPO_ROOT/py/utils/final_results_runs/oracle_la_vs_cache/$TIME"
  CSV_FILE="$OUT_DIR/sweep.csv"
fi
mkdir -p "$OUT_DIR"
MEMORY_LOG="$OUT_DIR/memory_log.txt"
echo "Cache Size,LRU Peak RAM (MB)" > "$MEMORY_LOG"

for C in 8 16 24 32 40 48 56 64; do
  case $C in
    8) LAs="1 2 3" ;;
    16) LAs="1 2 3 4" ;;
    24) LAs="1 2 3 4 5" ;;
    32) LAs="1 2 3 4 5 6" ;;
    40) LAs="1 2 4 5 6 7 8" ;;
    48) LAs="1 4 6 8 9 10" ;;
    56) LAs="1 4 8 10 11 12" ;;
    64) LAs="1 4 8 12 16 18" ;;
  esac

  echo "[oracle-la-vs-cache] Measuring base memory (LRU) for C=${C}..."
  TIME_LOG=$(mktemp)
  
  # Run a quick LRU-only sweep (1 prompt, 2 tokens) to measure peak RAM
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

  # Extract Max RSS in KB, convert to Bytes, add 100 MB
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

  ARGS=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --dataset oracle
    --oracle-trace-dir "$ORACLE_TRACE_DIR"
    --sweep-question oracle_baseline_sweep
    --oracle-full-union-only
    --no-actual-predictor
    --cache-sizes "$C"
    --lookaheads $LAs
    --num-prompts "$NUM_PROMPTS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --cold-per-prompt
    --drop-page-cache-between-prompts
    --disable-measurement
    --temperature 0.0
    --drop-page-cache-between-runs
    --out-dir "$OUT_DIR"
    --csv-file "$CSV_FILE"
    --append
  )

  if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
    ARGS+=(--retry-failed)
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  DRY RUN: ${ARGS[*]}"
  else
    "${ARGS[@]}"
  fi
done
