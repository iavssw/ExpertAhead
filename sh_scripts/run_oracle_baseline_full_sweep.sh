#!/usr/bin/env bash
# Recreate the ORACLE_BASELINE_SWEEP: LRU + RANDOM + Oracle Full Union per cache size,
# with cache-specific lookahead grids (matches the 2026-07-05/06 sweep CSV).
#
# Usage:
#   ./run_oracle_baseline_full_sweep.sh
#   DRY_RUN=1 ./run_oracle_baseline_full_sweep.sh
#   SWEEP_CSV=path/to/sweep.csv RETRY_FAILED=1 ./run_oracle_baseline_full_sweep.sh
#
# Analyze:
#   python3 py/utils/plot_oracle_cache_vs_lru.py --csv <sweep.csv>
#   python3 py/utils/plot_lookahead_sweep.py --csv <sweep.csv>

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
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
  OUT_DIR="${OUT_ROOT:-$REPO_ROOT/py/utils/final_results_runs/oracle_baseline_full_sweep}/$TIME"
  CSV_FILE="$OUT_DIR/sweep.csv"
fi
mkdir -p "$OUT_DIR"

# Lookahead grid per cache size (from the reference ORACLE_BASELINE_SWEEP CSV).
declare -A CACHE_LOOKAHEADS=(
  [8]="1 2 3"
  [16]="1 2 3 4"
  [24]="1 2 3 4 5"
  [32]="1 2 3 4 5 6"
  [40]="1 2 4 5 6 7 8"
  [48]="1 4 6 8 9 10"
  [56]="1 4 8 10 11 12"
  [64]="1 4 8 12 16 18"
)
CACHE_SIZES=(8 16 24 32 40 48 56 64)

run_one_cache() {
  local cache_size="$1"
  local lookaheads="${CACHE_LOOKAHEADS[$cache_size]}"
  local append_flag=()
  if [[ -f "$CSV_FILE" ]]; then
    append_flag=(--append)
  fi
  if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
    append_flag=(--retry-failed)
  fi

  local args=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --dataset oracle
    --oracle-trace-dir "$ORACLE_TRACE_DIR"
    --sweep-question oracle_baseline_sweep
    --oracle-full-union-only
    --no-actual-predictor
    --cache-sizes "$cache_size"
    --lookaheads $lookaheads
    --num-prompts "$NUM_PROMPTS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --cold-per-prompt
    --disable-measurement
    --temperature 0.0
    --out-dir "$OUT_DIR"
    --csv-file "$CSV_FILE"
    "${append_flag[@]}"
  )

  if [[ "${DROP_PAGE_CACHE:-0}" == "1" ]]; then
    args+=(--drop-page-cache-before-first-run --drop-page-cache-between-runs)
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[oracle-baseline] C=$cache_size LA=($lookaheads)"
    printf '  %q' "${args[@]}"
    echo
    return 0
  fi

  echo "============================================================"
  echo "[oracle-baseline] C=$cache_size  lookaheads=($lookaheads)"
  echo "  CSV: $CSV_FILE"
  echo "============================================================"
  "${args[@]}"
}

echo "Oracle baseline full sweep"
echo "  prompts=$NUM_PROMPTS  max_new_tokens=$MAX_NEW_TOKENS"
echo "  output=$CSV_FILE"
echo "  retry_failed=${RETRY_FAILED:-0}  drop_page_cache=${DROP_PAGE_CACHE:-0}"
echo

for C in "${CACHE_SIZES[@]}"; do
  run_one_cache "$C"
done

echo
echo "Done. Results: $CSV_FILE"
