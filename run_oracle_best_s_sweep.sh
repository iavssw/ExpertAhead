#!/usr/bin/env bash
# Find best Oracle Full Union lookahead S per cache on this machine.
# Uses the same 10 oracle traces as final_10prompt_oracle.
#
# Usage:
#   ./run_oracle_best_s_sweep.sh
#   CACHE_SIZE=32 ./run_oracle_best_s_sweep.sh
#   DRY_RUN=1 CACHE_SIZES="16 32" ./run_oracle_best_s_sweep.sh
#
# After the sweep:
#   python3 py/utils/analyze_oracle_best_s.py \
#     --run-dirs py/utils/final_results_runs/oracle_best_s_sweep/C16_* ...

set -euo pipefail
cd "$(dirname "$0")"
source utils/setup.sh

export HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0
export HETEROPREDICT_IO_THREADS="${HETEROPREDICT_IO_THREADS:-32}"

CACHE_SIZES="${CACHE_SIZES:-${CACHE_SIZE:-8 16 24 32 40 48 56 64}}"
ORACLE_TRACE_DIR="${ORACLE_TRACE_DIR:-trainingData/wikitext_test_traces}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-$(( 3600 + NUM_PROMPTS * MAX_NEW_TOKENS * 5 ))}"
BATCH_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-py/utils/final_results_runs/oracle_best_s_sweep}"

# Lookahead grids: dense at low S (slower SSD likely favors smaller windows).
declare -A CACHE_LOOKAHEADS=(
  [8]="1 2 3"
  [16]="1 2 3 4"
  [24]="1 2 3 4 5"
  [32]="1 2 3 4 5 6"
  [40]="1 2 3 4 5 6 7 8"
  [48]="1 2 3 4 5 6 8 10"
  [56]="1 2 3 4 6 8 10 11 12"
  [64]="1 2 3 4 6 8 12 16"
)

run_one_cache() {
  local C=$1
  local lookaheads="${CACHE_LOOKAHEADS[$C]:-}"
  if [[ -z "$lookaheads" ]]; then
    echo "No lookahead grid for C=$C" >&2
    exit 1
  fi

  local OUT_DIR="${OUT_DIR:-$OUT_ROOT/C${C}_${BATCH_TS}}"
  local CSV="${CSV:-$OUT_DIR/sweep.csv}"
  local LOG="${LOG:-$OUT_DIR/collection.log}"
  mkdir -p "$OUT_DIR"

  local -a cmd=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --dataset oracle
    --oracle-trace-dir "$ORACLE_TRACE_DIR"
    --sweep-question oracle_baseline_sweep
    --skip-baselines
    --oracle-full-union-only
    --no-actual-predictor
    --cache-sizes "$C"
    --lookaheads $lookaheads
    --num-prompts "$NUM_PROMPTS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --disable-measurement
    --drop-page-cache-between-runs
    --out-dir "$OUT_DIR"
    --csv-file "$CSV"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
    --log-file "$OUT_DIR/oracle_s_sweep.log"
  )
  if [[ -f "$CSV" ]]; then
    cmd+=(--append)
  fi
  if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
    cmd+=(--retry-failed)
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[C=$C] lookaheads=($lookaheads)"
    printf '  %q' "${cmd[@]}"
    echo
    return
  fi

  {
    echo "run_oracle_best_s_sweep.sh C=$C"
    echo "lookaheads=$lookaheads"
    echo "NUM_PROMPTS=$NUM_PROMPTS"
    echo "OUT_DIR=$OUT_DIR"
  } > "$OUT_DIR/command.txt"

  echo "=== C=$C → $OUT_DIR  LA=($lookaheads) ===" | tee -a "$LOG"
  echo "[C=$C] $(date -Is)" | tee -a "$LOG"
  "${cmd[@]}"
  echo "Done C=$C. CSV: $CSV" | tee -a "$LOG"
}

for C in $CACHE_SIZES; do
  unset OUT_DIR CSV LOG
  run_one_cache "$C"
done

echo
echo "Analyze best S:"
echo "  python3 py/utils/analyze_oracle_best_s.py --run-dirs $OUT_ROOT/C*_${BATCH_TS}"
