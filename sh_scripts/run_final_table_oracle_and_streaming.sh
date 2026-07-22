#!/usr/bin/env bash
# Collect SSD Streaming + Oracle Full Union rows used by Tab. end_to_end_results.
#
# Pair with:
#   CACHE_SIZES="16 32 48 64" MAX_NEW_TOKENS=150 SKIP_ORACLE=1 \
#     OUT_ROOT=py/utils/final_results_runs/final_10prompt_oracle_150 \
#     ./run_final_best_10prompt_oracle.sh
# then:
#   python3 py/utils/build_end_to_end_table.py ...
#
# Usage:
#   ./run_final_table_oracle_and_streaming.sh
#   DRY_RUN=1 ./run_final_table_oracle_and_streaming.sh
#   CACHE_SIZES="16 32" ./run_final_table_oracle_and_streaming.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source utils/setup.sh

CACHE_SIZES="${CACHE_SIZES:-16 32 48 64}"
ORACLE_TRACE_DIR="${ORACLE_TRACE_DIR:-trainingData/wikitext_test_traces}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"
REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-$(( 3600 + NUM_PROMPTS * MAX_NEW_TOKENS * 5 ))}"
BATCH_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-py/utils/final_results_runs/final_table_exact_150}"

# Paper table Oracle Predictor S (lookahead) per cache size.
declare -A ORACLE_TABLE_S=(
  [16]=2
  [32]=1
  [48]=1
  [64]=1
)

run_streaming() {
  local OUT_DIR="$OUT_ROOT/streaming_${BATCH_TS}"
  mkdir -p "$OUT_DIR"
  local -a cmd=(
    env FORCE_EXPERT_MISS=1
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --dataset oracle
    --oracle-trace-dir "$ORACLE_TRACE_DIR"
    --sweep-question custom_1_16_no_ppl
    --random-baseline-only
    --num-prompts "$NUM_PROMPTS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes 16
    --drop-page-cache-between-runs
    --out-dir "$OUT_DIR"
    --csv-file "$OUT_DIR/sweep.csv"
    --log-file "$OUT_DIR/streaming.log"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[streaming]"
    printf '  %q' "${cmd[@]}"
    echo
    return
  fi
  echo "=== SSD streaming (FORCE_EXPERT_MISS=1) → $OUT_DIR ==="
  "${cmd[@]}"
}

run_oracle() {
  local C=$1
  local S=${ORACLE_TABLE_S[$C]:-}
  if [[ -z "$S" ]]; then
    echo "No table Oracle S for C=$C" >&2
    exit 1
  fi
  local OUT_DIR="$OUT_ROOT/C${C}_${BATCH_TS}"
  mkdir -p "$OUT_DIR"
  local -a cmd=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --dataset oracle
    --oracle-trace-dir "$ORACLE_TRACE_DIR"
    --num-prompts "$NUM_PROMPTS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes "$C"
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --drop-page-cache-between-runs
    --sweep-question oracle_baseline_sweep
    --skip-baselines
    --oracle-full-union-only
    --no-actual-predictor
    --lookaheads "$S"
    --disable-measurement
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
    --out-dir "$OUT_DIR"
    --csv-file "$OUT_DIR/sweep.csv"
    --log-file "$OUT_DIR/oracle.log"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[oracle C=$C S=$S]"
    printf '  %q' "${cmd[@]}"
    echo
    return
  fi
  {
    echo "run_final_table_oracle_and_streaming.sh C=$C S=$S"
    echo "NUM_PROMPTS=$NUM_PROMPTS MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
    echo "OUT_DIR=$OUT_DIR"
  } > "$OUT_DIR/command.txt"
  echo "=== Oracle Full Union C=$C S=$S → $OUT_DIR ==="
  "${cmd[@]}"
}

run_streaming
for C in $CACHE_SIZES; do
  run_oracle "$C"
done

echo ""
echo "Done. OUT_ROOT=$OUT_ROOT  RUN_TS=$BATCH_TS"
echo "Build the paper table with:"
echo "  python3 py/utils/build_end_to_end_table.py \\"
echo "    --streaming-csv $OUT_ROOT/streaming_${BATCH_TS}/sweep.csv \\"
echo "    --oracle-root $OUT_ROOT --oracle-suffix ${BATCH_TS} \\"
echo "    --methods-root py/utils/final_results_runs/final_10prompt_oracle_150 \\"
echo "    --methods-suffix <METHODS_TS> \\"
echo "    --out-dir py/utils/final_results_runs/final_10prompt_oracle_150"
