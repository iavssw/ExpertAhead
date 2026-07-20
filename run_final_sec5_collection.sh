#!/usr/bin/env bash
# Final Section 5 data collection: Prefetch + Both + baselines + Cache-Cond + Gating
# at C=40,48,56,64 with cache-specific LA×B grids. 5 wikitext prompts per config.
#
# Writes NEW files only (timestamped directory). Safe to resume:
#   RETRY_FAILED=1 ./run_final_sec5_collection.sh
#
# Dry-run (print commands + run counts, no GPU):
#   DRY_RUN=1 ./run_final_sec5_collection.sh
#
# Run in tmux:
#   tmux new -s final_sec5
#   cd ~/heteroPredict && ./run_final_sec5_collection.sh
#   # detach: Ctrl-b d
#   # reattach: tmux attach -t final_sec5
#
# Monitor (from another shell):
#   tail -f py/utils/final_results_runs/final_sec5_collection/<ts>/collection.log

set -euo pipefail
cd "$(dirname "$0")"
source utils/setup.sh

PRED="${PRED:-trainingData/qwen3_30b/transformer_final_pfill_markov_emb}"
REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-7200}"

TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-py/utils/final_results_runs/final_sec5_collection}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/${TS}}"
CSV="${CSV:-$OUT_DIR/sweep.csv}"
LOG="${LOG:-$OUT_DIR/collection.log}"
mkdir -p "$OUT_DIR"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  exec > >(tee -a "$LOG") 2>&1
fi

budget_fractions() {
  local C=$1
  shift
  local b fracs=()
  for b in "$@"; do
    fracs+=("$(python3 -c "print($b/$C)")")
  done
  printf '%s\n' "${fracs[@]}"
}

count_grid() {
  local C=$1 LA_N=$2 B_N=$3
  local prefetch=$((LA_N * B_N))
  local both=$((LA_N * B_N * 2))   # J=5,6
  local cc=2 gating=4 baselines=2
  echo $((prefetch + both + cc + gating + baselines))
}

run_main() {
  local C=$1
  shift
  local -a LOOKAHEADS=()
  local -a BUDGETS=()
  local mode=la
  for x in "$@"; do
    if [[ "$x" == "--" ]]; then mode=budget; continue; fi
    if [[ "$mode" == la ]]; then LOOKAHEADS+=("$x"); else BUDGETS+=("$x"); fi
  done
  mapfile -t FRACS < <(budget_fractions "$C" "${BUDGETS[@]}")

  local -a EXTRA=()
  if [[ -f "$CSV" ]]; then
    EXTRA+=(--append)
  fi
  if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
    EXTRA+=(--retry-failed)
  fi

  local -a CMD=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --sweep-question custom_1_16_no_ppl
    --model qwen
    --predictor-base-dir "$PRED"
    --dataset wikitext
    --num-prompts "$NUM_PROMPTS"
    --prompt-max-chars "$PROMPT_MAX_CHARS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes "$C"
    --lookaheads "${LOOKAHEADS[@]}"
    --custom-explicit-prefetch-budgets
    --prefetch-budgets "${BUDGETS[@]}"
    --budget-fractions "${FRACS[@]}"
    --lambdas 1
    --routing-bias-top-ns 5 6
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --drop-page-cache-between-runs
    --out-dir "$OUT_DIR"
    --csv-file "$CSV"
    --log-file "$OUT_DIR/sweep_C${C}.log"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
    "${EXTRA[@]}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] C=$C main  LA=${LOOKAHEADS[*]}  B=${BUDGETS[*]}  (~$(count_grid "$C" "${#LOOKAHEADS[@]}" "${#BUDGETS[@]}") configs)"
    printf '  %q' "${CMD[@]}"
    echo
    return
  fi

  echo "[final-sec5] C=$C main  LA=${LOOKAHEADS[*]}  B=${BUDGETS[*]}" | tee -a "$LOG"
  "${CMD[@]}"
}

run_gating() {
  local C=$1
  local -a GATING_B=(2 4 6 8)

  local -a EXTRA=(--append)
  if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
    EXTRA+=(--retry-failed)
  fi

  local -a CMD=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --sweep-question gating_budget_sweep
    --model qwen
    --dataset wikitext
    --num-prompts "$NUM_PROMPTS"
    --prompt-max-chars "$PROMPT_MAX_CHARS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes "$C"
    --custom-explicit-prefetch-budgets
    --prefetch-budgets "${GATING_B[@]}"
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --drop-page-cache-between-runs
    --out-dir "$OUT_DIR"
    --csv-file "$CSV"
    --log-file "$OUT_DIR/gating_C${C}.log"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
    "${EXTRA[@]}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] C=$C gating  B=${GATING_B[*]}  (4 configs)"
    printf '  %q' "${CMD[@]}"
    echo
    return
  fi

  echo "[final-sec5] C=$C gating  B=${GATING_B[*]}" | tee -a "$LOG"
  "${CMD[@]}"
}

{
  echo "run_final_sec5_collection.sh"
  echo "started: $(date -Is)"
  echo "OUT_DIR=$OUT_DIR"
  echo "CSV=$CSV"
  echo "NUM_PROMPTS=$NUM_PROMPTS"
  echo "PRED=$PRED"
} > "$OUT_DIR/command.txt"

TOTAL=0
add_total() { TOTAL=$((TOTAL + $1)); }

echo "=== Final Sec5 collection ==="
echo "OUT_DIR:  $OUT_DIR"
echo "CSV:      $OUT_DIR/sweep.csv"
echo "Log:      $LOG"
echo "Prompts:  $NUM_PROMPTS  max_new_tokens=$MAX_NEW_TOKENS  (~281 configs, many hours)"
echo ""

# C=40: LA 1-3, B=10,16,20,24
add_total "$(count_grid 40 3 4)"
run_main 40 1 2 3 -- 10 16 20 24
run_gating 40

# C=48: LA 1-4, B=10,16,20,24
add_total "$(count_grid 48 4 4)"
run_main 48 1 2 3 4 -- 10 16 20 24
run_gating 48

# C=56: LA 1-5, B=16,20,22,24,28
add_total "$(count_grid 56 5 5)"
run_main 56 1 2 3 4 5 -- 16 20 22 24 28
run_gating 56

# C=64: LA 1-6, B=24,28,32,36,40
add_total "$(count_grid 64 6 5)"
run_main 64 1 2 3 4 5 6 -- 24 28 32 36 40
run_gating 64

echo ""
echo "Grid summary (~$TOTAL configs total across 4 cache sizes):"
echo "  per C: Prefetch (LA×B) + Both (LA×B×J{5,6}) + Cache-Cond J{5,6} + LRU + RANDOM + Gating B{2,4,6,8}"
echo "  C=40: 3×4 prefetch + 24 both + 8 shared = 44"
echo "  C=48: 4×4 prefetch + 32 both + 8 shared = 56"
echo "  C=56: 5×5 prefetch + 50 both + 8 shared = 83"
echo "  C=64: 6×5 prefetch + 60 both + 8 shared = 98"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""
  echo "DRY_RUN=1 — no jobs launched. Unset DRY_RUN to run."
  exit 0
fi

echo ""
echo "Done. Results: $CSV"
echo "Logs: $OUT_DIR/*.log"
