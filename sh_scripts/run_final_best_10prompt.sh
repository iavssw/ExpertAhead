#!/usr/bin/env bash
# Re-run best comparison configs with 10 prompts for thesis final results.
#
# Usage:
#   CACHE_SIZE=16 ./run_final_best_10prompt.sh
#   CACHE_SIZES="48 64" ./run_final_best_10prompt.sh
#   DRY_RUN=1 CACHE_SIZE=16 ./run_final_best_10prompt.sh
#
# Optional:
#   OUT_DIR=py/utils/final_results_runs/final_10prompt/C16
#   CSV=$OUT_DIR/sweep.csv
#   NUM_PROMPTS=10
#   MAX_NEW_TOKENS=150
#   SUBPROCESS_TIMEOUT=14400   # optional override; default scales with prompts × tokens
#   START_STEP=4          # resume from step 04 (01_random .. 05_expert_ahead_cc)

set -euo pipefail
cd "$(dirname "$0")/.."
source utils/setup.sh

CACHE_SIZES="${CACHE_SIZES:-${CACHE_SIZE:-}}"
if [[ -z "$CACHE_SIZES" ]]; then
  echo "Set CACHE_SIZE (single) or CACHE_SIZES (space-separated, e.g. \"48 64\")" >&2
  exit 1
fi

PRED="${PRED:-trainingData/qwen3_30b/transformer_final_pfill_markov_emb}"
REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
# Generous per-config ceiling: ~5s/token × prompts + 1h model/load margin.
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-$(( 3600 + NUM_PROMPTS * MAX_NEW_TOKENS * 5 ))}"
BATCH_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

run_one_cache() {
  local C=$1
  local TS="$BATCH_TS"
  local OUT_DIR="${OUT_DIR:-py/utils/final_results_runs/final_10prompt/C${C}_${TS}}"
  local CSV="${CSV:-$OUT_DIR/sweep.csv}"
  local LOG="${LOG:-$OUT_DIR/collection.log}"
  mkdir -p "$OUT_DIR"

  # Best configs from comparison sweeps (update when new data lands).
  local XL_B CC_J EA_S EA_B EACC_S EACC_B EACC_J
  case "$C" in
    16)
      XL_B=8
      CC_J=5
      EA_S=2; EA_B=4
      EACC_S=1; EACC_B=4; EACC_J=5
      ;;
    32)
      XL_B=6
      CC_J=5
      EA_S=2; EA_B=8
      EACC_S=1; EACC_B=8; EACC_J=5
      ;;
    48)
      XL_B=2
      CC_J=5
      EA_S=5; EA_B=24
      EACC_S=4; EACC_B=20; EACC_J=5
      ;;
    64)
      XL_B=2
      CC_J=5
      EA_S=6; EA_B=36
      EACC_S=6; EACC_B=36; EACC_J=5
      ;;
    *)
      echo "No best-config table for C=$C" >&2
      exit 1
      ;;
  esac


  # Each step below runs exactly ONE config (best point from comparison sweeps).

  local -a COMMON=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --dataset wikitext
    --num-prompts "$NUM_PROMPTS"
    --prompt-max-chars "$PROMPT_MAX_CHARS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes "$C"
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --drop-page-cache-between-runs
    --out-dir "$OUT_DIR"
    --csv-file "$CSV"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
  )

  run_step() {
    local name=$1
    local step_num="${name%%_*}"
    shift
    if [[ -n "${START_STEP:-}" ]] && (( 10#$step_num < START_STEP )); then
      echo "[$name] skipped (START_STEP=$START_STEP)" | tee -a "$LOG"
      return
    fi
    local -a cmd=( "${COMMON[@]}" "$@" )
    if [[ -f "$CSV" ]]; then
      cmd+=(--append)
    fi
    if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
      cmd+=(--retry-failed)
    fi
    cmd+=(--log-file "$OUT_DIR/${name}.log")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[C=$C $name]"
      printf '  %q' "${cmd[@]}"
      echo
      return
    fi

    echo "[C=$C $name] $(date -Is)" | tee -a "$LOG"
    "${cmd[@]}"
  }

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    {
      echo "run_final_best_10prompt.sh C=$C"
      echo "NUM_PROMPTS=$NUM_PROMPTS"
      echo "OUT_DIR=$OUT_DIR"
      echo "RANDOM baseline"
      echo "Cross-Layer B=$XL_B"
      echo "Cache-Cond J=$CC_J"
      echo "ExpertAhead S=$EA_S B=$EA_B"
      echo "ExpertAhead-CC S=$EACC_S B=$EACC_B J=$EACC_J"
    } > "$OUT_DIR/command.txt"
  fi

  echo "=== C=$C → $OUT_DIR ==="

  # 1) RANDOM baseline
  run_step "01_random" \
    --sweep-question custom_1_16_no_ppl \
    --random-baseline-only

  # 2) Cross-Layer / gating (best B)
  run_step "02_cross_layer" \
    --sweep-question gating_budget_sweep \
    --custom-explicit-prefetch-budgets \
    --no-budget-fractions \
    --prefetch-budgets "$XL_B"

  # 3) Cache-Cond only (best J) — baselines already in step 1
  run_step "03_cache_cond" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --cache-cond-only \
    --lambdas 1 \
    --routing-bias-top-ns "$CC_J"

  # 4) ExpertAhead / predictor-only (best S, B)
  run_step "04_expert_ahead" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --predictor-base-dir "$PRED" \
    --lookaheads "$EA_S" \
    --custom-explicit-prefetch-budgets \
    --no-budget-fractions \
    --prefetch-budgets "$EA_B" \
    --prefetch-only

  # 5) ExpertAhead-CC / hybrid (best S, B, J)
  run_step "05_expert_ahead_cc" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --both-only \
    --predictor-base-dir "$PRED" \
    --lookaheads "$EACC_S" \
    --custom-explicit-prefetch-budgets \
    --no-budget-fractions \
    --prefetch-budgets "$EACC_B" \
    --lambdas 1 \
    --routing-bias-top-ns "$EACC_J"

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo "Done C=$C. CSV: $CSV"
  fi
}

for C in $CACHE_SIZES; do
  # When batching multiple caches, ignore OUT_DIR/CSV/LOG from a prior iteration.
  unset OUT_DIR CSV LOG
  run_one_cache "$C"
done
