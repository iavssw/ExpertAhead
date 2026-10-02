#!/usr/bin/env bash
# Speed vs quality across domains at the paper's C=64 operating points.
#
# Methods (same prompts for all; RANDOM = default routing = quality reference):
#   01_random        RANDOM eviction, no prefetch
#   02_gating        Cross-layer gating prefetch, B=2
#   03_cache_cond    Cache-conditional routing only, λ=1 J=5
#   04_prefetch      ExpertAhead prefetch only (λ=0), S=6 B=36 — wikitext + multi predictors
#   05_hybrid        ExpertAhead + CC (λ=1 J=5), S=6 B=36   — wikitext + multi predictors
#   06_oracle        Oracle Full Union S=1 (per-prompt traces captured in step 0)
#
# Quality: reference perplexity (teacher-forced NLL of held-out / gold text given the
# prompt, under each method's routing) for every domain; plus exact match (gsm8k) and
# ROUGE-L (cnn_dailymail, orca). Speed: decode tokens/s. Both reported relative to RANDOM.
#
# Usage (tmux recommended):
#   ./sh_scripts/run_quality_speed_eval.sh
#   DRY_RUN=1 ./sh_scripts/run_quality_speed_eval.sh
#   DOMAINS="gsm8k" START_STEP=4 RUN_TS=<ts> ./sh_scripts/run_quality_speed_eval.sh   # resume
#
# Optional:
#   MAX_NEW_TOKENS=200 NUM_PROMPTS=5 CACHE_SIZE=64
#   PRED_WIKI=... PRED_MULTI=...
#   TRACE_ROOT=trainingData/domain_oracle_traces_t200
#   SKIP_ORACLE=1  SKIP_CAPTURE=1  RETRY_FAILED=1
#
# Outputs:
#   $OUT_ROOT/by_domain/<domain>.json         examples with ref_text
#   $OUT_ROOT/<domain>/C64_<ts>/              sweep.csv, transcripts.jsonl, generations.md, summary.md
#   $OUT_ROOT/summary_<ts>.md / .csv          cross-domain table

set -euo pipefail
cd "$(dirname "$0")/.."
source utils/setup.sh

export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1

C="${CACHE_SIZE:-64}"
DOMAINS="${DOMAINS:-wikitext fineweb orca gsm8k cnn_dailymail}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
PRED_WIKI="${PRED_WIKI:-trainingData/qwen3_30b/transformer_final_pfill_markov_emb}"
PRED_MULTI="${PRED_MULTI:-trainingData/qwen3_30b/multi_dataset/transformer_emb_markov_pfill_mixed}"
REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
SRC_SPLIT_DIR="${SRC_SPLIT_DIR:-py/utils/final_results_runs/expanded_prompt_space/by_domain}"
OUT_ROOT="${OUT_ROOT:-py/utils/final_results_runs/quality_speed}"
SPLIT_DIR="$OUT_ROOT/by_domain"
TRACE_ROOT="${TRACE_ROOT:-trainingData/domain_oracle_traces_t${MAX_NEW_TOKENS}}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-$(( 3600 + MAX_NEW_TOKENS * 10 ))}"
BATCH_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

# Paper C=64 operating points (run_final_best_10prompt_oracle.sh).
XL_B="${XL_B:-2}"
CC_J="${CC_J:-5}"
EA_S="${EA_S:-6}"
EA_B="${EA_B:-36}"

# shellcheck disable=SC2206
DOMAIN_ARR=($DOMAINS)

dry() { [[ "${DRY_RUN:-0}" == "1" ]]; }

prepare_examples() {
  local -a cmd=(python3 py/utils/prepare_ref_examples.py
    --src-dir "$SRC_SPLIT_DIR" --out-dir "$SPLIT_DIR"
    --domains "${DOMAIN_ARR[@]}" --prompt-max-chars "$PROMPT_MAX_CHARS")
  if dry; then
    echo "[step 0 prep]"; printf '  %q' "${cmd[@]}"; echo
    return
  fi
  local missing=0
  for d in "${DOMAIN_ARR[@]}"; do [[ -f "$SPLIT_DIR/$d.json" ]] || missing=1; done
  if [[ $missing == 0 && "${REFRESH_EXAMPLES:-0}" != "1" ]]; then
    echo "[prep] reusing $SPLIT_DIR (REFRESH_EXAMPLES=1 to rebuild — then re-capture traces)"
    return
  fi
  "${cmd[@]}"
}

capture_traces() {
  [[ "${SKIP_ORACLE:-0}" == "1" || "${SKIP_CAPTURE:-0}" == "1" ]] && return
  local -a cmd=(python3 py/utils/capture_domain_oracle_traces.py
    --examples-dir "$SPLIT_DIR" --out-root "$TRACE_ROOT"
    --domains "${DOMAIN_ARR[@]}" --num-prompts "$NUM_PROMPTS"
    --max-new-tokens "$MAX_NEW_TOKENS" --prompt-max-chars "$PROMPT_MAX_CHARS")
  if dry; then
    echo "[step 0 traces]"; printf '  %q' "${cmd[@]}"; echo
    return
  fi
  echo "[traces] capturing missing traces → $TRACE_ROOT (skips existing)"
  "${cmd[@]}"
}

run_domain() {
  local domain=$1
  local OUT_DIR="$OUT_ROOT/$domain/C${C}_${BATCH_TS}"
  local CSV="$OUT_DIR/sweep.csv"
  local LOG="$OUT_DIR/collection.log"
  mkdir -p "$OUT_DIR"

  local -a COMMON=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --examples-json "$SPLIT_DIR/$domain.json"
    --domain "$domain"
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
  local -a PREDS=(--predictor-base-dirs "$PRED_WIKI" "$PRED_MULTI" --predictor-tags wikitext multi)

  run_step() {
    local name=$1
    local step_num="${name%%_*}"
    shift
    if [[ -n "${START_STEP:-}" ]] && (( 10#$step_num < START_STEP )); then
      echo "[$domain $name] skipped (START_STEP=$START_STEP)"
      return
    fi
    local -a cmd=( "${COMMON[@]}" "$@" )
    [[ -f "$CSV" ]] && cmd+=(--append)
    [[ "${RETRY_FAILED:-0}" == "1" ]] && cmd+=(--retry-failed)
    cmd+=(--log-file "$OUT_DIR/${name}.log")
    [[ "${LIST_CONFIGS:-0}" == "1" ]] && cmd+=(--list-configs)
    if dry; then
      echo "[$domain $name]"; printf '  %q' "${cmd[@]}"; echo
      return
    fi
    echo "[$domain C=$C $name] $(date -Is)" | tee -a "$LOG"
    "${cmd[@]}"
  }

  if ! dry; then
    {
      echo "run_quality_speed_eval.sh domain=$domain C=$C"
      echo "EXAMPLES=$SPLIT_DIR/$domain.json NUM_PROMPTS=$NUM_PROMPTS MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
      echo "PRED_WIKI=$PRED_WIKI"
      echo "PRED_MULTI=$PRED_MULTI"
      echo "TRACE_DIR=$TRACE_ROOT/$domain"
      echo "Gating B=$XL_B | CC J=$CC_J | ExpertAhead S=$EA_S B=$EA_B | Hybrid S=$EA_S B=$EA_B J=$CC_J | Oracle Full Union S=1"
    } > "$OUT_DIR/command.txt"
  fi

  echo "=== domain=$domain C=$C → $OUT_DIR ==="

  run_step "01_random" \
    --sweep-question custom_1_16_no_ppl \
    --random-baseline-only

  run_step "02_gating" \
    --sweep-question gating_budget_sweep \
    --custom-explicit-prefetch-budgets \
    --no-budget-fractions \
    --prefetch-budgets "$XL_B"

  run_step "03_cache_cond" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --cache-cond-only \
    --lambdas 1 \
    --routing-bias-top-ns "$CC_J"

  run_step "04_prefetch" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    "${PREDS[@]}" \
    --lookaheads "$EA_S" \
    --custom-explicit-prefetch-budgets \
    --no-budget-fractions \
    --prefetch-budgets "$EA_B" \
    --prefetch-only

  run_step "05_hybrid" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --both-only \
    "${PREDS[@]}" \
    --lookaheads "$EA_S" \
    --custom-explicit-prefetch-budgets \
    --no-budget-fractions \
    --prefetch-budgets "$EA_B" \
    --lambdas 1 \
    --routing-bias-top-ns "$CC_J"

  if [[ "${SKIP_ORACLE:-0}" != "1" ]]; then
    run_step "06_oracle" \
      --sweep-question oracle_baseline_sweep \
      --skip-baselines \
      --oracle-full-union-only \
      --no-actual-predictor \
      --lookaheads 1 \
      --disable-measurement \
      --oracle-trace-dir "$TRACE_ROOT/$domain"
  fi

  if ! dry && [[ "${LIST_CONFIGS:-0}" != "1" ]]; then
    python3 py/utils/eval_quality_speed.py \
      --examples "$SPLIT_DIR/$domain.json" \
      --run-dir "$OUT_DIR" || true
  fi
}

prepare_examples
capture_traces

RUN_DIRS=()
for domain in "${DOMAIN_ARR[@]}"; do
  run_domain "$domain"
  RUN_DIRS+=("$OUT_ROOT/$domain/C${C}_${BATCH_TS}")
done

if ! dry && [[ "${LIST_CONFIGS:-0}" != "1" ]]; then
  python3 py/utils/eval_quality_speed.py \
    --summary-only \
    --run-dirs "${RUN_DIRS[@]}" \
    --out-prefix "$OUT_ROOT/summary_${BATCH_TS}" || true
  echo "All done. Cross-domain summary: $OUT_ROOT/summary_${BATCH_TS}.md"
fi
