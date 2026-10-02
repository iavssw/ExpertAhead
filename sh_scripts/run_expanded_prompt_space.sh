#!/usr/bin/env bash
# Per-domain expanded prompt-space: quality degradation vs normal operation.
#
# Normal operation (quality reference, runs fast):
#   RANDOM backend @ NORMAL_CACHE (default 64)
#
# Experimental modes (CACHE_SIZES, default 24 32):
#   Prefetch Only (λ=0, lossless — expect match_to_normal ≈ 100%)
#   Both (λ=1, J=5 hybrid — may degrade vs normal)
#
# Domains (5 prompts each): wikitext, fineweb, orca, gsm8k, cnn_dailymail
# Gold metrics (also reported):
#   wikitext/fineweb/orca → generation perplexity (↓ better)
#   gsm8k                 → exact match
#   cnn_dailymail         → ROUGE-L
#
# Usage (tmux recommended; set PRED when the new predictor lands):
#   PRED=path/to/new_predictor ./sh_scripts/run_expanded_prompt_space.sh
#   DRY_RUN=1 ./sh_scripts/run_expanded_prompt_space.sh
#   DOMAINS="gsm8k cnn_dailymail" CACHE_SIZES="24" ./sh_scripts/run_expanded_prompt_space.sh
#
# Optional:
#   NORMAL_CACHE=64          # normal-operation baseline cache size
#   SKIP_EXP_RANDOM=1        # skip RANDOM @ experimental C (default: run it for TPS)
#   NUM_PROMPTS=5 MAX_NEW_TOKENS=100
#   BUDGET_FRACTIONS="0.5" LOOKAHEADS="1 4 8 16"
#   OUT_ROOT=py/utils/final_results_runs/expanded_prompt_space
#
# Outputs:
#   <domain>/normal_C64_<ts>/     — normal baseline (transcripts + sweep)
#   <domain>/C<c>_<ts>/           — experimental run
#     comparison.md               — degradation table vs normal
#     correctness.json, sweep.csv, generations.md, transcripts.jsonl

set -euo pipefail
cd "$(dirname "$0")/.."
source utils/setup.sh

export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1

# C must satisfy expert-reuse mins (LA8≈45, LA16≈67). C=64 covers f1/f4/f8/f16.
CACHE_SIZES="${CACHE_SIZES:-${CACHE_SIZE:-64}}"
NORMAL_CACHE="${NORMAL_CACHE:-64}"
# Match trained folds on multi_dataset predictor (f1/f4/f8/f16).
LOOKAHEADS="${LOOKAHEADS:-1 4 8 16}"
DOMAINS="${DOMAINS:-wikitext fineweb orca gsm8k cnn_dailymail}"
PRED="${PRED:-trainingData/qwen3_30b/multi_dataset/transformer_emb_markov_pfill_mixed}"
PREDS="${PREDS:-}"
REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
BUDGET_FRACTIONS="${BUDGET_FRACTIONS:-0.5}"
CACHE_LOOKAHEAD_SLACK="${CACHE_LOOKAHEAD_SLACK:-3}"
OUT_ROOT="${OUT_ROOT:-py/utils/final_results_runs/expanded_prompt_space}"
EXAMPLES_JSON="${EXAMPLES_JSON:-$OUT_ROOT/examples_${NUM_PROMPTS}each.json}"
SPLIT_DIR="${SPLIT_DIR:-$OUT_ROOT/by_domain}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-$(( 3600 + NUM_PROMPTS * MAX_NEW_TOKENS * 5 ))}"
BATCH_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SKIP_EXP_RANDOM="${SKIP_EXP_RANDOM:-0}"

# shellcheck disable=SC2206
LA_ARR=($LOOKAHEADS)
# shellcheck disable=SC2206
BF_ARR=($BUDGET_FRACTIONS)
# shellcheck disable=SC2206
DOMAIN_ARR=($DOMAINS)

PRED_ARGS=()
if [[ -n "$PREDS" ]]; then
  # shellcheck disable=SC2206
  PRED_LIST=($PREDS)
  if [[ ${#PRED_LIST[@]} -gt 1 ]]; then
    PRED_ARGS=(--predictor-base-dirs "${PRED_LIST[@]}")
  else
    PRED_ARGS=(--predictor-base-dir "${PRED_LIST[0]}")
  fi
else
  PRED_ARGS=(--predictor-base-dir "$PRED")
fi

prepare_prompts() {
  mkdir -p "$OUT_ROOT" "$SPLIT_DIR"
  if [[ -f "$EXAMPLES_JSON" && -f "$SPLIT_DIR/gsm8k.json" && "${REFRESH_PROMPTS:-0}" != "1" ]]; then
    echo "[prompts] reusing $EXAMPLES_JSON (set REFRESH_PROMPTS=1 to rebuild)"
    return
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would cache examples → $EXAMPLES_JSON (+ $SPLIT_DIR)"
    return
  fi
  echo "[prompts] caching $NUM_PROMPTS example(s)/dataset → $EXAMPLES_JSON"
  set +e
  python3 py/utils/cache_eval_prompts.py \
    --out "$EXAMPLES_JSON" \
    --split-dir "$SPLIT_DIR" \
    --num-prompts "$NUM_PROMPTS" \
    --prompt-max-chars "$PROMPT_MAX_CHARS" \
    --datasets "${DOMAIN_ARR[@]}"
  local rc=$?
  set -e
  if [[ ! -f "$EXAMPLES_JSON" || ! -f "$SPLIT_DIR/gsm8k.json" ]]; then
    echo "[prompts] ERROR: cache failed (rc=$rc) and output missing" >&2
    exit 1
  fi
  if [[ $rc -ne 0 ]]; then
    echo "[prompts] warning: cache process exited rc=$rc but files look OK — continuing"
  fi
}

score_domain() {
  local domain=$1
  local out_dir=$2
  local normal_dir=$3
  local examples="$SPLIT_DIR/${domain}.json"
  local gens="$out_dir/generations.md"
  local csv="$out_dir/sweep.csv"
  local corr="$out_dir/correctness.json"
  local normal_tx="$normal_dir/transcripts.jsonl"
  if [[ ! -f "$gens" ]]; then
    echo "[score] no generations.md for $domain — skip"
    return
  fi
  local -a normal_args=()
  if [[ -f "$normal_tx" ]]; then
    normal_args=(--normal-transcripts "$normal_tx" --normal-cache "$NORMAL_CACHE")
  else
    echo "[score] warning: no normal transcripts at $normal_tx" >&2
  fi
  python3 py/utils/eval_correctness.py \
    --examples "$examples" \
    --generations "$gens" \
    --sweep-csv "$csv" \
    "${normal_args[@]}" \
    --out "$corr" || true
}

ppl_flags_for() {
  local domain=$1
  case "$domain" in
    wikitext|fineweb|orca) echo --enable-generation-perplexity ;;
  esac
}

run_normal_baseline() {
  local domain=$1
  local examples="$SPLIT_DIR/${domain}.json"
  local OUT_DIR="${OUT_ROOT}/${domain}/normal_C${NORMAL_CACHE}_${BATCH_TS}"
  local CSV="$OUT_DIR/sweep.csv"
  local LOG="$OUT_DIR/collection.log"
  mkdir -p "$OUT_DIR"

  # shellcheck disable=SC2206
  local -a PPL_FLAG=($(ppl_flags_for "$domain"))

  local -a cmd=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --examples-json "$examples"
    --domain "$domain"
    --num-prompts 0
    --prompt-max-chars "$PROMPT_MAX_CHARS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes "$NORMAL_CACHE"
    --lookaheads 1
    --cache-lookahead-slack "$CACHE_LOOKAHEAD_SLACK"
    --budget-fractions "${BF_ARR[@]}"
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --drop-page-cache-between-runs
    --out-dir "$OUT_DIR"
    --csv-file "$CSV"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
    --sweep-question custom_1_16_no_ppl
    --random-baseline-only
    --log-file "$OUT_DIR/00_normal.log"
    "${PPL_FLAG[@]}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[$domain normal C=$NORMAL_CACHE]"
    printf '  %q' "${cmd[@]}"
    echo
    return
  fi

  {
    echo "normal baseline domain=$domain C=$NORMAL_CACHE"
    echo "OUT_DIR=$OUT_DIR"
  } > "$OUT_DIR/command.txt"

  echo "=== NORMAL domain=$domain  C=$NORMAL_CACHE (RANDOM)  → $OUT_DIR ==="
  echo "[$domain normal] $(date -Is)" | tee -a "$LOG"
  "${cmd[@]}"
  echo "Normal baseline done: $OUT_DIR"
}

run_one() {
  local domain=$1
  local C=$2
  local normal_dir=$3
  local examples="$SPLIT_DIR/${domain}.json"
  local OUT_DIR="${OUT_ROOT}/${domain}/C${C}_${BATCH_TS}"
  local CSV="$OUT_DIR/sweep.csv"
  local LOG="$OUT_DIR/collection.log"
  mkdir -p "$OUT_DIR"

  # shellcheck disable=SC2206
  local -a PPL_FLAG=($(ppl_flags_for "$domain"))

  local -a COMMON=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --examples-json "$examples"
    --domain "$domain"
    --num-prompts 0
    --prompt-max-chars "$PROMPT_MAX_CHARS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes "$C"
    --lookaheads "${LA_ARR[@]}"
    --cache-lookahead-slack "$CACHE_LOOKAHEAD_SLACK"
    --budget-fractions "${BF_ARR[@]}"
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --drop-page-cache-between-runs
    "${PRED_ARGS[@]}"
    --out-dir "$OUT_DIR"
    --csv-file "$CSV"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
    "${PPL_FLAG[@]}"
  )

  run_step() {
    local name=$1
    shift
    local -a cmd=( "${COMMON[@]}" "$@" )
    if [[ -f "$CSV" ]]; then
      cmd+=(--append)
    fi
    if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
      cmd+=(--retry-failed)
    fi
    cmd+=(--log-file "$OUT_DIR/${name}.log")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[$domain C=$C $name]"
      printf '  %q' "${cmd[@]}"
      echo
      return
    fi

    echo "[$domain C=$C $name] $(date -Is)" | tee -a "$LOG"
    "${cmd[@]}"
  }

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    {
      echo "run_expanded_prompt_space.sh domain=$domain C=$C"
      echo "NORMAL_DIR=$normal_dir"
      echo "NORMAL_CACHE=$NORMAL_CACHE"
      echo "EXAMPLES=$examples"
      echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
      echo "LOOKAHEADS=${LA_ARR[*]}"
      echo "BUDGET_FRACTIONS=${BF_ARR[*]}"
      echo "PPL_FLAG=${PPL_FLAG[*]:-}"
      echo "PRED_ARGS=${PRED_ARGS[*]}"
      echo "OUT_DIR=$OUT_DIR"
    } > "$OUT_DIR/command.txt"
  fi

  echo "=== domain=$domain  C=$C  LA=${LA_ARR[*]}  → $OUT_DIR ==="

  if [[ "$SKIP_EXP_RANDOM" != "1" ]]; then
    run_step "01_random" \
      --sweep-question custom_1_16_no_ppl \
      --random-baseline-only
  fi

  run_step "02_expert_ahead" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --prefetch-only

  run_step "03_expert_ahead_cc" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --both-only \
    --lambdas 1 \
    --routing-bias-top-ns 5

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    score_domain "$domain" "$OUT_DIR" "$normal_dir"
    echo "Done $domain C=$C. CSV: $CSV  comparison: $OUT_DIR/comparison.md"
  fi
}

prepare_prompts

# Phase 1: normal operation baselines (RANDOM @ NORMAL_CACHE) — one per domain
declare -A NORMAL_DIRS=()
for domain in "${DOMAIN_ARR[@]}"; do
  run_normal_baseline "$domain"
  NORMAL_DIRS[$domain]="${OUT_ROOT}/${domain}/normal_C${NORMAL_CACHE}_${BATCH_TS}"
done

# Phase 2: experimental prefetch / both at each CACHE_SIZES
for domain in "${DOMAIN_ARR[@]}"; do
  for C in $CACHE_SIZES; do
    run_one "$domain" "$C" "${NORMAL_DIRS[$domain]}"
  done
done

echo "All runs finished under $OUT_ROOT/"
echo "Normal baselines: ${OUT_ROOT}/<domain>/normal_C${NORMAL_CACHE}_${BATCH_TS}/"
echo "Set PRED=... to the new predictor before running."
