#!/usr/bin/env bash
# Predictor parameter search (paper §4.2 / Fig. 5 protocol) for a new predictor at several C.
#
# Per cache size: Prefetch Only (λ=0) over stride S × prefetch budget B, plus RANDOM + LRU
# baselines. Records TPS, window recall ρ and window precision π. LFRU for non-baselines.
# Default protocol matches the paper: 5 WikiText-103 test prompts, 80 generated tokens.
#
# Default grids (override with S_<C>="..." B_<C>="..."):
#   C=16: S=1,2,3,4,6        × B=4,6,8,12            (20)
#   C=32: S=1,2,3,4,6,8      × B=4,8,12,16,20        (30)
#   C=48: S=1,2,3,4,5,6,8    × B=10,16,20,24,32      (35; S≤5 = paper Fig. 5 grid)
#   C=64: S=2,3,4,5,6,8,10,12 × B=24,28,32,36,40,48  (48)
#
# Usage (tmux):
#   ./sh_scripts/run_predictor_param_sweep.sh
#   DRY_RUN=1 ./sh_scripts/run_predictor_param_sweep.sh
#   LIST_CONFIGS=1 ./sh_scripts/run_predictor_param_sweep.sh     # print configs, no GPU
#   CACHE_SIZES="48" ./sh_scripts/run_predictor_param_sweep.sh
#   RUN_TS=<ts> RETRY_FAILED=1 ./sh_scripts/run_predictor_param_sweep.sh   # resume
#
# Optional:
#   PRED=...                 predictor base dir (default: multi-dataset predictor)
#   EXAMPLES_JSON=...        use structured examples instead of WikiText (e.g. mixed domains)
#   NUM_PROMPTS=5 MAX_NEW_TOKENS=80
#
# Outputs: $OUT_ROOT/<ts>/sweep_C<C>.csv (+ logs); plots/summary via
#   python3 py/utils/section_5_2/plot_param_sweep.py --run-dir $OUT_ROOT/<ts>

set -euo pipefail
cd "$(dirname "$0")/.."
source utils/setup.sh

PRED="${PRED:-trainingData/qwen3_30b/multi_dataset/transformer_emb_markov_pfill_mixed}"
REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
CACHE_SIZES="${CACHE_SIZES:-16 32 48 64}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-7200}"
EXAMPLES_JSON="${EXAMPLES_JSON:-}"
TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-py/utils/final_results_runs/predictor_param_sweep}"
OUT_DIR="$OUT_ROOT/$TS"

S_16="${S_16:-1 2 3 4 6}";          B_16="${B_16:-4 6 8 12}"
S_32="${S_32:-1 2 3 4 6 8}";        B_32="${B_32:-4 8 12 16 20}"
S_48="${S_48:-1 2 3 4 5 6 8}";      B_48="${B_48:-10 16 20 24 32}"
S_64="${S_64:-2 3 4 5 6 8 10 12}";  B_64="${B_64:-24 28 32 36 40 48}"

dry() { [[ "${DRY_RUN:-0}" == "1" ]]; }
listing() { [[ "${LIST_CONFIGS:-0}" == "1" ]]; }

if listing; then
  OUT_DIR="$(mktemp -d)"
elif ! dry; then
  mkdir -p "$OUT_DIR"
fi

if [[ -n "$EXAMPLES_JSON" ]]; then
  PROMPT_ARGS=(--examples-json "$EXAMPLES_JSON")
else
  PROMPT_ARGS=(--dataset wikitext)
fi

run_cache() {
  local C=$1
  local s_var="S_$C" b_var="B_$C"
  if [[ -z "${!s_var:-}" || -z "${!b_var:-}" ]]; then
    echo "No grid for C=$C — set S_$C=\"...\" and B_$C=\"...\"" >&2
    exit 1
  fi
  # shellcheck disable=SC2206
  local -a S_ARR=(${!s_var})
  # shellcheck disable=SC2206
  local -a B_ARR=(${!b_var})
  local csv="$OUT_DIR/sweep_C${C}.csv"

  local -a cmd=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --sweep-question custom_1_16_no_ppl
    --model qwen
    --predictor-base-dir "$PRED"
    "${PROMPT_ARGS[@]}"
    --num-prompts "$NUM_PROMPTS"
    --prompt-max-chars "$PROMPT_MAX_CHARS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes "$C"
    --lookaheads "${S_ARR[@]}"
    --custom-explicit-prefetch-budgets
    --no-budget-fractions
    --prefetch-budgets "${B_ARR[@]}"
    --prefetch-only
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --drop-page-cache-between-runs
    --out-dir "$OUT_DIR"
    --csv-file "$csv"
    --log-file "$OUT_DIR/sweep_C${C}.log"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
  )
  [[ -f "$csv" ]] && cmd+=(--append)
  [[ "${RETRY_FAILED:-0}" == "1" ]] && cmd+=(--retry-failed)
  listing && cmd+=(--list-configs)

  echo "=== C=$C  S=${S_ARR[*]}  B=${B_ARR[*]}  (${#S_ARR[@]}×${#B_ARR[@]} prefetch + RANDOM + LRU) ==="
  if dry; then
    printf '  %q' "${cmd[@]}"; echo
    return
  fi
  echo "[param-sweep] C=$C started $(date -Is)"
  "${cmd[@]}"
  echo "[param-sweep] C=$C done $(date -Is) → $csv"
}

if ! dry && ! listing; then
  {
    echo "run_predictor_param_sweep.sh  started $(date -Is)"
    echo "PRED=$PRED"
    echo "PROMPTS=${EXAMPLES_JSON:-wikitext} NUM_PROMPTS=$NUM_PROMPTS MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
    for C in $CACHE_SIZES; do s_var="S_$C"; b_var="B_$C"; echo "C=$C S=${!s_var} B=${!b_var}"; done
  } > "$OUT_DIR/command.txt"
  exec > >(tee -a "$OUT_DIR/collection.log") 2>&1
fi

for C in $CACHE_SIZES; do
  run_cache "$C"
done

if listing; then
  rm -rf "$OUT_DIR"
elif ! dry; then
  python3 py/utils/section_5_2/plot_param_sweep.py --run-dir "$OUT_DIR" || true
  echo "All done: $OUT_DIR  (summary: $OUT_DIR/best_configs.md)"
fi
