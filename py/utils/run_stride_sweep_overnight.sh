#!/usr/bin/env bash
# Overnight experiment: does predictor invoke stride affect cache hit rate / TPS?
# Decouples checkpoint depth (eh1_h32_fN) from --predictor-lookahead (invoke stride).
#
# Usage (from repo root, after setup.sh / build):
#   ./py/utils/run_stride_sweep_overnight.sh
#
# Optional env:
#   PREDICTOR_BASE_DIR  — directory containing eh1_h32_fN subdirs
#   EXPERT_WEIGHTS_DIR  — packed/unpacked expert bins
#   NUM_PROMPTS         — wikitext chunks (default 8)
#   MAX_NEW_TOKENS      — decode length per prompt (default 128)
#   CACHE_SIZE          — experts per layer (default 24)
#   PREFETCH_BUDGET     — explicit budget; else 75% of cache

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/home/michael/heteroPredict}"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/utils/setup.sh" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/utils/setup.sh"
fi
export PYTHONPATH="$REPO_ROOT/py/unified_llm_w4a16:$REPO_ROOT/build/py/unified_llm_w4a16:${PYTHONPATH:-}"

PREDICTOR_BASE_DIR="${PREDICTOR_BASE_DIR:-$REPO_ROOT/trainingData/qwen3_30b/final_multi_input_model}"
EXPERT_WEIGHTS_DIR="${EXPERT_WEIGHTS_DIR:-$REPO_ROOT/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CACHE_SIZE="${CACHE_SIZE:-24}"
MODEL_DEPTHS="${MODEL_DEPTHS:-4 8}"
STRIDES="${STRIDES:-1 2 4 8}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/py/utils/stride_sweep_runs}"
TS="$(date +%Y%m%d_%H%M%S)"
CSV_FILE="$OUT_DIR/stride_sweep_${TS}.csv"
LOG_FILE="$OUT_DIR/stride_sweep_${TS}.log"

mkdir -p "$OUT_DIR"

# B=12 (50% of cache=24) matched prior wikitext wins; B=18 over-prefetches and loses to LRU.
BUDGET_ARGS=(--custom-explicit-prefetch-budgets --prefetch-budgets 12)
if [[ -n "${PREFETCH_BUDGET:-}" ]]; then
  BUDGET_ARGS=(--custom-explicit-prefetch-budgets --prefetch-budgets "$PREFETCH_BUDGET")
fi

echo "Stride sweep starting at $(date -Is)"
echo "  CSV:      $CSV_FILE"
echo "  Log:      $LOG_FILE"
echo "  Models:   f${MODEL_DEPTHS// / f} under $PREDICTOR_BASE_DIR"
echo "  Strides:  ${STRIDES}"
echo "  Cache:    $CACHE_SIZE  prompts: $NUM_PROMPTS  max_new_tokens: $MAX_NEW_TOKENS"

python3 "$REPO_ROOT/py/utils/sweep_predict_cached_cache_metrics.py" \
  --sweep-question stride_sweep \
  --model qwen \
  --cache-sizes "$CACHE_SIZE" \
  --stride-sweep-model-depths $MODEL_DEPTHS \
  --predictor-strides $STRIDES \
  --predictor-base-dir "$PREDICTOR_BASE_DIR" \
  --expert-weights-dir "$EXPERT_WEIGHTS_DIR" \
  "${BUDGET_ARGS[@]}" \
  --dataset wikitext \
  --num-prompts "$NUM_PROMPTS" \
  --prompt-max-chars 2048 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --csv-file "$CSV_FILE" \
  --out-dir "$OUT_DIR" \
  --log-file "$LOG_FILE" \
  --subprocess-timeout "$((MAX_NEW_TOKENS * NUM_PROMPTS * 3 + 1200))" \
  2>&1 | tee -a "$LOG_FILE"

echo "Done at $(date -Is). Plots in $OUT_DIR (stride_sweep_C${CACHE_SIZE}_*.png)"
