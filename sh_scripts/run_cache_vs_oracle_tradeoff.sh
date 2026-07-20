#!/usr/bin/env bash
# Compare oracle prefetch (100% accurate "draft") at small cache vs LRU + actual
# predictor at large cache.  Builds a cache-scaling curve (LRU at C=8..40) for
# speedup-per-MB analysis.
#
# Usage:
#   ./run_cache_vs_oracle_tradeoff.sh
#   NUM_PROMPTS=3 MAX_NEW_TOKENS=100 ./run_cache_vs_oracle_tradeoff.sh
#   DRY_RUN=1 ./run_cache_vs_oracle_tradeoff.sh
#
# Append only missing Oracle Full Union rows to an existing sweep.csv:
#   SWEEP_CSV=py/utils/final_results_runs/cache_vs_oracle_tradeoff/<ts>/sweep.csv \
#   RETRY_FAILED=1 FULL_UNION_ONLY=1 ./run_cache_vs_oracle_tradeoff.sh
#
# Analyze:
#   python3 py/utils/analyze_cache_vs_oracle_tradeoff.py \
#     --csv py/utils/final_results_runs/cache_vs_oracle_tradeoff/<ts>/sweep.csv

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "$REPO_ROOT/utils/setup.sh"

export HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0
export HETEROPREDICT_IO_THREADS="${HETEROPREDICT_IO_THREADS:-32}"

PREDICTOR_BASE="${PREDICTOR_BASE:-$REPO_ROOT/trainingData/qwen3_30b/transformer_final_pfill_markov_emb}"
ORACLE_TRACE_DIR="${ORACLE_TRACE_DIR:-$REPO_ROOT/trainingData/wikitext_test_traces}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"

if [[ -n "${SWEEP_CSV:-}" ]]; then
  OUT_DIR="${RUN_DIR:-$(dirname "$SWEEP_CSV")}"
  CSV_FILE="$SWEEP_CSV"
else
  TIME=$(date +%Y%m%d_%H%M%S)
  OUT_DIR="${OUT_ROOT:-$REPO_ROOT/py/utils/final_results_runs/cache_vs_oracle_tradeoff}/$TIME"
  CSV_FILE="$OUT_DIR/sweep.csv"
fi
mkdir -p "$OUT_DIR"

SWEEP_ARGS=(
  python3 py/utils/sweep_predict_cached_cache_metrics.py
  --model qwen
  --dataset oracle
  --oracle-trace-dir "$ORACLE_TRACE_DIR"
  --sweep-question oracle_baseline_sweep
  --cache-sizes 8 16 24 32 40
  --lookaheads 1
  --budget-fractions 1.0
  --predictor-base-dir "$PREDICTOR_BASE"
  --expert-weights-dir py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed
  --num-prompts "$NUM_PROMPTS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --cold-per-prompt
  --drop-page-cache-between-prompts
  --disable-measurement
  --temperature 0.0
  --drop-page-cache-between-runs
  --out-dir "$OUT_DIR"
  --csv-file "$CSV_FILE"
)

if [[ "${FULL_UNION_ONLY:-0}" != "1" ]]; then
  SWEEP_ARGS+=(--include-oracle-full-union)
else
  SWEEP_ARGS+=(--oracle-full-union-only)
fi

if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
  SWEEP_ARGS+=(--retry-failed)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[cache-vs-oracle] DRY RUN — would run:"
  printf '  %q' "${SWEEP_ARGS[@]}"
  echo
  exit 0
fi

echo "============================================================"
echo "Cache vs Oracle tradeoff sweep"
if [[ "${FULL_UNION_ONLY:-0}" == "1" ]]; then
  echo "  Mode: Oracle Full Union only (append)"
else
  echo "  Oracle Top-B + Full Union + Actual Predictor @ C=8,16,24,32,40 (B=C, LA=1)"
  echo "  LRU baseline @ each C (for speedup/MB curve)"
fi
echo "  Output: $CSV_FILE"
echo "  retry_failed=${RETRY_FAILED:-0}"
echo "  prompts=$NUM_PROMPTS  max_new_tokens=$MAX_NEW_TOKENS"
echo "============================================================"

"${SWEEP_ARGS[@]}"

echo "Done. Analyze with:"
echo "  python3 py/utils/analyze_cache_vs_oracle_tradeoff.py --csv $CSV_FILE"
