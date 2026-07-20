#!/usr/bin/env bash
# Cross-layer gating budget sweep at C=64 (for thesis method comparison).
#
# Usage:
#   ./run_gating_c64.sh
#   DRY_RUN=1 ./run_gating_c64.sh
#
# Append into C64 comparison sweep (default):
#   CSV=py/utils/final_results_runs/C64_comparison_20260706_202859/sweep.csv \
#   OUT_DIR=$(dirname "$CSV") ./run_gating_c64.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source utils/setup.sh

REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
C=64
CSV="${CSV:-py/utils/final_results_runs/C64_comparison_20260706_202859/sweep.csv}"
OUT_DIR="${OUT_DIR:-$(dirname "$CSV")}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-7200}"
GATING_B=(2 4 6 8)

EXTRA=()
if [[ -f "$CSV" ]]; then
  EXTRA+=(--append)
fi
if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
  EXTRA+=(--retry-failed)
fi

CMD=(
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
  echo "[gating-c64] C=$C B=${GATING_B[*]} (4 configs) -> $CSV"
  printf '  %q' "${CMD[@]}"
  echo
  exit 0
fi

echo "[gating-c64] C=$C Cross-Layer B=${GATING_B[*]} prompts=$NUM_PROMPTS -> $CSV"
"${CMD[@]}"
