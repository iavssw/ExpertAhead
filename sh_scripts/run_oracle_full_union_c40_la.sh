#!/usr/bin/env bash
# Oracle Full Union @ C=40 for lookaheads 1–6 (oracle traces, cold-per-prompt).
#
# Usage:
#   ./run_oracle_full_union_c40_la.sh
#   SWEEP_CSV=path/to/sweep.csv RETRY_FAILED=1 ./run_oracle_full_union_c40_la.sh
#   DRY_RUN=1 ./run_oracle_full_union_c40_la.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
source "$REPO_ROOT/utils/setup.sh"

export HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0
export HETEROPREDICT_IO_THREADS="${HETEROPREDICT_IO_THREADS:-32}"

ORACLE_TRACE_DIR="${ORACLE_TRACE_DIR:-$REPO_ROOT/trainingData/wikitext_test_traces}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"

if [[ -n "${SWEEP_CSV:-}" ]]; then
  OUT_DIR="${RUN_DIR:-$(dirname "$SWEEP_CSV")}"
  CSV_FILE="$SWEEP_CSV"
else
  TIME=$(date +%Y%m%d_%H%M%S)
  OUT_DIR="$REPO_ROOT/py/utils/final_results_runs/oracle_full_union_c40_la/$TIME"
  CSV_FILE="$OUT_DIR/sweep.csv"
fi
mkdir -p "$OUT_DIR"

ARGS=(
  python3 py/utils/sweep_predict_cached_cache_metrics.py
  --model qwen
  --dataset oracle
  --oracle-trace-dir "$ORACLE_TRACE_DIR"
  --sweep-question oracle_baseline_sweep
  --oracle-full-union-only
  --no-actual-predictor
  --cache-sizes 40
  --lookaheads 1 2 3 4 5 6
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

if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
  ARGS+=(--retry-failed)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[oracle-c40-la] DRY RUN:"
  printf '  %q' "${ARGS[@]}"
  echo
  exit 0
fi

echo "[oracle-c40-la] Oracle Full Union @ C=40, LA=1..6 → $CSV_FILE"
"${ARGS[@]}"
