#!/usr/bin/env bash
# Section 5 test run at C=24:
#   - Neither (RANDOM), Neither (LRU), Cache-Cond λ=1  — once each
#   - Prefetch Only (predictor alone) — all lookaheads × budgets (B=6,12,18)
#   - Both λ=1 (Hybrid) — same LA×B grid
#
# Usage:
#   ./run_sec5_c24_test.sh
#   NUM_PROMPTS=3 MAX_NEW_TOKENS=100 ./run_sec5_c24_test.sh
#   DRY_RUN=1 ./run_sec5_c24_test.sh
#
# Append missing rows to an existing sweep.csv:
#   SWEEP_CSV=py/utils/final_results_runs/sec5_c24_test/sec5_c24_test/<ts>/sweep.csv \
#   RUN_DIR=py/utils/final_results_runs/sec5_c24_test/sec5_c24_test/<ts> \
#   RETRY_FAILED=1 ./run_sec5_c24_test.sh
#
# Plot (RANDOM baseline, C=24):
#   source utils/setup.sh
#   python3 py/utils/plot_thesis_finals.py \
#     --csv-sec5 py/utils/final_results_runs/sec5_c24_test/<timestamp>/sweep.csv \
#     --cache-size 24 \
#     --baseline random \
#     --out-dir py/utils/final_results_runs/sec5_c24_test/plots

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "$REPO_ROOT/utils/setup.sh"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/py/utils/final_results_runs/sec5_c24_test}"
PREDICTOR_BASE="${PREDICTOR_BASE:-$REPO_ROOT/trainingData/qwen3_30b/transformer_final_pfill_markov_emb}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
ROUTING_BIAS_TOP_N="${ROUTING_BIAS_TOP_N:-6}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-}"
SWEEP_CSV="${SWEEP_CSV:-}"
RUN_DIR="${RUN_DIR:-}"

RUNNER=(
  python3 py/utils/finals_experiment_runner.py
  --experiment sec5_c24_test
  --predictor-base-dir "$PREDICTOR_BASE"
  --num-prompts "$NUM_PROMPTS"
  --prompt-max-chars "$PROMPT_MAX_CHARS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --out-root "$OUT_ROOT"
)
if [[ -n "$SUBPROCESS_TIMEOUT" ]]; then
  RUNNER+=(--subprocess-timeout "$SUBPROCESS_TIMEOUT")
fi
if [[ -n "$SWEEP_CSV" ]]; then
  RUNNER+=(--csv-file "$SWEEP_CSV")
fi
if [[ -n "$RUN_DIR" ]]; then
  RUNNER+=(--run-dir "$RUN_DIR")
fi
if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
  RUNNER+=(--retry-failed)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RUNNER+=(--dry-run)
fi

echo "[sec5-c24] predictor=$PREDICTOR_BASE"
echo "[sec5-c24] grid: 3 baselines + 9 prefetch + 9 hybrid (LA=1,2,3 × B=6,12,18)  J=$ROUTING_BIAS_TOP_N"
echo "[sec5-c24] prompts=$NUM_PROMPTS  max_new_tokens=$MAX_NEW_TOKENS"
echo "[sec5-c24] out=$OUT_ROOT"
if [[ -n "$SWEEP_CSV" ]]; then
  echo "[sec5-c24] append to $SWEEP_CSV (retry_failed=${RETRY_FAILED:-0})"
fi
echo "[sec5-c24] $(date -Is)"

"${RUNNER[@]}" -- --routing-bias-top-n "$ROUTING_BIAS_TOP_N"

echo "[sec5-c24] Done. CSV under: $OUT_ROOT/sec5_c24_test/<timestamp>/sweep.csv"
