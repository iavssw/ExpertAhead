#!/usr/bin/env bash
# Section 2 predictor-effectiveness sweep × expert I/O mode A/B.
#
# Runs finals_experiment_runner sec2_predictor_effectiveness twice:
#   parallel  — HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0 (std::async preads within expert)
#   sequential — HETEROPREDICT_SEQUENTIAL_EXPERT_IO=1 (one pread at a time)
#
# Defaults: 8 wikitext prompts, 128 decode tokens (override via env).
#
# Usage:
#   ./run_sec2_expert_io_ab.sh
#   PREDICTOR_BASE=trainingData/qwen3_30b/transformer_final_pfill_markov_emb ./run_sec2_expert_io_ab.sh
#   DRY_RUN=1 ./run_sec2_expert_io_ab.sh
#   SKIP_SEQUENTIAL=1 ./run_sec2_expert_io_ab.sh   # parallel only
# Grid defaults (finals_experiment_runner sec2): C=8,16,24,32,40,48 × LA=1,2,3,4,6.
# Override if needed, e.g. EXTRA_SWEEP_ARGS="--cache-sizes 24 --lookaheads 1 2 3 4 6"
#
# Results:
#   py/utils/final_results_runs/sec2_expert_io_ab/parallel/sec2_predictor_effectiveness/<timestamp>/
#   py/utils/final_results_runs/sec2_expert_io_ab/sequential/sec2_predictor_effectiveness/<timestamp>/

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "$REPO_ROOT/utils/setup.sh"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/py/utils/final_results_runs/sec2_expert_io_ab}"
PREDICTOR_BASE="${PREDICTOR_BASE:-$REPO_ROOT/trainingData/qwen3_30b/transformer_final_pfill_markov_emb}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-}"
SKIP_PARALLEL="${SKIP_PARALLEL:-0}"
SKIP_SEQUENTIAL="${SKIP_SEQUENTIAL:-0}"
# shellcheck disable=SC2206
EXTRA_SWEEP_ARGS=(${EXTRA_SWEEP_ARGS:-})

RUNNER=(python3 py/utils/finals_experiment_runner.py
  --experiment sec2_predictor_effectiveness
  --predictor-base-dir "$PREDICTOR_BASE"
  --num-prompts "$NUM_PROMPTS"
  --prompt-max-chars "$PROMPT_MAX_CHARS"
  --max-new-tokens "$MAX_NEW_TOKENS"
)
if [[ -n "$SUBPROCESS_TIMEOUT" ]]; then
  RUNNER+=(--subprocess-timeout "$SUBPROCESS_TIMEOUT")
fi
if [[ ${#EXTRA_SWEEP_ARGS[@]} -gt 0 ]]; then
  RUNNER+=("${EXTRA_SWEEP_ARGS[@]}")
fi

run_sec2() {
  local label="$1"
  local sequential_io="$2"
  if [[ "$label" == "parallel" && "$SKIP_PARALLEL" == "1" ]]; then
    echo "[sec2-io-ab] Skipping parallel (SKIP_PARALLEL=1)"
    return 0
  fi
  if [[ "$label" == "sequential" && "$SKIP_SEQUENTIAL" == "1" ]]; then
    echo "[sec2-io-ab] Skipping sequential (SKIP_SEQUENTIAL=1)"
    return 0
  fi

  export HETEROPREDICT_SEQUENTIAL_EXPERT_IO="$sequential_io"
  local run_out="$OUT_ROOT/$label"

  echo ""
  echo "================================================================"
  echo "[sec2-io-ab] $label  HETEROPREDICT_SEQUENTIAL_EXPERT_IO=$sequential_io"
  echo "[sec2-io-ab] prompts=$NUM_PROMPTS  max_new_tokens=$MAX_NEW_TOKENS"
  echo "[sec2-io-ab] predictor=$PREDICTOR_BASE"
  echo "[sec2-io-ab] out=$run_out"
  echo "[sec2-io-ab] $(date -Is)"
  echo "================================================================"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    "${RUNNER[@]}" --dry-run --out-root "$run_out"
    return 0
  fi

  "${RUNNER[@]}" --out-root "$run_out"
}

echo "[sec2-io-ab] Section 2 A/B: parallel (0) vs sequential (1) expert SSD I/O"
echo "[sec2-io-ab] Base out: $OUT_ROOT"

FAIL=0
run_sec2 parallel 0 || FAIL=$((FAIL + 1))
run_sec2 sequential 1 || FAIL=$((FAIL + 1))

echo ""
echo "[sec2-io-ab] Done $(date -Is); failure count=$FAIL"
echo "[sec2-io-ab] CSVs under: $OUT_ROOT/{parallel,sequential}/sec2_predictor_effectiveness/"
exit "$FAIL"
