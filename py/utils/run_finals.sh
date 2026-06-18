#!/usr/bin/env bash
# Thesis results sweeps, in order:
#
#   sec1_cache_policy             -> section 1 (eviction policies × cache size)
#   sec2_predictor_effectiveness  -> sections 2 and 3 (run before 4/5 for ideal N,B)
#   sec4_routing_topj_vs_pm       -> section 4
#   sec5_all_methods              -> section 5
#
# Usage:
#   DRY_RUN=1 ./py/utils/run_finals.sh
#   nohup ./py/utils/run_finals.sh >> py/utils/final_results_runs/run_finals.log 2>&1 &

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
# Child sweeps use sys.executable — must be the ROCm venv Python.
# shellcheck source=/dev/null
source "$REPO_ROOT/utils/setup.sh"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/py/utils/final_results_runs}"
PREDICTOR_BASE="${PREDICTOR_BASE:-$REPO_ROOT/trainingData/qwen3_30b/final_predictor}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-}"
# Space-separated experiment keys to skip (e.g. SKIP="sec4_routing_topj_vs_pm").
SKIP="${SKIP:-}"

RUNNER=(python3 py/utils/finals_experiment_runner.py
  --out-root "$OUT_ROOT"
  --predictor-base-dir "$PREDICTOR_BASE"
  --num-prompts "$NUM_PROMPTS"
  --prompt-max-chars "$PROMPT_MAX_CHARS"
  --max-new-tokens "$MAX_NEW_TOKENS"
)
if [[ -n "$SUBPROCESS_TIMEOUT" ]]; then
  RUNNER+=(--subprocess-timeout "$SUBPROCESS_TIMEOUT")
fi

run_exp() {
  local key="$1"
  if [[ " $SKIP " == *" $key "* ]]; then
    echo "[finals] Skipping $key"
    return 0
  fi
  echo ""
  echo "================================================================"
  echo "[finals] $key at $(date -Is)"
  echo "================================================================"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    "${RUNNER[@]}" --dry-run --experiment "$key"
    return $?
  fi
  "${RUNNER[@]}" --experiment "$key"
  local rc=$?
  [[ $rc -ne 0 ]] && echo "[finals] WARNING: $key exited $rc" >&2
  return $rc
}

echo "[finals] NUM_PROMPTS=${NUM_PROMPTS} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} (smoke: NUM_PROMPTS=2)"
echo "[finals] Out: $OUT_ROOT | predictor: $PREDICTOR_BASE | skip: ${SKIP:-none}"

FAIL=0
run_exp sec1_cache_policy || FAIL=$((FAIL + 1))
run_exp sec2_predictor_effectiveness || FAIL=$((FAIL + 1))
run_exp sec4_routing_topj_vs_pm || FAIL=$((FAIL + 1))
run_exp sec5_all_methods || FAIL=$((FAIL + 1))

echo "[finals] Done $(date -Is); failure count=$FAIL"
exit "$FAIL"
