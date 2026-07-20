#!/usr/bin/env bash
# Same 5-method final collection as run_final_best_10prompt.sh, with per-config
# memory cgroup limits + HWM/RSS sampling.
#
# Uses sweep_predict_cached_cache_metrics.py:
#   --track-memory          → peak_rss_mb, peak_hwm_mb, mean_rss_mb in sweep.csv
#   --cgroup-memory-limit   → systemd-run MemoryMax (no swap), oom_killed flag
#
# Default caps extrapolate the post-memfix table in sweep.py (C8–C40) to C=48,64.
# Override with MEMORY_MAX_BY_CACHE or a single MEMORY_MAX for all caches.
#
# Usage:
#   ./run_final_best_10prompt_memlimit.sh
#   CACHE_SIZE=32 ./run_final_best_10prompt_memlimit.sh
#   DRY_RUN=1 ./run_final_best_10prompt_memlimit.sh
#
# Measure HWM only (no cgroup cap) — useful before picking limits:
#   TRACK_MEMORY_ONLY=1 CACHE_SIZE=48 ./run_final_best_10prompt_memlimit.sh
#
# Optional:
#   OUT_ROOT=py/utils/final_results_runs/final_10prompt_memlimit
#   CACHE_SIZES="48 64"
#   MEMORY_MAX_BY_CACHE="16:7G,32:9G,48:11G,64:12G"
#   MEMORY_MAX=8G                    # single cap for all caches (overrides per-C table)
#   START_STEP=3

set -euo pipefail
cd "$(dirname "$0")"
source utils/setup.sh

CACHE_SIZES="${CACHE_SIZES:-${CACHE_SIZE:-16 32 48 64}}"

PRED="${PRED:-trainingData/qwen3_30b/transformer_final_pfill_markov_emb}"
REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-$(( 3600 + NUM_PROMPTS * MAX_NEW_TOKENS * 5 ))}"
BATCH_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-py/utils/final_results_runs/final_10prompt_memlimit}"

# Default caps: LRU_peak_HWM + 1280 MiB, rounded up to GiB (see LRU_PEAK_HWM_MB in sweep.py).
# C=48 forced to 11G (formula alone stays at 10G) after overnight CC −6.8% at 10G.
MEMORY_MAX_BY_CACHE="${MEMORY_MAX_BY_CACHE:-16:7G,32:9G,48:11G,64:12G}"
TRACK_MEMORY_ONLY="${TRACK_MEMORY_ONLY:-0}"

if [[ "${TRACK_MEMORY_ONLY}" == "1" ]]; then
  echo "[memlimit] TRACK_MEMORY_ONLY=1 — sampling HWM/RSS only, no cgroup MemoryMax"
elif ! command -v systemd-run >/dev/null 2>&1; then
  echo "systemd-run not found; required for cgroup MemoryMax (or set TRACK_MEMORY_ONLY=1)" >&2
  exit 1
fi

run_one_cache() {
  local C=$1
  local TS="$BATCH_TS"
  local OUT_DIR="${OUT_DIR:-$OUT_ROOT/C${C}_${TS}}"
  local CSV="${CSV:-$OUT_DIR/sweep.csv}"
  local LOG="${LOG:-$OUT_DIR/collection.log}"
  mkdir -p "$OUT_DIR"

  local XL_B CC_J EA_S EA_B EACC_S EACC_B EACC_J
  case "$C" in
    16)
      XL_B=8; CC_J=5
      EA_S=2; EA_B=4
      EACC_S=1; EACC_B=4; EACC_J=5
      ;;
    32)
      XL_B=6; CC_J=5
      EA_S=2; EA_B=8
      EACC_S=1; EACC_B=8; EACC_J=5
      ;;
    48)
      XL_B=2; CC_J=5
      EA_S=5; EA_B=24
      EACC_S=4; EACC_B=20; EACC_J=5
      ;;
    64)
      XL_B=2; CC_J=5
      EA_S=6; EA_B=36
      EACC_S=6; EACC_B=36; EACC_J=5
      ;;
    *)
      echo "No best-config table for C=$C" >&2
      exit 1
      ;;
  esac

  local -a COMMON=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --dataset wikitext
    --num-prompts "$NUM_PROMPTS"
    --prompt-max-chars "$PROMPT_MAX_CHARS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes "$C"
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --drop-page-cache-between-runs
    --track-memory
    --out-dir "$OUT_DIR"
    --csv-file "$CSV"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
  )

  if [[ "${TRACK_MEMORY_ONLY}" != "1" ]]; then
    COMMON+=(--cgroup-memory-limit)
    if [[ -n "${MEMORY_MAX:-}" ]]; then
      COMMON+=(--memory-max "$MEMORY_MAX")
    else
      COMMON+=(--memory-max-by-cache "$MEMORY_MAX_BY_CACHE")
    fi
  fi

  run_step() {
    local name=$1
    local step_num="${name%%_*}"
    shift
    if [[ -n "${START_STEP:-}" ]] && (( 10#$step_num < START_STEP )); then
      echo "[$name] skipped (START_STEP=$START_STEP)" | tee -a "$LOG"
      return
    fi
    local -a cmd=( "${COMMON[@]}" "$@" )
    if [[ -f "$CSV" ]]; then
      cmd+=(--append)
    fi
    if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
      cmd+=(--retry-failed)
    fi
    cmd+=(--log-file "$OUT_DIR/${name}.log")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[C=$C $name]"
      printf '  %q' "${cmd[@]}"
      echo
      return
    fi

    echo "[C=$C $name] $(date -Is)" | tee -a "$LOG"
    "${cmd[@]}"
  }

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    {
      echo "run_final_best_10prompt_memlimit.sh C=$C"
      echo "NUM_PROMPTS=$NUM_PROMPTS MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
      echo "TRACK_MEMORY_ONLY=$TRACK_MEMORY_ONLY"
      echo "MEMORY_MAX=${MEMORY_MAX:-}"
      echo "MEMORY_MAX_BY_CACHE=${MEMORY_MAX_BY_CACHE}"
      echo "OUT_DIR=$OUT_DIR"
    } > "$OUT_DIR/command.txt"
  fi

  echo "=== C=$C (memlimit) → $OUT_DIR ==="

  run_step "01_random" \
    --sweep-question custom_1_16_no_ppl \
    --random-baseline-only

  run_step "02_cross_layer" \
    --sweep-question gating_budget_sweep \
    --custom-explicit-prefetch-budgets \
    --prefetch-budgets "$XL_B"

  run_step "03_cache_cond" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --cache-cond-only \
    --lambdas 1 \
    --routing-bias-top-ns "$CC_J"

  run_step "04_expert_ahead" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --predictor-base-dir "$PRED" \
    --lookaheads "$EA_S" \
    --custom-explicit-prefetch-budgets \
    --no-budget-fractions \
    --prefetch-budgets "$EA_B" \
    --prefetch-only

  run_step "05_expert_ahead_cc" \
    --sweep-question custom_1_16_no_ppl \
    --skip-baselines \
    --both-only \
    --predictor-base-dir "$PRED" \
    --lookaheads "$EACC_S" \
    --custom-explicit-prefetch-budgets \
    --no-budget-fractions \
    --prefetch-budgets "$EACC_B" \
    --lambdas 1 \
    --routing-bias-top-ns "$EACC_J"

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo "Done C=$C. CSV: $CSV"
    if [[ -f "$CSV" ]] && command -v python3 >/dev/null; then
      python3 - <<PY
import pandas as pd
df = pd.read_csv("$CSV")
cols = [c for c in ("label", "tokens_per_second", "peak_hwm_mb", "peak_rss_mb", "memory_max", "oom_killed") if c in df.columns]
if cols:
    print(df[cols].to_string(index=False))
PY
    fi
  fi
}

for C in $CACHE_SIZES; do
  unset OUT_DIR CSV LOG
  run_one_cache "$C"
done
