#!/usr/bin/env bash
# Oracle full-union lookahead sweep: cache C ∈ {8,16,24,32,40,48,56,64} × LA ∈ {1,2,3,4,6}
# plus LRU/RANDOM baselines, perfect oracle (A=1.0), and 7/8 draft-model proxy (A=0.875).
#
# Thesis question: at cache C, is a 7/8 routing predictor better than adding 32 cache slots?
#
# Resume after crash / hang / reboot:
#   bash py/utils/final_results_runs/oracle_full_union_la_sweep/run_oracle_full_union_c8_64.sh
#   (reuses paths in oracle_full_union_c8_64_active.env + --retry-failed)
#
# Fresh run (new CSV):
#   FRESH=1 bash py/utils/final_results_runs/oracle_full_union_la_sweep/run_oracle_full_union_c8_64.sh
#
# Check progress:
#   bash py/utils/final_results_runs/oracle_full_union_la_sweep/status_oracle_full_union_c8_64.sh
#
# tmux:
#   tmux new -s oracle_sweep 'bash py/utils/final_results_runs/oracle_full_union_la_sweep/run_oracle_full_union_c8_64.sh'

set -euo pipefail
cd "$(dirname "$0")/../../../.."
source utils/setup.sh

OUT_DIR="py/utils/final_results_runs/oracle_full_union_la_sweep"
STATE_FILE="${OUT_DIR}/oracle_full_union_c8_64_active.env"
mkdir -p "$OUT_DIR"

CACHE_SIZES=(8 16 24 32 40 48 56 64)
LOOKAHEADS=(1 2 3 4 6)
NUM_PROMPTS="${NUM_PROMPTS:-3}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-3600}"  # per-config wall clock (hang guard)

if [[ "${FRESH:-0}" == "1" && -f "$STATE_FILE" ]]; then
  echo "FRESH=1: removing prior active run state ($STATE_FILE)"
  rm -f "$STATE_FILE"
fi

if [[ -f "$STATE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  RESUME=1
  echo "=== Resuming oracle sweep ==="
  echo "  CSV:  $CSV"
  echo "  LOG:  $LOG"
  echo "  (set FRESH=1 to start a new timestamped CSV)"
else
  RESUME=0
  TS="$(date +%Y%m%d_%H%M%S)"
  CSV="${OUT_DIR}/oracle_full_union_c8_64_${TS}.csv"
  LOG="${OUT_DIR}/oracle_full_union_c8_64_${TS}.log"
  PLOT="${OUT_DIR}/oracle_full_union_c8_64_${TS}_tps_vs_cache.png"
  TRADEOFF_PLOT="${OUT_DIR}/oracle_full_union_c8_64_${TS}_predictor_vs_cache32.png"
  cat >"$STATE_FILE" <<EOF
# Active oracle C8-64 sweep — re-source this file to resume after reboot.
TS=$TS
CSV=$CSV
LOG=$LOG
PLOT=$PLOT
TRADEOFF_PLOT=$TRADEOFF_PLOT
EOF
  echo "=== Starting new oracle sweep ==="
  echo "  CSV: $CSV"
fi

SWEEP_CMD=(
  python py/utils/sweep_predict_cached_cache_metrics.py
  --sweep-question oracle_baseline_sweep
  --oracle-full-union-only
  --no-actual-predictor
  --model qwen
  --dataset oracle
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces
  --oracle-routing-agreements 0.875 1.0
  --oracle-noise-seed 42
  --num-prompts "$NUM_PROMPTS"
  --cache-sizes "${CACHE_SIZES[@]}"
  --lookaheads "${LOOKAHEADS[@]}"
  --max-new-tokens 64
  --temperature 0.0
  --cold-per-prompt
  --disable-measurement
  --subprocess-timeout "$SUBPROCESS_TIMEOUT"
  --out-dir "$OUT_DIR"
  --csv-file "$CSV"
  --log-file "$LOG"
  --retry-failed
)

printf '%s\n' "${SWEEP_CMD[@]}" > "${OUT_DIR}/oracle_full_union_c8_64_${TS:-resume}_command.txt"
echo "Command saved to ${OUT_DIR}/oracle_full_union_c8_64_${TS:-resume}_command.txt"
echo "Cache sizes: ${CACHE_SIZES[*]}"
echo "Lookaheads:  ${LOOKAHEADS[*]}"
echo "Prompts:     $NUM_PROMPTS"
echo "Timeout:     ${SUBPROCESS_TIMEOUT}s per config"
echo "Oracle:      0.875 (7/8 draft proxy), 1.0 (perfect)"

set +e
"${SWEEP_CMD[@]}"
SWEEP_RC=$?
set -e

if [[ $SWEEP_RC -ne 0 ]]; then
  echo ""
  echo "Sweep exited with code $SWEEP_RC (hang/crash/timeout?)."
  echo "After reboot, resume with:"
  echo "  bash py/utils/final_results_runs/oracle_full_union_la_sweep/run_oracle_full_union_c8_64.sh"
  echo "Progress:"
  bash py/utils/final_results_runs/oracle_full_union_la_sweep/status_oracle_full_union_c8_64.sh || true
  exit "$SWEEP_RC"
fi

python py/utils/plot_oracle_cache_vs_lru.py \
  --csv "$CSV" \
  --out "$PLOT"

python py/utils/plot_oracle_cache_vs_lru.py \
  --csv "$CSV" \
  --tradeoff \
  --cache-delta 32 \
  --tradeoff-lookahead 1 \
  --out "$TRADEOFF_PLOT"

echo ""
echo "Done."
echo "  CSV:            $CSV"
echo "  Lookahead plot: $PLOT"
echo "  Tradeoff plot:  $TRADEOFF_PLOT"
echo "  State file:     $STATE_FILE (kept for reference; FRESH=1 for next full run)"
