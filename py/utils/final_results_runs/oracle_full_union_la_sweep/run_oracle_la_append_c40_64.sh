#!/usr/bin/env bash
# Append LA=8,16 oracle rows for C ∈ {40,48,56,64} to the active C8-64 CSV.
#
# Resume after crash:
#   bash py/utils/final_results_runs/oracle_full_union_la_sweep/run_oracle_la_append_c40_64.sh
#
# tmux:
#   tmux new -s oracle_la_append 'bash py/utils/final_results_runs/oracle_full_union_la_sweep/run_oracle_la_append_c40_64.sh'

set -euo pipefail
cd "$(dirname "$0")/../../../.."
source utils/setup.sh

OUT_DIR="py/utils/final_results_runs/oracle_full_union_la_sweep"
STATE_FILE="${OUT_DIR}/oracle_full_union_c8_64_active.env"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "Missing active run state: $STATE_FILE"
  echo "Run the main sweep first or set CSV/LOG manually."
  exit 1
fi

# shellcheck disable=SC1090
source "$STATE_FILE"

CACHE_SIZES=(40 48 56 64)
LOOKAHEADS=(8 16)
NUM_PROMPTS="${NUM_PROMPTS:-3}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-3600}"
LOG="${OUT_DIR}/oracle_full_union_c8_64_${TS}_la_append.log"

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

echo "=== Appending LA=8,16 for C=40,48,56,64 ==="
echo "  CSV:  $CSV"
echo "  LOG:  $LOG"
echo "  Cache sizes: ${CACHE_SIZES[*]}"
echo "  Lookaheads:  ${LOOKAHEADS[*]}"

set +e
"${SWEEP_CMD[@]}"
SWEEP_RC=$?
set -e

if [[ $SWEEP_RC -ne 0 ]]; then
  echo ""
  echo "Append sweep exited with code $SWEEP_RC."
  echo "Resume with:"
  echo "  bash py/utils/final_results_runs/oracle_full_union_la_sweep/run_oracle_la_append_c40_64.sh"
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
