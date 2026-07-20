#!/usr/bin/env bash
# Method comparison plots from canonical final_10prompt runs (C*_20260709_001606).
#
# Usage:
#   source utils/setup.sh
#   ./run_plot_final_10prompt.sh
#
# Optional:
#   RUN_SUFFIX=20260709_001606
#   OUT_DIR=py/utils/final_results_runs/final_10prompt

set -euo pipefail
cd "$(dirname "$0")"
source utils/setup.sh

RUN_ROOT="${RUN_ROOT:-py/utils/final_results_runs/final_10prompt}"
RUN_SUFFIX="${RUN_SUFFIX:-20260709_001606}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT}"
CACHE_SIZES="${CACHE_SIZES:-16 32 48 64}"

RUN_DIRS=()
for C in $CACHE_SIZES; do
  RUN_DIRS+=("$RUN_ROOT/C${C}_${RUN_SUFFIX}")
done

for d in "${RUN_DIRS[@]}"; do
  if [[ ! -f "$d/sweep.csv" ]]; then
    echo "Missing $d/sweep.csv" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

python3 py/utils/plot_expert_ahead_evaluation.py \
  --run-dirs "${RUN_DIRS[@]}" \
  --out-dir "$OUT_DIR" \
  --cache-sizes $CACHE_SIZES \
  --baseline thesis \
  --dpi 300

python3 py/utils/plot_predictor_only_summary.py \
  --run-dirs "${RUN_DIRS[@]}" \
  --out-dir "$OUT_DIR" \
  --dpi 300

echo ""
echo "Outputs in $OUT_DIR:"
ls -1 "$OUT_DIR"/*.png "$OUT_DIR"/*.csv 2>/dev/null | sort || true
