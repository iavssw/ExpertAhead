#!/usr/bin/env bash
# Generate ExpertAhead evaluation + predictor-only summary plots from all sweeps.
#
# Usage:
#   source utils/setup.sh
#   ./run_plot_expert_ahead_all.sh
#
# Optional:
#   FINAL_SEC5_DIR=py/utils/final_results_runs/final_sec5_collection/<ts>
#   OUT_DIR=py/utils/final_results_runs/expert_ahead_plots

set -euo pipefail
cd "$(dirname "$0")"
source utils/setup.sh

ROOT="py/utils/final_results_runs"
FINAL_SEC5_DIR="${FINAL_SEC5_DIR:-$ROOT/final_sec5_collection/20260707_002948}"
OUT_DIR="${OUT_DIR:-$ROOT/expert_ahead_plots}"

RUN_DIRS=(
  "$ROOT/C8_comparison_20260703_132626"
  "$ROOT/C16_comparison_20260703_154927"
  "$ROOT/C24_comparison_20260704_120612"
  "$ROOT/C32_comparison_20260704_162727"
  "$ROOT/rerun_C32_cc_J5"
  "$ROOT/C40_comparison_20260704_220143"
  "$FINAL_SEC5_DIR"
  "$ROOT/C48_comparison_20260706_202859"
  "$ROOT/C56_refine_20260706_221747"
  "$ROOT/C56_comparison_20260706_202859"
  "$ROOT/C64_comparison_20260706_202859"
)

mkdir -p "$OUT_DIR"

python3 py/utils/plot_expert_ahead_evaluation.py \
  --run-dirs "${RUN_DIRS[@]}" \
  --out-dir "$OUT_DIR" \
  --cache-sizes 16 32 48 64 \
  --baseline thesis \
  --dpi 300

python3 py/utils/plot_predictor_only_summary.py \
  --run-dirs "${RUN_DIRS[@]}" \
  --out-dir "$OUT_DIR" \
  --dpi 300

echo ""
echo "Plots in $OUT_DIR:"
ls -1 "$OUT_DIR"/*.png "$OUT_DIR"/*.csv 2>/dev/null || true
