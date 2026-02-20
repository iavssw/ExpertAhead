#!/usr/bin/env bash
# run_train.sh — Expert Predictor Training Launcher
# ==================================================
# Usage:
#   bash run_train.sh <model>          # train all layers
#   bash run_train.sh <model> <layer>  # train a single layer
#   bash run_train.sh sweep <model>    # run hidden-dim sweep
#   bash run_train.sh plot  <model>    # plot sweep/train results
#
# Models: mixtral_8x7b | mixtral_8x22b | qwen3_30b | qwen3_480b

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${SCRIPT_DIR}/../../trainingData"

# Activate ROCm/PyTorch environment (same as rest of project)
source "${SCRIPT_DIR}/../../utils/setup.sh"

MODEL="${1:-mixtral_8x7b}"
EXTRA="${2:-}"

case "$MODEL" in
  sweep)
    TARGET="${EXTRA:-mixtral_8x7b}"
    echo "=== Sweep: $TARGET ==="
    python "${SCRIPT_DIR}/sweep.py" \
        --data_dir   "${DATA_ROOT}/${TARGET}" \
        --output_dir "${DATA_ROOT}/${TARGET}/sweep_results" \
        --model      "${TARGET}" \
        --hidden_dims 256 512 1024 2048 \
        --epochs 15
    ;;

  plot)
    TARGET="${EXTRA:-mixtral_8x7b}"
    echo "=== Plot: $TARGET ==="
    # Prefer sweep_results if it exists, else predictor_models
    RESULTS_DIR="${DATA_ROOT}/${TARGET}/sweep_results"
    if [ ! -d "$RESULTS_DIR" ]; then
        RESULTS_DIR="${DATA_ROOT}/${TARGET}/predictor_models"
    fi
    python "${SCRIPT_DIR}/plot.py" \
        --sweep_dir "${RESULTS_DIR}" \
        --mode all
    ;;

  *)
    # Default: train all layers (or single layer if EXTRA is a number)
    OUT_DIR="${DATA_ROOT}/${MODEL}/predictor_models"
    if [[ -n "$EXTRA" && "$EXTRA" =~ ^[0-9]+$ ]]; then
        echo "=== Training layer $EXTRA of $MODEL ==="
        python "${SCRIPT_DIR}/train.py" \
            --data_dir   "${DATA_ROOT}/${MODEL}" \
            --output_dir "${OUT_DIR}" \
            --model      "${MODEL}" \
            --layer_idx  "${EXTRA}"
    else
        echo "=== Training all layers of $MODEL ==="
        python "${SCRIPT_DIR}/train.py" \
            --data_dir   "${DATA_ROOT}/${MODEL}" \
            --output_dir "${OUT_DIR}" \
            --model      "${MODEL}"
    fi
    ;;
esac
