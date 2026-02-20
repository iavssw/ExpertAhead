#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_collect_training_data.sh
#
# Collect per-layer embeddings + router logits for expert prediction training.
# Run this script from the py/unified_llm_w4a16 directory so that the Python
# model wrappers (mixtral_8x7B_w4a16_model.py, qwen3_30B-A3B_w4a16_model.py)
# are importable.
#
# Usage:
#   bash run_collect_training_data.sh [mixtral_8x7b|mixtral_8x22b|qwen3_30b|qwen3_480b|small|all]
#
# small (default) = mixtral_8x7b + qwen3_30b sequentially
# all             = all four models sequentially
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Output directories ────────────────────────────────────────────────────────
MIXTRAL_8x7B_OUT="/mnt/storage/Michael/michaelg/heteroPredict/trainingData/mixtral_8x7b"
QWEN3_30B_OUT="/mnt/storage/Michael/michaelg/heteroPredict/trainingData/qwen3_30b"
MIXTRAL_8x22B_OUT="/mnt/storage/Michael/michaelg/heteroPredict/trainingData/mixtral_8x2b"
QWEN3_480B_OUT="/mnt/storage/Michael/michaelg/heteroPredict/trainingData/qwen3_480b"


# ── Collection parameters ─────────────────────────────────────────────────────
NUM_FINEWEB=100
NUM_ORCA=100
MIN_TOKENS=100
MAX_TOKENS=512
DEVICE="cuda"

COLLECT_MIXTRAL_8x7B=false
COLLECT_MIXTRAL_8x22B=false
COLLECT_QWEN3_30B=false
COLLECT_QWEN3_480B=false

TARGET="${1:-small}"
case "$TARGET" in
    mixtral_8x7b)  COLLECT_MIXTRAL_8x7B=true ;;
    mixtral_8x22b) COLLECT_MIXTRAL_8x22B=true ;;
    qwen3_30b)     COLLECT_QWEN3_30B=true ;;
    qwen3_480b)    COLLECT_QWEN3_480B=true ;;
    small)         COLLECT_MIXTRAL_8x7B=true; COLLECT_QWEN3_30B=true ;;
    all)           COLLECT_MIXTRAL_8x7B=true; COLLECT_MIXTRAL_8x22B=true; COLLECT_QWEN3_30B=true; COLLECT_QWEN3_480B=true ;;
    *)
        echo "Usage: $0 [mixtral_8x7b|mixtral_8x22b|qwen3_30b|qwen3_480b|small|all]"
        exit 1
        ;;
esac

# ── Mixtral 8x7B ─────────────────────────────────────────────────────────────
if $COLLECT_MIXTRAL_8x7B; then
    echo "========================================================"
    echo "  Collecting Mixtral 8x7B training data"
    echo "========================================================"
    python collect_training_data_unified.py \
        --model           mixtral_8x7b \
        --output-dir      "$MIXTRAL_8x7B_OUT" \
        --num-fineweb     "$NUM_FINEWEB" \
        --num-orca        "$NUM_ORCA" \
        --min-tokens      "$MIN_TOKENS" \
        --max-tokens      "$MAX_TOKENS" \
        --device          "$DEVICE"
    echo "Mixtral collection done → $MIXTRAL_8x7B_OUT"
fi


# ── Mixtral 8x22B ─────────────────────────────────────────────────────────────
if $COLLECT_MIXTRAL_8x22B; then
    echo "========================================================"
    echo "  Collecting Mixtral 8x22B training data"
    echo "========================================================"
    python collect_training_data_unified.py \
        --model           mixtral_8x22b \
        --output-dir      "$MIXTRAL_8x22B_OUT" \
        --num-fineweb     "$NUM_FINEWEB" \
        --num-orca        "$NUM_ORCA" \
        --min-tokens      "$MIN_TOKENS" \
        --max-tokens      "$MAX_TOKENS" \
        --device          "$DEVICE"
    echo "Mixtral collection done → $MIXTRAL_8x22B_OUT"
fi

# ── Qwen3 30B-A3B ────────────────────────────────────────────────────────────
if $COLLECT_QWEN3_30B; then
    echo "========================================================"
    echo "  Collecting Qwen3 30B-A3B training data"
    echo "========================================================"
    python collect_training_data_unified.py \
        --model           qwen3_30b \
        --output-dir      "$QWEN3_30B_OUT" \
        --num-fineweb     "$NUM_FINEWEB" \
        --num-orca        "$NUM_ORCA" \
        --min-tokens      "$MIN_TOKENS" \
        --max-tokens      "$MAX_TOKENS" \
        --device          "$DEVICE"
    echo "Qwen3 collection done → $QWEN3_30B_OUT"
fi

# ── Qwen3 480B ───────────────────────────────────────────────────────────────
if $COLLECT_QWEN3_480B; then
    echo "========================================================"
    echo "  Collecting Qwen3 480B training data"
    echo "========================================================"
    python collect_training_data_unified.py \
        --model           qwen3_480b \
        --output-dir      "$QWEN3_480B_OUT" \
        --num-fineweb     "$NUM_FINEWEB" \
        --num-orca        "$NUM_ORCA" \
        --min-tokens      "$MIN_TOKENS" \
        --max-tokens      "$MAX_TOKENS" \
        --device          "$DEVICE"
    echo "Qwen3 collection done → $QWEN3_480B_OUT"
fi

echo "========================================================"
echo "  ALL DONE"
echo "========================================================"
