#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_collect_resume.sh
#
# Resumes Qwen3-30B wikitext data collection from sample 3274 on GPU 1.
# Runs in 3 batches of 2500 to avoid OOM (process restarts between batches,
# clearing all Python/HuggingFace memory).
# Output goes to the same trainingDataExtended/qwen3_30b directory, so files
# will simply be numbered 03274–09999 and merge seamlessly with the existing ones.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b"
BATCH_SIZE=2000     # restart Python after every BATCH_SIZE samples to free RAM
START_IDX=5768      # resume after the 3274 already-collected files (0-indexed)
END_IDX=10000       # total target
MIN_TOKENS=100
MAX_TOKENS=1024

export CUDA_VISIBLE_DEVICES=1

echo "=========================================================="
echo "  Resuming Qwen3-30B collection on GPU 1"
echo "  Target range: $START_IDX → $END_IDX"
echo "  Batch size:   $BATCH_SIZE (process restarts between batches)"
echo "  Output:       $OUTPUT_DIR"
echo "=========================================================="

idx=$START_IDX
batch_num=1
while [ $idx -lt $END_IDX ]; do
    remaining=$(( END_IDX - idx ))
    count=$(( remaining < BATCH_SIZE ? remaining : BATCH_SIZE ))

    echo ""
    echo "--- Batch $batch_num: idx=$idx  count=$count ---"

    python collect_training_data_unified.py \
        --model           qwen3_30b \
        --output-dir      "$OUTPUT_DIR" \
        --num-wikitext    "$count" \
        --start-idx       "$idx" \
        --min-tokens      "$MIN_TOKENS" \
        --max-tokens      "$MAX_TOKENS" \
        --device          cuda

    idx=$(( idx + count ))
    batch_num=$(( batch_num + 1 ))
    echo "  → Batch done. Collected up to idx $idx"
done

echo ""
echo "=========================================================="
echo "  RESUME COMPLETE — all samples collected up to idx $END_IDX"
echo "=========================================================="
