#!/usr/bin/env bash
# Train ExpertAhead (transformer H=4, Markov + embedding + prefill) on the
# mixed Qwen3-30B traces: 200 each of wikitext, fineweb, orca, gsm8k, cnn_dailymail.
#
# Usage:
#   GPU=0 LAYERS="$(seq 0 23)"  bash run_mixed_expertahead.sh
#   GPU=1 LAYERS="$(seq 24 47)" bash run_mixed_expertahead.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/utils/setup.sh"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/trainingData/qwen3_30b_mixed_sharded}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/transformer_emb_markov_pfill_mixed}"
GPU="${GPU:-0}"
LAYERS="${LAYERS:-$(seq 0 47)}"
STRIDES="${STRIDES:-1 4 8 16}"

mkdir -p "${OUT_DIR}"
export PYTHONUNBUFFERED=1
export HIP_VISIBLE_DEVICES="${GPU}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[GPU ${GPU}] ExpertAhead mixed training"
echo "  data:    ${DATA_ROOT}"
echo "  out:     ${OUT_DIR}"
echo "  layers:  ${LAYERS}"
echo "  strides: ${STRIDES}"

for layer in ${LAYERS}; do
  DATA_DIR="${DATA_ROOT}/layer_${layer}"
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: missing ${DATA_DIR}" >&2
    exit 1
  fi
  for S in ${STRIDES}; do
    echo ""
    echo "[GPU ${GPU}] >>> layer=${layer}  S=${S}  $(date)"
    python "${SCRIPT_DIR}/expert_predictor_cross_token.py" train \
      --model         qwen3_30b \
      --data_dir      "${DATA_DIR}" \
      --output_dir    "${OUT_DIR}" \
      --layer         "${layer}" \
      --arch          transformer \
      --history       4 \
      --hidden        64 \
      --tx_layers     2 \
      --tx_heads      4 \
      --use_markov \
      --future_steps  "${S}" \
      --predict_k     8 \
      --prefetch_k    16 \
      --prefetch_ks   8 16 24 32 40 \
      --epochs        10 \
      --batch_size    128 \
      --device        cuda
  done
done

echo "[GPU ${GPU}] DONE $(date)"
