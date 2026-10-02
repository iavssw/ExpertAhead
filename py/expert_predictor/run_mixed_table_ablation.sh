#!/usr/bin/env bash
# Train the paper Table 1 / Table 2 ablation variants on mixed Qwen3-30B traces.
# Same protocol as the WikiText tables: layers 0, 23, 47; S in {1,4,8,16}; B=16.
#
#   GPU=0 TASKS=mlp  bash run_mixed_table_ablation.sh
#   GPU=1 TASKS=tx   bash run_mixed_table_ablation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/utils/setup.sh"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/trainingData/qwen3_30b_mixed_sharded}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/paper_tables_mixed}"
GPU="${GPU:-0}"
TASKS="${TASKS:-all}"   # mlp | tx | tx_h1 | all
LAYERS="${LAYERS:-0 23 47}"
STRIDES="${STRIDES:-1 4 8 16}"

mkdir -p "${OUT_DIR}"
export PYTHONUNBUFFERED=1
export HIP_VISIBLE_DEVICES="${GPU}"
export CUDA_VISIBLE_DEVICES="${GPU}"

train_one() {
  local layer="$1" S="$2" arch="$3" history="$4" hidden="$5"
  local use_markov="$6" use_pfill="$7" custom_name="$8"
  local data_dir="${DATA_ROOT}/layer_${layer}"
  local extra=()
  if [[ "${use_markov}" == "1" ]]; then extra+=(--use_markov); fi
  if [[ "${use_pfill}" == "0" ]]; then extra+=(--no-use_pfill); fi

  echo ""
  echo "[GPU ${GPU}] >>> ${custom_name}  layer=${layer}  S=${S}  $(date)"
  python "${SCRIPT_DIR}/expert_predictor_cross_token.py" train \
    --model         qwen3_30b \
    --data_dir      "${data_dir}" \
    --output_dir    "${OUT_DIR}" \
    --layer         "${layer}" \
    --arch          "${arch}" \
    --history       "${history}" \
    --hidden        "${hidden}" \
    --tx_layers     2 \
    --tx_heads      4 \
    --future_steps  "${S}" \
    --predict_k     8 \
    --prefetch_k    16 \
    --prefetch_ks   8 16 24 32 40 \
    --epochs        10 \
    --batch_size    128 \
    --device        cuda \
    --custom_name   "${custom_name}" \
    "${extra[@]}"
}

echo "[GPU ${GPU}] mixed paper-table ablation  tasks=${TASKS}"
echo "  data: ${DATA_ROOT}"
echo "  out:  ${OUT_DIR}"

for layer in ${LAYERS}; do
  for S in ${STRIDES}; do
    if [[ "${TASKS}" == "mlp" || "${TASKS}" == "all" ]]; then
      train_one "${layer}" "${S}" mlp 1 32 0 0 "ablation_emb_only_hist1_h32_f${S}"
      train_one "${layer}" "${S}" mlp 4 32 0 0 "ablation_emb_only_hist4_h32_f${S}"
    fi
    if [[ "${TASKS}" == "tx" || "${TASKS}" == "all" ]]; then
      train_one "${layer}" "${S}" transformer 4 64 0 0 "ablation_emb_only_hist4_h64_f${S}"
      train_one "${layer}" "${S}" transformer 4 64 1 0 "ablation_emb_markov_hist4_h64_f${S}"
      train_one "${layer}" "${S}" transformer 4 64 1 1 "ablation_emb_markov_pfill_hist4_h64_f${S}"
    fi
    if [[ "${TASKS}" == "tx_h1" ]]; then
      train_one "${layer}" "${S}" transformer 1 64 0 0 "ablation_emb_only_hist1_h64_f${S}"
    fi
    if [[ "${TASKS}" == "tx_h1_ablation" ]]; then
      train_one "${layer}" "${S}" transformer 1 64 1 0 "ablation_emb_markov_hist1_h64_f${S}"
      train_one "${layer}" "${S}" transformer 1 64 1 1 "ablation_emb_markov_pfill_hist1_h64_f${S}"
    fi
  done
done

echo "[GPU ${GPU}] DONE $(date)"
