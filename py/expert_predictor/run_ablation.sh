#!/bin/bash
# run_ablation.sh — full ablation study used for paper plots
# Generates data for: 
# 1. MLP architecture (History 1 & 4)
# 2. Transformer architecture (History 4)
# Note: The python script internally loops over all 8 feature variants (emb_only, pfill_only, etc.)

prefetch_budgets=(
  [1]=8  [2]=13 [3]=16 [4]=19
  [5]=22 [6]=25 [7]=27 [8]=29
  [9]=31 [10]=33 [11]=35 [12]=36
  [13]=38 [14]=39 [15]=41 [16]=42
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_MLP="../../trainingDataExtended/qwen3_30b/mlp_ablation_${TIMESTAMP}"
OUT_TX="../../trainingDataExtended/qwen3_30b/transformer_ablation_${TIMESTAMP}"

echo "=========================================================="
echo " Ablation Study — qwen3_30b (Paper Configuration)"
echo " Outputs: "
echo "   MLP: ${OUT_MLP}"
echo "   TX:  ${OUT_TX}"
echo "=========================================================="

# Activate ROCm/PyTorch environment
source "${SCRIPT_DIR:-../../utils}/setup.sh"

for future_steps in 1 4 8; do
    budget=${prefetch_budgets[$future_steps]}
    echo ""
    echo "=========================================================="
    echo " Ablation at future_steps=${future_steps}  prefetch_k=${budget}"
    echo "=========================================================="

    # 1. MLP: History 1 and 4
    # This generates data for 'mlp_emb_only' and 'mlp_hist4_emb_only'
    python3 expert_predictor_cross_token.py ablation \
        --model qwen3_30b \
        --data_dir "../../trainingDataExtended/qwen3_30b" \
        --predict_k 8 \
        --future_steps ${future_steps} \
        --prefetch_k ${budget} \
        --prefetch_ks 8 16 24 32 40 \
        --hiddens 32 \
        --histories 1 4 \
        --layers $(seq -s ' ' 0 47) \
        --epochs 10 \
        --batch_size 128 \
        --arch mlp \
        --output_dir "${OUT_MLP}"

    # 2. Transformer: History 4
    # This generates data for 'tx_emb_only'
    python3 expert_predictor_cross_token.py ablation \
        --model qwen3_30b \
        --data_dir "../../trainingDataExtended/qwen3_30b" \
        --predict_k 8 \
        --future_steps ${future_steps} \
        --prefetch_k ${budget} \
        --prefetch_ks 8 16 24 32 40 \
        --hiddens 32 \
        --histories 4 \
        --layers $(seq -s ' ' 0 47) \
        --epochs 10 \
        --batch_size 128 \
        --arch transformer \
        --output_dir "${OUT_TX}"
done

echo ""
echo "Ablation study complete!"
echo "Data available in:"
echo "  ${OUT_MLP}"
echo "  ${OUT_TX}"
echo ""
echo "To plot, you may need to update the directory paths in dump_table_data.py or plot_ablation.py."
