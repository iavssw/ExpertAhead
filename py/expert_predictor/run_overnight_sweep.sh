#!/bin/bash
# run_overnight_sweep.sh
# 
# This script runs a hyperparameter sweep over Mixtral and Qwen3 models,
# testing both architectures (MLP vs Transformer), different future step horizons,
# and different Top-K prediction counts.
#
# Inside the Python script, it will additionally loop over:
#   --histories 1 2 4
#   --hiddens  128 256

set -e

# Outer variables to sweep across
MODELS=("mixtral_8x7b" "qwen3_30b")
ARCHITECTURES=("mlp" "transformer")
FUTURE_STEPS=(1 2)
PREDICT_KS=(2 4)

for MODEL in "${MODELS[@]}"; do
    for ARCH in "${ARCHITECTURES[@]}"; do
        for FUTURE in "${FUTURE_STEPS[@]}"; do
            for K in "${PREDICT_KS[@]}"; do
                
                echo "=================================================="
                echo "🚀 STARTING TARGET: MODEL=$MODEL, ARCH=$ARCH, FUTURE_STEPS=$FUTURE, PREDICT_K=$K"
                echo "=================================================="
                
                # To finish overnight, we sample layers 0 (first), 15 (middle), and 31 (late)
                # rather than training ~20,000 models across all layers.
                LAYERS="0 15 31"
                
                python expert_predictor_multi_step.py sweep \
                    --model $MODEL \
                    --arch $ARCH \
                    --future_steps $FUTURE \
                    --predict_k $K \
                    --epochs 10 \
                    --histories 1 2 4 \
                    --hiddens 128 256 \
                    --layers $LAYERS

            done
        done
    done
done

echo "=================================================="
echo "🎉 OVERNIGHT SWEEP COMPLETED SUCCESSFULLY!"
echo "=================================================="
