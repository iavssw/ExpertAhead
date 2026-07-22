#!/bin/bash
# run_sweeps.sh — Phased Expert Predictor Sweep
# =============================================
# Priority order from analysis:
#   1. Baseline (predict_k=8 apples-to-apples)
#   2. future_steps sweep at correct predict_k=8
#   3. Regularization grid (dropout x smoothing)
#   4. History sweep at winning regularization
#   5. Feature ablation
#   6. Multi-layer sweep at winning config

set -e  # Exit on any failure

source /mnt/storage/Michael/michaelg/heteroPredict/utils/setup.sh
cd /mnt/storage/Michael/michaelg/heteroPredict/py/expert_predictor

MODEL="qwen3_30b"
SMOOTHING=0.1
EPOCHS=20
PATIENCE=4
BASE_ARGS="--model $MODEL --layer 5 --predict_k 8 --epochs $EPOCHS --patience $PATIENCE --smoothing $SMOOTHING"

# =============================================
# PHASE 1: Baseline (apples-to-apples, fs=3)
# =============================================
echo ""
echo "============================================================"
echo "Phase 1: Baseline — fs=3, predict_k=8, prefetch_k=24"
echo "============================================================"
python expert_predictor_multi_step.py train $BASE_ARGS \
  --future_steps 3 --prefetch_k 24 \
  --history 2 --hidden 128

# =============================================
# PHASE 2: future_steps sweep
# (prefetch_k scaled per union size data)
# =============================================
echo ""
echo "============================================================"
echo "Phase 2: Lookahead Horizon Sweep"
echo "============================================================"
declare -A PK_MAP; PK_MAP[2]=18; PK_MAP[3]=24; PK_MAP[4]=28; PK_MAP[5]=32

for FS in 2 3 4 5; do
  PK=${PK_MAP[$FS]}
  echo ">>> fs=$FS | prefetch_k=$PK"
  python expert_predictor_multi_step.py train $BASE_ARGS \
    --future_steps $FS --prefetch_k $PK \
    --history 2 --hidden 128
done

# =============================================
# PHASE 3: Regularization grid (biggest lever)
# =============================================
echo ""
echo "============================================================"
echo "Phase 3: Regularization Grid (dropout x smoothing)"
echo "============================================================"
for DROPOUT in 0.1 0.2 0.3; do
  for SM in 0.05 0.1 0.2; do
    echo ">>> dropout=$DROPOUT | smoothing=$SM"
    python expert_predictor_multi_step.py train \
      --model $MODEL --layer 5 --predict_k 8 \
      --future_steps 3 --prefetch_k 24 \
      --history 2 --hidden 128 \
      --epochs $EPOCHS --patience $PATIENCE \
      --dropout $DROPOUT --smoothing $SM
  done
done

# =============================================
# PHASE 4: History sweep at winning reg config
# (Update BEST_DROPOUT/BEST_SMOOTH from Phase 3)
# =============================================
echo ""
echo "============================================================"
echo "Phase 4: History Sweep"
echo "============================================================"
BEST_DROPOUT=0.2   # Update this after Phase 3
BEST_SMOOTH=0.1    # Update this after Phase 3

for HIST in 1 2 3 4; do
  echo ">>> history=$HIST"
  python expert_predictor_multi_step.py train \
    --model $MODEL --layer 5 --predict_k 8 \
    --future_steps 3 --prefetch_k 24 \
    --history $HIST --hidden 128 \
    --epochs $EPOCHS --patience $PATIENCE \
    --dropout $BEST_DROPOUT --smoothing $BEST_SMOOTH
done

# =============================================
# PHASE 5: Feature Ablation
# =============================================
echo ""
echo "============================================================"
echo "Phase 5: Feature Ablation — Layers 5 & 30"
echo "============================================================"
python expert_predictor_multi_step.py ablation \
  --model $MODEL \
  --layers 5 30 \
  --future_steps 3 --prefetch_k 24 \
  --histories 2 --hiddens 128 \
  --epochs $EPOCHS --patience $PATIENCE \
  --dropout $BEST_DROPOUT --smoothing $BEST_SMOOTH

# =============================================
# PHASE 6: Multi-layer sweep at winning config
# =============================================
echo ""
echo "============================================================"
echo "Phase 6: Multi-Layer Generalization Sweep"
echo "============================================================"
python expert_predictor_multi_step.py sweep \
  --model $MODEL \
  --layers 0 5 10 20 30 40 47 \
  --future_steps 3 --prefetch_k 24 \
  --histories 2 --hiddens 128 \
  --epochs $EPOCHS --patience $PATIENCE \
  --dropout $BEST_DROPOUT --smoothing $BEST_SMOOTH

echo ""
echo "============================================================"
echo "🎉 ALL SWEEPS COMPLETE"
echo "Results in: ../../trainingData/${MODEL}/sweep_*/"
echo "============================================================"
