#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Querying Python for PyTorch CMake path..."
if [ -z "$HOME_LIBS" ]; then
    echo "Warning: HOME_LIBS not set, guessing paths."
    TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
else
    TORCH_PATH="${HOME_LIBS}/libtorch_7.1.0"
fi

if [ ! -d "$TORCH_PATH" ]; then
    echo "Error: Could not find LibTorch at $TORCH_PATH"
    exit 1
fi

# We need the heteroPredict includes for the expert_predictor.h
PROJECT_INC="$(dirname "$DIR")/../include"
# We also need PyTorch includes
TORCH_INC="${TORCH_PATH}/include"
TORCH_API_INC="${TORCH_PATH}/include/torch/csrc/api/include"

echo "Compiling predictor_benchmark.cpp..."
g++ -O3 -std=c++17 \
    predictor_benchmark.cpp \
    -I"${PROJECT_INC}" \
    -I"${TORCH_INC}" \
    -I"${TORCH_API_INC}" \
    -L"${TORCH_PATH}/lib" \
    -Wl,-rpath,"${TORCH_PATH}/lib" \
    -ltorch -ltorch_cpu -lc10 -lpthread \
    -o run_predictor_benchmark

echo "Compilation successful!"
echo "Running benchmark..."
echo "=========================================================="

# Find a valid model file
DEFAULT_MODEL="/home/michael/heteroPredict/trainingData/qwen3_30b/final_predictor/ablation_emb_only_hist1_h32_f1/layer_0/best_jit.pt"

if [ "$#" -ge 1 ]; then
    MODEL_PATH="$1"
else
    MODEL_PATH="$DEFAULT_MODEL"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Could not find model at $MODEL_PATH"
    echo "Usage: ./run_predictor_benchmark.sh <path_to_best_jit.pt>"
    exit 1
fi

./run_predictor_benchmark "$MODEL_PATH" 1000

echo "=========================================================="
echo "Plotting results..."
python3 plot_latency.py
