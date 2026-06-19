#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Querying Python for PyTorch CMake path..."
# In many setups, users have PyTorch installed in python.
# We'll use the environment variable if available.
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

echo "Compiling expert_benchmark.cpp..."

g++ -O3 -std=c++17 \
    expert_benchmark.cpp \
    -I"${TORCH_PATH}/include" \
    -I"${TORCH_PATH}/include/torch/csrc/api/include" \
    -L"${TORCH_PATH}/lib" \
    -Wl,-rpath,"${TORCH_PATH}/lib" \
    -ltorch -ltorch_cpu -lc10 -lpthread \
    -o run_ssd_benchmark

echo "Compilation successful. Running benchmark (requires sudo for drop_caches)..."
echo

MIXTRAL_PACKED_DIR="/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/mixtral-8x7b-v0.1-AWQ_packed"
MIXTRAL_UNPACKED_DIR="/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/mixtral-8x7b-v0.1-AWQ_unpacked"

QWEN_PACKED_DIR="/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed"
QWEN_UNPACKED_DIR="/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_unpacked"

# Drop caches once before starting
sudo bash -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Run with sudo to ensure drop_caches works inside the benchmark
sudo ./run_ssd_benchmark "$QWEN_PACKED_DIR" "$QWEN_UNPACKED_DIR" 10
