#!/bin/bash
set -eo pipefail

# Thesis latency / TPS harness: micro (ssd/h2d/compute) + macro (base / cached / predict).
# Usage: ./run_benchmarks.sh
# Optional env:
#   PREDICTOR_DIR=/path/to/predictor_dir  (required for predict section)
#   SKIP_REBUILD=1                        (skip cmake if .so already built in this env)
#   EXPERT_DIR=...                        (packed/unpacked expert bins on SSD)
#   MODEL_PATH=...                        (HF id or local safetensors for attention + base MoE)

export REPO_ROOT="${REPO_ROOT:-/home/michael/heteroPredict}"
export EXPERT_DIR="${EXPERT_DIR:-$REPO_ROOT/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed}"
export MODEL_PATH="${MODEL_PATH:-QuixiAI/Qwen3-30B-A3B-AWQ}"
export LOG_FILE="$REPO_ROOT/benchmark_results_$(date +%Y%m%d_%H%M%S).log"
export PROMPT="${PROMPT:-In a detailed essay, explain the theory of relativity and its implications on modern physics.}"

echo "Starting benchmarks. Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "  EXPERT_DIR=$EXPERT_DIR" | tee -a "$LOG_FILE"
echo "  MODEL_PATH=$MODEL_PATH" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"

source "$REPO_ROOT/utils/setup.sh"
export PYTHONPATH="$REPO_ROOT/py/unified_llm_w4a16:$REPO_ROOT/build/py/unified_llm_w4a16:${PYTHONPATH:-}"

if [[ "${SKIP_REBUILD:-0}" != "1" ]]; then
  echo -e "\n=== Rebuilding libtorch extensions (must match rocmPytorch env) ===" | tee -a "$LOG_FILE"
  cmake -S "$REPO_ROOT" -B "$REPO_ROOT/build" -DCMAKE_BUILD_TYPE=Release 2>&1 | tee -a "$LOG_FILE"
  cmake --build "$REPO_ROOT/build" -j"$(nproc)" \
    --target unified_llm_w4a16_cached_libtorch \
            unified_llm_w4a16_predict_libtorch \
            unified_llm_w4a16_base_libtorch 2>&1 | tee -a "$LOG_FILE"
fi

echo -e "\n\n=== 1. MICROBENCHMARK: Expert loading (ssd / h2d / compute) ===" | tee -a "$LOG_FILE"
cd "$REPO_ROOT/py/utils"
python3 benchmark_expert_loading.py \
  --model qwen3 \
  --bin-dir "$EXPERT_DIR" \
  --num-layers 48 --num-experts 128 \
  --cache-size 8 --num-rounds 20 \
  --mode all --verbose 2>&1 | tee -a "$LOG_FILE"

echo -e "\n\n=== 2. MACROBENCHMARK: End-to-end decode ===" | tee -a "$LOG_FILE"
cd "$REPO_ROOT/py/unified_llm_w4a16"

echo -e "\n--- BASE (all experts permanently in System RAM — TPS ceiling) ---" | tee -a "$LOG_FILE"
python3 qwen3_30B-A3B_w4a16_model.py \
  --backend base \
  --model-path "$MODEL_PATH" \
  --text "$PROMPT" \
  --max-new-tokens 128 2>&1 | tee -a "$LOG_FILE"

echo -e "\n--- CACHED (SSD -> RAM LRU, cache=24) ---" | tee -a "$LOG_FILE"
python3 qwen3_30B-A3B_w4a16_model.py \
  --backend cached \
  --max-cached-experts 24 \
  --model-path "$MODEL_PATH" \
  --expert-weights-dir "$EXPERT_DIR" \
  --text "$PROMPT" \
  --max-new-tokens 128 2>&1 | tee -a "$LOG_FILE"

if [[ -n "${PREDICTOR_DIR:-}" ]]; then
  LOOKAHEAD=2
  PREDICTOR_MODEL="${PREDICTOR_DIR}/eh1_h32_f${LOOKAHEAD}"
  echo -e "\n--- PREDICT (SSD -> RAM Prefetch, prefetch=18, lookahead=${LOOKAHEAD}, cache=24) ---" | tee -a "$LOG_FILE"
  python3 qwen3_30B-A3B_w4a16_model.py \
    --backend predict \
    --max-cached-experts 24 \
    --prefetch-experts-count 18 \
    --predictor-lookahead ${LOOKAHEAD} \
    --predictor-model "$PREDICTOR_MODEL" \
    --model-path "$MODEL_PATH" \
    --expert-weights-dir "$EXPERT_DIR" \
    --text "$PROMPT" \
    --max-new-tokens 128 2>&1 | tee -a "$LOG_FILE"
else
  echo -e "\n--- PREDICT: skipped (export PREDICTOR_DIR=/path/to/predictor_dir) ---" | tee -a "$LOG_FILE"
fi

echo -e "\n======================================================" | tee -a "$LOG_FILE"
echo "Done. Grep $LOG_FILE for:" | tee -a "$LOG_FILE"
echo "  Micro:  per-expert ms / GB/s" | tee -a "$LOG_FILE"
echo "  Macro:  TPS, Bandwidth:*AvgLoadTime*, MoE Compute:*AvgComputeTime*" | tee -a "$LOG_FILE"
