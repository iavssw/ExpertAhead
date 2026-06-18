#!/usr/bin/env bash
# Greedy continuation of a short prompt: LRU (λ=0) vs cache-conditional (λ=1, FN=5).
# Matches sec4 Cache-Cond on the cached backend (no predictor / prefetch).
#
# Usage:
#   sudo -E bash py/utils/demo_cache_cond_prompt.sh
#   PROMPT="The capital of France is" CACHE_SIZE=16 bash py/utils/demo_cache_cond_prompt.sh
#
# Optional env: PROMPT, CACHE_SIZE, MAX_NEW_TOKENS, PYTHON (default: repo rocm venv python3)

set -euo pipefail
cd "$(dirname "$0")/../.."
source utils/setup.sh

PROMPT="${PROMPT:-The capital of France is}"
CACHE_SIZE="${CACHE_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
PYTHON="${PYTHON:-python3}"
MODEL="py/unified_llm_w4a16/qwen3_30B-A3B_w4a16_model.py"

common=(
  "$PYTHON" "$MODEL"
  --backend cached
  --device cuda
  --max-cached-experts "$CACHE_SIZE"
  --cache-policy LRU
  --text "$PROMPT"
  --generate
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature 0
)

run_case() {
  local title="$1"
  shift
  echo ""
  echo "################################################################"
  echo "# $title"
  echo "# prompt: $PROMPT"
  echo "# C=$CACHE_SIZE  max_new_tokens=$MAX_NEW_TOKENS"
  echo "################################################################"
  "${common[@]}" "$@"
}

run_case "LRU baseline (λ=0, no forced-top-n)" \
  --lambda-val 0 --forced-top-n 0

run_case "Cache-conditional (λ=1.0, forced-top-n=5)" \
  --lambda-val 1.0 --forced-top-n 5

echo ""
echo "Done. Compare 'Generated text only' above."
