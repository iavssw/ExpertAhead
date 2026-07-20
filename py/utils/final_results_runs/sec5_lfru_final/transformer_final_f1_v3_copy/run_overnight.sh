#!/usr/bin/env bash
# Overnight sweep: transformer_final_f1_v3
# Predictor: trainingData/qwen3_30b/transformer_final/transformer_eh4_h64_f1
# LA=1 | LFRU | routing-bias-top-n=5 | λ=1 | budgets=0.25,0.5,0.75,1.0 | 15 prompts
# Cache sizes: 8, 16, 24, 32
#
# Run order: RANDOM baseline first, then main sweep (--retry-failed keeps RANDOM rows)
# Resume after interrupt: re-run this script as-is (--retry-failed handles dedup)

set -euo pipefail

REPO=/home/michael/heteroPredict
OUT_DIR="${REPO}/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3_copy"
CSV="${OUT_DIR}/sweep.csv"
LOG="${OUT_DIR}/run.log"

cd "${REPO}"
source utils/setup.sh

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo " Step 1/2: RANDOM baseline"
echo "============================================================"
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen --dataset wikitext \
  --cache-sizes 8 16 24 32 \
  --lookaheads 1 \
  --constraint-expert-reuse-csv py/expert_predictor/expert_reuse_qwen3_30b.csv \
  --num-prompts 15 \
  --prompt-max-chars 4096 --max-new-tokens 256 --temperature 0.0 \
  --disable-measurement \
  --drop-page-cache-between-runs \
  --random-baseline-only \
  --log-file "${LOG}" \
  --out-dir "${OUT_DIR}" \
  --csv-file "${CSV}"

echo ""
echo "============================================================"
echo " Step 2/2: Main sweep (LRU, Cache-Cond, Prefetch, Hybrid)"
echo "============================================================"
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen --dataset wikitext \
  --cache-sizes 8 16 24 32 \
  --lookaheads 1 \
  --predictor-base-dir trainingData/qwen3_30b/transformer_final/transformer_eh4_h64_f1 \
  --constraint-expert-reuse-csv py/expert_predictor/expert_reuse_qwen3_30b.csv \
  --num-prompts 15 \
  --prompt-max-chars 4096 --max-new-tokens 256 --temperature 0.0 \
  --disable-measurement \
  --drop-page-cache-between-runs \
  --non-baseline-cache-policy LFRU \
  --lambdas 1.0 \
  --budget-fractions 0.25 0.5 0.75 1.0 \
  --routing-bias-top-n 5 \
  --retry-failed \
  --log-file "${LOG}" \
  --out-dir "${OUT_DIR}" \
  --csv-file "${CSV}"

echo ""
echo "============================================================"
echo " Done! Results saved to ${CSV}"
echo "============================================================"
