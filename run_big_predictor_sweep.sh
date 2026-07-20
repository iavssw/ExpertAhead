#!/bin/bash
set -e

# Enable parallel asynchronous expert loads from the NVMe SSD
export HETEROPREDICT_SEQUENTIAL_EXPERT_IO=0

TIME=$(date +%Y%m%d_%H%M%S)
OUT_DIR="py/utils/final_results_runs/big_predictor_sweep/sweep_${TIME}"
mkdir -p "$OUT_DIR"
CSV_FILE="$OUT_DIR/sweep.csv"

echo "============================================================"
echo "Starting Big Predictor Sweep"
echo "Output Directory: $OUT_DIR"
echo "============================================================"

source utils/setup.sh

for C in 8 16 24 32 40; do
  case $C in
    8)  LAs="1 2";   Bs="2 4" ;;
    16) LAs="1 2 3"; Bs="4 8" ;;
    24) LAs="1 2 3"; Bs="6 12" ;;
    32) LAs="1 2 3"; Bs="8 16" ;;
    40) LAs="1 2 3"; Bs="10 20" ;;
  esac

  echo "Running C=$C, LA=$LAs, B=$Bs"
  
  python3 py/utils/sweep_predict_cached_cache_metrics.py \
    --model qwen \
    --dataset wikitext \
    --cache-sizes "$C" \
    --lookaheads $LAs \
    --cache-lookahead-slack 15 \
    --custom-explicit-prefetch-budgets \
    --prefetch-budgets $Bs \
    --routing-bias-top-n 5 \
    --sweep-question custom_1_16_no_ppl \
    --lambdas 0 1 \
    --cache-cond-forced-top-ns 5 \
    --predictor-base-dir trainingData/qwen3_30b/transformer_final_pfill_markov_emb/ \
    --expert-weights-dir py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed \
    --num-prompts 1 \
    --prompt-max-chars 4096 \
    --max-new-tokens 80 \
    --temperature 0.0 \
    --disable-measurement \
    --non-baseline-cache-policy LFRU \
    --drop-page-cache-between-runs \
    --out-dir "$OUT_DIR" \
    --csv-file "$CSV_FILE" \
    --append
done

echo "Sweep completed! Results saved to $CSV_FILE"
