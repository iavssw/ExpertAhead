#!/bin/bash
# Sweep script to compare predictor at different lookaheads against baselines

CSV_OUT="sweep_results_$(date +%Y%m%d_%H%M%S).csv"
echo "Results will be saved to $CSV_OUT"

# Function to run the sweep for a given cache size and its corresponding lookaheads
run_sweep() {
    local cache_size=$1
    shift
    local lookaheads="$@"
    
    echo "Running sweep for Cache Size: $cache_size, Lookaheads: $lookaheads"
    
    python3 /home/michael/heteroPredict/py/utils/sweep_predict_cached_cache_metrics.py \
        --sweep-question custom_1_16_no_ppl \
        --prefetch-only \
        --num-prompts 2 \
        --budget-fractions 0.25 0.5 \
        --predictor-base-dir /home/michael/heteroPredict/trainingData/qwen3_30b/transformer_final_pfill_markov_emb \
        --cache-sizes "$cache_size" \
        --lookaheads $lookaheads \
        --csv-file "$CSV_OUT" \
        --append
}

# 8: 1,2
run_sweep 8 1 2

# 16: 1,2
run_sweep 16 1 2

# 24: 1,2,3
run_sweep 24 1 2 3

# 32: 1,2,3,4
run_sweep 32 1 2 3 4

# 40: 1,2,3,4,5
run_sweep 40 1 2 3 4 5

# 48: 1,2,3,4,5,6
run_sweep 48 1 2 3 4 5 6

# 56: 1,2,3,4,5,6
run_sweep 56 1 2 3 4 5 6

# 64: 1,2,3,4,5,7,8
run_sweep 64 1 2 3 4 5 7 8

echo "Sweep complete! Results are in $CSV_OUT"
