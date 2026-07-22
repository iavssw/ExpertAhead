#!/bin/bash
set -e

echo "==========================================================="
echo "0. Rerunning RANDOM baseline (all cache sizes)"
echo "==========================================================="
/home/michael/heteroPredict/utils/rocmPytorch/bin/python3 py/utils/run_missing_random.py

echo ""
echo "==========================================================="
echo "1. Running missing cache sizes (24, 40, 56) for Section 1"
echo "==========================================================="
/home/michael/heteroPredict/utils/rocmPytorch/bin/python3 py/utils/run_missing_sec1.py

echo ""
echo "==========================================================="
echo "2. Running Section 5 (All Methods) for cache sizes 24 and 32"
echo "   Predictor: transformer_final_pfill_markov_emb (lookahead 1)"
echo "==========================================================="
/home/michael/heteroPredict/utils/rocmPytorch/bin/python3 py/utils/finals_experiment_runner.py \
    --experiment sec5_all_methods \
    --predictor-base-dir /home/michael/heteroPredict/trainingData/qwen3_30b/transformer_final_pfill_markov_emb/transformer_eh4_h64_f1

echo ""
echo "All runs completed successfully!"
