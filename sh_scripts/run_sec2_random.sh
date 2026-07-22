#!/bin/bash
set -e


# echo "1/4: Running main predictor sweep (sec2_predictor_effectiveness)..."
# python3 py/utils/finals_experiment_runner.py \
#   --experiment sec2_predictor_effectiveness \
#   --predictor-base-dir /home/michael/heteroPredict/trainingData/qwen3_30b/transformer_final_not_actually \
#   --run-dir py/utils/final_results_runs/sec2_custom_run

echo "2/4: Appending RANDOM baselines to the sweep.csv..."
python3 py/utils/finals_experiment_runner.py \
  --experiment random_baseline_collection \
  --csv-file py/utils/final_results_runs/sec2_custom_run/sweep.csv \
  --num-prompts 8

echo "3/4: Post-processing to calculate predictor metrics vs RANDOM baseline..."
python3 py/utils/analyze_predictor_speedup_attribution.py \
  --csv py/utils/final_results_runs/sec2_custom_run/sweep.csv \
  --baseline-policy RANDOM

echo "4/4: Generating the scatterplot..."
python3 py/utils/final_results_runs/sec2_custom_run/plot_sec2.py \
  --csv py/utils/final_results_runs/sec2_custom_run/predictor_speedup_attribution.csv

echo "Done! Your plot has been generated successfully."
