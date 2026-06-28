#!/bin/bash
source utils/setup.sh

python py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question oracle_baseline_sweep \
  --cache-sizes 8 \
  --lookaheads 1 \
  --dataset oracle \
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/wikitext_test_traces \
  --num-prompts 1 \
  --oracle-full-union-only \
  --budget-fractions 1.0 \
  --csv-file wikitext_oracle_full_union.csv \
  --log-file wikitext_oracle_full_union.log
