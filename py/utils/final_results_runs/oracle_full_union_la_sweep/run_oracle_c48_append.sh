#!/usr/bin/env bash
# Append (or re-run) cache=48 rows with --cold-per-prompt to match fair per-prompt measurement.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source utils/setup.sh

OUT_DIR="py/utils/final_results_runs/oracle_full_union_la_sweep"
CSV="${OUT_DIR}/oracle_full_union_20260702_204246.csv"
LOG="${OUT_DIR}/oracle_full_union_c48_append.log"

# Drop any existing cache=48 rows so --retry-failed re-runs them.
python - <<'PY'
import pandas as pd
from pathlib import Path

csv = Path("py/utils/final_results_runs/oracle_full_union_la_sweep/oracle_full_union_20260702_204246.csv")
df = pd.read_csv(csv)
before = len(df)
df = df[df["cache_size"] != 48]
if len(df) < before:
    df.to_csv(csv, index=False)
    print(f"Removed {before - len(df)} cache=48 row(s) from {csv}")
else:
    print("No cache=48 rows to remove")
PY

python py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question oracle_baseline_sweep \
  --oracle-full-union-only \
  --no-actual-predictor \
  --model qwen \
  --dataset oracle \
  --oracle-trace-dir /home/michael/heteroPredict/trainingData/qwen_traces \
  --num-prompts 5 \
  --cache-sizes 48 \
  --lookaheads 1 2 3 4 6 \
  --max-new-tokens 64 \
  --temperature 0.0 \
  --cold-per-prompt \
  --disable-measurement \
  --out-dir "$OUT_DIR" \
  --csv-file "$CSV" \
  --log-file "$LOG" \
  --retry-failed

python py/utils/plot_oracle_cache_vs_lru.py \
  --csv "$CSV" \
  --out "${OUT_DIR}/oracle_full_union_20260702_204246_tps_vs_cache.png"

echo "Done. CSV: $CSV"
