# Utilities & Scripts

This folder contains helper scripts for environment setup, data collection, and plotting for the HeteroPredict project.

## Reproducing Tab. `end_to_end_results` (10 prompts × 150 tokens)

The paper end-to-end table combines **three** collections (all on the same oracle traces under `trainingData/wikitext_test_traces`):

| Table rows | How to collect |
|---|---|
| SSD Streaming | `FORCE_EXPERT_MISS=1` forced-miss baseline |
| Oracle Predictor | Oracle Full Union at table S: C16→2, C32/48/64→1 |
| Random / Cross-Layer / Cache-Cond / ExpertAhead / ExpertAhead-CC | `run_final_best_10prompt_oracle.sh` |

Run from the **repo root** (or `sh_scripts/`; scripts `cd` to root themselves).

### 1) SSD Streaming + Oracle Predictor

```bash
cd /home/michael/heteroPredict/sh_scripts
MAX_NEW_TOKENS=150 ./run_final_table_oracle_and_streaming.sh
```

This writes under `py/utils/final_results_runs/final_table_exact_150/`:
- `streaming_<TS>/sweep.csv` — SSD Streaming (Forced Miss Anchor)
- `C{16,32,48,64}_<TS>/sweep.csv` — Oracle Full Union at the table lookaheads

### 2) Method comparison (Random … ExpertAhead-CC)

```bash
cd /home/michael/heteroPredict/sh_scripts
OUT_ROOT=py/utils/final_results_runs/final_10prompt_oracle_150 \
MAX_NEW_TOKENS=150 \
CACHE_SIZES="16 32 48 64" \
SKIP_ORACLE=1 \
./run_final_best_10prompt_oracle.sh
```

`SKIP_ORACLE=1` avoids re-running Oracle here (already collected in step 1). Omit it if you want Oracle appended into each method `sweep.csv` as well.

Best configs encoded in the script (S/B/J):

| C | Cross-Layer B | Cache-Cond J | ExpertAhead S/B | ExpertAhead-CC S/B/J |
|---|---|---|---|---|
| 16 | 8 | 5 | 2/4 | 1/4/5 |
| 32 | 6 | 5 | 2/8 | 1/8/5 |
| 48 | 2 | 5 | 5/24 | 4/20/5 |
| 64 | 2 | 5 | 6/36 | 6/36/5 |

### 3) Build the LaTeX / CSV table

```bash
cd /home/michael/heteroPredict
python3 py/utils/build_end_to_end_table.py \
  --streaming-csv py/utils/final_results_runs/final_table_exact_150/streaming_<STREAM_TS>/sweep.csv \
  --oracle-root py/utils/final_results_runs/final_table_exact_150 \
  --oracle-suffix <STREAM_TS> \
  --methods-root py/utils/final_results_runs/final_10prompt_oracle_150 \
  --methods-suffix <METHODS_TS> \
  --out-dir py/utils/final_results_runs/final_10prompt_oracle_150
```

Outputs: `end_to_end_table.tex`, `end_to_end_table_combined.csv`, `end_to_end_table_speedups_wide.csv`.

Canonical prior run suffixes: streaming/oracle `20260715_150635`, methods `20260715_162440`.

### Dry run

```bash
DRY_RUN=1 MAX_NEW_TOKENS=150 ./sh_scripts/run_final_table_oracle_and_streaming.sh
DRY_RUN=1 CACHE_SIZE=16 MAX_NEW_TOKENS=150 ./sh_scripts/run_final_best_10prompt_oracle.sh
```

## Running Parameter Sweeps

To sweep across multiple lookaheads and prefetch budgets and find the best configuration for a specific cache size, use the `py/utils/sweep_predict_cached_cache_metrics.py` script.

You can pass multiple space-separated values to `--lookaheads` and `--prefetch-budgets`. The script will automatically test the cartesian product (every combination) of those values.

**Important:** You must include `--custom-explicit-prefetch-budgets` (and usually `--no-budget-fractions`) to use absolute budget numbers instead of the default fractions.

### Sweep Example (Cache Size 32)

```bash
python3 py/utils/sweep_predict_cached_cache_metrics.py \
    --sweep-question custom_1_16_no_ppl \
    --model qwen \
    --dataset wikitext \
    --num-prompts 10 \
    --cache-sizes 32 \
    --lookaheads 1 2 3 4 \
    --custom-explicit-prefetch-budgets \
    --no-budget-fractions \
    --prefetch-budgets 8 16 24 \
    --prefetch-only \
    --non-baseline-cache-policy LFRU \
    --csv-file "py/utils/final_results_runs/my_sweep_C32/sweep.csv"
```

Once the sweep completes, it outputs all of the configurations and their resulting TPS and hit rates to the CSV. Analyze the CSV (or use plotting scripts like `py/utils/section_5_2/plot_sec5_2.py`) to pick the highest-TPS config.

## Modeling Steady State Decode (Chapter 3 Plots)

From the repository root:

### Chapter 3.1 & 3.3: Naive Decode and Cache-Conditional Routing
```bash
python3 py/utils/modeling_cache_conditional.py
```

### Chapter 3.2: Predictive Prefetching
```bash
python3 py/plot_prefetch_theory.py
```

### Chapter 3.4: Unified Model
```bash
python3 py/plot_unified.py
```
