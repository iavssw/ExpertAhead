import pandas as pd
import os

sec5_path = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3_copy/sweep.csv'
final_random_path = '/home/michael/heteroPredict/py/utils/final_final_baseline_random/sweep.csv'
out_path = '/home/michael/heteroPredict/py/utils/final_results_runs/final_merged_lfru_with_final_random.csv'

df_sec5 = pd.read_csv(sec5_path)
df_new_random = pd.read_csv(final_random_path)

# Drop the buggy RANDOM runs from sec5
df_sec5_clean = df_sec5[df_sec5['label'] != 'Neither (RANDOM)'].copy()

# Take the new 10-prompt RANDOM runs
new_randoms = df_new_random[df_new_random['label'] == 'Neither (RANDOM)'].copy()

# Force the prompt hash to match so the analysis script can join them correctly
target_hash = 'c2c1f0bb4f74' # The hash from sec5
if 'prompt_hash' in new_randoms.columns:
    new_randoms['prompt_hash'] = target_hash

# Merge them together
merged = pd.concat([df_sec5_clean, new_randoms], ignore_index=True)
merged.to_csv(out_path, index=False)
print(f"Final Merged CSV created at {out_path}")
