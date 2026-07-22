import pandas as pd
import os

sec1_path = 'py/utils/final_results_runs/sec1_cache_policy/20260603_024931/sweep.csv'
random_path = 'py/utils/final_final_baseline_random/random_baseline.csv'

# Backup original
os.system(f"cp {sec1_path} {sec1_path}.bak")

df_sec1 = pd.read_csv(sec1_path)
df_random = pd.read_csv(random_path)

# Drop old buggy random
df_sec1_clean = df_sec1[df_sec1['policy'] != 'RANDOM'].copy()

# Format the new fixed random to match sec1 headers
new_randoms = pd.DataFrame({
    'policy': 'RANDOM',
    'cache_size': df_random['cache_size'],
    'lambda': 0.0,
    'prefill_top_n': -1.0, # sec1 uses -1.0 for standard baselines
    'tps': df_random['tokens_per_second'],
    'cache_hit_rate': df_random['hit_rate_pct']
})

# Merge and overwrite
df_patched = pd.concat([df_sec1_clean, new_randoms], ignore_index=True)
df_patched.to_csv(sec1_path, index=False)
print("Hot-swapped fixed RANDOM data into sec1 CSV.")
