import pandas as pd

force_miss_csv = "/home/michael/heteroPredict/py/utils/final_results_runs/sec2_expert_io_ab/parallel/sec5_all_methods/new_sweep_cold.csv"
sweep_csv = "/home/michael/heteroPredict/py/utils/final_results_runs/oracle_la_vs_cache/20260705_222605/sweep.csv"

df_force = pd.read_csv(force_miss_csv)
force_tps = df_force[df_force['label'] == 'Forced Miss Anchor (C=0)']['tokens_per_second'].values[0]

df = pd.read_csv(sweep_csv)

caches = sorted(df['cache_size'].unique())

print("| Cache Size | FORCE MISS EXPERT | RANDOM | Oracle S=1 | Oracle Best S (Max) | Best S (LA) |")
print("|---|---|---|---|---|---|")

for c in caches:
    df_c = df[df['cache_size'] == c]
    
    # RANDOM
    random_tps = df_c[df_c['label'] == 'Neither (RANDOM)']['tokens_per_second']
    random_val = random_tps.values[0] if len(random_tps) > 0 else 0
    
    # Oracle S=1
    s1_tps = df_c[df_c['label'] == 'Oracle Full Union LA=1']['tokens_per_second']
    s1_val = s1_tps.values[0] if len(s1_tps) > 0 else 0
    
    # Oracle Best S
    df_oracle = df_c[df_c['label'].str.startswith('Oracle Full Union LA=')]
    if len(df_oracle) > 0:
        best_row = df_oracle.loc[df_oracle['tokens_per_second'].idxmax()]
        best_s_val = best_row['tokens_per_second']
        best_s_name = best_row['label']
    else:
        best_s_val = 0
        best_s_name = "N/A"
        
    print(f"| {c} | {force_tps:.2f} | {random_val:.2f} | {s1_val:.2f} | {best_s_val:.2f} | {best_s_name} |")
