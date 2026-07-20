import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

f1 = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3/sweep.csv'
f2 = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3_copy/sweep.csv'

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)
df = pd.concat([df1, df2], ignore_index=True)

# 128 experts total
cache_sizes = [8, 16, 24, 32]
total_experts = 128

paper_data = [
    {'pct_experts': 10.0, 'label': 'MoE-Infinity', 'hit_rate_pct': 18.0},
    {'pct_experts': 20.0, 'label': 'MoE-Infinity', 'hit_rate_pct': 39.0},
    {'pct_experts': 30.0, 'label': 'MoE-Infinity', 'hit_rate_pct': 52.0},
    {'pct_experts': 10.0, 'label': 'MoE-Beyond', 'hit_rate_pct': 72.0},
    {'pct_experts': 20.0, 'label': 'MoE-Beyond', 'hit_rate_pct': 81.0},
    {'pct_experts': 30.0, 'label': 'MoE-Beyond', 'hit_rate_pct': 84.0},
]

records = []

for cs in cache_sizes:
    df_cs = df[df['cache_size'] == cs]
    if df_cs.empty: continue
    
    pct_exp = (cs / total_experts) * 100.0
    mean_hr = df_cs.groupby('label')['hit_rate_pct'].mean()
    
    if 'Neither (RANDOM)' in mean_hr:
        records.append({'pct_experts': pct_exp, 'label': 'RANDOM', 'hit_rate_pct': mean_hr['Neither (RANDOM)']})
    if 'Neither (LRU)' in mean_hr:
        records.append({'pct_experts': pct_exp, 'label': 'LRU', 'hit_rate_pct': mean_hr['Neither (LRU)']})
    if 'Cache-Cond Only λ=1.0' in mean_hr:
        records.append({'pct_experts': pct_exp, 'label': 'Cache Cond Only', 'hit_rate_pct': mean_hr['Cache-Cond Only λ=1.0']})
        
    prefetch = mean_hr[mean_hr.index.str.startswith('Prefetch Only')]
    if not prefetch.empty:
        records.append({'pct_experts': pct_exp, 'label': 'Best Prefetch', 'hit_rate_pct': prefetch.max()})
        
    both = mean_hr[mean_hr.index.str.startswith('Both')]
    if not both.empty:
        records.append({'pct_experts': pct_exp, 'label': 'Best Both', 'hit_rate_pct': both.max()})

records.extend(paper_data)
plot_df = pd.DataFrame(records)

hue_order = ['RANDOM', 'LRU', 'Cache Cond Only', 'Best Prefetch', 'Best Both', 'MoE-Infinity', 'MoE-Beyond']

sns.set_theme(style="whitegrid")
neutral_palette = sns.color_palette("muted", n_colors=len(hue_order))

plt.figure(figsize=(12, 8))
ax = sns.lineplot(data=plot_df, x='pct_experts', y='hit_rate_pct', hue='label', hue_order=hue_order,
                  palette=neutral_palette, marker='o', markersize=8, linewidth=2.5)

plt.title("Cache Hit Rate vs Capacity (% of Experts)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Capacity on Device (% of Total Experts)", fontsize=14)
plt.ylabel("Cache Hit Rate (%)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Method", title_fontsize='14', fontsize='12', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.ylim(0, 105)
plt.xlim(0, 35)

plt.tight_layout()
out_path = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/hit_rate_comparison_line.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved line plot to {out_path}")
