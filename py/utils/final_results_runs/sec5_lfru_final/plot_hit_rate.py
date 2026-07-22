import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

f1 = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3/sweep.csv'
f2 = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3_copy/sweep.csv'

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)
df = pd.concat([df1, df2], ignore_index=True)

# Map cache size to percentage based on the data points
cache_sizes = [8, 16, 24]
paper_data = [
    {'cache_size': 8, 'label': 'MoE-Infinity', 'hit_rate_pct': 18.0},
    {'cache_size': 16, 'label': 'MoE-Infinity', 'hit_rate_pct': 39.0},
    {'cache_size': 24, 'label': 'MoE-Infinity', 'hit_rate_pct': 52.0},
    {'cache_size': 8, 'label': 'MoE-Beyond', 'hit_rate_pct': 72.0},
    {'cache_size': 16, 'label': 'MoE-Beyond', 'hit_rate_pct': 81.0},
    {'cache_size': 24, 'label': 'MoE-Beyond', 'hit_rate_pct': 84.0},
]

records = []

for cs in cache_sizes:
    df_cs = df[df['cache_size'] == cs]
    if df_cs.empty: continue
    
    mean_hr = df_cs.groupby('label')['hit_rate_pct'].mean()
    
    if 'Neither (RANDOM)' in mean_hr:
        records.append({'cache_size': cs, 'label': 'RANDOM', 'hit_rate_pct': mean_hr['Neither (RANDOM)']})
    if 'Neither (LRU)' in mean_hr:
        records.append({'cache_size': cs, 'label': 'LRU', 'hit_rate_pct': mean_hr['Neither (LRU)']})
    if 'Cache-Cond Only λ=1.0' in mean_hr:
        records.append({'cache_size': cs, 'label': 'Cache Cond Only', 'hit_rate_pct': mean_hr['Cache-Cond Only λ=1.0']})
        
    prefetch = mean_hr[mean_hr.index.str.startswith('Prefetch Only')]
    if not prefetch.empty:
        records.append({'cache_size': cs, 'label': 'Best Prefetch', 'hit_rate_pct': prefetch.max()})
        
    both = mean_hr[mean_hr.index.str.startswith('Both')]
    if not both.empty:
        records.append({'cache_size': cs, 'label': 'Best Both', 'hit_rate_pct': both.max()})

records.extend(paper_data)
plot_df = pd.DataFrame(records)

cs_to_pct = {8: '10% (8 experts)', 16: '20% (16 experts)', 24: '30% (24 experts)'}
plot_df['Capacity'] = plot_df['cache_size'].map(cs_to_pct)

hue_order = ['RANDOM', 'LRU', 'Cache Cond Only', 'Best Prefetch', 'Best Both', 'MoE-Infinity', 'MoE-Beyond']

sns.set_theme(style="whitegrid")
neutral_palette = sns.color_palette("muted", n_colors=len(hue_order))

plt.figure(figsize=(14, 8))
ax = sns.barplot(data=plot_df, x='Capacity', y='hit_rate_pct', hue='label', hue_order=hue_order,
                 palette=neutral_palette, edgecolor='0.2', linewidth=1.5)

plt.title("Cache Hit Rate Comparison", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Device Cache Capacity", fontsize=14)
plt.ylabel("Cache Hit Rate (%)", fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Method", title_fontsize='14', fontsize='12', bbox_to_anchor=(1.02, 1), loc='upper left')

# Add labels above bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=4, fontsize=10, rotation=90)

# Set ylim a bit higher so the text fits
plt.ylim(0, 110)

plt.tight_layout()
out_path = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/hit_rate_comparison_bars.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved bar plot to {out_path}")
