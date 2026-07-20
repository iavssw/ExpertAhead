import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

f1 = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3/sweep.csv'
f2 = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3_copy/sweep.csv'

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)
df = pd.concat([df1, df2], ignore_index=True)

cache_sz = 24
df_filtered = df[(df['cache_size'] == cache_sz)]

mean_tps = df_filtered.groupby('label')['tokens_per_second'].mean()

if 'Neither (RANDOM)' in mean_tps:
    baseline = mean_tps['Neither (RANDOM)']
else:
    baseline = mean_tps.min()

speedups = mean_tps / baseline
speedups = speedups.reset_index()
speedups.columns = ['label', 'speedup']

# Find the best prefetch only and best both
prefetch_only = speedups[speedups['label'].str.startswith('Prefetch Only')]
best_prefetch = prefetch_only.loc[prefetch_only['speedup'].idxmax()] if not prefetch_only.empty else None

both = speedups[speedups['label'].str.startswith('Both')]
best_both = both.loc[both['speedup'].idxmax()] if not both.empty else None

# Build the final list of labels in the requested order
final_labels = []
if 'Neither (RANDOM)' in speedups['label'].values:
    final_labels.append('Neither (RANDOM)')
if 'Neither (LRU)' in speedups['label'].values:
    final_labels.append('Neither (LRU)')
if 'Cache-Cond Only λ=1.0' in speedups['label'].values:
    final_labels.append('Cache-Cond Only λ=1.0')

if best_prefetch is not None:
    final_labels.append(best_prefetch['label'])

if best_both is not None:
    final_labels.append(best_both['label'])

# Filter and reorder
speedups = speedups.set_index('label').loc[final_labels].reset_index()

# Rename labels to match what user requested for cleaner plot
rename_map = {
    'Neither (RANDOM)': 'RANDOM',
    'Neither (LRU)': 'LRU',
    'Cache-Cond Only λ=1.0': 'Cache Cond Only',
}
if best_prefetch is not None:
    rename_map[best_prefetch['label']] = f"Best Prefetch Only\n({best_prefetch['label'].replace('Prefetch Only ', '')})"
if best_both is not None:
    rename_map[best_both['label']] = f"Best Both\n({best_both['label'].replace('Both ', '')})"

speedups['plot_label'] = speedups['label'].map(rename_map)

sns.set_theme(style="whitegrid")
neutral_palette = sns.color_palette("muted")

plt.figure(figsize=(12, 8))
ax = sns.barplot(data=speedups, x='plot_label', y='speedup', 
                 palette=neutral_palette, edgecolor='0.2', linewidth=1.5, hue='plot_label', legend=False)

plt.title(f"Throughput Speedup vs RANDOM (Cache Size={cache_sz})", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Configuration Policy", fontsize=14)
plt.ylabel("Speedup (x)", fontsize=14)
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on bars
for i, v in enumerate(speedups['speedup']):
    ax.text(i, v + 0.05, f'{v:.2f}x', color='black', ha='center', fontsize=12, fontweight='bold')

# Add some top padding so text doesn't get cut off
plt.ylim(0, speedups['speedup'].max() * 1.1)

plt.tight_layout()
out_path = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/speedup_bar_plot_v3.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved bar plot to {out_path}")
