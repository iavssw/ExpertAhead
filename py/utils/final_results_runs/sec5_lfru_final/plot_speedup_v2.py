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
speedups = speedups.sort_values('speedup', ascending=False)

sns.set_theme(style="whitegrid")
neutral_palette = sns.color_palette("muted")

plt.figure(figsize=(14, 8))
ax = sns.barplot(data=speedups, x='label', y='speedup', 
                 palette=neutral_palette, edgecolor='0.2', linewidth=1.5, hue='label', legend=False)

plt.title(f"Throughput Speedup vs RANDOM (Cache Size={cache_sz})", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Configuration Policy", fontsize=14)
plt.ylabel("Speedup (x)", fontsize=14)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on bars
for i, v in enumerate(speedups['speedup']):
    ax.text(i, v + 0.05, f'{v:.2f}x', color='black', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
out_path = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/speedup_bar_plot_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved bar plot to {out_path}")
