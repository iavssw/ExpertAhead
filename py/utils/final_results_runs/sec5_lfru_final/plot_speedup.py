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
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=speedups, x='speedup', y='label', hue='label', legend=False, palette='viridis')
plt.title(f'Speedup relative to RANDOM (Cache Size={cache_sz})')
plt.xlabel('Speedup (x)')
plt.ylabel('Configuration')

for i, v in enumerate(speedups['speedup']):
    ax.text(v + 0.02, i, f'{v:.2f}x', color='black', va='center')

plt.tight_layout()
out_path = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/speedup_bar_plot.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved bar plot to {out_path}")
