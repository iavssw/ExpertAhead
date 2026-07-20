import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

f1 = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3/sweep.csv'
f2 = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/transformer_final_f1_v3_copy/sweep.csv'

df1 = pd.read_csv(f1)
df1['source'] = 'v3'

df2 = pd.read_csv(f2)
df2['source'] = 'v3_copy'

df = pd.concat([df1, df2], ignore_index=True)

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

sns.lineplot(data=df, x='cache_size', y='hit_rate_pct', hue='label', style='source', markers=True, ax=axes[0])
axes[0].set_title('Hit Rate % vs Cache Size')
axes[0].set_ylabel('Hit Rate (%)')
axes[0].set_xlabel('Cache Size')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.lineplot(data=df, x='cache_size', y='tokens_per_second', hue='label', style='source', markers=True, ax=axes[1])
axes[1].set_title('Tokens per Second vs Cache Size')
axes[1].set_ylabel('Tokens/s')
axes[1].set_xlabel('Cache Size')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
out_path = '/home/michael/heteroPredict/py/utils/final_results_runs/sec5_lfru_final/comparison_plot.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {out_path}")
