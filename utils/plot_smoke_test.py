import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Read the data
df = pd.read_csv('oracle_smoke_test.csv')

# Extract budget from label
def extract_budget(label):
    match = re.search(r'B=(\d+)', label)
    if match:
        return int(match.group(1))
    return None

def extract_type(label):
    if 'Oracle' in label:
        return 'Oracle'
    elif 'Actual' in label:
        return 'Actual Predictor'
    elif 'LRU' in label:
        return 'LRU'
    elif 'RANDOM' in label:
        return 'RANDOM'
    return label

df['Budget'] = df['label'].apply(extract_budget)
df['Predictor'] = df['label'].apply(extract_type)

# Separate baselines from sweeps
baselines = df[df['Budget'].isna()]
sweep_data = df[df['Budget'].notna()]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.set_theme(style="whitegrid")

# 1. Hit Rate Plot
sns.lineplot(data=sweep_data, x='Budget', y='hit_rate_pct', hue='Predictor', marker='o', ax=axes[0])
for _, row in baselines.iterrows():
    axes[0].axhline(y=row['hit_rate_pct'], linestyle='--', label=f"{row['Predictor']} Baseline")
axes[0].set_title('Cache Hit Rate vs Prefetch Budget (Cache Size = 32, Lookahead = 4)')
axes[0].set_ylabel('Hit Rate (%)')
axes[0].set_xlabel('Prefetch Budget (Experts per token)')
axes[0].legend()

# 2. TPS Plot
sns.lineplot(data=sweep_data, x='Budget', y='tokens_per_second', hue='Predictor', marker='o', ax=axes[1])
for _, row in baselines.iterrows():
    if pd.notna(row['tokens_per_second']):
        axes[1].axhline(y=row['tokens_per_second'], linestyle='--', label=f"{row['Predictor']} Baseline")
axes[1].set_title('Tokens Per Second vs Prefetch Budget')
axes[1].set_ylabel('Tokens / sec')
axes[1].set_xlabel('Prefetch Budget (Experts per token)')
axes[1].legend()

plt.tight_layout()
plt.savefig('oracle_smoke_test_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved to oracle_smoke_test_plot.png")
