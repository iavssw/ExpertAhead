import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the CSV
csv_path = "py/utils/final_results_runs/sec2_expert_io_ab/parallel/sec5_all_methods/oracle_hybrid_cache_32_la_1_b8_yahhhhh.csv"
df = pd.read_csv(csv_path)

# Filter for Lookahead 1 (to drop the LA=2 rows)
df = df[df['lookahead'] == 1].copy()

# We only want: Baseline (RANDOM), Oracle Top-B B=8, Actual Predictor B=8, Actual Predictor B=16
wanted_labels = [
    'Neither (RANDOM)',
    'Oracle Top-B B=8',
    'Actual Predictor B=8'
]
df = df[df['label'].isin(wanted_labels)]

# Rename RANDOM to Baseline
df.loc[df['label'].str.contains('RANDOM'), 'label'] = 'Baseline'

# Ensure the order matches our list
df['label_cat'] = pd.Categorical(df['label'], categories=['Baseline', 'Oracle Top-B B=8', 'Actual Predictor B=8'], ordered=True)
df = df.sort_values('label_cat')

# Extract data
labels = df['label'].tolist()
tps = df['tokens_per_second'].tolist()
hit_rate = df['hit_rate_pct'].tolist()

baseline_tps = tps[0] # Baseline is first

# Colors: Baseline (Blue), Oracle (Orange), Actual B=8 (Green), Actual B=16 (Red)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# --- Plot 1: TPS ---
plt.figure(figsize=(10, 6))
bars_tps = plt.bar(labels, tps, color=colors)
plt.title('Tokens Per Second (TPS) by Method (Cache=32, Lookahead=1)', fontsize=14)
plt.ylabel('Tokens Per Second (TPS)', fontsize=12)
plt.xlabel('Method', fontsize=12)
plt.xticks(rotation=15, ha='right')

# Add values on top of bars (relative speedup)
for bar in bars_tps:
    yval = bar.get_height()
    speedup = yval / baseline_tps
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{speedup:.2f}x", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
tps_output_path = "/home/michael/heteroPredict/py/tps_bar_plot_yahhh.png"
os.makedirs(os.path.dirname(tps_output_path), exist_ok=True)
plt.savefig(tps_output_path, dpi=300)
plt.close()

# --- Plot 2: Hit Rate ---
plt.figure(figsize=(10, 6))
bars_hit = plt.bar(labels, hit_rate, color=colors)
plt.title('Cache Hit Rate by Method (Cache=32, Lookahead=1)', fontsize=14)
plt.ylabel('Hit Rate (%)', fontsize=12)
plt.xlabel('Method', fontsize=12)
plt.xticks(rotation=15, ha='right')
plt.ylim(0, 110) # Leave room for annotations

# Add values on top of bars (raw percentage)
for bar in bars_hit:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
hitrate_output_path = "/home/michael/heteroPredict/py/tps_bar_plot_yahhh_hit_rate_tho.png"
plt.savefig(hitrate_output_path, dpi=300)
plt.close()

print(f"Plots saved to {tps_output_path} and {hitrate_output_path}")
