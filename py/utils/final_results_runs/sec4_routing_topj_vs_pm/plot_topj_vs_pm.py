import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.6,
})

def plot_topj_vs_pm(csv_path: str):
    df = pd.read_csv(csv_path)

    # Filter to lookahead = 4 to avoid duplicate points, or just pick the first lookahead found
    lookahead_val = df['lookahead'].dropna().unique()[0]
    if 4 in df['lookahead'].values:
        lookahead_val = 4
    
    df = df[df['lookahead'] == lookahead_val]

    # Aggregate over multiple prompts/runs if they exist
    agg_cols = ['tokens_per_second', 'gen_perplexity', 'forced_top_n', 'prob_mass_threshold']
    df = df.groupby('label')[agg_cols].mean().reset_index()

    # Separate data into categories
    baseline_df = df[df['label'] == 'LRU Baseline'].head(1)
    
    # Calculate normalization factors
    if not baseline_df.empty:
        baseline_ppl = baseline_df.iloc[0]['gen_perplexity']
        baseline_tps = baseline_df.iloc[0]['tokens_per_second']
        
        df['gen_perplexity_norm'] = df['gen_perplexity'] / baseline_ppl
        df['tokens_per_second_norm'] = df['tokens_per_second'] / baseline_tps
    else:
        df['gen_perplexity_norm'] = df['gen_perplexity']
        df['tokens_per_second_norm'] = df['tokens_per_second']

    baseline_df = df[df['label'] == 'LRU Baseline'].head(1)
    fn_df = df[df['label'].str.contains('FN=')].sort_values('forced_top_n')
    pm_df = df[df['label'].str.contains('PM=')].sort_values('prob_mass_threshold')

    plt.figure(figsize=(10, 6))

    # Plot FN (Forced Top J)
    plt.plot(fn_df['tokens_per_second_norm'], fn_df['gen_perplexity_norm'], marker='o', linestyle='-', linewidth=2, markersize=8, label='Forced Top J')
    
    # Annotate FN points
    for _, row in fn_df.iterrows():
        plt.annotate(f"J={int(row['forced_top_n'])}", 
                     (row['tokens_per_second_norm'], row['gen_perplexity_norm']),
                     textcoords="offset points", xytext=(0, -25), ha='center', va='bottom',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    # Plot PM (Cumulative Weight Threshold)
    plt.plot(pm_df['tokens_per_second_norm'], pm_df['gen_perplexity_norm'], marker='s', linestyle='-', linewidth=2, markersize=8, label='Cumulative Weight')
    
    # Annotate PM points
    for _, row in pm_df.iterrows():
        plt.annotate(f"{row['prob_mass_threshold']}", 
                     (row['tokens_per_second_norm'], row['gen_perplexity_norm']),
                     textcoords="offset points", xytext=(0, 30), ha='center', va='top',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    # Plot Baseline once
    if not baseline_df.empty:
        baseline = baseline_df.iloc[0]
        plt.scatter(baseline['tokens_per_second_norm'], baseline['gen_perplexity_norm'], 
                    color='red', marker='*', s=200, zorder=5, label='Baseline')
        plt.annotate("Baseline", 
                     (baseline['tokens_per_second_norm'], baseline['gen_perplexity_norm']),
                     textcoords="offset points", xytext=(25,15), ha='right', color='red', fontweight='bold')

        # Add 1% perplexity degradation line (normalized PPL = 1.01)
        plt.axhline(y=1.01, color='gray', linestyle='--', linewidth=1.5, zorder=1, label='1% PPL Degradation')

    plt.title("Forced Top J vs Cumulative Weight Threshold", fontsize=16, fontweight='bold')
    plt.xlabel("Normalized Tokens Per Second (TPS / Baseline TPS)", fontsize=14)
    plt.ylabel("Normalized Perplexity (PPL / Baseline PPL)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    output_path = Path(csv_path).parent / "topj_vs_pm_plot_normalized.png"
    plt.savefig(output_path, dpi=300)
    print(f"Normalized plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Top J vs PM")
    parser.add_argument("--csv", type=str, default="20260605_033735/sweep.csv", help="Path to sweep.csv")
    args = parser.parse_args()
    
    plot_topj_vs_pm(args.csv)
