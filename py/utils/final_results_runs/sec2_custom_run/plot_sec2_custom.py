import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to predictor_speedup_attribution.csv")
    parser.add_argument("--out", default="sec2_custom_plot.png")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found")
        return

    df = pd.read_csv(args.csv)

    # Filter for cache sizes 16 and 24
    df = df[df['cache_size'].isin([16, 24])].copy()

    if df.empty:
        print("No data found for cache sizes 16 and 24 in the CSV.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))

    # Plot speedup vs hit rate
    scatter = sns.scatterplot(
        data=df,
        x="hit_rate_pct",
        y="tps_speedup_vs_base",
        style="cache_size",
        hue="prefetch_budget",
        palette="viridis",
        s=150,
        edgecolor="w",
        linewidth=0.5
    )

    plt.title("Speedup over RANDOM for Cache Sizes 16 & 24")
    plt.xlabel("Cache Hit Rate (%)")
    plt.ylabel("Normalized TPS (Speedup vs RANDOM Baseline)")
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Prefetch & Cache Size")
    plt.tight_layout()
    
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()
