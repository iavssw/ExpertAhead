import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs='+', required=True, help="Path to sweep CSV files")
    parser.add_argument("--out", required=True, help="Output image file")
    return parser.parse_args()

def main():
    args = parse_args()
    dfs = []
    for f in args.csv:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    
    df = df[df["question"] == "ORACLE_BASELINE_SWEEP"].copy()
    df = df.dropna(subset=["tokens_per_second"])

    # Baselines
    lru_tps = df[df["label"] == "Neither (LRU)"].groupby("cache_size")["tokens_per_second"].max().to_dict()
    rnd_tps = df[df["label"] == "Neither (RANDOM)"].groupby("cache_size")["tokens_per_second"].max().to_dict()

    # Predictors
    top_b_df = df[df["label"].str.startswith("Oracle Top-B")]
    full_union_df = df[df["label"].str.startswith("Oracle Full Union")]
    
    best_top_b = top_b_df.loc[top_b_df.groupby("cache_size")["tokens_per_second"].idxmax()] if not top_b_df.empty else pd.DataFrame()
    best_full_union = full_union_df.loc[full_union_df.groupby("cache_size")["tokens_per_second"].idxmax()] if not full_union_df.empty else pd.DataFrame()
    
    plot_data = []
    
    def add_data(best_df, name):
        for _, row in best_df.iterrows():
            c = int(row["cache_size"])
            if c not in lru_tps or c not in rnd_tps: continue
            plot_data.append({
                "cache_size": c,
                "Predictor": name,
                "TPS": row["tokens_per_second"],
                "Speedup vs LRU": row["tokens_per_second"] / lru_tps[c],
                "Speedup vs RANDOM": row["tokens_per_second"] / rnd_tps[c],
                "Best LA": row["lookahead"]
            })

    add_data(best_top_b, "Oracle Top-B (Fixed Padding)")
    add_data(best_full_union, "Oracle Full Union (True Union)")

    plot_df = pd.DataFrame(plot_data).sort_values("cache_size")
    
    if plot_df.empty:
        print("No valid data to plot.")
        return

    # Figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # 1. Absolute TPS
    sns.lineplot(data=plot_df, x="cache_size", y="TPS", hue="Predictor", marker='o', ax=ax1, linewidth=2.5)
    cache_sizes = sorted(list(lru_tps.keys()))
    ax1.plot(cache_sizes, [lru_tps[c] for c in cache_sizes], label="LRU", linestyle="--", color="gray", linewidth=2)
    ax1.plot(cache_sizes, [rnd_tps[c] for c in cache_sizes], label="RANDOM", linestyle=":", color="black", linewidth=2)
    
    ax1.set_xlabel("Cache Size")
    ax1.set_ylabel("Tokens Per Second")
    ax1.set_title("Absolute Best Decode Speed")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Best Lookahead
    sns.barplot(data=plot_df, x="cache_size", y="Best LA", hue="Predictor", ax=ax2, palette="muted")
    ax2.set_xlabel("Cache Size")
    ax2.set_ylabel("Optimal Lookahead Window (Steps)")
    ax2.set_title("Best Performing Lookahead by Cache Size")
    ax2.grid(True, axis='y', linestyle="--", alpha=0.6)
    
    # 3. Speedup vs LRU
    sns.barplot(data=plot_df, x="cache_size", y="Speedup vs LRU", hue="Predictor", ax=ax3, palette="Set1")
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
    ax3.set_xlabel("Cache Size")
    ax3.set_ylabel("Speedup")
    ax3.set_title("Normalized Speedup vs LRU")
    ax3.grid(True, axis='y', linestyle="--", alpha=0.6)
    
    # 4. Speedup vs RANDOM
    sns.barplot(data=plot_df, x="cache_size", y="Speedup vs RANDOM", hue="Predictor", ax=ax4, palette="Set2")
    ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
    ax4.set_xlabel("Cache Size")
    ax4.set_ylabel("Speedup")
    ax4.set_title("Normalized Speedup vs RANDOM")
    ax4.grid(True, axis='y', linestyle="--", alpha=0.6)

    plt.suptitle("Predictor Performance Envelopes", fontsize=20, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
