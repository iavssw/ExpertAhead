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
    
    # Filter to oracle baseline sweep, only the Oracle runs, and specifically Cache Size 24
    df = df[(df["question"] == "ORACLE_BASELINE_SWEEP") & (df["backend"] == "predict")].copy()
    df = df[df["cache_size"] == 24]
    
    # Clean up any partial rows or NaNs
    df = df.dropna(subset=["lookahead", "hit_rate_pct", "tokens_per_second"])
    
    # Sort by lookahead
    df.sort_values(by="lookahead", inplace=True)
    
    # Group by label (Top-B vs Full Union)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.lineplot(data=df, x="lookahead", y="hit_rate_pct", hue="label", marker="o", ax=ax1, linewidth=2, markersize=8)
    ax1.set_xlabel("Lookahead Window (Steps)")
    ax1.set_ylabel("Cache Hit Rate (%)")
    ax1.set_title("Hit Rate vs Lookahead (Cache Size = 24)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    sns.lineplot(data=df, x="lookahead", y="tokens_per_second", hue="label", marker="s", ax=ax2, linewidth=2, markersize=8)
    ax2.set_xlabel("Lookahead Window (Steps)")
    ax2.set_ylabel("Tokens Per Second (TPS)")
    ax2.set_title("Decode Speed vs Lookahead (Cache Size = 24)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.suptitle("Impact of Lookahead Window on 24-Expert Cache", fontsize=16)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
