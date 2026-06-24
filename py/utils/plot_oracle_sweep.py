import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to sweep CSV file")
    parser.add_argument("--out", required=True, help="Output image file")
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    
    # Filter to only oracle baseline sweep if mixed
    df = df[df["question"] == "ORACLE_BASELINE_SWEEP"].copy()
    
    # Sort cache sizes to ensure X axis is ordered
    df.sort_values(by="cache_size", inplace=True)
    
    # Calculate Normalized Speedup relative to LRU baseline per cache size
    lru_df = df[df["label"] == "Neither (LRU)"][["cache_size", "tokens_per_second"]].drop_duplicates()
    lru_map = dict(zip(lru_df["cache_size"], lru_df["tokens_per_second"]))
    df["speedup"] = df.apply(lambda row: row["tokens_per_second"] / lru_map.get(row["cache_size"], row["tokens_per_second"]), axis=1)

    # Create figure with 3 subplots (TPS, HitRate, Normalized Speedup)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # We want to group the Top-B models by their fraction of the cache size.
    # Label is like 'Oracle Top-B B=16'
    def get_series_name(row):
        if row["label"].startswith("Neither"):
            return row["label"]
        elif row["label"].startswith("Oracle Full Union"):
            return "Oracle Full Union"
        elif row["label"].startswith("Oracle Top-B"):
            # B=X, cache_size=C -> frac = B/C
            b = float(row["prefetch_budget"])
            c = float(row["cache_size"])
            frac = int((b / c) * 100)
            return f"Oracle Top-B (Budget = {frac}% of Cache)"
        return row["label"]
        
    df["series"] = df.apply(get_series_name, axis=1)
    
    # Colors and markers
    markers = ['o', 's', '^', 'v', 'D', 'p', '*']
    unique_series = sorted(df["series"].unique())
    
    # Plot line charts for TPS and HitRate
    for i, s in enumerate(unique_series):
        subset = df[df["series"] == s]
        if subset.empty: continue
        ax1.plot(subset["cache_size"], subset["tokens_per_second"], marker=markers[i%len(markers)], label=s, linewidth=2)
        ax2.plot(subset["cache_size"], subset["hit_rate_pct"], marker=markers[i%len(markers)], label=s, linewidth=2)
        
    ax1.set_xlabel("Cache Size (Experts)")
    ax1.set_ylabel("Tokens Per Second (TPS)")
    ax1.set_title("Decode Speed vs Cache Size")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    ax2.set_xlabel("Cache Size (Experts)")
    ax2.set_ylabel("Cache Hit Rate (%)")
    ax2.set_title("Cache Hit Rate vs Cache Size")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    # Plot bar chart for Normalized Speedup
    sns.barplot(data=df, x="cache_size", y="speedup", hue="series", ax=ax3, palette="Set2")
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.8, linewidth=2, label="LRU Baseline (1.0x)")
    ax3.set_xlabel("Cache Size (Experts)")
    ax3.set_ylabel("Normalized Speedup (vs LRU)")
    ax3.set_title("Normalized Speedup vs Cache Size")
    ax3.legend()
    
    plt.suptitle("Oracle Prefetch Budget Sweep", fontsize=16)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
