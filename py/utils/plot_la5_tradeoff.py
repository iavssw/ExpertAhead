import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    
    # Filter for the relevant rows
    lru_df = df[(df['question'] == 'ORACLE_BASELINE_SWEEP') & (df['label'] == 'Neither (LRU)')]
    la5_df = df[(df['question'] == 'ORACLE_BASELINE_SWEEP') & (df['label'] == 'Oracle Full Union LA=5')]
    
    # Group by cache size to average out any step variance
    lru_agg = lru_df.groupby('cache_size')['tokens_per_second'].mean().reset_index()
    lru_agg.rename(columns={'tokens_per_second': 'LRU_TPS'}, inplace=True)
    
    la5_agg = la5_df.groupby('cache_size')['tokens_per_second'].mean().reset_index()
    la5_agg.rename(columns={'tokens_per_second': 'LA5_TPS'}, inplace=True)
    
    merged = pd.merge(lru_agg, la5_agg, on='cache_size')
    merged['Speedup'] = merged['LA5_TPS'] / merged['LRU_TPS']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left Plot: Absolute TPS
    ax1.plot(merged['cache_size'], merged['LRU_TPS'], marker='o', linestyle='--', color='gray', label='Naive Cache (LRU)', linewidth=2)
    ax1.plot(merged['cache_size'], merged['LA5_TPS'], marker='s', linestyle='-', color='blue', label='Oracle Predictor (Lookahead=5)', linewidth=2)
    ax1.axhline(y=20.0, color='red', linestyle=':', label='Hardware Compute Ceiling (~20 TPS)')
    
    ax1.set_xlabel("Expert Cache Capacity (Experts/Layer)")
    ax1.set_ylabel("Tokens Per Second (TPS)")
    ax1.set_title("Absolute Decode Speed: Lookahead=5 vs LRU")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    # Right Plot: Relative Speedup
    sns.barplot(data=merged, x='cache_size', y='Speedup', ax=ax2, color='cornflowerblue')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel("Expert Cache Capacity (Experts/Layer)")
    ax2.set_ylabel("Speedup over LRU baseline")
    ax2.set_title("Relative Speedup vs Cache Size for Lookahead=5")
    ax2.grid(True, axis='y', linestyle="--", alpha=0.6)
    
    plt.suptitle("The Memory Constraint Tradeoff for Deep Lookaheads", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
