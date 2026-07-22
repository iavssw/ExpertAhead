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
    
    # Filter for Oracle Full Union LA=5
    la5_df = df[(df['question'] == 'ORACLE_BASELINE_SWEEP') & (df['label'] == 'Oracle Full Union LA=5')]
    
    # Group by cache size
    agg = la5_df.groupby('cache_size').agg({
        'tokens_per_second': 'mean',
        'prefetch_loads': 'mean',
        'stall_loads': 'mean'
    }).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for loads
    width = 3.0
    ax1.bar(agg['cache_size'] - width/2, agg['prefetch_loads'], width, label='Hidden Prefetch Loads', color='lightsteelblue', edgecolor='black')
    ax1.bar(agg['cache_size'] + width/2, agg['stall_loads'], width, label='Blocking Stall Loads', color='salmon', edgecolor='black')
    
    ax1.set_xlabel('Expert Cache Capacity (Experts/Layer)', fontsize=12)
    ax1.set_ylabel('Total Expert Loads (over sweep window)', fontsize=12)
    ax1.set_xticks(agg['cache_size'])
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Line plot for TPS on a secondary Y-axis
    ax2 = ax1.twinx()
    ax2.plot(agg['cache_size'], agg['tokens_per_second'], marker='D', markersize=8, color='green', linewidth=3, label='Decode Throughput (TPS)')
    ax2.set_ylabel('Tokens Per Second (TPS)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 22)
    
    # Add Hardware Ceiling line
    ax2.axhline(y=20.0, color='darkgreen', linestyle=':', linewidth=2, label='Hardware Compute Ceiling')
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.title('The Overhead of Hidden Loads (Oracle Lookahead=5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
