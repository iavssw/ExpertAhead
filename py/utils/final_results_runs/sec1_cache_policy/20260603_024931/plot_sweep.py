import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def plot_sweep(csv_path):
    df = pd.read_csv(csv_path)
    
    # Filter out CLOCK policy
    df = df[df['policy'] != 'CLOCK']
    
    # Set style for a professional look
    sns.set_theme(style="whitegrid")
    
    # Use a neutral, professional color palette
    neutral_palette = sns.color_palette("muted")
    
    # Plot 1: Hit Rate vs Policy
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='policy', y='cache_hit_rate', hue='cache_size', 
                palette=neutral_palette, edgecolor='0.2', linewidth=1)
        
    plt.title("Cache Hit Rate by Policy", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Policy", fontsize=14)
    plt.ylabel("Cache Hit Rate (%)", fontsize=14)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.legend(title="Cache Size", title_fontsize='13', fontsize='12', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    out_name = str(csv_path).replace('.csv', '_hit_rate.png')
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {out_name}")
    
    # Plot 2: TPS vs Policy
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='policy', y='tps', hue='cache_size', 
                palette=neutral_palette, edgecolor='0.2', linewidth=1)
        
    plt.title("Throughput (TPS) by Policy", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Policy", fontsize=14)
    plt.ylabel("Throughput (TPS)", fontsize=14)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.legend(title="Cache Size", title_fontsize='13', fontsize='12', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    out_name = str(csv_path).replace('.csv', '_tps.png')
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help="Path to sweep.csv")
    args = parser.parse_args()
    
    plot_sweep(Path(args.csv))
