import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to predictor_speedup_attribution.csv")
    parser.add_argument("--out", default="sec2_plot.png")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found")
        return

    df = pd.read_csv(args.csv)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))

    # "recall on x axis, normalized TPS on y axis, different lookaheads using different shapes, and precision with datapoint color"
    scatter = sns.scatterplot(
        data=df,
        x="pred_recall_pct",
        y="tps_speedup_vs_base",
        style="lookahead",
        hue="pred_precision_pct",
        palette="viridis",
        s=100,
        edgecolor="w",
        linewidth=0.5
    )

    plt.title("Predictor Effectiveness: Recall vs Normalized TPS")
    plt.xlabel("Predictor Recall (%)")
    plt.ylabel("Normalized TPS (Speedup vs Baseline)")
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()
