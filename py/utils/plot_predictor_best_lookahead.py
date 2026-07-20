#!/usr/bin/env python3
"""Plots normalized speedup of Predictor LA=1 and Predictor Best LA vs Baseline."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to sweep.csv")
    p.add_argument("--out", default=None, help="Output PNG path prefix")
    return p.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    
    df = pd.read_csv(csv_path)
    # Ensure numeric types
    df["tokens_per_second"] = pd.to_numeric(df["tokens_per_second"], errors="coerce")
    df["cache_size"] = pd.to_numeric(df["cache_size"], errors="coerce")
    df["lookahead"] = pd.to_numeric(df["lookahead"], errors="coerce")
    
    # Filter valid rows
    df = df.dropna(subset=["tokens_per_second", "cache_size"])
    df = df[df["tokens_per_second"] > 0]
    
    # Extract baseline
    baseline_df = df[df["label"] == "Neither (RANDOM)"]
    if baseline_df.empty:
        # Fallback to LRU if RANDOM is not found
        baseline_df = df[df["label"] == "Neither (LRU)"]
    if baseline_df.empty:
        print("Error: No 'Neither (RANDOM)' or 'Neither (LRU)' baseline found in CSV.")
        return
        
    baseline_df = baseline_df.groupby("cache_size")["tokens_per_second"].mean().reset_index()
    baseline_df = baseline_df.sort_values("cache_size")
    baseline_map = dict(zip(baseline_df["cache_size"], baseline_df["tokens_per_second"]))
    
    # Extract predictor runs (we'll look at the 'Both' routing method as the main predictor performance)
    pred_df = df[df["label"].str.contains("Both", na=False)].copy()
    if pred_df.empty:
        # Fallback to Prefetch Only if Both is not available
        pred_df = df[df["label"].str.contains("Prefetch Only", na=False)].copy()
        
    if pred_df.empty:
        print("Error: No predictor runs found.")
        return
        
    pred_df["speedup"] = pred_df.apply(
        lambda r: r["tokens_per_second"] / baseline_map.get(r["cache_size"], 1.0),
        axis=1
    )
    
    # Calculate LA=1 (Best budget for LA=1 at each cache size)
    la1_rows = []
    for c, grp in pred_df[pred_df["lookahead"] == 1].groupby("cache_size"):
        best_row = grp.loc[grp["speedup"].idxmax()]
        la1_rows.append({
            "cache_size": c,
            "tokens_per_second": best_row["tokens_per_second"],
            "speedup": best_row["speedup"],
            "prefetch_budget": best_row.get("prefetch_budget", np.nan)
        })
    la1_df = pd.DataFrame(la1_rows).sort_values("cache_size")
    
    # Calculate Best LA (Best budget & best lookahead at each cache size)
    best_la_rows = []
    for c, grp in pred_df.groupby("cache_size"):
        best_row = grp.loc[grp["speedup"].idxmax()]
        best_la_rows.append({
            "cache_size": c,
            "tokens_per_second": best_row["tokens_per_second"],
            "speedup": best_row["speedup"],
            "lookahead": int(best_row["lookahead"]),
            "prefetch_budget": best_row.get("prefetch_budget", np.nan)
        })
    best_la_df = pd.DataFrame(best_la_rows).sort_values("cache_size")
    
    cache_sizes = sorted(list(baseline_map.keys()))
    
    # ---------------------------------------------------------
    # Plot: LA=1 vs Best LA vs Baseline
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 7))
    
    # Plot baseline
    plt.plot(baseline_df["cache_size"], baseline_df["tokens_per_second"], marker="^", color="#333333", linestyle="--", linewidth=2, markersize=8, label="Baseline")
    
    # Plot LA=1
    plt.plot(la1_df["cache_size"], la1_df["tokens_per_second"], marker="s", color="#4C72B0", linewidth=2, markersize=8, label="Predictor (LA=1)")
    
    # Plot Best LA
    plt.plot(best_la_df["cache_size"], best_la_df["tokens_per_second"], marker="o", color="#8172B3", linewidth=2, markersize=8, label="Predictor (Best LA)")
    
    # Annotate LA=1 points (show speedup and best budget)
    for _, row in la1_df.iterrows():
        b_str = f"B={int(row['prefetch_budget'])}" if not pd.isna(row['prefetch_budget']) else ""
        plt.annotate(
            f"{row['speedup']:.2f}x\n{b_str}", 
            (row["cache_size"], row["tokens_per_second"]),
            textcoords="offset points",
            xytext=(0, -25),
            ha='center',
            fontsize=9,
            color="#4C72B0"
        )
        
    # Annotate Best LA points
    for _, row in best_la_df.iterrows():
        b_str = f"B={int(row['prefetch_budget'])}" if not pd.isna(row['prefetch_budget']) else ""
        plt.annotate(
            f"LA={int(row['lookahead'])}\n{row['speedup']:.2f}x\n{b_str}", 
            (row["cache_size"], row["tokens_per_second"]),
            textcoords="offset points",
            xytext=(0, 15),
            ha='center',
            fontsize=9,
            fontweight='bold',
            color="#8172B3"
        )

    plt.xlabel("Cache Size (Experts)")
    plt.ylabel("Tokens per Second (TPS)")
    plt.title("Predictor Throughput vs Cache Size (Best LA & Budget)")
    plt.xticks(cache_sizes)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    out_name = args.out if args.out else "predictor_best_lookahead_speedup.png"
    out_path = str(csv_path.parent / out_name)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
