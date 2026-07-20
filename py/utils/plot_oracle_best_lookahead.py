#!/usr/bin/env python3
"""Plots normalized speedup of Oracle LA=1 and Oracle Best LA vs Baseline."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    p.add_argument("--out", default=None, help="Output PNG path")
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
        print("Error: No 'Neither (RANDOM)' baseline found in CSV.")
        return
    baseline_df = baseline_df.groupby("cache_size")["tokens_per_second"].mean().reset_index()
    baseline_df = baseline_df.sort_values("cache_size")
    baseline_map = dict(zip(baseline_df["cache_size"], baseline_df["tokens_per_second"]))
    
    # Extract oracle
    oracle_df = df[df["label"].str.startswith("Oracle Full Union")].copy()
    if oracle_df.empty:
        print("Error: No 'Oracle Full Union' rows found.")
        return
        
    oracle_df["speedup"] = oracle_df.apply(
        lambda r: r["tokens_per_second"] / baseline_map.get(r["cache_size"], 1.0),
        axis=1
    )
    
    # ---------------------------------------------------------
    # Theoretical Model Calculation
    # ---------------------------------------------------------
    T_ceil = 49.0
    L = 48
    K = 8
    E = L * K
    N = 1  # Oracle best lookahead acts effectively like stride=1 for prefetching window
    
    bench_df = pd.read_csv('/home/michael/heteroPredict/py/expert_loading_study/fast_ssd/expert_loading_benchmark.csv')
    bench_df = bench_df[bench_df['Config'].str.contains('Parallel')].sort_values('Num_Experts')
    benchmark_m = np.concatenate(([0], bench_df['Num_Experts'].values))
    benchmark_t_per_expert = np.concatenate(([0.0], bench_df['Time_per_Expert_ms'].values))
    
    def get_T_ssd(n_misses_per_layer):
        return np.interp(n_misses_per_layer, benchmark_m, benchmark_t_per_expert)

    # Get LRU hit rates to compute theoretical TPS
    hit_rate_map = df[df["label"] == "Neither (RANDOM)"].groupby("cache_size")["hit_rate_pct"].mean().to_dict()
    cache_sizes = sorted(list(baseline_map.keys()))
    
    theoretical_rows = []
    for c in cache_sizes:
        h = hit_rate_map.get(c, 0.0) / 100.0
        U = E * (1 - h)
        n_misses_per_layer = K * (1 - h)
        T_ssd = get_T_ssd(n_misses_per_layer)
        
        M_cap = (N * T_ceil) / T_ssd if T_ssd > 0 else float('inf')
        C_total = L * (c / L)  # c is max experts PER LAYER according to code... wait, is c total or per layer?
        # In HeteroPredict, cache_size is experts per layer! So C_total = L * c.
        M_prefetch = min(U, M_cap, L * c)
        E_prefetch = M_prefetch / N
        
        T_oracle = T_ceil + (U - E_prefetch) * T_ssd
        T_oracle = max(T_ceil, T_oracle)
        TPS_theoretical = 1000.0 / T_oracle
        
        theoretical_rows.append({
            "cache_size": c,
            "tokens_per_second": TPS_theoretical,
            "speedup": TPS_theoretical / baseline_map.get(c, 1.0)
        })
    theoretical_df = pd.DataFrame(theoretical_rows)
    # ---------------------------------------------------------

    # Calculate LA=1
    la1_df = oracle_df[oracle_df["lookahead"] == 1].groupby("cache_size")[["tokens_per_second", "speedup"]].mean().reset_index()

    la1_df = la1_df.sort_values("cache_size")
    
    # Calculate Best LA
    best_la_rows = []
    for c, grp in oracle_df.groupby("cache_size"):
        best_row = grp.loc[grp["speedup"].idxmax()]
        best_la_rows.append({
            "cache_size": c,
            "tokens_per_second": best_row["tokens_per_second"],
            "speedup": best_row["speedup"],
            "lookahead": int(best_row["lookahead"])
        })
    best_la_df = pd.DataFrame(best_la_rows).sort_values("cache_size")
    
    # ---------------------------------------------------------
    # Plot 1: Just S=1
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 7))
    
    # Plot baseline
    plt.plot(baseline_df["cache_size"], baseline_df["tokens_per_second"], marker="^", color="#333333", linestyle="--", linewidth=2, markersize=8, label="Baseline (RANDOM)")
    
    # Plot S=1
    plt.plot(la1_df["cache_size"], la1_df["tokens_per_second"], marker="s", color="#4C72B0", linewidth=2, markersize=8, label="Oracle (S=1)")
    
    # Plot Theoretical Max
    plt.plot(theoretical_df["cache_size"], theoretical_df["tokens_per_second"], marker="x", color="#55A868", linestyle=":", linewidth=2, markersize=8, label="Theoretical Oracle")
    
    # Plot Hardware Max TPS
    plt.axhline(1000.0 / 49.0, color="red", linestyle="--", linewidth=2, label=r"$T_{ceil}$ (All Experts Resident)")
    
    # Annotate S=1 points
    for _, row in la1_df.iterrows():
        plt.annotate(
            f"{row['speedup']:.2f}x", 
            (row["cache_size"], row["tokens_per_second"]),
            textcoords="offset points",
            xytext=(0, -15),
            ha='center',
            fontsize=9,
            color="#4C72B0"
        )

    plt.xlabel("Cache Size (Experts)")
    plt.ylabel("Tokens per Second (TPS)")
    plt.title("Oracle Throughput vs Cache Size (S=1)")
    plt.xticks(cache_sizes)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    out_path_s1 = str(csv_path.parent / "oracle_s1_speedup.png")
    plt.savefig(out_path_s1)
    print(f"Plot saved to {out_path_s1}")
    plt.close()
    
    # ---------------------------------------------------------
    # Plot 2: S=1 and Best Stride
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 7))
    
    # Plot baseline
    plt.plot(baseline_df["cache_size"], baseline_df["tokens_per_second"], marker="^", color="#333333", linestyle="--", linewidth=2, markersize=8, label="Baseline (RANDOM)")
    
    # Plot S=1
    plt.plot(la1_df["cache_size"], la1_df["tokens_per_second"], marker="s", color="#4C72B0", linewidth=2, markersize=8, label="Oracle (S=1)")
    
    # Plot Best LA
    plt.plot(best_la_df["cache_size"], best_la_df["tokens_per_second"], marker="o", color="#8172B3", linewidth=2, markersize=8, label="Oracle (Best Stride)")
    
    # Plot Theoretical Max
    plt.plot(theoretical_df["cache_size"], theoretical_df["tokens_per_second"], marker="x", color="#55A868", linestyle=":", linewidth=2, markersize=8, label="Theoretical Oracle")
    
    # Plot Hardware Max TPS
    plt.axhline(1000.0 / 49.0, color="red", linestyle="--", linewidth=2, label=r"$T_{ceil}$ (All Experts Resident)")
    
    # Annotate S=1 points
    for _, row in la1_df.iterrows():
        plt.annotate(
            f"{row['speedup']:.2f}x", 
            (row["cache_size"], row["tokens_per_second"]),
            textcoords="offset points",
            xytext=(0, -15),
            ha='center',
            fontsize=9,
            color="#4C72B0"
        )
        
    # Annotate Best LA points
    for _, row in best_la_df.iterrows():
        plt.annotate(
            f"S={int(row['lookahead'])}\n{row['speedup']:.2f}x", 
            (row["cache_size"], row["tokens_per_second"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            fontweight='bold',
            color="#8172B3"
        )

    plt.xlabel("Cache Size (Experts)")
    plt.ylabel("Tokens per Second (TPS)")
    plt.title("Oracle Throughput vs Cache Size (Best Stride)")
    plt.xticks(cache_sizes)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    out_path_best = str(csv_path.parent / "oracle_best_lookahead_speedup.png")
    plt.savefig(out_path_best)
    print(f"Plot saved to {out_path_best}")
    plt.close()
    
    # ---------------------------------------------------------
    # Plot 3: Custom Grouped Bar Chart (Cache Size vs TPS)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cluster_width = 0.8
    x_centers = np.arange(len(cache_sizes))
    
    # We will manually plot bars so they sit tightly together, 
    # even if different cache sizes have different numbers of lookaheads swept.
    import matplotlib.cm as cm
    
    for x_i, c in enumerate(cache_sizes):
        # Get baseline for this cache size
        base_tps = float(baseline_df[baseline_df["cache_size"] == c]["tokens_per_second"].iloc[0])
        
        # Get all oracle runs for this cache size, sorted by lookahead
        c_oracle = oracle_df[oracle_df["cache_size"] == c].sort_values("lookahead")
        
        n_bars = 1 + len(c_oracle)
        bar_width = cluster_width / n_bars
        
        # Plot Baseline
        ax.bar(
            x_centers[x_i] - cluster_width/2 + 0.5*bar_width,
            base_tps,
            width=bar_width*0.9,
            color="#333333",
            edgecolor="white",
            label="Baseline (RANDOM)" if x_i == 0 else None
        )
        
        # Plot Oracle Lookaheads with a color gradient
        cmap = cm.get_cmap("Blues")
        for i, (_, row) in enumerate(c_oracle.iterrows()):
            # Color based on relative lookahead depth for this cache size
            # i ranges from 0 to len(c_oracle)-1
            color_val = 0.3 + 0.7 * (i / max(1, len(c_oracle) - 1))
            
            ax.bar(
                x_centers[x_i] - cluster_width/2 + (i + 1.5)*bar_width,
                row["tokens_per_second"],
                width=bar_width*0.9,
                color=cmap(color_val),
                edgecolor="white"
            )
            
            # Annotate the lookahead value on top of the bar
            ax.text(
                x_centers[x_i] - cluster_width/2 + (i + 1.5)*bar_width,
                row["tokens_per_second"] + 0.2,
                f"S={int(row['lookahead'])}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90 if n_bars > 5 else 0
            )
            
    # Plot theoretical max line
    # Match the x_centers with the sorted cache sizes
    theoretical_y = [theoretical_df[theoretical_df["cache_size"] == c]["tokens_per_second"].iloc[0] for c in cache_sizes]
    ax.plot(x_centers, theoretical_y, marker="x", color="#55A868", linestyle=":", linewidth=2, markersize=8, label="Theoretical Oracle")
    
    # Hardware Max
    ax.axhline(1000.0 / 49.0, color="red", linestyle="--", linewidth=2, label=r"$T_{ceil}$ (All Experts Resident)")
    
    ax.set_xticks(x_centers)
    ax.set_xticklabels(cache_sizes)
    ax.set_xlabel("Cache Size (Experts)")
    ax.set_ylabel("Tokens per Second (TPS)")
    ax.set_title("Oracle Throughput by Lookahead and Cache Size")
    
    # Custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#333333', edgecolor='white', label='Baseline (RANDOM)'),
        Patch(facecolor=cm.get_cmap("Blues")(0.6), edgecolor='white', label='Oracle Lookahead (Gradient)'),
        Line2D([0], [0], marker="x", color="#55A868", linestyle=":", linewidth=2, label="Theoretical Oracle"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, label=r"$T_{ceil}$ (All Experts Resident)")
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    
    out_path_bars = str(csv_path.parent / "oracle_lookahead_bars.pdf")
    plt.savefig(out_path_bars)
    print(f"Plot saved to {out_path_bars}")
    plt.close()

if __name__ == "__main__":
    main()
