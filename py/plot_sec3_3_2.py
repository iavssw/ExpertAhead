import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Global Plot Style
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.6,
})

# =========================
# Data
# =========================
h = np.linspace(0, 1, 100)
T_ceil = 49

benchmark_file = r'/home/michael/heteroPredict/py/expert_loading_study/expert_loading_benchmark.csv'

bench_df = pd.read_csv(benchmark_file)
bench_df = bench_df[bench_df['Config'].str.contains('Parallel')]
bench_df = bench_df.sort_values('Num_Experts')

benchmark_m = np.concatenate(([0], bench_df['Num_Experts'].values))
benchmark_t_per_expert = np.concatenate(([0.0], bench_df['Time_per_Expert_ms'].values))
benchmark_stall_per_layer = benchmark_m * benchmark_t_per_expert

K = 8

# Naive Theoretical
expected_m_naive = K * (1 - h)
stall_per_layer_naive = np.interp(expected_m_naive, benchmark_m, benchmark_stall_per_layer)
T_model_naive = T_ceil + 48 * stall_per_layer_naive
TPS_model_naive = 1000 / T_model_naive

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(
    h * 100,
    TPS_model_naive,
    color="black",
    linewidth=2,
    label="Theoretical TPS (Naive)"
)

colors = {4: "blue", 5: "orange", 6: "green"}
markers = {4: "v", 5: "s", 6: "^"}

for j in [4, 5, 6]:
    h_eff_theory = (j/K) * h + (K - j)/K
    expected_m_cc = K * (1 - h_eff_theory)
    stall_per_layer_cc = np.interp(expected_m_cc, benchmark_m, benchmark_stall_per_layer)
    T_model_cc = T_ceil + 48 * stall_per_layer_cc
    TPS_model_cc = 1000 / T_model_cc
    
    ax.plot(
        h * 100,
        TPS_model_cc,
        color=colors[j],
        linewidth=1.5,
        linestyle="--",
        label=f"Theoretical TPS (CC J={j})"
    )

ax.axhline(
    y=1000 / T_ceil,
    linestyle="-.",
    alpha=0.7,
    label=f"Hardware Ceiling"
)

# =========================
# Empirical Data
# =========================
csv_path = sys.argv[1] if len(sys.argv) > 1 else "/home/michael/heteroPredict/py/utils/final_results_runs/sec3_3_2_cache_conditional_multi_J.csv"

try:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        cache_sizes = sorted(df["cache_size"].dropna().unique(), key=int)
        
        for i, c in enumerate(cache_sizes):
            sub = df[df["cache_size"] == c]
            
            rand_df = sub[sub["label"] == "Neither (RANDOM)"]
            if rand_df.empty:
                continue
                
            hr_naive = rand_df.iloc[0]["hit_rate_pct"]
            tps_naive = rand_df.iloc[0]["tokens_per_second"]
            
            # 1. Naive Marker
            ax.scatter(hr_naive, tps_naive, s=60, marker="o", color="black", zorder=6, 
                       label="Measured Naive" if i == 0 else "")
                       
            # Draw vertical line representing the cache size
            max_tps_for_c = tps_naive
            
            # Find max TPS for this cache size to draw the vertical grouping line
            for j in [4, 5, 6]:
                cc_df = sub[sub["label"] == f"Cache-Cond (J={j})"]
                if not cc_df.empty:
                    max_tps_for_c = max(max_tps_for_c, cc_df.iloc[0]["tokens_per_second"])
                    
            ax.plot([hr_naive, hr_naive], [tps_naive, max_tps_for_c], color="gray", alpha=0.3, zorder=3)
            ax.text(hr_naive - 0.5, (tps_naive + max_tps_for_c) / 2, f"C={c}", ha="right", va="center", fontsize=9, color="black")
            
            # Plot Cache-Cond points for each J
            for j in [4, 5, 6]:
                cc_df = sub[sub["label"] == f"Cache-Cond (J={j})"]
                if cc_df.empty:
                    continue
                    
                hr_cc = cc_df.iloc[0]["hit_rate_pct"]
                tps_cc = cc_df.iloc[0]["tokens_per_second"]

                # CC Marker at natural hit rate
                ax.scatter(hr_naive, tps_cc, s=60, marker=markers[j], color=colors[j], zorder=6, 
                           label=f"Measured CC J={j} (Natural HR)" if i == 0 else "")
                
                # Horizontal arrow pointing to ACTUAL hit rate
                ax.annotate("", xy=(hr_cc, tps_cc), xytext=(hr_naive, tps_cc),
                            arrowprops=dict(arrowstyle="->", color=colors[j], alpha=0.5, shrinkA=2, shrinkB=2))
                
                # Tiny dot at actual hit rate
                ax.scatter(hr_cc, tps_cc, s=15, color=colors[j], zorder=6,
                           label=f"Actual Achieved HR J={j}" if i == 0 else "")

    else:
        print(f"Warning: CSV file not found at {csv_path}. Plotting theoretical curves only.")
except Exception as e:
    print(f"Failed to load empirical data or plot: {e}")

# =========================
# Formatting
# =========================
ax.set_xlabel("Natural Cache Hit Rate (%)")
ax.set_ylabel("Tokens Per Second (TPS)")
ax.set_title("Cache-Conditional Routing Sweep (J=4, 5, 6)", fontweight="bold")

ax.legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=2)

fig.tight_layout()

out_path = "/home/michael/heteroPredict/py/modeling_cache_conditional_multi_J.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved to {out_path}")
