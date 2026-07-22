import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Global Plot Style (same as analyze_expert_reuse.py)
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


# SSD expert loading benchmark file
benchmark_file = r'/home/michael/heteroPredict/py/expert_loading_study/fast_ssd/expert_loading_benchmark.csv'

# Read from benchmark file for Parallel IOThreadPool
bench_df = pd.read_csv(benchmark_file)
bench_df = bench_df[bench_df['Config'].str.contains('Parallel')]
bench_df = bench_df.sort_values('Num_Experts')

# 0 misses = 0 stall time
benchmark_m = np.concatenate(([0], bench_df['Num_Experts'].values))
benchmark_t_per_expert = np.concatenate(([0.0], bench_df['Time_per_Expert_ms'].values))

# Calculate total stall time per layer for each N
benchmark_stall_per_layer = benchmark_m * benchmark_t_per_expert

# For a given hit rate h, expected misses per layer is 8 * (1 - h)
expected_m = 8 * (1 - h)

# Interpolate the stall time per layer for the continuous expected_m
stall_per_layer_interp = np.interp(expected_m, benchmark_m, benchmark_stall_per_layer)

# Total latency: hardware ceiling + 48 layers * stall time per layer
T_model = T_ceil + 48 * stall_per_layer_interp
TPS_model = 1000 / T_model

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    h * 100,
    TPS_model,
    color="black",
    linewidth=2,
    label="Theoretical TPS Model"
)

# Hardware ceiling
ax.axhline(
    y=1000 / T_ceil,
    linestyle="-.",
    alpha=0.7,
    label=f"Hardware Ceiling (T_ceil)"
)

# =========================
# =========================
csv_path = "/home/michael/heteroPredict/py/utils/final_results_runs/sec2_expert_io_ab/parallel/sec5_all_methods/new_sweep_cold.csv"
try:
    df = pd.read_csv(csv_path)

    # LRU/Random baseline
    lru_df = df[
        (df["label"] == "Neither (RANDOM)") &
        (df["lookahead"] == 1)
    ]

    if not lru_df.empty:
        ax.scatter(
            lru_df["hit_rate_pct"],
            lru_df["tokens_per_second"],
            s=70,
            label="Measured Naive Cache",
            zorder=5
        )

        for hr, tps, c in zip(
            lru_df["hit_rate_pct"],
            lru_df["tokens_per_second"],
            lru_df["cache_size"]
        ):
            ax.annotate(
                f"C={c}",
                (hr, tps),
                textcoords="offset points",
                xytext=(15, -10),
                fontsize=9
            )

    # Forced Miss Anchor
    anchor_df = df[df["label"] == "Forced Miss Anchor (C=0)"]
    if not anchor_df.empty:
        ax.scatter(
            anchor_df["hit_rate_pct"],
            anchor_df["tokens_per_second"],
            s=250,
            marker="*",
            color="red",
            label="Hardware Floor (SSD Streaming)",
            zorder=6
        )

except Exception as e:
    print(f"Failed to load empirical data: {e}")

# =========================
# Formatting
# =========================
ax.set_xlabel("Cache Hit Rate (%)")
ax.set_ylabel("Tokens Per Second (TPS)")
ax.set_title("Empirical vs Theoretical TPS", fontweight="bold")

ax.legend(loc="center left", framealpha=0.1)

fig.tight_layout()

out_path = "/home/michael/heteroPredict/py/modeling_naive.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved to {out_path}")

# =========================
# Theoretical vs Actual CSV
# =========================
try:
    df_csv = pd.read_csv(csv_path)
    emp_df = df_csv[
        ((df_csv["label"] == "Neither (RANDOM)") & (df_csv["lookahead"] == 1)) |
        (df_csv["label"] == "Forced Miss Anchor (C=0)")
    ].copy()

    if not emp_df.empty:
        # Compute theoretical TPS at each empirical hit rate
        emp_df = emp_df.copy()
        hr_frac = emp_df["hit_rate_pct"].values / 100.0
        m_at_hr = 8 * (1 - hr_frac)
        stall_at_hr = np.interp(m_at_hr, benchmark_m, benchmark_stall_per_layer)
        T_theory = T_ceil + 48 * stall_at_hr
        tps_theory = 1000.0 / T_theory

        emp_df["theoretical_tps"] = tps_theory
        emp_df["delta_tps"] = emp_df["tokens_per_second"] - emp_df["theoretical_tps"]
        emp_df["delta_pct"] = 100.0 * emp_df["delta_tps"] / emp_df["theoretical_tps"]

        out_cols = ["cache_size", "hit_rate_pct", "tokens_per_second", "theoretical_tps", "delta_tps", "delta_pct"]
        out_cols = [c for c in out_cols if c in emp_df.columns]
        diff_csv = "/home/michael/heteroPredict/py/modeling_vs_actual.csv"
        emp_df[out_cols].sort_values("cache_size").to_csv(diff_csv, index=False, float_format="%.4f")
        print(f"Saved theory-vs-actual diff to {diff_csv}")

        print("\n=== Theoretical vs Actual TPS ===")
        print(emp_df[out_cols].sort_values("cache_size").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("No RANDOM baseline rows found in CSV for diff output.")

except Exception as e:
    print(f"Failed to generate theory-vs-actual CSV: {e}")