import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# Configuration
# =========================
csv_path = "/home/michael/heteroPredict/py/utils/final_results_runs/oracle_la_vs_cache/20260705_222605/sweep.csv"
benchmark_file = '/home/michael/heteroPredict/py/expert_loading_study/fast_ssd/expert_loading_benchmark.csv'

T_ceil = 49.0
L = 48
K = 8
E = L * K # 384
N = 1 # predictor stride

# Load benchmark data for T_ssd,n
bench_df = pd.read_csv(benchmark_file)
bench_df = bench_df[bench_df['Config'].str.contains('Parallel')]
bench_df = bench_df.sort_values('Num_Experts')

benchmark_m = np.concatenate(([0], bench_df['Num_Experts'].values))
# benchmark_t_per_expert is the time PER EXPERT when loading m experts in parallel
benchmark_t_per_expert = np.concatenate(([0.0], bench_df['Time_per_Expert_ms'].values))

def get_T_ssd(n_misses_per_layer):
    # interpolate the time per expert based on the number of parallel loads
    return np.interp(n_misses_per_layer, benchmark_m, benchmark_t_per_expert)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Failed to load empirical data: {e}")
    exit(1)

# Filter for oracle runs, specifically focusing on Full Union to see unconstrained prefetch performance
df_oracle = df[df["label"].str.contains("Oracle Full Union", na=False)].copy()
if df_oracle.empty:
    print("No Oracle rows found in CSV yet. Waiting for script to finish.")
    exit(0)

# Find LRU baseline hit rate to know total U (un-cached experts requested)
# Find LRU baseline hit rate and TPS
lru_df = df[(df["label"] == "Neither (RANDOM)")]
hit_rate_map = {}
baseline_tps_map = {}
for _, row in lru_df.iterrows():
    c = int(row["cache_size"])
    if c not in hit_rate_map:
        hit_rate_map[c] = float(row["hit_rate_pct"]) / 100.0
        baseline_tps_map[c] = float(row["tokens_per_second"])

# Compute theoretical TPS for each Oracle row
theory_rows = []
for _, row in df_oracle.iterrows():
    c = int(row["cache_size"])
    b_str = str(row["prefetch_budget"])
    
    if b_str.lower() == "nan":
        b = float('inf') # Oracle Full Union has effectively infinite issue budget
    else:
        b = float(b_str)
        
    h = hit_rate_map.get(c, 0.0)
    
    # 1. Un-cached experts requested by the router over N=1 tokens
    U = E * (1 - h)
    
    # 2. n is the number of misses per layer (this determines the parallelization efficiency)
    n_misses_per_layer = K * (1 - h)
    
    # 3. T_ssd, n (Time PER EXPERT when loading n experts in parallel)
    T_ssd = get_T_ssd(n_misses_per_layer)
    
    # 4. M_cap (Maximum loads that can be hidden within the T_ceil compute window)
    M_cap = (N * T_ceil) / T_ssd if T_ssd > 0 else float('inf')
    
    # 5. M_prefetch (Number of experts successfully prefetched and hidden)
    # Oracle has rho = 1, pi = 1
    # Total cache capacity available for prefetching across the model is L * C
    C_total = L * c
    M_prefetch = min(U, L * b, M_cap, C_total)
    
    # 6. E_block (Number of blocking stalls remaining)
    E_miss = U
    E_prefetch = M_prefetch / N
    E_block = max(0, E_miss - E_prefetch)
    
    # 7. T_model (Total token latency)
    # The stall latency is E_block * Delta T, where Delta T is T_ssd
    # We ignore T_{mem} since it's extremely small.
    Delta_T = T_ssd
    T_model = T_ceil + (E_block * Delta_T)
    TPS_model = 1000.0 / T_model
    
    actual_tps = row["tokens_per_second"]
    baseline_tps = baseline_tps_map.get(c, actual_tps)
    actual_speedup = actual_tps / baseline_tps if baseline_tps > 0 else 1.0
    theoretical_speedup = TPS_model / baseline_tps if baseline_tps > 0 else 1.0
    
    theory_rows.append({
        "label": row["label"],
        "cache_size": c,
        "hit_rate_pct": h * 100,
        "actual_tps": actual_tps,
        "theoretical_tps": TPS_model,
        "actual_speedup": actual_speedup,
        "theoretical_speedup": theoretical_speedup,
        "M_cap": M_cap,
        "U": U,
        "M_prefetch": M_prefetch,
        "E_block": E_block,
        "expected_T_ssd": T_ssd,
        "actual_T_ssd": row["avg_ms_per_expert_load"]
    })

res_df = pd.DataFrame(theory_rows)
res_df = res_df.sort_values(by=["cache_size", "label"])

print("\n=== Theoretical vs Actual Oracle TPS ===")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

# Save to CSV
diff_csv = "/home/michael/heteroPredict/py/oracle_theory_vs_actual.csv"
res_df.to_csv(diff_csv, index=False)
print(f"\nSaved theory-vs-actual diff to {diff_csv}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

cache_sizes = sorted(res_df["cache_size"].unique())

colors = plt.cm.tab10.colors
for idx, label in enumerate(res_df["label"].unique()):
    sub_df = res_df[res_df["label"] == label]
    
    c = colors[idx % len(colors)]
    
    # Plot Actual
    ax.plot(
        sub_df["cache_size"],
        sub_df["actual_tps"],
        marker='o',
        color=c,
        label="Oracle (Measured)"
    )
    
    # Annotate speedup on the Actual points
    for cx, tps, spd in zip(sub_df["cache_size"], sub_df["actual_tps"], sub_df["actual_speedup"]):
        ax.annotate(
            f"{spd:.2f}x",
            (cx, tps),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            color=c,
            fontweight='bold'
        )

    # Plot Theoretical
    ax.plot(
        sub_df["cache_size"],
        sub_df["theoretical_tps"],
        linestyle='--',
        marker='x',
        color=c,
        alpha=0.7,
        label="Oracle (Theoretical)"
    )

# Plot RANDOM baseline
if not lru_df.empty:
    lru_df_sorted = lru_df.sort_values(by="cache_size")
    ax.plot(
        lru_df_sorted["cache_size"],
        lru_df_sorted["tokens_per_second"],
        marker='s',
        color='gray',
        linestyle=':',
        label="No Predictor"
    )

ax.axhline(
    y=1000 / T_ceil,
    linestyle="-.",
    alpha=0.7,
    color='red',
    label=f"Hardware Ceiling"
)

ax.set_xlabel("Expert Cache Capacity (Experts per Layer)")
ax.set_ylabel("Tokens Per Second (TPS)")
ax.set_title("Theoretical vs Measured Throughput (Oracle Prefetching)", fontweight="bold")
ax.legend(loc="best", framealpha=0.8)
if len(cache_sizes) > 0:
    ax.set_xticks(cache_sizes)

fig.tight_layout()
out_path = "/home/michael/heteroPredict/py/modeling_prefetch_oracle.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot to {out_path}\n")
