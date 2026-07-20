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
# Configuration
# =========================
csv_path = "/home/michael/heteroPredict/py/utils/final_results_runs/sec3_3_3_unified.csv"
benchmark_file = '/home/michael/heteroPredict/py/expert_loading_study/expert_loading_benchmark.csv'

T_ceil = 49.0
L = 48
K = 8
E = L * K # 384
N = 1 # predictor stride
J = 6 # forced top-J

# Load benchmark data for T_ssd,n
bench_df = pd.read_csv(benchmark_file)
bench_df = bench_df[bench_df['Config'].str.contains('Parallel')]
bench_df = bench_df.sort_values('Num_Experts')

benchmark_m = np.concatenate(([0], bench_df['Num_Experts'].values))
# benchmark_t_per_expert is the time PER EXPERT when loading m experts in parallel
benchmark_t_per_expert = np.concatenate(([0.0], bench_df['Time_per_Expert_ms'].values))
benchmark_stall_per_layer = benchmark_m * benchmark_t_per_expert

def get_T_ssd(n_misses_per_layer):
    # interpolate the time per expert based on the number of parallel loads
    return np.interp(n_misses_per_layer, benchmark_m, benchmark_t_per_expert)

def get_stall_per_layer(n_misses_per_layer):
    return np.interp(n_misses_per_layer, benchmark_m, benchmark_stall_per_layer)

if not os.path.exists(csv_path):
    print(f"Waiting for you to generate {csv_path}!")
    print("Run the following command to generate it:")
    print("python3 py/utils/sweep_predict_cached_cache_metrics.py --sweep-question unified_sweep --cache-sizes 8 16 24 40 64 --dataset oracle --num-prompts 5 --csv-file py/utils/final_results_runs/sec3_3_3_unified.csv")
    exit(0)

df = pd.read_csv(csv_path)

lru_df = df[(df["label"] == "Neither (RANDOM)")]
oracle_cc_df = df[df["label"].str.contains("Oracle Full Union \+ CC", na=False)]

hit_rate_map = {}
baseline_tps_map = {}
for _, row in lru_df.iterrows():
    c = int(row["cache_size"])
    hit_rate_map[c] = float(row["hit_rate_pct"]) / 100.0
    baseline_tps_map[c] = float(row["tokens_per_second"])

theory_rows = []
for _, row in oracle_cc_df.iterrows():
    c = int(row["cache_size"])
    h = hit_rate_map.get(c, 0.0)
    
    # ---------------------------------------------------------
    # Naive Theoretical Model
    # ---------------------------------------------------------
    E_miss_naive = E * (1 - h)
    n_misses_naive = K * (1 - h)
    stall_naive = get_stall_per_layer(n_misses_naive)
    T_model_naive = T_ceil + 48 * stall_naive
    TPS_model_naive = 1000.0 / T_model_naive
    
    # ---------------------------------------------------------
    # Unified Theoretical Model (Equation 1)
    # T = T_ceil + (E'_miss - E_prefetch) * Delta T
    # ---------------------------------------------------------
    
    # 1. Effective Hit Rate (from Cache-Cond J=6)
    h_eff = (J / K) * h + (K - J) / K
    
    # 2. E'_miss: Total misses across the network after Cache-Cond
    E_prime_miss = E * (1 - h_eff)
    
    # 3. Delta T (T_ssd): Time per expert, derived from parallel misses per layer
    n_prime = K * (1 - h_eff)
    T_ssd = get_T_ssd(n_prime)
    
    # 4. M_cap: Maximum prefetch capacity hidden by compute window
    M_cap = (N * T_ceil) / T_ssd if T_ssd > 0 else float('inf')
    
    # 5. E_prefetch: Number of experts prefetched successfully
    C_total = L * c
    M_prefetch = min(E_prime_miss, M_cap, C_total) # B = infinity for Oracle Full Union
    E_prefetch = M_prefetch / N
    
    # 6. E_block: Remaining blocking stalls
    E_block = max(0, E_prime_miss - E_prefetch)
    
    # 7. Final Token Latency
    T_model_unified = T_ceil + E_block * T_ssd
    TPS_model_unified = 1000.0 / T_model_unified
    
    theory_rows.append({
        "cache_size": c,
        "actual_tps_naive": baseline_tps_map.get(c, 0),
        "theoretical_tps_naive": TPS_model_naive,
        "actual_tps_unified": row["tokens_per_second"],
        "theoretical_tps_unified": TPS_model_unified,
        "E_prime_miss": E_prime_miss,
        "E_prefetch": E_prefetch,
        "E_block": E_block
    })

res_df = pd.DataFrame(theory_rows).sort_values(by="cache_size")

print("\n=== Theoretical vs Actual Unified TPS ===")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(8, 5))

cache_sizes = res_df["cache_size"].tolist()

# Plot Naive
ax.plot(
    res_df["cache_size"],
    res_df["actual_tps_naive"],
    marker='o',
    color='black',
    label="Naive (Measured)"
)
ax.plot(
    res_df["cache_size"],
    res_df["theoretical_tps_naive"],
    marker='x',
    linestyle=':',
    color='gray',
    alpha=0.7,
    label="Naive (Theoretical)"
)

# Plot Unified
c_unified = plt.cm.tab10.colors[2] # Green
ax.plot(
    res_df["cache_size"],
    res_df["actual_tps_unified"],
    marker='^',
    color=c_unified,
    label="Unified (Measured)"
)
ax.plot(
    res_df["cache_size"],
    res_df["theoretical_tps_unified"],
    marker='+',
    linestyle='--',
    color=c_unified,
    label="Unified (Theoretical)"
)

# Hardware Ceiling
ax.axhline(
    y=1000 / T_ceil,
    linestyle="-.",
    alpha=0.7,
    color='red',
    label=f"Hardware Ceiling"
)

# Formatting
ax.set_xlabel("Expert Cache Capacity (Experts per Layer)")
ax.set_ylabel("Tokens Per Second (TPS)")
ax.set_title("Unified Model: Oracle Prefetching + Cache-Cond (J=6)", fontweight="bold")
ax.legend(loc="lower right", framealpha=0.9)
if len(cache_sizes) > 0:
    ax.set_xticks(cache_sizes)

fig.tight_layout()
out_path = "/home/michael/heteroPredict/py/modeling_unified.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved plot to {out_path}\n")
