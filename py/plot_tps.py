import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h = np.linspace(0, 1, 100)
E = 384
T_ceil = 49

# 1. Theoretical Curve (Constant T_SSD = 0.67 ms)
E_miss = E * (1 - h)
T_067 = T_ceil + E_miss * 0.67
TPS_067 = 1000 / T_067

plt.figure(figsize=(9, 6))
plt.plot(h * 100, TPS_067, label=r'Theoretical TPS ($\Delta T = 0.67$ ms)', color='blue', linewidth=2)

# Mark the hardware ceiling
plt.axhline(y=1000/T_ceil, color='r', linestyle='-.', alpha=0.5, label='Hardware Ceiling (20.4 TPS)')

# 2. Empirical Data
csv_path = "trainingData/qwen_traces/final_empirical/sweep_20260623_184257_final.csv"
try:
    df = pd.read_csv(csv_path)
    
    # LRU baseline at N=1
    lru_df = df[(df["label"] == "Neither (LRU)") & (df["lookahead"] == 1)]
    if not lru_df.empty:
        plt.scatter(lru_df["hit_rate_pct"], lru_df["tokens_per_second"], color='orange', s=100, zorder=5, label='Empirical LRU')
        for hr, tps, c in zip(lru_df["hit_rate_pct"], lru_df["tokens_per_second"], lru_df["cache_size"]):
            plt.annotate(f'C={c}', (hr, tps), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='orange')
            
    # Oracle Full Union LA=1
    oracle_df = df[(df["label"] == "Oracle Full Union LA=1")]
    if not oracle_df.empty:
        plt.scatter(oracle_df["hit_rate_pct"], oracle_df["tokens_per_second"], color='green', s=100, marker='^', zorder=5, label='Empirical Oracle Full Union (LA=1)')
        for hr, tps, c in zip(oracle_df["hit_rate_pct"], oracle_df["tokens_per_second"], oracle_df["cache_size"]):
            plt.annotate(f'C={c}', (hr, tps), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='green')

except Exception as e:
    print(f"Failed to load empirical data: {e}")

plt.xlabel('Cache Hit Rate (%)', fontsize=12)
plt.ylabel('Expected Tokens Per Second (TPS)', fontsize=12)
plt.title('Validation: Empirical TPS vs. Theoretical Model', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

out_path = "/home/michael/.gemini/antigravity-ide/brain/148329c1-d05d-48f6-a1d9-533aa556cc72/tps_vs_hitrate.png"
plt.savefig(out_path, dpi=300)
print(f"Saved to {out_path}")
