# Plot Best Prefetch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Global Plot Style
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.6,
})

# =========================
# File Paths
# =========================
parallel_file = "/home/michael/heteroPredict/py/utils/final_results_runs/sec2_expert_io_ab/oracle_sec2_sweep_parallel.csv"
sequential_file = "/home/michael/heteroPredict/py/utils/final_results_runs/sec2_expert_io_ab/oracle_sec2_sweep_sequential.csv"
oracle_file = "/home/michael/heteroPredict/py/utils/final_results_runs/sec2_expert_io_ab/research_oracle_only_parallel.csv"

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    if 'cache_size' in df.columns:
        df = df[df['cache_size'] != 32].copy()
    return df

par_df = load_data(parallel_file)
seq_df = load_data(sequential_file)
oracle_df = load_data(oracle_file)

if par_df.empty or seq_df.empty:
    print("Warning: Missing or empty CSV files.")
    exit(1)

cache_sizes = sorted(set(par_df['cache_size'].dropna().unique()).intersection(seq_df['cache_size'].dropna().unique()), key=int)

# =========================
# Helpers
# =========================
def get_seq_baseline(df, cache_size):
    sub = df[(df['cache_size'] == cache_size) & (df['label'] == "Neither (RANDOM)")]
    if sub.empty:
        return None
    return sub.iloc[0]

def get_best_par_prefetch(df, cache_size):
    sub = df[(df['cache_size'] == cache_size) & (df['label'].str.startswith("Prefetch Only "))]
    if sub.empty:
        return None
    best_idx = sub['tokens_per_second'].idxmax()
    return df.loc[best_idx]
    
def get_best_par_both(df, cache_size):
    sub = df[(df['cache_size'] == cache_size) & (df['label'].str.startswith("Both λ=1.0 "))]
    if sub.empty:
        return None
    best_idx = sub['tokens_per_second'].idxmax()
    return df.loc[best_idx]

def get_best_oracle(df, cache_size):
    if df.empty:
        return None
    sub = df[(df['cache_size'] == cache_size) & (df['label'].str.startswith("Oracle Full Union"))]
    if sub.empty:
        return None
    best_idx = sub['tokens_per_second'].idxmax()
    return df.loc[best_idx]

# =========================
# Plotting
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

metrics = [
    ("tokens_per_second", "Throughput", "Tokens Per Second (TPS)"),
    ("stall_loads", "Blocking IO", "Total Stalling Loads"),
    ("hit_rate_pct", "Cache Hit Rate", "Cache Hit Rate (%)")
]

bar_width = 0.2
x = np.arange(len(cache_sizes))

for col_idx, (metric, title, ylabel) in enumerate(metrics):
    ax = axes[col_idx]
    
    seq_vals = []
    par_vals = []
    both_vals = []
    oracle_vals = []
    
    par_annotations = []
    both_annotations = []
    oracle_annotations = []
    
    for c in cache_sizes:
        seq_row = get_seq_baseline(seq_df, c)
        par_row = get_best_par_prefetch(par_df, c)
        both_row = get_best_par_both(par_df, c)
        oracle_row = get_best_oracle(oracle_df, c)
        
        seq_vals.append(seq_row[metric] if seq_row is not None else np.nan)
        
        if par_row is not None:
            par_vals.append(par_row[metric])
            b = par_row['prefetch_budget']
            la = par_row['lookahead']
            par_annotations.append(f"B={int(b) if not pd.isna(b) else '?'}\nLA={int(la) if not pd.isna(la) else '?'}")
        else:
            par_vals.append(np.nan)
            par_annotations.append("")
            
        if both_row is not None:
            both_vals.append(both_row[metric])
            b = both_row['prefetch_budget']
            la = both_row['lookahead']
            both_annotations.append(f"B={int(b) if not pd.isna(b) else '?'}\nLA={int(la) if not pd.isna(la) else '?'}")
        else:
            both_vals.append(np.nan)
            both_annotations.append("")
            
        if oracle_row is not None:
            oracle_vals.append(oracle_row[metric])
            la = oracle_row['lookahead']
            oracle_annotations.append(f"Oracle\nLA={int(la) if not pd.isna(la) else '?'}")
        else:
            oracle_vals.append(np.nan)
            oracle_annotations.append("")
            
    # Plot bars
    bars1 = ax.bar(x - 1.5 * bar_width, seq_vals, bar_width, label="Sequential Baseline", color="darkred", alpha=0.8)
    bars2 = ax.bar(x - 0.5 * bar_width, par_vals, bar_width, label="Parallel Best Prefetch", color="orange", alpha=0.9)
    bars3 = ax.bar(x + 0.5 * bar_width, both_vals, bar_width, label="Parallel Best Both", color="purple", alpha=0.8)
    bars4 = ax.bar(x + 1.5 * bar_width, oracle_vals, bar_width, label="Parallel Best Oracle", color="darkgreen", alpha=0.8)
    
    # Helper for annotations
    def annotate_bars(bars, annotations, color):
        for i, bar in enumerate(bars):
            val = bar.get_height()
            if not pd.isna(val) and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val / 2 if metric != 'hit_rate_pct' else val - 15, annotations[i],
                        ha='center', va='center', fontsize=7, color='white' if metric != 'hit_rate_pct' else 'black', fontweight='bold')
                
                seq_val = seq_vals[i]
                if metric == "tokens_per_second" and not pd.isna(seq_val) and seq_val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val, f"{(val / seq_val):.2f}x", ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)
                elif metric == "stall_loads" and not pd.isna(seq_val) and seq_val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val, f"{(val / seq_val):.2f}x", ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)
                elif metric == "hit_rate_pct" and not pd.isna(seq_val):
                    ax.text(bar.get_x() + bar.get_width()/2, val, f"+{(val - seq_val):.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

    annotate_bars(bars2, par_annotations, 'orange')
    annotate_bars(bars3, both_annotations, 'purple')
    annotate_bars(bars4, oracle_annotations, 'darkgreen')

    ax.set_xlabel("Cache Size (Experts)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(cache_sizes)
    
    ymin, ymax = ax.get_ylim()
    if metric == "hit_rate_pct":
        ax.set_ylim(0, 115)
    else:
        ax.set_ylim(ymin, ymax * 1.15)
        
    if col_idx == 0:
        ax.legend(loc='upper left', fontsize=10)

fig.suptitle("Sequential Baseline vs. Parallel Predictor vs. Oracle", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

out_path = "/home/michael/heteroPredict/py/research_meeting_best_prefetch.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved best prefetch plots to {out_path}")
