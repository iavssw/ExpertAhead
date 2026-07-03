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
# File Paths
# =========================
parallel_file = "/home/michael/heteroPredict/py/utils/final_results_runs/sec2_expert_io_ab/oracle_sec2_sweep_parallel.csv"
sequential_file = "/home/michael/heteroPredict/py/utils/final_results_runs/sec2_expert_io_ab/oracle_sec2_sweep_sequential.csv"

# =========================
# Load Data
# =========================
def load_and_filter(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    
    # We want RANDOM, Cache-Cond, Prefetch, and Both
    def should_keep(lbl):
        return (lbl == "Neither (RANDOM)") or \
               lbl.startswith("Cache-Cond Only") or \
               lbl.startswith("Prefetch Only ") or \
               lbl.startswith("Both ")
               
    df = df[df['label'].apply(should_keep)].copy()
    
    # Exclude incomplete cache size 32
    df = df[df['cache_size'] != 32].copy()
    
    # For predictor methods, fix lookahead=1.0 for consistency
    df['lookahead'] = df['lookahead'].fillna(1.0)
    df = df[df['lookahead'] == 1.0].copy()
    
    return df

par_df = load_and_filter(parallel_file)
seq_df = load_and_filter(sequential_file)

if par_df.empty or seq_df.empty:
    print("Warning: Missing or empty CSV files.")
    exit(1)

cache_sizes = sorted(set(par_df['cache_size'].dropna().unique()).intersection(seq_df['cache_size'].dropna().unique()), key=int)

# =========================
# Helpers
# =========================
def get_best_label(df, cache_size, prefix):
    if prefix == "Neither (RANDOM)":
        return "Neither (RANDOM)"
    sub = df[(df['cache_size'] == cache_size) & (df['label'].str.startswith(prefix))]
    if sub.empty:
        return None
    # Best is defined by highest throughput
    best_idx = sub['tokens_per_second'].idxmax()
    return df.loc[best_idx, 'label']

def get_value(df, cache_size, label, metric):
    if label is None:
        return np.nan
    sub = df[(df['cache_size'] == cache_size) & (df['label'] == label)]
    if not sub.empty:
        return sub.iloc[0][metric]
    return np.nan

# =========================
# Plotting
# =========================
fig, axes = plt.subplots(3, 3, figsize=(18, 16))

metrics = [
    ("tokens_per_second", "Throughput vs. Cache Size", "Tokens Per Second (TPS)"),
    ("stall_loads", "Blocking IO (Stalling Loads)", "Total Stalling Loads"),
    ("hit_rate_pct", "Cache Hit Rate vs. Cache Size", "Cache Hit Rate (%)")
]

# Configure the 3 rows
rows_config = [
    {
        "mode": "Parallel Baseline",
        "dfs": [par_df, par_df, par_df, par_df],
        "prefixes": ["Neither (RANDOM)", "Cache-Cond Only", "Prefetch Only ", "Both "],
        "labels": ["Par RANDOM", "Par Cache-Cond", "Par Best Prefetch", "Par Best Both"],
        "colors": ["darkgreen", "blue", "orange", "purple"]
    },
    {
        "mode": "Sequential Baseline",
        "dfs": [seq_df, seq_df, seq_df, seq_df],
        "prefixes": ["Neither (RANDOM)", "Cache-Cond Only", "Prefetch Only ", "Both "],
        "labels": ["Seq RANDOM", "Seq Cache-Cond", "Seq Best Prefetch", "Seq Best Both"],
        "colors": ["darkgreen", "blue", "orange", "purple"]
    },
    {
        "mode": "Absolute Speedup (Parallel vs Sequential)",
        "dfs": [seq_df, par_df, par_df, par_df, par_df],
        "prefixes": ["Neither (RANDOM)", "Neither (RANDOM)", "Cache-Cond Only", "Prefetch Only ", "Both "],
        "labels": ["Seq RANDOM (Base)", "Par RANDOM", "Par Cache-Cond", "Par Best Prefetch", "Par Best Both"],
        "colors": ["darkred", "darkgreen", "blue", "orange", "purple"]
    }
]

for row_idx, rcfg in enumerate(rows_config):
    mode = rcfg["mode"]
    dfs = rcfg["dfs"]
    prefixes = rcfg["prefixes"]
    labels = rcfg["labels"]
    colors = rcfg["colors"]
    
    num_bars = len(labels)
    bar_width = 0.8 / num_bars
    
    for col_idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[row_idx, col_idx]
        
        x = np.arange(len(cache_sizes))
        
        # We need to collect the baseline values for each cache size to compute speedups
        baseline_vals = []
        for c in cache_sizes:
            base_lbl = get_best_label(dfs[0], c, prefixes[0])
            baseline_vals.append(get_value(dfs[0], c, base_lbl, metric))
            
        for b_idx, (df_src, prefix, label, color) in enumerate(zip(dfs, prefixes, labels, colors)):
            
            vals = []
            for c in cache_sizes:
                best_lbl = get_best_label(df_src, c, prefix)
                vals.append(get_value(df_src, c, best_lbl, metric))
                
            bars = ax.bar(x + b_idx * bar_width, vals, bar_width, label=label, color=color, alpha=0.8)
            
            # Annotate speedups
            for i, bar in enumerate(bars):
                val = bar.get_height()
                base_val = baseline_vals[i]
                
                if pd.isna(val) or pd.isna(base_val) or base_val == 0 or val == 0:
                    continue
                    
                if metric == "tokens_per_second":
                    speedup = val / base_val
                    text = f"{speedup:.2f}x"
                elif metric == "stall_loads":
                    ratio = val / base_val
                    text = f"{ratio:.2f}x"
                else:
                    diff = val - base_val
                    text = f"{diff:+.1f}%"
                    
                # Don't annotate baseline if it's identical
                if abs(val - base_val) < 1e-5:
                    text = "-"
                    
                ax.text(bar.get_x() + bar.get_width()/2, val, text,
                        ha='center', va='bottom', fontsize=7, rotation=90)

        ax.set_xlabel("Cache Size (Experts)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{mode} - {title}", fontsize=11)
        ax.set_xticks(x + (num_bars - 1) * bar_width / 2)
        ax.set_xticklabels(cache_sizes)
        
        # Adjust ylim to make room for text labels
        ymin, ymax = ax.get_ylim()
        if metric == "hit_rate_pct":
            ax.set_ylim(0, max(105, ymax * 1.15))
        else:
            ax.set_ylim(ymin, ymax * 1.3)
            
        # Add legend to the first column of each row
        if col_idx == 0:
            ax.legend(fontsize=8, loc='upper left')

plt.subplots_adjust(bottom=0.1, hspace=0.4, wspace=0.25)

out_path = "/home/michael/heteroPredict/py/research_meeting_summary_no_oracle.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved meeting plots to {out_path}")
