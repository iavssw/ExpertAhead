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
parallel_file = "/home/michael/heteroPredict/py/utils/final_results_runs/sec2_expert_io_ab/meeting_parallel.csv"
sequential_file = "/home/michael/heteroPredict/py/utils/final_results_runs/sec2_expert_io_ab/meeting_sequential.csv"

# =========================
# Load Data
# =========================
def load_and_filter(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    return df

par_df = load_and_filter(parallel_file)
seq_df = load_and_filter(sequential_file)

if par_df.empty or seq_df.empty:
    print("Warning: Missing or empty CSV files. Please run './run_meeting_sweep.sh' first.")
    exit(1)

cache_sizes = sorted(par_df['cache_size'].dropna().unique(), key=int)

# =========================
# Image 1: Main Speedup Bar Chart
# =========================
# =========================
# Image 1: Main Speedup Comparison (3-panel)
# =========================
def plot_speedup_bars():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    width = 0.3
    x = np.arange(len(cache_sizes))

    metrics = [
        ("tokens_per_second", "Throughput", "Tokens Per Second (TPS)"),
        ("stall_loads", "Blocking IO", "Total Stalling Loads"),
        ("hit_rate_pct", "Cache Hit Rate", "Cache Hit Rate (%)")
    ]

    for col_idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[col_idx]

        seq_vals = []
        par_pred_vals = []
        pred_labels = []

        for c in cache_sizes:
            # Sequential RANDOM
            sub = seq_df[(seq_df['cache_size'] == c) & (seq_df['label'] == "Neither (RANDOM)")]
            seq_vals.append(sub.iloc[0][metric] if not sub.empty else np.nan)

            # Best Parallel Predictor (selected by tokens_per_second, consistent across panels)
            sub = par_df[(par_df['cache_size'] == c) & (par_df['label'].str.startswith("Predictor"))]
            if not sub.empty:
                best_idx = sub['tokens_per_second'].idxmax()
                best_row = sub.loc[best_idx]
                par_pred_vals.append(best_row[metric])
                la = best_row['lookahead']
                b = best_row['prefetch_budget']
                pred_labels.append(f"LA={int(la)}\nB={int(b)}")
            else:
                par_pred_vals.append(np.nan)
                pred_labels.append("")

        bars1 = ax.bar(x - 0.5*width, seq_vals, width, label='Sequential RANDOM', color='darkred', alpha=0.8)
        bars2 = ax.bar(x + 0.5*width, par_pred_vals, width, label='Parallel Predictor (Best)', color='purple', alpha=0.8)

        # Annotate config inside the predictor bar
        for i, bar in enumerate(bars2):
            val = bar.get_height()
            if not pd.isna(val) and val > 0 and pred_labels[i]:
                ytext = val / 2 if metric != 'hit_rate_pct' else val - 15
                ax.text(bar.get_x() + bar.get_width()/2, ytext, pred_labels[i],
                         ha='center', va='center', fontsize=7,
                         color='white' if metric != 'hit_rate_pct' else 'black',
                         fontweight='bold')

            seq_val = seq_vals[i]
            if not pd.isna(val) and val > 0 and not pd.isna(seq_val) and seq_val > 0:
                if metric in ("tokens_per_second", "stall_loads"):
                    ax.text(bar.get_x() + bar.get_width()/2, val, f"{(val / seq_val):.2f}x",
                             ha='center', va='bottom', fontsize=8, fontweight='bold', color='purple')
                elif metric == "hit_rate_pct":
                    ax.text(bar.get_x() + bar.get_width()/2, val, f"+{(val - seq_val):.1f}%",
                             ha='center', va='bottom', fontsize=8, fontweight='bold', color='purple')

        ax.set_xlabel("Cache Size (Experts)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(cache_sizes)

        if metric == "hit_rate_pct":
            ax.set_ylim(0, 115)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 1.15)

        if col_idx == 0:
            ax.legend(loc='upper left', fontsize=10)

    fig.suptitle("Sequential Baseline vs. Parallel Predictor", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = "/home/michael/heteroPredict/py/meeting_summary_speedup.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

# =========================
# Image 2: Budget Scaling
# =========================
def plot_budget_scaling():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    line_colors = ['darkred', 'purple', 'darkgreen', 'orange']
    markers = ['o', 's', '^', 'D']

    for i, c in enumerate([8, 16, 24]):
        if c not in cache_sizes:
            continue
        ax = axes[i]

        # Parallel RANDOM baseline (flat line)
        sub_rand = par_df[(par_df['cache_size'] == c) & (par_df['label'] == "Neither (RANDOM)")]
        rand_tps = sub_rand.iloc[0]['tokens_per_second'] if not sub_rand.empty else 0

        # Predictor runs
        sub_pred = par_df[(par_df['cache_size'] == c) & (par_df['label'].str.startswith("Predictor"))]
        if not sub_pred.empty:
            lookaheads = sorted(sub_pred['lookahead'].unique())
            for j, la in enumerate(lookaheads):
                la_df = sub_pred[sub_pred['lookahead'] == la].sort_values('prefetch_budget')
                ax.plot(la_df['prefetch_budget'], la_df['tokens_per_second'],
                         marker=markers[j % len(markers)],
                         color=line_colors[j % len(line_colors)],
                         label=f'Predictor LA={int(la)}')

        ax.axhline(rand_tps, color='darkred', linestyle='--', alpha=0.8, label='Parallel RANDOM')

        ax.set_title(f"Cache Size = {c}")
        ax.set_xlabel("Prefetch Budget (Experts)")
        if i == 0:
            ax.set_ylabel("Tokens Per Second (TPS)")
        ax.legend(fontsize=9)

    out_path = "/home/michael/heteroPredict/py/meeting_summary_budget.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

plot_speedup_bars()
plot_budget_scaling()