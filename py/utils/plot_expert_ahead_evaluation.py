#!/usr/bin/env python3
"""
ExpertAhead evaluation plots from one or more sweep.csv run directories.

Merges per-cache data (later / more-complete runs win on ties). Produces a
thesis-ready method comparison figure at representative cache sizes.

Usage:
  source utils/setup.sh
  python py/utils/plot_expert_ahead_evaluation.py \\
    --run-dirs py/utils/final_results_runs/C8_comparison_20260703_132626 ... \\
    --out-dir py/utils/final_results_runs/expert_ahead_plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

THESIS_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "text.color": "#111111",
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
}

METHOD_COLORS = {
    "RANDOM": "#999999",
    "Cross-Layer": "#56B4E9",
    "ExpertAhead": "#0072B2",
    "Cache-Cond": "#E69F00",
    "ExpertAhead-CC": "#009E73",
}

METHOD_ORDER = [
    "RANDOM",
    "Cross-Layer",
    "ExpertAhead",
    "Cache-Cond",
    "ExpertAhead-CC",
]

DEFAULT_THESIS_CACHES = [16, 32, 48, 64]


def _load_sweep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (
        "cache_size",
        "lookahead",
        "prefetch_budget",
        "lambda_val",
        "forced_top_n",
        "tokens_per_second",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _row_key(row: pd.Series) -> tuple:
    return (
        int(row["cache_size"]) if pd.notna(row["cache_size"]) else -1,
        str(row.get("label", "")),
        float(row["lookahead"]) if pd.notna(row.get("lookahead")) else -1.0,
        float(row["prefetch_budget"]) if pd.notna(row.get("prefetch_budget")) else -1.0,
        float(row["forced_top_n"]) if pd.notna(row.get("forced_top_n")) else -1.0,
        float(row["lambda_val"]) if pd.notna(row.get("lambda_val")) else -1.0,
        str(row.get("question", "")),
    )


def load_merged_sweeps(run_dirs: List[str]) -> pd.DataFrame:
    """Load sweep.csv from each run dir; merge rows, later dirs override identical configs."""
    frames: List[pd.DataFrame] = []
    for d in run_dirs:
        csv_path = os.path.join(d, "sweep.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing sweep.csv in {d}")
        df = _load_sweep(csv_path)
        df["_run_dir"] = os.path.basename(d.rstrip("/"))
        q = df.get("question", pd.Series("", index=df.index)).astype(str)
        df = df.loc[
            q.str.contains("CUSTOM_1_16", na=False) | (q == "GATING_BUDGET_SWEEP")
        ].copy()
        df = df.loc[df["tokens_per_second"].notna()].copy()
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df["_row_key"] = all_df.apply(_row_key, axis=1)
    all_df = all_df.drop_duplicates(subset=["_row_key"], keep="last").drop(columns=["_row_key"])
    return all_df


def _lru_tps(sub: pd.DataFrame) -> Optional[float]:
    row = sub.loc[sub["label"].astype(str).str.contains("Neither (LRU)", regex=False)]
    if row.empty:
        return None
    return float(row["tokens_per_second"].iloc[0])


def _random_tps(sub: pd.DataFrame) -> float:
    row = sub.loc[sub["label"].astype(str).str.contains("Neither (RANDOM)", regex=False)]
    if row.empty:
        raise ValueError(f"No RANDOM row for cache_size={sub['cache_size'].iloc[0]}")
    return float(row["tokens_per_second"].iloc[0])


def _ref_tps(s: Dict, baseline: str) -> float:
    return s["random_tps"] if baseline == "random" else s["lru_tps"]


def _cross_layer_tps(sub: pd.DataFrame) -> Tuple[Optional[float], float]:
    """Best Cross-Layer (gating) TPS over budget sweep."""
    rows = sub.loc[sub["question"].astype(str) == "GATING_BUDGET_SWEEP"]
    rows = rows.dropna(subset=["tokens_per_second"])
    if rows.empty:
        return None, float("nan")
    best = rows.loc[rows["tokens_per_second"].idxmax()]
    return float(best["tokens_per_second"]), float(best["prefetch_budget"])


def _cache_cond_best(sub: pd.DataFrame) -> Tuple[float, Optional[int]]:
    rows = sub.loc[sub["label"].astype(str).str.startswith("Cache-Cond Only")]
    rows = rows.dropna(subset=["tokens_per_second"])
    if rows.empty:
        return float("nan"), None
    best = rows.loc[rows["tokens_per_second"].idxmax()]
    j = int(best["forced_top_n"]) if pd.notna(best.get("forced_top_n")) else None
    return float(best["tokens_per_second"]), j


def _best_expert_ahead(sub: pd.DataFrame, *, lambda_val: float, prefetch_only: bool) -> Tuple[int, float, float, Optional[int]]:
    """Global best over all swept (S, B[, J]) configs."""
    if prefetch_only:
        rows = sub.loc[
            (sub["backend"] == "predict")
            & (sub["lambda_val"] == 0.0)
            & sub["label"].astype(str).str.startswith("Prefetch Only")
        ]
    else:
        rows = sub.loc[
            (sub["backend"] == "predict")
            & (sub["lambda_val"] == lambda_val)
            & sub["label"].astype(str).str.startswith("Both")
        ]
    rows = rows.dropna(subset=["lookahead", "prefetch_budget", "tokens_per_second"])
    if rows.empty:
        return 1, float("nan"), float("nan"), None
    rows = rows.sort_values(["tokens_per_second", "lookahead"], ascending=[False, True])
    row = rows.iloc[0]
    j = int(row["forced_top_n"]) if "forced_top_n" in row and pd.notna(row["forced_top_n"]) else None
    return int(row["lookahead"]), float(row["tokens_per_second"]), float(row["prefetch_budget"]), j


def _summarize_cache(sub: pd.DataFrame) -> Dict:
    cs = int(sub["cache_size"].iloc[0])
    lru = _lru_tps(sub)
    random = _random_tps(sub)
    cc, cc_j = _cache_cond_best(sub)

    x_ea, tps_ea, b_ea, _ = _best_expert_ahead(sub, lambda_val=0.0, prefetch_only=True)
    x_eacc, tps_eacc, b_eacc, j_eacc = _best_expert_ahead(sub, lambda_val=1.0, prefetch_only=False)

    la1_rows = sub.loc[
        (sub["backend"] == "predict")
        & (sub["lambda_val"] == 0.0)
        & sub["label"].astype(str).str.startswith("Prefetch Only")
        & (sub["lookahead"] == 1)
    ].dropna(subset=["tokens_per_second"])
    if not la1_rows.empty:
        la1 = la1_rows.loc[la1_rows["tokens_per_second"].idxmax()]
        tps_la1 = float(la1["tokens_per_second"])
        b_la1 = float(la1["prefetch_budget"])
    else:
        tps_la1, b_la1 = float("nan"), float("nan")

    tps_cross_layer, b_cross_layer = _cross_layer_tps(sub)

    return {
        "cache_size": cs,
        "lru_tps": lru if lru is not None else float("nan"),
        "random_tps": random,
        "cc_tps": cc,
        "cc_j": cc_j,
        "x_ea": x_ea,
        "tps_ea": tps_ea,
        "b_ea": b_ea,
        "tps_la1": tps_la1,
        "b_la1": b_la1,
        "x_eacc": x_eacc,
        "tps_eacc": tps_eacc,
        "b_eacc": b_eacc,
        "j_eacc": j_eacc,
        "tps_cross_layer": tps_cross_layer,
        "b_cross_layer": b_cross_layer,
        # Back-compat aliases used elsewhere
        "tps_gating": tps_cross_layer,
        "b_gating": b_cross_layer,
    }


def summarize_all(df: pd.DataFrame) -> List[Dict]:
    cache_sizes = sorted(df["cache_size"].dropna().unique().astype(int).tolist())
    return [_summarize_cache(df.loc[df["cache_size"] == cs].copy()) for cs in cache_sizes]


def select_summaries(summaries: List[Dict], cache_sizes: Optional[List[int]]) -> List[Dict]:
    by_c = {s["cache_size"]: s for s in summaries}
    if cache_sizes is None:
        cache_sizes = DEFAULT_THESIS_CACHES
    selected = [by_c[c] for c in cache_sizes if c in by_c]
    if not selected:
        raise ValueError(f"No summaries for cache sizes {cache_sizes}")
    return selected


def _method_tps(s: Dict, method: str) -> float:
    if method == "RANDOM":
        return s["random_tps"]
    if method == "Cross-Layer":
        tg = s.get("tps_cross_layer")
        return float(tg) if tg is not None else float("nan")
    if method == "Cache-Cond":
        return s["cc_tps"]
    if method == "ExpertAhead":
        return s["tps_ea"]
    if method == "ExpertAhead-CC":
        return s["tps_eacc"]
    raise KeyError(method)


def _display_label(method: str) -> str:
    return method


def plot_methods_panel(ax, s: Dict, *, baseline: str, show_ylabel: bool) -> None:
    ref = _ref_tps(s, baseline)
    x = np.arange(len(METHOD_ORDER), dtype=float)
    vals = []
    for method in METHOD_ORDER:
        tps = _method_tps(s, method)
        if baseline == "absolute":
            vals.append(tps)
        else:
            vals.append((tps / ref) if pd.notna(tps) and ref > 0 else np.nan)

    bars = ax.bar(
        x,
        vals,
        0.72,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )

    if baseline != "absolute":
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([_display_label(m) for m in METHOD_ORDER], rotation=28, ha="right")
    ax.set_title(f"C = {s['cache_size']}", pad=8)
    ax.grid(True, axis="y", zorder=0)
    ax.set_ylim(bottom=0)

    ymax = max([v for v in vals if pd.notna(v)] or [1.0])
    ax.set_ylim(top=ymax * 1.16)

    if show_ylabel:
        if baseline == "absolute":
            ax.set_ylabel("Throughput (tokens/s)")
        else:
            pass
            # ax.set_ylabel(f"Normalized throughput")

    for bar, val in zip(bars, vals):
        if pd.isna(val):
            bar.set_alpha(0.25)
            bar.set_hatch("//")


def render_evaluation_figure(
    summaries: List[Dict],
    out_path: str,
    *,
    baseline: str = "lru",
    dpi: int = 300,
) -> None:
    n = len(summaries)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    with plt.style.context(THESIS_STYLE):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.05 * ncols, 2.85 * nrows + 0.55),
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for ax, s in zip(flat_axes, summaries):
            plot_methods_panel(ax, s, baseline=baseline, show_ylabel=(ax is flat_axes[0]))

        for ax in flat_axes[len(summaries) :]:
            ax.set_visible(False)

        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[m], edgecolor="black", linewidth=0.4)
            for m in METHOD_ORDER
        ]
        fig.legend(
            handles,
            [_display_label(m) for m in METHOD_ORDER],
            loc="upper center",
            ncol=5,
            frameon=True,
            bbox_to_anchor=(0.5, 1.08),
            columnspacing=1.0,
            handletextpad=0.4,
        )

        fig.subplots_adjust(top=0.82, bottom=0.22, left=0.07, right=0.99, wspace=0.28, hspace=0.55)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def summaries_to_table(summaries: List[Dict]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        xl = s.get("tps_cross_layer")
        rows.append(
            {
                "cache_size": s["cache_size"],
                "random_tps": s["random_tps"],
                "lru_tps": s["lru_tps"],
                "cross_layer_tps": xl,
                "cross_layer_budget": s["b_cross_layer"],
                "cache_cond_tps": s["cc_tps"],
                "cache_cond_j": s.get("cc_j"),
                "ea_s": s["x_ea"],
                "ea_tps": s["tps_ea"],
                "ea_budget": s["b_ea"],
                "ea_s1_tps": s["tps_la1"],
                "ea_s1_budget": s["b_la1"],
                "eacc_s": s["x_eacc"],
                "eacc_tps": s["tps_eacc"],
                "eacc_budget": s["b_eacc"],
                "eacc_j": s["j_eacc"],
                "ea_vs_cross_layer": (s["tps_ea"] / xl) if xl else np.nan,
                "eacc_vs_cache_cond": (s["tps_eacc"] / s["cc_tps"])
                if pd.notna(s["tps_eacc"]) and s["cc_tps"]
                else np.nan,
                "eacc_vs_cross_layer": (s["tps_eacc"] / xl) if xl and pd.notna(s["tps_eacc"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True, help="Run dirs with sweep.csv")
    p.add_argument("--out-dir", required=True, help="Directory for output PNGs")
    p.add_argument(
        "--cache-sizes",
        nargs="+",
        type=int,
        default=None,
        help=f"Representative cache sizes for thesis figure (default: {DEFAULT_THESIS_CACHES})",
    )
    p.add_argument(
        "--baseline",
        choices=["lru", "random", "absolute", "both", "all", "thesis"],
        default="thesis",
        help="Y-axis mode; 'thesis' writes RANDOM-normalized + absolute plots",
    )
    p.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    args = p.parse_args()

    df = load_merged_sweeps(args.run_dirs)
    if df.empty:
        raise SystemExit("No usable sweep rows after merge.")

    all_summaries = summarize_all(df)
    selected = select_summaries(all_summaries, args.cache_sizes)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.baseline in ("lru", "both", "all"):
        if any(pd.notna(s.get("lru_tps")) for s in selected):
            out = os.path.join(args.out_dir, "expert_ahead_evaluation_lru.png")
            render_evaluation_figure(selected, out, baseline="lru", dpi=args.dpi)
            print(f"Saved {out}")
        else:
            print("Skipping LRU-normalized plot (no Neither (LRU) rows in data).")

    if args.baseline in ("random", "both", "all", "thesis"):
        out = os.path.join(args.out_dir, "expert_ahead_evaluation_random.png")
        render_evaluation_figure(selected, out, baseline="random", dpi=args.dpi)
        print(f"Saved {out}")

    if args.baseline in ("absolute", "all", "thesis"):
        out = os.path.join(args.out_dir, "expert_ahead_evaluation_absolute.png")
        render_evaluation_figure(selected, out, baseline="absolute", dpi=args.dpi)
        print(f"Saved {out}")

    summary_csv = os.path.join(args.out_dir, "expert_ahead_summary.csv")
    summaries_to_table(all_summaries).to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv}")

    thesis_csv = os.path.join(args.out_dir, "expert_ahead_thesis_table.csv")
    table = summaries_to_table(selected)
    table.to_csv(thesis_csv, index=False)
    print(f"Saved {thesis_csv}")
    print("\nBest-config speedups (thesis caches):")
    for _, r in table.iterrows():
        xl = r["ea_vs_cross_layer"]
        cc = r["eacc_vs_cache_cond"]
        xl_s = f"{xl:.2f}x vs Cross-Layer" if pd.notna(xl) else "n/a vs Cross-Layer"
        cc_s = f"{cc:.2f}x vs Cache-Cond" if pd.notna(cc) else "n/a vs Cache-Cond"
        xl_b = int(r["cross_layer_budget"]) if pd.notna(r["cross_layer_budget"]) else "?"
        eacc_s = int(r["eacc_s"]) if pd.notna(r["eacc_s"]) else "?"
        eacc_b = int(r["eacc_budget"]) if pd.notna(r["eacc_budget"]) else "?"
        print(
            f"  C={int(r['cache_size'])}: ExpertAhead {xl_s}; "
            f"ExpertAhead-CC {cc_s} "
            f"(EA S={int(r['ea_s'])} B={int(r['ea_budget'])}, "
            f"EA-CC S={eacc_s} B={eacc_b}, XL B={xl_b})"
        )


if __name__ == "__main__":
    main()
