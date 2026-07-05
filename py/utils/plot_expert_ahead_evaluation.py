#!/usr/bin/env python3
"""
Bar plots for ExpertAhead evaluation (sections 1–3 of the thesis comparison).

Loads fixed-cache comparison sweeps (C8/C16/C24/C32) and produces:
  1) Lookahead ablation: ExpertAhead vs ExpertAhead-CC from 1-token to x tokens
  2) Method comparison: LRU, Cache-Cond, Gating (prior work), ExpertAhead*, ExpertAhead-CC*

Usage:
  source utils/setup.sh
  python py/utils/plot_expert_ahead_evaluation.py \\
    --run-dirs \\
      py/utils/final_results_runs/C8_comparison_20260703_132626 \\
      py/utils/final_results_runs/C16_comparison_20260703_154927 \\
      py/utils/final_results_runs/C24_comparison_20260704_120612 \\
      py/utils/final_results_runs/C32_comparison_20260704_162727 \\
    --out py/utils/final_results_runs/expert_ahead_evaluation.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "text.color": "#111111",
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "font.size": 10,
}

METHOD_COLORS = {
    "RANDOM": "#333333",
    "LRU": "#666666",
    "Cache-Cond": "#E69F00",
    "Gating": "#56B4E9",
    "ExpertAhead*": "#0072B2",
    "ExpertAhead-CC*": "#009E73",
}
EA_COLOR = "#0072B2"
EACC_COLOR = "#009E73"


def _load_sweep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (
        "cache_size", "lookahead", "prefetch_budget", "lambda_val",
        "forced_top_n", "tokens_per_second",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_runs(run_dirs: List[str]) -> pd.DataFrame:
    frames = []
    for d in run_dirs:
        csv_path = os.path.join(d, "sweep.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing sweep.csv in {d}")
        df = _load_sweep(csv_path)
        df["_run_dir"] = os.path.basename(d.rstrip("/"))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    q = out.get("question", pd.Series("", index=out.index)).astype(str)
    out = out.loc[
        q.str.contains("CUSTOM_1_16", na=False) | (q == "GATING_BUDGET_SWEEP")
    ].copy()
    out = out.loc[out["tokens_per_second"].notna()].copy()
    return out


def _lru_tps(sub: pd.DataFrame) -> float:
    row = sub.loc[sub["label"].astype(str).str.contains("Neither (LRU)", regex=False)]
    if row.empty:
        raise ValueError(f"No LRU row for cache_size={sub['cache_size'].iloc[0]}")
    return float(row["tokens_per_second"].iloc[0])


def _random_tps(sub: pd.DataFrame) -> float:
    row = sub.loc[sub["label"].astype(str).str.contains("Neither (RANDOM)", regex=False)]
    if row.empty:
        raise ValueError(f"No RANDOM row for cache_size={sub['cache_size'].iloc[0]}")
    return float(row["tokens_per_second"].iloc[0])


def _ref_tps(s: Dict, baseline: str) -> float:
    return s["random_tps"] if baseline == "random" else s["lru_tps"]


def _gating_tps(sub: pd.DataFrame) -> Optional[float]:
    """Cross-layer gating heuristic (prior work: one token ahead)."""
    rows = sub.loc[sub["question"].astype(str) == "GATING_BUDGET_SWEEP"]
    rows = rows.dropna(subset=["tokens_per_second"])
    if rows.empty:
        return None
    return float(rows["tokens_per_second"].max())


def _cache_cond_tps(sub: pd.DataFrame, forced_top_n: int = 5) -> float:
    mask = (
        sub["label"].astype(str).str.startswith("Cache-Cond Only")
        & (sub["forced_top_n"] == forced_top_n)
    )
    row = sub.loc[mask]
    if row.empty:
        row = sub.loc[sub["label"].astype(str).str.startswith("Cache-Cond Only")]
    return float(row["tokens_per_second"].max())


def _best_per_lookahead(
    sub: pd.DataFrame,
    *,
    lambda_val: float,
    forced_top_n: int = 5,
    prefetch_only: bool,
) -> pd.DataFrame:
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
            & (sub["forced_top_n"] == forced_top_n)
            & sub["label"].astype(str).str.startswith("Both")
        ]
    rows = rows.dropna(subset=["lookahead", "tokens_per_second"])
    if rows.empty:
        return pd.DataFrame()
    idx = rows.groupby("lookahead")["tokens_per_second"].idxmax()
    best = rows.loc[idx].sort_values("lookahead")
    return best[["lookahead", "prefetch_budget", "tokens_per_second"]].copy()


def _pick_optimal_x(curve: pd.DataFrame) -> Tuple[int, float, float]:
    """Return (lookahead, tps, budget) at peak TPS; tie-break toward smaller lookahead."""
    if curve.empty:
        return 1, float("nan"), float("nan")
    curve = curve.sort_values(["tokens_per_second", "lookahead"], ascending=[False, True])
    row = curve.iloc[0]
    return int(row["lookahead"]), float(row["tokens_per_second"]), float(row["prefetch_budget"])


def _summarize_cache(sub: pd.DataFrame) -> Dict:
    cs = int(sub["cache_size"].iloc[0])
    lru = _lru_tps(sub)
    random = _random_tps(sub)
    cc = _cache_cond_tps(sub)

    ea_curve = _best_per_lookahead(sub, lambda_val=0.0, prefetch_only=True)
    eacc_curve = _best_per_lookahead(sub, lambda_val=1.0, prefetch_only=False)

    x_ea, tps_ea, b_ea = _pick_optimal_x(ea_curve)
    x_eacc, tps_eacc, b_eacc = _pick_optimal_x(eacc_curve)

    tps_gating = _gating_tps(sub)
    gating_rows = sub.loc[sub["question"].astype(str) == "GATING_BUDGET_SWEEP"].dropna(subset=["tokens_per_second"])
    gating_b = float(gating_rows.loc[gating_rows["tokens_per_second"].idxmax(), "prefetch_budget"]) if not gating_rows.empty else float("nan")

    return {
        "cache_size": cs,
        "lru_tps": lru,
        "random_tps": random,
        "cc_tps": cc,
        "ea_curve": ea_curve,
        "eacc_curve": eacc_curve,
        "x_ea": x_ea,
        "tps_ea": tps_ea,
        "b_ea": b_ea,
        "x_eacc": x_eacc,
        "tps_eacc": tps_eacc,
        "b_eacc": b_eacc,
        "tps_gating": tps_gating,
        "b_gating": gating_b,
    }


def plot_ablation_panels(ax_row, summaries: List[Dict], baseline: str) -> None:
    ylab = f"TPS / {baseline.upper()}"
    for ax, s in zip(ax_row, summaries):
        cs = s["cache_size"]
        ref = _ref_tps(s, baseline)
        lookaheads = sorted(set(s["ea_curve"]["lookahead"].tolist()) | set(s["eacc_curve"]["lookahead"].tolist()))
        if not lookaheads:
            ax.set_visible(False)
            continue

        x = np.arange(len(lookaheads), dtype=float)
        w = 0.36

        ea_y = []
        eacc_y = []
        for la in lookaheads:
            ea_row = s["ea_curve"].loc[s["ea_curve"]["lookahead"] == la]
            eacc_row = s["eacc_curve"].loc[s["eacc_curve"]["lookahead"] == la]
            ea_y.append(float(ea_row["tokens_per_second"].iloc[0]) / ref if not ea_row.empty else np.nan)
            eacc_y.append(float(eacc_row["tokens_per_second"].iloc[0]) / ref if not eacc_row.empty else np.nan)

        bars_ea = ax.bar(x - w / 2, ea_y, w, label="ExpertAhead", color=EA_COLOR, edgecolor="black", linewidth=0.4)
        bars_eacc = ax.bar(x + w / 2, eacc_y, w, label="ExpertAhead-CC", color=EACC_COLOR, edgecolor="black", linewidth=0.4)

        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.2, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(la)}" for la in lookaheads])
        ax.set_xlabel("Lookahead (future tokens)")
        ax.set_ylabel(ylab)
        ax.set_title(f"C={cs}")
        ax.grid(True, axis="y")
        ax.set_ylim(bottom=0)

        # Mark optimal x*
        if s["x_ea"] in lookaheads:
            i = lookaheads.index(float(s["x_ea"]))
            bars_ea[i].set_edgecolor("#003366")
            bars_ea[i].set_linewidth(2.0)
        if s["x_eacc"] in lookaheads:
            i = lookaheads.index(float(s["x_eacc"]))
            bars_eacc[i].set_edgecolor("#004d33")
            bars_eacc[i].set_linewidth(2.0)


def plot_comparison_panel(ax, summaries: List[Dict], baseline: str) -> None:
    if baseline == "random":
        methods = ["RANDOM", "LRU", "Cache-Cond", "Gating", "ExpertAhead*", "ExpertAhead-CC*"]
        title = "Final methods vs RANDOM / LRU / Cache-Cond / Gating (prior work)"
    else:
        methods = ["LRU", "Cache-Cond", "Gating", "ExpertAhead*", "ExpertAhead-CC*"]
        title = "Final methods vs LRU / Cache-Cond / Gating (prior work)"
    ylab = f"TPS / {baseline.upper()}"
    cache_sizes = [s["cache_size"] for s in summaries]
    n_c = len(cache_sizes)
    n_m = len(methods)
    x = np.arange(n_c, dtype=float)
    w = 0.13 if baseline == "random" else 0.15
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * w

    for mi, method in enumerate(methods):
        vals = []
        for s in summaries:
            ref = _ref_tps(s, baseline)
            if method == "RANDOM":
                v = 1.0
            elif method == "LRU":
                v = s["lru_tps"] / ref
            elif method == "Cache-Cond":
                v = s["cc_tps"] / ref
            elif method == "Gating":
                tg = s.get("tps_gating")
                v = (tg / ref) if tg is not None else np.nan
            elif method == "ExpertAhead*":
                v = s["tps_ea"] / ref
            else:
                v = s["tps_eacc"] / ref
            vals.append(v)
        ax.bar(
            x + offsets[mi], vals, w,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.35,
        )

    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C={cs}" for cs in cache_sizes])
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y")
    ax.set_ylim(bottom=0)


def _annotation_text(summaries: List[Dict]) -> str:
    lines = [
        "Gating = cross-layer router heuristic, one token ahead (prior work).",
        "Optimal lookahead x* for learned predictor (best budget per LA):",
    ]
    for s in summaries:
        gating_part = ""
        if s.get("tps_gating") is not None:
            gating_part = f"; Gating B={int(s['b_gating'])} ({s['tps_gating']:.2f} tok/s)"
        lines.append(
            f"  C={s['cache_size']}: ExpertAhead x*={s['x_ea']} (B={int(s['b_ea'])}, "
            f"{s['tps_ea']:.2f} tok/s); "
            f"ExpertAhead-CC x*={s['x_eacc']} (B={int(s['b_eacc'])}, {s['tps_eacc']:.2f} tok/s)"
            f"{gating_part}"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-dirs", nargs="+", required=True,
        help="Comparison run directories containing sweep.csv",
    )
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument(
        "--baseline", choices=["lru", "random"], default="lru",
        help="Normalize TPS to this baseline (default: lru)",
    )
    args = p.parse_args()

    df = _load_runs(args.run_dirs)
    cache_sizes = sorted(
        df.loc[df["question"].astype(str).str.contains("CUSTOM_1_16", na=False), "cache_size"]
        .dropna().unique().astype(int).tolist()
    )
    summaries = []
    for cs in cache_sizes:
        sub = df.loc[df["cache_size"] == cs].copy()
        summaries.append(_summarize_cache(sub))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with plt.style.context(PLOT_STYLE):
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.38)

        gs_ab = gs[0].subgridspec(1, len(summaries), wspace=0.28)
        ab_axes = [fig.add_subplot(gs_ab[0, i]) for i in range(len(summaries))]
        plot_ablation_panels(ab_axes, summaries, args.baseline)
        if ab_axes:
            ab_axes[0].legend(loc="upper left", fontsize=8)
        fig.text(
            0.5, 0.965,
            f"(1) Lookahead ablation: end-to-end TPS normalized to {args.baseline.upper()}",
            ha="center", va="top", fontsize=12, fontweight="bold",
        )

        ax_cmp = fig.add_subplot(gs[1])
        plot_comparison_panel(ax_cmp, summaries, args.baseline)
        fig.text(
            0.5, 0.47,
            f"(2–3) Gating (prior work) vs multi-token ExpertAhead* vs {args.baseline.upper()} / Cache-Cond",
            ha="center", va="top", fontsize=12, fontweight="bold",
        )

        fig.text(
            0.5, 0.02, _annotation_text(summaries),
            ha="center", va="bottom", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#cccccc"),
        )

        fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved {args.out}")
    print(_annotation_text(summaries))


if __name__ == "__main__":
    main()
