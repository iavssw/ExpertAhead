#!/usr/bin/env python3
"""
Thesis results figures, in a fixed-cache x lookahead layout.

Each sweep fixes the expert cache size C, puts predictor lookahead N on the x-axis,
and draws one line per prefetch budget B (or routing variant) with LRU as the dashed
baseline. The CSVs come from ``finals_experiment_runner.py`` (see ``run_finals.sh``):

  --csv-sec2  <- sec2_predictor_effectiveness/<ts>/sweep.csv   (sections 2 and 3)
  --csv-sec4  <- sec4_routing_topj_vs_pm/<ts>/sweep.csv        (section 4)
  --csv-sec5  <- sec5_all_methods/<ts>/sweep.csv               (section 5)

Run::

  source utils/setup.sh
  python py/utils/plot_thesis_finals.py \\
    --csv-sec2 py/utils/final_results_runs/sec2_predictor_effectiveness/<ts>/sweep.csv \\
    --csv-sec4 py/utils/final_results_runs/sec4_routing_topj_vs_pm/<ts>/sweep.csv \\
    --csv-sec5 py/utils/final_results_runs/sec5_all_methods/<ts>/sweep.csv \\
    --out-dir py/utils/final_results_runs/thesis_plots

Outputs:
  sec2_predictor_effectiveness.png   — TPS, recall, precision vs N (fixed C)
  sec2_stalls_vs_tps.png             — attribution scatter (prefetch rows)
  sec3_predictor_speedup_vs_lru.png  — max prefetch TPS / LRU vs N
  sec4_routing_fn_pm.png             — forced-top-J vs PM vs LRU (panels per lookahead)
  sec5_methods_tps_ppl_vs_lookahead.png — four policies vs N at ~50% B (baseline = RANDOM by default)
  sec5_tps_ppl_tradeoff.png          — TPS vs perplexity scatter per cache size
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from sweep_predict_cached_cache_metrics import (  # noqa: E402
    LRU_STYLE,
    PLOT_STYLE,
    plot_custom_1_16,
    plot_prefetch_speedup_attribution,
    save_plot,
    style_ax,
)

POLICY_ORDER = ["LRU", "Prefetch", "Cache-Cond", "Hybrid"]
BASELINE_LABELS = {
    "lru": "Neither (LRU)",
    "random": "Neither (RANDOM)",
}
BASELINE_PLOT_NAMES = {
    "lru": "LRU",
    "random": "Random",
}
POLICY_STYLES = {
    "LRU": dict(**LRU_STYLE, marker="D"),
    "Random": dict(**LRU_STYLE, marker="D"),
    "Prefetch": dict(color="#0072B2", marker="^", linewidth=2),
    "Cache-Cond": dict(color="#E69F00", marker="o", linewidth=2, linestyle=":"),
    "Hybrid": dict(color="#009E73", marker="s", linewidth=2, linestyle="-"),
}
ROUTING_ORDER = ["LRU", "FN=4", "FN=6", "PM=0.5", "PM=0.7", "PM=0.9"]
ROUTING_COLORS = {
    "LRU": "#666666",
    "FN=4": "#56B4E9",
    "FN=6": "#0072B2",
    "PM=0.5": "#009E73",
    "PM=0.7": "#E69F00",
    "PM=0.9": "#D55E00",
}


def _load_csv_raw(path: str) -> pd.DataFrame:
    """Load sweep CSV without normalizing TPS to LRU."""
    df = pd.read_csv(path)
    for col in (
        "cache_size", "lookahead", "lambda_val", "prefetch_budget", "forced_top_n",
        "prob_mass_threshold", "tokens_per_second", "gen_perplexity",
        "pred_hit_rate_routed_topk_pct", "pred_requested_rate_topk_pct",
        "stall_loads", "prefetch_loads",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_csv(path: str) -> pd.DataFrame:
    df = _load_csv_raw(path)

    # Normalize tokens_per_second to LRU baseline
    if "tokens_per_second" in df.columns:
        lru_mask = (df["label"].astype(str).str.contains("LRU", na=False))
        if "backend" in df.columns:
            lru_mask |= (df["backend"] == "cached") & (pd.to_numeric(df.get("lambda_val", 0.0), errors="coerce").fillna(0.0) == 0.0)
        
        df_lru = df[lru_mask]
        lru_map = {}
        for c in df_lru["cache_size"].dropna().unique():
            sub = df_lru[df_lru["cache_size"] == c]
            m = sub["tokens_per_second"].dropna().mean()
            if pd.notna(m) and m > 0:
                lru_map[c] = m
                
        def norm(r):
            tps = r.get("tokens_per_second")
            if pd.isna(tps): return tps
            ref = lru_map.get(r.get("cache_size"))
            return float(tps) / ref if ref else tps

        df["tokens_per_second"] = df.apply(norm, axis=1)

    return df


def _filter_custom(df: pd.DataFrame) -> pd.DataFrame:
    q = df.get("question", pd.Series("", index=df.index)).astype(str)
    return df.loc[
        q.str.contains("CUSTOM_1_16", na=False) | q.str.match(r"SEC5_C\d+_HYBRID_GRID", na=False)
    ].copy()


def _filter_routing(df: pd.DataFrame) -> pd.DataFrame:
    q = df.get("question", pd.Series("", index=df.index)).astype(str)
    return df.loc[q == "LAMBDA_FN_SWEEP"].copy()


def _at_cache(df: pd.DataFrame, cache_size: int) -> pd.DataFrame:
    return df.loc[df["cache_size"] == cache_size].copy()


def _budget_fraction(row: pd.Series) -> Optional[float]:
    b, c = row.get("prefetch_budget"), row.get("cache_size")
    if pd.isna(b) or pd.isna(c) or float(c) <= 0:
        return None
    return float(b) / float(c)


def _policy_label(label: str, lambda_val: float) -> Optional[str]:
    s = str(label)
    if s == "Neither (LRU)":
        return "LRU"
    if s == "Neither (RANDOM)":
        return "Random"
    if s.startswith("Prefetch Only"):
        return "Prefetch" if float(lambda_val) == 0.0 else None
    if s.startswith("Cache-Cond Only"):
        return "Cache-Cond"
    if s.startswith("Both"):
        return "Hybrid"
    return None


def _routing_short_label(row: pd.Series) -> str:
    if str(row.get("label", "")).startswith("LRU"):
        return "LRU"
    fn = int(row.get("forced_top_n", 0) or 0)
    if fn > 0:
        return f"FN={fn}"
    pm = float(row.get("prob_mass_threshold", -1))
    if pm >= 0:
        return f"PM={pm:g}"
    return "?"


def _rename_plot(out_dir: str, src_name: str, dst_name: str) -> None:
    src = os.path.join(out_dir, src_name)
    dst = os.path.join(out_dir, dst_name)
    if os.path.isfile(src):
        os.replace(src, dst)
        print(f"[thesis_plot] Saved {dst}", flush=True)


def _recall_col(df: pd.DataFrame) -> str:
    if "pred_hit_rate_routed_topk_pct" in df.columns and df["pred_hit_rate_routed_topk_pct"].notna().any():
        return "pred_hit_rate_routed_topk_pct"
    return "pred_hit_rate_pct"


def _precision_col(df: pd.DataFrame) -> str:
    if "pred_requested_rate_topk_pct" in df.columns and df["pred_requested_rate_topk_pct"].notna().any():
        return "pred_requested_rate_topk_pct"
    return "pred_requested_rate_forced_n_pct"


def plot_fixed_cache_lookahead_grid(
    df: pd.DataFrame,
    cache_size: int,
    out_dir: str,
    filename: str,
    *,
    metrics: Sequence[Tuple[str, str]],
    title: str,
    include_lru: bool = True,
    include_prefetch_lines: bool = True,
    include_routing_lines: Optional[Sequence[str]] = None,
    annotate_tps_speedup: bool = True,
) -> None:
    """
  One column, multiple metric rows — same layout as ``custom_1_16_summary`` at fixed C.

  Prefetch curves: one line per prefetch budget B.
  Optional ``include_routing_lines``: list of label prefixes for cache-cond variants.
    """
    sub = _at_cache(df, cache_size)
    if sub.empty:
        print(f"[thesis_plot] No rows for cache_size={cache_size}", flush=True)
        return

    lookaheads = sorted(sub["lookahead"].dropna().unique())
    pref_mask = sub["label"].astype(str).str.startswith("Prefetch Only", na=False)
    budgets = sorted(sub.loc[pref_mask, "prefetch_budget"].dropna().unique(), key=float)
    n_budgets = max(len(budgets), 1)
    budget_cmap = plt.cm.Blues

    with plt.style.context(PLOT_STYLE):
        nrows = len(metrics)
        fig, axes = plt.subplots(nrows, 1, figsize=(8.5, 4.2 * nrows), squeeze=False)
        fig.suptitle(f"{title}  (C = {cache_size} experts/layer)", fontsize=13, fontweight="bold", y=1.01)

        for row, (metric_col, ylabel) in enumerate(metrics):
            ax = axes[row][0]
            is_tps = metric_col == "tokens_per_second"
            lru_vals: Dict[float, float] = {}

            if include_lru:
                s_lru = (
                    sub[sub["label"] == "Neither (LRU)"]
                    .dropna(subset=[metric_col, "lookahead"])
                    .sort_values("lookahead")
                )
                if not s_lru.empty:
                    ax.plot(
                        s_lru["lookahead"], s_lru[metric_col],
                        marker="D", **LRU_STYLE, label="LRU (full router)",
                    )
                    for _, r in s_lru.iterrows():
                        lru_vals[float(r["lookahead"])] = float(r[metric_col])

            if include_prefetch_lines:
                for b_idx, budget in enumerate(budgets):
                    frac = budget / cache_size if cache_size else 0
                    color = budget_cmap(0.3 + 0.6 * b_idx / (n_budgets - 1) if n_budgets > 1 else 0.7)
                    s_b = (
                        sub[pref_mask & (sub["prefetch_budget"] == budget)]
                        .dropna(subset=[metric_col, "lookahead"])
                        .sort_values("lookahead")
                    )
                    if s_b.empty:
                        continue
                    ax.plot(
                        s_b["lookahead"], s_b[metric_col],
                        marker="^", color=color, linewidth=2,
                        label=f"Prefetch B={int(budget)} ({frac:.0%})",
                    )
                    if annotate_tps_speedup and is_tps and lru_vals:
                        for _, r in s_b.iterrows():
                            la = float(r["lookahead"])
                            ref = lru_vals.get(la)
                            if ref and ref > 0:
                                val = float(r[metric_col])
                                ax.annotate(
                                    f"×{val/ref:.2f}",
                                    xy=(la, val), xytext=(0, 6),
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=6, color=color, fontweight="bold",
                                )

            if include_routing_lines:
                for pfx in include_routing_lines:
                    mask = sub["label"].astype(str).str.startswith(pfx, na=False)
                    s = sub.loc[mask].dropna(subset=[metric_col, "lookahead"]).sort_values("lookahead")
                    if s.empty:
                        continue
                    short = "Cache-cond" if "Cache-Cond" in pfx else "Hybrid"
                    ax.plot(
                        s["lookahead"], s[metric_col],
                        marker="o", linewidth=1.8, linestyle=":",
                        label=short,
                    )

            ax.set_xticks(lookaheads)
            style_ax(ax, "", "Lookahead N (predictor horizon)", ylabel)
            if row == 0:
                ax.legend(fontsize=7, loc="best", ncol=2)

        plt.tight_layout()
        save_plot(fig, out_dir, filename)


# ---------------------------------------------------------------------------
# §2 Predictor effectiveness (fixed cache, lossless → TPS + ρ + π)
# ---------------------------------------------------------------------------
def plot_sec2_predictor_effectiveness(df: pd.DataFrame, cache_size: int, out_dir: str) -> None:
    sub = _filter_custom(_at_cache(df, cache_size))
    if sub.empty:
        return
    rc, pc = _recall_col(sub), _precision_col(sub)
    plot_fixed_cache_lookahead_grid(
        sub, cache_size, out_dir, "sec2_predictor_effectiveness.png",
        metrics=[
            ("tokens_per_second", "TPS Speedup (vs LRU)"),
            (rc, "Predictor recall ρ on routed experts (%)"),
            (pc, "Predictor precision π on prefetches (%)"),
        ],
        title="Predictor effectiveness",
        include_lru=True,
        include_prefetch_lines=True,
        annotate_tps_speedup=True,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_prefetch_speedup_attribution(sub, out_dir, ts)
    _rename_plot(out_dir, f"prefetch_speedup_attribution_{ts}.png", "sec2_stalls_vs_tps.png")


# ---------------------------------------------------------------------------
# §3 Predictor-only vs LRU — max lossless speedup at each lookahead
# ---------------------------------------------------------------------------
def plot_sec3_predictor_speedup(df: pd.DataFrame, cache_size: int, out_dir: str) -> None:
    sub = _filter_custom(_at_cache(df, cache_size))
    lru = sub[sub["label"] == "Neither (LRU)"].dropna(subset=["tokens_per_second", "lookahead"])
    pref = sub[sub["label"].astype(str).str.startswith("Prefetch Only", na=False)]

    rows: List[dict] = []
    for la in sorted(pref["lookahead"].dropna().unique()):
        lru_row = lru.loc[lru["lookahead"] == la]
        if lru_row.empty:
            continue
        tps_lru = float(lru_row["tokens_per_second"].iloc[0])
        if tps_lru <= 0:
            continue
        p_la = pref.loc[pref["lookahead"] == la]
        if p_la.empty:
            continue
        best_idx = p_la["tokens_per_second"].idxmax()
        best = p_la.loc[best_idx]
        rows.append({
            "lookahead": int(la),
            "tps_lru": tps_lru,
            "tps_best": float(best["tokens_per_second"]),
            "speedup": float(best["tokens_per_second"]) / tps_lru,
            "best_B": int(best["prefetch_budget"]) if pd.notna(best["prefetch_budget"]) else None,
        })
    if not rows:
        return
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "sec3_best_prefetch_per_lookahead.csv"), index=False)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(tab["lookahead"], tab["speedup"], marker="^", color="#0072B2", linewidth=2.5, label="Best prefetch / LRU")
        ax.axhline(1.0, **LRU_STYLE, label="No speedup")
        for _, r in tab.iterrows():
            ax.annotate(
                f"B={int(r['best_B'])}" if r["best_B"] is not None else "",
                xy=(r["lookahead"], r["speedup"]),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, color="#0072B2",
            )
        style_ax(ax, f"Max predictor speedup vs LRU (C = {cache_size})", "Lookahead N", "TPS speedup ×")
        ax.legend(fontsize=9)
        plt.tight_layout()
        save_plot(fig, out_dir, "sec3_predictor_speedup_vs_lru.png")


def plot_best_lookahead_per_cache(df: pd.DataFrame, out_dir: str) -> None:
    """Plot the best lookahead (and speedup) associated with each cache size."""
    df = df.copy()
    if df.empty:
        return

    # Filter to CUSTOM questions
    df_custom = _filter_custom(df)
    if df_custom.empty:
        return

    # Get LRU baselines
    lru_mask = (df_custom["label"] == "Neither (LRU)") | (df_custom["label"] == "LRU Baseline")
    if "backend" in df_custom.columns:
        lru_mask |= (df_custom["backend"] == "cached") & (df_custom.get("lambda_val", 0.0).fillna(0.0) == 0.0)
    df_lru = df_custom[lru_mask]

    # Predictor rows
    pred_mask = df_custom["backend"] == "predict"
    df_pred = df_custom[pred_mask].dropna(subset=["tokens_per_second", "lookahead"])

    cache_sizes = sorted(df_pred["cache_size"].dropna().unique(), key=int)
    rows = []
    for c in cache_sizes:
        sub_lru = df_lru[df_lru["cache_size"] == c]
        if sub_lru.empty or sub_lru["tokens_per_second"].isna().all():
            continue
        lru_tps = sub_lru["tokens_per_second"].mean()

        sub_pred = df_pred[df_pred["cache_size"] == c]
        if sub_pred.empty:
            continue

        # Find the row that maximizes tokens_per_second
        best_row_idx = sub_pred["tokens_per_second"].idxmax()
        best_row = sub_pred.loc[best_row_idx]

        best_la = int(best_row["lookahead"])
        best_b = int(best_row["prefetch_budget"]) if pd.notna(best_row["prefetch_budget"]) else None
        best_tps = float(best_row["tokens_per_second"])
        speedup = best_tps / lru_tps

        rows.append({
            "cache_size": int(c),
            "best_lookahead": best_la,
            "best_budget": best_b,
            "lru_tps": lru_tps,
            "best_tps": best_tps,
            "speedup": speedup
        })

    if not rows:
        print("[thesis_plot] No cache sizes with both LRU and predictor data found for Section 3 plot.", flush=True)
        return

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "sec3_best_lookahead_per_cache.csv"), index=False)
    print(f"[thesis_plot] Saved best lookahead table to {os.path.join(out_dir, 'sec3_best_lookahead_per_cache.csv')}", flush=True)

    with plt.style.context(PLOT_STYLE):
        # Plot 1: Best Lookahead depth vs Cache Size
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.plot(tab["cache_size"], tab["best_lookahead"], marker="o", color="#009E73", linewidth=2.5, label="Best Lookahead Depth")
        
        # Annotate with speedup and budget
        for _, r in tab.iterrows():
            b_str = f", B={int(r['best_budget'])}" if r["best_budget"] is not None else ""
            lbl = f"LA={int(r['best_lookahead'])}{b_str}\n({r['speedup']:.2f}x speedup)"
            ax.annotate(
                lbl,
                xy=(r["cache_size"], r["best_lookahead"]),
                xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8,
                color="#0072B2", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="#cccccc", alpha=0.8)
            )
        
        # Add some padding to top of y-limit for labels
        ax.set_ylim(bottom=0, top=max(tab["best_lookahead"]) + 4)
        style_ax(ax, "Optimal Lookahead Depth vs Expert Cache Size", "Expert Cache Size (C) [experts/layer]", "Best Lookahead Depth (N) [tokens]")
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        save_plot(fig, out_dir, "sec3_best_lookahead_vs_cache.png")

        # Plot 2: Speedup vs Cache Size
        fig2, ax2 = plt.subplots(figsize=(8.5, 5.0))
        ax2.plot(tab["cache_size"], tab["speedup"], marker="^", color="#0072B2", linewidth=2.5, label="Max Speedup")
        ax2.axhline(1.0, **LRU_STYLE, label="LRU Baseline (1.00x)")
        
        for _, r in tab.iterrows():
            b_str = f", B={int(r['best_budget'])}" if r["best_budget"] is not None else ""
            lbl = f"LA={int(r['best_lookahead'])}{b_str}"
            ax2.annotate(
                lbl,
                xy=(r["cache_size"], r["speedup"]),
                xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8,
                color="#009E73", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="#cccccc", alpha=0.8)
            )
            
        ax2.set_ylim(bottom=0.8, top=max(tab["speedup"]) + 0.15)
        style_ax(ax2, "Maximum Prefetch Speedup vs Expert Cache Size", "Expert Cache Size (C) [experts/layer]", "TPS Speedup (relative to LRU) [x]")
        ax2.grid(True, alpha=0.5)
        plt.tight_layout()
        save_plot(fig2, out_dir, "sec3_best_speedup_vs_cache.png")


# ---------------------------------------------------------------------------
# §4 Perplexity degradation: FN vs PM (exp 4 — panel per lookahead)
# ---------------------------------------------------------------------------
def plot_sec4_routing_fn_pm(df: pd.DataFrame, out_dir: str) -> None:
    sub = _filter_routing(df).copy()
    if sub.empty:
        return
    sub["routing_lbl"] = sub.apply(_routing_short_label, axis=1)
    lookaheads = sorted(sub["lookahead"].dropna().unique())
    n = len(lookaheads)
    if n == 0:
        return

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), squeeze=False)
        fig.suptitle(
            "Cache-conditional routing: forced top-J vs probability-mass (λ=1, B=C per grid point)",
            fontsize=12, fontweight="bold", y=1.02,
        )
        for col, la in enumerate(lookaheads):
            panel = sub[sub["lookahead"] == la]
            c = int(panel["cache_size"].iloc[0]) if not panel.empty else 0
            labels = [lbl for lbl in ROUTING_ORDER if (panel["routing_lbl"] == lbl).any()]
            x = np.arange(len(labels))
            for row, (metric, ylab) in enumerate((
                ("tokens_per_second", "TPS Speedup (vs LRU)"),
                ("gen_perplexity", "Gen perplexity (↓)"),
            )):
                ax = axes[row][col]
                vals = [
                    float(panel.loc[panel["routing_lbl"] == lbl, metric].iloc[0])
                    if (panel["routing_lbl"] == lbl).any() else np.nan
                    for lbl in labels
                ]
                colors = [ROUTING_COLORS.get(lbl, "#888") for lbl in labels]
                ax.bar(x, vals, color=colors, edgecolor="#3a3d4d")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
                style_ax(ax, f"N={int(la)}, C={c}", "", ylab)
                ax.grid(True, axis="y", alpha=0.5)
        plt.tight_layout()
        save_plot(fig, out_dir, "sec4_routing_fn_pm.png")


# ---------------------------------------------------------------------------
# §5 Four-way comparison (exp 3) — canonical one row per (N, policy) at ~50% B
# ---------------------------------------------------------------------------
def _add_policy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["policy"] = [
        _policy_label(l, lam) for l, lam in zip(out.get("label", ""), out.get("lambda_val", 0))
    ]
    out["budget_frac"] = out.apply(_budget_fraction, axis=1)
    return out


def _sec5_policy_order(baseline: str) -> List[str]:
    """Baseline eviction policy + the three active policies (prefetch / cache-cond / hybrid)."""
    base = BASELINE_PLOT_NAMES.get(baseline.lower(), BASELINE_PLOT_NAMES["random"])
    return [base, "Prefetch", "Cache-Cond", "Hybrid"]


def _canonical_at_budget_frac(
    df: pd.DataFrame,
    budget_frac: float,
    *,
    baseline: str = "random",
    tol: float = 0.06,
) -> pd.DataFrame:
    """One row per (lookahead, policy): closest prefetch budget fraction to target."""
    baseline_policy = BASELINE_PLOT_NAMES.get(baseline.lower(), BASELINE_PLOT_NAMES["random"])
    rows: List[pd.Series] = []
    for la in sorted(df["lookahead"].dropna().unique()):
        for pol in _sec5_policy_order(baseline):
            m = (df["lookahead"] == la) & (df["policy"] == pol)
            if pol != baseline_policy:
                m &= df["budget_frac"].notna()
            sub = df.loc[m]
            if sub.empty:
                continue
            if pol == baseline_policy:
                rows.append(sub.iloc[0])
                continue
            j = (sub["budget_frac"] - budget_frac).abs().idxmin()
            rows.append(sub.loc[j])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_sec5_methods(
    df: pd.DataFrame,
    out_dir: str,
    *,
    budget_frac: float = 0.5,
    baseline: str = "random",
    cache_size: Optional[int] = None,
) -> None:
    baseline_policy = BASELINE_PLOT_NAMES.get(baseline.lower(), BASELINE_PLOT_NAMES["random"])
    baseline_style = POLICY_STYLES.get(baseline_policy, POLICY_STYLES["Random"])
    df = _add_policy_columns(_filter_custom(df))
    if cache_size is not None:
        df = df[df["cache_size"] == cache_size]
    canon = _canonical_at_budget_frac(df, budget_frac, baseline=baseline)
    if canon.empty:
        print("[thesis_plot] sec5: no canonical policy rows", flush=True)
        return
    canon.to_csv(os.path.join(out_dir, "sec5_canonical_configs.csv"), index=False)

    cache_sizes = sorted(canon["cache_size"].dropna().unique())
    c_title = (
        f"C = {int(cache_sizes[0])}"
        if cache_size is not None or len(cache_sizes) == 1
        else "C = expert-reuse min per lookahead"
    )

    # (A) Lookahead sweep at canonical B/C per policy.
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), squeeze=False)
        fig.suptitle(
            f"Four policies @ ≈{budget_frac:.0%} B/C  ({c_title}, baseline = {baseline_policy})",
            fontsize=12, fontweight="bold", y=1.01,
        )
        las = sorted(canon["lookahead"].dropna().unique())
        for row, (metric, ylab) in enumerate((
            ("tokens_per_second", f"TPS (× vs {baseline_policy})"),
            ("gen_perplexity", "Gen perplexity (↓)"),
        )):
            ax = axes[row][0]
            base_by_la: Dict[float, float] = {}
            base_sub = canon[canon["policy"] == baseline_policy].sort_values("lookahead")
            if not base_sub.empty:
                ax.plot(
                    base_sub["lookahead"], base_sub[metric],
                    label=baseline_policy, **baseline_style,
                )
                for _, r in base_sub.iterrows():
                    base_by_la[float(r["lookahead"])] = float(r[metric])
            for pol in ("Prefetch", "Cache-Cond", "Hybrid"):
                s = canon[canon["policy"] == pol].sort_values("lookahead")
                if s.empty:
                    continue
                st = POLICY_STYLES[pol]
                ax.plot(s["lookahead"], s[metric], label=pol, **st)
                for _, r in s.iterrows():
                    la = float(r["lookahead"])
                    c = int(r["cache_size"]) if pd.notna(r.get("cache_size")) else None
                    ref = base_by_la.get(la) if metric == "tokens_per_second" else None
                    if metric == "tokens_per_second" and ref and ref > 0:
                        ax.annotate(
                            f"×{float(r[metric])/ref:.2f}",
                            xy=(la, float(r[metric])), xytext=(0, 5),
                            textcoords="offset points", ha="center", fontsize=6,
                            color=st.get("color", "#ccc"),
                        )
                    elif c is not None and row == 0:
                        ax.annotate(
                            f"C={c}",
                            xy=(la, float(r[metric])), xytext=(0, -10),
                            textcoords="offset points", ha="center", fontsize=5,
                            color="#888", alpha=0.85,
                        )
            ax.set_xticks(las)
            style_ax(ax, "", "Lookahead N", ylab)
            ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        save_plot(fig, out_dir, "sec5_methods_tps_ppl_vs_lookahead.png")

    # (B) TPS–PPL tradeoff: one panel per distinct cache size in canonical set
    sub = canon[canon["tokens_per_second"].notna() & canon["gen_perplexity"].notna()]
    tradeoff_cache_sizes = sorted(sub["cache_size"].dropna().unique())
    n = len(tradeoff_cache_sizes)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols)) if n else 1
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
        fig.suptitle("TPS vs gen-PPL at canonical configs (each point labeled by N)", fontsize=12, fontweight="bold")
        for i, c in enumerate(tradeoff_cache_sizes):
            ax = axes.flat[i]
            panel = sub[sub["cache_size"] == c]
            for pol in _sec5_policy_order(baseline):
                m = panel["policy"] == pol
                if not m.any():
                    continue
                st = POLICY_STYLES[pol]
                ax.scatter(
                    panel.loc[m, "gen_perplexity"],
                    panel.loc[m, "tokens_per_second"],
                    c=st.get("color", "#fff"),
                    marker=st.get("marker", "o"),
                    s=80,
                    label=pol,
                    edgecolors="#333333",
                    linewidths=0.5,
                )
                for _, r in panel.loc[m].iterrows():
                    ax.annotate(
                        f"N={int(r['lookahead'])}",
                        (float(r["gen_perplexity"]), float(r["tokens_per_second"])),
                        fontsize=6, alpha=0.85,
                    )
            style_ax(ax, f"C={int(c)}", "Gen perplexity (↓)", "TPS (↑)")
            ax.legend(fontsize=7)
        for j in range(i + 1, nrows * ncols):
            axes.flat[j].set_visible(False)
        plt.tight_layout()
        save_plot(fig, out_dir, "sec5_tps_ppl_tradeoff.png")


def _discover_sweep_csvs(results_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _dirs, files in os.walk(results_dir):
        if "sweep.csv" not in files:
            continue
        if "_broken" in root or root.endswith("_partial"):
            continue
        paths.append(os.path.join(root, "sweep.csv"))
    return sorted(paths)


def _sweep_row_key(row: pd.Series) -> Tuple[Any, ...]:
    """Identity for one sweep config (matches sweep.csv rows)."""
    b = row.get("prefetch_budget")
    bookend = row.get("bookend_pass")
    if pd.isna(bookend) or bookend == "":
        bookend = "main"
    return (
        row.get("cache_size"),
        str(row.get("label", "")),
        row.get("backend"),
        float(row.get("lambda_val", 0.0)),
        int(row.get("forced_top_n", 0) or 0),
        row.get("lookahead"),
        None if pd.isna(b) else float(b),
        bookend,
    )


def _analysis_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop end-of-cache verification bookends from primary plots."""
    if "bookend_pass" not in df.columns:
        return df
    bp = df["bookend_pass"].fillna("main").astype(str)
    return df.loc[~bp.eq("end")].copy()


def _dedupe_prefer_newest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """When the same config was re-run, keep the row with the latest timestamp."""
    if df.empty or "row_timestamp" not in df.columns:
        return df
    out = df.copy()
    out["_ts"] = pd.to_datetime(out["row_timestamp"], errors="coerce")
    out["_key"] = out.apply(_sweep_row_key, axis=1)
    idx = out.groupby("_key", sort=False)["_ts"].idxmax()
    return out.loc[idx].drop(columns=["_ts", "_key"], errors="ignore")


def _best_row_per_policy(sub: pd.DataFrame, policy: str) -> Optional[pd.Series]:
    """Return the row with max TPS for ``policy`` within ``sub``."""
    pols = _add_policy_columns(sub)
    m = pols["policy"] == policy
    if policy not in ("LRU", "Random"):
        m &= pols["tokens_per_second"].notna()
    rows = pols.loc[m]
    if rows.empty:
        return None
    return rows.loc[rows["tokens_per_second"].idxmax()]


def _baseline_row_for_cache(
    df: pd.DataFrame,
    cache_size: int,
    *,
    baseline: str = "lru",
) -> Optional[pd.Series]:
    """Single eviction baseline per cache size (LRU start bookend or RANDOM main row)."""
    baseline = baseline.lower()
    label = BASELINE_LABELS.get(baseline, BASELINE_LABELS["lru"])
    sub = df[df["cache_size"] == cache_size]
    rows = sub[sub["label"].astype(str) == label]
    if rows.empty:
        return None
    if baseline == "lru" and "bookend_pass" in rows.columns:
        start = rows[rows["bookend_pass"].fillna("main").astype(str) == "start"]
        if not start.empty:
            rows = start
    if "row_timestamp" in rows.columns and rows["row_timestamp"].notna().any():
        return rows.sort_values("row_timestamp", ascending=False).iloc[0]
    return rows.iloc[0]


def _lru_row_for_cache(df: pd.DataFrame, cache_size: int) -> Optional[pd.Series]:
    return _baseline_row_for_cache(df, cache_size, baseline="lru")


def _best_lookahead_for_cache(sub_c: pd.DataFrame) -> Optional[int]:
    """Lookahead N with highest best Hybrid (Both) TPS at this cache size."""
    lookaheads = sorted(sub_c["lookahead"].dropna().unique(), key=int)
    if not lookaheads:
        return None
    best_la = lookaheads[0]
    best_score = float("-inf")
    for la in lookaheads:
        sub_la = sub_c[sub_c["lookahead"] == la]
        both = _best_row_per_policy(sub_la, "Hybrid")
        score = float(both["tokens_per_second"]) if both is not None else float("-inf")
        if score > best_score:
            best_score = score
            best_la = int(la)
    return best_la


def _speedup(tps: float, lru_tps: float) -> float:
    if not lru_tps or lru_tps <= 0 or not np.isfinite(tps):
        return np.nan
    return float(tps) / float(lru_tps)


def _lfru_unified_table(df: pd.DataFrame, *, baseline: str = "lru") -> pd.DataFrame:
    """One row per cache size: best lookahead (by Hybrid TPS), best config per policy."""
    baseline = baseline.lower()
    df = _dedupe_prefer_newest_rows(_add_policy_columns(_filter_custom(_analysis_rows(df))))
    if df.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for c in sorted(df["cache_size"].dropna().unique(), key=int):
        sub_c = df[df["cache_size"] == c]
        base_row = _baseline_row_for_cache(sub_c, int(c), baseline=baseline)
        base_tps = float(base_row["tokens_per_second"]) if base_row is not None else np.nan
        best_la = _best_lookahead_for_cache(sub_c)
        if best_la is None:
            continue

        sub = sub_c[sub_c["lookahead"] == best_la]
        entry: dict = {
            "cache_size": int(c),
            "best_lookahead": int(best_la),
            "baseline": baseline,
            "baseline_tps": base_tps,
            "baseline_speedup": 1.0 if base_tps and base_tps > 0 else np.nan,
            # Legacy column names (LRU) kept for downstream readers when baseline=lru.
            "lru_tps": base_tps,
            "lru_speedup": 1.0 if base_tps and base_tps > 0 else np.nan,
        }
        policy_scope = {
            "Cache-Cond": sub_c,
            "Prefetch": sub,
            "Hybrid": sub,
        }
        for pol, tps_col, sp_col, budget_col in (
            ("Cache-Cond", "cache_cond_tps", "cache_cond_speedup", None),
            ("Prefetch", "prefetch_tps", "prefetch_speedup", "prefetch_budget"),
            ("Hybrid", "both_tps", "both_speedup", "both_budget"),
        ):
            row = _best_row_per_policy(policy_scope[pol], pol)
            tps = float(row["tokens_per_second"]) if row is not None else np.nan
            entry[tps_col] = tps
            entry[sp_col] = _speedup(tps, base_tps)
            if budget_col and row is not None:
                b = row.get("prefetch_budget")
                entry[budget_col] = int(b) if pd.notna(b) else np.nan
        rows.append(entry)
    return pd.DataFrame(rows)


def _lfru_speedup_annot(speedup: float, *, budget: Optional[float] = None) -> str:
    """Annotation text: normalized speedup (and prefetch budget if any)."""
    if not np.isfinite(speedup):
        return ""
    lines = [f"×{speedup:.2f}"]
    if budget is not None and pd.notna(budget):
        lines.append(f"B={int(budget)}")
    return "\n".join(lines)


def plot_lfru_final_unified(
    df: pd.DataFrame,
    out_dir: str,
    *,
    title_suffix: str = "",
    baseline: str = "lru",
) -> None:
    """
    One point per cache size C: pick best lookahead N (max Hybrid TPS), then plot
    eviction baseline + best Cache-Cond / Prefetch / Both at that N (best B per policy).
    Y-axis is raw TPS; × speedup vs baseline annotated on non-baseline points.
    """
    baseline = baseline.lower()
    tab = _lfru_unified_table(df, baseline=baseline)
    if tab.empty:
        print("[thesis_plot] lfru_final: no CUSTOM rows to plot", flush=True)
        return
    if tab["baseline_tps"].isna().all():
        print(
            f"[thesis_plot] lfru_final: no {BASELINE_PLOT_NAMES.get(baseline, baseline)} "
            f"baseline rows found (label={BASELINE_LABELS.get(baseline)!r}).",
            flush=True,
        )
        return

    os.makedirs(out_dir, exist_ok=True)
    suffix_tag = "" if baseline == "lru" else f"_vs_{baseline}"
    csv_path = os.path.join(out_dir, f"sec5_lfru_unified_summary{suffix_tag}.csv")
    tab.to_csv(csv_path, index=False)
    print(f"[thesis_plot] Saved {csv_path}", flush=True)

    x_labels = [f"C={int(r.cache_size)}\nN={int(r.best_lookahead)}" for r in tab.itertuples()]
    x = np.arange(len(tab))
    suffix = f" — {title_suffix}" if title_suffix else ""
    base_name = BASELINE_PLOT_NAMES.get(baseline, baseline.title())
    base_style_key = "LRU" if baseline == "lru" else "Random"

    series = [
        (base_name, "baseline_tps", None, None, POLICY_STYLES[base_style_key]),
        ("Cache-Cond", "cache_cond_tps", "cache_cond_speedup", None, POLICY_STYLES["Cache-Cond"]),
        ("Prefetch", "prefetch_tps", "prefetch_speedup", "prefetch_budget", POLICY_STYLES["Prefetch"]),
        ("Both", "both_tps", "both_speedup", "both_budget", POLICY_STYLES["Hybrid"]),
    ]

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(tab)), 5.5))
        fig.suptitle(
            f"LFRU: four-policy TPS vs {base_name} (best N per C, best B per policy){suffix}",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        for pol_name, tps_col, sp_col, budget_col, st in series:
            ys = tab[tps_col].to_numpy(dtype=float)
            ax.plot(
                x, ys,
                label=pol_name,
                marker=st.get("marker", "o"),
                color=st.get("color", "#333"),
                linestyle=st.get("linestyle", "-"),
                linewidth=st.get("linewidth", 2),
            )
            if pol_name == base_name or sp_col is None:
                continue
            for i, yi in enumerate(ys):
                if not np.isfinite(yi):
                    continue
                sp_val = float(tab.iloc[i][sp_col])
                budget = tab.iloc[i].get(budget_col) if budget_col else None
                txt = _lfru_speedup_annot(sp_val, budget=budget)
                if txt:
                    ax.annotate(
                        txt,
                        xy=(x[i], yi),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        fontsize=6,
                        color=st.get("color", "#333"),
                        alpha=0.95,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        style_ax(ax, "", "Expert cache size C  (best lookahead N)", "Tokens / sec")
        ax.legend(fontsize=9, ncol=2, loc="upper left")
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        save_plot(fig, out_dir, f"sec5_lfru_unified_tps{suffix_tag}.png")

        width = 0.18
        fig2, ax2 = plt.subplots(figsize=(max(8, 2.0 * len(tab)), 5.5))
        fig2.suptitle(
            f"LFRU: four-policy TPS vs {base_name} (grouped){suffix}",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        for i, (pol_name, tps_col, sp_col, budget_col, st) in enumerate(series):
            offset = (i - 1.5) * width
            ys = tab[tps_col].to_numpy(dtype=float)
            bars = ax2.bar(
                x + offset,
                ys,
                width=width,
                label=pol_name,
                color=st.get("color", "#333"),
                edgecolor="#333333",
                linewidth=0.4,
                alpha=0.92,
            )
            if pol_name != base_name and sp_col:
                for bar, row_idx in zip(bars, range(len(tab))):
                    h = ys[row_idx]
                    if not np.isfinite(h):
                        continue
                    sp_val = float(tab.iloc[row_idx][sp_col])
                    budget = tab.iloc[row_idx].get(budget_col) if budget_col else None
                    txt = _lfru_speedup_annot(sp_val, budget=budget)
                    if txt:
                        ax2.annotate(
                            txt,
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            fontsize=5,
                            color=st.get("color", "#333"),
                        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels)
        style_ax(ax2, "", "Expert cache size C  (best lookahead N)", "Tokens / sec")
        ax2.legend(fontsize=9, ncol=2)
        ax2.set_ylim(bottom=0)
        plt.tight_layout()
        save_plot(fig2, out_dir, f"sec5_lfru_unified_tps_bars{suffix_tag}.png")


def _ppl_winners_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per C: LRU + best Cache-Cond + Hybrid rows (matches PPL phase selection)."""
    df = _dedupe_prefer_newest_rows(_add_policy_columns(_filter_custom(_analysis_rows(df))))
    if df.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for c in sorted(df["cache_size"].dropna().unique(), key=int):
        sub_c = df[df["cache_size"] == c]
        lru = _baseline_row_for_cache(sub_c, int(c), baseline="lru")
        best_la = _best_lookahead_for_cache(sub_c)
        if best_la is None:
            continue
        sub_la = sub_c[sub_c["lookahead"] == best_la]
        entry: dict = {"cache_size": int(c), "best_lookahead": int(best_la)}
        for pol, col in (("LRU", "lru"), ("Cache-Cond", "cache_cond"), ("Hybrid", "hybrid")):
            scope = sub_c if pol in ("LRU", "Cache-Cond") else sub_la
            if pol == "LRU":
                row = lru
            else:
                row = _best_row_per_policy(scope, pol)
            ppl = float(row["gen_perplexity"]) if row is not None and pd.notna(row.get("gen_perplexity")) else np.nan
            tps = float(row["tokens_per_second"]) if row is not None and pd.notna(row.get("tokens_per_second")) else np.nan
            entry[f"{col}_ppl"] = ppl
            entry[f"{col}_tps"] = tps
        lru_ppl = entry.get("lru_ppl")
        for col in ("cache_cond", "hybrid"):
            ppl = entry.get(f"{col}_ppl")
            if lru_ppl and lru_ppl > 0 and np.isfinite(ppl):
                entry[f"{col}_ppl_delta_pct"] = 100.0 * (ppl - lru_ppl) / lru_ppl
            else:
                entry[f"{col}_ppl_delta_pct"] = np.nan
        rows.append(entry)
    return pd.DataFrame(rows)


def plot_ppl_winners_vs_lru(df: pd.DataFrame, out_dir: str) -> None:
    """Grouped WikiText-103 PPL for LRU vs best Cache-Cond vs Hybrid (winner configs per C)."""
    tab = _ppl_winners_table(df)
    if tab.empty:
        print("[thesis_plot] ppl_winners: no CUSTOM rows", flush=True)
        return

    has_any = tab[["lru_ppl", "cache_cond_ppl", "hybrid_ppl"]].notna().any(axis=None)
    if not has_any:
        print("[thesis_plot] ppl_winners: no gen_perplexity values yet", flush=True)
        return

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ppl_winners_vs_lru_summary.csv")
    tab.to_csv(csv_path, index=False)
    print(f"[thesis_plot] Saved {csv_path}", flush=True)

    x = np.arange(len(tab))
    x_labels = [f"C={int(r.cache_size)}" for r in tab.itertuples()]
    width = 0.25
    series = [
        ("LRU", "lru_ppl", POLICY_STYLES["LRU"]),
        ("Cache-Cond", "cache_cond_ppl", POLICY_STYLES["Cache-Cond"]),
        ("Hybrid", "hybrid_ppl", POLICY_STYLES["Hybrid"]),
    ]

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(tab)), 5.5))
        fig.suptitle(
            "WikiText-103 PPL vs LRU (Cache-Cond: cached · Hybrid: predict)",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        for i, (name, col, st) in enumerate(series):
            offset = (i - 1) * width
            ys = tab[col].to_numpy(dtype=float)
            bars = ax.bar(
                x + offset,
                ys,
                width=width,
                label=name,
                color=st.get("color", "#333"),
                edgecolor="#333333",
                linewidth=0.4,
                alpha=0.92,
            )
            for bar, yi, row_idx in zip(bars, ys, range(len(tab))):
                if not np.isfinite(yi):
                    continue
                txt = f"{yi:.2f}"
                if name != "LRU":
                    delta = tab.iloc[row_idx].get(
                        "cache_cond_ppl_delta_pct" if name == "Cache-Cond" else "hybrid_ppl_delta_pct"
                    )
                    if pd.notna(delta):
                        sign = "+" if delta >= 0 else ""
                        txt = f"{yi:.2f}\n({sign}{delta:.1f}%)"
                ax.annotate(
                    txt,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color=st.get("color", "#333"),
                )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        style_ax(ax, "", "Expert cache size C", "WikiText-103 PPL (↓ better)")
        ax.legend(fontsize=9, ncol=3, loc="upper right")
        plt.tight_layout()
        save_plot(fig, out_dir, "ppl_winners_vs_lru.png")


def plot_sec5_fixed_cache_if_present(
    df_all: pd.DataFrame,
    df_fixed: Optional[pd.DataFrame],
    cache_size: int,
    out_dir: str,
    *,
    budget_frac: float = 0.5,
) -> None:
    """If a fixed-cache all-methods CSV exists, emit true fixed-C policy grid via plot_custom_1_16."""
    if df_fixed is None:
        return
    sub = _add_policy_columns(_filter_custom(_at_cache(df_fixed, cache_size)))
  # Need all four policies — fixed_cache32 is usually no_ppl prefetch only
    has_cc = sub["label"].astype(str).str.contains("Cache-Cond", na=False).any()
    has_both = sub["label"].astype(str).str.contains("Both", na=False).any()
    if not (has_cc and has_both):
        print(
            "[thesis_plot] sec5: --csv-fixed-cache has no cache-cond/hybrid rows; "
            "use exp-3 sec5_methods_* plots for the four-way comparison.",
            flush=True,
        )
        return
    sub_ppl = sub.copy()
    sub_ppl["gen_perplexity"] = sub_ppl.get("gen_perplexity", np.nan)
    if sub_ppl["gen_perplexity"].notna().any():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_custom_1_16(sub_ppl, out_dir, ts)
        src = os.path.join(out_dir, f"custom_1_16_summary_{ts}.png")
        dst = os.path.join(out_dir, "sec5_methods_fixed_cache_summary.png")
        if os.path.isfile(src):
            os.replace(src, dst)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--csv-sec2",
        default="",
        help="sec2_predictor_effectiveness sweep.csv (drives sections 2 and 3).",
    )
    p.add_argument("--csv-sec4", default="", help="sec4_routing_topj_vs_pm sweep.csv (section 4).")
    p.add_argument("--csv-sec5", default="", help="sec5_all_methods sweep.csv (section 5).")
    p.add_argument(
        "--csv-lfru-dir",
        default="",
        help="Directory tree with sec5_lfru_final sweep.csv files (unified four-policy plot).",
    )
    p.add_argument("--cache-size", type=int, default=32, help="Fixed cache size C the sections 2/3 plots key on.")
    p.add_argument("--out-dir", default="py/utils/final_results_runs/thesis_plots")
    p.add_argument("--budget-frac", type=float, default=0.5, help="Section 5: match policies at this B/C fraction.")
    p.add_argument(
        "--baseline",
        choices=["lru", "random"],
        default="random",
        help="Eviction baseline for section 5 plots and LFRU speedup annotations (default: RANDOM).",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    c = args.cache_size

    if args.csv_sec2 and os.path.isfile(args.csv_sec2):
        df_fix = _load_csv(args.csv_sec2)
        plot_sec2_predictor_effectiveness(df_fix, c, args.out_dir)
        plot_sec3_predictor_speedup(df_fix, c, args.out_dir)
        plot_best_lookahead_per_cache(df_fix, args.out_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_custom_1_16(_filter_custom(_at_cache(df_fix, c)), args.out_dir, ts)
        _rename_plot(args.out_dir, f"custom_1_16_summary_{ts}.png", "sec2_prefetch_lru_summary.png")
    else:
        print("[thesis_plot] Skipping sections 2-3 (no --csv-sec2)", flush=True)

    if args.csv_sec4 and os.path.isfile(args.csv_sec4):
        plot_sec4_routing_fn_pm(_load_csv(args.csv_sec4), args.out_dir)

    if args.csv_sec5 and os.path.isfile(args.csv_sec5):
        df5 = _load_csv(args.csv_sec5)
        plot_sec5_methods(
            df5,
            args.out_dir,
            budget_frac=args.budget_frac,
            baseline=args.baseline,
            cache_size=c,
        )
        df_fix = _load_csv(args.csv_sec2) if args.csv_sec2 and os.path.isfile(args.csv_sec2) else None
        plot_sec5_fixed_cache_if_present(df5, df_fix, c, args.out_dir, budget_frac=args.budget_frac)

    if args.csv_lfru_dir and os.path.isdir(args.csv_lfru_dir):
        csvs = _discover_sweep_csvs(args.csv_lfru_dir)
        if not csvs:
            print(f"[thesis_plot] No sweep.csv under {args.csv_lfru_dir}", flush=True)
        else:
            frames = [_load_csv_raw(p) for p in csvs]
            df_lfru = pd.concat(frames, ignore_index=True)
            plot_lfru_final_unified(
                df_lfru,
                args.out_dir,
                title_suffix="LFRU cache policy",
                baseline=args.baseline,
            )
            plot_ppl_winners_vs_lru(df_lfru, args.out_dir)

    print(f"[thesis_plot] Done → {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
