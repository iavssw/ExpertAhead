#!/usr/bin/env python3
"""
plot_emb_markov_paper.py
========================
Paper-quality figures for a single expert-predictor variant sweep.

Works for any variant (emb_only, emb_markov, etc.) — pass --variant to
control which files are loaded from the sweep directory.

Usage
-----
python plot_emb_markov_paper.py \\
    --sweep_dir /data/qwen3_30b/emb_only_sweep \\
    --variant emb_only \\
    [--out_dir plots/emb_only] \\
    [--budgets 8 16 32 64] \\
    [--eval_topks 8 16 32] \\
    [--main_eval_topk 8] \\
    [--thresholds 0.3 0.5 0.7]

Produces (all as .png + .pdf unless noted)
------------------------------------------
recall_prec_B{K}                  PRIMARY: Recall@B + Precision@B dual-subplot vs horizon
recall_vs_horizon                 Fixed top-K recall for multiple K values
precision_vs_horizon              Fixed top-K precision for multiple K values
f1_vs_horizon                     Fixed top-K F1 for multiple K values
best_k_prf                        Best-K Precision / Recall / F1 vs horizon
best_topk_vs_best_threshold_f1   Oracle best-K vs best-τ F1
prf_vs_horizon_adaptive           Adaptive P/R/F1 vs horizon
f1_vs_horizon_adaptive            Adaptive F1 (same as above, F1-only)
thresh_recall_vs_horizon          Threshold recall vs horizon
thresh_precision_vs_horizon       Threshold precision vs horizon
thresh_f1_vs_horizon              Threshold F1 vs horizon
heatmap_adaptive_recall           Recall heatmap: horizon × layer
heatmap_adaptive_f1               F1 heatmap: horizon × layer
heatmap_recall_B{K}               Recall@B heatmap: horizon × layer
heatmap_precision_B{K}            Precision@B heatmap: horizon × layer
<variant>_metrics_per_layer.csv
<variant>_metrics_mean_over_layers.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import (
    apply_style,
    aggregate,
    best_param_aggregate,
    get_best_k_prf,
    get_num_experts_from_records,
    discover_metric_keys,
    load_sweep_full,
    save_fig,
    style_for,
)

apply_style()

DEFAULT_BUDGETS = [8, 16, 32, 64]


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def write_csvs(
    records: dict,
    out_dir: Path,
    variant: str,
    eval_ks: list[int],
    thresholds: list[float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(records)
    layers   = sorted({l for fs in records for l in records[fs]})

    # ── per-layer long CSV ────────────────────────────────────────────────────
    rows: list[dict] = []
    for fs in horizons:
        for layer in layers:
            if layer not in records[fs]:
                continue
            m   = records[fs][layer]["metrics"]
            row = {
                "horizon_N":           fs,
                "layer":               layer,
                "label_density":       m.get("label_density"),
                "loss":                m.get("loss"),
                "adaptive_recall":     m.get("adaptive_recall"),
                "adaptive_precision":  m.get("adaptive_precision"),
                "adaptive_f1":         m.get("adaptive_f1"),
                "adaptive_avg_union":  m.get("adaptive_avg_union"),
            }
            for k in eval_ks:
                row[f"recall@{k}"]    = m.get(f"topk_recall@{k}",    m.get(f"union_recall@{k}"))
                row[f"precision@{k}"] = m.get(f"topk_precision@{k}", m.get(f"union_precision@{k}"))
                row[f"f1@{k}"]        = m.get(f"topk_f1@{k}",        m.get(f"union_f1@{k}"))
            for t in thresholds:
                tag = f"{t:g}"
                row[f"recall_thr_{tag}"]    = m.get(f"thresh_recall@{tag}")
                row[f"precision_thr_{tag}"] = m.get(f"thresh_precision@{tag}")
                row[f"f1_thr_{tag}"]        = m.get(f"thresh_f1@{tag}")
                row[f"avg_pred_thr_{tag}"]  = m.get(f"thresh_avg_pred@{tag}")
            rows.append(row)

    if rows:
        p = out_dir / f"{variant}_metrics_per_layer.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {p.name}")

    # ── mean-over-layers CSV ──────────────────────────────────────────────────
    rows_mean: list[dict] = []
    for fs in horizons:
        row = {"horizon_N": fs, "n_layers": len(records[fs])}
        for key in ("adaptive_recall", "adaptive_precision", "adaptive_f1", "adaptive_avg_union"):
            mn, sd, _ = aggregate(records[fs], key)
            row[f"{key}_mean"] = mn
            row[f"{key}_std"]  = sd
        for k in eval_ks:
            for pfx in ("topk_recall", "topk_precision", "topk_f1"):
                mn, sd, _ = aggregate(records[fs], f"{pfx}@{k}")
                row[f"{pfx}@{k}_mean"] = mn
                row[f"{pfx}@{k}_std"]  = sd
        row["label_density_mean"] = aggregate(records[fs], "label_density")[0]
        rows_mean.append(row)

    if rows_mean:
        p = out_dir / f"{variant}_metrics_mean_over_layers.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows_mean[0].keys())
            w.writeheader()
            w.writerows(rows_mean)
        print(f"  Wrote {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive P / R / F1
# ─────────────────────────────────────────────────────────────────────────────

def plot_adaptive_prf_vs_horizon(records: dict, out_dir: Path, variant: str) -> None:
    """
    Adaptive score at K_i = |true union_i| per prompt.
    When the predictor outputs exactly |union| experts per sample,
    per-sample precision == recall, so F1 = recall = precision.
    We plot all three and warn if they diverge.
    """
    horizons = sorted(records)
    s        = style_for(variant)

    keys = ("adaptive_recall", "adaptive_precision", "adaptive_f1")
    f1_means, f1_stds = [], []
    for fs in horizons:
        mn, sd, _ = aggregate(records[fs], "adaptive_f1")
        f1_means.append(mn)
        f1_stds.append(sd)

    # Sanity check — if they differ, log it
    for fs in horizons:
        vals = [aggregate(records[fs], k)[0] for k in keys]
        if max(vals) - min(vals) > 1e-6:
            print(f"  Note: adaptive P/R/F1 differ at N={fs}: {vals}")

    # ── Combined P/R/F1 figure ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        horizons, f1_means, yerr=f1_stds,
        marker=s["marker"], ls=s["ls"], color=s["color"],
        linewidth=2.5, capsize=3, markersize=7,
        label="F1 (= P = R at K = |union|)",
    )
    ax.set_xlabel("Lookahead $N$")
    ax.set_ylabel("F1 / Recall / Precision")
    ax.set_title(f"{s['label']} — score at K_i = |true union_i| per prompt")
    ax.set_xticks(horizons)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / "prf_vs_horizon_adaptive")

    # ── F1-only figure ────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.errorbar(
        horizons, f1_means, yerr=f1_stds,
        marker=s["marker"], ls=s["ls"], color=s["color"],
        linewidth=2.5, capsize=3, markersize=7,
        label="Adaptive F1",
    )
    ax2.set_xlabel("Lookahead $N$")
    ax2.set_ylabel("F1")
    ax2.set_title(f"{s['label']} — Adaptive F1 at K = |true union| per prompt")
    ax2.set_xticks(horizons)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(linestyle="--", alpha=0.35)
    fig2.tight_layout()
    save_fig(fig2, out_dir / "f1_vs_horizon_adaptive")


# ─────────────────────────────────────────────────────────────────────────────
# Fixed top-K metric curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_topk_metric_vs_horizon(
    records: dict, out_dir: Path, eval_ks: list[int],
    metric_suffix: str, ylabel: str, fname: str, variant: str,
) -> None:
    """Fixed top-K recall / precision / F1 curves (mean ± std over layers)."""
    horizons = sorted(records)
    colors   = plt.cm.viridis(np.linspace(0.2, 0.85, max(len(eval_ks), 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for ki, k in enumerate(eval_ks):
        means, stds = [], []
        for fs in horizons:
            mn, sd, _ = aggregate(records[fs], f"topk_{metric_suffix}@{k}")
            means.append(mn)
            stds.append(sd)
        ax.errorbar(horizons, means, yerr=stds,
                    marker="o", capsize=3, color=colors[ki],
                    linewidth=2, markersize=6, label=f"K={k}")

    s = style_for(variant)
    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{s['label']} — {ylabel} vs Horizon")
    ax.set_xticks(horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Top-K")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / fname)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold-based curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_vs_horizon(
    records: dict, out_dir: Path, thresholds: list[float], metric: str, variant: str
) -> None:
    """Threshold-based recall / precision / F1 curves, mean over layers."""
    if not thresholds:
        return
    horizons = sorted(records)
    colors   = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(thresholds), 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for ti, t in enumerate(thresholds):
        means = [aggregate(records[fs], f"thresh_{metric}@{t:g}")[0] for fs in horizons]
        ax.plot(horizons, means, "o-", linewidth=1.8, markersize=6,
                color=colors[ti], label=f"τ={t:g}")

    s = style_for(variant)
    ax.set_xlabel("Lookahead $N$")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{s['label']} — Threshold {metric.capitalize()} vs horizon (sigmoid ≥ τ)")
    ax.set_xticks(horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Threshold", fontsize=8)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / f"thresh_{metric}_vs_horizon")


# ─────────────────────────────────────────────────────────────────────────────
# Recall@B + Precision@B dual-subplot figure (PRIMARY plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_recall_precision_at_budgets(
    records: dict, out_dir: Path, budgets: list[int], variant: str
) -> None:
    """
    For each budget B in *budgets*: one figure with two vertically-stacked
    subplots — recall@B (top) and precision@B (bottom) vs lookahead horizon N.
    Values are mean ± std over layers.
    """
    horizons = sorted(records)
    if not horizons:
        return
    s = style_for(variant)

    for B in budgets:
        rec_means, rec_stds   = [], []
        prec_means, prec_stds = [], []
        valid_horizons = []

        for fs in horizons:
            r_mn, r_sd, r_n = aggregate(records[fs], f"topk_recall@{B}")
            p_mn, p_sd, p_n = aggregate(records[fs], f"topk_precision@{B}")
            if r_n == 0 and p_n == 0:
                continue
            valid_horizons.append(fs)
            rec_means.append(r_mn);  rec_stds.append(r_sd)
            prec_means.append(p_mn); prec_stds.append(p_sd)

        if not valid_horizons:
            print(f"  No topk_recall/precision data for B={B}, skipping.")
            continue

        xs = np.array(valid_horizons)
        fig = plt.figure(figsize=(7, 6))
        gs = gridspec.GridSpec(2, 1, hspace=0.45)
        ax_rec  = fig.add_subplot(gs[0])
        ax_prec = fig.add_subplot(gs[1])

        for ax, means, stds, ylabel, title in [
            (ax_rec,  rec_means,  rec_stds,
             f"Recall @ B={B}",    f"{s['label']} — Recall@{B} vs Horizon"),
            (ax_prec, prec_means, prec_stds,
             f"Precision @ B={B}", f"{s['label']} — Precision@{B} vs Horizon"),
        ]:
            ys = np.array(means)
            sd = np.array(stds)
            ax.plot(xs, ys, marker=s["marker"], ls=s["ls"], color=s["color"],
                    lw=s["lw"], markersize=7, label=f"B={B}")
            ax.fill_between(
                xs,
                np.clip(ys - sd, 0, 1),
                np.clip(ys + sd, 0, 1),
                color=s["color"], alpha=0.12,
            )
            ax.set_xlabel("Lookahead $N$ (tokens)")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks(valid_horizons)
            ax.set_xticklabels([str(h) for h in valid_horizons])
            ax.set_ylim(0, 1.05)
            ax.grid(linestyle="--", alpha=0.35)

        fig.suptitle(
            f"{s['label']} — Prefetch Budget B={B}",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        save_fig(fig, out_dir / f"recall_prec_B{B}")


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap (horizon × layer)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    records: dict, out_dir: Path, metric_key: str, title_suffix: str
) -> None:
    horizons = sorted(records)
    layers   = sorted({l for fs in records for l in records[fs]})
    if not horizons or not layers:
        return

    mat = np.full((len(horizons), len(layers)), np.nan)
    for i, fs in enumerate(horizons):
        for j, layer in enumerate(layers):
            if layer in records[fs]:
                val = records[fs][layer]["metrics"].get(metric_key)
                if val is not None:
                    mat[i, j] = float(val)

    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.35), 3 + 0.15 * len(horizons)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(horizons)))
    ax.set_yticklabels([f"N={fs}" for fs in horizons])
    step = max(1, len(layers) // 16)
    ax.set_xticks(range(0, len(layers), step))
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), step)], fontsize=8)
    ax.set_xlabel("Decoder layer")
    ax.set_title(f"{title_suffix} (best val)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    slug = metric_key.replace("adaptive_", "").replace("_", "")
    save_fig(fig, out_dir / f"heatmap_{slug}")


def plot_topk_heatmaps(
    records: dict, out_dir: Path, budgets: list[int], variant: str
) -> None:
    """Heatmap of recall@B and precision@B (horizon × layer) for each B."""
    horizons = sorted(records)
    layers   = sorted({l for fs in records for l in records[fs]})
    if not horizons or not layers:
        return
    s = style_for(variant)

    for B in budgets:
        for metric_key, label in [
            (f"topk_recall@{B}",    f"{s['label']} — Recall@{B}"),
            (f"topk_precision@{B}", f"{s['label']} — Precision@{B}"),
        ]:
            mat = np.full((len(horizons), len(layers)), np.nan)
            for i, fs in enumerate(horizons):
                for j, layer in enumerate(layers):
                    if layer in records[fs]:
                        val = records[fs][layer]["metrics"].get(metric_key)
                        if val is not None:
                            mat[i, j] = float(val)

            if np.all(np.isnan(mat)):
                continue

            fig, ax = plt.subplots(
                figsize=(max(12, len(layers) * 0.35), 3 + 0.15 * len(horizons))
            )
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_yticks(range(len(horizons)))
            ax.set_yticklabels([f"N={fs}" for fs in horizons])
            step = max(1, len(layers) // 16)
            ax.set_xticks(range(0, len(layers), step))
            ax.set_xticklabels(
                [layers[i] for i in range(0, len(layers), step)], fontsize=8
            )
            ax.set_xlabel("Decoder layer")
            ax.set_ylabel("Lookahead horizon N")
            ax.set_title(f"{label} — Horizon × Layer Heatmap")
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            fig.tight_layout()
            slug = "recall" if "recall" in metric_key else "precision"
            save_fig(fig, out_dir / f"heatmap_{slug}_B{B}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(records: dict, variant: str) -> None:
    horizons = sorted(records)
    print(f"\n{'='*80}")
    print(f"  {variant} — adaptive P/R/F1 at K_i = |union_i|")
    print(f"{'='*80}")
    print(f"  {'N':>4}  {'Recall':>8}  {'Prec':>8}  {'F1':>8}  {'|union|':>8}  {'layers':>6}")
    print(f"  {'-'*52}")
    for fs in horizons:
        r   = aggregate(records[fs], "adaptive_recall")[0]
        p   = aggregate(records[fs], "adaptive_precision")[0]
        f1  = aggregate(records[fs], "adaptive_f1")[0]
        u   = aggregate(records[fs], "adaptive_avg_union")[0]
        n   = len(records[fs])
        fmt = lambda v: f"{v:>8.4f}" if v == v else f"{'N/A':>8}"  # noqa: E731
        print(f"  {fs:>4}  {fmt(r)}  {fmt(p)}  {fmt(f1)}  {fmt(u)}  {n:>6}")
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Best top-K vs best threshold F1
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_topk_vs_best_threshold_f1(records: dict, out_dir: Path, variant: str) -> None:
    """
    Two curves vs lookahead horizon for the chosen variant:
      • solid  — oracle best-K F1:  per horizon/layer, max over all K of topk_f1@K
      • dashed — oracle best-τ F1:  per horizon/layer, max over all τ of thresh_f1@T

    Shows whether threshold selection has higher ceiling than top-K (or vice versa)
    without committing to any specific K or τ value.
    """
    horizons = sorted(records)
    s        = style_for(variant)

    has_topk   = any(
        any(k.startswith("topk_f1@")   for k in ent["metrics"])
        for fs_d in records.values() for ent in fs_d.values()
    )
    has_thresh = any(
        any(k.startswith("thresh_f1@") for k in ent["metrics"])
        for fs_d in records.values() for ent in fs_d.values()
    )
    if not has_topk and not has_thresh:
        print("  Skipping best top-K vs threshold plot: no topk_f1 or thresh_f1 keys found.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if has_topk:
        means_k = [best_param_aggregate(records[fs], "topk_f1")[0] for fs in horizons]
        stds_k  = [best_param_aggregate(records[fs], "topk_f1")[1] for fs in horizons]
        ax.errorbar(horizons, means_k, yerr=stds_k,
                    marker=s["marker"], ls="-", color=s["color"],
                    linewidth=s["lw"], capsize=4, markersize=7,
                    label="Best-K F1  (oracle top-K)")

    if has_thresh:
        means_t = [best_param_aggregate(records[fs], "thresh_f1")[0] for fs in horizons]
        stds_t  = [best_param_aggregate(records[fs], "thresh_f1")[1] for fs in horizons]
        # Use a contrasting but harmonious color for the threshold curve
        thresh_color = "#FF6B6B" if s["color"] != "#FF6B6B" else "#457B9D"
        ax.errorbar(horizons, means_t, yerr=stds_t,
                    marker=s["marker"], ls="--", color=thresh_color,
                    linewidth=s["lw"] * 0.85, capsize=4, markersize=6,
                    label="Best-τ F1  (oracle threshold)")

    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel("F1")
    ax.set_title(f"{s['label']} — Oracle Best-K vs Best-τ F1")
    ax.set_xticks(horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / "best_topk_vs_best_threshold_f1")


def plot_best_k_prf(records: dict, out_dir: Path, variant: str) -> None:
    """
    Plot Precision, Recall, and F1 at the best K value (maximizing F1)
    for each lookahead horizon. The K values are annotated on the plot.
    """
    horizons = sorted(records)
    s        = style_for(variant)

    # Verify we actually have topk metrics
    has_topk = any(
        any(k.startswith("topk_f1@") for k in ent["metrics"])
        for fs_d in records.values() for ent in fs_d.values()
    )
    if not has_topk:
        print("  Skipping best-K PRF plot: no topk_f1 keys found.")
        return

    best_ks = []
    means = {"precision": [], "recall": [], "f1": []}
    stds = {"precision": [], "recall": [], "f1": []}

    for fs in horizons:
        k, prf = get_best_k_prf(records[fs])
        best_ks.append(k)
        for metric in ("precision", "recall", "f1"):
            mean, std = prf.get(metric, (float("nan"), 0.0))
            means[metric].append(mean)
            stds[metric].append(std)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {"precision": "#2eca7f", "recall": "#3a86c8", "f1": "#e63946"}
    markers = {"precision": "s", "recall": "o", "f1": "^"}
    linestyles = {"precision": "--", "recall": "--", "f1": "-"}

    for metric in ("precision", "recall", "f1"):
        m_vals = means[metric]
        s_vals = stds[metric]

        x_vals = [h for h, val in zip(horizons, m_vals) if not np.isnan(val)]
        y_vals = [val for val in m_vals if not np.isnan(val)]
        std_y = [s for val, s in zip(m_vals, s_vals) if not np.isnan(val)]

        if not x_vals:
            continue

        ax.plot(x_vals, y_vals, color=colors[metric], marker=markers[metric],
                ls=linestyles[metric], lw=2.2 if metric == "f1" else 1.8,
                label=metric.capitalize())
        
        y_low = [y - sd for y, sd in zip(y_vals, std_y)]
        y_high = [y + sd for y, sd in zip(y_vals, std_y)]
        ax.fill_between(x_vals, y_low, y_high, color=colors[metric], alpha=0.08)

    # Plot Random F1 baseline
    random_f1s = []
    for fs, k in zip(horizons, best_ks):
        if fs in records and k is not None:
            dens_val, _, _ = aggregate(records[fs], "label_density")
            E = get_num_experts_from_records(records[fs])
            ratio = k / E
            denom = dens_val + ratio
            rand_f1 = (2 * dens_val * ratio) / denom if denom > 1e-6 else 0.0
            random_f1s.append(rand_f1)
        else:
            random_f1s.append(float("nan"))

    x_vals_rand = [h for h, val in zip(horizons, random_f1s) if not np.isnan(val)]
    y_vals_rand = [val for val in random_f1s if not np.isnan(val)]
    if x_vals_rand:
        ax.plot(x_vals_rand, y_vals_rand, color="gray", ls=":", lw=1.5,
                label="Random baseline (F1)")

    # Annotate K values on the F1 line
    f1_vals = means["f1"]
    for idx, (fs, k, f1_val) in enumerate(zip(horizons, best_ks, f1_vals)):
        if k is not None and not np.isnan(f1_val):
            offset_y = 12 if idx % 2 == 0 else 24
            ax.annotate(f"B={k}", (fs, f1_val), textcoords="offset points",
                        xytext=(0, offset_y), ha='center', fontsize=9, color="#333333",
                        fontweight='bold', bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    ax.set_xlabel("Lookahead Horizon (N)")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"{s['label']} — Precision, Recall, & F1 for Best-K per Horizon")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    save_fig(fig, out_dir / "best_k_prf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    ap = argparse.ArgumentParser(
        description="Paper figures for a single expert-predictor variant sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--sweep_dir", required=True,
                    help="Root directory containing training_metrics.json files")
    ap.add_argument("--variant",   default="emb_only",
                    help="Variant tag used to filter files by path substring "
                         "(default: 'emb_only').  Pass '' to load all files.")
    ap.add_argument("--out_dir",   default=None,
                    help="Output directory (default: sweep_dir/paper_plots)")
    ap.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS,
                    help=f"Prefetch budget B values for recall/precision plots "
                         f"(default: {DEFAULT_BUDGETS})")
    ap.add_argument("--eval_topks", nargs="+", type=int, default=None,
                    help="Fixed top-K values for multi-K curves (default: auto-detected)")
    ap.add_argument("--main_eval_topk", type=int, default=8,
                    help="Primary reference K (default: 8)")
    ap.add_argument("--thresholds", nargs="+", type=float, default=None,
                    help="Sigmoid τ values (default: auto-detected from data)")
    args = ap.parse_args()

    sweep   = Path(args.sweep_dir)
    out     = Path(args.out_dir) if args.out_dir else sweep / "paper_plots"
    variant = args.variant
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    variant_tag = variant if variant else None
    records = load_sweep_full(sweep, variant_tag=variant_tag)
    if not records:
        tag_msg = f"(variant tag: '{variant_tag}')" if variant_tag else "(no filter)"
        print(f"No training_metrics.json found under {sweep} {tag_msg}.")
        return

    horizons = sorted(records)
    layers   = sorted({l for fs in records for l in records[fs]})
    print(f"Loaded: {len(horizons)} horizons {horizons}, {len(layers)} layers {layers}")
    print(f"Variant filter: '{variant_tag}' | Output: {out}\n")

    # ── Auto-discover metric keys ─────────────────────────────────────────────
    # Build a flat records dict for discover_metric_keys (it expects records[fs][layer] = metrics)
    flat_records = {fs: {l: ent["metrics"] for l, ent in ld.items()} for fs, ld in records.items()}

    thresholds = args.thresholds or [
        t for t in discover_metric_keys(flat_records, "thresh_recall")
    ]
    eval_ks = args.eval_topks
    if eval_ks is None:
        eval_ks = [int(k) for k in discover_metric_keys(flat_records, "topk_recall")]
    eval_ks = sorted(set(eval_ks) | {args.main_eval_topk})

    print(f"  Eval top-K: {eval_ks}")
    print(f"  Thresholds: {thresholds}\n")

    write_csvs(records, out, variant, eval_ks, thresholds)
    print_summary(records, variant)

    # ── PRIMARY: Recall@B + Precision@B at fixed budgets ─────────────────────
    print("--- Recall / Precision at fixed budgets ---")
    plot_recall_precision_at_budgets(records, out, args.budgets, variant)

    # ── Budget heatmaps (horizon × layer) ────────────────────────────────────
    print("--- Recall / Precision heatmaps (horizon × layer) ---")
    plot_topk_heatmaps(records, out, args.budgets, variant)

    # ── Best top-K Precision/Recall/F1 ────────────────────────────────────────
    print("--- Best-K P/R/F1 ---")
    plot_best_k_prf(records, out, variant)

    # ── Fixed top-K multi-K curves ────────────────────────────────────────────
    print("--- Fixed top-K curves ---")
    for metric_suffix, ylabel, fname in [
        ("recall",    "Recall",    "recall_vs_horizon"),
        ("precision", "Precision", "precision_vs_horizon"),
        ("f1",        "F1",        "f1_vs_horizon"),
    ]:
        plot_topk_metric_vs_horizon(
            records, out, eval_ks, metric_suffix, ylabel, fname, variant
        )

    # ── Adaptive P/R/F1 (secondary) ───────────────────────────────────────────
    print("--- Adaptive F1 ---")
    plot_adaptive_prf_vs_horizon(records, out, variant)

    # ── Threshold curves ──────────────────────────────────────────────────────
    print("--- Threshold curves ---")
    for metric in ("recall", "precision", "f1"):
        plot_threshold_vs_horizon(records, out, thresholds, metric, variant)

    # ── Best top-K vs best threshold F1 comparison ────────────────────────────
    print("--- Best-K vs Best-τ F1 ---")
    plot_best_topk_vs_best_threshold_f1(records, out, variant)

    # ── Adaptive heatmaps ─────────────────────────────────────────────────────
    print("--- Adaptive heatmaps ---")
    plot_heatmap(records, out, "adaptive_recall", f"{variant} — Adaptive Recall")
    plot_heatmap(records, out, "adaptive_f1",     f"{variant} — Adaptive F1")

    print(f"\nAll outputs in {out}")


if __name__ == "__main__":
    main()
