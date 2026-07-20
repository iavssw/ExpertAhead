#!/usr/bin/env python3
"""
plot_ablation_comparison.py
============================
Canonical ablation study plotter.  Compares any set of expert-predictor
variants side-by-side across lookahead horizons and decoder layers.

Typical paper variants: emb_only / markov_only / emb_markov
But any variant whose sweep directory you point to will work.

Usage
-----
python plot_ablation_comparison.py \\
    --variant_dirs emb_only:/data/emb_only_sweep \\
                   markov_only:/data/markov_only_sweep \\
                   emb_markov:/data/emb_markov_sweep \\
    [--out_dir plots/ablation] \\
    [--eval_ks 8 16 32] \\
    [--main_eval_topk 8]

Produces (all as .png + .pdf)
-----------------------------
ablation_adaptive_f1_vs_horizon        Primary: adaptive F1 mean±std over layers
ablation_adaptive_f1_per_layer         One subplot per layer
ablation_precision_topk_vs_horizon     Fixed-K precision vs horizon
ablation_recall_topk_vs_horizon        Fixed-K recall vs horizon
ablation_f1_topk_vs_horizon            Fixed-K F1 vs horizon
ablation_threshold_recall_vs_horizon   Threshold-based recall vs horizon
ablation_threshold_precision_vs_horizon Threshold-based precision vs horizon
ablation_threshold_f1_vs_horizon       Threshold-based F1 vs horizon
ablation_density_confound_main         Raw F1 + random (density) baseline
ablation_density_confound_skill        Absolute skill = F1 - density
ablation_density_confound_norm         Normalized skill = (F1 - ρ) / (1 - ρ)
ablation_density_confound_combined     3-panel summary figure
ablation_label_density_vs_horizon      Label density sanity check
ablation_metrics_mean.csv
ablation_metrics_per_layer.csv
"""
from __future__ import annotations

import argparse
import csv
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
    load_sweep,
    ordered_variants,
    save_fig,
    set_horizon_xticks,
    style_for,
    VARIANT_LABELS,
)

apply_style()


# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────

def write_csvs(all_records: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted({fs for recs in all_records.values() for fs in recs})

    # ── per-layer long CSV ────────────────────────────────────────────────────
    rows: list[dict] = []
    for variant, recs in all_records.items():
        for fs in sorted(recs):
            for layer in sorted(recs[fs]):
                m = recs[fs][layer]
                rows.append({
                    "variant":             variant,
                    "horizon_N":           fs,
                    "layer":               layer,
                    "adaptive_f1":         m.get("adaptive_f1"),
                    "adaptive_recall":     m.get("adaptive_recall"),
                    "adaptive_precision":  m.get("adaptive_precision"),
                    "adaptive_avg_union":  m.get("adaptive_avg_union"),
                    "label_density":       m.get("label_density"),
                    "loss":                m.get("loss"),
                })
    if rows:
        p = out_dir / "ablation_metrics_per_layer.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {p.name}")

    # ── mean-over-layers CSV ──────────────────────────────────────────────────
    rows_mean: list[dict] = []
    for variant, recs in all_records.items():
        for fs in sorted(recs):
            row: dict = {"variant": variant, "horizon_N": fs, "n_layers": len(recs[fs])}
            for key in ("adaptive_f1", "adaptive_recall", "adaptive_precision",
                        "adaptive_avg_union", "label_density"):
                mn, sd, _ = aggregate(recs[fs], key)
                row[f"{key}_mean"] = mn
                row[f"{key}_std"]  = sd
            rows_mean.append(row)
    if rows_mean:
        p = out_dir / "ablation_metrics_mean.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows_mean[0].keys())
            w.writeheader()
            w.writerows(rows_mean)
        print(f"  Wrote {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive F1 plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_adaptive_f1_vs_horizon(all_records: dict, out_dir: Path) -> None:
    """Primary ablation figure: adaptive F1 mean ± std over layers."""
    horizons = sorted({fs for recs in all_records.values() for fs in recs})
    variants = ordered_variants(list(all_records))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in variants:
        recs  = all_records[variant]
        s     = style_for(variant)
        means = [aggregate(recs[fs], "adaptive_f1")[0] if fs in recs else float("nan")
                 for fs in horizons]
        stds  = [aggregate(recs[fs], "adaptive_f1")[1] if fs in recs else 0.0
                 for fs in horizons]
        ax.errorbar(
            horizons, means, yerr=stds,
            marker=s["marker"], ls=s["ls"], color=s["color"],
            linewidth=s["lw"], capsize=4, markersize=7,
            label=s["label"],
        )

    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel("Adaptive F1")
    ax.set_title("Expert Predictor Ablation — Adaptive F1 vs Horizon\n"
                 "(mean ± std over layers)")
    set_horizon_xticks(ax, horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / "ablation_adaptive_f1_vs_horizon")


def plot_adaptive_f1_per_layer(all_records: dict, out_dir: Path) -> None:
    """One subplot per decoder layer."""
    all_layers = sorted({
        l for recs in all_records.values()
        for fs_d in recs.values() for l in fs_d
    })
    horizons = sorted({fs for recs in all_records.values() for fs in recs})
    variants = ordered_variants(list(all_records))
    n = len(all_layers)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True, squeeze=False)
    for ax, layer in zip(axes[0], all_layers):
        for variant in variants:
            recs  = all_records[variant]
            s     = style_for(variant)
            vals  = [recs[fs].get(layer, {}).get("adaptive_f1", float("nan"))
                     if fs in recs else float("nan")
                     for fs in horizons]
            ax.plot(horizons, vals,
                    marker=s["marker"], ls=s["ls"], color=s["color"],
                    linewidth=2, markersize=6, label=s["label"])
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Horizon $N$")
        ax.set_ylim(0, 1.05)
        ax.grid(linestyle="--", alpha=0.35)
        set_horizon_xticks(ax, horizons)

    axes[0][0].set_ylabel("Adaptive F1")
    axes[0][-1].legend(framealpha=0.9)
    fig.suptitle("Adaptive F1 per decoder layer", fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_dir / "ablation_adaptive_f1_per_layer")


# ─────────────────────────────────────────────────────────────────────────────
# Fixed top-K curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_recall_precision_subplots(all_records: dict, out_dir: Path, eval_ks: list[int]) -> None:
    """
    For each B in eval_ks, creates a figure with 2 subplots: Recall@B and Precision@B.
    Plots mean with shaded standard deviation.
    """
    horizons = sorted({fs for recs in all_records.values() for fs in recs})
    variants = ordered_variants(list(all_records))

    for k in eval_ks:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        ax_rec = axes[0]
        ax_prec = axes[1]
        
        for variant in variants:
            recs = all_records[variant]
            s = style_for(variant)
            
            # Recall
            m_rec = [aggregate(recs[fs], f"topk_recall@{k}")[0] if fs in recs else float("nan") for fs in horizons]
            s_rec = [aggregate(recs[fs], f"topk_recall@{k}")[1] if fs in recs else 0.0 for fs in horizons]
            
            # Precision
            m_prec = [aggregate(recs[fs], f"topk_precision@{k}")[0] if fs in recs else float("nan") for fs in horizons]
            s_prec = [aggregate(recs[fs], f"topk_precision@{k}")[1] if fs in recs else 0.0 for fs in horizons]
            
            # Plot Recall
            ax_rec.plot(horizons, m_rec, marker=s["marker"], ls=s["ls"], color=s["color"], 
                        linewidth=s["lw"], markersize=6, label=s["label"])
            ax_rec.fill_between(horizons, 
                                [m - e for m, e in zip(m_rec, s_rec)], 
                                [m + e for m, e in zip(m_rec, s_rec)], 
                                color=s["color"], alpha=0.15)
                                
            # Plot Precision
            ax_prec.plot(horizons, m_prec, marker=s["marker"], ls=s["ls"], color=s["color"], 
                         linewidth=s["lw"], markersize=6, label=s["label"])
            ax_prec.fill_between(horizons, 
                                 [m - e for m, e in zip(m_prec, s_prec)], 
                                 [m + e for m, e in zip(m_prec, s_prec)], 
                                 color=s["color"], alpha=0.15)
                                 
        ax_rec.set_title(f"Recall@{k}")
        ax_rec.set_xlabel("Lookahead horizon $N$ (tokens)")
        ax_rec.set_ylabel("Score")
        ax_rec.set_ylim(0, 1.05)
        set_horizon_xticks(ax_rec, horizons)
        ax_rec.grid(linestyle="--", alpha=0.35)
        
        ax_prec.set_title(f"Precision@{k}")
        ax_prec.set_xlabel("Lookahead horizon $N$ (tokens)")
        ax_prec.set_ylim(0, 1.05)
        set_horizon_xticks(ax_prec, horizons)
        ax_prec.grid(linestyle="--", alpha=0.35)
        
        ax_prec.legend(fontsize=8, framealpha=0.9, loc="lower right")
        
        fig.suptitle(f"Recall and Precision @ {k} vs Horizon\n(mean ± std over layers)", fontsize=12)
        fig.tight_layout()
        save_fig(fig, out_dir / f"ablation_recall_precision_B{k}_subplots")


def plot_fixed_topk(all_records: dict, out_dir: Path, eval_ks: list[int], metric: str) -> None:
    """Fixed-K recall / precision / F1 curves, mean over layers."""
    horizons = sorted({fs for recs in all_records.values() for fs in recs})
    variants = ordered_variants(list(all_records))
    linestyles = ["-", "--", ":"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for ki, k in enumerate(eval_ks):
        for variant in variants:
            recs  = all_records[variant]
            s     = style_for(variant)
            means = [aggregate(recs[fs], f"topk_{metric}@{k}")[0] if fs in recs else float("nan")
                     for fs in horizons]
            ls = linestyles[ki % len(linestyles)]
            ax.plot(horizons, means,
                    marker=s["marker"], ls=ls, color=s["color"],
                    linewidth=1.8, markersize=5, alpha=0.85,
                    label=f"{s['label']}  K={k}")

    ax.set_xlabel("Lookahead horizon $N$")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Fixed-K {metric.capitalize()} vs Horizon — Ablation\n"
                 f"(mean over layers, K ∈ {eval_ks})")
    set_horizon_xticks(ax, horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / f"ablation_{metric}_topk_vs_horizon")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold-based curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_metric_vs_horizon(
    all_records: dict, out_dir: Path, thresholds: list[float], metric: str
) -> None:
    """Threshold-based recall / precision / F1 vs horizon, mean over layers."""
    if not thresholds:
        return
    horizons = sorted({fs for recs in all_records.values() for fs in recs})
    variants = ordered_variants(list(all_records))
    linestyles = ["-", "--", ":"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for ti, t in enumerate(thresholds):
        tag = f"{t:g}"
        for variant in variants:
            recs  = all_records[variant]
            s     = style_for(variant)
            means = [aggregate(recs[fs], f"thresh_{metric}@{tag}")[0] if fs in recs else float("nan")
                     for fs in horizons]
            ls = linestyles[ti % len(linestyles)]
            ax.plot(horizons, means,
                    marker=s["marker"], ls=ls, color=s["color"],
                    linewidth=1.8, markersize=5, alpha=0.85,
                    label=f"{s['label']}  τ={tag}")

    ax.set_xlabel("Lookahead horizon $N$")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Threshold {metric.capitalize()} vs Horizon — Ablation\n"
                 f"(sigmoid ≥ τ, mean over layers)")
    set_horizon_xticks(ax, horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / f"ablation_threshold_{metric}_vs_horizon")


# ─────────────────────────────────────────────────────────────────────────────
# Label density
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_density(all_records: dict, out_dir: Path) -> None:
    """Label density vs horizon — sanity check (should be same across variants)."""
    horizons = sorted({fs for recs in all_records.values() for fs in recs})
    variants = ordered_variants(list(all_records))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for variant in variants:
        recs  = all_records[variant]
        s     = style_for(variant)
        means = [aggregate(recs[fs], "label_density")[0] if fs in recs else float("nan")
                 for fs in horizons]
        ax.plot(horizons, means, marker=s["marker"], ls=s["ls"],
                color=s["color"], linewidth=1.8, markersize=6, label=s["label"])

    ax.set_xlabel("Lookahead horizon $N$")
    ax.set_ylabel("Label density (fraction of experts in union)")
    ax.set_title("Union label density vs horizon")
    set_horizon_xticks(ax, horizons)
    ax.set_ylim(0, 0.6)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / "ablation_label_density_vs_horizon")


# ─────────────────────────────────────────────────────────────────────────────
# Density confound / skill decomposition
# ─────────────────────────────────────────────────────────────────────────────

def _build_skill_table(all_records: dict) -> tuple[dict, dict]:
    """
    Returns:
      table[variant][fs]  = {f1, f1_std, density, density_std, skill, norm_skill}
      densities[fs]       = (mean, std)
    Density is a dataset property (same across variants); we average over all variants.
    """
    horizons = sorted({fs for recs in all_records.values() for fs in recs})

    # Use the first available variant for density (it's the same for all)
    density_src = next(iter(all_records.values()))
    densities: dict[int, tuple[float, float]] = {}
    for fs in horizons:
        if fs in density_src:
            mn, sd, _ = aggregate(density_src[fs], "label_density")
        else:
            mn, sd = float("nan"), float("nan")
        densities[fs] = (mn, sd)

    table: dict[str, dict[int, dict]] = {}
    for variant, recs in all_records.items():
        table[variant] = {}
        for fs in horizons:
            if fs not in recs:
                continue
            f1, f1s, _ = aggregate(recs[fs], "adaptive_f1")
            density, density_std = densities[fs]
            skill     = f1 - density
            headroom  = 1.0 - density
            norm_skill = skill / headroom if headroom > 1e-6 else float("nan")
            table[variant][fs] = {
                "f1": f1, "f1_std": f1s,
                "density": density, "density_std": density_std,
                "skill": skill, "norm_skill": norm_skill,
            }
    return table, densities


def _xticks(ax: plt.Axes, horizons: list) -> None:
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])


def plot_density_confound_main(table: dict, densities: dict, out_dir: Path) -> None:
    """Raw adaptive F1 with random (= density) baseline shaded."""
    horizons = sorted(densities)
    dens     = [densities[fs][0] for fs in horizons]
    variants = ordered_variants(list(table))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(horizons, 0, dens, color="gray", alpha=0.18,
                    label="Random baseline")
    ax.plot(horizons, dens, color="gray", lw=1.5, ls=":", label="_nolegend_")

    for variant in variants:
        if variant not in table:
            continue
        s   = style_for(variant)
        xs  = sorted(table[variant])
        ys  = [table[variant][fs]["f1"]     for fs in xs]
        yes = [table[variant][fs]["f1_std"] for fs in xs]
        ax.errorbar(xs, ys, yerr=yes,
                    marker=s["marker"], ls=s["ls"], color=s["color"],
                    lw=s["lw"], capsize=4, markersize=7, label=s["label"])

    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel("Adaptive F1")
    ax.set_title("Adaptive F1 vs Horizon with Random Baseline\n"
                 "(mean ± std over layers)")
    _xticks(ax, horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / "ablation_density_confound_main")


def plot_density_confound_skill(table: dict, horizons: list, out_dir: Path) -> None:
    """Absolute skill = F1 − label density."""
    variants = ordered_variants(list(table))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="gray", lw=1, ls=":")
    for variant in variants:
        if variant not in table:
            continue
        s  = style_for(variant)
        xs = sorted(table[variant])
        ys = [table[variant][fs]["skill"] for fs in xs]
        ax.plot(xs, ys, marker=s["marker"], ls=s["ls"],
                color=s["color"], lw=s["lw"], markersize=7, label=s["label"])

    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel("Skill = Adaptive F1 − Label Density")
    ax.set_title("Predictor Skill Above Random Baseline")
    _xticks(ax, horizons)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / "ablation_density_confound_skill")


def plot_density_confound_norm(table: dict, horizons: list, out_dir: Path) -> None:
    """Normalized skill = (F1 − density) / (1 − density)."""
    variants = ordered_variants(list(table))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.axhline(1, color="gray", lw=1, ls="--", alpha=0.4, label="Perfect predictor")
    for variant in variants:
        if variant not in table:
            continue
        s  = style_for(variant)
        xs = sorted(table[variant])
        ys = [table[variant][fs]["norm_skill"] for fs in xs]
        ax.plot(xs, ys, marker=s["marker"], ls=s["ls"],
                color=s["color"], lw=s["lw"], markersize=7, label=s["label"])

    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel("Normalized Skill\n$(F1 - \\rho) / (1 - \\rho)$")
    ax.set_title("Fraction of Headroom Above Random Captured\n"
                 r"$\rho$ = label density (= random baseline F1)")
    _xticks(ax, horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / "ablation_density_confound_norm")


def plot_density_confound_combined(table: dict, densities: dict, out_dir: Path) -> None:
    """3-panel paper figure: raw F1 | skill | norm_skill."""
    horizons = sorted(densities)
    dens     = [densities[fs][0] for fs in horizons]
    variants = ordered_variants(list(table))

    fig = plt.figure(figsize=(15, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── Panel A: raw F1 + baseline ──────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(horizons, 0, dens, color="gray", alpha=0.18,
                    label="Random\n(= density)")
    ax.plot(horizons, dens, color="gray", lw=1.5, ls=":")
    for variant in variants:
        if variant not in table:
            continue
        s   = style_for(variant)
        xs  = sorted(table[variant])
        ys  = [table[variant][fs]["f1"]     for fs in xs]
        yes = [table[variant][fs]["f1_std"] for fs in xs]
        ax.errorbar(xs, ys, yerr=yes, marker=s["marker"], ls=s["ls"],
                    color=s["color"], lw=2, capsize=3, markersize=6, label=s["label"])
    ax.set_title("(a) Raw Adaptive F1")
    ax.set_xlabel("Lookahead $N$")
    ax.set_ylabel("Adaptive F1")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(ls="--", alpha=0.3)
    _xticks(ax, horizons)

    # ── Panel B: skill ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(0, color="gray", lw=1, ls=":")
    for variant in variants:
        if variant not in table:
            continue
        s  = style_for(variant)
        xs = sorted(table[variant])
        ys = [table[variant][fs]["skill"] for fs in xs]
        ax.plot(xs, ys, marker=s["marker"], ls=s["ls"],
                color=s["color"], lw=2, markersize=6, label=s["label"])
    ax.set_title("(b) Absolute Skill\n$F1 - \\rho$")
    ax.set_xlabel("Lookahead $N$")
    ax.set_ylabel("F1 − Label Density $\\rho$")
    ax.grid(ls="--", alpha=0.3)
    _xticks(ax, horizons)
    ax.legend(fontsize=9, framealpha=0.9)

    # ── Panel C: normalized skill ────────────────────────────────────────────
    ax = axes[2]
    ax.axhline(1, color="gray", lw=1, ls="--", alpha=0.4)
    ax.axhline(0, color="gray", lw=1, ls=":")
    for variant in variants:
        if variant not in table:
            continue
        s  = style_for(variant)
        xs = sorted(table[variant])
        ys = [table[variant][fs]["norm_skill"] for fs in xs]
        ax.plot(xs, ys, marker=s["marker"], ls=s["ls"],
                color=s["color"], lw=2, markersize=6, label=s["label"])
    ax.set_title("(c) Normalized Skill\n$(F1 - \\rho)\\,/\\,(1 - \\rho)$")
    ax.set_xlabel("Lookahead $N$")
    ax.set_ylabel("Fraction of headroom above random")
    ax.set_ylim(0, 1.05)
    ax.grid(ls="--", alpha=0.3)
    _xticks(ax, horizons)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Disentangling predictor skill from label-density confound\n"
        "(mean over decoder layers)",
        fontsize=11, y=1.04,
    )
    save_fig(fig, out_dir / "ablation_density_confound_combined")


# ─────────────────────────────────────────────────────────────────────────────
# Best top-K vs best threshold F1 comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_topk_vs_best_threshold_f1(all_records: dict, out_dir: Path) -> None:
    """
    For each variant, plot two curves vs lookahead horizon:
      • solid  — oracle best-K F1: for each horizon/layer, max over all K of topk_f1@K
      • dashed — oracle best-τ F1: for each horizon/layer, max over all τ of thresh_f1@T

    Variants are distinguished by color; the two strategies by linestyle.
    This reveals whether threshold or top-K selection has higher ceiling,
    independently of any fixed K or τ choice.
    """
    horizons = sorted({fs for recs in all_records.values() for fs in recs})
    variants = ordered_variants(list(all_records))

    # Check that the data actually has these keys
    has_topk   = any(
        any(k.startswith("topk_f1@") for k in (ent.get("metrics", ent) if isinstance(ent, dict) else ent))
        for recs in all_records.values() for fs_d in recs.values() for ent in fs_d.values()
    )
    has_thresh = any(
        any(k.startswith("thresh_f1@") for k in (ent.get("metrics", ent) if isinstance(ent, dict) else ent))
        for recs in all_records.values() for fs_d in recs.values() for ent in fs_d.values()
    )
    if not has_topk and not has_thresh:
        print("  Skipping best top-K vs threshold plot: no topk_f1 or thresh_f1 keys found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    legend_handles = []
    for variant in variants:
        recs = all_records[variant]
        s    = style_for(variant)

        if has_topk:
            means_k = [best_param_aggregate(recs[fs], "topk_f1")[0] if fs in recs else float("nan")
                       for fs in horizons]
            stds_k  = [best_param_aggregate(recs[fs], "topk_f1")[1] if fs in recs else 0.0
                       for fs in horizons]
            line_k, = ax.plot(horizons, means_k,
                              marker=s["marker"], ls="-", color=s["color"],
                              linewidth=s["lw"], markersize=7,
                              label=f"{s['label']} — best K")
            ax.fill_between(horizons,
                            [m - e for m, e in zip(means_k, stds_k)],
                            [m + e for m, e in zip(means_k, stds_k)],
                            color=s["color"], alpha=0.10)
            legend_handles.append(line_k)

        if has_thresh:
            means_t = [best_param_aggregate(recs[fs], "thresh_f1")[0] if fs in recs else float("nan")
                       for fs in horizons]
            stds_t  = [best_param_aggregate(recs[fs], "thresh_f1")[1] if fs in recs else 0.0
                       for fs in horizons]
            line_t, = ax.plot(horizons, means_t,
                              marker=s["marker"], ls="--", color=s["color"],
                              linewidth=s["lw"] * 0.8, markersize=6, alpha=0.85,
                              label=f"{s['label']} — best τ")
            ax.fill_between(horizons,
                            [m - e for m, e in zip(means_t, stds_t)],
                            [m + e for m, e in zip(means_t, stds_t)],
                            color=s["color"], alpha=0.07)
            legend_handles.append(line_t)

    # Linestyle legend entries (strategy)
    from matplotlib.lines import Line2D
    strategy_handles = [
        Line2D([0], [0], color="gray", ls="-",  lw=2, label="Best K (top-K)"),
        Line2D([0], [0], color="gray", ls="--", lw=2, label="Best τ (threshold)"),
    ]

    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel("F1")
    ax.set_title("Oracle Best-K vs Best-τ F1 per Horizon\n"
                 "(solid = top-K, dashed = threshold; mean ± std over layers)")
    set_horizon_xticks(ax, horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(handles=legend_handles + strategy_handles,
              fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_dir / "ablation_best_topk_vs_best_threshold_f1")


def plot_best_k_prf(all_records: dict, out_dir: Path) -> None:
    """
    For each variant, plot Precision, Recall, and F1 at the best K value
    for each lookahead horizon. The best K value (maximizing F1) is annotated
    on the plot.

    Generates a single multi-panel figure (one subplot per variant) to compare
    the metrics.
    """
    variants = ordered_variants(list(all_records))
    horizons = sorted({fs for recs in all_records.values() for fs in recs})

    # Verify we actually have topk metrics
    has_topk = any(
        any(k.startswith("topk_f1@") for k in (ent.get("metrics", ent) if isinstance(ent, dict) else ent))
        for recs in all_records.values() for fs_d in recs.values() for ent in fs_d.values()
    )
    if not has_topk:
        print("  Skipping best-K PRF plot: no topk_f1 keys found.")
        return

    n_variants = len(variants)
    fig, axes = plt.subplots(1, n_variants, figsize=(5 * n_variants, 4.2), sharey=True)
    if n_variants == 1:
        axes = [axes]

    colors = {"precision": "#2eca7f", "recall": "#3a86c8", "f1": "#e63946"}
    markers = {"precision": "s", "recall": "o", "f1": "^"}
    linestyles = {"precision": "--", "recall": "--", "f1": "-"}

    for i, variant in enumerate(variants):
        ax = axes[i]
        recs = all_records[variant]

        best_ks = []
        means = {"precision": [], "recall": [], "f1": []}
        stds = {"precision": [], "recall": [], "f1": []}

        for fs in horizons:
            if fs in recs:
                k, prf = get_best_k_prf(recs[fs])
                best_ks.append(k)
                for metric in ("precision", "recall", "f1"):
                    mean, std = prf.get(metric, (float("nan"), 0.0))
                    means[metric].append(mean)
                    stds[metric].append(std)
            else:
                best_ks.append(None)
                for metric in ("precision", "recall", "f1"):
                    means[metric].append(float("nan"))
                    stds[metric].append(0.0)

        # Plot Precision, Recall, F1
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

            y_low = [y - s for y, s in zip(y_vals, std_y)]
            y_high = [y + s for y, s in zip(y_vals, std_y)]
            ax.fill_between(x_vals, y_low, y_high, color=colors[metric], alpha=0.08)

        # Plot Random F1 baseline
        random_f1s = []
        for fs, k in zip(horizons, best_ks):
            if fs in recs and k is not None:
                dens_val, _, _ = aggregate(recs[fs], "label_density")
                E = get_num_experts_from_records(recs[fs])
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
                            xytext=(0, offset_y), ha='center', fontsize=8, color="#333333",
                            fontweight='bold', bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

        ax.set_title(VARIANT_LABELS.get(variant, variant))
        ax.set_xlabel("Lookahead Horizon (N)")
        set_horizon_xticks(ax, horizons)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, ls=":", alpha=0.5)

        if i == 0:
            ax.set_ylabel("Metric Value")
            ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    save_fig(fig, out_dir / "ablation_best_k_prf")


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(all_records: dict) -> None:
    horizons = sorted({fs for recs in all_records.values() for fs in recs})
    variants = ordered_variants(list(all_records))
    col_w    = 16
    print(f"\n{'='*72}")
    print("  Ablation — Adaptive F1 (mean ± std over layers)")
    print(f"{'='*72}")
    header = f"  {'N':>4}" + "".join(f"  {v:>{col_w}}" for v in variants)
    print(header)
    print(f"  {'-'*68}")
    for fs in horizons:
        row = f"  {fs:>4}"
        for variant in variants:
            recs = all_records.get(variant, {})
            if fs not in recs:
                row += f"  {'—':>{col_w}}"
                continue
            mn, sd, _ = aggregate(recs[fs], "adaptive_f1")
            cell = f"{mn:.4f}±{sd:.4f}" if not (mn != mn) else "—"
            row += f"  {cell:>{col_w}}"
        print(row)
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_variant_dirs(raw: list[str]) -> dict[str, tuple[Path, str]]:
    """Parse 'variant_name:/path/to/dir' or 'variant_name:tag:/path/to/dir' tokens into a dict."""
    result: dict[str, tuple[Path, str]] = {}
    for token in raw:
        parts = token.split(":")
        if len(parts) == 2:
            name, path_str = parts
            tag = f"_{name}_hist"
        elif len(parts) == 3:
            name, tag, path_str = parts
        else:
            raise ValueError(
                f"--variant_dirs values must be 'variant_name:/path/to/dir' or 'name:tag:path', got: {token!r}"
            )
        name = name.strip()
        tag = tag.strip()
        p    = Path(path_str.strip())
        if not p.exists():
            print(f"  Warning: directory does not exist: {p}")
        result[name] = (p, tag)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ablation study plotter — compare any set of expert-predictor variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--variant_dirs", nargs="+", required=True, metavar="NAME:/path",
        help=(
            "One or more 'variant_name:/path/to/sweep_dir' pairs. "
            "Example: emb_only:/data/emb_only markov_only:/data/markov_only"
        ),
    )
    ap.add_argument("--out_dir",  default=None,
                    help="Output directory (default: first variant dir / paper_plots / ablation)")
    ap.add_argument("--eval_ks",  nargs="+", type=int, default=[8, 16, 32],
                    help="Fixed-K values for recall/precision/F1 plots (default: 8 16 32)")
    ap.add_argument("--main_eval_topk", type=int, default=8,
                    help="Primary reference K for top-K plots (default: 8)")
    ap.add_argument("--thresholds", nargs="+", type=float, default=None,
                    help="Sigmoid thresholds τ for threshold-based plots "
                         "(default: auto-discovered from data)")
    ap.add_argument("--no_skill", action="store_true",
                    help="Skip density-confound / skill decomposition plots")
    args = ap.parse_args()

    variant_dirs = parse_variant_dirs(args.variant_dirs)
    out = (
        Path(args.out_dir)
        if args.out_dir
        else next(iter(variant_dirs.values()))[0] / "paper_plots" / "ablation"
    )
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading sweep data…")
    all_records: dict[str, dict] = {}
    for variant, (d, tag) in variant_dirs.items():
        recs = load_sweep(d, variant_tag=tag)   # load all files matching the variant tag exactly
        layers = sorted({l for fs_d in recs.values() for l in fs_d})
        print(f"  {variant}: {len(recs)} horizons, layers {layers}")
        all_records[variant] = recs

    if not all_records or not any(recs for recs in all_records.values()):
        print("No data found — check your --variant_dirs paths.")
        return

    # ── Auto-discover thresholds ──────────────────────────────────────────────
    if args.thresholds:
        thresholds = args.thresholds
    else:
        flat_records = {}
        for recs in all_records.values():
            for fs, layers in recs.items():
                flat_records.setdefault(fs, {}).update(layers)
        thresholds = discover_metric_keys(flat_records, "thresh_recall")
    print(f"  Thresholds: {thresholds}")

    # ── Ensure main_eval_topk is in eval_ks ───────────────────────────────────
    eval_ks = sorted(set(args.eval_ks) | {args.main_eval_topk})

    print(f"\nOutputting to: {out}\n")
    write_csvs(all_records, out)
    print_table(all_records)

    # ── Adaptive F1 plots ─────────────────────────────────────────────────────
    plot_adaptive_f1_vs_horizon(all_records, out)
    plot_adaptive_f1_per_layer(all_records, out)

    # Added requested 2-panel subplots for B=8, 16, 32
    plot_recall_precision_subplots(all_records, out, args.eval_ks)

    # ── Fixed top-K plots ─────────────────────────────────────────────────────
    for metric in ("recall", "precision", "f1"):
        plot_fixed_topk(all_records, out, eval_ks, metric)

    # ── Threshold-based plots ─────────────────────────────────────────────────
    for metric in ("recall", "precision", "f1"):
        plot_threshold_metric_vs_horizon(all_records, out, thresholds, metric)

    # ── Label density sanity check ────────────────────────────────────────────
    plot_label_density(all_records, out)

    # ── Best top-K vs best threshold F1 comparison ────────────────────────────
    plot_best_topk_vs_best_threshold_f1(all_records, out)

    # ── Best top-K Precision/Recall/F1 ────────────────────────────────────────
    plot_best_k_prf(all_records, out)

    # ── Density confound / skill decomposition ────────────────────────────────
    if not args.no_skill:
        table, densities = _build_skill_table(all_records)
        horizons = sorted(densities)
        plot_density_confound_main(table, densities, out)
        plot_density_confound_skill(table, horizons, out)
        plot_density_confound_norm(table, horizons, out)
        plot_density_confound_combined(table, densities, out)

    print(f"\nDone. All outputs in: {out}")



if __name__ == "__main__":
    main()
