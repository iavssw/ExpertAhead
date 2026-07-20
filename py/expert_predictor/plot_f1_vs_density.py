#!/usr/bin/env python3
"""
plot_f1_vs_density.py
======================
Separates genuine predictor skill from the label-density confound.

For the adaptive-K metric (K_i = |union_i| per prompt), a random predictor
achieves F1 = label_density.  We compute:

  skill      = F1 - density          (absolute gain over random)
  norm_skill = (F1 - density) / (1 - density)   (fraction of headroom captured)

Produces:
  density_confound_main.png/.pdf      — raw F1 + random baseline
  density_confound_skill.png/.pdf     — absolute skill (F1 - density)
  density_confound_norm.png/.pdf      — normalized skill
  density_confound_combined.png/.pdf  — 3-panel summary figure (for paper)
  density_confound_table.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

VARIANTS = {
    "emb_only":    dict(color="#E63946", marker="s", ls="-",  label="Embeddings only"),
    "markov_only": dict(color="#457B9D", marker="^", ls="--", label="Markov only"),
    "emb_markov":  dict(color="#2EC4B6", marker="o", ls="-",  label="Embeddings + Markov"),
}


# ── Data loading ───────────────────────────────────────────────────────────────
def load_sweep(sweep_dir: Path, variant_tag: str) -> dict:
    records: dict[int, dict[int, dict]] = {}
    for p in sweep_dir.rglob("training_metrics.json"):
        if variant_tag not in str(p):
            continue
        with open(p) as f:
            d = json.load(f)
        fs    = d.get("future_steps")
        layer = d.get("layer_idx")
        if fs is None or layer is None:
            continue
        records.setdefault(fs, {})[layer] = d.get("best_val_recalls", {})
    return records


def agg(records_fs: dict, key: str):
    vals = [v for ent in records_fs.values() if (v := ent.get(key)) is not None]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def build_table(all_records: dict) -> dict:
    """
    Returns table[variant][fs] = {f1, f1_std, density, density_std,
                                   skill, norm_skill}
    density is averaged from all variants (it's a dataset property, not model).
    """
    horizons = sorted({fs for recs in all_records.values() for fs in recs})

    # Density is identical across variants; use emb_markov if available, else first
    density_src = all_records.get("emb_markov") or next(iter(all_records.values()))
    densities = {}
    for fs in horizons:
        if fs in density_src:
            d, ds = agg(density_src[fs], "label_density")
        else:
            d, ds = np.nan, np.nan
        densities[fs] = (d, ds)

    table = {}
    for variant, recs in all_records.items():
        table[variant] = {}
        for fs in horizons:
            if fs not in recs:
                continue
            f1, f1s = agg(recs[fs], "adaptive_f1")
            density, density_std = densities[fs]
            skill      = f1 - density
            headroom   = 1.0 - density
            norm_skill = skill / headroom if headroom > 1e-6 else np.nan
            table[variant][fs] = {
                "f1": f1, "f1_std": f1s,
                "density": density, "density_std": density_std,
                "skill": skill,
                "norm_skill": norm_skill,
            }
    return table, densities


def write_csv(table: dict, densities: dict, out_dir: Path):
    rows = []
    for variant, fs_dict in table.items():
        for fs, v in sorted(fs_dict.items()):
            rows.append({
                "variant": variant, "horizon_N": fs,
                "adaptive_f1": v["f1"], "adaptive_f1_std": v["f1_std"],
                "label_density": v["density"],
                "skill_f1_minus_density": v["skill"],
                "norm_skill": v["norm_skill"],
            })
    p = out_dir / "density_confound_table.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Wrote {p.name}")


# ── Plots ──────────────────────────────────────────────────────────────────────
def _xticks(ax, horizons):
    ax.set_xticks(horizons)
    ax.set_xticklabels(horizons)


def _save(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {base.name}")


def plot_raw_with_baseline(table, densities, out_dir):
    """Raw F1 + random baseline shaded region."""
    horizons = sorted(densities)
    dens = [densities[fs][0] for fs in horizons]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Random baseline band
    ax.fill_between(horizons, 0, dens, color="gray", alpha=0.18, label="Random baseline (= density)")
    ax.plot(horizons, dens, color="gray", lw=1.5, ls=":", label="_nolegend_")

    for variant, fs_dict in table.items():
        style = VARIANTS[variant]
        xs = sorted(fs_dict)
        ys  = [fs_dict[fs]["f1"]     for fs in xs]
        yes = [fs_dict[fs]["f1_std"] for fs in xs]
        ax.errorbar(xs, ys, yerr=yes,
                    marker=style["marker"], ls=style["ls"], color=style["color"],
                    lw=2.2, capsize=4, ms=7, label=style["label"])

    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel("Adaptive F1")
    ax.set_title("Adaptive F1 vs Horizon with Random Baseline\n(Qwen3-30B, mean ± std over layers 0/23/47)")
    _xticks(ax, horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    _save(fig, out_dir / "density_confound_main")


def plot_skill(table, horizons, out_dir):
    """Absolute skill = F1 - density."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="gray", lw=1, ls=":")

    for variant, fs_dict in table.items():
        style = VARIANTS[variant]
        xs = sorted(fs_dict)
        ys = [fs_dict[fs]["skill"] for fs in xs]
        ax.plot(xs, ys, marker=style["marker"], ls=style["ls"],
                color=style["color"], lw=2.2, ms=7, label=style["label"])

    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel("Skill = Adaptive F1 − Label Density")
    ax.set_title("Predictor Skill Above Random Baseline")
    _xticks(ax, horizons)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    _save(fig, out_dir / "density_confound_skill")


def plot_norm_skill(table, horizons, out_dir):
    """Normalized skill = (F1 - density) / (1 - density)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.axhline(1, color="gray", lw=1, ls="--", alpha=0.4, label="Perfect predictor")

    for variant, fs_dict in table.items():
        style = VARIANTS[variant]
        xs = sorted(fs_dict)
        ys = [fs_dict[fs]["norm_skill"] for fs in xs]
        ax.plot(xs, ys, marker=style["marker"], ls=style["ls"],
                color=style["color"], lw=2.2, ms=7, label=style["label"])

    ax.set_xlabel("Lookahead horizon $N$ (tokens)")
    ax.set_ylabel("Normalized Skill\n$(F1 - \\rho) / (1 - \\rho)$")
    ax.set_title("Fraction of Headroom Above Random Captured\n"
                 r"$\rho$ = label density (= random baseline F1)")
    _xticks(ax, horizons)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    _save(fig, out_dir / "density_confound_norm")


def plot_combined(table, densities, out_dir):
    """3-panel paper figure: raw F1 | skill | norm_skill."""
    horizons = sorted(densities)
    dens = [densities[fs][0] for fs in horizons]

    fig = plt.figure(figsize=(15, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── Panel A: Raw F1 + baseline ──
    ax = axes[0]
    ax.fill_between(horizons, 0, dens, color="gray", alpha=0.18,
                    label="Random\n(= density)")
    ax.plot(horizons, dens, color="gray", lw=1.5, ls=":")
    for variant, fs_dict in table.items():
        style = VARIANTS[variant]
        xs = sorted(fs_dict)
        ys  = [fs_dict[fs]["f1"]     for fs in xs]
        yes = [fs_dict[fs]["f1_std"] for fs in xs]
        ax.errorbar(xs, ys, yerr=yes, marker=style["marker"], ls=style["ls"],
                    color=style["color"], lw=2, capsize=3, ms=6,
                    label=style["label"])
    ax.set_title("(a) Raw Adaptive F1")
    ax.set_xlabel("Lookahead $N$")
    ax.set_ylabel("Adaptive F1")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(ls="--", alpha=0.3)
    _xticks(ax, horizons)

    # ── Panel B: Skill ──
    ax = axes[1]
    ax.axhline(0, color="gray", lw=1, ls=":")
    for variant, fs_dict in table.items():
        style = VARIANTS[variant]
        xs = sorted(fs_dict)
        ys = [fs_dict[fs]["skill"] for fs in xs]
        ax.plot(xs, ys, marker=style["marker"], ls=style["ls"],
                color=style["color"], lw=2, ms=6, label=style["label"])
    ax.set_title("(b) Absolute Skill\n$F1 - \\rho$")
    ax.set_xlabel("Lookahead $N$")
    ax.set_ylabel("F1 − Label Density $\\rho$")
    ax.grid(ls="--", alpha=0.3)
    _xticks(ax, horizons)
    ax.legend(fontsize=9, framealpha=0.9)

    # ── Panel C: Normalized skill ──
    ax = axes[2]
    ax.axhline(1, color="gray", lw=1, ls="--", alpha=0.4)
    ax.axhline(0, color="gray", lw=1, ls=":")
    for variant, fs_dict in table.items():
        style = VARIANTS[variant]
        xs = sorted(fs_dict)
        ys = [fs_dict[fs]["norm_skill"] for fs in xs]
        ax.plot(xs, ys, marker=style["marker"], ls=style["ls"],
                color=style["color"], lw=2, ms=6, label=style["label"])
    ax.set_title("(c) Normalized Skill\n$(F1 - \\rho)\\,/\\,(1 - \\rho)$")
    ax.set_xlabel("Lookahead $N$")
    ax.set_ylabel("Fraction of headroom above random")
    ax.set_ylim(0, 1.05)
    ax.grid(ls="--", alpha=0.3)
    _xticks(ax, horizons)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle("Qwen3-30B — Disentangling predictor skill from label-density confound\n"
                 "(mean over decoder layers 0 / 23 / 47)",
                 fontsize=11, y=1.04)
    _save(fig, out_dir / "density_confound_combined")


def print_summary(table, densities):
    horizons = sorted(densities)
    print(f"\n{'='*80}")
    print("  Skill decomposition: F1 = density (random) + skill")
    print(f"{'='*80}")
    print(f"  {'N':>4}  {'density':>8}  "
          + "  ".join(f"{'F1':>8} {'skill':>7} {'norm':>6} ({v})"
                      for v in table))
    print(f"  {'-'*78}")
    for fs in horizons:
        d = densities[fs][0]
        row = f"  {fs:>4}  {d:>8.4f}"
        for fs_dict in table.values():
            if fs not in fs_dict:
                row += f"  {'—':>8} {'—':>7} {'—':>6}"
                continue
            v = fs_dict[fs]
            row += f"  {v['f1']:>8.4f} {v['skill']:>7.4f} {v['norm_skill']:>6.4f}"
        print(row)
    print(f"{'='*80}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_markov_dir",  required=True)
    ap.add_argument("--emb_only_dir",    required=True)
    ap.add_argument("--markov_only_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    sweep_dirs = {
        "emb_only":    Path(args.emb_only_dir),
        "markov_only": Path(args.markov_only_dir),
        "emb_markov":  Path(args.emb_markov_dir),
    }
    out = Path(args.out_dir) if args.out_dir else \
          Path(args.emb_markov_dir) / "paper_plots" / "ablation"
    out.mkdir(parents=True, exist_ok=True)

    print("\nLoading sweep data…")
    all_records = {}
    for variant, d in sweep_dirs.items():
        recs = load_sweep(d, variant)
        print(f"  {variant}: {len(recs)} horizons")
        all_records[variant] = recs

    table, densities = build_table(all_records)
    horizons = sorted(densities)

    print_summary(table, densities)
    write_csv(table, densities, out)

    print(f"\nGenerating plots → {out}")
    plot_raw_with_baseline(table, densities, out)
    plot_skill(table, horizons, out)
    plot_norm_skill(table, horizons, out)
    plot_combined(table, densities, out)
    print("\nDone.")


if __name__ == "__main__":
    main()
