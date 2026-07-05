#!/usr/bin/env python3
"""Analyze cache expansion vs oracle/predictor tradeoff from oracle_baseline_sweep CSV."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Qwen3-30B-A3B: 48 MoE layers, ~2.47 MB per expert (packed AWQ).
NUM_LAYERS = 48
EXPERT_MB = 2.47


def cache_mb(cache_size: int) -> float:
    return cache_size * NUM_LAYERS * EXPERT_MB


def draft_model_mb(num_draft_experts: int, quant_frac: float = 0.25) -> float:
  """DRAM for draft-model expert weights (fraction of full expert size)."""
  return num_draft_experts * NUM_LAYERS * EXPERT_MB * quant_frac


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--baseline-cache", type=int, default=8)
    p.add_argument("--draft-experts", type=int, default=32,
                   help="Draft model expert slots (default: 32 @ 1/4 size ≈ 8 full experts).")
    p.add_argument("--draft-quant-frac", type=float, default=0.25)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["question"] == "ORACLE_BASELINE_SWEEP"].copy()
    if df.empty:
        print("No ORACLE_BASELINE_SWEEP rows in CSV.", file=sys.stderr)
        return 1

    df = df[df["lookahead"] == 1].copy()
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.csv), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    def pick(label_prefix: str, cache_size: int) -> pd.Series | None:
        rows = df[(df["cache_size"] == cache_size) & (df["label"].str.startswith(label_prefix))]
        if rows.empty:
            return None
        return rows.iloc[0]

    # LRU scaling curve
    lru_rows = []
    for c in sorted(df["cache_size"].unique()):
        row = pick("Neither (LRU)", int(c))
        if row is not None:
            lru_rows.append(row)
    lru = pd.DataFrame(lru_rows).sort_values("cache_size")
    if lru.empty:
        print("No LRU rows found.", file=sys.stderr)
        return 1

    base_c = args.baseline_cache
    base_lru_tps = float(lru.loc[lru["cache_size"] == base_c, "tokens_per_second"].iloc[0])
    lru["cache_mb"] = lru["cache_size"].apply(cache_mb)
    lru["speedup_vs_lru_c8"] = lru["tokens_per_second"] / base_lru_tps

    # Marginal speedup per MB (finite differences on LRU curve)
    lru = lru.reset_index(drop=True)
    marginal = []
    for i in range(1, len(lru)):
        d_tps = lru.loc[i, "tokens_per_second"] - lru.loc[i - 1, "tokens_per_second"]
        d_mb = lru.loc[i, "cache_mb"] - lru.loc[i - 1, "cache_mb"]
        marginal.append({
            "from_c": int(lru.loc[i - 1, "cache_size"]),
            "to_c": int(lru.loc[i, "cache_size"]),
            "d_mb": d_mb,
            "d_tps": d_tps,
            "speedup_per_mb": d_tps / d_mb if d_mb > 0 else np.nan,
            "mid_c": (lru.loc[i - 1, "cache_size"] + lru.loc[i, "cache_size"]) / 2,
        })
    marg_df = pd.DataFrame(marginal)

    # Key operating points
    oracle8 = pick("Oracle Top-B", 8)
    pred40 = pick("Actual Predictor", 40)
    lru40 = pick("Neither (LRU)", 40)
    oracle40 = pick("Oracle Top-B", 40)

    def fmt_row(name: str, row: pd.Series | None) -> None:
        if row is None:
            print(f"  {name}: (missing)")
            return
        c = int(row["cache_size"])
        tps = float(row["tokens_per_second"])
        hr = float(row["hit_rate_pct"])
        sp = tps / base_lru_tps
        mb = cache_mb(c)
        print(f"  {name}: C={c}  TPS={tps:.3f}  hit={hr:.1f}%  "
              f"cache={mb:.0f} MB  speedup vs LRU@8={sp:.2f}x")

    print("=" * 72)
    print("Cache vs Oracle tradeoff")
    print(f"  baseline: LRU @ C={base_c}  TPS={base_lru_tps:.3f}  ({cache_mb(base_c):.0f} MB)")
    print("-" * 72)
    print("Operating points:")
    fmt_row("Oracle (100% draft)", oracle8)
    fmt_row("Actual predictor", pred40)
    fmt_row("LRU (large cache)", lru40)
    fmt_row("Oracle ceiling @ C=40", oracle40)

  # Interpolate: LRU cache size needed to match a target TPS
    def equiv_cache_size(target_tps: float) -> float | None:
        xs = lru["cache_size"].to_numpy(dtype=float)
        ys = lru["tokens_per_second"].to_numpy(dtype=float)
        if target_tps <= ys[0]:
            return float(xs[0])
        if target_tps >= ys[-1]:
            return None
        return float(np.interp(target_tps, ys, xs))

    draft_mb = draft_model_mb(args.draft_experts, args.draft_quant_frac)
    draft_equiv_experts = args.draft_experts * args.draft_quant_frac

    print("-" * 72)
    print("Memory framing (expert cache only, MB = C × 48 layers × 2.47 MB):")
    print(f"  C=8 cache:  {cache_mb(8):.0f} MB")
    print(f"  C=40 cache: {cache_mb(40):.0f} MB  (Δ = {cache_mb(40) - cache_mb(8):.0f} MB vs C=8)")
    print(f"  Draft model ({args.draft_experts} experts @ {args.draft_quant_frac:.0%} size): "
          f"{draft_mb:.0f} MB  (≈ {draft_equiv_experts:.0f} full experts)")

    if oracle8 is not None:
        o8_tps = float(oracle8["tokens_per_second"])
        eq = equiv_cache_size(o8_tps)
        print("-" * 72)
        print(f"Oracle@C=8 delivers TPS={o8_tps:.3f} ({o8_tps / base_lru_tps:.2f}x vs LRU@8)")
        if eq is not None:
            print(f"  LRU needs C≈{eq:.1f} ({cache_mb(int(round(eq))):.0f} MB) to match that TPS")
            print(f"  'Virtual' cache gain from perfect predictor: {eq - 8:.1f} experts "
                  f"({cache_mb(int(round(eq))) - cache_mb(8):.0f} MB) at zero draft cost")
            total_with_draft = cache_mb(8) + draft_mb
            print(f"  With draft memory ({total_with_draft:.0f} MB total): "
                  f"break-even LRU cache ≈ C={total_with_draft / (NUM_LAYERS * EXPERT_MB):.1f}")

    if pred40 is not None and lru40 is not None:
        p40 = float(pred40["tokens_per_second"])
        l40 = float(lru40["tokens_per_second"])
        print("-" * 72)
        print(f"Actual predictor@C=40: TPS={p40:.3f}  vs  LRU@C=40: TPS={l40:.3f}  "
              f"(predictor lift {p40 / l40:.2f}x)")
        eq_p = equiv_cache_size(p40)
        if eq_p is not None:
            print(f"  LRU needs C≈{eq_p:.1f} to match predictor@40 TPS (without predictor)")

    if oracle8 is not None and pred40 is not None:
        o8 = float(oracle8["tokens_per_second"])
        p40 = float(pred40["tokens_per_second"])
        print("-" * 72)
        print(f"Oracle@C=8 vs Actual@C=40: {o8:.3f} vs {p40:.3f} TPS  "
              f"(oracle {'wins' if o8 > p40 else 'loses'} by {abs(o8 - p40):.3f} TPS)")
        mem_oracle = cache_mb(8)
        mem_pred = cache_mb(40) + draft_mb
        print(f"  Memory: oracle path {mem_oracle:.0f} MB cache vs "
              f"predictor path {cache_mb(40):.0f} MB cache + {draft_mb:.0f} MB draft = {mem_pred:.0f} MB")

    print("-" * 72)
    print("LRU marginal speedup per MB (ΔTPS / ΔMB):")
    for _, r in marg_df.iterrows():
        print(f"  C={int(r['from_c'])}→{int(r['to_c'])}: "
              f"+{r['d_mb']:.0f} MB → +{r['d_tps']:.3f} TPS  "
              f"({r['speedup_per_mb'] * 1000:.4f} mTPS/MB)")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(lru["cache_mb"], lru["tokens_per_second"], "o-", label="LRU", linewidth=2)
    for name, row, marker in [
        ("Oracle@C=8", oracle8, "*"),
        ("Predictor@C=40", pred40, "D"),
        ("Oracle@C=40", oracle40, "s"),
    ]:
        if row is not None:
            ax.scatter(cache_mb(int(row["cache_size"])), row["tokens_per_second"],
                       s=120, marker=marker, label=name, zorder=5)
    ax.set_xlabel("Expert cache (MB)")
    ax.set_ylabel("TPS")
    ax.set_title("TPS vs cache capacity")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(lru["cache_mb"], lru["speedup_vs_lru_c8"], "o-", label="LRU", linewidth=2)
    if oracle8 is not None:
        ax.scatter(cache_mb(8), float(oracle8["tokens_per_second"]) / base_lru_tps,
                   s=120, marker="*", label="Oracle@C=8", zorder=5)
    if pred40 is not None:
        ax.scatter(cache_mb(40), float(pred40["tokens_per_second"]) / base_lru_tps,
                   s=120, marker="D", label="Predictor@C=40", zorder=5)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Expert cache (MB)")
    ax.set_ylabel("Speedup vs LRU@C=8")
    ax.set_title("Normalized speedup")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(marg_df["mid_c"], marg_df["speedup_per_mb"] * 1000, width=6, alpha=0.75)
    ax.set_xlabel("Cache size (experts, midpoint)")
    ax.set_ylabel("Marginal TPS gain per MB (×1000)")
    ax.set_title("LRU: diminishing returns per MB")
    ax.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "cache_vs_oracle_tradeoff.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot: {plot_path}")

    summary_path = os.path.join(out_dir, "summary.csv")
    lru_out = lru[["cache_size", "cache_mb", "tokens_per_second", "hit_rate_pct", "speedup_vs_lru_c8"]]
    lru_out.to_csv(summary_path, index=False)
    marg_df.to_csv(os.path.join(out_dir, "marginal_speedup_per_mb.csv"), index=False)
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
