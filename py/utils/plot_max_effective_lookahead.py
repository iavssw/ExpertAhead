#!/usr/bin/env python3
"""Compare max effective lookahead: perfect oracle vs actual ML predictor.

For each cache size C, we take the best TPS at each lookahead N (oracle: Full Union;
predictor: best Prefetch Only budget B). Then:

  peak_la       — N with highest TPS among tested lookaheads
  max_eff_la    — largest N where TPS(N) >= (1 - tolerance) * peak TPS

This captures how far lookahead can be pushed before longer horizons stop helping
(predictor recall decay) vs the oracle ceiling (routing is perfect).

Usage:
  python py/utils/plot_max_effective_lookahead.py \\
    --oracle-csv py/utils/final_results_runs/oracle_full_union_la_sweep/oracle_full_union_c8_64_20260703_151716.csv \\
    --predictor-csv py/utils/final_results_runs/ml_predictor_la_sweep/ml_predictor_20260702_212656.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

ORACLE_COLOR = "#0072B2"
PREDICTOR_COLOR = "#E69F00"
LRU_COLOR = "#333333"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oracle-csv",
        default="py/utils/final_results_runs/oracle_full_union_la_sweep/"
        "oracle_full_union_c8_64_20260703_151716.csv",
    )
    p.add_argument(
        "--predictor-csv",
        default="py/utils/final_results_runs/ml_predictor_la_sweep/"
        "ml_predictor_20260702_212656.csv",
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: oracle CSV dir)")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Max effective LA: TPS must be >= (1 - tol) * peak (default 0.01 = 1%%).",
    )
    p.add_argument(
        "--cache-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these cache sizes (default: intersection of both CSVs).",
    )
    p.add_argument(
        "--include-noisy-oracle",
        action="store_true",
        help="Also plot 7/8 noisy oracle curve alongside perfect oracle.",
    )
    return p.parse_args()


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def load_oracle_tps_by_la(
    csv_path: Path,
    *,
    perfect: bool = True,
    agreement: float = 0.875,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["question"] == "ORACLE_BASELINE_SWEEP"].copy()
    df["cache_size"] = _numeric(df, "cache_size")
    df["lookahead"] = _numeric(df, "lookahead")
    df["tokens_per_second"] = _numeric(df, "tokens_per_second")

    if perfect:
        mask = df["label"].astype(str).str.startswith("Oracle Full Union", na=False)
    else:
        mask = df["label"].astype(str).str.startswith("Oracle Noisy", na=False)
        if "oracle_routing_agreement" in df.columns:
            agr = _numeric(df, "oracle_routing_agreement")
            mask = mask | (agr.notna() & np.isclose(agr, agreement, rtol=0, atol=1e-3))

    df = df[mask].dropna(subset=["cache_size", "lookahead", "tokens_per_second"])
    rows = []
    for (c, la), grp in df.groupby(["cache_size", "lookahead"], sort=True):
        rows.append({
            "cache_size": int(c),
            "lookahead": int(la),
            "tokens_per_second": float(grp["tokens_per_second"].max()),
        })
    return pd.DataFrame(rows)


def load_predictor_tps_by_la(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    q = df.get("question", pd.Series("", index=df.index)).astype(str)
    lab = df.get("label", pd.Series("", index=df.index)).astype(str)
    lam = _numeric(df, "lambda_val").fillna(0.0)
    backend = df.get("backend", pd.Series("", index=df.index)).astype(str)

    mask = (
        (backend == "predict")
        & (lam == 0.0)
        & lab.str.startswith("Prefetch Only", na=False)
        & q.str.contains("CUSTOM_1_16", na=False)
    )
    df = df.loc[mask].copy()
    df["cache_size"] = _numeric(df, "cache_size")
    df["lookahead"] = _numeric(df, "lookahead")
    df["prefetch_budget"] = _numeric(df, "prefetch_budget")
    df["tokens_per_second"] = _numeric(df, "tokens_per_second")
    df = df.dropna(subset=["cache_size", "lookahead", "tokens_per_second"])

    rows = []
    for (c, la), grp in df.groupby(["cache_size", "lookahead"], sort=True):
        best = grp.loc[grp["tokens_per_second"].idxmax()]
        rows.append({
            "cache_size": int(c),
            "lookahead": int(la),
            "tokens_per_second": float(best["tokens_per_second"]),
            "prefetch_budget": int(best["prefetch_budget"]) if pd.notna(best["prefetch_budget"]) else None,
        })
    return pd.DataFrame(rows)


def load_lru_tps(csv_path: Path, *, question: str) -> dict[int, float]:
    df = pd.read_csv(csv_path)
    df = df[df["question"] == question].copy()
    df["cache_size"] = _numeric(df, "cache_size")
    df["tokens_per_second"] = _numeric(df, "tokens_per_second")
    lru = df[df["label"] == "Neither (LRU)"].dropna(subset=["cache_size", "tokens_per_second"])
    lru = lru.drop_duplicates("cache_size")
    return dict(zip(lru["cache_size"].astype(int), lru["tokens_per_second"].astype(float)))


def max_effective_lookahead(
    curve: pd.DataFrame,
    *,
    tolerance: float,
) -> dict[str, Optional[int | float]]:
    """Return peak and max-effective lookahead for one cache-size curve."""
    if curve.empty:
        return {
            "peak_la": None,
            "peak_tps": np.nan,
            "max_eff_la": None,
            "max_eff_tps": np.nan,
        }

    tab = curve.sort_values("lookahead")
    peak_row = tab.loc[tab["tokens_per_second"].idxmax()]
    peak_tps = float(peak_row["tokens_per_second"])
    peak_la = int(peak_row["lookahead"])
    floor = peak_tps * (1.0 - tolerance)

    eligible = tab[tab["tokens_per_second"] >= floor]
    max_eff_row = eligible.loc[eligible["lookahead"].idxmax()]
    return {
        "peak_la": peak_la,
        "peak_tps": peak_tps,
        "max_eff_la": int(max_eff_row["lookahead"]),
        "max_eff_tps": float(max_eff_row["tokens_per_second"]),
    }


def summarize(
    oracle: pd.DataFrame,
    predictor: pd.DataFrame,
    *,
    cache_sizes: list[int],
    tolerance: float,
) -> pd.DataFrame:
    rows = []
    for c in cache_sizes:
        o_curve = oracle[oracle["cache_size"] == c]
        p_curve = predictor[predictor["cache_size"] == c]
        o_stats = max_effective_lookahead(o_curve, tolerance=tolerance)
        p_stats = max_effective_lookahead(p_curve, tolerance=tolerance)
        rows.append({
            "cache_size": c,
            "oracle_peak_la": o_stats["peak_la"],
            "oracle_peak_tps": o_stats["peak_tps"],
            "oracle_max_eff_la": o_stats["max_eff_la"],
            "oracle_max_eff_tps": o_stats["max_eff_tps"],
            "predictor_peak_la": p_stats["peak_la"],
            "predictor_peak_tps": p_stats["peak_tps"],
            "predictor_max_eff_la": p_stats["max_eff_la"],
            "predictor_max_eff_tps": p_stats["max_eff_tps"],
            "peak_la_gap": (
                (o_stats["peak_la"] - p_stats["peak_la"])
                if o_stats["peak_la"] is not None and p_stats["peak_la"] is not None
                else np.nan
            ),
            "max_eff_la_gap": (
                (o_stats["max_eff_la"] - p_stats["max_eff_la"])
                if o_stats["max_eff_la"] is not None and p_stats["max_eff_la"] is not None
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def plot_tps_vs_lookahead_panels(
    oracle: pd.DataFrame,
    predictor: pd.DataFrame,
    lru_map: dict[int, float],
    cache_sizes: list[int],
    *,
    out_path: Path,
    noisy_oracle: Optional[pd.DataFrame] = None,
    tolerance: float = 0.01,
) -> None:
    n = len(cache_sizes)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows), squeeze=False)

    for idx, c in enumerate(cache_sizes):
        ax = axes[idx // ncols][idx % ncols]
        o_curve = oracle[oracle["cache_size"] == c].sort_values("lookahead")
        p_curve = predictor[predictor["cache_size"] == c].sort_values("lookahead")

        if c in lru_map:
            ax.axhline(lru_map[c], color=LRU_COLOR, linestyle=":", linewidth=1.8, label=f"LRU ({lru_map[c]:.1f})")

        if not o_curve.empty:
            ax.plot(
                o_curve["lookahead"],
                o_curve["tokens_per_second"],
                "o-",
                color=ORACLE_COLOR,
                linewidth=2.2,
                label="Oracle Full Union",
            )
            o_stats = max_effective_lookahead(o_curve, tolerance=tolerance)
            if o_stats["max_eff_la"] is not None:
                ax.axvline(
                    o_stats["max_eff_la"],
                    color=ORACLE_COLOR,
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.2,
                )

        if noisy_oracle is not None:
            n_curve = noisy_oracle[noisy_oracle["cache_size"] == c].sort_values("lookahead")
            if not n_curve.empty:
                ax.plot(
                    n_curve["lookahead"],
                    n_curve["tokens_per_second"],
                    "s--",
                    color="#56B4E9",
                    linewidth=1.6,
                    alpha=0.85,
                    label="Oracle 7/8",
                )

        if not p_curve.empty:
            ax.plot(
                p_curve["lookahead"],
                p_curve["tokens_per_second"],
                "^-",
                color=PREDICTOR_COLOR,
                linewidth=2.2,
                label="ML Predictor (best B)",
            )
            p_stats = max_effective_lookahead(p_curve, tolerance=tolerance)
            if p_stats["max_eff_la"] is not None:
                ax.axvline(
                    p_stats["max_eff_la"],
                    color=PREDICTOR_COLOR,
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.2,
                )

        ax.set_title(f"C = {c}")
        ax.set_xlabel("Lookahead N")
        ax.set_ylabel("TPS")
        ax.legend(fontsize=8, loc="lower right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "TPS vs lookahead: oracle ceiling vs ML predictor\n"
        f"Dashed verticals = max effective LA (within {100*tolerance:.0f}% of peak TPS)",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_max_eff_la_vs_cache(
    summary: pd.DataFrame,
    *,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = summary["cache_size"].astype(int)
    x_arr = np.array(list(x), dtype=float)
    w = 1.4

    ax.bar(
        x_arr - w / 4,
        summary["oracle_peak_la"],
        width=w / 2,
        color=ORACLE_COLOR,
        alpha=0.95,
        label="Oracle peak LA (max TPS)",
    )
    ax.bar(
        x_arr + w / 4,
        summary["predictor_peak_la"],
        width=w / 2,
        color=PREDICTOR_COLOR,
        alpha=0.95,
        label="Predictor peak LA (max TPS)",
    )
    ax.plot(
        x,
        summary["oracle_max_eff_la"],
        "o--",
        color=ORACLE_COLOR,
        linewidth=1.6,
        markersize=7,
        alpha=0.7,
        label="Oracle max effective LA",
    )
    ax.plot(
        x,
        summary["predictor_max_eff_la"],
        "^--",
        color=PREDICTOR_COLOR,
        linewidth=1.6,
        markersize=7,
        alpha=0.7,
        label="Predictor max effective LA",
    )

    for _, r in summary.iterrows():
        gap = r["peak_la_gap"]
        if pd.notna(gap) and gap != 0:
            y = max(r["oracle_peak_la"], r["predictor_peak_la"])
            ax.annotate(
                f"Δpeak={int(gap):+d}",
                (r["cache_size"], y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="#444444",
            )

    ax.set_xlabel("Expert cache size C")
    ax.set_ylabel("Lookahead N")
    ax.set_title("Peak vs max-effective lookahead: oracle vs ML predictor")
    ax.set_xticks(list(x))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    oracle_path = Path(args.oracle_csv)
    predictor_path = Path(args.predictor_csv)
    out_dir = Path(args.out_dir) if args.out_dir else oracle_path.parent

    oracle = load_oracle_tps_by_la(oracle_path, perfect=True)
    predictor = load_predictor_tps_by_la(predictor_path)
    noisy = (
        load_oracle_tps_by_la(oracle_path, perfect=False)
        if args.include_noisy_oracle
        else None
    )

    oracle_caches = set(oracle["cache_size"].unique())
    pred_caches = set(predictor["cache_size"].unique())
    if args.cache_sizes:
        cache_sizes = sorted(set(args.cache_sizes) & oracle_caches & pred_caches)
    else:
        cache_sizes = sorted(oracle_caches & pred_caches)

    if not cache_sizes:
        raise SystemExit(
            f"No overlapping cache sizes.\n"
            f"  Oracle:    {sorted(oracle_caches)}\n"
            f"  Predictor: {sorted(pred_caches)}"
        )

    lru_oracle = load_lru_tps(oracle_path, question="ORACLE_BASELINE_SWEEP")
    lru_pred = load_lru_tps(predictor_path, question="CUSTOM_1_16_NO_PPL")
    lru_map = {**lru_pred, **lru_oracle}

    summary = summarize(oracle, predictor, cache_sizes=cache_sizes, tolerance=args.tolerance)
    csv_out = out_dir / "max_effective_lookahead_summary.csv"
    summary.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")
    print(summary.to_string(index=False))

    stem = oracle_path.stem
    plot_tps_vs_lookahead_panels(
        oracle,
        predictor,
        lru_map,
        cache_sizes,
        out_path=out_dir / f"{stem}_max_eff_la_curves.png",
        noisy_oracle=noisy,
        tolerance=args.tolerance,
    )
    plot_max_eff_la_vs_cache(
        summary,
        out_path=out_dir / f"{stem}_max_eff_la_vs_cache.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
