#!/usr/bin/env python3
"""Plot LRU baseline vs best Prefetch Only (LA=1 and best LA) across cache sizes."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
})

DEFAULT_DIRS = {
    8: "/home/michael/heteroPredict/py/utils/final_results_runs/C8_comparison_20260703_132626",
    16: "/home/michael/heteroPredict/py/utils/final_results_runs/C16_comparison_20260703_154927",
    24: "/home/michael/heteroPredict/py/utils/final_results_runs/C24_comparison_20260704_120612",
    32: "/home/michael/heteroPredict/py/utils/final_results_runs/C32_comparison_20260704_162727",
    40: "/home/michael/heteroPredict/py/utils/final_results_runs/C40_comparison_20260704_220143",
}


def _prefetch_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label"].astype(str).str.startswith("Prefetch Only")].copy()


def _lru_baseline(df: pd.DataFrame) -> Optional[pd.Series]:
    rows = df[df["label"] == "Neither (LRU)"]
    return rows.iloc[0] if not rows.empty else None


def best_prefetch_at_la(df: pd.DataFrame, la: int) -> Optional[pd.Series]:
    sub = _prefetch_only(df)
    sub = sub[sub["lookahead"] == la]
    if sub.empty:
        return None
    return sub.loc[sub["tokens_per_second"].idxmax()]


def best_prefetch_any_la(df: pd.DataFrame) -> Optional[pd.Series]:
    sub = _prefetch_only(df)
    if sub.empty:
        return None
    return sub.loc[sub["tokens_per_second"].idxmax()]


def load_comparison_dirs(dirs: Dict[int, str]) -> pd.DataFrame:
    rows: List[dict] = []
    for cache_size in sorted(dirs):
        path = os.path.join(dirs[cache_size], "sweep.csv")
        df = pd.read_csv(path)
        lru = _lru_baseline(df)
        la1 = best_prefetch_at_la(df, 1)
        best = best_prefetch_any_la(df)

        if lru is not None:
            rows.append({
                "cache_size": cache_size,
                "series": "LRU baseline",
                "tps": float(lru["tokens_per_second"]),
                "label": lru["label"],
                "lookahead": None,
                "budget": None,
            })
        if la1 is not None:
            rows.append({
                "cache_size": cache_size,
                "series": "Prefetch Only (best B, LA=1)",
                "tps": float(la1["tokens_per_second"]),
                "label": la1["label"],
                "lookahead": int(la1["lookahead"]),
                "budget": int(la1["prefetch_budget"]),
            })
        if best is not None:
            rows.append({
                "cache_size": cache_size,
                "series": "Prefetch Only (best LA & B)",
                "tps": float(best["tokens_per_second"]),
                "label": best["label"],
                "lookahead": int(best["lookahead"]),
                "budget": int(best["prefetch_budget"]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/home/michael/heteroPredict/py/utils/final_results_runs/cache_comparison_prefetch.png")
    p.add_argument("--csv-out", default="/home/michael/heteroPredict/py/utils/final_results_runs/cache_comparison_prefetch_summary.csv")
    args = p.parse_args()

    summary = load_comparison_dirs(DEFAULT_DIRS)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary.to_csv(args.csv_out, index=False)

    print("Best prefetch configs per cache size (Prefetch Only, ignoring CC/Hybrid):")
    print("-" * 72)
    for c in sorted(DEFAULT_DIRS):
        sub = summary[summary["cache_size"] == c]
        lru = sub[sub["series"] == "LRU baseline"]
        la1 = sub[sub["series"] == "Prefetch Only (best B, LA=1)"]
        best = sub[sub["series"] == "Prefetch Only (best LA & B)"]
        print(f"C={c}:")
        if not lru.empty:
            print(f"  LRU:        TPS={lru.iloc[0]['tps']:.3f}")
        if not la1.empty:
            r = la1.iloc[0]
            print(f"  LA=1 best:  TPS={r['tps']:.3f}  (B={int(r['budget'])})")
        if not best.empty:
            r = best.iloc[0]
            print(f"  Best LA:    TPS={r['tps']:.3f}  (LA={int(r['lookahead'])}, B={int(r['budget'])})")
    print("-" * 72)
    print(f"Summary CSV: {args.csv_out}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        "LRU baseline": ("o-", "#444444"),
        "Prefetch Only (best B, LA=1)": ("s-", "#1f77b4"),
        "Prefetch Only (best LA & B)": ("D-", "#d62728"),
    }
    for series, (fmt, color) in styles.items():
        sub = summary[summary["series"] == series].sort_values("cache_size")
        ax.plot(sub["cache_size"], sub["tps"], fmt, color=color, label=series)

    ax.set_xlabel("Expert cache size (experts per layer)")
    ax.set_ylabel("Tokens per second (TPS)")
    ax.set_title("Prefetch-only vs LRU across cache sizes")
    ax.set_xticks(sorted(DEFAULT_DIRS))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Plot saved: {args.out}")


if __name__ == "__main__":
    main()
