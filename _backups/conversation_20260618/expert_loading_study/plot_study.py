#!/usr/bin/env python3
"""Generate presentation plots from expert_loading_study CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _read_csvs(results_dir: Path) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = {"raw_io": [], "cpp_prewarm": []}
    for path in sorted(results_dir.glob("*.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        if path.name.startswith("raw_io"):
            buckets["raw_io"].extend(rows)
        elif path.name.startswith("cpp_prewarm"):
            buckets["cpp_prewarm"].extend(rows)
    return buckets


def _to_bool(v: str) -> bool:
    return v in ("True", "true", "1", "yes")


def _to_float(v: str) -> float:
    return float(v) if v not in ("", None) else 0.0


def plot_intra_expert(raw_rows: List[dict], out_dir: Path) -> None:
    by_fmt: Dict[str, List[dict]] = {}
    for r in raw_rows:
        if r.get("scenario") != "intra_expert":
            continue
        by_fmt.setdefault(r["format"], []).append(r)

    if not by_fmt:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    formats = sorted(by_fmt)
    x = np.arange(len(formats))
    width = 0.35

    seq_vals, par_vals = [], []
    for fmt in formats:
        rows = by_fmt[fmt]
        seq = next(r for r in rows if not _to_bool(r["parallel_intra"]))
        par = next(r for r in rows if _to_bool(r["parallel_intra"]))
        seq_vals.append(_to_float(seq["ms_per_expert"]))
        par_vals.append(_to_float(par["ms_per_expert"]))

    ax.bar(x - width / 2, seq_vals, width, label="sequential tensor reads", color="#c44e52")
    ax.bar(x + width / 2, par_vals, width, label="parallel tensor reads (C++ default)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(formats)
    ax.set_ylabel("ms / expert (SSD only)")
    ax.set_title("Intra-expert I/O: packing vs parallel preads")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "intra_expert_packing_vs_parallel.png", dpi=150)
    plt.close(fig)


def plot_inter_expert(raw_rows: List[dict], out_dir: Path) -> None:
    by_fmt: Dict[str, Dict[int, Dict[bool, dict]]] = {}
    for r in raw_rows:
        if r.get("scenario") != "inter_expert":
            continue
        fmt = r["format"]
        n = int(r["num_experts"])
        inter = _to_bool(r["parallel_inter"])
        by_fmt.setdefault(fmt, {}).setdefault(n, {})[inter] = r

    if not by_fmt:
        return

    fig, axes = plt.subplots(1, len(by_fmt), figsize=(6 * len(by_fmt), 4), squeeze=False)
    for ax, (fmt, by_n) in zip(axes[0], sorted(by_fmt.items())):
        ns = sorted(by_n)
        seq = [_to_float(by_n[n][False]["avg_ms"]) for n in ns]
        par = [_to_float(by_n[n][True]["avg_ms"]) for n in ns]

        ax.plot(ns, seq, "o-", label="sequential experts", color="#c44e52")
        ax.plot(ns, par, "s-", label="parallel experts", color="#4c72b0")
        ax.set_xlabel("experts loaded per batch")
        ax.set_ylabel("total ms (SSD only)")
        ax.set_title(f"Inter-expert parallelism ({fmt})")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "inter_expert_parallelism.png", dpi=150)
    plt.close(fig)


def plot_inter_speedup(raw_rows: List[dict], out_dir: Path) -> None:
    """Speedup of parallel vs sequential expert loads, by N."""
    data: Dict[str, Dict[int, float]] = {}
    for fmt in set(r["format"] for r in raw_rows if r.get("scenario") == "inter_expert"):
        for n in set(int(r["num_experts"]) for r in raw_rows
                     if r.get("scenario") == "inter_expert" and r["format"] == fmt):
            seq = next((r for r in raw_rows if r["scenario"] == "inter_expert"
                        and r["format"] == fmt and int(r["num_experts"]) == n
                        and not _to_bool(r["parallel_inter"])), None)
            par = next((r for r in raw_rows if r["scenario"] == "inter_expert"
                        and r["format"] == fmt and int(r["num_experts"]) == n
                        and _to_bool(r["parallel_inter"])), None)
            if seq and par:
                data.setdefault(fmt, {})[n] = _to_float(seq["avg_ms"]) / _to_float(par["avg_ms"])

    if not data:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for fmt, by_n in sorted(data.items()):
        ns = sorted(by_n)
        ax.plot(ns, [by_n[n] for n in ns], "o-", label=fmt)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("experts per batch")
    ax.set_ylabel("speedup (sequential / parallel)")
    ax.set_title("Inter-expert parallel speedup (SSD queue depth)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "inter_expert_speedup.png", dpi=150)
    plt.close(fig)


def plot_cpp_prewarm(cpp_rows: List[dict], out_dir: Path) -> None:
    if not cpp_rows:
        return
    by_fmt: Dict[str, List[dict]] = {}
    for r in cpp_rows:
        by_fmt.setdefault(r["format"], []).append(r)

    fig, ax = plt.subplots(figsize=(7, 4))
    formats = sorted(by_fmt)
    x = np.arange(len(formats))
    width = 0.35
    seq_vals, par_vals = [], []
    for fmt in formats:
        rows = by_fmt[fmt]
        seq = next(r for r in rows if _to_bool(str(r.get("sequential_intra", ""))))
        par = next(r for r in rows if not _to_bool(str(r.get("sequential_intra", ""))))
        seq_vals.append(_to_float(seq["ms_per_expert"]))
        par_vals.append(_to_float(par["ms_per_expert"]))

    ax.bar(x - width / 2, seq_vals, width, label="sequential intra (env=1)", color="#c44e52")
    ax.bar(x + width / 2, par_vals, width, label="parallel intra (default)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(formats)
    ax.set_ylabel("ms / expert (SSD + H2D)")
    ax.set_title("C++ prewarm: intra-expert parallel preads")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cpp_prewarm_intra_parallel.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()
    results_dir = Path(args.results_dir)
    buckets = _read_csvs(results_dir)

    if not HAS_MPL:
        print("matplotlib not installed — skipping plots. Install with: pip install matplotlib")
        return

    plot_intra_expert(buckets["raw_io"], results_dir)
    plot_inter_expert(buckets["raw_io"], results_dir)
    plot_inter_speedup(buckets["raw_io"], results_dir)
    plot_cpp_prewarm(buckets["cpp_prewarm"], results_dir)
    print(f"Plots written to {results_dir}")


if __name__ == "__main__":
    main()
