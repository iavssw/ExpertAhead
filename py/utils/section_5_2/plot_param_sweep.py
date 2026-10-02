"""
Fig. 5-style predictor parameter sweep plots + best (S, B) per cache size.

Reads sweep_C*.csv from a run_predictor_param_sweep.sh run dir. The baseline for each C
is that CSV's Neither (RANDOM) row. Optionally compares against the paper's
WikiText-predictor sweeps (sec_5_2_raw_C{C}.csv).

Usage:
    python3 py/utils/section_5_2/plot_param_sweep.py --run-dir <out>/<ts>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_sec5_2 import MARKER_CYCLE, plot_cache_panel, save_figure  # noqa: E402

MARKERS = MARKER_CYCLE + ["h", "p", "<", ">", "d"]
NEEDED = ["lookahead", "prefetch_budget", "tokens_per_second",
          "pred_window_recall_pct", "pred_window_precision_pct"]
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))


def load_run(run_dirs: list[str]) -> pd.DataFrame:
    paths = [p for d in run_dirs for p in sorted(glob.glob(os.path.join(d, "sweep_C*.csv")))]
    if not paths:
        raise FileNotFoundError(f"No sweep_C*.csv in {run_dirs}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _row_key(rec: dict) -> tuple | None:
    label = rec.get("label") or ""
    if label.startswith("Neither (RANDOM)"):
        return (int(rec["cache_size"]), "random", None, None)
    if label.startswith("Prefetch Only"):
        return (int(rec["cache_size"]), "prefetch", int(rec["lookahead"]), int(rec["prefetch_budget"]))
    return None


def robust_tps(run_dirs: list[str]) -> dict[tuple, tuple[float, int]]:
    """Per config: mean over prompts of the median TPS across passes (one pass per run dir).

    Returns {(C, kind, S, B): (tps, max passes for any prompt)}. Robust to sporadic slow runs.
    """
    per_pass = []
    for d in run_dirs:
        path = os.path.join(d, "transcripts.jsonl")
        if not os.path.exists(path):
            continue
        vals: dict[tuple, float] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = _row_key(rec)
                tps = rec.get("tokens_per_second")
                if key is not None and tps is not None:
                    vals[key + (int(rec["prompt_ordinal"]),)] = float(tps)
        per_pass.append(vals)
    samples: dict[tuple, list[float]] = {}
    for vals in per_pass:
        for k, v in vals.items():
            samples.setdefault(k, []).append(v)
    by_cfg: dict[tuple, list[tuple[float, int]]] = {}
    for k, vs in samples.items():
        by_cfg.setdefault(k[:4], []).append((statistics.median(vs), len(vs)))
    return {k: (sum(m for m, _ in v) / len(v), max(n for _, n in v)) for k, v in by_cfg.items()}


def split(df: pd.DataFrame, cs: int,
          robust: dict[tuple, tuple[float, int]] | None = None) -> tuple[pd.DataFrame, float | None]:
    sub = df[df["cache_size"] == cs]
    rnd = sub[sub["label"].str.startswith("Neither (RANDOM)", na=False)]
    rnd_tps = pd.to_numeric(rnd["tokens_per_second"], errors="coerce").dropna()
    pf = sub[sub["label"].str.startswith("Prefetch Only", na=False)].copy()
    for col in NEEDED:
        pf[col] = pd.to_numeric(pf[col], errors="coerce")
    pf = pf.dropna(subset=NEEDED)
    for col in ("lookahead", "prefetch_budget"):
        pf[col] = pf[col].astype(int)
    pf = pf.drop_duplicates(subset=["lookahead", "prefetch_budget"], keep="last")
    base = float(rnd_tps.iloc[-1]) if len(rnd_tps) else None
    if robust:
        keys = [(cs, "prefetch", int(s), int(b)) for s, b in zip(pf["lookahead"], pf["prefetch_budget"])]
        pf["tokens_per_second"] = [robust.get(k, (t, 1))[0] for k, t in zip(keys, pf["tokens_per_second"])]
        pf["n_passes"] = [robust.get(k, (0, 1))[1] for k in keys]
        if (cs, "random", None, None) in robust:
            base = robust[(cs, "random", None, None)][0]
    return pf, base


def paper_best(cs: int) -> tuple[pd.Series | None, float | None]:
    path = os.path.join(PAPER_DIR, f"sec_5_2_raw_C{cs}.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    df = df[df["cache_size"] == cs] if "cache_size" in df else df
    pf, base = split(df, cs)
    if pf.empty:
        return None, base
    return pf.loc[pf["tokens_per_second"].idxmax()], base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", nargs="+", required=True,
                    help="One or more run dirs; several dirs = repeated passes, merged by "
                         "per-prompt median TPS")
    ap.add_argument("--out-dir", default=None,
                    help="Default: the run dir (single) or <first run dir>_merged (several)")
    ap.add_argument("--cache-sizes", type=int, nargs="*", default=None)
    ap.add_argument("--strides", type=int, nargs="*", default=None)
    ap.add_argument("--annotate-strides", type=int, nargs="*", default=None,
                    help="Strides to label every budget for (default: best stride per C)")
    args = ap.parse_args()

    run_dirs = [os.path.normpath(d) for d in args.run_dir]
    out_dir = args.out_dir or (run_dirs[0] if len(run_dirs) == 1 else run_dirs[0] + "_merged")
    os.makedirs(out_dir, exist_ok=True)
    args.run_dir = out_dir
    robust = robust_tps(run_dirs) if len(run_dirs) > 1 else None

    df = load_run(run_dirs)
    cache_sizes = sorted(int(c) for c in df["cache_size"].dropna().unique())
    if args.cache_sizes:
        cache_sizes = [c for c in cache_sizes if c in set(args.cache_sizes)]
    panels = []
    for cs in cache_sizes:
        pf, base = split(df, cs, robust)
        if args.strides:
            pf = pf[pf["lookahead"].isin(args.strides)]
        if pf.empty:
            print(f"[C={cs}] no prefetch rows yet")
            continue
        if base is None:
            print(f"[C={cs}] no RANDOM row yet — normalizing to 1.0 TPS")
        panels.append((cs, pf, base or 1.0, base is not None))
    if not panels:
        raise SystemExit("Nothing to plot.")

    all_pf = pd.concat([p for _, p, _, _ in panels])
    lookaheads = sorted(all_pf["lookahead"].unique())
    marker_for = {la: MARKERS[i % len(MARKERS)] for i, la in enumerate(lookaheads)}
    norm = Normalize(vmin=all_pf["pred_window_precision_pct"].min(),
                     vmax=all_pf["pred_window_precision_pct"].max())
    cmap = plt.get_cmap("viridis")

    def draw(ax, cs, pf, base):
        best = pf.loc[pf["tokens_per_second"].idxmax()]
        full = set(args.annotate_strides) if args.annotate_strides else {int(best["lookahead"])}
        plot_cache_panel(ax, cs=cs, sub=pf, lookaheads=lookaheads, marker_for=marker_for,
                         norm=norm, cmap=cmap, baseline=base, annotate_full_strides=full,
                         tps_decimals=2)
        ax.set_title(f"C = {cs}")

    for cs, pf, base, _ in panels:
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        draw(ax, cs, pf, base)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02).set_label("Window Precision (%)")
        save_figure(fig, os.path.join(args.run_dir, f"tps_vs_recall_precision_C{cs}.png"))
        plt.close(fig)

    ncols = 2 if len(panels) > 1 else 1
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5.5 * nrows),
                             squeeze=False, constrained_layout=True)
    flat = axes.ravel()
    for ax, (cs, pf, base, _) in zip(flat, panels):
        draw(ax, cs, pf, base)
    for ax in flat[len(panels):]:
        ax.axis("off")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=flat.tolist(), shrink=0.85, pad=0.02).set_label("Window Precision (%)")
    save_figure(fig, os.path.join(args.run_dir, "tps_vs_recall_precision_all.png"))
    plt.close(fig)

    lines = ["# Predictor parameter sweep — best (S, B) per cache size", "",
             "| C | RANDOM tok/s | Best S | Best B | tok/s | Speedup | Recall ρ | Precision π "
             "| Paper (WikiText pred.) best | Paper speedup |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    per_stride = []
    best_rows = []
    for cs, pf, base, has_base in panels:
        b = pf.loc[pf["tokens_per_second"].idxmax()]
        pb, pbase = paper_best(cs)
        paper_cell = (f"S={int(pb['lookahead'])} B={int(pb['prefetch_budget'])} "
                      f"({pb['tokens_per_second']:.2f} tok/s)") if pb is not None else "—"
        paper_sp = (f"{pb['tokens_per_second'] / pbase:.2f}×"
                    if pb is not None and pbase else "—")
        sp = f"{b['tokens_per_second'] / base:.2f}×" if has_base else "—"
        lines.append(
            f"| {cs} | {base:.2f} | {int(b['lookahead'])} | {int(b['prefetch_budget'])} | "
            f"{b['tokens_per_second']:.2f} | {sp} | {b['pred_window_recall_pct']:.1f}% | "
            f"{b['pred_window_precision_pct']:.1f}% | {paper_cell} | {paper_sp} |"
            if has_base else
            f"| {cs} | — | {int(b['lookahead'])} | {int(b['prefetch_budget'])} | "
            f"{b['tokens_per_second']:.2f} | — | {b['pred_window_recall_pct']:.1f}% | "
            f"{b['pred_window_precision_pct']:.1f}% | {paper_cell} | {paper_sp} |"
        )
        best_rows.append({"cache_size": cs, "lookahead": int(b["lookahead"]),
                          "prefetch_budget": int(b["prefetch_budget"]),
                          "tokens_per_second": b["tokens_per_second"],
                          "random_tps": base if has_base else None,
                          "speedup": b["tokens_per_second"] / base if has_base else None,
                          "window_recall_pct": b["pred_window_recall_pct"],
                          "window_precision_pct": b["pred_window_precision_pct"]})
        per_stride += ["", f"### C = {cs}: best budget per stride", "",
                       "| S | B | tok/s | Speedup | Recall ρ | Precision π | Passes |",
                       "|---:|---:|---:|---:|---:|---:|---:|"]
        for s in sorted(pf["lookahead"].unique()):
            g = pf[pf["lookahead"] == s]
            r = g.loc[g["tokens_per_second"].idxmax()]
            per_stride.append(
                f"| {s} | {int(r['prefetch_budget'])} | {r['tokens_per_second']:.2f} | "
                f"{(r['tokens_per_second'] / base):.2f}× | {r['pred_window_recall_pct']:.1f}% | "
                f"{r['pred_window_precision_pct']:.1f}% | {int(r.get('n_passes', 1))} |")
    header = []
    if robust:
        header = [f"Merged {len(run_dirs)} passes ({', '.join(os.path.basename(d) for d in run_dirs)}): "
                  "TPS = mean over prompts of the per-prompt median across passes.", ""]
    md = "\n".join(header + lines + per_stride) + "\n"
    with open(os.path.join(args.run_dir, "best_configs.md"), "w", encoding="utf-8") as f:
        f.write(md)
    pd.DataFrame(best_rows).to_csv(os.path.join(args.run_dir, "best_configs.csv"), index=False)
    print(md)


if __name__ == "__main__":
    main()
