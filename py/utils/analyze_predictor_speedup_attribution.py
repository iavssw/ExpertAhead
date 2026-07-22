#!/usr/bin/env python3
"""Join LRU vs prefetch-only rows and summarize predictor recall/precision + load stats.

Post-processes a ``sweep.csv`` from the ``sec2_predictor_effectiveness`` experiment
(``finals_experiment_runner.py``), backing the section-3 predictor-vs-LRU story. Writes:

- ``predictor_speedup_attribution.csv`` — one row per prefetch config with baseline TPS /
  cache hit rate for the same (cache_size, lookahead, prompt_hash).

Metric glossary (serving decode, not training accuracy):

- **Recall (routed):** ``pred_hit_rate_routed_topk_pct`` if present, else ``pred_hit_rate_pct``.
  Fraction of *router-selected* experts that were already in the predictor’s prefetch set.

- **Precision (requested):** ``pred_requested_rate_topk_pct`` when the model prints it:
  fraction of *prefetched* experts that the router actually asked for on that step.

- **Stalls / prefetch loads:** from the ``Bandwidth: StallLoads=…`` line — fewer stalls with
  higher TPS usually means the predictor hid on-demand expert load latency.

Usage::

  python analyze_predictor_speedup_attribution.py --csv path/to/sweep.csv
"""

from __future__ import annotations

import argparse
import os
import textwrap

import pandas as pd


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--baseline-policy", default="LRU", help="Policy to normalize against (LRU or RANDOM)")
    args = ap.parse_args()

    path = os.path.abspath(args.csv)
    if not os.path.isfile(path):
        print(f"Not found: {path}")
        return 1
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(path)
    lam = _numeric(df, "lambda_val").fillna(0.0)
    df = df.assign(_lam=lam)

    lab = df["label"].astype(str) if "label" in df.columns else pd.Series("", index=df.index)
    if args.baseline_policy.upper() == "RANDOM":
        is_base = (df.get("backend", "").astype(str) == "cached") & lab.str.contains("RANDOM", case=False, na=False) & (df["_lam"] == 0.0)
    else:
        # Default to LRU
        is_base = (df.get("backend", "").astype(str) == "cached") & lab.str.contains("LRU", case=False, na=False) & (df["_lam"] == 0.0)
        
    is_pref = (
        (df.get("backend", "").astype(str) == "predict")
        & lab.str.startswith("Prefetch Only", na=False)
        & (df["_lam"] == 0.0)
    )

    merge_keys = ["cache_size", "lookahead"]
    if "prompt_hash" in df.columns:
        merge_keys.append("prompt_hash")

    base = df.loc[is_base, merge_keys + ["tokens_per_second", "hit_rate_pct"]].copy()
    for c in ("tokens_per_second", "hit_rate_pct"):
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.rename(columns={"tokens_per_second": "base_tps", "hit_rate_pct": "base_cache_hit_pct"})

    pref = df.loc[is_pref].copy()
    if pref.empty:
        print("[attribution] No prefetch-only λ=0 rows found.")
        return 0

    recall = _numeric(pref, "pred_hit_rate_routed_topk_pct")
    if recall.isna().all():
        recall = _numeric(pref, "pred_hit_rate_pct")
    pref = pref.assign(
        pred_recall_pct=recall,
        pred_precision_pct=_numeric(pref, "pred_requested_rate_topk_pct"),
        tps=_numeric(pref, "tokens_per_second"),
        stall_loads=_numeric(pref, "stall_loads"),
        prefetch_loads=_numeric(pref, "prefetch_loads"),
        prefetch_budget=_numeric(pref, "prefetch_budget"),
    )

    merged = pref.merge(base, on=merge_keys, how="left", suffixes=("", "_dup"))
    merged["tps_speedup_vs_base"] = merged["tps"] / merged["base_tps"]

    cols = [
        "question",
        "label",
        "cache_size",
        "lookahead",
        "prefetch_budget",
        "tps",
        "base_tps",
        "tps_speedup_vs_base",
        "base_cache_hit_pct",
        "hit_rate_pct",
        "pred_recall_pct",
        "pred_precision_pct",
        "stall_loads",
        "prefetch_loads",
        "avg_ms_per_expert_load",
        "pred_hits",
        "pred_total",
    ]
    cols = [c for c in cols if c in merged.columns]
    out_csv = os.path.join(out_dir, "predictor_speedup_attribution.csv")
    merged[cols].sort_values(["cache_size", "lookahead", "prefetch_budget"]).to_csv(out_csv, index=False)
    print(f"[attribution] Wrote {out_csv} ({len(merged)} rows)")

    readme = os.path.join(out_dir, "PREDICTOR_METRICS_README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(__doc__).strip() + "\n")
    print(f"[attribution] Wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
