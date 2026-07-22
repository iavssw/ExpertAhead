"""Merge C64 comparison + refine sweep CSVs into one plot-ready file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_COMPARISON = (
    "/home/michael/heteroPredict/py/utils/final_results_runs/"
    "C64_comparison_20260706_202859/sweep.csv"
)
DEFAULT_REFINE = (
    "/home/michael/heteroPredict/py/utils/final_results_runs/"
    "C64_refine_20260706_221747/sweep.csv"
)
DEFAULT_OUTPUT = (
    "/home/michael/heteroPredict/py/utils/section_5_2/sec_5_2_raw_C64.csv"
)


def merge_sweeps(comparison_path: str, refine_path: str) -> pd.DataFrame:
    df_cmp = pd.read_csv(comparison_path)
    df_ref = pd.read_csv(refine_path)

    pred_cmp = df_cmp[df_cmp["backend"] == "predict"].copy()
    pred_ref = df_ref[df_ref["backend"] == "predict"].copy()
    pred_cmp["_merge_order"] = 0
    pred_ref["_merge_order"] = 1

    predict = pd.concat([pred_cmp, pred_ref], ignore_index=True)
    predict["lookahead"] = predict["lookahead"].astype(int)
    predict["prefetch_budget"] = predict["prefetch_budget"].astype(int)
    predict = (
        predict.sort_values(["lookahead", "prefetch_budget", "_merge_order"])
        .drop_duplicates(["lookahead", "prefetch_budget"], keep="last")
        .drop(columns="_merge_order")
    )

    baselines = df_cmp[df_cmp["backend"] == "cached"].copy()
    # Keep one RANDOM/LRU pair (comparison run, S=1).
    baselines = baselines[baselines["lookahead"] == 1]

    return pd.concat([predict, baselines], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    parser.add_argument("--refine", default=DEFAULT_REFINE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    merged = merge_sweeps(args.comparison, args.refine)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    n_predict = (merged["backend"] == "predict").sum()
    print(f"Wrote {out} ({n_predict} predict rows)")


if __name__ == "__main__":
    main()
