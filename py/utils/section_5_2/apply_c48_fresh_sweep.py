"""Replace all C=48 prefetch rows from a fresh sweep CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SECTION_DIR = Path(__file__).resolve().parent
FRESH_SWEEP_CSV = SECTION_DIR / "fresh_C48_S1to5_2prompt" / "sweep.csv"
RAW_C48 = SECTION_DIR / "sec_5_2_raw_C48.csv"
PREFETCH_ONLY = Path(
    "/home/michael/heteroPredict/py/utils/final_results_runs/"
    "final_sec5_collection/20260707_002948/prefetch_only.csv"
)
PLOT_BUDGETS = [10, 16, 20, 24]


def _prefetch_rows(sweep_csv: Path) -> pd.DataFrame:
    patch = pd.read_csv(sweep_csv)
    patch = patch[
        (patch["backend"] == "predict")
        & patch["label"].str.startswith("Prefetch Only", na=False)
        & (patch["cache_size"] == 48)
    ]
    patch = patch[patch["prefetch_budget"].isin(PLOT_BUDGETS)].copy()
    if patch.empty:
        raise ValueError(f"No rows for budgets {PLOT_BUDGETS} in {sweep_csv}")
    return patch.sort_values(["lookahead", "prefetch_budget"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-csv", type=Path, default=FRESH_SWEEP_CSV)
    args = parser.parse_args()

    patch = _prefetch_rows(args.sweep_csv)
    print(f"Using {args.sweep_csv} ({len(patch)} prefetch rows)")

    for path in (RAW_C48, PREFETCH_ONLY):
        df = pd.read_csv(path)
        other = df[df["cache_size"] != 48]
        out = pd.concat([other, patch], ignore_index=True)
        out = out.sort_values(["cache_size", "lookahead", "prefetch_budget"]).reset_index(drop=True)
        out.to_csv(path, index=False)
        print(f"Updated {path}")

    for _, row in patch.iterrows():
        print(
            f"  S={int(row['lookahead'])} B={int(row['prefetch_budget'])}: "
            f"TPS={row['tokens_per_second']:.3f} "
            f"recall={row['pred_window_recall_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
