"""Append C=48 S=5 prefetch rows from sweep CSV into plot data files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SECTION_DIR = Path(__file__).resolve().parent
S5_RERUN_CSV = SECTION_DIR / "rerun_C48_S5_all" / "sweep.csv"
LEGACY_SWEEP_CSV = SECTION_DIR / "C48_S5_prefetch" / "sweep.csv"
RAW_C48 = SECTION_DIR / "sec_5_2_raw_C48.csv"
PREFETCH_ONLY = Path(
    "/home/michael/heteroPredict/py/utils/final_results_runs/"
    "final_sec5_collection/20260707_002948/prefetch_only.csv"
)
PATCH_KEYS = [(48, 5, 10), (48, 5, 16), (48, 5, 20), (48, 5, 24), (48, 5, 32)]


def default_sweep_csv() -> Path:
    if S5_RERUN_CSV.exists():
        return S5_RERUN_CSV
    return LEGACY_SWEEP_CSV


def _prefetch_rows(sweep_csv: Path) -> pd.DataFrame:
    patch = pd.read_csv(sweep_csv)
    patch = patch[
        (patch["backend"] == "predict")
        & patch["label"].str.startswith("Prefetch Only", na=False)
        & (patch["cache_size"] == 48)
        & (patch["lookahead"] == 5)
    ]
    if patch.empty:
        raise ValueError(f"No C=48 S=5 prefetch rows in {sweep_csv}")
    return patch


def _upsert_rows(target: pd.DataFrame, patch: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    for cache_size, stride, budget in PATCH_KEYS:
        new_rows = patch[
            (patch["cache_size"] == cache_size)
            & (patch["lookahead"] == stride)
            & (patch["prefetch_budget"] == budget)
        ]
        if new_rows.empty:
            raise ValueError(f"Missing sweep row for C={cache_size} S={stride} B={budget}")
        mask = (
            (out["cache_size"] == cache_size)
            & (out["lookahead"] == stride)
            & (out["prefetch_budget"] == budget)
        )
        out = out[~mask]
        out = pd.concat([out, new_rows], ignore_index=True)
    sort_cols = ["cache_size", "lookahead", "prefetch_budget"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-csv", type=Path, default=None)
    args = parser.parse_args()

    sweep_csv = args.sweep_csv or default_sweep_csv()
    patch = _prefetch_rows(sweep_csv)
    print(f"Using {sweep_csv}")

    for path in (RAW_C48, PREFETCH_ONLY):
        df = pd.read_csv(path)
        updated = _upsert_rows(df, patch)
        updated.to_csv(path, index=False)
        print(f"Updated {path}")

    for cache_size, stride, budget in PATCH_KEYS:
        row = patch[
            (patch["cache_size"] == cache_size)
            & (patch["lookahead"] == stride)
            & (patch["prefetch_budget"] == budget)
        ].iloc[0]
        print(
            f"  S={int(row['lookahead'])} B={int(row['prefetch_budget'])}: "
            f"TPS={row['tokens_per_second']:.3f} "
            f"recall={row['pred_window_recall_pct']:.1f}% "
            f"prec={row['pred_window_precision_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
