"""Patch C=48 rerun rows into plot data files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_C48 = Path(__file__).with_name("sec_5_2_raw_C48.csv")
PREFETCH_ONLY = Path(
    "/home/michael/heteroPredict/py/utils/final_results_runs/"
    "final_sec5_collection/20260707_002948/prefetch_only.csv"
)
SECTION_DIR = Path(__file__).resolve().parent
RERUN_PATCHES: list[tuple[Path, list[tuple[int, int, int]]]] = [
    (
        SECTION_DIR / "rerun_C48_S2S3_B16" / "sweep.csv",
        [(48, 2, 16), (48, 3, 16)],
    ),
    (
        SECTION_DIR / "rerun_C48_S1_B10" / "sweep.csv",
        [(48, 1, 10)],
    ),
]


def _replace_rows(
    target: pd.DataFrame,
    patch: pd.DataFrame,
    patch_keys: list[tuple[int, int, int]],
) -> pd.DataFrame:
    out = target.copy()
    for cache_size, stride, budget in patch_keys:
        new_rows = patch[
            (patch["cache_size"] == cache_size)
            & (patch["lookahead"] == stride)
            & (patch["prefetch_budget"] == budget)
        ]
        if new_rows.empty:
            raise ValueError(f"Missing rerun row for C={cache_size} S={stride} B={budget}")
        mask = (
            (out["cache_size"] == cache_size)
            & (out["lookahead"] == stride)
            & (out["prefetch_budget"] == budget)
        )
        out = out[~mask]
        out = pd.concat([out, new_rows], ignore_index=True)
    sort_cols = ["cache_size", "lookahead", "prefetch_budget"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _prefetch_rows(rerun_csv: Path) -> pd.DataFrame:
    patch = pd.read_csv(rerun_csv)
    patch = patch[
        (patch["backend"] == "predict")
        & patch["label"].str.startswith("Prefetch Only", na=False)
    ]
    if patch.empty:
        raise ValueError(f"No prefetch rows in {rerun_csv}")
    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun-csv",
        type=Path,
        action="append",
        dest="rerun_csvs",
        help="Optional rerun CSV (default: all known reruns that exist on disk)",
    )
    args = parser.parse_args()

    if args.rerun_csvs:
        sources = [(path, keys) for path, keys in RERUN_PATCHES if path in args.rerun_csvs]
        missing = set(args.rerun_csvs) - {path for path, _ in sources}
        if missing:
            raise ValueError(f"Unknown rerun CSV(s): {missing}")
    else:
        sources = [(path, keys) for path, keys in RERUN_PATCHES if path.exists()]

    if not sources:
        print("No rerun CSVs found; nothing to patch.")
        return

    for path in (RAW_C48, PREFETCH_ONLY):
        df = pd.read_csv(path)
        for rerun_csv, patch_keys in sources:
            patch = _prefetch_rows(rerun_csv)
            df = _replace_rows(df, patch, patch_keys)
        df.to_csv(path, index=False)
        print(f"Patched {path}")

    for rerun_csv, patch_keys in sources:
        patch = _prefetch_rows(rerun_csv)
        for cache_size, stride, budget in patch_keys:
            row = patch[
                (patch["cache_size"] == cache_size)
                & (patch["lookahead"] == stride)
                & (patch["prefetch_budget"] == budget)
            ].iloc[0]
            print(
                f"  S={int(row['lookahead'])} B={int(row['prefetch_budget'])}: "
                f"TPS={row['tokens_per_second']:.3f} "
                f"recall={row['pred_window_recall_pct']:.1f}%"
            )


if __name__ == "__main__":
    main()
