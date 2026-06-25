#!/usr/bin/env python3
"""
Orchestrate the full expert-loading study and write results under results/.

Runs:
  1. Raw SSD microbench (packed + unpacked if both dirs exist)
  2. C++ prewarm microbench (optional, requires CUDA + built extension)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

STUDY_DIR = Path(__file__).parent
DEFAULT_PACKED = STUDY_DIR.parent / "unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed"
DEFAULT_UNPACKED = STUDY_DIR.parent / "unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_unpacked"


def _run(cmd: list[str], cwd: Path) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> None:
    p = argparse.ArgumentParser(description="Run full expert loading study")
    p.add_argument("--packed-dir", type=str, default=str(DEFAULT_PACKED))
    p.add_argument("--unpacked-dir", type=str, default=str(DEFAULT_UNPACKED))
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--expert-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--skip-cpp", action="store_true", help="Skip C++ prewarm (no GPU)")
    p.add_argument("--skip-plot", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else STUDY_DIR / "results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        "--num-rounds", str(args.num_rounds),
        "--warmup", str(args.warmup),
        "--out-dir", str(out_dir),
    ]

    for label, bin_dir in [("packed", args.packed_dir), ("unpacked", args.unpacked_dir)]:
        if not Path(bin_dir).is_dir():
            print(f"[skip] {label} dir not found: {bin_dir}")
            continue
        rc = _run(
            [sys.executable, str(STUDY_DIR / "microbench_raw_io.py"),
             "--bin-dir", bin_dir,
             "--expert-counts", *map(str, args.expert_counts),
             *common[1:]],
            STUDY_DIR,
        )
        if rc != 0:
            sys.exit(rc)

    if not args.skip_cpp:
        for label, bin_dir in [("packed", args.packed_dir), ("unpacked", args.unpacked_dir)]:
            if not Path(bin_dir).is_dir():
                continue
            rc = _run(
                [sys.executable, str(STUDY_DIR / "microbench_cpp_prewarm.py"),
                 "--bin-dir", bin_dir,
                 *common[1:]],
                STUDY_DIR,
            )
            if rc != 0:
                print(f"[warn] cpp prewarm failed for {label} (rc={rc})")

    if not args.skip_plot:
        rc = _run(
            [sys.executable, str(STUDY_DIR / "summarize_results.py"), "--results-dir", str(out_dir)],
            STUDY_DIR,
        )
        rc = _run(
            [sys.executable, str(STUDY_DIR / "plot_study.py"), "--results-dir", str(out_dir)],
            STUDY_DIR,
        )
        if rc != 0:
            print(f"[warn] plot_study failed (rc={rc})")

    print(f"\nStudy complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
