#!/usr/bin/env python3
"""Print a presentation-ready text summary from study CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _to_bool(v: str) -> bool:
    return v in ("True", "true", "1")


def _f(v: str) -> float:
    return float(v)


def load_rows(results_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(results_dir.glob("raw_io_*.csv")):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()
    rows = load_rows(Path(args.results_dir))
    if not rows:
        print("No raw_io_*.csv found.")
        return

    print("=" * 72)
    print("EXPERT LOADING STUDY — KEY FINDINGS")
    print("=" * 72)

    for fmt in sorted(set(r["format"] for r in rows)):
        print(f"\n## {fmt.upper()} format\n")

        intra = [r for r in rows if r["scenario"] == "intra_expert" and r["format"] == fmt]
        if len(intra) == 2:
            seq = next(r for r in intra if not _to_bool(r["parallel_intra"]))
            par = next(r for r in intra if _to_bool(r["parallel_intra"]))
            sp = _f(seq["ms_per_expert"]) / _f(par["ms_per_expert"])
            print("Intra-expert (parallel tensor preads within one expert):")
            print(f"  sequential: {_f(seq['ms_per_expert']):.2f} ms/expert")
            print(f"  parallel:   {_f(par['ms_per_expert']):.2f} ms/expert")
            print(f"  speedup:    {sp:.2f}x")
            if sp > 1.2:
                print("  → Parallel preads within expert HELP (esp. many small files).")
            elif sp < 0.9:
                print("  → Parallel preads within expert HURT (overhead > benefit for single file).")
            else:
                print("  → Intra-expert parallelism is neutral at this expert size.")

        print("\nInter-expert (load N experts sequentially vs in parallel):")
        print(f"  {'N':>3}  {'seq ms':>8}  {'par ms':>8}  {'speedup':>8}  {'par ms/exp':>10}")
        for n in sorted(set(int(r["num_experts"]) for r in rows
                            if r["scenario"] == "inter_expert" and r["format"] == fmt)):
            seq = next(r for r in rows if r["scenario"] == "inter_expert"
                       and r["format"] == fmt and int(r["num_experts"]) == n
                       and not _to_bool(r["parallel_inter"]))
            par = next(r for r in rows if r["scenario"] == "inter_expert"
                       and r["format"] == fmt and int(r["num_experts"]) == n
                       and _to_bool(r["parallel_inter"]))
            sp = _f(seq["avg_ms"]) / _f(par["avg_ms"])
            print(f"  {n:3d}  {_f(seq['avg_ms']):8.1f}  {_f(par['avg_ms']):8.1f}  {sp:8.2f}x  {_f(par['ms_per_expert']):10.2f}")

    print("\n" + "=" * 72)
    print("IMPLICATIONS FOR C++ BACKEND")
    print("=" * 72)
    print("""
Current behavior (unified_llm_w4a16.cpp):
  • WITHIN expert: parallel preads (9 async for packed, 6+3 for unpacked) ✓
  • ACROSS experts: sequential loop in load_predicted_experts() ✗

If inter-expert speedup >> 1x (see packed results above):
  → Big win from parallelizing expert loads across slots, not more packing.

If intra-expert speedup >> 1x only for unpacked:
  → Packing (EXPK) already solves the multi-file open problem.

Set HETEROPREDICT_SEQUENTIAL_EXPERT_IO=1 to A/B test intra-expert parallelism
in the real C++ path via microbench_cpp_prewarm.py.
""")


if __name__ == "__main__":
    main()
