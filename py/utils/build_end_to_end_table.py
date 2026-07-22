#!/usr/bin/env python3
"""Build Tab. end_to_end_results from collected sweep CSVs.

Inputs (10 prompts, 150 tokens, matched prompt_hash):
  - SSD Streaming: FORCE_EXPERT_MISS sweep (one TPS reused for all C)
  - Oracle Predictor: final_table_exact_150/C{C}_*/sweep.csv
  - Other methods: final_10prompt_oracle_150/C{C}_*/sweep.csv

Example:
  python3 py/utils/build_end_to_end_table.py \\
    --streaming-csv py/utils/final_results_runs/final_table_exact_150/streaming_20260715_150635/sweep.csv \\
    --oracle-root py/utils/final_results_runs/final_table_exact_150 \\
    --oracle-suffix 20260715_150635 \\
    --methods-root py/utils/final_results_runs/final_10prompt_oracle_150 \\
    --methods-suffix 20260715_162440 \\
    --out-dir py/utils/final_results_runs/final_10prompt_oracle_150
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

CACHE_SIZES = [16, 32, 48, 64]

METHOD_ORDER = [
    "SSD Streaming",
    "Random Eviction",
    "Oracle Predictor",
    "Cross-Layer Prefetching",
    "Cache-Conditional Routing",
    "ExpertAhead",
    "ExpertAhead-CC",
]

# Display names in the paper LaTeX (citations on middle rows).
LATEX_METHOD = {
    "SSD Streaming": "SSD Streaming",
    "Random Eviction": "Random Eviction",
    "Oracle Predictor": "Oracle Predictor",
    "Cross-Layer Prefetching": r"Cross-Layer ~\cite{eliseev2023fastinferencemixtureofexpertslanguage}",
    "Cache-Conditional Routing": r"Cache-Conditional~\cite{skliar2025mixturecacheconditionalexpertsefficient}",
    "ExpertAhead": r"\textbf{ExpertAhead}",
    "ExpertAhead-CC": r"\textbf{ExpertAhead-CC}",
}

BOLD_METHODS = {"ExpertAhead", "ExpertAhead-CC"}


def _num(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt_sbj(s: Optional[float], b: Optional[float], j: Optional[float], kind: str) -> str:
    def part(v: Optional[float]) -> str:
        if v is None:
            return "--"
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:g}"

    if kind == "ssd":
        return "--"
    if kind == "random":
        return "--/--/--"
    if kind == "oracle":
        return f"{part(s)}/--/--"
    if kind == "cross":
        return f"--/{part(b)}/--"
    if kind == "cc":
        return f"--/--/{part(j)}"
    if kind == "ea":
        return f"{part(s)}/{part(b)}/--"
    if kind == "eacc":
        return f"{part(s)}/{part(b)}/{part(j)}"
    return "--/--/--"


def _pick_row(df: pd.DataFrame, pred) -> pd.Series:
    sub = df[pred(df)].copy()
    if sub.empty:
        raise ValueError("no matching row")
    # Prefer rows with measured TPS.
    sub = sub[sub["tokens_per_second"].notna()]
    if sub.empty:
        raise ValueError("matching rows lack tokens_per_second")
    return sub.iloc[0]


def _relpath(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def load_methods_csv(path: str) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(path)
    out: Dict[str, Dict[str, Any]] = {}

    def add(method: str, kind: str, row: pd.Series) -> None:
        out[method] = {
            "tps": float(row["tokens_per_second"]),
            "S": _num(row.get("lookahead")),
            "B": _num(row.get("prefetch_budget")),
            "J": _num(row.get("forced_top_n")) if kind in ("cc", "eacc") else None,
            "kind": kind,
            "prompt_hash": str(row.get("prompt_hash", "")),
            "source": path,
        }
        if kind == "cross":
            out[method]["S"] = None
            out[method]["J"] = None
        if kind == "cc":
            out[method]["S"] = None
            out[method]["B"] = None
        if kind == "random":
            out[method]["S"] = out[method]["B"] = out[method]["J"] = None
        if kind == "ea":
            out[method]["J"] = None

    add(
        "Random Eviction",
        "random",
        _pick_row(df, lambda d: d["label"].astype(str).str.startswith("Neither (RANDOM)")),
    )
    add(
        "Cross-Layer Prefetching",
        "cross",
        _pick_row(df, lambda d: d["label"].astype(str).str.startswith("Gating")),
    )
    add(
        "Cache-Conditional Routing",
        "cc",
        _pick_row(df, lambda d: d["label"].astype(str).str.startswith("Cache-Cond Only")),
    )
    add(
        "ExpertAhead",
        "ea",
        _pick_row(df, lambda d: d["label"].astype(str).str.startswith("Prefetch Only")),
    )
    add(
        "ExpertAhead-CC",
        "eacc",
        _pick_row(df, lambda d: d["label"].astype(str).str.startswith("Both")),
    )
    return out


def load_oracle_csv(path: str) -> Dict[str, Any]:
    df = pd.read_csv(path)
    row = _pick_row(df, lambda d: d["label"].astype(str).str.contains("Oracle Full Union", na=False))
    return {
        "tps": float(row["tokens_per_second"]),
        "S": _num(row.get("lookahead")),
        "B": None,
        "J": None,
        "kind": "oracle",
        "prompt_hash": str(row.get("prompt_hash", "")),
        "source": path,
    }


def load_streaming_tps(path: str) -> Tuple[float, str, str]:
    df = pd.read_csv(path)
    row = _pick_row(
        df,
        lambda d: d["label"].astype(str).str.contains("Forced Miss|Neither \\(RANDOM\\)", regex=True, na=False),
    )
    return float(row["tokens_per_second"]), str(row.get("prompt_hash", "")), path


def build_rows(
    *,
    streaming_csv: str,
    oracle_root: str,
    oracle_suffix: str,
    methods_root: str,
    methods_suffix: str,
    cache_sizes: List[int],
    num_prompts: int,
    max_new_tokens: int,
    repo_root: str,
) -> pd.DataFrame:
    ssd_tps, ssd_hash, ssd_src = load_streaming_tps(streaming_csv)
    rows: List[Dict[str, Any]] = []

    for c in cache_sizes:
        methods_csv = os.path.join(methods_root, f"C{c}_{methods_suffix}", "sweep.csv")
        oracle_csv = os.path.join(oracle_root, f"C{c}_{oracle_suffix}", "sweep.csv")
        methods = load_methods_csv(methods_csv)
        oracle = load_oracle_csv(oracle_csv)
        random_tps = methods["Random Eviction"]["tps"]

        block: Dict[str, Dict[str, Any]] = {
            "SSD Streaming": {
                "tps": ssd_tps,
                "S": None,
                "B": None,
                "J": None,
                "kind": "ssd",
                "prompt_hash": ssd_hash,
                "source": ssd_src,
            },
            **methods,
            "Oracle Predictor": oracle,
        }

        for method in METHOD_ORDER:
            info = block[method]
            tps = info["tps"]
            rows.append(
                {
                    "cache_size": c,
                    "method": method,
                    "S_B_J": _fmt_sbj(info["S"], info["B"], info["J"], info["kind"]),
                    "S": info["S"],
                    "B": info["B"],
                    "J": info["J"],
                    "tps": tps,
                    "speedup_vs_ssd": tps / ssd_tps,
                    "speedup_vs_random": tps / random_tps,
                    "ssd_baseline_tps": ssd_tps,
                    "random_tps": random_tps,
                    "num_prompts": num_prompts,
                    "max_new_tokens": max_new_tokens,
                    "prompt_hash": info.get("prompt_hash", ""),
                    "source_dir": _relpath(os.path.dirname(info["source"]), os.path.join(repo_root, "py/utils/final_results_runs")),
                }
            )
    return pd.DataFrame(rows)


def wide_speedups(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method in METHOD_ORDER:
        rec: Dict[str, Any] = {"method": method}
        for c in sorted(df["cache_size"].unique()):
            row = df[(df["method"] == method) & (df["cache_size"] == c)].iloc[0]
            rec[f"C{c}_S_B_J"] = row["S_B_J"]
            rec[f"C{c}_tps"] = row["tps"]
            rec[f"C{c}_ssd"] = row["speedup_vs_ssd"]
            rec[f"C{c}_rand"] = row["speedup_vs_random"]
        records.append(rec)
    return pd.DataFrame(records)


def _fmt_tps(x: float) -> str:
    return f"{x:.2f}"


def _fmt_speedup(x: float, bold: bool) -> str:
    s = f"{x:.2f}$\\times$"
    return f"\\textbf{{{s}}}" if bold else s


def render_latex(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append("")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccccccccccccc}")
    lines.append(r"\toprule")
    lines.append("")
    lines.append(r"&")
    lines.append(r"\multicolumn{4}{c}{$\mathbf{C=16}$} &")
    lines.append(r"\multicolumn{4}{c}{$\mathbf{C=32}$} &")
    lines.append(r"\multicolumn{4}{c}{$\mathbf{C=48}$} &")
    lines.append(r"\multicolumn{4}{c}{$\mathbf{C=64}$} \\")
    lines.append("")
    lines.append(r"\cmidrule(lr){2-5}")
    lines.append(r"\cmidrule(lr){6-9}")
    lines.append(r"\cmidrule(lr){10-13}")
    lines.append(r"\cmidrule(lr){14-17}")
    lines.append("")
    lines.append(r"\textbf{Method}")
    for _ in range(4):
        lines.append(r"&")
        lines.append(r"\textbf{$S/B/J$} &")
        lines.append(r"\textbf{TPS} &")
        lines.append(r"\textbf{SSD} &")
        lines.append(r"\textbf{Rand}")
    lines.append(r"\\")
    lines.append("")
    lines.append(r"\midrule")
    lines.append("")

    def cells(method: str) -> str:
        bold = method in BOLD_METHODS
        parts = []
        for c in CACHE_SIZES:
            row = df[(df["method"] == method) & (df["cache_size"] == c)].iloc[0]
            parts.append(
                f"{row['S_B_J']} & {_fmt_tps(row['tps'])} & "
                f"{_fmt_speedup(row['speedup_vs_ssd'], bold)} & "
                f"{_fmt_speedup(row['speedup_vs_random'], bold)}"
            )
        return "\n&\n".join(parts)

    mid_breaks = {
        "Oracle Predictor",
        "Cache-Conditional Routing",
    }
    for method in METHOD_ORDER:
        lines.append(LATEX_METHOD[method])
        lines.append("&")
        lines.append(cells(method))
        lines.append(r"\\")
        lines.append("")
        if method in mid_breaks:
            lines.append(r"\midrule")
            lines.append("")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{End-to-end throughput across different expert cache sizes ($C$). "
        r"Speedups are reported relative to both the SSD Streaming baseline and the "
        r"Random Eviction baseline. Cross-Layer  reproduces~\cite{eliseev2023fastinferencemixtureofexpertslanguage} "
        r"and Cache-Conditional  reproduces~\cite{skliar2025mixturecacheconditionalexpertsefficient}. "
        r"Oracle Predictor represents the theoretical upper bound with best-case perfect prediction. "
        r'"-" is N/A. (10 prompts, 150 generated tokens each)}'
    )
    lines.append(r"\label{tab:end_to_end_results}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--streaming-csv", required=True)
    p.add_argument("--oracle-root", required=True)
    p.add_argument("--oracle-suffix", required=True, help="Timestamp suffix, e.g. 20260715_150635")
    p.add_argument("--methods-root", required=True)
    p.add_argument("--methods-suffix", required=True, help="Timestamp suffix, e.g. 20260715_162440")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-sizes", type=int, nargs="+", default=CACHE_SIZES)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=150)
    args = p.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.makedirs(args.out_dir, exist_ok=True)

    df = build_rows(
        streaming_csv=args.streaming_csv,
        oracle_root=args.oracle_root,
        oracle_suffix=args.oracle_suffix,
        methods_root=args.methods_root,
        methods_suffix=args.methods_suffix,
        cache_sizes=list(args.cache_sizes),
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
        repo_root=repo_root,
    )
    wide = wide_speedups(df)
    tex = render_latex(df)

    combined = os.path.join(args.out_dir, "end_to_end_table_combined.csv")
    wide_path = os.path.join(args.out_dir, "end_to_end_table_speedups_wide.csv")
    tex_path = os.path.join(args.out_dir, "end_to_end_table.tex")
    df.to_csv(combined, index=False)
    wide.to_csv(wide_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Wrote {combined}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
