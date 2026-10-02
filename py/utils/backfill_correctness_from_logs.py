#!/usr/bin/env python3
"""Backfill generations.md + correctness.json from existing sweep step logs.

Cold-per-prompt logs contain model stdout only (no sweep banners). We map
generation blocks onto sweep.csv rows in run order:
  01_random.log           → RANDOM row(s)
  02_expert_ahead.log     → Prefetch Only rows (LA ascending)
  03_expert_ahead_cc.log  → Both rows (LA ascending)

Each CSV row corresponds to ``n_examples`` consecutive generation blocks.

Usage:
  python3 py/utils/backfill_correctness_from_logs.py \\
    --root py/utils/final_results_runs/expanded_prompt_space \\
    --timestamp 20260917_225231
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_correctness import (  # noqa: E402
    aggregate_by_label,
    merge_into_sweep_csv,
    parse_generations_md,
    score_examples,
)

GEN_RE = re.compile(r"Generated text only:\n={60}\n(.*?)\n={60}", re.DOTALL)


def extract_gens(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return [g.strip() for g in GEN_RE.findall(text)]


def write_generations_md(
    blocks: list[tuple[str, str, dict]],
    out_path: Path,
) -> int:
    """blocks: (label, gen_text, meta). Prompt index resets per label change."""
    if not blocks:
        return 0
    lines: list[str] = []
    prompt_i = 0
    prev_label = None
    for label, gen, meta in blocks:
        if prev_label is not None and label != prev_label:
            prompt_i = 0
        prev_label = label
        prompt_i += 1
        lines.append(f"### Prompt {prompt_i}\n")
        lines.append(f"**Label:** {label}\n")
        lines.append(
            f"**Config:** Cache={meta.get('cache_size')} | Top-J={meta.get('forced_top_n')} | "
            f"Lam={meta.get('lambda_val')} | T={meta.get('temperature', 0)} | "
            f"Lookahead={meta.get('lookahead')}\n\n"
        )
        lines.append(f"```text\n{gen}\n```\n\n---\n\n")
    out_path.write_text("".join(lines), encoding="utf-8")
    return len(blocks)


def _row_meta(row: dict) -> dict:
    def f(key, default=None):
        v = row.get(key, "")
        if v is None or v == "":
            return default
        try:
            return float(v) if "." in str(v) or key in ("lambda_val", "temperature") else int(float(v))
        except ValueError:
            return default

    return {
        "cache_size": f("cache_size"),
        "forced_top_n": f("forced_top_n", 0),
        "lambda_val": f("lambda_val", 0.0),
        "temperature": f("temperature", 0.0),
        "lookahead": f("lookahead"),
    }


def assign_gens_to_rows(
    rows: list[dict],
    gens: list[str],
    n_examples: int,
) -> list[tuple[str, str, dict]]:
    need = len(rows) * n_examples
    if len(gens) < need:
        print(
            f"[backfill] WARNING: {len(gens)} gens for {len(rows)} rows×{n_examples} "
            f"(need {need}); truncating",
            flush=True,
        )
    out: list[tuple[str, str, dict]] = []
    gi = 0
    for row in rows:
        label = row.get("label", "")
        meta = _row_meta(row)
        for _ in range(n_examples):
            if gi >= len(gens):
                break
            out.append((label, gens[gi], meta))
            gi += 1
    return out


def process_run_dir(run_dir: Path, examples_path: Path) -> None:
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    n_examples = len(examples)
    sweep_path = run_dir / "sweep.csv"
    if not sweep_path.exists():
        print(f"[backfill] no sweep.csv in {run_dir}")
        return

    rows = list(csv.DictReader(sweep_path.open()))
    random_rows = [r for r in rows if "RANDOM" in r.get("label", "")]
    prefetch_rows = [r for r in rows if r.get("label", "").startswith("Prefetch Only")]
    both_rows = [r for r in rows if r.get("label", "").startswith("Both")]
    # Stable LA order
    prefetch_rows.sort(key=lambda r: float(r.get("lookahead") or 0))
    both_rows.sort(key=lambda r: float(r.get("lookahead") or 0))

    mapping = [
        ("01_random.log", random_rows),
        ("02_expert_ahead.log", prefetch_rows),
        ("03_expert_ahead_cc.log", both_rows),
    ]

    all_blocks: list[tuple[str, str, dict]] = []
    for log_name, step_rows in mapping:
        gens = extract_gens(run_dir / log_name)
        print(
            f"[backfill] {log_name}: {len(gens)} gen(s) → {len(step_rows)} row(s) "
            f"× {n_examples} prompts",
            flush=True,
        )
        all_blocks.extend(assign_gens_to_rows(step_rows, gens, n_examples))

    gen_path = run_dir / "generations.md"
    n = write_generations_md(all_blocks, gen_path)
    print(f"[backfill] wrote {n} block(s) → {gen_path}", flush=True)
    if n == 0:
        return

    generations = parse_generations_md(gen_path)
    per_prompt = score_examples(examples, generations)
    aggregates = aggregate_by_label(per_prompt)
    payload = {
        "domain": examples[0]["domain"] if examples else None,
        "metric": examples[0].get("metric") if examples else None,
        "n_examples": n_examples,
        "n_generation_blocks": len(generations),
        "per_prompt": per_prompt,
        "by_label": aggregates,
    }
    corr_path = run_dir / "correctness.json"
    corr_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[backfill] wrote {corr_path}", flush=True)
    merge_into_sweep_csv(sweep_path, aggregates)
    print(f"[backfill] annotated {sweep_path}", flush=True)
    for a in aggregates:
        print(
            f"  {a['label']}: {a['metric']}={a['correctness_mean']:.4f} (n={a['n']})",
            flush=True,
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--examples", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--timestamp", type=str, default="20260917_225231")
    p.add_argument(
        "--domains",
        nargs="+",
        default=["gsm8k", "mbpp", "cnn_dailymail"],  # gold-scorable by default
    )
    p.add_argument("--caches", nargs="+", type=int, default=[24, 32])
    args = p.parse_args()

    if args.run_dir:
        if not args.examples:
            print("--examples required with --run-dir", file=sys.stderr)
            return 1
        process_run_dir(Path(args.run_dir), Path(args.examples))
        return 0

    if not args.root:
        print("Provide --run-dir or --root", file=sys.stderr)
        return 1

    root = Path(args.root)
    split = root / "by_domain"
    for domain in args.domains:
        examples = split / f"{domain}.json"
        if not examples.exists():
            print(f"[backfill] skip {domain}: missing {examples}")
            continue
        for C in args.caches:
            run_dir = root / domain / f"C{C}_{args.timestamp}"
            if not run_dir.exists():
                print(f"[backfill] skip missing {run_dir}")
                continue
            print(f"\n=== {domain} C={C} ===", flush=True)
            process_run_dir(run_dir, examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
