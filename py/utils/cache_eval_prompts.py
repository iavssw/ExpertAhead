#!/usr/bin/env python3
"""Cache HF eval examples (text + gold + metric) for expanded_prompt_space sweeps.

Writes a JSON list of objects:
  {domain, idx, text, gold, metric}

Use ``--split-dir`` to also write one JSON file per domain for per-domain runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from prompt_datasets import DATASET_NAMES, load_eval_examples  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output examples.json path")
    p.add_argument("--num-prompts", type=int, default=5, help="Examples per HF source")
    p.add_argument("--prompt-max-chars", type=int, default=4096)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_NAMES),
        help="HF sources to include (default: all catalog names)",
    )
    p.add_argument(
        "--split-dir",
        default=None,
        help="If set, also write <domain>.json under this directory",
    )
    args = p.parse_args()

    examples = []
    for name in args.datasets:
        print(f"[cache_eval_prompts] loading {name} …", flush=True)
        part = load_eval_examples(
            name,
            args.num_prompts,
            max_chars=args.prompt_max_chars,
        )
        if not part:
            print(f"[cache_eval_prompts] WARNING: {name} returned 0 examples", flush=True)
            continue
        examples.extend(part)
        print(f"[cache_eval_prompts] {name}: {len(part)}", flush=True)

    if not examples:
        print("[cache_eval_prompts] ERROR: no examples collected", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[cache_eval_prompts] wrote {len(examples)} examples → {args.out}", flush=True)

    if args.split_dir:
        os.makedirs(args.split_dir, exist_ok=True)
        by_domain: dict = {}
        for ex in examples:
            by_domain.setdefault(ex["domain"], []).append(ex)
        for domain, rows in by_domain.items():
            # Re-index within domain for stable prompt ordinals in generations.md
            for i, row in enumerate(rows):
                row["idx"] = i
            path = os.path.join(args.split_dir, f"{domain}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"[cache_eval_prompts] {domain}: {len(rows)} → {path}", flush=True)

    # datasets/pyarrow can abort during interpreter teardown after a successful run
    # ("terminate called without an active exception"). Force a clean exit.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
