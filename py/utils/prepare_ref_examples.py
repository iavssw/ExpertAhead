#!/usr/bin/env python3
"""Build per-domain examples with a ``ref_text`` for teacher-forced reference perplexity.

- orca / gsm8k / cnn_dailymail: ``ref_text`` is the gold response / solution / highlights.
- wikitext / fineweb (no gold): the tail of each document is held out as ``ref_text``
  and removed from the prompt, split at a whitespace boundary.

Usage:
  python3 py/utils/prepare_ref_examples.py \\
    --src-dir py/utils/final_results_runs/expanded_prompt_space/by_domain \\
    --out-dir py/utils/final_results_runs/quality_speed/by_domain
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

METRIC_BY_DOMAIN = {
    "wikitext": "ref_ppl",
    "fineweb": "ref_ppl",
    "orca": "rouge_l",
    "gsm8k": "exact_match",
    "cnn_dailymail": "rouge_l",
}


def split_held_out_tail(text: str, ref_chars: int, max_fraction: float) -> tuple[str, str]:
    """Return (prompt, ref) with ref ≈ min(ref_chars, max_fraction·len) chars, cut at whitespace."""
    target = min(ref_chars, int(len(text) * max_fraction))
    if target <= 0:
        return text, ""
    cut = len(text) - target
    while cut < len(text) and not text[cut].isspace():
        cut += 1
    if cut >= len(text):
        return text, ""
    return text[:cut].rstrip(), text[cut:].rstrip()


def build(examples: List[Dict[str, Any]], domain: str, prompt_max_chars: int,
          ref_chars: int, max_fraction: float) -> List[Dict[str, Any]]:
    out = []
    for ex in examples:
        text = (ex.get("text") or "")[:prompt_max_chars]
        gold = ex.get("gold")
        rec = dict(ex)
        rec["metric"] = METRIC_BY_DOMAIN.get(domain, ex.get("metric"))
        if isinstance(gold, str) and gold.strip():
            ref = re.sub(r"<<[^>]*>>", "", gold) if domain == "gsm8k" else gold
            rec["text"] = text
            rec["ref_text"] = ref if ref[:1].isspace() else " " + ref
        else:
            prompt, ref = split_held_out_tail(text, ref_chars, max_fraction)
            rec["text"] = prompt
            rec["ref_text"] = ref
        out.append(rec)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--domains", nargs="+", default=list(METRIC_BY_DOMAIN))
    p.add_argument("--prompt-max-chars", type=int, default=4096)
    p.add_argument("--ref-chars", type=int, default=600,
                   help="Held-out tail length for wikitext/fineweb (≈130 tokens)")
    p.add_argument("--max-ref-fraction", type=float, default=0.3)
    args = p.parse_args()

    src, dst = Path(args.src_dir), Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        examples = json.loads((src / f"{domain}.json").read_text(encoding="utf-8"))
        built = build(examples, domain, args.prompt_max_chars, args.ref_chars, args.max_ref_fraction)
        (dst / f"{domain}.json").write_text(json.dumps(built, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
        ref_lens = [len(r["ref_text"]) for r in built]
        print(f"[prep] {domain}: {len(built)} examples, metric={built[0]['metric'] if built else None}, "
              f"ref chars={ref_lens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
