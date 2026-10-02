#!/usr/bin/env python3
"""Capture oracle expert traces for per-domain examples JSON files (one model load).

Prompts are tokenized exactly as the sweep does for ``--examples-json``
(``model.tokenize(text[:prompt_max_chars])``), so ``--oracle-trace-dir <trace_root>/<domain>``
replays the matching trace for prompt index i. Capture uses a full expert cache (128) and λ=0.

Usage:
  python3 py/utils/capture_domain_oracle_traces.py \\
    --examples-dir py/utils/final_results_runs/quality_speed/by_domain \\
    --out-root trainingData/domain_oracle_traces_200 \\
    --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "unified_llm_w4a16"))

from oracle_trace import load_oracle_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture oracle traces for per-domain examples")
    parser.add_argument("--examples-dir", required=True, help="Directory with <domain>.json examples")
    parser.add_argument("--out-root", required=True, help="Traces go to <out-root>/<domain>/")
    parser.add_argument("--domains", nargs="+",
                        default=["wikitext", "fineweb", "orca", "gsm8k", "cnn_dailymail"])
    parser.add_argument("--num-prompts", type=int, default=0, help="0 = all examples per domain")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--prompt-max-chars", type=int, default=4096)
    parser.add_argument("--max-cached-experts", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-path", default="QuixiAI/Qwen3-30B-A3B-AWQ")
    args = parser.parse_args()

    examples_dir, out_root = Path(args.examples_dir), Path(args.out_root)
    jobs = []
    for domain in args.domains:
        examples = json.loads((examples_dir / f"{domain}.json").read_text(encoding="utf-8"))
        if args.num_prompts > 0:
            examples = examples[: args.num_prompts]
        for i, ex in enumerate(examples):
            text = ex.get("text") if isinstance(ex, dict) else str(ex)
            out_path = out_root / domain / f"oracle_trace_qwen3_30b_{i:05d}.txt"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {out_path} exists")
                continue
            jobs.append((domain, i, (text or "")[: args.prompt_max_chars], out_path))

    if not jobs:
        print("All traces present; nothing to capture.")
        return 0

    spec = importlib.util.spec_from_file_location(
        "qwen3_model", ROOT / "unified_llm_w4a16" / "qwen3_30B-A3B_w4a16_model.py"
    )
    assert spec and spec.loader
    qwen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen_mod)

    print(f"Loading model (predict backend, cache={args.max_cached_experts}) for {len(jobs)} trace(s)...")
    model = qwen_mod.Qwen3_30BA3BW4A16Model(
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        device="cuda",
        backend="predict",
        max_cached_experts_per_layer=args.max_cached_experts,
        prefetch_experts_count=args.max_cached_experts,
        config_path=str(ROOT / "unified_llm_w4a16" / "configs" / "configs_strixH_qwen3_30B_A3B.json5"),
        expert_weights_dir=str(ROOT / "unified_llm_w4a16" / "model_weights" / "Qwen3-30B-A3B-AWQ_packed"),
    )

    failures = 0
    for n, (domain, i, text, out_path) in enumerate(jobs, 1):
        input_ids = model.tokenize(text)
        print(f"[{n}/{len(jobs)}] {domain} #{i}: {input_ids.size(1)} prompt tokens -> {out_path}")
        model.model.begin_oracle_trace_capture()
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens, temperature=0.0)
        gen_ids = output[0, input_ids.size(1):].tolist()
        generated_text = model.tokenizer.decode(gen_ids, skip_special_tokens=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = model.model.write_oracle_trace_file(
            str(out_path), input_ids, output, text, generated_text, "qwen3_30b"
        )
        model.model.cancel_oracle_trace_capture()
        if not ok:
            failures += 1
            print(f"  !! failed to write {out_path}")
            continue
        written = load_oracle_trace(out_path)
        print(f"  steps={len(written.expert_trace)} generated={len(written.generated_token_ids)}")

    print(f"Done: {len(jobs) - failures}/{len(jobs)} traces written under {out_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
