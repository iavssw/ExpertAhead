#!/usr/bin/env python3
"""
Capture an exact oracle trace from the predict/base backend using model.generate().

Uses PROMPT TOKEN IDS in the output so replay does not depend on text re-tokenization.
For a true upper-bound prefetch experiment, capture with all experts resident in cache
(--max-cached-experts >= num_experts) and lambda=0.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "unified_llm_w4a16"))

from oracle_trace import load_oracle_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture exact oracle expert trace via generate()")
    parser.add_argument("--output", required=True, help="Output .txt trace path")
    parser.add_argument("--text", default="", help="Prompt text (ignored if --input-trace is set)")
    parser.add_argument(
        "--input-trace",
        default="",
        help="Optional existing trace: reuse PROMPT TOKEN IDS / PROMPT TEXT from it",
    )
    parser.add_argument("--backend", default="predict", choices=["base", "predict", "cached"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-cached-experts", type=int, default=128,
                        help="Use full expert cache during capture (recommended: 128 for Qwen)")
    parser.add_argument("--model-path", default="QuixiAI/Qwen3-30B-A3B-AWQ")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--expert-weights-dir", default=None)
    args = parser.parse_args()

    if args.input_trace:
        src = load_oracle_trace(args.input_trace)
        prompt_text = src.prompt_text or args.text
        if src.prompt_token_ids:
            prompt_ids_list = src.prompt_token_ids
        elif prompt_text:
            prompt_ids_list = None
        else:
            print("Error: input trace has neither PROMPT TOKEN IDS nor PROMPT TEXT", file=sys.stderr)
            return 1
    else:
        prompt_text = args.text
        prompt_ids_list = None

    if args.config_path is None:
        args.config_path = str(
            ROOT / "unified_llm_w4a16" / "configs" / "configs_strixH_qwen3_30B_A3B.json5"
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "qwen3_model",
        ROOT / "unified_llm_w4a16" / "qwen3_30B-A3B_w4a16_model.py",
    )
    assert spec and spec.loader
    qwen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen_mod)
    Qwen3Model = qwen_mod.Qwen3_30BA3BW4A16Model

    print(f"Loading model (backend={args.backend}, cache={args.max_cached_experts})...")
    model = Qwen3Model(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path or args.model_path,
        device=args.device,
        backend=args.backend,
        max_cached_experts_per_layer=args.max_cached_experts,
        prefetch_experts_count=args.max_cached_experts,
        config_path=args.config_path,
        expert_weights_dir=args.expert_weights_dir,
    )

    if prompt_ids_list is None:
        if not prompt_text:
            print("Error: provide --text or --input-trace with prompt content", file=sys.stderr)
            return 1
        input_ids = model.tokenize(prompt_text)
    else:
        input_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=args.device)
        if not prompt_text and model.tokenizer is not None:
            prompt_text = model.tokenizer.decode(prompt_ids_list, skip_special_tokens=False)

    if not hasattr(model.model, "begin_oracle_trace_capture"):
        print("Error: backend does not support oracle trace capture (rebuild predict libtorch)", file=sys.stderr)
        return 1

    print(f"Capturing greedy decode ({args.max_new_tokens} tokens)...")
    model.model.begin_oracle_trace_capture()
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    prompt_len = input_ids.size(1)
    gen_ids = output[0, prompt_len:].tolist()
    generated_text = ""
    if model.tokenizer is not None:
        generated_text = model.tokenizer.decode(gen_ids, skip_special_tokens=False)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = model.model.write_oracle_trace_file(
        str(out_path),
        input_ids,
        output,
        prompt_text,
        generated_text,
        "qwen3_30b",
    )
    model.model.cancel_oracle_trace_capture()

    if not ok:
        return 1

    written = load_oracle_trace(out_path)
    print(f"Wrote {out_path}")
    print(f"  prompt tokens: {len(written.prompt_token_ids)}")
    print(f"  generated tokens: {len(written.generated_token_ids)}")
    print(f"  expert trace steps: {len(written.expert_trace)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
