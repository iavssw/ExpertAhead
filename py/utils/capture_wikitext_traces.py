import argparse
import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "unified_llm_w4a16"))

import importlib.util

def load_wikitext_test_text():
    print("Loading WikiText-103 test split...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    lines = [line for line in ds["text"] if line and line.strip()]
    return "\n\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Capture oracle traces from the WikiText-103 test set")
    parser.add_argument("--out-dir", required=True, help="Directory to save traces")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of traces to generate")
    parser.add_argument("--prompt-tokens", type=int, default=512, help="Context length of each prompt")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Number of tokens to generate per trace")
    parser.add_argument("--model-path", default="QuixiAI/Qwen3-30B-A3B-AWQ")
    args = parser.parse_args()

    config_path = str(ROOT / "unified_llm_w4a16" / "configs" / "configs_strixH_qwen3_30B_A3B.json5")
    
    spec = importlib.util.spec_from_file_location("qwen3_model", ROOT / "unified_llm_w4a16" / "qwen3_30B-A3B_w4a16_model.py")
    qwen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen_mod)
    Qwen3Model = qwen_mod.Qwen3_30BA3BW4A16Model

    print("Loading model for capture (cache=128 to ensure pure trace)...")
    model = Qwen3Model(
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        device="cuda",
        backend="predict",
        max_cached_experts_per_layer=128,
        prefetch_experts_count=128,
        config_path=config_path,
        expert_weights_dir=None
    )

    text = load_wikitext_test_text()
    print("Tokenizing entire wikitext test set...")
    all_ids = model.tokenize(text)[0]
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    stride = len(all_ids) // args.num_prompts
    
    for i in range(args.num_prompts):
        start = i * stride
        end = start + args.prompt_tokens
        if end >= len(all_ids): break
        
        prompt_ids = all_ids[start:end].unsqueeze(0)
        prompt_text = model.tokenizer.decode(prompt_ids[0].tolist(), skip_special_tokens=False)
        
        print(f"[{i+1}/{args.num_prompts}] Capturing trace...")
        model.model.begin_oracle_trace_capture()
        output = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        gen_ids = output[0, args.prompt_tokens:].tolist()
        generated_text = model.tokenizer.decode(gen_ids, skip_special_tokens=False)
        
        out_path = out_dir / f"trace_{i:05d}.txt"
        model.model.write_oracle_trace_file(
            str(out_path),
            prompt_ids,
            output,
            prompt_text,
            generated_text,
            "qwen3_30b"
        )
        model.model.cancel_oracle_trace_capture()
        print(f"  -> Saved {out_path}")

if __name__ == "__main__":
    main()
