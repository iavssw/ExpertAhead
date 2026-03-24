"""
experiment_forced_top_n.py
==========================
Experiment to determine how many of the top-K unbiased experts (for Qwen: K=8,
for Mixtral: K=2) must be "correct" for perplexity to remain acceptable.

Method:
  For each value of `forced_top_n` from 0 to num_experts_per_tok:
    - Keep the top `forced_top_n` experts as chosen by the unbiased router.
    - Fill the remaining (K - forced_top_n) active-expert slots with randomly
      selected experts (drawn uniformly from the full expert pool).
    - Measure perplexity on a set of prompts.

IMPORTANT — --weights-dir must be the HF model repo ID (e.g. QuixiAI/Qwen3-30B-A3B-AWQ)
or a local directory containing the .safetensors files.  Do NOT pass the
_unpacked subdirectory — the model resolves that automatically from the config.

Usage:
  source /path/to/heteroPredict/utils/setup.sh
  python experiment_forced_top_n.py \\
      --model qwen \\
      --weights-dir /path/to/weights \\
      --tokenizer-path Qwen/Qwen3-30B-A3B \\
      --prompts-file ../prompts.txt \\
      --max-prompt-tokens 200 \\
      --output experiment_topn_qwen.csv

Notes:
  - This script uses the *cached* backend with cache-size=0 so that the
    random_fill_mode_ substitution is controlled entirely by forced_top_n.
  - lambda is set to 0 (no bias) during the experiment so only the random-fill
    substitution is active.
  - The model is loaded once and only the forced_top_n value is changed between
    runs, so load time is minimised.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch

_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.parent
_build_dir = _project_root / "build" / "py" / "unified_llm_w4a16"
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))
else:
    _alt = _project_root / "build" / "py"
    if _alt.exists():
        sys.path.insert(0, str(_alt))

# ── Model imports ──────────────────────────────────────────────────────────────
def _import_model(model_name: str):
    if model_name == "qwen":
        sys.path.insert(0, str(_script_dir.parent / "unified_llm_w4a16"))
        from qwen3_30B_A3B_w4a16_model import Qwen3_30BA3BW4A16Model  # type: ignore
        return Qwen3_30BA3BW4A16Model, 8, "Qwen3-30B-A3B"
    elif model_name == "mixtral":
        sys.path.insert(0, str(_script_dir.parent / "unified_llm_w4a16"))
        from mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model  # type: ignore
        return Mixtral8x7BW4A16Model, 2, "Mixtral-8x7B"
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'qwen' or 'mixtral'.")


# ── Import model file path ──────────────────────────────────────────────────────
def _import_from_file(model_name: str):
    """Import model class directly from the sibling directory."""
    import importlib.util
    models_dir = _script_dir.parent / "unified_llm_w4a16"
    if model_name == "qwen":
        spec = importlib.util.spec_from_file_location(
            "qwen_model", models_dir / "qwen3_30B-A3B_w4a16_model.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Qwen3_30BA3BW4A16Model, 8, "Qwen3-30B-A3B"
    elif model_name == "mixtral":
        spec = importlib.util.spec_from_file_location(
            "mixtral_model", models_dir / "mixtral_8x7B_w4a16_model.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Mixtral8x7BW4A16Model, 2, "Mixtral-8x7B"
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Prompt loading ─────────────────────────────────────────────────────────────
def load_prompts(prompts_file: str, max_tokens: int, tokenizer) -> List[str]:
    """Load prompts, splitting by newline or using the first max_tokens tokens."""
    path = Path(prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    text = path.read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        # Treat the whole file as one prompt, split into chunks
        tokens = tokenizer.encode(text, add_special_tokens=False)
        prompts = []
        for i in range(0, len(tokens), max_tokens):
            chunk = tokens[i : i + max_tokens]
            if len(chunk) >= 10:
                prompts.append(tokenizer.decode(chunk))
        return prompts
    return lines


# ── Perplexity measurement ─────────────────────────────────────────────────────
def measure_perplexity(model, tokenizer, prompts: List[str], max_tokens: int) -> float:
    """Compute average perplexity across all prompts."""
    total_nll = 0.0
    total_tokens = 0
    import torch.nn.functional as F

    for prompt in prompts:
        try:
            input_ids = model.tokenize(prompt)
            if input_ids.size(1) > max_tokens + 1:
                input_ids = input_ids[:, : max_tokens + 1]
            if input_ids.size(1) < 3:
                continue

            result = model.perplexity(input_ids)
            ppl_val = result.get("perplexity", 0.0)
            n_tok = result.get("num_tokens", 0)
            if n_tok > 0 and ppl_val > 0:
                # Convert ppl back to NLL for averaging
                import math
                total_nll += math.log(ppl_val) * n_tok
                total_tokens += n_tok
        except Exception as e:
            print(f"  [warn] prompt failed: {e}")

    if total_tokens == 0:
        return float("inf")
    import math
    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Top-N expert forcing perplexity experiment")
    parser.add_argument("--model", default="qwen", choices=["qwen", "mixtral"],
                        help="Model architecture to test")
    parser.add_argument("--weights-dir", required=True,
                        help="Path to model weights directory")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Tokenizer path (HF model ID or local path)")
    parser.add_argument("--prompts-file", default=None,
                        help="Text file with prompts (one per line, or one long passage)")
    parser.add_argument("--dataset", default=None,
                        choices=["wikitext", "fineweb", "orca"],
                        help="HuggingFace dataset to stream prompts from. "
                             "Overrides --prompts-file and the built-in fallback.")
    parser.add_argument("--max-prompt-tokens", type=int, default=1024,
                        help="Max tokens per prompt for perplexity calculation")
    parser.add_argument("--cache-size", type=int, default=0,
                        help="Expert cache size per layer (0 = load all on demand)")
    parser.add_argument("--num-prompts", type=int, default=5,
                        help="Number of prompts to evaluate per forced_top_n value")
    parser.add_argument("--output", default="experiment_topn_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--config-path", default=None,
                        help="Path to model config JSON")
    args = parser.parse_args()

    # ── Import model class ─────────────────────────────────────────────────────
    ModelClass, num_experts_per_tok, model_label = _import_from_file(args.model)

    print(f"\n{'='*60}")
    print(f"Experiment: Top-N Expert Forcing Perplexity")
    print(f"Model: {model_label}  |  num_experts_per_tok={num_experts_per_tok}")
    print(f"Weights: {args.weights_dir}")
    print(f"{'='*60}\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    # Use the cached backend with cache_size=num_experts_per_tok (e.g. 8 for Qwen).
    # This allocates exactly the right number of slots for one token's active set.
    # We set prewarm=0 (cache_size in prewarm call) so no pre-loading happens;
    # experts are loaded on demand during the first forward pass of each prompt.
    # lambda=0 so the only routing change is random_fill_mode_.
    #
    # NOTE: pass the HF repo ID or parent safetensors dir as --weights-dir, NOT
    # the _unpacked subdirectory. The model resolves that via usePreSavedWeights.
    print("Loading model (base backend, all experts in RAM, lambda=0)...")
    kwargs = dict(
        model_path=args.weights_dir,
        tokenizer_path=args.tokenizer_path,
        backend="base",
    )
    # Only pass config_path if explicitly specified by the user.
    # For Qwen, leaving it as None lets the model auto-discover
    # configs/configs_strixH_qwen3_30B_A3B.json5 which has
    # usePreSavedWeights=true → loads from bins, not safetensors.
    if args.model == "qwen" and args.config_path:
        kwargs["config_path"] = args.config_path
    model = ModelClass(**kwargs)
    print("Model loaded.\n")

    # ── Load tokenizer and prompts ─────────────────────────────────────────────
    tokenizer = model.tokenizer
    if args.dataset:
        try:
            from datasets import load_dataset
            print(f"Loading '{args.dataset}' dataset from HuggingFace...")
            if args.dataset == "wikitext":
                hf_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
            elif args.dataset == "fineweb":
                hf_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            elif args.dataset == "orca":
                hf_ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            prompts = []
            for item in hf_ds:
                if args.dataset == "orca":
                    text = f"{item.get('system_prompt', '')}\n{item.get('question', '')}\n{item.get('response', '')}".strip()
                else:
                    text = item.get("text", "")
                if len(text) > 100:
                    prompts.append(text)
                if len(prompts) >= args.num_prompts:
                    break
        except ImportError:
            print("Error: 'datasets' library not found. Install with: pip install datasets")
            sys.exit(1)
    elif args.prompts_file:
        prompts = load_prompts(args.prompts_file, args.max_prompt_tokens, tokenizer)
    else:
        # Fallback: built-in passages
        prompts = [
            "The transformer architecture has revolutionized natural language processing. "
            "Self-attention mechanisms allow models to weigh relevance of different tokens "
            "when encoding representations. This has enabled unprecedented performance on "
            "tasks such as translation, summarization, and question answering.",
            "Recent advances in mixture-of-experts models have demonstrated that sparsely "
            "activated networks can dramatically scale model capacity without proportionally "
            "increasing compute requirements at inference time. Expert routing decisions "
            "are made by lightweight router networks that learn to assign tokens to specialists.",
            "The perplexity of a language model is defined as the exponential of the average "
            "negative log-likelihood per token. Lower perplexity indicates that the model "
            "assigns higher probability to the observed token sequence, reflecting better "
            "fit to the underlying language distribution.",
        ]
    prompts = prompts[:args.num_prompts]
    print(f"Using {len(prompts)} prompts from '{args.dataset or (args.prompts_file or 'built-in fallback')}'.")

    # ── Sweep forced_top_n ─────────────────────────────────────────────────────
    results = []
    print(f"\n{'forced_top_n':>14} | {'avg_perplexity':>16} | {'delta_vs_baseline':>18}")
    print("-" * 56)

    baseline_ppl = None

    # Sweep order: K down to 0 (baseline first at K, then degrade)
    for n in range(num_experts_per_tok, -1, -1):
        model.set_forced_top_n(n)
        model.set_random_fill_mode(n < num_experts_per_tok)

        ppl = measure_perplexity(model, tokenizer, prompts, args.max_prompt_tokens)

        if n == num_experts_per_tok:
            baseline_ppl = ppl
            delta = 0.0
        else:
            delta = ((ppl - baseline_ppl) / baseline_ppl * 100.0) if baseline_ppl else float("inf")

        results.append({
            "forced_top_n": n,
            "avg_perplexity": round(ppl, 4),
            "delta_pct": round(delta, 2),
        })

        tag = " <- baseline" if n == num_experts_per_tok else ""
        print(f"{n:>14} | {ppl:>16.4f} | {delta:>+17.2f}%{tag}")

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["forced_top_n", "avg_perplexity", "delta_pct"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {out_path.resolve()}")
    print("\nInterpretation:")
    print(f"  forced_top_n={num_experts_per_tok}: all correct experts → baseline perplexity")
    print(f"  forced_top_n=N: top-N correct + {num_experts_per_tok-num_experts_per_tok+0}-ish random")
    print(f"  Choose the smallest N where delta_pct stays within your tolerance (e.g. <5%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
