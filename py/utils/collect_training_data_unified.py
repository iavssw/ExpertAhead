#!/usr/bin/env python3
"""
Unified training data collection script for MoE expert prediction.

Collects per-layer embeddings (post-attention-norm, pre-router) and router logits
during the **generation (decode) phase** of inference. Also captures the prefill
expert distribution as a summary feature per layer. Supported models:
  - mixtral_8x7b  : Mixtral 8x7B   (TheBloke/mixtral-8x7b-v0.1-AWQ)
  - mixtral_8x22b : Mixtral 8x22B  (MaziyarPanahi/Mixtral-8x22B-v0.1-AWQ)
  - qwen3_30b     : Qwen3 30B-A3B  (QuixiAI/Qwen3-30B-A3B-AWQ)
  - qwen3_480b    : Qwen3 480B-A35B (QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ)

To add a new model, add one entry to MODEL_REGISTRY below — no other changes needed.

Data collection procedure per sample:
  1. Prefill phase  — run one forward pass on the full prompt with collection
                      enabled; extract prefill_expert_dist per layer; flush buffers.
  2. Decode phase   — autoregressively call forward() for up to max_gen_tokens
                      steps with collection enabled; each step appends one row of
                      (embedding, router_logits) per layer to the C++ buffer.
  3. After the loop — call get_training_data() to retrieve [gen_len, *] tensors.

The predictor is then trained on consecutive decode steps:
  input:  embedding[t]  +  prefill_expert_dist
  label:  top-k experts from router_logits[t+1]

Each output .pt file contains:
  {
    'sample_idx':    int,
    'dataset':       str,
    'token_count':   int,     # number of generation tokens collected
    'model':         str,
    'layers': [
        {
            'layer_idx':            int,
            'embeddings':           Tensor[gen_len, hidden_size],   # fp32 CPU
            'router_logits':        Tensor[gen_len, num_experts],   # fp32 CPU
            'prev_expert_ids':      Tensor[gen_len, top_k],         # int64 CPU — experts chosen at step t
            'prefill_expert_count': Tensor[num_experts],            # int counts
        },
        ...
    ]
  }

Usage:
  python collect_training_data_unified.py \\
      --model <model_tag> \\
      --output-dir <path> \\
      --num-wikitext 100 \\
      --max-tokens 512 \\
      --max-gen-tokens 128
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def _stream_texts(dataset_name: str, min_tokens: int, max_tokens: int, tokenizer):
    """
    Yield (text, token_ids_tensor) pairs from the given streaming dataset,
    already filtered to [min_tokens, max_tokens] and truncated if needed.
    """
    if dataset_name == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    elif dataset_name == "orca":
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    elif dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    for example in ds:
        if dataset_name == "fineweb":
            text = example.get("text", "").strip()
        elif dataset_name == "orca":
            system   = example.get("system_prompt", "")
            question = example.get("question", "")
            text = f"{system}\n{question}\n{example.get('response', '')}".strip()
        elif dataset_name == "wikitext":
            text = example.get("text", "").strip()
            if not text or text.startswith(" = "): # Skip section headers
                continue

        if not text:
            continue

        enc = tokenizer(
            [text],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_tokens,
        )
        ids = enc["input_ids"]  # [1, seq_len]
        seq_len = ids.shape[1]
        if seq_len < min_tokens:
            continue

        yield text, ids


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# Each entry: model_tag → (py_filename, class_name, default_hf_path)
# Filenames with hyphens are handled transparently via importlib.
# To add a new model: add one row here, nothing else to change.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mixtral_8x7b":  ("unified_llm_w4a16/mixtral_8x7B_w4a16_model.py",       "Mixtral8x7BW4A16Model",      "TheBloke/mixtral-8x7b-v0.1-AWQ"),
    "mixtral_8x22b": ("unified_llm_w4a16/mixtral_8x22B_w4a16_model.py",      "Mixtral8x22BW4A16Model",     "MaziyarPanahi/Mixtral-8x22B-v0.1-AWQ"),
    "qwen3_30b":     ("unified_llm_w4a16/qwen3_30B-A3B_w4a16_model.py",      "Qwen3_30BA3BW4A16Model",     "QuixiAI/Qwen3-30B-A3B-AWQ"),
    "qwen3_480b":    ("unified_llm_w4a16/qwen3_480B-A35B_w4a16_model.py",    "Qwen3_480BA35BW4A16Model",   "Qwen/Qwen3-480B-A35B-AWQ"),
}


def _load_model_from_registry(model_tag: str, model_path: str, config_path=None, device="cuda"):
    """
    Generic model loader. Looks up model_tag in MODEL_REGISTRY,
    imports the wrapper module (handling hyphens via importlib), and
    instantiates the model class with the base backend.
    """
    if model_tag not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model tag {model_tag!r}. "
            f"Valid choices: {list(MODEL_REGISTRY.keys())}"
        )

    filename, class_name, _ = MODEL_REGISTRY[model_tag]
    # Wrapper paths are relative to py/ (one level above this utils/ script)
    py_root = Path(__file__).parent.parent
    wrapper_path = py_root / filename

    if not wrapper_path.exists():
        raise FileNotFoundError(
            f"Model wrapper not found: {wrapper_path}\n"
            f"Please create the Python wrapper for '{model_tag}' at that path."
        )

    import importlib.util
    spec = importlib.util.spec_from_file_location(filename.replace("-", "_").replace(".py", ""), str(wrapper_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ModelClass = getattr(mod, class_name)

    if "qwen" in model_tag:
        cache_size = 128
    else:
        cache_size = 8

    print(f"Loading {model_tag} from {model_path} with cache_size={cache_size} …")
    model = ModelClass(
        model_path=model_path,
        backend="cached",           # 'base' doesn't expose training data collection API;
                                    # 'cached' does, and caching is a bonus during collection.
        max_cached_experts_per_layer=cache_size,
        device=device,
        config_path=config_path,
    )
    print(f"{model_tag} loaded.\n")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Core collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_from_dataset(
    model,
    tokenizer,
    dataset_name: str,
    num_samples: int,
    output_dir: Path,
    model_tag: str,
    min_tokens: int,
    max_tokens: int,
    start_sample_idx: int = 0,
    max_gen_tokens: int = 128,
):
    """
    Run inference on `num_samples` texts from `dataset_name`, collecting
    per-layer embeddings and router logits from the **generation (decode) phase**.
    Saves one .pt file per sample.

    Procedure per sample:
      1. Prefill (collection enabled)  → capture prefill_expert_count per layer
      2. Clear buffers
      3. Decode loop (up to max_gen_tokens)  → accumulate one row per step
      4. get_training_data() → [gen_len, hidden] embeddings + [gen_len, E] logits

    Returns the number of successfully collected samples.
    """
    if not HAS_DATASETS:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Collecting from: {dataset_name}  ({num_samples} samples)")
    print(f"Prompt token range: {min_tokens}–{max_tokens}  |  max_gen_tokens={max_gen_tokens}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    eos_id = -1
    if tokenizer.eos_token_id is not None:
        eos_id = tokenizer.eos_token_id

    top_k_experts = getattr(model, "num_experts_per_tok", 2)

    collected = 0
    skipped   = 0

    # Enable C++ backend collection for the entire session
    model.model.enable_training_data_collection()

    try:
        for text, input_ids in _stream_texts(dataset_name, min_tokens, max_tokens, tokenizer):
            global_idx  = start_sample_idx + collected
            prefill_len = input_ids.shape[1]

            print(f"  [{collected + 1}/{num_samples}] prefill={prefill_len} tokens", end="", flush=True)

            input_ids_dev = input_ids.to(model.device)

            # ── Phase 1: Prefill ──────────────────────────────────────────────
            # Run with collection on so we can compute prefill_expert_count.
            model.model.clear_training_data()
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    prefill_logits = model.model.forward(input_ids_dev, 0)
            except Exception as e:
                print(f"  ← PREFILL ERROR: {e}")
                skipped += 1
                continue

            prefill_data = model.model.get_training_data()   # [(emb, rlogits), ...] per layer
            model.model.clear_training_data()                # reset before decode

            if not prefill_data:
                print("  ← WARNING: no prefill data, skipping")
                skipped += 1
                continue 

            # Compute per-layer prefill expert count from the collected prefill data
            prefill_expert_counts = {}   # layer_idx -> count tensor
            for l_idx, (emb_p, rlogits_p) in enumerate(prefill_data):
                rlogits_cpu = rlogits_p.float().cpu()        # [prefill_len, num_experts]
                top_k_idx   = rlogits_cpu.topk(top_k_experts, dim=-1).indices
                prefill_count = torch.bincount(top_k_idx.flatten(), minlength=rlogits_cpu.shape[-1])
                prefill_expert_counts[l_idx] = prefill_count
                del emb_p, rlogits_p, rlogits_cpu
            del prefill_data

            # ── Phase 2: Decode loop ─────────────────────────────────────────
            # Greedily sample the first new token from prefill logits.
            next_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            del prefill_logits

            gen_steps = 0
            gen_data_per_layer = {}  # layer_idx -> {"embeddings": [], "router_logits": []}
            t_decode_start = time.perf_counter()
            try:
                with torch.no_grad():
                    for step in range(max_gen_tokens):
                        decode_logits = model.model.forward(next_token, prefill_len + step)
                        step_data = model.model.get_training_data()
                        model.model.clear_training_data()
                        
                        for l_idx, (emb_p, rlogits_p) in enumerate(step_data):
                            if l_idx not in gen_data_per_layer:
                                gen_data_per_layer[l_idx] = {"embeddings": [], "router_logits": [], "prev_expert_ids": []}
                            rlogits_cpu = rlogits_p.float().cpu()
                            # top-k expert indices for this step (the "current" choice, used as feature for t+1)
                            step_topk_idx = rlogits_cpu.topk(top_k_experts, dim=-1).indices  # [1, top_k] or [top_k]
                            gen_data_per_layer[l_idx]["embeddings"].append(emb_p.squeeze(0).cpu())
                            gen_data_per_layer[l_idx]["router_logits"].append(rlogits_cpu)
                            gen_data_per_layer[l_idx]["prev_expert_ids"].append(step_topk_idx.squeeze(0).cpu())

                        next_token = decode_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen_steps += 1
                        if eos_id >= 0 and next_token.item() == eos_id:
                            break
            except Exception as e:
                print(f"  ← DECODE ERROR after {gen_steps} steps: {e}")
                if gen_steps < 2:
                    skipped += 1
                    model.model.clear_training_data()
                    continue
                # otherwise keep the partial data

            elapsed = time.perf_counter() - t0

            if not gen_data_per_layer:
                print(f"  ← WARNING: no generation data after {gen_steps} steps, skipping")
                skipped += 1
                continue

            layers = []
            for l_idx, data_dict in gen_data_per_layer.items():
                if not data_dict["embeddings"]: continue
                emb          = torch.cat(data_dict["embeddings"], dim=0).float()      # [gen_steps, hidden_size]
                rlogits      = torch.cat(data_dict["router_logits"], dim=0).float()   # [gen_steps, num_experts]
                prev_exp_ids = torch.stack(data_dict["prev_expert_ids"], dim=0).long() # [gen_steps, top_k]

                # If gen_data has more layers than prefill_data, default to uniform
                prefill_ct = prefill_expert_counts.get(l_idx)
                if prefill_ct is None:
                    prefill_ct = torch.ones(rlogits.shape[-1], dtype=torch.float)

                layers.append({
                    "layer_idx":            l_idx,
                    "embeddings":           emb,
                    "router_logits":        rlogits,
                    "prev_expert_ids":      prev_exp_ids,
                    "prefill_expert_count": prefill_ct,
                })
            torch.cuda.empty_cache()

            sample_record = {
                "sample_idx":  global_idx,
                "dataset":     dataset_name,
                "token_count": gen_steps,   # number of generation tokens collected
                "model":       model_tag,
                "layers":      layers,
            }

            fname = output_dir / f"{dataset_name}_{global_idx:05d}.pt"
            torch.save(sample_record, fname)

            print(f"  ← {len(layers)} layers, {gen_steps} gen tokens, {elapsed:.2f}s  → {fname.name}")

            collected += 1
            if collected >= num_samples:
                break

    finally:
        model.model.disable_training_data_collection()

    print(f"\n  Done: {collected} collected, {skipped} skipped\n")
    return collected


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helper
# ─────────────────────────────────────────────────────────────────────────────

def write_metadata(output_dir: Path, model_tag: str, num_fineweb: int, num_orca: int,
                   num_wikitext: int, min_tokens: int, max_tokens: int, max_gen_tokens: int,
                   model_path: str):
    meta = {
        "model":           model_tag,
        "model_path":      model_path,
        "num_fineweb":     num_fineweb,
        "num_orca":        num_orca,
        "num_wikitext":    num_wikitext,
        "min_tokens":      min_tokens,
        "max_tokens":      max_tokens,
        "max_gen_tokens":  max_gen_tokens,
        "collection_phase": "generation",
        "file_format":     "per_sample_pt",
        "description":     (
            "Each .pt file contains embeddings and router_logits from the GENERATION phase. "
            "'token_count' is the number of generation tokens collected. "
            "'layers' is a list of dicts with: layer_idx, embeddings [gen_len, hidden_size], "
            "router_logits [gen_len, num_experts] (generation steps only), "
            "prev_expert_ids [gen_len, top_k] (expert IDs selected at each step, int64), "
            "prefill_expert_count [num_experts] (expert usage counts over full prefill). "
            "Training pairs: embedding[t] + prev_expert_ids[t] + prefill_expert_count -> top-k experts from router_logits[t+1]."
        ),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {meta_path}")



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect per-layer embeddings + router logits for MoE expert prediction training."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which model to collect data for. Choices: " + ", ".join(MODEL_REGISTRY.keys())
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="HuggingFace repo ID or local path. Defaults to the registry default for each model."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where per-sample .pt files will be saved."
    )
    parser.add_argument(
        "--num-fineweb", type=int, default=0,
        help="Number of samples from FineWeb (default: 100)."
    )
    parser.add_argument(
        "--num-orca", type=int, default=0,
        help="Number of samples from OpenOrca (default: 100)."
    )
    parser.add_argument(
        "--num-wikitext", type=int, default=1000,
        help="Number of samples from Wikitext-103 (default: 0)."
    )
    parser.add_argument(
        "--min-tokens", type=int, default=100,
        help="Minimum token count per sample (default: 100)."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum token count per sample (default: 512)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)."
    )
    parser.add_argument(
        "--config-path", type=str, default=None,
        help="Path to the backend JSON5 config file. Uses model-specific defaults."
    )
    parser.add_argument(
        "--max-gen-tokens", type=int, default=128,
        help="Maximum number of generation (decode) tokens to collect per sample (default: 128)."
    )
    parser.add_argument(
        "--skip-fineweb", action="store_true",
        help="Skip FineWeb collection (useful for resuming Orca-only)."
    )
    parser.add_argument(
        "--skip-orca", action="store_true",
        help="Skip Orca collection (useful for resuming FineWeb-only)."
    )

    args = parser.parse_args()

    # ── resolve defaults ──────────────────────────────────────────────────────
    model_tag  = args.model
    _, _, default_path = MODEL_REGISTRY[model_tag]
    model_path = args.model_path or default_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    try:
        model = _load_model_from_registry(
            model_tag, model_path, config_path=args.config_path, device=args.device
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    tokenizer = model.tokenizer
    if tokenizer is None:
        print("ERROR: model has no tokenizer loaded.", file=sys.stderr)
        return 1

    # ── collect FineWeb ───────────────────────────────────────────────────────
    fineweb_collected = 0
    if not args.skip_fineweb and args.num_fineweb > 0:
        fineweb_collected = collect_from_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name="fineweb",
            num_samples=args.num_fineweb,
            output_dir=output_dir,
            model_tag=model_tag,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            start_sample_idx=0,
            max_gen_tokens=args.max_gen_tokens,
        )

    # ── collect Orca ──────────────────────────────────────────────────────────
    orca_collected = 0
    if not args.skip_orca and args.num_orca > 0:
        orca_collected = collect_from_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name="orca",
            num_samples=args.num_orca,
            output_dir=output_dir,
            model_tag=model_tag,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            start_sample_idx=args.num_fineweb,
            max_gen_tokens=args.max_gen_tokens,
        )

    # ── collect Wikitext ──────────────────────────────────────────────────────
    wikitext_collected = 0
    if args.num_wikitext > 0:
        wikitext_collected = collect_from_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name="wikitext",
            num_samples=args.num_wikitext,
            output_dir=output_dir,
            model_tag=model_tag,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            start_sample_idx=args.num_fineweb + args.num_orca,
            max_gen_tokens=args.max_gen_tokens,
        )

    # ── write metadata ────────────────────────────────────────────────────────
    write_metadata(
        output_dir=output_dir,
        model_tag=model_tag,
        num_fineweb=fineweb_collected,
        num_orca=orca_collected,
        num_wikitext=wikitext_collected,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_gen_tokens=args.max_gen_tokens,
        model_path=model_path,
    )

    total = fineweb_collected + orca_collected + wikitext_collected
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print(f"  FineWeb: {fineweb_collected} samples")
    print(f"  Orca:    {orca_collected} samples")
    print(f"  Total:   {total} .pt files in {output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
