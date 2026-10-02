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
      --num-wikitext 200 --num-orca 200 --num-gsm8k 200 \\
      --max-tokens 512 \\
      --max-gen-tokens 128

Sources (see prompt_datasets.py): wikitext, fineweb, orca, gsm8k, mbpp, cnn_dailymail.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from prompt_datasets import (
    DATASET_NAMES,
    add_collection_args,
    iter_train_texts,
    requested_collection_counts,
)

try:
    from datasets import load_dataset  # noqa: F401  (presence check)
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


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
    # Qwen wrappers take a single run_config dict; Mixtral wrappers take kwargs.
    import inspect
    init_params = inspect.signature(ModelClass.__init__).parameters
    if "run_config" in init_params:
        run_config = {
            "model_path": model_path,
            "tokenizer_path": model_path,
            "backend": "cached",  # 'base' doesn't expose training data collection API
            "device": device,
            "max_cached_experts": cache_size,
            # Cached backend only accepts a single weights-dir argument.
            "expert_weights_dir": "",
        }
        if config_path is not None:
            run_config["config_path"] = config_path
        model = ModelClass(run_config)
    else:
        model = ModelClass(
            model_path=model_path,
            backend="cached",
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
    stream_skip: int = 0,
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
        for text, input_ids in iter_train_texts(
            dataset_name, tokenizer, min_tokens, max_tokens, stream_skip=stream_skip
        ):
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

def write_metadata(output_dir: Path, model_tag: str, counts: dict,
                   min_tokens: int, max_tokens: int, max_gen_tokens: int,
                   model_path: str):
    meta = {
        "model":           model_tag,
        "model_path":      model_path,
        "counts":          counts,
        "num_fineweb":     counts.get("fineweb", 0),
        "num_orca":        counts.get("orca", 0),
        "num_wikitext":    counts.get("wikitext", 0),
        "min_tokens":      min_tokens,
        "max_tokens":      max_tokens,
        "max_gen_tokens":  max_gen_tokens,
        "collection_phase": "generation",
        "file_format":     "per_sample_pt",
        "datasets":        list(DATASET_NAMES),
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
    add_collection_args(parser)
    parser.add_argument(
        "--start-idx", type=int, default=0,
        help="File-index offset and stream skip for the first dataset collected "
             "(resume a single-dataset run from this sample index).",
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

    requested = requested_collection_counts(args)
    collected = {name: 0 for name in DATASET_NAMES}
    running_idx = args.start_idx
    first_dataset = True

    for name in DATASET_NAMES:
        num = requested[name]
        if num <= 0:
            continue
        stream_skip = args.start_idx if first_dataset else 0
        collected[name] = collect_from_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=name,
            num_samples=num,
            output_dir=output_dir,
            model_tag=model_tag,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            start_sample_idx=running_idx,
            max_gen_tokens=args.max_gen_tokens,
            stream_skip=stream_skip,
        )
        running_idx += collected[name]
        first_dataset = False

    write_metadata(
        output_dir=output_dir,
        model_tag=model_tag,
        counts=collected,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_gen_tokens=args.max_gen_tokens,
        model_path=model_path,
    )

    total = sum(collected.values())
    print("=" * 60)
    print("COLLECTION COMPLETE")
    for name in DATASET_NAMES:
        print(f"  {name:<16} {collected[name]} samples")
    print(f"  Total:           {total} .pt files in {output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
