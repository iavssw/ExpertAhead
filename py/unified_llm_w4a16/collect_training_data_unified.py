#!/usr/bin/env python3
"""
Unified training data collection script for MoE expert prediction.

Collects per-layer embeddings (post-attention-norm, pre-router) and router logits
for each sample, saving one .pt file per sample. Supported models:
  - mixtral_8x7b  : Mixtral 8x7B   (TheBloke/mixtral-8x7b-v0.1-AWQ)
  - mixtral_8x22b : Mixtral 8x22B  (TheBloke/Mixtral-8x22B-v0.1-AWQ)
  - qwen3_30b     : Qwen3 30B-A3B  (QuixiAI/Qwen3-30B-A3B-AWQ)
  - qwen3_480b    : Qwen3 480B-A35B (Qwen/Qwen3-480B-A35B-AWQ)

To add a new model, add one entry to MODEL_REGISTRY below — no other changes needed.

Each output .pt file contains:
  {
    'sample_idx':    int,
    'dataset':       str,     # 'fineweb' or 'orca'
    'token_count':   int,
    'model':         str,     # e.g. 'mixtral_8x7b'
    'layers': [
        {
            'layer_idx':     int,
            'embeddings':    Tensor[seq_len, hidden_size],   # fp32 on CPU
            'router_logits': Tensor[seq_len, num_experts],   # fp32 on CPU
        },
        ...
    ]
  }

Usage:
  python collect_training_data_unified.py \\
      --model <model_tag> \\
      --output-dir <path> \\
      --num-fineweb 100 \\
      --num-orca 100 \\
      --max-tokens 512
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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    for example in ds:
        if dataset_name == "fineweb":
            text = example.get("text", "").strip()
        elif dataset_name == "orca":
            system   = example.get("system_prompt", "")
            question = example.get("question", "")
            response = example.get("response", "")
            text = f"{system}\n{question}\n{response}".strip()

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
    "mixtral_8x7b":  ("mixtral_8x7B_w4a16_model.py",       "Mixtral8x7BW4A16Model",      "TheBloke/mixtral-8x7b-v0.1-AWQ"),
    "mixtral_8x22b": ("mixtral_8x22B_w4a16_model.py",      "Mixtral8x22BW4A16Model",     "TheBloke/Mixtral-8x22B-v0.1-AWQ"),
    "qwen3_30b":     ("qwen3_30B-A3B_w4a16_model.py",      "Qwen3_30BA3BW4A16Model",     "QuixiAI/Qwen3-30B-A3B-AWQ"),
    "qwen3_480b":    ("qwen3_480B-A35B_w4a16_model.py",    "Qwen3_480BA35BW4A16Model",   "Qwen/Qwen3-480B-A35B-AWQ"),
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
    wrapper_path = Path(__file__).parent / filename

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

    print(f"Loading {model_tag} from {model_path} …")
    model = ModelClass(
        model_path=model_path,
        backend="cached",           # 'base' doesn't expose training data collection API;
                                    # 'cached' does, and caching is a bonus during collection.
        max_cached_experts_per_layer=8,
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
):
    """
    Run inference on `num_samples` texts from `dataset_name`, collecting
    per-layer embeddings and router logits. Saves one .pt file per sample.

    Returns the number of successfully collected samples.
    """
    if not HAS_DATASETS:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Collecting from: {dataset_name}  ({num_samples} samples)")
    print(f"Token range: {min_tokens}–{max_tokens}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    collected = 0
    skipped   = 0

    # Enable C++ backend collection
    model.model.enable_training_data_collection()

    try:
        for text, input_ids in _stream_texts(dataset_name, min_tokens, max_tokens, tokenizer):
            global_idx = start_sample_idx + collected
            token_count = input_ids.shape[1]

            print(f"  [{collected + 1}/{num_samples}] tokens={token_count}", end="", flush=True)

            # Move input_ids to the model device
            input_ids_dev = input_ids.to(model.device)

            # Clear previous layer data
            model.model.clear_training_data()

            # Forward pass (prefill only – no generation)
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    _ = model(input_ids_dev, start_pos=0)
            except Exception as e:
                print(f"  ← ERROR: {e}")
                skipped += 1
                continue
            elapsed = time.perf_counter() - t0

            # Retrieve collected data: list of (embeddings, router_logits) per layer
            training_data = model.model.get_training_data()
            # Clear C++ buffer immediately to free GPU memory before we process
            model.model.clear_training_data()

            if not training_data:
                print("  ← WARNING: no data collected, skipping")
                skipped += 1
                continue

            # Build per-sample record — move each tensor to CPU immediately and
            # delete the GPU reference so the allocator can reclaim the memory.
            layers = []
            for layer_idx, (embeddings, router_logits) in enumerate(training_data):
                # embeddings:    [batch, seq_len, hidden_size]  → squeeze batch dim → [seq_len, hidden_size]
                # router_logits: [batch*seq_len, num_experts]   → [seq_len, num_experts]
                emb     = embeddings.squeeze(0).float().cpu()   # [seq_len, hidden_size]
                rlogits = router_logits.float().cpu()           # [seq_len, num_experts]
                del embeddings, router_logits                   # release GPU tensors now
                layers.append({
                    "layer_idx":     layer_idx,
                    "embeddings":    emb,
                    "router_logits": rlogits,
                })
            del training_data                                   # release the list
            torch.cuda.empty_cache()                            # return freed pages to allocator

            sample_record = {
                "sample_idx":  global_idx,
                "dataset":     dataset_name,
                "token_count": token_count,
                "model":       model_tag,
                "layers":      layers,
            }

            # Save as individual file: <dataset>_<sample_idx:05d>.pt
            fname = output_dir / f"{dataset_name}_{global_idx:05d}.pt"
            torch.save(sample_record, fname)

            print(f"  ← {len(layers)} layers, {elapsed:.2f}s  → {fname.name}")

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
                   min_tokens: int, max_tokens: int, model_path: str):
    meta = {
        "model":          model_tag,
        "model_path":     model_path,
        "num_fineweb":    num_fineweb,
        "num_orca":       num_orca,
        "min_tokens":     min_tokens,
        "max_tokens":     max_tokens,
        "file_format":    "per_sample_pt",
        "description":    (
            "Each .pt file is a dict with keys: sample_idx, dataset, token_count, model, layers. "
            "'layers' is a list of dicts keyed by layer_idx, embeddings [seq_len, hidden_size], "
            "router_logits [seq_len, num_experts]. Embeddings are post-attention-norm (pre-router)."
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
        "--num-fineweb", type=int, default=100,
        help="Number of samples from FineWeb (default: 100)."
    )
    parser.add_argument(
        "--num-orca", type=int, default=100,
        help="Number of samples from OpenOrca (default: 100)."
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
        )

    # ── collect Orca ──────────────────────────────────────────────────────────
    orca_collected = 0
    if not args.skip_orca and args.num_orca > 0:
        # Offset sample_idx so filenames don't collide
        orca_collected = collect_from_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name="orca",
            num_samples=args.num_orca,
            output_dir=output_dir,
            model_tag=model_tag,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            start_sample_idx=args.num_fineweb,   # offset so orca files start after fineweb
        )

    # ── write metadata ────────────────────────────────────────────────────────
    write_metadata(
        output_dir=output_dir,
        model_tag=model_tag,
        num_fineweb=fineweb_collected,
        num_orca=orca_collected,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        model_path=model_path,
    )

    total = fineweb_collected + orca_collected
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print(f"  FineWeb: {fineweb_collected} samples")
    print(f"  Orca:    {orca_collected} samples")
    print(f"  Total:   {total} .pt files in {output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
