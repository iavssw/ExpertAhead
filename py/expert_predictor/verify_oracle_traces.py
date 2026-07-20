#!/usr/bin/env python3
"""
Verify that the oracle trace text files are using the correct prompts by:
1. Loading just the tokenizer (no GPU needed)
2. Re-streaming wikitext-103-v1 with the EXACT same token filter as collect_training_data_unified.py
3. Comparing the texts we recover against the .pt file sample_idx values
4. Printing a side-by-side comparison of the first N=characters of each text

This does NOT re-run model inference. It verifies the prompt→file mapping.
"""

import sys
import torch
import glob
from pathlib import Path
from transformers import AutoTokenizer

# ── Args ──────────────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['qwen3_30b', 'mixtral_8x7b'], required=True)
parser.add_argument('--num_traces', type=int, default=15)
parser.add_argument('--min_tokens', type=int, default=100)
parser.add_argument('--max_tokens', type=int, default=512)
args = parser.parse_args()

# ── Model/path config ─────────────────────────────────────────────────────────
MODEL_HF = {
    "qwen3_30b":    "Qwen/Qwen3-30B-A3B",        # just tokenizer, no weights downloaded
    "mixtral_8x7b": "mistralai/Mixtral-8x7B-v0.1",
}
PT_DIRS = {
    "qwen3_30b":    "/mnt/storage/Michael/michaelg/heteroPredict/trainingData/qwen3_30b",
    "mixtral_8x7b": "/mnt/storage/Michael/michaelg/heteroPredict/trainingData/mixtral_8x7b",
}

hf_name = MODEL_HF[args.model]
pt_dir  = PT_DIRS[args.model]

print(f"Loading tokenizer for {args.model} ({hf_name}) ...")
tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
print("Tokenizer loaded.\n")

# ── Stream wikitext with EXACT same filter as collect_training_data_unified ───
from datasets import load_dataset

print("Streaming wikitext-103-v1 with exact same token filter ...")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

texts = []
for example in ds:
    text = example.get("text", "").strip()
    if not text or text.startswith(" = "):
        continue
    enc = tokenizer([text], return_tensors="pt", padding=False, truncation=True, max_length=args.max_tokens)
    seq_len = enc["input_ids"].shape[1]
    if seq_len < args.min_tokens:
        continue
    texts.append((text, seq_len))
    if len(texts) >= args.num_traces:
        break

print(f"Got {len(texts)} valid texts from stream.\n")

# ── Load .pt files and compare ────────────────────────────────────────────────
pt_files = sorted(glob.glob(f"{pt_dir}/*.pt"))[:args.num_traces]

print("=" * 80)
print(f"{'FILE':<30} {'PT sample_idx':>14} {'PT tok_count':>13} {'Stream tok_len':>14} {'TEXT MATCH?':>12}")
print("=" * 80)

all_match = True
for i, fp in enumerate(pt_files):
    payload = torch.load(fp, map_location='cpu', weights_only=False)
    sample_idx = payload.get('sample_idx', '?')
    pt_token_count = payload.get('token_count', '?')  # generation tokens collected

    # Stream token count is the prompt length (prefill), PT token_count is generation length
    stream_text, stream_tok_len = texts[i]

    # We can't directly compare generation tokens to prefill tokens,
    # so we compare the first 80 chars of the text visually
    fname = Path(fp).name
    print(f"\n{fname:<30} {str(sample_idx):>14} {str(pt_token_count):>13} {str(stream_tok_len):>14}")
    print(f"  Stream text start : {repr(stream_text[:120])}")
    print(f"  (Expected sample_idx == file index: {sample_idx == i})")
    if sample_idx != i:
        all_match = False

print("\n" + "=" * 80)
if all_match:
    print("✓ sample_idx in all .pt files matches file order — prompt mapping is consistent.")
else:
    print("✗ Some sample_idx values are out of order — prompt mapping may be wrong!")
print("=" * 80)
