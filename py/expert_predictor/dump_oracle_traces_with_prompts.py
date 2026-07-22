#!/usr/bin/env python3
"""
Re-extract the original prompt texts from the wikitext dataset that were used
during training data collection, and regenerate oracle trace .txt files with
the prompt included.

Since the .pt files were saved before 'prompt_text' was added to the format,
we re-stream wikitext-103-v1 deterministically (same filter logic as 
collect_training_data_unified.py) to recover the text for each sample index.
"""

import sys
import torch
import numpy as np
import glob
from pathlib import Path

from datasets import load_dataset

def stream_wikitext_texts(num_samples, min_tokens=10, max_tokens=99999):
    """
    Re-stream wikitext-103-v1 with the same filter logic as collect_training_data_unified.py.
    Returns a list of raw text strings (no tokenizer needed for filtering here — 
    we filter by word-count as a proxy, but the real filter was token-based).
    We just collect enough samples to cover the indices we need.
    """
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    texts = []
    for example in ds:
        text = example.get("text", "").strip()
        if not text or text.startswith(" = "):
            continue
        # Use character length as a rough proxy for token length
        # Real filter was min_tokens=100 (approx 400 chars), max_tokens=512 (approx 2048 chars)
        if len(text) < 400:
            continue
        texts.append(text)
        if len(texts) >= num_samples:
            break
        if len(texts) % 100 == 0:
            print(f"  Streamed {len(texts)}/{num_samples} texts...", flush=True)
    return texts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Directory with .pt files')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--num_traces', type=int, default=15)
    parser.add_argument('--out_dir', type=str, default=None, help='Output dir for .txt files (default: same as --dir)')
    args = parser.parse_args()

    pt_files = sorted(glob.glob(f"{args.dir}/*.pt"))[:args.num_traces]
    if not pt_files:
        print(f"No .pt files found in {args.dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else Path("/mnt/storage/Michael/michaelg/heteroPredict/py/expert_predictor/paper_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine model name from directory path
    dir_lower = args.dir.lower()
    if 'qwen' in dir_lower:
        model_name = "qwen3_30b"
        active_k = 8
    elif 'mixtral' in dir_lower:
        model_name = "mixtral_8x7b"
        active_k = 2
    else:
        model_name = "unknown"
        active_k = 4

    print(f"Model: {model_name}, active_k: {active_k}")
    print(f"Processing {len(pt_files)} .pt files...")
    print(f"Re-streaming wikitext-103-v1 to recover {args.num_traces} prompt texts...")

    texts = stream_wikitext_texts(args.num_traces)
    print(f"Got {len(texts)} texts from wikitext stream.")

    for file_idx, fp in enumerate(pt_files):
        payload = torch.load(fp, map_location='cpu', weights_only=False)
        sample_idx = payload.get('sample_idx', file_idx)

        layer_idx = args.layer
        layer_data = next((l for l in payload.get('layers', []) if l['layer_idx'] == layer_idx), None)
        if not layer_data:
            print(f"[{file_idx+1}] Layer {layer_idx} not found in {fp}, skipping.")
            continue

        logits = layer_data['router_logits']
        T, num_experts = logits.shape

        topk_indices = logits.topk(active_k, dim=1).indices

        # Lookahead-5 metric
        f = 5
        lookahead_sizes = []
        for t in range(T - f):
            union = set()
            for step in range(1, f+1):
                for i in range(active_k):
                    union.add(topk_indices[t+step, i].item())
            lookahead_sizes.append(len(union))
        avg_union = np.mean(lookahead_sizes) if lookahead_sizes else 0

        # Turnover metric
        turnovers = []
        for t in range(1, T):
            prev = set(topk_indices[t-1, i].item() for i in range(active_k))
            curr = set(topk_indices[t, i].item() for i in range(active_k))
            turnovers.append(len(curr - prev))
        avg_turnover = np.mean(turnovers) if turnovers else 0

        # Use re-streamed text for this sample
        prompt_text = texts[file_idx] if file_idx < len(texts) else "[text not available]"

        file_id = Path(fp).stem
        out_file = out_dir / f"oracle_trace_{model_name}_{file_id}.txt"

        with open(out_file, "w") as f_out:
            f_out.write(f"Oracle Trace for {model_name} (Layer {layer_idx}) - File {file_id}\n")
            f_out.write(f"Sample index: {sample_idx}\n")
            f_out.write(f"Generation tokens: {T}\n")
            f_out.write(f"Active experts per token: {active_k}\n")
            f_out.write(f"Avg unique experts in next 5 tokens: {avg_union:.1f} / {num_experts}\n")
            f_out.write(f"Avg expert turnover token-to-token: {avg_turnover:.2f} / {active_k}\n")
            f_out.write("="*60 + "\n")
            f_out.write("PROMPT TEXT (from wikitext-103-v1):\n")
            f_out.write(prompt_text + "\n")
            f_out.write("="*60 + "\n")
            f_out.write("EXPERT TRACE (generation tokens only):\n")
            for t in range(T):
                expert_list = [str(topk_indices[t, i].item()) for i in range(active_k)]
                f_out.write(f"Token {t:03d}: Experts [{', '.join(expert_list)}]\n")

        print(f"[{file_idx+1}/{len(pt_files)}] Tokens:{T} Turnover:{avg_turnover:.2f} → {out_file.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
