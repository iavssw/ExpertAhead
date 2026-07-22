import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import argparse
import sys
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/mnt/storage/Michael/michaelg/heteroPredict/trainingData/qwen3_30b")
parser.add_argument('--layer', type=int, default=0)
parser.add_argument('--num_traces', type=int, default=15)
args = parser.parse_args()

files = sorted(glob.glob(f"{args.dir}/*.pt"))[:args.num_traces]

for fp_idx, fp in enumerate(files):
    print(f"\n[{fp_idx+1}/{args.num_traces}] Loading {fp} ...")
    payload = torch.load(fp, map_location='cpu', weights_only=False)

    # Let's inspect the requested layer's trace
    layer_idx = args.layer
    layer_data = next((l for l in payload.get('layers', []) if l['layer_idx'] == layer_idx), None)

    if not layer_data:
        print(f"Layer {layer_idx} not found in the file.")
        continue

    logits = layer_data['router_logits'] # [T, num_experts]
    T, num_experts = logits.shape

    # Automatically detect active_k
    if 'qwen' in fp.lower():
        active_k = 8
        model_name = "qwen3_30b"
    elif 'mixtral' in fp.lower():
        active_k = 2
        model_name = "mixtral_8x7b"
    else:
        active_k = 4 # default fallback
        model_name = "unknown"

    topk_indices = logits.topk(active_k, dim=1).indices # [T, active_k]

    # Calculate how many experts we'd need to load if we looked ahead `f` steps
    f = 5
    lookahead_union_sizes = []
    for t in range(T - f):
        union_set = set()
        for step in range(1, f + 1):
            for i in range(active_k):
                union_set.add(topk_indices[t + step, i].item())
        lookahead_union_sizes.append(len(union_set))
    avg_union = np.mean(lookahead_union_sizes) if lookahead_union_sizes else 0

    # Calculate average expert turnover token-to-token
    turnovers = []
    for t in range(1, T):
        prev_experts = set(topk_indices[t-1, i].item() for i in range(active_k))
        curr_experts = set(topk_indices[t, i].item() for i in range(active_k))
        new_experts = len(curr_experts - prev_experts)
        turnovers.append(new_experts)
    avg_turnover = np.mean(turnovers) if turnovers else 0

    print(f"Tokens: {T} | Turnover: {avg_turnover:.2f} | Avg {f}-lookahead: {avg_union:.1f}")

    # Write per-token expert trace to a text file
    file_id = Path(fp).stem
    txt_out_file = f"/mnt/storage/Michael/michaelg/heteroPredict/py/expert_predictor/paper_plots/oracle_trace_{model_name}_{file_id}.txt"
    prompt_text    = payload.get('prompt_text', '[not stored]')
    generated_text = payload.get('generated_text', '[not stored]')
    with open(txt_out_file, "w") as f_out:
        f_out.write(f"Oracle Trace for {model_name} (Layer {layer_idx}) - File {file_id}\n")
        f_out.write(f"Prompt Length: {T} tokens\n")
        f_out.write(f"Active experts per token: {active_k}\n")
        f_out.write(f"Avg unique experts in next {f} tokens: {avg_union:.1f} / {num_experts}\n")
        f_out.write(f"Avg expert turnover token-to-token: {avg_turnover:.2f} / {active_k}\n")
        f_out.write("="*50 + "\n")
        f_out.write("PROMPT TEXT:\n")
        f_out.write(prompt_text + "\n")
        f_out.write("-"*50 + "\n")
        f_out.write("GENERATED TEXT:\n")
        f_out.write(generated_text + "\n")
        f_out.write("="*50 + "\n")
        f_out.write("EXPERT TRACE (generation tokens only):\n")
        for t in range(T):
            expert_list = [str(topk_indices[t, i].item()) for i in range(active_k)]
            f_out.write(f"Token {t:03d}: Experts [{', '.join(expert_list)}]\n")

