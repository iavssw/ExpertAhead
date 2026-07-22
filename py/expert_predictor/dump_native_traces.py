#!/usr/bin/env python3
import torch
import glob
import os
from pathlib import Path

# Config
OUT_DIR = "/mnt/storage/Michael/michaelg/heteroPredict/py/expert_predictor/paper_plots"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "mixtral_8x7b": "/mnt/storage/Michael/michaelg/heteroPredict/trainingData/fresh_traces_mixtral_8x7b",
    "qwen3_30b": "/mnt/storage/Michael/michaelg/heteroPredict/trainingData/fresh_traces_qwen3_30b"
}

for model_name, pt_dir in MODELS.items():
    pt_files = sorted(glob.glob(f"{pt_dir}/*.pt"))
    print(f"Found {len(pt_files)} for {model_name}")
    
    for i, pt_file in enumerate(pt_files):
        payload = torch.load(pt_file, map_location="cpu", weights_only=False)
        
        prompt_text = payload.get("prompt_text", "<MISSING PROMPT>")
        generated_text = payload.get("generated_text", "<MISSING GEN>")
        token_count = payload.get("token_count", 0)
        layers = payload["layers"]
        
        # Sort layers to ensure correct order
        layers = sorted(layers, key=lambda x: x["layer_idx"])
        num_layers = len(layers)
        
        # Expert traces
        active_k = layers[0]["prev_expert_ids"].shape[1] if "prev_expert_ids" in layers[0] else 2
        
        # Reconstruct expert traces shape: [token_count, num_layers, active_k]
        # Or just write token-by-token
        
        out_file = f"{OUT_DIR}/oracle_trace_{model_name}_{i:05d}.txt"
        
        with open(out_file, "w") as f:
            f.write(f"MODEL: {model_name}\n")
            f.write(f"TOKEN_COUNT: {token_count}\n")
            f.write(f"ACTIVE_EXPERTS: {active_k}\n")
            f.write("="*80 + "\n")
            f.write("PROMPT TEXT:\n")
            f.write(prompt_text + "\n")
            f.write("="*80 + "\n")
            f.write("GENERATED TEXT:\n")
            f.write(generated_text + "\n")
            f.write("="*80 + "\n")
            f.write("EXPERT TRACE (Per token, list of layer expert IDs):\n")
            f.write("="*80 + "\n")
            
            for t in range(token_count):
                f.write(f"Token {t:3d}: ")
                layer_experts = []
                for layer in layers:
                    if "prev_expert_ids" in layer:
                        experts = layer["prev_expert_ids"][t].tolist()
                    else:
                        # Fallback to deriving from topk of router_logits
                        logits = layer["router_logits"][t]
                        experts = logits.topk(active_k, dim=-1).indices.tolist()
                    experts.sort() # Sort to have deterministic representation
                    layer_experts.append("[" + ",".join(map(str, experts)) + "]")
                f.write(" ".join(layer_experts) + "\n")
                
        print(f"  Wrote {out_file}")

print("All done!")
