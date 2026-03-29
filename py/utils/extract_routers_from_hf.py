import os
import torch
from safetensors import safe_open

MODELS = {
    "mixtral_8x7b": os.path.expanduser("~/.cache/huggingface/hub/models--TheBloke--mixtral-8x7b-v0.1-AWQ/snapshots"),
    "qwen3_30b": os.path.expanduser("~/.cache/huggingface/hub/models--QuixiAI--Qwen3-30B-A3B-AWQ/snapshots"),
}

OUTPUT_BASE = "/mnt/storage/Michael/michaelg/heteroPredict/trainingData"

for model_name, hub_path in MODELS.items():
    if not os.path.exists(hub_path):
        print(f"Skipping {model_name}, path {hub_path} not found.")
        continue
    
    # Get the latest snapshot
    snapshots = [d for d in os.listdir(hub_path) if os.path.isdir(os.path.join(hub_path, d))]
    if not snapshots:
        print(f"No snapshots for {model_name}")
        continue
    
    snapshot_dir = os.path.join(hub_path, snapshots[0])
    
    out_dir = os.path.join(OUTPUT_BASE, model_name, "router_bins")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n--- Extracting {model_name} from {snapshot_dir} ---")
    st_files = [f for f in os.listdir(snapshot_dir) if f.endswith(".safetensors")]
    pt_files = [f for f in os.listdir(snapshot_dir) if f.endswith(".bin") and f.startswith("pytorch_model")]
    
    count = 0
    if st_files:
        for stf in st_files:
            try:
                with safe_open(os.path.join(snapshot_dir, stf), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if "gate" in key and "weight" in key:
                            # Mixtral: block_sparse_moe.gate.weight
                            # Qwen might have a different name, standard is gate.weight
                            parts = key.split('.')
                            if "layers" in parts:
                                idx = parts.index("layers")
                                layer_idx = parts[idx + 1]
                                W = f.get_tensor(key)
                                out_file = os.path.join(out_dir, f"layer_{layer_idx}_gate.pt")
                                torch.save(W.clone().half(), out_file)
                                print(f"Saved {out_file} shape {W.shape}")
                                count += 1
            except Exception as e:
                print(f"Error reading {stf}: {e}")
    elif pt_files:
        for ptf in pt_files:
            try:
                state_dict = torch.load(os.path.join(snapshot_dir, ptf), map_location="cpu", mmap=True)
                for key, W in state_dict.items():
                    if "gate" in key and "weight" in key:
                        parts = key.split('.')
                        if "layers" in parts:
                            idx = parts.index("layers")
                            layer_idx = parts[idx + 1]
                            out_file = os.path.join(out_dir, f"layer_{layer_idx}_gate.pt")
                            torch.save(W.clone().half(), out_file)
                            print(f"Saved {out_file} shape {W.shape}")
                            count += 1
                del state_dict
            except Exception as e:
                print(f"Error reading {ptf}: {e}")
    else:
        print(f"No valid format found in {snapshot_dir}")
        
    print(f"Total gate weights extracted for {model_name}: {count}")
