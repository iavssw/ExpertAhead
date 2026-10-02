#!/usr/bin/env python3
import os
import sys
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def process_file(args):
    src_fp, dst_root = args
    try:
        payload = torch.load(src_fp, map_location='cpu', weights_only=False)
        for l_data in payload.get('layers', []):
            l_idx = l_data['layer_idx']
            l_dir = dst_root / f"layer_{l_idx}"
            l_dir.mkdir(parents=True, exist_ok=True)
            dst_fp = l_dir / src_fp.name
            torch.save({'layers': [l_data]}, dst_fp)
        return True
    except Exception as e:
        print(f"Error processing {src_fp.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Shard dataset by layer for faster loading.")
    parser.add_argument("--src", type=str, default="../../trainingDataExtended/qwen3_30b")
    parser.add_argument("--dst", type=str, default="../../trainingDataExtended/qwen3_30b_sharded")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--include_datasets",
        nargs="+",
        default=None,
        help="Only shard files named {dataset}_*.pt (e.g. fineweb orca gsm8k). "
             "Default: all .pt files in --src.",
    )
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    
    files = list(src_root.glob("*.pt"))
    if args.include_datasets:
        prefixes = tuple(f"{d}_" for d in args.include_datasets)
        files = [f for f in files if f.name.startswith(prefixes)]
        print(
            f"Filtered to {len(files)} files matching {list(args.include_datasets)} "
            f"from {src_root}"
        )
    print(f"Found {len(files)} files to shard from {src_root} to {dst_root}...")
    if not files:
        print("No matching .pt files; nothing to do.")
        return

    args_list = [(fp, dst_root) for fp in files]
    
    success = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_file, a): a for a in args_list}
        for fut in tqdm(as_completed(futures), total=len(files)):
            if fut.result():
                success += 1
                
    print(f"Successfully sharded {success} / {len(files)} files.")

if __name__ == '__main__':
    main()
