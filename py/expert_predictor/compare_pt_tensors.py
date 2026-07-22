#!/usr/bin/env python3
"""
Compare re-run .pt files against original .pt files.
Checks router_logits (allclose) and prev_expert_ids (exact match) for all layers.
"""
import sys
import torch
import glob
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--orig',   required=True, help='Original .pt dir')
parser.add_argument('--rerun',  required=True, help='Re-run .pt dir')
parser.add_argument('--n',      type=int, default=3)
parser.add_argument('--atol',   type=float, default=1e-4, help='Tolerance for logit comparison')
args = parser.parse_args()

orig_files   = sorted(glob.glob(f"{args.orig}/wikitext_*.pt"))[:args.n]
verify_files = sorted(glob.glob(f"{args.rerun}/wikitext_*.pt"))[:args.n]

print(f"Comparing {args.n} files pair-wise:")
print(f"  Original : {args.orig}")
print(f"  Re-run   : {args.rerun}")
print()

if len(orig_files) < args.n:
    print(f"ERROR: only {len(orig_files)} original files found (expected {args.n})")
    sys.exit(1)
if len(verify_files) < args.n:
    print(f"ERROR: only {len(verify_files)} re-run files found (expected {args.n})")
    sys.exit(1)

all_pass = True
for i, (ofp, vfp) in enumerate(zip(orig_files, verify_files)):
    orig  = torch.load(ofp, map_location='cpu', weights_only=False)
    rerun = torch.load(vfp, map_location='cpu', weights_only=False)

    orig_sidx  = orig.get('sample_idx', '?')
    rerun_sidx = rerun.get('sample_idx', '?')

    print(f"[{i+1}/{args.n}] {Path(ofp).name} (orig idx={orig_sidx}) vs {Path(vfp).name} (rerun idx={rerun_sidx})")

    if orig_sidx != rerun_sidx:
        print(f"  ✗ sample_idx mismatch: {orig_sidx} vs {rerun_sidx}")
        all_pass = False
        continue

    # Token count
    orig_tc  = orig.get('token_count', 0)
    rerun_tc = rerun.get('token_count', 0)
    tc_match = (orig_tc == rerun_tc)
    print(f"  token_count  orig={orig_tc}  rerun={rerun_tc}  {'✓' if tc_match else '✗ MISMATCH'}")
    if not tc_match:
        all_pass = False
        continue

    # Prompt text (if both have it)
    if 'prompt_text' in orig and 'prompt_text' in rerun:
        pt_match = (orig['prompt_text'] == rerun['prompt_text'])
        print(f"  prompt_text  {'✓ MATCH' if pt_match else '✗ MISMATCH'}")
        if not pt_match:
            print(f"    orig:  {repr(orig['prompt_text'][:100])}")
            print(f"    rerun: {repr(rerun['prompt_text'][:100])}")
            all_pass = False

    # Per-layer tensor comparison
    orig_layers  = {l['layer_idx']: l for l in orig['layers']}
    rerun_layers = {l['layer_idx']: l for l in rerun['layers']}
    common_layers = sorted(set(orig_layers) & set(rerun_layers))

    layer_failures = []
    for lid in common_layers:
        ol = orig_layers[lid]
        rl = rerun_layers[lid]

        rl_match = torch.allclose(ol['router_logits'].float(), rl['router_logits'].float(), atol=args.atol)

        # Derive top-k expert selections from logits and compare
        # (This is what actually matters for the oracle trace)
        # Infer active_k from prev_expert_ids if available, else use 2 for Mixtral, 8 for Qwen
        if 'prev_expert_ids' in ol:
            active_k = ol['prev_expert_ids'].shape[-1]
        else:
            active_k = 2  # default Mixtral

        orig_topk  = ol['router_logits'].float().topk(active_k, dim=-1).indices
        rerun_topk = rl['router_logits'].float().topk(active_k, dim=-1).indices
        # Sort so order doesn't matter
        orig_topk_sorted  = orig_topk.sort(dim=-1).values
        rerun_topk_sorted = rerun_topk.sort(dim=-1).values
        topk_match = torch.equal(orig_topk_sorted, rerun_topk_sorted)

        # prev_expert_ids was added later — only compare if both files have it
        if 'prev_expert_ids' in ol and 'prev_expert_ids' in rl:
            ei_match = torch.equal(ol['prev_expert_ids'], rl['prev_expert_ids'])
        elif 'prev_expert_ids' not in ol and 'prev_expert_ids' not in rl:
            ei_match = True  # Both missing — consistent
        else:
            ei_match = None  # One has it, one doesn't — flag it

        failed = (not rl_match) or (ei_match is False) or (ei_match is None) or (not topk_match)
        if failed:
            diff = (ol['router_logits'].float() - rl['router_logits'].float()).abs()
            # Count how many tokens have at least one expert mismatch
            mismatch_tokens = (~torch.all(orig_topk_sorted == rerun_topk_sorted, dim=-1)).sum().item()
            layer_failures.append({
                'lid': lid,
                'rl': rl_match,
                'topk': topk_match,
                'ei': ei_match,
                'max_diff': diff.max().item(),
                'mean_diff': diff.mean().item(),
                'mismatch_tokens': mismatch_tokens,
                'total_tokens': orig_topk.shape[0],
            })

    if layer_failures:
        for f in layer_failures:
            print(f"  Layer {f['lid']:2d}: ✗  logits_match={f['rl']}  topk_match={f['topk']}  "
                  f"expert_ids_match={f['ei']}  max_diff={f['max_diff']:.4f}  "
                  f"token_mismatches={f['mismatch_tokens']}/{f['total_tokens']}")
        all_pass = False
    else:
        print(f"  All {len(common_layers)} layers: router_logits ✓  top-k experts ✓  prev_expert_ids ✓")

    print()

print("=" * 60)
if all_pass:
    print("✓ VERIFICATION PASSED — router logits and expert IDs match exactly!")
else:
    print("✗ VERIFICATION FAILED — some tensors did not match.")
    sys.exit(1)
print("=" * 60)
