#!/usr/bin/env python3
"""
Verify that the cross-layer expert predictor outputs raw logits [B, 128],
not a multi-hot vector.  Optionally load a trained checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from expert_predictor_cross_layer import MultiStepExpertPredictor, MODEL_DEFAULTS


def describe_output(name: str, logits: torch.Tensor):
    probs = torch.sigmoid(logits)
    print(f"\n{name}")
    print(f"  shape      : {tuple(logits.shape)}")
    print(f"  dtype      : {logits.dtype}")
    print(f"  logits     : min={logits.min():.4f}  max={logits.max():.4f}  mean={logits.mean():.4f}")
    print(f"  sigmoid    : min={probs.min():.4f}  max={probs.max():.4f}  mean={probs.mean():.4f}")
    print(f"  unique vals (rounded 3dp): {len(torch.unique(logits.round(decimals=3)))}")
    is_binary = torch.all((logits == 0) | (logits == 1))
    print(f"  all 0/1 (multi-hot)?   : {bool(is_binary)}")
    top8 = logits[0].topk(8)
    print(f"  top-8 indices          : {top8.indices.tolist()}")
    print(f"  top-8 logit scores     : {[round(x, 4) for x in top8.values.tolist()]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to best.pt (optional)")
    p.add_argument("--variant", choices=["emb_only", "markov_only", "all_features"],
                   default="all_features")
    p.add_argument("--hidden", type=int, default=32)
    args = p.parse_args()

    emb_dim, n_exp, active_k, n_layers, _, predict_k = MODEL_DEFAULTS["qwen3_30b"]

    flags = {
        "emb_only":      (True,  False, False, False),
        "markov_only":   (False, False, False, True),
        "all_features":  (True,  True,  True,  True),
    }[args.variant]
    use_emb, use_pfill, use_prev, use_markov = flags

    model = MultiStepExpertPredictor(
        history=1, future_steps=8, emb_dim=emb_dim, num_experts=n_exp,
        hidden_dim=args.hidden, use_embedding=use_emb, use_prefill=use_pfill,
        use_prev=use_prev, use_markov=use_markov, layer_idx=0,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state"])
        print(f"Loaded: {args.checkpoint}")
        if "config" in ckpt:
            print(f"Config: {json.dumps(ckpt['config'], indent=2)}")

    model.eval()
    B = 4
    emb = torch.randn(B, emb_dim)
    pfill = torch.softmax(torch.randn(B, n_exp), dim=-1)
    prev = torch.zeros(B, n_exp)
    prev[:, :active_k] = 1.0 / active_k

    with torch.no_grad():
        logits = model(emb, pfill, prev)

    describe_output("Model forward() output (raw logits)", logits)

    # What training / metrics use: top-K indices, not thresholded multi-hot
    pred_idx = logits.topk(12, dim=-1).indices
    pred_mh = torch.zeros_like(logits).scatter_(1, pred_idx, 1.0)
    print("\nDerived cache set (top-12 → multi-hot, only for eval):")
    print(f"  sum per row (should be 12): {pred_mh.sum(dim=1).tolist()}")

    # BCE target path (loss uses logits + multi-hot target internally)
    fake_target = torch.zeros(B, n_exp)
    fake_target[:, :20] = 1.0
    loss = F.binary_cross_entropy_with_logits(logits, fake_target)
    print(f"\nBCEWithLogitsLoss (training loss) on fake union target: {loss.item():.4f}")
    print("\nConclusion: forward() returns 1×128 real-valued logits (router-style scores).")
    print("Multi-hot appears only in targets and when top-K is applied for cache selection.")


if __name__ == "__main__":
    main()
