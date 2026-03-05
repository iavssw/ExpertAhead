#!/usr/bin/env python3
"""
expert_predictor.py — Unified Expert Predictor
==============================================
A single script for training, sweeping, and ablation studies to predict 
the top-1 router expert at token t+1.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# (emb_dim, num_experts, active_k, num_layers, prefetch_k)
MODEL_DEFAULTS = {
    "mixtral_8x7b":  (4096, 8,   2, 32, 1),
    "mixtral_8x22b": (4096, 8,   2, 56, 1),
    "qwen3_30b":     (2048, 128, 8, 48, 4),
    "qwen3_480b":    (7168, 128, 8, 94, 4),
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. Model & Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ExpertPredictor(nn.Module):
    """Unified predictor handling single tokens, history, and prefill branches."""
    def __init__(
        self, 
        hidden_size: int = 4096, 
        num_experts: int = 8, 
        hidden_dim: int = 256, 
        dropout: float = 0.1,
        use_embedding: bool = True, 
        use_prefill: bool = True, 
        use_prev: bool = False
    ):
        super().__init__()
        self.config = {
            'hidden_size': hidden_size, 'num_experts': num_experts, 
            'hidden_dim': hidden_dim, 'use_embedding': use_embedding, 
            'use_prefill': use_prefill, 'use_prev': use_prev
        }
        self.use_embedding, self.use_prefill, self.use_prev = use_embedding, use_prefill, use_prev
        self.num_experts = num_experts

        fused_dim = 0
        if use_embedding:
            self.emb_branch = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
            )
            fused_dim += hidden_dim
        
        if use_prefill:
            p_dim = max(hidden_dim // 4, 8)
            self.pfill_branch = nn.Sequential(nn.Linear(num_experts, p_dim), nn.GELU())
            fused_dim += p_dim
            
        if use_prev:
            p_dim = max(hidden_dim // 4, 8)
            self.prev_branch = nn.Sequential(nn.Linear(num_experts, p_dim), nn.GELU())
            fused_dim += p_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_experts)
        )

    def forward(self, embedding, prefill_dist=None, prev_onehot=None):
        feats = []
        if self.use_embedding:
            feats.append(self.emb_branch(embedding))
        if self.use_prefill:
            if prefill_dist is None:
                prefill_dist = torch.full((embedding.size(0), self.num_experts), 1/self.num_experts, device=embedding.device)
            feats.append(self.pfill_branch(prefill_dist))
        if self.use_prev:
            if prev_onehot is None:
                prev_onehot = torch.zeros((embedding.size(0), self.num_experts), device=embedding.device)
            feats.append(self.prev_branch(prev_onehot))
            
        return self.classifier(torch.cat(feats, dim=-1))

    def save(self, path: Path):
        torch.save({'state': self.state_dict(), 'config': self.config}, path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

class JITWrapper(nn.Module):
    """Ensures a fixed (emb, pfill, prev) signature for C++ LibTorch."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, emb, pfill, prev):
        return self.model(emb, pfill, prev)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ExpertDataset(Dataset):
    def __init__(self, files, layer_idx, num_experts, history):
        self.samples = []
        self.num_experts = num_experts
        
        for fp in tqdm(files, desc=f"Loading L{layer_idx}"):
            payload = torch.load(fp, map_location='cpu', weights_only=False)
            layer_data = next((l for l in payload.get('layers', []) if l['layer_idx'] == layer_idx), None)
            if not layer_data: continue

            embs, logits = layer_data['embeddings'], layer_data['router_logits']
            prev_ids = layer_data.get('prev_expert_ids')
            
            # Extract or compute prefill distribution
            if 'prefill_expert_dist' in layer_data:
                pfill = layer_data['prefill_expert_dist']
            elif 'prefill_expert_count' in layer_data:
                c = layer_data['prefill_expert_count'].float()
                pfill = c / c.sum().clamp(min=1)
            else:
                pfill = torch.softmax(logits.float(), dim=-1).mean(dim=0)

            for t in range(len(embs) - 1):
                # Build history window
                window = []
                for h in range(history):
                    src = t - (history - 1 - h)
                    window.append(embs[src] if src >= 0 else torch.zeros_like(embs[0]))
                
                # Previous expert one-hot (feature for t+1)
                prev_oh = torch.zeros(num_experts)
                prev_oh[prev_ids[t, 0] if prev_ids is not None else logits[t].argmax()] = 1.0

                self.samples.append({
                    'emb': torch.cat(window),
                    'target': logits[t+1].argmax(),
                    'pfill': pfill,
                    'prev': prev_oh
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

# ──────────────────────────────────────────────────────────────────────────────
# 3. Training Logic (The "Engine")
# ──────────────────────────────────────────────────────────────────────────────

def run_predictor_task(args, layer_idx, hidden_dim, history, 
                       use_emb=True, use_pfill=True, use_prev=False, 
                       custom_name=None):
    """Central engine for training a single model configuration."""
    
    # 1. Setup paths
    config_name = custom_name or f"eh{history}_h{hidden_dim}"
    out_dir = Path(args.output_dir) / config_name / f"layer_{layer_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Prepare Data
    files = sorted(list(Path(args.data_dir).glob("*.pt")))
    random.seed(42); random.shuffle(files)
    split = int(0.8 * len(files))
    
    train_ds = ExpertDataset(files[:split], layer_idx, args.n_exp, history)
    val_ds = ExpertDataset(files[split:] or [files[-1]], layer_idx, args.n_exp, history)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Init Model
    model = ExpertPredictor(
        hidden_size=args.emb_dim * history, num_experts=args.n_exp, 
        hidden_dim=hidden_dim, use_embedding=use_emb, 
        use_prefill=use_pfill, use_prev=use_prev
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc, logs = 0.0, []

    # 4. Loop
    for epoch in range(args.epochs):
        for phase in ['train', 'val']:
            model.train(phase == 'train')
            hits, total, loss_sum = 0, 0, 0.0
            
            with torch.set_grad_enabled(phase == 'train'):
                loader = train_loader if phase == 'train' else val_loader
                for b in tqdm(loader, desc=f"L{layer_idx} Ep{epoch+1} {phase}"):
                    b = {k: v.to(args.device) for k, v in b.items()}
                    logits = model(b['emb'], b['pfill'], b['prev'])
                    loss = criterion(logits, b['target'])
                    
                    if phase == 'train':
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                        
                    loss_sum += loss.item() * len(b['target'])
                    hits += (logits.argmax(-1) == b['target']).sum().item()
                    total += len(b['target'])
            
            metrics = {'loss': loss_sum / total, 'acc': hits / total}
            if phase == 'val':
                scheduler.step(metrics['acc'])
                logs.append({'epoch': epoch+1, 'val': metrics})
                if metrics['acc'] > best_acc:
                    best_acc = metrics['acc']
                    model.save(out_dir / "best.pt")
                    # JIT Trace for C++
                    try:
                        wrapper = JITWrapper(model).eval()
                        ex_emb = torch.randn(1, args.emb_dim*history).to(args.device)
                        ex_pf = torch.zeros(1, args.n_exp).to(args.device)
                        ex_pr = torch.zeros(1, args.n_exp).to(args.device)
                        torch.jit.trace(wrapper, (ex_emb, ex_pf, ex_pr)).save(str(out_dir / "best_jit.pt"))
                    except Exception as e: print(f"JIT failed: {e}")
                
                print(f"  {phase.capitalize()} Acc: {metrics['acc']:.4f} (Best: {best_acc:.4f})")

    with open(out_dir / "summary.json", 'w') as f:
        json.dump({'best_acc': best_acc, 'history': logs}, f, indent=2)
    
    return best_acc

# ──────────────────────────────────────────────────────────────────────────────
# 4. CLI & Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def resolve_args(args):
    """Inject model defaults if a model name was provided."""
    if args.model:
        d = MODEL_DEFAULTS[args.model]
        args.emb_dim, args.n_exp, args.active_k, args.n_layers, args.pf_k = d
    else:
        # Defaults if no model specified
        args.emb_dim, args.n_exp, args.n_layers = 4096, 8, 32
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Base Parser
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument('--data_dir', required=True)
    base.add_argument('--output_dir', required=True)
    base.add_argument('--model', choices=MODEL_DEFAULTS.keys())
    base.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    base.add_argument('--epochs', type=int, default=10)
    base.add_argument('--batch_size', type=int, default=128)
    base.add_argument('--lr', type=float, default=1e-3)

    # Commands
    p_train = subparsers.add_parser('train', parents=[base])
    p_train.add_argument('--layer', type=int)
    p_train.add_argument('--history', type=int, default=1)
    p_train.add_argument('--hidden', type=int, default=128)

    p_sweep = subparsers.add_parser('sweep', parents=[base])
    p_sweep.add_argument('--hiddens', nargs='+', type=int, default=[64, 128, 256])
    p_sweep.add_argument('--histories', nargs='+', type=int, default=[1, 2])
    p_sweep.add_argument('--layers', nargs='+', type=int)
    p_sweep.add_argument('--max_mb', type=float, default=70.0)

    p_abl = subparsers.add_parser('ablation', parents=[base])
    p_abl.add_argument('--layers', nargs='+', type=int, default=[15])
    p_abl.add_argument('--hidden', type=int, default=256)

    args = resolve_args(parser.parse_args())

    if args.command == 'train':
        layers = [args.layer] if args.layer is not None else list(range(args.n_layers))
        for l in layers:
            run_predictor_task(args, l, args.hidden, args.history)

    elif args.command == 'sweep':
        layers = args.layers if args.layers else list(range(args.n_layers))
        for hists in args.histories:
            for h in args.hiddens:
                # Basic size filter: (weights * 4 bytes) / 1MB
                size = ((args.emb_dim*hists*h + h*h//2 + h//2*args.n_exp) * 4) / 1e6
                if size > args.max_mb: continue
                for l in layers:
                    run_predictor_task(args, l, h, hists)

    elif args.command == 'ablation':
        variants = [
            ("emb_only", True, False, False),
            ("pfill_only", False, True, False),
            ("prev_only", False, False, True),
            ("all_features", True, True, True),
        ]
        for l in args.layers:
            for name, e, pf, pr in variants:
                run_predictor_task(args, l, args.hidden, 1, 
                                   use_emb=e, use_pfill=pf, use_prev=pr, 
                                   custom_name=f"ablation_{name}")
