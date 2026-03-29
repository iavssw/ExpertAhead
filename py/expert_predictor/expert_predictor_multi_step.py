#!/usr/bin/env python3
"""
expert_predictor_multi_step.py — Unified Multi-Step Expert Predictor
====================================================================
Trains a lightweight MLP or Transformer to predict the top-K router experts
for the next `N` tokens (t+1, t+2, ..., t+N), given features from token t.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

MODEL_DEFAULTS = {
    "mixtral_8x7b":  (4096, 8,   2, 32, 1, 1),
    "mixtral_8x22b": (4096, 8,   2, 56, 1, 1),
    "qwen3_30b":     (2048, 128, 8, 48, 4, 4),
    "qwen3_480b":    (7168, 128, 8, 94, 4, 4),
}

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.net(self.ln(x))

class MultiStepExpertPredictor(nn.Module):
    def __init__(
        self,
        history: int = 1,
        future_steps: int = 2,
        emb_dim: int = 4096,
        num_experts: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_embedding: bool = True,
        use_prefill: bool = True,
        use_prev: bool = True,
        use_markov: bool = False,
        transition_matrix: Optional[torch.Tensor] = None,
        noise_std: float = 0.01
    ):
        super().__init__()
        self.config = {
            'arch': 'mlp',
            'history': history, 'future_steps': future_steps,
            'emb_dim': emb_dim, 'hidden_size': history * emb_dim,
            'num_experts': num_experts, 'hidden_dim': hidden_dim,
            'use_embedding': use_embedding, 'use_prefill': use_prefill,
            'use_prev': use_prev, 'use_markov': use_markov, 'noise_std': noise_std
        }
        self.history = history
        self.future_steps = future_steps
        self.use_embedding = use_embedding
        self.use_prefill   = use_prefill
        self.use_prev      = use_prev
        self.use_markov    = use_markov
        self.num_experts   = num_experts
        self.noise_std     = noise_std

        if transition_matrix is not None:
            self.register_buffer("transition_matrix", transition_matrix)
        else:
            self.register_buffer("transition_matrix", torch.zeros((num_experts, num_experts)))

        fused_dim = 0
        hidden_size = history * emb_dim

        if use_embedding:
            self.emb_branch = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
            )
            fused_dim += hidden_dim

        if use_prefill:
            p_dim = max(hidden_dim // 4, 16)
            self.pfill_branch = nn.Sequential(
                nn.Linear(num_experts, p_dim),
                nn.LayerNorm(p_dim), nn.GELU()
            )
            fused_dim += p_dim

        if use_prev:
            p_dim = max(hidden_dim // 4, 16)
            self.prev_branch = nn.Sequential(
                nn.Linear(num_experts, p_dim),
                nn.LayerNorm(p_dim), nn.GELU()
            )
            fused_dim += p_dim

        if use_markov:
            p_dim = max(hidden_dim // 4, 16)
            self.markov_branch = nn.Sequential(
                nn.Linear(num_experts, p_dim),
                nn.LayerNorm(p_dim), nn.GELU()
            )
            fused_dim += p_dim

        self.post_fusion_norm = nn.LayerNorm(fused_dim)
        
        # Predicts a single Union multi-hot target
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            ResidualBlock(fused_dim, dropout),
            nn.Linear(fused_dim, num_experts)
        )

    def forward(self, embedding, prefill_dist=None, prev_onehot=None):
        B = embedding.size(0)

        if self.training and self.noise_std > 0:
            embedding = embedding + torch.randn_like(embedding) * self.noise_std

        feats = []

        if self.use_embedding:
            feats.append(self.emb_branch(embedding))
        if self.use_prefill:
            if prefill_dist is None:
                prefill_dist = torch.full((B, self.num_experts), 1.0 / self.num_experts, device=embedding.device)
            feats.append(self.pfill_branch(prefill_dist))
        if self.use_prev:
            if prev_onehot is None:
                prev_onehot = torch.zeros(B, self.num_experts, device=embedding.device)
            feats.append(self.prev_branch(prev_onehot))
        if self.use_markov:
            if prev_onehot is None:
                prev_onehot = torch.zeros(B, self.num_experts, device=embedding.device)
            markov_prior = prev_onehot @ self.transition_matrix
            feats.append(self.markov_branch(markov_prior))

        fused = self.post_fusion_norm(torch.cat(feats, dim=-1))
        logits = self.classifier(fused)
        return logits

    def save(self, path: Path):
        torch.save({'state': self.state_dict(), 'config': self.config}, path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.config, f, indent=2)


class TransformerMultiStepPredictor(nn.Module):
    def __init__(
        self,
        history: int = 1,
        future_steps: int = 2,
        emb_dim: int = 4096,
        num_experts: int = 8,
        hidden_dim: int = 256,
        tx_layers: int = 2,
        tx_heads: int = 4,
        dropout: float = 0.1,
        use_embedding: bool = True,
        use_prefill: bool = True,
        use_prev: bool = True,
        use_markov: bool = False,
        transition_matrix: Optional[torch.Tensor] = None,
        noise_std: float = 0.01
    ):
        super().__init__()
        self.config = {
            'arch': 'transformer',
            'history': history, 'future_steps': future_steps,
            'emb_dim': emb_dim, 'hidden_size': history * emb_dim, 
            'num_experts': num_experts, 'hidden_dim': hidden_dim,
            'tx_layers': tx_layers, 'tx_heads': tx_heads,
            'use_embedding': use_embedding, 'use_prefill': use_prefill,
            'use_prev': use_prev, 'use_markov': use_markov, 'noise_std': noise_std
        }
        self.history = history
        self.future_steps = future_steps
        self.emb_dim = emb_dim
        self.use_embedding = use_embedding
        self.use_prefill   = use_prefill
        self.use_prev      = use_prev
        self.use_markov    = use_markov
        self.num_experts   = num_experts
        self.noise_std     = noise_std

        if transition_matrix is not None:
            self.register_buffer("transition_matrix", transition_matrix)
        else:
            self.register_buffer("transition_matrix", torch.zeros((num_experts, num_experts)))

        fused_dim = 0

        if use_embedding:
            self.emb_proj = nn.Linear(emb_dim, hidden_dim)
            self.pos_emb = nn.Parameter(torch.zeros(history, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=tx_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tx_layers)
            fused_dim += hidden_dim

        if use_prefill:
            p_dim = max(hidden_dim // 4, 16)
            self.pfill_branch = nn.Sequential(nn.Linear(num_experts, p_dim), nn.LayerNorm(p_dim), nn.GELU())
            fused_dim += p_dim

        if use_prev:
            p_dim = max(hidden_dim // 4, 16)
            self.prev_branch = nn.Sequential(nn.Linear(num_experts, p_dim), nn.LayerNorm(p_dim), nn.GELU())
            fused_dim += p_dim

        if use_markov:
            p_dim = max(hidden_dim // 4, 16)
            self.markov_branch = nn.Sequential(nn.Linear(num_experts, p_dim), nn.LayerNorm(p_dim), nn.GELU())
            fused_dim += p_dim

        self.post_fusion_norm = nn.LayerNorm(fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            ResidualBlock(fused_dim, dropout),
            nn.Linear(fused_dim, num_experts)
        )

    def forward(self, embedding, prefill_dist=None, prev_onehot=None):
        B = embedding.size(0)

        if self.training and self.noise_std > 0:
            embedding = embedding + torch.randn_like(embedding) * self.noise_std

        feats = []

        if self.use_embedding:
            seq = embedding.view(B, self.history, self.emb_dim)
            x = self.emb_proj(seq)
            x = x + self.pos_emb.unsqueeze(0)
            out_seq = self.transformer(x)
            final_token = out_seq[:, -1, :] 
            feats.append(final_token)

        if self.use_prefill:
            if prefill_dist is None:
                prefill_dist = torch.full((B, self.num_experts), 1.0 / self.num_experts, device=embedding.device)
            feats.append(self.pfill_branch(prefill_dist))

        if self.use_prev:
            if prev_onehot is None:
                prev_onehot = torch.zeros(B, self.num_experts, device=embedding.device)
            feats.append(self.prev_branch(prev_onehot))

        if self.use_markov:
            if prev_onehot is None:
                prev_onehot = torch.zeros(B, self.num_experts, device=embedding.device)
            markov_prior = prev_onehot @ self.transition_matrix
            feats.append(self.markov_branch(markov_prior))

        fused = self.post_fusion_norm(torch.cat(feats, dim=-1))
        logits = self.classifier(fused)
        return logits

    def save(self, path: Path):
        torch.save({'state': self.state_dict(), 'config': self.config}, path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

class JITWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, emb, pfill, prev):
        return self.model(emb, pfill, prev)


class ExpertDatasetMultiStep(Dataset):
    def __init__(self, files, layer_idx, num_experts, active_k, predict_k, history, future_steps):
        self.samples = []
        self.num_experts = num_experts
        self.predict_k   = predict_k
        self.future_steps = future_steps

        for fp in tqdm(files, desc=f"Loading L{layer_idx}"):
            payload = torch.load(fp, map_location='cpu', weights_only=False)
            layer_data = next(
                (l for l in payload.get('layers', []) if l['layer_idx'] == layer_idx), None
            )
            if not layer_data:
                continue

            embs     = layer_data['embeddings']     # [T, hidden_size]
            logits   = layer_data['router_logits']  # [T, num_experts]
            prev_ids = layer_data.get('prev_expert_ids')  # [T, active_k] or None

            # Prefill distribution
            if 'prefill_expert_dist' in layer_data:
                pfill = layer_data['prefill_expert_dist'].float()
            elif 'prefill_expert_count' in layer_data:
                c = layer_data['prefill_expert_count'].float()
                pfill = c / c.sum().clamp(min=1)
            else:
                pfill = torch.softmax(logits.float(), dim=-1).mean(dim=0)

            T = len(embs)
            
            # Predict labels up to t + future_steps
            for t in range(T - future_steps):
                window = []
                for h in range(history):
                    src = t - (history - 1 - h)
                    window.append(embs[src] if src >= 0 else torch.zeros_like(embs[0]))

                prev_oh = torch.zeros(num_experts)
                if prev_ids is not None:
                    prev_oh[prev_ids[t]] = 1.0
                else:
                    prev_oh[logits[t].topk(active_k).indices] = 1.0
                s = prev_oh.sum()
                if s > 0:
                    prev_oh = prev_oh / s

                target_union = torch.zeros(num_experts, dtype=torch.float32)
                target_weights = torch.ones(num_experts, dtype=torch.float32)
                for f in range(1, future_steps + 1):
                    t_idx = logits[t + f].topk(predict_k).indices
                    target_union[t_idx] = 1.0
                    target_weights[t_idx] += 1.0

                self.samples.append({
                    'emb':    torch.cat(window),
                    'target': target_union,
                    'weight': target_weights,
                    'pfill':  pfill,
                    'prev':   prev_oh,
                })

        if len(self.samples) > 0:
            avg_union = sum(s['target'].sum().item() for s in self.samples) / len(self.samples)
            print(f"  [{layer_idx}] Avg union size: {avg_union:.1f} / {num_experts} ({100*avg_union/num_experts:.1f}% dense)")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def compute_recall_at_k(logits: torch.Tensor, targets_mh: torch.Tensor, k: int) -> float:
    # logits/targets shape: [B, num_experts]
    pred_idx = logits.topk(k, dim=-1).indices
    pred_mh  = torch.zeros_like(targets_mh).scatter_(-1, pred_idx, 1.0)
    true_pos = (pred_mh * targets_mh).sum()
    total    = targets_mh.sum().clamp(min=1)
    return (true_pos / total).item()

def smooth_bce_loss(logits, targets, weights=None, smoothing=0.1):
    # Only smooth the negatives — don't penalize confident positives
    smoothed = targets * (1 - smoothing) + (1 - targets) * smoothing
    return F.binary_cross_entropy_with_logits(logits, smoothed, weight=weights)

def compute_transition_matrix(loader, num_experts: int) -> torch.Tensor:
    counts = torch.zeros((num_experts, num_experts))
    for batch in loader:
        curr = batch['prev'] 
        tgt  = batch['target']
        counts += curr.T.float() @ tgt.float()
    counts += 1e-5
    return counts / counts.sum(dim=1, keepdim=True)


def run_predictor_task(args, layer_idx, hidden_dim, history,
                       use_emb=True, use_pfill=True, use_prev=False,
                       use_markov=False, custom_name=None):

    predict_k    = args.predict_k
    future_steps = args.future_steps
    multilabel   = (predict_k > 1)
    
    arch = getattr(args, 'arch', 'mlp')
    arch_prefix = f"{arch}_" if arch != 'mlp' else ""
    if custom_name:
        config_name = f"{arch_prefix}{custom_name}"
    else:
        config_name = f"{arch_prefix}eh{history}_h{hidden_dim}_f{future_steps}"

    out_dir = Path(args.output_dir) / config_name / f"layer_{layer_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "="*60)
    print(f"Starting Task: {config_name} | Layer: {layer_idx}")
    print("="*60)

    data_path = Path(args.data_dir)
    if not data_path.exists():
        sys.exit(f"ERROR: data_dir does not exist: {data_path.resolve()}")
    files = sorted(data_path.glob("*.pt"))
    if not files:
        sys.exit(f"ERROR: No .pt files found in: {data_path.resolve()}")
    print(f"  Found {len(files)} .pt files in {data_path.resolve()}")

    ds_kwargs = dict(
        layer_idx=layer_idx, num_experts=args.n_exp,
        active_k=args.active_k, predict_k=predict_k, 
        history=history, future_steps=future_steps
    )
    random.seed(42); random.shuffle(files)
    split = max(1, int(0.8 * len(files)))
    train_files = list(files)[:split]
    val_files   = list(files)[split:] if split < len(files) else train_files[-1:]

    train_ds = ExpertDatasetMultiStep(train_files, **ds_kwargs)
    val_ds   = ExpertDatasetMultiStep(val_files,   **ds_kwargs)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    train_loader  = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader    = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    trans_mat = None
    if use_markov:
        trans_mat = compute_transition_matrix(train_loader, args.n_exp)

    model_kwargs = dict(
        future_steps=future_steps,
        num_experts=args.n_exp, hidden_dim=hidden_dim,
        dropout=getattr(args, 'dropout', 0.1),
        use_embedding=use_emb, use_prefill=use_pfill, 
        use_prev=use_prev, use_markov=use_markov,
        transition_matrix=trans_mat, noise_std=args.noise
    )

    if arch == 'transformer':
        model = TransformerMultiStepPredictor(
            history=history, emb_dim=args.emb_dim,
            tx_layers=args.tx_layers, tx_heads=args.tx_heads,
            **model_kwargs
        ).to(args.device)
    else:
        model = MultiStepExpertPredictor(
            history=history, emb_dim=args.emb_dim,
            **model_kwargs
        ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture     : {arch.upper()} (Predicting {future_steps} steps ahead)")
    print(f"  Trainable Params : {total_params:,}\n")

    # Given the Union Lookahead Target, we strictly evaluate BCE
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_metric, logs = 0.0, []
    eval_k = getattr(args, 'prefetch_k', predict_k)
    metric_name = f"union_recall@{eval_k}"

    patience = getattr(args, 'patience', 3)
    no_improve = 0

    for epoch in range(args.epochs):
        epoch_log = {'epoch': epoch + 1}

        for phase in ('train', 'val'):
            model.train(phase == 'train')
            loss_sum, tp_sum, tot_sum = 0.0, 0.0, 0.0
            total = 0

            with torch.set_grad_enabled(phase == 'train'):
                loader = train_loader if phase == 'train' else val_loader
                pbar = tqdm(loader, desc=f"L{layer_idx} Ep{epoch+1} {phase}")
                for b in pbar:
                    b = {k: v.to(args.device) for k, v in b.items()}
                    out_logits = model(b['emb'], b['pfill'], b['prev'])

                    loss = smooth_bce_loss(out_logits, b['target'], weights=b.get('weight'), smoothing=args.smoothing)
                    eval_k = getattr(args, 'prefetch_k', predict_k)
                    
                    with torch.no_grad():
                        pred_idx = out_logits.topk(eval_k, dim=-1).indices
                        pred_mh  = torch.zeros_like(b['target']).scatter_(-1, pred_idx, 1.0)
                        batch_tp = (pred_mh * b['target']).sum().item()
                        batch_tot = b['target'].sum().clamp(min=1).item()
                        batch_metric = batch_tp / max(batch_tot, 1)

                    if phase == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn_utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    B = len(b['target'])
                    loss_sum    += loss.item() * B
                    tp_sum      += batch_tp
                    tot_sum     += batch_tot
                    total       += B
                    
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", metric_name: f"{batch_metric:.4f}"})

            if phase == 'train':
                scheduler.step()

            avg_loss = loss_sum / total
            avg_metric = tp_sum / max(tot_sum, 1)
            epoch_log[phase] = {'loss': avg_loss, metric_name: avg_metric}

            if phase == 'val':
                train_loss = epoch_log['train']['loss']
                train_metric = epoch_log['train'][metric_name]
                print(f"\n  [Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train {metric_name}: {train_metric:.4f}")
                print(f"            Val Loss:   {avg_loss:.4f} | Val {metric_name}:   {avg_metric:.4f}  (Best: {best_metric:.4f})")
                if avg_metric > best_metric:
                    best_metric = avg_metric
                    model.save(out_dir / "best.pt")
                    try:
                        wrapper = JITWrapper(model).eval()
                        ex_emb = torch.randn(1, args.emb_dim * history).to(args.device)
                        ex_pf  = torch.zeros(1, args.n_exp).to(args.device)
                        ex_pr  = torch.zeros(1, args.n_exp).to(args.device)
                        torch.jit.trace(wrapper, (ex_emb, ex_pf, ex_pr), check_trace=False, strict=False).save(str(out_dir / "best_jit.pt"))
                    except Exception as e:
                        print(f"  JIT failed: {e}")
                    no_improve = 0
                else:
                    no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break

        logs.append(epoch_log)

    with open(out_dir / "training_metrics.json", 'w') as f:
        json.dump({
            'layer_idx': layer_idx, 'hidden_dim': hidden_dim, 'history': history,
            'future_steps': future_steps, 'config_name': config_name, 
            'predict_k': predict_k, 'metric': metric_name, 
            'best': best_metric, 'best_acc': best_metric,
            'epochs': logs
        }, f, indent=2)

    return best_metric

def resolve_args(args):
    if args.model:
        emb_dim, n_exp, active_k, n_layers, pf_k, predict_k = MODEL_DEFAULTS[args.model]
        args.emb_dim, args.n_exp, args.active_k = emb_dim, n_exp, active_k
        args.n_layers, args.pf_k = n_layers, pf_k
        if not getattr(args, 'predict_k', None):
            args.predict_k = predict_k
        if not getattr(args, 'prefetch_k', None):
            args.prefetch_k = args.pf_k
    else:
        if not hasattr(args, 'emb_dim'):   args.emb_dim  = 4096
        if not hasattr(args, 'n_exp'):     args.n_exp    = 8
        if not hasattr(args, 'active_k'):  args.active_k = 2
        if not hasattr(args, 'n_layers'):  args.n_layers = 32
        if not hasattr(args, 'predict_k') or not args.predict_k:
            args.predict_k = args.active_k
            
    model_name = args.model if args.model else "mixtral_8x7b"
    if not getattr(args, 'data_dir', None):
        args.data_dir = f"../../trainingData/{model_name}"
    if not getattr(args, 'output_dir', None):
        args.output_dir = f"../../trainingData/{model_name}/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument('--data_dir',   default=None)
    base.add_argument('--output_dir', default=None)
    base.add_argument('--model',      choices=MODEL_DEFAULTS.keys())
    base.add_argument('--predict_k',  type=int, default=None)
    base.add_argument('--prefetch_k', type=int, default=None, help="Cache budget for recall metric")
    base.add_argument('--future_steps', type=int, default=2, help="How many steps ahead to predict.")
    base.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    base.add_argument('--epochs',     type=int, default=10)
    base.add_argument('--batch_size', type=int, default=128)
    base.add_argument('--lr',         type=float, default=1e-3)
    base.add_argument('--noise',      type=float, default=0.01)
    base.add_argument('--smoothing',  type=float, default=0.1)
    base.add_argument('--dropout',    type=float, default=0.1, help="Dropout rate in model layers")
    base.add_argument('--patience',   type=int, default=3)
    base.add_argument('--weight_decay', type=float, default=0.01)

    base.add_argument('--arch',       choices=['mlp', 'transformer'], default='mlp')
    base.add_argument('--tx_layers',  type=int, default=2)
    base.add_argument('--tx_heads',   type=int, default=4)

    p_train = subparsers.add_parser('train', parents=[base])
    p_train.add_argument('--layer',   type=int)
    p_train.add_argument('--history', type=int, default=1)
    p_train.add_argument('--hidden',  type=int, default=128)

    p_sweep = subparsers.add_parser('sweep', parents=[base])
    p_sweep.add_argument('--hiddens',   nargs='+', type=int, default=[64, 128, 256])
    p_sweep.add_argument('--histories', nargs='+', type=int, default=[1, 2])
    p_sweep.add_argument('--layers',    nargs='+', type=int, default=list(range(32)))
    p_sweep.add_argument('--max_mb',    type=float, default=70.0)

    p_abl = subparsers.add_parser('ablation', parents=[base])
    p_abl.add_argument('--layers',    nargs='+', type=int, default=list(range(32)))
    p_abl.add_argument('--hiddens',   nargs='+', type=int, default=[128, 256])
    p_abl.add_argument('--histories', nargs='+', type=int, default=[1, 2])

    args = resolve_args(parser.parse_args())

    ABLATION_VARIANTS = [
        ("emb_only",      True,  False, False, False),
        ("pfill_only",    False, True,  False, False),
        ("prev_only",     False, False, True,  False),
        ("markov_only",   False, False, False, True ),
        ("emb_prev",      True,  False, True,  False),
        ("emb_markov",    True,  False, False, True ),
        ("all_features",  True,  True,  True,  True ),
    ]

    if args.command == 'train':
        layers = [args.layer] if args.layer is not None else list(range(args.n_layers))
        for l in layers:
            run_predictor_task(args, l, args.hidden, args.history)

    elif args.command == 'sweep':
        layers = args.layers or list(range(args.n_layers))
        for hist in args.histories:
            for h in args.hiddens:
                size_mb = ((args.emb_dim * hist * h + h * h // 2 + h // 2 * args.n_exp) * 4) / 1e6
                if size_mb > args.max_mb:
                    continue
                for l in layers:
                    run_predictor_task(args, l, h, hist)

    elif args.command == 'ablation':
        layers = args.layers or list(range(args.n_layers))
        for l in layers:
            for hist in args.histories:
                for h in args.hiddens:
                    for name, e, pf, pr, mk in ABLATION_VARIANTS:
                        run_predictor_task(
                            args, l, h, hist,
                            use_emb=e, use_pfill=pf, use_prev=pr, use_markov=mk,
                            custom_name=f"ablation_{name}_hist{hist}_h{h}_f{args.future_steps}"
                        )
