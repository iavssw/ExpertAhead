#!/usr/bin/env python3
"""
expert_predictor.py — Unified Expert Predictor
==============================================
Trains a lightweight MLP to predict the top-K router experts at token t+1,
given features from token t. Supports:
  - predict_k=1  → CrossEntropyLoss + top-1 accuracy   (Mixtral style)
  - predict_k>1  → BCEWithLogitsLoss + recall@K         (Qwen style)
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
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# (emb_dim, num_experts, active_k, num_layers, prefetch_k, predict_k)
MODEL_DEFAULTS = {
    "mixtral_8x7b":  (4096, 8,   2, 32, 1, 1),
    "mixtral_8x22b": (4096, 8,   2, 56, 1, 1),
    "qwen3_30b":     (2048, 128, 8, 48, 4, 4),
    "qwen3_480b":    (7168, 128, 8, 94, 4, 4),
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. Model & Wrapper
# ──────────────────────────────────────────────────────────────────────────────

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

class ExpertPredictor(nn.Module):
    """
    Predicts the top-K router experts at the next token.
    
    Inputs (all optional except embedding):
      - embedding:    [B, hidden_size]   hidden state at token t
      - prefill_dist: [B, num_experts]   expert usage distribution from prefill
      - prev_onehot:  [B, num_experts]   multi-hot of active experts at token t
                                         (normalized to sum=1)
    Output:
      - logits: [B, num_experts]  → argmax for top-1, topk for top-K
    """
    def __init__(
        self,
        hidden_size: int = 4096,
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
            'hidden_size': hidden_size, 'num_experts': num_experts,
            'hidden_dim': hidden_dim, 'use_embedding': use_embedding,
            'use_prefill': use_prefill, 'use_prev': use_prev,
            'use_markov': use_markov, 'noise_std': noise_std
        }
        self.use_embedding = use_embedding
        self.use_prefill   = use_prefill
        self.use_prev      = use_prev
        self.use_markov    = use_markov
        self.num_experts   = num_experts
        self.noise_std     = noise_std

        # Markov transition matrix: [num_experts, num_experts]
        if transition_matrix is not None:
            self.register_buffer("transition_matrix", transition_matrix)
        else:
            self.register_buffer("transition_matrix", torch.zeros((num_experts, num_experts)))

        fused_dim = 0

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
            # prev_onehot is a (normalized) distribution over active experts.
            # Weighted sum of their transition rows gives expected next-expert dist.
            if prev_onehot is None:
                prev_onehot = torch.zeros(B, self.num_experts, device=embedding.device)
            markov_prior = prev_onehot @ self.transition_matrix  # [B, num_experts]
            feats.append(self.markov_branch(markov_prior))

        fused = self.post_fusion_norm(torch.cat(feats, dim=-1))
        return self.classifier(fused)

    def save(self, path: Path):
        torch.save({'state': self.state_dict(), 'config': self.config}, path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

class TransformerPredictor(nn.Module):
    """
    Predicts the top-K router experts at the next token using a Transformer.
    """
    def __init__(
        self,
        history: int = 1,
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
            'history': history, 'emb_dim': emb_dim,
            'hidden_size': history * emb_dim, 
            'num_experts': num_experts, 'hidden_dim': hidden_dim,
            'tx_layers': tx_layers, 'tx_heads': tx_heads,
            'use_embedding': use_embedding, 'use_prefill': use_prefill,
            'use_prev': use_prev, 'use_markov': use_markov, 'noise_std': noise_std
        }
        self.history = history
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
        return self.classifier(fused)

    def save(self, path: Path):
        torch.save({'state': self.state_dict(), 'config': self.config}, path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

class JITWrapper(nn.Module):
    """Fixed (emb, pfill, prev) signature for C++ LibTorch."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, emb, pfill, prev):
        return self.model(emb, pfill, prev)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ExpertDataset(Dataset):
    """
    Builds (embedding, prefill_dist, prev_multihot) → target samples.

    predict_k=1  → target is a scalar class index (for CrossEntropyLoss)
    predict_k>1  → target is a float multi-hot vector (for BCEWithLogitsLoss)

    prev_onehot is always multi-hot over all active_k experts, normalized
    to sum=1 so it acts as a prior distribution.
    """
    def __init__(self, files, layer_idx, num_experts, active_k, predict_k, history):
        self.samples = []
        self.num_experts = num_experts
        self.predict_k   = predict_k

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

            # Prefill distribution: [num_experts]
            if 'prefill_expert_dist' in layer_data:
                pfill = layer_data['prefill_expert_dist'].float()
            elif 'prefill_expert_count' in layer_data:
                c = layer_data['prefill_expert_count'].float()
                pfill = c / c.sum().clamp(min=1)
            else:
                pfill = torch.softmax(logits.float(), dim=-1).mean(dim=0)

            T = len(embs)
            for t in range(T - 1):
                # ── History window ─────────────────────────────────────────
                window = []
                for h in range(history):
                    src = t - (history - 1 - h)
                    window.append(embs[src] if src >= 0 else torch.zeros_like(embs[0]))

                # ── Previous experts: multi-hot, normalized ─────────────────
                prev_oh = torch.zeros(num_experts)
                if prev_ids is not None:
                    prev_oh[prev_ids[t]] = 1.0        # all active_k bits
                else:
                    prev_oh[logits[t].topk(active_k).indices] = 1.0
                s = prev_oh.sum()
                if s > 0:
                    prev_oh = prev_oh / s

                # ── Target ─────────────────────────────────────────────────
                if predict_k == 1:
                    target = logits[t + 1].argmax().long()          # scalar
                else:
                    top_k_idx = logits[t + 1].topk(predict_k).indices
                    target = torch.zeros(num_experts, dtype=torch.float32)
                    target[top_k_idx] = 1.0                         # multi-hot

                self.samples.append({
                    'emb':    torch.cat(window),
                    'target': target,
                    'pfill':  pfill,
                    'prev':   prev_oh,
                })

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

# ──────────────────────────────────────────────────────────────────────────────
# 3. Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_recall_at_k(logits: torch.Tensor, targets_mh: torch.Tensor, k: int) -> float:
    """
    Recall@K for multi-label prediction.
    targets_mh: [B, num_experts] float multi-hot
    Returns fraction of true experts captured in the top-K predictions.
    """
    pred_idx = logits.topk(k, dim=-1).indices          # [B, k]
    pred_mh  = torch.zeros_like(targets_mh).scatter_(1, pred_idx, 1.0)
    true_pos = (pred_mh * targets_mh).sum()
    total    = targets_mh.sum().clamp(min=1)
    return (true_pos / total).item()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Transition Matrix
# ──────────────────────────────────────────────────────────────────────────────

def compute_transition_matrix(loader, num_experts: int) -> torch.Tensor:
    """
    Build a [num_experts, num_experts] row-stochastic transition matrix
    from training data. Works for both scalar and multi-hot targets.
    """
    counts = torch.zeros((num_experts, num_experts))
    for batch in loader:
        curr = batch['prev']    # [B, num_experts] distribution
        tgt  = batch['target']  # scalar or multi-hot
        if tgt.dim() == 1 and tgt.dtype == torch.long:
            # scalar → one-hot next
            nxt = torch.zeros(len(tgt), num_experts).scatter_(1, tgt.unsqueeze(1), 1.0)
        else:
            nxt = tgt.float()
        # Outer product accumulation: curr^T @ nxt
        counts += curr.T.float() @ nxt.float()
    counts += 1e-5
    return counts / counts.sum(dim=1, keepdim=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Training Engine
# ──────────────────────────────────────────────────────────────────────────────

def run_predictor_task(args, layer_idx, hidden_dim, history,
                       use_emb=True, use_pfill=True, use_prev=False,
                       use_markov=False, custom_name=None):
    """Train a single model configuration and return best validation metric."""

    predict_k   = args.predict_k
    multilabel   = (predict_k > 1)
    
    arch = getattr(args, 'arch', 'mlp')
    arch_prefix = f"{arch}_" if arch != 'mlp' else ""
    if custom_name:
        config_name = f"{arch_prefix}{custom_name}"
    else:
        config_name = f"{arch_prefix}eh{history}_h{hidden_dim}"

    out_dir      = Path(args.output_dir) / config_name / f"layer_{layer_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path = Path(args.data_dir)
    if not data_path.exists():
        sys.exit(f"ERROR: data_dir does not exist: {data_path.resolve()}")
    files = sorted(data_path.glob("*.pt"))
    if not files:
        sys.exit(f"ERROR: No .pt files found in: {data_path.resolve()}")
    print(f"  Found {len(files)} .pt files in {data_path.resolve()}")

    random.seed(42); random.shuffle(files)
    split = max(1, int(0.8 * len(files)))   # at least 1 train file

    ds_kwargs = dict(
        layer_idx=layer_idx, num_experts=args.n_exp,
        active_k=args.active_k, predict_k=predict_k, history=history
    )
    train_files = list(files)[:split]
    val_files   = list(files)[split:] if split < len(files) else [files[-1]]  # reuse last if tiny set
    train_ds = ExpertDataset(train_files, **ds_kwargs)
    val_ds   = ExpertDataset(val_files,   **ds_kwargs)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    train_loader  = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader    = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    trans_mat = None
    if use_markov:
        trans_mat = compute_transition_matrix(train_loader, args.n_exp)

    model_kwargs = dict(
        num_experts=args.n_exp, hidden_dim=hidden_dim, 
        use_embedding=use_emb, use_prefill=use_pfill, 
        use_prev=use_prev, use_markov=use_markov,
        transition_matrix=trans_mat, noise_std=args.noise
    )

    if getattr(args, 'arch', 'mlp') == 'transformer':
        model = TransformerPredictor(
            history=history, emb_dim=args.emb_dim,
            tx_layers=args.tx_layers, tx_heads=args.tx_heads,
            **model_kwargs
        ).to(args.device)
    else:
        model = ExpertPredictor(
            hidden_size=args.emb_dim * history,
            **model_kwargs
        ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture     : {getattr(args, 'arch', 'mlp').upper()}")
    print(f"  Trainable Params : {total_params:,}\n")

    # ── Loss ──────────────────────────────────────────────────────────────────
    multilabel = (predict_k > 1)
    if multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_metric, logs = 0.0, []
    metric_name = f"recall@{predict_k}" if multilabel else "top1_acc"

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        epoch_log = {'epoch': epoch + 1}

        for phase in ('train', 'val'):
            model.train(phase == 'train')
            loss_sum, metric_sum, total = 0.0, 0.0, 0

            with torch.set_grad_enabled(phase == 'train'):
                loader = train_loader if phase == 'train' else val_loader
                pbar = tqdm(loader, desc=f"L{layer_idx} Ep{epoch+1} {phase}")
                for b in pbar:
                    b = {k: v.to(args.device) for k, v in b.items()}
                    out_logits = model(b['emb'], b['pfill'], b['prev'])

                    if multilabel:
                        loss = criterion(out_logits, b['target'])
                        batch_metric = compute_recall_at_k(out_logits, b['target'], predict_k)
                    else:
                        loss = criterion(out_logits, b['target'])
                        batch_metric = (out_logits.argmax(-1) == b['target']).float().mean().item()

                    if phase == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn_utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    B = len(b['target'])
                    loss_sum    += loss.item() * B
                    metric_sum  += batch_metric * B
                    total       += B
                    
                    pbar.set_postfix({'loss': loss.item(), metric_name: batch_metric})

            if phase == 'train':
                scheduler.step()

            avg_loss = loss_sum / total
            avg_metric = metric_sum / total
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

        logs.append(epoch_log)

    with open(out_dir / "training_metrics.json", 'w') as f:
        json.dump({
            'layer_idx': layer_idx, 'hidden_dim': hidden_dim, 'history': history,
            'config_name': config_name, 'predict_k': predict_k,
            'metric': metric_name, 'best': best_metric, 'best_acc': best_metric,  # best_acc alias for plot compat
            'epochs': logs
        }, f, indent=2)

    return best_metric

# ──────────────────────────────────────────────────────────────────────────────
# 6. CLI & Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def resolve_args(args):
    """Inject model defaults if a model name was provided."""
    if args.model:
        emb_dim, n_exp, active_k, n_layers, pf_k, predict_k = MODEL_DEFAULTS[args.model]
        args.emb_dim, args.n_exp, args.active_k = emb_dim, n_exp, active_k
        args.n_layers, args.pf_k = n_layers, pf_k
        # CLI --predict_k overrides the model default
        if not getattr(args, 'predict_k', None):
            args.predict_k = predict_k
    else:
        if not hasattr(args, 'emb_dim'):   args.emb_dim  = 4096
        if not hasattr(args, 'n_exp'):     args.n_exp    = 8
        if not hasattr(args, 'active_k'):  args.active_k = 2
        if not hasattr(args, 'n_layers'):  args.n_layers = 32
        if not hasattr(args, 'predict_k') or not args.predict_k:
            args.predict_k = args.active_k
            
    # Auto-resolve directories based on model if not provided
    model_name = args.model if args.model else "mixtral_8x7b"
    if not getattr(args, 'data_dir', None):
        args.data_dir = f"../../trainingData/{model_name}"
    if not getattr(args, 'output_dir', None):
        args.output_dir = f"../../trainingData/{model_name}/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    return args


if __name__ == "__main__":
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared base args
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument('--data_dir',   default=None)
    base.add_argument('--output_dir', default=None)
    base.add_argument('--model',      choices=MODEL_DEFAULTS.keys())
    base.add_argument('--predict_k',  type=int, default=None,
                      help="How many top experts to predict. Defaults to model's active_k. "
                           "predict_k=1 uses CrossEntropyLoss; predict_k>1 uses BCEWithLogitsLoss + Recall@K.")
    base.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    base.add_argument('--epochs',     type=int, default=10)
    base.add_argument('--batch_size', type=int, default=128)
    base.add_argument('--lr',         type=float, default=1e-3)
    base.add_argument('--noise',      type=float, default=0.01)
    base.add_argument('--smoothing',  type=float, default=0.1,
                      help="Label smoothing for CrossEntropyLoss (predict_k=1 only).")
    base.add_argument('--weight_decay', type=float, default=0.01)

    # Architecture
    base.add_argument('--arch',       choices=['mlp', 'transformer'], default='mlp')
    base.add_argument('--tx_layers',  type=int, default=2)
    base.add_argument('--tx_heads',   type=int, default=4)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser('train', parents=[base])
    p_train.add_argument('--layer',   type=int)
    p_train.add_argument('--history', type=int, default=1)
    p_train.add_argument('--hidden',  type=int, default=128)

    # ── sweep ─────────────────────────────────────────────────────────────────
    p_sweep = subparsers.add_parser('sweep', parents=[base])
    p_sweep.add_argument('--hiddens',   nargs='+', type=int, default=[64, 128, 256])
    p_sweep.add_argument('--histories', nargs='+', type=int, default=[1, 2])
    p_sweep.add_argument('--layers',    nargs='+', type=int, default=list(range(32)))
    p_sweep.add_argument('--max_mb',    type=float, default=70.0)

    # ── ablation ──────────────────────────────────────────────────────────────
    p_abl = subparsers.add_parser('ablation', parents=[base])
    p_abl.add_argument('--layers',    nargs='+', type=int, default=list(range(32)))
    p_abl.add_argument('--hiddens',   nargs='+', type=int, default=[128, 256])
    p_abl.add_argument('--histories', nargs='+', type=int, default=[1, 2])

    args = resolve_args(parser.parse_args())

    # Ablation variants: (name, use_emb, use_pfill, use_prev, use_markov)
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
                            custom_name=f"ablation_{name}_hist{hist}_h{h}"
                        )
