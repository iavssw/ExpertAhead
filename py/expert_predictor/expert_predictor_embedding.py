#!/usr/bin/env python3
"""
expert_predictor_embedding.py — Embedding-Targeted Expert Predictor
===================================================================
Instead of predicting the router experts (classification), this script trains a
model to predict the ACTUAL embedding at token t+1.
The predicted embedding is then multiplied by the exact recovered router weights 
to get the predicted router logits and compute top-K accuracy/recall.
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

class EmbeddingPredictor(nn.Module):
    """
    Predicts the next EXACT embedding using features from token t.
    """
    def __init__(
        self,
        emb_dim: int = 4096,     # size of the target predicted embedding
        history: int = 1,        # number of history steps
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
            'target': 'embedding_mse',
            'history': history, 'emb_dim': emb_dim, 
            'num_experts': num_experts, 'hidden_dim': hidden_dim, 
            'use_embedding': use_embedding,
            'use_prefill': use_prefill, 'use_prev': use_prev,
            'use_markov': use_markov, 'noise_std': noise_std
        }
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
            self.history = history
            self.emb_dim = emb_dim
            self.step_proj = nn.Linear(emb_dim, hidden_dim)
            self.emb_branch = nn.Sequential(
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

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            ResidualBlock(fused_dim, dropout),
            nn.Linear(fused_dim, emb_dim) # Output the full targeted embedding
        )

    def forward(self, embedding, prefill_dist=None, prev_onehot=None):
        B = embedding.size(0)

        if self.training and self.noise_std > 0:
            embedding = embedding + torch.randn_like(embedding) * self.noise_std

        feats = []

        if self.use_embedding:
            seq = embedding.view(B, self.history, self.emb_dim) # [B, history, emb_dim]
            proj = self.step_proj(seq) # [B, history, hidden_dim]
            pooled = proj.mean(dim=1)  # [B, hidden_dim]
            feats.append(self.emb_branch(pooled))

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
        return self.regressor(fused)

    def forward_lookahead(self, embedding, prefill_dist, prev_onehot, router_W_ref, active_k, steps=1):
        predictions = []
        current_emb = self.forward(embedding, prefill_dist, prev_onehot)
        predictions.append(current_emb)

        for _ in range(steps - 1):
            with torch.no_grad():
                pred_logits = current_emb @ router_W_ref
                prev_onehot_next = torch.zeros_like(prev_onehot)
                prev_onehot_next.scatter_(1, pred_logits.topk(active_k).indices, 1.0)
            current_emb = self.forward(current_emb, prefill_dist, prev_onehot_next)
            predictions.append(current_emb)

        return predictions

    def save(self, path: Path):
        torch.save({'state': self.state_dict(), 'config': self.config}, path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

class JITWrapper(nn.Module):
    """Fixed (emb, pfill, prev) positional signature required for C++ LibTorch tracing."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, emb, pfill, prev):
        return self.model(emb, pfill, prev)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """
    Builds (embedding, prefill, prev_multihot) → target_embedding at t+1 samples.
    Loads the explicit literal router weights from router_bins to compute validation metrics.
    """
    def __init__(self, files, data_dir, layer_idx, num_experts, active_k, predict_k, history, router_W=None, lookahead_k=1):
        self.samples = []
        self.num_experts = num_experts
        self.predict_k = predict_k
        self.lookahead_k = lookahead_k
        self.router_W = router_W # The exact actual router weights [emb_dim, num_experts]

        if self.router_W is None:
            gate_path = Path(data_dir) / "router_bins" / f"layer_{layer_idx}_gate.pt"
            if gate_path.exists():
                # HF Linear weights are [num_experts, emb_dim]. Transpose to [emb_dim, num_experts]
                try:
                    self.router_W = torch.load(gate_path, map_location='cpu', weights_only=False).T.float()
                    print(f"Loaded literal router weights for layer {layer_idx}")
                except Exception as e:
                    print(f"Error loading gate weights: {e}")

        all_embs = []
        all_logits = []

        for fp in tqdm(files, desc=f"Loading L{layer_idx}"):
            payload = torch.load(fp, map_location='cpu', weights_only=False)
            layer_data = next(
                (l for l in payload.get('layers', []) if l['layer_idx'] == layer_idx), None
            )
            if not layer_data:
                continue

            embs     = layer_data['embeddings']     # [T, emb_dim]
            logits   = layer_data['router_logits']  # [T, num_experts]
            prev_ids = layer_data.get('prev_expert_ids')  

            if self.router_W is None:
                all_embs.append(embs)
                all_logits.append(logits)

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
                # Multi-step Autoregressive lookahead
                target_emb_list = []
                target_cls_list = []
                for step in range(self.lookahead_k):
                    idx = min(t + 1 + step, T - 1)
                    
                    tgt_emb = embs[idx]
                    if predict_k == 1:
                        tgt_cls = logits[idx].argmax().long()          # scalar
                    else:
                        top_k_idx = logits[idx].topk(predict_k).indices
                        tgt_cls = torch.zeros(num_experts, dtype=torch.float32)
                        tgt_cls[top_k_idx] = 1.0                         # multi-hot
                    
                    target_emb_list.append(tgt_emb)
                    target_cls_list.append(tgt_cls)

                sample = {
                    'emb':         torch.cat(window),
                    'target_emb':  target_emb_list[0],
                    'target_cls':  target_cls_list[0],
                    'pfill':       pfill,
                    'prev':        prev_oh,
                }
                for step in range(self.lookahead_k):
                    sample[f'target_emb_{step}'] = target_emb_list[step]
                    sample[f'target_cls_{step}'] = target_cls_list[step]
                
                self.samples.append(sample)

        if self.router_W is None and len(all_embs) > 0:
            print(f"WARNING: No true router weights found. Falling back to global pseudo-inverse approximation for L{layer_idx}. Validation metrics may be unreliable.")
            global_embs = torch.cat(all_embs, dim=0).float()
            global_logits = torch.cat(all_logits, dim=0).float()
            self.router_W = torch.linalg.lstsq(global_embs, global_logits).solution

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

# ──────────────────────────────────────────────────────────────────────────────
# 3. Metrics
# ──────────────────────────────────────────────────────────────────────────────

def cache_coverage_loss(pred_logits, true_experts_mh, prefetch_k, temp=1.0):
    """Maximize probability mass on true experts within prefetch_k budget."""
    cache_probs = F.softmax(pred_logits / temp, dim=-1)
    coverage = (cache_probs * true_experts_mh).sum(dim=-1)
    return (1.0 - coverage).mean()

def routing_kl_loss(pred_emb, target_emb, router_W, temp=1.0):
    """KL divergence between predicted and true routing distributions."""
    pred_logits   = pred_emb @ router_W
    target_logits = target_emb @ router_W
    return F.kl_div(
        F.log_softmax(pred_logits / temp, dim=-1),
        F.softmax(target_logits / temp, dim=-1),
        reduction='batchmean'
    )

def cache_hit_rate(pred_logits, true_experts_mh, prefetch_k):
    """What fraction of true expert activations are covered by our cache of size prefetch_k?"""
    cache_idx = pred_logits.topk(prefetch_k, dim=-1).indices      # [B, prefetch_k]
    cache_mh  = torch.zeros_like(true_experts_mh).scatter_(1, cache_idx, 1.0)
    hits  = (cache_mh * true_experts_mh).sum()
    total = true_experts_mh.sum().clamp(min=1)
    return (hits / total).item()

def compute_transition_matrix(samples, num_experts: int) -> torch.Tensor:
    counts = torch.zeros((num_experts, num_experts))
    for s in samples:
        curr = s['prev']    
        tgt  = s['target_cls']
        if tgt.dim() == 0 and tgt.dtype == torch.long:
            nxt = torch.zeros(num_experts).scatter_(0, tgt, 1.0)
        else:
            nxt = tgt.float()
        counts += curr.unsqueeze(1) * nxt.unsqueeze(0)
    counts += 1e-5
    return counts / counts.sum(dim=1, keepdim=True)

def prefetch_urgency(pred_logits, prefetch_k=None):
    """
    Returns urgency in [0, 1]. High urgency = issue prefetch immediately.
    """
    probs   = F.softmax(pred_logits, dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(pred_logits.size(-1)), device=pred_logits.device))
    normalized_entropy = entropy / max_entropy
    return 1.0 - normalized_entropy

# ──────────────────────────────────────────────────────────────────────────────
# 4. Training Engine
# ──────────────────────────────────────────────────────────────────────────────

def run_predictor_task(args, layer_idx, hidden_dim, history,
                       use_emb=True, use_pfill=True, use_prev=False,
                       use_markov=False, custom_name=None):
    
    predict_k   = args.predict_k
    prefetch_k  = args.prefetch_k
    multilabel   = (predict_k > 1)
    
    if custom_name:
        config_name = f"emb_target_{custom_name}"
    else:
        config_name = f"emb_target_eh{history}_h{hidden_dim}"

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
    split = max(1, int(0.8 * len(files)))

    if len(files) <= 1:
        raise ValueError("Only 1 data file found — cannot create a separate validation split.")

    train_files = list(files)[:split]
    val_files   = list(files)[split:]
    
    ds_kwargs = dict(
        data_dir=args.data_dir,
        layer_idx=layer_idx, num_experts=args.n_exp,
        active_k=args.active_k, predict_k=predict_k, history=history,
        lookahead_k=getattr(args, 'lookahead_k', 1)
    )
    
    train_ds = EmbeddingDataset(train_files, **ds_kwargs)
    val_ds   = EmbeddingDataset(val_files, router_W=train_ds.router_W, **ds_kwargs)
    
    # We must have router weights recovered
    router_W = train_ds.router_W.to(args.device)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    train_loader  = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader    = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    trans_mat = None
    if use_markov:
        trans_mat = compute_transition_matrix(train_ds.samples, args.n_exp)

    model = EmbeddingPredictor(
        emb_dim=args.emb_dim, history=history,
        num_experts=args.n_exp, hidden_dim=hidden_dim, 
        use_embedding=use_emb, use_prefill=use_pfill, 
        use_prev=use_prev, use_markov=use_markov,
        transition_matrix=trans_mat, noise_std=args.noise
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture     : Embedding-Targeted MLP")
    print(f"  Trainable Params : {total_params:,}\n")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # The crucial change: we use MSELoss against the actual next embedding!
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup    = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=1)
    cosine    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 1), eta_min=args.lr * 0.01)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1])

    best_metric, logs = 0.0, []
    metric_name = f"cache_hit_rate@{prefetch_k}"

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        epoch_log = {'epoch': epoch + 1}

        for phase in ('train', 'val'):
            model.train(phase == 'train')
            loss_sum, tp_sum, tot_sum = 0.0, 0.0, 0.0
            pred_tp_sum = 0.0
            total = 0

            with torch.set_grad_enabled(phase == 'train'):
                loader = train_loader if phase == 'train' else val_loader
                pbar = tqdm(loader, desc=f"L{layer_idx} Ep{epoch+1} {phase}")
                for b in pbar:
                    b = {k: v.to(args.device) for k, v in b.items()}
                    
                    
                    lookahead_steps = getattr(args, 'lookahead_k', 1)
                    
                    # 1. Forward pass - Multi step Autoregressive
                    pred_embs = model.forward_lookahead(
                        b['emb'], b['pfill'], b['prev'], 
                        router_W_ref=router_W, active_k=args.active_k, 
                        steps=lookahead_steps
                    )
                    pred_emb = pred_embs[0]

                    # 2. Compute Cache Coverage & Routing KL Loss exponentially over steps
                    gamma = 0.7
                    total_loss = 0.0
                    
                    for step, p_emb in enumerate(pred_embs):
                        t_emb = b[f'target_emb_{step}']
                        t_cls = b[f'target_cls_{step}']
                        
                        
                        if getattr(args, 'normalize_embeddings', False):
                            t_emb = torch.nn.functional.normalize(t_emb, p=2, dim=-1)
                            p_emb_norm = torch.nn.functional.normalize(p_emb, p=2, dim=-1)
                        else:
                            p_emb_norm = p_emb

                        p_logits = p_emb_norm @ router_W
                        step_loss = (
                            cache_coverage_loss(p_logits, t_cls, args.prefetch_k, temp=args.temp)
                            + 0.3 * routing_kl_loss(p_emb_norm, t_emb, router_W, temp=args.temp)
                        )
                        total_loss += (gamma ** step) * step_loss
                    
                    loss = total_loss

                    # 3. Validation Metric: Multiply predicted embedding by literal Router Weights
                    # to obtain predicted router logits, and compute hit rate!
                    with torch.no_grad():
                        eval_emb = pred_emb
                        if getattr(args, 'normalize_embeddings', False):
                            eval_emb = torch.nn.functional.normalize(eval_emb, p=2, dim=-1)

                        pred_logits = eval_emb @ router_W
                        urgency_batch = prefetch_urgency(pred_logits).mean().item()
                        
                        B_dim = len(b['target_cls'])
                        if multilabel:
                            # Print a debug line occasionally to satisfy user cache observability request
                            if pbar.n == 0:
                                cache_idx = pred_logits[0].topk(args.prefetch_k, dim=-1).indices.tolist()
                                sample_tgt = b['target_cls'][0].nonzero(as_tuple=True)[0].tolist()
                                hit_list = [x for x in sample_tgt if x in cache_idx]
                                miss = [x for x in sample_tgt if x not in cache_idx]
                                print(f"\n  [Cache Debug] Target: {sample_tgt} -> Hits: {hit_list} | Missed: {miss}")
                            
                            batch_metric = cache_hit_rate(pred_logits, b['target_cls'], prefetch_k)
                            batch_targets = b['target_cls'].sum().clamp(min=1).item()
                            tp_sum += batch_metric * batch_targets
                            tot_sum += batch_targets

                            # Also track predict_k for supplementary logging
                            pred_idx = pred_logits.topk(predict_k, dim=-1).indices
                            pred_mh  = torch.zeros_like(b['target_cls']).scatter_(1, pred_idx, 1.0)
                            pred_tp_sum += (pred_mh * b['target_cls']).sum().item()
                        else:
                            batch_hits = (pred_logits.argmax(-1) == b['target_cls']).float().sum().item()
                            tp_sum += batch_hits
                            tot_sum += B_dim
                            batch_metric = batch_hits / max(B_dim, 1)

                    if phase == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn_utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    loss_sum += loss.item() * B_dim
                    total    += B_dim
                    
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", metric_name: f"{batch_metric:.4f}", 'urgency': f"{urgency_batch:.2f}"})

            if phase == 'train':
                scheduler.step()

            avg_loss = loss_sum / total
            avg_metric = tp_sum / max(tot_sum, 1)
            epoch_log[phase] = {'loss': avg_loss, metric_name: avg_metric}
            
            if multilabel:
                epoch_log[phase][f"hit_rate@{predict_k}"] = pred_tp_sum / max(tot_sum, 1)

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
                        if not (out_dir / "best_jit.pt").exists():
                            print("  WARNING: JIT export failed and no prior best_jit.pt exists. Inference will not work.")

        logs.append(epoch_log)

    with open(out_dir / "training_metrics.json", 'w') as f:
        json.dump({
            'layer_idx': layer_idx, 'hidden_dim': hidden_dim, 'history': history,
            'config_name': config_name, 'predict_k': predict_k,
            'metric': metric_name, 'best': best_metric,
            'epochs': logs
        }, f, indent=2)

    return best_metric

# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI & Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def resolve_args(args):
    if args.model:
        emb_dim, n_exp, active_k, n_layers, pf_k, predict_k = MODEL_DEFAULTS[args.model]
        args.emb_dim, args.n_exp, args.active_k = emb_dim, n_exp, active_k
        args.n_layers = n_layers
        if not args.predict_k:
            args.predict_k = predict_k
        if not getattr(args, 'prefetch_k', None):
            args.prefetch_k = pf_k
    else:
        assert args.emb_dim and args.n_exp and args.active_k and args.n_layers, \
            "Must supply --model or manual hypers"
        if not args.predict_k:
            args.predict_k = args.active_k
        if not getattr(args, 'prefetch_k', None):
            args.prefetch_k = args.predict_k
            
    assert args.prefetch_k >= args.predict_k, f"prefetch_k ({args.prefetch_k}) must be >= predict_k ({args.predict_k})"
    
    model_name = args.model if args.model else "mixtral_8x7b"
    if not getattr(args, 'data_dir', None):
        args.data_dir = f"../../trainingData/{model_name}"
    if not getattr(args, 'output_dir', None):
        args.output_dir = f"../../trainingData/{model_name}/sweep_emb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument('--data_dir',   default=None)
    base.add_argument('--output_dir', default=None)
    base.add_argument('--model',      choices=MODEL_DEFAULTS.keys())
    base.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    base.add_argument('--predict_k',  type=int, default=None, help="Experts to predictably route to")
    base.add_argument('--prefetch_k', type=int, default=None, help="Cache budget per token")
    base.add_argument('--epochs',     type=int, default=10)
    base.add_argument('--batch_size', type=int, default=128)
    base.add_argument('--lr',         type=float, default=1e-3)
    base.add_argument('--noise',      type=float, default=0.01)
    base.add_argument('--weight_decay', type=float, default=0.01)
    base.add_argument('--temp',       type=float, default=2.0, help="Temperature for Cache Coverage loss")
    base.add_argument('--normalize_embeddings', action='store_true', default=True, help="L2 normalize embeddings")
    base.add_argument('--lookahead_k', type=int, default=1, help="Autoregressive prediction depth")

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser('train', parents=[base])
    p_train.add_argument('--layer',   type=int)
    p_train.add_argument('--history', type=int, default=1)
    p_train.add_argument('--hidden',  type=int, default=128)

    args = parser.parse_args()

    if args.command == 'train':
        args = resolve_args(args)
        layers = [args.layer] if args.layer is not None else list(range(args.n_layers))
        for l in layers:
            run_predictor_task(args, l, args.hidden, args.history)

