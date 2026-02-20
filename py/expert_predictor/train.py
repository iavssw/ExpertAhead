#!/usr/bin/env python3
"""
train.py — Expert Predictor Training Script
============================================
Trains a per-layer MLP that predicts which experts will be selected at token t+1
given the post-attention-norm embeddings at token t (+ optional history window).

Usage
-----
# Train all layers for Mixtral 8x7B:
python train.py --data_dir ../../trainingData/mixtral_8x7b \
                --output_dir ../../trainingData/mixtral_8x7b/predictor_models \
                --embedding_dim 4096 --num_experts 8 --top_k 2

# Train a single layer:
python train.py ... --layer_idx 15

# Qwen3 30B-A3B:
python train.py --data_dir ../../trainingData/qwen3_30b \
                --output_dir ../../trainingData/qwen3_30b/predictor_models \
                --embedding_dim 2048 --num_experts 128 --top_k 8 --num_layers 48
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mlp_predictor import ExpertPredictor


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in expert selection."""

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            loss = loss * targets * self.alpha + loss * (1 - targets)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingHistoryDataset(Dataset):
    """
    Dataset for expert prediction using embeddings (current + optional history).

    Handles two .pt schemas:
      NEW  (from collect_training_data_unified.py):
           payload['layers'] = [{'layer_idx', 'embeddings' [seq,H], 'router_logits' [seq,E]}, ...]
      LEGACY: payload['data'] = {layer_idx: {token_pos: {'post_attn_post_norm_embedding', ...}}}
    """

    def __init__(
        self,
        data_source: Union[str, Path, List[Path]],
        layer_idx: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        embedding_history_size: int = 1,
    ):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.embedding_history_size = embedding_history_size
        self.samples = []

        if isinstance(data_source, (str, Path)):
            path_obj = Path(data_source)
            files = [path_obj] if path_obj.is_file() else sorted(
                list(path_obj.glob("*.pt")) or list(path_obj.glob("*.jsonl"))
            )
        else:
            files = [Path(p) for p in data_source]

        label = f"layer {layer_idx}" if layer_idx is not None else "ALL layers"
        print(f"Loading data from {len(files)} files for {label}...")
        for fp in tqdm(files):
            if str(fp).endswith('.pt'):
                self._load_pt_file(fp)
            elif str(fp).endswith('.jsonl'):
                self._load_jsonl_file(fp)

        print(f"Loaded {len(self.samples)} training samples "
              f"(history={embedding_history_size})")

    def _load_pt_file(self, pt_file: Path):
        try:
            payload = torch.load(pt_file, map_location='cpu', weights_only=False)

            # ── New unified schema ─────────────────────────────────────────────
            if 'layers' in payload:
                for layer_dict in payload['layers']:
                    l_idx = layer_dict['layer_idx']
                    if self.layer_idx is not None and l_idx != self.layer_idx:
                        continue

                    embeddings    = layer_dict['embeddings']    # [seq_len, hidden_size]
                    router_logits = layer_dict['router_logits'] # [seq_len, num_experts]
                    seq_len, hidden_size = embeddings.shape

                    for t in range(seq_len - 1):
                        emb_parts = []
                        for h in range(self.embedding_history_size):
                            src = t - (self.embedding_history_size - 1 - h)
                            emb_parts.append(
                                embeddings[src] if src >= 0 else torch.zeros(hidden_size)
                            )
                        self.samples.append({
                            'embedding_features': torch.cat(emb_parts, dim=0),  # [H*hidden]
                            'next_token_router_logits': router_logits[t + 1],
                        })
                return

            # ── Legacy schema ──────────────────────────────────────────────────
            if 'data' not in payload:
                return
            data_by_layer = payload['data']
            target_layers = (
                [self.layer_idx] if self.layer_idx is not None
                else list(data_by_layer.keys())
            )
            for l_idx in target_layers:
                if l_idx not in data_by_layer:
                    continue
                layer_data = data_by_layer[l_idx]
                sorted_tokens = sorted(layer_data.keys())
                for i in range(len(sorted_tokens) - 1):
                    t_curr, t_next = sorted_tokens[i], sorted_tokens[i + 1]
                    if t_next != t_curr + 1:
                        continue
                    emb_list = []
                    for h in range(self.embedding_history_size):
                        t_h = t_curr - (self.embedding_history_size - 1 - h)
                        emb_list.extend(
                            layer_data[t_h]['post_attn_post_norm_embedding']
                            if t_h in layer_data else [0.0] * 4096
                        )
                    self.samples.append({
                        'embedding_features': emb_list,
                        'next_token_router_logits': layer_data[t_next]['current_router_logits'],
                    })
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")

    def _load_jsonl_file(self, jsonl_file: Path):
        records = {}
        with open(jsonl_file) as f:
            for line in f:
                d = json.loads(line)
                if d.get('is_output_token', False):
                    if self.layer_idx is not None and d.get('layer_idx') != self.layer_idx:
                        continue
                    records[d['token_position']] = d

        sorted_tokens = sorted(records.keys())
        for i in range(len(sorted_tokens) - 1):
            t_curr, t_next = sorted_tokens[i], sorted_tokens[i + 1]
            if t_next != t_curr + 1:
                continue
            emb_list = []
            for h in range(self.embedding_history_size):
                t_h = t_curr - (self.embedding_history_size - 1 - h)
                rec = records.get(t_h)
                emb_list.extend(
                    rec['post_attn_post_norm_embedding'] if rec else [0.0] * 4096
                )
            s = {'embedding_features': emb_list}
            nxt = records[t_next]
            if 'selected_experts' in nxt:
                s['next_token_selected_experts'] = nxt['selected_experts']
            else:
                s['next_token_router_logits'] = nxt['current_router_logits']
            self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        emb = s['embedding_features']
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float32)

        if 'next_token_selected_experts' in s:
            top_k_idx = torch.tensor(s['next_token_selected_experts'], dtype=torch.long)
        else:
            probs = s['next_token_router_logits']
            if not isinstance(probs, torch.Tensor):
                probs = torch.tensor(probs)
            _, top_k_idx = torch.topk(torch.softmax(probs.float(), dim=0), self.top_k)

        label = torch.zeros(self.num_experts)
        label[top_k_idx] = 1.0
        return {
            'post_attn_embedding': emb.float(),
            'label': label,
            'top_k_indices': top_k_idx,
            'context_tokens': torch.zeros(1, dtype=torch.long),
            'router_logits_history': torch.zeros(1, dtype=torch.float32),
        }

    def calculate_class_weights(self):
        expert_counts = torch.zeros(self.num_experts)
        for s in self.samples:
            if 'next_token_selected_experts' in s:
                idx = s['next_token_selected_experts']
            else:
                probs = s['next_token_router_logits']
                if not isinstance(probs, torch.Tensor):
                    probs = torch.tensor(probs)
                _, idx = torch.topk(torch.softmax(probs.float(), dim=0), self.top_k)
            expert_counts[idx] += 1
        expert_counts = expert_counts.clamp(min=1)
        return (len(self.samples) - expert_counts) / expert_counts


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_logits: torch.Tensor,
                    true_logits: torch.Tensor,
                    top_k: int) -> dict:
    """
    Compute a rich set of prediction accuracy metrics for one batch.

    Both pred_logits and true_logits are [batch, num_experts] — higher = more
    confident/important.  Ranking is highest-first in both cases.

    Returned keys (all are sums over the batch; caller divides by n):
      top1_exact          — predicted #1 == actual #1
      topN_in_pred[N]     — actual top-N experts all appear in predicted top-k
                            (dict: N = 1..top_k)
      exact_set_k[k]      — set(pred[:k]) == set(true[:k])
                            (dict: k = 1..top_k)
      overlap_count       — |intersection(pred_top_k, true_top_k)| (partial credit)
      consecutive_prefix  — longest j where ordered pred[:j] == ordered true[:j]
    """
    B = pred_logits.size(0)
    # Ranked indices, best-first
    pred_ranked = torch.argsort(pred_logits, dim=1, descending=True)[:, :top_k]  # [B, top_k]
    true_ranked = torch.argsort(true_logits, dim=1, descending=True)[:, :top_k]  # [B, top_k]

    sums = {
        'top1_exact': 0,
        'topN_in_pred': {n: 0 for n in range(1, top_k + 1)},
        'exact_set_k':  {k: 0 for k in range(1, top_k + 1)},
        'overlap_count': 0,
        'any_correct': 0,
        'consecutive_prefix': 0,
    }

    pred_list = pred_ranked.tolist()
    true_list = true_ranked.tolist()

    for pred_row, true_row in zip(pred_list, true_list):
        pred_set = set(pred_row)
        true_set = set(true_row)

        # top-1 exact
        if pred_row[0] == true_row[0]:
            sums['top1_exact'] += 1

        # top-N-in-pred: are the actual top-N experts all in the predicted set?
        for n in range(1, top_k + 1):
            if set(true_row[:n]).issubset(pred_set):
                sums['topN_in_pred'][n] += 1

        # exact set match at each prefix k
        for k in range(1, top_k + 1):
            if set(pred_row[:k]) == set(true_row[:k]):
                sums['exact_set_k'][k] += 1

        # overlap count (how many of pred top-k are in true top-k)
        n_overlap = len(pred_set & true_set)
        sums['overlap_count'] += n_overlap
        if n_overlap >= 1:
            sums['any_correct'] += 1

        # consecutive ordered prefix length
        prefix = 0
        for p, t in zip(pred_row, true_row):
            if p == t:
                prefix += 1
            else:
                break
        sums['consecutive_prefix'] += prefix

    return sums, B


def _merge_metrics(acc, new_sums, new_n):
    """Accumulate metric sums returned by compute_metrics."""
    acc['n'] = acc.get('n', 0) + new_n
    acc['total_loss'] = acc.get('total_loss', 0.0) + new_sums.get('total_loss', 0.0)
    acc['top1_exact'] = acc.get('top1_exact', 0) + new_sums['top1_exact']
    acc['overlap_count'] = acc.get('overlap_count', 0.0) + new_sums['overlap_count']
    acc['any_correct']   = acc.get('any_correct', 0)     + new_sums['any_correct']
    acc['consecutive_prefix'] = acc.get('consecutive_prefix', 0.0) + new_sums['consecutive_prefix']
    for n, v in new_sums['topN_in_pred'].items():
        acc.setdefault('topN_in_pred', {})
        acc['topN_in_pred'][n] = acc['topN_in_pred'].get(n, 0) + v
    for k, v in new_sums['exact_set_k'].items():
        acc.setdefault('exact_set_k', {})
        acc['exact_set_k'][k] = acc['exact_set_k'].get(k, 0) + v
    return acc


def _finalise_metrics(acc, num_batches, top_k):
    """Divide accumulated sums by N to get averages."""
    n = acc['n']
    result = {
        'loss':               acc['total_loss'] / max(num_batches, 1),
        # exact full-set match (= exact_set_k[top_k]) for backward compat
        'acc':                acc['exact_set_k'].get(top_k, 0) / n,
        'top1_exact':         acc['top1_exact'] / n,
        'mean_overlap':       acc['overlap_count'] / n,
        'mean_prefix':        acc['consecutive_prefix'] / n,
        'any_correct':        acc['any_correct'] / n,
    }
    for n_val, v in acc.get('topN_in_pred', {}).items():
        result[f'top{n_val}_in_pred'] = v / acc['n']
    for k_val, v in acc.get('exact_set_k', {}).items():
        result[f'exact_set_{k_val}'] = v / acc['n']
    return result


def _print_metrics(phase: str, m: dict, top_k: int):
    print(f"  {phase}:")
    print(f"    loss={m['loss']:.4f}  full_match={m['acc']:.4f}  top1_exact={m['top1_exact']:.4f}")
    # top-N in pred
    topN_str = '  '.join(f"top{n}∈pred={m[f'top{n}_in_pred']:.3f}" for n in range(1, top_k + 1))
    print(f"    {topN_str}")
    # exact set match at each k
    set_str = '  '.join(f"set@{k}={m[f'exact_set_{k}']:.3f}" for k in range(1, top_k + 1))
    print(f"    {set_str}")
    print(f"    mean_overlap={m['mean_overlap']:.3f}  mean_prefix={m['mean_prefix']:.3f}  any_correct={m['any_correct']:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Train / Eval loops
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    top_k = model.top_k
    acc = {}
    num_batches = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        emb  = batch['post_attn_embedding'].to(device)
        lbl  = batch['label'].to(device)
        true = batch['top_k_indices'].to(device)  # [B, top_k] actual ranked indices

        optimizer.zero_grad()
        logits = model(post_attn_embedding=emb)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()

        # Build true_logits from top_k_indices (one-hot so argsort gives right order)
        true_logits = lbl  # already encodes which experts are active
        sums, n = compute_metrics(logits.detach(), true_logits, top_k)
        sums['total_loss'] = loss.item()
        _merge_metrics(acc, sums, n)
        num_batches += 1

    return _finalise_metrics(acc, num_batches, top_k)


def evaluate(model, loader, criterion, device):
    model.eval()
    top_k = model.top_k
    acc = {}
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            emb  = batch['post_attn_embedding'].to(device)
            lbl  = batch['label'].to(device)

            logits = model(post_attn_embedding=emb)
            loss   = criterion(logits, lbl)

            true_logits = lbl
            sums, n = compute_metrics(logits, true_logits, top_k)
            sums['total_loss'] = loss.item()
            _merge_metrics(acc, sums, n)
            num_batches += 1

    return _finalise_metrics(acc, num_batches, top_k)


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_embedding_predictor(
    data_dir: str,
    output_dir: str,
    layer_idx: int,
    embedding_history_size: int = 1,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cuda',
    base_embedding_dim: int = 4096,
    num_experts: int = 8,
    top_k: int = 2,
    hidden_dim: Optional[int] = None,
):
    """Train one per-layer expert predictor MLP."""
    _base_out = Path(output_dir) / f"layer_{layer_idx}"

    all_files = sorted(
        list(Path(data_dir).glob("*.pt")) + list(Path(data_dir).glob("*.jsonl"))
    )
    random.seed(42)
    random.shuffle(all_files)
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    val_files   = all_files[split:] or [all_files[-1]]
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    ds_kwargs = dict(
        layer_idx=layer_idx,
        num_experts=num_experts,
        top_k=top_k,
        embedding_history_size=embedding_history_size,
    )
    train_ds = EmbeddingHistoryDataset(train_files, **ds_kwargs)
    val_ds   = EmbeddingHistoryDataset(val_files,   **ds_kwargs)
    pos_weights = train_ds.calculate_class_weights().to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    input_dim        = base_embedding_dim * embedding_history_size
    _hidden_dim      = hidden_dim if hidden_dim else min(2048, max(256, input_dim // 4))
    _emb_output_dim  = min(512, max(64, input_dim // 8))
    _fusion_hidden   = min(1024, max(128, _emb_output_dim * 2))

    output_dir = _base_out / f"hidden_{_hidden_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Layer {layer_idx} — input_dim={input_dim}, hidden_dim={_hidden_dim}, "
        f"emb_out={_emb_output_dim}, fusion={_fusion_hidden}, "
        f"num_experts={num_experts}, top_k={top_k}"
    )

    model = ExpertPredictor(
        pretrained_embeddings=None,
        vocab_size=32000,
        embedding_dim=input_dim,
        num_experts=num_experts,
        top_k=top_k,
        embedding_output_dim=_emb_output_dim,
        fusion_hidden_dim=_fusion_hidden,
        use_context_tokens=False,
        use_embedding=True,
        use_router_history=False,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha=pos_weights)

    best_acc = 0.0
    metrics  = []
    metrics_path = output_dir / "training_metrics.json"

    run_config = {
        "layer_idx": layer_idx,
        "hidden_dim": _hidden_dim,
        "emb_output_dim": _emb_output_dim,
        "fusion_hidden_dim": _fusion_hidden,
        "embedding_history_size": embedding_history_size,
        "base_embedding_dim": base_embedding_dim,
        "num_experts": num_experts,
        "top_k": top_k,
    }

    def _save_metrics():
        with open(metrics_path, 'w') as f:
            json.dump({**run_config, "best_val_acc": best_acc, "epochs": metrics}, f, indent=2)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, val_loader, criterion, device)

        _print_metrics("Train", train_m, top_k)
        _print_metrics("Val",   val_m,   top_k)

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_m,
            "val":   val_m,
            # Keep flat aliases for backward compat with plot.py
            "train_loss": train_m['loss'], "train_acc": train_m['acc'],
            "val_loss":   val_m['loss'],   "val_acc":   val_m['acc'],
        }
        metrics.append(epoch_record)
        _save_metrics()

        if val_m['acc'] > best_acc:
            best_acc = val_m['acc']
            save_path = output_dir / "embedding_predictor_best.pt"
            model.save(save_path)
            print(f"  ↑ New best full_match={best_acc:.4f} — saved to {save_path}")
            _save_metrics()

    return best_acc


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DEFAULTS = {
    # model_tag: (embedding_dim, num_experts, top_k, num_layers)
    "mixtral_8x7b":  (4096, 8,   2, 32),
    "mixtral_8x22b": (4096, 8,   2, 56),
    "qwen3_30b":     (2048, 128, 8, 48),
    "qwen3_480b":    (7168, 128, 8, 94),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train expert predictor MLPs.")
    parser.add_argument('--data_dir',      required=True)
    parser.add_argument('--output_dir',    required=True)
    parser.add_argument('--layer_idx',     type=int, default=None,
                        help="Single layer to train (default: all layers)")
    parser.add_argument('--num_layers',    type=int, default=32,
                        help="Total number of layers when training all (default: 32)")
    parser.add_argument('--model',         choices=list(MODEL_DEFAULTS.keys()), default=None,
                        help="Use preset embedding_dim/num_experts/top_k/num_layers for a model")
    parser.add_argument('--history',       type=int, default=1)
    parser.add_argument('--epochs',        type=int, default=10)
    parser.add_argument('--device',        type=str, default='cuda')
    parser.add_argument('--embedding_dim', type=int, default=4096)
    parser.add_argument('--num_experts',   type=int, default=8)
    parser.add_argument('--top_k',         type=int, default=2)
    parser.add_argument('--hidden_dim',    type=int, default=None)
    parser.add_argument('--batch_size',    type=int, default=32)
    parser.add_argument('--lr',            type=float, default=1e-3)
    args = parser.parse_args()

    # Apply model preset if given
    emb_dim, n_exp, k, n_layers = (
        MODEL_DEFAULTS[args.model] if args.model else
        (args.embedding_dim, args.num_experts, args.top_k, args.num_layers)
    )

    shared = dict(
        embedding_history_size=args.history,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        base_embedding_dim=emb_dim,
        num_experts=n_exp,
        top_k=k,
        hidden_dim=args.hidden_dim,
    )

    layers = [args.layer_idx] if args.layer_idx is not None else list(range(n_layers))
    layer_accs = {}
    for i in layers:
        print(f"\n{'='*60}\nTraining Layer {i}/{n_layers-1}\n{'='*60}")
        acc = train_embedding_predictor(args.data_dir, args.output_dir, i, **shared)
        layer_accs[i] = acc

    # Summary table
    print("\n" + "="*60)
    print("Layer-wise best validation accuracy:")
    print("-"*30)
    for i, acc in sorted(layer_accs.items()):
        print(f"  Layer {i:3d}: {acc:.4f}")
    print("="*60)
