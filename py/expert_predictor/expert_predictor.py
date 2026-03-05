#!/usr/bin/env python3
"""
expert_predictor.py — Expert Predictor Training and Sweeping
============================================================
A unified script containing model definitions, data loaders, training loops,
and sweeping capabilities.

Usage Examples
--------------
Train all layers for Mixtral 8x7B:
  python expert_predictor.py train --data_dir ../../trainingData/mixtral_8x7b \\
                                   --output_dir ../../trainingData/mixtral_8x7b/predictor_models \\
                                   --model mixtral_8x7b

Comprehensive Sweep:
  python expert_predictor.py sweep --data_dir ../../trainingData/mixtral_8x7b \\
                                   --output_dir ../../trainingData/mixtral_8x7b/sweep_results \\
                                   --model mixtral_8x7b \\
                                   --hidden_dims 256 512 1024 \\
                                   --max_size_mb 50
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model Definition
# ──────────────────────────────────────────────────────────────────────────────

class DualMLPPredictor(nn.Module):
    """
    Dual-branch expert predictor for MoE models.

    Branch 1 — EmbeddingBranch:
        Input:  post-attention-norm embedding for token t  [batch, hidden_size]
        Learns the local, token-level semantic context.

    Branch 2 — PrefillBranch:
        Input:  per-layer expert usage distribution over the full prefill
                sequence for this prompt  [batch, num_experts]
                (= mean softmax of router_logits over all prefill tokens)
        Learns the global "expert prior" for this prompt.

    Both branches are projected to feature vectors, concatenated, then
    passed through a classifier head that outputs logits over all experts.
    The predicted top-1 (or top-k) expert is used for preload decisions.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_experts: int = 8,
        branch_dim: int = 256,
        prefetch_k: int = 1,
        dropout: float = 0.1,
        use_embedding: bool = True,
        use_prefill: bool = True,
    ):
        super().__init__()
        if not use_embedding and not use_prefill:
            raise ValueError("Must use at least one branch (embedding or prefill).")

        self.hidden_size   = hidden_size
        self.num_experts   = num_experts
        self.branch_dim    = branch_dim
        self.top_k         = prefetch_k   # kept as .top_k for train loop compat
        self.prefetch_k    = prefetch_k
        self.use_embedding = use_embedding
        self.use_prefill   = use_prefill

        prefill_dim = max(branch_dim // 4, num_experts)
        fused_dim = 0

        # Branch 1: token embedding → branch_dim features
        if self.use_embedding:
            self.embedding_branch = nn.Sequential(
                nn.Linear(self.hidden_size, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fused_dim += branch_dim

        # Branch 2: prefill expert dist → small feature vector
        if self.use_prefill:
            self.prefill_branch = nn.Sequential(
                nn.Linear(num_experts, prefill_dim),
                nn.GELU(),
            )
            fused_dim += prefill_dim

        # Fused classifier
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_experts),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        prefill_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embedding:    [batch, hidden_size]  post-attn-norm embedding at token t
            prefill_dist: [batch, num_experts]  expert usage distribution over the
                          full prefill. If None, uniform distribution is used.
        Returns:
            logits: [batch, num_experts]
        """
        features = []
        
        if self.use_embedding:
            features.append(self.embedding_branch(embedding))

        if self.use_prefill:
            if prefill_dist is None:
                prefill_dist = torch.full(
                    (embedding.size(0), self.num_experts),
                    1.0 / self.num_experts,
                    dtype=embedding.dtype,
                    device=embedding.device,
                )
            features.append(self.prefill_branch(prefill_dist))

        fused = torch.cat(features, dim=-1)
        return self.classifier(fused)

    def predict_top_k(
        self,
        embedding: torch.Tensor,
        prefill_dist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(embedding, prefill_dist)
        probs  = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        return top_k_indices, top_k_probs

    def get_config(self) -> Dict:
        return {
            'hidden_size':   self.hidden_size,
            'num_experts':   self.num_experts,
            'branch_dim':    self.branch_dim,
            'prefetch_k':    self.prefetch_k,
            'use_embedding': self.use_embedding,
            'use_prefill':   self.use_prefill
        }


    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_path = path.with_suffix('.pth') if path.suffix == '.pt' else path
        torch.save({'model_state_dict': self.state_dict(), 'config': self.get_config()}, state_path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'DualMLPPredictor':
        path_obj = Path(path)
        if path_obj.suffix == '.pt' and path_obj.with_suffix('.pth').exists():
            checkpoint = torch.load(path_obj.with_suffix('.pth'), map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class HistoryAwareDualMLPPredictor(nn.Module):
    """
    Three-branch expert predictor.

    Branch 1 — EmbeddingBranch:   post-attn-norm embedding at token t   [batch, hidden_size]
    Branch 2 — PrevExpertBranch:  one-hot expert selection at token t    [batch, num_experts]
                                  (scatter of the top-k IDs chosen at step t)
    Branch 3 — PrefillBranch:     expert usage counts over prefill        [batch, num_experts]

    For ablation, each branch can be independently enabled/disabled.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_experts: int = 8,
        branch_dim: int = 256,
        prefetch_k: int = 1,
        dropout: float = 0.1,
        use_embedding: bool = True,
        use_prev_experts: bool = True,
        use_prefill: bool = True,
    ):
        super().__init__()
        if not any([use_embedding, use_prev_experts, use_prefill]):
            raise ValueError("Must enable at least one branch.")

        self.hidden_size      = hidden_size
        self.num_experts      = num_experts
        self.branch_dim       = branch_dim
        self.top_k            = prefetch_k   # kept as .top_k for train loop compat
        self.prefetch_k       = prefetch_k
        self.use_embedding    = use_embedding
        self.use_prev_experts = use_prev_experts
        self.use_prefill      = use_prefill

        prefill_dim    = max(branch_dim // 4, num_experts)
        prev_exp_dim   = max(branch_dim // 4, num_experts)
        fused_dim = 0

        if self.use_embedding:
            self.embedding_branch = nn.Sequential(
                nn.Linear(self.hidden_size, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fused_dim += branch_dim

        if self.use_prev_experts:
            # Input: one-hot/count vector over num_experts indicating which were active last step
            self.prev_expert_branch = nn.Sequential(
                nn.Linear(num_experts, prev_exp_dim),
                nn.GELU(),
            )
            fused_dim += prev_exp_dim

        if self.use_prefill:
            self.prefill_branch = nn.Sequential(
                nn.Linear(num_experts, prefill_dim),
                nn.GELU(),
            )
            fused_dim += prefill_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_experts),
        )

    def forward(
        self,
        embedding: Optional[torch.Tensor] = None,
        prev_expert_onehot: Optional[torch.Tensor] = None,
        prefill_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        embedding:          [batch, hidden_size]
        prev_expert_onehot: [batch, num_experts]  — one-hot/count of experts at step t
        prefill_dist:       [batch, num_experts]
        """
        features = []
        ref = next(x for x in [embedding, prev_expert_onehot, prefill_dist] if x is not None)
        batch = ref.size(0)

        if self.use_embedding:
            assert embedding is not None
            features.append(self.embedding_branch(embedding))

        if self.use_prev_experts:
            if prev_expert_onehot is None:
                prev_expert_onehot = torch.zeros(batch, self.num_experts,
                                                  dtype=ref.dtype, device=ref.device)
            features.append(self.prev_expert_branch(prev_expert_onehot))

        if self.use_prefill:
            if prefill_dist is None:
                prefill_dist = torch.full((batch, self.num_experts),
                                          1.0 / self.num_experts,
                                          dtype=ref.dtype, device=ref.device)
            features.append(self.prefill_branch(prefill_dist))

        fused = torch.cat(features, dim=-1)
        return self.classifier(fused)

    def predict_top_k(self, embedding, prev_expert_onehot=None, prefill_dist=None):
        logits = self.forward(embedding, prev_expert_onehot, prefill_dist)
        probs  = F.softmax(logits, dim=-1)
        return torch.topk(probs, self.top_k, dim=-1)

    def get_config(self) -> Dict:
        return {
            'hidden_size':      self.hidden_size,
            'num_experts':      self.num_experts,
            'branch_dim':       self.branch_dim,
            'prefetch_k':       self.prefetch_k,
            'use_embedding':    self.use_embedding,
            'use_prev_experts': self.use_prev_experts,
            'use_prefill':      self.use_prefill,
        }

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_path = path.with_suffix('.pth') if path.suffix == '.pt' else path
        torch.save({'model_state_dict': self.state_dict(), 'config': self.get_config(),
                    'model_class': 'HistoryAwareDualMLPPredictor'}, state_path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'HistoryAwareDualMLPPredictor':
        path_obj = Path(path)
        ckpt_path = path_obj.with_suffix('.pth') if (path_obj.suffix == '.pt' and path_obj.with_suffix('.pth').exists()) else path_obj
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class ExpertPredictor(nn.Module):
    # Dummy fallback in case ExpertPredictor is needed (was referenced in train.py but not defined there)
    pass

class PredictorJITWrapper(torch.nn.Module):
    """
    Thin JIT-traceable wrapper that presents a single fixed signature:
        forward(embedding, prefill_dist, prev_expert_onehot) -> logits
    Branches that are disabled in the underlying model receive a zero tensor
    and are short-circuited internally, so the traced graph is always correct.
    """
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.uses_prefill      = getattr(predictor, 'use_prefill',      False)
        self.uses_prev_experts = getattr(predictor, 'use_prev_experts', False)
        self.is_history_aware  = isinstance(predictor, HistoryAwareDualMLPPredictor)

    def forward(
        self,
        embedding:          torch.Tensor,          # [B, hidden]
        prefill_dist:       torch.Tensor,          # [B, num_experts]  (zeros if unused)
        prev_expert_onehot: torch.Tensor,          # [B, num_experts]  (zeros if unused)
    ) -> torch.Tensor:
        # Always pass tensors (never None) so torch.jit.trace can follow the graph.
        # The model's branches are gated by use_prefill / use_prev_experts flags,
        # so passing zeros for unused inputs has no effect on output.
        if self.is_history_aware:
            return self.predictor(
                embedding=embedding,
                prev_expert_onehot=prev_expert_onehot,
                prefill_dist=prefill_dist,
            )
        else:
            return self.predictor(embedding=embedding, prefill_dist=prefill_dist)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Loss & Datasets
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
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        return loss


class EmbeddingHistoryDataset(Dataset):
    def __init__(
        self,
        data_source: Union[str, Path, List[Path]],
        layer_idx: Optional[int] = None,
        num_experts: int = 8,
        active_k: int = 2,     # how many experts the router picks (defines the label)
        prefetch_k: int = 1,   # how many we predict/prefetch (used for metrics only)
        embedding_history_size: int = 1,
    ):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.active_k = active_k
        self.prefetch_k = prefetch_k
        self.top_k = prefetch_k       # kept for train loop compat
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

        print(f"Loaded {len(self.samples)} training samples (history={embedding_history_size})")

    def _load_pt_file(self, pt_file: Path):
        try:
            payload = torch.load(pt_file, map_location='cpu', weights_only=False)
            if 'layers' in payload:
                for layer_dict in payload['layers']:
                    l_idx = layer_dict['layer_idx']
                    if self.layer_idx is not None and l_idx != self.layer_idx: continue

                    embeddings    = layer_dict['embeddings']
                    router_logits = layer_dict['router_logits']
                    prev_expert_ids = layer_dict.get('prev_expert_ids')  # [gen_len, top_k] or None
                    seq_len, hidden_size = embeddings.shape

                    for t in range(seq_len - 1):
                        emb_parts = []
                        for h in range(self.embedding_history_size):
                            src = t - (self.embedding_history_size - 1 - h)
                            emb_parts.append(embeddings[src] if src >= 0 else torch.zeros(hidden_size))

                        prefill_expert_dist = layer_dict.get('prefill_expert_dist')
                        if prefill_expert_dist is None and 'prefill_expert_count' in layer_dict:
                            c = layer_dict['prefill_expert_count'].float()
                            prefill_expert_dist = c / c.sum().clamp(min=1)
                        if prefill_expert_dist is None:
                            prefill_expert_dist = torch.softmax(router_logits[:t+1].float(), dim=-1).mean(dim=0)

                        # Build one-hot for the experts chosen AT step t (feature for predicting t+1)
                        if prev_expert_ids is not None:
                            ids_t = prev_expert_ids[t]  # [top_k]
                            prev_onehot = torch.zeros(self.num_experts)
                            prev_onehot.scatter_(0, ids_t.long(), 1.0)
                        else:
                            prev_onehot = None  # will be filled with zeros in __getitem__

                        sample = {
                            'embedding_features':      torch.cat(emb_parts, dim=0),
                            'next_token_router_logits': router_logits[t + 1],
                            'prefill_expert_dist':      prefill_expert_dist,
                            'prev_expert_onehot':       prev_onehot,
                        }
                        self.samples.append(sample)
                return

            if 'data' not in payload: return
            data_by_layer = payload['data']
            target_layers = [self.layer_idx] if self.layer_idx is not None else list(data_by_layer.keys())
            for l_idx in target_layers:
                if l_idx not in data_by_layer: continue
                layer_data = data_by_layer[l_idx]
                sorted_tokens = sorted(layer_data.keys())
                for i in range(len(sorted_tokens) - 1):
                    t_curr, t_next = sorted_tokens[i], sorted_tokens[i + 1]
                    if t_next != t_curr + 1: continue
                    emb_list = []
                    for h in range(self.embedding_history_size):
                        t_h = t_curr - (self.embedding_history_size - 1 - h)
                        emb_list.extend(layer_data[t_h]['post_attn_post_norm_embedding'] if t_h in layer_data else [0.0] * 4096)
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
                    if self.layer_idx is not None and d.get('layer_idx') != self.layer_idx: continue
                    records[d['token_position']] = d
        sorted_tokens = sorted(records.keys())
        for i in range(len(sorted_tokens) - 1):
            t_curr, t_next = sorted_tokens[i], sorted_tokens[i + 1]
            if t_next != t_curr + 1: continue
            emb_list = []
            for h in range(self.embedding_history_size):
                t_h = t_curr - (self.embedding_history_size - 1 - h)
                rec = records.get(t_h)
                emb_list.extend(rec['post_attn_post_norm_embedding'] if rec else [0.0] * 4096)
            s = {'embedding_features': emb_list}
            nxt = records[t_next]
            if 'selected_experts' in nxt: s['next_token_selected_experts'] = nxt['selected_experts']
            else: s['next_token_router_logits'] = nxt['current_router_logits']
            self.samples.append(s)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        emb = s['embedding_features']
        if not isinstance(emb, torch.Tensor): emb = torch.tensor(emb, dtype=torch.float32)

        if 'next_token_selected_experts' in s:
            active_idx = torch.tensor(s['next_token_selected_experts'], dtype=torch.long)
        else:
            probs = s['next_token_router_logits']
            if not isinstance(probs, torch.Tensor): probs = torch.tensor(probs)
            # Label: the active_k experts that actually fire at t+1
            _, active_idx = torch.topk(torch.softmax(probs.float(), dim=0), self.active_k)

        # Multi-hot label over all num_experts: 1 for each of the active_k experts
        label = torch.zeros(self.num_experts)
        label[active_idx] = 1.0
        prev_onehot = s.get('prev_expert_onehot')
        if prev_onehot is None:
            prev_onehot = torch.zeros(self.num_experts)
        return {
            'post_attn_embedding':  emb.float(),
            'label':                label,
            'active_indices':       active_idx,   # [active_k] ground-truth experts that fire
            'prefill_expert_dist':  s.get('prefill_expert_dist', torch.full((self.num_experts,), 1.0 / self.num_experts)),
            'prev_expert_onehot':   prev_onehot.float(),
        }

    def calculate_class_weights(self):
        expert_counts = torch.zeros(self.num_experts)
        for s in self.samples:
            if 'next_token_selected_experts' in s:
                idx = s['next_token_selected_experts']
            else:
                probs = s['next_token_router_logits']
                if not isinstance(probs, torch.Tensor): probs = torch.tensor(probs)
                _, idx = torch.topk(torch.softmax(probs.float(), dim=0), self.active_k)
            expert_counts[idx] += 1
        expert_counts = expert_counts.clamp(min=1)
        return (len(self.samples) - expert_counts) / expert_counts


# ──────────────────────────────────────────────────────────────────────────────
# 3. Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_logits: torch.Tensor, true_label: torch.Tensor,
                    prefetch_k: int, active_k: Optional[int] = None) -> dict:
    """
    pred_logits:  [B, num_experts]  raw logits from model
    true_label:   [B, num_experts]  multi-hot: 1.0 at each of the active_k experts that fired
    prefetch_k:   how many experts we predict/prefetch (our budget)
    active_k:     how many experts actually fire (inferred from true_label if None)

    Positive class = expert is in the active set (true_label==1).
    Predicted positive = expert is in our top-prefetch_k predictions.

    top1_exact: our single top-1 prediction is in the active set.
    any_correct: at least 1 of our prefetch_k predictions is in the active set.
    mean_overlap: average |intersection| between our prefetch_k picks and active set.
    """
    B = pred_logits.size(0)
    num_experts = pred_logits.size(1)
    if active_k is None:
        active_k = int(true_label[0].sum().item())

    # Our prediction: top-prefetch_k experts
    pred_ranked  = torch.argsort(pred_logits, dim=1, descending=True)[:, :prefetch_k]
    pred_mask    = torch.zeros(B, num_experts, dtype=torch.bool, device=pred_logits.device)
    pred_mask.scatter_(1, pred_ranked, True)

    # Ground truth: the multi-hot active set
    true_mask = true_label.bool().to(pred_logits.device)

    # Per-expert TP/FP/FN/TN -> [num_experts]
    tp_per = (pred_mask &  true_mask).sum(0).cpu().float()
    fp_per = (pred_mask & ~true_mask).sum(0).cpu().float()
    fn_per = (~pred_mask &  true_mask).sum(0).cpu().float()
    tn_per = (~pred_mask & ~true_mask).sum(0).cpu().float()

    sums = {
        'top1_exact': 0, 'any_correct': 0, 'overlap_count': 0,
        'tp_per': tp_per, 'fp_per': fp_per, 'fn_per': fn_per, 'tn_per': tn_per,
    }

    pred_list = pred_ranked.tolist()
    for i, pred_row in enumerate(pred_list):
        active_set = true_mask[i].nonzero(as_tuple=True)[0].tolist()
        active_set_s = set(active_set)
        # top1_exact: is our top-1 in the active set?
        if pred_row[0] in active_set_s:
            sums['top1_exact'] += 1
        n_overlap = len(set(pred_row) & active_set_s)
        sums['overlap_count'] += n_overlap
        if n_overlap >= 1:
            sums['any_correct'] += 1

    return sums, B


def _merge_metrics(acc, new_sums, new_n):
    acc['n'] = acc.get('n', 0) + new_n
    acc['total_loss']    = acc.get('total_loss', 0.0) + new_sums.get('total_loss', 0.0)
    acc['top1_exact']    = acc.get('top1_exact', 0)   + new_sums['top1_exact']
    acc['overlap_count'] = acc.get('overlap_count', 0.0) + new_sums['overlap_count']
    acc['any_correct']   = acc.get('any_correct', 0)  + new_sums['any_correct']
    for key in ('tp_per', 'fp_per', 'fn_per', 'tn_per'):
        if key in new_sums:
            acc[key] = acc.get(key, torch.zeros_like(new_sums[key])) + new_sums[key]
    return acc

def _finalise_metrics(acc, num_batches, prefetch_k, active_k):
    n = acc['n']
    result = {
        'loss':         acc['total_loss'] / max(num_batches, 1),
        'top1_exact':   acc['top1_exact'] / n,
        'mean_overlap': acc['overlap_count'] / n,
        'any_correct':  acc['any_correct'] / n,
        # convenience alias
        'acc':          acc['top1_exact'] / n,
    }
    # F1 / TP/FP etc
    if 'tp_per' in acc:
        tp, fp, fn, tn = acc['tp_per'], acc['fp_per'], acc['fn_per'], acc['tn_per']
        micro_tp = tp.sum().item()
        micro_fp = fp.sum().item()
        micro_fn = fn.sum().item()
        micro_prec = micro_tp / max(micro_tp + micro_fp, 1e-9)
        micro_rec  = micro_tp / max(micro_tp + micro_fn, 1e-9)
        micro_f1   = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-9)
        per_prec = tp / (tp + fp).clamp(min=1e-9)
        per_rec  = tp / (tp + fn).clamp(min=1e-9)
        per_f1   = 2 * per_prec * per_rec / (per_prec + per_rec).clamp(min=1e-9)
        total    = (tp + fp + fn + tn).clamp(min=1)
        result.update({
            'micro_precision': micro_prec,
            'micro_recall':    micro_rec,
            'micro_f1':        micro_f1,
            'macro_f1':        per_f1.mean().item(),
            'per_expert_f1':   per_f1.tolist(),
            'per_expert_acc':  ((tp + tn) / total).tolist(),
            'per_expert_tp':   tp.long().tolist(),
            'per_expert_fp':   fp.long().tolist(),
            'per_expert_fn':   fn.long().tolist(),
            'per_expert_tn':   tn.long().tolist(),
        })
    return result

def _print_metrics(phase: str, m: dict, prefetch_k: int):
    print(f"  {phase}:")
    top1 = m['top1_exact']
    print(f"    loss={m['loss']:.4f}")
    print(f"    ─── Prefetch accuracy (primary system metric) ─────────────")
    print(f"    top1_exact (prefetch hit rate) = {top1:.4f}  [random=0.125]")
    # With Mixtral top-2 routing + LRU cache: we load 1 expert speculatively.
    # If we're right, only 1 load needed instead of 2.
    # Expected loads/token = 2 - top1_exact  (save top1_exact loads on average).
    if prefetch_k >= 2:
        exp_loads = 2.0 - top1
        reduction_pct = (2.0 - exp_loads) / 2.0 * 100
        print(f"    expected loads/token ≈ {exp_loads:.3f}  "
              f"({reduction_pct:.1f}% load reduction vs no predictor)")
    print(f"    any_correct={m['any_correct']:.4f}  "
          f"mean_overlap={m['mean_overlap']:.3f}")
    if 'micro_f1' in m:
        print(f"    ─── ML metrics (multi-label, top-{prefetch_k} as positive class) ──")
        print(f"    micro_F1={m['micro_f1']:.4f}  macro_F1={m['macro_f1']:.4f}  "
              f"prec={m['micro_precision']:.4f}  rec={m['micro_recall']:.4f}")
        if 'per_expert_f1' in m and len(m['per_expert_f1']) <= 16:
            pef = '  '.join(f"E{i}:{v:.3f}" for i, v in enumerate(m['per_expert_f1']))
            pea = '  '.join(f"E{i}:{v:.3f}" for i, v in enumerate(m.get('per_expert_acc', [])))
            print(f"    per-expert F1:  {pef}")
            print(f"    per-expert acc: {pea}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training Engine
# ──────────────────────────────────────────────────────────────────────────────

def _model_forward(model, batch_on_device):
    """Unified forward dispatch for all predictor model types."""
    emb   = batch_on_device.get('post_attn_embedding')
    pdist = batch_on_device.get('prefill_expert_dist')
    prev  = batch_on_device.get('prev_expert_onehot')
    if isinstance(model, HistoryAwareDualMLPPredictor):
        return model(embedding=emb, prev_expert_onehot=prev, prefill_dist=pdist)
    elif isinstance(model, DualMLPPredictor):
        return model(embedding=emb, prefill_dist=pdist)
    else:
        return model(post_attn_embedding=emb)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    prefetch_k = model.prefetch_k
    acc = {}
    num_batches = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        lbl    = batch_dev['label']             # multi-hot [B, num_experts]
        active = batch_dev['active_indices']    # [B, active_k] ground truth experts

        optimizer.zero_grad()
        logits = _model_forward(model, batch_dev)

        if isinstance(criterion, nn.CrossEntropyLoss):
            loss = criterion(logits, active[:, 0])  # supervise on top-1 active expert
        else:
            loss = criterion(logits, lbl)

        loss.backward()
        optimizer.step()

        active_k = active.size(1)
        sums, n = compute_metrics(logits.detach(), lbl, prefetch_k, active_k)
        sums['total_loss'] = loss.item()
        _merge_metrics(acc, sums, n)
        num_batches += 1

    return _finalise_metrics(acc, num_batches, prefetch_k, active_k)

def evaluate(model, loader, criterion, device):
    model.eval()
    prefetch_k = model.prefetch_k
    acc = {}
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            lbl    = batch_dev['label']
            active = batch_dev['active_indices']

            logits = _model_forward(model, batch_dev)

            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(logits, active[:, 0])
            else:
                loss = criterion(logits, lbl)

            active_k = active.size(1)
            sums, n = compute_metrics(logits, lbl, prefetch_k, active_k)
            sums['total_loss'] = loss.item()
            _merge_metrics(acc, sums, n)
            num_batches += 1

    return _finalise_metrics(acc, num_batches, prefetch_k, active_k)

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
    active_k: int = 2,       # how many experts the router selects (label size)
    prefetch_k: int = 1,     # how many experts we predict and prefetch
    hidden_dim: Optional[int] = None,
    loss_type: str = 'ce',
    best_by: str = 'top1_exact',
    model_type: str = 'dual_mlp',
    use_embedding: bool = True,
    use_prefill: bool = True,
    use_prev_experts: bool = False,
):
    _base_out = Path(output_dir) / f"layer_{layer_idx}"

    all_files = sorted(list(Path(data_dir).glob("*.pt")) + list(Path(data_dir).glob("*.jsonl")))
    random.seed(42)
    random.shuffle(all_files)
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    val_files   = all_files[split:] or [all_files[-1]]
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    ds_kwargs = dict(
        layer_idx=layer_idx, num_experts=num_experts,
        active_k=active_k, prefetch_k=prefetch_k,
        embedding_history_size=embedding_history_size,
    )
    train_ds = EmbeddingHistoryDataset(train_files, **ds_kwargs)
    val_ds   = EmbeddingHistoryDataset(val_files,   **ds_kwargs)
    pos_weights = train_ds.calculate_class_weights().to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    input_dim   = base_embedding_dim * embedding_history_size
    _hidden_dim = hidden_dim if hidden_dim else min(2048, max(256, input_dim // 4))
    output_dir  = _base_out / f"hidden_{_hidden_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Layer {layer_idx} — input_dim={input_dim}, hidden_dim={_hidden_dim}, "
          f"num_experts={num_experts}, active_k={active_k}, prefetch_k={prefetch_k}, "
          f"model_type={model_type}, use_embedding={use_embedding}, "
          f"use_prefill={use_prefill}, use_prev_experts={use_prev_experts}")

    if use_prev_experts:
        model = HistoryAwareDualMLPPredictor(
            hidden_size=input_dim, num_experts=num_experts,
            branch_dim=_hidden_dim, prefetch_k=prefetch_k,
            use_embedding=use_embedding, use_prefill=use_prefill,
            use_prev_experts=True,
        ).to(device)
    elif model_type == 'dual_mlp':
        model = DualMLPPredictor(
            hidden_size=input_dim, num_experts=num_experts,
            branch_dim=_hidden_dim, prefetch_k=prefetch_k,
            use_embedding=use_embedding, use_prefill=use_prefill,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    if loss_type == 'ce': criterion = nn.CrossEntropyLoss()
    elif loss_type == 'bce': criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else: criterion = FocalLoss(alpha=pos_weights)

    best_acc = 0.0
    best_metrics = {}
    metrics  = []
    metrics_path = output_dir / "training_metrics.json"

    run_config = {
        "layer_idx": layer_idx, "hidden_dim": _hidden_dim,
        "embedding_history_size": embedding_history_size,
        "base_embedding_dim": base_embedding_dim,
        "num_experts": num_experts, "active_k": active_k, "prefetch_k": prefetch_k,
        "use_embedding": use_embedding,
        "use_prefill": use_prefill,
        "use_prev_experts": use_prev_experts,
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_m.get(best_by, val_m['acc']))

        _print_metrics("Train", train_m, prefetch_k)
        _print_metrics("Val",   val_m,   prefetch_k)

        epoch_record = {
            "epoch": epoch + 1, "train": train_m, "val": val_m,
            "train_loss": train_m['loss'], "train_acc": train_m['acc'],
            "val_loss": val_m['loss'], "val_acc": val_m['acc'],
        }
        metrics.append(epoch_record)

        current_score = val_m.get(best_by, val_m['acc'])
        if current_score > best_acc:
            best_acc = current_score
            best_metrics = val_m
            save_path = output_dir / "embedding_predictor_best.pt"
            model.save(save_path)
            
            try:
                model.eval()
                wrapper = PredictorJITWrapper(model)
                wrapper.eval()
                # Always provide all three inputs so trace captures full graph.
                # Disabled branches will receive zeros and ignore them internally.
                num_exp = model.num_experts
                example_emb    = torch.randn(1, input_dim).to(device)
                example_pfill  = torch.zeros(1, num_exp).to(device)
                example_prev   = torch.zeros(1, num_exp).to(device)
                with torch.no_grad():
                    traced_model = torch.jit.trace(
                        wrapper, (example_emb, example_pfill, example_prev)
                    )
                traced_model.save(str(save_path))
                model.train()
            except Exception as e:
                print(f"  [warn] Failed to auto-save JIT model: {e}")

            print(f"  ↑ New best {best_by}={best_acc:.4f} — saved to {save_path} (and JIT)")
        
        with open(metrics_path, 'w') as f:
            json.dump({**run_config, "best_val_acc": best_acc, "best_by": best_by, "epochs": metrics}, f, indent=2)

    return best_metrics


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI Definition
# ──────────────────────────────────────────────────────────────────────────────

# emb_dim, num_experts, active_k (router top-k), num_layers, default_prefetch_k
MODEL_DEFAULTS = {
    "mixtral_8x7b":  (4096, 8,   2, 32, 1),
    "mixtral_8x22b": (4096, 8,   2, 56, 1),
    "qwen3_30b":     (2048, 128, 8, 48, 4),
    "qwen3_480b":    (7168, 128, 8, 94, 4),
}

def calculate_model_size_mb(input_dim, hidden_dim, output_dim):
    params = (input_dim * hidden_dim) + hidden_dim
    params += (hidden_dim * (hidden_dim // 2)) + (hidden_dim // 2)
    params += ((hidden_dim // 2) * output_dim) + output_dim
    params += (hidden_dim * 2) + ((hidden_dim // 2) * 2)
    return (params * 4) / (1024 * 1024)

def run_train(args):
    if args.model:
        emb_dim, n_exp, default_active_k, n_layers, default_prefetch_k = MODEL_DEFAULTS[args.model]
    else:
        emb_dim, n_exp, default_active_k, n_layers, default_prefetch_k = args.embedding_dim, args.num_experts, 2, args.num_layers, 1
    active_k   = args.active_k   if args.active_k   is not None else default_active_k
    prefetch_k = args.prefetch_k if args.prefetch_k is not None else default_prefetch_k
    shared = dict(
        embedding_history_size=args.history, num_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device, base_embedding_dim=emb_dim, num_experts=n_exp,
        active_k=active_k, prefetch_k=prefetch_k, hidden_dim=args.hidden_dim, loss_type=args.loss_type,
        model_type=args.model_type, best_by=args.best_by
    )
    layers = [args.layer_idx] if args.layer_idx is not None else list(range(n_layers))
    layer_accs = {}
    for i in layers:
        print(f"\n{'='*60}\nTraining Layer {i}/{n_layers-1}\n{'='*60}")
        acc = train_embedding_predictor(args.data_dir, args.output_dir, i, **shared)
        layer_accs[i] = acc
    print("\n" + "="*60 + "\nLayer-wise best validation accuracy:\n" + "-"*30)
    for i, acc in sorted(layer_accs.items()): print(f"  Layer {i:3d}: {acc:.4f}")
    print("="*60)

def run_sweep(args):
    if args.model:
        emb_dim, n_exp, default_active_k, n_layers, default_prefetch_k = MODEL_DEFAULTS[args.model]
    else:
        emb_dim, n_exp, default_active_k, n_layers, default_prefetch_k = args.embedding_dim, args.num_experts, 2, args.num_layers, 1
    active_k   = args.active_k   if args.active_k   is not None else default_active_k
    prefetch_k = args.prefetch_k if args.prefetch_k is not None else default_prefetch_k
    layers = args.layers if args.layers is not None else list(range(n_layers))
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_configs = []
    OVERHEAD = 1.25
    for history in args.histories:
        for h_dim in args.hidden_dims:
            est = calculate_model_size_mb(emb_dim * history, h_dim, n_exp) * OVERHEAD
            if est <= args.max_size_mb:
                valid_configs.append((history, h_dim, est))
                print(f"Valid: emb_hist={history}, hidden={h_dim} (Est Per-Layer: {est:.1f}MB)")
            else:
                print(f"Skipping: hist={history}, hidden={h_dim} ({est:.1f}MB > {args.max_size_mb}MB)")

    total, done = len(layers) * len(valid_configs), 0
    results = []
    for history, hidden_dim, size_mb in valid_configs:
        for layer_idx in layers:
            done += 1
            print(f"\n{'='*60}\n Run {done}/{total}: Layer {layer_idx}, EmbHist {history}, Hidden {hidden_dim} ({size_mb:.1f}MB)\n{'='*60}")
            out_dir = output_path / f"eh{history}_h{hidden_dim}"
            try:
                metrics = train_embedding_predictor(
                    data_dir=args.data_dir, output_dir=str(out_dir), layer_idx=layer_idx,
                    embedding_history_size=history, hidden_dim=hidden_dim, num_epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, device=args.device, base_embedding_dim=emb_dim,
                    num_experts=n_exp, active_k=active_k, prefetch_k=prefetch_k,
                    loss_type=args.loss_type, best_by=args.best_by, model_type=args.model_type
                )
                results.append({
                    "layer": layer_idx, "history": history, "hidden_dim": hidden_dim,
                    "size_mb": size_mb, **{f"val_{key}": v for key, v in metrics.items()}
                })
                with open(output_path / "sweep_summary.json", "w") as f: json.dump(results, f, indent=2)
            except Exception as e:
                print(f"[ERROR] Run failed: {e}", file=sys.stderr)

    print(f"\nSweep complete. Best architectures per layer (by {args.best_by}):")
    for layer in layers:
        layer_res = [r for r in results if r['layer'] == layer]
        if not layer_res: continue
        best = max(layer_res, key=lambda x: x[f'val_{args.best_by}'])
        print(f"  Layer {layer:2d}: Best is EmbHist={best['history']}, Hidden={best['hidden_dim']} "
              f"(Top-1 Exact: {best['val_top1_exact']:.4f}, Mean Overlap: {best.get('val_mean_overlap', 0):.4f})")

def run_ablation(args):
    if args.model:
        emb_dim, n_exp, default_active_k, n_layers, default_prefetch_k = MODEL_DEFAULTS[args.model]
    else:
        emb_dim, n_exp, default_active_k, n_layers, default_prefetch_k = args.embedding_dim, args.num_experts, 2, args.num_layers, 1
    active_k   = args.active_k   if args.active_k   is not None else default_active_k
    prefetch_k = args.prefetch_k if args.prefetch_k is not None else default_prefetch_k
    layers = args.layers if args.layers is not None else [15]  # default middle layer
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Each variant specifies which feature branches to enable.
    # use_prev_experts=True  -> HistoryAwareDualMLPPredictor
    # use_prev_experts=False -> DualMLPPredictor (original)
    variants = [
        # ── Baselines (no prev-expert branch) ────────────────────────────
        {"name": "emb_only",             "use_embedding": True,  "use_prev_experts": False, "use_prefill": False},
        {"name": "prefill_only",         "use_embedding": False, "use_prev_experts": False, "use_prefill": True},
        {"name": "prev_only",            "use_embedding": False, "use_prev_experts": True,  "use_prefill": False},
        # ── Two-branch combos ─────────────────────────────────────────────
        {"name": "emb_prefill",          "use_embedding": True,  "use_prev_experts": False, "use_prefill": True},
        {"name": "emb_prev",             "use_embedding": True,  "use_prev_experts": True,  "use_prefill": False},
        {"name": "prev_prefill",         "use_embedding": False, "use_prev_experts": True,  "use_prefill": True},
        # ── Full three-branch model ────────────────────────────────────────
        {"name": "emb_prev_prefill",     "use_embedding": True,  "use_prev_experts": True,  "use_prefill": True},
    ]

    total = len(layers) * len(variants)
    done = 0
    results = []

    print(f"\n=== Starting Ablation Study ===")
    print(f"  Layers:   {layers}")
    print(f"  Variants: {[v['name'] for v in variants]}\n")

    for layer_idx in layers:
        for var in variants:
            done += 1
            var_name = var["name"]
            print(f"\n{'='*60}\n Run {done}/{total}: Layer {layer_idx}, Variant {var_name}\n{'='*60}")
            out_dir = output_path / f"ablation_layer_{layer_idx}_{var_name}"

            try:
                metrics = train_embedding_predictor(
                    data_dir=args.data_dir, output_dir=str(out_dir), layer_idx=layer_idx,
                    embedding_history_size=args.history, hidden_dim=args.hidden_dim, num_epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, device=args.device, base_embedding_dim=emb_dim,
                    num_experts=n_exp, active_k=active_k, prefetch_k=prefetch_k,
                    loss_type=args.loss_type, best_by=args.best_by,
                    model_type="dual_mlp",
                    use_embedding=var["use_embedding"],
                    use_prefill=var["use_prefill"],
                    use_prev_experts=var["use_prev_experts"],
                )
                results.append({
                    "layer": layer_idx, "variant": var_name,
                    "use_embedding": var["use_embedding"],
                    "use_prev_experts": var["use_prev_experts"],
                    "use_prefill": var["use_prefill"],
                    **{f"val_{key}": v for key, v in metrics.items()}
                })
                with open(output_path / "ablation_summary.json", "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                import traceback
                print(f"[ERROR] Run failed: {e}", file=sys.stderr)
                traceback.print_exc()

    print(f"\n=== Ablation Study Complete ===")
    for layer in layers:
        print(f"\nLayer {layer}:")
        layer_res = [r for r in results if r['layer'] == layer]
        layer_res = sorted(layer_res, key=lambda x: x[f'val_{args.best_by}'], reverse=True)
        if prefetch_k == 1:
            # top_k=1: accuracy is the single meaningful metric
            header = f"  {'Variant':<22} {'accuracy':>10}  {'macro_F1':>10}  {'loads/tok':>10}  {'reduction':>10}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for r in layer_res:
                acc = r.get(f'val_{args.best_by}', 0)
                mf1 = r.get('val_macro_f1', 0)
                exp_loads = 2.0 - acc
                reduc = (2.0 - exp_loads) / 2.0 * 100
                print(f"  {r['variant']:<22} {acc:>10.4f}  {mf1:>10.4f}  "
                      f"{exp_loads:>10.3f}  {reduc:>9.1f}%")
        else:
            header = f"  {'Variant':<22} {args.best_by:>12}  {'any_correct':>12}  {'mean_overlap':>12}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for r in layer_res:
                print(f"  {r['variant']:<22} {r[f'val_{args.best_by}']:>12.4f}  "
                      f"{r.get('val_any_correct', 0):>12.4f}  {r.get('val_mean_overlap', 0):>12.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expert Predictor Training and Sweeping")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Base arguments shared between both commands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data_dir', required=True)
    parent_parser.add_argument('--output_dir', required=True)
    parent_parser.add_argument('--model', choices=list(MODEL_DEFAULTS.keys()), default=None)
    parent_parser.add_argument('--num_layers', type=int, default=32)
    parent_parser.add_argument('--embedding_dim', type=int, default=4096)
    parent_parser.add_argument('--num_experts', type=int, default=8)
    parent_parser.add_argument('--active_k',   type=int, default=None,
                               help='Experts the router picks per token (defines the label). Overrides model default.')
    parent_parser.add_argument('--prefetch_k', type=int, default=None,
                               help='Experts to predict and prefetch. Overrides model default (1 for Mixtral, 4 for Qwen3).')
    parent_parser.add_argument('--device', type=str, default='cuda')
    parent_parser.add_argument('--epochs', type=int, default=10)
    parent_parser.add_argument('--batch_size', type=int, default=32)
    parent_parser.add_argument('--lr', type=float, default=1e-3)
    parent_parser.add_argument('--loss_type', choices=['ce', 'bce', 'focal'], default='ce')
    parent_parser.add_argument('--best_by', type=str, default="top1_exact")
    parent_parser.add_argument('--model_type', type=str, default="dual_mlp", choices=["dual_mlp"])

    # train command
    parser_train = subparsers.add_parser('train', parents=[parent_parser])
    parser_train.add_argument('--layer_idx', type=int, default=None)
    parser_train.add_argument('--history', type=int, default=1)
    parser_train.add_argument('--hidden_dim', type=int, default=None)

    # sweep command
    parser_sweep = subparsers.add_parser('sweep', parents=[parent_parser])
    parser_sweep.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 128, 256])
    parser_sweep.add_argument("--histories", nargs="+", type=int, default=[1, 2])
    parser_sweep.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
    parser_sweep.add_argument("--max_size_mb", type=float, default=70.0)

    # ablation command
    parser_ablation = subparsers.add_parser('ablation', parents=[parent_parser])
    parser_ablation.add_argument('--layers', nargs="+", type=int, default=[15], help="Layers specifically to run ablation on")
    parser_ablation.add_argument('--history', type=int, default=1)
    parser_ablation.add_argument('--hidden_dim', type=int, default=256)

    args = parser.parse_args()
    if args.command == 'train': run_train(args)
    elif args.command == 'sweep': run_sweep(args)
    elif args.command == 'ablation': run_ablation(args)
