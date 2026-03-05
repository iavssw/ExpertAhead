#!/usr/bin/env python3
"""
expert_predictor.py — Expert Predictor: Models, Training, and Sweeping
=======================================================================
A single unified script containing model definitions, data loaders, training
loops, and sweep/ablation orchestration.

Goal: predict which SINGLE expert (the top-1 router expert) will be selected
      at token t+1, given the post-attention-norm embedding at token t and the
      prefill expert-usage distribution.

Usage Examples
--------------
Train all layers for Mixtral 8x7B:
  python expert_predictor.py train \\
      --data_dir ../../trainingData/mixtral_8x7b \\
      --output_dir ../../trainingData/mixtral_8x7b/predictor_models \\
      --model mixtral_8x7b

Train a single layer:
  python expert_predictor.py train ... --layer_idx 15

Comprehensive sweep (hidden_dim × embedding_history × layers):
  python expert_predictor.py sweep \\
      --data_dir ../../trainingData/mixtral_8x7b \\
      --output_dir ../../trainingData/mixtral_8x7b/sweep_results \\
      --model mixtral_8x7b \\
      --hidden_dims 32 64 128 256 \\
      --histories 1 2 \\
      --max_size_mb 70

Ablation study (which feature branches matter):
  python expert_predictor.py ablation \\
      --data_dir ../../trainingData/mixtral_8x7b \\
      --output_dir ../../trainingData/mixtral_8x7b/ablation \\
      --model mixtral_8x7b \\
      --layers 0 15 31
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
# 1. Model Defaults
# ──────────────────────────────────────────────────────────────────────────────

# emb_dim, num_experts, active_k (router top-k), num_layers, default_prefetch_k
MODEL_DEFAULTS = {
    "mixtral_8x7b":  (4096, 8,   2, 32, 1),
    "mixtral_8x22b": (4096, 8,   2, 56, 1),
    "qwen3_30b":     (2048, 128, 8, 48, 4),
    "qwen3_480b":    (7168, 128, 8, 94, 4),
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model Definitions
# ──────────────────────────────────────────────────────────────────────────────

class DualMLPPredictor(nn.Module):
    """
    Dual-branch expert predictor for MoE models.

    Predicts the SINGLE top-1 expert that will be selected at token t+1.
    Training uses CrossEntropyLoss against the true top-1 expert index.

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
    The argmax gives the single predicted top-1 expert to prefetch.
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

        # Fused classifier → logits over all experts
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
            logits: [batch, num_experts]  — argmax gives predicted top-1 expert
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

    def predict_top1(
        self,
        embedding: torch.Tensor,
        prefill_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the single predicted top-1 expert index. Shape: [batch]."""
        logits = self.forward(embedding, prefill_dist)
        return logits.argmax(dim=-1)

    def get_config(self) -> Dict:
        return {
            'hidden_size':   self.hidden_size,
            'num_experts':   self.num_experts,
            'branch_dim':    self.branch_dim,
            'prefetch_k':    self.prefetch_k,
            'use_embedding': self.use_embedding,
            'use_prefill':   self.use_prefill,
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

    Predicts the SINGLE top-1 expert at token t+1.

    Branch 1 — EmbeddingBranch:   post-attn-norm embedding at token t   [batch, hidden_size]
    Branch 2 — PrevExpertBranch:  one-hot of expert selected at token t  [batch, num_experts]
                                  (scatter of the top-1 ID chosen at step t)
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
        self.prefetch_k       = prefetch_k
        self.use_embedding    = use_embedding
        self.use_prev_experts = use_prev_experts
        self.use_prefill      = use_prefill

        prefill_dim  = max(branch_dim // 4, num_experts)
        prev_exp_dim = max(branch_dim // 4, num_experts)
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
            # One-hot indicating which expert was top-1 at token t
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

        # Classifier → logits over all experts; argmax = predicted top-1 expert
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
        prev_expert_onehot: [batch, num_experts]  — one-hot of top-1 expert at step t
        prefill_dist:       [batch, num_experts]
        Returns:
            logits: [batch, num_experts]
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

    def predict_top1(self, embedding=None, prev_expert_onehot=None, prefill_dist=None) -> torch.Tensor:
        """Returns the single predicted top-1 expert index. Shape: [batch]."""
        return self.forward(embedding, prev_expert_onehot, prefill_dist).argmax(dim=-1)

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


class PredictorJITWrapper(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.is_history_aware = isinstance(predictor, HistoryAwareDualMLPPredictor)

    def forward(
        self,
        embedding:          torch.Tensor,
        prefill_dist:       torch.Tensor,
        prev_expert_onehot: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_history_aware:
            return self.predictor(
                embedding=embedding,
                prev_expert_onehot=prev_expert_onehot,
                prefill_dist=prefill_dist,
            )
        else:
            return self.predictor(embedding=embedding, prefill_dist=prefill_dist)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ExpertPredictorDataset(Dataset):
    """
    Dataset for top-1 expert prediction.

    Each sample is:
      - post_attn_embedding:  [hidden_size * embedding_history_size]  (float32)
      - top1_expert:          scalar int64  — the TRUE top-1 router expert at t+1
                              This is the training target for CrossEntropyLoss.
      - prefill_expert_dist:  [num_experts]  (float32)
      - prev_expert_onehot:   [num_experts]  (float32)  — one-hot of top-1 at t

    Handles two .pt schemas:
      NEW  (collect_training_data_unified.py):
           payload['layers'] = [{'layer_idx', 'embeddings' [seq,H],
                                 'router_logits' [seq,E], optionally
                                 'prev_expert_ids' [seq, active_k],
                                 'prefill_expert_dist' [E] or
                                 'prefill_expert_count' [E]}, ...]
      LEGACY: payload['data'] = {layer_idx: {token_pos: {'post_attn_post_norm_embedding', ...}}}
    """

    def __init__(
        self,
        data_source: Union[str, Path, List[Path]],
        layer_idx: Optional[int] = None,
        num_experts: int = 8,
        embedding_history_size: int = 1,
    ):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
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

            # ── New unified schema ─────────────────────────────────────────────
            if 'layers' in payload:
                for layer_dict in payload['layers']:
                    l_idx = layer_dict['layer_idx']
                    if self.layer_idx is not None and l_idx != self.layer_idx:
                        continue

                    embeddings      = layer_dict['embeddings']      # [seq_len, hidden_size]
                    router_logits   = layer_dict['router_logits']   # [seq_len, num_experts]
                    prev_expert_ids = layer_dict.get('prev_expert_ids')  # [seq_len, active_k] or None
                    seq_len, hidden_size = embeddings.shape

                    # Prefill distribution (prompt-level feature)
                    prefill_expert_dist = layer_dict.get('prefill_expert_dist')
                    if prefill_expert_dist is None and 'prefill_expert_count' in layer_dict:
                        c = layer_dict['prefill_expert_count'].float()
                        prefill_expert_dist = c / c.sum().clamp(min=1)
                    if prefill_expert_dist is None:
                        # Fallback: use softmax mean of all available router logits
                        prefill_expert_dist = torch.softmax(router_logits.float(), dim=-1).mean(dim=0)

                    for t in range(seq_len - 1):
                        # Build embedding history window
                        emb_parts = []
                        for h in range(self.embedding_history_size):
                            src = t - (self.embedding_history_size - 1 - h)
                            emb_parts.append(embeddings[src] if src >= 0 else torch.zeros(hidden_size))

                        # Top-1 expert at t+1 (the training target)
                        top1_next = int(torch.argmax(router_logits[t + 1]).item())

                        # One-hot of the top-1 expert USED at step t (feature for predicting t+1)
                        if prev_expert_ids is not None:
                            # prev_expert_ids[t, 0] is the top-1 expert at step t
                            top1_t = int(prev_expert_ids[t, 0].item())
                        else:
                            top1_t = int(torch.argmax(router_logits[t]).item())
                        prev_onehot = torch.zeros(self.num_experts)
                        prev_onehot[top1_t] = 1.0

                        self.samples.append({
                            'embedding_features': torch.cat(emb_parts, dim=0),
                            'top1_expert':        top1_next,
                            'prefill_expert_dist': prefill_expert_dist,
                            'prev_expert_onehot':  prev_onehot,
                        })
                return

            # ── Legacy schema ──────────────────────────────────────────────────
            if 'data' not in payload:
                return
            data_by_layer = payload['data']
            target_layers = [self.layer_idx] if self.layer_idx is not None else list(data_by_layer.keys())
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
                    next_logits = torch.tensor(layer_data[t_next]['current_router_logits'])
                    top1_next = int(torch.argmax(next_logits).item())
                    self.samples.append({
                        'embedding_features': emb_list,
                        'top1_expert':        top1_next,
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
                emb_list.extend(rec['post_attn_post_norm_embedding'] if rec else [0.0] * 4096)
            nxt = records[t_next]
            if 'selected_experts' in nxt:
                top1_next = nxt['selected_experts'][0]
            else:
                logits = torch.tensor(nxt['current_router_logits'])
                top1_next = int(torch.argmax(logits).item())
            self.samples.append({'embedding_features': emb_list, 'top1_expert': top1_next})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        emb = s['embedding_features']
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float32)

        prev_onehot = s.get('prev_expert_onehot')
        if prev_onehot is None:
            prev_onehot = torch.zeros(self.num_experts)

        return {
            'post_attn_embedding': emb.float(),
            'top1_expert':         torch.tensor(s['top1_expert'], dtype=torch.long),
            'prefill_expert_dist': s.get('prefill_expert_dist',
                                         torch.full((self.num_experts,), 1.0 / self.num_experts)),
            'prev_expert_onehot':  prev_onehot.float(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_logits: torch.Tensor, true_top1: torch.Tensor) -> dict:
    """
    Compute accuracy metrics for top-1 expert prediction.

    pred_logits: [B, num_experts]  raw logits from model
    true_top1:   [B]               ground-truth top-1 expert index (int64)

    Returns a dict of *sums* (caller divides by N):
      top1_exact  — our argmax matches the true top-1 expert exactly
      any_top2    — our argmax is within the true top-2 experts (softer metric)
    """
    B = pred_logits.size(0)
    num_experts = pred_logits.size(1)

    pred_top1 = pred_logits.argmax(dim=-1)   # [B]  our prediction

    exact = (pred_top1 == true_top1).sum().item()

    # Soft bonus: are we within top-2?   (true_top1 is rank-1; rank-2 is next best)
    pred_top2_mask = torch.zeros(B, num_experts, dtype=torch.bool, device=pred_logits.device)
    pred_top2_mask.scatter_(1, pred_logits.topk(2, dim=-1).indices, True)
    true_top1_in_pred2 = pred_top2_mask[torch.arange(B), true_top1].sum().item()

    return {'top1_exact': exact, 'any_top2': true_top1_in_pred2}, B


def _merge_metrics(acc: dict, new_sums: dict, new_n: int) -> dict:
    acc['n']          = acc.get('n', 0)   + new_n
    acc['total_loss'] = acc.get('total_loss', 0.0) + new_sums.get('total_loss', 0.0)
    acc['top1_exact'] = acc.get('top1_exact', 0)   + new_sums['top1_exact']
    acc['any_top2']   = acc.get('any_top2',   0)   + new_sums['any_top2']
    return acc


def _finalise_metrics(acc: dict, num_batches: int) -> dict:
    n = acc['n']
    return {
        'loss':       acc['total_loss'] / max(num_batches, 1),
        'top1_exact': acc['top1_exact'] / n,   # PRIMARY metric: exact top-1 hit rate
        'any_top2':   acc['any_top2']   / n,   # softer: prediction is within true top-2
        'acc':        acc['top1_exact'] / n,   # alias for backward compat
    }


def _print_metrics(phase: str, m: dict):
    random_baseline = 1.0 / 8   # placeholder; will be 1/num_experts at runtime
    print(f"  {phase}:")
    print(f"    loss={m['loss']:.4f}")
    print(f"    top1_exact={m['top1_exact']:.4f}  any_top2={m['any_top2']:.4f}")
    exp_loads = 2.0 - m['top1_exact']   # expected loads/token (Mixtral has active_k=2)
    reduction = m['top1_exact'] * 100
    print(f"    expected_loads/tok≈{exp_loads:.3f}  ({reduction:.1f}% load reduction vs no predictor)")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Training Engine
# ──────────────────────────────────────────────────────────────────────────────

def _model_forward(model, batch_on_device: dict) -> torch.Tensor:
    """Unified forward dispatch for all predictor model types."""
    emb   = batch_on_device['post_attn_embedding']
    pdist = batch_on_device['prefill_expert_dist']
    prev  = batch_on_device['prev_expert_onehot']
    if isinstance(model, HistoryAwareDualMLPPredictor):
        return model(embedding=emb, prev_expert_onehot=prev, prefill_dist=pdist)
    else:
        return model(embedding=emb, prefill_dist=pdist)


def train_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    acc = {}
    num_batches = 0
    for batch in tqdm(loader, desc="Training", leave=True):
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        target = batch_dev['top1_expert']   # [B]  true top-1 expert index

        optimizer.zero_grad()
        logits = _model_forward(model, batch_dev)   # [B, num_experts]
        loss = criterion(logits, target)             # CrossEntropyLoss
        loss.backward()
        optimizer.step()

        sums, n = compute_metrics(logits.detach(), target)
        sums['total_loss'] = loss.item()
        _merge_metrics(acc, sums, n)
        num_batches += 1

    return _finalise_metrics(acc, num_batches)


def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    acc = {}
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=True):
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            target = batch_dev['top1_expert']

            logits = _model_forward(model, batch_dev)
            loss = criterion(logits, target)

            sums, n = compute_metrics(logits, target)
            sums['total_loss'] = loss.item()
            _merge_metrics(acc, sums, n)
            num_batches += 1

    return _finalise_metrics(acc, num_batches)


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
    active_k: int = 2,       # how many experts the router selects (for reference / future use)
    prefetch_k: int = 1,     # currently always 1; we predict a single top-1 expert
    hidden_dim: Optional[int] = None,
    loss_type: str = 'ce',   # only 'ce' is meaningful for single-class prediction
    best_by: str = 'top1_exact',
    model_type: str = 'dual_mlp',
    use_embedding: bool = True,
    use_prefill: bool = True,
    use_prev_experts: bool = False,
) -> dict:
    """Train one per-layer expert predictor MLP to predict the top-1 router expert."""
    _base_out = Path(output_dir) / f"layer_{layer_idx}"

    all_files = sorted(list(Path(data_dir).glob("*.pt")) + list(Path(data_dir).glob("*.jsonl")))
    random.seed(42)
    random.shuffle(all_files)
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    val_files   = all_files[split:] or [all_files[-1]]
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    ds_kwargs = dict(
        layer_idx=layer_idx,
        num_experts=num_experts,
        embedding_history_size=embedding_history_size,
    )
    train_ds = ExpertPredictorDataset(train_files, **ds_kwargs)
    val_ds   = ExpertPredictorDataset(val_files,   **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    input_dim   = base_embedding_dim * embedding_history_size
    _hidden_dim = hidden_dim if hidden_dim else min(2048, max(256, input_dim // 4))
    output_dir  = _base_out / f"hidden_{_hidden_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Layer {layer_idx} — input_dim={input_dim}, hidden_dim={_hidden_dim}, "
        f"num_experts={num_experts}, active_k={active_k}, prefetch_k={prefetch_k}, "
        f"model_type={model_type}, use_embedding={use_embedding}, "
        f"use_prefill={use_prefill}, use_prev_experts={use_prev_experts}"
    )

    # Build model
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

    # CrossEntropyLoss is the only correct loss for single-class top-1 prediction
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc     = 0.0
    best_metrics = {}
    epoch_log    = []
    metrics_path = output_dir / "training_metrics.json"

    run_config = {
        "layer_idx": layer_idx, "hidden_dim": _hidden_dim,
        "embedding_history_size": embedding_history_size,
        "base_embedding_dim": base_embedding_dim,
        "num_experts": num_experts, "active_k": active_k, "prefetch_k": prefetch_k,
        "use_embedding": use_embedding,
        "use_prefill": use_prefill,
        "use_prev_experts": use_prev_experts,
        "target": "top1_expert_exact",
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model,   val_loader,   criterion, device)

        scheduler.step(val_m.get(best_by, val_m['top1_exact']))

        _print_metrics("Train", train_m)
        _print_metrics("Val",   val_m)

        epoch_log.append({
            "epoch": epoch + 1, "train": train_m, "val": val_m,
            "train_loss": train_m['loss'], "train_acc": train_m['top1_exact'],
            "val_loss":   val_m['loss'],   "val_acc":   val_m['top1_exact'],
        })

        current_score = val_m.get(best_by, val_m['top1_exact'])
        if current_score > best_acc:
            best_acc     = current_score
            best_metrics = val_m
            save_path    = output_dir / "embedding_predictor_best.pt"
            model.save(save_path)

            # Also save as TorchScript JIT for inference
            try:
                model.eval()
                wrapper = PredictorJITWrapper(model)
                wrapper.eval()
                # Always provide 3 tensors for the trace to ensure fixed C++ signature.
                ex_emb   = torch.randn(1, input_dim).to(device)
                ex_pfill = torch.zeros(1, num_experts).to(device)
                ex_prev  = torch.zeros(1, num_experts).to(device)
                with torch.no_grad():
                    traced = torch.jit.trace(wrapper, (ex_emb, ex_pfill, ex_prev))
                traced.save(str(save_path))
                model.train()
            except Exception as e:
                print(f"  [warn] JIT trace failed: {e}")

            print(f"  ↑ New best {best_by}={best_acc:.4f} — saved to {save_path}")

        with open(metrics_path, 'w') as f:
            json.dump({**run_config, "best_val_acc": best_acc, "best_by": best_by,
                       "epochs": epoch_log}, f, indent=2)

    return best_metrics


# ──────────────────────────────────────────────────────────────────────────────
# 6. Sweep Helpers
# ──────────────────────────────────────────────────────────────────────────────

def calculate_model_size_mb(input_dim: int, hidden_dim: int, output_dim: int) -> float:
    """Estimate the size of a DualMLPPredictor in megabytes."""
    # Branch 1: Linear(input_dim, hidden_dim) + LayerNorm
    params  = (input_dim * hidden_dim) + hidden_dim   # weights + bias
    params += hidden_dim * 2                          # LayerNorm weight + bias
    # Classifier: Linear(hidden_dim, hidden_dim//2) + Linear(hidden_dim//2, output_dim)
    params += (hidden_dim * (hidden_dim // 2)) + (hidden_dim // 2)
    params += ((hidden_dim // 2) * output_dim)   + output_dim
    return (params * 4) / (1024 * 1024)   # float32 bytes → MB


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLI Commands
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_model_args(args):
    """Return (emb_dim, n_experts, active_k, n_layers, prefetch_k) from CLI args."""
    if args.model:
        emb_dim, n_exp, default_active_k, n_layers, default_prefetch_k = MODEL_DEFAULTS[args.model]
    else:
        emb_dim, n_exp, default_active_k, n_layers, default_prefetch_k = (
            args.embedding_dim, args.num_experts, 2, args.num_layers, 1
        )
    active_k   = args.active_k   if getattr(args, 'active_k',   None) is not None else default_active_k
    prefetch_k = args.prefetch_k if getattr(args, 'prefetch_k', None) is not None else default_prefetch_k
    return emb_dim, n_exp, active_k, n_layers, prefetch_k


def run_train(args):
    emb_dim, n_exp, active_k, n_layers, prefetch_k = _resolve_model_args(args)
    shared = dict(
        embedding_history_size=args.history, num_epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
        base_embedding_dim=emb_dim, num_experts=n_exp,
        active_k=active_k, prefetch_k=prefetch_k,
        hidden_dim=args.hidden_dim, loss_type=args.loss_type,
        model_type=args.model_type, best_by=args.best_by,
    )
    layers = [args.layer_idx] if args.layer_idx is not None else list(range(n_layers))
    layer_results = {}
    for i in layers:
        print(f"\n{'='*60}\nTraining Layer {i}/{n_layers-1}\n{'='*60}")
        layer_results[i] = train_embedding_predictor(args.data_dir, args.output_dir, i, **shared)
    print("\n" + "="*60 + "\nLayer-wise best val top1_exact:\n" + "-"*30)
    for i, m in sorted(layer_results.items()):
        print(f"  Layer {i:3d}: top1_exact={m.get('top1_exact', 0):.4f}")
    print("="*60)


def run_sweep(args):
    emb_dim, n_exp, active_k, n_layers, prefetch_k = _resolve_model_args(args)
    layers = args.layers if args.layers is not None else list(range(n_layers))
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    OVERHEAD = 1.25   # 25% buffer for metadata in .pt file
    valid_configs = []
    for history in args.histories:
        for h_dim in args.hidden_dims:
            est = calculate_model_size_mb(emb_dim * history, h_dim, n_exp) * OVERHEAD
            if est <= args.max_size_mb:
                valid_configs.append((history, h_dim, est))
                print(f"Valid:    emb_hist={history}, hidden={h_dim}  (est {est:.1f} MB/layer)")
            else:
                print(f"Skipping: emb_hist={history}, hidden={h_dim}  ({est:.1f} MB > {args.max_size_mb} MB)")

    total, done = len(layers) * len(valid_configs), 0
    print(f"\nSweep Summary:\n  Layers: {layers}\n  Valid archs: {len(valid_configs)}\n  Total runs: {total}")

    results = []
    for history, hidden_dim, size_mb in valid_configs:
        for layer_idx in layers:
            done += 1
            print(f"\n{'='*60}\n Run {done}/{total}: Layer {layer_idx}, "
                  f"EmbHist {history}, Hidden {hidden_dim} ({size_mb:.1f} MB)\n{'='*60}")
            out_dir = output_path / f"eh{history}_h{hidden_dim}"
            try:
                metrics = train_embedding_predictor(
                    data_dir=args.data_dir, output_dir=str(out_dir), layer_idx=layer_idx,
                    embedding_history_size=history, hidden_dim=hidden_dim,
                    num_epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, device=args.device, base_embedding_dim=emb_dim,
                    num_experts=n_exp, active_k=active_k, prefetch_k=prefetch_k,
                    loss_type=args.loss_type, best_by=args.best_by,
                    model_type=args.model_type,
                )
                results.append({
                    "layer": layer_idx, "history": history, "hidden_dim": hidden_dim,
                    "size_mb": size_mb,
                    **{f"val_{k}": v for k, v in metrics.items()},
                })
                with open(output_path / "sweep_summary.json", "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"[ERROR] Run failed: {e}", file=sys.stderr)

    print(f"\nSweep complete. Best architectures per layer (by {args.best_by}):")
    for layer in layers:
        layer_res = [r for r in results if r['layer'] == layer]
        if not layer_res:
            continue
        best = max(layer_res, key=lambda x: x.get(f'val_{args.best_by}', 0))
        print(f"  Layer {layer:2d}: EmbHist={best['history']}, Hidden={best['hidden_dim']}  "
              f"top1_exact={best.get('val_top1_exact', 0):.4f}  "
              f"any_top2={best.get('val_any_top2', 0):.4f}")


def run_ablation(args):
    emb_dim, n_exp, active_k, n_layers, prefetch_k = _resolve_model_args(args)
    layers = args.layers if args.layers is not None else [15]
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    variants = [
        # ── Single-branch baselines ────────────────────────────────────────────
        {"name": "emb_only",         "use_embedding": True,  "use_prev_experts": False, "use_prefill": False},
        {"name": "prefill_only",     "use_embedding": False, "use_prev_experts": False, "use_prefill": True},
        {"name": "prev_only",        "use_embedding": False, "use_prev_experts": True,  "use_prefill": False},
        # ── Two-branch combos ─────────────────────────────────────────────────
        {"name": "emb_prefill",      "use_embedding": True,  "use_prev_experts": False, "use_prefill": True},
        {"name": "emb_prev",         "use_embedding": True,  "use_prev_experts": True,  "use_prefill": False},
        {"name": "prev_prefill",     "use_embedding": False, "use_prev_experts": True,  "use_prefill": True},
        # ── Full three-branch ─────────────────────────────────────────────────
        {"name": "emb_prev_prefill", "use_embedding": True,  "use_prev_experts": True,  "use_prefill": True},
    ]

    total, done = len(layers) * len(variants), 0
    results = []
    print(f"\n=== Ablation Study ===\n  Layers:   {layers}")
    print(f"  Variants: {[v['name'] for v in variants]}\n")

    for layer_idx in layers:
        for var in variants:
            done += 1
            print(f"\n{'='*60}\n Run {done}/{total}: Layer {layer_idx}, Variant {var['name']}\n{'='*60}")
            out_dir = output_path / f"ablation_layer_{layer_idx}_{var['name']}"
            try:
                metrics = train_embedding_predictor(
                    data_dir=args.data_dir, output_dir=str(out_dir), layer_idx=layer_idx,
                    embedding_history_size=args.history, hidden_dim=args.hidden_dim,
                    num_epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, device=args.device, base_embedding_dim=emb_dim,
                    num_experts=n_exp, active_k=active_k, prefetch_k=prefetch_k,
                    loss_type=args.loss_type, best_by=args.best_by,
                    model_type="dual_mlp",
                    use_embedding=var["use_embedding"],
                    use_prefill=var["use_prefill"],
                    use_prev_experts=var["use_prev_experts"],
                )
                results.append({
                    "layer": layer_idx, "variant": var["name"],
                    **var,
                    **{f"val_{k}": v for k, v in metrics.items()},
                })
                with open(output_path / "ablation_summary.json", "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                import traceback
                print(f"[ERROR] Run failed: {e}", file=sys.stderr)
                traceback.print_exc()

    print("\n=== Ablation Complete ===")
    for layer in layers:
        print(f"\nLayer {layer}:")
        layer_res = sorted(
            [r for r in results if r['layer'] == layer],
            key=lambda x: x.get(f'val_{args.best_by}', 0), reverse=True
        )
        header = f"  {'Variant':<22} {'top1_exact':>12}  {'any_top2':>10}  {'loads/tok':>10}  {'reduction':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in layer_res:
            acc = r.get(f'val_{args.best_by}', 0)
            at2 = r.get('val_any_top2', 0)
            exp_loads = 2.0 - acc
            reduc     = acc * 100
            print(f"  {r['variant']:<22} {acc:>12.4f}  {at2:>10.4f}  "
                  f"{exp_loads:>10.3f}  {reduc:>9.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expert Predictor: predict the top-1 router expert at token t+1."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Shared parent parser ───────────────────────────────────────────────────
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('--data_dir',      required=True,  help="Directory with .pt training files")
    parent.add_argument('--output_dir',    required=True,  help="Where to save models/metrics")
    parent.add_argument('--model',         choices=list(MODEL_DEFAULTS.keys()), default=None,
                        help="Use preset emb_dim/num_experts/active_k/num_layers for a known model")
    parent.add_argument('--num_layers',    type=int, default=32)
    parent.add_argument('--embedding_dim', type=int, default=4096)
    parent.add_argument('--num_experts',   type=int, default=8)
    parent.add_argument('--active_k',      type=int, default=None,
                        help="Experts the router picks per token (overrides model default)")
    parent.add_argument('--prefetch_k',    type=int, default=None,
                        help="Experts to prefetch — currently must be 1 (overrides model default)")
    parent.add_argument('--epochs',        type=int,   default=10)
    parent.add_argument('--batch_size',    type=int,   default=128)
    parent.add_argument('--lr',            type=float, default=1e-3)
    parent.add_argument('--device',        type=str,   default='cuda')
    parent.add_argument('--loss_type',     choices=['ce'], default='ce',
                        help="Loss function (only 'ce' is valid for top-1 prediction)")
    parent.add_argument('--best_by',       type=str,   default='top1_exact')
    parent.add_argument('--model_type',    type=str,   default='dual_mlp', choices=['dual_mlp'])

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser('train', parents=[parent],
                                    help="Train a predictor for one or all layers")
    p_train.add_argument('--layer_idx',  type=int, default=None,
                         help="Single layer to train (default: all layers)")
    p_train.add_argument('--history',    type=int, default=1,
                         help="Embedding history window size")
    p_train.add_argument('--hidden_dim', type=int, default=None,
                         help="Branch projection dimension (default: auto)")

    # ── sweep ─────────────────────────────────────────────────────────────────
    p_sweep = subparsers.add_parser('sweep', parents=[parent],
                                    help="Grid search over hidden_dim × history × layers")
    p_sweep.add_argument('--hidden_dims', nargs='+', type=int, default=[64, 128, 256],
                         help="Hidden (branch) dims to sweep")
    p_sweep.add_argument('--histories',   nargs='+', type=int, default=[1, 2],
                         help="Embedding history sizes to sweep")
    p_sweep.add_argument('--layers',      nargs='+', type=int,
                         default=list(range(32)),
                         help="Layer indices to sweep")
    p_sweep.add_argument('--max_size_mb', type=float, default=70.0,
                         help="Max estimated model size per layer in MB")

    # ── ablation ──────────────────────────────────────────────────────────────
    p_abl = subparsers.add_parser('ablation', parents=[parent],
                                  help="Ablation study over feature branches")
    p_abl.add_argument('--layers',      nargs='+', type=int, default=[15],
                       help="Layers to run ablation on")
    p_abl.add_argument('--history',     type=int, default=1)
    p_abl.add_argument('--hidden_dim',  type=int, default=256)

    args = parser.parse_args()
    if   args.command == 'train':    run_train(args)
    elif args.command == 'sweep':    run_sweep(args)
    elif args.command == 'ablation': run_ablation(args)
