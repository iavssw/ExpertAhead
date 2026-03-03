"""
Modular MLP Expert Predictor Using Post-attention, post-RMS norm embeddings


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import json
from pathlib import Path



class EmbeddingMLP(nn.Module):
    """MLP for processing post-attention, post-RMS norm embeddings"""
    
    def __init__(
        self,
        embedding_dim: int = 4096,  # Mixtral hidden size
        embedding_history_size: int = 1,
        hidden_dim: int = 1024,
        output_dim: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        
        input_dim = embedding_dim * embedding_history_size
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: [batch_size, embedding_dim] post-attention embedding
            
        Returns:
            [batch_size, output_dim] feature vector
        """
        return self.mlp(embedding)


class RouterHistoryMLP(nn.Module):
    """MLP for processing router logit history"""
    def __init__(self, num_experts: int, history_size: int, hidden_dim: int):
        super().__init__()
        # Flattened input: [batch, history_size * num_experts]
        self.mlp = nn.Sequential(
            nn.Linear(num_experts * history_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        
    def forward(self, x):
        return self.mlp(x.view(x.size(0), -1))

class ExpertPredictor(nn.Module):
    """
    Expert prediction model with Embedding + Router History fusion.
    """
    def __init__(
        self,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        vocab_size: int = 32000,
        base_embedding_dim: int = 4096,
        embedding_history_size: int = 1,
        router_history_size: int = 0, # New: history of past router decisions
        num_experts: int = 8,
        top_k: int = 2,
        context_window_k: int = 10,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_embedding: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.base_embedding_dim = base_embedding_dim
        self.embedding_history_size = embedding_history_size
        self.embedding_dim = base_embedding_dim * embedding_history_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.context_window_k = context_window_k
        self.hidden_dim = hidden_dim
        self.use_embedding = use_embedding
        self.router_history_size = router_history_size
        
        # 1. Feature extractors
        feat_dim = 0
        if use_embedding:
            self.embedding_mlp = nn.Sequential(
                nn.Linear(base_embedding_dim * embedding_history_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            feat_dim += hidden_dim
            
        if router_history_size > 0:
            self.router_mlp = nn.Sequential(
                nn.Linear(num_experts * router_history_size, hidden_dim // 4),
                nn.GELU()
            )
            feat_dim += (hidden_dim // 4)
            
        # 2. Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts)
        )
    
    def forward(
        self,
        post_attn_embedding: Optional[torch.Tensor] = None,
        router_logits_history: Optional[torch.Tensor] = None,
        expert_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = []
        
        if self.use_embedding:
            features.append(self.embedding_mlp(post_attn_embedding))
            
        if self.router_history_size > 0 and router_logits_history is not None:
            features.append(self.router_mlp(router_logits_history.view(router_logits_history.size(0), -1)))
            
        combined = torch.cat(features, dim=-1)
        logits = self.classifier(combined)
        
        if expert_bias is not None:
            logits = logits + expert_bias
            
        return logits
    
    def predict_top_k(
        self,
        post_attn_embedding: Optional[torch.Tensor] = None,
        router_logits_history: Optional[torch.Tensor] = None,
        expert_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k experts.
        
        Returns:
            top_k_indices: [batch_size, top_k] indices of top experts
            top_k_probs: [batch_size, top_k] probabilities for top experts
        """
        logits = self.forward(post_attn_embedding, router_logits_history, expert_bias)
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        return top_k_indices, top_k_probs
    
    def get_config(self) -> Dict:
        """Get model configuration for saving/loading"""
        return {
            'vocab_size': self.vocab_size,
            'base_embedding_dim': self.base_embedding_dim,
            'embedding_history_size': self.embedding_history_size,
            'embedding_dim': self.embedding_dim,
            'num_experts': self.num_experts,
            'router_history_size': self.router_history_size,
            'top_k': self.top_k,
            'context_window_k': self.context_window_k,
            'hidden_dim': self.hidden_dim,
            'use_embedding': self.use_embedding,
        }
    
    def save(self, path: str):
        """Save model and configuration"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # We save the state dict to .pth to avoid overwriting the TorchScript model
        state_path = path.with_suffix('.pth') if path.suffix == '.pt' else path
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
        }, state_path)
        
        # Also save config as JSON for easy inspection
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f, indent=2)
            
    @classmethod
    def load(cls, path: str, device: str = 'cpu', pretrained_embeddings: Optional[torch.Tensor] = None):
        """
        Load model from checkpoint.
        
        Args:
             path: Path to checkpoint file
             device: Device to load model on
             pretrained_embeddings: Pretrained embeddings tensor (required if model uses context tokens)
        """
        path_obj = Path(path)
        # If trying to load .pt but it might be a traced TorchScript model, fallback to .pth state dict
        if path_obj.suffix == '.pt' and path_obj.with_suffix('.pth').exists():
            try:
                checkpoint = torch.load(path_obj.with_suffix('.pth'), map_location=device, weights_only=False)
            except Exception:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Check if we need embeddings and warn if missing
        config = checkpoint['config']
        if config.get('use_context_tokens', False) and pretrained_embeddings is None:
             print("WARNING: Loading model that uses context tokens but pretrained_embeddings not provided.")
             print("Please provide pretrained_embeddings to load() to avoid initialization errors or random embeddings.")
             
        model = cls(pretrained_embeddings=pretrained_embeddings, **config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) # strict=False to handle missing embedding buffer if it was stripped
        return model


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
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.branch_dim  = branch_dim
        self.top_k       = top_k

        prefill_dim = max(branch_dim // 4, num_experts)

        # Branch 1: token embedding → branch_dim features
        self.embedding_branch = nn.Sequential(
            nn.Linear(hidden_size, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Branch 2: prefill expert dist → small feature vector
        self.prefill_branch = nn.Sequential(
            nn.Linear(num_experts, prefill_dim),
            nn.GELU(),
        )

        # Fused classifier
        fused_dim = branch_dim + prefill_dim
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
                          full prefill (mean softmax of router_logits).
                          If None, a uniform distribution is used as a neutral prior.
        Returns:
            logits: [batch, num_experts]
        """
        emb_feat = self.embedding_branch(embedding)

        if prefill_dist is None:
            prefill_dist = torch.full(
                (embedding.size(0), self.num_experts),
                1.0 / self.num_experts,
                dtype=embedding.dtype,
                device=embedding.device,
            )
        pre_feat = self.prefill_branch(prefill_dist)

        fused  = torch.cat([emb_feat, pre_feat], dim=-1)
        return self.classifier(fused)

    def predict_top_k(
        self,
        embedding: torch.Tensor,
        prefill_dist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            top_k_indices: [batch, top_k]
            top_k_probs:   [batch, top_k]
        """
        logits = self.forward(embedding, prefill_dist)
        probs  = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        return top_k_indices, top_k_probs

    def get_config(self) -> Dict:
        return {
            'hidden_size':  self.hidden_size,
            'num_experts':  self.num_experts,
            'branch_dim':   self.branch_dim,
            'top_k':        self.top_k,
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


def load_pretrained_embeddings(
    embeddings_path: str,
    device: str = 'cpu',
    target_vocab_size: int = 32000
) -> torch.Tensor:
    """
    Load pretrained Mixtral embeddings from disk and pad to full vocabulary if needed.
    
    Args:
        embeddings_path: Path to saved embeddings file
        device: Device to load embeddings on (keep 'cpu' to avoid ROCm issues)
        target_vocab_size: Target vocabulary size (32000 for Mixtral)
        
    Returns:
        Pretrained embeddings tensor [vocab_size, embed_dim]
    """
    embeddings = torch.load(embeddings_path, map_location=device)
    
    if isinstance(embeddings, dict):
        # Handle case where embeddings are saved in a dict
        if 'embeddings' in embeddings:
            embeddings = embeddings['embeddings']
        elif 'weight' in embeddings:
            embeddings = embeddings['weight']
    
    
    # Check if this is bfloat16 data misinterpreted as float32
    # If we have 16000 entries but expect 32000, and bytes match bf16 storage
    vocab_size, embed_dim = embeddings.shape
    expected_bf16_bytes = target_vocab_size * embed_dim * 2  # 2 bytes per bf16
    actual_bytes = embeddings.element_size() * embeddings.numel()
    
    if vocab_size < target_vocab_size and actual_bytes == expected_bf16_bytes:
        print(f"Detected bfloat16 data stored as float32. Reinterpreting...")
        # Reinterpret the bytes as bfloat16
        import numpy as np
        raw_bytes = embeddings.cpu().numpy().tobytes()
        # Reinterpret as uint16 first, then convert to bfloat16
        uint16_data = np.frombuffer(raw_bytes, dtype=np.uint16)
        # Reshape to [32000, 4096]
        uint16_data = uint16_data.reshape(target_vocab_size, embed_dim)
        # Convert to bfloat16 tensor
        embeddings = torch.from_numpy(uint16_data.view(np.uint16)).view(torch.bfloat16).to(device)
        # Convert to float32 for computation
        embeddings = embeddings.float()
        print(f"Reinterpreted to shape: {embeddings.shape}")
    elif vocab_size < target_vocab_size:
        # Pad if still incomplete
        print(f"WARNING: Embeddings have {vocab_size} entries, padding to {target_vocab_size}")
        missing_count = target_vocab_size - vocab_size
        padding = torch.randn(missing_count, embed_dim, device=device, dtype=torch.float32) * embeddings.std()
        embeddings = torch.cat([embeddings, padding], dim=0)
        print(f"Padded embeddings to shape: {embeddings.shape}")
    
    # Ensure float32 for computation
    if embeddings.dtype != torch.float32:
        embeddings = embeddings.float()
    
    print(f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}, device={embeddings.device}")
    return embeddings


def create_ablation_models(
    pretrained_embeddings: torch.Tensor,
    vocab_size: int = 32000,
    embedding_dim: int = 4096,
    num_experts: int = 8,
    top_k: int = 2,
    context_window_k: int = 10,
) -> Dict[str, ExpertPredictor]:
    """
    Create all models needed for ablation study.
    
    Returns a dictionary with the following models:
    - 'all': Uses all features
    - 'context_only': Uses only context tokens
    - 'embedding_only': Uses only embeddings
    - 'router_history_only': Uses only router logits history
    - 'context_embedding': Uses context tokens + embeddings
    - 'context_router': Uses context tokens + router history
    - 'embedding_router': Uses embeddings + router history
    """
    models = {
        'embedding_only': ExpertPredictor(
            pretrained_embeddings=pretrained_embeddings,
            vocab_size=vocab_size,
            base_embedding_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            context_window_k=context_window_k,
        )
    }
    
    return models


if __name__ == "__main__":
    # Example usage
    print("Loading pretrained Mixtral embeddings...")
    embeddings_path = "/home/michaelg/mixtral_project/mixtral_embeddings.pt"
    pretrained_embeddings = load_pretrained_embeddings(embeddings_path)
    
    print("\nCreating ablation study models...")
    models = create_ablation_models(pretrained_embeddings)
    
    print(f"\nCreated {len(models)} models:")
    for name, model in models.items():
        config = model.get_config()
        features = []

        if config['use_embedding']:
            features.append('embedding')
        
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:20s}: {', '.join(features):40s} ({total_params:,} params)")
    
    # Test forward pass
    print("\nTesting forward pass with 'all' model...")
    model = models['all']
    batch_size = 4
    
    # Create dummy inputs
    context_tokens = torch.randint(0, 32000, (batch_size, 10))
    post_attn_embedding = torch.randn(batch_size, 4096)
    router_logits_history = torch.randn(batch_size, 9, 8)
    
    # Forward pass
    logits = model(context_tokens, post_attn_embedding, router_logits_history)
    print(f"Output shape: {logits.shape}")
    
    # Predict top-k
    top_k_indices, top_k_probs = model.predict_top_k(
        context_tokens, post_attn_embedding, router_logits_history
    )
    print(f"Top-k indices shape: {top_k_indices.shape}")
    print(f"Top-k probs shape: {top_k_probs.shape}")
    print(f"\nExample predictions:")
    print(f"  Predicted experts: {top_k_indices[0].tolist()}")
    print(f"  Probabilities: {top_k_probs[0].tolist()}")
