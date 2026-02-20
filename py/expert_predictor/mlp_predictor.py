"""
Modular MLP Expert Predictor with Ablation Study Support

This module implements a multi-stage MLP architecture for predicting which experts
will be selected by the router in Mixtral 8x7B. It supports ablation studies by
allowing selective enabling/disabling of different input features:

1. Context tokens (t-k to t-1)
2. Post-attention, post-RMS norm embeddings
3. Router logits history

Architecture:
    Input Features → Feature-specific MLPs → Fusion MLP → Expert Predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import json
from pathlib import Path


class ContextTokenMLP(nn.Module):
    """MLP for processing context tokens (t-k to t-1) using frozen Mixtral embeddings"""
    
    def __init__(
        self, 
        pretrained_embeddings: torch.Tensor,
        context_window_k: int = 10,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Args:
            pretrained_embeddings: Frozen Mixtral embeddings [vocab_size, embed_dim]
            context_window_k: Number of context tokens
            hidden_dim: Hidden dimension for MLP
            output_dim: Output feature dimension
            dropout: Dropout rate
            use_attention: Whether to use attention pooling over context
        """
        super().__init__()
        self.context_window_k = context_window_k
        self.use_attention = use_attention
        
        # Frozen Mixtral embeddings (keep on CPU to avoid ROCm issues)
        # We use a non-persistent buffer so it is NOT saved in the state_dict
        self.register_buffer('embedding_weight', pretrained_embeddings, persistent=False)
        
        embed_dim = pretrained_embeddings.shape[1]
        
        if use_attention:
            # Attention pooling over context tokens
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            input_size = embed_dim
        else:
            # Simple flattening
            input_size = context_window_k * embed_dim
        
        # MLP to process context
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def _apply(self, fn):
        """Override to keep embedding on CPU even when model is moved to GPU"""
        # Store embedding device
        embedding_device = self.embedding_weight.device
        
        # Apply to all other parameters
        super()._apply(fn)
        
        # Move embedding back to CPU if it was moved
        if self.embedding_weight.device != embedding_device:
            self.embedding_weight = self.embedding_weight.cpu()
        
        return self
    
    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_tokens: [batch_size, context_window_k] token IDs
            
        Returns:
            [batch_size, output_dim] feature vector
        """
        # Store original device
        original_device = context_tokens.device
        
        # Embed tokens on CPU (frozen embeddings stay on CPU to avoid ROCm issues)
        # Ensure tokens are long dtype for indexing
        context_tokens_cpu = context_tokens.cpu().long()
        # Use F.embedding with the registered buffer
        embedded = F.embedding(context_tokens_cpu, self.embedding_weight)  # [batch, k, embed_dim]
        embedded = embedded.to(original_device)
        
        if self.use_attention:
            # Attention pooling: [batch, k, embed_dim] -> [batch, embed_dim]
            attn_scores = self.attention(embedded)  # [batch, k, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, k, 1]
            pooled = (embedded * attn_weights).sum(dim=1)  # [batch, embed_dim]
            return self.mlp(pooled)
        else:
            # Flatten: [batch, k, embed_dim] -> [batch, k * embed_dim]
            flattened = embedded.view(embedded.size(0), -1)
            return self.mlp(flattened)


class EmbeddingMLP(nn.Module):
    """MLP for processing post-attention, post-RMS norm embeddings"""
    
    def __init__(
        self,
        embedding_dim: int = 4096,  # Mixtral hidden size
        hidden_dim: int = 1024,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
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


class RouterLogitsHistoryMLP(nn.Module):
    """MLP for processing router logits history"""
    
    def __init__(
        self,
        num_experts: int = 8,
        context_window_k: int = 10,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.context_window_k = context_window_k
        
        # Input is (k-1) previous router logits, each with num_experts values
        input_size = (context_window_k - 1) * num_experts
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, router_logits_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            router_logits_history: [batch_size, k-1, num_experts] router logits
            
        Returns:
            [batch_size, output_dim] feature vector
        """
        # Flatten: [batch, k-1, num_experts] -> [batch, (k-1) * num_experts]
        batch_size = router_logits_history.size(0)
        flattened = router_logits_history.view(batch_size, -1)
        
        return self.mlp(flattened)


class FusionMLP(nn.Module):
    """MLP for fusing features from different sources and predicting experts"""
    
    def __init__(
        self,
        feature_dims: Dict[str, int],
        num_experts: int = 8,
        top_k: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dims: Dictionary mapping feature names to their dimensions
                         e.g., {'context': 256, 'embedding': 256, 'router_history': 256}
            num_experts: Number of experts to predict
            top_k: Number of top experts to predict
            hidden_dim: Hidden dimension for fusion MLP
            dropout: Dropout rate
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.feature_names = sorted(feature_dims.keys())
        
        # Calculate total input dimension
        total_input_dim = sum(feature_dims.values())
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts),
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Dictionary of feature tensors, each [batch_size, feature_dim]
            
        Returns:
            [batch_size, num_experts] logits for expert selection
        """
        # Concatenate features in consistent order
        feature_list = [features[name] for name in self.feature_names]
        concatenated = torch.cat(feature_list, dim=1)
        
        # Predict expert logits
        return self.fusion_mlp(concatenated)


class ExpertPredictor(nn.Module):
    """
    Complete expert prediction model with ablation study support.
    
    This model can selectively enable/disable different input features to study
    their individual contributions to prediction accuracy.
    """
    
    def __init__(
        self,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        vocab_size: int = 32000,
        embedding_dim: int = 4096,
        num_experts: int = 8,
        top_k: int = 2,
        context_window_k: int = 10,
        # Feature-specific MLP dimensions
        context_output_dim: int = 256,
        embedding_output_dim: int = 256,
        router_history_output_dim: int = 256,
        # Fusion MLP dimensions
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
        # Ablation flags
        use_context_tokens: bool = True,
        use_embedding: bool = True,
        use_router_history: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.context_window_k = context_window_k
        
        # Ablation flags
        self.use_context_tokens = use_context_tokens
        self.use_embedding = use_embedding
        self.use_router_history = use_router_history
        
        # Feature-specific MLPs
        if use_context_tokens:
            if pretrained_embeddings is None:
                raise ValueError("pretrained_embeddings required when use_context_tokens=True")
            self.context_mlp = ContextTokenMLP(
                pretrained_embeddings=pretrained_embeddings,
                context_window_k=context_window_k,
                hidden_dim=512,
                output_dim=context_output_dim,
                dropout=dropout,
                use_attention=use_attention
            )
        
        if use_embedding:
            self.embedding_mlp = EmbeddingMLP(
                embedding_dim=embedding_dim,
                hidden_dim=1024,
                output_dim=embedding_output_dim,
                dropout=dropout
            )
        
        if use_router_history:
            self.router_history_mlp = RouterLogitsHistoryMLP(
                num_experts=num_experts,
                context_window_k=context_window_k,
                hidden_dim=512,
                output_dim=router_history_output_dim,
                dropout=dropout
            )
        
        # Build feature dimensions dict for fusion
        feature_dims = {}
        if use_context_tokens:
            feature_dims['context'] = context_output_dim
        if use_embedding:
            feature_dims['embedding'] = embedding_output_dim
        if use_router_history:
            feature_dims['router_history'] = router_history_output_dim
        
        # Fusion MLP
        self.fusion_mlp = FusionMLP(
            feature_dims=feature_dims,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        context_tokens: Optional[torch.Tensor] = None,
        post_attn_embedding: Optional[torch.Tensor] = None,
        router_logits_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the expert predictor.
        
        Args:
            context_tokens: [batch_size, k] token IDs (if use_context_tokens=True)
            post_attn_embedding: [batch_size, embedding_dim] (if use_embedding=True)
            router_logits_history: [batch_size, k-1, num_experts] (if use_router_history=True)
            
        Returns:
            [batch_size, num_experts] logits for expert selection
        """
        features = {}
        
        # Process each feature type if enabled
        if self.use_context_tokens:
            if context_tokens is None:
                raise ValueError("context_tokens required when use_context_tokens=True")
            features['context'] = self.context_mlp(context_tokens)
        
        if self.use_embedding:
            if post_attn_embedding is None:
                raise ValueError("post_attn_embedding required when use_embedding=True")
            features['embedding'] = self.embedding_mlp(post_attn_embedding)
        
        if self.use_router_history:
            if router_logits_history is None:
                raise ValueError("router_logits_history required when use_router_history=True")
            features['router_history'] = self.router_history_mlp(router_logits_history)
        
        # Fuse features and predict
        return self.fusion_mlp(features)
    
    def predict_top_k(
        self,
        context_tokens: Optional[torch.Tensor] = None,
        post_attn_embedding: Optional[torch.Tensor] = None,
        router_logits_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k experts.
        
        Returns:
            top_k_indices: [batch_size, top_k] indices of top experts
            top_k_probs: [batch_size, top_k] probabilities for top experts
        """
        logits = self.forward(context_tokens, post_attn_embedding, router_logits_history)
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        return top_k_indices, top_k_probs
    
    def get_config(self) -> Dict:
        """Get model configuration for saving/loading"""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'context_window_k': self.context_window_k,
            'use_context_tokens': self.use_context_tokens,
            'use_embedding': self.use_embedding,
            'use_router_history': self.use_router_history,
        }
    
    def save(self, path: str):
        """Save model and configuration"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
        }, path)
        
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
        checkpoint = torch.load(path, map_location=device)
        
        # Check if we need embeddings and warn if missing
        config = checkpoint['config']
        if config.get('use_context_tokens', False) and pretrained_embeddings is None:
             print("WARNING: Loading model that uses context tokens but pretrained_embeddings not provided.")
             print("Please provide pretrained_embeddings to load() to avoid initialization errors or random embeddings.")
             
        model = cls(pretrained_embeddings=pretrained_embeddings, **config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) # strict=False to handle missing embedding buffer if it was stripped
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
        'all': ExpertPredictor(
            pretrained_embeddings=pretrained_embeddings,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            context_window_k=context_window_k,
            use_context_tokens=True,
            use_embedding=True,
            use_router_history=True,
        ),
        'context_only': ExpertPredictor(
            pretrained_embeddings=pretrained_embeddings,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            context_window_k=context_window_k,
            use_context_tokens=True,
            use_embedding=False,
            use_router_history=False,
        ),
        'embedding_only': ExpertPredictor(
            pretrained_embeddings=pretrained_embeddings,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            context_window_k=context_window_k,
            use_context_tokens=False,
            use_embedding=True,
            use_router_history=False,
        ),
        'router_history_only': ExpertPredictor(
            pretrained_embeddings=pretrained_embeddings,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            context_window_k=context_window_k,
            use_context_tokens=False,
            use_embedding=False,
            use_router_history=True,
        ),
        'context_embedding': ExpertPredictor(
            pretrained_embeddings=pretrained_embeddings,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            context_window_k=context_window_k,
            use_context_tokens=True,
            use_embedding=True,
            use_router_history=False,
        ),
        'context_router': ExpertPredictor(
            pretrained_embeddings=pretrained_embeddings,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            context_window_k=context_window_k,
            use_context_tokens=True,
            use_embedding=False,
            use_router_history=True,
        ),
        'embedding_router': ExpertPredictor(
            pretrained_embeddings=pretrained_embeddings,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            top_k=top_k,
            context_window_k=context_window_k,
            use_context_tokens=False,
            use_embedding=True,
            use_router_history=True,
        ),
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
        if config['use_context_tokens']:
            features.append('context')
        if config['use_embedding']:
            features.append('embedding')
        if config['use_router_history']:
            features.append('router_history')
        
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
