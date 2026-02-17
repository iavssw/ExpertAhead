
import torch
import torch.nn as nn
import os
import glob

# Define the model structure matching the state_dict
class SubMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        
        # Block 1: Linear -> Norm -> ReLU -> Dropout
        layers.append(nn.Linear(input_dim, hidden_dims[0])) # 0
        layers.append(nn.LayerNorm(hidden_dims[0]))         # 1
        layers.append(nn.ReLU())                            # 2
        layers.append(nn.Dropout(0.1))                      # 3
        
        # Block 2: Linear -> Norm -> ReLU -> Dropout
        layers.append(nn.Linear(hidden_dims[0], hidden_dims[1])) # 4
        layers.append(nn.LayerNorm(hidden_dims[1]))              # 5
        layers.append(nn.ReLU())                                 # 6
        layers.append(nn.Dropout(0.1))                           # 7
        
        # Block 3: Linear (Output)
        layers.append(nn.Linear(hidden_dims[1], output_dim))     # 8
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class FusionSubMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        # Block 1
        layers.append(nn.Linear(input_dim, hidden_dims[0])) # 0
        layers.append(nn.LayerNorm(hidden_dims[0]))         # 1
        layers.append(nn.ReLU())                            # 2
        layers.append(nn.Dropout(0.1))                      # 3
        
        # Block 2
        layers.append(nn.Linear(hidden_dims[0], hidden_dims[1])) # 4
        layers.append(nn.LayerNorm(hidden_dims[1]))              # 5
        layers.append(nn.ReLU())                                 # 6
        layers.append(nn.Dropout(0.1))                           # 7
        
        # Block 3
        layers.append(nn.Linear(hidden_dims[1], output_dim))     # 8
        
        self.fusion_mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.fusion_mlp(x)

class PredictorModel(nn.Module):
    def __init__(self, vocab_size=32000, embedding_dim=4096, num_experts=8):
        super().__init__()
        # Initialize embedding with zeros or small random values
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Embedding MLP: 4096 -> 1024 -> 512 -> 256
        self.embedding_mlp = SubMLP(embedding_dim, [1024, 512], 256)
        
        # Fusion MLP: 256 -> 512 -> 256 -> 8
        self.fusion_mlp = FusionSubMLP(256, [512, 256], num_experts)
        
    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        # Take the last token
        last_token = input_ids[:, -1]
        emb = self.embedding(last_token) # [batch, 4096]
        
        h = self.embedding_mlp.mlp(emb)
        logits = self.fusion_mlp.fusion_mlp(h)
        return logits

def convert_all():
    base_dir = "/mnt/storage/Michael/michaelg/mixtral_project/expert_prediction_full/results"
    
    for layer_idx in range(32):
        layer_dir = os.path.join(base_dir, f"layer_{layer_idx}")
        pt_file = os.path.join(layer_dir, "embedding_only_best.pt")
        bak_file = pt_file + ".bak"
        
        if not os.path.exists(pt_file) and not os.path.exists(bak_file):
            print(f"Skipping layer {layer_idx}: {pt_file} not found")
            continue
            
        # Helper to load from bak if exists (idempotency)
        load_path = bak_file if os.path.exists(bak_file) else pt_file
            
        print(f"Processing layer {layer_idx} (Loading from {load_path})...")
        try:
            checkpoint = torch.load(load_path, map_location='cpu')
            
            # Check if already scripted (if we run this script twice)
            if isinstance(checkpoint, torch.jit.ScriptModule):
                 print(f"  Layer {layer_idx} is already a ScriptModule. Skipping.")
                 continue
                 
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            config = checkpoint.get('config', {})
            
            model = PredictorModel(
                vocab_size=config.get('vocab_size', 32000),
                embedding_dim=config.get('embedding_dim', 4096),
                num_experts=config.get('num_experts', 8)
            )
            
            # Load state dict
            keys = model.load_state_dict(state_dict, strict=False)
            # print(f"  Keys: {keys}")
            
            model.eval()
            
            # Trace
            example_input = torch.zeros((1, 32), dtype=torch.long)
            traced = torch.jit.trace(model, example_input)
            
            # Save
            if not os.path.exists(bak_file):
                 os.rename(pt_file, bak_file)
            
            traced.save(pt_file)
            print(f"  Saved TorchScript model to {pt_file}")
            
        except Exception as e:
            print(f"  Error converting layer {layer_idx}: {e}")
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    convert_all()
