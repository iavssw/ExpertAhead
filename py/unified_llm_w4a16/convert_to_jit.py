import torch
import os
import sys

# Add path so we can import the model definition
sys.path.append('/home/michael/heteroPredict/py')
from expert_predictor.mlp_predictor import ExpertPredictor

base_dir = '/home/michael/mixtral_project/expert_prediction_full/embedding_only_predictors'

class PredictorWrapper(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        
    def forward(self, embedding):
        return self.predictor(None, embedding, None)

def process_layer(layer_idx):
    layer_dir = os.path.join(base_dir, f'layer_{layer_idx}')
    if not os.path.exists(layer_dir):
        return
        
    pt_path = os.path.join(layer_dir, 'embedding_predictor_best.pt')
    backup_path = os.path.join(layer_dir, 'embedding_predictor_best.pt.original')
    
    # If backup doesn't exist, it means pt_path is the original state dict
    if not os.path.exists(backup_path):
        os.rename(pt_path, backup_path)
    
    # Check if backup exists now
    if not os.path.exists(backup_path):
        return
        
    # Load model
    try:
        checkpoint = torch.load(backup_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading {backup_path}: {e}")
        return
        
    if 'config' not in checkpoint:
        print(f"Skipping {layer_idx}, no config found")
        return
        
    config = checkpoint['config']
    # Instantiate the model architecture
    model = ExpertPredictor(pretrained_embeddings=None, **config)
    
    # Check for unexpected missing keys (strict=False is fine but good to know)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Wrap it to only accept ONE positional argument
    wrapper = PredictorWrapper(model)
    wrapper.eval()
    
    # Trace the module
    # Create dummy input that matches training time embedding size
    embed_dim = config.get('embedding_dim', 4096)
    example_input = torch.randn(1, embed_dim)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, example_input)
        
    # Save the TorchScript model back to the original .pt filename
    traced_model.save(pt_path)
    print(f"Layer {layer_idx} converted and saved to TorchScript.")

if __name__ == "__main__":
    print("Starting TorchScript conversion...")
    for i in range(32):
        process_layer(i)
    print("Done!")
