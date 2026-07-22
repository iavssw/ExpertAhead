import torch
import sys
from pathlib import Path

# Add current directory to path if needed, though we are running it here
sys.path.append(str(Path(__file__).parent))

from expert_predictor_cross_layer import MultiStepExpertPredictor, TransformerMultiStepPredictor, JITWrapper

def export_onnx(base_dir):
    base_path = Path(base_dir)
    checkpoints = list(base_path.rglob("best.pt"))
    print(f"Found {len(checkpoints)} models to ONNX compile.")
    
    success_count = 0
    fail_count = 0
    
    for ckpt_path in checkpoints:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            config = checkpoint['config']
            
            arch = config.get('arch', 'mlp')
            layer_idx = 0
            
            if ckpt_path.parent.name.startswith("layer_"):
                layer_idx = int(ckpt_path.parent.name.split("_")[1])
            
            if arch == 'transformer':
                model = TransformerMultiStepPredictor(
                    history=config['history'], future_steps=config['future_steps'],
                    emb_dim=config['emb_dim'], num_experts=config['num_experts'],
                    hidden_dim=config['hidden_dim'], tx_layers=config.get('tx_layers', 2),
                    tx_heads=config.get('tx_heads', 4), use_embedding=config['use_embedding'],
                    use_prefill=config['use_prefill'], use_prev=config['use_prev'],
                    use_markov=config['use_markov'], noise_std=config['noise_std'],
                    layer_idx=layer_idx, use_prev_layers=config.get('use_prev_layers', False)
                )
            else:
                model = MultiStepExpertPredictor(
                    history=config['history'], future_steps=config['future_steps'],
                    emb_dim=config['emb_dim'], num_experts=config['num_experts'],
                    hidden_dim=config['hidden_dim'], use_embedding=config['use_embedding'],
                    use_prefill=config['use_prefill'], use_prev=config['use_prev'],
                    use_markov=config['use_markov'], noise_std=config['noise_std'],
                    layer_idx=layer_idx, use_prev_layers=config.get('use_prev_layers', False)
                )
            
            model.load_state_dict(checkpoint['state'], strict=False)
            model.eval()
            
            wrapper = JITWrapper(model).eval()
            
            hs = int(config.get("hidden_size", config['history'] * config['emb_dim']))
            ne = int(config["num_experts"])
            
            ex_emb = torch.randn(1, hs)
            ex_pf = torch.zeros(1, ne)
            ex_pr = torch.zeros(1, ne)
            
            out_path = ckpt_path.parent / "best_model.onnx"
            
            if config.get('use_prev_layers', False) and layer_idx > 0:
                ex_prev_layers = torch.zeros(1, layer_idx * ne)
                inputs = (ex_emb, ex_pf, ex_pr, ex_prev_layers)
                input_names = ["embedding", "prefill_dist", "prev_expert", "prev_layers"]
            else:
                inputs = (ex_emb, ex_pf, ex_pr)
                input_names = ["embedding", "prefill_dist", "prev_expert"]
            
            torch.onnx.export(
                wrapper, 
                inputs, 
                str(out_path), 
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=input_names,
                output_names=["prediction"]
            )
            success_count += 1
            
        except Exception as e:
            print(f"Failed to ONNX compile {ckpt_path}: {e}")
            fail_count += 1

    print(f"\nSummary: {success_count} models ONNX exported successfully, {fail_count} failures.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "../../trainingData/qwen3_30b/final_predictor"
    export_onnx(base_dir)
