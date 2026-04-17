import torch
import sys
from pathlib import Path

# Add current directory to path if needed, though we are running it here
sys.path.append(str(Path(__file__).parent))

from expert_predictor_cross_layer import MultiStepExpertPredictor, TransformerMultiStepPredictor, JITWrapper

def fix_checkpoints(base_dir):
    base_path = Path(base_dir)
    checkpoints = list(base_path.rglob("best.pt"))
    print(f"Found {len(checkpoints)} models to JIT compile.")
    
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
            
            # Use strict=False for State Dict as some config keys might be slightly mismatched over versions
            # but for this script we just want to apply state_dict on same model architecture.
            model.load_state_dict(checkpoint['state'], strict=False)
            model.eval()
            
            wrapper = JITWrapper(model).eval()
            
            hs = int(config.get("hidden_size", config['history'] * config['emb_dim']))
            ne = int(config["num_experts"])
            
            ex_emb = torch.randn(1, hs)
            ex_pf = torch.zeros(1, ne)
            ex_pr = torch.zeros(1, ne)
            
            if config.get('use_prev_layers', False) and layer_idx > 0:
                ex_prev_layers = torch.zeros(1, layer_idx * ne)
                trace = torch.jit.trace(wrapper, (ex_emb, ex_pf, ex_pr, ex_prev_layers), check_trace=False, strict=False)
            else:
                trace = torch.jit.trace(wrapper, (ex_emb, ex_pf, ex_pr), check_trace=False, strict=False)
            
            out_path = ckpt_path.parent / "best_jit.pt"
            trace.save(str(out_path))
            success_count += 1
            
        except Exception as e:
            print(f"Failed to JIT compile {ckpt_path}: {e}")
            fail_count += 1

    print(f"\nSummary: {success_count} models JIT exported successfully, {fail_count} failures.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "../../trainingData/qwen3_30b/massive_baseline_sweep"
    fix_checkpoints(base_dir)
