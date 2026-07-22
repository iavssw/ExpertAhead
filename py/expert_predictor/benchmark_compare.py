import torch
import time
import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from expert_predictor_cross_layer import MultiStepExpertPredictor, JITWrapper

def benchmark():
    onnx_path = "/home/michael/heteroPredict/trainingData/qwen3_30b/final_predictor/ablation_emb_only_hist1_h32_f10/layer_0/best_model.onnx"
    pt_path = "/home/michael/heteroPredict/trainingData/qwen3_30b/final_predictor/ablation_emb_only_hist1_h32_f10/layer_0/best.pt"
    
    print("Loading models...")
    
    # 1. Setup ONNX Runtime
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    session_onnx = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])
    
    inputs_onnx = session_onnx.get_inputs()
    dummy_inputs_np = {}
    for inp in inputs_onnx:
        shape = [1 if type(s) == str else s for s in inp.shape]
        dummy_inputs_np[inp.name] = np.random.randn(*shape).astype(np.float32)
        
    # 2. Setup PyTorch model
    checkpoint = torch.load(pt_path, map_location='cpu')
    config = checkpoint['config']
    
    model = MultiStepExpertPredictor(
        history=config['history'], future_steps=config['future_steps'],
        emb_dim=config['emb_dim'], num_experts=config['num_experts'],
        hidden_dim=config['hidden_dim'], use_embedding=config['use_embedding'],
        use_prefill=config['use_prefill'], use_prev=config['use_prev'],
        use_markov=config['use_markov'], noise_std=config['noise_std'],
        layer_idx=0, use_prev_layers=config.get('use_prev_layers', False)
    )
    model.load_state_dict(checkpoint['state'], strict=False)
    model.eval()
    
    # We will trace it to match LibTorch behavior in C++ exactly
    wrapper = JITWrapper(model).eval()
    dummy_inputs_pt = tuple(torch.from_numpy(dummy_inputs_np[inp.name]) for inp in inputs_onnx)
    model_pt = torch.jit.trace(wrapper, dummy_inputs_pt)
    
    n_iters = 1000
    
    print("\nBenchmarking TorchScript (libtorch style)...")
    with torch.no_grad():
        # Warmup
        for _ in range(100):
            model_pt(*dummy_inputs_pt)
            
        start = time.time()
        for _ in range(n_iters):
            model_pt(*dummy_inputs_pt)
        end = time.time()
        
    pt_time = (end - start) * 1000 / n_iters
    print(f"TorchScript Average Latency: {pt_time:.3f} ms")
    
    print("\nBenchmarking ONNX Runtime...")
    # Warmup
    for _ in range(100):
        session_onnx.run(None, dummy_inputs_np)
        
    start = time.time()
    for _ in range(n_iters):
        session_onnx.run(None, dummy_inputs_np)
    end = time.time()
    
    onnx_time = (end - start) * 1000 / n_iters
    print(f"ONNX Average Latency: {onnx_time:.3f} ms")
    
    if onnx_time < pt_time:
        print(f"\nResult: ONNX Runtime is {pt_time / onnx_time:.2f}x faster than TorchScript!")
    else:
        print(f"\nResult: TorchScript is {onnx_time / pt_time:.2f}x faster than ONNX Runtime!")

if __name__ == "__main__":
    benchmark()
