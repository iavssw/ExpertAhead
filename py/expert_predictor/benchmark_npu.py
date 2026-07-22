import time
import numpy as np
import onnxruntime as ort

def benchmark():
    model_path = "/home/michael/heteroPredict/trainingData/qwen3_30b/final_predictor/ablation_emb_only_hist1_h32_f10/layer_0/best_model.onnx"
    
    # 1. Test CPU ONNX Runtime
    print("Testing ONNX Runtime (CPU)...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    session_cpu = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])
    
    inputs = session_cpu.get_inputs()
    dummy_inputs = {}
    for inp in inputs:
        shape = inp.shape
        # if dynamic, replace with 1
        shape = [1 if type(s) == str else s for s in shape]
        dummy_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        
    # Warmup
    for _ in range(100):
        session_cpu.run(None, dummy_inputs)
        
    start = time.time()
    n_iters = 1000
    for _ in range(n_iters):
        session_cpu.run(None, dummy_inputs)
    end = time.time()
    
    cpu_time = (end - start) * 1000 / n_iters
    print(f"CPU Average Latency: {cpu_time:.3f} ms")
    
    # 2. Test NPU (VitisAIExecutionProvider)
    print("\nTesting ONNX Runtime (NPU - Vitis AI)...")
    try:
        session_npu = ort.InferenceSession(model_path, sess_opts, providers=["VitisAIExecutionProvider"])
        
        # Warmup
        for _ in range(100):
            session_npu.run(None, dummy_inputs)
            
        start = time.time()
        for _ in range(n_iters):
            session_npu.run(None, dummy_inputs)
        end = time.time()
        
        npu_time = (end - start) * 1000 / n_iters
        print(f"NPU Average Latency: {npu_time:.3f} ms")
        print(f"Speedup: {cpu_time / npu_time:.2f}x")
    except Exception as e:
        print(f"Could not run on NPU. Error: {e}")
        print("\nNote: Running on the AMD NPU requires installing the Ryzen AI software stack, "
              "including the 'onnxruntime-vitisai' Python package, and setting up the Vitis AI execution provider.")

if __name__ == "__main__":
    benchmark()
