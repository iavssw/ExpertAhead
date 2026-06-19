import subprocess
import re
import matplotlib.pyplot as plt
import sys
import os

MODELS = {
    "Mixtral": {
        "packed": "/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/mixtral-8x7b-v0.1-AWQ_packed",
        "unpacked": "/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/mixtral-8x7b-v0.1-AWQ_unpacked"
    },
    "Qwen": {
        "packed": "/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed",
        "unpacked": "/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_unpacked"
    }
}

ITERS = 10
MAX_SSD_BW = 11.6 # From previous fio measurement

def compile_benchmark():
    print("Compiling benchmark...")
    try:
        torch_path = subprocess.check_output(
            ["python3", "-c", "import torch; import os; print(os.path.dirname(torch.__file__))"],
            text=True
        ).strip()
    except Exception:
        torch_path = os.environ.get("HOME_LIBS", "") + "/libtorch_7.1.0"

    cmd = [
        "g++", "-O3", "-std=c++17", "expert_benchmark.cpp",
        f"-I{torch_path}/include",
        f"-I{torch_path}/include/torch/csrc/api/include",
        f"-L{torch_path}/lib",
        f"-Wl,-rpath,{torch_path}/lib",
        "-ltorch", "-ltorch_cpu", "-lc10", "-lpthread",
        "-o", "run_ssd_benchmark"
    ]
    subprocess.run(cmd, check=True)
    print("Compilation successful.")

def run_benchmark(num_experts, packed_dir, unpacked_dir):
    cmd = [
        "sudo", "./run_ssd_benchmark",
        packed_dir, unpacked_dir, str(ITERS), str(num_experts)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running benchmark for {num_experts} experts:")
        print(result.stderr)
        sys.exit(1)
    
    data = {}
    lines = result.stdout.split('\n')
    
    regex = re.compile(r'\[(.*?)\](.*?)\s+mean=\s+(\d+)\s+us.*?BW=\s*([\d.]+)\s*GB/s')
    
    for line in lines:
        match = regex.search(line)
        if match:
            fmt = match.group(1).strip() # "Packed" or "Unpacked"
            desc = match.group(2).strip() # e.g. "K Experts (Seq Loop, Par Tensors)"
            mean_us = float(match.group(3))
            bw_gbps = float(match.group(4))
            
            if num_experts == 1:
                if "1 Expert Sequential" in desc:
                    data[f"{fmt} Seq"] = (mean_us, bw_gbps)
                elif "1 Expert Parallel" in desc:
                    data[f"{fmt} Par"] = (mean_us, bw_gbps)
            else:
                if "Seq Loop" in desc:
                    data[f"{fmt} Seq"] = (mean_us, bw_gbps)
                elif "Par Loop" in desc:
                    data[f"{fmt} Par"] = (mean_us, bw_gbps)
                    
    return data

def main():
    compile_benchmark()
    
    model_expert_counts = {
        "Mixtral": [1, 2, 4, 8],
        "Qwen": [1, 2, 4, 8, 16, 32, 64]
    }
    
    # Store data for both models
    all_data = {
        "Mixtral": { "Unpacked Seq": {"time": [], "bw": []}, "Unpacked Par": {"time": [], "bw": []}, "Packed Seq": {"time": [], "bw": []}, "Packed Par": {"time": [], "bw": []} },
        "Qwen":    { "Unpacked Seq": {"time": [], "bw": []}, "Unpacked Par": {"time": [], "bw": []}, "Packed Seq": {"time": [], "bw": []}, "Packed Par": {"time": [], "bw": []} }
    }
    
    for model_name, dirs in MODELS.items():
        print(f"\n--- Running benchmarks for {model_name} ---")
        for count in model_expert_counts[model_name]:
            print(f"Loading {count} experts...")
            raw_data = run_benchmark(count, dirs["packed"], dirs["unpacked"])
            
            for key in all_data[model_name].keys():
                if key in raw_data:
                    mean_us, bw_gbps = raw_data[key]
                    time_per_expert_ms = (mean_us / 1000.0) / count
                    all_data[model_name][key]["time"].append(time_per_expert_ms)
                    all_data[model_name][key]["bw"].append(bw_gbps)
                else:
                    print(f"Warning: Missing data for {model_name} {key} at {count} experts")
                    all_data[model_name][key]["time"].append(None)
                    all_data[model_name][key]["bw"].append(None)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    
    colors = {
        "Unpacked Seq": "lightcoral",
        "Unpacked Par": "red",
        "Packed Seq": "lightblue",
        "Packed Par": "blue",
        "Unpacked Speedup": "red",
        "Packed Speedup": "blue"
    }
    
    markers = {
        "Unpacked Seq": "o",
        "Unpacked Par": "s",
        "Packed Seq": "^",
        "Packed Par": "D"
    }
    
    for row_idx, model_name in enumerate(["Mixtral", "Qwen"]):
        data_series = all_data[model_name]
        counts = model_expert_counts[model_name]
        ax_time = axes[row_idx, 0]
        ax_bw = axes[row_idx, 1]
        ax_speedup = axes[row_idx, 2]
        
        # 1. Time per Expert
        for key, series in data_series.items():
            ax_time.plot(counts, series["time"], label=key, color=colors[key], 
                         linestyle='-', marker=markers[key], markersize=6, linewidth=2, alpha=0.8)

        ax_time.set_xscale('log', base=2)
        ax_time.set_xticks(counts)
        ax_time.set_xticklabels(counts)
        ax_time.set_xlabel('Number of Experts Loaded Concurrently', fontsize=12)
        ax_time.set_ylabel('Time per Expert (ms)', fontsize=12)
        ax_time.set_title(f'{model_name} Time per Expert vs Parallelization', fontsize=14)
        ax_time.grid(True, which="both", ls="--", alpha=0.6)
        ax_time.legend(fontsize=9, loc='upper right', ncol=2)

        # 2. Bandwidth
        for key, series in data_series.items():
            ax_bw.plot(counts, series["bw"], label=key, color=colors[key], 
                       linestyle='-', marker=markers[key], markersize=6, linewidth=2, alpha=0.8)

        ax_bw.axhline(y=MAX_SSD_BW, color='green', linestyle='-', linewidth=2, label=f"Max SSD Bandwidth ({MAX_SSD_BW} GB/s)")
        
        ax_bw.set_xscale('log', base=2)
        ax_bw.set_xticks(counts)
        ax_bw.set_xticklabels(counts)
        ax_bw.set_xlabel('Number of Experts Loaded Concurrently', fontsize=12)
        ax_bw.set_ylabel('Throughput (GB/s)', fontsize=12)
        ax_bw.set_title(f'{model_name} SSD Bandwidth Saturation', fontsize=14)
        ax_bw.grid(True, which="both", ls="--", alpha=0.6)
        ax_bw.legend(fontsize=9, loc='lower right', ncol=2)
        
        # 3. Speedup
        packed_speedup = []
        unpacked_speedup = []
        for i in range(len(counts)):
            p_seq = data_series["Packed Seq"]["time"][i]
            p_par = data_series["Packed Par"]["time"][i]
            if p_seq and p_par and p_par > 0:
                packed_speedup.append(p_seq / p_par)
            else:
                packed_speedup.append(None)
                
            u_seq = data_series["Unpacked Seq"]["time"][i]
            u_par = data_series["Unpacked Par"]["time"][i]
            if u_seq and u_par and u_par > 0:
                unpacked_speedup.append(u_seq / u_par)
            else:
                unpacked_speedup.append(None)
        
        ax_speedup.plot(counts, unpacked_speedup, label="Unpacked Speedup", color=colors["Unpacked Speedup"], 
                 linestyle='-', marker='s', markersize=6, linewidth=2, alpha=0.8)
        ax_speedup.plot(counts, packed_speedup, label="Packed Speedup", color=colors["Packed Speedup"], 
                 linestyle='-', marker='D', markersize=6, linewidth=2, alpha=0.8)

        ax_speedup.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax_speedup.set_xscale('log', base=2)
        ax_speedup.set_xticks(counts)
        ax_speedup.set_xticklabels(counts)
        ax_speedup.set_xlabel('Number of Experts Loaded Concurrently', fontsize=12)
        ax_speedup.set_ylabel('Speedup Factor (Seq / Par)', fontsize=12)
        ax_speedup.set_title(f'{model_name} Parallelization Speedup', fontsize=14)
        ax_speedup.grid(True, which="both", ls="--", alpha=0.6)
        ax_speedup.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    
    import csv
    csv_file = "expert_loading_benchmark.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Num_Experts", "Configuration", "Time_per_Expert_ms", "Total_Bandwidth_GBps"])
        for model_name in MODELS.keys():
            counts = model_expert_counts[model_name]
            for i, count in enumerate(counts):
                for key in all_data[model_name].keys():
                    t = all_data[model_name][key]["time"][i]
                    b = all_data[model_name][key]["bw"][i]
                    if t is not None:
                        writer.writerow([model_name, count, key, f"{t:.4f}", f"{b:.4f}"])
    print(f"\nData saved successfully to {csv_file}!")
    
    out_file = "expert_loading_scaling.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved successfully to {out_file}!")

if __name__ == "__main__":
    main()
