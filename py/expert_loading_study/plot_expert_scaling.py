import subprocess
import re
import matplotlib.pyplot as plt
import sys
import os

MODELS = {
    "Qwen": {
        "packed": "/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed",
    }
}

ITERS = 50
MAX_SSD_BW = 11.6  # GB/s from fio measurement

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

def run_benchmark(num_experts, packed_dir, num_threads):
    """
    New binary signature: run_ssd_benchmark <packed_dir> <iters> [num_experts] [num_threads]
    """
    cmd = [
        "sudo", "./run_ssd_benchmark",
        packed_dir, str(ITERS), str(num_experts), str(num_threads)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running benchmark for {num_experts} experts:")
        print(result.stderr or result.stdout)
        sys.exit(1)

    data = {}
    # Match lines like:
    #   Sequential (K experts, serial loop)    mean=   858 us  ...  BW= 2.88 GB/s
    #   Parallel   (K experts, IOThreadPool)   mean=   858 us  ...  BW= 2.88 GB/s
    regex = re.compile(
        r'(Sequential|Parallel)\s.*?mean=\s*(\d+)\s+us.*?BW=\s*([\d.]+)\s*GB/s'
    )
    for line in result.stdout.split('\n'):
        m = regex.search(line)
        if m:
            variant = m.group(1)   # "Sequential" or "Parallel"
            mean_us = float(m.group(2))
            bw_gbps = float(m.group(3))
            key = "Seq" if variant == "Sequential" else "Par"
            data[key] = (mean_us, bw_gbps)

    return data

def main():
    plot_only = "--plot-only" in sys.argv
    if not plot_only:
        compile_benchmark()

    expert_counts = [1,2,3,4,5,6,7,8,12,16,24,32,48,64]
    fixed_threads = 16

    all_data = {
        "Seq": {"time": [], "total_time": []},
        "Par": {"time": [], "total_time": []},
    }

    if plot_only:
        import csv
        csv_file = "expert_loading_benchmark.csv"
        print(f"Reading data from {csv_file}...")
        parsed = {"Seq": {}, "Par": {}}
        expert_counts_set = set()
        
        try:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    k = int(row["Num_Experts"])
                    expert_counts_set.add(k)
                    config = "Seq" if "Seq" in row["Config"] else "Par"
                    
                    t_per_exp = float(row["Time_per_Expert_ms"])
                    if "Total_Time_ms" in row:
                        t_total = float(row["Total_Time_ms"])
                    else:
                        # Infer from older CSV format
                        t_total = t_per_exp * (k * 48)
                    
                    parsed[config][k] = (t_per_exp, t_total)
            
            expert_counts = sorted(list(expert_counts_set))
            for k in expert_counts:
                for key in ("Seq", "Par"):
                    if k in parsed[key]:
                        t, tt = parsed[key][k]
                        all_data[key]["time"].append(t)
                        all_data[key]["total_time"].append(tt)
                    else:
                        all_data[key]["time"].append(None)
                        all_data[key]["total_time"].append(None)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            sys.exit(1)
    else:
        for model_name, dirs in MODELS.items():
            print(f"\n--- Running benchmarks for {model_name} (T={fixed_threads}) ---")
            for count in expert_counts:
                print(f"Loading {count} experts...")
                raw_data = run_benchmark(count, dirs["packed"], fixed_threads)
    
                for key in ("Seq", "Par"):
                    if key in raw_data:
                        mean_us, bw_gbps = raw_data[key]
                        total_time_ms = mean_us / 1000.0
                        time_per_expert_ms = total_time_ms / (count * 48)
                        all_data[key]["time"].append(time_per_expert_ms)
                        all_data[key]["total_time"].append(total_time_ms)
                    else:
                        print(f"  Warning: missing {key} data for K={count}")
                        all_data[key]["time"].append(None)
                        all_data[key]["total_time"].append(None)

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors  = {"Seq": "steelblue",  "Par": "darkorange"}
    markers = {"Seq": "^",          "Par": "D"}
    labels  = {"Seq": "Sequential (serial loop)",
               "Par": "Parallel (IOThreadPool)"}

    ax_time, ax_total = axes

    # 1. Time per expert
    for key in ("Seq", "Par"):
        valid = [(x, y) for x, y in zip(expert_counts, all_data[key]["time"]) if y is not None]
        if valid:
            xs, ys = zip(*valid)
            ax_time.plot(xs, ys, label=labels[key], color=colors[key],
                         marker=markers[key], markersize=7, linewidth=2)

    ax_time.set_xlabel("Number of experts loaded (K)", fontsize=12)
    ax_time.set_ylabel("Time per expert (ms)", fontsize=12)
    ax_time.set_title("Expert load latency vs number of experts", fontsize=13)
    ax_time.set_xticks(expert_counts)
    ax_time.grid(True, ls="--", alpha=0.5)
    ax_time.legend(fontsize=10)

    # 2. Total time
    for key in ("Seq", "Par"):
        valid = [(x, y) for x, y in zip(expert_counts, all_data[key]["total_time"]) if y is not None]
        if valid:
            xs, ys = zip(*valid)
            ax_total.plot(xs, ys, label=labels[key], color=colors[key],
                          marker=markers[key], markersize=7, linewidth=2)

    ax_total.set_xlabel("Number of experts loaded (K)", fontsize=12)
    ax_total.set_ylabel("Total time (ms)", fontsize=12)
    ax_total.set_title("Total load latency vs number of experts", fontsize=13)
    ax_total.set_xticks(expert_counts)
    ax_total.grid(True, ls="--", alpha=0.5)
    ax_total.legend(fontsize=10)

    plt.tight_layout()

    # ── CSV output ────────────────────────────────────────────────────────────
    import csv
    csv_file = "expert_loading_benchmark.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Num_Experts", "Config", "Time_per_Expert_ms", "Total_Time_ms"])
        for i, count in enumerate(expert_counts):
            for key in ("Seq", "Par"):
                t = all_data[key]["time"][i]
                tt = all_data[key]["total_time"][i]
                if t is not None:
                    writer.writerow([count, labels[key], f"{t:.4f}", f"{tt:.2f}"])
    print(f"\nData saved to {csv_file}")

    out_file = "expert_loading_scaling.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
