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

def run_benchmark(num_experts, packed_dir):
    """
    New binary signature: run_ssd_benchmark <packed_dir> <iters> [num_experts]
    Output labels:
        "Sequential (K experts, serial loop)"
        "Parallel   (K experts, IOThreadPool)"
    """
    cmd = [
        "sudo", "./run_ssd_benchmark",
        packed_dir, str(ITERS), str(num_experts)
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
    compile_benchmark()

    expert_counts = [1, 2, 3, 4, 5, 6, 7, 8]

    all_data = {
        "Seq": {"time": [], "bw": []},
        "Par": {"time": [], "bw": []},
    }

    for model_name, dirs in MODELS.items():
        print(f"\n--- Running benchmarks for {model_name} ---")
        for count in expert_counts:
            print(f"Loading {count} experts...")
            raw_data = run_benchmark(count, dirs["packed"])

            for key in ("Seq", "Par"):
                if key in raw_data:
                    mean_us, bw_gbps = raw_data[key]
                    time_per_expert_ms = (mean_us / 1000.0) / (count * 48)
                    all_data[key]["time"].append(time_per_expert_ms)
                    all_data[key]["bw"].append(bw_gbps)
                else:
                    print(f"  Warning: missing {key} data for K={count}")
                    all_data[key]["time"].append(None)
                    all_data[key]["bw"].append(None)

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    colors  = {"Seq": "steelblue",  "Par": "darkorange"}
    markers = {"Seq": "^",          "Par": "D"}
    labels  = {"Seq": "Sequential (serial loop)",
               "Par": "Parallel (IOThreadPool, 16 threads)"}

    ax_time, ax_bw, ax_speedup = axes

    # 1. Time per expert
    for key in ("Seq", "Par"):
        valid = [(x, y) for x, y in zip(expert_counts, all_data[key]["time"]) if y is not None]
        xs, ys = zip(*valid) if valid else ([], [])
        ax_time.plot(xs, ys, label=labels[key], color=colors[key],
                     marker=markers[key], markersize=7, linewidth=2)

    ax_time.set_xlabel("Number of experts loaded (K)", fontsize=12)
    ax_time.set_ylabel("Time per expert (ms)", fontsize=12)
    ax_time.set_title("Expert load latency vs parallelism", fontsize=13)
    ax_time.set_xticks(expert_counts)
    ax_time.grid(True, ls="--", alpha=0.5)
    ax_time.legend(fontsize=10)

    # 2. Aggregate bandwidth
    for key in ("Seq", "Par"):
        valid = [(x, y) for x, y in zip(expert_counts, all_data[key]["bw"]) if y is not None]
        xs, ys = zip(*valid) if valid else ([], [])
        ax_bw.plot(xs, ys, label=labels[key], color=colors[key],
                   marker=markers[key], markersize=7, linewidth=2)

    ax_bw.axhline(y=MAX_SSD_BW, color="green", linestyle="--", linewidth=1.5,
                  label=f"SSD peak ({MAX_SSD_BW} GB/s)")
    ax_bw.set_xlabel("Number of experts loaded (K)", fontsize=12)
    ax_bw.set_ylabel("Aggregate bandwidth (GB/s)", fontsize=12)
    ax_bw.set_title("SSD bandwidth utilisation", fontsize=13)
    ax_bw.set_xticks(expert_counts)
    ax_bw.grid(True, ls="--", alpha=0.5)
    ax_bw.legend(fontsize=10)

    # 3. Speedup (Seq wall-time / Par wall-time)
    speedup_vals = []
    for i in range(len(expert_counts)):
        ts = all_data["Seq"]["time"][i]  # time per expert (ms)
        tp = all_data["Par"]["time"][i]
        if ts is not None and tp is not None and tp > 0:
            # wall-clock ratio: (ts * K) / (tp * K) = ts / tp
            speedup_vals.append(ts / tp)
        else:
            speedup_vals.append(None)

    valid_s = [(x, y) for x, y in zip(expert_counts, speedup_vals) if y is not None]
    xs, ys = zip(*valid_s) if valid_s else ([], [])
    ax_speedup.plot(xs, ys, color="purple", marker="D", markersize=7, linewidth=2,
                    label="Speedup (Seq / Par)")
    ax_speedup.axhline(y=1.0, color="black", linestyle="-", linewidth=1, alpha=0.4)
    ax_speedup.set_xlabel("Number of experts loaded (K)", fontsize=12)
    ax_speedup.set_ylabel("Speedup", fontsize=12)
    ax_speedup.set_title("Parallelisation speedup", fontsize=13)
    ax_speedup.set_xticks(expert_counts)
    ax_speedup.grid(True, ls="--", alpha=0.5)
    ax_speedup.legend(fontsize=10)

    plt.tight_layout()

    # ── CSV output ────────────────────────────────────────────────────────────
    import csv
    csv_file = "expert_loading_benchmark.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Num_Experts", "Config", "Time_per_Expert_ms",
                         "Total_Bandwidth_GBps"])
        for i, count in enumerate(expert_counts):
            for key in ("Seq", "Par"):
                t = all_data[key]["time"][i]
                b = all_data[key]["bw"][i]
                if t is not None:
                    writer.writerow([count, labels[key], f"{t:.4f}", f"{b:.4f}"])
    print(f"\nData saved to {csv_file}")

    out_file = "expert_loading_scaling.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
