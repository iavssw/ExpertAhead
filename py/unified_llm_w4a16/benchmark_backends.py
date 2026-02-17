
import subprocess
import time
import re
import sys
import os
import statistics
import tempfile

# Configuration
MIXTRAL_SCRIPT = "mixtral_8x7B_w4a16_model.py"
PREDICTOR_MODEL_PATH = "/mnt/storage/Michael/michaelg/mixtral_project/expert_prediction_full/results"
PROMPTS_FILE = "prompts.txt"
NUM_PROMPTS = 20  # Number of prompts to run
MAX_NEW_TOKENS = 64

# Base command
BASE_CMD = [
    "python3", MIXTRAL_SCRIPT,
    "--max-new-tokens", str(MAX_NEW_TOKENS),
    "--no-generate" # We override this later
]


def run_benchmark(name, extra_args):
    print(f"Starting benchmark: {name}")
    
    # We now pass the prompts file directly to the script
    cmd = BASE_CMD + ["--benchmark-prompts", PROMPTS_FILE] + extra_args
    
    try:
        # Run the command (single process for all prompts in this config)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse the special output markers we added
        avg_match = re.search(r"BENCHMARK_RESULT_TPS:\s+([\d\.]+)", output)
        std_match = re.search(r"BENCHMARK_RESULT_STD:\s+([\d\.]+)", output)
        
        if avg_match:
            avg_tps = float(avg_match.group(1))
            std_tps = float(std_match.group(1)) if std_match else 0.0
            print(f"  {name} Result: {avg_tps:.2f} +/- {std_tps:.2f} tokens/sec")
            return avg_tps, std_tps
        else:
            print("  Could not parse benchmark results from output.")
            # print(output[-1000:]) 
            return 0, 0
            
    except subprocess.CalledProcessError as e:
        print(f"    Error running benchmark (Exit {e.returncode}):")
        if e.stderr:
            print(e.stderr[-500:])
        else:
            print("    No stderr output captured.")
        return 0, 0
    except Exception as e:
        print(f"    Unexpected error: {e}")
        return 0, 0

def main():
    # Make sure prompts file exists where the script expects it
    if not os.path.exists(PROMPTS_FILE):
        print(f"Error: Prompts file {PROMPTS_FILE} not found.")
        # Try to locate it relative to this script if running from elsewhere
        current_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(current_dir, "prompts.txt")
        if os.path.exists(alt_path):
             PROMPTS_FILE_PATH = alt_path
             print(f"Found prompts at {PROMPTS_FILE_PATH}")
        else:
             sys.exit(1)
    else:
        PROMPTS_FILE_PATH = PROMPTS_FILE
        
    print(f"Using prompts from: {PROMPTS_FILE_PATH}")
    
    results = {}
    
    # 1. Base implementation (Whole Model)
    avg, std = run_benchmark("Base (Whole Model)", ["--backend", "base"])
    results["Base"] = (avg, std)
    
    # 2. Cached (LRU, Cache=2)
    avg, std = run_benchmark("Cached (LRU, Cache=2)", ["--backend", "cached", "--expert-cache", "2"])
    results["Cached_2"] = (avg, std)
    
    # 3. Predictor (Cache=2)
    avg, std = run_benchmark("Predictor (Cache=2)", ["--backend", "predict", "--predictor-model", PREDICTOR_MODEL_PATH, "--expert-cache", "2"])
    results["Predictor_2"] = (avg, std)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Configuration':<25} | {'TPS (Avg)':<10} | {'Std Dev':<10}")
    print("-" * 49)
    for name, (avg, std) in results.items():
        print(f"{name:<25} | {avg:<10.2f} | {std:<10.2f}")

    # Plotting
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        names = list(results.keys())
        avgs = [results[n][0] for n in names]
        stds = [results[n][1] for n in names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, avgs, yerr=stds, capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        plt.title('Token Generation Performance by Backend')
        plt.ylabel('Tokens Per Second (TPS)')
        plt.xlabel('Backend Configuration')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
                     
        output_file = "benchmark_tps_comparison.png"
        plt.savefig(output_file)
        print(f"\nPlot saved to {output_file}")
        
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation.")
    except Exception as e:
        print(f"\nError generating plot: {e}")

if __name__ == "__main__":
    main()
