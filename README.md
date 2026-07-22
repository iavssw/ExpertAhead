# Libraries Setup Guide

This guide describes how to replicate the `libraries` directory in a directory of your choosing, which contains the necessary external dependencies for building and running the project.

## Prerequisites

### 1. ROCm 7.1.1
Ensure ROCm 7.1.1 is installed on your system.
Reference: [ROCm Quick Start Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.1/install/quick-start.html)

### 2. AOCL (AMD Optimizing CPU Libraries)
Instructions and downloads can be found here:
- [AOCL Archives](https://www.amd.com/en/developer/aocl/aocl-archives.html)
- [AOCL Building from Source](https://docs.amd.com/r/en-US/57404-AOCL-user-guide/3.1.-Building-from-Source)

---

## Environment Configuration

Before setting up the dependencies, choose a location where you want to install the `libraries`. You should export the `HOME_LIBS` environment variable to point to this directory.

Add the following to your `~/.bashrc` or equivalent:

```bash
export HOME_LIBS="/path/to/your/libraries"  # Replace with your desired path
mkdir -p "$HOME_LIBS"
```

Then reload your shell or run:
```bash
source ~/.bashrc
```

---

## Setup Steps

### 1. Setup LibTorch
```bash
cd "$HOME_LIBS"
wget https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-2.10.0%2Brocm7.1.zip
unzip libtorch-shared-with-deps-2.10.0+rocm7.1.zip
mv libtorch libtorch_7.1.0
rm libtorch-shared-with-deps-2.10.0+rocm7.1.zip
```

### 2. Setup AOCL
Ensure you have the AOCL tarball (e.g., `aocl-linux-gcc-5.1.0.tar.gz`) available.
```bash
cd "$HOME_LIBS"
mkdir aocl
# Assuming the tarball is in your current directory or adjust path
tar -xvf /path/to/aocl-linux-gcc-5.1.0.tar.gz -C aocl --strip-components=1
cd aocl-linux-gcc-5.1.0/
./install.sh -t $HOME_LIBS/aocl
cd ..
```

### 3. Setup XDNA Driver
```bash
cd "$HOME_LIBS"
git clone https://github.com/amd/xdna-driver.git xdna-driver-05-19-25
cd xdna-driver-05-19-25
git checkout 0e6d303b2cc2b3fe1cf10aba0acbf57a422588fb
cd ..
```

---

## Project Build Instructions

Follow these steps to build the project. Note that a GPU (ROCm) setup and **CMake 3.25 or higher** are required.



### 1. Initial Environment Setup
Navigate to the `utils` directory and run the initialization script. This will set up the virtual environment, install Python dependencies, and install a modern version of CMake (3.25+) inside the environment.
```bash
cd utils
source new_setup.sh
```
> [!IMPORTANT]
> This script will fail if ROCm is not correctly set up.

### 2. Build the Project
After the initial setup is complete, **close your current terminal** and open a new one.

In the new terminal, navigate to the project root and follow these steps:

1. **Source the setup script**:
   ```bash
   source utils/setup.sh
   ```

2. **Run CMake and Build**:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make -j$(nproc)
   ```

---

## Running the Model

### 1. Authenticate
Log in to your Hugging Face account:
```bash
hf auth login
```

### 2. Run the Python Script

Navigate to the Python directory and run the model script:
```bash
cd py/unified_llm_w4a16
python3 mixtral_8x7B_w4a16_model.py
```

**Common Arguments:**

- `--text "Your prompt here"`: The input text to process.
- `--model-path "TheBloke/mixtral-8x7b-v0.1-AWQ"`: The Hugging Face repository ID or local path to the model.
- `--config-path "configs/configs_strixH_mixtral7x8B.json5"`: Path to the NPU configuration file. This file controls heterogeneity settings.
- `--device "cuda"`: The device to run on (`cuda` or `cpu`).
- `--max-new-tokens 16`: The maximum number of new tokens to generate.

**Example Command:**

```bash
python3 mixtral_8x7B_w4a16_model.py \
    --text "What is the meaning of life?" \
    --config-path "configs/configs_strixH_mixtral7x8B.json5" \
    --max-new-tokens 32
```

### 3. Configuration

Runtime behavior is controlled by the JSON5 config file at `py/unified_llm_w4a16/configs/configs_strixH_mixtral7x8B.json5`. See this file for available options such as heterogeneity mode, warmup, MoE kernel preloading, debug verbosity, and more.

> [!NOTE]
> The HIP kernels in this project are primarily optimized for **RDNA (gfx11)** architectures. While they support CDNA (gfx9) devices via Wave64 adaptation, performance on CDNA may not be optimal compared to RDNA.

---

## Reproducing ExpertAhead (Paper Experiments)

Follow these steps to reproduce the evaluation and analysis found in the ExpertAhead paper from scratch.

### 1. Model Preparation & Expert Packing
The inference runtime requires expert weights to be packed into a custom binary format for efficient SSD loading.
After downloading the Qwen3-30B-A3B-AWQ weights and extracting them to `py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_unpacked`, run the packing script to generate the SSD-optimized binaries:
```bash
cd py/unified_llm_w4a16
python3 pack_experts.py
```
This script relies on default arguments to process the 48 layers and 128 experts, outputting to the `Qwen3-30B-A3B-AWQ_packed` directory.

### 2. Training Data Collection
Generate the training traces for the predictor using the WikiText dataset.
```bash
cd py/utils
bash run_collect_training_data.sh qwen3_30b
```

### 3. Predictor Training
Train the lightweight cross-token transformer predictors for all layers.
```bash
cd py/expert_predictor
bash run_train.sh qwen3_30b
```
The resulting predictor models will be saved to the training data directory.

### 4. End-to-End Evaluation (Sweeps)
Execute the inference evaluation scripts located in the `sh_scripts/` directory to generate the final performance metrics (e.g., Tokens Per Second).
```bash
# Run the 10-prompt Oracle sweep across all configurations
CACHE_SIZES="16 32 48 64" ./sh_scripts/run_final_best_10prompt_oracle.sh
```

### 5. Generating Plots
Use the Python plotting scripts to recreate the graphs from the paper based on your collected CSV data.
```bash
# Example: Generate Chapter 3 theoretical throughput plots
python3 py/utils/modeling_cache_conditional.py

# Example: Generate best lookahead scaling plots
python3 py/utils/plot_oracle_best_lookahead.py
python3 py/utils/plot_predictor_best_lookahead.py

# Example: Generate final end-to-end evaluation plots
python3 py/utils/plot_thesis_finals.py \
    --csv-sec2 py/utils/final_results_runs/<your_sec2_dir>/sweep.csv \
    --csv-sec4 py/utils/final_results_runs/<your_sec4_dir>/sweep.csv \
    --csv-sec5 py/utils/final_results_runs/<your_sec5_dir>/sweep.csv \
    --out-dir py/utils/final_results_runs/thesis_plots
```
