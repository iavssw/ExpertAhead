# ExpertAhead

ExpertAhead is a research runtime for Mixture-of-Experts (MoE) LLM inference on AMD Ryzen AI. It keeps expert weights primarily on SSD, stages them through a small unified memory expert cache, and uses lightweight predictors (ML, gating heuristics, or oracle) to prefetch experts ahead of decode—so large MoE models can run under tight memory budgets.

This repository includes the W4A16 C++/HIP inference backends, expert packing and SSD loading, predictor training, and the scripts used for the ExpertAhead paper experiments.

---

# Libraries Setup Guide

This guide describes how to replicate the `libraries` directory in a directory of your choosing, which contains the necessary external dependencies for building and running the project.

## Prerequisites

### 1. ROCm 7.1.1
Ensure ROCm 7.1.1 is installed.
Reference: [ROCm Quick Start Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.1/install/quick-start.html)

### 2. AOCL (AMD Optimizing CPU Libraries)
Instructions and downloads can be found here:
- [AOCL Archives](https://www.amd.com/en/developer/aocl/aocl-archives.html)
- [AOCL Building from Source](https://docs.amd.com/r/en-US/57404-AOCL-user-guide/3.1.-Building-from-Source)

### 3. CMake
CMake **3.25+** is required (installed into the project venv by `utils/new_setup.sh`).

---

## External libraries (`HOME_LIBS`)

Before setting up the dependencies, choose a location where you want to install the `libraries`. You should export the `HOME_LIBS` environment variable to point to this directory.

Add the following to your `~/.bashrc` or equivalent:

```bash
export HOME_LIBS="/path/to/your/libraries"  # Replace with your path
mkdir -p "$HOME_LIBS"
```

Add that export to `~/.bashrc` (or equivalent), then `source ~/.bashrc`.

### 1. LibTorch
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

### 2. Run Qwen3-30B-A3B
```bash
cd py/unified_llm_w4a16
python3 qwen3_30B-A3B_w4a16_model.py
```

**Common Arguments:**

- `--text "Your prompt here"`: The input text to process.
- `--model-path "QuixiAI/Qwen3-30B-A3B-AWQ"`: The Hugging Face repository ID or local path to the model.
- `--config-path "configs/configs_strixH_qwen3_30B_A3B.json5"`: Path to the configuration file. This file controls heterogeneity settings.
- `--device "cuda"`: The device to run on (`cuda` or `cpu`).
- `--max-new-tokens 16`: The maximum number of new tokens to generate.

**Example Command:**

```bash
python3 qwen3_30B-A3B_w4a16_model.py \
    --text "What is the meaning of life?" \
    --config-path "configs/configs_strixH_qwen3_30B_A3B.json5" \
    --max-new-tokens 32
```

**Common arguments:**

| Flag | Meaning |
|------|---------|
| `--text "..."` | Prompt |
| `--model-path "..."` | HF repo ID or local model path (default AWQ checkpoint) |
| `--config-path "..."` | Device / heterogeneity JSON5 (under `configs/`) |
| `--device cuda\|cpu` | Execution device |
| `--max-new-tokens N` | Generation length |

Device configs live at `py/unified_llm_w4a16/configs/` (e.g. `configs_strixH_qwen3_30B_A3B.json5`). For ExpertAhead cache / prefetch behavior, use a **run config** JSON (`backend`, `max_cached_experts`, `predictor_type`, …) as described in `README_run_config.md`.

> [!NOTE]
> HIP kernels are primarily optimized for **RDNA (gfx11)**. CDNA (gfx9) is supported via Wave64 adaptation but may be slower.

---

## Reproducing ExpertAhead (paper experiments)

### 1. Pack experts for SSD loading
Download and unpack Qwen3-30B-A3B-AWQ weights to  
`py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_unpacked`, then:

### 1. Model Preparation & Expert Packing
The inference runtime requires expert weights to be packed into a custom binary format for efficient SSD loading.
After downloading the Qwen3-30B-A3B-AWQ weights and extracting them to `py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_unpacked`, run the packing script to generate the SSD-optimized binaries:
```bash
cd py/unified_llm_w4a16
python3 pack_experts.py
```

Default packing covers 48 MoE layers × 128 experts and writes `Qwen3-30B-A3B-AWQ_packed`.

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

Trained models are written under the training-data tree.

### 4. End-to-end evaluation
Scripts under `sh_scripts/` drive sweeps and paper tables, for example:

### 4. End-to-End Evaluation (Sweeps)
Execute the inference evaluation scripts located in the `sh_scripts/` directory to generate the final performance metrics (e.g., Tokens Per Second).
```bash
# Run the 10-prompt Oracle sweep across all configurations
CACHE_SIZES="16 32 48 64" ./sh_scripts/run_final_best_10prompt_oracle.sh
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
