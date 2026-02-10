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

> [!WARNING]
> The `heteroPredict` (NPU/Heterogeneous) backend is currently non-functional and under development. The `predict` backend should be considered experimental and may not work as expected.


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
