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

### 3. Kernel and GTT Memory (before library setup)

Install the recommended kernel and configure GRUB for large GTT (Graphics Translation Table) memory. Do this **before** running the library setup steps below.

**Install kernel 6.11.0-26-generic (e.g. Ubuntu 24.04):**

```bash
sudo apt update
sudo apt install -y \
  linux-image-6.11.0-26-generic \
  linux-headers-6.11.0-26-generic \
  linux-modules-6.11.0-26-generic \
  linux-modules-extra-6.11.0-26-generic \
  linux-firmware \
  dkms build-essential
```

To boot this kernel by default, set in `/etc/default/grub`:

```bash
GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.11.0-26-generic"
```

**Configure large GTT memory** by editing `/etc/default/grub` and setting `GRUB_CMDLINE_LINUX_DEFAULT`. Use one of the following (replace the existing `GRUB_CMDLINE_LINUX_DEFAULT` line):

```bash
# Option A (FOR 64G Systems)
GRUB_CMDLINE_LINUX_DEFAULT="cma=0 transparent_hugepage=madvise zswap.enabled=0 ttm.pages_limit=15728640 ttm.page_pool_size=15728640 amdttm.pages_limit=15728640 amdttm.page_pool_size=15728640"

# Option B (FOR 96G Systems)
GRUB_CMDLINE_LINUX_DEFAULT="cma=0 transparent_hugepage=madvise zswap.enabled=0 ttm.pages_limit=23068672 ttm.page_pool_size=23068672 amdttm.pages_limit=23068672 amdttm.page_pool_size=23068672"

# Option C (FOR 128G Systems)
GRUB_CMDLINE_LINUX_DEFAULT="cma=0 transparent_hugepage=madvise zswap.enabled=0 ttm.pages_limit=29296875 ttm.page_pool_size=2097152 amdttm.pages_limit=29296875 amdttm.page_pool_size=2097152"
```

Then update GRUB and reboot:

```bash
sudo update-grub
sudo reboot
```

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
Ensure you have the AOCL tarball (e.g., `aocl-linux-gcc-5.1.0.tar.gz`) available. Select 1 when Prompted.
```bash
cd "$HOME_LIBS"
mkdir aocl
# Assuming the tarball is in your current directory or adjust path
tar -xvf /path/to/aocl-linux-gcc-5.1.0.tar.gz
cd aocl-linux-gcc-5.1.0/
./install.sh -t $HOME_LIBS/aocl
cd ..
```

### 3. Setup XDNA Driver

**Step A — Install .deb packages**

From the project root, install the XDNA/XRT packages in `utils/xdna_deb_051925`:

```bash
cd /path/to/heteroPredict/utils/xdna_deb_051925/ # project root
sudo apt install ./xrt_202520.2.20.0_24.04-amd64-base.deb
sudo apt install ./xrt_202520.2.20.0_24.04-amd64-base-dev.deb
sudo apt install ./xrt_202520.2.20.0_24.04-amd64-npu.deb
sudo apt install ./xrt_plugin.2.20.0_ubuntu24.04-x86_64-amdxdna.deb
```

If `dpkg -i` reports missing dependencies, run `sudo apt install -f` and then re-run the `dpkg -i` commands as needed.

**Step B Clone Source so that the project can link to it**

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
python3 llama3_8b_w4a16_model.py
```

**Common Arguments:**

- `--text "Your prompt here"`: The input text to process.
- `--model-path "meta-llama/Llama-3.1-8B-Instruct"`: The Hugging Face repository ID or local path to the model.
- `--config-path "configs/configs_strixP_llama3.json5"`: Path to the NPU configuration file. This file controls heterogeneity settings.
- `--device "cuda"`: The device to run on (`cuda` or `cpu`).
- `--max-new-tokens 16`: The maximum number of new tokens to generate.

**Example Command:**

```bash
python3 llama3_8b_w4a16_model.py \
    --text "What is the meaning of life?" \
    --config-path "configs/configs_strixP_llama3.json5" \
    --max-new-tokens 32
```

### 3. Configuration

Runtime behavior is controlled by the JSON5 config file at `py/unified_llm_w4a16/configs/configs_strixP_llama3.json5`. See this file for available options such as heterogeneity mode, warmup, MoE kernel preloading, debug verbosity, and more.

> [!NOTE]
> The HIP kernels in this project are primarily optimized for **RDNA (gfx11)** architectures. They do not work for CDNA architectures with a wave size of 64.
