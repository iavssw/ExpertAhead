#!/usr/bin/env bash
# export HOME_LIBS="/home/greg/libraries"

# Need to set this environment variable
# Check GFX version using rocminfo
if ! ROCMINFO_OUT=$(rocminfo 2>&1); then
  echo "Error: rocminfo failed or is not installed." >&2
  return 1
fi

# Extract GFX version (e.g., gfx90a)
CURRENT_GFX=$(echo "$ROCMINFO_OUT" | grep "Name:" | grep -o "gfx[0-9a-f]\+" | head -n 1)

# Whitelist of GFX versions that DON'T need HSA_OVERRIDE
WHITELIST="gfx950 gfx1201 gfx1101 gfx1200 gfx1030 gfx942 gfx90a gfx908 gfx1151"

# If current GFX is not in the whitelist, set the override
if ! echo "$WHITELIST" | grep -qwq "$CURRENT_GFX"; then
  export HSA_OVERRIDE_GFX_VERSION=11.0.0
fi
#export PYTHONPYCACHEPREFIX=/scratch/gregoryj/.cache

sudo apt install libopencv-dev python3-opencv libstdc++-12-dev 
sudo apt install -y python3 python3-venv python3-pip python3-virtualenv

# Remove to avoid conflicts
sudo apt-get remove -y libamdhip64-dev

rm -rf rocmPytorch

python3 -m virtualenv rocmPytorch
# The real path to source might depend on the virtualenv version
if [ -r sandbox/local/bin/activate ]; then
  source rocmPytorch/local/bin/activate
else
  source rocmPytorch/bin/activate
fi
python3 -m pip install --upgrade pip

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing requirements for ROCm 7.1.1..."
python3 -m pip install -r "$SCRIPT_DIR/requirements_rocm7.1.txt"
pip install timm torchsummary transformers
pip install sentencepiece
pip install matplotlib

# Install Huggingface
curl -LsSf https://hf.co/cli/install.sh | bash

# Kernel 24.04 with 6.11.0-26-generic
# GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.11.0-26-generic"
# sudo apt update
# sudo apt install -y \
#   linux-image-6.11.0-26-generic \
#   linux-headers-6.11.0-26-generic \
#   linux-modules-6.11.0-26-generic \
#   linux-modules-extra-6.11.0-26-generic \
#   linux-firmware \
#   dkms build-essential

# libtorch / rocm
# wget https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-2.10.0%2Brocm7.1.zip
# wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb

# xdna-driver-05-19-25
# 0e6d303b2cc2b3fe1cf10aba0acbf57a422588fb

# XDNA-09-19-24 export PDI
# xclbinutil --dump-section AIE_PARTITION:JSON:aie_partition.json --force --input build/final        

# ENV Variables
# export HOME_LIBS="/home/greg/libraries"

# GRUB Large GTT Mem
# GRUB_CMDLINE_LINUX_DEFAULT="cma=0 transparent_hugepage=madvise zswap.enabled=0 ttm.pages_limit=23068672 ttm.page_pool_size=23068672 ttm.dma32_pages_limit=23068672 amdttm.pages_limit=23068672 amdttm.page_pool_size=23068672 amdttm.dma32_pages_limit=23068672"
# GRUB_CMDLINE_LINUX_DEFAULT="cma=0 transparent_hugepage=madvise zswap.enabled=0 ttm.pages_limit=29296875 ttm.page_pool_size=2097152 amdttm.pages_limit=29296875 amdttm.page_pool_size=2097152"
# Hugging Face CLI

