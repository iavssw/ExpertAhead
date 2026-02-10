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

# Optional: Enable experimental memory-efficient attention on AMD GPUs
# This can improve performance and reduce memory usage, but is still experimental
# Uncomment the line below to enable it:
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Set repo root directory
export HETEROMOSAIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$HETEROMOSAIC_ROOT/utils/rocmPytorch/bin/activate"

# Add local torch lib to LD_LIBRARY_PATH
if [ -d "$HETEROMOSAIC_ROOT/utils/rocmPytorch/lib/python3.12/site-packages/torch/lib" ]; then
    export LD_LIBRARY_PATH="$HETEROMOSAIC_ROOT/utils/rocmPytorch/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
fi

export ROCM_PATH=/opt/rocm-7.1.1
echo "Using ROCm 7.1.1 ($ROCM_PATH)"

export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH
export CPATH=$ROCM_PATH/include:$CPATH
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH

if [ -f "/opt/xilinx/xrt/setup.sh" ]; then
    source /opt/xilinx/xrt/setup.sh
fi