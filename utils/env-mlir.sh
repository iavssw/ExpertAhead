# export PATH=$PATH:/tools/Xilinx/Vitis/2023.2/aietools/bin:/tools/Xilinx/Vitis/2023.2/bin
export AIETOOLS_ROOT=/home/greg/libraries/vitis_aie_essentials
export PATH=$PATH:${AIETOOLS_ROOT}/bin
export LM_LICENSE_FILE=/home/greg/libraries/Xilinx.lic

cd mlir-aie
source ironenv/bin/activate
source utils/env_setup.sh install llvm/install
source /opt/xilinx/xrt/setup.sh 