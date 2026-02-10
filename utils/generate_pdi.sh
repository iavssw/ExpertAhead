#!/bin/bash

# File to store logs
output_file="pdi_gen.log"
output_folder="generated_pdi_insts"

mkdir -p "$output_folder"

#squre
gen_sizes=(
  "8196 4096 1024"
  "8196 4096 4096"
  "8196 4096 14336"
  "8196 14336 4096"
)

TILE="64x128x64"
COLS="8c"
DT="bf16_int4AWQ_bf16"

step_val=256  # M_SPLIT step must be 256
mkdir -p "$DATA_DIR"
for size in "${gen_sizes[@]}"; do
    read M K N <<< "$size"
    
    # Create subdirectory for this size
    size_dir="$output_folder/${M}x${K}x${N}"
    mkdir -p "$size_dir"

    for M_SPLIT in $(seq $step_val $step_val $M); do
        echo "Running with M=${M}, K=${K}, N=${N}, M_SPLIT=${M_SPLIT}"

        # make -f Makefile.chess devicename=npu2 M=$M_SPLIT K=$K N=$N | tee -a $output_file
        make devicename=npu2 M=$M_SPLIT K=$K N=$N | tee -a $output_file
        xclbinutil --dump-section AIE_PARTITION:JSON:aie_partition.json --force --input build/final_${M_SPLIT}x${K}x${N}_${TILE}_${COLS}.xclbin

        latest_pdi=$(ls -t *.pdi 2>/dev/null | head -n 1)
        if [[ -z "$latest_pdi" ]]; then
            echo "No .pdi files found in the current directory."
            exit 1
        fi
        echo "generated PDI" $latest_pdi
        mv "$latest_pdi" "$size_dir/final_${M_SPLIT}x${K}x${N}_${TILE}_${COLS}_${DT}.pdi"
        cp "build/insts_${M_SPLIT}x${K}x${N}_${TILE}_${COLS}.txt" "$size_dir/insts_${M_SPLIT}x${K}x${N}_${TILE}_${COLS}_${DT}.txt"
    done
done