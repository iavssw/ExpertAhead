#!/bin/bash
echo "Dropping page cache..."
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Repacking the experts using our updated python script
echo "Please repack your experts by running:"
echo "python3 py/unified_llm_w4a16/pack_experts.py --src model_weights/Qwen3-30B-A3B-AWQ_unpacked --dst model_weights/Qwen3-30B-A3B-AWQ_packed --num-layers 48 --num-experts 128 --workers 16"
echo "python3 py/unified_llm_w4a16/pack_experts.py --src model_weights/mixtral-8x7b-v0.1-AWQ_unpacked --dst model_weights/mixtral-8x7b-v0.1-AWQ_packed --num-layers 32 --num-experts 8 --workers 8"
