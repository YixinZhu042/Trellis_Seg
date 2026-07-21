#!/bin/bash
# Multi-node multi-GPU step1 rendering.
# Same sharding logic as step2.
#
# Usage (per node, via cluster scheduler):
#   bash scripts/run_step1_multi_node.sh NODE_RANK NUM_NODES
#
# Example (4 nodes × 8 GPUs = 32 workers):
#   Node 0: bash scripts/run_step1_multi_node.sh 0 4
#   Node 1: bash scripts/run_step1_multi_node.sh 1 4
#   Node 2: bash scripts/run_step1_multi_node.sh 2 4
#   Node 3: bash scripts/run_step1_multi_node.sh 3 4
#
# Cluster scheduler command:
#   cd . && bash scripts/run_step1_multi_node.sh ${NODE_RANK} 4

set -e

NODE_RANK=${1:-0}
NUM_NODES=${2:-1}
GPUS_PER_NODE=${3:-8}

DATA_DIR="/path/to/data"
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "============================================"
echo " Step1: Render seg_views (Multi-Node)"
echo " Node: ${NODE_RANK}/${NUM_NODES}"
echo " GPUs per node: ${GPUS_PER_NODE}"
echo " Total workers: ${WORLD_SIZE}"
echo " Output: ${DATA_DIR}/seg_views/"
echo "============================================"

for LOCAL_RANK in $(seq 0 $((GPUS_PER_NODE - 1))); do
    GLOBAL_RANK=$((NODE_RANK * GPUS_PER_NODE + LOCAL_RANK))
    echo "[Node ${NODE_RANK}] Launching GPU ${LOCAL_RANK} → global rank ${GLOBAL_RANK}/${WORLD_SIZE}"
    CUDA_VISIBLE_DEVICES=$LOCAL_RANK python preprocess/step1_render_seg_views.py \
        --output_dir ${DATA_DIR} \
        --resolution 512 \
        --rank $GLOBAL_RANK \
        --world_size $WORLD_SIZE &
done

wait
echo "[Node ${NODE_RANK}] All ${GPUS_PER_NODE} workers finished."
