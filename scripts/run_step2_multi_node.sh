#!/bin/bash
# Multi-node multi-GPU step2 encoding.
# Each node runs this script with different NODE_RANK.
# Work is sharded: total world_size = NUM_NODES * GPUS_PER_NODE
#
# Usage (per node, via cluster scheduler):
#   bash scripts/run_step2_multi_node.sh NODE_RANK NUM_NODES
#
# Example (4 nodes × 8 GPUs = 32 workers):
#   Node 0: bash scripts/run_step2_multi_node.sh 0 4
#   Node 1: bash scripts/run_step2_multi_node.sh 1 4
#   Node 2: bash scripts/run_step2_multi_node.sh 2 4
#   Node 3: bash scripts/run_step2_multi_node.sh 3 4
#
# Cluster scheduler command (paste for each node, set NODE_RANK=0,1,2,3):
#   cd . && bash scripts/run_step2_multi_node.sh ${NODE_RANK} 4

set -e

# Launcher appends: <NUM_NODES> <NODE_RANK> <MASTER_ADDR>
NUM_NODES=${1:-1}
NODE_RANK=${2:-0}
MASTER_ADDR=${3:-"localhost"}  # unused for data-parallel preprocessing
GPUS_PER_NODE=8

DATA_DIR="/path/to/data"
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "============================================"
echo " Step2: Encode mv_seg_latent (Multi-Node)"
echo " Node: ${NODE_RANK}/${NUM_NODES}"
echo " GPUs per node: ${GPUS_PER_NODE}"
echo " Total workers: ${WORLD_SIZE}"
echo " Data: ${DATA_DIR}"
echo "============================================"

# Launch one process per GPU on this node
for LOCAL_RANK in $(seq 0 $((GPUS_PER_NODE - 1))); do
    GLOBAL_RANK=$((NODE_RANK * GPUS_PER_NODE + LOCAL_RANK))
    echo "[Node ${NODE_RANK}] Launching GPU ${LOCAL_RANK} → global rank ${GLOBAL_RANK}/${WORLD_SIZE}"
    CUDA_VISIBLE_DEVICES=$LOCAL_RANK python preprocess/step2_encode_mv_seg_latent.py \
        --data_dir ${DATA_DIR} \
        --num_classes 20 \
        --rank $GLOBAL_RANK \
        --world_size $WORLD_SIZE &
done

wait
echo "[Node ${NODE_RANK}] All ${GPUS_PER_NODE} workers finished."
