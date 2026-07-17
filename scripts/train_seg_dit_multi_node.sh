#!/bin/bash
# Multi-node training for segmentation DIT
#
# Usage (on each node):
#   bash scripts/train_seg_dit_multi_node.sh NODE_RANK NUM_NODES MASTER_ADDR
#
# Example (2 nodes, 8 GPUs each):
#   Node 0: bash scripts/train_seg_dit_multi_node.sh 0 2 10.0.0.1
#   Node 1: bash scripts/train_seg_dit_multi_node.sh 1 2 10.0.0.1

set -e

NODE_RANK=${1:-0}
NUM_NODES=${2:-1}
MASTER_ADDR=${3:-"localhost"}
MASTER_PORT=${4:-"12345"}
NUM_GPUS=${5:-8}

DATA_DIR=${DATA_DIR:-"/path/to/seg_data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/seg_dit_finetune"}
CONFIG="configs/slat_flow_seg_dit_1_3B_512_bf16.json"

echo "============================================"
echo "Training Segmentation DIT (Multi-Node)"
echo "  Node: ${NODE_RANK}/${NUM_NODES}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs per node: ${NUM_GPUS}"
echo "  Config: ${CONFIG}"
echo "============================================"

python train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --data_dir ${DATA_DIR} \
    --num_gpus ${NUM_GPUS} \
    --num_nodes ${NUM_NODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --auto_retry 3
