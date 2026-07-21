#!/bin/bash
# Train segmentation DIT, fine-tuning from TRELLIS.2 shape DIT
#
# Usage:
#   bash scripts/train_seg_dit.sh [NUM_GPUS] [DATA_DIR] [OUTPUT_DIR]

set -e

NUM_GPUS=${1:-8}
DATA_DIR=${2:-"/path/to/seg_data"}
OUTPUT_DIR=${3:-"./output/seg_dit_finetune"}

CONFIG="configs/slat_flow_seg_dit_1_3B_512_bf16.json"

echo "============================================"
echo "Training Segmentation DIT"
echo "  Config: ${CONFIG}"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs: ${NUM_GPUS}"
echo "============================================"

python train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --data_dir ${DATA_DIR} \
    --num_gpus ${NUM_GPUS} \
    --auto_retry 3
