#!/bin/bash
# Rerun step1 + step2 with label-0 fix.
# Step1: re-renders seg_views with face_labels+1 (so 0=background, 1+=parts)
# Step2: re-encodes mv_seg_latent with -1 correction (restoring original label space)
#
# Usage:
#   bash scripts/rerun_step1_step2.sh [NUM_GPUS]
#
# Example (8 GPUs):
#   bash scripts/rerun_step1_step2.sh 8

set -e

NUM_GPUS=${1:-8}
DATA_DIR="/path/to/data"

echo "============================================"
echo " Rerunning step1 + step2 (label-0 fix)"
echo " DATA_DIR: ${DATA_DIR}"
echo " NUM_GPUS: ${NUM_GPUS}"
echo "============================================"

# Clear old outputs (optional - uncomment if you want a clean rerun)
# rm -rf ${DATA_DIR}/seg_views
# rm -rf ${DATA_DIR}/mv_seg_latent

# ============================================================
# Step 1: Re-render seg_views (with +1 fix)
# ============================================================
echo ""
echo "[Step 1] Rendering seg_views (face_labels + 1)..."
echo "  Output: ${DATA_DIR}/seg_views/"
echo ""

for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$RANK python preprocess/step1_render_seg_views.py \
        --output_dir ${DATA_DIR} \
        --resolution 512 \
        --rank $RANK \
        --world_size $NUM_GPUS &
done
wait
echo "[Step 1] Done."

# ============================================================
# Step 2: Re-encode mv_seg_latent (with -1 fix)
# ============================================================
echo ""
echo "[Step 2] Encoding mv_seg_latent (with -1 restore)..."
echo "  Output: ${DATA_DIR}/mv_seg_latent/"
echo ""

for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$RANK python preprocess/step2_encode_mv_seg_latent.py \
        --data_dir ${DATA_DIR} \
        --num_classes 20 \
        --rank $RANK \
        --world_size $NUM_GPUS &
done
wait
echo "[Step 2] Done."

echo ""
echo "============================================"
echo " All done! Next steps:"
echo "  1. Verify: python -c \"import numpy as np; d=np.load('${DATA_DIR}/seg_views/\$(ls ${DATA_DIR}/seg_views|head -1)'); print('min:', d['seg_maps'].min(), 'max:', d['seg_maps'].max())\""
echo "     (min should be 0, max should be num_parts)"
echo "  2. Retrain DIT: python train.py --config configs/slat_flow_seg_dit_1_3B_512_bf16.json ..."
echo "============================================"
