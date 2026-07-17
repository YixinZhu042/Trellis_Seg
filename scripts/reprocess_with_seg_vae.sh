#!/bin/bash
# ============================================================
# 用新 Seg VAE (one-hot 20ch) 重新预处理 + 训练 DIT
#
# 前置条件:
#   - Seg VAE encoder 已训好: seg_vae_ft_2/ckpts/encoder_step0001000.pt
#   - seg_voxels 已生成 (step4b)
#   - seg_views 已生成 (step1)
#   - shape_latent 已生成 (step3, 不需要重跑)
#
# Usage:
#   bash scripts/reprocess_with_seg_vae.sh [NUM_GPUS]
# ============================================================

set -e

NUM_GPUS=${1:-8}
PYTHON="python"
PROJECT_DIR="."
DATA_DIR="/path/to/data"

cd ${PROJECT_DIR}

echo "============================================"
echo " Reprocess with Seg VAE (one-hot 20ch)"
echo " GPUs: ${NUM_GPUS}"
echo " Data: ${DATA_DIR}"
echo "============================================"

# ============================================================
# Step 2: Re-encode mv_seg_latent with Seg VAE
# ============================================================
echo ""
echo "[Step 2] Re-encoding mv_seg_latent..."
rm -rf ${DATA_DIR}/mv_seg_latent
mkdir -p ${DATA_DIR}/mv_seg_latent

for RANK in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${RANK} ${PYTHON} preprocess/step2_encode_mv_seg_latent.py \
        --data_dir ${DATA_DIR} \
        --num_classes 20 \
        --rank ${RANK} \
        --world_size ${NUM_GPUS} &
done
wait
echo "[Step 2] Done."

# ============================================================
# Step 4: Re-encode seg_latent with Seg VAE
# ============================================================
echo ""
echo "[Step 4] Re-encoding seg_latent..."
rm -rf ${DATA_DIR}/seg_latent
mkdir -p ${DATA_DIR}/seg_latent

for RANK in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${RANK} ${PYTHON} preprocess/step4_encode_seg_latent.py \
        --data_dir ${DATA_DIR} \
        --num_classes 20 \
        --rank ${RANK} \
        --world_size ${NUM_GPUS} &
done
wait
echo "[Step 4] Done."

# ============================================================
# Step 5: Recompute normalization
# ============================================================
echo ""
echo "[Step 5] Computing normalization stats..."
${PYTHON} scripts/compute_normalization.py \
    --data_dir ${DATA_DIR} \
    --config configs/slat_flow_seg_dit_1_3B_512_bf16.json
echo "[Step 5] Done."

# ============================================================
echo ""
echo "============================================"
echo " All preprocessing complete!"
echo " Next: train DIT with new latents"
echo "   python train.py --config configs/slat_flow_seg_dit_1_3B_512_bf16.json ..."
echo "============================================"
