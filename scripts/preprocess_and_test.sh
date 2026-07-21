#!/bin/bash
# Step 1: Preprocess PartNeXt data to seg_labels
# Step 2: Test texture VAE reconstruction on real data

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="./data/seg_labels"

echo "=========================================="
echo "Step 1: Preprocess PartNeXt → seg_labels"
echo "=========================================="

python preprocess_partnext_to_seg_labels.py \
    --arrow_dir /path/to/PartNeXt \
    --mesh_dir /path/to/PartNeXt_mesh/glbs \
    --output_dir ${OUTPUT_DIR} \
    --resolution 256 \
    --points_per_face 10 \
    --num_samples 100 \
    --min_voxels 5000

echo ""
echo "=========================================="
echo "Step 2: Test Texture VAE reconstruction"
echo "=========================================="

python test_texture_vae_for_seg.py \
    --seg_dir ${OUTPUT_DIR} \
    --num_classes 20 \
    --encoding all \
    --max_samples 50 \
    --encoder_path /path/to/trellis2_ckpts/tex_enc \
    --decoder_path /path/to/trellis2_ckpts/tex_dec \
    --output_dir ./output/tex_vae_recon_real \
    --device cuda

echo ""
echo "Done! Check ./output/tex_vae_recon_real/ for detailed results."
