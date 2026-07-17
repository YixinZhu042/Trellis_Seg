#!/bin/bash
# Test TRELLIS.2 original Texture VAE for part-id reconstruction.
# Run with synthetic data first to verify the pipeline works.

cd "$(dirname "$0")/.."

# Use synthetic data (no real data needed)
python test_texture_vae_for_seg.py \
    --synthetic \
    --num_classes 20 \
    --num_voxels 50000 \
    --resolution 256 \
    --encoding all \
    --encoder_path /path/to/trellis2_ckpts/tex_enc \
    --decoder_path /path/to/trellis2_ckpts/tex_dec \
    --output_dir ./output/tex_vae_recon_test \
    --device cuda

# If you have real seg_labels data, use:
# python test_texture_vae_for_seg.py \
#     --seg_dir /path/to/seg_labels/ \
#     --num_classes 20 \
#     --encoding all \
#     --output_dir ./output/tex_vae_recon_test_real
