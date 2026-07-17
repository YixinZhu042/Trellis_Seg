#!/bin/bash
# ============================================================
# Push Seg VAE checkpoints to Hugging Face Hub
#
# Usage:
#   bash scripts/push_vae_to_hf.sh [STEP] [REPO_ID]
#
# Examples:
#   bash scripts/push_vae_to_hf.sh 1000 your-username/trellis-seg-vae
#   bash scripts/push_vae_to_hf.sh latest your-username/trellis-seg-vae
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login
# ============================================================

set -e

# ============================================================
# Configuration
# ============================================================
CKPT_DIR="./checkpoints/seg_vae"
CONFIG_FILE="./configs/seg_vae_train.json"

STEP=${1:-"latest"}
REPO_ID=${2:-"GilgameshYX/Trellis_segPart"}

# ============================================================
# Resolve step
# ============================================================
if [ "${STEP}" = "latest" ]; then
    STEP=$(ls ${CKPT_DIR}/misc_step*.pt 2>/dev/null | sed 's/.*step\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
    if [ -z "${STEP}" ]; then
        echo "Error: No checkpoints found in ${CKPT_DIR}"
        exit 1
    fi
    echo "Resolved latest step: ${STEP}"
fi

# Zero-pad to 7 digits
STEP_PAD=$(printf "%07d" ${STEP})

# ============================================================
# Verify files exist
# ============================================================
FILES=(
    "encoder_step${STEP_PAD}.pt"
    "decoder_step${STEP_PAD}.pt"
)

echo "============================================"
echo " Push Seg VAE to Hugging Face"
echo "  Repo: ${REPO_ID}"
echo "  Step: ${STEP} (${STEP_PAD})"
echo "  CKPT dir: ${CKPT_DIR}"
echo "============================================"

for f in "${FILES[@]}"; do
    if [ ! -f "${CKPT_DIR}/${f}" ]; then
        echo "Error: ${CKPT_DIR}/${f} not found"
        exit 1
    fi
    echo "  Found: ${f} ($(du -h ${CKPT_DIR}/${f} | cut -f1))"
done

# ============================================================
# Create temporary upload directory with clean structure
# ============================================================
UPLOAD_DIR=$(mktemp -d)
trap "rm -rf ${UPLOAD_DIR}" EXIT

echo ""
echo "Preparing upload directory..."

# Copy non-EMA weights
cp "${CKPT_DIR}/encoder_step${STEP_PAD}.pt" "${UPLOAD_DIR}/encoder.pt"
cp "${CKPT_DIR}/decoder_step${STEP_PAD}.pt" "${UPLOAD_DIR}/decoder.pt"

# Copy config
cp "${CONFIG_FILE}" "${UPLOAD_DIR}/config.json"

# Create model card
cat > "${UPLOAD_DIR}/README.md" << 'EOF'
---
tags:
  - 3d
  - segmentation
  - vae
  - sparse-voxel
license: other
---

# Trellis Seg VAE

Sparse 3D VAE for segmentation label encoding/decoding, fine-tuned from TRELLIS.2 Texture VAE.

## Architecture

- **Encoder**: SparseUnetVaeEncoder (20ch one-hot → 32ch latent)
- **Decoder**: SparseUnetVaeDecoder (32ch latent → 20ch logits)
- **Resolution**: 512³ input → 32³ latent grid (16x downsampling)
- **Backbone**: SparseConvNeXtBlock3d, channels [64, 128, 256, 512, 1024]

## Usage

```python
import torch
from trellis2 import models as trellis2_models
from trellis2.modules.sparse.basic import SparseTensor

# Load encoder/decoder
encoder_state = torch.load("encoder.pt", map_location="cuda")
decoder_state = torch.load("decoder.pt", map_location="cuda")

encoder = trellis2_models.SparseUnetVaeEncoder(
    in_channels=20, latent_channels=32,
    model_channels=[64, 128, 256, 512, 1024],
    num_blocks=[0, 4, 8, 16, 4],
    # ... see config.json for full args
).cuda()
encoder.load_state_dict(encoder_state)

decoder = trellis2_models.SparseUnetVaeDecoder(
    out_channels=20, latent_channels=32,
    model_channels=[1024, 512, 256, 128, 64],
    num_blocks=[4, 16, 8, 4, 0],
    # ... see config.json for full args
).cuda()
decoder.load_state_dict(decoder_state)
```

## Files

- `encoder.pt` — Encoder weights
- `decoder.pt` — Decoder weights
- `config.json` — Training configuration

## Training Details

- Fine-tuned from TRELLIS.2 Texture VAE (6ch → 20ch expansion)
- Loss: Cross-entropy + KL (λ=1e-5)
- Optimizer: AdamW, lr=1e-5
- Mixed precision: fp16 (inflat_all mode)
EOF

# Add step info
echo "" >> "${UPLOAD_DIR}/README.md"
echo "## Checkpoint Info" >> "${UPLOAD_DIR}/README.md"
echo "" >> "${UPLOAD_DIR}/README.md"
echo "- Training step: ${STEP}" >> "${UPLOAD_DIR}/README.md"

echo "Upload directory contents:"
ls -lh "${UPLOAD_DIR}"

# ============================================================
# Upload to Hugging Face
# ============================================================
echo ""
echo "Uploading to https://huggingface.co/${REPO_ID} ..."

python3 -c "
from huggingface_hub import HfApi, create_repo

repo_id = '${REPO_ID}'
upload_dir = '${UPLOAD_DIR}'

api = HfApi()

# Create repo if it doesn't exist
try:
    create_repo(repo_id, repo_type='model', exist_ok=True, private=True)
    print(f'Repo ready: {repo_id}')
except Exception as e:
    print(f'Repo creation note: {e}')

# Upload entire folder
api.upload_folder(
    folder_path=upload_dir,
    repo_id=repo_id,
    repo_type='model',
    commit_message='Upload Seg VAE step ${STEP}',
)
print(f'Done! https://huggingface.co/{repo_id}')
"

echo ""
echo "============================================"
echo " Upload complete!"
echo " https://huggingface.co/${REPO_ID}"
echo "============================================"
