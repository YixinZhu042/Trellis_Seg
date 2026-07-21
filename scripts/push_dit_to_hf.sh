#!/bin/bash
# ============================================================
# Push Seg DIT checkpoint to Hugging Face Hub
#
# Usage:
#   bash scripts/push_dit_to_hf.sh [REPO_ID]
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login
# ============================================================

set -e

# ============================================================
# Configuration
# ============================================================
CKPT_PATH="/apdcephfs_zwfy/share_303204533/easonyzhu/trellis_seg_run/seg_dit_full_0717/ckpts/denoiser_ema0.9999_step0050000.pt"
CONFIG_FILE="./configs/slat_flow_seg_dit_1_3B_512_bf16.json"
REPO_ID=${1:-"GilgameshYX/Trellis_segPart"}

# ============================================================
# Verify files exist
# ============================================================
echo "============================================"
echo " Push Seg DIT to Hugging Face"
echo "  Repo: ${REPO_ID}"
echo "  CKPT: ${CKPT_PATH}"
echo "============================================"

if [ ! -f "${CKPT_PATH}" ]; then
    echo "Error: ${CKPT_PATH} not found"
    exit 1
fi
echo "  Found: $(basename ${CKPT_PATH}) ($(du -h ${CKPT_PATH} | cut -f1))"

# ============================================================
# Upload to Hugging Face
# ============================================================
echo ""
echo "Uploading to https://huggingface.co/${REPO_ID} ..."

python3 -c "
from huggingface_hub import HfApi, create_repo

repo_id = '${REPO_ID}'
ckpt_path = '${CKPT_PATH}'

api = HfApi()

# Create repo if it doesn't exist
try:
    create_repo(repo_id, repo_type='model', exist_ok=True, private=True)
    print(f'Repo ready: {repo_id}')
except Exception as e:
    print(f'Repo creation note: {e}')

# Upload the checkpoint file
api.upload_file(
    path_or_fileobj=ckpt_path,
    path_in_repo='denoiser_ema0.9999_step0050000.pt',
    repo_id=repo_id,
    repo_type='model',
    commit_message='Upload Seg DIT ema0.9999 step 50000',
)
print(f'Done! https://huggingface.co/{repo_id}')
"

echo ""
echo "============================================"
echo " Upload complete!"
echo " https://huggingface.co/${REPO_ID}"
echo "============================================"
