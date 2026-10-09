#!/usr/bin/env bash
# Use --index_annotations once, then launch all TAIJI ranks; merge after completion.
set -euo pipefail
export VAE_V2_PROJECT=${VAE_V2_PROJECT:-/path/to/Trellis_Seg}
export VAE_V2_CONFIG=${VAE_V2_CONFIG:-configs/seg_vae_v2_parts64.json}
exec bash "$VAE_V2_PROJECT/scripts/prepare_seg_vae_v2_taiji.sh" "$@"
