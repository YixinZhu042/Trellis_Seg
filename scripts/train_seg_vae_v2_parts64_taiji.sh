#!/usr/bin/env bash
# New 64-class experiment. Resume only after this experiment has its own checkpoint.
set -euo pipefail
export VAE_V2_PROJECT=${VAE_V2_PROJECT:-/path/to/Trellis_Seg}
export VAE_V2_CONFIG=${VAE_V2_CONFIG:-configs/seg_vae_v2_parts64.json}
export VAE_V2_LOG_DIR=${VAE_V2_LOG_DIR:-${VAE_V2_PROJECT}/logs/seg_vae_v2_parts64}
if [[ $# -ne 3 ]]; then echo 'Usage: train_seg_vae_v2_parts64_taiji.sh HOST_NUM NODE_RANK MASTER_ADDR' >&2; exit 2; fi
if [[ ! "$2" =~ ^(0|[1-9][0-9]*)$ ]]; then exit 2; fi
if [[ "${VAE_V2_DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "$VAE_V2_LOG_DIR"
    exec > >(tee -a "$VAE_V2_LOG_DIR/launcher_node${2}.log") 2>&1
    printf '[%s] parts64 / latent32 training node %s\n' "$(date '+%F %T')" "$2"
fi
exec bash "$VAE_V2_PROJECT/scripts/train_seg_vae_v2_taiji.sh" "$@"
