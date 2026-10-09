#!/usr/bin/env bash
# Resume the existing 5-node x 8-GPU VAE v2 run to 5000 total optimizer steps.
# TAIJI appends HOST_NUM NODE_RANK MASTER_ADDR automatically.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo 'Usage: bash scripts/resume_seg_vae_v2_taiji.sh HOST_NUM NODE_RANK MASTER_ADDR' >&2
    exit 2
fi
VAE_RESUME_NODES=$1
VAE_RESUME_RANK=$2
VAE_RESUME_MASTER=$3
if [[ "$VAE_RESUME_NODES" != 5 || "${VAE_V2_GPUS:-8}" != 8 ]]; then
    echo 'This checkpoint requires the original topology: 5 nodes x 8 GPUs (40 ranks).' >&2
    exit 2
fi
if [[ ! "$VAE_RESUME_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || (( VAE_RESUME_RANK >= 5 )); then
    echo 'NODE_RANK must be 0, 1, 2, 3, or 4.' >&2
    exit 2
fi
if [[ -z "$VAE_RESUME_MASTER" || "$VAE_RESUME_MASTER" == -* ]]; then
    echo 'MASTER_ADDR must be a nonempty address.' >&2
    exit 2
fi

export VAE_V2_PROJECT=${VAE_V2_PROJECT:-/path/to/Trellis_Seg}
export VAE_V2_CONFIG=${VAE_V2_CONFIG:-configs/seg_vae_v2.json}
export VAE_V2_OUTPUT=${VAE_V2_OUTPUT:-/path/to/trellisseg_run/checkpoints/seg_vae_v2}
export VAE_V2_LOG_DIR=${VAE_V2_LOG_DIR:-${VAE_V2_PROJECT}/logs/seg_vae_v2}
export VAE_V2_STEPS=${VAE_V2_STEPS:-5000}
export VAE_V2_GPUS=8
export VAE_V2_RESUME=1

cd "$VAE_V2_PROJECT"
if [[ "${VAE_V2_DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "$VAE_V2_LOG_DIR"
    # Capture preflight, rendezvous, worker output and failures from every node.
    exec > >(tee -a "$VAE_V2_LOG_DIR/launcher_resume_node${VAE_RESUME_RANK}.log") 2>&1
    printf '[%s] VAE v2 resume: node=%s/5 master=%s:%s total_steps=%s\n' \
        "$(date '+%F %T')" "$VAE_RESUME_RANK" "$VAE_RESUME_MASTER" "${VAE_V2_PORT:-12755}" "$VAE_V2_STEPS"
    printf 'checkpoint=%s/trainer_latest.pt\n' "$VAE_V2_OUTPUT"
    if [[ ! -f "$VAE_V2_OUTPUT/trainer_latest.pt" ]]; then
        echo 'Missing trainer_latest.pt; a full optimizer/EMA/RNG checkpoint is required.' >&2
        exit 1
    fi
fi

exec bash "$VAE_V2_PROJECT/scripts/train_seg_vae_v2_taiji.sh" \
    "$VAE_RESUME_NODES" "$VAE_RESUME_RANK" "$VAE_RESUME_MASTER"
