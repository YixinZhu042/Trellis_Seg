#!/usr/bin/env bash
# TAIJI appends HOST_NUM NODE_RANK MASTER_ADDR; this only launches training.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'Usage: train_seg_vae_v2_taiji.sh HOST_NUM NODE_RANK MASTER_ADDR' >&2
    exit 2
fi
VAE_NODES=$1
VAE_RANK=$2
VAE_MASTER=$3
VAE_PROJECT=${VAE_V2_PROJECT:-/path/to/Trellis_Seg}
VAE_PYTHON=${VAE_V2_PYTHON:-/path/to/trellis-env/bin/python}
VAE_CONFIG=${VAE_V2_CONFIG:-configs/seg_vae_v2.json}
VAE_OUTPUT=${VAE_V2_OUTPUT:-}
VAE_GPUS=${VAE_V2_GPUS:-8}
VAE_PORT=${VAE_V2_PORT:-12755}
VAE_STEPS=${VAE_V2_STEPS:-}
[[ "$VAE_NODES" =~ ^[1-9][0-9]*$ && "$VAE_RANK" =~ ^(0|[1-9][0-9]*)$ && "$VAE_GPUS" =~ ^[1-9][0-9]*$ && "$VAE_PORT" =~ ^[1-9][0-9]*$ ]] || exit 2
(( VAE_RANK < VAE_NODES && VAE_PORT < 65536 )) || exit 2
if [[ -n "$VAE_STEPS" && ! "$VAE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'VAE_V2_STEPS must be a positive integer (total optimizer steps).' >&2
    exit 2
fi
cd "$VAE_PROJECT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
args=(scripts/train_seg_vae_v2.py --config "$VAE_CONFIG")
if [[ -n "$VAE_OUTPUT" ]]; then args+=(--output_dir "$VAE_OUTPUT"); fi
if [[ -n "$VAE_STEPS" ]]; then args+=(--steps "$VAE_STEPS"); fi
if [[ -n "${VAE_V2_LOG_DIR:-}" ]]; then args+=(--log_dir "$VAE_V2_LOG_DIR"); fi
if [[ "${VAE_V2_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
if [[ -n "${VAE_V2_RESUME_CACHE:-}" ]]; then args+=(--resume_cache_dir "$VAE_V2_RESUME_CACHE"); fi
command=("$VAE_PYTHON" -m torch.distributed.run --nnodes "$VAE_NODES" --node_rank "$VAE_RANK"
         --nproc_per_node "$VAE_GPUS" --master_addr "$VAE_MASTER" --master_port "$VAE_PORT" --max_restarts 0 "${args[@]}")
if [[ "${VAE_V2_DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
"$VAE_PYTHON" "${args[@]}" --check_only
exec "${command[@]}"
