#!/usr/bin/env bash
# CPU preprocessing: independent rank per node, worker pool within each node.
set -euo pipefail
if [[ $# -ne 3 ]]; then echo 'Usage: prepare_seg_vae_v2_taiji.sh HOST_NUM NODE_RANK MASTER_ADDR' >&2; exit 2; fi
VAE_NODES=$1
VAE_RANK=$2
[[ "$VAE_NODES" =~ ^[1-9][0-9]*$ && "$VAE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || exit 2
(( VAE_RANK < VAE_NODES )) || exit 2
VAE_PROJECT=${VAE_V2_PROJECT:-/path/to/Trellis_Seg}
VAE_PYTHON=${VAE_V2_PYTHON:-/path/to/trellis-env/bin/python}
VAE_CONFIG=${VAE_V2_CONFIG:-configs/seg_vae_v2.json}
VAE_WORKERS=${VAE_V2_PREP_WORKERS:-16}
[[ "$VAE_WORKERS" =~ ^[1-9][0-9]*$ ]] || exit 2
cd "$VAE_PROJECT"
if [[ "${VAE_V2_MERGE_ONLY:-0}" == 1 ]]; then
    if (( VAE_RANK == 0 )); then
        exec "$VAE_PYTHON" scripts/prepare_seg_vae_v2.py --config "$VAE_CONFIG" --world_size "$VAE_NODES" --workers "$VAE_WORKERS" --merge
    fi
else
    exec "$VAE_PYTHON" scripts/prepare_seg_vae_v2.py --config "$VAE_CONFIG" --world_size "$VAE_NODES" --rank "$VAE_RANK" --workers "$VAE_WORKERS"
fi
