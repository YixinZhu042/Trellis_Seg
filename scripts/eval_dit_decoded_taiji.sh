#!/usr/bin/env bash
# Run distributed read-only evaluation after training. TAIJI appends 3 args.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'Usage: bash scripts/eval_dit_decoded_taiji.sh HOST_NUM NODE_RANK MASTER_ADDR' >&2
    exit 2
fi
SEG_DIT_PROJECT_DIR=${SEG_DIT_PROJECT_DIR:-/path/to/Trellis_Seg}
SEG_DIT_PYTHON=${SEG_DIT_PYTHON:-/path/to/trellis-env/bin/python}
SEG_DIT_CONFIG=${SEG_DIT_CONFIG:-configs/dit_decoded_aux_v1.json}
SEG_DIT_GPUS_PER_NODE=${SEG_DIT_GPUS_PER_NODE:-8}
SEG_DIT_EVAL_SPLIT=${SEG_DIT_EVAL_SPLIT:-validation}
SEG_DIT_EVAL_OUTPUT=${SEG_DIT_EVAL_OUTPUT:?Set SEG_DIT_EVAL_OUTPUT to a new shared output directory}
SEG_DIT_MASTER_PORT=${SEG_DIT_MASTER_PORT:-12357}
for value in "$1" "$SEG_DIT_GPUS_PER_NODE" "$SEG_DIT_MASTER_PORT"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Nodes, GPUs/node and port must be positive integers' >&2; exit 2; }
done
[[ "$2" =~ ^(0|[1-9][0-9]*)$ ]] && (( $2 < $1 )) || { echo 'Invalid NODE_RANK' >&2; exit 2; }
[[ -n "$3" && "$3" != -* ]] || { echo 'Invalid MASTER_ADDR' >&2; exit 2; }
(( SEG_DIT_MASTER_PORT <= 65535 )) || { echo 'Invalid port' >&2; exit 2; }
case "$SEG_DIT_EVAL_SPLIT" in validation|test) ;; *) echo 'Invalid evaluation split' >&2; exit 2;; esac
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_SL=${NCCL_IB_SL:-3}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
cd "$SEG_DIT_PROJECT_DIR"
args=(scripts/evaluate_dit_decoded.py --config "$SEG_DIT_CONFIG"
    --output_dir "$SEG_DIT_EVAL_OUTPUT" --split "$SEG_DIT_EVAL_SPLIT")
if [[ -n "${SEG_DIT_EVAL_CHECKPOINT:-}" ]]; then
    args+=(--checkpoint "$SEG_DIT_EVAL_CHECKPOINT")
fi
command=("$SEG_DIT_PYTHON" -m torch.distributed.run --nnodes "$1" --node_rank "$2"
    --nproc_per_node "$SEG_DIT_GPUS_PER_NODE" --master_addr "$3" --master_port "$SEG_DIT_MASTER_PORT"
    --max_restarts 0 "${args[@]}")
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${SEG_DIT_DRY_RUN:-0}" == 1 ]]; then exit 0; fi
exec "${command[@]}"
