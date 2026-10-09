#!/usr/bin/env bash
# TAIJI appends: <HOST_NUM> <NODE_RANK> <MASTER_ADDR>
# All nodes must use identical SEG_DIT_* experiment settings.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/train_dit_decoded_taiji.sh HOST_NUM NODE_RANK MASTER_ADDR" >&2
    exit 2
fi
SEG_DIT_NODES=$1
SEG_DIT_NODE_RANK=$2
SEG_DIT_MASTER_ADDR=$3
for value in "$SEG_DIT_NODES" "$SEG_DIT_NODE_RANK"; do
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Node arguments must be decimal integers' >&2; exit 2; }
done
if (( SEG_DIT_NODES < 1 || SEG_DIT_NODE_RANK >= SEG_DIT_NODES )); then
    echo 'Require HOST_NUM >= 1 and 0 <= NODE_RANK < HOST_NUM' >&2
    exit 2
fi
[[ -n "$SEG_DIT_MASTER_ADDR" && "$SEG_DIT_MASTER_ADDR" != -* ]] || { echo 'Invalid MASTER_ADDR' >&2; exit 2; }

SEG_DIT_PROJECT_DIR=${SEG_DIT_PROJECT_DIR:-/path/to/Trellis_Seg}
SEG_DIT_PYTHON=${SEG_DIT_PYTHON:-/path/to/trellis-env/bin/python}
SEG_DIT_CONFIG=${SEG_DIT_CONFIG:-configs/dit_decoded_aux_full_v1.json}
SEG_DIT_MODE=${SEG_DIT_MODE:-decoded}
case "$SEG_DIT_MODE" in
    decoded) SEG_DIT_DEFAULT_PORT=12555 ;;
    flow) SEG_DIT_DEFAULT_PORT=12556 ;;
    *) echo 'SEG_DIT_MODE must be decoded or flow' >&2; exit 2 ;;
esac
SEG_DIT_GPUS_PER_NODE=${SEG_DIT_GPUS_PER_NODE:-8}
SEG_DIT_STEPS=${SEG_DIT_STEPS:-3000}
# One object per microbatch. Effective batch = nodes * GPUs/node * accumulate.
SEG_DIT_ACCUMULATE=${SEG_DIT_ACCUMULATE:-1}
SEG_DIT_MASTER_PORT=${SEG_DIT_MASTER_PORT:-$SEG_DIT_DEFAULT_PORT}
SEG_DIT_TIMEOUT=${SEG_DIT_TIMEOUT:-1800}
SEG_DIT_RUN_NAME=${SEG_DIT_RUN_NAME:-dit_${SEG_DIT_MODE}_full_v1}
SEG_DIT_OUTPUT_DIR=${SEG_DIT_OUTPUT_DIR:-${SEG_DIT_PROJECT_DIR}/experiments/${SEG_DIT_RUN_NAME}}
SEG_DIT_LOG_DIR=${SEG_DIT_LOG_DIR:-${SEG_DIT_PROJECT_DIR}/logs/${SEG_DIT_RUN_NAME}}
SEG_DIT_RESUME=${SEG_DIT_RESUME:-}
SEG_DIT_DRY_RUN=${SEG_DIT_DRY_RUN:-0}
SEG_DIT_CHECK_ONLY=${SEG_DIT_CHECK_ONLY:-0}
SEG_DIT_EPOCHS=${SEG_DIT_EPOCHS:-}
for value in "$SEG_DIT_GPUS_PER_NODE" "$SEG_DIT_STEPS" "$SEG_DIT_ACCUMULATE" "$SEG_DIT_MASTER_PORT" "$SEG_DIT_TIMEOUT"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'GPU count, steps, accumulation, port and timeout must be positive integers' >&2; exit 2; }
done
(( SEG_DIT_MASTER_PORT <= 65535 )) || { echo 'Invalid port' >&2; exit 2; }

cd "$SEG_DIT_PROJECT_DIR"
mkdir -p "$SEG_DIT_LOG_DIR"
# Capture launcher checks and rendezvous errors as well as torchrun output. The
# previous pipe started too late to explain a worker that never reached torchrun.
exec > >(tee -a "$SEG_DIT_LOG_DIR/launcher_node${SEG_DIT_NODE_RANK}.log") 2>&1
printf '[%s] launcher start: host=%s nodes=%s node_rank=%s master=%s:%s\n' \
    "$(date '+%F %T')" "$(hostname)" "$SEG_DIT_NODES" "$SEG_DIT_NODE_RANK" \
    "$SEG_DIT_MASTER_ADDR" "$SEG_DIT_MASTER_PORT"

# Reusing a port from a still-running torchrun can mix two jobs into one static
# rendezvous and produce duplicate global ranks. Fail before loading data.
if (( SEG_DIT_NODE_RANK == 0 )) && [[ "$SEG_DIT_DRY_RUN" != 1 ]]; then
    "$SEG_DIT_PYTHON" - "$SEG_DIT_MASTER_PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(('0.0.0.0', port))
except OSError as exc:
    raise SystemExit(f'Master port {port} is already occupied; stop the stale TAIJI job or choose SEG_DIT_MASTER_PORT: {exc}')
finally:
    sock.close()
PY
fi

if [[ -n "$SEG_DIT_EPOCHS" ]]; then
    [[ "$SEG_DIT_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'SEG_DIT_EPOCHS must be a positive integer' >&2; exit 2; }
    train_count=$($SEG_DIT_PYTHON - "$SEG_DIT_CONFIG" <<'PY'
import json,sys
from pathlib import Path
config=json.load(open(sys.argv[1]));data=Path(config['data_dir']);data=data if data.is_absolute() else Path.cwd()/data
selection=json.load(open(data/'selection.json'));print(selection['split_counts']['train'])
PY
)
    global_batch=$(( SEG_DIT_NODES * SEG_DIT_GPUS_PER_NODE * SEG_DIT_ACCUMULATE ))
    SEG_DIT_STEPS=$(( (train_count * SEG_DIT_EPOCHS + global_batch - 1) / global_batch ))
fi

# Defaults match this repository's H20/RDMA deployment. Every value can be
# overridden before launch (e.g. NCCL_IB_DISABLE=1 for a TCP-only environment).
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_SL=${NCCL_IB_SL:-3}
export NCCL_CHECK_DISABLE=${NCCL_CHECK_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-bond1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}
export NCCL_LL_THRESHOLD=${NCCL_LL_THRESHOLD:-16384}
export NCCL_IB_CUDA_SUPPORT=${NCCL_IB_CUDA_SUPPORT:-1}
export NCCL_TOPO_AFFINITY=${NCCL_TOPO_AFFINITY:-0}
export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
export SHARP_COLL_ENABLE_SAT=${SHARP_COLL_ENABLE_SAT:-0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
export NCCL_IB_TC=${NCCL_IB_TC:-160}
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-600}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-1}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

train_args=(scripts/train_dit_decoded.py
    --config "$SEG_DIT_CONFIG" --output_dir "$SEG_DIT_OUTPUT_DIR"
    --mode "$SEG_DIT_MODE" --steps "$SEG_DIT_STEPS" --accumulate "$SEG_DIT_ACCUMULATE"
    --distributed_timeout "$SEG_DIT_TIMEOUT")
if [[ -n "$SEG_DIT_RESUME" ]]; then
    train_args+=(--resume "$SEG_DIT_RESUME")
fi
command=("$SEG_DIT_PYTHON" -m torch.distributed.run
    --nnodes "$SEG_DIT_NODES" --node_rank "$SEG_DIT_NODE_RANK"
    --nproc_per_node "$SEG_DIT_GPUS_PER_NODE"
    --master_addr "$SEG_DIT_MASTER_ADDR" --master_port "$SEG_DIT_MASTER_PORT"
    --max_restarts 0 "${train_args[@]}")

printf 'mode=%s nodes=%s rank=%s GPUs/node=%s accumulate/GPU=%s global_batch=%s steps=%s\n' \
    "$SEG_DIT_MODE" "$SEG_DIT_NODES" "$SEG_DIT_NODE_RANK" "$SEG_DIT_GPUS_PER_NODE" \
    "$SEG_DIT_ACCUMULATE" "$(( SEG_DIT_NODES * SEG_DIT_GPUS_PER_NODE * SEG_DIT_ACCUMULATE ))" "$SEG_DIT_STEPS"
printf 'output=%s\n' "$SEG_DIT_OUTPUT_DIR"
printf 'command: '
printf '%q ' "${command[@]}"
printf '\n'

if [[ "$SEG_DIT_DRY_RUN" == 1 ]]; then
    exit 0
fi

# Training must not start from an incomplete preparation. This check is cheap
# and read-only; every node executes it before rendezvous.
SEG_DIT_DATA_DIR=$($SEG_DIT_PYTHON - "$SEG_DIT_CONFIG" <<'PY'
import json,sys
from pathlib import Path
d=Path(json.load(open(sys.argv[1]))['data_dir']);print((d if d.is_absolute() else Path.cwd()/d).resolve())
PY
)
"$SEG_DIT_PYTHON" scripts/prepare_dit_supervision.py --check --require_prepared \
    --output_dir "$SEG_DIT_DATA_DIR"
printf '[%s] prepared-data check passed on node_rank=%s\n' "$(date '+%F %T')" "$SEG_DIT_NODE_RANK"

# Read-only input validation runs on every node before torchrun rendezvous.
# It does not construct a process group or load model weights.
WORLD_SIZE=$(( SEG_DIT_NODES * SEG_DIT_GPUS_PER_NODE )) RANK=0 LOCAL_RANK=0 \
    "$SEG_DIT_PYTHON" "${train_args[@]}" --check_only
printf '[%s] training input check passed on node_rank=%s; entering torchrun\n' \
    "$(date '+%F %T')" "$SEG_DIT_NODE_RANK"
if [[ "$SEG_DIT_CHECK_ONLY" == 1 ]]; then
    exit 0
fi

# Logs live outside OUTPUT_DIR so rank zero can safely require an empty run
# directory. Do not kill unrelated train.py processes or occupied ports.
"${command[@]}"
