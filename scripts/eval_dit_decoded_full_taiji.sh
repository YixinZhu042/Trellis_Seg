#!/usr/bin/env bash
# Full 5-node validation selection followed by paired test evaluation.
# TAIJI appends: HOST_NUM NODE_RANK MASTER_ADDR.
set -euo pipefail
if [[ $# -ne 3 ]]; then echo 'Usage: bash scripts/eval_dit_decoded_full_taiji.sh HOST_NUM NODE_RANK MASTER_ADDR' >&2; exit 2; fi
NODES=$1; NODE_RANK=$2; MASTER_ADDR=$3
[[ "$NODES" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid HOST_NUM' >&2; exit 2; }
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] && (( NODE_RANK < NODES )) || { echo 'Invalid NODE_RANK' >&2; exit 2; }
PROJECT=${SEG_DIT_PROJECT_DIR:-/path/to/Trellis_Seg}
PYTHON=${SEG_DIT_PYTHON:-/path/to/trellis-env/bin/python}
CONFIG=${SEG_DIT_CONFIG:-configs/dit_decoded_aux_full_v1.json}
GPUS=${SEG_DIT_GPUS_PER_NODE:-8}
ROOT=${SEG_DIT_EVAL_ROOT:-${PROJECT}/eval/dit_decoded_full_v1}
BASE_PORT=${SEG_DIT_EVAL_BASE_PORT:-12655}
WORKER_TIMEOUT=${SEG_DIT_EVAL_WORKER_TIMEOUT:-2400}
WAIT_TIMEOUT=${SEG_DIT_EVAL_WAIT_TIMEOUT:-3600}
mkdir -p "$ROOT/logs"
cd "$PROJECT"
exec > >(tee -a "$ROOT/logs/launcher_node${NODE_RANK}.log") 2>&1

export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_SL=${NCCL_IB_SL:-3}
export NCCL_CHECK_DISABLE=${NCCL_CHECK_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-bond1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}
export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
export NCCL_IB_TC=${NCCL_IB_TC:-160}
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

run_eval() {
    local name=$1 split=$2 port=$3 checkpoint=$4 seeds=$5
    local output="$ROOT/$split/$name"
    if [[ -f "$output/metrics.json" ]]; then echo "Skip completed $split/$name"; return; fi
    mkdir -p "$output/logs"
    args=(scripts/evaluate_dit_decoded.py --config "$CONFIG" --output_dir "$output"
          --split "$split" --seeds $seeds --no_save_predictions --resume --no_merge
          --independent_worker --conv_algorithm masked_implicit_gemm_splitk --watchdog_seconds 600)
    if [[ -n "$checkpoint" ]]; then args+=(--checkpoint "$checkpoint"); fi
    echo "[$(date '+%F %T')] $split/$name independent workers checkpoint=${checkpoint:-pretrained}"
    if [[ "${SEG_DIT_DRY_RUN:-0}" == 1 ]]; then
        for local_rank in $(seq 0 $((GPUS-1))); do
            global_rank=$((NODE_RANK * GPUS + local_rank))
            printf 'CUDA_VISIBLE_DEVICES=%q timeout %q ' "$local_rank" "$WORKER_TIMEOUT"
            printf '%q ' "$PYTHON" "${args[@]}" --device cuda:0 \
                --worker_rank "$global_rank" --worker_world_size "$((NODES*GPUS))"
            printf '\n'
        done
        return
    fi
    rm -f "$output/node${NODE_RANK}.done"
    pids=()
    for local_rank in $(seq 0 $((GPUS-1))); do
        global_rank=$((NODE_RANK * GPUS + local_rank))
        CUDA_VISIBLE_DEVICES=$local_rank timeout --signal=TERM --kill-after=60 "$WORKER_TIMEOUT" \
            "$PYTHON" "${args[@]}" --device cuda:0 \
            --worker_rank "$global_rank" --worker_world_size "$((NODES*GPUS))" \
            > "$output/logs/worker_rank$(printf '%04d' "$global_rank").log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    if (( status != 0 )); then
        echo "One or more local evaluation workers failed or exceeded ${WORKER_TIMEOUT}s; inspect $output/logs" >&2
        exit 1
    fi
    marker="$output/node${NODE_RANK}.done"
    printf '%s\n' "$(date '+%F %T')" > "$marker.tmp"
    mv "$marker.tmp" "$marker"
    deadline=$((SECONDS + WAIT_TIMEOUT))
    if (( NODE_RANK == 0 )); then
        while (( SECONDS < deadline )); do
            count=$(find "$output" -maxdepth 1 -name 'node*.done' -type f | wc -l)
            (( count == NODES )) && break
            sleep 2
        done
        count=$(find "$output" -maxdepth 1 -name 'node*.done' -type f | wc -l)
        (( count == NODES )) || { echo "Timed out waiting for all node markers: $count/$NODES" >&2; exit 1; }
        merge_args=(scripts/evaluate_dit_decoded.py --config "$CONFIG" --output_dir "$output"
                    --split "$split" --seeds $seeds --merge_only --worker_rank 0
                    --worker_world_size "$((NODES*GPUS))" --no_save_predictions)
        if [[ -n "$checkpoint" ]]; then merge_args+=(--checkpoint "$checkpoint"); fi
        "$PYTHON" "${merge_args[@]}"
    else
        while (( SECONDS < deadline )); do [[ -f "$output/metrics.json" ]] && break; sleep 2; done
        [[ -f "$output/metrics.json" ]] || { echo "Timed out waiting for merged metrics: $output" >&2; exit 1; }
    fi
}

EXP=${SEG_DIT_CKPT_DIR:-/path/to/legacy_experiments}
names=(pretrained regular_500 regular_1000 regular_1095 ema_500 ema_1000 ema_1095)
checkpoints=("" "$EXP/dit_step000500.pt" "$EXP/dit_step001000.pt" "$EXP/dit_step001095.pt"
             "$EXP/dit_ema_step000500.pt" "$EXP/dit_ema_step001000.pt" "$EXP/dit_ema_step001095.pt")
for i in "${!names[@]}"; do run_eval "${names[$i]}" validation "$((BASE_PORT+i))" "${checkpoints[$i]}" '42'; done

if [[ "${SEG_DIT_DRY_RUN:-0}" == 1 ]]; then
    echo 'Dry run stops before validation selection and test evaluation.'
    exit 0
fi

if (( NODE_RANK == 0 )); then
    "$PYTHON" scripts/select_dit_decoded_full_checkpoint.py \
        --validation_dir "$ROOT/validation" --output "$ROOT/selection.json"
fi
for attempt in $(seq 1 600); do [[ -f "$ROOT/selection.json" ]] && break; sleep 2; done
[[ -f "$ROOT/selection.json" ]] || { echo 'Timed out waiting for validation selection' >&2; exit 1; }
selected=$($PYTHON -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_continuation"]["checkpoint"])' "$ROOT/selection.json")
run_eval pretrained test "$((BASE_PORT+7))" '' '42 123 2026'
run_eval selected test "$((BASE_PORT+8))" "$selected" '42 123 2026'

if (( NODE_RANK == 0 )); then
    "$PYTHON" scripts/summarize_dit_decoded_full_eval.py --root "$ROOT"
fi
echo "[$(date '+%F %T')] evaluation complete on node_rank=$NODE_RANK"
