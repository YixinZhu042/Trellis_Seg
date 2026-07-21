#!/bin/bash
# ============================================================
# 多机训练启动脚本 (Multi-node training launcher)
#
# Usage:
#   bash scripts/train_seg_dit_multi_node_launcher.sh <NUM_NODES> <NODE_RANK> <MASTER_ADDR>
#
# Example (2 nodes):
#   Node 0: bash scripts/train_seg_dit_multi_node_launcher.sh 2 0 10.x.x.x
#   Node 1: bash scripts/train_seg_dit_multi_node_launcher.sh 2 1 10.x.x.x
# ============================================================

set -e

NUM_NODES=${1:-1}
NODE_RANK=${2:-0}
MASTER_ADDR=${3:-"localhost"}
MASTER_PORT=12345
NUM_GPUS=8

# ============================================================
# 配置
# ============================================================
PYTHON="python"
PROJECT_DIR="."
CONFIG="configs/slat_flow_seg_dit_1_3B_512_bf16.json"
DATA_DIR="/path/to/data"
OUTPUT_DIR="./output/seg_dit"

# ============================================================
# 清理残留进程
# ============================================================
if ps -ef | grep "train.py --config" | grep -v grep > /dev/null 2>&1; then
    echo "[Warn] Old train.py detected, killing..."
    pkill -9 -f "train.py --config" 2>/dev/null || true
    sleep 3
fi
fuser -k ${MASTER_PORT}/tcp 2>/dev/null || true
sleep 1

# ============================================================
# NCCL 环境变量 (InfiniBand cluster)
# ============================================================
export NCCL_NVLS_ENABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_TOPO_AFFINITY=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0

export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=600
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
echo "============================================"
echo " Seg DIT Training (Multi-Node via multi-node launcher)"
echo "  Nodes: ${NUM_NODES}, Rank: ${NODE_RANK}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs/node: ${NUM_GPUS}"
echo "  Config: ${CONFIG}"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

cd ${PROJECT_DIR}

# 非 master 节点等待 master 就绪
if [ "${NODE_RANK}" != "0" ]; then
    echo "Checking master connectivity..."
    for i in $(seq 1 60); do
        if timeout 2 bash -c "echo > /dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then
            echo "  Master reachable."
            break
        fi
        echo "  Waiting for master... (${i}/60)"
        sleep 2
    done
fi

LOG_FILE="${OUTPUT_DIR}/log_node${NODE_RANK}.txt"
mkdir -p ${OUTPUT_DIR}

${PYTHON} train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --data_dir ${DATA_DIR} \
    --num_gpus ${NUM_GPUS} \
    --num_nodes ${NUM_NODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --auto_retry 0 \
    2>&1 | tee ${LOG_FILE}
