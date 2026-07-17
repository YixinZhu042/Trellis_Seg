#!/bin/bash
# ============================================================
# Seg VAE 多机训练启动脚本 (Multi-node training launcher)
#
# Usage:
#   bash scripts/train_seg_vae_multi_node_launcher.sh <NUM_NODES> <NODE_RANK> <MASTER_ADDR>
#
# Example (2 nodes):
#   Node 0: bash scripts/train_seg_vae_multi_node_launcher.sh 2 0 10.x.x.x
#   Node 1: bash scripts/train_seg_vae_multi_node_launcher.sh 2 1 10.x.x.x
# ============================================================

set -e

NUM_NODES=${1:-1}
NODE_RANK=${2:-0}
MASTER_ADDR=${3:-"localhost"}
MASTER_PORT=12346
NUM_GPUS=8

# ============================================================
# 配置
# ============================================================
PYTHON="python"
PROJECT_DIR="."
CONFIG="configs/seg_vae_train.json"
DATA_DIR="/path/to/data"
OUTPUT_DIR="./output/seg_vae"

# ============================================================
# NCCL 环境变量：防止卡住
# ============================================================
export NCCL_TIMEOUT=600
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

# ------------------------------------------------------------
# 快速失败 + 可观测性（避免任一 rank 出错后其余 rank 死等 10 分钟超时）
#   ASYNC_ERROR_HANDLING=1: 任一 rank 集合通信出错/超时立即 tear-down 整个 job
#   DUMP_ON_TIMEOUT=1     : 超时时 dump 调试信息
#   TRACE_BUFFER_SIZE     : 开启 FlightRecorder，超时时能打印卡住那次 collective 的栈
# ------------------------------------------------------------
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}

# ------------------------------------------------------------
# 调试模式：DEBUG=1 时把 hang 定位到具体 CUDA 算子（会显著变慢，仅排查坏样本时用）
#   用法: DEBUG=1 bash scripts/train_seg_vae_multi_node_launcher.sh
# ------------------------------------------------------------
if [ "${DEBUG:-0}" = "1" ]; then
    echo "[DEBUG] CUDA_LAUNCH_BLOCKING enabled — kernel errors will surface synchronously (slow)."
    export CUDA_LAUNCH_BLOCKING=1
    export NCCL_DEBUG=INFO
fi

# ============================================================
echo "============================================"
echo " Seg VAE Fine-tuning (Multi-Node via multi-node launcher)"
echo "  Nodes: ${NUM_NODES}, Rank: ${NODE_RANK}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs/node: ${NUM_GPUS}"
echo "  Config: ${CONFIG}"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

cd ${PROJECT_DIR}

# 日志落盘：stdout/stderr 同时写入 output_dir（这次卡死的教训——报错只打到了终端，没留文件）
mkdir -p ${OUTPUT_DIR}
LOG_FILE="${OUTPUT_DIR}/full_node${NODE_RANK}.log"
echo "  Full log -> ${LOG_FILE}"
# 让 tee 之后仍以 python 的退出码为准，便于外层判断成败
set -o pipefail

# 检查 master 可达性
if [ "${NODE_RANK}" != "0" ]; then
    echo "Checking master connectivity..."
    for i in $(seq 1 30); do
        if timeout 2 bash -c "echo > /dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then
            echo "  Master reachable."
            break
        fi
        echo "  Waiting for master... (${i}/30)"
        sleep 2
    done
fi

${PYTHON} train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --data_dir ${DATA_DIR} \
    --num_gpus ${NUM_GPUS} \
    --num_nodes ${NUM_NODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --auto_retry 0 2>&1 | tee "${LOG_FILE}"
