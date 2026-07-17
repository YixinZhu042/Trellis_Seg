#!/bin/bash
# 单独跑 step5：8 卡并行，每卡一个 rank，OptiX 单进程批处理
set -e
cd "$(dirname "$0")/.."

DATA_DIR="/path/to/data"
BLENDER="blender"
NUM_GPUS=8

pids=()
for rank in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$rank python preprocess/step5_render_cond_images.py \
        --data_dir ${DATA_DIR} \
        --mesh_dir /path/to/PartNeXt_mesh/glbs \
        --resolution 512 \
        --rank $rank \
        --world_size ${NUM_GPUS} \
        --renderer blender \
        --blender_path ${BLENDER} &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait $pid || fail=1; done
[ $fail -eq 0 ] || { echo "ERROR: a Step 5 rank failed"; exit 1; }
echo "Step 5 done: $(ls ${DATA_DIR}/render_cond/*.png 2>/dev/null | wc -l) images"
