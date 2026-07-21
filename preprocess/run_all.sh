#!/bin/bash
# ============================================================
# 数据预处理完整流程
# Step 1/5: CPU 多进程渲染
# Step 2/3/4: 8 卡 GPU 并行编码
# Step 5: CPU 渲染
# ============================================================

set -e
cd "$(dirname "$0")/.."

DATA_DIR="/path/to/data"
# 处理样本数上限。留空 = 处理全部实例（全量约 23500 个）。
# 用法：  全量 -> NUM_SAMPLES=""      ；  小规模测试 -> NUM_SAMPLES=100
NUM_SAMPLES=""
NUM_GPUS=8

# 构造 --num_samples 参数：有值才传，空则各步默认 None（处理全部）
SAMPLE_ARG=${NUM_SAMPLES:+--num_samples ${NUM_SAMPLES}}

# ============================================================
echo "============================================================"
echo " Step 1/5: 渲染 6 视角分割图 (${NUM_GPUS} GPUs 并行, nvdiffrast)"
echo " Output: ${DATA_DIR}/seg_views/{instance}.npz"
echo "============================================================"
pids=()
for rank in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$rank python preprocess/step1_render_seg_views.py \
        --arrow_dir /path/to/PartNeXt \
        --mesh_dir /path/to/PartNeXt_mesh/glbs \
        --output_dir ${DATA_DIR} \
        --resolution 512 \
        --rank $rank \
        --world_size ${NUM_GPUS} \
        ${SAMPLE_ARG} &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait $pid || fail=1; done
[ $fail -eq 0 ] || { echo "ERROR: a Step 1 rank failed"; exit 1; }
echo "  Step 1 done."

# ============================================================
echo ""
echo "============================================================"
echo " Step 2/5: 编码 mv_seg_latent (${NUM_GPUS} GPUs 并行)"
echo " Output: ${DATA_DIR}/mv_seg_latent/{instance}.npz"
echo "============================================================"
pids=()
for rank in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$rank python preprocess/step2_encode_mv_seg_latent.py \
        --data_dir ${DATA_DIR} \
        --resolution 64 \
        --num_classes 20 \
        --rank $rank \
        --world_size ${NUM_GPUS} \
        ${SAMPLE_ARG} &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait $pid || fail=1; done
[ $fail -eq 0 ] || { echo "ERROR: a Step 2 rank failed"; exit 1; }
echo "  Step 2 done."

# ============================================================
echo ""
echo "============================================================"
echo " Step 3/5: 编码 shape_latent (${NUM_GPUS} GPUs 并行, o_voxel)"
echo " Output: ${DATA_DIR}/shape_latent/{instance}.npz"
echo "============================================================"
pids=()
for rank in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$rank python preprocess/step3_encode_shape_latent.py \
        --data_dir ${DATA_DIR} \
        --mesh_dir /path/to/PartNeXt_mesh/glbs \
        --resolution 512 \
        --rank $rank \
        --world_size ${NUM_GPUS} \
        ${SAMPLE_ARG} &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait $pid || fail=1; done
[ $fail -eq 0 ] || { echo "ERROR: a Step 3 rank failed"; exit 1; }
echo "  Step 3 done."

# ============================================================
echo ""
echo "============================================================"
echo " Step 4/5: 编码 seg_latent (${NUM_GPUS} GPUs 并行, 训练目标)"
echo " Output: ${DATA_DIR}/seg_latent/{instance}.npz"
echo "============================================================"
pids=()
for rank in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$rank python preprocess/step4_encode_seg_latent.py \
        --data_dir ${DATA_DIR} \
        --resolution 512 \
        --num_classes 20 \
        --rank $rank \
        --world_size ${NUM_GPUS} \
        ${SAMPLE_ARG} &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait $pid || fail=1; done
[ $fail -eq 0 ] || { echo "ERROR: a Step 4 rank failed"; exit 1; }
echo "  Step 4 done."

# ============================================================
echo ""
echo "============================================================"
echo " Step 5/5: 渲染参考图 (Blender Eevee, 用于 DINO 条件, ${STEP5_WORKERS} 路并行)"
echo " Output: ${DATA_DIR}/render_cond/{instance}.png"
echo "============================================================"
BLENDER_PATH="blender"
# Blender 是 CPU 渲染、单实例串行会成为全量瓶颈；按 rank 分片并行。
STEP5_WORKERS=${STEP5_WORKERS:-16}
pids=()
for rank in $(seq 0 $((STEP5_WORKERS-1))); do
    python preprocess/step5_render_cond_images.py \
        --data_dir ${DATA_DIR} \
        --resolution 512 \
        --rank $rank \
        --world_size ${STEP5_WORKERS} \
        --renderer blender \
        --blender_path ${BLENDER_PATH} \
        ${SAMPLE_ARG} &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait $pid || fail=1; done
[ $fail -eq 0 ] || { echo "ERROR: a Step 5 worker failed"; exit 1; }
echo "  Step 5 done."

# ============================================================
echo ""
echo "============================================================"
echo " 验证输出"
echo "============================================================"

python -c "
import numpy as np, os
data_dir = '${DATA_DIR}'
for subdir in ['seg_views', 'mv_seg_latent', 'shape_latent', 'seg_latent']:
    d = os.path.join(data_dir, subdir)
    if not os.path.isdir(d): continue
    files = [f for f in os.listdir(d) if f.endswith('.npz')]
    if not files: continue
    f = os.path.join(d, files[0])
    data = np.load(f)
    print(f'{subdir}/ ({len(files)} files):')
    for k in data.files:
        print(f'  {k}: shape={data[k].shape}, dtype={data[k].dtype}')
    print()
"

echo "============================================================"
echo " 统计汇总"
echo "============================================================"
echo "seg_views:     $(ls ${DATA_DIR}/seg_views/*.npz 2>/dev/null | wc -l) files"
echo "mv_seg_latent: $(ls ${DATA_DIR}/mv_seg_latent/*.npz 2>/dev/null | wc -l) files"
echo "shape_latent:  $(ls ${DATA_DIR}/shape_latent/*.npz 2>/dev/null | wc -l) files"
echo "seg_latent:    $(ls ${DATA_DIR}/seg_latent/*.npz 2>/dev/null | wc -l) files"
echo "render_cond:   $(ls ${DATA_DIR}/render_cond/*.png 2>/dev/null | wc -l) files"
echo ""
echo "============================================================"
echo " 全部完成！"
echo "============================================================"
