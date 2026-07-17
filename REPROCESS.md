# 重新预处理 & 训练流程

本次修改了颜色 palette（更高对比度）和 mv_seg 编码方式（逐 view VAE encode at 512），需要重跑预处理和训练。

## 环境

```bash
export PYTHON="python"
export PROJECT_DIR="."
export DATA_DIR="/path/to/data"
cd ${PROJECT_DIR}
```

## Step 1: 渲染 seg views（不需要重跑）

seg_views 保存的是 integer label maps，与 palette 无关，无需重跑。

## Step 2: 重新编码 mv_seg_latent（逐 view VAE encode at 512）

新方案：每个 view 独立编码（前景 pixels 量化到 512³ → tex_encoder → latent tokens on 32³）。6 个 view 的 latent tokens 合并去重。

```bash
# 删除旧 mv_seg_latent
rm -rf ${DATA_DIR}/mv_seg_latent

# 多卡并行（8卡，每卡独立处理一批样本）
for RANK in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${RANK} ${PYTHON} preprocess/step2_encode_mv_seg_latent.py \
        --data_dir ${DATA_DIR} \
        --num_classes 20 \
        --rank ${RANK} \
        --world_size 8 &
done
wait
echo "Step 2 done."
```

输出格式：`coords (M, 3) uint8 [0, 31]` + `feats (M, 32) float32`，平均 ~1000-1500 tokens/样本。

## Step 3: shape_latent（不需要重跑）

shape_latent 与 palette 无关，无需重跑。

## Step 4: 重新编码 seg_latent（新 palette）

```bash
# 删除旧 seg_latent
rm -rf ${DATA_DIR}/seg_latent

# 多卡并行（8卡示例）
for RANK in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${RANK} ${PYTHON} preprocess/step4_encode_seg_latent.py \
        --data_dir ${DATA_DIR} \
        --resolution 512 \
        --num_classes 20 \
        --rank ${RANK} \
        --world_size 8 &
done
wait
echo "Step 4 done."
```

## Step 5: 重算 normalization + metadata

```bash
${PYTHON} scripts/compute_normalization.py \
    --data_dir ${DATA_DIR} \
    --config configs/slat_flow_seg_dit_1_3B_512_bf16.json
```

这会自动更新 config 中的 `normalization`, `mv_cond_normalization`, `shape_cond_normalization` 和生成 `metadata.csv`。

## Step 6: 训练

旧 checkpoint 不兼容（palette 变了，latent 分布完全不同），需要从头训练（但仍从 texture DIT finetune）。

```bash
# 单机 8 卡
${PYTHON} train.py \
    --config configs/slat_flow_seg_dit_1_3B_512_bf16.json \
    --output_dir ./output/seg_dit \
    --data_dir ${DATA_DIR} \
    --num_gpus 8 \
    --auto_retry 3

# 多机（via cluster scheduler）
# train:bash ./scripts/train_seg_dit_multi_node_launcher.sh
# 注意修改 OUTPUT_DIR 为新路径，避免覆盖旧实验
```

## Step 7: 测试推理

```bash
${PYTHON} scripts/test_inference.py \
    --data_dir ${DATA_DIR} \
    --ckpt <新checkpoint路径> \
    --num_samples 4 \
    --decode \
    --mesh_dir /path/to/PartNeXt_mesh/glbs
```

## 本次变更摘要

| 项目 | 旧 | 新 |
|------|----|----|
| 颜色 palette | HSV 等距 20 色（红橙黄难分） | 高对比度 20 色（红绿蓝黄紫橙青...） |
| mv_seg 编码 | 6 views 合并到 64³ → 一次 VAE encode → 4³ latent (~43 tokens) | 逐 view 独立 VAE encode at 512 → 32³ latent → 合并去重 (~1000-1500 tokens) |
| mv_seg latent space | 4³ grid, 坐标 [0, 3] | 32³ grid, 坐标 [0, 31]（与 seg/shape latent 同空间） |
| step4 num_classes | 可变（per instance） | 固定 20 |
| DINOv3 cond image | 训练时 fallback 到 colorized seg view | 统一用 colorized seg view |

## 预计耗时

- Step 2（8卡 GPU）：~2-3 小时（逐 view VAE encode）
- Step 4（8卡）：~1-2 小时
- Step 5：~10 分钟
- 训练 50K steps（8×A100）：~24 小时
