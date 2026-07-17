# 数据预处理流程

## 概览

将 PartNeXt 数据集（mesh + 分割标注）预处理为 DIT 训练所需的 5 种数据。

```
PartNeXt (mesh + face labels)
    │
    ├─ Step 1 ─→ seg_views/          6视角分割图
    │
    ├─ Step 2 ─→ mv_seg_latent/      多视角分割 latent (Texture VAE)
    │
    ├─ Step 3 ─→ shape_latent/       几何 latent (Shape VAE, o_voxel)
    │
    ├─ Step 4 ─→ seg_latent/         3D分割 latent (训练目标 x₀)
    │
    └─ Step 5 ─→ render_cond/        参考图 (DINO cross-attention 条件)
```

## 数据源

| 来源 | 路径 |
|------|------|
| PartNeXt 标注 (arrow) | `/path/to/PartNeXt/` |
| PartNeXt 网格 (glb) | `/path/to/PartNeXt_mesh/glbs/` |
| Texture VAE Encoder | `/path/to/trellis2_ckpts/tex_enc` |
| Shape VAE Encoder | `/path/to/trellis2_ckpts/shape_enc` |

---

## Step 1: 渲染 6 视角分割图

**做什么**：将 mesh 的 per-face 分割标注渲染为 6 个正交视角的 2D 分割图（front, right, back, left, top, bottom）。

**输入**：
- PartNeXt arrow 标注文件（`masks` 字段 → per-face part labels）
- GLB 网格文件

**输出**：
```
{data_dir}/seg_views/{instance}.npz
    seg_maps: (6, H, W) int32    — 每像素 part label（0=背景）
    num_parts: int                — part 数量（不含背景）
```

**命令**：
```bash
python preprocess/step1_render_seg_views.py \
    --arrow_dir /path/to/PartNeXt \
    --mesh_dir /path/to/PartNeXt_mesh/glbs \
    --output_dir ./data \
    --resolution 512 \
    --num_samples 5        # 冒烟测试；全量去掉此参数
```

**耗时**：~10s/sample

---

## Step 2: 编码多视角分割条件 → mv_seg_latent

**做什么**：将 6 视角 seg map 通过 color encoding → 3D 坐标映射 → Texture VAE encode，得到多视角分割条件 latent。

**流程细节**：
1. 每视角 seg map `(H, W)` → color encoding → 6ch features `(H*W, 6)`
2. `box_surface_image_ids()` 将像素映射到单位立方体表面 3D 坐标
3. 量化到 `resolution` 大小的体素网格
4. 构建 SparseTensor → Texture VAE encode → 32ch latent
5. 6 视角合并 → 重叠体素均值融合

**输入**：
- `{data_dir}/seg_views/{instance}.npz`（Step 1 输出）

**输出**：
```
{data_dir}/mv_seg_latent/{instance}.npz
    coords: (M, 3) uint8     — latent 体素坐标
    feats:  (M, 32) float32  — latent 特征
```

**命令**：
```bash
python preprocess/step2_encode_mv_seg_latent.py \
    --data_dir ./data \
    --resolution 64 \
    --num_classes 20 \
    --num_samples 5
```

**模型**：Texture VAE Encoder (SparseUnetVaeEncoder, 6ch→32ch)  
**耗时**：~60s/sample（GPU）

---

## Step 3: 编码几何 → shape_latent

**做什么**：使用 o_voxel 将 mesh 转为 FlexiDualGrid 体素表示，再通过 TRELLIS.2 原版 Shape VAE 编码为几何 latent。**与 TRELLIS.2 完全一致。**

**流程细节**：
1. 加载 mesh，归一化到 [-0.5, 0.5]
2. `o_voxel.convert.mesh_to_flexible_dual_grid()` → 体素坐标 + dual vertices + intersection flags
3. 构建 `vertices` SparseTensor (3ch) 和 `intersected` SparseTensor (3ch bool)
4. `FlexiDualGridVaeEncoder(vertices, intersected)` → 32ch latent

**输入**：
- GLB 网格文件

**输出**：
```
{data_dir}/shape_latent/{instance}.npz
    coords: (M, 3) uint8     — latent 体素坐标
    feats:  (M, 32) float32  — 几何 latent 特征
```

**命令**：
```bash
python preprocess/step3_encode_shape_latent.py \
    --data_dir ./data \
    --mesh_dir /path/to/PartNeXt_mesh/glbs \
    --resolution 512 \
    --num_samples 5
```

**模型**：Shape VAE Encoder (FlexiDualGridVaeEncoder, 6ch→32ch)  
**依赖**：`o_voxel` 库  
**耗时**：~5s/sample（GPU）

**Resolution 选择**：
| Resolution | Latent Tokens | Chamfer | 建议 |
|------------|---------------|---------|------|
| 256 | 200-1100 | ~5‰ | 快速实验 |
| **512** | **900-5300** | **~5‰** | **推荐（精度=1024，token少4x）** |
| 1024 | 3700-22000 | ~5‰ | 与原版一致但很慢 |

---

## Step 4: 编码 3D 分割标签 → seg_latent（训练目标）

**做什么**：将 mesh 体素化并赋予 per-voxel 分割标签，通过 color encoding → Texture VAE 编码为 seg_latent。这是 DIT 训练的目标 x₀。

**流程细节**：
1. Mesh 表面采样 → 量化到体素网格 → majority vote 得到 per-voxel label
2. Labels → color encoding: `[R, G, B, label_norm, label_norm, 1.0]` → `*2-1` 映射到 [-1, 1]
3. 构建 SparseTensor → Texture VAE encode → 32ch latent

**输入**：
- PartNeXt arrow 标注 + GLB 网格

**输出**：
```
{data_dir}/seg_latent/{instance}.npz
    coords: (M, 3) uint8     — latent 体素坐标
    feats:  (M, 32) float32  — 分割 latent 特征
```

**命令**：
```bash
python preprocess/step4_encode_seg_latent.py \
    --data_dir ./data \
    --resolution 512 \
    --num_classes 20 \
    --num_samples 5
```

**模型**：Texture VAE Encoder (SparseUnetVaeEncoder, 6ch→32ch)  
**耗时**：~20s/sample（GPU）

**重建质量**（encode→decode 验证）：
| 类别数 | 准确率 |
|--------|--------|
| ≤5 | 99.7% |
| ≤10 | 99.4% |
| ≤20 | 96.3% |

---

## Step 5: 渲染参考图（DINO 条件）

**做什么**：从随机视角渲染 1 张 RGBA 图像，用于 DIT 训练时的 DINO cross-attention 条件。

**DINO 特征在训练时在线提取**（trainer 自动处理），无需离线预计算。

**输入**：
- GLB 网格文件

**输出**：
```
{data_dir}/render_cond/{instance}.png    — RGBA 图像 (512×512)
```

**命令**：
```bash
# Blender 高质量渲染（推荐）
python preprocess/step5_render_cond_images.py \
    --data_dir ./data \
    --resolution 512 \
    --renderer blender \
    --num_samples 5

# 无 Blender 时 fallback
python preprocess/step5_render_cond_images.py \
    --data_dir ./data \
    --renderer pyrender \
    --num_samples 5
```

**可选**：离线预提取 DINO 特征（加速训练）
```bash
python preprocess/step5_render_cond_images.py \
    --data_dir ./data \
    --renderer blender \
    --extract_dino \
    --dino_model dinov2_vitl14_reg
```

**耗时**：Blender ~5s/sample，pyrender ~1s/sample

---

## 一键运行（冒烟测试）

```bash
bash preprocess/run_all.sh
```

默认处理 100 个样本，全量运行时去掉 `--num_samples` 参数。

---

## 多卡分布式处理

Step 2/3/4 支持 `--rank` 和 `--world_size` 参数，可在多卡机器上并行：

```bash
# 4 卡并行 Step 2（每张卡处理 1/4 数据）
for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$rank python preprocess/step2_encode_mv_seg_latent.py \
        --rank $rank --world_size 4 &
done
wait

# 4 卡并行 Step 3
for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$rank python preprocess/step3_encode_shape_latent.py \
        --rank $rank --world_size 4 &
done
wait

# 4 卡并行 Step 4
for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$rank python preprocess/step4_encode_seg_latent.py \
        --rank $rank --world_size 4 &
done
wait
```

也可以跨机器：每台机器跑不同的 rank 区间，输出到同一个共享存储路径即可（所有脚本都支持断点续跑）。

---

## 最终数据目录结构

```
/path/to/data/
├── seg_views/{instance}.npz         # Step 1: (6, 512, 512) int32
├── mv_seg_latent/{instance}.npz     # Step 2: coords(M,3) + feats(M,32)
├── shape_latent/{instance}.npz      # Step 3: coords(M,3) + feats(M,32)
├── seg_latent/{instance}.npz        # Step 4: coords(M,3) + feats(M,32)
└── render_cond/{instance}.png       # Step 5: RGBA 512x512
```

---

## DIT 训练时数据使用

```
DIT 输入 (on shape_latent coords):
  ┌──────────────┬──────────────────┬─────────────────────┐
  │ noise (32ch) │ shape_latent(32) │ mv_seg_latent (32ch)│
  └──────────────┴──────────────────┴─────────────────────┘
  = 96ch total (concat on channel dim)

Cross-Attention:
  render_cond.png → DINO → feature tokens

Target:
  seg_latent (32ch) — flow matching velocity prediction
```

---

## 注意事项

1. **Step 3 的 resolution 和 Step 4 的 resolution 是独立的**：
   - Step 3 (shape): 推荐 512（几何精度足够）
   - Step 4 (seg): 推荐 256（保证 VAE 压缩比合理，token 数 ~200-1000）

2. **Step 4 体素数限制**：体素过多会导致 Texture VAE 压缩比过高、重建质量下降。脚本内置 `max_voxels=30000`。

3. **Color Encoding 的 num_classes 需要全局一致**：训练和推理时必须用相同的 `num_classes`（决定调色板），推荐 20。

4. **推理时**：输入 mesh 不需要降采样，o_voxel 会按 resolution 参数自动体素化。
