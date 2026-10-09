# Seg VAE v2：几何约束的多点表面解码

各项损失的通俗解释、具体公式、数字例子和日志读法，见
[VAE v2 损失函数详解](seg_vae_v2_losses_explained.md)。

当前修正原始标签的 64 类版本、128/128 留出集及三机启动命令，见
[parts64 训练指南](seg_vae_v2_parts64.md)。本页其余部分保留原 20 类实验说明。

本实现新增独立的 `vae_v2/` 包与训练入口，旧 VAE/DiT 推理路径不自动切换。
默认初始化自旧 encoder/decoder step 1000；联合训练 encoder、decoder 和表面头。
只有完成独立验证与测试后，才应把新权重接入后续 DiT 流程。

所有新数据缓存、预测标签缓存和 checkpoint 统一写入：
`/path/to/trellisseg_run`。
配置 `storage_root` 控制根目录；默认子目录为 `data/seg_vae_v2`、
`checkpoints/<配置名>` 和 `predictions/<评估名称>`。训练入口会拒绝在该根目录外
写 checkpoint 或恢复缓存。代码、训练日志和 GLB/PNG 可视化仍保存在项目中。
原始已审计数据与旧初始化权重只读使用，不自动搬迁。

## 已实现的变化

| 环节 | v2 实现 |
| --- | --- |
| 数据来源 | 审计过的 `data/dit_decoded_full_v1` 原始网格、逐面标签、固定划分 |
| 表面采样 | 面积采样＋每面中心锚点；覆盖小面、背面与不可见区域 |
| 体素冲突 | 保存 20 通道标签分布作为 encoder 输入及软监督，不丢弃少数标签 |
| 标签 0 | 表示未标注；从重建监督中忽略。与标注混合的体素在监督时重新归一化 |
| 几何邻域 | 只在同一几何连通分量中查锚点；法线筛选；距离、法线、法向位移参与注意力 |
| 面内查询 | 每张面 4 个重心坐标点，聚合概率得到面标签；不再只查询面中心 |
| 表面头 | 解码器最后一层 64 维特征＋查询点位置、法线、fallback 标志，残差预测 20 类 |
| 小部件 | 50% 面积分布＋50% 部件均衡分布采样；均衡 CE＋macro Dice |
| 连续表面 | 同 GT 部件的邻接面使用 JS 一致性，法线变化处减弱；不同标签绝不施加平滑 |
| 分离约束 | 除共享边外，加入几何上接近的跨连通分量面配对，弥补无跨部件共享边的模型 |
| 投影监督 | 面像素计数加权 CE，通过采样概率修正，形成采样估计；不是每步全像素精确求和 |
| KL | 前 1000 步 posterior mean 的确定性重建；之后采样 posterior，2000 步渐增 KL 到 1e-5 |
| 选型 | 固定 validation 子集，对比普通权重与 EMA 0.999；先表面 mIoU，再小部件召回，再孤立错误面积 |

几何采样、连通分量、法线筛选与表面头输入均不读取 GT 标签。GT 只用于 encoder
重建输入及训练监督/训练采样。解码已有 latent 可使用 `model.geometry_template(coords)`、
`decode_voxels(latent)`、`SurfaceQueries.build(face_ids)` 和 `decode_surface(...)`，不需要 GT。
锚点虽限制在同一几何分量，其稀疏体素特征仍可能包含同格/感受野内的多部件信息，
因此这是抑制串扰的机制，不是严格无串扰保证。若全部候选法线不合，使用同分量最近
锚点，并显式记录 fallback 比例。

表面最终输出仍是一面一标签。面内四点采样缓解中心查询敏感，但不提供面内连续纹理；
输入本身的面内语义边界、错误连接拓扑或不一致法线仍需单独处理。

## 配置与消融

默认配置：`configs/seg_vae_v2.json`。保持 32 维 latent、四次下采样（stride 16）。

- `seg_vae_v2_voxel_baseline.json`：相同新数据，仅普通 voxel CE＋KL；固定几何插值用于评估。
- `seg_vae_v2_loss_only.json`：加入新任务损失，但无可学习表面头。
- `seg_vae_v2.json`：可学习表面头＋全部任务损失。
- `seg_vae_v2_no_kl.json`：确定性 AE 对照，默认 10000 步内不采样 posterior。
- `seg_vae_v2_latent64.json`：64 维 latent；重建不兼容的 latent 投影层，其余层从 v1 初始化。
- `seg_vae_v2_stride8.json`：减少一级空间下采样；骨干从随机初始化，**不是与预训练默认组完全匹配的容量对照**。

先比较同样初始化、数据和采样协议的前三组，再决定容量实验。权重默认值是起点，
不是经过超参搜索的最优值；不要根据这把已反复检查的椅子挑选最终权重。

默认骨干 LR=1e-5、表面头 LR=1e-4、每 GPU 每 microbatch 一个对象。
有效 batch 为 `world_size × accumulate`。默认 `query_faces=2048`、`pair_count=1024`。
首次 GPU 检查发现现有 `flex_gemm explicit_gemm` 在 BF16 稀疏反向有 dtype 不匹配；
实现已将稀疏 encoder/decoder 固定 FP32，`precision=bf16` 仅控制表面头计算。
主干启用梯度 checkpointing。

## 准备新缓存

从项目根目录执行（无需 GPU）：

```bash
python3 scripts/prepare_seg_vae_v2.py --config configs/seg_vae_v2.json --workers 16
python3 scripts/prepare_seg_vae_v2.py --config configs/seg_vae_v2.json --workers 16 --merge
```

单机也可使用 `--workers 16 --merge_after`，全量准备成功后自动并行校验合并。
默认 worker 数是 CPU 可用核数的一半、最多 16；通过 `--workers 1` 可恢复串行执行。
每个 worker 限制 BLAS/OpenMP 为一线程。清单每 128 个结果或每 5 秒原子保存一次
（`--flush_every` / `--progress_seconds`），不再每个对象重写累计清单。并行不改变采样 seed、
数据标签或最终清单顺序。完成文件先验证再复用，中断后即使少量文件尚未记入清单也可恢复。
SIGINT/SIGTERM 会停止提交新样本、完成在途样本并保存进度。文件锁阻止同一 rank 重复写入，
合并期间禁止新的预处理任务。升级前已启动的旧进程不使用该锁，需先停止旧进程再重启。

非流形网格的预算溢出修复：原 `trimesh.face_adjacency` 忽略一条边由三个及以上面
共享的情况，大量重复面会因此被误拆成孤立分量。若原计算需要的分量锚点超过体素
预算，v2 会对精确共享边的全部入射面建立确定性的线性数量连接，重新计算分量后
重试。只在原本会失败的分支启用；正常样本仍沿用原算法与缓存。不会按距离焊接
不同壳体，也不删除重复面、改变面顺序或改写 GT。修复样本记录 `topology_repair=1`
和 `legacy_component_count`；清单记录修复方法与修复前分量数。若重算后仍超预算，
仍明确报错。重复面间的 GT 冲突属于原数据问题，此修复不自行更改标注。

多机 TAIJI：

```bash
train: bash /path/to/Trellis_Seg/scripts/prepare_seg_vae_v2_taiji.sh
```

所有节点完成后，用相同节点数提交合并：

```bash
train: VAE_V2_MERGE_ONLY=1 bash /path/to/Trellis_Seg/scripts/prepare_seg_vae_v2_taiji.sh
```

`VAE_V2_PREP_WORKERS` 可设置每个节点内的并行进程数，默认 16。

各 rank 原子写样本和清单；已完整、契约一致的样本会验证并跳过。配置或源文件变化会
拒绝复用，请使用新的 `prepared_dir`。`--instance`/`--limit` 仅生成诊断缓存，不可合并
成正式训练清单。合并要求所有 22909 对象完整，保留 train 21885 / validation 512 / test 512。
训练检查 manifest、分割、样本大小及 mtime；不会静默丢样本或随机替换失败对象。

## 训练、恢复和验证

```bash
python scripts/train_seg_vae_v2.py \
  --config configs/seg_vae_v2.json --check_only

torchrun --standalone --nproc_per_node 8 scripts/train_seg_vae_v2.py \
  --config configs/seg_vae_v2.json
```

使用已安装 TRELLIS 依赖的 Python 环境。TAIJI 命令：

```bash
train: bash /path/to/Trellis_Seg/scripts/train_seg_vae_v2_taiji.sh
```

可覆盖 `VAE_V2_CONFIG`、`VAE_V2_OUTPUT`、`VAE_V2_GPUS`、`VAE_V2_PORT`、`VAE_V2_STEPS`、`VAE_V2_LOG_DIR`。
`VAE_V2_STEPS` 指总优化步数，包含已完成的步数。
恢复时加 `--resume` 或 `VAE_V2_RESUME=1`；恢复 optimizer、EMA、各 rank 的 Torch/CUDA/NumPy
随机状态，要求数据、world size 和配置一致，允许延长 `--steps`。
默认直接从指定存储目录读取 checkpoint。若需要额外副本，可将 `--resume_cache_dir`
或 `VAE_V2_RESUME_CACHE` 设为同一 `storage_root` 下的子目录，例如 `cache/resume`；
不再写 `/tmp`。文件锁避免重复复制，校验来源 size/mtime。
迁移现有 v2 缓存时保留文件 mtime；恢复允许目录变化，但数据摘要、world size 和训练
超参数仍必须一致。

本轮 5 机 × 8 卡从现有完整断点恢复、训练到总第 5000 步，可以直接在 TAIJI 向全部
5 个节点发送：

```bash
train: bash /path/to/Trellis_Seg/scripts/resume_seg_vae_v2_taiji.sh
```

TAIJI 自动追加三个拓扑参数，无需 `bash -c` 或手写 `$1`。该脚本读取指定存储盘下
`checkpoints/seg_vae_v2/trainer_latest.pt`，恢复 optimizer、EMA 和随机状态；不运行预处理。
每个节点的启动检查、集合连接和 worker 输出追加到
`logs/seg_vae_v2/launcher_resume_node0.log` 至 `launcher_resume_node4.log`，逐步指标仍写入
`train_rankXXXX.jsonl`。若需要调整总步数，设置 `VAE_V2_STEPS`。
仅检查生成命令而不启动训练：

```bash
VAE_V2_DRY_RUN=1 bash scripts/resume_seg_vae_v2_taiji.sh 5 0 node0
```

每 500 步保存普通与 EMA portable checkpoint，另保存 `trainer_latest.pt` 用于恢复。
每 500 步及结束时在固定的前 64 个 validation ID 上选型；输出 `selection.json`。
每 rank 有独立 `logs/<运行名>/train_rankXXXX.jsonl`。这是训练期快速选型，正式 test 前建议用完整 validation：

```bash
python scripts/evaluate_seg_vae_v2.py --config configs/seg_vae_v2.json \
  --checkpoint /path/to/trellisseg_run/checkpoints/seg_vae_v2/regular_step001000.pt /path/to/trellisseg_run/checkpoints/seg_vae_v2/ema_step001000.pt \
  --split validation --output_dir eval/seg_vae_v2_full_validation

python scripts/evaluate_seg_vae_v2.py --config configs/seg_vae_v2.json \
  --selection eval/seg_vae_v2_full_validation/selection.json \
  --split test --output_dir eval/seg_vae_v2_test --export
```

评估输出体素/表面/投影 mIoU、小部件召回、隐藏表面 mIoU、真实边界 F1、同一 GT
表面内的伪边界率、孤立错误面积、邻近部件分离率、表面 mIoU 的第 10 百分位。
没有有效边界或小部件时返回 null，而不是把这些对象算成满分。
`--export` 将逐面/体素标签保存到指定存储盘的 `predictions/` 子目录，将着色 GLB
和 GT/预测六视图对比保存到评估输出目录。

## 实现验证，不等于训练结论

CPU 回归测试：

```bash
python3 -m unittest discover -s tests -p test_vae_v2.py -v
```

真实椅子的单次更新检查：

```bash
python scripts/prepare_seg_vae_v2.py --instance 007d2949e5b0450c995c54809b43af0f
python scripts/smoke_seg_vae_v2.py \
  --sample /path/to/trellisseg_run/data/seg_vae_v2/samples/007d2949e5b0450c995c54809b43af0f.npz \
  --output_dir results/vae_v2_smoke_new
```

该诊断允许在明确指定的测试对象上做一次临时更新，仅检查计算链；不产生正式训练
checkpoint，不用于选型。正式训练严格只读取 train split。

训练器集成检查可用 `--smoke --steps 1`，先准备至少一个 train 和一个 validation
对象；它会生成带 `smoke_only=true` 的 checkpoint，正式测试入口明确拒绝这类权重。

## 与 DiT 的衔接

encoder 更新后，即使维数保持 32，latent 分布也会改变。不得直接把旧 DiT 输出
交给新 VAE 当作已兼容。新训练完成后需要重新编码 3D 目标、六视图条件并计算各自的
归一化统计，再适配/重训 DiT。若使用 64 维或 stride 8，还要更新 DiT 通道/坐标契约。
本次实现 VAE v2 训练与重建评估，不自动覆写旧 latent 缓存、旧 checkpoint 或生产推理默认值。

## 当前验证记录

- CPU 单元测试覆盖几何邻域不依赖标签、软标签碰撞、未标注忽略、多点概率聚合、
  梯度、表面关系、指标、固定采样顺序、缓存审计及迁移约束。
- 真实椅子的一次联合更新通过：encoder/decoder/表面头梯度非零且有限，峰值显存约 13.4 GB。
- 自定义 decoder 特征提取与原 decoder 输出一致；几何专用缓存解码相同 latent 的最大 logit 差为 0。
- 两卡 DDP 第 1 步、普通/EMA 保存与独立 validation 选型通过。
- 数据与约 19 GB 的验证 checkpoint 迁移到指定存储盘后，两卡成功恢复 optimizer、
  EMA 与随机状态，完成第 2 步、保存与 validation 选型；数据 manifest 摘要保持一致。
- 全量预处理与合并已完成：22909 对象、0 失败，保留 train 21885 / validation 512 / test 512。
  两个非流形预算溢出对象已修复，原 22907 份缓存未重建；训练入口文件身份检查通过。
- 未启动正式全量训练；示例重建不能作为 v2 泛化提升结论。修复与校验详情见
  `eval/seg_vae_v2_topology_repair.json`。
