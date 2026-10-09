# 使用新 VAE 缓存训练 DiT（2026-09-30）

推理参数和条件消融：[2026-10-08 验证集 20/50 步、CFG 与条件检查](dit_inference_check_20261008.md)。

训练完成后的推理验证：[2026-10-08 DiT 生成与 VAE 重建对照报告](dit_vae_v2_test_20261008.md)。

## 当前结果

全量缓存已生成并通过深度审计；正式 DiT 训练已通过 Remote API 启动。
2026-09-30 18:11（Asia/Shanghai）启动验收确认全部 24 rank 至少完成 23 步，
已读取的 loss、梯度均为有限值；所有 rank 的三项辅助损失梯度均非零且有限。
这只是启动验收，不是收敛或生成质量验收；后续进度以实时日志为准。

有效会话记录：`logs/dit_vae_v2_parts64/train/remote_20260930_180639.json`。
启动验收：`logs/dit_vae_v2_parts64/train/startup_verification.json`。

## 固定模型与数据

- VAE：验证集选出的 `seg_vae_v2_parts64_permute/regular_step009500.pt`，64 通道（含 0）、latent32。
- VAE SHA256：`0a6172b072feecc35c8d4c957c0b90eb5874f1fd5098c86bad9efdf3bcfc3fb3`。
- 原始 VAE 数据 manifest digest：`8c020edead3defedc050c6493d0432e43b2b2d94d0b21fed9d938d599ca85867`。
- 新 DiT 缓存 manifest digest：`939af815a109a4c037c24238abd1524f1a6e71f2472e07451539759f50dfd17b`。
- 22,889 个对象：train **22,633**、validation **128**、test **128**；沿用本轮 VAE 的划分。
- 20 个不可可靠解析的原始对象仍按原划分清单隔离，不新增删除、截断或按效果筛选。
- DiT：现有 `SegSLatFlowModel` 1.3B 结构（30 blocks、1536 hidden），latent 输出仍为 32 维。
- 从旧 `seg_dit_full_0717/ckpts/denoiser_ema0.9999_step0050000.pt` 初始化权重。
  新建 optimizer、EMA 和训练步数，不恢复旧 latent 空间下的 optimizer；不是从随机权重训练。
- 不沿用旧 DiT teacher replay：旧教师对应旧 VAE latent 空间，不能作为新空间的直接速度目标。
- VAE encoder、decoder、surface head 全冻结，不加入 optimizer；仅通过它们计算输入梯度。

历史初始化模型可能见过部分对象，测试集也已参与开发诊断；本轮 train/validation/test
成员互斥不等于所有历史训练和调参过程都未接触测试信息。

## 缓存是成套重新生成的

使用同一份修正后的原始面标签及 VAE 几何缓存：

1. 用新 VAE encoder 的**后验均值**编码目标 3D soft voxel labels，不采样 posterior。
2. 从缓存的 6 个 face raster 重新生成正确 64 类分割图；图像背景为 0，前景编码使用
   渲染专用 `label+1`，送 encoder 前准确减去该偏移一次。
3. 使用同一个新 VAE encoder 将每个视图映射到 512³ 盒子表面，再编码为 32 维条件 token。
   六个视图按固定顺序拼接，不把不同视角的 token 混为目标体素坐标。
4. shape latent 是标签无关的已有 shape-VAE 几何条件，按同对象的 32³ 坐标精确匹配。
   未命中位置设置为 0 并保存 `shape_valid`；归一化后再次保持未命中位置为 0，禁止最近邻填充。
5. 目标 latent、六视图 latent、shape 条件的 mean/std **仅使用训练集有效 token 重算**，
   float64 统计，不复用旧 VAE 统计、不使用验证或测试统计。

每对象新缓存含 `latent_coords/latent_feats`、`shape_feats/shape_valid`、
`mv_coords/mv_feats`、`view_offsets` 和嵌入式来源元数据。新编码文件合计
**15,771,837,637 bytes（约 15.8 GB）**，不含原有 VAE 几何/标注缓存。

辅助监督读取已经位于快速存储的 VAE 几何缓存，不使用旧 20 类 NPZ 中的标签、
旧目标 latent、旧六视图 latent 或旧面中心 KNN。

### token 容量

全量检查发现 32 个对象超过旧的 8,192-token 接受上限，最大 **11,873**，六视图最多
**6,144** 条条件 token。新实验上限为 **16,384**，不改变 DiT 的 32³ 空间分辨率。
没有删除对象，没有裁掉 token，没有重新体素采样。最长三个样本均通过真实训练压力测试。

首次预处理因旧上限暂停，原始 `plan.json` 和已完成文件保持不变，
`capacity.json` 明确记录从 8192→16384 的接受上限扩展、全量长度检查及不重新采样政策。
已完成且身份一致的文件直接复用；暂停产生的一个未封存 NPZ 保留在
`unsealed_20260930_1750/` 后重新编码，未删除用户原始文件。

### shape 条件覆盖限制

按对象平均精确匹配率 **98.69%**；474 个对象低于 90%，18 个低于 50%，2 个低于 10%，
最低为 6.42%。这些对象未剔除或填入其他位置的特征，缺失部分为掩码零值，六视图条件仍保留。
这属于后续生成质量分析需要关注的数据限制；精确坐标命中不能单独证明历史 shape
缓存语义/几何来源全部无误，也没有在本轮重新运行 shape VAE。

## 正式训练设置

配置：`configs/dit_vae_v2_parts64.json`。

- 三台 H20 节点各 8 卡：`node0`、`node1`、`node2`。
- 5,000 optimizer steps，global batch 24，单卡每次一个对象。
- LR `1e-5`，500 步 warmup，AdamW（betas 0.9/0.95，weight decay 0.01），梯度裁剪 1。
- flow matching 目标仍为 velocity；BF16 autocast，FP32 主权重。
- 六视图条件 dropout 0.1；shape 几何条件保留。
- 在条件未丢弃且 `t<=0.6` 时，将估计 clean latent 送入冻结的新 VAE decoder + surface head。
- segmentation / boundary / reprojection 权重 `0.1 / 0.02 / 0.1`，辅助损失乘
  `(1-t)^2` 及 500 步 warmup。边界项使用新表面查询上的同/异部件关系，不做标签平滑。
- 每次抽 2048 个监督面、最多 1024 个关系对；沿用几何/法线/连通区域约束的 4 点表面查询。
- EMA 0.9999；每 500 步保存普通权重、EMA 和完整 optimizer/RNG 状态。
- 每 500 步对完整 128 validation 对象分别评估普通和 EMA 权重：固定每对象噪声种子，
  从 Gaussian noise 做 20 步 Euler 生成，不用 GT latent 值初始化、不做后处理、不做标签匹配。
  依据生成结果的表面 mIoU、小部件召回和孤立错误面积选型；不使用 test 选 checkpoint。
- 首次模型保存/验证在 step 500。训练过程中 VAE 仍保持原始固定 checkpoint，不再更新。

DiT 缓存使用**原始修正编号**。本轮没有直接对 32 维 latent 通道做标签置换：latent
通道不是 64 个 part ID；若未来增加 DiT 标签增强，必须在原标签上置换并同步重新编码
目标与六视图条件。本轮仅使用已经通过置换训练的 VAE。

## 已执行的验收

- 6 对象编码/回解码：目标 latent 坐标与 label-free 几何模板一致，logit 最大误差 0。
- 三机各一卡 2 步真实训练：DiT 更新，冻结 VAE 无参数梯度；3 项辅助梯度到达 DiT。
- 最长三个训练样本（11,873 / 10,732 / 10,245 token）2 步压力训练：均通过，
  观测单进程 PyTorch 显存峰值约 37.3 GiB，不含其他进程及驱动占用。
- 从压力测试完整 checkpoint 恢复至 step 3，并完成噪声生成和表面解码评估，三台退出码 0。
  小样 checkpoint 标记 `smoke_only`，没有作为正式训练初始化或质量选型依据。
- 全量 22,889 文件深审：数组形状、有限值、坐标范围/唯一性、文件 SHA256、split、
  嵌入式计划身份、训练统计全部通过；24 shard 无失败、漏项、额外对象。
- 4 项新缓存契约测试、57 项 VAE 相关测试通过。
- 正式 24 卡训练已实际推进，所有 rank 的辅助梯度检查通过；完整初始验收见上述 JSON。

## 路径与命令

```text
/path/to/trellisseg_run/
  data/dit_vae_v2_parts64/
    plan.json
    capacity.json
    normalization.json
    manifest.json
    audit.json
    samples/*.npz
    records/*.json
  checkpoints/dit_vae_v2_parts64/
  runtime_cache/dit_vae_v2_parts64/
```

项目内日志：`logs/dit_vae_v2_parts64/prepare/` 和 `logs/dit_vae_v2_parts64/train/`。
原实验、试验记录和 checkpoint 都保留，未改已有数据或训练任务。

**当前已在训练，请只查看状态，不要重复启动：**

```bash
bash scripts/train_dit_vae_v2_remote.sh status
```

新实验启动及后续中断恢复（本实验已有 checkpoint 且没有活动任务时）：

```bash
bash scripts/train_dit_vae_v2_remote.sh launch
bash scripts/train_dit_vae_v2_remote.sh resume
```

缓存检查、正式训练输入检查：

```bash
python3 scripts/dit_vae_v2_remote.py --phase prepare --status
/path/to/trellis-env/bin/python -B scripts/train_dit_vae_v2.py --check-only
```

预处理入口：`scripts/prepare_dit_vae_v2.py`，`--plan` 只用于新目录建立计划；
`--worker 0..23` 由 Remote API 节点启动器下发，`--merge` 在全部完成后深度审计封存。
不使用 TAIJI 命令或 SSH，不修改 Remote API 看板或 WriteGuard。
