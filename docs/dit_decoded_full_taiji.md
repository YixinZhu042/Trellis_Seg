# 全量固定 VAE + DiT 辅助损失训练

## 已确认的数据问题

- 当前 metadata、`seg_views`、`seg_latent`、`mv_seg_latent` 均有 22,909 个对象。
- 其中 29 个对象的旧 latent 超过 8192 token。新流程会自适应降低这些对象的
  体素采样预算并重新编码，因此仍纳入训练。
- `seg_voxels` 和 `shape_latent` 各有 23,008 个文件，多出的 99 个对象不在当前 metadata 训练清单。
- 10,363 个对象存在未标注面；这些面保持模型 channel 0，并在 segmentation、
  boundary、reprojection 辅助损失中被显式忽略，不会当成一个真实部件学习。
- 旧 `step4b_save_seg_voxels.py` 使用默认 `max_parts=15`，而 `seg_views` 和
  `seg_latent` 使用 `num_classes-1=19`；层级过宽时解析器还可能返回超过
  `max_parts` 的部件，旧脚本随后会静默截断越界 ID。即使实际对象最终映射
  有时相同，这个生成契约也不可靠。因此旧 `seg_voxels` 不与 19-part 缓存
  混合作为辅助监督。脚本已改为显式传 19 上限、超限报错、禁止静默截断；
  新实验不读取旧 `seg_voxels`。
- 外部/API 分割使用 0=背景、1..19=部件；旧 wild inference 错误地再次减 1。
  训练渲染的 `+1` 文件偏移与外部 ID 约定现已分别处理并有回归测试。

全量只读审计完成后以 `eval/seg_training_data_audit.json` 为准。

## 新数据布局

新缓存写入项目盘：`data/dit_decoded_full_v1`，不覆盖旧数据盘。固定划分为：

- train 21,885
- validation 512
- test 512
- 总计 22,909；旧 latent 超限并需自适应重采样 29

每个对象从同一面标签与同一次表面采样生成体素、目标 latent、shape 条件、
六视图条件、面到体素 KNN、边界和重投影计数。清单只有 ID/类别/路径/split，
各 rank 只读取自己负责的 Arrow 标注。文件采用临时文件原子替换，可断点续跑。
若新对齐采样得到的 latent 超过 8192 token，预处理会确定性降低表面体素采样
预算并重试，避免因轻微采样差异让已选对象在准备阶段被静默遗漏。

## 第一步：多机多卡预处理

在 TAIJI WebUI 发送：

```bash
train: bash /path/to/Trellis_Seg/scripts/prepare_dit_supervision_taiji.sh
```

建议使用两机或更多机器、每机 8 卡。所有 rank 完成后，再单独执行合并检查：

```bash
train: SEG_PREP_MERGE_ONLY=1 bash /path/to/Trellis_Seg/scripts/prepare_dit_supervision_taiji.sh
```

合并成功会生成 `data/dit_decoded_full_v1/prepared_manifest.json`，其中
`prepared=expected=22909`、failures/missing/extra/wrong_split/orphan_files 均为空，才可训练。
`missing_ranks/extra_ranks/duplicate_ranks` 也必须全部为空。
同一命令随后逐文件深度审计并生成 `prepared_audit.json`；训练脚本会核对审计
中的清单、汇总文件和所有样本的 size/mtime 摘要，缓存被修改后必须重新审计。
重复运行预处理会验证并跳过已完整文件。若要强制重建某对象，应先把该对象
文件移动到备份目录，再重跑；脚本不会覆盖完整文件。

## 第二步：全量辅助 loss 训练

默认启动命令：

```bash
train: SEG_DIT_EPOCHS=2 bash /path/to/Trellis_Seg/scripts/train_dit_decoded_taiji.sh
```

脚本默认使用新端口 `12555`，避免与退出不完整的旧 `12455` rendezvous 冲突。

两机 × 8 卡时 global batch=16，2 epochs 自动换算为 2736 optimizer steps。
默认配置 `configs/dit_decoded_aux_full_v1.json`：

- 从原 step-50000 EMA DiT 继续训练，VAE 编解码器冻结。
- LR `2e-6`，500 步 warmup。
- segmentation / boundary / reprojection 权重为 `0.1 / 0.02 / 0.1`。
- 辅助损失仅用于 `t<=0.6` 且条件未丢弃，乘 `(1-t)^2` 和 500 步 warmup。
- rank 0 维护 EMA 0.9999，并同时保存普通权重与 EMA 权重。
- 10% batch 加入原始 DiT 教师速度约束，权重 0.1，用于减少续训漂移。
- 每 500 步保存；断点续训使用 `SEG_DIT_RESUME=latest`，world size 必须一致。

本轮建议先训练 2 epochs，不直接跑 3 epochs。每 500 步在 validation 上比较
普通与 EMA checkpoint，选择后只评估一次 test。必须同时报告原始模型、
本轮模型和冻结 VAE oracle，全部关闭后处理和标签匹配。

若教师模型导致单卡显存不足，可以新建配置把
`teacher_replay.weight/probability` 设为 0；不要临时在各节点使用不一致配置。

## 审计与容量

320 对象试验缓存平均约 1.75MB；全量预计约 40GB。项目盘当前可用空间约
19TB。原数据盘接近满载，新缓存不得写到原数据目录。
