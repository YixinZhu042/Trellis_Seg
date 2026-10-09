# VAE parts64：恢复 5,000 步状态并随机置换部件编号

训练结束后的完整对比：[2026-09-30 置换续训重建检查](seg_vae_v2_parts64_permute_test_20260930.md)。

2026-09-30 按用户要求，新增在线编号置换续训分支。

## 本轮设置

- 父实验：`seg_vae_v2_parts64` 的 `trainer_latest.pt`，step 5,000。
- 从 step **5,001** 继续到 **10,000**，即再训练 5,000 个 optimizer steps。
- 三机 24 卡，global batch 24；保留模型、optimizer、EMA、24 rank 的 RNG 状态及训练步数。
- 不改变 latent 32、网络结构、loss 权重、学习率、KL 计划或对象采样比例。
- 不重新 warmup；每 500 步保存普通权重、EMA、完整续训状态，并评估全部 128 个验证对象。
- 数据沿用 22,633 / 128 / 128 的划分，不重新生成缓存、不改原始标注。

配置：`configs/seg_vae_v2_parts64_permute.json`。

## 置换的含义

训练每次取到一个对象，生成对 **1–63 全部非零通道**的随机一一映射，0 始终不变。
不是仅在该对象的 1～部件数之间打乱。因此，两部件对象也能使用 47、60 这类高编号，
使每个通道都有机会接收真实部件的正监督。

设 `p[old] = new`：

- 面标签：`new_face_labels = p[old_face_labels]`。
- 体素整数标签：同一个正向映射。
- 输入软标签：`new_target[:, p] = old_target`，实际按逆置换重新排列通道。
- 所有 query labels、point/surface CE、Dice、投影监督均从置换后的面标签生成。
- 几何、邻居、光栅、面积/像素权重不变；same/different 部件关系不变。
- 不置换模型参数，也不置换 optimizer 动量；这是训练数据增强，不是重排模型权重。

置换仅存在于当前样本的内存副本，不写回缓存。每次置换由 seed 和全局样本 index
确定，重启 resume 可以复现；采用独立无状态 RNG，不额外消耗原有 query/posterior RNG。
训练日志记录实际对象、全局 index 和已出现标签的 `from/to` 映射，便于验收。

## 验证和风险

验证/测试仍使用原始编号，不启用随机增强。编号置换可能使初始训练 loss 大幅升高，
它与原编号下旧训练末期的 loss 不是同一分布，不能直接作好坏比较。
保留原父实验的 5,000 步权重作为外部基线；本分支的 `selection.json` 只在本分支
完成验证的新 checkpoint 中选优，首轮预定在 step 5,500，绝不覆盖父实验选型结果。

这轮动机来自已查看的测试吉他，因此旧测试集已参与开发诊断，后续在同一测试集的
提升应视为开发回归结果，不能当作完全独立的最终泛化估计。验证集选型仍不使用测试分数。
增强本身不保证细小部件或空间压缩问题全部解决，本轮不叠加其他结构改动。

## Remote API 命令

只在控制节点执行一次：

```bash
bash scripts/resume_seg_vae_v2_parts64_permute_remote.sh
```

查看进度，不启动新任务：

```bash
bash scripts/resume_seg_vae_v2_parts64_permute_remote.sh status
```

打印命令，不连接 API、不运行训练：

```bash
bash scripts/resume_seg_vae_v2_parts64_permute_remote.sh dry-run
```

本分支产生自己的 `trainer_latest.pt` 后，若任务中断，可用：

```bash
bash scripts/resume_seg_vae_v2_parts64_permute_remote.sh resume
```

正常 `--resume` 的严格兼容检查不放宽。首次分支使用显式 `--resume_from`，仅允许
从无增强的 64 类同构实验启用 `all_nonzero_v1` 并延长步数；若模型/loss/队列/卡数等
发生变化直接拒绝。源 checkpoint 通过 private mmap 加载以避免每 rank 先复制完整
13 GB 到 CPU；写入已映射 CPU 张量采用 copy-on-write，不修改父 checkpoint。

## 输出位置

```text
/path/to/trellisseg_run/
  checkpoints/seg_vae_v2_parts64_permute/
  runtime_cache/seg_vae_v2_parts64_permute/
```

日志在项目 `logs/seg_vae_v2_parts64_permute/`：

- `remote_launch_*.json`：三节点会话、命令及预检结果。
- `launcher_*_nodeN.log`：数据检查、状态恢复、DDP 和训练日志。
- `resume_rank0000.json`～`resume_rank0023.json`：原步数、optimizer step、EMA/RNG 恢复验收。
- `train_rank*.jsonl`：实际 loss、梯度及编号置换记录。
- checkpoint 目录的 `continuation.json`：源 checkpoint 身份和本次唯一配置变化。

54 项测试通过，包括非自反置换方向、零标签保留、全 63 通道覆盖、缓存不变、所有
loss/梯度在一致重编号下不变、严格 resume 兼容和并行文件身份检查。

## 实际启动验收（2026-09-30）

有效任务记录：`logs/seg_vae_v2_parts64_permute/remote_launch_20260930_113646_71d16e.json`。
11:41（Asia/Shanghai）已确认全部 24 rank 从 step 5,001 开始，至少完成到 5,017；
观察到的 loss 和梯度全为有限值。每 rank 恢复 576 份 optimizer 参数状态，内部 step
均为 5,000，EMA/RNG 恢复通过。源 checkpoint 大小及修改时间未改变。

验收快照：`logs/seg_vae_v2_parts64_permute/startup_verification.json`。
已观察到全部 1–63 编号参与监督；两部件对象的实际映射包括 `[1,2] -> [51,42]`
和 `[1,2] -> [56,20]`。此为启动验收，不是质量提升结论，首个原编号验证在 step 5,500。

首次尝试中两个 rank 进入异常退出清理，NCCL 清理等待掩盖了原始异常；没有完成训练
step。只停止了这次尝试的三个会话，保留日志及空目录
`checkpoints/seg_vae_v2_parts64_permute_unstarted_20260930_112646`。训练入口已改为
异常时先打印 traceback，再让 torchrun 终止本任务其他 rank，避免异常清理无限等待。
重新启动后通过上述验收；首次原始异常未被独立还原，不推断其唯一根因。
