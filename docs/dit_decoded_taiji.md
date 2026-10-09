# 固定 VAE 的 DiT 解码监督实验：TAIJI 启动与结果

> 发布说明：本页保留实验记录；机器路径已改为占位符，批量图片、GLB、日志和数据缓存不包含在 GitHub 源码中。
本实验已经完成。两组均使用两机共 16 张 H20、global batch 16、训练
300 步。实验组 16/16 rank 均确认三项辅助损失能反传到 DiT，VAE 前后
权重摘要一致。默认模型仍保留原始 checkpoint；完整结果见
`eval/dit_decoded_report_v1/README.md`（本地运行产物，未随源码发布：`../eval/dit_decoded_report_v1/README.md`）。

测试集原始输出（32 对象 × 3 种子，无后处理）：

| 模型 | 投影 mIoU | 表面 mIoU | 边界 F1 |
| --- | ---: | ---: | ---: |
| 原始 checkpoint | 87.392% | 84.711% | 22.185% |
| 纯 flow-MSE 续训 | 86.675% | 83.535% | 19.702% |
| 解码任务损失续训 | 86.930% | 83.877% | 20.286% |

新增损失相对纯 MSE 续训提升投影 mIoU 0.255、表面 mIoU 0.342、边界 F1
0.583 个百分点；三项按对象 bootstrap 的 95% 区间均略高于零。但实验组
相对原始 checkpoint 的投影和表面 mIoU 分别下降 0.463 和 0.834 个百分点，
所以本轮不替换默认权重。以下启动说明保留用于复现或后续消融。

## 直接启动

在 TAIJI 多机发送命令 WebUI 中，对所有分配节点发送以下命令。
WebUI 自动追加 `<HOST_NUM> <NODE_RANK> <MASTER_ADDR>`，无需手工写这三个参数。

**新增损失组：**

```bash
train: bash /path/to/Trellis_Seg/scripts/train_dit_decoded_taiji.sh
```

**纯 flow-MSE 续训对照组：**

```bash
train: SEG_DIT_MODE=flow bash /path/to/Trellis_Seg/scripts/train_dit_decoded_taiji.sh
```

建议两组顺序运行，使用相同节点数、每节点 GPU 数、步数和梯度累积。
两组有不同输出目录和 rendezvous 端口，不共享训练状态。

手工启动示例（两机、每机八卡）：

```bash
# 节点 0
bash scripts/train_dit_decoded_taiji.sh 2 0 10.x.x.x

# 节点 1
bash scripts/train_dit_decoded_taiji.sh 2 1 10.x.x.x
```

## 默认参数与数据

| 项目 | 默认值 |
| --- | --- |
| Python | `/path/to/trellis-env/bin/python` |
| 配置 | `configs/dit_decoded_aux_v1.json` |
| 已准备数据 | `data/dit_decoded_v1`，256 train / 32 validation / 32 test |
| 每节点 GPU | 8 |
| 每卡微批量 | 1 个对象 |
| 每卡梯度累积 | 1，TAIJI 脚本显式覆盖 JSON 中的单卡检查值 4 |
| 总 batch | 节点数 × 8 × 梯度累积；两机时为 16 |
| 更新步数 | 300 |
| AdamW 学习率 | `5e-6`，30 步线性 warmup |
| 精度 | DiT BF16 autocast、FP32 参数/优化器；VAE 权重冻结 |
| 辅助损失权重 | segmentation 0.1，boundary 0.05，reprojection 0.1 |
| 辅助损失门控 | `t <= 0.7` 且多视图条件未丢弃；乘 `(1-t)^2` 与 50 步 warmup |
| 保存间隔 | 每 100 步，训练结束也保存 |

该数据集专门为本实验生成，每个对象使用同一次采样和统一标签定义建立
体素、潜变量、面标签、边界、相机与可见像素对应。没有将旧的独立采样
`seg_voxels` 与 `seg_latent` 强行按行匹配。所有 320 个样例准备成功。
这是小规模受控续训实验；原始 checkpoint 可能已在预训练时见过这些对象，
因此不能把 validation/test 结果直接宣称为全新的数据集泛化表现。

300 步在两机八卡、累积 1 时处理 4,800 个对象次，约 18.75 个训练集遍历。
不要根据机器数自动线性放大学习率；先用此小规模实验判断相对收益。

两组启动均从现有 step-50000 DiT EMA 权重开始。对照组和实验组保持相同
数据顺序、rank 划分、噪声、时间步、条件丢弃概率和优化器，只改变辅助损失。
VAE 参数不更新，解码器输入的梯度保留。训练与评估都不使用投影覆盖、图优化等后处理。

## 修改参数

所有节点必须使用相同环境变量。例：每卡累积 2 次，总 batch 为两机 × 八卡 × 2 = 32：

```bash
train: SEG_DIT_RUN_NAME=dit_decoded_ddp_acc2 SEG_DIT_ACCUMULATE=2 SEG_DIT_STEPS=300 bash /path/to/Trellis_Seg/scripts/train_dit_decoded_taiji.sh
```

支持 `SEG_DIT_GPUS_PER_NODE`、`SEG_DIT_CONFIG`、`SEG_DIT_OUTPUT_DIR`、
`SEG_DIT_LOG_DIR`、`SEG_DIT_MASTER_PORT`、`SEG_DIT_TIMEOUT`、`SEG_DIT_PYTHON`。
新实验请设置新的 `SEG_DIT_RUN_NAME`，或者明确给出新输出目录。
正式训练拒绝覆盖非空目录。

脚本默认使用仓库 H20 集群的 `bond1` / RDMA 参数，NCCL 环境变量均允许覆盖。
若另一集群仅支持 TCP，需根据该集群接口设置，例如
`NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0`；不要在不知道接口的情况下照搬。
脚本不会杀其他训练进程，不会清理占用端口。

## 启动前检查

只打印 torchrun 命令，不加载模型、不占 GPU、不创建实验目录：

```bash
SEG_DIT_DRY_RUN=1 bash scripts/train_dit_decoded_taiji.sh 2 0 10.x.x.x
```

只读检查配置、checkpoint 路径、完整数据与 split，仍不加载模型、不训练：

```bash
SEG_DIT_CHECK_ONLY=1 bash scripts/train_dit_decoded_taiji.sh 2 0 10.x.x.x
```

正式启动会在每个节点先执行该数据检查。共享项目、数据及 checkpoint 路径
需要在所有节点可读，输出和日志目录需要可写。

## 输出与断点恢复

新增损失组默认输出：`experiments/dit_decoded_ddp_v1/`。
对照组默认输出：`experiments/dit_flow_ddp_v1/`。

- `train.jsonl`：rank 0 汇总的全局损失与有效样本数。
- `ranks/rankXXXX.jsonl`：每卡的样例 ID、时间步、条件丢弃、各项损失。
- `ranks/gradient_check_rankXXXX.json`：该卡首次确认三项辅助梯度非零的记录。
- `dit_step000100.pt` 等：可直接用于评估/推理的 DiT state_dict，无 `module.` 前缀。
- `trainer_latest.pt`：最新完整模型、AdamW 状态和各 rank 随机状态。
- `checkpoint_latest.json`：最新保存步数；`manifest.json`：配置、split、冻结权重检查。
- 终端日志：`logs/dit_decoded_ddp_v1/node0.log` 等，每个节点独立追加。

权重约 5.2 GB/份，完整 AdamW 恢复状态更大；目录需要预留几十 GB 空间。
恢复文件保存采用临时文件再原子替换，只有 rank 0 写入共享 checkpoint。

中断后，在原输出目录继续到总计 300 步：

```bash
train: SEG_DIT_RESUME=latest bash /path/to/Trellis_Seg/scripts/train_dit_decoded_taiji.sh
```

恢复要求相同 world size、配置、累积次数、数据划分和模式。若要从已完成
300 步继续到 600 步，可在同样命令中加 `SEG_DIT_STEPS=600`。
恢复文件应使用本实验自己产生的 `trainer_latest.pt`。

## 训练后评估

同样支持 TAIJI 多机多卡，每个 rank 评估不同对象并自动合并指标。
先看 validation，选定 checkpoint 后再运行 test；不要用 test 选参数。

```bash
# 原始 checkpoint 的验证集基线（不指定 SEG_DIT_EVAL_CHECKPOINT）
train: SEG_DIT_EVAL_OUTPUT=eval/dit_pretrained_val_ddp bash /path/to/Trellis_Seg/scripts/eval_dit_decoded_taiji.sh

# 新增损失组，step 300
train: SEG_DIT_EVAL_CHECKPOINT=experiments/dit_decoded_ddp_v1/dit_step000300.pt SEG_DIT_EVAL_OUTPUT=eval/dit_decoded_val_ddp bash /path/to/Trellis_Seg/scripts/eval_dit_decoded_taiji.sh

# 对照组，step 300
train: SEG_DIT_EVAL_CHECKPOINT=experiments/dit_flow_ddp_v1/dit_step000300.pt SEG_DIT_EVAL_OUTPUT=eval/dit_flow_val_ddp bash /path/to/Trellis_Seg/scripts/eval_dit_decoded_taiji.sh
```

最终测试时设置 `SEG_DIT_EVAL_SPLIT=test` 并更换输出目录。
检查 `metrics.json` 中的原始输出投影 mIoU、表面 mIoU、边界 F1 和小部件召回。
标签 ID 固定、不做匹配、不做后处理；同时记录 VAE oracle，区分压缩瓶颈与 DiT 误差。

## 本次已验证范围

两步真实单卡 DiT/VAE 梯度验证已完成，VAE 确认未更新。
TAIJI 参数展开、输入只读检查、Python/Shell 语法检查通过。
两进程 CPU/Gloo 测试覆盖条件分支未用参数、梯度检查点、累积、全局批量等价性、
rank 0 checkpoint 写入和模型/优化器恢复。CPU 测试使用微型模型，不是正式训练。
真实多节点 NCCL/RDMA 与 16 卡大模型训练已完成。训练、split、梯度、冻结
权重和测试协议均已审计；检查点仍可用于进一步消融，但不是当前默认模型。
