# VAE v2：正确原始标注、64 类、32 维 latent

5,000 步训练后的完整 128 对象测试及 GLB/渲染结果：
[2026-09-30 重建测试报告](seg_vae_v2_parts64_test_20260930.md)。

针对编号偏向的后续实验：[从 5,000 步恢复并随机置换 1–63 编号](seg_vae_v2_parts64_permute.md)。

训练配置：[seg_vae_v2_parts64.json](../configs/seg_vae_v2_parts64.json)。
64 类包含标签 0，最多 63 个有效部件。标签来自原始 GLB 子网格与 maskId 解析，
不复用旧版错误的面标签。旧 VAE 主干用于初始化，语义输入／输出层重新初始化；
这是新实验，不从旧 20 类模型恢复 optimizer 或 EMA。

## 数据划分

根据用户 2026-09-29 指定的划分，使用已通过检查的 22,889 个对象：

| 划分 | 对象数 |
| --- | ---: |
| train | 22,633 |
| validation | 128 |
| test | 128 |
| 隔离，不参与训练或评估 | 20 |

20 个隔离对象的原始标注引用非三角图元或缺失几何，不能可靠匹配面标签。保留原始
数据与错误原因，不制造三角面，不删除原始文件。原验证集中隔离 1 个，原训练集中
隔离 19 个，原测试集无此类失败对象。

规则为 `nested_holdout_sha256_v1`，seed 为 `20260929`：分别在原来的有效 validation、
test 集内按 `SHA256(seed:split:instance)` 排序取前 128 个，避免将旧训练集对象转为
新测试对象。其余可用对象均进入 train，包括原 validation 中剩余 383 个和原 test
中剩余 384 个。规则不读取预测或质量指标。

这是新的数据划分，不能再把旧版 512 个测试对象全部视为这轮的留出测试集；其中部分
已进入本轮训练。旧初始化模型可能见过这些对象，因此只能保证本轮更新的划分互斥。

缓存 NPZ 中的 `metadata.split` 保留当时预处理划分，便于溯源。当前实验的成员关系
以最终 `manifest.json` 的 `split` 为准，并以 `preparation_split` 记录原划分。
重新划分不重写 27.2 GiB 的缓存张量。训练和评估加载器会重算哈希划分并检查互斥、
对象总数、隔离记录与文件身份，拒绝与配置不匹配的 manifest。

数据根目录：

```text
/path/to/trellisseg_run/data/seg_vae_v2_parts64/
├── samples/             # 64 通道数据缓存
├── manifest.json        # 当前训练/验证/测试清单，训练就绪标志
├── split_128.json       # 划分规则和逐对象分配
├── quarantine.json      # 20 个隔离对象及原因
└── prep_rank*.json      # 三机预处理历史记录，含原划分和原失败记录
```

## 三机 24 卡训练（Remote API）

按用户要求，直接使用 `../remote_api` SDK，不需要 TAIJI 的 `train:` 命令，
也不使用 SSH。**只在当前控制节点执行一次**：

```bash
python3 /path/to/Trellis_Seg/scripts/train_seg_vae_v2_remote.py --launch
```

启动器连接已登记的 `master` 服务，使用已授权看板 `your-agent-board-id`。
节点依次为 `node0`、`node1`、`node2`。
认证沿用本机 Remote API 客户端密钥，不复制或输出私钥。

先通过 WriteGuard 内的只读检查核对环境、路径、重复进程和空闲主端口，
然后逐节点 `create_session` 后立即 `exec_command`，使用持久会话前台运行
`torch.distributed.run`。不会修改看板、放松 WriteGuard、终止占卡或其他任务。
提交成功不等于训练成功，需检查各 rank 的实际 step。默认设置：

- 每节点 8 卡，总计 24 卡，global batch 24。
- 总 5,000 个 optimizer steps。
- 每 500 步保存普通权重和 EMA，使用全部 128 个 validation 对象评估选型。
- 默认新建训练，不要使用旧的 `resume_seg_vae_v2_taiji.sh`，也不要设置
  `VAE_V2_RESUME=1` 来恢复旧 20 类权重。

checkpoint 路径：

```text
/path/to/trellisseg_run/checkpoints/seg_vae_v2_parts64/
```

日志路径：

```text
/path/to/Trellis_Seg/logs/seg_vae_v2_parts64/
```

每次启动创建 `remote_launch_<时间戳>.json`，记录三台的命令、会话 ID 和预检结果；
启动日志为 `launcher_<时间戳>_node0.log` 至 `node2.log`，退出后写
`exit_<时间戳>_nodeN.txt`。各 rank 指标为 `train_rank0000.jsonl` 至
`train_rank0023.jsonl`。启动时每个 rank 会打印 manifest 校验和分布式初始化进度。
首次启动会对缓存、旧源 NPZ、GLB 和标注逐文件检查大小及修改时间；远端节点的源
文件元数据访问可能耗时数分钟以上。在 `Checking prepared manifest` 阶段没有 loss
是正常的，但不能据此判断训练成功。只有全部 rank 写出 step 后才算完成训练启动验收。

检查最近一次任务（不提交新任务）：

```bash
python3 scripts/train_seg_vae_v2_remote.py --status
```

运行时缓存（Triton、Torch、CUDA、临时缓存等）统一在
`/path/to/trellisseg_run/runtime_cache/seg_vae_v2_parts64/nodeN/`。
重复执行 `--launch` 不会恢复或覆盖已有训练；恢复训练需要明确使用本实验自己的
checkpoint。若提交阶段失败，启动器仅清理本次创建的会话，保留日志及退出信息。

本次划分已封存后无需再次预处理或执行旧 `--merge`；该命令面向原始分片划分。
若重建同样的可用集及新划分，使用独立的封存工具：

```bash
python3 scripts/finalize_seg_vae_v2_split.py --config configs/seg_vae_v2_parts64.json --workers 16
```

只读检查训练输入，不启动 GPU：

```bash
python scripts/train_seg_vae_v2.py --config configs/seg_vae_v2_parts64.json --check_only
```

离线查看三节点命令，不连接 API、不启动训练：

```bash
python3 scripts/train_seg_vae_v2_remote.py
```

## 2026-09-29 Remote API 启动验收

有效任务：`remote_launch_20260929_200219_923665.json`（位于上述日志目录）。
2026-09-29 20:08（Asia/Shanghai）确认全部 24 个 rank 至少完成 18 个 optimizer
steps，已读到的 loss 各项和梯度范数均有限；主节点随后打印 step 20。
逐 rank 验收快照在 `startup_verification.json`，后续进度以实时日志为准。

首次任务在 DDP 的 `verify_params_across_processes` 阶段长时间等待，未进入训练。
已仅停止首次任务自身的三个会话，保留日志；无权重的旧输出目录保留为
`checkpoints/seg_vae_v2_parts64_unstarted_20260929_193006`。
独立短测试 `ddp_check_20260929_195847/plan.json` 记录全部 24 rank 完成 GPU
all-reduce、整数 all-gather、DDP 初始化和两步反向传播，三节点退出码均为 0。
增加 VAE 上 GPU 和 EMA 创建后的显式 CUDA 同步、逐 rank 初始化阶段日志后重新启动，
实际 VAE 训练通过上述验收。尚未单独隔离首次等待的唯一根因，不能据此认定数据或
模型结构有误，也没有修改集群网络或停止既有占卡任务。

当前配置在第 500 步首次保存普通权重、EMA 和可续训状态，并评估完整 128 个验证
对象；启动验收不等于收敛或重建质量验收。不要再次运行 `--launch` 来查看进度。
