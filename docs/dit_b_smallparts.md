# DiT B：最佳权重续接、同步标签置换与小部件优化

日期：2026-10-08。配置：`configs/dit_vae_v2_parts64_b_smallparts.json`。

## 实验边界

用户选择 B，并要求加入过采样等小部件优化。本轮不是原版 TRELLIS 重开，
不是 VAE 训练，也不覆盖此前 DiT、数据缓存或推理结果。

- 初始 DiT：`checkpoints/dit_vae_v2_parts64/regular_step004500.pt`，原验证集选出的最佳普通权重。
- 冻结 VAE：`checkpoints/seg_vae_v2_parts64_permute/regular_step009500.pt`。
- 模型结构、latent32、六视图条件和 shape 条件不变。
- 新建 optimizer 和 EMA，局部训练步数从 1 开始，计划新增 5,000 步。
  不能将它称为完整恢复旧 step 4,500 的训练状态；该步没有对应的完整 optimizer。
- 初始权重加载前严格校验 source step/kind、VAE SHA256、数据 manifest digest、模型配置 digest，
  禁止加载 smoke 或不同 latent 空间的权重。`initialization.json` 记录迁移来源。
- train 22,633，validation 128，test 128，保持原划分。训练采样和增强仅使用 train。

## 1. 50% 同步标签置换，在线重编码

每次抽到训练对象后，以 50% 概率生成一个 `0→0、1..63 全范围随机双射`。
同一份 old→new 映射同时作用于 soft voxel label 通道、hard voxel label、mesh face label。
随后从置换后的面标签生成全部六个视图，用同一个冻结 VAE 的后验均值重新编码：

```text
同一对象 + 同一份编号映射
  ├─ 三维标签 → 冻结 VAE encoder → 目标 latent
  ├─ 六视图标签 → 冻结 VAE encoder → 六视图条件 latent
  └─ 面标签 → 面 CE / 投影 CE / 边界辅助监督
```

形状、坐标、相机、shape latent 不变。六视图不独立置换，不混合对象，不打乱 latent32
通道，不改变特征与位置对应。渲染图中背景为 0，mesh 前景使用 `label+1`；编码入口只去掉
一次这个渲染偏移。未知标签 0 与图像背景仍保持原有区分。

未置换的 50% 直接复用原缓存。增强后的 latent 仅在内存中使用，不写入原缓存，不需要预先
制作数倍全量缓存。代价是增强样本增加一次三维编码和六次视图编码；实际吞吐须用正式日志测量。
初始化小样通过恒等映射在线重编码，与原缓存核对数值及坐标/六视图边界。

增强随机数由训练 seed + 全局 sample index 决定，独立于噪声与面查询随机数；重启后同一步
同 rank 得到同一编号映射。日志保存 sample index、是否置换、映射摘要和编码时间。

### 为什么保持原归一化统计

本轮保持原缓存由训练集计算的 mean/std：更换统计会立即改变已训练 DiT 的输入、输出坐标系。
置换数据也用这套固定统计，不使用验证/测试统计，不声称新增强分布已被重新标准化为零均值单位方差。
如后续要重新估计统计，应独立实验并正确迁移模型接口，不能在 B 中悄悄替换。

## 2. 对象级多部件过采样

| 对象部件数 | 训练对象数 | 原对象占比 | 本轮抽样目标 |
| --- | ---: | ---: | ---: |
| 1–20 | 19,864 | 87.77% | 70% |
| 21–32 | 1,832 | 8.09% | 15% |
| 33–63 | 937 | 4.14% | 15% |

每个采样周期仍是 22,633 个 draw，取整后分别为 15,843 / 3,395 / 3,395。
组内先不放回打乱，耗尽后开启新的打乱周期；最终把所有组的 draw 混合打乱。
因此高部件数组单对象每周期平均约 3.62 次，中间组约 1.85 次，低组约 0.80 次。
低组在后续周期重新抽样，不是永久删掉未抽中的对象。

三个节点的全部 rank 使用同一全局序列，按 sample index 分配；周期与序列可确定性重建。
这是有意改变训练分布，不再乘反向对象抽样权重把过采样抵消。
部件多不等于一定含小部件，所以还需要下一层面监督。

## 3. 面查询过采样 + 真正的部件等权监督

每次仍查询 2,048 个监督面、最多 1,024 个关系对，不扩大面查询显存预算。
面抽样分布从 `50% 按面积 + 50% 按部件均衡` 改为：

```text
p_draw(f) = 0.25 × area(f) / total_area
          + 0.75 × area(f) / (该部件总面积 × 部件数)
```

只在有标注且面积为正的面上抽样。均衡项先给每个部件相同抽样概率，再在部件内部按面积抽面，
不受某个部件被三角化成多少个面的影响。零面积面和未知标签不会获得额外权重。

之前 `area/p_draw` 的损失重要性修正使优化目标仍是面积平均；仅增加小面采样并不能给小部件
相同的总贡献。本轮额外计算：

```text
p_macro(f) = area(f) / (该部件总面积 × 部件数)
L_macro   = mean[ CE(f) × p_macro(f) / p_draw(f) ]
L_surface = 0.5 × L_area + 0.5 × L_macro
L_seg     = 0.5 × L_voxel + 0.5 × L_surface
```

`L_macro` 是按部件等权、部件内按面积平均 CE 的 Monte Carlo 估计。分母来自整个对象的真实
部件面积，不从当前随机抽中的面估计；不再用 batch 权重和归一化改变这个估计。
在本轮 75% 均衡采样下，`p_macro/p_draw` 最大不超过 4/3，避免极小部件的重要性权重爆炸。

保留一半面积 CE，是为了不让主体大块的稳定性完全让位于小部件。
体素 CE 仍用原来的 soft-label 平衡 CE；边界和投影项不改。
日志单独记录 `surface_area_ce` 与 `surface_macro_ce`。

## 4. 其他固定设置和评估

- LR `1e-5`，warmup 500，AdamW，梯度裁剪 1，global batch 24。
- EMA 从当前 regular 初始化，衰减系数改为 `0.999`，不沿用旧 `0.9999` 长惯性 EMA。
- flow loss、辅助权重 `segmentation=.1 / boundary=.02 / reprojection=.1` 不变。
- 辅助监督仍只在条件未丢弃且 `t<=.6` 生效，乘 `(1-t)^2` 及 warmup。
- 记录辅助损失到预测 velocity 的梯度与 flow 梯度；它们不是全模型参数梯度，不能混淆。
- 每 500 步保存普通权重、EMA、完整 optimizer/RNG；验证完整 128 对象，原编号、固定噪声、
  20 步、CFG=1，保证与旧训练选型口径一致。此次不悄悄改成 CFG=3。
- 验证同时输出 1–20 / 21–32 / 33–63 分组，观察总体、小部件、隐藏面及普通对象退化。
- 不依据测试集选择 checkpoint。这是“B+小部件优化”的组合实验，不能把结果变化单独归因于置换。

## 存储与命令

所有缓存和 checkpoint 均在：

```text
/path/to/trellisseg_run/
  data/dit_vae_v2_parts64/                              # 只读复用
  checkpoints/dit_vae_v2_parts64_b_smallparts/           # 正式新实验
  checkpoints/dit_vae_v2_parts64_b_smallparts_smoke/     # 小样测试，不能作为正式初始化
  runtime_cache/dit_vae_v2_parts64_b_smallparts/node*/
```

只在控制节点执行一次，通过用户指定的 Remote API 管理三机24卡；不使用 TAIJI/SSH。

```bash
bash scripts/train_dit_b_smallparts_remote.sh status
bash scripts/train_dit_b_smallparts_remote.sh dry-run
# 确认无活动同类训练且目标目录为空时首次启动：
bash scripts/train_dit_b_smallparts_remote.sh launch
# 本实验中断后，保持配置、划分、VAE、world size 不变恢复：
bash scripts/train_dit_b_smallparts_remote.sh resume
```

日志：`logs/dit_vae_v2_parts64_b_smallparts/train/`。

## 验收状态

真实三机24卡两步测试及恢复第3步、Gaussian-start生成均已通过，三台均退出码0。
24 rank恒等重编码目标/六视图最大误差全为0，VAE无参数梯度，DiT真实更新，辅助梯度均非零且有限。
覆盖9/29/62部件普通训练样本及最大的11,873/10,732/10,245-token三个对象；峰值37.27GiB以内。
报告：`logs/dit_vae_v2_parts64_b_smallparts_smoke/train/smoke_verification.json`。
另修复显式 smoke 子集只有单一部件数组时错误要求所有采样组齐全的问题，生产采样不受影响。

实现通过57项VAE回归测试及25项DiT/编号相关测试；代码独立复查未发现正式训练阻断项。
恢复测试是状态加载/继续更新/推理验收，不声称做过完整不间断训练与恢复训练的逐张量bitwise对照。
正式训练于2026-10-08 15:11通过Remote API提交；15:16启动验收通过：全部24 rank完成至少15步，
已审计369条样本记录（原编号202、置换167、条件丢弃37、有辅助211、无辅助158），全部采样序列/
置换摘要与确定性重放一致，所有已读loss/梯度有限，24项辅助梯度检查通过，无节点失败退出。
当时观测的正式训练显存峰值34.19GiB以内。该短窗口的比例有随机波动，不能替代整周期比例。
启动验收：`logs/dit_vae_v2_parts64_b_smallparts/train/startup_verification.json`。
启动记录：`logs/dit_vae_v2_parts64_b_smallparts/train/remote_20261008_151107.json`。
启动成功不等于小部件质量已改善，首次正式checkpoint/全量验证在局部step500。

可复用检查命令：

```bash
python3 -B scripts/verify_dit_b_run.py --mode smoke
python3 -B scripts/verify_dit_b_run.py --mode startup --min-steps 10
```
