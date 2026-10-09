# B＋小部件优化 DiT：测试集推理（2026-10-09）

> 发布说明：本页保留实验记录；机器路径已改为占位符，批量图片、GLB、日志和数据缓存不包含在 GitHub 源码中。
主要入口：GT / 旧DiT / 新DiT / VAE 四排对照（本地运行产物，未随源码发布：`../results/dit_b_smallparts_test_20261009/comparison.html`）。
新模型入口：GT / VAE / DiT 六视图与 GLB（本地运行产物，未随源码发布：`../results/dit_b_smallparts_test_20261009/index.html`）。
逐对象配对：paired_comparison.json（本地运行产物，未随源码发布：`../results/dit_b_smallparts_test_20261009/paired_comparison.json`）。
数值记录：metrics.json（本地运行产物，未随源码发布：`../results/dit_b_smallparts_test_20261009/metrics.json`）。
导出验收：artifacts.json（本地运行产物，未随源码发布：`../results/dit_b_smallparts_test_20261009/artifacts.json`）。

## 结论

固定128个测试对象，B的表面mIoU为 **85.05%**，旧DiT为 **83.58%**，提升 **1.46个百分点**。
小部件召回 **57.03%→60.61%**，隐藏面mIoU **71.67%→73.48%**。
整体和多部件组有所改善，第一把吉他的高编号琴身已基本恢复，但大量细部件仍丢失。
同一VAE的重建表面mIoU为95.96%，DiT仍相差10.91个百分点；VAE参考不是逐对象的严格上界。

## 固定模型和协议

- 新DiT：`checkpoints/dit_vae_v2_parts64_b_smallparts/ema_step005000.pt`。
- 旧DiT：`checkpoints/dit_vae_v2_parts64/regular_step004500.pt`。
- VAE：`checkpoints/seg_vae_v2_parts64_permute/regular_step009500.pt`，全冻结。
- 新checkpoint由128个validation对象选出；本次不根据test重新挑选checkpoint或采样设置。
- 同一个原始正确编号缓存、训练集归一化统计、shape条件与GT衍生六视图条件，使用已知mesh。
- 从Gaussian噪声开始，20步Euler，CFG=1；每对象seed为 `20260930 + int(instance[:8],16)`。
- 目标GT latent值不用于DiT初始化或条件，单独解码为VAE重建参考。
- 无Hungarian编号匹配，无后处理、平滑、连通域清理，几何和面顺序不变。
- 新旧使用同一批128个test对象。另有3个train历史诊断、3个validation复算、3个validation退化
  诊断，共137个对象。只有128个test计入主要指标。
- 条件来自正确GT六视图分割，并非外部分割API/真实照片的端到端表现。
- 该测试集此前参与开发诊断，因此是固定开发回归评估，不是新的独立盲测。
- 本轮为“额外训练＋编号置换＋过采样＋部件等权损失＋EMA调整”的组合结果，没有同预算
  单因素对照，不能把提升单独归因于其中一种改动。每对象一个固定种子，未测多种子方差。

## 128个test对象指标

先按对象计算，再平均有效指标；没有小部件的对象不参与小部件召回平均。
小部件为占对象总标注表面积不到1%的部件；隐藏面为六视图中无可见像素覆盖的面。

| 指标 | 旧DiT | B EMA5000 | 变化（百分点） |
| --- | ---: | ---: | ---: |
| 表面mIoU | 83.58% | 85.05% | +1.46 |
| 表面面积准确率 | 93.86% | 94.95% | +1.09 |
| 投影mIoU | 88.51% | 89.72% | +1.21 |
| 投影准确率 | 97.79% | 98.79% | +1.00 |
| 隐藏面mIoU | 71.67% | 73.48% | +1.81 |
| 表面小部件召回 | 57.03% | 60.61% | +3.59 |
| 体素mIoU | 79.70% | 81.10% | +1.39 |
| 边界F1 | 30.71% | 31.93% | +1.22 |
| 孤立错误面积比例（低为好） | 1.276% | 1.228% | −0.047 |

新模型表面mIoU≥90%为73/128，≥95%为53/128，≥99%为28/128，低于50%仍有10个。

### 按真实部件数分组

| 部件数 | test数量 | 旧表面mIoU | 新表面mIoU | 变化 | 旧小部件召回 | 新小部件召回 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–20 | 108 | 89.20% | 90.13% | +0.93pp | 66.44% | 69.40% |
| 21–32 | 12 | 65.57% | 68.87% | +3.31pp | 46.45% | 53.19% |
| 33–63 | 8 | 34.74% | 40.60% | +5.86pp | 29.33% | 31.13% |

test高部件组的小部件召回仅提高约1.80pp，明显小于此前validation高部件组的改善。
不能直接把validation的收益照搬为test收益；这组只有8个对象，类别、可见性和部件面积未受控。

## 吉他：主体修复不等于细部件修复

| 对象 | 旧表面mIoU | 新表面mIoU | 旧面积准确率 | 新面积准确率 | 完全漏掉的部件数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `251c8125765149c0bd46b3a1a51ff4b6`，60部件 | 5.28% | 8.77% | 18.87% | 74.69% | 46→44 / 60 |
| `d1d58f05c346470e9805612d562e9dde`，54部件 | 10.55% | 14.30% | 78.50% | 85.75% | 40→39 / 54 |

第一把吉他的编号60琴身召回从 **0.35%提高到90.56%**，琴身IoU为88.84%。观察六视图，
大块琴身颜色恢复，琴颈、弦、拾音器和局部细节仍有合并/错标。44/60部件召回为0，因而
即使主体面积正确，按部件等权的mIoU依然只有8.77%。不能说“吉他已修好”。

第二把吉他的编号7琴身召回97.97%，主体一直较好，但39/54部件仍完全漏掉。
同VAE参考mIoU分别39.70%与38.10%；VAE本身也不擅长这些极端细部件，DiT还有额外损失。

## 改善与退化

- 34个test对象提升超过1pp，6个下降超过1pp，其中1个下降超过5pp。
- 最大改善：`5b22289d076f47a1878392a7a6c6414f`，59.52%→77.70%，+18.18pp。
- 最大退化：台灯 `630d2463e3e446f4a952f7bee7981c93`，83.66%→78.53%，−5.13pp。
  灯罩和底座主体仍正常，标签3的IoU从95.46%降至76.50%，标签4从78.38%降至71.56%。
  面积准确率仍约98.57%，因此仅看大块颜色/面积准确率容易忽略退化。这里部分部件略大于1%
  阈值，按本评估定义该对象没有“小部件召回”数值，不能误报为小部件指标下降。

### 历史训练对象（不计入test）

| 对象 | 旧表面mIoU | 新表面mIoU |
| --- | ---: | ---: |
| 椅子 `007d2949e5b0450c995c54809b43af0f` | 91.44% | 93.60% |
| 床 `11c88bc30d37479281c30cc49d25084a` | 76.42% | 78.73% |
| 沙发 `141329918e4945ed82d0f48ba5903b8f` | 89.22% | 93.46% |

### 此前验证退化对象（不计入test）

三者复算与训练保存的EMA5000验证指标完全一致：

| 对象 | 原DiT表面mIoU | B EMA5000表面mIoU |
| --- | ---: | ---: |
| `cfa158a124a34f00a999ef7c64aa4f44` | 84.62% | 78.60% |
| `f6039123baf74ea5acaddbba3eddf61e` | 90.56% | 85.47% |
| `c98ed0dc410649209278c834a0fdc9f9` | 82.70% | 80.68% |

## 运行与产物验收

- Remote API三机24worker；所有worker和节点退出码均为0，137/137个对象完成。
- 6个validation对象（复算与退化诊断合计）相对训练验证的最大指标差异为0。
- 128个test＋3个train诊断的VAE参考逐对象指标与旧推理完全一致，汇总也完全一致。
- 411个GLB，逐对象GT/VAE/DiT各1个，导出时回读核对三角形顺序、几何及精确颜色。
- 1,096张PNG（每对象6张DiT分视图、1张三排对照、1张错误图），8张总览JPG。
- 137份数值预测NPZ；权重大小与修改时间在运行前后保持一致。
- 新增134张GT/旧DiT/新DiT/VAE四排六视图对照，PNG总数1,230，单图1536×1160。
  覆盖128test＋3train诊断＋3validation诊断，3个纯parity对象不重复比较。
- 比较前逐对象核对GT/camera像素、缓存校验和、部件标签与面积、prediction计划摘要与seed，
  test/train诊断的新旧VAE重建face/voxel数组完全一致。验证诊断的旧预测取旧checkpoint的
  `correct_s20_cfg1`，不是上次CFG=3最优设置。
- 对照工具正确区分原始NPZ preparation split与封存manifest当前split；核对源manifest摘要，
  不修改原缓存。所有本地HTML图片/GLB链接通过存在性检查。
- 推理、对照及新增诊断相关40项单元测试通过。

数值预测及运行缓存均在用户指定的快速存储，未覆写旧缓存或旧预测：

```text
/path/to/trellisseg_run/
  predictions/dit_b_smallparts_test_20261009/
  runtime_cache/dit_b_smallparts_test_20261009/
```

图片、GLB与报告：`results/dit_b_smallparts_test_20261009/`。
推理记录：该目录的 `plan.json`、`remote_launch.json`、`logs/worker*_exit.json`。

关键案例：第一把吉他（本地运行产物，未随源码发布：`../results/dit_b_smallparts_test_20261009/before_after/251c8125765149c0bd46b3a1a51ff4b6.png`）、
第二把吉他（本地运行产物，未随源码发布：`../results/dit_b_smallparts_test_20261009/before_after/d1d58f05c346470e9805612d562e9dde.png`）、
最大退化台灯（本地运行产物，未随源码发布：`../results/dit_b_smallparts_test_20261009/before_after/630d2463e3e446f4a952f7bee7981c93.png`）。

可复用命令（重新测试时使用新的output-dir，不重复启动已有结果）：

```bash
python3 -B scripts/evaluate_dit_vae_v2.py --config configs/dit_vae_v2_parts64_b_smallparts.json \
  --output-dir results/dit_b_smallparts_test_20261009 --workers 24 \
  --diagnostic-ids cfa158a124a34f00a999ef7c64aa4f44 f6039123baf74ea5acaddbba3eddf61e c98ed0dc410649209278c834a0fdc9f9 --prepare
python3 -B scripts/evaluate_dit_remote.py --output-dir results/dit_b_smallparts_test_20261009 --launch
python3 -B scripts/evaluate_dit_vae_v2.py --output-dir results/dit_b_smallparts_test_20261009 --merge
python3 -B scripts/compare_dit_runs.py --before results/dit_vae_v2_test_20261008 \
  --after results/dit_b_smallparts_test_20261009 --validation-baseline results/dit_inference_check_20261008
```

本轮只运行冻结模型推理与可视化，没有重新训练、修改推理默认CFG或基于test切换checkpoint。
