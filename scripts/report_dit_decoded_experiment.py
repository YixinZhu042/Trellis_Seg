"""Write a reviewable experiment report and raw-output visual comparisons."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_seg_quality import generate_class_colors
from scripts.refine_seg_projection import export_labels
from utils.seg_projection import render_labels, VIEW_NAMES


def percentage(value):
    return '—' if value is None else f'{value*100:.3f}%'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--test_dir', type=Path, required=True)
    ap.add_argument('--output_dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.comparison.read_text())
    training = json.loads((ROOT / 'eval/dit_decoded_training_audit.json').read_text())
    labels = {'pretrained': 'Original', 'flow': 'Flow-only continuation', 'decoded': 'Decoded-loss continuation'}
    text = ['# 固定 VAE + DiT 解码任务监督：实验结果', '',
            '两组均从相同 DiT checkpoint 续训，VAE 编码器/解码器固定。以下全部为原始模型输出，关闭投影覆盖、图优化和其他标签后处理。', '',
            '**结论：本轮保留原始 checkpoint，不替换默认模型。** 新增任务损失相对相同条件的纯 MSE 续训有小幅改善，但整体表现仍低于原始模型。', '',
            '本次共完成 7 组验证评估（32 对象 × 1 种子）和 3 组测试评估（32 对象 × 3 种子），无失败。', '',
            '## 训练核验', '',
            '- 两机共 16 卡，global batch 16，各 300 步、4,800 个对象次。',
            '- 逐 rank 核对样例顺序、t、条件丢弃与配置一致。',
            '- VAE 前后摘要一致；实验组 16/16 rank 的三项辅助梯度均非零。',
            f"- 训练耗时：对照 {training['training_summary']['flow']['seconds']/60:.1f} 分钟，实验组 {training['training_summary']['decoded']['seconds']/60:.1f} 分钟。", '',
            '## 验证集选型', '',
            '按 32 个 validation 对象的投影 mIoU 选择；同分依次看表面 mIoU、更早步数。测试集不参与选型。', '',
            '| 检查点 | 投影 mIoU | 表面 mIoU | 边界 F1 |', '| --- | ---: | ---: | ---: |']
    for run in report['validation']:
        s = run['summary']
        text.append(f"| {run['name']} | {percentage(s['projection']['miou'])} | {percentage(s['surface']['miou'])} | {percentage(s['boundary_f1'])} |")
    text += ['', f"选中：flow step {report['selection']['flow']['step']}；decoded step {report['selection']['decoded']['step']}。", '',
             '## 测试集：32 个对象，每个对象 3 个种子', '',
             '| 模型 | 投影 mIoU | 表面 mIoU | 体素 mIoU | 表面小部件召回 | 边界 F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in ['pretrained', 'flow', 'decoded']:
        s = report['test_summaries'][name]
        text.append(f"| {labels[name]} | {percentage(s['projection']['miou'])} | {percentage(s['surface']['miou'])} | "
                    f"{percentage(s['voxel']['miou'])} | {percentage(s['surface']['small_part_recall'])} | {percentage(s['boundary_f1'])} |")
    text += ['', '## 新增损失的配对增益：decoded − flow', '',
             '先对每个对象的 3 个种子取平均，再对对象做 10,000 次 bootstrap。区间单位为百分点。', '',
             '| 指标 | 平均差值 | 95% 区间 | 提升 / 回退 / 不变对象 |', '| --- | ---: | --- | --- |']
    for key, data in report['decoded_vs_flow'].items():
        if data['mean_delta'] is None:
            continue
        lo, hi = data['ci95']
        text.append(f"| {key} | {data['mean_delta']*100:+.3f} | [{lo*100:+.3f}, {hi*100:+.3f}] | "
                    f"{data['improved']} / {data['regressed']} / {data['unchanged']} |")
    text += ['', '## 相对原始模型：decoded − pretrained', '',
             '| 指标 | 平均差值（百分点） | 95% 区间 |', '| --- | ---: | --- |']
    for key in ['projection.miou', 'surface.miou', 'boundary_f1']:
        data = report['decoded_vs_pretrained'][key]
        lo, hi = data['ci95']
        text.append(f"| {key} | {data['mean_delta']*100:+.3f} | [{lo*100:+.3f}, {hi*100:+.3f}] |")
    text += ['', '## 对结果的解释与下一轮建议', '',
             '两组续训都低于原始模型，说明不能把本轮回退直接归因于新增损失。新增损失缓解了回退，但没有带来超过原始模型的净收益。', '',
             '可确认的事实是：训练集只有 256 个对象，300 步约重复 18.75 遍；原始模型是 EMA 权重，当前续训输出为原始优化器权重；新增损失门控后在末 30 步约占总 flow 损失数值的 3.9%。这些都值得排查，但尚未通过消融确定因果。', '',
             '下一轮建议先扩大对齐训练数据并保留独立测试集，同时加入 EMA、原始数据 replay/教师约束，降低续训漂移；再分别测试 segmentation、boundary、reprojection，而不是直接增加本轮训练步数。', '',
             '小部件表面召回仅 7 个测试对象有定义，边界 F1 仅 20 个对象有定义，不能将它们当作 32 个对象都支持的稳定结论。', '',
             '本轮未自动启动新训练，也未修改默认推理 checkpoint。']
    text += ['', '## 协议范围', '',
             '- 本轮 train/validation/test 为对象级独立划分，但原模型预训练可能见过这些对象。',
             '- 标签采用固定原始 ID，不做 Hungarian 匹配；label 0 为未标注表面，不纳入任务指标。',
             '- 边界 F1 为原始网格邻接边的精确边界，具有真实边界的对象才计入均值。',
             '- 2D 条件从同一 3D 标注渲染，因此本实验衡量干净监督下的任务能力；不能直接外推到冲突的 API 分割。',
             '- 三项损失同时启用；该实验不能单独识别分割、边界或重投影中哪项产生收益。',
             '- 训练样例仅 256 个，采用约 18.75 个对象遍历；不能据此宣布大规模泛化或正式替换生产权重。', '',
             '## 视觉诊断', '',
             '下面按测试集投影 mIoU 的配对变化，展示两个提升最多和两个回退最多的对象。它们仅用于诊断，不用于重新选 checkpoint。', '']
    delta = report['decoded_vs_flow']['projection.miou']['per_object_delta']
    ordered = sorted(delta, key=delta.get)
    selected = list(dict.fromkeys(ordered[:2] + ordered[-2:]))
    palette = (generate_class_colors(21) * 255).astype(np.uint8)
    palette[0] = 255
    # Input image IDs are model labels + 1, including unannotated mesh class 0.
    palette[1:] = (generate_class_colors(20) * 255).astype(np.uint8)
    for uid in selected:
        with np.load(ROOT / 'data/dit_decoded_v1/samples' / f'{uid}.npz') as data:
            raster, seg = data['face_raster'], data['seg_maps']
            mesh = trimesh.Trimesh(vertices=data['mesh_vertices'], faces=data['mesh_faces'], process=False)
        maps = [seg]
        names = ['Input 2D']
        for name in ['pretrained', 'flow', 'decoded']:
            with np.load(args.test_dir / name / f'{uid}_seed42.npz') as prediction:
                faces = prediction['face_labels']
            maps.append(render_labels(faces + 1, raster))
            names.append(labels[name])
            export_labels(mesh, faces, args.output_dir / f'{uid}_{name}.glb')
        size = 256
        canvas = Image.new('RGB', (size*6, (size+24)*4), 'white')
        draw = ImageDraw.Draw(canvas)
        for row, (label, images) in enumerate(zip(names, maps)):
            for view, view_name in enumerate(VIEW_NAMES):
                x, y = size*view, (size+24)*row
                draw.text((x+4, y+5), f'{label}: {view_name}', fill='black')
                panel = Image.fromarray(palette[images[view]]).resize((size, size), Image.Resampling.NEAREST)
                canvas.paste(panel, (x, y+24))
        canvas.save(args.output_dir / f'{uid}_comparison.png')
        text.append(f"- `{uid}`：平均投影 mIoU 变化 {delta[uid]*100:+.3f} pp；[对比图]({uid}_comparison.png)，"
                    f"[原始]({uid}_pretrained.glb) / [对照]({uid}_flow.glb) / [实验组]({uid}_decoded.glb)。")
    (args.output_dir / 'README.md').write_text('\n'.join(text) + '\n')
    print(f'Wrote {args.output_dir / "README.md"} and four paired visual diagnostics')


if __name__ == '__main__':
    main()
