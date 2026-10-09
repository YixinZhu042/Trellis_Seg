"""Summarize full validation selection and paired test evidence."""
import argparse
import json
from pathlib import Path

import numpy as np


METRICS = [('projection', 'miou'), ('surface', 'miou'), ('voxel', 'miou'),
           ('projection', 'accuracy'), ('surface', 'accuracy'),
           ('surface', 'small_part_recall'), ('voxel', 'small_part_recall'),
           (None, 'boundary_f1')]


def value(row, group, name):
    return row[group][name] if group else row[name]


def paired_comparison(reference, candidate):
    left = {(row['instance'], row['seed']): row for row in reference}
    right = {(row['instance'], row['seed']): row for row in candidate}
    if left.keys() != right.keys() or len(left) != len(reference) or len(right) != len(candidate):
        raise ValueError('Test object/seed sets differ or contain duplicates')
    output = {}
    for group, name in METRICS:
        per_object = {}
        for key in sorted(left):
            a, b = value(left[key], group, name), value(right[key], group, name)
            if (a is None) != (b is None):
                raise ValueError(f'Metric validity differs for {key}: {group}/{name}')
            if a is not None:
                per_object.setdefault(key[0], []).append(b - a)
        delta = np.array([np.mean(values) for values in per_object.values()])
        metric = f'{group}.{name}' if group else name
        if not len(delta):
            output[metric] = {'objects': 0, 'mean_delta': None}
            continue
        rng = np.random.default_rng(20260921)
        means = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
        output[metric] = {'objects': len(delta), 'mean_delta': float(delta.mean()),
                          'ci95': np.quantile(means, [.025, .975]).tolist(),
                          'improved': int((delta > 1e-10).sum()),
                          'regressed': int((delta < -1e-10).sum()),
                          'unchanged': int((np.abs(delta) <= 1e-10).sum())}
    return output


def percentage(value):
    return '—' if value is None else f'{100 * value:.3f}%'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    selection = json.loads((args.root / 'selection.json').read_text())
    original = json.loads((args.root / 'test/pretrained/metrics.json').read_text())
    selected = json.loads((args.root / 'test/selected/metrics.json').read_text())
    if original['split'] != 'test' or selected['split'] != 'test':
        raise ValueError('Expected test metrics')
    if original['seeds'] != [42, 123, 2026] or selected['seeds'] != [42, 123, 2026]:
        raise ValueError('Expected fixed three-seed test protocol')
    if len(original['samples']) != 1536 or len(selected['samples']) != 1536:
        raise ValueError('Expected 512 test objects x 3 seeds')
    # The VAE oracle must be identical for both model checkpoints.
    a = {row['instance']: row for row in original['oracle']}
    b = {row['instance']: row for row in selected['oracle']}
    if a.keys() != b.keys():
        raise ValueError('VAE oracle object sets differ')
    for uid in a:
        for group, name in METRICS:
            x, y = value(a[uid], group, name), value(b[uid], group, name)
            if (x is None) != (y is None) or (x is not None and abs(x-y) > 1e-5):
                raise ValueError(f'VAE oracle mismatch: {uid}, {group}/{name}')
    comparison = paired_comparison(original['samples'], selected['samples'])
    report = {'selection': selection, 'test_summaries': {
                  'pretrained': original['summary'],
                  selection['selected_continuation']['name']: selected['summary']},
              'oracle_summary': original['oracle_summary'],
              'selected_vs_pretrained': comparison,
              'postprocessing': False, 'label_matching': False,
              'test_used_for_selection': False,
              'caveat': 'Continuation validation/test objects may have appeared in original pretraining.'}
    (args.root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    names = [('Original step-50000 EMA', original['summary']),
             (selection['selected_continuation']['name'], selected['summary']),
             ('Frozen VAE oracle', original['oracle_summary'])]
    lines = ['# 全量固定 VAE + DiT 辅助损失评估', '',
             f"Validation 选择的续训 checkpoint：`{selection['selected_continuation']['name']}`。",
             f"Validation 总体最佳：`{selection['best_overall_on_validation']['name']}`。", '',
             '## Validation checkpoint 选择', '',
             '| Checkpoint | Projection mIoU | Surface mIoU | Boundary F1 |',
             '| --- | ---: | ---: | ---: |']
    for row in selection['results']:
        s = row['summary']
        lines.append(f"| {row['name']} | {percentage(s['projection']['miou'])} | "
                     f"{percentage(s['surface']['miou'])} | {percentage(s['boundary_f1'])} |")
    lines += ['', '## Test：512 对象 × 3 seeds', '',
              '| Model | Projection mIoU | Surface mIoU | Voxel mIoU | Small-part recall | Boundary F1 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label, s in names:
        lines.append(f"| {label} | {percentage(s['projection']['miou'])} | {percentage(s['surface']['miou'])} | "
                     f"{percentage(s['voxel']['miou'])} | {percentage(s['surface']['small_part_recall'])} | "
                     f"{percentage(s['boundary_f1'])} |")
    lines += ['', '## Selected continuation − original', '',
              '| Metric | Mean delta | 95% object-bootstrap CI | Improved / regressed / unchanged |',
              '| --- | ---: | --- | --- |']
    for metric, row in comparison.items():
        if row['mean_delta'] is None:
            continue
        lo, hi = row['ci95']
        lines.append(f"| {metric} | {100*row['mean_delta']:+.3f} pp | "
                     f"[{100*lo:+.3f}, {100*hi:+.3f}] | "
                     f"{row['improved']} / {row['regressed']} / {row['unchanged']} |")
    lines += ['', '所有指标使用固定原始标签 ID，关闭后处理；test 未参与 checkpoint 选择。']
    (args.root / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'selected': selection['selected_continuation']['name'],
                      'test': report['test_summaries'],
                      'projection_delta': comparison['projection.miou'],
                      'surface_delta': comparison['surface.miou'],
                      'boundary_delta': comparison['boundary_f1']}, indent=2))


if __name__ == '__main__':
    main()
