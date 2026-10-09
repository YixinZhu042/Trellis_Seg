"""Summarize paired raw-output validation/test evidence for decoded DiT losses."""
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


def paired_comparison(left, right):
    """Average seeds within objects, then bootstrap objects (not seed replicas)."""
    left_map = {(r['instance'], r['seed']): r for r in left}
    right_map = {(r['instance'], r['seed']): r for r in right}
    if len(left_map) != len(left) or len(right_map) != len(right) or left_map.keys() != right_map.keys():
        raise ValueError('Evaluation object/seed sets differ or contain duplicates')
    output = {}
    for group, name in METRICS:
        per_object = {}
        for key in sorted(left_map):
            a, b = value(left_map[key], group, name), value(right_map[key], group, name)
            if (a is None) != (b is None):
                raise ValueError(f'Metric validity differs: {key}, {group}/{name}')
            if a is not None:
                if not np.isfinite([a, b]).all():
                    raise ValueError('Nonfinite metric')
                per_object.setdefault(key[0], []).append(b-a)
        delta = np.array([np.mean(v) for v in per_object.values()])
        key = f'{group}.{name}' if group else name
        if not len(delta):
            output[key] = {'objects': 0, 'mean_delta': None}
            continue
        rng = np.random.default_rng(20260921)
        means = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
        output[key] = {'objects': len(delta), 'mean_delta': float(delta.mean()),
                       'ci95': np.quantile(means, [.025, .975]).tolist(),
                       'improved': int((delta > 1e-10).sum()),
                       'regressed': int((delta < -1e-10).sum()),
                       'unchanged': int((abs(delta) <= 1e-10).sum()),
                       'per_object_delta': {uid: float(np.mean(v)) for uid, v in per_object.items()}}
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation_dir', type=Path, required=True)
    ap.add_argument('--test_dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    validation = json.loads((args.validation_dir / 'sweep.json').read_text())
    test = json.loads((args.test_dir / 'sweep.json').read_text())
    if validation['failures'] or test['failures']:
        raise ValueError('Cannot summarize a sweep containing failed jobs')
    runs = {name: json.loads((args.test_dir / name / 'metrics.json').read_text())
            for name in ['pretrained', 'flow', 'decoded']}
    if any(d['postprocessing'] for d in runs.values()):
        raise ValueError('Comparison requires raw model outputs')
    # Confirm that every arm used the same VAE and evaluation geometry via
    # their independently computed oracle scores, up to numerical tolerance.
    oracles = {name: {r['instance']: r for r in d['oracle']} for name, d in runs.items()}
    for name in ['flow', 'decoded']:
        if oracles[name].keys() != oracles['pretrained'].keys():
            raise ValueError('Oracle object sets differ')
        for uid, reference in oracles['pretrained'].items():
            for group, metric in METRICS:
                a, b = value(reference, group, metric), value(oracles[name][uid], group, metric)
                if (a is None) != (b is None) or (a is not None and abs(a-b) > 1e-5):
                    raise ValueError(f'Oracle mismatch: {name}, {uid}, {group}/{metric}')
    report = {'validation_protocol': str(args.validation_dir / 'protocol.json'),
              'test_protocol': str(args.test_dir / 'protocol.json'), 'selection': test['selection'],
              'validation': [{k: r[k] for k in ['name', 'summary']} for r in validation['results']],
              'test_summaries': {name: d['summary'] for name, d in runs.items()},
              'oracle_summary': runs['pretrained']['oracle_summary'],
              'decoded_vs_flow': paired_comparison(runs['flow']['samples'], runs['decoded']['samples']),
              'decoded_vs_pretrained': paired_comparison(runs['pretrained']['samples'], runs['decoded']['samples']),
              'flow_vs_pretrained': paired_comparison(runs['pretrained']['samples'], runs['flow']['samples']),
              'postprocessing': False,
              'caveat': '32 test objects are excluded from continuation and selection, not necessarily original pretraining. No out-of-distribution claim.'}
    against_original = report['decoded_vs_pretrained']
    below_original = all(against_original[k]['ci95'][1] < 0
                         for k in ['projection.miou', 'surface.miou'])
    report['decision'] = {
        'default_checkpoint': 'retain_original' if below_original else 'requires_further_review',
        'reason': ('Decoded auxiliary losses improve over matched flow-only continuation, but projection '
                   'and surface mIoU remain below the original checkpoint on the held-out continuation test split.'
                   if below_original else 'Review all paired metrics before selecting a production checkpoint.'),
        'automatic_checkpoint_replacement': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'test': report['test_summaries'],
                      'decoded_vs_flow': {k: {q:v[q] for q in v if q!='per_object_delta'}
                                          for k,v in report['decoded_vs_flow'].items()}}, indent=2))


if __name__ == '__main__':
    main()
