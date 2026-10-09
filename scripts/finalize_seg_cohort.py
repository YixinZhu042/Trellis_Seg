"""Replay current shared postprocessing on raw labels from completed neural runs.

This avoids resampling the DiT while verifying that final artifacts use exactly
the deployed refinement function, from the original mesh through GLB export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_seg_quality import load_mesh
from scripts.refine_seg_projection import export_labels, save_comparison
from utils.seg_projection import collect_evidence, consistency_metrics, refine_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory')
    source = json.loads((args.source / 'report.json').read_text())
    if source.get('failures'):
        parser.error('Source cohort contains failed neural runs')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'source': str(args.source.resolve()), 'mode': 'shared_refinement_replay',
              'gt_used_for_predictions': False, 'device': args.device,
              'settings': {'subdivide': 5, 'graph_cleanup': True, 'accuracy_budget': .003,
                           'miou_budget': .01, 'hidden_confidence': .8, 'hidden_smooth': .25,
                           'part_iou_refine': True},
              'samples': [], 'failures': []}
    for sample in source['samples']:
        uid = sample['instance']
        mesh = load_mesh(sample['mesh'])
        label_path = args.source / f'{uid}_model_labels.npz'
        with np.load(label_path) as data:
            model_labels = data['face_labels']
        with np.load(sample['seg_maps']) as data:
            seg = data['seg_maps']
        refined, labels, parents, evidence, baseline, metrics = refine_projection(
            mesh, model_labels, seg, device=args.device, **report['settings'])
        output = args.output_dir / f'{uid}.glb'
        delivered = export_labels(refined, labels, output)
        np.savez_compressed(args.output_dir / f'{uid}_labels.npz', face_labels=labels, parent_faces=parents)
        np.savez_compressed(args.output_dir / f'{uid}_model_labels.npz', face_labels=model_labels)
        exported = collect_evidence(load_mesh(str(output)), seg, device=args.device)
        metrics['exported'] = consistency_metrics(delivered, seg, exported)
        save_comparison(args.output_dir / f'{uid}_comparison.png', seg, baseline, labels, evidence)
        row = {**sample, **metrics, 'glb_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'source_raw_labels_sha256': hashlib.sha256(label_path.read_bytes()).hexdigest()}
        report['samples'].append(row)
        (args.output_dir / f'{uid}_consistency.json').write_text(json.dumps(row, indent=2) + '\n')
        print(f"{uid}: acc={metrics['exported']['accuracy']:.5f}, mIoU={metrics['exported']['miou']:.5f}", flush=True)
    report['summary'] = {'num_samples': len(report['samples']), 'num_failures': 0}
    for kind in ['baseline', 'exported']:
        report['summary'][kind] = {k: float(np.mean([r[kind][k] for r in report['samples']]))
                                  for k in ['accuracy', 'miou', 'gap_to_ceiling']}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
