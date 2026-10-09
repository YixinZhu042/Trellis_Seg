"""Paired fixed-mesh mIoU refinement of an already validated output cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_seg_quality import load_mesh
from scripts.refine_seg_projection import export_labels, save_comparison
from utils.seg_projection import collect_evidence, consistency_metrics
from utils.seg_miou import refine_part_iou


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--accuracy_budget', type=float, default=.001)
    ap.add_argument('--boundary_budget', type=float, default=.01)
    ap.add_argument('--ids', nargs='+')
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error('Use an empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source / 'report.json').read_text())
    report = {'source': str(args.source.resolve()), 'device': args.device,
              'gt_used_for_predictions': False, 'samples': [], 'failures': []}
    for row in source['samples']:
        uid = row['instance']
        if args.ids and uid not in args.ids:
            continue
        mesh = load_mesh(str(args.source / f'{uid}.glb'))
        with np.load(args.source / f'{uid}_labels.npz') as d:
            labels, parents = d['face_labels'], d['parent_faces']
        with np.load(row['seg_maps']) as d:
            seg = d['seg_maps']
        evidence = collect_evidence(mesh, seg, device=args.device)
        candidate, detail = refine_part_iou(mesh, labels, seg, evidence,
                                            args.accuracy_budget, args.boundary_budget)
        output = args.output_dir / f'{uid}.glb'
        export_labels(mesh, candidate, output)
        np.savez_compressed(args.output_dir / f'{uid}_labels.npz', face_labels=candidate, parent_faces=parents)
        with np.load(args.source / f'{uid}_model_labels.npz') as d:
            np.savez_compressed(args.output_dir / f'{uid}_model_labels.npz', face_labels=d['face_labels'])
        save_comparison(args.output_dir / f'{uid}_comparison.png', seg, labels, candidate, evidence)
        exported = collect_evidence(load_mesh(str(output)), seg, device=args.device)
        result = {**row, 'miou_refinement': detail, 'refined': detail['after'],
                  'exported': consistency_metrics(candidate, seg, exported),
                  'source_glb_sha256': row['glb_sha256'],
                  'glb_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
        report['samples'].append(result)
        (args.output_dir / f'{uid}_consistency.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f"{uid}: acc {detail['before']['accuracy']:.5f}->{detail['after']['accuracy']:.5f}, "
              f"mIoU {detail['before']['miou']:.5f}->{detail['after']['miou']:.5f}, "
              f"changed={detail['changed_faces']}, accepted={detail['accepted']}", flush=True)
    report['summary'] = {'num_samples': len(report['samples'])}
    for kind in ['before', 'after']:
        report['summary'][kind] = {
            k: float(np.mean([r['miou_refinement'][kind][k] for r in report['samples']]))
            for k in ['accuracy', 'miou']}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
