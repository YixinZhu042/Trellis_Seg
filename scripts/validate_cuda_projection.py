"""Compare the delivered CPU results with the production CUDA rasterizer."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_seg_quality import load_mesh, recover_face_labels_from_colors
from utils.seg_projection import collect_evidence, consistency_metrics, render_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    source = json.loads((args.prediction_dir / 'report.json').read_text())
    report = {'samples': [], 'device': args.device}
    for sample in source['samples']:
        uid = sample['instance']
        mesh = load_mesh(str(args.prediction_dir / f'{uid}.glb'))
        labels = recover_face_labels_from_colors(str(args.prediction_dir / f'{uid}.glb'), 20)
        seg = np.load(sample['seg_maps'])['seg_maps']
        cpu = collect_evidence(mesh, seg)
        cuda = collect_evidence(mesh, seg, device=args.device)
        overlap = (cpu.face_ids >= 0) & (cuda.face_ids >= 0)
        union = (cpu.face_ids >= 0) | (cuda.face_ids >= 0)
        cpu_labels, cuda_labels = render_labels(labels, cpu.face_ids), render_labels(labels, cuda.face_ids)
        row = {'instance': uid, 'cpu': consistency_metrics(labels, seg, cpu),
               'cuda': consistency_metrics(labels, seg, cuda),
               'silhouette_iou': float(overlap.sum() / union.sum()),
               'face_agreement': float(np.mean(cpu.face_ids[overlap] == cuda.face_ids[overlap])),
               'label_agreement': float(np.mean(cpu_labels[overlap] == cuda_labels[overlap]))}
        report['samples'].append(row)
        print(f"{uid}: CPU acc={row['cpu']['accuracy']:.5f}, CUDA acc={row['cuda']['accuracy']:.5f}, "
              f"face agreement={row['face_agreement']:.5f}, label agreement={row['label_agreement']:.5f}", flush=True)
    report['summary'] = {kind: {k: float(np.mean([r[kind][k] for r in report['samples']]))
                               for k in ['accuracy', 'miou', 'fixed_mesh_accuracy_ceiling', 'gap_to_ceiling']}
                         for kind in ['cpu', 'cuda']}
    for k in ['face_agreement', 'label_agreement', 'silhouette_iou']:
        report['summary'][k] = float(np.mean([r[k] for r in report['samples']]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
