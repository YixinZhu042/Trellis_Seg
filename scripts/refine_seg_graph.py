"""Paired graph cleanup of exported projection results with hard fidelity limits."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_seg_quality import load_mesh
from scripts.refine_seg_projection import export_labels, save_comparison
from utils.seg_projection import collect_evidence, consistency_metrics
from utils.seg_graph import optimize_with_budget


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--output_dir', required=True, type=Path)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--accuracy_budget', type=float, default=.003)
    ap.add_argument('--miou_budget', type=float, default=.01)
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error('Use an empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.source / 'report.json').read_text())
    report = {'samples': [], 'failures': [], 'source': str(args.source), 'device': args.device,
              'gt_used_for_predictions': False}
    for row in original['samples']:
        uid = row['instance']
        mesh = load_mesh(str(args.source / f'{uid}.glb'))
        with np.load(args.source / f'{uid}_labels.npz') as d:
            labels, parents = d['face_labels'], d['parent_faces']
        seg = np.load(row['seg_maps'])['seg_maps']
        evidence = collect_evidence(mesh, seg, device=args.device)
        before = consistency_metrics(labels, seg, evidence)
        # Refine on the actual renderer when requested, preserving inherited
        # labels only where the production raster has no foreground evidence.
        labels[evidence.visible] = evidence.votes[evidence.visible].argmax(1)
        cleaned, diagnostics = optimize_with_budget(mesh, labels, seg, evidence,
                                                     args.accuracy_budget, args.miou_budget)
        score = consistency_metrics(cleaned, seg, evidence)
        delivered = export_labels(mesh, cleaned, args.output_dir / f'{uid}.glb')
        np.savez_compressed(args.output_dir / f'{uid}_labels.npz', face_labels=cleaned, parent_faces=parents)
        save_comparison(args.output_dir / f'{uid}_comparison.png', seg, labels, delivered, evidence)
        result = dict(row, source_projection=before, refined=score, exported=score, graph=diagnostics)
        report['samples'].append(result)
        (args.output_dir / f'{uid}_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f"{uid}: {before['accuracy']:.4f}/{before['miou']:.4f} -> {score['accuracy']:.4f}/{score['miou']:.4f}; "
              f"boundary {diagnostics['before']['boundary_length_pixels']:.0f} -> "
              f"{diagnostics['after']['boundary_length_pixels']:.0f}", flush=True)
    report['summary'] = {kind: {k: float(np.mean([r[kind][k] for r in report['samples']]))
                               for k in ['accuracy', 'miou']} for kind in ['source_projection', 'refined']}
    report['summary']['mean_boundary_ratio'] = float(np.mean([
        r['graph']['after']['boundary_length_pixels'] / max(r['graph']['before']['boundary_length_pixels'], 1e-12)
        for r in report['samples']]))
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
