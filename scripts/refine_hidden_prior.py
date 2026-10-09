"""Preserve confident model part assignments only on unobserved mesh faces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_seg_quality import load_mesh
from scripts.refine_seg_projection import export_labels
from utils.seg_projection import collect_evidence, consistency_metrics, restore_confident_hidden
from utils.seg_graph import smooth_hidden_surfaces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--confidence', type=float, default=.8)
    ap.add_argument('--smooth_hidden', type=float, default=0.)
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error('Use an empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.source / 'report.json').read_text())
    report.update(source=str(args.source), hidden_confidence=args.confidence)
    for row in report['samples']:
        uid = row['instance']
        mesh = load_mesh(str(args.source / f'{uid}.glb'))
        data = np.load(args.source / f'{uid}_labels.npz')
        labels, parents = data['face_labels'], data['parent_faces']
        model = np.load(args.source / f'{uid}_model_labels.npz')['face_labels']
        seg = np.load(row['seg_maps'])['seg_maps']
        e = collect_evidence(mesh, seg, args.device)
        labels, detail = restore_confident_hidden(labels, model[parents], e, args.confidence)
        changed = detail['changed_faces']
        if args.smooth_hidden > 0:
            labels, detail['cleanup'] = smooth_hidden_surfaces(mesh, labels, e, args.smooth_hidden)
        score = consistency_metrics(labels, seg, e)
        assert abs(score['accuracy'] - row['exported']['accuracy']) < 1e-10
        export_labels(mesh, labels, args.output_dir / f'{uid}.glb')
        np.savez_compressed(args.output_dir / f'{uid}_labels.npz', face_labels=labels, parent_faces=parents)
        np.savez_compressed(args.output_dir / f'{uid}_model_labels.npz', face_labels=model)
        row.update(refined=score, exported=score, hidden_prior=detail,
                   source_glb_sha256=row.get('glb_sha256'),
                   glb_sha256=hashlib.sha256((args.output_dir / f'{uid}.glb').read_bytes()).hexdigest())
        (args.output_dir / f'{uid}_consistency.json').write_text(json.dumps(row, indent=2)+'\n')
        print(f'{uid}: restored {changed} unseen faces, input score unchanged', flush=True)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
