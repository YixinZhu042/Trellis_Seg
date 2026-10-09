"""Refine frozen predictions with cached input views, without model/API/GT access.

Single: --mesh INPUT --prediction PRED --seg_maps MAPS --output_dir OUT
Batch:  --mesh_dir DIR --pred_dir DIR --seg_root /tmp/po_run --output_dir OUT
Writes colored GLBs, exact face labels, paired 2D metrics and comparison images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.seg_projection import (
    collect_evidence, consistency_metrics, refine_projection,
    render_labels, validate_seg_maps, VIEW_NAMES,
)
from scripts.evaluate_seg_quality import generate_class_colors, load_mesh, recover_face_labels_from_colors


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_labels(mesh, labels, path):
    palette = (generate_class_colors(int(labels.max()) + 1) * 255).astype(np.uint8)
    colors = np.column_stack([palette[labels], np.full(len(labels), 255, np.uint8)])
    vertices = mesh.triangles.reshape(-1, 3)
    result = trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3),
                             vertex_colors=np.repeat(colors, 3, axis=0), process=False)
    result.export(str(path))
    # Reload the delivered artifact to verify labels and face order, not just arrays.
    recovered = recover_face_labels_from_colors(str(path), len(palette), len(labels))
    if not np.array_equal(recovered, labels):
        raise ValueError('GLB label/color roundtrip failed')
    loaded = load_mesh(str(path))
    if not np.allclose(mesh.triangles, loaded.triangles, rtol=1e-6,
                       atol=max(float(np.max(mesh.extents)), 1e-12) * 1e-6):
        raise ValueError('GLB export changed face geometry or order')
    return recovered


def save_comparison(path, seg_maps, baseline, refined, evidence):
    palette = (generate_class_colors(evidence.votes.shape[1]) * 255).astype(np.uint8)
    palette[0] = 255
    panels = [seg_maps, render_labels(baseline, evidence.face_ids), render_labels(refined, evidence.face_ids)]
    width = 256
    height = max(1, round(seg_maps.shape[1] * width / seg_maps.shape[2]))
    canvas = Image.new('RGB', (6 * width, 3 * (height + 24)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (title, maps) in enumerate(zip(('Input 2D', 'Baseline (global ID alignment)', 'Refined'), panels)):
        for view, name in enumerate(VIEW_NAMES):
            x, y = view * width, row * (height + 24)
            draw.text((x + 4, y + 5), f'{title}: {name}', fill='black')
            im = Image.fromarray(palette[maps[view]]).resize((width, height), Image.Resampling.NEAREST)
            canvas.paste(im, (x, y + 24))
    canvas.save(path)


def refine_one(mesh_path, pred_path, seg_path, output_dir, device='cpu', subdivide=0,
               graph_cleanup=False, accuracy_budget=.003, miou_budget=.01,
               hidden_confidence=.8, hidden_smooth=.25, part_iou_refine=True):
    uid = Path(pred_path).stem
    mesh = load_mesh(str(mesh_path))
    pred_mesh = load_mesh(str(pred_path))
    if len(mesh.faces) != len(pred_mesh.faces) or not np.allclose(
            mesh.triangles, pred_mesh.triangles, rtol=1e-6,
            atol=max(float(np.max(mesh.extents)), 1e-12) * 1e-6):
        raise ValueError('Input and prediction mesh geometry/face order differ')
    model_labels = recover_face_labels_from_colors(str(pred_path), 20, len(mesh.faces))
    with np.load(seg_path) as data:
        seg_maps = validate_seg_maps(data['seg_maps'], max_parts=19)
    mesh, labels, parent_faces, evidence, baseline, result = refine_projection(
        mesh, model_labels, seg_maps, device=device, subdivide=subdivide,
        graph_cleanup=graph_cleanup, accuracy_budget=accuracy_budget, miou_budget=miou_budget,
        hidden_confidence=hidden_confidence, hidden_smooth=hidden_smooth, part_iou_refine=part_iou_refine)
    delivered = export_labels(mesh, labels, output_dir / f'{uid}.glb')
    np.savez_compressed(output_dir / f'{uid}_labels.npz', face_labels=labels, parent_faces=parent_faces)
    save_comparison(output_dir / f'{uid}_comparison.png', seg_maps, baseline, delivered, evidence)
    result.update({
        'instance': uid, 'mesh': str(mesh_path), 'prediction': str(pred_path),
        'seg_maps': str(seg_path), 'mesh_sha256': sha256(mesh_path),
        'prediction_sha256': sha256(pred_path), 'seg_maps_sha256': sha256(seg_path),
    })
    # Re-rasterize the delivered float32 GLB, not just the in-memory mesh.
    exported_mesh = load_mesh(str(output_dir / f'{uid}.glb'))
    exported_evidence = collect_evidence(exported_mesh, seg_maps, device=device)
    result['exported'] = consistency_metrics(delivered, seg_maps, exported_evidence)
    # Existing occupancy is independently produced by the actual GPU renderer.
    occupancy_path = Path(seg_path).parent.parent / 'renders/occupancy.npy'
    if occupancy_path.exists():
        occupancy = np.load(occupancy_path).astype(bool)
        if occupancy.shape == evidence.face_ids.shape:
            hit = evidence.face_ids >= 0
            result['cached_render_silhouette_iou'] = float((hit & occupancy).sum() / (hit | occupancy).sum())
    with (output_dir / f'{uid}_metrics.json').open('w') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh')
    parser.add_argument('--prediction')
    parser.add_argument('--seg_maps')
    parser.add_argument('--mesh_dir')
    parser.add_argument('--pred_dir')
    parser.add_argument('--seg_root', default='/tmp/po_run')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--subdivide', type=int, default=5, help='Adaptive boundary subdivision rounds (default 5; 0 preserves face count)')
    parser.add_argument('--cleanup', choices=['graph', 'ties'], default='graph')
    parser.add_argument('--accuracy_budget', type=float, default=.003)
    parser.add_argument('--miou_budget', type=float, default=.01)
    parser.add_argument('--part_iou_refine', choices=['on', 'off'], default='on')
    parser.add_argument('--hidden_prior_confidence', type=float, default=.8)
    parser.add_argument('--hidden_smooth', type=float, default=.25)
    args = parser.parse_args()
    if not 0 <= args.subdivide <= 5:
        parser.error('--subdivide must be between 0 and 5')
    if not 0 <= args.accuracy_budget <= 1 or not 0 <= args.miou_budget <= 1:
        parser.error('Fidelity budgets must be between 0 and 1')
    if not 0 <= args.hidden_prior_confidence <= 1:
        parser.error('--hidden_prior_confidence must be between 0 and 1')
    if not np.isfinite(args.hidden_smooth) or args.hidden_smooth < 0:
        parser.error('--hidden_smooth must be finite and nonnegative')
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new or empty output directory to preserve paired experiments')
    single = (args.mesh, args.prediction, args.seg_maps)
    if all(single) and not (args.mesh_dir or args.pred_dir):
        inputs = [tuple(Path(p).resolve() for p in single)]
    elif args.mesh_dir and args.pred_dir and not any(single):
        inputs = [(Path(args.mesh_dir).resolve() / p.name, p,
                   Path(args.seg_root).resolve() / p.stem / 'seg/seg_maps.npz')
                  for p in sorted(Path(args.pred_dir).resolve().glob('*.glb'))]
    else:
        parser.error('Provide the three single-sample paths or --mesh_dir and --pred_dir')
    if not inputs:
        parser.error('No predictions found')
    output.mkdir(parents=True, exist_ok=True)
    report = {'gt_used_for_predictions': False, 'device': args.device, 'subdivide': args.subdivide,
              'cleanup': args.cleanup, 'accuracy_budget': args.accuracy_budget,
              'miou_budget': args.miou_budget, 'hidden_prior_confidence': args.hidden_prior_confidence,
              'hidden_smooth': args.hidden_smooth, 'part_iou_refine': args.part_iou_refine,
              'samples': [], 'failures': []}
    for mesh, pred, seg in inputs:
        try:
            result = refine_one(mesh, pred, seg, output, args.device, args.subdivide,
                                args.cleanup == 'graph', args.accuracy_budget, args.miou_budget,
                                args.hidden_prior_confidence, args.hidden_smooth, args.part_iou_refine == 'on')
            report['samples'].append(result)
            print(f"{pred.stem}: acc {result['baseline']['accuracy']:.4f} -> "
                  f"{result['refined']['accuracy']:.4f}; mIoU {result['baseline']['miou']:.4f} -> "
                  f"{result['refined']['miou']:.4f}; ceiling {result['refined']['fixed_mesh_accuracy_ceiling']:.4f}", flush=True)
        except Exception as exc:
            report['failures'].append({'instance': pred.stem, 'error': str(exc)})
            print(f'FAIL {pred.stem}: {exc}', flush=True)
    report['summary'] = {'num_samples': len(report['samples'])}
    for kind in ('baseline', 'refined'):
        report['summary'][kind] = {key: float(np.mean([r[kind][key] for r in report['samples']]))
                                 for key in ('accuracy', 'miou', 'foreground_coverage',
                                             'fixed_mesh_accuracy_ceiling', 'gap_to_ceiling')} if report['samples'] else {}
    with (output / 'report.json').open('w') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report['summary'], indent=2))
    if report['failures']:
        raise RuntimeError(f"{len(report['failures'])} samples failed; see report.json")


if __name__ == '__main__':
    main()
