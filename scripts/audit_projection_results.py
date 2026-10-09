"""Read-only artifact audit: reproject exported GLBs and check parent surface mapping.

Optional --gt_root evaluates PartObjaverse surface metrics on inherited parent GT.
GT is accessed here only, after predictions have been saved. Face-count metrics
are omitted because adaptive subdivision changes their weighting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.seg_projection import align_model_labels, collect_evidence, consistency_metrics
from scripts.evaluate_seg_quality import load_mesh, recover_face_labels_from_colors, region_metrics


def closest_points(triangles, points):
    """Triangle projection with explicit segment fallback for degenerate faces."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normal = np.cross(b - a, c - a)
    norm2 = np.einsum('ij,ij->i', normal, normal)
    projected = points - normal * (np.einsum('ij,ij->i', points - a, normal) /
                                  np.maximum(norm2, 1e-300))[:, None]
    wb = np.einsum('ij,ij->i', np.cross(projected - a, c - a), normal) / np.maximum(norm2, 1e-300)
    wc = np.einsum('ij,ij->i', np.cross(b - a, projected - a), normal) / np.maximum(norm2, 1e-300)
    inside = (norm2 > 1e-30) & (wb >= -1e-10) & (wc >= -1e-10) & (wb + wc <= 1 + 1e-10)
    result = a.copy()
    best = np.full(len(points), np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = np.einsum('ij,ij->i', points - start, edge) / np.maximum(np.einsum('ij,ij->i', edge, edge), 1e-300)
        candidate = start + np.clip(t, 0, 1)[:, None] * edge
        distance = np.linalg.norm(points - candidate, axis=1)
        replace = distance < best
        result[replace], best[replace] = candidate[replace], distance[replace]
    result[inside] = projected[inside]
    return result


def audit_sample(sample, prediction_dir, gt_root=None, device='cpu', comparisons=False):
    uid = sample['instance']
    original = load_mesh(sample['mesh'])
    exported = load_mesh(str(prediction_dir / f'{uid}.glb'))
    with np.load(prediction_dir / f'{uid}_labels.npz') as d:
        labels, parents = d['face_labels'], d['parent_faces']
    if parents.shape != (len(exported.faces),) or parents.min() < 0 or parents.max() >= len(original.faces):
        raise ValueError('Invalid parent face indices')
    recovered = recover_face_labels_from_colors(str(prediction_dir / f'{uid}.glb'), 20, len(labels))
    np.testing.assert_array_equal(labels, recovered)

    # Every exported vertex must lie on its own parent triangle, and children
    # must preserve each parent's total area and orientation.
    extent = max(float(np.max(original.extents)), 1e-12)
    max_distance = 0.
    for start in range(0, len(parents), 20000):
        end = min(start + 20000, len(parents))
        points = exported.triangles[start:end].reshape(-1, 3)
        triangles = np.repeat(original.triangles[parents[start:end]], 3, axis=0)
        # Normalize each parent for stable tests on small and skinny triangles.
        origin = triangles[:, 0].copy()
        scale = np.maximum(np.linalg.norm(triangles - origin[:, None], axis=2).max(1), 1e-15)
        local_triangles = (triangles - origin[:, None]) / scale[:, None, None]
        local_points = (points - origin) / scale[:, None]
        nearest = closest_points(local_triangles, local_points) * scale[:, None] + origin
        if not np.isfinite(nearest).all():
            raise ValueError('Nonfinite nearest-surface point')
        max_distance = max(max_distance, float(np.linalg.norm(nearest - points, axis=1).max() / extent))
    if max_distance > 2e-6:
        raise ValueError(f'Exported vertices left original surface: {max_distance}')
    child_areas = np.bincount(parents, weights=exported.area_faces, minlength=len(original.faces))
    relative_area_error = float(np.abs(child_areas - original.area_faces).sum() / original.area)
    if relative_area_error > 1e-5:
        raise ValueError(f'Parent areas changed: {relative_area_error}')
    substantial = exported.area_faces > original.area * 1e-12
    def stable_normals(triangles):
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-30)
    normal_dot = np.einsum('ij,ij->i', stable_normals(exported.triangles[substantial]),
                           stable_normals(original.triangles[parents[substantial]]))
    if len(normal_dot) and normal_dot.min() < 0.99:
        raise ValueError('Child face orientation changed')
    with np.load(sample['seg_maps']) as data:
        seg_maps = data['seg_maps']
    evidence = collect_evidence(exported, seg_maps, device=device)
    metrics = consistency_metrics(labels, seg_maps, evidence)
    if metrics['accuracy'] + 0.001 < sample['refined']['accuracy']:
        raise ValueError('Exported GLB projection regressed from the in-memory result')
    result = {
        'instance': uid, 'original_faces': len(original.faces), 'exported_faces': len(exported.faces),
        'max_surface_distance_relative_to_extent': max_distance,
        'parent_area_relative_l1_error': relative_area_error,
        'glb_sha256': hashlib.sha256((prediction_dir / f'{uid}.glb').read_bytes()).hexdigest(),
        'exported_projection': metrics,
    }
    if comparisons:
        from scripts.refine_seg_projection import save_comparison
        raw_path = prediction_dir / f'{uid}_model_labels.npz'
        if raw_path.exists():
            with np.load(raw_path) as data:
                raw = data['face_labels']
        else:
            raw = recover_face_labels_from_colors(sample['prediction'], 20, len(original.faces))
        baseline_aligned, _ = align_model_labels(raw[parents], evidence.votes)
        save_comparison(prediction_dir / f'{uid}_comparison.png', seg_maps,
                        baseline_aligned, labels, evidence)
    if gt_root is not None:
        gt = np.load(gt_root / 'PartObjaverse-Tiny_semantic_gt' / f'{uid}.npy').astype(np.int32)
        if gt.shape != (len(original.faces),):
            raise ValueError('GT face count does not match original mesh')
        inherited_gt = gt[parents]
        result['gt_surface'] = region_metrics(inherited_gt, labels, inherited_gt >= 0,
                                               weights=exported.area_faces)
        model_path = prediction_dir / f'{uid}_model_labels.npz'
        if model_path.exists():
            with np.load(model_path) as data:
                baseline = data['face_labels']
            if baseline.shape != gt.shape:
                raise ValueError('Raw model label count differs from the input mesh')
            result['baseline_source'] = str(model_path)
        else:
            baseline = recover_face_labels_from_colors(sample['prediction'], 20, len(original.faces))
            result['baseline_source'] = sample['prediction']
        result['baseline_gt_surface'] = region_metrics(gt, baseline, gt >= 0, weights=original.area_faces)
        # Report visible and unseen original surfaces separately to expose
        # regressions hidden by the observed-view objective.
        original_evidence = collect_evidence(original, seg_maps, device=device)
        for name, mask in [('observed', original_evidence.visible), ('unobserved', ~original_evidence.visible)]:
            if mask.any() and original.area_faces[mask].sum() > 0:
                result[f'{name}_gt_surface'] = region_metrics(
                    inherited_gt, labels, (inherited_gt >= 0) & mask[parents], weights=exported.area_faces)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction_dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gt_root', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--comparisons', action='store_true')
    args = parser.parse_args()
    source = json.loads((args.prediction_dir / 'report.json').read_text())
    report = {'prediction_dir': str(args.prediction_dir.resolve()), 'device': args.device,
              'samples': [], 'failures': []}
    for sample in source['samples']:
        try:
            result = audit_sample(sample, args.prediction_dir, args.gt_root, args.device, args.comparisons)
            report['samples'].append(result)
            print(f"{sample['instance']}: artifact OK, accuracy {result['exported_projection']['accuracy']:.5f}", flush=True)
        except Exception as exc:
            report['failures'].append({'instance': sample['instance'], 'error': str(exc)})
            print(f"FAIL {sample['instance']}: {exc}", flush=True)
    rows = report['samples']
    summary = {'num_samples': len(rows), 'num_failures': len(report['failures'])}
    if rows:
        summary['exported_projection'] = {k: float(np.mean([r['exported_projection'][k] for r in rows]))
                                          for k in ('accuracy', 'miou', 'gap_to_ceiling')}
        summary['mean_face_multiplier'] = float(np.mean([r['exported_faces'] / r['original_faces'] for r in rows]))
        summary['max_exported_faces'] = max(r['exported_faces'] for r in rows)
        if args.gt_root:
            for kind in ('baseline_gt_surface', 'gt_surface', 'observed_gt_surface', 'unobserved_gt_surface'):
                summary[kind] = {k: float(np.mean([r[kind][k] for r in rows if kind in r]))
                                 for k in ('face_acc', 'miou_ca', 'miou_h', 'ap50')}
    report['summary'] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))
    if report['failures']:
        raise RuntimeError('Artifact audit failed; see report')


if __name__ == '__main__':
    main()
