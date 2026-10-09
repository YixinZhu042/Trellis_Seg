"""
Ablation runner for improving Trellis-Seg inference WITHOUT retraining.

Reuses cached API seg_maps under /tmp/po_run/<uid>/seg/seg_maps.npz so no new
API calls are made and GT is never used to guide predictions. Each variant runs
scripts/inference_wild.py with a different flag set into its own output dir, so
the baseline results in results_po/ are left untouched.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PO_ROOT = '/path/to/PartObjaverse-Tiny'
MESH_DIR = f'{PO_ROOT}/PartObjaverse-Tiny_mesh'
CACHE = '/tmp/po_run'


def postprocess_cached_predictions(ids, args):
    """Paired cleanup ablation on frozen predictions, using input mesh topology only."""
    import hashlib
    import numpy as np
    import trimesh
    from evaluate_seg_quality import load_mesh, recover_face_labels_from_colors
    from inference_wild import generate_class_colors, postprocess_face_labels

    source = Path(args.postprocess_from).resolve()
    output = Path(args.out_dir).resolve()
    if source == output or any(output.glob('*.glb')):
        raise ValueError('Use a separate output directory with no existing GLB files')
    palette = (generate_class_colors(20) * 255).astype(np.uint8)
    manifest = {
        'mode': 'postprocess_frozen_predictions', 'source_dir': str(source),
        'mesh_dir': str(Path(args.mesh_dir).resolve()), 'sample_ids': ids,
        'smooth_iterations': args.postprocess_iterations,
        'min_component_faces': args.postprocess_min_faces,
        'gt_used_for_predictions': False, 'samples': [], 'failures': [],
    }
    for uid in ids:
        try:
            pred_path = source / f'{uid}.glb'
            mesh = load_mesh(str(Path(args.mesh_dir) / f'{uid}.glb'))
            pred_mesh = load_mesh(str(pred_path))
            tolerance = max(float(np.max(mesh.extents)), 1e-12) * 1e-6
            if len(mesh.faces) != len(pred_mesh.faces) or not np.allclose(
                    mesh.triangles, pred_mesh.triangles, rtol=1e-6, atol=tolerance):
                raise ValueError('Prediction and input mesh face order/geometry differ')
            labels = recover_face_labels_from_colors(str(pred_path), 20, len(mesh.faces))
            cleaned = postprocess_face_labels(
                labels, mesh.face_adjacency, seg_prior_mode='components',
                postprocess_after_prior=True,
                min_component_faces=args.postprocess_min_faces,
                smooth_iterations=args.postprocess_iterations,
            )
            vertices = mesh.triangles.reshape(-1, 3)
            faces = np.arange(len(vertices)).reshape(-1, 3)
            colors = np.column_stack([palette[cleaned], np.full(len(cleaned), 255, np.uint8)])
            result = trimesh.Trimesh(vertices=vertices, faces=faces,
                                     vertex_colors=np.repeat(colors, 3, axis=0), process=False)
            out = output / f'{uid}.glb'
            result.export(str(out))
            recovered = recover_face_labels_from_colors(str(out), 20, len(mesh.faces))
            if not np.array_equal(recovered, cleaned):
                raise ValueError('Label/color roundtrip failed')
            manifest['samples'].append({
                'instance': uid, 'source_sha256': hashlib.sha256(pred_path.read_bytes()).hexdigest(),
                'changed_faces': int(np.count_nonzero(labels != cleaned)),
                'num_faces': len(labels),
            })
            print(f'{uid}: changed {np.count_nonzero(labels != cleaned)}/{len(labels)} faces', flush=True)
        except Exception as exc:
            manifest['failures'].append({'instance': uid, 'error': str(exc)})
            print(f'FAIL {uid}: {exc}', flush=True)
    with open(output / 'postprocess_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    if manifest['failures']:
        raise RuntimeError(f"Postprocessing failed for {len(manifest['failures'])} samples")
    print(f'Postprocessed {len(manifest["samples"])} frozen predictions without model/API calls')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', required=True, help='eval metrics.json to read instance ids from')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mesh_dir', default=MESH_DIR)
    ap.add_argument('--postprocess_from', help='Apply cleanup to frozen predicted GLBs; no API/model calls')
    ap.add_argument('--postprocess_iterations', type=int, default=4)
    ap.add_argument('--postprocess_min_faces', type=int, default=50)
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                    help='extra flags passed verbatim to inference_wild.py')
    args = ap.parse_args()
    if args.postprocess_from and args.extra:
        ap.error('--extra cannot be used with --postprocess_from')
    if args.postprocess_iterations < 0 or args.postprocess_min_faces < 0:
        ap.error('Postprocessing counts must be non-negative')

    with open(args.ids) as f:
        ids = [r['instance'] for r in json.load(f)['samples']]
    if len(ids) != len(set(ids)):
        ap.error('Duplicate instance IDs')
    os.makedirs(args.out_dir, exist_ok=True)
    if args.postprocess_from:
        postprocess_cached_predictions(ids, args)
        return
    done = fail = skip = 0
    for uid in ids:
        seg = f'{CACHE}/{uid}/seg/seg_maps.npz'
        mesh = f'{MESH_DIR}/{uid}.glb'
        out = f'{args.out_dir}/{uid}.glb'
        if not os.path.exists(seg) or not os.path.exists(mesh):
            print(f'SKIP {uid} (missing cache/mesh)'); skip += 1; continue
        if os.path.exists(out):
            done += 1; continue
        cmd = [sys.executable, str(PROJECT / 'scripts' / 'inference_wild.py'),
               '--mesh', mesh, '--seg_maps', seg, '--output', out, '--steps', '50'] + args.extra
        r = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
        if r.returncode == 0:
            done += 1
        else:
            fail += 1
            print(f'FAIL {uid}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?"}')
    print(f'\n{args.out_dir}: done={done} fail={fail} skip={skip}')


if __name__ == '__main__':
    main()
