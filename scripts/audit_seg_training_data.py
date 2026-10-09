"""Read-only full audit of PartNeXt segmentation training caches.

Detects missing/extra objects, label-granularity disagreement, malformed arrays,
out-of-range coordinates/labels, duplicate sparse coordinates, NaN/Inf features,
and metadata/token mismatches. No cache files are modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess.step4_encode_seg_latent import parse_face_labels
from scripts.evaluate_seg_quality import load_partnext_records


def duplicate_rows(coords):
    if len(coords) < 2:
        return 0
    base = int(coords.max(initial=0)) + 1
    packed = (coords[:, 0].astype(np.int64) * base + coords[:, 1]) * base + coords[:, 2]
    return int(len(coords) - len(np.unique(packed)))


def audit_one(root, mesh_dir, row, record):
    uid = str(row.sha256)
    errors, warnings = [], []
    result = {'instance': uid, 'metadata_tokens': int(row.seg_latent_tokens)}
    import trimesh
    mesh_path = mesh_dir / record['type_id'] / f'{uid}.glb'
    scene = trimesh.load(mesh_path, force='scene', process=False)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    labels19, parts19 = parse_face_labels(record['masks'], len(mesh.faces), record.get('hierarchy'), max_parts=19)
    labels15, parts15 = parse_face_labels(record['masks'], len(mesh.faces), record.get('hierarchy'), max_parts=15)
    face_disagreement = int(np.count_nonzero(labels19 != labels15))
    result.update(parts_max19=int(parts19), parts_max15=int(parts15), granularity_diff=bool(parts19 != parts15),
                  granularity_face_disagreement=face_disagreement,
                  annotation_unlabeled_faces=int(np.count_nonzero(labels19 == 0)),
                  annotation_faces=int(len(labels19)))
    try:
        with np.load(root / 'seg_views' / f'{uid}.npz') as data:
            maps = data['seg_maps']; stored_parts = int(data['num_parts'])
            if maps.shape != (6, 512, 512) or maps.dtype.kind not in 'iu': errors.append('invalid seg_views shape/dtype')
            if stored_parts != parts19: errors.append(f'seg_views num_parts={stored_parts}, annotation19={parts19}')
            if maps.min(initial=0) < 0 or maps.max(initial=0) > parts19 + 1: errors.append('seg_views label out of range')
            visible = set((np.unique(maps[maps > 0]) - 1).tolist())
            if not visible.issubset(set(np.unique(labels19).tolist())): errors.append('seg_views labels disagree with annotation mapping')
            result.update(seg_view_min=int(maps.min()), seg_view_max=int(maps.max()),
                          seg_view_foreground_pixels=int(np.count_nonzero(maps)))
    except Exception as exc: errors.append(f'seg_views: {exc}')
    try:
        with np.load(root / 'seg_voxels' / f'{uid}.npz') as data:
            coords, labels = data['coords'], data['labels']
            if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) != len(labels): errors.append('invalid seg_voxels shapes')
            if coords.dtype.kind not in 'iu' or labels.dtype.kind not in 'iu': errors.append('invalid seg_voxels dtypes')
            if coords.size and (coords.min() < 0 or coords.max() > 511): errors.append('seg_voxels coordinate out of range')
            if labels.size and (labels.min() < 0 or labels.max() > 19): errors.append('seg_voxels label out of range')
            result.update(voxels=int(len(coords)), voxel_label_min=int(labels.min(initial=0)),
                          voxel_label_max=int(labels.max(initial=0)), voxel_duplicate_coords=duplicate_rows(coords))
            if face_disagreement: warnings.append('seg_voxels likely generated with max_parts=15; incompatible with 19-part caches')
    except Exception as exc: errors.append(f'seg_voxels: {exc}')
    for name, expected_tokens in [('seg_latent', int(row.seg_latent_tokens)), ('mv_seg_latent', None), ('shape_latent', None)]:
        try:
            with np.load(root / name / f'{uid}.npz') as data:
                coords, feats = data['coords'], data['feats']
                if coords.ndim != 2 or coords.shape[1] != 3 or feats.ndim != 2 or feats.shape[1] != 32 or len(coords) != len(feats): errors.append(f'{name} invalid shapes')
                if expected_tokens is not None and len(coords) != expected_tokens: errors.append(f'{name} tokens differ from metadata')
                if coords.size and (coords.min() < 0 or coords.max() > 31): errors.append(f'{name} coordinate out of range')
                if not np.isfinite(feats).all(): errors.append(f'{name} has NaN/Inf')
                duplicates = duplicate_rows(coords)
                # mv tokens intentionally preserve per-view duplicates.
                if name != 'mv_seg_latent' and duplicates: errors.append(f'{name} duplicate coordinates={duplicates}')
                result[f'{name}_tokens'] = int(len(coords));result[f'{name}_duplicate_coords'] = duplicates
        except Exception as exc: errors.append(f'{name}: {exc}')
    result['errors'], result['warnings'] = errors, warnings
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--arrow_dir', type=Path, default=Path('/path/to/PartNeXt'))
    ap.add_argument('--mesh_dir', type=Path, default=Path('/path/to/PartNeXt_mesh/glbs'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    metadata = pd.read_csv(args.data_dir / 'metadata.csv')
    ids = metadata.sha256.astype(str).tolist()
    records = load_partnext_records(str(args.arrow_dir), target_ids=ids)
    missing_records = sorted(set(ids) - set(records))
    if missing_records: raise ValueError(f'Missing annotations: {missing_records[:10]}')
    missing_meshes = [uid for uid, record in records.items()
                      if not (args.mesh_dir / record['type_id'] / f'{uid}.glb').is_file()]
    if missing_meshes: raise FileNotFoundError(f'Missing meshes: {missing_meshes[:10]}')
    sets = {}
    for name in ['seg_views', 'seg_voxels', 'seg_latent', 'mv_seg_latent', 'shape_latent']:
        sets[name] = {p.stem for p in (args.data_dir / name).glob('*.npz') if not p.name.startswith('_')}
    expected = set(ids)
    start = time.monotonic();results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_one, args.data_dir, args.mesh_dir, row, records[str(row.sha256)])
                   for row in metadata.itertuples(index=False)]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 1000 == 0: print(f'Audited {i}/{len(futures)}', flush=True)
    results.sort(key=lambda x:x['instance'])
    errors = [x for x in results if x['errors']]
    granularity = [x for x in results if x['granularity_face_disagreement']]
    tokens = metadata.seg_latent_tokens.to_numpy()
    report = {'data_dir':str(args.data_dir.resolve()), 'objects':len(results), 'seconds':time.monotonic()-start,
        'set_checks':{name:{'files':len(values),'missing':sorted(expected-values),'extra':sorted(values-expected)} for name,values in sets.items()},
        'token_stats':{'min':int(tokens.min()),'max':int(tokens.max()),'over_8192':int(np.count_nonzero(tokens>8192)),
                       'percentiles':{str(p):float(np.percentile(tokens,p)) for p in [50,90,95,99,99.9]}},
        'granularity_mismatch_count':len(granularity),'granularity_mismatch_examples':[x['instance'] for x in granularity[:100]],
        'errors_count':len(errors),'error_examples':errors[:100],
        'label_histograms':{'seg_view_max':np.bincount([x.get('seg_view_max',0) for x in results],minlength=21).tolist(),
                            'voxel_label_max':np.bincount([x.get('voxel_label_max',0) for x in results],minlength=21).tolist()},
        'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['objects','seconds','token_stats','granularity_mismatch_count','errors_count']},indent=2))


if __name__=='__main__':main()
