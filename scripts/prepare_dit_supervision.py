"""Build coordinate-aligned supervision for a frozen-VAE DiT experiment.

One mesh/part mapping produces voxels, target latent, six input views, projected
pixel votes and surface adjacency. Existing independently sampled seg_voxels
and seg_latent are not paired. Shape features reuse the existing geometry cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'TRELLIS.2'))
sys.path.insert(0, str(ROOT))


def voxelize(mesh, labels, seed=42, max_voxels=30000):
    points, face = trimesh.sample.sample_surface(mesh, min(len(mesh.faces) * 10, 100000), seed=seed)
    center = (points.min(0) + points.max(0)) / 2
    extent = (points.max(0) - points.min(0)).max()
    if extent <= 0:
        raise ValueError('Zero sample extent')
    coords = np.clip(((points - center) / extent * 509 + 256).astype(np.int32), 0, 511)
    linear = coords[:, 0].astype(np.int64) * 512**2 + coords[:, 1] * 512 + coords[:, 2]
    unique, inverse = np.unique(linear, return_inverse=True)
    votes = np.bincount(inverse * 20 + labels[face], minlength=len(unique) * 20).reshape(-1, 20)
    target = votes.argmax(1).astype(np.int64)
    coords = np.column_stack([unique // 512**2, unique // 512 % 512, unique % 512]).astype(np.int32)
    if len(coords) > max_voxels:
        take = np.random.default_rng(len(coords)).choice(len(coords), max_voxels, replace=False)
        coords, target = coords[take], target[take]
    return coords, target, center, extent


def select_records(args):
    import pandas as pd
    from scripts.evaluate_seg_quality import load_partnext_records
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'selection.json'
    if manifest.exists() and not args.replace_selection:
        raise ValueError('Selection already exists; use a new output directory')
    metadata = pd.read_csv(args.data_dir / 'metadata.csv')
    eligible = metadata.loc[metadata.seg_latent_tokens <= args.max_tokens, 'sha256'].astype(str).tolist()
    over_limit = metadata.loc[metadata.seg_latent_tokens > args.max_tokens,
                              ['sha256', 'seg_latent_tokens']].to_dict('records')
    # Fixed split chosen before any loss or outcome is computed. These objects
    # may have been seen by the original pretrained model; only the continuation
    # train/validation/test splits are disjoint.
    eligible.sort(key=lambda x: hashlib.sha256((args.split_seed + ':' + x).encode()).hexdigest())
    if args.use_all:
        if args.val_samples + args.test_samples >= len(eligible):
            raise ValueError('Validation and test consume the entire eligible dataset')
        # Preserve the fixed validation/test cohort from the <=max_tokens pool.
        # Old over-limit latents are still usable: aligned preparation can reduce
        # the deterministic voxel budget until the new target fits max_tokens.
        base_train = len(eligible) - args.val_samples - args.test_samples
        train_ids = eligible[:base_train]
        validation_ids = eligible[base_train:base_train + args.val_samples]
        test_ids = eligible[base_train + args.val_samples:]
        adaptive_ids = sorted(
            (row['sha256'] for row in over_limit),
            key=lambda x: hashlib.sha256((args.split_seed + ':' + x).encode()).hexdigest())
        train_ids.extend(adaptive_ids)
        chosen = train_ids + validation_ids + test_ids
        args.train_samples = len(train_ids)
        excluded = []
    else:
        n = args.train_samples + args.val_samples + args.test_samples
        if n > len(eligible): raise ValueError(f'Requested {n} samples but only {len(eligible)} are eligible')
        chosen = eligible[:n]
        over_ids = {row['sha256'] for row in over_limit}
        excluded = [row for row in over_limit if row['sha256'] in over_ids]
    records = load_partnext_records(str(args.arrow_dir), target_ids=chosen)
    entries = []
    for i, uid in enumerate(chosen):
        record = records[uid]
        entries.append({'model_id': uid, 'type_id': record['type_id'],
                        'mesh_path': str(args.mesh_dir / record['type_id'] / f'{uid}.glb'),
                        'split': 'train' if i < args.train_samples else
                                 ('validation' if i < args.train_samples + args.val_samples else 'test')})
    payload = {'format_version': 2, 'samples': entries, 'data_dir': str(args.data_dir.resolve()),
               'metadata_objects': int(len(metadata)), 'max_tokens': args.max_tokens,
               'excluded_over_max_tokens': excluded,
               'adaptive_resample_candidates': over_limit if args.use_all else [],
               'split_seed': args.split_seed,
               'split_counts': {'train': args.train_samples, 'validation': args.val_samples, 'test': args.test_samples},
               'split_note': 'Disjoint for continuation only; pretrained checkpoint may have seen these objects.'}
    temporary = manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(manifest)
    print(f'Selected {len(entries)} objects: train={args.train_samples}, validation={args.val_samples}, '
          f'test={args.test_samples}; adaptive resample={len(payload["adaptive_resample_candidates"])}; '
          f'excluded={len(excluded)}', flush=True)


@torch.no_grad()
def prepare(args):
    import trellis2.models as models
    from trellis2.modules.sparse.basic import SparseTensor
    from trellis2.modules.sparse.conv import config as conv_config
    from scripts.inference_wild import SEG_VAE_ENC_CKPT, SEG_VAE_DEC_CKPT, lookup_feats_at_coords
    from scripts.evaluate_seg_quality import load_mesh, parse_face_labels
    from preprocess.step2_encode_mv_seg_latent import encode_one_view
    from utils.seg_projection import rasterize_face_ids, welded_face_adjacency
    torch.cuda.set_device(args.device)
    conv_config.FLEX_GEMM_ALGO = 'explicit_gemm'
    cfg = json.loads((ROOT / 'configs/seg_vae_train.json').read_text())['models']
    enc = models.SparseUnetVaeEncoder(**cfg['encoder']['args']).eval()
    dec = models.SparseUnetVaeDecoder(**cfg['decoder']['args']).eval()
    enc.load_state_dict(torch.load(SEG_VAE_ENC_CKPT, map_location='cpu', weights_only=True))
    dec.load_state_dict(torch.load(SEG_VAE_DEC_CKPT, map_location='cpu', weights_only=True))
    enc, dec = enc.to(args.device).requires_grad_(False), dec.to(args.device).requires_grad_(False)
    selection = json.loads((args.output_dir / 'selection.json').read_text())
    if selection.get('format_version') != 2:
        raise ValueError('Unsupported selection format; regenerate it with the current --select code')
    if not 0 <= args.rank < args.world_size:
        raise ValueError('Require 0 <= rank < world_size')
    samples = selection['samples'][args.rank::args.world_size]
    from scripts.evaluate_seg_quality import load_partnext_records
    assigned_ids = [row['model_id'] for row in samples]
    annotations = load_partnext_records(str(args.arrow_dir), target_ids=assigned_ids)
    if set(annotations) != set(assigned_ids):
        raise ValueError('Assigned annotations are incomplete')
    (args.output_dir / 'samples').mkdir(exist_ok=True)
    result = {'rank': args.rank, 'world_size': args.world_size, 'samples': [], 'failures': [],
              'encoder': SEG_VAE_ENC_CKPT, 'decoder': SEG_VAE_DEC_CKPT}
    for selected in samples:
        uid = selected['model_id']
        rec = annotations[uid]
        rec['mesh_path'] = selected['mesh_path']
        rec['split'] = selected['split']
        path = args.output_dir / 'samples' / f'{uid}.npz'
        try:
            if path.exists() and not args.force:
                with np.load(path) as existing:
                    required = {'coords','voxel_labels','latent_coords','latent_feats','shape_feats','mv_coords','mv_feats',
                                'shape_valid',
                                'face_labels','face_area','face_knn','face_knn_weights','adjacency','boundary','face_pixels',
                                'face_raster','seg_maps','sample_center','sample_extent','mesh_vertices','mesh_faces'}
                    if required.issubset(existing.files) and len(existing['latent_coords']) <= args.max_tokens:
                        result['samples'].append({'instance':uid,'split':rec['split'],'path':str(path.resolve()),
                                                  'tokens':int(len(existing['latent_coords'])),'resumed_existing':True})
                        continue
            mesh = load_mesh(rec['mesh_path'])
            labels, parts = parse_face_labels(rec['masks'], len(mesh.faces), rec.get('hierarchy'), max_parts=19)
            if labels.min() < 0 or labels.max() >= 20 or parts > 19:
                raise ValueError('Invalid part mapping (20 model channels)')
            voxel_budget = 30000
            while True:
                coords, target, center, extent = voxelize(mesh, labels, max_voxels=voxel_budget)
                c = torch.tensor(np.column_stack([np.zeros(len(coords), np.int32), coords]), device=args.device)
                x = SparseTensor(feats=F.one_hot(torch.tensor(target, device=args.device), 20).float(), coords=c)
                z = enc(x, sample_posterior=False)
                if len(z.feats) <= args.max_tokens:
                    break
                if voxel_budget <= 5000:
                    raise ValueError(f'Too many latent tokens after adaptive sampling: {len(z.feats)} > {args.max_tokens}')
                next_budget = max(5000, min(voxel_budget - 1000,
                    int(voxel_budget * args.max_tokens / len(z.feats) * .9)))
                print(f'{uid}: {len(z.feats)} latent tokens exceeds {args.max_tokens}; '
                      f'retrying with voxel budget {next_budget}', flush=True)
                voxel_budget = next_budget
            reconstructed = dec(z)
            if not torch.equal(reconstructed.coords, c):
                raise ValueError('Decoder output coordinates do not match targets')
            # Establish cache again later during training, with all-zero dummy
            # labels, so target labels never enter the decoder through its cache.
            query = ((mesh.triangles_center - center) / extent * 509 + 256).astype(np.float32)
            distance, nearest = cKDTree(coords.astype(float)).query(query, k=min(8, len(coords)))
            if nearest.ndim == 1:
                nearest, distance = nearest[:, None], distance[:, None]
            weights = 1 / np.maximum(distance, 1e-3)
            weights /= weights.sum(1, keepdims=True)
            raster = rasterize_face_ids(mesh, (512, 512), args.device)
            seg = np.zeros(raster.shape, np.int32)
            hit = raster >= 0
            seg[hit] = labels[raster[hit]] + 1
            face_pixels = np.bincount(raster[hit], minlength=len(mesh.faces))
            mv_coords, mv_feats = [], []
            for vi in range(6):
                vc, vf = encode_one_view(seg[vi], vi, enc, 20, args.device)
                if vc is not None:
                    mv_coords.append(vc); mv_feats.append(vf)
            shape = np.load(args.data_dir / 'shape_latent' / f'{uid}.npz')
            shape_feats, shape_valid = lookup_feats_at_coords(z.coords[:, 1:].int(),
                        torch.tensor(shape['coords'].astype(np.int32), device=args.device),
                        torch.tensor(shape['feats'], device=args.device), return_mask=True)
            shape_feats = shape_feats.cpu().numpy()
            adjacency = welded_face_adjacency(mesh).astype(np.int32)
            edge_target = (labels[adjacency[:, 0]] != labels[adjacency[:, 1]]).astype(np.float32)
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, coords=coords, voxel_labels=target, latent_coords=z.coords[:, 1:].cpu().numpy(),
                latent_feats=z.feats.float().cpu().numpy(), shape_feats=shape_feats,
                shape_valid=shape_valid.cpu().numpy(),
                mv_coords=torch.cat(mv_coords).numpy(), mv_feats=torch.cat(mv_feats).float().numpy(),
                face_labels=labels, face_area=mesh.area_faces.astype(np.float32), face_knn=nearest.astype(np.int32),
                face_knn_weights=weights.astype(np.float32), adjacency=adjacency, boundary=edge_target,
                face_pixels=face_pixels.astype(np.int32), face_raster=raster, seg_maps=seg,
                sample_center=center, sample_extent=np.array(extent), mesh_vertices=mesh.vertices, mesh_faces=mesh.faces)
            temporary.replace(path)
            logits = reconstructed.feats.float().cpu().numpy()
            face_pred = (logits[nearest] * weights[..., None]).sum(1).argmax(1)
            summary = {'instance': uid, 'split': rec['split'], 'mesh': rec['mesh_path'],
                'path': str(path.resolve()), 'num_parts': int(parts), 'voxels': len(coords),
                'voxel_budget': voxel_budget, 'tokens': len(z.feats),
                'faces': len(labels), 'boundary_edges': int(edge_target.sum()),
                'voxel_oracle_accuracy': float((logits.argmax(1) == target).mean()),
                'face_oracle_accuracy': float(np.average(face_pred == labels, weights=mesh.area_faces)),
                'projection_oracle_accuracy': float(np.average(face_pred == labels, weights=face_pixels)),
                'voxel_class_counts': np.bincount(target, minlength=20).tolist(),
                'face_class_counts': np.bincount(labels, minlength=20).tolist(),
                'visible_pixel_class_counts': np.bincount(seg[seg > 0] - 1, minlength=20).tolist(),
                'mesh_sha256': hashlib.sha256(Path(rec['mesh_path']).read_bytes()).hexdigest()}
            result['samples'].append(summary)
            print(f'{uid}: {rec["split"]}, tokens={len(z.feats)}, oracle2d={summary["projection_oracle_accuracy"]:.4f}', flush=True)
            (args.output_dir / f'prep_rank{args.rank}.json').write_text(json.dumps(result, indent=2) + '\n')
        except Exception as exc:
            result['failures'].append({'instance': uid, 'split': rec['split'], 'error': str(exc)})
            print(f'FAIL {uid}: {exc}', flush=True)
    (args.output_dir / f'prep_rank{args.rank}.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['failures'] and args.fail_on_error:
        raise RuntimeError(f"{len(result['failures'])} preparation failures; inspect the manifest")


def merge(args):
    selection = json.loads((args.output_dir / 'selection.json').read_text())
    expected = {row['model_id']: row['split'] for row in selection['samples']}
    rows, failures = [], []
    manifests = sorted(args.output_dir.glob('prep_rank*.json'))
    if not manifests:
        raise FileNotFoundError(f'No prep_rank*.json manifests found in {args.output_dir}')
    world_sizes = {data['world_size'] for data in (json.loads(path.read_text()) for path in manifests)}
    if len(world_sizes) != 1:
        raise ValueError(f'Preparation manifests disagree on world_size: {sorted(world_sizes)}')
    declared_world = world_sizes.pop()
    ranks = []
    for path in manifests:
        data = json.loads(path.read_text())
        ranks.append(data['rank'])
        rows.extend(data['samples'])
        failures.extend(data['failures'])
    expected_ranks = set(range(declared_world))
    missing_ranks = sorted(expected_ranks - set(ranks))
    extra_ranks = sorted(set(ranks) - expected_ranks)
    duplicate_ranks = sorted(rank for rank in set(ranks) if ranks.count(rank) > 1)
    by_id = {}
    for row in rows:
        if row['instance'] in by_id: raise ValueError(f'Duplicate prepared sample {row["instance"]}')
        by_id[row['instance']] = row
    missing = sorted(set(expected) - set(by_id) - {x['instance'] for x in failures})
    extra = sorted(set(by_id) - set(expected))
    wrong_split = sorted(uid for uid,row in by_id.items() if row['split'] != expected[uid])
    files = {p.stem for p in (args.output_dir/'samples').glob('*.npz')}
    report = {'selection':str((args.output_dir/'selection.json').resolve()),'rank_manifests':[str(x) for x in manifests],
              'world_size':declared_world,'missing_ranks':missing_ranks,'extra_ranks':extra_ranks,
              'duplicate_ranks':duplicate_ranks,
              'expected':len(expected),'prepared':len(by_id),'failures':failures,'missing':missing,'extra':extra,
              'wrong_split':wrong_split,'orphan_files':sorted(files-set(expected)),
              'split_counts':{s:sum(row['split']==s for row in by_id.values()) for s in ['train','validation','test']}}
    (args.output_dir/'prepared_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['expected','prepared','split_counts','missing','extra','wrong_split']},indent=2))
    if (failures or missing or extra or wrong_split or report['orphan_files'] or
            missing_ranks or extra_ranks or duplicate_ranks):
        raise RuntimeError('Prepared dataset is incomplete; inspect prepared_manifest.json')


def check(args):
    selection = json.loads((args.output_dir / 'selection.json').read_text())
    if selection.get('format_version') != 2:
        raise ValueError('Invalid selection format')
    samples = selection['samples']
    ids = [row['model_id'] for row in samples]
    if len(ids) != len(set(ids)) or sum(selection['split_counts'].values()) != len(ids):
        raise ValueError('Selection IDs/split counts are inconsistent')
    # Source meshes/shape latents are needed only while preparing. Once the
    # self-contained cache has passed a deep audit, training should not depend
    # on tens of thousands of slow source-filesystem stat calls.
    if args.require_prepared:
        missing_mesh, missing_shape = [], []
    else:
        def missing(entries):
            with ThreadPoolExecutor(max_workers=args.check_workers) as pool:
                present = list(pool.map(lambda item: Path(item[1]).is_file(), entries))
            return [uid for (uid, _), exists in zip(entries, present) if not exists]
        missing_mesh = missing([(row['model_id'], row['mesh_path']) for row in samples])
        missing_shape = missing([(uid, args.data_dir/'shape_latent'/f'{uid}.npz') for uid in ids])
    prepared = args.output_dir / 'prepared_manifest.json'
    prepared_state = json.loads(prepared.read_text()) if prepared.exists() else None
    audit = args.output_dir / 'prepared_audit.json'
    audit_state = json.loads(audit.read_text()) if audit.exists() else None
    if args.require_prepared:
        if prepared_state is None:
            raise FileNotFoundError(prepared)
        if (prepared_state['prepared'] != prepared_state['expected'] or prepared_state['failures'] or
                prepared_state['missing'] or prepared_state['extra'] or prepared_state['wrong_split'] or
                prepared_state.get('orphan_files') or
                prepared_state.get('missing_ranks') or prepared_state.get('extra_ranks') or
                prepared_state.get('duplicate_ranks')):
            raise ValueError('prepared_manifest.json is not complete')
        if audit_state is None:
            raise FileNotFoundError(audit)
        if (audit_state['objects'] != len(ids) or audit_state['errors_count'] or
                audit_state['split_counts'] != selection['split_counts'] or
                audit_state['selection_sha256'] != hashlib.sha256((args.output_dir/'selection.json').read_bytes()).hexdigest() or
                audit_state['prepared_manifest_sha256'] != hashlib.sha256(prepared.read_bytes()).hexdigest()):
            raise ValueError('prepared_audit.json does not validate the current prepared dataset')
        current_file_state=[]
        for uid in sorted(ids):
            path=args.output_dir/'samples'/f'{uid}.npz'
            if not path.is_file():raise FileNotFoundError(f'Missing prepared sample {uid}')
            stat=path.stat();current_file_state.append((uid,stat.st_size,stat.st_mtime_ns))
        digest=hashlib.sha256(json.dumps(current_file_state,separators=(',',':')).encode()).hexdigest()
        if audit_state.get('file_state_sha256')!=digest:
            raise ValueError('Prepared sample files changed after the deep audit')
    audit_summary = None if audit_state is None else {
        key:audit_state[key] for key in ('objects','errors_count','split_counts','size_bytes')}
    payload = {'selection':str((args.output_dir/'selection.json').resolve()),'objects':len(ids),
               'split_counts':selection['split_counts'],'excluded_over_max_tokens':len(selection['excluded_over_max_tokens']),
               'missing_mesh_count':len(missing_mesh),'missing_shape_count':len(missing_shape),
               'prepared_manifest':prepared_state,'prepared_audit':audit_summary,
               'require_prepared':args.require_prepared}
    print(json.dumps(payload,indent=2))
    if missing_mesh or missing_shape:
        raise RuntimeError('Selection references missing source data')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--select', action='store_true')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--data_dir', type=Path, default=Path('/path/to/trellis_partnext_data'))
    ap.add_argument('--arrow_dir', type=Path, default=Path('/path/to/PartNeXt'))
    ap.add_argument('--mesh_dir', type=Path, default=Path('/path/to/PartNeXt_mesh/glbs'))
    ap.add_argument('--train_samples', type=int, default=256)
    ap.add_argument('--val_samples', type=int, default=32)
    ap.add_argument('--test_samples', type=int, default=32)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world_size', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--use_all', action='store_true')
    ap.add_argument('--max_tokens', type=int, default=8192)
    ap.add_argument('--split_seed', default='decoded-dit-full-v1')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--fail_on_error', action='store_true')
    ap.add_argument('--replace_selection', action='store_true')
    ap.add_argument('--require_prepared', action='store_true')
    ap.add_argument('--check_workers', type=int, default=32)
    args = ap.parse_args()
    if args.check_workers < 1:
        ap.error('--check_workers must be positive')
    if sum([args.select,args.merge,args.check]) > 1:
        ap.error('--select, --merge and --check are mutually exclusive')
    if args.select:
        select_records(args)
    elif args.merge:
        merge(args)
    elif args.check:
        check(args)
    else:
        prepare(args)


if __name__ == '__main__':
    main()
