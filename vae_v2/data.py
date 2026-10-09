"""Versioned surface-supervision cache and deterministic train/val/test access."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .geometry import FORMAT_VERSION


def source_rows(cfg):
    if cfg.get('annotation_policy'):
        from .raw_data import raw_source_rows
        return raw_source_rows(cfg)
    source = Path(cfg['source_dir'])
    selection = json.loads((source / 'selection.json').read_text())
    prepared = json.loads((source / 'prepared_manifest.json').read_text())
    audit = json.loads((source / 'prepared_audit.json').read_text())
    if prepared['expected'] != prepared['prepared'] or audit['errors_count']:
        raise ValueError('Source supervision is incomplete or failed audit')
    for key in ('failures', 'missing', 'extra', 'wrong_split', 'orphan_files', 'missing_ranks', 'extra_ranks', 'duplicate_ranks'):
        if prepared.get(key):
            raise ValueError(f'Source preparation has {key}')
    if audit['selection_sha256'] != hashlib.sha256((source / 'selection.json').read_bytes()).hexdigest():
        raise ValueError('Source selection differs from its audit')
    if audit['prepared_manifest_sha256'] != hashlib.sha256((source / 'prepared_manifest.json').read_bytes()).hexdigest():
        raise ValueError('Source prepared manifest differs from audit')
    rows = []
    audited = {r['instance']:r for r in audit['rows']}
    for r in selection['samples']:
        uid = r['model_id']
        if uid not in audited or audited[uid]['errors']:
            raise ValueError(f'Source object not covered by passing audit: {uid}')
        rows.append({'instance': r['model_id'], 'split': r['split'], 'type_id': r['type_id'],
                     'source': str(source / 'samples' / f'{r["model_id"]}.npz'),
                     'source_bytes':audited[uid]['size'], 'source_mtime_ns':audited[uid]['mtime_ns']})
    if len(rows) != len({r['instance'] for r in rows}):
        raise ValueError('Duplicate source IDs')
    return sorted(rows, key=lambda r: r['instance'])


def cache_contract(cfg):
    result = {'format_version': FORMAT_VERSION, 'geometry': cfg['geometry'],
            'seed': cfg['seed'], 'source_dir': cfg['source_dir'],
            'target_policy':'soft-input_annotated-hard-metric-v1'}
    if cfg.get('annotation_policy'):
        result.update(num_classes=cfg['num_classes'],annotation_policy=cfg['annotation_policy'],
                      annotation_dir=cfg['annotation_dir'],target_policy='raw_corrected_soft_labels_v1')
    return result


def load_manifest(cfg, verify_files=True, verify_workers=1):
    root = Path(cfg['prepared_dir'])
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['contract'] != cache_contract(cfg) or manifest['failures']:
        raise ValueError('Prepared VAE v2 cache contract mismatch or failures')
    sources = source_rows(cfg)
    if cfg.get('dataset_split'):
        from .splits import validate_split_manifest
        assignments = validate_split_manifest(manifest, sources, cfg['dataset_split'])
        expected = set(assignments.items())
    else:
        if manifest.get('dataset_split'):
            raise ValueError('A repartitioned manifest requires its matching dataset_split config')
        expected = {(r['instance'], r['split']) for r in sources}
    actual = {(r['instance'], r['split']) for r in manifest['samples']}
    if expected != actual or len(actual) != len(manifest['samples']):
        raise ValueError('Incomplete, duplicated, or changed VAE v2 split')
    source_by_id={r['instance']:r for r in sources}
    for row in manifest['samples']:
        source=source_by_id[row['instance']]
        if any(row[k]!=source[k] for k in ('source','source_bytes','source_mtime_ns')):
            raise ValueError('VAE v2 source provenance differs from audited source')
        if cfg.get('annotation_policy'):
            from .raw_data import raw_source_identity
            if raw_source_identity(row)!=raw_source_identity(source):
                raise ValueError('Raw annotation provenance changed')
    if verify_files:
        def verify(row):
            p = root / row['path']
            stat = p.stat()
            if stat.st_size != row['bytes'] or stat.st_mtime_ns != row['mtime_ns']:
                raise ValueError(f'Prepared cache changed after audit: {p}')
            source_stat = Path(row['source']).stat()
            if source_stat.st_size != row['source_bytes'] or source_stat.st_mtime_ns != row['source_mtime_ns']:
                raise ValueError(f'Source geometry changed: {row["source"]}')
            if cfg.get('annotation_policy'):
                from .raw_data import check_raw_source
                check_raw_source(row)
        if verify_workers < 1:
            raise ValueError('verify_workers must be positive')
        if verify_workers == 1:
            for row in manifest['samples']:
                verify(row)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=verify_workers) as pool:
                for _ in pool.map(verify, manifest['samples']):
                    pass
    return manifest


def load_sample(path):
    with np.load(path, allow_pickle=False) as data:
        sample = {k: data[k] for k in data.files if k != 'metadata'}
        metadata = json.loads(str(data['metadata']))
    if metadata['format_version'] != FORMAT_VERSION:
        raise ValueError('Unsupported VAE v2 sample')
    classes=metadata.get('num_classes',20)
    if sample['voxel_target'].shape[-1]!=classes:
        raise ValueError('Cache metadata class count differs from target')
    if classes!=20 and ('annotation_audit' not in sample or metadata.get('annotation_policy')!='gltf_mesh_local_maskid_v1'):
        raise ValueError('Expanded-class cache lacks corrected-annotation provenance')
    return sample


def smoke_manifest(cfg):
    """Small explicit trainer integration fixture; never eligible for test selection."""
    available = []
    if cfg.get('dataset_split'):
        full = load_manifest(cfg)
        for split in ('train', 'validation'):
            row = next(r for r in full['samples'] if r['split'] == split)
            validate_sample(load_sample(Path(cfg['prepared_dir']) / row['path']), cfg.get('num_classes', 20))
            available.append(row)
        return {'contract': cache_contract(cfg), 'samples': available, 'failures': [], 'smoke_only': True}
    rows = source_rows(cfg)
    for split in ('train', 'validation'):
        for row in (r for r in rows if r['split'] == split):
            p = Path(cfg['prepared_dir'])/'samples'/f'{row["instance"]}.npz'
            if not p.is_file():
                continue
            with np.load(p,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']))
            if any(meta.get(k)!=v for k,v in cache_contract(cfg).items()) or meta['instance']!=row['instance'] or meta['split']!=split:
                raise ValueError('Smoke cache contract mismatch')
            validate_sample(load_sample(p))
            stat=p.stat()
            available.append(dict(row,path=str(p.relative_to(cfg['prepared_dir'])),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
            break
        else:
            raise ValueError(f'Prepare at least one {split} object for --smoke')
    return {'contract':cache_contract(cfg),'samples':available,'failures':[],'smoke_only':True}


def validate_sample(sample, num_classes=None):
    coords = sample['coords']
    target = sample['voxel_target']
    labels = sample['face_labels']
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or coords.min() < 0 or coords.max() > 511:
        raise ValueError('Bad voxel coordinates')
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('Duplicate voxels')
    classes = num_classes if num_classes is not None else (target.shape[-1] if target.ndim==2 else 0)
    if not 2 <= classes <= 256 or target.shape != (len(coords), classes) or not np.isfinite(target).all() or target.min() < 0 or not np.allclose(target.sum(1), 1):
        raise ValueError('Invalid soft voxel labels')
    if labels.shape != (len(sample['mesh_faces']),) or labels.min() < 0 or labels.max() >= classes:
        raise ValueError('Invalid face labels')
    if not np.issubdtype(labels.dtype,np.integer) or not np.issubdtype(coords.dtype,np.integer):
        raise ValueError('Labels and coordinates must be integer')
    if not ((labels>0)&(sample['face_area']>0)).any() or target[:,1:].sum()<=0:
        raise ValueError('No annotated geometry: fail preparation instead of silently dropping this object')
    metric_target=target.copy();metric_target[:,0]=0
    if not np.array_equal(sample['voxel_labels'],metric_target.argmax(1)):
        raise ValueError('Hard metric labels must select annotated soft target channels')
    if len(sample['anchor_voxel']) != len(sample['anchor_faces']) or sample['anchor_voxel'].min() < 0 or sample['anchor_voxel'].max() >= len(coords):
        raise ValueError('Invalid anchors')
    if sample['anchor_faces'].min() < 0 or sample['anchor_faces'].max() >= len(labels):
        raise ValueError('Invalid anchor face indices')
    components = sample['face_components']
    if components.shape != labels.shape or not np.issubdtype(components.dtype, np.integer) or components.min() < 0:
        raise ValueError('Invalid face components')
    if not np.array_equal(np.unique(components), np.unique(components[sample['anchor_faces']])):
        raise ValueError('A geometry component has no retained surface anchor')
    if 'topology_repair' in sample and (int(sample['topology_repair']) != 1 or
            int(sample['legacy_component_count']) <= len(np.unique(components))):
        raise ValueError('Invalid topology repair provenance')
    for key in ('adjacency', 'near_pairs'):
        pairs = sample[key]
        if pairs.ndim != 2 or pairs.shape[1] != 2 or (pairs.size and (pairs.min() < 0 or pairs.max() >= len(labels))):
            raise ValueError(f'Invalid {key}')
    for key in ('face_normals', 'face_area', 'anchor_points', 'mesh_vertices'):
        if not np.isfinite(sample[key]).all():
            raise ValueError(f'Nonfinite {key}')
    if sample['face_pixels'].shape != labels.shape or (sample['face_pixels'] < 0).any():
        raise ValueError('Invalid pixel counts')
    if 'annotation_audit' in sample:
        audit=json.loads(str(sample['annotation_audit']))
        if audit['parts']>=classes or audit['annotated_faces']!=int((labels>0).sum()):
            raise ValueError('Annotation coverage audit differs from labels')
    if 'face_raster' in sample:
        raster=sample['face_raster']
        if raster.ndim!=3 or raster.shape[0]!=6 or raster.min() < -1 or raster.max()>=len(labels):
            raise ValueError('Invalid face raster')
        if not np.array_equal(np.bincount(raster[raster>=0],minlength=len(labels)),sample['face_pixels']):
            raise ValueError('Visibility raster does not match face pixel counts')


def to_device(values, device):
    import torch
    result = {}
    integer_keys = {'coords', 'voxel_labels', 'neighbor_index', 'face_ids', 'query_face_ids',
                    'query_labels', 'pair_index', 'face_labels'}
    bool_keys = {'neighbor_mask', 'pair_same'}
    for key, value in values.items():
        if not isinstance(value, np.ndarray):
            result[key] = value
            continue
        dtype = torch.long if key in integer_keys else torch.bool if key in bool_keys else torch.float32
        result[key] = torch.as_tensor(value, dtype=dtype, device=device)
    return result


def voxel_tensors(sample, device):
    return to_device({k: sample[k] for k in ('coords', 'voxel_target', 'voxel_labels')}, device)


def training_order(rows, seed, epoch):
    return np.random.default_rng(seed + epoch).permutation(len(rows))
