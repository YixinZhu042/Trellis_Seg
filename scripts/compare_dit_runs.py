"""CPU-only paired frozen DiT report; run after both evaluations are merged.

Usage: python3 -B scripts/compare_dit_runs.py --before OLD --after NEW
       [--validation-baseline results/dit_inference_check_20261008]

Writes NEW/paired_comparison.json, NEW/comparison.html and
NEW/before_after/{instance}.png. Never loads a model or selects a checkpoint.
"""
import argparse
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vae_v2.config import atomic_json, digest

BASELINE = 'correct_s20_cfg1'
CONDITION = 'GT-derived six-view segmentation latents + native shape features; known mesh geometry'
SEED_POLICY = 'training_seed_plus_first8_hex_instance'
ROLES = {'test': 'test', 'train_diagnostic': 'train',
         'validation_diagnostic': 'validation', 'validation_parity': 'validation'}
COMMON = ('cache_digest', 'vae_checkpoint', 'vae_sha256', 'source_manifest_digest',
          'normalization', 'base_config', 'base_digest', 'vae_config')
CACHE_FIELDS = ('instance', 'split', 'type_id', 'path', 'bytes', 'mtime_ns', 'sha256',
                'plan_digest', 'tokens', 'mv_tokens', 'shape_coverage', 'shape_source',
                'source_path', 'source_bytes', 'source_mtime_ns', 'parts')
METRICS = tuple(f'{group}.{key}' for group in ('surface', 'projection', 'voxel', 'hidden_surface')
                for key in ('miou', 'accuracy', 'small_part_recall')) + (
    'boundary_f1', 'false_internal_boundary_rate', 'isolated_error_area_fraction',
    'near_part_separation_accuracy', 'query_fallback_fraction', 'unknown_prediction_area')
GUITARS = ('251c8125765149c0bd46b3a1a51ff4b6', 'd1d58f05c346470e9805612d562e9dde')
NOTE = ('Developmental regression evaluation on previously inspected objects; '
        'not a fresh blind test and no checkpoint selection from test results. '
        'One fixed generation seed per object, paired across runs; this does not '
        'measure variation across seeds. Labels are compared on the same known '
        'object geometry with GT-derived conditioning. Training and validation '
        'diagnostics are excluded from every test aggregate and ranking.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read {path}; wait for completed worker export and merge: {exc}') from exc


def index_rows(rows, context):
    result = {}
    for row in rows:
        uid = row.get('instance', '')
        require(isinstance(uid, str) and re.fullmatch('[0-9a-f]{32}', uid), f'{context}: invalid instance {uid}')
        require(uid not in result, f'{context}: duplicate instance {uid}')
        result[uid] = row
    return result


def unique_ids(ids, context, count=128):
    require(len(ids) == count and len(set(ids)) == count, f'{context}: expected exactly {count} unique IDs')
    return set(ids)


def source_identity(row):
    require(all(k in row for k in CACHE_FIELDS), f'{row.get("instance")}: incomplete source/cache identity')
    # A sweep sample has no role; a validation sample can acquire a diagnostic role.
    return {k: v for k, v in row.items() if k != 'role'}


def expected_seed(plan, uid):
    return plan['config']['seed'] + int(uid[:8], 16)


def check_plan(plan, sweep=False):
    require(all(k in plan for k in COMMON), 'Incomplete frozen plan contract')
    require(plan['base_digest'] == digest(plan['base_config']), 'base_config/base_digest mismatch')
    require(plan['vae_config'].get('num_classes') == 64, 'Expected 64-class VAE')
    require(plan.get('label_matching') is False and plan.get('postprocessing') is False,
            'Label matching and postprocessing must be disabled')
    require(type(plan['config'].get('seed')) is int, 'Missing integer generation seed')
    for scope in (plan, plan['config']):
        for key in ('cfg', 'cfg_scale', 'guidance_scale'):
            require(scope.get(key, 1) == 1 and not isinstance(scope.get(key), bool), 'Expected no CFG (scale 1)')
    selected = plan['selection']
    require(selected.get('split') == 'validation' and selected.get('test_used_for_selection') is False,
            'Checkpoint selection must use validation only')
    unique_ids(selected['ids'], 'selection validation cohort')
    require(selected['checkpoint'] == plan['checkpoint']['path'], 'Selection checkpoint differs from plan')
    require(selected['sampling_steps'] == 20, 'Expected 20 sampling steps')
    rows = index_rows(plan['samples'], 'plan samples')
    for row in rows.values():
        source_identity(row)
        require(type(row['parts']) is int and 1 <= row['parts'] <= 63, 'Part count outside 1..63')
        if sweep:
            require(row['split'] == 'validation', 'Sweep contains a non-validation object')
        else:
            require(row.get('role') in ROLES and ROLES[row['role']] == row['split'], 'Invalid role/split')
    if sweep:
        require(plan.get('split') == 'validation' and plan.get('test_used') is False, 'Sweep must be validation only')
        require(set(rows) == unique_ids(selected['ids'], 'sweep selection') and len(rows) == 128,
                'Sweep selection/cohort mismatch')
        require(plan.get('seed_policy') == f"{plan['config']['seed']} + int(instance[:8],16); identical x1 across settings",
                'Unknown sweep seed policy')
        names = [s['name'] for s in plan['settings']]
        require(len(names) == len(set(names)), 'Duplicate sweep setting')
        settings = [s for s in plan['settings'] if s['name'] == BASELINE]
        require(len(settings) == 1 and settings[0] == {
            'name': BASELINE, 'steps': 20, 'cfg': 1.0, 'condition': 'correct', 'eligible': True},
            'Sweep baseline must be correct_s20_cfg1, correct conditioning, 20 steps, no CFG')
    else:
        require(plan.get('test_used_for_selection') is False, 'Test used for checkpoint selection')
        require(plan.get('seed_policy') == SEED_POLICY, 'Unknown seed policy')
        require(plan.get('condition') == CONDITION, 'Unexpected condition')
        require(plan.get('sampling_steps') == 20, 'Expected 20 sampling steps')
        tests = unique_ids([u for u, r in rows.items() if r['role'] == 'test'], 'test cohort')
        require(not tests.intersection(selected['ids']), 'Test IDs leaked into checkpoint selection')
        require(all(r['instance'] in selected['ids'] for r in rows.values() if r['split'] == 'validation'),
                'Validation diagnostic outside selection cohort')
    return rows


def validate_plans(before, after, baseline=None):
    a, b = check_plan(before), check_plan(after)
    for key in COMMON + ('seed_policy', 'sampling_steps', 'condition', 'generation', 'oracle'):
        require(key in before and key in after and before[key] == after[key], f'Frozen contract mismatch: {key}')
    for key in ('seed', 'cache_dir', 'max_tokens'):
        require(before['config'][key] == after['config'][key], f'Frozen config mismatch: {key}')
    require(set(before['selection']['ids']) == set(after['selection']['ids']), 'Validation selection cohort changed')
    for role in ('test', 'train_diagnostic'):
        require({u for u, r in a.items() if r['role'] == role} == {u for u, r in b.items() if r['role'] == role},
                f'{role} cohort changed')
    for uid in a.keys() & b.keys():
        require(source_identity(a[uid]) == source_identity(b[uid]), f'Source/cache identity changed: {uid}')
        if a[uid]['split'] != 'validation':
            require(a[uid]['role'] == b[uid]['role'], f'Role changed: {uid}')
    s = None
    if baseline is not None:
        s = check_plan(baseline, sweep=True)
        for key in COMMON + ('checkpoint',):
            require(before[key] == baseline[key], f'Validation baseline mismatch: {key}')
        for key in ('seed', 'cache_dir', 'max_tokens'):
            require(before['config'][key] == baseline['config'][key], f'Validation baseline config mismatch: {key}')
        require(set(s) == set(before['selection']['ids']), 'Baseline validation cohort changed')
        for uid, row in b.items():
            if row['role'] == 'validation_diagnostic':
                require(uid in s and source_identity(row) == source_identity(s[uid]),
                        f'Validation baseline source/cache identity changed: {uid}')
    return a, b, s


def metric_value(record, path):
    if record is None:
        return None
    value = record
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    require(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(value)), f'Invalid metric {path}: {value}')
    return value


def part_identity(metrics):
    parts = metrics.get('parts')
    require(isinstance(parts, list) and parts, 'Missing per-part metrics')
    labels = [p['label'] for p in parts]
    require(len(labels) == len(set(labels)) and all(type(k) is int and 1 <= k <= 63 for k in labels),
            'Duplicate or invalid part labels')
    for p in parts:
        require(isinstance(p['area_fraction'], (int, float)) and math.isfinite(p['area_fraction'])
                and 0 <= p['area_fraction'] <= 1, 'Invalid part area fraction')
    return sorted((p['label'], p['area_fraction']) for p in parts)


def check_record(record, row, plan, sweep=False):
    uid = row['instance']
    for key in ('instance', 'split', 'parts') + (() if sweep else ('role', 'type_id', 'shape_coverage')):
        require(record.get(key) == row[key], f'Metric/source {key} mismatch: {uid}')
    require(record.get('plan_digest') == digest(plan), f'Metric plan_digest mismatch: {uid}')
    require(record.get('seed') == expected_seed(plan, uid), f'Metric seed mismatch: {uid}')
    if sweep:
        require(record.get('setting') == next(s for s in plan['settings'] if s['name'] == BASELINE),
                f'Baseline metric setting mismatch: {uid}')
        require(record.get('nfe') == 20, f'Baseline is not no-CFG/20 steps: {uid}')
        scores = [record['metrics']]
    else:
        scores = [record['dit'], record['oracle']]
    for score in scores:
        for key in METRICS:
            metric_value(score, key)
        require(len(part_identity(score)) == row['parts'], f'Part count mismatch: {uid}')
    if not sweep:
        require(part_identity(scores[0]) == part_identity(scores[1]), f'DiT/oracle part labels or area differ: {uid}')


def report_rows(report, plan):
    rows = index_rows(plan['samples'], 'plan')
    for key in ('checkpoint', 'vae_checkpoint', 'cache_digest', 'sampling_steps', 'condition', 'prediction_dir'):
        require(report.get(key) == plan[key], f'Merged report/plan mismatch: {key}')
    require(report.get('test_used_for_selection') is False, 'Merged report used test selection')
    ids = unique_ids(report['ids'], 'merged test IDs')
    require(ids == {u for u, r in rows.items() if r['role'] == 'test'}, 'Merged test IDs differ from plan')
    collected = []
    for section, role in (('samples', 'test'), ('diagnostics', 'train_diagnostic'),
                          ('validation_diagnostics', 'validation_diagnostic')):
        section_rows = report.get(section, [])
        indexed = index_rows(section_rows, section)
        require(set(indexed) == {u for u, r in rows.items() if r['role'] == role}, f'Incomplete {section} cohort')
        for uid, record in indexed.items():
            check_record(record, rows[uid], plan)
        collected.extend(section_rows)
    return index_rows(collected, 'merged metrics')


def make_pair(row, old, new, old_source, before_plan, after_plan):
    a = None if old is None else old.get('dit', old.get('metrics'))
    b = new['dit']
    if a is not None:
        require(part_identity(a) == part_identity(b), f'Old/new part labels or area differ: {row["instance"]}')
    def delta(key):
        x, y = metric_value(a, key), metric_value(b, key)
        return None if x is None or y is None else y - x
    return {'instance': row['instance'], 'role': row['role'], 'split': row['split'],
            'type_id': row['type_id'], 'parts': row['parts'], 'seed': new['seed'],
            'before_source': old_source, 'paired': old is not None,
            'before_plan_digest': digest(before_plan) if before_plan is not None else None,
            'after_plan_digest': digest(after_plan),
            'before': a, 'after': b,
            'oracle_before': None if old is None else old.get('oracle'), 'oracle_after': new['oracle'],
            'metrics': {key: {'before': metric_value(a, key), 'after': metric_value(b, key),
                              'delta': delta(key)} for key in METRICS},
            'surface_miou_delta': delta('surface.miou'),
            'surface_accuracy_delta': delta('surface.accuracy')}


def prepare_pairs(before_plan, after_plan, before_report, after_report, baseline_plan=None, baseline_rows=None):
    a, b, s = validate_plans(before_plan, after_plan, baseline_plan)
    old, new = report_rows(before_report, before_plan), report_rows(after_report, after_plan)
    baseline_rows = {} if baseline_rows is None else baseline_rows
    pairs = []
    for uid, row in sorted(b.items()):
        if row['role'] == 'validation_parity':
            continue
        prior, source, plan = old.get(uid), 'before', before_plan
        if prior is None and row['role'] == 'validation_diagnostic' and s is not None:
            require(uid in baseline_rows, f'Missing validation baseline metrics: {uid}')
            prior = baseline_rows[uid]
            check_record(prior, s[uid], baseline_plan, sweep=True)
            source, plan = 'validation_baseline', baseline_plan
        if prior is None:
            require(row['role'] == 'validation_diagnostic', f'Missing paired object: {uid}')
            source, plan = None, None
        pairs.append(make_pair(row, prior, new[uid], source, plan, after_plan))
    return pairs


def average(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def aggregate(pairs, oracle=False):
    """Common finite support for each metric; missing values cannot change pairing."""
    result = {}
    for key in METRICS + ('surface_miou_p10',):
        lookup = 'surface.miou' if key == 'surface_miou_p10' else key
        sides = ('oracle_before', 'oracle_after') if oracle else ('before', 'after')
        values = [(metric_value(p[sides[0]], lookup), metric_value(p[sides[1]], lookup)) for p in pairs]
        valid = [(a, b) for a, b in values if a is not None and b is not None]
        reducer = (lambda x: float(np.quantile(x, .1)) if x else None) if key == 'surface_miou_p10' else average
        a, b = (reducer([v[i] for v in valid]) for i in (0, 1))
        result[key] = {'before': a, 'after': b, 'delta': None if a is None else b - a,
                       'paired_count': len(valid), 'before_count': sum(a is not None for a, _ in values),
                       'after_count': sum(b is not None for _, b in values)}
    return result


def changes(pairs):
    ds = [p['surface_miou_delta'] for p in pairs if p['surface_miou_delta'] is not None]
    # Decimal thresholds must not turn a mathematically exact 1pp into >1pp.
    return {f'{direction}_over_{pp}pp': sum(sign * d > pp / 100 and
            not math.isclose(sign * d, pp / 100, rel_tol=0, abs_tol=1e-12) for d in ds)
            for direction, sign in (('improved', 1), ('regressed', -1)) for pp in (1, 5)}


def build_report(pairs, before_plan, after_plan, baseline_plan=None):
    tests = [p for p in pairs if p['role'] == 'test']
    unique_ids([p['instance'] for p in tests], 'aggregate tests')
    ranked = [p for p in tests if p['surface_miou_delta'] is not None]
    result = {'created_at': datetime.now().astimezone().isoformat(), 'note': NOTE,
              'before_checkpoint': before_plan['checkpoint'], 'after_checkpoint': after_plan['checkpoint'],
              'vae_checkpoint': after_plan['vae_checkpoint'], 'cache_digest': after_plan['cache_digest'],
              'test_used_for_selection': False, 'seed_count_per_object': 1,
              'seed_policy': after_plan['seed_policy'], 'sampling_steps': 20, 'cfg': 1,
              'condition': after_plan['condition'], 'test_count': len(tests),
              'metric_policy': 'Reuse validated per-object metrics; recompute aggregates on common non-None pairs. '
                               'Deltas are after-minus-before fractions (multiply by 100 for percentage points). '
                               'Small-part recall uses the evaluator definition (<1% of valid area/weight).',
              'metrics': aggregate(tests), 'oracle_metrics': aggregate(tests, oracle=True),
              'groups_by_parts': [{'parts': f'{lo}-{hi}', 'count': len(group), 'metrics': aggregate(group),
                                   **changes(group)} for lo, hi in ((1, 20), (21, 32), (33, 63))
                                  for group in [[p for p in tests if lo <= p['parts'] <= hi]]],
              **changes(tests),
              'gains': sorted([p for p in ranked if p['surface_miou_delta'] > 0], key=lambda p: -p['surface_miou_delta'])[:10],
              'regressions': sorted([p for p in ranked if p['surface_miou_delta'] < 0], key=lambda p: p['surface_miou_delta'])[:10],
              'pairs': pairs}
    for role, name in (('train_diagnostic', 'training_diagnostics'), ('validation_diagnostic', 'validation_diagnostics')):
        group = [p for p in pairs if p['role'] == role]
        result[name] = {'excluded_from_test': True, 'count': len(group),
                        'paired_count': sum(p['paired'] for p in group), 'metrics': aggregate(group),
                        'ids': [p['instance'] for p in group],
                        'unpaired_ids': [p['instance'] for p in group if not p['paired']]}
    result['validation_diagnostics']['baseline_setting'] = BASELINE if baseline_plan is not None else None
    result['validation_diagnostics']['baseline_checkpoint'] = None if baseline_plan is None else baseline_plan['checkpoint']
    result['validation_parity'] = 'Excluded from comparison; parity is checked by the evaluator merge.'
    return result


def load_geometry(row):
    from vae_v2.data import load_sample
    path=Path(row['source_path']);stat=path.stat()
    require((stat.st_size,stat.st_mtime_ns)==(row['source_bytes'],row['source_mtime_ns']),
            f'Geometry file changed: {row["instance"]}')
    return load_sample(path)


def verify_geometry_metadata(meta, row, source_row):
    """Sealed manifest owns membership; immutable NPZ retains preparation split."""
    uid=row['instance']
    require(meta.get('instance')==uid==source_row['instance'],f'GT geometry instance mismatch: {uid}')
    require(source_row['split']==row['split'],f'Current source manifest split mismatch: {uid}')
    require(meta.get('split')==source_row.get('preparation_split',source_row['split']),
            f'GT preparation split mismatch: {uid}')
    require(meta.get('num_classes')==64,f'GT geometry class count mismatch: {uid}')
    require((source_row['bytes'],source_row['mtime_ns'])==(row['source_bytes'],row['source_mtime_ns']),
            f'Geometry source file identity mismatch: {uid}')


def check_geometry(row, plan, geometry, source_row):
    uid = row['instance']
    with np.load(row['source_path'], allow_pickle=False) as source:
        meta = json.loads(str(source['metadata']))
    verify_geometry_metadata(meta,row,source_row)
    require((Path(plan['vae_config']['prepared_dir'])/source_row['path']).resolve()==Path(row['source_path']).resolve(),
            f'Geometry source path mismatch: {uid}')
    path = Path(plan['config']['cache_dir']) / row['path']
    stat = path.stat()
    require((stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns']), f'Cache file identity changed: {uid}')
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    require(h.hexdigest() == row['sha256'], f'Cache checksum changed: {uid}')
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata']))
        for key in ('instance', 'split', 'plan_digest'):
            require(metadata.get(key) == row[key], f'Cache metadata {key} mismatch: {uid}')
        require(metadata.get('vae_sha256') == plan['vae_sha256'], f'Cache VAE mismatch: {uid}')
        coords = cache['latent_coords']
    require(coords.shape == (row['tokens'], 3) and np.issubdtype(coords.dtype, np.integer), f'Bad latent coordinates: {uid}')
    require(len(np.unique(coords, axis=0)) == len(coords), f'Duplicate latent coordinates: {uid}')
    require(np.array_equal(np.unique(geometry['coords'] // 16, axis=0), np.unique(coords, axis=0)),
            f'GT/latent coordinate identity mismatch: {uid}')
    raster = geometry['face_raster']
    require(raster.ndim == 3 and raster.shape[0] == 6 and np.issubdtype(raster.dtype, np.integer)
            and raster.min() >= -1 and raster.max() < len(geometry['face_labels']), f'Invalid GT six-view raster: {uid}')
    return meta


def load_prediction(plan, row, geometry, sweep=False):
    uid = row['instance']
    root = Path(plan['prediction_dir'])
    path = (root / BASELINE if sweep else root) / f'{uid}.npz'
    with np.load(path, allow_pickle=False) as archive:
        require(archive['plan_digest'].shape == () and str(archive['plan_digest']) == digest(plan),
                f'Prediction plan_digest mismatch: {uid}')
        require(archive['seed'].shape == () and np.issubdtype(archive['seed'].dtype, np.integer)
                and int(archive['seed']) == expected_seed(plan, uid), f'Prediction seed mismatch: {uid}')
        names = ('face_labels', 'voxel_labels') + (() if sweep else ('oracle_faces', 'oracle_voxels'))
        result = {key: archive[key] for key in names}
        for key in ('coords', 'voxel_coords'):
            if key in archive:
                require(np.array_equal(archive[key], geometry['coords']), f'Prediction coordinate mismatch: {uid}')
    for key, value in result.items():
        size = len(geometry['voxel_labels'] if 'voxel' in key else geometry['face_labels'])
        require(value.shape == (size,) and np.issubdtype(value.dtype, np.integer)
                and np.all((value >= 0) & (value < 64)), f'Invalid prediction {key}: {uid}')
    return result


def verify_parts(pair, geometry):
    labels = geometry['face_labels']
    area = geometry['face_area'].astype(np.float64)
    expected = [(int(k), float(area[labels == k].sum() / area.sum())) for k in np.unique(labels[labels > 0])]
    for key in ('before', 'after', 'oracle_before', 'oracle_after'):
        if pair[key] is not None:
            actual = part_identity(pair[key])
            require([k for k, _ in actual] == [k for k, _ in expected] and
                    np.allclose([a for _, a in actual], [a for _, a in expected], rtol=0, atol=1e-12),
                    f'GT geometry part labels/area mismatch: {pair["instance"]} {key}')


def verify_oracles(uid, old, new):
    for key in ('oracle_faces', 'oracle_voxels'):
        require(np.array_equal(old[key], new[key]), f'Oracle {key} changed: {uid}')


def gt_tiles(geometry):
    from PIL import Image
    from utils.seg_projection import render_labels
    from vae_v2.visuals import render_palette
    maps = render_labels(geometry['face_labels'] + 1, geometry['face_raster'])
    return [Image.fromarray(render_palette(64)[view]).resize((256, 256), Image.Resampling.NEAREST) for view in maps]


def verify_gt_views(directory, uid, tiles, sweep=False):
    from PIL import Image
    path = Path(directory) / 'objects' / uid / ('comparison.png' if sweep else f'{uid}_comparison.png')
    with Image.open(path) as image:
        require(image.width == 1536 and image.height >= (320 if sweep else 890), f'Unexpected GT panel size: {path}')
        image = image.convert('RGB')
        y = 64 if sweep else 74
        for i, tile in enumerate(tiles):
            require(image.crop((256*i, y, 256*(i+1), y+256)).tobytes() == tile.tobytes(),
                    f'GT views/camera/palette differ: {path}, view {i}')


def artifact_links(pair, before, after, baseline=None):
    uid = pair['instance']
    paths = {'new DiT GLB': Path(after) / 'objects' / uid / f'{uid}_dit.glb',
             'GT GLB': Path(after) / 'objects' / uid / f'{uid}_gt.glb',
             'VAE GLB': Path(after) / 'objects' / uid / f'{uid}_vae.glb'}
    if pair['before_source'] == 'before':
        paths['old DiT GLB'] = Path(before) / 'objects' / uid / f'{uid}_dit.glb'
    elif pair['before_source'] == 'validation_baseline':
        paths['old DiT GLB (validation baseline)'] = Path(baseline) / 'objects' / uid / f'{BASELINE}.glb'
    return {label: os.path.relpath(path, after) for label, path in paths.items()}


def format_metric(value, delta=False):
    if value is None:
        return 'N/A'
    return f'{100*value:+.2f} pp' if delta else f'{100*value:.2f}%'


def render_panel(path, pair, geometry, old, new):
    from PIL import Image, ImageDraw
    from utils.seg_projection import render_labels, VIEW_NAMES
    from vae_v2.visuals import render_palette
    canvas = Image.new('RGB', (1536, 1160), 'white')
    draw = ImageDraw.Draw(canvas)
    score = pair['metrics']['surface.miou']
    draw.text((8, 8), f'{pair["instance"]} | {pair["role"]} | {pair["parts"]} parts | '
              f'surface mIoU {format_metric(score["before"])} -> {format_metric(score["after"])} '
              f'({format_metric(score["delta"], True)})', fill='black')
    palette = render_palette(64)
    rows = [('GT', geometry['face_labels']), ('oldDiT', None if old is None else old['face_labels']),
            ('newDiT', new['face_labels']), ('VAE', new['oracle_faces'])]
    for row, (name, labels) in enumerate(rows):
        maps = None if labels is None else render_labels(labels + 1, geometry['face_raster'])
        y = 40 + row * 280
        for i, view in enumerate(VIEW_NAMES):
            draw.text((i*256+4, y+3), f'{name}: {view}', fill='black')
            if maps is None:
                draw.text((i*256+8, y+100), 'Old validation baseline unavailable', fill='black')
            else:
                tile = Image.fromarray(palette[maps[i]]).resize((256, 256), Image.Resampling.NEAREST)
                canvas.paste(tile, (i*256, y+24))
    canvas.save(path)


def html_document(report):
    def esc(value):
        return html.escape(str(value), quote=True)
    def table(metrics):
        return ('<table><tr><th>Metric</th><th>Old</th><th>New</th><th>Delta</th><th>Pairs</th></tr>' +
                ''.join(f'<tr><td>{esc(k)}</td><td>{format_metric(v["before"])}</td>'
                        f'<td>{format_metric(v["after"])}</td><td>{format_metric(v["delta"], True)}</td>'
                        f'<td>{v["paired_count"]}</td></tr>' for k, v in metrics.items()) + '</table>')
    def card(pair):
        uid = pair['instance']
        score = pair['metrics']['surface.miou']
        links = ' · '.join(f'<a href="{esc(path)}">{esc(label)}</a>' for label, path in pair['links'].items())
        return (f'<section><h3>{uid} | {pair["parts"]} parts | {esc(pair["role"])}</h3>'
                f'<p>Surface mIoU: {format_metric(score["before"])} to {format_metric(score["after"])} '
                f'({format_metric(score["delta"], True)}). {links}</p>'
                f'<p>Old source: {esc(pair["before_source"] or "unavailable; supply --validation-baseline")}. '
                'VAE row uses the new run oracle; paired run oracles must match exactly.</p>'
                f'<a href="before_after/{uid}.png"><img loading="lazy" alt="GT / oldDiT / newDiT / VAE six views: {uid}" '
                f'src="before_after/{uid}.png"></a></section>')
    pairs = report['pairs']
    body = ('<!doctype html><html><head><meta charset="utf-8"><title>Paired DiT comparison</title>'
            '<style>body{max-width:1536px;margin:auto;padding:24px;font:16px sans-serif}'
            'img{width:100%}section{margin:32px 0}table{border-collapse:collapse}'
            'td,th{padding:7px;border:1px solid #ccc}</style></head><body><h1>Old / new DiT on fixed geometry</h1>'
            f'<p>{esc(report["note"])}</p><p>Rows: GT / oldDiT / newDiT / VAE. Six cached views, original labels, '
            'no relabeling, alignment, or smoothing. 20 Euler steps; no CFG.</p>'
            f'<p>Old checkpoint: {esc(report["before_checkpoint"]["path"])}<br>'
            f'New checkpoint: {esc(report["after_checkpoint"]["path"])}</p>'
            '<p><a href="paired_comparison.json">Full paired metrics JSON</a></p><h2>128 test objects only</h2>' +
            f'<p>{esc(report["metric_policy"])}</p>' + table(report['metrics']) +
            '<p>' + esc(', '.join(f'{k}: {report[k]}' for k in changes([]))) + '</p>')
    for group in report['groups_by_parts']:
        body += f'<h3>{group["parts"]} parts ({group["count"]} test objects)</h3>' + table(group['metrics'])
    body += '<h2>Guitars</h2>' + ''.join(card(p) for p in pairs if p['instance'] in GUITARS)
    body += '<h2>Historical training diagnostics — excluded from test</h2>' + ''.join(card(p) for p in pairs if p['role'] == 'train_diagnostic')
    body += '<h2>Largest test regressions</h2>' + ''.join(card(p) for p in report['regressions'])
    body += '<h2>All 128 test objects</h2>' + ''.join(card(p) for p in pairs if p['role'] == 'test')
    body += ('<h2>Validation diagnostics — separate, excluded from test</h2>'
             '<p>The optional old baseline is only correct_s20_cfg1 from the validation sweep at the old checkpoint. '
             'The sweep has no oracle; VAE uses the new run oracle. Missing old baselines are explicitly unpaired. '
             'Validation parity objects are not compared.</p>' +
             table(report['validation_diagnostics']['metrics']) +
             ''.join(card(p) for p in pairs if p['role'] == 'validation_diagnostic'))
    return body + '</body></html>\n'


def compare_runs(before, after, validation_baseline=None):
    before, after = Path(before).resolve(), Path(after).resolve()
    baseline = None if validation_baseline is None else Path(validation_baseline).resolve()
    pa, pb = read_json(before / 'plan.json'), read_json(after / 'plan.json')
    ma, mb = read_json(before / 'metrics.json'), read_json(after / 'metrics.json')
    ps = None if baseline is None else read_json(baseline / 'plan.json')
    # Validate complete metadata before opening any predictions or writing outputs.
    a, b, s = validate_plans(pa, pb, ps)
    sweep_rows = {} if baseline is None else {
        uid: read_json(baseline / 'metrics' / BASELINE / f'{uid}.json')
        for uid, row in b.items() if row['role'] == 'validation_diagnostic'}
    pairs = prepare_pairs(pa, pb, ma, mb, ps, sweep_rows)
    for pair in pairs:
        pair['links'] = artifact_links(pair, before, after, baseline)
        for path in pair['links'].values():
            artifact = after / path
            require(artifact.is_file() and artifact.stat().st_size > 0, f'Missing GLB export: {artifact}')
    report = build_report(pairs, pa, pb, ps)
    source_manifest=read_json(Path(pb['vae_config']['prepared_dir'])/'manifest.json')
    require(digest(source_manifest)==pb['source_manifest_digest'],'Sealed VAE source manifest changed')
    source_rows={r['instance']:r for r in source_manifest['samples']}
    require(len(source_rows)==len(source_manifest['samples']),'Duplicate sealed geometry IDs')
    require(all(p['instance'] in source_rows for p in pairs),'Missing selected geometry IDs')
    # Keep only label arrays after verification, and render only after ALL objects pass.
    # Geometry is reloaded one object at a time to bound memory on the CPU host.
    predictions = {}
    for pair in pairs:
        uid = pair['instance']
        geometry = load_geometry(b[uid])
        check_geometry(b[uid], pb, geometry, source_rows[uid])
        verify_parts(pair, geometry)
        new = load_prediction(pb, b[uid], geometry)
        old = None
        tiles = gt_tiles(geometry)
        verify_gt_views(after, uid, tiles)
        if pair['before_source'] is not None:
            sweep = pair['before_source'] == 'validation_baseline'
            old = load_prediction(ps if sweep else pa, s[uid] if sweep else a[uid], geometry, sweep)
            verify_gt_views(baseline if sweep else before, uid, tiles, sweep)
            if not sweep:
                verify_oracles(uid, old, new)
        pair['verification'] = {'prediction_plan_digests': True, 'seeds': True, 'cache_checksum': True,
                                'geometry_coordinates': True, 'gt_views': True, 'part_labels_and_area': True,
                                'oracle_exact_match': None if old is None or pair['before_source'] == 'validation_baseline' else True}
        predictions[uid] = (old, new)
    target = after / 'before_after'
    target.mkdir(exist_ok=True)
    for pair in pairs:
        uid = pair['instance']
        geometry = load_geometry(b[uid])
        old, new = predictions.pop(uid)
        render_panel(target / f'{uid}.png', pair, geometry, old, new)
    report['artifacts'] = {'json': str(after / 'paired_comparison.json'), 'html': str(after / 'comparison.html'),
                           'images': str(target), 'image_size': [1536, 1160], 'image_count': len(pairs)}
    atomic_json(after / 'paired_comparison.json', report)
    (after / 'comparison.html').write_text(html_document(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--validation-baseline', type=Path,
                        help='Old validation sweep; reads only correct_s20_cfg1, never the selected best setting')
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.before, args.after, args.validation_baseline)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Comparison refused: {exc}\n')
    print(json.dumps({'test_count': report['test_count'], **changes([p for p in report['pairs'] if p['role'] == 'test']),
                      'artifacts': report['artifacts']}, indent=2))


if __name__ == '__main__':
    main()
