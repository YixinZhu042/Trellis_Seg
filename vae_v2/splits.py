"""Reproducible subset splits over audited caches, with explicit exclusions.

Cache tensors do not depend on train/validation/test membership. The sealed
manifest owns the current split; preparation_split preserves each NPZ's original
provenance. Existing caches and the original dataset selection are never edited.
"""
from collections import Counter
import hashlib

POLICY = 'nested_holdout_sha256_v1'
SPLITS = {'train', 'validation', 'test'}


def validate_spec(spec):
    if set(spec) != {'policy', 'seed', 'validation', 'test', 'quarantine_count'} or spec['policy'] != POLICY:
        raise ValueError('Unknown or incomplete dataset split policy')
    for key in ('seed', 'validation', 'test', 'quarantine_count'):
        if isinstance(spec[key], bool) or not isinstance(spec[key], int):
            raise ValueError(f'dataset_split.{key} must be an integer')
    if min(spec['validation'], spec['test']) < 1 or spec['quarantine_count'] < 0:
        raise ValueError('Invalid holdout size or quarantine count')


def assign_splits(sources, excluded_ids, spec):
    """Hash-select within original holdouts; every other usable object is train."""
    validate_spec(spec)
    sources = list(sources)
    by_id = {r['instance']: r for r in sources}
    if len(by_id) != len(sources) or any(r['split'] not in SPLITS for r in sources):
        raise ValueError('Duplicate source IDs or invalid original split')
    excluded = set(excluded_ids)
    if len(excluded) != spec['quarantine_count'] or not excluded <= by_id.keys():
        raise ValueError('Quarantine IDs do not match the declared policy')
    assigned = {uid: 'train' for uid in by_id if uid not in excluded}
    for split in ('validation', 'test'):
        pool = [r['instance'] for r in sources if r['split'] == split and r['instance'] not in excluded]
        pool.sort(key=lambda uid: (hashlib.sha256(f'{spec["seed"]}:{split}:{uid}'.encode()).hexdigest(), uid))
        if len(pool) < spec[split]:
            raise ValueError(f'Not enough usable objects in the original {split} pool')
        for uid in pool[:spec[split]]:
            assigned[uid] = split
    if len(assigned) <= spec['validation'] + spec['test']:
        raise ValueError('No training objects remain')
    return assigned


def seal_split_manifest(contract, sources, prepared_rows, failures, spec, audit):
    sources = list(sources)
    prepared_rows = list(prepared_rows)
    failures = list(failures)
    source_by_id = {r['instance']: r for r in sources}
    good_ids = [r['instance'] for r in prepared_rows]
    bad_ids = [r['instance'] for r in failures]
    combined = good_ids + bad_ids
    if len(combined) != len(set(combined)) or set(combined) != source_by_id.keys():
        raise ValueError('Successful and quarantined objects must account for every source exactly once')
    assignments = assign_splits(sources, bad_ids, spec)
    samples = []
    for row in prepared_rows:
        source = source_by_id[row['instance']]
        if any(row.get(k) != v for k, v in source.items()):
            raise ValueError(f'Prepared source provenance changed: {row["instance"]}')
        samples.append(dict({k: v for k, v in row.items() if k != 'reused'},
                            preparation_split=row['split'], split=assignments[row['instance']]))
    excluded = []
    for failure in failures:
        if not failure.get('error'):
            raise ValueError('Quarantined objects require an explicit error')
        source = source_by_id[failure['instance']]
        excluded.append(dict(source, error=failure['error']))
    counts = dict(Counter(r['split'] for r in samples))
    return {
        'contract': contract,
        'failures': [],
        'samples': sorted(samples, key=lambda r: r['instance']),
        'dataset_split': dict(spec),
        'split_counts': counts,
        'expected_source_objects': len(sources),
        'excluded': sorted(excluded, key=lambda r: r['instance']),
        'split_note': 'New parts64 training membership. Validation/test are hash-selected subsets of original holdouts; all other usable objects train. Historical v1 pretraining may include holdout objects.',
        'cache_split_note': 'NPZ metadata.split retains preparation provenance; manifest.split is authoritative for this experiment.',
        'source_cache_audit': audit,
    }


def validate_split_manifest(manifest, sources, spec):
    """Reject changed exclusions, overlap, manual membership edits and lost objects."""
    if manifest.get('dataset_split') != spec:
        raise ValueError('Sealed dataset split differs from training config')
    excluded = manifest.get('excluded', [])
    original = {r['instance']: r for r in sources}
    samples = manifest['samples']
    excluded_ids = [r['instance'] for r in excluded]
    all_ids = [r['instance'] for r in samples] + excluded_ids
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != original.keys():
        raise ValueError('Dataset split omits, duplicates, or adds source objects')
    if manifest.get('expected_source_objects') != len(original):
        raise ValueError('Source object count changed')
    assignments = assign_splits(sources, excluded_ids, spec)
    for row in excluded:
        if not row.get('error') or any(row.get(k) != v for k, v in original[row['instance']].items()):
            raise ValueError('Quarantine record differs from its raw source')
    for row in samples:
        source = original[row['instance']]
        if row.get('preparation_split') != source['split'] or row['split'] != assignments[row['instance']]:
            raise ValueError(f'Non-reproducible split assignment: {row["instance"]}')
        if any(row.get(k) != v for k, v in source.items() if k != 'split'):
            raise ValueError('Dataset source provenance changed')
    actual_counts = dict(Counter(row['split'] for row in samples))
    if manifest.get('split_counts') != actual_counts:
        raise ValueError('Manifest split count summary is incorrect')
    return assignments
