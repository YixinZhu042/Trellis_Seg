"""Seal an explicitly requested parts64 split after auditing successful caches.

Never mutates raw annotations, cached tensors or historical preparation shards.
Every excluded object must match the audited preparation failure list exactly.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vae_v2.config import load_config, atomic_json
from vae_v2.data import cache_contract, source_rows
from vae_v2.splits import seal_split_manifest, validate_split_manifest
from scripts.prepare_seg_vae_v2 import output_lock, parallel_results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'configs/seg_vae_v2_parts64.json')
    ap.add_argument('--audit', type=Path, default=ROOT/'eval/seg_vae_v2_parts64_cache_audit.json')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('workers must be positive')
    cfg = load_config(args.config)
    spec = cfg.get('dataset_split')
    if not spec:
        ap.error('Configuration requires an explicit dataset_split policy')
    root = Path(cfg['prepared_dir'])
    sources = source_rows(cfg)
    audit = json.loads(args.audit.read_text())
    if audit['cache_root'] != str(root) or audit['cache_errors']:
        raise ValueError('Wrong or failing cache audit')
    with output_lock(root, 0, merge=True):
        if (root/'manifest.json').exists():
            manifest = json.loads((root/'manifest.json').read_text())
            validate_split_manifest(manifest, sources, spec)
            print('Already sealed:', manifest['split_counts'], flush=True)
            return
        shards = [json.loads(p.read_text()) for p in sorted(root.glob('prep_rank*.json'))]
        world = audit['world_size']
        if len(shards) != world or {s['rank'] for s in shards} != set(range(world)):
            raise ValueError('Incomplete rank reports')
        for shard in shards:
            if shard['world_size'] != world or shard['contract'] != cache_contract(cfg):
                raise ValueError('Shard contract changed')
            expected = {r['instance'] for r in sources[shard['rank']::world]}
            actual = [r['instance'] for r in shard['samples']] + [r['instance'] for r in shard['failures']]
            if set(actual) != expected or len(actual) != len(expected):
                raise ValueError('Rank object accounting changed')
        rows = [r for s in shards for r in s['samples']]
        failures = [f for s in shards for f in s['failures']]
        if audit['verified_objects'] != len(rows) or audit['expected_objects'] != len(sources):
            raise ValueError('Audit object counts differ from preparation')
        audited_failures = {r['instance']: r['error'] for r in audit['source_failures']}
        if audited_failures != {r['instance']: r['error'] for r in failures}:
            raise ValueError('Quarantine differs from audited failures')
        provenance = {'path': str(args.audit.resolve()),
                      'sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest()}
        manifest = seal_split_manifest(cache_contract(cfg), sources, rows, failures, spec, provenance)
        validate_split_manifest(manifest, sources, spec)
        started = time.monotonic()
        verified = 0
        for row, error in parallel_results((('verify', r, cfg, root) for r in rows), args.workers):
            if error:
                raise ValueError(f'Cache no longer passes audit: {error}')
            verified += 1
            if verified % 2000 == 0:
                print(f'Checked {verified}/{len(rows)}; elapsed={time.monotonic()-started:.1f}s', flush=True)
        if verified != len(rows):
            raise ValueError('Incomplete cache validation')
        quarantine = {'excluded_objects': len(failures),
                      'reason': 'User approved using valid caches only; non-triangle/missing geometry cannot be safely mapped.',
                      'split_counts': dict(Counter(r['split'] for r in manifest['excluded'])),
                      'samples': manifest['excluded'], 'source_cache_audit': provenance}
        atomic_json(root/'quarantine.json', quarantine)
        split_file = {'dataset_split': spec, 'split_counts': manifest['split_counts'],
                      'samples': [{'instance': r['instance'], 'split': r['split'],
                                   'preparation_split': r['preparation_split']} for r in manifest['samples']]}
        atomic_json(root/'split_128.json', split_file)
        # Write manifest last: this is the only file consumed as a readiness signal.
        atomic_json(root/'manifest.json', manifest)
        print(json.dumps({'sealed': str(root/'manifest.json'), 'objects': len(rows),
                          'split_counts': manifest['split_counts'], 'quarantined': len(failures),
                          'cache_tensors_rewritten': 0}, indent=2), flush=True)


if __name__ == '__main__':
    main()
