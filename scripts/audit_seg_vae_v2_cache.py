"""Audit every successful preparation cache without dropping failures or sealing a dataset."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vae_v2.config import load_config, atomic_json
from vae_v2.data import source_rows, cache_contract
from scripts.prepare_seg_vae_v2 import parallel_results, output_lock


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'configs/seg_vae_v2_parts64.json')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('workers must be positive')
    cfg = load_config(args.config)
    root = Path(cfg['prepared_dir'])
    expected = {r['instance']: r for r in source_rows(cfg)}
    with output_lock(root, 0, merge=True):
        paths = sorted(root.glob('prep_rank*.json'))
        shards = [json.loads(p.read_text()) for p in paths]
        if not shards:
            raise ValueError('No preparation shards')
        world = shards[0]['world_size']
        if {r['rank'] for r in shards} != set(range(world)) or len(shards) != world:
            raise ValueError('Missing or duplicate rank reports')
        for shard in shards:
            if shard['world_size'] != world or shard['contract'] != cache_contract(cfg):
                raise ValueError('Mismatched shard contract')
        rows = [r for s in shards for r in s['samples']]
        failures = [f for s in shards for f in s['failures']]
        ids = [r['instance'] for r in rows] + [r['instance'] for r in failures]
        if len(ids) != len(set(ids)) or set(ids) != expected.keys():
            raise ValueError('Object accounting is incomplete or duplicated')
        for r in rows:
            source = expected[r['instance']]
            if any(r[k] != v for k, v in source.items()):
                raise ValueError(f'Changed source provenance: {r["instance"]}')
        start = time.monotonic()
        errors = []
        verified = 0
        tasks = (('verify', r, cfg, root) for r in rows)
        for result, error in parallel_results(tasks, args.workers):
            if error:
                errors.append(error)
            else:
                verified += 1
            if (verified + len(errors)) % 2000 == 0:
                print(f'Audit {verified}/{len(rows)}, errors={len(errors)}, '
                      f'elapsed={time.monotonic()-start:.1f}s', flush=True)
        pending = [dict(f, split=expected[f['instance']]['split'],
                        mesh_path=expected[f['instance']]['mesh_path'],
                        annotation_path=expected[f['instance']]['annotation_path']) for f in failures]
        summary = {
            'status': 'successful_caches_verified_pending_source_decision' if not errors and failures else
                      'all_caches_verified' if not errors else 'cache_errors',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'cache_root': str(root), 'world_size': world,
            'expected_objects': len(expected), 'verified_objects': verified,
            'successful_split_counts': dict(Counter(r['split'] for r in rows)),
            'source_failure_split_counts': dict(Counter(r['split'] for r in pending)),
            'source_failures': pending, 'cache_errors': errors,
            'cache_gib': sum(r['bytes'] for r in rows)/1024**3,
            'max_effective_parts': max(r['annotation_summary']['parts'] for r in rows),
            'mean_labeled_surface_fraction': sum(r['annotation_summary']['labeled_area_fraction'] for r in rows)/len(rows),
            'mean_labeled_pixel_fraction': sum(r['annotation_summary']['labeled_visible_pixel_fraction'] for r in rows)/len(rows),
            'seconds': time.monotonic()-start,
            'manifest_sealed': (root/'manifest.json').exists(),
            'note': 'No failed objects excluded; no training manifest created by this audit.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('source_failures','cache_errors')}, indent=2), flush=True)
        if errors:
            raise RuntimeError('Cache integrity errors')


if __name__ == '__main__':
    main()
