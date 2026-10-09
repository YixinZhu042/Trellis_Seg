"""Build geometry-aware VAE v2 caches from audited aligned mesh/face supervision."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vae_v2.config import load_config, atomic_json
from vae_v2.data import source_rows, cache_contract, validate_sample, load_sample
from vae_v2.geometry import build_geometry, attach_targets, near_surface_pairs


def default_workers():
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    return min(16, max(1, available // 2))


@contextmanager
def output_lock(output, rank, merge=False, diagnostic=False):
    """Different ranks can prepare together; merge/diagnostics must have exclusive access."""
    with (output / '.preparation.lock').open('a') as root_lock:
        mode = fcntl.LOCK_EX if merge or diagnostic else fcntl.LOCK_SH
        try:
            fcntl.flock(root_lock, mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another preparation/merge is active; wait for it to exit') from exc
        with (output / f'.rank{rank:04d}.lock').open('a') as rank_lock:
            try:
                fcntl.flock(rank_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Preparation rank {rank} is already running') from exc
            yield


def worker_init():
    # The parent handles termination, stops queuing, drains in-flight work, and
    # flushes its manifest. Ctrl-C must not corrupt worker output.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def execute_task(task):
    kind, row, cfg, output = task
    try:
        if kind == 'prepare':
            return prepare_one(row, cfg, output), None
        verify_one(row, cfg, output)
        return row, None
    except Exception as exc:
        return None, {'instance': row['instance'], 'error': f'{type(exc).__name__}: {exc}'}


def parallel_results(tasks, workers, stopped=lambda: False):
    """Bound memory and queued work; collect fast objects without head-of-line blocking."""
    if workers == 1:
        for task in tasks:
            if stopped():
                break
            yield execute_task(task)
        return
    # Set before spawn so NumPy/BLAS initialization also sees the limits.
    variables = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous = {key: os.environ.get(key) for key in variables}
    for key in variables:
        os.environ[key] = '1'
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=worker_init) as pool:
            iterator = iter(tasks)
            active = set()
            exhausted = False
            while active or not exhausted:
                while not exhausted and not stopped() and len(active) < workers * 2:
                    try:
                        task = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    active.add(pool.submit(execute_task, task))
                if not active:
                    break
                completed, active = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    yield future.result()
                if stopped():
                    exhausted = True
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def prepare_one(row, cfg, output):
    source = Path(row['source'])
    stat = source.stat()
    if stat.st_size != row['source_bytes'] or stat.st_mtime_ns != row['source_mtime_ns']:
        raise ValueError(f'Source sample changed after audit: {source}')
    metadata = dict(cache_contract(cfg), instance=row['instance'], split=row['split'],
                    source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns)
    if cfg.get('annotation_policy'):
        from vae_v2.raw_data import check_raw_source,raw_source_identity
        check_raw_source(row)
        metadata.update(raw_source_identity(row))
    path = output / 'samples' / f'{row["instance"]}.npz'
    reused = path.exists()
    if reused:
        with np.load(path, allow_pickle=False) as d:
            if json.loads(str(d['metadata'])) != metadata:
                raise ValueError(f'Existing sample contract changed: {path}; use a new cache directory')
        sample = load_sample(path)
    else:
        seed = (cfg['seed'] + int(hashlib.sha256(row['instance'].encode()).hexdigest()[:8], 16)) % 2**32
        if cfg.get('annotation_policy'):
            from vae_v2.raw_data import corrected_geometry
            vertices,faces,labels,raster,pixels,audit=corrected_geometry(row,cfg)
            geometry=build_geometry(vertices,faces,cfg['geometry'],seed)
            sample=attach_targets(geometry,labels,cfg['num_classes'])
            sample['face_raster']=raster
            sample['face_pixels']=pixels
            sample['annotation_audit']=np.array(json.dumps(audit,sort_keys=True))
            sample['near_pairs']=near_surface_pairs(geometry,cfg['geometry']['near_pair_radius_voxels'])
        else:
          with np.load(source, allow_pickle=False) as data:
            geometry = build_geometry(data['mesh_vertices'], data['mesh_faces'], cfg['geometry'], seed)
            sample = attach_targets(geometry, data['face_labels'])
            sample['face_pixels'] = data['face_pixels'].astype(np.int32)
            sample['near_pairs'] = near_surface_pairs(geometry, cfg['geometry']['near_pair_radius_voxels'])
        temporary = path.with_suffix('.npz.tmp')
        validate_sample(sample,cfg.get('num_classes',20))
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, metadata=np.array(json.dumps(metadata, sort_keys=True)), **sample)
        temporary.replace(path)
    if reused:
        validate_sample(sample,cfg.get('num_classes',20))
    stat = path.stat()
    entry = dict(row, path=str(path.relative_to(output)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                voxels=len(sample['coords']), faces=len(sample['face_labels']),
                mixed_voxels=int(((sample['voxel_target'] > 0).sum(1) > 1).sum()),
                components=int(sample['face_components'].max())+1, near_pairs=len(sample['near_pairs']),
                reused=reused)
    if int(sample.get('topology_repair', 0)):
        entry['topology_repair'] = 'nonmanifold_edges_on_budget_overflow_v1'
        entry['legacy_components'] = int(sample['legacy_component_count'])
    if 'annotation_audit' in sample:
        audit=json.loads(str(sample['annotation_audit']))
        entry['annotation_summary']={k:audit[k] for k in ('raw_parts','parts','annotated_faces','unannotated_faces',
                                                         'labeled_area_fraction','labeled_visible_pixel_fraction')}
    return entry


def verify_one(row, cfg, output):
    """Same full merge validation, now safely parallelized across distinct files."""
    path = output / row['path']
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
    if any(meta.get(k) != v for k, v in cache_contract(cfg).items()) or meta['instance'] != row['instance'] or meta['split'] != row['split']:
        raise ValueError('Cached sample contract differs from preparation shard')
    for key in ('source_bytes', 'source_mtime_ns'):
        if meta[key] != row[key]:
            raise ValueError('Cached sample source identity mismatch')
    if cfg.get('annotation_policy'):
        from vae_v2.raw_data import check_raw_source,raw_source_identity
        check_raw_source(row)
        if any(meta[k]!=v for k,v in raw_source_identity(row).items()):
            raise ValueError('Raw annotation identity mismatch in cache')
    validate_sample(load_sample(path),cfg.get('num_classes',20))
    stat = path.stat()
    if stat.st_size != row['bytes'] or stat.st_mtime_ns != row['mtime_ns']:
        raise ValueError('Sample changed during preparation')


def merge_prepared(rows, cfg, output, world_size, workers, stopped=lambda: False):
    expected_shards = {f'prep_rank{r:04d}.json' for r in range(world_size)}
    if {p.name for p in output.glob('prep_rank*.json')} != expected_shards:
        raise ValueError('Preparation rank files do not match --world_size')
    all_rows, failures = [], []
    for rank in range(world_size):
        shard = json.loads((output / f'prep_rank{rank:04d}.json').read_text())
        if shard['contract'] != cache_contract(cfg) or shard['rank'] != rank or shard['world_size'] != world_size:
            raise ValueError('Preparation shard contract mismatch')
        if shard.get('status') not in (None, 'complete'):
            raise ValueError(f'Rank {rank} is not complete')
        expected_ids = {r['instance'] for r in rows[rank::world_size]}
        if {r['instance'] for r in shard['samples']} != expected_ids:
            raise ValueError(f'Rank {rank} is incomplete or owns wrong objects')
        all_rows.extend(shard['samples'])
        failures.extend(shard['failures'])
    if failures or {(r['instance'], r['split']) for r in rows} != {(r['instance'], r['split']) for r in all_rows} or len(rows) != len(all_rows):
        raise ValueError('Cannot seal an incomplete cache; limited smoke caches cannot become training manifests')
    source_by_id = {r['instance']: r for r in rows}
    for row in all_rows:
        source_row = source_by_id.get(row['instance'])
        if source_row is None or any(row[k] != source_row[k] for k in ('source', 'source_bytes', 'source_mtime_ns')):
            raise ValueError('Stale source metadata in preparation shard')
        if cfg.get('annotation_policy'):
            from vae_v2.raw_data import raw_source_identity
            if raw_source_identity(row)!=raw_source_identity(source_row):
                raise ValueError('Stale raw annotation metadata in shard')
    start = last_print = time.monotonic()
    verified = 0
    for result, failure in parallel_results((('verify', r, cfg, output) for r in all_rows), workers, stopped):
        if failure:
            failures.append(failure)
            print(f'FAILED merge: {failure}', flush=True)
        else:
            verified += 1
        if time.monotonic() - last_print >= 5:
            print(f'Merge validated {verified}/{len(all_rows)}; failures={len(failures)}', flush=True)
            last_print = time.monotonic()
    if stopped() or verified != len(all_rows) or failures:
        raise RuntimeError(f'Merge incomplete: verified={verified}/{len(all_rows)}, failures={failures[:5]}')
    # Reuse bookkeeping is run-specific, not part of dataset identity.
    clean_rows = [{k: v for k, v in r.items() if k != 'reused'} for r in all_rows]
    atomic_json(output / 'manifest.json', {'contract': cache_contract(cfg), 'failures': [],
                                          'samples': sorted(clean_rows, key=lambda r: r['instance']),
                                          'split_note': 'Disjoint for VAE v2 updates; v1 may have seen these objects.'})
    print(f'Sealed {len(all_rows)} samples in {time.monotonic()-start:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/seg_vae_v2.json')
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world_size', type=int, default=1)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--instance')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--index_annotations',action='store_true',help='Index raw Arrow annotations before corrected preparation')
    ap.add_argument('--workers', type=int, default=default_workers(), help='CPU processes per rank (default up to 16)')
    ap.add_argument('--flush_every', type=int, default=128, help='Save accumulated manifest after this many results')
    ap.add_argument('--progress_seconds', type=float, default=5, help='Also save manifest/progress at this time interval')
    ap.add_argument('--merge_after', action='store_true', help='After a full single-rank preparation, validate and seal automatically')
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.index_annotations:
        if not cfg.get('annotation_policy') or args.merge or args.merge_after or args.instance or args.limit:
            ap.error('--index_annotations requires raw-annotation config and no other mode')
        from vae_v2.raw_data import index_annotations
        index_annotations(cfg)
        return
    if not 0 <= args.rank < args.world_size or (args.limit is not None and args.limit < 1):
        ap.error('Invalid rank/world_size or limit')
    if args.workers < 1 or args.flush_every < 1 or not np.isfinite(args.progress_seconds) or args.progress_seconds <= 0:
        ap.error('workers, flush_every and progress_seconds must be positive')
    if args.merge_after and (args.world_size != 1 or args.instance or args.limit or args.merge):
        ap.error('--merge_after requires full single-rank preparation')
    rows = source_rows(cfg)
    output = Path(cfg['prepared_dir'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'samples').mkdir(exist_ok=True)
    interrupted = []
    def request_stop(signum, frame):
        interrupted.append(signum)
        print('Stopping submission; finishing in-flight samples and saving progress...', flush=True)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    if args.merge:
        with output_lock(output, args.rank, merge=True):
            merge_prepared(rows, cfg, output, args.world_size, args.workers, lambda: bool(interrupted))
        return
    if args.instance:
        rows = [r for r in rows if r['instance'] == args.instance]
        if not rows:
            raise ValueError('Unknown instance')
    if args.limit:
        rows = rows[:args.limit]
    shard_name = f'diagnostic_rank{args.rank:04d}.json' if args.instance or args.limit else f'prep_rank{args.rank:04d}.json'
    cohort = rows[args.rank::args.world_size]
    report = {'contract': cache_contract(cfg), 'rank': args.rank, 'world_size': args.world_size,
              'samples': [], 'failures': [], 'workers': args.workers, 'status': 'running'}
    start = last_save = time.monotonic()
    since_save = reused = 0
    first_new = None
    print(f'Preparing {len(cohort)} objects with {args.workers} CPU workers; existing files are validated and reused', flush=True)
    def save_progress(status):
        elapsed = time.monotonic() - start
        count = len(report['samples']) + len(report['failures'])
        report.update(status=status, progress={'completed':count,'expected':len(cohort),'reused':reused,
                      'seconds':elapsed,'objects_per_second':count/max(elapsed,1e-9)})
        if first_new is not None:
            new_count = len(report['samples']) - reused
            report['progress']['new_objects_per_second'] = max(0, new_count-1) / max(time.monotonic()-first_new,1e-9)
        atomic_json(output / shard_name, report)
        print(f'{status}: {count}/{len(cohort)}; reused={reused}; failures={len(report["failures"])}; '
              f'{count/max(elapsed,1e-9):.2f} objects/s; elapsed={elapsed/60:.1f}min', flush=True)
    with output_lock(output, args.rank, diagnostic=bool(args.instance or args.limit)):
        try:
            tasks = (('prepare', r, cfg, output) for r in cohort)
            for result, failure in parallel_results(tasks, args.workers, lambda: bool(interrupted)):
                if failure:
                    report['failures'].append(failure)
                    print(f'FAILED: {failure}', flush=True)
                else:
                    report['samples'].append(result)
                    reused += int(result['reused'])
                    if not result['reused'] and first_new is None:
                        first_new = time.monotonic()
                since_save += 1
                if since_save >= args.flush_every or time.monotonic() - last_save >= args.progress_seconds:
                    save_progress('running')
                    last_save = time.monotonic(); since_save = 0
        finally:
            done = len(report['samples']) == len(cohort) and not report['failures'] and not interrupted
            save_progress('complete' if done else 'interrupted' if interrupted else 'incomplete')
    if interrupted:
        raise SystemExit(128 + interrupted[0])
    if report['status'] != 'complete':
        raise RuntimeError('Preparation incomplete; inspect rank report')
    if args.merge_after:
        with output_lock(output, args.rank, merge=True):
            merge_prepared(rows, cfg, output, args.world_size, args.workers, lambda: bool(interrupted))


if __name__ == '__main__':
    main()
