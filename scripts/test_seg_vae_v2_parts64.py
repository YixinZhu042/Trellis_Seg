"""Frozen validation-selected VAE reconstruction test, GLBs, and diagnostic gallery.

Prepare once, execute independent GPU shards through Remote API, then merge.
Historical problem cases are excluded from the held-out test summary.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vae_v2.config import load_config, digest, atomic_json, storage_path

CASES = ['007d2949e5b0450c995c54809b43af0f',
         '11c88bc30d37479281c30cc49d25084a',
         '141329918e4945ed82d0f48ba5903b8f']


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def select_rows(manifest, selection):
    if (selection['split'] != 'validation' or selection['test_used_for_selection']
            or selection.get('smoke_only') or selection['manifest_digest'] != digest(manifest)):
        raise ValueError('Invalid validation-only checkpoint selection')
    rows = {r['instance']: r for r in manifest['samples']}
    val_ids = {r['instance'] for r in rows.values() if r['split'] == 'validation'}
    if len(rows) != len(manifest['samples']) or set(selection['ids']) != val_ids or len(selection['ids']) != 128:
        raise ValueError('Selection must use exactly the 128 validation objects')
    tests = sorted((r for r in rows.values() if r['split'] == 'test'), key=lambda r:r['instance'])
    if len(tests) != 128:
        raise ValueError('Expected all 128 held-out test objects')
    samples = [dict(r, role='test') for r in tests]
    samples += [dict(rows[u], role='diagnostic') for u in CASES if rows[u]['split'] != 'test']
    return samples


def prepare(args):
    from vae_v2.data import load_manifest
    from vae_v2.raw_data import check_raw_source
    cfg = load_config(args.config)
    if cfg.get('num_classes') != 64 or not cfg.get('annotation_policy'):
        raise ValueError('Expected corrected 64-class configuration')
    print('Verifying full data manifest once before sharding', flush=True)
    manifest = load_manifest(cfg, verify_files=False)
    def verify(row):
        for path, size, mtime in (
                (Path(cfg['prepared_dir'])/row['path'], row['bytes'], row['mtime_ns']),
                (Path(row['source']), row['source_bytes'], row['source_mtime_ns'])):
            stat = path.stat()
            if (stat.st_size,stat.st_mtime_ns) != (size,mtime):
                raise ValueError(f'File changed since audit: {path}')
        check_raw_source(row)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for index, _ in enumerate(pool.map(verify,manifest['samples']),1):
            if index%5000==0:
                print(f'Verified {index}/{len(manifest["samples"])} objects',flush=True)
    selection = json.loads(args.selection.read_text())
    rows = select_rows(manifest, selection)
    ckpt = Path(selection['checkpoint']).resolve()
    pred = storage_path(Path(cfg['storage_root'])/'predictions'/args.output_dir.name, cfg)
    plan = {'task': 'GT soft labels -> posterior mean -> VAE decoder/surface head; no DiT, no postprocessing',
            'created_at': datetime.now().astimezone().isoformat(), 'config': cfg,
            'manifest_digest': digest(manifest),
            'manifest_identity': identity(Path(cfg['prepared_dir'])/'manifest.json'),
            'checkpoint': identity(ckpt), 'selection': selection,
            'selection_source': str(args.selection.resolve()), 'samples': rows,
            'prediction_dir': str(pred), 'workers': args.workers,
            'diagnostic_ids': CASES, 'test_used_for_selection': False}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a fresh output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ('objects', 'metrics', 'logs'):
        (args.output_dir/folder).mkdir()
    atomic_json(args.output_dir/'plan.json', plan)
    print(f'Prepared {len(rows)} objects, {args.workers} workers; checkpoint={ckpt.name}', flush=True)


def extra_visuals(out, uid, sample, pred, classes):
    import numpy as np
    from PIL import Image, ImageDraw
    from vae_v2.visuals import render_palette
    from utils.seg_projection import render_labels, VIEW_NAMES
    palette = render_palette(classes)
    raster = sample['face_raster']
    maps = render_labels(pred+1, raster)
    error = Image.new('RGB', (1536, 280), 'white')
    draw = ImageDraw.Draw(error)
    for i, name in enumerate(VIEW_NAMES):
        Image.fromarray(palette[maps[i]]).save(out/f'{name}.png')
        rr = raster[i]
        hit = rr >= 0
        gt = sample['face_labels'][rr.clip(min=0)]
        bad = hit & (gt > 0) & (pred[rr.clip(min=0)] != gt)
        rgb = np.full((*rr.shape, 3), 255, np.uint8)
        rgb[hit] = [205, 213, 221]
        rgb[bad] = [229, 55, 74]
        draw.text((i*256+4, 3), f'{name}: red = wrong', fill='black')
        error.paste(Image.fromarray(rgb).resize((256,256), Image.Resampling.NEAREST), (i*256,24))
    error.save(out/f'{uid}_errors.png')


def worker(args):
    import numpy as np
    import torch
    from vae_v2.data import load_sample, validate_sample
    from vae_v2.evaluation import reconstruct, export_prediction
    from vae_v2.model import SegVaeV2
    plan = json.loads((args.output_dir/'plan.json').read_text())
    rank = args.worker_rank
    if not 0 <= rank < plan['workers']:
        raise ValueError('Invalid worker index')
    for key in ('checkpoint', 'manifest_identity'):
        if identity(plan[key]['path']) != plan[key]:
            raise ValueError(f'{key} changed since preparation')
    cfg = plan['config']
    device = 'cuda:0'
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    print(f'Worker {rank}: loading {plan["checkpoint"]["path"]}', flush=True)
    state = torch.load(plan['checkpoint']['path'], map_location='cpu', weights_only=False)
    if state.get('smoke_only') or state['manifest_digest'] != plan['manifest_digest']:
        raise ValueError('Wrong checkpoint provenance')
    checkpoint_cfg = state['config']
    if checkpoint_cfg['num_classes'] != cfg['num_classes'] or checkpoint_cfg['annotation_policy'] != cfg['annotation_policy']:
        raise ValueError('Wrong checkpoint annotation semantics')
    model = SegVaeV2(checkpoint_cfg, initialize=False)
    model.load_state_dict(state['model'])
    del state
    model = model.to(device).eval()
    torch.cuda.synchronize()
    print(f'Worker {rank}: model ready', flush=True)
    assigned = plan['samples'][rank::plan['workers']]
    for index, row in enumerate(assigned):
        uid = row['instance']
        start = time.monotonic()
        path = Path(cfg['prepared_dir'])/row['path']
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row['bytes'], row['mtime_ns']):
            raise ValueError(f'Changed sample {uid}')
        atomic_json(args.output_dir/f'progress{rank}.json', {'phase':'inference', 'instance':uid, 'done':index, 'total':len(assigned)})
        sample = load_sample(path)
        validate_sample(sample, cfg['num_classes'])
        metric, faces, voxels = reconstruct(model, sample, checkpoint_cfg, device)
        out = args.output_dir/'objects'/uid
        export_prediction(out, uid, sample, faces, voxels, row['source'], plan['prediction_dir'])
        extra_visuals(out, uid, sample, faces, cfg['num_classes'])
        labels, area = sample['face_labels'], sample['face_area'].astype(np.float64)
        total = area[labels>0].sum()
        parts = []
        for label in np.unique(labels[labels>0]):
            gt, guess = labels==label, faces==label
            intersection = area[gt & guess].sum()
            parts.append({'label':int(label), 'area_fraction':float(area[gt].sum()/total),
                          'recall':float(intersection/max(area[gt].sum(),1e-12)),
                          'iou':float(intersection/max(area[gt | guess].sum(),1e-12))})
        record = dict(instance=uid, split=row['split'], role=row['role'], type_id=row['type_id'],
                      plan_digest=digest(plan), parts=parts, seconds=time.monotonic()-start,
                      glb_roundtrip_verified=True, **metric)
        atomic_json(args.output_dir/'metrics'/f'{uid}.json', record)
        print(f'Worker {rank} {index+1}/{len(assigned)} {uid}: surface={metric["surface"]["miou"]:.5f} projection={metric["projection"]["miou"]:.5f}', flush=True)
    atomic_json(args.output_dir/f'progress{rank}.json', {'phase':'complete', 'done':len(assigned), 'total':len(assigned)})


def merge(args):
    from PIL import Image, ImageDraw
    from vae_v2.evaluation import summarize
    from utils.seg_projection import VIEW_NAMES
    plan = json.loads((args.output_dir/'plan.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((args.output_dir/'metrics').glob('*.json'))]
    expected = {r['instance']: r for r in plan['samples']}
    if len(records) != len(expected) or {r['instance'] for r in records} != expected.keys():
        raise ValueError('Incomplete or duplicate results')
    for r in records:
        if r['plan_digest'] != digest(plan) or r['role'] != expected[r['instance']]['role']:
            raise ValueError('Mixed inference plans')
        uid = r['instance']
        folder = args.output_dir/'objects'/uid
        files = [f'{uid}.glb',f'{uid}_gt.glb',f'{uid}_comparison.png',f'{uid}_errors.png',f'{uid}_palette.json']
        files += [f'{v}.png' for v in VIEW_NAMES]
        if not all((folder/f).is_file() and (folder/f).stat().st_size for f in files):
            raise ValueError(f'Missing artifacts: {uid}')
        if not (Path(plan['prediction_dir'])/f'{uid}.npz').is_file():
            raise ValueError(f'Missing prediction cache: {uid}')
    tests = [r for r in records if r['role']=='test']
    diagnostics = [r for r in records if r['role']=='diagnostic']
    report = {'checkpoint': plan['checkpoint']['path'], 'task': plan['task'], 'split':'test',
              'ids':[r['instance'] for r in tests], 'manifest_digest':plan['manifest_digest'],
              'summary':summarize(tests), 'samples':tests, 'test_used_for_selection':False,
              'diagnostics':diagnostics, 'prediction_dir':plan['prediction_dir'],
              'completed_at':datetime.now().astimezone().isoformat()}
    atomic_json(args.output_dir/'metrics.json', report)
    pages = []
    for number, start in enumerate(range(0,len(tests),16),1):
        subset = tests[start:start+16]
        canvas = Image.new('RGB',(1536,310*((len(subset)+1)//2)),'white')
        draw = ImageDraw.Draw(canvas)
        for i, r in enumerate(subset):
            uid = r['instance']
            x,y=(i%2)*768,(i//2)*310
            draw.text((x+8,y+3),f'{start+i+1:03d} {uid}  mIoU={r["surface"]["miou"]:.4f}  GT top / VAE bottom',fill='black')
            with Image.open(args.output_dir/'objects'/uid/f'{uid}_comparison.png') as im:
                canvas.paste(im.resize((768,280)),(x,y+24))
        name = f'overview_{number:02d}.jpg'
        canvas.save(args.output_dir/name,quality=90)
        pages.append(name)
    worst = sorted(tests,key=lambda r:r['surface']['miou'])[:10]
    atomic_json(args.output_dir/'worst10.json', worst)
    def card(r):
        uid=r['instance'];base=f'objects/{uid}/{uid}'
        return (f'<section><h3>{uid} — {r["split"]} / {r["role"]}</h3>'
                f'<p>Surface mIoU {r["surface"]["miou"]:.4f}; accuracy {r["surface"]["accuracy"]:.4f}; '
                f'<a href="{base}.glb">VAE GLB</a> · <a href="{base}_gt.glb">GT GLB</a> · '
                f'<a href="{base}_errors.png">Error views</a></p>'
                f'<a href="{base}_comparison.png"><img loading="lazy" src="{base}_comparison.png"></a></section>')
    markup = ('<!doctype html><meta charset="utf-8"><title>VAE parts64 reconstruction test</title>'
              '<style>body{max-width:1540px;margin:auto;font:16px sans-serif;padding:24px}img{width:100%}section{margin:30px 0}pre{background:#eee;padding:15px}</style>'
              '<h1>VAE v2 parts64: 128 held-out reconstruction tests</h1>'
              f'<p>{html.escape(plan["task"])}</p><p>Checkpoint: {html.escape(plan["checkpoint"]["path"])}</p>'
              '<p>Same label colors and cameras for GT (top) and reconstruction (bottom). No smoothing. '
              'Historical diagnostics are training objects, not held-out tests.</p>'
              f'<pre>{html.escape(json.dumps(report["summary"],indent=2))}</pre>'
              '<h2>Historical diagnostic objects</h2>'+''.join(card(r) for r in diagnostics)+
              '<h2>10 lowest surface-mIoU test objects (post-evaluation diagnostic)</h2>'+''.join(card(r) for r in worst)+
              '<h2>All 128 tests in fixed ID order</h2>'+''.join(card(r) for r in tests))
    (args.output_dir/'index.html').write_text(markup)
    print(json.dumps(report['summary'],indent=2))
    print(f'Export verified: {len(records)} objects, {len(records)*2} GLBs, {len(pages)} overview pages')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/seg_vae_v2_parts64.json')
    ap.add_argument('--selection',type=Path,default=Path('/path/to/trellisseg_run/checkpoints/seg_vae_v2_parts64/selection.json'))
    ap.add_argument('--output-dir',type=Path,required=True)
    modes=ap.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--worker-rank',type=int)
    modes.add_argument('--merge',action='store_true')
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    args.output_dir=args.output_dir.resolve()
    if args.workers<1:ap.error('workers must be positive')
    if args.prepare:prepare(args)
    elif args.merge:merge(args)
    else:worker(args)


if __name__=='__main__':main()
