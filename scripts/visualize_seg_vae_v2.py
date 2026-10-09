"""Reconstruct a fixed test gallery and separate checkpoint-comparison diagnostics.

This is GT -> encoder posterior mean -> decoder/surface-head reconstruction,
not DiT generation. It never changes the validation checkpoint selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vae_v2.config import load_config, digest, atomic_json, storage_path
from vae_v2.data import load_manifest

CHAIR = '007d2949e5b0450c995c54809b43af0f'
SMALL_PART_CASE = '1b626ff2bc4d3f08834bcd739bba2251'


def prepare(args):
    cfg = load_config(args.config)
    if cfg.get('num_classes',20)!=20 or cfg.get('annotation_policy'):
        raise ValueError('This historical gallery compares legacy 20-class checkpoints. Use evaluate_seg_vae_v2.py --export for corrected parts64 checkpoints.')
    data = load_manifest(cfg)
    rows = {r['instance']: r for r in data['samples']}
    selection = json.loads(args.selection.read_text())
    if selection['split'] != 'validation' or selection['test_used_for_selection'] or selection.get('smoke_only'):
        raise ValueError('Expected a real checkpoint selected using validation only')
    if selection['manifest_digest'] != digest(data):
        raise ValueError('Checkpoint selection/data manifest mismatch')
    if not selection['ids'] or any(rows[u]['split'] != 'validation' for u in selection['ids']):
        raise ValueError('Selection contains non-validation objects')
    prior = json.loads(args.cohort.read_text())
    ids = [r['instance'] for r in prior['samples']]
    if len(ids) != 50 or len(set(ids)) != 50 or any(rows[u]['split'] != 'test' for u in ids):
        raise ValueError('Expected the previous fixed 50-object test gallery')
    selected = Path(selection['checkpoint']).resolve()
    baseline = selected.parent / 'regular_step002500.pt'
    checkpoints = {}
    for name, path in [('selected', selected), ('step2500', baseline)]:
        stat = path.stat()
        checkpoints[name] = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    all_ids = list(dict.fromkeys(ids + [CHAIR, SMALL_PART_CASE]))
    prediction_root = storage_path(Path(cfg['storage_root']) / 'predictions' / args.output_dir.name, cfg)
    plan = {
        'task': 'VAE reconstruction from GT soft labels; posterior mean, no DiT, no postprocessing',
        'config': cfg, 'data_manifest_digest': digest(data),
        'selection_source': str(args.selection.resolve()), 'selection': selection,
        'cohort_source': str(args.cohort.resolve()), 'test_ids': ids,
        'diagnostic_ids': [CHAIR, SMALL_PART_CASE],
        'diagnostic_note': 'Step comparisons are exploratory; excluded from fixed test summary and do not select checkpoints',
        'samples': [rows[u] for u in all_ids], 'checkpoints': checkpoints,
        'prediction_root': str(prediction_root), 'gpus': args.gpus,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / 'manifest.json'
    if target.exists():
        if json.loads(target.read_text()) != plan:
            raise ValueError('Output manifest differs; choose a new directory')
    elif any(args.output_dir.iterdir()):
        raise ValueError('Output directory is not empty')
    else:
        atomic_json(target, plan)
    return plan


def worker(args):
    import numpy as np
    import torch
    from vae_v2.data import load_sample, validate_sample
    from vae_v2.evaluation import reconstruct
    from vae_v2.model import SegVaeV2

    plan = json.loads((args.output_dir / 'manifest.json').read_text())
    rank, world = args.worker_rank, len(plan['gpus'])
    if not 0 <= rank < world:
        raise ValueError('Invalid worker rank')
    cfg = plan['config']
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    prediction_root = storage_path(plan['prediction_root'], cfg)
    shard = args.output_dir / f'metrics_rank{rank:04d}.json'
    report = json.loads(shard.read_text()) if shard.exists() else {'rank': rank, 'plan_digest': digest(plan), 'samples': []}
    if report['plan_digest'] != digest(plan):
        raise ValueError('Worker resume manifest mismatch')
    done = {(r['checkpoint'], r['instance']): r for r in report['samples']}
    for name in ('selected', 'step2500'):
        cohort = plan['samples'] if name == 'selected' else [r for r in plan['samples'] if r['instance'] in plan['diagnostic_ids']]
        assigned = cohort[rank::world]
        pending = [r for r in assigned if (name, r['instance']) not in done or
                   not (prediction_root / name / f'{r["instance"]}.npz').is_file()]
        if not pending:
            continue
        descriptor = plan['checkpoints'][name]
        path = Path(descriptor['path'])
        stat = path.stat()
        if stat.st_size != descriptor['bytes'] or stat.st_mtime_ns != descriptor['mtime_ns']:
            raise ValueError('Checkpoint changed since gallery preparation')
        print(f'[rank {rank}] Loading {name}: {path}', flush=True)
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('smoke_only') or state['manifest_digest'] != plan['data_manifest_digest']:
            raise ValueError('Checkpoint is a smoke run or uses another dataset')
        model_cfg = state['config']
        model = SegVaeV2(model_cfg, initialize=False).to('cuda:0')
        model.load_state_dict(state['model'])
        del state
        target = prediction_root / name
        target.mkdir(parents=True, exist_ok=True)
        for row in pending:
            uid = row['instance']
            atomic_json(args.output_dir / f'progress_rank{rank:04d}.json',
                        {'instance': uid, 'checkpoint': name, 'phase': 'reconstruction', 'updated': time.time()})
            start = time.monotonic()
            sample = load_sample(Path(cfg['prepared_dir']) / row['path'])
            validate_sample(sample)
            metric, faces, voxel = reconstruct(model, sample, model_cfg, 'cuda:0')
            path = target / f'{uid}.npz'
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, face_labels=faces, voxel_labels=voxel,
                                    checkpoint=np.array(descriptor['path']), instance=np.array(uid),
                                    posterior_mean=np.array(True))
            temporary.replace(path)
            record = dict(instance=uid, split=row['split'], checkpoint=name,
                          prediction=str(path), seconds=time.monotonic()-start, **metric)
            done[name, uid] = record
            report['samples'] = list(done.values())
            atomic_json(shard, report)
            print(f'[rank {rank}] {name} {uid}: surface mIoU={metric["surface"]["miou"]:.5f} '
                  f'projection mIoU={metric["projection"]["miou"]:.5f}', flush=True)
        del model
        torch.cuda.empty_cache()
    atomic_json(args.output_dir / f'progress_rank{rank:04d}.json', {'phase': 'complete', 'updated': time.time()})


def launch(args, plan):
    logs = args.output_dir / 'logs'
    logs.mkdir(exist_ok=True)
    def run(rank):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['gpus'][rank]),
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--output_dir', str(args.output_dir),
                   '--worker_rank', str(rank)]
        with (logs / f'worker{rank:04d}.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f'Worker {rank} exited {result.returncode}', flush=True)
        return result.returncode
    with ThreadPoolExecutor(max_workers=len(plan['gpus'])) as pool:
        statuses = list(pool.map(run, range(len(plan['gpus']))))
    if any(statuses):
        raise RuntimeError(f'Workers failed: {statuses}; see {logs}')


def font(size):
    from PIL import ImageFont
    path = Path('/usr/share/fonts/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def panel_grid(image_sets, column_titles, row_titles, path, subtitle, cell=512):
    from PIL import Image, ImageDraw
    gap, margin, header, label_height = 18, 24, 120, 32
    width = margin*2 + len(column_titles)*cell + (len(column_titles)-1)*gap
    height = header + len(row_titles)*(cell+label_height+gap) + margin
    canvas = Image.new('RGB', (width, height), '#edf1f5')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 15), 'VAE reconstruction comparison', font=font(26), fill='#17212b')
    draw.text((margin, 53), subtitle, font=font(16), fill='#465568')
    for col, name in enumerate(column_titles):
        draw.text((margin+col*(cell+gap), 86), name, font=font(21), fill='#17212b')
    for row, title in enumerate(row_titles):
        y = header + row*(cell+label_height+gap)
        for col in range(len(column_titles)):
            x = margin + col*(cell+gap)
            draw.text((x+6, y+4), title, font=font(16), fill='#465568')
            panel = image_sets[col][row]
            scale = min(cell/panel.width, cell/panel.height)
            panel = panel.resize((round(panel.width*scale), round(panel.height*scale)), Image.Resampling.NEAREST)
            draw.rectangle((x, y+label_height, x+cell-1, y+label_height+cell-1), fill='white')
            canvas.paste(panel, (x+(cell-panel.width)//2, y+label_height+(cell-panel.height)//2))
    canvas.save(path)


def export(args, plan):
    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw
    from vae_v2.data import load_sample
    from vae_v2.evaluation import summarize
    from scripts.evaluate_seg_quality import generate_class_colors
    from scripts.refine_seg_projection import export_labels
    from utils.seg_projection import render_labels, VIEW_NAMES

    records = {}
    for rank in range(len(plan['gpus'])):
        shard = json.loads((args.output_dir / f'metrics_rank{rank:04d}.json').read_text())
        if shard['plan_digest'] != digest(plan):
            raise ValueError('Mismatched inference shard')
        for row in shard['samples']:
            key = row['checkpoint'], row['instance']
            if key in records:
                raise ValueError('Duplicate prediction row')
            records[key] = row
    expected = {('selected', r['instance']) for r in plan['samples']} | {('step2500', u) for u in plan['diagnostic_ids']}
    if set(records) != expected:
        raise ValueError('Incomplete prediction set')
    palette = np.full((21, 3), 255, np.uint8)
    palette[1:] = (generate_class_colors(20)*255).astype(np.uint8)
    glb = args.output_dir / 'glb'
    renders = args.output_dir / 'renders'
    diagnostic = args.output_dir / 'diagnostics'
    for path in (glb, renders, diagnostic):
        path.mkdir(exist_ok=True)
    artifacts = []
    overview = Image.new('RGB', (1600, 250*((len(plan['test_ids'])+4)//5)), 'white')
    overview_draw = ImageDraw.Draw(overview)
    case_details = {}
    for row in plan['samples']:
        uid = row['instance']
        sample = load_sample(Path(plan['config']['prepared_dir']) / row['path'])
        mesh = trimesh.Trimesh(sample['mesh_vertices'], sample['mesh_faces'], process=False)
        with np.load(row['source']) as source:
            raster = source['face_raster']
            np.testing.assert_array_equal(sample['mesh_faces'], source['mesh_faces'])
            np.testing.assert_array_equal(sample['mesh_vertices'], source['mesh_vertices'].astype(np.float32))
        with np.load(records['selected', uid]['prediction']) as data:
            faces = data['face_labels']
            if str(data['checkpoint']) != plan['checkpoints']['selected']['path']:
                raise ValueError('Prediction checkpoint mismatch')
        delivered = export_labels(mesh, faces, glb / f'{uid}.glb')
        export_labels(mesh, sample['face_labels'], glb / f'{uid}_gt.glb')
        predicted_images = [Image.fromarray(palette[m]) for m in render_labels(delivered+1, raster)]
        gt_images = [Image.fromarray(palette[m]) for m in render_labels(sample['face_labels']+1, raster)]
        individual = renders / uid
        individual.mkdir(exist_ok=True)
        grid = Image.new('RGB', (1536, 1072), 'white')
        draw = ImageDraw.Draw(grid)
        comparison = Image.new('RGB', (1536, 560), 'white')
        comp_draw = ImageDraw.Draw(comparison)
        for v, name in enumerate(VIEW_NAMES):
            predicted_images[v].save(individual / f'{name}.png')
            x, y = (v%3)*512, (v//3)*536
            draw.text((x+6, y+3), name, font=font(15), fill='black')
            grid.paste(predicted_images[v], (x, y+24))
            for ri, (title, images) in enumerate([('GT', gt_images), ('VAE v2 step 5000', predicted_images)]):
                comp_draw.text((v*256+4, ri*280+4), f'{title}: {name}', fill='black')
                comparison.paste(images[v].resize((256,256), Image.Resampling.NEAREST), (v*256, ri*280+24))
        view_path = renders / f'{uid}_views.png'
        comp_path = renders / f'{uid}_comparison.png'
        grid.save(view_path)
        comparison.save(comp_path)
        if uid in plan['test_ids']:
            index = plan['test_ids'].index(uid)
            x, y = (index%5)*320, (index//5)*250
            overview_draw.text((x+5,y+5), f'{index+1:02d} {uid[:12]}', fill='black')
            overview.paste(grid.resize((320,223),Image.Resampling.LANCZOS), (x,y+25))
        artifact = {'instance':uid, 'split':row['split'], 'glb':str((glb/f'{uid}.glb').relative_to(args.output_dir)),
                    'gt_glb':str((glb/f'{uid}_gt.glb').relative_to(args.output_dir)),
                    'views':str(view_path.relative_to(args.output_dir)),
                    'comparison':str(comp_path.relative_to(args.output_dir)),
                    'individual_views':[str((individual/f'{v}.png').relative_to(args.output_dir)) for v in VIEW_NAMES],
                    'prediction':records['selected',uid]['prediction'], 'glb_roundtrip_verified':True}
        artifacts.append(artifact)
        if uid in plan['diagnostic_ids']:
            folder = diagnostic / uid
            folder.mkdir(exist_ok=True)
            with np.load(records['step2500',uid]['prediction']) as data:
                early = data['face_labels']
            export_labels(mesh, early, folder/'step2500.glb')
            labels = [sample['face_labels'], early, faces]
            titles = ['GT', 'VAE v2 step 2500', 'VAE v2 step 5000']
            all_images = [[Image.fromarray(palette[m]) for m in render_labels(l+1,raster)] for l in labels]
            panel_grid(all_images,titles,VIEW_NAMES,folder/'comparison.png',uid)
            error_images = []
            for lab in labels:
                views = []
                for rr in raster:
                    hit = rr >= 0
                    a = np.full((*rr.shape,3),255,np.uint8)
                    a[hit] = [210,218,225]
                    wrong = hit & (sample['face_labels'][rr.clip(min=0)] > 0) & (lab[rr.clip(min=0)] != sample['face_labels'][rr.clip(min=0)])
                    a[wrong] = [229,55,74]
                    views.append(Image.fromarray(a))
                error_images.append(views)
            panel_grid(error_images,titles,VIEW_NAMES,folder/'errors.png','Red = incorrect label; gray = correct')
            case = {'step2500':records['step2500',uid], 'step5000':records['selected',uid]}
            if uid == CHAIR:
                with np.load(ROOT/'results/chair_007d_gt_vae_dit/vae_reconstruction.npz') as old:
                    old_faces = old['face_labels']
                export_labels(mesh, old_faces, folder/'vae_v1.glb')
                labels = [sample['face_labels'], old_faces, early, faces]
                titles = ['GT', 'VAE v1', 'VAE v2 step 2500', 'VAE v2 step 5000']
                views = [[Image.fromarray(palette[m]) for m in render_labels(l+1,raster)] for l in labels]
                crops = [[] for _ in labels]
                errors = {}
                for vi in [4,5,0]:
                    rr = raster[vi];hit = rr>=0
                    seat = hit & (sample['face_labels'][rr.clip(min=0)]==14)
                    yy,xx=np.where(seat)
                    box=(max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(512,int(xx.max())+13),min(512,int(yy.max())+13))
                    for ci in range(len(labels)):
                        crops[ci].append(views[ci][vi].crop(box))
                    errors[VIEW_NAMES[vi]]={t:float(100*np.mean(l[rr[seat]]!=14)) for t,l in zip(titles,labels)}
                panel_grid(crops,titles,['Seat top','Seat bottom','Seat front'],folder/'seat_closeup.png',
                           'Same mesh/GT; v1 and v2 use their respective reconstruction pipelines',cell=400)
                case['seat_error_pixel_percent']=errors
                case['top_triangles']=[{'face':i,**{t:int(l[i]) for t,l in zip(titles,labels)}} for i in [2410,2251,2534]]
            else:
                area=sample['face_area'];gt=sample['face_labels']
                part_fraction={int(c):float(area[gt==c].sum()/area[gt>0].sum()) for c in np.unique(gt[gt>0])}
                small=min(part_fraction,key=part_fraction.get)
                visibility=[]
                for vi,rr in enumerate(raster):
                    mask=(rr>=0)&(gt[rr.clip(min=0)]==small)
                    visibility.append((int(mask.sum()),vi,mask))
                chosen=sorted(visibility,key=lambda x:-x[0])[:3]
                crops=[[],[],[]];crop_titles=[]
                for pixels,vi,mask in chosen:
                    if pixels==0:continue
                    yy,xx=np.where(mask)
                    x0,x1=int(xx.min()),int(xx.max());y0,y1=int(yy.min()),int(yy.max())
                    side=max(48,x1-x0+17,y1-y0+17)
                    cx=(x0+x1)//2;cy=(y0+y1)//2
                    left=max(0,min(512-side,cx-side//2));top=max(0,min(512-side,cy-side//2))
                    box=(left,top,min(512,left+side),min(512,top+side))
                    for ci in range(3):
                        marked=all_images[ci][vi].copy()
                        ImageDraw.Draw(marked).rectangle((max(0,x0-3),max(0,y0-3),min(511,x1+3),min(511,y1+3)),outline='black',width=1)
                        crops[ci].append(marked.crop(box))
                    crop_titles.append(f'{VIEW_NAMES[vi]}: GT part {small}, {pixels} pixels')
                if crop_titles:
                    panel_grid(crops,titles,crop_titles,folder/'small_part_closeup.png','Black box marks the GT small-part location; nearest-neighbor zoom')
                case['small_part_id']=small;case['gt_area_fraction']=part_fraction
                case['small_part_area_recall']={title:float(area[(gt==small)&(lab==small)].sum()/area[gt==small].sum()) for title,lab in zip(titles,labels)}
            atomic_json(folder/'report.json',case)
            case_details[uid]=case
        print(f'Exported {len(artifacts)}/{len(plan["samples"])}: {uid}',flush=True)
    overview.save(args.output_dir/'overview.png')
    fixed=[records['selected',u] for u in plan['test_ids']]
    report={'task':plan['task'],'checkpoint':plan['checkpoints']['selected']['path'],
            'test_objects':len(fixed),'test_summary':summarize(fixed),'test_samples':fixed,
            'diagnostics':case_details,'artifacts':artifacts,'completed_at':datetime.now(timezone.utc).isoformat(),
            'prediction_root':plan['prediction_root'],'rendering':'Cached exact mesh visibility, flat part colors, six 512px orthographic views'}
    atomic_json(args.output_dir/'report.json',report)
    cards=[]
    for a in artifacts:
        uid=a['instance'];metric=records['selected',uid]
        score=metric['surface']['miou']
        cards.append(f'<article><h2>{uid}</h2><p>{a["split"]} · Surface mIoU {score*100:.3f}% · '
                     f'<a href="{a["glb"]}">VAE GLB</a> · <a href="{a["gt_glb"]}">GT GLB</a> · '
                     f'<a href="{a["comparison"]}">GT comparison</a></p>'
                     f'<a href="{a["views"]}"><img loading="lazy" src="{a["views"]}" alt="Six views"></a></article>')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page+='<title>VAE v2 step 5000 reconstruction</title><style>body{font:16px system-ui;background:#f1f4f7;color:#182332;max-width:1450px;margin:28px auto;padding:0 20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:22px}article{background:white;padding:16px;border-radius:10px}h2{font-size:15px;overflow-wrap:anywhere}img{width:100%}a{color:#145ec0}</style>'
    page+='<h1>VAE v2 · step 5000 重建可视化</h1><p>GT 标签 → Encoder 均值 → Decoder／表面头；不是 DiT 生成。固定 50 个测试对象，另附 1 个验证诊断对象。</p>'
    page+=f'<p><a href="overview.png">50 对象总览</a> · <a href="diagnostics/{CHAIR}/seat_closeup.png">椅子：GT／旧 VAE／2500／5000</a> · <a href="diagnostics/{SMALL_PART_CASE}/small_part_closeup.png">小部件退化放大</a> · <a href="report.json">指标</a></p>'
    page+='<main>'+''.join(cards)+'</main></html>'
    (args.output_dir/'index.html').write_text(page+'\n')
    summary=report['test_summary']
    readme=['# VAE v2 step 5000 重建推理与可视化','',
            '使用验证集已选定的普通权重，GT 软体素标签编码后取 posterior mean，再解码为网格面标签。无 DiT、无后处理。','',
            f'权重：`{plan["checkpoints"]["selected"]["path"]}`。',
            '与此前 DiT gallery 使用相同 50 个测试对象；额外验证案例不计入测试均值。2500 步比较仅用于探索性诊断，不重新选择 checkpoint。','',
            '- [浏览页面](index.html)','- [50 对象总览](overview.png)',
            f'- [椅子四列座面放大](diagnostics/{CHAIR}/seat_closeup.png)',
            f'- [小部件三列放大](diagnostics/{SMALL_PART_CASE}/small_part_closeup.png)',
            '- [完整指标与产物清单](report.json)','',
            '| 指标 | 固定 50 对象 |','| --- | ---: |']
    for g,k in [('surface','miou'),('projection','miou'),('voxel','miou'),('surface','small_part_recall')]:
        readme.append(f'| {g}.{k} | {summary[g][k]*100:.3f}% |')
    readme+=['',f'预测标签缓存：`{plan["prediction_root"]}`。',
             'GLB 全部通过逐面标签与几何回读校验。每个对象保存 6 张 512×512 渲染图、六视角拼图与 GT 对比图。']
    (args.output_dir/'README.md').write_text('\n'.join(readme)+'\n')
    print('Gallery ready:',args.output_dir/'index.html',flush=True)
    print(json.dumps(report['test_summary'],indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/seg_vae_v2.json')
    ap.add_argument('--selection',type=Path,default=Path('/path/to/trellisseg_run/checkpoints/seg_vae_v2/selection.json'))
    ap.add_argument('--cohort',type=Path,default=ROOT/'results/dit_decoded_full_v1_ema1095_test50/manifest.json')
    ap.add_argument('--output_dir',type=Path,required=True)
    ap.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    ap.add_argument('--worker_rank',type=int)
    ap.add_argument('--prepare_only',action='store_true')
    ap.add_argument('--export_only',action='store_true')
    args=ap.parse_args();args.output_dir=args.output_dir.resolve()
    if args.worker_rank is not None:
        worker(args);return
    if min(args.gpus)<0 or len(set(args.gpus))!=len(args.gpus):
        ap.error('GPU indices must be unique and nonnegative')
    plan=prepare(args)
    if args.prepare_only:
        print(json.dumps({'checkpoint':plan['checkpoints']['selected']['path'],'test_objects':len(plan['test_ids']),
                          'diagnostic_ids':plan['diagnostic_ids'],'prediction_root':plan['prediction_root']},indent=2));return
    if not args.export_only:
        launch(args,plan)
    export(args,plan)


if __name__=='__main__':main()
