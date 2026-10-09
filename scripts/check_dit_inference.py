"""Validation-only inference sweep with paired noise and frozen model identities.

prepare -> Remote API launch -> merge -> export. No training/checkpoint writes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import html
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import digest,atomic_json,storage_path
from utils.dit_vae_v2 import config,load_cache,identity,file_sha256
from utils.dit_inference_check import experiments,generate,metric_difference


def prepare(args):
    cfg,vcfg=config(args.config);source,cache=load_cache(cfg)
    selected=json.loads((Path(cfg['output_dir'])/'selection.json').read_text())
    rows=sorted([r for r in cache['samples'] if r['split']=='validation'],key=lambda r:r['instance'])
    if (len(rows)!=128 or len(selected['ids'])!=128 or set(selected['ids'])!={r['instance'] for r in rows}
            or selected['split']!='validation' or selected['test_used_for_selection']):
        raise ValueError('Expected validation-only selected checkpoint / cohort')
    reference_path=Path(cfg['output_dir'])/'validation'/f'{selected["kind"]}_{selected["step"]:06d}.json'
    reference=json.loads(reference_path.read_text())
    for k in ('step','kind','checkpoint','summary','ids'):
        if reference[k]!=selected[k]:raise ValueError('Selection/reference mismatch')
    if selected['sampling_steps']!=20:raise ValueError('Baseline reference must use 20 steps')
    base=json.loads(Path(cfg['base_config']).read_text());trained=json.loads((Path(cfg['output_dir'])/'manifest.json').read_text())
    if trained['smoke_only'] or trained['cache_digest']!=digest(cache) or trained['base_digest']!=digest(base) or trained['vae_sha256']!=source['vae_sha256']:
        raise ValueError('Training/cache/architecture identity mismatch')
    plan={'created_at':datetime.now().astimezone().isoformat(),'config':cfg,'vae_config':vcfg,
          'checkpoint':identity(selected['checkpoint']),'vae_checkpoint':source['vae_checkpoint'],
          'vae_sha256':source['vae_sha256'],'cache_digest':digest(cache),'source_manifest_digest':source['source_manifest_digest'],
          'base_config':base,'base_digest':digest(base),'normalization':cache['normalization'],
          'selection':selected,'reference':reference['samples'],
          'samples':rows,'settings':experiments(),'workers':args.gpus*3,'gpus_per_node':args.gpus,
          'prediction_dir':str(storage_path(Path(vcfg['storage_root'])/'predictions'/args.output_dir.name,vcfg)),
          'split':'validation','test_used':False,'postprocessing':False,'label_matching':False,
          'seed_policy':'20260930 + int(instance[:8],16); identical x1 across settings',
          'cfg_formula':'shape_only + scale*(conditioned-shape_only); shape kept in both branches',
          'shuffle':'shuffle features within each fixed view, keep coords/offsets/counts/feature multiset',
          'selection_rule':'correct-condition settings only: surface mIoU; small-part recall; lower isolated-error area',
          'limitations':'one fixed seed per object; validation is used for settings selection, not an unbiased test estimate'}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Use new result directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for folder in ('metrics','logs'):(args.output_dir/folder).mkdir()
    atomic_json(args.output_dir/'plan.json',plan)
    print(f'Prepared 128 validation objects x {len(plan["settings"])} settings, {plan["workers"]} workers',flush=True)


def worker(args):
    import numpy as np
    import torch
    from models.seg_flow_model import SegSLatFlowModel
    from vae_v2.model import SegVaeV2
    from vae_v2.geometry import SurfaceQueries
    from scripts.train_dit_vae_v2 import load_geometry
    from scripts.evaluate_dit_vae_v2 import decode
    from utils.dit_vae_v2 import load_latents,unnormalize
    plan=json.loads((args.output_dir/'plan.json').read_text());contract=digest(plan)
    rank=args.worker
    if not 0<=rank<plan['workers']:raise ValueError('Invalid worker')
    torch.cuda.set_device(args.device_index);device=f'cuda:{args.device_index}'
    torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(0)
    with (args.output_dir/f'.worker{rank}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in ('checkpoint','vae_checkpoint'):
            if identity(plan[key]['path'])!=plan[key]:raise ValueError('Frozen weights changed')
        print(f'worker {rank}: loading frozen weights',flush=True)
        state=torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
        m=state['metadata']
        if (m['smoke_only'] or m['cache_digest']!=plan['cache_digest'] or m['vae_sha256']!=plan['vae_sha256']
                or m['base_digest']!=plan['base_digest'] or state['step']!=plan['selection']['step'] or state['kind']!=plan['selection']['kind']):
            raise ValueError('Checkpoint not compatible with frozen sweep')
        model=SegSLatFlowModel(**dict(plan['base_config']['models']['denoiser']['args'],use_checkpoint=False))
        model.load_state_dict(state['model']);del state
        model=model.to(device).eval().requires_grad_(False);model.dtype=torch.bfloat16
        state=torch.load(plan['vae_checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
        if state['manifest_digest']!=plan['source_manifest_digest']:raise ValueError('Wrong VAE source')
        vae=SegVaeV2(state['config'],initialize=False);vae.load_state_dict(state['model']);del state
        vae=vae.to(device).eval().requires_grad_(False);torch.cuda.synchronize()
        predroot=storage_path(plan['prediction_dir'],plan['vae_config']);predroot.mkdir(parents=True,exist_ok=True)
        rows=plan['samples'][rank::plan['workers']];reference={r['instance']:r for r in plan['reference']};completed=0
        for row in rows:
            uid=row['instance'];seed=plan['config']['seed']+int(uid[:8],16)
            data=load_latents(row,plan['config']['cache_dir']);geometry=load_geometry(row)
            template=vae.geometry_template(torch.as_tensor(geometry['coords'],device=device))
            if not np.array_equal(template.coords[:,1:].cpu().numpy(),data['latent_coords']):raise ValueError('Coordinates differ')
            qb=SurfaceQueries(geometry,vae.cfg['geometry'])
            for setting in plan['settings']:
                name=setting['name'];folder=args.output_dir/'metrics'/name;folder.mkdir(exist_ok=True)
                target=folder/f'{uid}.json';pf=predroot/name;pf.mkdir(exist_ok=True);prediction=pf/f'{uid}.npz'
                if target.exists():
                    r=json.loads(target.read_text())
                    if r['plan_digest']!=contract or not prediction.exists():raise ValueError('Incomplete/changed resume artifact')
                    completed+=1;continue
                atomic_json(args.output_dir/f'progress{rank:02d}.json',{'phase':'inference','done':completed,'total':len(rows)*len(plan['settings']),'instance':uid,'setting':name,'updated':time.time()})
                torch.cuda.synchronize();start=time.monotonic()
                x,nfe=generate(model,template,data,plan['normalization'],seed,setting,device)
                torch.cuda.synchronize();sampling_seconds=time.monotonic()-start
                result,faces,voxels=decode(vae,template,unnormalize(x,plan['normalization']),geometry,qb,device)
                parity=None
                if name=='correct_s20_cfg1':
                    parity=metric_difference(result,reference[uid])
                    if parity>1e-5:raise ValueError(f'Baseline parity changed: {uid}: {parity}')
                temporary=prediction.with_suffix('.npz.tmp')
                with temporary.open('wb') as h:
                    np.savez_compressed(h,face_labels=faces,voxel_labels=voxels,plan_digest=np.array(contract),seed=np.array(seed))
                temporary.replace(prediction)
                record={'instance':uid,'split':'validation','parts':row['parts'],'seed':seed,'plan_digest':contract,
                        'setting':setting,'metrics':result,'sampling_seconds':sampling_seconds,
                        'seconds':time.monotonic()-start,'nfe':nfe,'baseline_parity':parity}
                atomic_json(target,record);completed+=1
                print(f'worker{rank} {completed}/{len(rows)*len(plan["settings"])} {uid} {name}: {result["surface"]["miou"]:.4f}',flush=True)
            del template,data,geometry,qb
        atomic_json(args.output_dir/f'progress{rank:02d}.json',{'phase':'complete','done':completed,'total':len(rows)*len(plan['settings']),'updated':time.time()})


def run_node(args):
    from scripts.train_seg_vae_v2_remote import PYTHON
    from scripts.dit_vae_v2_remote import environment
    plan=json.loads((args.output_dir/'plan.json').read_text());ngpu=plan['gpus_per_node']
    cache=Path(plan['vae_config']['storage_root'])/'runtime_cache'/'dit_vae_v2_parts64'/f'node{args.node_rank}'
    env=dict(os.environ,**environment(cache))
    def run(local):
        rank=args.node_rank*ngpu+local
        cmd=[PYTHON,'-B','-u',str(Path(__file__).resolve()),'--worker',str(rank),'--device-index',str(local),'--output-dir',str(args.output_dir)]
        with (args.output_dir/'logs'/f'worker{rank:02d}.log').open('a') as log:
            r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(args.output_dir/'logs'/f'worker{rank:02d}_exit.json',{'worker':rank,'exit_code':r.returncode})
        print('Worker exited',rank,r.returncode,flush=True);return r.returncode
    with ThreadPoolExecutor(max_workers=ngpu) as pool:codes=list(pool.map(run,range(ngpu)))
    if any(codes):raise RuntimeError(f'Sweep worker failures: {codes}')


def launch(args):
    from scripts.train_seg_vae_v2_remote import api_client,API_ROOT,NODES,BOARD,dispatch,remote_worker_prefix
    plan=json.loads((args.output_dir/'plan.json').read_text());c=api_client(API_ROOT,'master')
    if (args.output_dir/'remote_launch.json').exists():raise ValueError('Already launched; inspect state before resubmitting')
    if any(s['session_name'].startswith('ditcheck_') for s in c.list_sessions()):raise ValueError('Existing inference check')
    board=next(b for b in c.list_boards() if b['id']==BOARD);nodes={m['ip']:m for m in board['machines']}
    if any(nodes[ip]['status']!='online' or nodes[ip]['gpu_count']<plan['gpus_per_node'] for ip in NODES):raise ValueError('Nodes unavailable')
    from common.security import check_agent_command
    tag=datetime.now().strftime('%Y%m%d_%H%M%S');jobs={'status':'planned','jobs':[],'board':BOARD}
    for rank,ip in enumerate(NODES):
        cmd=shlex.join([*remote_worker_prefix(),str(Path(__file__).resolve()),'--node-rank',str(rank),'--output-dir',str(args.output_dir)])
        log=args.output_dir/'logs'/f'node{rank}.log';ex=args.output_dir/'logs'/f'node{rank}.exit'
        cmd+=' > '+shlex.quote(str(log))+' 2>&1; dit_rc=$?; printf "%s\\n" "$dit_rc" > '+shlex.quote(str(ex))
        safe,reason=check_agent_command(cmd,protected_dirs=board.get('agent_readonly_dirs',[]))
        if not safe:raise ValueError(reason)
        jobs['jobs'].append({'rank':rank,'ip':ip,'session_name':f'ditcheck_{tag}_n{rank}','command':cmd,'log':str(log),'exit_file':str(ex)})
    atomic_json(args.output_dir/'remote_launch.json',jobs);dispatch(c,jobs,args.output_dir/'remote_launch.json')


def merge(args):
    import numpy as np
    from vae_v2.evaluation import summarize,selection_score
    plan=json.loads((args.output_dir/'plan.json').read_text());contract=digest(plan);ids={r['instance'] for r in plan['samples']}
    reports={};all_rows={}
    for setting in plan['settings']:
        name=setting['name'];rows=[json.loads(f.read_text()) for f in sorted((args.output_dir/'metrics'/name).glob('*.json'))]
        if len(rows)!=128 or {r['instance'] for r in rows}!=ids:raise ValueError('Incomplete '+name)
        for r in rows:
            if r['plan_digest']!=contract or r['setting']!=setting or r['split']!='validation':raise ValueError('Mixed experiment')
            if not (Path(plan['prediction_dir'])/name/f'{r["instance"]}.npz').is_file():raise ValueError('Missing predictions')
        summary=summarize([r['metrics'] for r in rows])
        reports[name]={'setting':setting,'summary':summary,'sampling_seconds_mean':float(np.mean([r['sampling_seconds'] for r in rows])),
                       'nfe':rows[0]['nfe'],'samples':rows}
        all_rows[name]={r['instance']:r for r in rows}
    baseline='correct_s20_cfg1'
    if max(r['baseline_parity'] for r in reports[baseline]['samples'])>1e-5:raise ValueError('Baseline failed parity')
    rng=np.random.default_rng(20261008);bootstrap=rng.integers(0,128,size=(2000,128))
    for name,report in reports.items():
        pairs=[]
        for uid in sorted(ids):
            a=all_rows[baseline][uid]['metrics'];b=all_rows[name][uid]['metrics']
            pairs.append({'instance':uid,'parts':all_rows[name][uid]['parts'],'before':a['surface']['miou'],
                          'after':b['surface']['miou'],'delta':b['surface']['miou']-a['surface']['miou']})
        deltas=np.array([r['delta'] for r in pairs]);ci=np.quantile(deltas[bootstrap].mean(1),[.025,.975]).tolist()
        report['paired']={'surface_miou_delta':float(deltas.mean()),'bootstrap95_object_resampling':ci,
                          'improved_gt_1pp':int((deltas>.01).sum()),'regressed_gt_1pp':int((deltas<-.01).sum()),
                          'regressed_gt_5pp':int((deltas<-.05).sum()),'pairs':pairs,
                          'largest_regressions':sorted(pairs,key=lambda r:r['delta'])[:8],
                          'largest_gains':sorted(pairs,key=lambda r:-r['delta'])[:8]}
        report['groups_by_parts']=[]
        for lo,hi in [(1,20),(21,32),(33,63)]:
            group=[r for r in pairs if lo<=r['parts']<=hi]
            report['groups_by_parts'].append({'parts':[lo,hi],'count':len(group),
                'baseline':float(np.mean([r['before'] for r in group])) if group else None,
                'miou':float(np.mean([r['after'] for r in group])) if group else None})
    best=max((s['name'] for s in plan['settings'] if s['eligible']),key=lambda n:selection_score(reports[n]['summary']))
    result={'checkpoint':plan['checkpoint'],'split':'validation','count':128,'baseline':baseline,'best_setting':best,
            'test_used':False,'seed_count_per_object':1,'selection_rule':plan['selection_rule'],
            'baseline_max_parity_error':max(r['baseline_parity'] for r in reports[baseline]['samples']),
            'note':'Exploratory paired validation check; CI resamples objects only, not random generation seeds; selected best is not an independent test result.',
            'experiments':reports,'completed_at':datetime.now().astimezone().isoformat()}
    atomic_json(args.output_dir/'summary.json',result)
    table=[]
    for name,r in reports.items():
        s=r['summary'];table.append({'name':name,'surface_miou':s['surface']['miou'],'projection_miou':s['projection']['miou'],
                                   'hidden_miou':s['hidden_surface']['miou'],'small_recall':s['surface']['small_part_recall'],
                                   'boundary_f1':s['boundary_f1'],'seconds':r['sampling_seconds_mean'],'nfe':r['nfe'],
                                   'delta':r['paired']['surface_miou_delta'],'regressions_gt1pp':r['paired']['regressed_gt_1pp']})
    atomic_json(args.output_dir/'table.json',table)
    print(json.dumps({'best':best,'table':table},indent=2),flush=True)


def export(args):
    import numpy as np
    import trimesh
    from PIL import Image,ImageDraw,ImageFont
    from scripts.train_dit_vae_v2 import load_geometry
    from vae_v2.visuals import render_palette,export_labels
    from utils.seg_projection import render_labels,VIEW_NAMES
    plan=json.loads((args.output_dir/'plan.json').read_text());summary=json.loads((args.output_dir/'summary.json').read_text())
    contract=digest(plan);base=summary['baseline'];best=summary['best_setting'];predroot=Path(plan['prediction_dir'])
    palette=render_palette(64);folder=args.output_dir/'objects';folder.mkdir(exist_ok=True)
    font_path=Path('/usr/share/fonts/dejavu/DejaVuSans.ttf');font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    display=list(dict.fromkeys([base,best,'correct_s50_cfg1','no_mv_s20','shuffled_s20']))
    def one(row):
        uid=row['instance'];sample=load_geometry(row);out=folder/uid;out.mkdir(exist_ok=True)
        mesh=trimesh.Trimesh(sample['mesh_vertices'],sample['mesh_faces'],process=False)
        labels={'GT':sample['face_labels']}
        for name in display:
            with np.load(predroot/name/f'{uid}.npz',allow_pickle=False) as data:
                if str(data['plan_digest'])!=contract:raise ValueError('Wrong prediction provenance')
                labels[name]=data['face_labels']
        canvas=Image.new('RGB',(1536,40+280*len(labels)),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,8),f'{uid} | validation | {row["parts"]} parts',font=font,fill='black')
        for i,(name,lab) in enumerate(labels.items()):
            maps=render_labels(lab+1,sample['face_raster'])
            for v,view in enumerate(VIEW_NAMES):
                y=40+i*280;draw.text((v*256+3,y+3),name+': '+view,fill='black')
                canvas.paste(Image.fromarray(palette[maps[v]]).resize((256,256),Image.Resampling.NEAREST),(v*256,y+24))
        canvas.save(out/'comparison.png')
        for name in dict.fromkeys(['GT',base,best]):export_labels(mesh,labels[name],out/f'{name}.glb',64)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(one,plan['samples']))
    table=json.loads((args.output_dir/'table.json').read_text())
    headings='<tr><th>setting</th><th>surface mIoU</th><th>projection mIoU</th><th>hidden mIoU</th><th>small recall</th><th>mean sampling s</th><th>NFE</th></tr>'
    lines=''.join('<tr><td>'+r['name']+'</td>'+''.join(f'<td>{100*r[k]:.2f}%</td>' for k in ['surface_miou','projection_miou','hidden_miou','small_recall'])+f'<td>{r["seconds"]:.2f}</td><td>{r["nfe"]}</td></tr>' for r in table)
    rows={r['instance']:r for r in plan['samples']}
    pairs={r['instance']:r for r in summary['experiments'][best]['paired']['pairs']}
    def card(uid):
        r=pairs[uid]
        return f'<section><h3>{uid} | {rows[uid]["parts"]} parts</h3><p>baseline {r["before"]:.4f} → {best} {r["after"]:.4f} ({r["delta"]*100:+.2f} pp). <a href="objects/{uid}/{base}.glb">baseline GLB</a> · <a href="objects/{uid}/{best}.glb">best GLB</a> · <a href="objects/{uid}/GT.glb">GT GLB</a></p><a href="objects/{uid}/comparison.png"><img loading="lazy" src="objects/{uid}/comparison.png"></a></section>'
    focus=list(dict.fromkeys([r['instance'] for r in summary['experiments'][best]['paired']['largest_gains'][:4]]+[r['instance'] for r in summary['experiments'][best]['paired']['largest_regressions'][:4]]))
    body=('<!doctype html><meta charset="utf-8"><title>DiT inference check</title><style>body{max-width:1550px;margin:auto;padding:24px;font:16px sans-serif}img{width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}section{margin:32px 0}</style>'
          '<h1>Validation-only DiT inference check</h1><p>Frozen weights; identical seeds; GT-derived MV conditioning. No training, no test-set selection.</p>'
          f'<p>Best among correct-condition settings: {html.escape(best)}. {html.escape(summary["note"])}</p>'
          '<p>Shuffled = feature-to-coordinate mapping randomized within each view; shape retained. No-MV is a diagnostic, not a candidate.</p>'
          '<table>'+headings+lines+'</table><h2>Largest gains and regressions</h2>'+''.join(card(u) for u in focus)+
          '<h2>All 128 validation objects</h2>'+''.join(card(u) for u in sorted(rows)))
    (args.output_dir/'index.html').write_text(body)
    atomic_json(args.output_dir/'artifacts.json',{'objects':128,'displayed_settings':display,'glb_count':len(list(folder.rglob('*.glb'))),'comparison_count':len(list(folder.rglob('comparison.png'))),'prediction_count':len(list(predroot.rglob('*.npz'))),'glb_roundtrip_verified':True})
    print('Gallery saved',args.output_dir/'index.html',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/dit_vae_v2_parts64.json')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--gpus',type=int,default=8);ap.add_argument('--device-index',type=int,default=0)
    modes=ap.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--worker',type=int)
    modes.add_argument('--node-rank',type=int);modes.add_argument('--launch',action='store_true')
    modes.add_argument('--merge',action='store_true');modes.add_argument('--export',action='store_true')
    args=ap.parse_args();args.output_dir=args.output_dir.resolve()
    if not 1<=args.gpus<=8:ap.error('gpus must be 1..8')
    if args.prepare:prepare(args)
    elif args.worker is not None:worker(args)
    elif args.node_rank is not None:
        if not 0<=args.node_rank<3:ap.error('node-rank must be 0..2')
        run_node(args)
    elif args.launch:launch(args)
    elif args.merge:merge(args)
    else:export(args)


if __name__=='__main__':main()
