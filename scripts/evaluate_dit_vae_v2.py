"""Frozen validation-selected DiT test with GT / cached VAE oracle / generated labels.

Independent Remote API GPU shards; no training, GT latent initialization, label
matching, postprocessing or checkpoint selection from test results.
"""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import atomic_json,digest,storage_path
from utils.dit_vae_v2 import config,load_cache,identity,file_sha256

DIAGNOSTICS=['007d2949e5b0450c995c54809b43af0f','11c88bc30d37479281c30cc49d25084a',
             '141329918e4945ed82d0f48ba5903b8f']


def select_cohort(manifest,selection,validation_report,diagnostic_ids=()):
    by_id={r['instance']:r for r in manifest['samples']}
    valid={u for u,r in by_id.items() if r['split']=='validation'}
    if (selection['split']!='validation' or selection['test_used_for_selection']
            or set(selection['ids'])!=valid or len(selection['ids'])!=128):
        raise ValueError('Expected a checkpoint selected only using all 128 validation objects')
    for key in ('checkpoint','step','kind','sampling_steps','summary','ids'):
        if validation_report[key]!=selection[key]:raise ValueError('Selection and saved validation disagree')
    if {r['instance'] for r in validation_report['samples']}!=valid:raise ValueError('Invalid validation report membership')
    rows=[dict(r,role='test') for r in manifest['samples'] if r['split']=='test']
    if len(rows)!=128 or len({r['instance'] for r in rows})!=128:raise ValueError('Incomplete test cohort')
    rows += [dict(by_id[u],role='train_diagnostic') for u in DIAGNOSTICS if by_id[u]['split']=='train']
    # Fixed validation parity checks, not an additional test or checkpoint sweep.
    parity_ids=sorted(valid)[:2]+[min(validation_report['samples'],key=lambda r:r['surface']['miou'])['instance']]
    rows += [dict(by_id[u],role='validation_parity') for u in dict.fromkeys(parity_ids)]
    if len(set(diagnostic_ids))!=len(diagnostic_ids) or any(u not in valid for u in diagnostic_ids):
        raise ValueError('Additional diagnostic IDs must be unique validation objects')
    present={r['instance'] for r in rows}
    rows += [dict(by_id[u],role='validation_diagnostic') for u in diagnostic_ids if u not in present]
    return rows


def prepare(args):
    cfg,vcfg=config(args.config);cache_plan,manifest=load_cache(cfg)
    selection_path=Path(cfg['output_dir'])/'selection.json'
    selected=json.loads(selection_path.read_text())
    validation_path=Path(cfg['output_dir'])/'validation'/f'{selected["kind"]}_{selected["step"]:06d}.json'
    validation=json.loads(validation_path.read_text())
    rows=select_cohort(manifest,selected,validation,getattr(args,'diagnostic_ids',()))
    trained=json.loads((Path(cfg['output_dir'])/'manifest.json').read_text())
    base=json.loads(Path(cfg['base_config']).read_text())
    if (trained['smoke_only'] or trained['cache_digest']!=digest(manifest)
            or trained['vae_sha256']!=cache_plan['vae_sha256'] or trained['base_digest']!=digest(base)):
        raise ValueError('Training/cache/VAE/architecture provenance mismatch')
    pred=storage_path(Path(vcfg['storage_root'])/'predictions'/args.output_dir.name,vcfg)
    if pred.exists() and any(pred.iterdir()):raise ValueError('Use a new numerical prediction directory')
    result={'config':cfg,'vae_config':vcfg,'checkpoint':identity(selected['checkpoint']),
            'vae_checkpoint':cache_plan['vae_checkpoint'],'vae_sha256':cache_plan['vae_sha256'],
            'cache_digest':digest(manifest),'source_manifest_digest':cache_plan['source_manifest_digest'],
            'base_config':base,'base_digest':digest(base),'normalization':manifest['normalization'],
            'selection':selected,'selection_source':str(selection_path),
            'validation_reference':validation['samples'],'samples':rows,'workers':args.workers,
            'sampling_steps':selected['sampling_steps'],'seed_policy':'training_seed_plus_first8_hex_instance',
            'prediction_dir':str(pred),'created_at':datetime.now().astimezone().isoformat(),
            'condition':'GT-derived six-view segmentation latents + native shape features; known mesh geometry',
            'generation':'Gaussian noise -> DiT velocity Euler sampling -> frozen VAE surface decode',
            'oracle':'cached GT posterior-mean latent -> same frozen VAE surface decode',
            'postprocessing':False,'label_matching':False,'test_used_for_selection':False}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Use a new evaluation directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for folder in ('metrics','objects','logs'):(args.output_dir/folder).mkdir()
    atomic_json(args.output_dir/'plan.json',result)
    print('Prepared',len(rows),'objects: 128 test + training diagnostics + validation parity; steps',result['sampling_steps'],flush=True)


def normalize_conditions(data,stats,device):
    """Target latent VALUES are never consumed by the DiT generation path."""
    import torch
    out={}
    for k in ('shape_feats','mv_feats'):
        x=torch.as_tensor(data[k],dtype=torch.float32,device=device)
        x=(x-torch.tensor(stats[k]['mean'],device=device))/torch.tensor(stats[k]['std'],device=device)
        if k=='shape_feats':x=x.masked_fill(~torch.as_tensor(data['shape_valid'],device=device)[:,None],0)
        out[k]=x
    return out


def generate_latent(model,template,data,stats,seed,steps,device):
    import torch
    from utils.dit_vae_v2 import conditions
    normalized=normalize_conditions(data,stats,device)
    rng=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn((len(data['latent_coords']),32),device=device,generator=rng)
    cond=conditions(template,data,normalized,False)
    with torch.no_grad():
        for i in range(steps):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                v=model(template.replace(x),torch.tensor([(1-i/steps)*1000],device=device),**cond).feats.float()
            if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite DiT velocity')
            x=x-v/steps
    return x


def metrics(sample,face,voxel):
    import numpy as np
    from vae_v2.evaluation import classification,surface_diagnostics
    target=sample['face_labels'];area=sample['face_area'].astype(np.float64)
    out={'surface':classification(target,face,area),
         'projection':classification(target,face,sample['face_pixels']),
         'voxel':classification(sample['voxel_labels'],voxel,np.ones(len(voxel))),
         **surface_diagnostics(sample,face)}
    out['parts']=[]
    for k in np.unique(target[target>0]):
        a,b=target==k,face==k;intersection=area[a&b].sum()
        out['parts'].append({'label':int(k),'area_fraction':float(area[a].sum()/area.sum()),
                             'recall':float(intersection/max(area[a].sum(),1e-12)),
                             'iou':float(intersection/max(area[a|b].sum(),1e-12))})
    out['unknown_prediction_area']=float(area[face==0].sum()/area.sum())
    return out


def decode(vae,template,raw,geometry,query_builder,device):
    import numpy as np
    import torch
    from vae_v2.data import to_device
    with torch.no_grad():
        features,logits,coords=vae.decode_voxels(template.replace(raw))
        if not np.array_equal(coords[:,1:].cpu().numpy(),geometry['coords']):raise ValueError('Decoded coordinates differ')
        labels=[];fallback=0
        for start in range(0,len(geometry['face_labels']),1024):
            ids=np.arange(start,min(start+1024,len(geometry['face_labels'])))
            q=query_builder.build(ids);fallback+=int(q['query_fallback'].sum())
            with torch.autocast('cuda',dtype=torch.bfloat16):_,scores=vae.decode_surface(features,logits,to_device(q,device))
            if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite surface probabilities')
            labels.append(scores.argmax(1).cpu().numpy())
        face=np.concatenate(labels);voxel=logits.argmax(1).cpu().numpy()
    result=metrics(geometry,face,voxel);result['query_fallback_fraction']=fallback/(4*len(face))
    return result,face,voxel


def export(out,uid,sample,oracle,generated,dit_score,oracle_score):
    import numpy as np
    import trimesh
    from PIL import Image,ImageDraw,ImageFont
    from vae_v2.visuals import export_labels,render_palette
    from utils.seg_projection import render_labels,VIEW_NAMES
    out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.Trimesh(sample['mesh_vertices'],sample['mesh_faces'],process=False)
    for name,labels in [('gt',sample['face_labels']),('vae',oracle),('dit',generated)]:
        export_labels(mesh,labels,out/f'{uid}_{name}.glb',64)
    palette=render_palette(64);raster=sample['face_raster']
    canvas=Image.new('RGB',(1536,890),'white');draw=ImageDraw.Draw(canvas)
    font_path=Path('/usr/share/fonts/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    draw.text((8,8),f'{uid} | surface mIoU: VAE={oracle_score:.4f} / DiT={dit_score:.4f}',font=font,fill='black')
    error=Image.new('RGB',(1536,560),'white');ed=ImageDraw.Draw(error)
    for row,(name,labels) in enumerate([('GT',sample['face_labels']),('VAE reconstruction',oracle),('DiT generation',generated)]):
        maps=render_labels(labels+1,raster)
        for i,view in enumerate(VIEW_NAMES):
            im=Image.fromarray(palette[maps[i]])
            y=50+row*280;draw.text((i*256+4,y+3),name+': '+view,fill='black')
            canvas.paste(im.resize((256,256),Image.Resampling.NEAREST),(i*256,y+24))
            if row==2:im.save(out/f'dit_{view}.png')
            if row:
                rr=raster[i];hit=rr>=0;gt=sample['face_labels'][rr.clip(min=0)]
                wrong=hit&(gt>0)&(labels[rr.clip(min=0)]!=gt)
                rgb=np.full((*rr.shape,3),255,np.uint8);rgb[hit]=[205,213,221];rgb[wrong]=[229,55,74]
                ey=(row-1)*280;ed.text((i*256+4,ey+3),name+': '+view,fill='black')
                error.paste(Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST),(i*256,ey+24))
    canvas.save(out/f'{uid}_comparison.png');error.save(out/f'{uid}_errors.png')


def worker(args):
    import numpy as np
    import torch
    from utils.dit_vae_v2 import load_latents,unnormalize
    from scripts.train_dit_vae_v2 import load_geometry
    from vae_v2.model import SegVaeV2
    from vae_v2.geometry import SurfaceQueries
    from models.seg_flow_model import SegSLatFlowModel
    plan=json.loads((args.output_dir/'plan.json').read_text());rank=args.worker
    if not 0<=rank<plan['workers']:raise ValueError('Invalid worker index')
    for key in ('checkpoint','vae_checkpoint'):
        if identity(plan[key]['path'])!=plan[key]:raise ValueError(key+' changed after freezing evaluation plan')
    device='cuda:0';torch.cuda.set_device(0);torch.set_num_threads(2);torch.set_grad_enabled(False)
    torch.manual_seed(0)
    print(f'worker{rank}: loading selected DiT and frozen VAE',flush=True)
    state=torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
    metadata=state['metadata']
    if (metadata['smoke_only'] or metadata['cache_digest']!=plan['cache_digest']
            or metadata['base_digest']!=plan['base_digest'] or metadata['vae_sha256']!=plan['vae_sha256']
            or state['step']!=plan['selection']['step'] or state['kind']!=plan['selection']['kind']):
        raise ValueError('DiT checkpoint does not match frozen evaluation plan')
    model=SegSLatFlowModel(**dict(plan['base_config']['models']['denoiser']['args'],use_checkpoint=False))
    model.load_state_dict(state['model']);del state
    model=model.to(device).eval();model.dtype=torch.bfloat16
    state=torch.load(plan['vae_checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
    if state['manifest_digest']!=plan['source_manifest_digest'] or state.get('smoke_only'):raise ValueError('VAE provenance mismatch')
    vae=SegVaeV2(state['config'],initialize=False);vae.load_state_dict(state['model']);del state
    vae=vae.to(device).eval().requires_grad_(False);torch.cuda.synchronize()
    predroot=storage_path(plan['prediction_dir'],plan['vae_config']);predroot.mkdir(parents=True,exist_ok=True)
    rows=plan['samples'][rank::plan['workers']];plan_digest=digest(plan)
    for index,row in enumerate(rows):
        uid=row['instance'];start=time.monotonic()
        atomic_json(args.output_dir/f'progress{rank}.json',{'phase':'inference','done':index,'total':len(rows),'instance':uid})
        data=load_latents(row,plan['config']['cache_dir']);geometry=load_geometry(row)
        template=vae.geometry_template(torch.as_tensor(geometry['coords'],device=device))
        if not np.array_equal(template.coords[:,1:].cpu().numpy(),data['latent_coords']):raise ValueError('Template/latent coords mismatch')
        seed=plan['config']['seed']+int(uid[:8],16)
        generated=generate_latent(model,template,data,plan['normalization'],seed,plan['sampling_steps'],device)
        query_builder=SurfaceQueries(geometry,vae.cfg['geometry'])
        dit,faces,voxels=decode(vae,template,unnormalize(generated,plan['normalization']),geometry,query_builder,device)
        oracle,ofaces,ovoxels=decode(vae,template,torch.as_tensor(data['latent_feats'],device=device),geometry,query_builder,device)
        export(args.output_dir/'objects'/uid,uid,geometry,ofaces,faces,dit['surface']['miou'],oracle['surface']['miou'])
        temp=predroot/f'{uid}.npz.tmp';target=predroot/f'{uid}.npz'
        with temp.open('wb') as h:
            np.savez_compressed(h,face_labels=faces,voxel_labels=voxels,oracle_faces=ofaces,oracle_voxels=ovoxels,
                                plan_digest=np.array(plan_digest),seed=np.array(seed))
        temp.replace(target)
        record={'instance':uid,'role':row['role'],'split':row['split'],'type_id':row['type_id'],
                'parts':row['parts'],'shape_coverage':row['shape_coverage'],'seed':seed,'plan_digest':plan_digest,
                'dit':dit,'oracle':oracle,'seconds':time.monotonic()-start,'glb_roundtrip_verified':True}
        atomic_json(args.output_dir/'metrics'/f'{uid}.json',record)
        print(f'worker{rank} {index+1}/{len(rows)} {uid}: DiT={dit["surface"]["miou"]:.4f} VAE={oracle["surface"]["miou"]:.4f}',flush=True)
        del data,geometry,template,generated,query_builder
    atomic_json(args.output_dir/f'progress{rank}.json',{'phase':'complete','done':len(rows),'total':len(rows)})


def merge(args):
    from PIL import Image,ImageDraw
    from vae_v2.evaluation import summarize
    from utils.seg_projection import VIEW_NAMES
    plan=json.loads((args.output_dir/'plan.json').read_text());p_digest=digest(plan)
    rows=[json.loads(p.read_text()) for p in sorted((args.output_dir/'metrics').glob('*.json'))]
    expected={r['instance']:r for r in plan['samples']}
    if len(rows)!=len(expected) or {r['instance'] for r in rows}!=expected.keys():raise ValueError('Incomplete/duplicated inference')
    for r in rows:
        uid=r['instance'];folder=args.output_dir/'objects'/uid
        if r['plan_digest']!=p_digest or r['role']!=expected[uid]['role'] or r['split']!=expected[uid]['split']:raise ValueError('Mixed inference plans')
        names=[f'{uid}_{k}.glb' for k in ('gt','vae','dit')]+[f'{uid}_comparison.png',f'{uid}_errors.png']+[f'dit_{v}.png' for v in VIEW_NAMES]
        if not all((folder/n).is_file() and (folder/n).stat().st_size for n in names):raise ValueError('Missing artifacts '+uid)
        if not (Path(plan['prediction_dir'])/f'{uid}.npz').is_file():raise ValueError('Missing numerical prediction '+uid)
    tests=[r for r in rows if r['role']=='test'];diag=[r for r in rows if r['role']=='train_diagnostic']
    parity=[];reference={r['instance']:r for r in plan['validation_reference']}
    for r in rows:
        if r['role'] in ('validation_parity','validation_diagnostic'):
            a=reference[r['instance']];keys=['surface','projection','voxel','hidden_surface']
            delta=max(abs(r['dit'][g][k]-a[g][k]) for g in keys for k in ('accuracy','miou'))
            parity.append({'instance':r['instance'],'max_metric_abs_diff':delta,'passed':delta<1e-5})
    if not all(r['passed'] for r in parity):raise ValueError(f'Training validation parity failed: {parity}')
    result={'checkpoint':plan['checkpoint'],'vae_checkpoint':plan['vae_checkpoint'],'sampling_steps':plan['sampling_steps'],
            'cache_digest':plan['cache_digest'],'condition':plan['condition'],'test_used_for_selection':False,
            'summary':summarize([r['dit'] for r in tests]),'oracle_summary':summarize([r['oracle'] for r in tests]),
            'ids':[r['instance'] for r in tests],'samples':tests,'diagnostics':diag,'validation_parity':parity,
            'validation_diagnostics':[r for r in rows if r['role']=='validation_diagnostic'],
            'completed_at':datetime.now().astimezone().isoformat(),'prediction_dir':plan['prediction_dir']}
    atomic_json(args.output_dir/'metrics.json',result)
    worst=sorted(tests,key=lambda r:r['dit']['surface']['miou'])[:10]
    atomic_json(args.output_dir/'worst10.json',worst)
    for page,start in enumerate(range(0,len(tests),16),1):
        group=tests[start:start+16];canvas=Image.new('RGB',(1536,475*((len(group)+1)//2)),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(group):
            uid=r['instance'];x,y=(i%2)*768,(i//2)*475
            draw.text((x+6,y+3),f'{start+i+1:03d} {uid} DiT mIoU={r["dit"]["surface"]["miou"]:.3f}',fill='black')
            with Image.open(args.output_dir/'objects'/uid/f'{uid}_comparison.png') as im:canvas.paste(im.resize((768,445)),(x,y+24))
        canvas.save(args.output_dir/f'overview_{page:02d}.jpg',quality=90)
    lookup={r['instance']:r for r in rows}
    def card(r):
        uid=r['instance'];base=f'objects/{uid}/{uid}'
        return (f'<section><h3>{uid} — {r["split"]}/{r["role"]}, {r["parts"]} parts</h3>'
                f'<p>DiT surface mIoU {r["dit"]["surface"]["miou"]:.4f}; VAE oracle {r["oracle"]["surface"]["miou"]:.4f}. '
                + ' · '.join(f'<a href="{base}_{k}.glb">{k.upper()} GLB</a>' for k in ('dit','vae','gt'))+
                f' · <a href="{base}_errors.png">Errors</a></p><a href="{base}_comparison.png"><img loading="lazy" src="{base}_comparison.png"></a></section>')
    focus=['251c8125765149c0bd46b3a1a51ff4b6','d1d58f05c346470e9805612d562e9dde']+DIAGNOSTICS
    document=('<!doctype html><meta charset="utf-8"><title>DiT with VAE v2 test</title>'
              '<style>body{max-width:1536px;margin:auto;padding:24px;font:16px sans-serif}img{width:100%}section{margin:30px 0}pre{background:#eee;padding:16px}</style>'
              '<h1>DiT generation with frozen VAE v2</h1>'
              '<p>Rows: GT / VAE reconstruction / DiT generation. No smoothing or label matching.</p>'
              f'<p>{html.escape(plan["condition"])}. {plan["sampling_steps"]} Euler steps, fixed per-object seed, no CFG.</p>'
              '<p>This is GT-derived conditioning, not an external segmentation API end-to-end evaluation. Training diagnostics and validation parity objects are excluded from test metrics.</p>'
              f'<pre>{html.escape(json.dumps({"DiT":result["summary"],"VAE_oracle":result["oracle_summary"]},indent=2))}</pre>'
              '<h2>Previously identified cases</h2>'+''.join(card(lookup[u]) for u in focus if u in lookup)+
              '<h2>10 lowest DiT surface-mIoU test objects</h2>'+''.join(card(r) for r in worst)+
              '<h2>Validation diagnostics (excluded from test)</h2>'+''.join(card(r) for r in rows if r['role']=='validation_diagnostic')+
              '<h2>All 128 test objects in fixed ID order</h2>'+''.join(card(r) for r in tests))
    (args.output_dir/'index.html').write_text(document)
    print(json.dumps({'DiT':result['summary'],'VAE_oracle':result['oracle_summary'],'validation_parity':parity},indent=2))
    print('Verified',len(rows),'objects,',len(rows)*3,'GLBs',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/dit_vae_v2_parts64.json')
    ap.add_argument('--output-dir',type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--worker',type=int);mode.add_argument('--merge',action='store_true')
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--diagnostic-ids',nargs='*',default=[])
    args=ap.parse_args();args.output_dir=args.output_dir.resolve()
    if args.workers<1:ap.error('workers must be positive')
    if args.prepare:prepare(args)
    elif args.merge:merge(args)
    else:worker(args)


if __name__=='__main__':main()
