"""Versioned target + six-view latents from one frozen, validation-selected VAE v2."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import atomic_json,digest
from utils.dit_vae_v2 import (config,identity,file_sha256,align_shape,moments,normalization,
                              validate_latents,FORMAT,capacity)


def prepare_plan(args,cfg,vcfg):
    from vae_v2.data import load_manifest
    print('Checking corrected VAE source manifest',flush=True)
    source=load_manifest(vcfg,verify_workers=16)
    selection=json.loads(Path(cfg['vae_selection']).read_text())
    val={r['instance'] for r in source['samples'] if r['split']=='validation'}
    if (selection['split']!='validation' or selection['test_used_for_selection'] or selection.get('smoke_only')
            or selection['manifest_digest']!=digest(source) or set(selection['ids'])!=val):
        raise ValueError('Invalid frozen VAE selection')
    checkpoint=identity(selection['checkpoint'])
    samples=source['samples']
    if args.ids:
        by_id={r['instance']:r for r in samples}
        if len(set(args.ids))!=len(args.ids):raise ValueError('Duplicate pilot IDs')
        samples=[by_id[u] for u in args.ids]
    elif args.limit:
        if args.limit<3:raise ValueError('Pilot requires >=3 objects')
        # Two complex problem examples plus fixed samples; never a training release.
        special=['251c8125765149c0bd46b3a1a51ff4b6','d1d58f05c346470e9805612d562e9dde']
        ids=list(dict.fromkeys(special+[r['instance'] for r in samples]))[:args.limit]
        by_id={r['instance']:r for r in samples};samples=[by_id[u] for u in ids]
    def shape_info(row):
        result=dict(row)
        result['shape_source']=identity(Path(cfg['shape_dir'])/f'{row["instance"]}.npz')
        return result
    with ThreadPoolExecutor(max_workers=16) as pool:samples=list(pool.map(shape_info,samples))
    plan={'format':FORMAT,'config':cfg,'vae_config':vcfg,'vae_selection':selection,
          'vae_checkpoint':checkpoint,'vae_sha256':file_sha256(checkpoint['path']),
          'source_manifest_digest':digest(source),'source_manifest_identity':identity(Path(vcfg['prepared_dir'])/'manifest.json'),
          'posterior':'mean','label_policy':'original_corrected_ids_0_to_63',
          'mv_policy':'six_cached_face_rasters_render_label_plus_one_box_surface_512_no_merge',
          'shape_policy':'reuse_same_object_native_shape_latent_exact_32_grid_join_mask_missing',
          'samples':samples,'workers':args.workers,'pilot_only':bool(args.limit or args.ids),
          'normalization_policy':'token_weighted_training_only_float64_moments'}
    root=Path(cfg['cache_dir'])
    if root.exists() and any(root.iterdir()):raise ValueError('Use fresh cache root, or --worker/--merge existing plan')
    root.mkdir(parents=True)
    for folder in ('samples','records'):(root/folder).mkdir()
    atomic_json(root/'plan.json',plan)
    print('PLAN_READY',root,len(samples),'objects',args.workers,'workers',flush=True)


def encode(args,cfg):
    import torch
    from vae_v2.model import SegVaeV2
    from vae_v2.data import load_sample,validate_sample,voxel_tensors
    from preprocess.step2_encode_mv_seg_latent import encode_one_view
    from utils.seg_projection import render_labels
    root=Path(cfg['cache_dir']);plan=json.loads((root/'plan.json').read_text());contract=digest(plan)
    capacity(cfg,plan)
    rank=args.worker;world=plan['workers']
    if not 0<=rank<world:raise ValueError('Invalid worker rank')
    gpu=rank%8;torch.cuda.set_device(gpu);device=f'cuda:{gpu}'
    torch.set_grad_enabled(False);torch.set_num_threads(2)
    with (root/f'.worker{rank}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if identity(plan['vae_checkpoint']['path'])!=plan['vae_checkpoint']:raise ValueError('VAE changed')
        print(f'worker{rank}: loading frozen VAE {plan["vae_checkpoint"]["path"]}',flush=True)
        state=torch.load(plan['vae_checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
        if state['manifest_digest']!=plan['source_manifest_digest'] or state.get('smoke_only'):raise ValueError('Wrong VAE checkpoint')
        vae=SegVaeV2(state['config'],initialize=False);vae.load_state_dict(state['model']);del state
        vae=vae.to(device).eval().requires_grad_(False)
        torch.cuda.synchronize()
        assigned=plan['samples'][rank::world];done=[];failures=[]
        for index,row in enumerate(assigned):
            uid=row['instance'];t0=time.monotonic();target=root/'samples'/f'{uid}.npz';record=root/'records'/f'{uid}.json'
            try:
                if target.exists() and record.exists():
                    r=json.loads(record.read_text());s=target.stat()
                    if r['plan_digest']!=contract or (s.st_size,s.st_mtime_ns)!=(r['bytes'],r['mtime_ns']):raise ValueError('Existing cache differs')
                    done.append(uid);continue
                if target.exists():raise ValueError('Unsealed existing cache; inspect/move before retry')
                source=Path(plan['vae_config']['prepared_dir'])/row['path'];s=source.stat()
                if (s.st_size,s.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):raise ValueError('VAE sample changed')
                if identity(row['shape_source']['path'])!=row['shape_source']:raise ValueError('Shape source changed')
                sample=load_sample(source);validate_sample(sample,64)
                z,_,_=vae.encode(voxel_tensors(sample,device),sample_posterior=False)
                coords=z.coords[:,1:].cpu().numpy().astype(np.int32);feats=z.feats.float().cpu().numpy()
                if len(coords)>cfg['max_tokens']:raise ValueError(f'Target {len(coords)} tokens exceeds {cfg["max_tokens"]}')
                if not np.array_equal(np.unique(sample['coords']//16,axis=0),np.unique(coords,axis=0)):
                    raise ValueError('Unexpected spatial downsampling contract')
                # Pilot verifies GT-free geometry cache and actual decoder output order.
                parity=None
                if plan['pilot_only']:
                    template=vae.geometry_template(torch.as_tensor(sample['coords'],device=device))
                    if not torch.equal(template.coords,z.coords):raise ValueError('Template coords differ')
                    _,a,ac=vae.decode_voxels(z);_,b,bc=vae.decode_voxels(template.replace(z.feats))
                    parity=float((a-b).abs().max());torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
                    if not np.array_equal(ac[:,1:].cpu().numpy(),sample['coords']):raise ValueError('Decoder order differs')
                    del template,a,b,ac,bc
                del z
                maps=render_labels(sample['face_labels']+1,sample['face_raster'])
                mv_coords=[];mv_feats=[];offset=[0]
                for view in range(6):
                    vc,vf=encode_one_view(maps[view],view,vae.encoder,64,device)
                    if vc is not None:
                        mv_coords.append(vc.numpy().astype(np.int32));mv_feats.append(vf.float().numpy())
                        offset.append(offset[-1]+len(vc))
                    else:offset.append(offset[-1])
                if not mv_feats:raise ValueError('No visible conditioning pixels')
                with np.load(row['shape_source']['path'],allow_pickle=False) as shape:
                    sf,valid=align_shape(coords,shape['coords'],shape['feats'])
                data={'latent_coords':coords,'latent_feats':feats,'shape_feats':sf,'shape_valid':valid,
                      'mv_coords':np.concatenate(mv_coords),'mv_feats':np.concatenate(mv_feats),
                      'view_offsets':np.asarray(offset,np.int32)}
                validate_latents(data,cfg['max_tokens'])
                meta={'instance':uid,'split':row['split'],'plan_digest':contract,'format':FORMAT,
                      'vae_sha256':plan['vae_sha256'],'posterior':'mean','num_classes':64}
                tmp=target.with_suffix('.tmp')
                with tmp.open('wb') as handle:np.savez_compressed(handle,**data,metadata=np.array(json.dumps(meta)))
                tmp.replace(target);stat=target.stat()
                r={'instance':uid,'split':row['split'],'type_id':row['type_id'],'path':str(target.relative_to(root)),
                   'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':file_sha256(target),'plan_digest':contract,
                   'tokens':len(coords),'mv_tokens':len(data['mv_coords']),'shape_coverage':float(valid.mean()),
                   'shape_source':row['shape_source'],'source_path':str(source),'source_bytes':row['bytes'],
                   'source_mtime_ns':row['mtime_ns'],'parts':row['annotation_summary']['parts'],
                   'moments':{k:moments(data[k][valid] if k=='shape_feats' else data[k]) for k in ('latent_feats','mv_feats','shape_feats')},
                   'seconds':time.monotonic()-t0,'pilot_decode_max_error':parity}
                atomic_json(record,r);done.append(uid)
                if index<3 or (index+1)%25==0:print(f'worker{rank} {index+1}/{len(assigned)} {uid} tokens={len(coords)} mv={len(data["mv_coords"])} shape={valid.mean():.3f} {r["seconds"]:.2f}s',flush=True)
            except Exception as exc:
                failures.append({'instance':uid,'error':repr(exc)})
                print(f'FAILED worker{rank} {uid}: {exc}',flush=True)
                torch.cuda.empty_cache()
            finally:
                atomic_json(root/f'progress{rank:02d}.json',{'rank':rank,'total':len(assigned),'done':len(done),
                            'failures':failures,'last':uid,'updated':time.time(),'complete':index+1==len(assigned)})
        if failures:raise RuntimeError(f'worker{rank}: {len(failures)} failures')


def seal(args,cfg):
    root=Path(cfg['cache_dir']);plan=json.loads((root/'plan.json').read_text());contract=digest(plan)
    ceiling=capacity(cfg,plan)
    expected={r['instance']:r for r in plan['samples']}
    records=list((root/'records').glob('*.json'))
    if len(records)!=len(expected):raise ValueError(f'Expected {len(expected)} records, found {len(records)}')
    print('Deep checking arrays, identities, checksum, all shards and train-only statistics',flush=True)
    for rank in range(plan['workers']):
        r=json.loads((root/f'progress{rank:02d}.json').read_text())
        if r['failures'] or not r['complete'] or r['done']!=r['total']:raise ValueError(f'Incomplete worker {rank}')
    def check(path):
        r=json.loads(path.read_text());src=expected[r['instance']];target=root/r['path'];st=target.stat()
        if r['plan_digest']!=contract or r['split']!=src['split']:raise ValueError('Mixed shard/selection')
        if (st.st_size,st.st_mtime_ns)!=(r['bytes'],r['mtime_ns']) or file_sha256(target)!=r['sha256']:raise ValueError('Changed cache')
        with np.load(target,allow_pickle=False) as d:
            data={k:d[k] for k in d.files if k!='metadata'};meta=json.loads(str(d['metadata']))
        validate_latents(data,cfg['max_tokens'])
        if meta['plan_digest']!=contract or meta['instance']!=r['instance'] or meta['split']!=r['split']:raise ValueError('Bad embedded provenance')
        for k in ('latent_feats','mv_feats','shape_feats'):
            measured=moments(data[k][data['shape_valid']] if k=='shape_feats' else data[k])
            if measured!=r['moments'][k]:raise ValueError('Statistics mismatch')
        return r
    with ThreadPoolExecutor(max_workers=16) as pool:rows=sorted(pool.map(check,records),key=lambda r:r['instance'])
    if {r['instance'] for r in rows}!=expected.keys():raise ValueError('Duplicate or missing objects')
    if {p.stem for p in (root/'samples').glob('*.npz')}!=expected.keys():raise ValueError('Orphan cached object')
    splits={s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}
    if not plan['pilot_only'] and splits!={'train':22633,'validation':128,'test':128}:raise ValueError('Changed split')
    norm=normalization(rows)
    atomic_json(root/'normalization.json',norm)
    # Full per-token moments remain in record files; do not duplicate into a huge training manifest.
    manifest={'format':FORMAT,'status':'complete','plan_digest':contract,'expected':len(expected),'failures':[],
              'split_counts':splits,'pilot_only':plan['pilot_only'],'normalization':norm,'max_tokens':ceiling,
              'samples':[{k:v for k,v in r.items() if k!='moments'} for r in rows]}
    atomic_json(root/'manifest.json',manifest)
    atomic_json(root/'audit.json',{'plan_digest':contract,'manifest_sha256':file_sha256(root/'manifest.json'),
                                  'objects':len(rows),'all_arrays_checked':True,'all_checksums_checked':True,
                                  'source':'records and embedded NPZ metadata','split_counts':splits})
    print('SEALED',len(rows),splits,'max tokens',max(r['tokens'] for r in rows),
          'min shape coverage',min(r['shape_coverage'] for r in rows),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/dit_vae_v2_parts64.json')
    ap.add_argument('--cache-dir',type=Path)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true');mode.add_argument('--worker',type=int);mode.add_argument('--merge',action='store_true')
    ap.add_argument('--workers',type=int,default=24);ap.add_argument('--limit',type=int)
    ap.add_argument('--ids',nargs='+',help='Explicit implementation/stress pilot IDs; never a production cache')
    args=ap.parse_args();cfg,vcfg=config(args.config)
    if args.cache_dir:
        from vae_v2.config import storage_path
        cfg['cache_dir']=str(storage_path(args.cache_dir,vcfg))
    if args.workers<1:ap.error('workers must be positive')
    if args.plan:prepare_plan(args,cfg,vcfg)
    elif args.merge:seal(args,cfg)
    else:encode(args,cfg)


if __name__=='__main__':main()
