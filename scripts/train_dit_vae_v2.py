"""DiT in the new frozen parts64 VAE space, with train-only normalization and surface supervision."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'TRELLIS.2'))
from vae_v2.config import digest,atomic_json,storage_path
from utils.decoded_dit_runtime import FeatureOutputModel,master_action,atomic_torch_save,distributed_topology
from utils.dit_vae_v2 import (config,load_cache,identity,load_latents,normalize_tensors,conditions,
                             frozen_aux_losses,unnormalize)
from vae_v2.data import load_sample,to_device,training_order
from vae_v2.geometry import training_queries,SurfaceQueries
from vae_v2.evaluation import classification,surface_diagnostics,summarize,selection_score
from trainers.decoded_seg_losses import recover_clean_latent
from utils.dit_training_b import (object_order,training_permutation,reencode_labels,
                                  checked_fork_state,selected_row_index,BINS)


def load_geometry(row):
    path=Path(row['source_path']);s=path.stat()
    if (s.st_size,s.st_mtime_ns)!=(row['source_bytes'],row['source_mtime_ns']):raise ValueError('Geometry cache changed')
    return load_sample(path)


def sparse(data,feats,device):
    from trellis2.modules.sparse.basic import SparseTensor
    coords=torch.as_tensor(data['latent_coords'],dtype=torch.int32,device=device)
    return SparseTensor(feats,torch.cat([torch.zeros_like(coords[:,:1]),coords],1))


def ema_update(ema,model,rate):
    with torch.no_grad():
        for k,v in model.state_dict().items():
            if v.is_floating_point():ema[k].lerp_(v,1-rate)
            else:ema[k].copy_(v)


@torch.no_grad()
def validate(model,vae,rows,cfg,stats,device,rank,world):
    model.eval();results=[]
    for row in rows[rank::world]:
        data=load_latents(row,cfg['cache_dir']);norm=normalize_tensors(data,stats,device)
        geometry=load_geometry(row)
        template=vae.geometry_template(torch.as_tensor(geometry['coords'],device=device))
        if not np.array_equal(template.coords[:,1:].cpu().numpy(),data['latent_coords']):raise ValueError('Validation coordinates mismatch')
        seed=cfg['seed']+int(row['instance'][:8],16)
        rng=torch.Generator(device=device).manual_seed(seed)
        feats=torch.randn(data['latent_feats'].shape,device=device,generator=rng)
        nsteps=cfg['train'].get('validation_sampling_steps',20)
        cond=conditions(template,data,norm,False)
        # Pure Gaussian start; target latent is used only for its known geometry shape.
        for i in range(nsteps):
            t=1-i/nsteps
            with torch.autocast('cuda',dtype=torch.bfloat16):
                velocity=model(template.replace(feats),torch.tensor([t*1000],device=device),**cond).feats.float()
            feats=feats-velocity/nsteps
        features,logits,coords=vae.decode_voxels(template.replace(unnormalize(feats,stats)))
        if not np.array_equal(coords[:,1:].cpu().numpy(),geometry['coords']):raise ValueError('Validation decoded coordinates mismatch')
        query=SurfaceQueries(geometry,vae.cfg['geometry']);faces=[]
        for start in range(0,len(geometry['face_labels']),1024):
            q=to_device(query.build(np.arange(start,min(start+1024,len(geometry['face_labels'])))),device)
            with torch.autocast('cuda',dtype=torch.bfloat16):_,score=vae.decode_surface(features,logits,q)
            faces.append(score.argmax(1).cpu().numpy())
        faces=np.concatenate(faces);voxels=logits.argmax(1).cpu().numpy()
        results.append({'instance':row['instance'],'parts':row['parts'],
            'surface':classification(geometry['face_labels'],faces,geometry['face_area']),
            'projection':classification(geometry['face_labels'],faces,geometry['face_pixels']),
            'voxel':classification(geometry['voxel_labels'],voxels,np.ones(len(voxels))),
            'query_fallback_fraction':None,**surface_diagnostics(geometry,faces)})
    if world>1:
        parts=[None]*world;dist.all_gather_object(parts,results);results=[r for p in parts for r in p]
    return {'summary':summarize(results),'samples':sorted(results,key=lambda r:r['instance']),
            'part_groups':{f'{lo}-{hi}':{'count':sum(lo<=r['parts']<=hi for r in results),
                 'summary':summarize([r for r in results if lo<=r['parts']<=hi])}
                 for lo,hi in BINS if any(lo<=r['parts']<=hi for r in results)}}


def train(args,cfg,vcfg,plan,manifest,rank,local,world):
    from vae_v2.model import SegVaeV2
    from models.seg_flow_model import SegSLatFlowModel
    device=torch.device(f'cuda:{local}');torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(seconds=1200))
    output=Path(cfg['output_dir']);logs=Path(cfg['log_dir'])
    def setup():
        if not args.resume and output.exists() and any(output.iterdir()):raise ValueError('Output exists; use resume or new directory')
        output.mkdir(parents=True,exist_ok=True);(output/'validation').mkdir(exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    master_action(setup,rank,world>1)
    train_rows=[r for r in manifest['samples'] if r['split']=='train']
    val_rows=[r for r in manifest['samples'] if r['split']=='validation']
    if args.smoke_ids:
        requested=set(args.smoke_ids)
        if len(requested)!=len(args.smoke_ids) or not requested<={r['instance'] for r in train_rows}:
            raise ValueError('Smoke IDs must be unique members of TRAIN only')
        train_rows=[r for r in train_rows if r['instance'] in requested]
        val_rows=[]
    if not train_rows or (not val_rows and not args.smoke):raise ValueError('Missing training/validation')
    base=json.loads(Path(cfg['base_config']).read_text())
    metadata={'config':cfg,'cache_digest':digest(manifest),'world_size':world,'smoke_only':args.smoke,
              'vae_sha256':plan['vae_sha256'],'base_digest':digest(base),'dit_init':identity(cfg['dit_init'])}
    if args.smoke_ids:metadata['smoke_train_ids']=sorted(args.smoke_ids)
    checkpoint=torch.load(output/'trainer_latest.pt',map_location='cpu',weights_only=False,mmap=True) if args.resume else None
    if checkpoint:
        a=json.loads(json.dumps(checkpoint['metadata']));b=json.loads(json.dumps(metadata))
        for m in (a,b):m['config']['train'].pop('steps',None)
        if a!=b:raise ValueError('Resume requires identical data/VAE/model/settings/world size')
    print(f'[rank {rank}] loading frozen VAE',flush=True)
    if identity(plan['vae_checkpoint']['path'])!=plan['vae_checkpoint']:raise ValueError('VAE file changed')
    frozen=torch.load(plan['vae_checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
    if frozen['manifest_digest']!=plan['source_manifest_digest']:raise ValueError('VAE provenance changed')
    vae=SegVaeV2(frozen['config'],initialize=False);vae.load_state_dict(frozen['model']);del frozen
    vae=vae.to(device).eval().requires_grad_(False)
    if args.smoke and cfg['train'].get('label_augmentation'):
        probe=train_rows[rank%len(train_rows)];cached=load_latents(probe,cfg['cache_dir'])
        encoded,_=reencode_labels(vae,load_geometry(probe),cached,np.arange(64),device)
        errors={k:float(np.max(np.abs(encoded[k]-cached[k]))) for k in ('latent_feats','mv_feats')}
        for key in errors:
            np.testing.assert_allclose(encoded[key],cached[key],rtol=1e-5,atol=1e-5,
                                       err_msg='Identity online encoding differs from original cache')
        atomic_json(logs/f'online_parity_rank{rank:04d}.json',dict(instance=probe['instance'],max_errors=errors,
                     geometry_unchanged=True,vae_parameter_grads_absent=all(p.grad is None for p in vae.parameters())))
        print(f'[rank {rank}] online identity encoding parity: {errors}',flush=True)
        del encoded,cached
    print(f'[rank {rank}] loading DiT initialization',flush=True)
    torch.manual_seed(cfg['seed'])
    model=SegSLatFlowModel(**dict(base['models']['denoiser']['args'],use_checkpoint=True))
    if checkpoint:model.load_state_dict(checkpoint['model'])
    else:
        state=torch.load(cfg['dit_init'],map_location='cpu',weights_only=True,mmap=True)
        weights,initialization=checked_fork_state(state,cfg,manifest,plan['vae_sha256'],digest(base))
        model.load_state_dict(weights)
        print(f'[rank {rank}] initialization: {initialization}',flush=True)
        master_action(lambda:atomic_json(output/'initialization.json',dict(initialization,source=identity(cfg['dit_init']))),rank,world>1)
        del state,weights
    model=model.to(device);model.dtype=torch.bfloat16;torch.cuda.synchronize(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['train']['lr'],betas=(.9,.95),weight_decay=.01,foreach=False)
    if checkpoint:optimizer.load_state_dict(checkpoint['optimizer'])
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    if checkpoint:
        for k,v in ema.items():v.copy_(checkpoint['ema'][k])
    forward=FeatureOutputModel(model)
    if world>1:forward=DistributedDataParallel(forward,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
    print(f'[rank {rank}] DDP ready',flush=True)
    noise_rng=torch.Generator(device=device).manual_seed(cfg['seed']+100003*rank)
    rng=np.random.default_rng(cfg['seed']+1+100003*rank)
    start_step=1;gradient_checked=False
    if checkpoint:
        start_step=checkpoint['step']+1;s=checkpoint['rank_rng'][rank]
        noise_rng.set_state(s['noise']);rng.bit_generator.state=s['numpy']
        torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda'],device)
        gradient_checked=s['gradient_checked'];del checkpoint
        print(f'[rank {rank}] Restored model/EMA/optimizer/RNG; next step {start_step}',flush=True)
    if start_step>cfg['train']['steps']:raise ValueError('Requested steps already complete')
    master_action(lambda:atomic_json(output/'manifest.json',dict(metadata,train_count=len(train_rows),validation_count=len(val_rows),global_batch=world*cfg['train']['accumulate'],start_step=start_step,requested_steps=cfg['train']['steps'])),rank,world>1)
    stats=manifest['normalization'];initial_state={k:v.detach().cpu().clone() for k,v in list(model.state_dict().items())[:1]} if args.smoke else None
    best=json.loads((output/'selection.json').read_text()) if args.resume and (output/'selection.json').exists() else None
    t0=time.monotonic();epoch_old=None;order=None
    with (logs/f'train_rank{rank:04d}.jsonl').open('a',buffering=1) as log:
        for step in range(start_step,cfg['train']['steps']+1):
            model.train();optimizer.zero_grad(set_to_none=True);batch=[]
            for g in optimizer.param_groups:g['lr']=cfg['train']['lr']*min(1.,step/cfg['train']['warmup_steps'])
            for micro in range(cfg['train']['accumulate']):
                idx=((step-1)*cfg['train']['accumulate']+micro)*world+rank
                epoch,offset=divmod(idx,len(train_rows))
                if not args.smoke_ids and epoch!=epoch_old:
                    order=object_order(train_rows,cfg['seed'],epoch,cfg['train'].get('object_sampling'));epoch_old=epoch
                # Smoke visits every selected geometry, including longest samples,
                # independently of production oversampling; never selects a model.
                row=train_rows[selected_row_index(train_rows,order,offset,idx,bool(args.smoke_ids))]
                data=load_latents(row,cfg['cache_dir']);geometry=None
                mapping=training_permutation(cfg['seed'],idx,cfg['train'].get('label_augmentation'),row['split'])
                if args.smoke and cfg['train'].get('label_augmentation'):
                    from vae_v2.augmentation import label_permutation
                    mapping=label_permutation(64,cfg['seed'],idx) if (step+rank)%2 else None
                augmentation_seconds=0.
                if mapping is not None:
                    augmentation_start=time.monotonic()
                    geometry=load_geometry(row)
                    data,geometry=reencode_labels(vae,geometry,data,mapping,device)
                    augmentation_seconds=time.monotonic()-augmentation_start
                norm=normalize_tensors(data,stats,device)
                clean=norm['latent_feats'];template=sparse(data,clean,device)
                noise=torch.randn(clean.shape,device=device,generator=noise_rng)
                t=.3 if args.smoke else float(rng.random());drop=False if args.smoke else bool(rng.random()<cfg['train']['condition_dropout'])
                sigma=cfg['train']['sigma_min'];noisy=(1-t)*clean+(sigma+(1-sigma)*t)*noise;target=(1-sigma)*noise-clean
                use_aux=t<=cfg['train']['aux_t_max'] and not drop
                query=None;aux_diagnostics={}
                if use_aux:
                    if geometry is None:geometry=load_geometry(row)
                    template=vae.geometry_template(torch.as_tensor(geometry['coords'],device=device))
                    if not np.array_equal(template.coords[:,1:].cpu().numpy(),data['latent_coords']):raise ValueError('Latent/decoder coordinate mismatch')
                    query=to_device(training_queries(geometry,vae.cfg['geometry'],cfg['train'],rng),device)
                with forward.no_sync() if world>1 and micro<cfg['train']['accumulate']-1 else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=forward(template.replace(noisy),torch.tensor([t*1000],device=device),**conditions(template,data,norm,drop))
                    flow=F.mse_loss(pred.float(),target);loss=flow;terms={};gate=0.
                    if use_aux:
                        estimated=recover_clean_latent(noisy,pred.float(),t,sigma)
                        terms=frozen_aux_losses(vae,template,estimated,data,geometry,query,stats,cfg['train'],aux_diagnostics)
                        gate=(1-t)**2*min(1.,step/cfg['train']['aux_warmup_steps'])
                        loss=loss+gate*sum(cfg['train']['aux_weights'][k]*v for k,v in terms.items())
                        if not gradient_checked:
                            grads={k:float(torch.autograd.grad(v,pred,retain_graph=True)[0].norm()) for k,v in terms.items()}
                            if not all(np.isfinite(v) for v in grads.values()):raise ValueError('Nonfinite auxiliary gradient')
                            if all(v>0 for v in grads.values()):
                                flow_grad=float(torch.autograd.grad(flow,pred,retain_graph=True)[0].norm())
                                aux_grad=float(torch.autograd.grad(gate*sum(cfg['train']['aux_weights'][k]*v for k,v in terms.items()),pred,retain_graph=True)[0].norm())
                                atomic_json(logs/f'gradient_rank{rank:04d}.json',dict(grads,flow_gradient=flow_grad,
                                    weighted_aux_gradient=aux_grad,measurement='gradient_wrt_predicted_velocity',step=step));gradient_checked=True
                                print(f'[rank {rank}] surface auxiliary gradients reach DiT: {grads}',flush=True)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    (loss/cfg['train']['accumulate']).backward()
                if any(p.grad is not None for p in vae.parameters()):raise ValueError('Frozen VAE received parameter gradients')
                batch.append({'instance':row['instance'],'parts':row['parts'],'sample_index':idx,
                    'label_permuted':mapping is not None,'permutation_sha256':digest(mapping.tolist()) if mapping is not None else None,
                    'augmentation_seconds':augmentation_seconds,'flow':float(flow.detach()),'loss':float(loss.detach()),
                    't':t,'drop':drop,'aux_gate':gate,**aux_diagnostics,**{k:float(v.detach()) for k,v in terms.items()}})
                del template,pred,loss,flow,terms,geometry,query,data,norm,clean,noise,noisy,target
                if use_aux:del estimated
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();ema_update(ema,model,cfg['train']['ema_rate'])
            log.write(json.dumps({'step':step,'rank':rank,'seconds':time.monotonic()-t0,'grad_norm':float(grad),'batch':batch,'max_gpu_gb':torch.cuda.max_memory_allocated(device)/1024**3})+'\n')
            if rank==0 and (step==start_step or step%10==0):print(f'step={step} flow={batch[-1]["flow"]:.5f} loss={batch[-1]["loss"]:.5f}',flush=True)
            save=step%cfg['train']['save_every']==0 or step==cfg['train']['steps']
            if save:
                local_rng={'noise':noise_rng.get_state().cpu(),'numpy':rng.bit_generator.state,
                           'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device).cpu(),'gradient_checked':gradient_checked}
                all_rng=[None]*world
                if world>1:dist.all_gather_object(all_rng,local_rng)
                else:all_rng[0]=local_rng
                def write():
                    regular={k:v.detach().cpu() for k,v in model.state_dict().items()};averaged={k:v.cpu() for k,v in ema.items()}
                    for kind,state in [('regular',regular),('ema',averaged)]:
                        atomic_torch_save(output/f'{kind}_step{step:06d}.pt',{'model':state,'step':step,'kind':kind,'metadata':metadata})
                    atomic_torch_save(output/'trainer_latest.pt',{'model':regular,'ema':averaged,'optimizer':optimizer.state_dict(),'rank_rng':all_rng,'step':step,'metadata':metadata})
                master_action(write,rank,world>1)
            if not args.smoke and (step%cfg['train']['validate_every']==0 or step==cfg['train']['steps']):
                original=None
                for kind in ('regular','ema'):
                    if kind=='ema':
                        original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        model.load_state_dict(ema)
                    result=validate(model,vae,val_rows,cfg,stats,device,rank,world)
                    record={'checkpoint':str(output/f'{kind}_step{step:06d}.pt'),'step':step,'kind':kind,
                            'split':'validation','ids':[r['instance'] for r in val_rows],'test_used_for_selection':False,
                            'sampling_steps':cfg['train'].get('validation_sampling_steps',20),**result}
                    if best is None or selection_score(record['summary'])>selection_score(best['summary']):best={k:v for k,v in record.items() if k!='samples'}
                    def write_validation():
                        atomic_json(output/'validation'/f'{kind}_{step:06d}.json',record);atomic_json(output/'selection.json',best)
                    master_action(write_validation,rank,world>1)
                    if kind=='ema':model.load_state_dict(original);del original
                if rank==0:print('VALIDATION',best['checkpoint'],best['summary']['surface'],flush=True)
    verified={'rank':rank,'completed_steps':cfg['train']['steps'],'gradient_checked':gradient_checked,
              'vae_parameter_grads_absent':all(p.grad is None for p in vae.parameters())}
    if args.smoke:
        verified['dit_weights_changed']=any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in initial_state.items())
        if not verified['dit_weights_changed'] or not gradient_checked:raise ValueError('Smoke did not update weights / verify gradients')
        # Exercise actual Gaussian-start generation through the frozen surface
        # decoder. This uses pilot TRAIN objects only, not validation selection.
        if args.resume:
            result=validate(model,vae,train_rows[:world],cfg,stats,device,rank,world)
            master_action(lambda:atomic_json(logs/'smoke_generation.json',
                {'smoke_only':True,'split':'pilot_train','used_for_selection':False,
                 'sampling_steps':cfg['train'].get('validation_sampling_steps',20),**result}),rank,world>1)
            verified['gaussian_generation_evaluation_passed']=True
    atomic_json(logs/f'complete_rank{rank:04d}.json',verified)
    print(f'[rank {rank}] COMPLETE {verified}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/dit_vae_v2_parts64.json')
    ap.add_argument('--cache-dir',type=Path);ap.add_argument('--output-dir',type=Path);ap.add_argument('--log-dir',type=Path)
    ap.add_argument('--steps',type=int);ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--smoke-ids',nargs='+',help='Audited full-cache TRAIN IDs for implementation/stress tests only')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();cfg,vcfg=config(args.config)
    for arg,k in [(args.cache_dir,'cache_dir'),(args.output_dir,'output_dir')]:
        if arg:cfg[k]=str(storage_path(arg,vcfg))
    if args.log_dir:cfg['log_dir']=str(args.log_dir.resolve())
    if args.steps:cfg['train']['steps']=args.steps
    if args.smoke_ids and not args.smoke:ap.error('--smoke-ids requires --smoke')
    if args.smoke:
        if cfg['train']['steps']>3:ap.error('Smoke must run <=3 total steps')
        if args.smoke_ids:plan,manifest=load_cache(cfg)
        else:
            root=Path(cfg['cache_dir']);plan=json.loads((root/'plan.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
            if not manifest['pilot_only'] or manifest['status']!='complete' or manifest['plan_digest']!=digest(plan):raise ValueError('Invalid pilot cache')
    else:plan,manifest=load_cache(cfg)
    if args.check_only:print(json.dumps({'config':cfg,'counts':manifest['split_counts'],'cache_digest':digest(manifest),'vae_sha256':plan['vae_sha256']},indent=2));return
    rank,local,world=distributed_topology()
    try:train(args,cfg,vcfg,plan,manifest,rank,local,world)
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
        if world>1:os._exit(1)
        raise
    else:
        if world>1 and dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
