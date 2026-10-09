"""Joint VAE v2 training with deterministic warmup, DDP, EMA, and validation selection."""
import argparse
from contextlib import nullcontext
import copy
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
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import load_config, digest, atomic_json, storage_path, default_run_dir, resume_metadata_compatible, permutation_resume_compatible
from vae_v2.augmentation import POLICY, label_permutation, permute_sample
from vae_v2.data import load_manifest, load_sample, voxel_tensors, to_device, training_order, smoke_manifest
from vae_v2.geometry import training_queries
from vae_v2.losses import reconstruction_losses, kl_weight
from vae_v2.evaluation import reconstruct, summarize, selection_score
from utils.decoded_dit_runtime import distributed_topology, master_action, atomic_torch_save


def validate(model, cfg, rows, device, rank, world):
    results = []
    for row in rows[rank::world]:
        sample = load_sample(Path(cfg['prepared_dir'])/row['path'])
        metric, _, _ = reconstruct(model,sample,cfg,device)
        results.append(dict(instance=row['instance'],**metric))
    if world>1:
        shards=[None]*world
        dist.all_gather_object(shards,results)
        results=[r for shard in shards for r in shard]
    return {'summary':summarize(results),'samples':sorted(results,key=lambda r:r['instance'])}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/seg_vae_v2.json')
    ap.add_argument('--output_dir',type=Path,help='Defaults to storage_root/checkpoints/<config name>')
    ap.add_argument('--log_dir',type=Path,help='Defaults to project logs/<run name>')
    resume_mode=ap.add_mutually_exclusive_group()
    resume_mode.add_argument('--resume',action='store_true')
    resume_mode.add_argument('--resume_from',type=Path,help='Fork full trainer state into a fresh directory; only permits enabling label permutation')
    ap.add_argument('--manifest_workers',type=int,default=1,help='Parallel file identity checks; no checks skipped')
    ap.add_argument('--check_only',action='store_true')
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--steps',type=int,help='Total optimizer steps, including resumed steps')
    ap.add_argument('--smoke',action='store_true',help='Explicit integration run on one prepared train/validation object; not a release checkpoint')
    ap.add_argument('--resume_cache_dir',type=Path,help='Optional staged checkpoint directory, under storage_root')
    args=ap.parse_args()
    if args.manifest_workers < 1:ap.error('manifest_workers must be positive')
    cfg=load_config(args.config)
    output = storage_path(args.output_dir, cfg) if args.output_dir else default_run_dir(args.config, cfg)
    log_dir = (args.log_dir or ROOT/'logs'/output.name).resolve()
    if args.resume_cache_dir:
        args.resume_cache_dir = storage_path(args.resume_cache_dir, cfg)
    if args.resume_from:
        args.resume_from=storage_path(args.resume_from,cfg)
        if output==args.resume_from.parent:
            ap.error('resume_from requires a new output directory')
    if args.steps is not None and args.steps < 1:
        ap.error('steps must be positive')
    if args.steps is not None:
        cfg['train']['steps']=args.steps
    if args.smoke and (args.steps is None or args.steps>2):
        ap.error('--smoke requires --steps 1 or 2')
    rank,local,world=distributed_topology()
    if not args.check_only:
        print(f'[rank {rank}/{world}] Checking prepared manifest: {cfg["prepared_dir"]}',flush=True)
    manifest=smoke_manifest(cfg) if args.smoke else load_manifest(cfg,verify_workers=args.manifest_workers)
    train_rows=[r for r in manifest['samples'] if r['split']=='train']
    val_rows=[r for r in manifest['samples'] if r['split']=='validation'][:cfg['train']['validation_limit']]
    if not train_rows or not val_rows:
        raise ValueError('Both training and validation cohorts required')
    if not args.check_only:
        print(f'[rank {rank}/{world}] Manifest valid: train={len(train_rows)}, validation={len(val_rows)}',flush=True)
    metadata={'config':cfg,'manifest_digest':digest(manifest),'world_size':world,'smoke_only':args.smoke,
              'global_batch':world*cfg['train']['accumulate'],'validation_ids':[r['instance'] for r in val_rows]}
    if args.check_only:
        print(json.dumps({'train':len(train_rows),'validation':len(val_rows),'metadata':metadata},indent=2));return
    device=torch.device(f'cuda:{local}' if world>1 else args.device)
    torch.cuda.set_device(device)
    if world>1:
        dist.init_process_group('nccl',timeout=timedelta(seconds=1800))
    print(f'[rank {rank}/{world}] Distributed initialized on {device}',flush=True)
    try:
        def setup():
            if not args.resume and output.exists() and any(output.iterdir()):
                raise ValueError('Use a new run directory or --resume')
            output.mkdir(parents=True,exist_ok=True)
            (output/'validation').mkdir(exist_ok=True)
            log_dir.mkdir(parents=True,exist_ok=True)
        master_action(setup,rank,world>1)
        resume_path=args.resume_from or output/'trainer_latest.pt'
        source_identity=None
        if args.resume or args.resume_from:
            source_stat=resume_path.stat()
            source_identity={'path':str(resume_path),'bytes':source_stat.st_size,'mtime_ns':source_stat.st_mtime_ns}
        if (args.resume or args.resume_from) and args.resume_cache_dir:
            from vae_v2.checkpoint import stage_checkpoint
            if rank==0:print('Staging optimizer checkpoint into local cache',flush=True)
            resume_path=stage_checkpoint(resume_path,args.resume_cache_dir)
        # Private mmap avoids eagerly copying the 13GB checkpoint into each rank.
        # Writes to optimizer step tensors are copy-on-write, not source-file edits.
        checkpoint=torch.load(resume_path,map_location='cpu',weights_only=False,mmap=True) if (args.resume or args.resume_from) else None
        continuation=checkpoint.get('continuation') if checkpoint else None
        if checkpoint:
            compatible=permutation_resume_compatible if args.resume_from else resume_metadata_compatible
            if not compatible(checkpoint['metadata'],metadata):
                raise ValueError('Resume config, cohort, or world size changed')
            if len(checkpoint['rank_rng'])!=world:
                raise ValueError('Checkpoint RNG rank count differs')
            if args.resume_from:
                continuation={'source':source_identity,'source_step':checkpoint['step'],
                              'parent_metadata_digest':digest(checkpoint['metadata']),
                              'change':{'train.label_permutation':POLICY},
                              'restored':['model','optimizer','ema','rank_rng','step']}
            if rank==0:print(f'Resuming optimizer/EMA/RNG from step {checkpoint["step"]}',flush=True)
        torch.manual_seed(cfg['seed'])
        from vae_v2.model import SegVaeV2
        print(f'[rank {rank}/{world}] Loading VAE backbone on CPU',flush=True)
        model=SegVaeV2(cfg,initialize=checkpoint is None)
        print(f'[rank {rank}/{world}] Moving VAE to {device}',flush=True)
        model=model.to(device)
        torch.cuda.synchronize(device)
        print(f'[rank {rank}/{world}] VAE on GPU',flush=True)
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
        groups=[{'params':list(model.encoder.parameters())+list(model.decoder.parameters()),'lr':cfg['train']['lr']}]
        if model.surface is not None:
            groups.append({'params':model.surface.parameters(),'lr':cfg['train']['surface_lr']})
        optimizer=torch.optim.AdamW(groups,weight_decay=cfg['train']['weight_decay'])
        base_lrs=[g['lr'] for g in optimizer.param_groups]
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        torch.cuda.synchronize(device)
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer']);ema.load_state_dict(checkpoint['ema'])
            torch.cuda.synchronize(device)
            optimizer_steps=[float(s['step']) for s in optimizer.state.values() if 'step' in s]
            restore={'rank':rank,'source_step':checkpoint['step'],'next_step':checkpoint['step']+1,
                     'optimizer_states':len(optimizer.state),'optimizer_step_min':min(optimizer_steps),
                     'optimizer_step_max':max(optimizer_steps),'lrs':[g['lr'] for g in optimizer.param_groups],
                     'ema_restored':True,'rng_restored':False}
        print(f'[rank {rank}/{world}] Model/EMA/optimizer ready; entering DDP initialization',flush=True)
        forward=DistributedDataParallel(model,device_ids=[device.index],find_unused_parameters=True) if world>1 else model
        print(f'[rank {rank}/{world}] DDP ready',flush=True)
        rng=np.random.default_rng(cfg['seed']+rank*100003)
        start_step=1
        if checkpoint:
            start_step=checkpoint['step']+1
            saved_rng=checkpoint['rank_rng'][rank]
            rng.bit_generator.state=saved_rng['numpy']
            torch.set_rng_state(saved_rng['torch'])
            torch.cuda.set_rng_state(saved_rng['cuda'],device)
            restore['rng_restored']=True
            atomic_json(log_dir/f'resume_rank{rank:04d}.json',restore)
        else:
            torch.manual_seed(cfg['seed']+rank*100003)
        if start_step>cfg['train']['steps']:
            raise ValueError('Requested final step already completed')
        del checkpoint
        if continuation:
            master_action(lambda:atomic_json(output/'continuation.json',continuation),rank,world>1)
        if not args.resume and not args.resume_from:
            master_action(lambda:atomic_json(output/'initialization.json',model.initialization),rank,world>1)
        master_action(lambda:atomic_json(output/'manifest.json',metadata),rank,world>1)
        log=(log_dir/f'train_rank{rank:04d}.jsonl').open('a',buffering=1)
        best=json.loads((output/'selection.json').read_text()) if args.resume and (output/'selection.json').exists() else None
        if best and Path(best['checkpoint']).parent != output:
            relocated=output/Path(best['checkpoint']).name
            if not relocated.is_file() or best['manifest_digest'] != metadata['manifest_digest']:
                raise ValueError('Selected checkpoint could not be relocated safely')
            best['checkpoint']=str(relocated)
        started=time.monotonic()
        cached_epoch=None;order=None
        try:
            for step in range(start_step,cfg['train']['steps']+1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                warm=min(1.,step/max(1,cfg['train']['warmup_steps']))
                for group,lr in zip(optimizer.param_groups,base_lrs):group['lr']=lr*warm
                metrics=[];augmentation=[]
                for micro in range(cfg['train']['accumulate']):
                    index=((step-1)*cfg['train']['accumulate']+micro)*world+rank
                    epoch,offset=divmod(index,len(train_rows))
                    if epoch!=cached_epoch:
                        order=training_order(train_rows,cfg['seed'],epoch);cached_epoch=epoch
                    row=train_rows[int(order[offset])]
                    sample=load_sample(Path(cfg['prepared_dir'])/row['path'])
                    if cfg['train'].get('label_permutation')==POLICY:
                        mapping=label_permutation(cfg.get('num_classes',20),cfg['seed'],index)
                        present=np.unique(sample['face_labels'])
                        augmentation.append({'instance':row['instance'],'sample_index':index,
                                             'from':present.tolist(),'to':mapping[present].tolist()})
                        sample=permute_sample(sample,mapping)
                    use_surface=any(cfg['loss'][k]>0 for k in ('point_ce','surface_ce','surface_dice','reprojection','consistency','boundary'))
                    q=training_queries(sample,cfg['geometry'],cfg['train'],rng) if use_surface else None
                    query=to_device(q,device) if q is not None else None
                    voxels=voxel_tensors(sample,device)
                    sync=forward.no_sync() if world>1 and micro<cfg['train']['accumulate']-1 else nullcontext()
                    with sync:
                        with (torch.autocast('cuda',dtype=torch.bfloat16) if cfg['train']['precision']=='bf16' else nullcontext()):
                            result=forward(voxels,query,sample_posterior=step>cfg['train']['deterministic_steps'])
                        loss,terms=reconstruction_losses(result,voxels,query,cfg['loss'],kl_weight(step,cfg['train']),
                                                         cfg['train']['balanced_voxel_ce'])
                        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss: {row["instance"]}')
                        (loss/cfg['train']['accumulate']).backward()
                    metrics.append({k:float(v.detach()) for k,v in terms.items()})
                    del result,loss,terms,voxels,query,q,sample
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['grad_clip'],error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,1-cfg['train']['ema_rate'])
                stats={k:sum(r[k] for r in metrics)/len(metrics) for k in metrics[0]}
                log.write(json.dumps({'step':step,'seconds':time.monotonic()-started,'loss':stats,'grad_norm':float(norm),
                                     'kl_weight':kl_weight(step,cfg['train']),
                                     **({'label_permutation':augmentation} if augmentation else {})})+'\n')
                if rank==0 and (step==start_step or step%10==0):
                    print(f'step={step} loss={stats["loss"]:.5f} ce={stats["voxel_ce"]:.5f} surface={stats["surface_ce"]:.5f}',flush=True)
                validate_now=step%cfg['train']['validate_every']==0 or step==cfg['train']['steps']
                save_now=step%cfg['train']['save_every']==0 or validate_now
                if save_now:
                    local_rng={'numpy':rng.bit_generator.state,'torch':torch.get_rng_state(),
                               'cuda':torch.cuda.get_rng_state(device).cpu()}
                    all_rng=[None]*world
                    if world>1:dist.all_gather_object(all_rng,local_rng)
                    else:all_rng[0]=local_rng
                    def save():
                        model_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
                        ema_state={k:v.detach().cpu() for k,v in ema.state_dict().items()}
                        for name,state in [('regular',model_state),('ema',ema_state)]:
                            atomic_torch_save(output/f'{name}_step{step:06d}.pt',{'model':state,'config':cfg,'step':step,
                                                                               'kind':name,'manifest_digest':metadata['manifest_digest'],
                                                                               'smoke_only':args.smoke})
                        atomic_torch_save(output/'trainer_latest.pt',{'model':model_state,'ema':ema_state,
                             'optimizer':optimizer.state_dict(),'rank_rng':all_rng,'step':step,'metadata':metadata,
                             **({'continuation':continuation} if continuation else {})})
                    master_action(save,rank,world>1)
                if validate_now:
                    for name,candidate in [('regular',model),('ema',ema)]:
                        result=validate(candidate,cfg,val_rows,device,rank,world)
                        path=output/f'{name}_step{step:06d}.pt'
                        selection={'checkpoint':str(path),'kind':name,'step':step,'split':'validation',
                                   'ids':[r['instance'] for r in val_rows],'summary':result['summary'],
                                   'rule':'surface mIoU; then small-part recall; then lower isolated-error area',
                                   'test_used_for_selection':False,'smoke_only':args.smoke,
                                   'manifest_digest':metadata['manifest_digest']}
                        if best is None or selection_score(result['summary'])>selection_score(best['summary']):best=selection
                        def save_validation():
                            atomic_json(output/'validation'/f'{name}_{step:06d}.json',dict(selection,samples=result['samples']))
                            atomic_json(output/'selection.json',best)
                        master_action(save_validation,rank,world>1)
                    if rank==0:print('Best validation:',best['checkpoint'],best['summary']['surface'],flush=True)
        finally:log.close()
    except BaseException:
        # Destroying NCCL while peers wait in another collective can hide the
        # original Python exception indefinitely. Let torchrun stop peer ranks.
        traceback.print_exc()
        sys.stdout.flush();sys.stderr.flush()
        if world>1:os._exit(1)
        raise
    else:
        if world>1 and dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
