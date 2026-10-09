"""Prepare the exact validation-selected VAE/DiT release; no Hub mutations.

All staged weights and generated publication metadata stay under trellisseg_run.
Only explicit files are staged. This is not a recursive workspace upload.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import atomic_json,digest,storage_path
from utils.dit_vae_v2 import config,load_cache,identity,file_sha256

PREFIX='releases/parts64_20261009'
REPO='GilgameshYX/Trellis_segPart'


def require(ok,message):
    if not ok:raise ValueError(message)


def prepare():
    import torch
    cfg,vcfg=config(ROOT/'configs/dit_vae_v2_parts64_b_smallparts.json')
    cache_plan,manifest=load_cache(cfg)
    selected=json.loads((Path(cfg['output_dir'])/'selection.json').read_text())
    vae_selected=json.loads(Path(cfg['vae_selection']).read_text())
    expected={'vae':(vae_selected,9500,'regular'),'dit':(selected,5000,'ema')}
    val_ids={r['instance'] for r in manifest['samples'] if r['split']=='validation'}
    for label,(selection,step,kind) in expected.items():
        require(selection['step']==step and selection['kind']==kind,label+' selection changed')
        require(selection['split']=='validation' and not selection['test_used_for_selection'],label+' selection used test')
        require(set(selection['ids'])==val_ids and len(selection['ids'])==128,label+' validation membership changed')
    result_dir=ROOT/'results/dit_b_smallparts_test_20261009'
    result=json.loads((result_dir/'metrics.json').read_text())
    eval_plan=json.loads((result_dir/'plan.json').read_text())
    paired=json.loads((result_dir/'paired_comparison.json').read_text())
    require(result['checkpoint']==identity(selected['checkpoint']), 'Tested DiT differs from selected checkpoint')
    require(result['vae_checkpoint']==cache_plan['vae_checkpoint']==identity(vae_selected['checkpoint']),'Tested VAE differs')
    require(result['cache_digest']==digest(manifest)==eval_plan['cache_digest'],'Test cache differs')
    require(len(result['ids'])==len(set(result['ids']))==128 and not result['test_used_for_selection'],'Invalid test cohort')
    release_root=storage_path(Path(vcfg['storage_root'])/'releases/parts64_20261009',vcfg)
    stage=release_root/'upload'
    require(not stage.exists() or not any(stage.iterdir()),'Staging exists; inspect rather than overwriting')
    stage.mkdir(parents=True,exist_ok=True)
    release_dir=stage/PREFIX
    for name in ('vae','dit','configs'):(release_dir/name).mkdir(parents=True,exist_ok=True)
    states={};checkpoint_files={}
    for label,(selection,step,kind) in expected.items():
        source=Path(selection['checkpoint']);before=identity(source)
        print('Checking checkpoint',label,source.name,flush=True)
        state=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
        require(state['step']==step and state['kind']==kind,'Checkpoint step/kind differs')
        require('optimizer' not in state and 'model' in state,'Expected inference model checkpoint')
        require(not state.get('smoke_only',state.get('metadata',{}).get('smoke_only',False)),'Smoke checkpoint')
        model=state['model']
        require(all(torch.is_tensor(v) for v in model.values()),'Non-tensor model state')
        if label=='vae':
            require(state['manifest_digest']==cache_plan['source_manifest_digest'],'VAE source mismatch')
            require(state['config']['num_classes']==64 and any(k.startswith('surface.') for k in model),'Incomplete VAE v2')
        else:
            m=state['metadata']
            require(m['vae_sha256']==cache_plan['vae_sha256'] and m['cache_digest']==digest(manifest),'DiT/VAE cache mismatch')
            require(m['base_digest']==digest(eval_plan['base_config']),'DiT architecture mismatch')
            require(tuple(model['input_layer.weight'].shape)==(1536,64),'Wrong DiT input dimensions')
            require(tuple(model['out_layer.weight'].shape)==(32,1536),'Wrong DiT output dimensions')
        sha=file_sha256(source)
        require(identity(source)==before,'Checkpoint changed while hashing')
        if label=='vae':require(sha==cache_plan['vae_sha256'],'VAE SHA256 differs from tested artifact')
        # Same-filesystem immutable input link: no checkpoint rewrite/copy or dtype conversion.
        target=release_dir/label/source.name
        os.link(source,target)
        checkpoint_files[label]={'path':str(target.relative_to(stage)),'filename':source.name,
            'bytes':source.stat().st_size,'sha256':sha,'step':step,'kind':kind,
            'tensor_count':len(model),'tensor_elements':sum(x.numel() for x in model.values()),
            'tensor_dtypes':sorted({str(x.dtype) for x in model.values()}),'optimizer_included':False}
        states[label]=state
        print('CHECKPOINT_READY',label,sha,flush=True)
    # Runtime-only config: no local paths, dataset secrets, training initializers, or optimizer settings.
    vae_cfg={k:copy.deepcopy(states['vae']['config'][k]) for k in ('num_classes','model','geometry')}
    vae_cfg['vae_config']='configs/vae_backbone.json'
    vae_cfg['model']['initialize_backbone']=False
    vae_cfg['model']['gradient_checkpointing']=False
    backbone={'models':copy.deepcopy(json.loads(Path(vcfg['vae_config']).read_text())['models'])}
    for name in ('encoder','decoder'):
        args=backbone['models'][name]['args'];args['use_fp16']=False
        args['latent_channels']=32
        args['in_channels' if name=='encoder' else 'out_channels']=64
        args['block_args']=[{'use_checkpoint':False} for _ in args['num_blocks']]
    dit_args=copy.deepcopy(eval_plan['base_config']['models']['denoiser']['args'])
    dit_args['use_checkpoint']=False
    dit_cfg={'name':'SegSLatFlowModel','args':dit_args}
    for filename,value in [('configs/vae_v2.json',vae_cfg),('configs/vae_backbone.json',backbone),
                            ('configs/dit_model.json',dit_cfg),('normalization.json',manifest['normalization'])]:
        require('/apdcephfs' not in json.dumps(value) and '/cos_' not in json.dumps(value),'Nonportable config')
        atomic_json(release_dir/filename,value)
    eval_summary={'test_count':128,'validation_count':128,'train_count':22633,
        'sampling_steps':20,'cfg':1,'seed_count_per_object':1,'seed_policy':eval_plan['seed_policy'],
        'seed':cfg['seed'],'condition':result['condition'],'postprocessing':False,'label_matching':False,
        'test_used_for_selection':False,'cache_digest':result['cache_digest'],
        'before_model':'dit_vae_v2_parts64/regular_step004500.pt','after_model':checkpoint_files['dit']['path'],
        'dit_summary':result['summary'],'vae_reconstruction_summary':result['oracle_summary'],
        'paired_metrics':paired['metrics'],'groups_by_parts':paired['groups_by_parts'],
        'improved_over_1pp':paired['improved_over_1pp'],'regressed_over_1pp':paired['regressed_over_1pp'],
        'regressed_over_5pp':paired['regressed_over_5pp'],
        'validation_max_metric_abs_diff':max(r['max_metric_abs_diff'] for r in result['validation_parity']),
        'limitations':['Development regression, not a new blind test','GT-derived conditioning, not external API end-to-end',
                       'Single seed per object','Combined training changes; no single-factor causal attribution']}
    atomic_json(release_dir/'evaluation_summary.json',eval_summary)
    model_card=ROOT/'docs/hf_readme_parts64_20261009.md'
    require('85.05%' in model_card.read_text() and checkpoint_files['vae']['path'] in model_card.read_text(),'README/version mismatch')
    # Copy bytes exactly from the apply_patch-authored model card.
    import shutil
    shutil.copyfile(model_card,stage/'README.md')
    files=[]
    for p in sorted(stage.rglob('*')):
        if p.is_file():
            known=next((v for v in checkpoint_files.values() if v['path']==str(p.relative_to(stage))),None)
            files.append({'path':str(p.relative_to(stage)),'bytes':p.stat().st_size,
                          'sha256':known['sha256'] if known else file_sha256(p)})
    release={'release':'parts64_20261009','created_at':datetime.now().astimezone().isoformat(),'repo_id':REPO,
        'checkpoints':checkpoint_files,'files':files,'num_classes_including_unknown':64,'latent_channels':32,
        'unknown_id':0,'nonzero_part_ids':[1,63],'surface_head_included':True,
        'selection_policy':'Both checkpoints selected on128validation objects; exact pair evaluated on128test objects',
        'latest_vae_training_step':10000,'recommended_vae_step':9500,'dit_additional_training_steps':5000,
        'source_manifest_digest':cache_plan['source_manifest_digest'],'dit_cache_digest':digest(manifest),
        'normalization_policy':'Original corrected train-only statistics, retained for B weight continuation',
        'configs':{'vae':'configs/vae_v2.json','vae_backbone':'configs/vae_backbone.json','dit':'configs/dit_model.json'},
        'checkpoint_format':'Original PyTorch dictionary; load model entry; unchanged bytes; no optimizer',
        'local_paths_in_checkpoint_metadata':'Historical provenance only; use portable companion configs instead',
        'required_code':['vae_v2.model.SegVaeV2','vae_v2.geometry.SurfaceQueries','models.seg_flow_model.SegSLatFlowModel'],
        'upstream_dependencies':['TRELLIS.2','PyTorch CUDA','flex_gemm','flash_attn'],
        'evaluation':'evaluation_summary.json','no_training_data_or_optimizer_uploaded':True}
    atomic_json(release_dir/'release.json',release)
    allfiles=files+[{'path':PREFIX+'/release.json','bytes':(release_dir/'release.json').stat().st_size,
                    'sha256':file_sha256(release_dir/'release.json')}]
    atomic_json(release_root/'upload_plan.json',{'repo_id':REPO,'stage':str(stage),'files':allfiles,
        'source_identities':{k:identity(s['checkpoint']) for k,(s,_,_) in expected.items()},'status':'prepared_not_uploaded'})
    print('STAGED',len(allfiles),'files',sum(x['bytes'] for x in allfiles),'bytes',stage,flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
