"""One real-sample VAE v2 optimizer update and reconstruction/export smoke check."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import load_config,atomic_json,storage_path
from vae_v2.data import load_sample,validate_sample,voxel_tensors,to_device
from vae_v2.geometry import training_queries
from vae_v2.losses import reconstruction_losses
from vae_v2.evaluation import reconstruct,export_prediction


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/seg_vae_v2.json')
    ap.add_argument('--sample',type=Path,required=True)
    ap.add_argument('--output_dir',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--forward_only',action='store_true')
    args=ap.parse_args();cfg=load_config(args.config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Use an empty smoke directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(args.device);torch.manual_seed(cfg['seed'])
    from vae_v2.model import SegVaeV2
    model=SegVaeV2(cfg).to(args.device)
    sample=load_sample(args.sample);validate_sample(sample,cfg.get('num_classes',20))
    settings=dict(cfg['train'],query_faces=128,pair_count=64)
    query=to_device(training_queries(sample,cfg['geometry'],settings,np.random.default_rng(42)),args.device)
    voxels=voxel_tensors(sample,args.device)
    model.train()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['train']['precision']=='bf16'):
        output=model(voxels,query,sample_posterior=False)
    loss,terms=reconstruction_losses(output,voxels,query,cfg['loss'],0)
    if not torch.isfinite(loss):raise ValueError('Nonfinite smoke loss')
    gradients={}
    if not args.forward_only:
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,foreach=False)
        loss.backward()
        for name in ['encoder','decoder','surface']:
            module=getattr(model,name)
            if module is None:continue
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError(f'Invalid {name} gradients')
            gradients[name]=sum(float(g.float().square().sum()) for g in grads)**.5
            if gradients[name]<=0:raise ValueError(f'Zero {name} gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del optimizer
    losses={k:float(v.detach()) for k,v in terms.items()}
    del output,loss,terms,query,voxels
    torch.cuda.empty_cache()
    # Check that decoder feature extraction and the label-free spatial template
    # preserve the backbone output, instead of leaking encoder label features.
    model.eval()
    with torch.no_grad():
        voxels=voxel_tensors(sample,args.device)
        latent,_,_=model.encode(voxels,False)
        _,custom,coords=model.decode_voxels(latent)
        native=model.decoder(latent).feats.float()
        if not torch.allclose(custom,native,atol=1e-5,rtol=1e-5):raise ValueError('Custom decoder differs from native decoder')
        template=model.geometry_template(voxels['coords'])
        if not torch.equal(template.coords,latent.coords):raise ValueError('Label-free template coordinates differ')
        _,blind,_=model.decode_voxels(template.replace(latent.feats))
        parity=float((blind-custom).abs().max())
        if not torch.allclose(blind,custom,atol=1e-4,rtol=1e-4):raise ValueError('Geometry-only decode depends on GT cache')
        del voxels,latent,custom,native,template,blind
    metrics,faces,voxel=reconstruct(model,sample,cfg,args.device)
    uid=args.sample.stem
    prediction_dir=storage_path(Path(cfg['storage_root'])/'predictions'/args.output_dir.name,cfg)
    export_prediction(args.output_dir,uid,sample,faces,voxel,Path(cfg['source_dir'])/'samples'/f'{uid}.npz',prediction_dir)
    report={'instance':uid,'config':cfg,'losses':losses,'gradient_norms':gradients,'metrics':metrics,
            'optimizer_steps':0 if args.forward_only else 1,'note':'Implementation smoke only, not a trained v2 quality result',
            'max_gpu_memory_gb':torch.cuda.max_memory_allocated()/1024**3,
            'native_decoder_parity':True,'geometry_only_template_max_logit_difference':parity}
    report['num_classes']=cfg.get('num_classes',20)
    report['latent_channels']=cfg['model']['latent_channels']
    report['initialization']=model.initialization
    report['prediction_cache']=str(prediction_dir/f'{uid}.npz')
    atomic_json(args.output_dir/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
