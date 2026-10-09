"""Frozen VAE-v2 / DiT cache and runtime contracts, separate from legacy 20-class data."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FORMAT='dit_vae_v2_parts64_v1'


def identity(path):
    p=Path(path).resolve();s=p.stat()
    return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def config(path):
    from vae_v2.config import load_config,storage_path,read_config
    cfg=read_config(path)
    for k in ('vae_config','vae_selection','base_config','dit_init','shape_dir','cache_dir','output_dir','log_dir'):
        cfg[k]=str((ROOT/cfg[k]).resolve())
    vae_cfg=load_config(cfg['vae_config'])
    if vae_cfg.get('num_classes')!=64 or vae_cfg['model']['latent_channels']!=32:
        raise ValueError('Expected corrected 64-class / latent32 VAE')
    for k in ('cache_dir','output_dir'):storage_path(cfg[k],vae_cfg)
    t=cfg['train']
    for key in ('steps','warmup_steps','accumulate','aux_warmup_steps','query_faces','pair_count','save_every','validate_every'):
        if not isinstance(t[key],int) or isinstance(t[key],bool) or t[key]<1:raise ValueError('Invalid train.'+key)
    if not isinstance(cfg['max_tokens'],int) or cfg['max_tokens']<1:raise ValueError('Invalid token limit')
    for key in ('condition_dropout','sigma_min','aux_t_max','ema_rate'):
        if not 0<=t[key]<1:raise ValueError('Invalid train.'+key)
    if not 0<=t['balanced_fraction']<=1 or not np.isfinite(t['lr']) or t['lr']<=0:raise ValueError('Invalid optimizer/sampling')
    if set(t['aux_weights'])!={'segmentation','boundary','reprojection'} or any(not np.isfinite(v) or v<0 for v in t['aux_weights'].values()):
        raise ValueError('Invalid auxiliary weights')
    from utils.dit_training_b import validate_options
    validate_options(t)
    return cfg,vae_cfg


def align_shape(target_coords, coords, feats):
    """Exact 32^3 coordinate join; absent geometry stays masked zero, no nearest fill."""
    def checked(c):
        c=np.asarray(c)
        if c.ndim!=2 or c.shape[1]!=3 or not np.issubdtype(c.dtype,np.integer) or (len(c) and (c.min()<0 or c.max()>31)):
            raise ValueError('Expected integer 32^3 coords')
        return c.astype(np.int64)
    target,source=checked(target_coords),checked(coords)
    if feats.shape!=(len(source),32) or not np.isfinite(feats).all():raise ValueError('Invalid shape features')
    sh=source@np.array([1024,32,1]);th=target@np.array([1024,32,1])
    if len(np.unique(sh))!=len(sh):raise ValueError('Duplicate shape coordinates')
    result=np.zeros((len(target),32),np.float32);valid=np.zeros(len(target),bool)
    if len(source):
        order=np.argsort(sh);pos=np.searchsorted(sh[order],th).clip(max=len(sh)-1)
        valid=sh[order[pos]]==th;result[valid]=feats[order[pos[valid]]]
    return result,valid


def moments(array):
    a=np.asarray(array,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=32 or not np.isfinite(a).all():raise ValueError('Invalid latent array')
    return {'n':len(a),'sum':a.sum(0).tolist(),'sum_sq':np.square(a).sum(0).tolist()}


def normalization(rows):
    result={}
    for key in ('latent_feats','mv_feats','shape_feats'):
        selected=[r['moments'][key] for r in rows if r['split']=='train']
        n=sum(r['n'] for r in selected)
        if not n:raise ValueError('No training tokens for '+key)
        mean=np.sum([r['sum'] for r in selected],0)/n
        variance=np.maximum(np.sum([r['sum_sq'] for r in selected],0)/n-mean**2,0)
        std=np.sqrt(variance).clip(min=1e-6)
        result[key]={'mean':mean.tolist(),'std':std.tolist(),'tokens':n,'split':'train'}
    return result


def capacity(cfg,plan):
    """A token acceptance ceiling can be raised without changing encoded tensors.

    Preserve the immutable original plan and all existing sample checksums.
    A separate audited acceptance policy is mandatory when the limit changes.
    """
    from vae_v2.config import digest
    value=cfg['max_tokens'];original=plan['config']['max_tokens']
    if value==original:return value
    path=Path(cfg['cache_dir'])/'capacity.json'
    change=json.loads(path.read_text())
    if (change['plan_digest']!=digest(plan) or change['original_max_tokens']!=original
            or change['max_tokens']!=value or value<original or change['resampling']):
        raise ValueError('Invalid token capacity extension')
    return value


def validate_latents(data,max_tokens):
    for c,f in [('latent_coords','latent_feats'),('mv_coords','mv_feats')]:
        co,fe=data[c],data[f]
        if co.ndim!=2 or co.shape[1]!=3 or not len(co) or not np.issubdtype(co.dtype,np.integer) or co.min()<0 or co.max()>31:
            raise ValueError('Invalid '+c)
        if fe.shape!=(len(co),32) or not np.isfinite(fe).all():raise ValueError('Invalid '+f)
    if len(data['latent_coords'])>max_tokens:raise ValueError('Latent token budget exceeded; no silent truncation')
    if len(np.unique(data['latent_coords'],axis=0))!=len(data['latent_coords']):raise ValueError('Duplicate target coords')
    if data['shape_feats'].shape!=data['latent_feats'].shape or not np.isfinite(data['shape_feats']).all():raise ValueError('Bad shape alignment')
    if data['shape_valid'].shape!=(len(data['latent_coords']),) or data['shape_valid'].dtype!=np.bool_:raise ValueError('Bad shape mask')
    if not data['shape_valid'].any() or np.any(data['shape_feats'][~data['shape_valid']]!=0):raise ValueError('Empty or unmasked shape condition')
    offsets=data['view_offsets']
    if offsets.shape!=(7,) or offsets[0]!=0 or offsets[-1]!=len(data['mv_coords']) or (np.diff(offsets)<0).any():raise ValueError('Bad view offsets')


def load_cache(cfg):
    from vae_v2.config import digest
    root=Path(cfg['cache_dir']);plan=json.loads((root/'plan.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    if manifest['plan_digest']!=digest(plan) or manifest['status']!='complete' or manifest['failures']:
        raise ValueError('Cache not sealed/compatible')
    if plan['pilot_only'] or manifest['pilot_only']:raise ValueError('Pilot cache is not a production dataset')
    ceiling=capacity(cfg,plan)
    if manifest.get('max_tokens',plan['config']['max_tokens'])!=ceiling:raise ValueError('Cache capacity differs')
    if manifest['split_counts']!={'train':22633,'validation':128,'test':128}:raise ValueError('Wrong split')
    # The full audit checks all arrays/checksums; every subsequent load checks file identity.
    if manifest['expected']!=len(manifest['samples']) or manifest['expected']!=len(plan['samples']):raise ValueError('Incomplete cache')
    if len({r['instance'] for r in manifest['samples']})!=len(plan['samples']):raise ValueError('Duplicate IDs')
    expected={r['instance']:r['split'] for r in plan['samples']}
    if {r['instance']:r['split'] for r in manifest['samples']}!=expected:raise ValueError('Cache split membership differs')
    selection=json.loads(Path(cfg['vae_selection']).read_text())
    if identity(selection['checkpoint'])!=plan['vae_checkpoint']:raise ValueError('Training selected another VAE')
    audit=json.loads((root/'audit.json').read_text())
    if audit['manifest_sha256']!=file_sha256(root/'manifest.json') or audit['plan_digest']!=digest(plan):raise ValueError('Cache not covered by final audit')
    if audit.get('objects')!=len(plan['samples']) or not audit.get('all_arrays_checked') or not audit.get('all_checksums_checked'):
        raise ValueError('Full deep audit required')
    if manifest['format']!=FORMAT or plan['format']!=FORMAT:raise ValueError('Wrong cache format')
    if manifest['normalization']!=json.loads((root/'normalization.json').read_text()):raise ValueError('Normalization differs from audit')
    return plan,manifest


def load_latents(row,cache_root):
    path=Path(cache_root)/row['path'];s=path.stat()
    if (s.st_size,s.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):raise ValueError('Changed cache '+str(path))
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']))
        if meta['instance']!=row['instance'] or meta['split']!=row['split'] or meta['plan_digest']!=row['plan_digest']:
            raise ValueError('Cached sample metadata differs')
        return {k:d[k] for k in d.files if k!='metadata'}


def normalize_tensors(data,stats,device):
    import torch
    out={}
    for key in ('latent_feats','shape_feats','mv_feats'):
        x=torch.as_tensor(data[key],dtype=torch.float32,device=device)
        mean=torch.tensor(stats[key]['mean'],device=device);std=torch.tensor(stats[key]['std'],device=device)
        x=(x-mean)/std
        if key=='shape_feats':x=x.masked_fill(~torch.as_tensor(data['shape_valid'],device=device)[:,None],0)
        out[key]=x
    return out


def conditions(template,data,normalized,drop):
    import torch
    return {'cond':torch.zeros(1,1029,1024,device=template.device),
            'concat_cond':template.replace(normalized['shape_feats']),
            'mv_cond':[None if drop else normalized['mv_feats']],
            'mv_coords':[None if drop else torch.as_tensor(data['mv_coords'],dtype=torch.float32,device=template.device)]}


def unnormalize(x,stats):
    import torch
    return x*torch.tensor(stats['latent_feats']['std'],device=x.device)+torch.tensor(stats['latent_feats']['mean'],device=x.device)


def frozen_aux_losses(vae,template,predicted_clean,data,sample,query,stats,train=None,diagnostics=None):
    """Differentiable clean latent -> new VAE surface head -> corrected supervision."""
    import torch
    from vae_v2.losses import balanced_soft_ce,weighted_ce,relation_losses,macro_surface_ce
    features,logits,coords=vae.decode_voxels(template.replace(unnormalize(predicted_clean.float(),stats)))
    if not torch.equal(coords[:,1:].int(),torch.as_tensor(sample['coords'],device=coords.device).int()):raise ValueError('Decoder coords changed')
    with torch.autocast('cuda',dtype=torch.bfloat16):_,face=vae.decode_surface(features,logits,query)
    n=query['supervised_count'];labels=query['query_labels'][:n]
    target=torch.as_tensor(sample['voxel_target'],device=logits.device)
    consistency,boundary=relation_losses(face,query['pair_index'],query['pair_same'],query['pair_geometry_weight'])
    area_ce=weighted_ce(face[:n],labels,query['surface_importance'])
    fraction=(train or {}).get('surface_macro_fraction',0.)
    macro_ce=macro_surface_ce(face[:n],labels,query['surface_macro_importance']) if fraction else None
    surface_ce=((1-fraction)*area_ce+fraction*macro_ce) if fraction else area_ce
    if diagnostics is not None:
        diagnostics['surface_area_ce']=float(area_ce.detach())
        diagnostics['surface_macro_ce']=float(macro_ce.detach()) if macro_ce is not None else None
    return {'segmentation':.5*balanced_soft_ce(logits,target)+.5*surface_ce,
            'boundary':.5*(consistency+boundary),
            'reprojection':weighted_ce(face[:n],labels,query['pixel_importance'])}
