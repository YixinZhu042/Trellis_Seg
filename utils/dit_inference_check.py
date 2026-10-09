"""Inference-only, paired CFG / steps / spatial-conditioning diagnostics."""
import math
import numpy as np


def experiments():
    settings=[]
    for steps in (20,50):
        for scale in (1.,1.5,2.,3.):
            name=f'correct_s{steps}_cfg{scale:g}'.replace('.','p')
            settings.append({'name':name,'steps':steps,'cfg':scale,'condition':'correct','eligible':True})
    settings.extend([
        {'name':'no_mv_s20','steps':20,'cfg':1.,'condition':'no_mv','eligible':False},
        {'name':'shuffled_s20','steps':20,'cfg':1.,'condition':'shuffle_within_view','eligible':False},
    ])
    return settings


def shuffled_view_features(data,seed):
    """Destroy within-view spatial correspondence, NOT paired-token list order.

    Coordinates stay fixed. Each view's feature multiset and token count are
    preserved. No label renaming, cross-object mixing, or GT-dependent sampling.
    """
    features=np.asarray(data['mv_feats']);offsets=np.asarray(data['view_offsets'])
    if offsets.shape!=(7,) or offsets[0]!=0 or offsets[-1]!=len(features) or (np.diff(offsets)<0).any():
        raise ValueError('Invalid view offsets')
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),0x53485546]))
    order=np.arange(len(features))
    for start,end in zip(offsets[:-1],offsets[1:]):
        if end-start>1:
            local=rng.permutation(end-start)
            # Avoid accidental identity for tiny views; this is a diagnostic,
            # not an attempt at sampling uniform permutations including identity.
            if np.array_equal(local,np.arange(end-start)):local=np.roll(local,1)
            order[start:end]=start+local
    return features[order].copy(),order


def guided_velocity(conditional,shape_only,scale):
    if not math.isfinite(scale) or scale<0:raise ValueError('Invalid CFG scale')
    if scale==1:return conditional
    if scale==0:return shape_only
    return shape_only+scale*(conditional-shape_only)


def generate(model,template,data,stats,seed,setting,device):
    import torch
    from scripts.evaluate_dit_vae_v2 import normalize_conditions
    from utils.dit_vae_v2 import conditions
    mode=setting['condition'];scale=setting['cfg'];steps=setting['steps']
    if mode not in ('correct','no_mv','shuffle_within_view') or steps<1:raise ValueError('Invalid experiment')
    altered=data
    if mode=='shuffle_within_view':
        features,_=shuffled_view_features(data,seed)
        altered=dict(data,mv_feats=features)
    normalized=normalize_conditions(altered,stats,device)
    conditional=conditions(template,altered,normalized,mode=='no_mv')
    shape_only=conditions(template,altered,normalized,True)
    # Same initial noise in EVERY setting, independent of shuffle RNG.
    generator=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn((len(data['latent_coords']),32),device=device,generator=generator)
    evaluations=0
    with torch.no_grad():
        for step in range(steps):
            t=torch.tensor([(1-step/steps)*1000],device=device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
                vc=model(template.replace(x),t,**conditional).feats.float();evaluations+=1
                if scale!=1 and mode!='no_mv':
                    vu=model(template.replace(x),t,**shape_only).feats.float();evaluations+=1
                    v=guided_velocity(vc,vu,scale)
                else:v=vc
            if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite guided velocity')
            x=x-v/steps
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite generated latent')
    return x,evaluations


def metric_difference(left,right):
    values=[]
    for group in ('surface','projection','voxel','hidden_surface'):
        for key in ('accuracy','miou','small_part_recall'):
            a,b=left[group][key],right[group][key]
            if a is None or b is None:
                if a is not None or b is not None:raise ValueError('Metric coverage changed')
            else:values.append(abs(a-b))
    return max(values,default=0.)
