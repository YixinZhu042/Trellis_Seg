"""Read-only B progress/acceptance; --record writes a project-local audit report."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import atomic_json,digest
from utils.dit_vae_v2 import config,load_cache,identity
from utils.dit_training_b import object_order,training_permutation,BINS


def read_lines(path):
    lines=path.read_bytes().splitlines(keepends=True)
    # A concurrent incomplete final line is not a completed step.
    return [json.loads(line) for line in lines if line.endswith(b'\n')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['smoke','startup'],required=True)
    ap.add_argument('--min-steps',type=int,default=10)
    ap.add_argument('--record',action='store_true')
    args=ap.parse_args()
    cfg,_=config(ROOT/'configs/dit_vae_v2_parts64_b_smallparts.json')
    plan,manifest=load_cache(cfg)
    name=Path(cfg['output_dir']).name+('_smoke' if args.mode=='smoke' else '')
    logs=ROOT/'logs'/name/'train';output=Path(cfg['output_dir']).with_name(name)
    launches=sorted(logs.glob('remote_*.json'))
    if not launches:raise ValueError('No launch record')
    launch=json.loads(launches[-1].read_text())
    rows=[r for r in manifest['samples'] if r['split']=='train']
    orders={};records=[];batches=[];errors=[]
    for rank in range(24):
        file=logs/f'train_rank{rank:04d}.jsonl'
        trace=read_lines(file) if file.exists() else []
        if not trace:
            errors.append(f'rank{rank}: no completed steps');continue
        if [r['step'] for r in trace]!=list(range(1,trace[-1]['step']+1)):
            errors.append(f'rank{rank}: noncontiguous/duplicated steps')
        for r in trace:
            if r['rank']!=rank or not math.isfinite(r['grad_norm']):errors.append(f'rank{rank}: invalid rank/gradient')
            for micro,b in enumerate(r['batch']):
                for key in ('flow','loss','t','aux_gate','augmentation_seconds','segmentation','boundary','reprojection','surface_area_ce','surface_macro_ce'):
                    if key in b and b[key] is not None and not math.isfinite(b[key]):errors.append(f'rank{rank}: nonfinite {key}')
                if args.mode=='startup':
                    index=((r['step']-1)*cfg['train']['accumulate']+micro)*24+rank
                    epoch,offset=divmod(index,len(rows))
                    if epoch not in orders:orders[epoch]=object_order(rows,cfg['seed'],epoch,cfg['train']['object_sampling'])
                    expected=rows[int(orders[epoch][offset])]
                    mapping=training_permutation(cfg['seed'],index,cfg['train']['label_augmentation'])
                    if (b['sample_index']!=index or b['instance']!=expected['instance'] or b['parts']!=expected['parts']
                            or b['label_permuted']!=(mapping is not None)
                            or b['permutation_sha256']!=(digest(mapping.tolist()) if mapping is not None else None)):
                        errors.append(f'rank{rank}, step{r["step"]}: sampling/augmentation replay mismatch')
                batches.append(b)
        records.append({'rank':rank,'step':trace[-1]['step'],'loss':trace[-1]['batch'][-1]['loss'],
                        'seconds':trace[-1]['seconds'],'max_gpu_gb':trace[-1]['max_gpu_gb']})
    minimum=3 if args.mode=='smoke' else args.min_steps
    if len(records)!=24 or any(r['step']<minimum for r in records):errors.append(f'Not all24 ranks have {minimum} steps')
    gradients=[]
    for rank in range(24):
        file=logs/f'gradient_rank{rank:04d}.json'
        if not file.exists():errors.append(f'rank{rank}: missing gradient check');continue
        g=json.loads(file.read_text())
        if not all(math.isfinite(g[k]) and g[k]>0 for k in ('segmentation','boundary','reprojection','flow_gradient','weighted_aux_gradient')):
            errors.append(f'rank{rank}: invalid auxiliary gradient')
        gradients.append(g)
    exits=[]
    for job in launch['jobs']:
        file=Path(job['exit_file']);code=int(file.read_text().strip()) if file.exists() else None
        exits.append({'ip':job['ip'],'exit_code':code})
        if code not in (None,0):errors.append(f'node {job["ip"]}: failed exit {code}')
    coverage={'original':sum(not b['label_permuted'] for b in batches),'permuted':sum(b['label_permuted'] for b in batches),
              'condition_dropped':sum(b['drop'] for b in batches),'aux':sum('segmentation' in b for b in batches),
              'no_aux':sum('segmentation' not in b for b in batches)}
    if args.mode=='smoke':
        for rank in range(24):
            parity_file=logs/f'online_parity_rank{rank:04d}.json'
            done_file=logs/f'complete_rank{rank:04d}.json'
            if not parity_file.exists() or not done_file.exists():errors.append(f'rank{rank}: missing smoke evidence');continue
            parity=json.loads(parity_file.read_text());done=json.loads(done_file.read_text())
            if max(parity['max_errors'].values())!=0:errors.append(f'rank{rank}: nonzero identity parity')
            if (done['completed_steps']!=3 or not all(done.get(k) for k in ('gradient_checked','dit_weights_changed',
                    'vae_parameter_grads_absent','gaussian_generation_evaluation_passed'))):errors.append(f'rank{rank}: resume incomplete')
        if any(r['exit_code']!=0 for r in exits):errors.append('Smoke nodes have not all exited successfully')
        if not (logs/'smoke_generation.json').exists():errors.append('Missing smoke Gaussian generation')
    elif not all(coverage.values()):errors.append('Production branches not all observed yet')
    init=json.loads((output/'initialization.json').read_text())
    if init['source']!=identity(cfg['dit_init']) or init['source_step']!=4500 or init['source_kind']!='regular':
        errors.append('Initialization source differs')
    report={'time':datetime.now().astimezone().isoformat(),'mode':args.mode,'passed':not errors,'errors':errors,
            'launch_record':str(launches[-1]),'output_dir':str(output),'initialization':init,
            'cache_digest':digest(manifest),'vae_sha256':plan['vae_sha256'],'split_counts':manifest['split_counts'],
            'ranks':sorted(records,key=lambda r:r['rank']),'coverage':coverage,'gradient_checks':len(gradients),
            'sampled_part_groups':{f'{lo}-{hi}':sum(lo<=b['parts']<=hi for b in batches) for lo,hi in BINS},
            'node_exits':exits,'quality_improvement_verified':False}
    if args.record:
        if errors:raise ValueError('Refuse passing acceptance record: '+str(errors))
        atomic_json(logs/f'{args.mode}_verification.json',report)
    print(json.dumps(report,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
