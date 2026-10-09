"""Evaluate VAE v2 checkpoints on validation; test requires a saved validation selection."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import load_config, digest, atomic_json, storage_path
from vae_v2.data import load_manifest,load_sample
from vae_v2.evaluation import reconstruct,summarize,selection_score,export_prediction


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/seg_vae_v2.json')
    ap.add_argument('--checkpoint',type=Path,nargs='+')
    ap.add_argument('--selection',type=Path)
    ap.add_argument('--split',choices=['validation','test'],default='validation')
    ap.add_argument('--output_dir',type=Path,required=True)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--export',action='store_true')
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args();cfg=load_config(args.config)
    if args.limit is not None and args.limit<1:ap.error('limit must be positive')
    manifest=load_manifest(cfg)
    rows=[r for r in manifest['samples'] if r['split']==args.split]
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('Empty evaluation cohort')
    paths=args.checkpoint or []
    if args.split=='test':
        if not args.selection:ap.error('Test requires --selection from validation')
        selected=json.loads(args.selection.read_text())
        if selected['split']!='validation' or selected['test_used_for_selection']:
            raise ValueError('Invalid selection provenance')
        if selected.get('smoke_only') or selected.get('manifest_digest')!=digest(manifest):
            raise ValueError('Smoke runs or mismatched validation provenance cannot select test checkpoints')
        validation_ids={r['instance'] for r in manifest['samples'] if r['split']=='validation'}
        if not selected['ids'] or not set(selected['ids']).issubset(validation_ids):
            raise ValueError('Selection includes non-validation IDs')
        chosen=Path(selected['checkpoint']).resolve()
        if paths and [p.resolve() for p in paths]!=[chosen]:raise ValueError('Test checkpoint differs from validation choice')
        paths=[chosen]
    if not paths:ap.error('Supply --checkpoint or a test --selection')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Use a new evaluation directory')
    args.output_dir.mkdir(parents=True)
    torch.cuda.set_device(args.device)
    from vae_v2.model import SegVaeV2
    best=None
    for index,path in enumerate(paths):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('smoke_only'):raise ValueError('Smoke checkpoint is not eligible for official evaluation')
        checkpoint_cfg=state['config']
        if state['manifest_digest']!=digest(manifest):raise ValueError('Checkpoint was trained on another cache manifest')
        # Identical audited caches may have moved to the designated storage disk.
        checkpoint_cfg['prepared_dir']=cfg['prepared_dir']
        model=SegVaeV2(checkpoint_cfg,initialize=False).to(args.device)
        model.load_state_dict(state['model']);del state
        output=args.output_dir/f'{index:03d}_{path.stem}';output.mkdir()
        results=[]
        for row in rows:
            sample=load_sample(Path(cfg['prepared_dir'])/row['path'])
            metric,faces,voxels=reconstruct(model,sample,checkpoint_cfg,args.device)
            results.append(dict(instance=row['instance'],**metric))
            if args.export:
                prediction_dir=storage_path(Path(cfg['storage_root'])/'predictions'/args.output_dir.name/output.name,cfg)
                export_prediction(output,row['instance'],sample,faces,voxels,row['source'],prediction_dir)
            print(path.name,row['instance'],metric['surface'],flush=True)
        report={'checkpoint':str(path.resolve()),'split':args.split,'ids':[r['instance'] for r in rows],
                'summary':summarize(results),'samples':results,'test_used_for_selection':False,
                'manifest_digest':digest(manifest),'smoke_only':False}
        if args.export:report['prediction_dir']=str(prediction_dir)
        atomic_json(output/'metrics.json',report)
        if args.split=='validation' and (best is None or selection_score(report['summary'])>selection_score(best['summary'])):
            best={k:v for k,v in report.items() if k!='samples'}
            atomic_json(args.output_dir/'selection.json',best)
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':main()
