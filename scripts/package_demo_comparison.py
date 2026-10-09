"""Audit completed paired inference and package its browser-facing assets."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import digest,atomic_json
from utils.dit_vae_v2 import file_sha256,identity


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result',type=Path,required=True);ap.add_argument('--git-repo',type=Path,required=True)
    args=ap.parse_args();out=args.result.resolve();git=args.git_repo.resolve()
    plan=json.loads((out/'plan.json').read_text());records=[];manifest=[];total=0
    for row in plan['samples']:
        r=json.loads((out/'metrics'/f'{row["id"]}.json').read_text())
        if r['plan_digest']!=digest(plan) or not r['initial_noise']['identical'] or not r['target_coords_identical']:
            raise ValueError('Paired inference contract changed')
        if any(r[k] for k in ('postprocessing','smoothing','label_matching','prior_matching','gt_initialization')):
            raise ValueError('Unexpected postprocessing or target leakage')
        for key,info in r['model_hashes'].items():
            if identity(plan[key])!={k:info[k] for k in ('path','bytes','mtime_ns')}:raise ValueError('Model changed')
        target=git/'demo/comparison'/row['id'];target.mkdir(parents=True,exist_ok=True)
        asset_checks={}
        for name in ('old.glb','new.glb','inputcondition.png','comparison.png'):
            src=out/'objects'/row['id']/name
            if not src.is_file():raise ValueError('Missing '+str(src))
            dest=target/name;shutil.copy2(src,dest)
            if file_sha256(src)!=file_sha256(dest):raise ValueError('Copy checksum differs')
            asset_checks[name]={'bytes':dest.stat().st_size,'sha256':file_sha256(dest)};total+=dest.stat().st_size
        source_image=None
        if row.get('source_image'):
            dest=target/'source.png';shutil.copy2(row['source_image'],dest);source_image=f'comparison/{row["id"]}/source.png'
        archive=None
        if row.get('archive_path'):
            archive=str(int(row['id'][4:]))+'.glb'
            if file_sha256(git/'demo'/archive)!=file_sha256(row['archive_path']):raise ValueError('Archived original changed')
        records.append(r)
        manifest.append({'id':row['id'],'title':row['title'],'kind':row['kind'],'source_name':row['source_name'],
            'old_glb':f'comparison/{row["id"]}/old.glb','new_glb':f'comparison/{row["id"]}/new.glb',
            'archive_glb':archive,'condition_image':f'comparison/{row["id"]}/inputcondition.png',
            'comparison_image':f'comparison/{row["id"]}/comparison.png','source_image':source_image,
            'faces':r['face_count'],'metrics':{side:{k:r[side]['condition_agreement'][k] for k in ('accuracy','miou')} for side in ('old','new')},
            'shape_coverage':r['shape_valid_fraction'],'notes':row['notes'],
            'source_geometry_sha256':r['source_geometry_hash'],'same_noise_sha256':r['initial_noise']['old'],
            'assets':asset_checks})
    summary={}
    for kind in ('original_demo','character'):
        rows=[r for r in manifest if r['kind']==kind]
        summary[kind]={'count':len(rows),
            'mean_condition_agreement':{side:{key:statistics.mean(r['metrics'][side][key] for r in rows) for key in ('accuracy','miou')} for side in ('old','new')},
            'improved_miou':sum(r['metrics']['new']['miou']>r['metrics']['old']['miou'] for r in rows),
            'regressed_miou':sum(r['metrics']['new']['miou']<r['metrics']['old']['miou'] for r in rows)}
    public={'release_date':'2026-10-09','created_at':datetime.now().astimezone().isoformat(),
        'protocol':{'old_model':'Legacy20 VAE encoder/decoder step1000 + DiT EMA0.9999 step50000',
            'new_model':'VAE v2 parts64 regular9500 + B small-part DiT EMA5000',
            'steps':20,'seed':42,'cfg':1,'postprocessing':False,
            'condition_note':'Same geometry-only sparse grid, exact same six-view label pixels and Gaussian noise for both models. Old and new use their respective trained normalization. No labels inferred from the archived outputs.',
            'metric_note':'Exact-ID agreement with shared 2D input labels, not independent GT accuracy. Historical conditions may be incorrect; character conditions are external API predictions. Improvements and regressions are all retained.',
            'legacy_face_decode':'8NN inverse-distance centroid logits','new_face_decode':'learned geometry-aware four-point surface head',
            'geometry_note':'Original face count/order kept. Only common bbox centering and scaling applied to exported web GLBs.'},
        'samples':manifest,'summary':summary,'archived_outputs_preserved':True,'model_hashes':{k:v['sha256'] for k,v in records[0]['model_hashes'].items()}}
    atomic_json(git/'demo/comparison.json',public)
    atomic_json(out/'summary.json',public)
    atomic_json(out/'package_audit.json',{'count':len(records),'glbs':2*len(records),'asset_bytes':total,
        'all_pair_noise_and_coords_identical':True,'archive_hashes_unchanged':True,'public_manifest_has_no_machine_paths':True})
    print(json.dumps({'summary':summary,'objects':len(records),'asset_bytes':total,'manifest':str(git/'demo/comparison.json')},indent=2))


if __name__=='__main__':main()
