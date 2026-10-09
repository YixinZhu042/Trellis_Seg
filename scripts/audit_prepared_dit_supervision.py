"""Deep read-only validation of every aligned DiT supervision file."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np


REQUIRED = {'coords','voxel_labels','latent_coords','latent_feats','shape_feats','shape_valid','mv_coords','mv_feats',
            'face_labels','face_area','face_knn','face_knn_weights','adjacency','boundary','face_pixels',
            'face_raster','seg_maps','sample_center','sample_extent','mesh_vertices','mesh_faces'}


def audit(path, split):
    errors=[]
    try:
        with np.load(path) as d:
            missing=REQUIRED-set(d.files)
            if missing:errors.append(f'missing keys {sorted(missing)}');return {'instance':path.stem,'split':split,'errors':errors}
            coords,latent=d['coords'],d['latent_coords'];faces=d['face_labels'];raster=d['face_raster'];hit=raster>=0
            if coords.ndim!=2 or coords.shape[1]!=3 or len(coords)!=len(d['voxel_labels']):errors.append('invalid voxel shapes')
            if (latent.ndim!=2 or latent.shape[1]!=3 or d['latent_feats'].ndim!=2 or d['latent_feats'].shape[1]!=32 or
                    d['shape_feats'].ndim!=2 or d['shape_feats'].shape[1]!=32 or
                    len(latent)!=len(d['latent_feats']) or len(latent)!=len(d['shape_feats'])):errors.append('invalid latent shapes')
            if d['shape_valid'].ndim!=1 or len(d['shape_valid'])!=len(latent):errors.append('invalid shape validity mask')
            if d['shape_valid'].size and not np.isin(d['shape_valid'],[0,1,False,True]).all():errors.append('invalid shape validity values')
            if len(latent) and np.any(d['shape_feats'][~d['shape_valid'].astype(bool)]!=0):errors.append('unmatched shape features are not zero')
            if (d['mv_coords'].ndim!=2 or d['mv_coords'].shape[1]!=3 or d['mv_feats'].ndim!=2 or
                    d['mv_feats'].shape[1]!=32 or len(d['mv_coords'])!=len(d['mv_feats'])):errors.append('invalid multi-view shapes')
            if len(latent)>8192:errors.append('latent tokens >8192')
            if coords.size and (coords.min()<0 or coords.max()>511):errors.append('voxel coords out of range')
            if len(coords) and len(np.unique(coords,axis=0))!=len(coords):errors.append('duplicate voxel coords')
            if len(latent) and len(np.unique(latent,axis=0))!=len(latent):errors.append('duplicate latent coords')
            for key in ['latent_coords','mv_coords']:
                if d[key].size and (d[key].min()<0 or d[key].max()>31):errors.append(f'{key} out of range')
            for key in ['latent_feats','shape_feats','mv_feats','face_area','face_knn_weights']:
                if not np.isfinite(d[key]).all():errors.append(f'{key} nonfinite')
            if faces.ndim!=1 or len(d['face_area'])!=len(faces) or len(d['face_pixels'])!=len(faces):errors.append('invalid face supervision shapes')
            if faces.size and (faces.min()<0 or faces.max()>19):errors.append('face labels out of range')
            if d['voxel_labels'].size and (d['voxel_labels'].min()<0 or d['voxel_labels'].max()>19):errors.append('voxel labels out of range')
            knn=d['face_knn'];weights=d['face_knn_weights'];adj=d['adjacency']
            if knn.shape!=weights.shape or len(knn)!=len(faces):errors.append('invalid KNN shapes')
            if knn.size and (knn.min()<0 or knn.max()>=len(coords)):errors.append('KNN out of range')
            if weights.size and (weights<0).any():errors.append('negative KNN weights')
            if len(weights) and not np.allclose(weights.sum(1),1,atol=1e-5):errors.append('KNN weights do not sum to 1')
            if adj.ndim!=2 or adj.shape[1]!=2 or len(d['boundary'])!=len(adj) or (adj.size and (adj.min()<0 or adj.max()>=len(faces))):errors.append('adjacency invalid')
            expected=(faces[adj[:,0]]!=faces[adj[:,1]]).astype(np.float32)
            if not np.array_equal(d['boundary'],expected):errors.append('boundary target mismatch')
            if raster.shape!=(6,512,512) or d['seg_maps'].shape!=raster.shape:errors.append('raster/map shape mismatch')
            if raster.size and (raster.min() < -1 or raster.max(initial=-1)>=len(faces)):errors.append('face raster out of range')
            if hit.any() and not np.array_equal(d['seg_maps'][hit],faces[raster[hit]]+1):errors.append('render labels mismatch')
            if (~hit).any() and np.any(d['seg_maps'][~hit]!=0):errors.append('render background mismatch')
            if d['seg_maps'].size and (d['seg_maps'].min()<0 or d['seg_maps'].max()>20):errors.append('render labels out of range')
            if not np.array_equal(d['face_pixels'],np.bincount(raster[hit],minlength=len(faces))):errors.append('face pixel counts mismatch')
            if len(d['mesh_faces'])!=len(faces):errors.append('mesh face count mismatch')
            if d['mesh_faces'].size and (d['mesh_faces'].min()<0 or d['mesh_faces'].max()>=len(d['mesh_vertices'])):errors.append('mesh topology invalid')
            if d['face_area'].size and (d['face_area']<0).any():errors.append('negative face area')
            if np.asarray(d['sample_extent']).size!=1 or float(d['sample_extent'])<=0:errors.append('invalid sample extent')
            for key in ['sample_center','sample_extent','mesh_vertices']:
                if not np.isfinite(d[key]).all():errors.append(f'{key} nonfinite')
            result={'instance':path.stem,'split':split,'errors':errors,'size':path.stat().st_size,
                    'mtime_ns':path.stat().st_mtime_ns,
                    'voxels':len(coords),'tokens':len(latent),'faces':len(faces),'parts':int(len(np.unique(faces[faces>0])))}
    except Exception as exc:
        result={'instance':path.stem,'split':split,'errors':[str(exc)]}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=16)
    args=ap.parse_args();selection=json.load(open(args.data_dir/'selection.json'));expected={r['model_id']:r['split'] for r in selection['samples']}
    start=time.monotonic();rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(audit,args.data_dir/'samples'/f'{uid}.npz',split) for uid,split in expected.items()]
        for i,future in enumerate(as_completed(futures),1):
            rows.append(future.result())
            if i%1000==0:print(f'Audited prepared files {i}/{len(futures)}',flush=True)
    rows.sort(key=lambda x:x['instance']);errors=[r for r in rows if r['errors']]
    selection_path=args.data_dir/'selection.json';prepared_path=args.data_dir/'prepared_manifest.json'
    file_state=[(row['instance'],row.get('size'),row.get('mtime_ns')) for row in rows]
    report={'objects':len(rows),'seconds':time.monotonic()-start,'errors_count':len(errors),'error_examples':errors[:100],
            'split_counts':{s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},
            'size_bytes':sum(r.get('size',0) for r in rows),
            'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            'prepared_manifest_sha256':hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
            'file_state_sha256':hashlib.sha256(json.dumps(file_state,separators=(',',':')).encode()).hexdigest(),
            'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_name(args.output.name+'.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(args.output)
    print(json.dumps({k:report[k] for k in ['objects','seconds','errors_count','split_counts','size_bytes']},indent=2))
    if errors:raise RuntimeError('Prepared data audit failed')


if __name__=='__main__':main()
