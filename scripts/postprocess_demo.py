"""Geometry-only cleanup for the 18 paired demo outputs.

The transformation never reads GT labels and never copies the 2D condition onto
the mesh.  Conditions are used only to report/choose one global development-demo
setting. Raw predictions remain untouched and are always exported beside cleanup.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.demo_paired_inference import projection_agreement
from vae_v2.config import atomic_json
from vae_v2.visuals import export_labels,render_palette
from utils.seg_projection import render_labels


CANDIDATES=(
    {'name':'area_0p002','component_area_fraction':2e-5,'component_dominance':.78,'smooth':False},
    {'name':'area_0p005','component_area_fraction':5e-5,'component_dominance':.78,'smooth':False},
    {'name':'area_0p01','component_area_fraction':1e-4,'component_dominance':.78,'smooth':False},
    {'name':'area_0p02','component_area_fraction':2e-4,'component_dominance':.78,'smooth':False},
    {'name':'area_0p02_boundary','component_area_fraction':2e-4,'component_dominance':.78,'smooth':True},
    {'name':'area_0p05_boundary','component_area_fraction':5e-4,'component_dominance':.8,'smooth':True},
)


def label_components(labels,adjacency,face_area):
    labels=np.asarray(labels);adjacency=np.asarray(adjacency,dtype=np.int64)
    same=labels[adjacency[:,0]]==labels[adjacency[:,1]] if len(adjacency) else np.zeros(0,bool)
    edges=adjacency[same]
    graph=coo_matrix((np.ones(len(edges)*2,np.uint8),
        (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(labels),len(labels))).tocsr()
    count,component=connected_components(graph,directed=False)
    area=np.bincount(component,weights=np.asarray(face_area,dtype=np.float64),minlength=count)
    size=np.bincount(component,minlength=count)
    return component,area,size


def diagnostics(labels,adjacency,area,condition_ids):
    comp,comp_area,comp_size=label_components(labels,adjacency,area)
    total=float(np.asarray(area,dtype=np.float64).sum())
    tiny=comp_area<total*2e-4
    absent=~np.isin(labels,np.asarray(condition_ids,dtype=labels.dtype))
    return {'components':int(len(comp_area)),'tiny_components_at_0p02pct':int(tiny.sum()),
        'tiny_component_area_fraction':float(comp_area[tiny].sum()/total),
        'unique_ids':int(len(np.unique(labels))),'absent_id_area_fraction':float(np.asarray(area)[absent].sum()/total),
        'largest_component_area_fraction':float(comp_area.max()/total),'single_face_components':int((comp_size==1).sum())}


def cleanup_components(labels,adjacency,area,normals,threshold,dominance,iterations=2):
    labels=np.asarray(labels).copy();adjacency=np.asarray(adjacency,dtype=np.int64)
    area=np.asarray(area,dtype=np.float64);normals=np.asarray(normals,dtype=np.float64);total=area.sum()
    changed=np.zeros(len(labels),bool);history=[]
    for iteration in range(iterations):
        component,comp_area,_=label_components(labels,adjacency,area)
        small=comp_area<total*threshold
        boundary=labels[adjacency[:,0]]!=labels[adjacency[:,1]]
        edges=adjacency[boundary]
        if not len(edges) or not small.any():break
        cosine=np.maximum(np.einsum('ij,ij->i',normals[edges[:,0]],normals[edges[:,1]]),0.)
        weights=(np.sqrt(area[edges[:,0]]*area[edges[:,1]])+1e-12)*(.05+.95*cosine**2)
        votes={}
        for side in (0,1):
            face=edges[:,side];other=edges[:,1-side]
            candidate=small[component[face]]
            for comp,target,w in zip(component[face[candidate]],labels[other[candidate]],weights[candidate]):
                bucket=votes.setdefault(int(comp),{});bucket[int(target)]=bucket.get(int(target),0.)+float(w)
        replace={}
        for comp,bucket in votes.items():
            target,top=max(bucket.items(),key=lambda x:(x[1],-x[0]));mass=sum(bucket.values())
            if mass>0 and top/mass>=dominance:replace[comp]=target
        if not replace:break
        updated=labels.copy();round_changed=0
        for comp,target in replace.items():
            hit=component==comp;round_changed+=int(np.count_nonzero(labels[hit]!=target));updated[hit]=target;changed|=hit
        labels=updated;history.append({'iteration':iteration+1,'components_replaced':len(replace),'faces_changed':round_changed})
    return labels,changed,history


def smooth_isolated_boundaries(labels,adjacency,area,normals):
    """One synchronous high-confidence vote; avoids sharp geometric creases."""
    labels=np.asarray(labels);adjacency=np.asarray(adjacency,dtype=np.int64);area=np.asarray(area,dtype=np.float64)
    normals=np.asarray(normals,dtype=np.float64);votes=[{} for _ in range(len(labels))]
    if not len(adjacency):return labels.copy(),np.zeros(len(labels),bool)
    cosine=np.maximum(np.einsum('ij,ij->i',normals[adjacency[:,0]],normals[adjacency[:,1]]),0.)
    weight=(np.sqrt(area[adjacency[:,0]]*area[adjacency[:,1]])+1e-12)*cosine**4
    for side in (0,1):
        face=adjacency[:,side];other=adjacency[:,1-side]
        for f,target,w in zip(face,labels[other],weight):
            if w<=0:continue
            bucket=votes[int(f)];bucket[int(target)]=bucket.get(int(target),0.)+float(w)
    result=labels.copy();changed=np.zeros(len(labels),bool)
    for face,bucket in enumerate(votes):
        if not bucket:continue
        total=sum(bucket.values());target,top=max(bucket.items(),key=lambda x:(x[1],-x[0]))
        current=bucket.get(int(labels[face]),0.)
        if target!=labels[face] and top/total>=.86 and current/total<=.08 and top>=3*max(current,1e-20):
            result[face]=target;changed[face]=True
    return result,changed


def apply_candidate(raw,geometry,candidate):
    labels,component_changed,history=cleanup_components(raw,geometry['adjacency'],geometry['face_area'],
        geometry['face_normals'],candidate['component_area_fraction'],candidate['component_dominance'])
    boundary_changed=np.zeros(len(labels),bool)
    if candidate['smooth']:labels,boundary_changed=smooth_isolated_boundaries(labels,geometry['adjacency'],geometry['face_area'],geometry['face_normals'])
    changed=labels!=raw;area=np.asarray(geometry['face_area'],dtype=np.float64)
    return labels,{'changed_faces':int(changed.sum()),'changed_area_fraction':float(area[changed].sum()/area.sum()),
        'component_changed_faces':int(component_changed.sum()),'boundary_changed_faces':int(boundary_changed.sum()),'component_history':history}


def load_all(result):
    result=Path(result);plan=json.loads((result/'plan.json').read_text());rows=[]
    pred=Path(plan['prediction_dir'])
    for row in plan['samples']:
        with np.load(row['geometry_path'],allow_pickle=False) as data:geometry={k:data[k] for k in data.files}
        with np.load(row['condition_path'],allow_pickle=False) as data:maps=data['render_maps']
        with np.load(pred/f'{row["id"]}.npz',allow_pickle=False) as data:raw=data['new_faces']
        rows.append((row,geometry,maps,raw))
    return plan,rows


def scan(args):
    plan,rows=load_all(args.result);report=[]
    raw_metrics={row['id']:projection_agreement(raw,g['face_raster'],maps) for row,g,maps,raw in rows}
    for candidate in CANDIDATES:
        samples=[]
        for row,g,maps,raw in rows:
            cleaned,change=apply_candidate(raw,g,candidate);before=raw_metrics[row['id']];after=projection_agreement(cleaned,g['face_raster'],maps)
            samples.append({'id':row['id'],'kind':row['kind'],'before':before,'after':after,
                'miou_delta':None if before['miou'] is None else after['miou']-before['miou'],
                'accuracy_delta':None if before['accuracy'] is None else after['accuracy']-before['accuracy'],
                'change':change,'before_fragmentation':diagnostics(raw,g['adjacency'],g['face_area'],before['condition_ids']),
                'after_fragmentation':diagnostics(cleaned,g['adjacency'],g['face_area'],before['condition_ids'])})
        deltas=[x['miou_delta'] for x in samples if x['miou_delta'] is not None];acc=[x['accuracy_delta'] for x in samples if x['accuracy_delta'] is not None]
        before_comp=sum(x['before_fragmentation']['components'] for x in samples);after_comp=sum(x['after_fragmentation']['components'] for x in samples)
        summary={'name':candidate['name'],'settings':candidate,'mean_miou_delta':float(np.mean(deltas)),
            'mean_accuracy_delta':float(np.mean(acc)),'worst_miou_delta':float(min(deltas)),
            'improved':sum(x>1e-12 for x in deltas),'regressed':sum(x< -1e-12 for x in deltas),
            'regressed_over_1pp':sum(x<-.01 for x in deltas),'changed_area_mean':float(np.mean([x['change']['changed_area_fraction'] for x in samples])),
            'components_before':before_comp,'components_after':after_comp,'component_reduction_fraction':1-after_comp/before_comp,'samples':samples}
        # One global setting: avoid any >1pp regression and cap average area/accuracy impact.
        summary['eligible']=summary['worst_miou_delta']>=-.01 and summary['mean_accuracy_delta']>=-.001 and summary['changed_area_mean']<=.02
        report.append(summary)
    eligible=[x for x in report if x['eligible']]
    if not eligible:raise ValueError('No conservative global candidate passed')
    # Prefer input agreement first, then component reduction. The operation itself never consumes condition IDs.
    chosen=max(eligible,key=lambda x:(x['mean_miou_delta'],x['component_reduction_fraction'],-x['changed_area_mean']))
    result={'created_at':datetime.now().astimezone().isoformat(),'selection_split':'18 development demo objects, not held-out test',
        'selection_rule':'One global geometry-only setting; maximize mean input-condition mIoU among candidates with worst delta >=-1pp, mean accuracy delta >=-0.1pp and mean changed area <=2%',
        'gt_used':False,'condition_used_by_transformation':False,'chosen':chosen['name'],'candidates':report}
    atomic_json(Path(args.result)/'postprocess_scan.json',result);print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2));
    for x in report:print(x['name'],'eligible',x['eligible'],'miou',x['mean_miou_delta'],'worst',x['worst_miou_delta'],'components',x['component_reduction_fraction'],'area',x['changed_area_mean'])


def export(args):
    import trimesh
    from PIL import Image,ImageDraw
    result=Path(args.result);plan,rows=load_all(result);scan_report=json.loads((result/'postprocess_scan.json').read_text())
    chosen=next(x for x in scan_report['candidates'] if x['name']==scan_report['chosen']);candidate=chosen['settings'];all_records=[]
    for row,g,maps,raw in rows:
        cleaned,change=apply_candidate(raw,g,candidate);before=projection_agreement(raw,g['face_raster'],maps);after=projection_agreement(cleaned,g['face_raster'],maps)
        folder=result/'objects'/row['id'];center,extent=g['center'],float(g['extent']);mesh=trimesh.Trimesh((g['mesh_vertices']-center)/extent,g['mesh_faces'],process=False)
        export_labels(mesh,cleaned,folder/'post.glb',64)
        palette=render_palette(64);raster=g['face_raster'];hit=raster>=0;canvas=Image.new('RGB',(1536,840),'white');draw=ImageDraw.Draw(canvas)
        for y,(name,labels) in enumerate((('Input condition',None),('New raw',raw),('New postprocessed',cleaned))):
            render=maps if labels is None else np.where(hit,labels[raster.clip(min=0)]+1,0)
            for view in range(6):
                draw.text((view*256+4,y*280+4),name,fill='black');canvas.paste(Image.fromarray(palette[render[view]]),(view*256,y*280+24))
        canvas.save(folder/'post_comparison.png')
        record={'id':row['id'],'kind':row['kind'],'candidate':candidate,'raw':before,'post':after,
            'miou_delta':after['miou']-before['miou'],'accuracy_delta':after['accuracy']-before['accuracy'],'change':change,
            'raw_fragmentation':diagnostics(raw,g['adjacency'],g['face_area'],before['condition_ids']),
            'post_fragmentation':diagnostics(cleaned,g['adjacency'],g['face_area'],before['condition_ids'])}
        all_records.append(record);atomic_json(result/'metrics'/f'{row["id"]}_postprocess.json',record)
    atomic_json(result/'postprocess_summary.json',{'candidate':candidate,'samples':all_records,
        'mean_miou_delta':float(np.mean([x['miou_delta'] for x in all_records])),
        'mean_accuracy_delta':float(np.mean([x['accuracy_delta'] for x in all_records])),
        'improved':sum(x['miou_delta']>1e-12 for x in all_records),'regressed':sum(x['miou_delta']< -1e-12 for x in all_records),
        'components_before':sum(x['raw_fragmentation']['components'] for x in all_records),
        'components_after':sum(x['post_fragmentation']['components'] for x in all_records)})
    print('POSTPROCESS_EXPORTED',len(all_records))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--result',type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--scan',action='store_true');mode.add_argument('--export',action='store_true');a=ap.parse_args()
    scan(a) if a.scan else export(a)


if __name__=='__main__':main()
