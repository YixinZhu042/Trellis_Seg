"""Independent raw-annotation index and corrected sample construction."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .annotations import ANNOTATION_POLICY, parse_raw_mesh
from .config import atomic_json


def index_annotations(cfg):
    import pyarrow.ipc as ipc
    from .annotations import canonical_masks
    from .config import digest
    selection_path = Path(cfg['source_dir'])/'selection.json'
    selected = json.loads(selection_path.read_text())['samples']
    expected = {r['model_id']:r for r in selected}
    if len(expected) != len(selected):
        raise ValueError('Duplicate split IDs')
    root = Path(cfg['annotation_dir'])
    root.mkdir(parents=True,exist_ok=True)
    (root/'records').mkdir(exist_ok=True)
    # Rebuild only missing identical records; never overwrite changed annotation.
    records = {}
    for arrow in sorted(Path(cfg['arrow_dir']).glob('data-*.arrow')):
        with arrow.open('rb') as handle:
            for batch in ipc.open_stream(handle):
                for i,uid in enumerate(batch.column('model_id').to_pylist()):
                    if uid not in expected:
                        continue
                    if uid in records:
                        raise ValueError(f'Duplicate annotation record: {uid}')
                    rec={'instance':uid,'type_id':batch.column('type_id')[i].as_py(),
                         'masks':batch.column('masks')[i].as_py(),
                         'hierarchy':batch.column('hierarchyList')[i].as_py(),
                         'mesh_face_num':batch.column('mesh_face_num')[i].as_py()}
                    masks=canonical_masks(rec['masks'])
                    if rec['type_id'] != expected[uid]['type_id']:
                        raise ValueError('Raw annotation category differs from selected object')
                    mesh=Path(cfg['mesh_dir'])/rec['type_id']/f'{uid}.glb'
                    stat=mesh.stat()
                    path=root/'records'/f'{uid}.json'
                    if path.exists():
                        if json.loads(path.read_text())!=rec:
                            raise ValueError(f'Raw annotation changed: {path}; use a new index')
                    else:
                        atomic_json(path,rec)
                    ann_stat=path.stat()
                    source=Path(cfg['source_dir'])/'samples'/f'{uid}.npz'
                    ss=source.stat()
                    records[uid]={'instance':uid,'split':expected[uid]['split'],'type_id':rec['type_id'],
                                  'source':str(source),'source_bytes':ss.st_size,'source_mtime_ns':ss.st_mtime_ns,
                                  'mesh_path':str(mesh),'mesh_bytes':stat.st_size,'mesh_mtime_ns':stat.st_mtime_ns,
                                  'annotation_path':str(path),'annotation_bytes':ann_stat.st_size,
                                  'annotation_mtime_ns':ann_stat.st_mtime_ns,'annotation_digest':digest(rec),
                                  'raw_parts':len(masks)}
        print(f'Indexed {len(records)}/{len(expected)} raw objects from {arrow.name}',flush=True)
    if records.keys()!=expected.keys():
        raise ValueError('Raw annotation index is incomplete')
    payload={'annotation_policy':ANNOTATION_POLICY,
             'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
             'samples':sorted(records.values(),key=lambda r:r['instance'])}
    atomic_json(root/'manifest.json',payload)
    return payload


def raw_source_rows(cfg):
    path=Path(cfg['annotation_dir'])/'manifest.json'
    if not path.is_file():
        raise FileNotFoundError('Run prepare_seg_vae_v2.py --index_annotations with the parts64 config first')
    indexed=json.loads(path.read_text())
    selection_path=Path(cfg['source_dir'])/'selection.json'
    if indexed['annotation_policy']!=cfg['annotation_policy'] or indexed['selection_sha256']!=hashlib.sha256(selection_path.read_bytes()).hexdigest():
        raise ValueError('Raw annotation policy/split index changed')
    rows=indexed['samples']
    selected=json.loads(selection_path.read_text())['samples']
    if len(rows)!=len(selected) or {r['instance']:r['split'] for r in rows}!={r['model_id']:r['split'] for r in selected}:
        raise ValueError('Raw index is incomplete or duplicates split IDs')
    return rows


def check_raw_source(row):
    for field,prefix in [('mesh_path','mesh'),('annotation_path','annotation')]:
        stat=Path(row[field]).stat()
        if stat.st_size!=row[prefix+'_bytes'] or stat.st_mtime_ns!=row[prefix+'_mtime_ns']:
            raise ValueError(f'Changed raw source: {row[field]}')


def raw_source_identity(row):
    return {k:row[k] for k in ('mesh_path','mesh_bytes','mesh_mtime_ns','annotation_path',
                              'annotation_bytes','annotation_mtime_ns','annotation_digest')}


def corrected_geometry(row,cfg):
    """Read raw GT, reuse old visibility only after checking face order and geometry."""
    from .config import digest
    from utils.seg_projection import rasterize_face_ids
    check_raw_source(row)
    record=json.loads(Path(row['annotation_path']).read_text())
    if digest(record)!=row['annotation_digest']:
        raise ValueError('Raw annotation content changed')
    mesh,labels,audit=parse_raw_mesh(row['mesh_path'],record,cfg['num_classes']-1)
    with np.load(row['source'],allow_pickle=False) as source:
        original_triangles=source['mesh_vertices'][source['mesh_faces']]
        same=(original_triangles.shape==mesh.triangles.shape and
              np.allclose(original_triangles,mesh.triangles,rtol=1e-6,atol=max(float(mesh.extents.max()),1e-12)*1e-6))
        if same:
            # Use the established float32 normalization/raster; labels always come from raw annotations.
            vertices=source['mesh_vertices'].astype(np.float32)
            faces=source['mesh_faces'].astype(np.int32)
            raster=source['face_raster'].astype(np.int32)
            pixels=source['face_pixels'].astype(np.int32)
            old_labels=source['face_labels']
            audit['legacy_face_label_changes']=int((labels!=old_labels).sum())
            audit['legacy_unlabeled_faces']=int((old_labels==0).sum())
        else:
            vertices=np.asarray(mesh.vertices,np.float32)
            faces=np.asarray(mesh.faces,np.int32)
            raster=rasterize_face_ids(trimesh_mesh(vertices,faces),(512,512),'cpu')
            pixels=np.bincount(raster[raster>=0],minlength=len(faces)).astype(np.int32)
            audit['legacy_face_label_changes']=None
    audit['visibility_reused_after_geometry_check']=same
    audit['labeled_visible_pixel_fraction']=float(pixels[labels>0].sum()/max(1,pixels.sum()))
    return vertices,faces,labels,raster,pixels,audit


def trimesh_mesh(vertices,faces):
    import trimesh
    return trimesh.Trimesh(vertices,faces,process=False)
