"""Raw reconstruction evaluation, surface failure diagnostics, and artifact export."""
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .data import voxel_tensors, to_device
from .geometry import SurfaceQueries


def classification(target, pred, weight):
    valid = (target > 0) & (weight > 0)
    target, pred, weight = target[valid], pred[valid], weight[valid].astype(np.float64)
    if not len(target):
        return {'accuracy': None, 'miou': None, 'small_part_recall': None}
    iou, small = [], []
    for label in np.unique(target):
        a, b = target == label, pred == label
        intersection = weight[a & b].sum()
        iou.append(intersection / weight[a | b].sum())
        if weight[a].sum() < .01 * weight.sum():
            small.append(intersection / weight[a].sum())
    return {'accuracy': float(weight[target == pred].sum() / weight.sum()),
            'miou': float(np.mean(iou)), 'small_part_recall': float(np.mean(small)) if small else None}


def surface_diagnostics(sample, pred):
    gt, area = sample['face_labels'], sample['face_area'].astype(np.float64)
    valid = (gt > 0) & (area > 0)
    adjacency = sample['adjacency']
    left, right = adjacency.T
    edges_valid = valid[left] & valid[right]
    truth = (gt[left] != gt[right]) & edges_valid
    guess = (pred[left] != pred[right]) & edges_valid
    boundary = float(2 * (truth & guess).sum() / max(1, truth.sum()+guess.sum())) if truth.any() else None
    internal = edges_valid & (gt[left] == gt[right])
    false_internal = float(guess[internal].mean()) if internal.any() else None
    kept = adjacency[pred[left] == pred[right]]
    edge = np.concatenate([kept, kept[:, ::-1]])
    graph = coo_matrix((np.ones(len(edge)), (edge[:, 0], edge[:, 1])), shape=(len(gt), len(gt)))
    _, comp = connected_components(graph, directed=False)
    comp_area = np.bincount(comp, weights=area * valid)
    total = area[valid].sum()
    small = comp_area[comp] < .001 * max(total, 1e-12)
    wrong = (pred != gt) & valid
    near = sample['near_pairs']
    near_gt = gt[near]
    cross = (near_gt > 0).all(1) & (near_gt[:, 0] != near_gt[:, 1])
    separation = float((pred[near[cross, 0]] != pred[near[cross, 1]]).mean()) if cross.any() else None
    return {'boundary_f1': boundary, 'false_internal_boundary_rate': false_internal,
            'isolated_error_area_fraction': float(area[wrong & small].sum() / max(total, 1e-12)),
            'near_part_separation_accuracy': separation,
            'hidden_surface': classification(gt, pred, area * (sample['face_pixels'] == 0))}


@torch.no_grad()
def reconstruct(model, sample, cfg, device, chunk_faces=1024):
    model.eval()
    context = torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') and cfg['train']['precision']=='bf16' else nullcontext()
    with context:
        output = model(voxel_tensors(sample, device), sample_posterior=False)
    query_builder = SurfaceQueries(sample, cfg['geometry'])
    labels, fallback = [], 0
    for start in range(0, len(sample['face_labels']), chunk_faces):
        ids = np.arange(start, min(start+chunk_faces, len(sample['face_labels'])))
        query = query_builder.build(ids)
        fallback += int(query['query_fallback'].sum())
        with (torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') and cfg['train']['precision']=='bf16' else nullcontext()):
            _, score = model.decode_surface(output['features'], output['voxel_logits'], to_device(query, device))
        labels.append(score.argmax(-1).cpu().numpy())
    faces = np.concatenate(labels)
    voxel = output['voxel_logits'].argmax(-1).cpu().numpy()
    truth = sample['voxel_labels']
    row = {'voxel': classification(truth, voxel, np.ones(len(truth))),
           'surface': classification(sample['face_labels'], faces, sample['face_area']),
           'projection': classification(sample['face_labels'], faces, sample['face_pixels']),
           'query_fallback_fraction': fallback / (4 * len(faces)), **surface_diagnostics(sample, faces)}
    if cfg.get('annotation_policy'):
        area=sample['face_area'].astype(np.float64);pixels=sample['face_pixels'].astype(np.float64)
        valid=sample['face_labels']>0
        row['annotation_coverage']={'face_fraction':float(valid.mean()),
            'surface_fraction':float(area[valid].sum()/max(area.sum(),1e-12)),
            'visible_pixel_fraction':float(pixels[valid].sum()/max(pixels.sum(),1)),
            'unannotated_predicted_labels':np.unique(faces[~valid]).tolist()}
    return row, faces, voxel


def summarize(rows):
    def average(key, group=None):
        vals = [(r[group][key] if group else r[key]) for r in rows]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    result = {g: {k:average(k,g) for k in ('accuracy','miou','small_part_recall')}
              for g in ('voxel','surface','projection','hidden_surface')}
    for k in ('boundary_f1','false_internal_boundary_rate','isolated_error_area_fraction',
              'near_part_separation_accuracy','query_fallback_fraction'):
        result[k] = average(k)
    values = [r['surface']['miou'] for r in rows if r['surface']['miou'] is not None]
    result['surface_miou_p10'] = float(np.quantile(values, .1)) if values else None
    if rows and all('annotation_coverage' in row for row in rows):
        result['annotation_coverage']={k:float(np.mean([r['annotation_coverage'][k] for r in rows]))
                                       for k in ('face_fraction','surface_fraction','visible_pixel_fraction')}
    return result


def selection_score(summary):
    def value(v):
        return -1. if v is None else v
    return (value(summary['surface']['miou']), value(summary['surface']['small_part_recall']),
            -value(summary['isolated_error_area_fraction']))


def export_prediction(output, uid, sample, face_labels, voxel_labels, source_path, prediction_dir):
    import trimesh
    from PIL import Image, ImageDraw
    from scripts.refine_seg_projection import export_labels
    from scripts.evaluate_seg_quality import generate_class_colors
    from utils.seg_projection import render_labels, VIEW_NAMES
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(sample['mesh_vertices'], sample['mesh_faces'], process=False)
    classes=sample['voxel_target'].shape[-1]
    corrected='annotation_audit' in sample
    if corrected:
        from .visuals import export_labels as export_corrected,render_palette
        export_corrected(mesh,face_labels,output/f'{uid}.glb',classes)
        export_corrected(mesh,sample['face_labels'],output/f'{uid}_gt.glb',classes)
        palette=render_palette(classes)
    else:
        export_labels(mesh, face_labels, output / f'{uid}.glb')
        palette = np.full((classes+1,3),255,np.uint8)
        palette[1:] = (generate_class_colors(classes)*255).astype(np.uint8)
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_dir / f'{uid}.npz', face_labels=face_labels, voxel_labels=voxel_labels)
    if 'face_raster' in sample:
        raster=sample['face_raster']
    else:
        with np.load(source_path) as cache:
            raster = cache['face_raster']
    canvas = Image.new('RGB',(1536,560),'white')
    draw = ImageDraw.Draw(canvas)
    for row,(name,labels) in enumerate([('GT',sample['face_labels']),('VAE v2',face_labels)]):
        maps = render_labels(labels+1,raster)
        for v,view in enumerate(VIEW_NAMES):
            draw.text((v*256+4,row*280+4),f'{name}: {view}',fill='black')
            canvas.paste(Image.fromarray(palette[maps[v]]).resize((256,256),Image.Resampling.NEAREST),(v*256,row*280+24))
    canvas.save(output / f'{uid}_comparison.png')
    if corrected:
        from .config import atomic_json
        atomic_json(output/f'{uid}_palette.json',{'unknown_label':0,'unknown_color':'gray',
                     'num_classes':classes,'colors':palette[1:].tolist(),
                     'annotation_audit':__import__('json').loads(str(sample['annotation_audit']))})
