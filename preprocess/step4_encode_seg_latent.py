"""
Step 4: Encode 3D voxel part labels → seg_latent (training target x₀).

Pipeline:
  1. Load mesh + face annotations → voxelize with per-voxel labels
  2. Labels → color encoding (6ch)
  3. Build SparseTensor → Texture VAE encode → seg_latent (32ch)

Input:
    PartNeXt annotations + meshes (same as step1)

Output:
    {data_dir}/seg_latent/{instance}.npz  (coords: (M,3), feats: (M,32))

Usage:
    python preprocess/step4_encode_seg_latent.py \
        --data_dir ./data \
        --resolution 512 \
        --num_samples 5
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import trimesh
import pyarrow.ipc as ipc
from glob import glob
from tqdm import tqdm

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

import trellis2.models as models
import trellis2.modules.sparse as sp

TEX_ENC_PATH = '/path/to/trellis2_ckpts/tex_enc'
SEG_VAE_ENC_PATH = './checkpoints/seg_vae/encoder_step0001000.pt'


def generate_class_colors(num_classes):
    """Generate a maximally-distinguishable palette for segmentation."""
    BASE_PALETTE = np.array([
        [0.900, 0.098, 0.294],  # red
        [0.235, 0.706, 0.294],  # green
        [0.263, 0.388, 0.847],  # blue
        [0.957, 0.831, 0.118],  # yellow
        [0.522, 0.078, 0.694],  # purple
        [0.957, 0.522, 0.114],  # orange
        [0.275, 0.941, 0.941],  # cyan
        [0.941, 0.196, 0.902],  # magenta
        [0.824, 0.961, 0.235],  # lime
        [0.980, 0.745, 0.831],  # pink
        [0.000, 0.502, 0.502],  # teal
        [0.902, 0.745, 1.000],  # lavender
        [0.667, 0.431, 0.157],  # brown
        [1.000, 0.980, 0.784],  # beige
        [0.502, 0.000, 0.000],  # maroon
        [0.667, 1.000, 0.765],  # mint
        [0.502, 0.502, 0.000],  # olive
        [1.000, 0.843, 0.706],  # apricot
        [0.000, 0.000, 0.502],  # navy
        [0.502, 0.502, 0.502],  # grey
    ], dtype=np.float32)

    if num_classes <= 20:
        return BASE_PALETTE[:num_classes]
    colors = np.zeros((num_classes, 3), dtype=np.float32)
    colors[:20] = BASE_PALETTE
    for i in range(20, num_classes):
        colors[i] = BASE_PALETTE[i % 20] * 0.6
    return colors


def _as_face_list(face_indices):
    """Normalize a mask's face-index entry to an iterable of ints (some are bare int)."""
    if isinstance(face_indices, int):
        return (face_indices,)
    return face_indices


def parse_face_labels(masks_json, num_faces, hierarchy_json=None, max_parts=19):
    """
    Parse masks annotation to per-face labels.
    If num_parts > max_parts and hierarchy is available, merge using hierarchy.
    """
    masks = json.loads(masks_json) if isinstance(masks_json, str) else masks_json
    num_raw_parts = len(masks)
    
    if num_raw_parts <= max_parts or hierarchy_json is None:
        face_labels = np.full(num_faces, 0, dtype=np.int32)
        part_id = 1
        for part_key in sorted(masks.keys(), key=int):
            for submesh_key, face_indices in masks[part_key].items():
                for fi in _as_face_list(face_indices):
                    if fi < num_faces:
                        face_labels[fi] = part_id
            part_id += 1
        return face_labels, part_id - 1
    
    # Merge using hierarchy
    hierarchy = json.loads(hierarchy_json) if isinstance(hierarchy_json, str) else hierarchy_json
    
    all_node_ids = []
    def collect_nodes(node, nodes_list):
        nodes_list.append(node['nodeId'])
        for child in node.get('children', []):
            collect_nodes(child, nodes_list)
    if isinstance(hierarchy, list):
        for h in hierarchy:
            collect_nodes(h, all_node_ids)
    
    def get_nodes_at_depth(node, target_depth, current_depth=0):
        if current_depth == target_depth or not node.get('children'):
            return [node]
        result = []
        for child in node['children']:
            result.extend(get_nodes_at_depth(child, target_depth, current_depth + 1))
        return result
    
    def get_all_descendant_node_ids(node):
        ids = [node['nodeId']]
        for child in node.get('children', []):
            ids.extend(get_all_descendant_node_ids(child))
        return ids
    
    roots = hierarchy if isinstance(hierarchy, list) else [hierarchy]
    best_groups = None
    for depth in range(1, 10):
        groups = []
        for root in roots:
            groups.extend(get_nodes_at_depth(root, depth))
        if len(groups) <= max_parts:
            best_groups = groups
            if len(groups) >= 2:
                break
        else:
            if best_groups is None:
                best_groups = groups
            break
    if best_groups is None:
        best_groups = roots
    
    face_labels = np.full(num_faces, 0, dtype=np.int32)
    part_id = 1
    for group_node in best_groups:
        descendant_ids = set(get_all_descendant_node_ids(group_node))
        for idx, mask_key in enumerate(sorted(masks.keys(), key=int)):
            if idx < len(all_node_ids) and all_node_ids[idx] in descendant_ids:
                for submesh_key, face_indices in masks[mask_key].items():
                    for fi in _as_face_list(face_indices):
                        if fi < num_faces:
                            face_labels[fi] = part_id
        part_id += 1
    
    return face_labels, part_id - 1


def _parse_face_labels_simple(masks_json, num_faces):
    """Simple version without hierarchy (kept for reference)."""
    masks = json.loads(masks_json) if isinstance(masks_json, str) else masks_json
    face_labels = np.full(num_faces, 0, dtype=np.int32)
    part_id = 1
    for part_key in sorted(masks.keys(), key=int):
        for submesh_key, face_indices in masks[part_key].items():
            for fi in _as_face_list(face_indices):
                if fi < num_faces:
                    face_labels[fi] = part_id
        part_id += 1
    return face_labels, part_id - 1


def voxelize_with_labels(mesh, face_labels, resolution=256, num_samples=None):
    """Voxelize mesh with per-voxel part labels.
    
    Note: num_samples controls the sampling density. Too many samples relative
    to resolution creates too many unique voxels, making VAE compression ratio 
    too high and destroying reconstruction quality.
    Rule of thumb: target ~5k-30k unique voxels for resolution=512.
    """
    if num_samples is None:
        # Adaptive: ~10 points per face, capped to keep unique voxels reasonable
        num_samples = min(len(mesh.faces) * 10, 100000)
    points, face_idx = mesh.sample(num_samples, return_index=True)
    point_labels = face_labels[face_idx]

    # Normalize
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = (bbox_max - bbox_min).max()
    if bbox_size < 1e-8:
        return None, None
    center = (bbox_min + bbox_max) / 2
    points_norm = (points - center) / bbox_size * (resolution - 3) + resolution / 2
    coords = points_norm.astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)
    
    # Deduplicate with majority vote (vectorized)
    coord_hash = coords[:, 0].astype(np.int64) * resolution * resolution + \
                 coords[:, 1].astype(np.int64) * resolution + coords[:, 2].astype(np.int64)
    unique_hash, inverse, counts = np.unique(coord_hash, return_inverse=True, return_counts=True)

    n_unique = len(unique_hash)

    # Recover coords from hash
    unique_coords = np.stack([
        unique_hash // (resolution * resolution),
        (unique_hash % (resolution * resolution)) // resolution,
        unique_hash % resolution
    ], axis=1).astype(np.int32)

    # Majority vote via sparse matrix (voxel × num_labels)
    max_label = int(point_labels.max()) + 1
    from scipy.sparse import csr_matrix
    vote_matrix = csr_matrix(
        (np.ones(len(point_labels), dtype=np.int32), (inverse, point_labels)),
        shape=(n_unique, max_label)
    )
    unique_labels = np.asarray(vote_matrix.argmax(axis=1)).flatten().astype(np.int32)
    
    # Subsample if too many voxels (VAE compression ratio must stay reasonable)
    # Target: ~5k-30k voxels for good reconstruction at resolution=512.
    # Seeded so preprocessing is reproducible; log how many voxels are dropped.
    max_voxels = 30000
    if len(unique_coords) > max_voxels:
        rng = np.random.default_rng(seed=len(unique_coords))
        idx = rng.choice(len(unique_coords), max_voxels, replace=False)
        print(f'    [voxelize] subsampling {len(unique_coords)} → {max_voxels} voxels '
              f'({len(unique_coords) - max_voxels} dropped)')
        unique_coords = unique_coords[idx]
        unique_labels = unique_labels[idx]
    
    return unique_coords, unique_labels


def labels_to_onehot(labels, num_classes):
    """Convert integer labels → one-hot (N, num_classes) float."""
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Segmentation labels must be integers')
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(f'Labels must be in [0, {num_classes - 1}]')
    feats = np.zeros((len(labels), num_classes), dtype=np.float32)
    feats[np.arange(len(labels)), labels] = 1.0
    return feats


@torch.no_grad()
def encode_seg_latent(coords, labels, encoder, num_classes, device):
    """Encode voxelized labels → seg_latent using Seg VAE (one-hot 20ch input)."""
    feats_onehot = labels_to_onehot(labels, num_classes)

    coords_t = torch.tensor(coords, dtype=torch.int32, device=device)
    feats_t = torch.tensor(feats_onehot, dtype=torch.float32, device=device)

    batch_coords = torch.cat([
        torch.zeros(coords_t.shape[0], 1, dtype=torch.int32, device=device),
        coords_t
    ], dim=-1)

    x = sp.SparseTensor(feats_t, batch_coords)
    z = encoder(x)

    return z.coords[:, 1:].cpu().numpy().astype(np.uint8), z.feats.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/path/to/data')
    parser.add_argument('--arrow_dir', type=str, default='/path/to/PartNeXt')
    parser.add_argument('--mesh_dir', type=str, default='/path/to/PartNeXt_mesh/glbs')
    parser.add_argument('--encoder_path', type=str, default=SEG_VAE_ENC_PATH,
                        help='Seg VAE encoder checkpoint (.pt)')
    parser.add_argument('--encoder_config', type=str, default='configs/seg_vae_train.json',
                        help='Config with encoder model args')
    parser.add_argument('--resolution', type=int, default=512, help='Voxelization resolution')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--rank', type=int, default=0, help='Worker rank for distributed processing')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of workers')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    output_dir = os.path.join(args.data_dir, 'seg_latent')
    os.makedirs(output_dir, exist_ok=True)
    
    # Multi-GPU: set device by rank
    if args.world_size > 1:
        args.device = f'cuda:{args.rank % torch.cuda.device_count()}'
    
    # Load Seg VAE encoder
    print(f"[Rank {args.rank}/{args.world_size}] Loading Seg VAE encoder on {args.device}...")
    import json as _json
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, project_root)
    enc_cfg_path = os.path.join(project_root, args.encoder_config)
    enc_cfg = _json.load(open(enc_cfg_path))['models']['encoder']
    from models import get_model
    encoder = get_model(enc_cfg['name'], **enc_cfg['args'])
    encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu', weights_only=True))
    encoder.convert_to_fp16()
    encoder = encoder.eval().to(args.device)
    print(f"Seg VAE encoder loaded (in_channels={enc_cfg['args']['in_channels']}).")
    
    # Get instances we need (from seg_views)
    seg_views_dir = os.path.join(args.data_dir, 'seg_views')
    if not os.path.isdir(seg_views_dir):
        print(f"Error: {seg_views_dir} not found. Run step1 first.")
        return
    target_instances = set(f[:-4] for f in os.listdir(seg_views_dir) if f.endswith('.npz'))
    
    # Load PartNeXt annotations
    arrow_files = sorted(glob(os.path.join(args.arrow_dir, 'data-*.arrow')))
    records = {}
    for af in arrow_files:
        with open(af, 'rb') as f:
            reader = ipc.open_stream(f)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                for i in range(batch.num_rows):
                    mid = batch.column('model_id')[i].as_py()
                    if mid in target_instances:
                        records[mid] = {
                            'type_id': batch.column('type_id')[i].as_py(),
                            'masks': batch.column('masks')[i].as_py(),
                            'hierarchy': batch.column('hierarchyList')[i].as_py(),
                        }
        if len(records) >= len(target_instances):
            break
    
    instances = sorted(records.keys())
    if args.num_samples:
        instances = instances[:args.num_samples]
    
    # Shard by rank
    instances = instances[args.rank::args.world_size]
    
    print(f"Processing {len(instances)} instances...")
    success, fail = 0, 0
    
    # Pre-load meshes in parallel threads while GPU encodes
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    
    load_queue = Queue(maxsize=16)
    
    def loader(instance):
        rec = records[instance]
        mesh_path = os.path.join(args.mesh_dir, rec['type_id'], f'{instance}.glb')
        if not os.path.exists(mesh_path):
            load_queue.put((instance, None, None, None, None))
            return
        try:
            scene = trimesh.load(mesh_path, force='scene')
            mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
            if not isinstance(mesh, trimesh.Trimesh):
                load_queue.put((instance, None, None, None, None))
                return
            # Merge to at most num_classes-1 parts so labels stay in [0, num_classes-1]
            # (must match step1's rule, or seg_latent labels won't align with seg_views).
            max_parts = args.num_classes - 1
            face_labels, num_parts = parse_face_labels(
                rec['masks'], len(mesh.faces), hierarchy_json=rec.get('hierarchy'), max_parts=max_parts)
            if num_parts > max_parts:
                load_queue.put((instance, None, None, None, f"too many parts ({num_parts} > {max_parts})"))
                return
            coords, labels = voxelize_with_labels(mesh, face_labels, args.resolution)
            load_queue.put((instance, coords, labels, num_parts, None))
        except Exception as e:
            load_queue.put((instance, None, None, None, str(e)))
    
    loader_executor = ThreadPoolExecutor(max_workers=8)
    loader_executor.map(loader, instances)
    
    for _ in tqdm(range(len(instances)), desc="Step4 encode seg_latent"):
        instance, coords, labels, num_parts, err = load_queue.get()
        
        out_path = os.path.join(output_dir, f'{instance}.npz')
        if os.path.exists(out_path):
            success += 1
            continue
        
        if coords is None:
            fail += 1
            if fail <= 5 and err:
                print(f"  Error {instance[:8]}: {err}")
            continue
        
        try:
            lat_coords, lat_feats = encode_seg_latent(
                coords, labels, encoder, args.num_classes, args.device
            )
            
            np.savez_compressed(out_path, coords=lat_coords, feats=lat_feats)
            success += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  Error {instance}: {e}")
            torch.cuda.empty_cache()
    
    loader_executor.shutdown(wait=False)
    print(f"\nDone! Success: {success}, Failed: {fail}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
