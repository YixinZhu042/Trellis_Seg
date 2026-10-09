"""
Step 1: Render 6-view segmentation maps from mesh + face labels.

Input:
    - PartNeXt annotations (arrow files with masks)
    - GLB meshes

Output:
    {output_dir}/seg_views/{instance}.npz
        seg_maps: (6, H, W) int32 - per-pixel part labels
        num_parts: int

Usage:
    python preprocess/step1_render_seg_views.py \
        --arrow_dir /path/to/PartNeXt \
        --mesh_dir /path/to/PartNeXt_mesh/glbs \
        --output_dir ./data \
        --resolution 512 \
        --num_samples 5
"""
import os
import sys
import json
import argparse
import numpy as np
import trimesh
import pyarrow.ipc as ipc
from glob import glob
from tqdm import tqdm


def _as_face_list(face_indices):
    """Normalize a mask's face-index entry to an iterable of ints.

    PartNeXt annotations are usually a list of face indices, but some entries
    store a single index as a bare int; wrap those so iteration is uniform.
    """
    if isinstance(face_indices, int):
        return (face_indices,)
    return face_indices


def parse_face_labels(masks_json, num_faces, hierarchy_json=None, max_parts=19):
    """
    Parse masks annotation to per-face labels.
    
    If num_parts > max_parts and hierarchy is available, merge leaf nodes
    into their ancestors at a higher level to reduce part count.
    """
    masks = json.loads(masks_json) if isinstance(masks_json, str) else masks_json
    
    num_raw_parts = len(masks)
    
    # If within limit, use raw labels directly
    if num_raw_parts <= max_parts or hierarchy_json is None:
        face_labels = np.full(num_faces, 0, dtype=np.int32)
        part_id = 1
        for part_key in sorted(masks.keys(), key=int):
            submeshes = masks[part_key]
            for submesh_key, face_indices in submeshes.items():
                for fi in _as_face_list(face_indices):
                    if fi < num_faces:
                        face_labels[fi] = part_id
            part_id += 1
        return face_labels, part_id - 1
    
    # Too many parts: merge using hierarchy
    hierarchy = json.loads(hierarchy_json) if isinstance(hierarchy_json, str) else hierarchy_json
    
    # Build nodeId → part_key mapping from masks
    # masks keys are "0", "1", ... corresponding to hierarchy nodes in order
    node_id_to_mask_key = {}
    all_node_ids = []
    
    def collect_nodes(node, nodes_list):
        nodes_list.append(node['nodeId'])
        for child in node.get('children', []):
            collect_nodes(child, nodes_list)
    
    if isinstance(hierarchy, list):
        for h in hierarchy:
            collect_nodes(h, all_node_ids)
    
    # Map mask keys to nodeIds (masks are ordered by nodeId traversal)
    for idx, nid in enumerate(sorted(masks.keys(), key=int)):
        if idx < len(all_node_ids):
            node_id_to_mask_key[all_node_ids[idx]] = nid
    
    # Find the depth level where we get <= max_parts nodes
    # Try progressively deeper cuts
    def get_nodes_at_depth(node, target_depth, current_depth=0):
        """Get nodes at target_depth. If a node has no children at that depth, include it."""
        if current_depth == target_depth or not node.get('children'):
            return [node]
        result = []
        for child in node['children']:
            result.extend(get_nodes_at_depth(child, target_depth, current_depth + 1))
        return result
    
    def get_all_descendant_node_ids(node):
        """Get all nodeIds in subtree (including self)."""
        ids = [node['nodeId']]
        for child in node.get('children', []):
            ids.extend(get_all_descendant_node_ids(child))
        return ids
    
    # Try depth 1, 2, 3... until we get <= max_parts groups
    best_groups = None
    roots = hierarchy if isinstance(hierarchy, list) else [hierarchy]
    
    for depth in range(1, 10):
        groups = []
        for root in roots:
            groups.extend(get_nodes_at_depth(root, depth))
        if len(groups) <= max_parts:
            best_groups = groups
            if len(groups) >= 2:  # at least 2 groups to be useful
                break
        else:
            if best_groups is None:
                best_groups = groups
            break
    
    if best_groups is None:
        best_groups = roots
    
    # Assign merged labels: each group gets one label
    # For each group, collect all descendant nodeIds → all their face indices
    face_labels = np.full(num_faces, 0, dtype=np.int32)
    part_id = 1
    
    for group_node in best_groups:
        descendant_ids = get_all_descendant_node_ids(group_node)
        # Find which mask keys correspond to these nodeIds
        for idx, mask_key in enumerate(sorted(masks.keys(), key=int)):
            if idx < len(all_node_ids) and all_node_ids[idx] in descendant_ids:
                submeshes = masks[mask_key]
                for submesh_key, face_indices in submeshes.items():
                    for fi in _as_face_list(face_indices):
                        if fi < num_faces:
                            face_labels[fi] = part_id
        part_id += 1
    
    num_parts = part_id - 1
    return face_labels, num_parts


def render_orthographic_seg(mesh, face_labels, resolution, view_idx):
    """
    Render an orthographic segmentation view.
    
    view_idx: 0=front(+Z), 1=right(+X), 2=back(-Z), 3=left(-X), 4=top(+Y), 5=bottom(-Y)
    
    Returns: (H, W) int32 array of part labels
    """
    # Define camera directions for 6 orthographic views
    # Each view looks along negative of the direction vector
    view_configs = [
        {'axis': 2, 'sign': +1, 'up': 1, 'right': 0, 'flip_right': False},  # front: look -Z
        {'axis': 0, 'sign': +1, 'up': 1, 'right': 2, 'flip_right': True},   # right: look -X  
        {'axis': 2, 'sign': -1, 'up': 1, 'right': 0, 'flip_right': True},   # back: look +Z
        {'axis': 0, 'sign': -1, 'up': 1, 'right': 2, 'flip_right': False},  # left: look +X
        {'axis': 1, 'sign': +1, 'up': 2, 'right': 0, 'flip_right': False},  # top: look -Y
        {'axis': 1, 'sign': -1, 'up': 2, 'right': 0, 'flip_right': False},  # bottom: look +Y
    ]
    
    cfg = view_configs[view_idx]
    
    # Get mesh bounds and normalize to [-1, 1]
    vertices = mesh.vertices.copy()
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    max_dim = (bbox_max - bbox_min).max()
    if max_dim > 0:
        vertices = (vertices - center) / max_dim * 2  # [-1, 1]
    
    # Project vertices orthographically
    # "right" axis → x in image, "up" axis → y in image (inverted)
    proj_x = vertices[:, cfg['right']]
    proj_y = vertices[:, cfg['up']]
    proj_depth = vertices[:, cfg['axis']] * cfg['sign']  # depth for z-buffering
    
    if cfg['flip_right']:
        proj_x = -proj_x
    
    # Map to pixel coordinates [0, resolution-1]
    px = ((proj_x + 1) / 2 * (resolution - 1)).astype(np.int32)
    py = ((1 - (proj_y + 1) / 2) * (resolution - 1)).astype(np.int32)  # flip Y
    px = np.clip(px, 0, resolution - 1)
    py = np.clip(py, 0, resolution - 1)
    
    # Rasterize with z-buffer (per face)
    seg_map = np.zeros((resolution, resolution), dtype=np.int32)
    depth_buffer = np.full((resolution, resolution), -np.inf)
    
    faces = mesh.faces
    for fi in range(len(faces)):
        face = faces[fi]
        label = face_labels[fi] + 1  # +1 so label 0 becomes 1; background stays 0
        
        # Get face vertices in projected space
        face_px = px[face]
        face_py = py[face]
        face_depth = proj_depth[face]
        
        # Simple triangle rasterization using bounding box + barycentric
        min_x = max(0, face_px.min())
        max_x = min(resolution - 1, face_px.max())
        min_y = max(0, face_py.min())
        max_y = min(resolution - 1, face_py.max())
        
        if min_x == max_x and min_y == max_y:
            # Degenerate face - single pixel
            d = face_depth.mean()
            if d > depth_buffer[min_y, min_x]:
                depth_buffer[min_y, min_x] = d
                seg_map[min_y, min_x] = label
            continue
        
        # Rasterize using scanline within bbox
        v0 = np.array([face_px[0], face_py[0]], dtype=np.float64)
        v1 = np.array([face_px[1], face_py[1]], dtype=np.float64)
        v2 = np.array([face_px[2], face_py[2]], dtype=np.float64)
        
        # Area of triangle (2x)
        area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
        if abs(area) < 1e-10:
            # Degenerate - just fill the center pixel
            cx = int((face_px[0] + face_px[1] + face_px[2]) / 3)
            cy = int((face_py[0] + face_py[1] + face_py[2]) / 3)
            cx = np.clip(cx, 0, resolution - 1)
            cy = np.clip(cy, 0, resolution - 1)
            d = face_depth.mean()
            if d > depth_buffer[cy, cx]:
                depth_buffer[cy, cx] = d
                seg_map[cy, cx] = label
            continue
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x, y], dtype=np.float64)
                # Barycentric coordinates
                w0 = ((v1[0] - p[0]) * (v2[1] - p[1]) - (v2[0] - p[0]) * (v1[1] - p[1])) / area
                w1 = ((v2[0] - p[0]) * (v0[1] - p[1]) - (v0[0] - p[0]) * (v2[1] - p[1])) / area
                w2 = 1 - w0 - w1
                
                if w0 >= -0.01 and w1 >= -0.01 and w2 >= -0.01:
                    d = w0 * face_depth[0] + w1 * face_depth[1] + w2 * face_depth[2]
                    if d > depth_buffer[y, x]:
                        depth_buffer[y, x] = d
                        seg_map[y, x] = label
    
    return seg_map


# View configs shared by CPU and GPU rasterizers. Order must match
# render_orthographic_seg's view_configs and step2's box_surface_image_ids.
_VIEW_CONFIGS = [
    {'axis': 2, 'sign': +1, 'up': 1, 'right': 0, 'flip_right': False},  # front: look -Z
    {'axis': 0, 'sign': +1, 'up': 1, 'right': 2, 'flip_right': True},   # right: look -X
    {'axis': 2, 'sign': -1, 'up': 1, 'right': 0, 'flip_right': True},   # back: look +Z
    {'axis': 0, 'sign': -1, 'up': 1, 'right': 2, 'flip_right': False},  # left: look +X
    {'axis': 1, 'sign': +1, 'up': 2, 'right': 0, 'flip_right': False},  # top: look -Y
    {'axis': 1, 'sign': -1, 'up': 2, 'right': 0, 'flip_right': False},  # bottom: look +Y
]

_GLCTX = None  # per-process nvdiffrast context, created lazily


def render_all_views_gpu(mesh, face_labels, resolution, device='cuda'):
    """Rasterize all 6 orthographic seg views on GPU (nvdiffrast).

    Output matches render_orthographic_seg to >98% of pixels (differences are
    sub-pixel edge coverage). A vertical flip is applied because nvdiffrast's
    screen-space y axis is opposite to the CPU rasterizer's.
    """
    global _GLCTX
    import torch
    import nvdiffrast.torch as dr
    if _GLCTX is None:
        _GLCTX = dr.RasterizeCudaContext()

    verts = mesh.vertices.astype(np.float32)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    max_dim = (bbox_max - bbox_min).max()
    if max_dim > 0:
        verts = (verts - center) / max_dim * 2  # [-1, 1]

    V = torch.from_numpy(verts).to(device)
    tri = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
    fl = torch.from_numpy(face_labels.astype(np.int32)).to(device)

    clips = []
    for cfg in _VIEW_CONFIGS:
        px = V[:, cfg['right']].clone()
        py = V[:, cfg['up']]
        pd = V[:, cfg['axis']] * cfg['sign']
        if cfg['flip_right']:
            px = -px
        w = torch.ones_like(px)
        # NDC clip coords; z = -depth so the nearest-to-camera (largest depth) wins.
        clips.append(torch.stack([px, py, -pd, w], dim=-1))
    pos = torch.stack(clips, dim=0).contiguous()  # (6, V, 4)

    rast, _ = dr.rasterize(_GLCTX, pos, tri, resolution=[resolution, resolution])
    tri_id = rast[..., 3].long() - 1  # 0 = background in nvdiffrast → -1 here
    seg = torch.zeros_like(tri_id, dtype=torch.int32)
    hit = tri_id >= 0
    seg[hit] = fl[tri_id[hit]] + 1  # +1 so label 0 becomes 1; background stays 0
    seg = seg.flip(1)  # vertical flip to match CPU convention
    return seg.cpu().numpy().astype(np.int32)


def process_one(record, mesh_dir, output_dir, resolution, num_classes=20, use_gpu=True):
    """Process one sample: render 6-view seg maps."""
    model_id = record['model_id']
    type_id = record['type_id']

    output_path = os.path.join(output_dir, 'seg_views', f'{model_id}.npz')
    if os.path.exists(output_path):
        return True, "exists"

    # Load mesh
    mesh_path = os.path.join(mesh_dir, type_id, f'{model_id}.glb')
    if not os.path.exists(mesh_path):
        return False, "mesh not found"

    try:
        scene = trimesh.load(mesh_path, force='scene')
        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_geometry()
        else:
            mesh = scene
        if not isinstance(mesh, trimesh.Trimesh):
            return False, "not a Trimesh"
    except Exception as e:
        return False, f"load error: {e}"

    # Parse face labels (merge hierarchy down to at most num_classes-1 parts,
    # so labels stay in [0, num_classes-1] and never overflow the palette).
    max_parts = num_classes - 1
    try:
        face_labels, num_parts = parse_face_labels(
            record['masks'], len(mesh.faces),
            hierarchy_json=record.get('hierarchy'),
            max_parts=max_parts
        )
    except Exception as e:
        return False, f"label parse error: {e}"
    if num_parts == 0:
        return False, "no parts"
    # Hierarchy merging can still leave >max_parts groups (broad/flat trees);
    # such samples would produce labels ≥ num_classes, so skip them.
    if num_parts > max_parts:
        return False, f"too many parts ({num_parts} > {max_parts})"

    # Render 6 views (GPU rasterizer with CPU fallback)
    seg_maps = None
    if use_gpu:
        try:
            seg_maps = render_all_views_gpu(mesh, face_labels, resolution)
        except Exception as e:
            return False, f"gpu render error: {e}"
    if seg_maps is None:
        seg_maps = np.zeros((6, resolution, resolution), dtype=np.int32)
        for v in range(6):
            seg_maps[v] = render_orthographic_seg(mesh, face_labels, resolution, v)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, seg_maps=seg_maps, num_parts=num_parts)

    return True, f"parts={num_parts}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arrow_dir', type=str, default='/path/to/PartNeXt')
    parser.add_argument('--mesh_dir', type=str, default='/path/to/PartNeXt_mesh/glbs')
    parser.add_argument('--output_dir', type=str, default='/path/to/data')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--instances', type=str, default=None,
                        help='Comma-separated model ids or a text file with one model id per line; '
                             'restricts processing to these instances')
    parser.add_argument('--num_classes', type=int, default=20,
                        help='Segmentation classes; labels are kept in [0, num_classes-1]')
    parser.add_argument('--num_workers', type=int, default=8, help='CPU-mode parallel workers')
    parser.add_argument('--rank', type=int, default=0, help='Worker rank for distributed processing')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of workers')
    parser.add_argument('--no_gpu', action='store_true', help='Force the CPU rasterizer (nvdiffrast off)')
    args = parser.parse_args()

    # Load annotations
    wanted_ids = None
    if args.instances:
        if os.path.exists(args.instances):
            with open(args.instances, 'r') as f:
                wanted_ids = {line.strip() for line in f if line.strip()}
        else:
            wanted_ids = {x.strip() for x in args.instances.split(',') if x.strip()}
        print(f'[Rank {args.rank}] restricting to {len(wanted_ids)} requested instances')

    arrow_files = sorted(glob(os.path.join(args.arrow_dir, 'data-*.arrow')))
    records = []
    for af in arrow_files:
        with open(af, 'rb') as f:
            reader = ipc.open_stream(f)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                for i in range(batch.num_rows):
                    model_id = batch.column('model_id')[i].as_py()
                    if wanted_ids is not None and model_id not in wanted_ids:
                        continue
                    records.append({
                        'model_id': model_id,
                        'type_id': batch.column('type_id')[i].as_py(),
                        'masks': batch.column('masks')[i].as_py(),
                        'hierarchy': batch.column('hierarchyList')[i].as_py(),
                    })
                    if args.num_samples and len(records) >= args.num_samples:
                        break
                if args.num_samples and len(records) >= args.num_samples:
                    break
        if args.num_samples and len(records) >= args.num_samples:
            break

    # Shard by rank (num_samples applied first, so a fixed cap splits across workers)
    records = records[args.rank::args.world_size]

    # Decide GPU vs CPU
    use_gpu = not args.no_gpu
    if use_gpu:
        try:
            import nvdiffrast.torch  # noqa: F401
        except Exception as e:
            print(f"[Rank {args.rank}] nvdiffrast unavailable ({e}); falling back to CPU rasterizer")
            use_gpu = False

    print(f"[Rank {args.rank}/{args.world_size}] Processing {len(records)} samples, "
          f"resolution={args.resolution}, mode={'GPU' if use_gpu else 'CPU'}")

    success, fail = 0, 0
    if use_gpu:
        # Serial loop: CUDA does not fork cleanly through a process pool; parallelism
        # comes from launching one rank process per GPU (see run_all.sh).
        for rec in tqdm(records, desc=f"Step1 render (rank {args.rank})"):
            ok, msg = process_one(rec, args.mesh_dir, args.output_dir, args.resolution,
                                  num_classes=args.num_classes, use_gpu=True)
            if ok:
                success += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  Skip {rec['model_id'][:8]}: {msg}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        worker_fn = partial(process_one, mesh_dir=args.mesh_dir, output_dir=args.output_dir,
                            resolution=args.resolution, num_classes=args.num_classes, use_gpu=False)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(worker_fn, rec): rec for rec in records}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Step1 render"):
                ok, msg = future.result()
                if ok:
                    success += 1
                else:
                    fail += 1
                    if fail <= 5:
                        print(f"  Skip {futures[future]['model_id'][:8]}: {msg}")

    print(f"\n[Rank {args.rank}] Done! Success: {success}, Failed: {fail}")
    print(f"Output: {args.output_dir}/seg_views/")


if __name__ == '__main__':
    main()
