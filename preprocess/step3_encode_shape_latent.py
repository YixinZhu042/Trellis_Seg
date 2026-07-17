"""
Step 3: Encode mesh → shape_latent using TRELLIS.2 原版 pipeline.

Uses o_voxel for FlexiDualGrid voxelization + FlexiDualGridVaeEncoder,
保持与 TRELLIS.2 完全一致的 shape latent 表示。

Input:
    GLB meshes from PartNeXt

Output:
    {data_dir}/shape_latent/{instance}.npz  (coords: (M,3), feats: (M,32))

Usage:
    python preprocess/step3_encode_shape_latent.py \
        --data_dir ./data \
        --mesh_dir /path/to/PartNeXt_mesh/glbs \
        --resolution 512 \
        --num_samples 5
"""
import os
import sys
import argparse
import numpy as np
import torch
import trimesh
from tqdm import tqdm

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

import trellis2.models as models
import trellis2.modules.sparse as sp
import o_voxel

torch.set_grad_enabled(False)

SHAPE_ENC_PATH = '/path/to/trellis2_ckpts/shape_enc'


@torch.no_grad()
def encode_shape(mesh_path, encoder, resolution, device):
    """
    Encode mesh → shape_latent using o_voxel FlexiDualGrid + FlexiDualGridVaeEncoder.
    Exactly matches TRELLIS.2 data_toolkit pipeline (dual_grid.py + encode_shape_latent.py).
    """
    from o_voxel.convert import mesh_to_flexible_dual_grid
    
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    
    # Normalize to [-0.5, 0.5] (same as TRELLIS.2 dual_grid.py)
    v_min = vertices.min(dim=0)[0]
    v_max = vertices.max(dim=0)[0]
    center = (v_min + v_max) / 2
    scale = 0.99999 / (v_max - v_min).max()
    vertices = (vertices - center) * scale
    
    # FlexiDualGrid voxelization
    voxel_indices, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        vertices, faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
    )
    
    # Post-process (same as TRELLIS.2)
    dual_vertices = dual_vertices * resolution - voxel_indices
    dual_vertices = torch.clamp(dual_vertices, 0, 1)
    dual_vertices_uint8 = (dual_vertices * 255).type(torch.uint8)
    intersected_packed = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
    
    # Build encoder inputs (same as TRELLIS.2 encode_shape_latent.py loader)
    coords = voxel_indices.int()
    batch_coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=-1)
    
    vertices_feats = (dual_vertices_uint8.float() / 255.0)  # (N, 3) float
    vertices_sp = sp.SparseTensor(vertices_feats, batch_coords)
    
    intersected_bits = torch.cat([
        intersected_packed % 2,
        intersected_packed // 2 % 2,
        intersected_packed // 4 % 2,
    ], dim=-1).bool()
    intersected_sp = vertices_sp.replace(intersected_bits)
    
    # Encode
    vertices_sp = vertices_sp.to(device)
    intersected_sp = intersected_sp.to(device)
    z = encoder(vertices_sp, intersected_sp)
    
    return z.coords[:, 1:].cpu().numpy().astype(np.uint8), z.feats.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/path/to/data')
    parser.add_argument('--mesh_dir', type=str,
                        default='/path/to/PartNeXt_mesh/glbs')
    parser.add_argument('--encoder_path', type=str, default=SHAPE_ENC_PATH)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--rank', type=int, default=0, help='Worker rank for distributed processing')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of workers')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    output_dir = os.path.join(args.data_dir, 'shape_latent')
    os.makedirs(output_dir, exist_ok=True)
    
    # Multi-GPU: set device by rank
    if args.world_size > 1:
        args.device = f'cuda:{args.rank % torch.cuda.device_count()}'
    
    # Load Shape VAE encoder (FlexiDualGridVaeEncoder)
    print(f"[Rank {args.rank}/{args.world_size}] Loading Shape VAE encoder on {args.device}...")
    encoder = models.from_pretrained(args.encoder_path).eval().to(args.device)
    print(f"Shape encoder loaded: {type(encoder).__name__}")
    
    # Get instances from seg_views
    seg_views_dir = os.path.join(args.data_dir, 'seg_views')
    if not os.path.isdir(seg_views_dir):
        print(f"Error: {seg_views_dir} not found. Run step1 first.")
        return
    
    target_instances = sorted([f[:-4] for f in os.listdir(seg_views_dir) if f.endswith('.npz')])
    if args.num_samples:
        target_instances = target_instances[:args.num_samples]
    
    # Shard by rank
    target_instances = target_instances[args.rank::args.world_size]
    
    # Build id→type mapping
    import pyarrow.ipc as ipc
    from glob import glob
    
    arrow_dir = '/path/to/PartNeXt'
    id_to_type = {}
    for af in sorted(glob(os.path.join(arrow_dir, 'data-*.arrow'))):
        with open(af, 'rb') as f:
            reader = ipc.open_stream(f)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                for i in range(batch.num_rows):
                    mid = batch.column('model_id')[i].as_py()
                    tid = batch.column('type_id')[i].as_py()
                    id_to_type[mid] = tid
        if len(id_to_type) > 50000:
            break
    
    print(f"Processing {len(target_instances)} instances (resolution={args.resolution})...")
    success, fail = 0, 0
    
    # Build work list (skip existing)
    work_list = []
    for instance in target_instances:
        out_path = os.path.join(output_dir, f'{instance}.npz')
        if os.path.exists(out_path):
            success += 1
            continue
        type_id = id_to_type.get(instance)
        if not type_id:
            fail += 1
            continue
        mesh_path = os.path.join(args.mesh_dir, type_id, f'{instance}.glb')
        if not os.path.exists(mesh_path):
            fail += 1
            continue
        work_list.append((instance, mesh_path))
    
    print(f"  Skipped {success} existing, {fail} missing. Encoding {len(work_list)}...")
    
    for instance, mesh_path in tqdm(work_list, desc="Step3 encode shape"):
        out_path = os.path.join(output_dir, f'{instance}.npz')
        try:
            coords, feats = encode_shape(mesh_path, encoder, args.resolution, args.device)
            np.savez_compressed(out_path, coords=coords, feats=feats)
            success += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  Error {instance[:8]}: {e}")
            torch.cuda.empty_cache()
    
    print(f"\nDone! Success: {success}, Failed: {fail}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
