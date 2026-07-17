"""
Pre-encode 3D meshes into shape latent using TRELLIS.2 Shape VAE.

This script takes 3D meshes (or pre-voxelized data) and encodes them through
the TRELLIS.2 Shape VAE encoder to produce geometry latent representations.
These shape latents serve as geometry conditions for the segmentation DIT.

This is analogous to how TRELLIS.2 texture generation uses shape_slat as
concat_cond — the geometry latent tells the DIT where the 3D structure exists
and provides rich geometric features at each voxel.

Input data format (option A - from TRELLIS.2 pipeline):
    {data_dir}/slat/{instance}.npz
        coords: (N, 3) int - structured latent voxel coordinates
        feats: (N, 32) float - structured latent features
    (If you already have TRELLIS.2 shape latents, just copy/symlink them)

Input data format (option B - from raw mesh):
    {data_dir}/meshes/{instance}.obj/glb/ply
    → voxelize → Shape VAE encode → shape_latent

Output:
    {data_dir}/shape_latent/{instance}.npz
        coords: (M, 3) int - latent voxel coordinates
        feats: (M, 32) float - geometry latent features

Usage:
    # Option A: Copy from existing TRELLIS.2 shape latents
    python datasets/preprocess_shape_latent.py \
        --data_dir /path/to/data \
        --source_slat_dir /path/to/trellis2/slat \
        --mode copy

    # Option B: Encode from meshes (requires Shape VAE)
    python datasets/preprocess_shape_latent.py \
        --data_dir /path/to/data \
        --shape_vae_ckpt /path/to/shape_vae.pt \
        --mode encode
"""
import os
import sys
import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)


def copy_from_existing(data_dir, source_slat_dir, output_dir):
    """
    Copy/symlink existing TRELLIS.2 shape latents.
    
    If you've already run TRELLIS.2's pipeline and have shape latents,
    this is the easiest path - just reuse them.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of instances we need (from seg_latent dir)
    seg_latent_dir = os.path.join(data_dir, 'seg_latent')
    if os.path.isdir(seg_latent_dir):
        instances = [f[:-4] for f in os.listdir(seg_latent_dir) if f.endswith('.npz')]
    else:
        instances = [f[:-4] for f in os.listdir(source_slat_dir) if f.endswith('.npz')]
    
    print(f'Copying {len(instances)} shape latents from {source_slat_dir}...')
    copied, missing = 0, 0
    
    for instance in tqdm(instances):
        src = os.path.join(source_slat_dir, f'{instance}.npz')
        dst = os.path.join(output_dir, f'{instance}.npz')
        
        if os.path.exists(dst):
            copied += 1
            continue
            
        if os.path.exists(src):
            # Symlink for efficiency (or copy if cross-filesystem)
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            if missing <= 5:
                print(f'  Warning: missing {src}')
    
    print(f'Done! Copied: {copied}, Missing: {missing}')


@torch.no_grad()
def encode_from_meshes(data_dir, shape_vae_ckpt, output_dir, device='cuda'):
    """
    Encode meshes through Shape VAE to get geometry latents.
    
    NOTE: This requires the TRELLIS.2 Shape VAE and mesh voxelization pipeline.
    The exact implementation depends on your Shape VAE architecture.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Loading Shape VAE from {shape_vae_ckpt}...')
    
    # TODO: Load Shape VAE encoder
    # The Shape VAE is part of TRELLIS.2's pipeline.
    # Example loading (adjust based on actual TRELLIS.2 API):
    #
    # from trellis2.pipelines import TrellisImageTo3DPipeline
    # pipeline = TrellisImageTo3DPipeline.from_pretrained(...)
    # shape_vae_encoder = pipeline.models['shape_slat_encoder']
    #
    # OR if you have the encoder weights directly:
    # from trellis2.models import SparseStructureVaeEncoder
    # encoder = SparseStructureVaeEncoder(**encoder_args)
    # encoder.load_state_dict(torch.load(shape_vae_ckpt))
    # encoder = encoder.to(device).eval()
    
    print('ERROR: Shape VAE encoder loading not yet configured.')
    print('Please either:')
    print('  1. Use --mode copy with --source_slat_dir pointing to existing TRELLIS.2 shape latents')
    print('  2. Implement the Shape VAE encoder loading in this script')
    print('')
    print('If using TRELLIS.2 pipeline, shape latents are typically at:')
    print('  {trellis2_output}/slat/{instance}.npz')
    return
    
    # Process meshes
    mesh_dir = os.path.join(data_dir, 'meshes')
    instances = [f.rsplit('.', 1)[0] for f in os.listdir(mesh_dir) 
                 if f.endswith(('.obj', '.glb', '.ply', '.stl'))]
    
    print(f'Processing {len(instances)} meshes...')
    for instance in tqdm(instances):
        output_path = os.path.join(output_dir, f'{instance}.npz')
        if os.path.exists(output_path):
            continue
        
        # TODO: Voxelize mesh and encode
        # mesh_path = os.path.join(mesh_dir, f'{instance}.obj')
        # sparse_voxels = voxelize_mesh(mesh_path, resolution=...)
        # latent = encoder(sparse_voxels)
        # coords = latent.coords[:, 1:].cpu().numpy()
        # feats = latent.feats.cpu().numpy()
        # np.savez_compressed(output_path, coords=coords, feats=feats)
        pass
    
    print(f'Done!')


def validate_shape_latents(data_dir, output_dir):
    """Validate shape latents match seg_latent instances."""
    seg_latent_dir = os.path.join(data_dir, 'seg_latent')
    
    if not os.path.isdir(seg_latent_dir):
        print('Warning: seg_latent dir not found, skipping validation')
        return
    
    seg_instances = set(f[:-4] for f in os.listdir(seg_latent_dir) if f.endswith('.npz'))
    shape_instances = set(f[:-4] for f in os.listdir(output_dir) if f.endswith('.npz'))
    
    missing = seg_instances - shape_instances
    extra = shape_instances - seg_instances
    
    print(f'\nValidation:')
    print(f'  seg_latent instances: {len(seg_instances)}')
    print(f'  shape_latent instances: {len(shape_instances)}')
    print(f'  Matched: {len(seg_instances & shape_instances)}')
    
    if missing:
        print(f'  MISSING shape_latent for {len(missing)} instances!')
        for m in list(missing)[:5]:
            print(f'    - {m}')
    
    if extra:
        print(f'  Extra shape_latent (no matching seg): {len(extra)}')
    
    # Check a sample
    sample = list(seg_instances & shape_instances)[:3]
    for inst in sample:
        seg = np.load(os.path.join(seg_latent_dir, f'{inst}.npz'))
        shape = np.load(os.path.join(output_dir, f'{inst}.npz'))
        print(f'\n  Sample "{inst}":')
        print(f'    seg_latent:   coords {seg["coords"].shape}, feats {seg["feats"].shape}')
        print(f'    shape_latent: coords {shape["coords"].shape}, feats {shape["feats"].shape}')
        
        # Check if coords match (they should for same instance)
        if np.array_equal(seg['coords'], shape['coords']):
            print(f'    coords: MATCH ✓')
        else:
            print(f'    coords: DIFFERENT (seg={seg["coords"].shape[0]} vs shape={shape["coords"].shape[0]} tokens)')
            print(f'    Note: This is OK if shape has more/fewer tokens than seg')


def main():
    parser = argparse.ArgumentParser(description='Pre-encode shape latents for geometry conditioning')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output dir (default: data_dir/shape_latent)')
    parser.add_argument('--mode', type=str, choices=['copy', 'encode', 'validate'], 
                        default='copy', help='Mode: copy existing latents, encode from mesh, or validate')
    parser.add_argument('--source_slat_dir', type=str, default=None,
                        help='Source dir with existing TRELLIS.2 shape latents (for copy mode)')
    parser.add_argument('--shape_vae_ckpt', type=str, default=None,
                        help='Shape VAE checkpoint path (for encode mode)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'shape_latent')
    
    if args.mode == 'copy':
        if args.source_slat_dir is None:
            print('Error: --source_slat_dir required for copy mode')
            print('This should point to TRELLIS.2 shape latent directory, e.g.:')
            print('  /path/to/trellis2_data/slat/')
            return
        copy_from_existing(args.data_dir, args.source_slat_dir, args.output_dir)
        validate_shape_latents(args.data_dir, args.output_dir)
        
    elif args.mode == 'encode':
        if args.shape_vae_ckpt is None:
            print('Error: --shape_vae_ckpt required for encode mode')
            return
        encode_from_meshes(args.data_dir, args.shape_vae_ckpt, args.output_dir, args.device)
        
    elif args.mode == 'validate':
        if not os.path.isdir(args.output_dir):
            print(f'Error: {args.output_dir} does not exist')
            return
        validate_shape_latents(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
