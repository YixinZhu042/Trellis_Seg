"""
Pre-process multi-view segmentation maps into sparse latent tokens.

This script encodes 6-view segmentation maps (one-hot) into sparse latent
representations using the TRELLIS.2 Sparse UNet VAE, mapping each pixel
to 3D box surface coordinates.

Usage:
    python datasets/preprocess_mv_seg.py \
        --data_dir /path/to/data \
        --vae_ckpt /path/to/seg_vae_ckpt.pt \
        --output_dir /path/to/data/mv_seg_latent \
        --resolution 32 \
        --num_classes 20
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.modules.sparse.basic import SparseTensor


def box_surface_image_ids(h, w, rotation_result=0):
    """Generate 3D coordinates for pixels on box surface faces."""
    img_ids = torch.zeros(h, w, 3)

    if rotation_result == 0:  # front (z=1)
        img_ids[..., 0] = torch.arange(w)[None, :] / (w - 1)
        img_ids[..., 1] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)
        img_ids[..., 2] = 1
    elif rotation_result == 1:  # right (x=1)
        img_ids[..., 0] = 1
        img_ids[..., 1] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)
        img_ids[..., 2] = (w - torch.arange(1, w + 1)[None, :]) / (w - 1)
    elif rotation_result == 2:  # back (z=0)
        img_ids[..., 0] = (w - torch.arange(1, w + 1)[None, :]) / (w - 1)
        img_ids[..., 1] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)
        img_ids[..., 2] = 0
    elif rotation_result == 3:  # left (x=0)
        img_ids[..., 0] = 0
        img_ids[..., 1] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)
        img_ids[..., 2] = torch.arange(w)[None, :] / (w - 1)
    elif rotation_result == 4:  # top (y=1)
        img_ids[..., 0] = torch.arange(w)[None, :] / (w - 1)
        img_ids[..., 1] = 1
        img_ids[..., 2] = torch.arange(h)[:, None] / (h - 1)
    elif rotation_result == 5:  # bottom (y=0)
        img_ids[..., 0] = torch.arange(w)[None, :] / (w - 1)
        img_ids[..., 1] = 0
        img_ids[..., 2] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)

    return rearrange(img_ids, "h w c -> (h w) c")


def quantize_coords(coords_float, resolution):
    """Quantize float [0,1] coordinates to integer grid coordinates.
    
    Args:
        coords_float: (N, 3) float tensor in [0, 1]
        resolution: grid resolution
    Returns:
        (N, 3) int tensor in [0, resolution-1]
    """
    coords_int = (coords_float * (resolution - 1)).round().int()
    coords_int = coords_int.clamp(0, resolution - 1)
    return coords_int


def deduplicate_voxels(coords_int, feats):
    """Merge features at duplicate voxel positions by averaging.
    
    Args:
        coords_int: (N, 3) int tensor
        feats: (N, C) float tensor
    Returns:
        unique_coords: (M, 3) int tensor
        unique_feats: (M, C) float tensor
    """
    # Convert to linear indices for dedup
    res = coords_int.max().item() + 1
    linear_idx = coords_int[:, 0] * res * res + coords_int[:, 1] * res + coords_int[:, 2]
    
    unique_idx, inverse_idx = torch.unique(linear_idx, return_inverse=True)
    
    # Average features per unique voxel
    unique_feats = torch.zeros(unique_idx.shape[0], feats.shape[1], device=feats.device)
    count = torch.zeros(unique_idx.shape[0], 1, device=feats.device)
    unique_feats.scatter_add_(0, inverse_idx.unsqueeze(1).expand_as(feats), feats)
    count.scatter_add_(0, inverse_idx.unsqueeze(1), torch.ones_like(feats[:, :1]))
    unique_feats = unique_feats / count
    
    # Recover coords
    unique_coords = torch.stack([
        unique_idx // (res * res),
        (unique_idx % (res * res)) // res,
        unique_idx % res
    ], dim=1).int()
    
    return unique_coords, unique_feats


@torch.no_grad()
def encode_mv_seg_maps(seg_maps, vae_encoder, resolution, num_classes, device='cuda'):
    """
    Encode 6-view segmentation maps into a single sparse latent.
    
    Args:
        seg_maps: (6, H, W) integer labels tensor
        vae_encoder: VAE encoder model
        resolution: target grid resolution for quantization
        num_classes: number of seg classes
        device: compute device
    
    Returns:
        coords: (M, 3) int tensor - voxel coordinates
        feats: (M, 32) float tensor - latent features
    """
    n_views, h, w = seg_maps.shape
    
    all_coords = []
    all_feats = []
    
    for view_idx in range(n_views):
        seg_map = seg_maps[view_idx]  # (H, W)
        
        # Get non-background mask
        mask = (seg_map > 0).reshape(-1)  # (H*W,)
        if mask.sum() == 0:
            continue
        
        # One-hot encode
        one_hot = F.one_hot(seg_map.long(), num_classes).float()  # (H, W, C)
        one_hot = rearrange(one_hot, 'h w c -> (h w) c')  # (H*W, C)
        
        # Get 3D positions for this view's pixels
        positions_3d = box_surface_image_ids(h, w, rotation_result=view_idx)  # (H*W, 3)
        
        # Filter by mask
        valid_positions = positions_3d[mask].to(device)  # (N_valid, 3)
        valid_features = one_hot[mask].to(device)        # (N_valid, C)
        
        # Encode through VAE
        # The VAE expects sparse input: convert to SparseTensor
        coords_int = quantize_coords(valid_positions, resolution)
        
        # Create batch coords (batch_idx=0)
        batch_coords = torch.cat([
            torch.zeros(coords_int.shape[0], 1, dtype=torch.int32, device=device),
            coords_int.to(device)
        ], dim=-1)
        
        sparse_input = SparseTensor(coords=batch_coords, feats=valid_features)
        sparse_input._shape = torch.Size([1, valid_features.shape[-1]])
        
        # Encode
        latent = vae_encoder(sparse_input)  # SparseTensor with feats (N, latent_dim)
        
        all_coords.append(latent.coords[:, 1:].cpu())  # Remove batch dim
        all_feats.append(latent.feats.cpu())
    
    if len(all_coords) == 0:
        # Return empty
        return torch.zeros(0, 3, dtype=torch.int32), torch.zeros(0, 32)
    
    # Merge all views
    merged_coords = torch.cat(all_coords, dim=0)
    merged_feats = torch.cat(all_feats, dim=0)
    
    # Deduplicate overlapping voxels (from different views)
    final_coords, final_feats = deduplicate_voxels(merged_coords, merged_feats)
    
    return final_coords, final_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory containing seg_views/')
    parser.add_argument('--vae_ckpt', type=str, required=True, help='Path to seg VAE encoder checkpoint')
    parser.add_argument('--vae_config', type=str, default=None, help='Path to seg VAE config (optional)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output dir (default: data_dir/mv_seg_latent)')
    parser.add_argument('--resolution', type=int, default=32, help='Grid resolution for quantization')
    parser.add_argument('--num_classes', type=int, default=20, help='Number of seg classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'mv_seg_latent')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VAE encoder
    print(f'Loading VAE from {args.vae_ckpt}...')
    # TODO: Initialize and load VAE encoder based on your specific seg VAE architecture
    # For now, placeholder - user needs to provide the VAE model
    # Example:
    # from trellis2 import models
    # vae = models.from_pretrained(args.vae_ckpt)
    # encoder = vae.encoder.to(args.device).eval()
    
    print(f'Placeholder: VAE encoder loading not yet configured.')
    print(f'Please configure the VAE encoder path and architecture.')
    print(f'Expected input: SparseTensor with {args.num_classes}ch one-hot features')
    print(f'Expected output: SparseTensor with 32ch latent features')
    return
    
    # Process all instances
    seg_views_dir = os.path.join(args.data_dir, 'seg_views')
    instances = [f[:-4] for f in os.listdir(seg_views_dir) if f.endswith('.npz')]
    
    print(f'Processing {len(instances)} instances...')
    for instance in tqdm(instances):
        output_path = os.path.join(args.output_dir, f'{instance}.npz')
        if os.path.exists(output_path):
            continue
        
        # Load seg views
        seg_data = np.load(os.path.join(seg_views_dir, f'{instance}.npz'))
        seg_maps = torch.tensor(seg_data['seg_maps']).long()  # (6, H, W)
        
        # Encode
        coords, feats = encode_mv_seg_maps(
            seg_maps, encoder, args.resolution, args.num_classes, args.device
        )
        
        # Save
        np.savez_compressed(
            output_path,
            coords=coords.numpy(),
            feats=feats.numpy()
        )
    
    print(f'Done! Saved to {args.output_dir}')


if __name__ == '__main__':
    main()
