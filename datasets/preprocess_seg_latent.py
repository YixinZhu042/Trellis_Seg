"""
Pre-encode 3D segmentation labels into structured latent for DIT training.

This script takes per-voxel segmentation labels and encodes them through
the trained Seg VAE to produce the target latent (x_0) for flow matching.

Input data format:
    {data_dir}/seg_labels/{instance}.npz
        coords: (N, 3) int - voxel coordinates
        labels: (N,) int - per-voxel class labels [0, num_classes)

Output:
    {data_dir}/seg_latent/{instance}.npz
        coords: (M, 3) int - latent voxel coordinates (downsampled)
        feats: (M, 32) float - latent features

Usage:
    python datasets/preprocess_seg_latent.py \
        --data_dir /path/to/data \
        --vae_ckpt /path/to/seg_vae_encoder.pt \
        --num_classes 20
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.modules.sparse.basic import SparseTensor


@torch.no_grad()
def encode_seg_labels(coords, labels, encoder, num_classes, device='cuda'):
    """
    Encode per-voxel segmentation labels to latent space.
    
    Args:
        coords: (N, 3) int numpy array - voxel positions
        labels: (N,) int numpy array - class labels
        encoder: VAE encoder model
        num_classes: number of classes
        device: compute device
    
    Returns:
        latent_coords: (M, 3) int tensor
        latent_feats: (M, 32) float tensor
    """
    coords_t = torch.tensor(coords, dtype=torch.int32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    
    # One-hot encode labels
    one_hot = F.one_hot(labels_t, num_classes).float()  # (N, num_classes)
    
    # Create SparseTensor with batch dim
    batch_coords = torch.cat([
        torch.zeros(coords_t.shape[0], 1, dtype=torch.int32, device=device),
        coords_t
    ], dim=-1)
    
    sparse_input = SparseTensor(coords=batch_coords, feats=one_hot)
    sparse_input._shape = torch.Size([1, num_classes])
    
    # Encode
    latent = encoder(sparse_input)
    
    # Get mean (deterministic encoding for latent dataset)
    # If encoder outputs mean+logvar, take only mean
    latent_feats = latent.feats
    if latent_feats.shape[-1] == 64:  # 2 * latent_channels (mean + logvar)
        latent_feats = latent_feats[:, :32]  # Take mean only
    
    latent_coords = latent.coords[:, 1:]  # Remove batch dim
    
    return latent_coords.cpu(), latent_feats.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--vae_ckpt', type=str, required=True, help='Path to trained seg VAE encoder')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'seg_latent')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load encoder
    print(f'Loading VAE encoder from {args.vae_ckpt}...')
    # TODO: Load your specific VAE encoder
    # Example:
    # from trellis2 import models
    # encoder = models.from_pretrained(args.vae_ckpt).encoder
    # encoder = encoder.to(args.device).eval()
    print('Placeholder: Configure VAE encoder loading.')
    return
    
    # Process
    seg_labels_dir = os.path.join(args.data_dir, 'seg_labels')
    instances = [f[:-4] for f in os.listdir(seg_labels_dir) if f.endswith('.npz')]
    
    print(f'Processing {len(instances)} instances...')
    for instance in tqdm(instances):
        output_path = os.path.join(args.output_dir, f'{instance}.npz')
        if os.path.exists(output_path):
            continue
        
        data = np.load(os.path.join(seg_labels_dir, f'{instance}.npz'))
        coords = data['coords']  # (N, 3)
        labels = data['labels']  # (N,)
        
        latent_coords, latent_feats = encode_seg_labels(
            coords, labels, encoder, args.num_classes, args.device
        )
        
        np.savez_compressed(
            output_path,
            coords=latent_coords.numpy(),
            feats=latent_feats.numpy()
        )
    
    print(f'Done! Saved to {args.output_dir}')


if __name__ == '__main__':
    main()
