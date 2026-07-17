"""
Compute normalization statistics (mean/std) for seg_latent, mv_seg_latent, shape_latent.
Update config JSON and generate metadata.csv.

Usage:
    python scripts/compute_normalization.py \
        --data_dir /path/to/data \
        --config configs/slat_flow_seg_dit_1_3B_512_bf16.json
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm


def compute_stats(data_dir, subdir, max_samples=None):
    """Compute per-channel mean and std using Welford's online algorithm."""
    latent_dir = os.path.join(data_dir, subdir)
    if not os.path.isdir(latent_dir):
        print(f"  [WARN] {latent_dir} does not exist, skipping.")
        return None, None
    
    files = sorted(glob(os.path.join(latent_dir, '*.npz')))
    if max_samples:
        files = files[:max_samples]
    
    if len(files) == 0:
        print(f"  [WARN] No .npz files in {latent_dir}")
        return None, None
    
    print(f"  Computing stats for {subdir}/ ({len(files)} files)...")
    
    # Two-pass for numerical stability:
    # Pass 1: compute global mean
    total_sum = None
    total_count = 0
    
    for f in tqdm(files, desc=f"  {subdir} pass1(mean)"):
        data = np.load(f)
        feats = data['feats'].astype(np.float64)  # (N, C)
        if total_sum is None:
            total_sum = np.zeros(feats.shape[1], dtype=np.float64)
        total_sum += feats.sum(axis=0)
        total_count += feats.shape[0]
    
    mean = total_sum / total_count  # (C,)
    
    # Pass 2: compute variance
    total_sq_diff = np.zeros_like(mean)
    
    for f in tqdm(files, desc=f"  {subdir} pass2(std)"):
        data = np.load(f)
        feats = data['feats'].astype(np.float64)
        diff = feats - mean[None, :]
        total_sq_diff += (diff ** 2).sum(axis=0)
    
    std = np.sqrt(total_sq_diff / total_count)
    
    # Avoid division by zero
    std = np.maximum(std, 1e-6)
    
    print(f"    Channels: {len(mean)}")
    print(f"    Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"    Std range:  [{std.min():.4f}, {std.max():.4f}]")
    print(f"    Total tokens: {total_count}")
    
    return mean.tolist(), std.tolist()


def generate_metadata(data_dir, output_path):
    """Generate metadata.csv with instance info and token counts."""
    seg_latent_dir = os.path.join(data_dir, 'seg_latent')
    if not os.path.isdir(seg_latent_dir):
        print(f"  [WARN] {seg_latent_dir} not found, cannot generate metadata.")
        return
    
    files = sorted(glob(os.path.join(seg_latent_dir, '*.npz')))
    print(f"  Generating metadata for {len(files)} instances...")
    
    records = []
    for f in tqdm(files, desc="  metadata"):
        instance = os.path.basename(f)[:-4]  # remove .npz
        data = np.load(f)
        n_tokens = data['coords'].shape[0]
        records.append({
            'sha256': instance,
            'seg_latent_tokens': n_tokens,
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"  Saved metadata: {output_path} ({len(df)} rows)")
    if len(df) > 0:
        print(f"    Token stats: min={df['seg_latent_tokens'].min()}, "
              f"max={df['seg_latent_tokens'].max()}, "
              f"mean={df['seg_latent_tokens'].mean():.1f}, "
              f"median={df['seg_latent_tokens'].median():.1f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='/path/to/data')
    parser.add_argument('--config', type=str,
                        default='configs/slat_flow_seg_dit_1_3B_512_bf16.json')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of files for stats computation (for testing)')
    args = parser.parse_args()
    
    print("=" * 60)
    print(" Computing normalization statistics")
    print(f" Data dir: {args.data_dir}")
    print(f" Config:   {args.config}")
    print("=" * 60)
    
    # 1. Compute seg_latent normalization
    print("\n[1/4] seg_latent (training target x_0):")
    seg_mean, seg_std = compute_stats(args.data_dir, 'seg_latent', args.max_samples)
    
    # 2. Compute mv_seg_latent normalization
    print("\n[2/4] mv_seg_latent (multi-view condition):")
    mv_mean, mv_std = compute_stats(args.data_dir, 'mv_seg_latent', args.max_samples)
    
    # 3. Compute shape_latent normalization
    print("\n[3/4] shape_latent (geometry condition):")
    shape_mean, shape_std = compute_stats(args.data_dir, 'shape_latent', args.max_samples)
    
    # 4. Generate metadata.csv
    print("\n[4/4] Generating metadata.csv:")
    metadata_path = os.path.join(args.data_dir, 'metadata.csv')
    generate_metadata(args.data_dir, metadata_path)
    
    # 5. Update config
    print("\n" + "=" * 60)
    print(" Updating config...")
    
    config_path = args.config
    if not os.path.isabs(config_path):
        # Relative to script's parent dir (project root)
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        config_path = os.path.join(project_root, config_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset_args = config['dataset']['args']
    
    if seg_mean is not None:
        dataset_args['normalization'] = {
            'mean': seg_mean,
            'std': seg_std,
        }
        print(f"  Updated normalization (seg_latent): {len(seg_mean)} channels")
    
    if mv_mean is not None:
        dataset_args['mv_cond_normalization'] = {
            'mean': mv_mean,
            'std': mv_std,
        }
        print(f"  Updated mv_cond_normalization: {len(mv_mean)} channels")
    
    if shape_mean is not None:
        dataset_args['shape_cond_normalization'] = {
            'mean': shape_mean,
            'std': shape_std,
        }
        print(f"  Updated shape_cond_normalization: {len(shape_mean)} channels")
    
    # Write back
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  Config saved: {config_path}")
    
    print("\n" + "=" * 60)
    print(" Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
