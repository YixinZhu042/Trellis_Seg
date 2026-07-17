"""
Test TRELLIS.2 original Texture VAE for part-id reconstruction.

Goal: Verify whether the pretrained texture VAE (6ch PBR in/out) can
      adequately reconstruct part-id labels when we encode them as
      color-coded features, without needing to train a separate Seg VAE.

Strategy:
    1. Load seg_labels (per-voxel integer labels)
    2. Convert labels to a 6-channel representation that fits the texture VAE:
       - Option A: one-hot padded to 6ch (only works for <=6 classes)
       - Option B: normalized label value repeated across channels
       - Option C: color-coded (assign a unique RGB to each class, fill remaining 3ch with 0)
    3. Feed through texture VAE encoder → decoder
    4. Decode back to labels and measure accuracy

Usage:
    python test_texture_vae_for_seg.py \
        --seg_data /path/to/seg_labels/instance.npz \
        --num_classes 20 \
        --device cuda

    Or batch mode:
    python test_texture_vae_for_seg.py \
        --seg_dir /path/to/seg_labels/ \
        --num_classes 20 \
        --max_samples 50
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

import trellis2.models as models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)


# ============================================================
# Label encoding strategies (labels -> 6ch features in [-1, 1])
# ============================================================

def generate_class_colors(num_classes):
    """Generate distinct colors for each class using a fixed palette."""
    # Use a spread-out hue palette for maximal separation
    colors = np.zeros((num_classes, 3), dtype=np.float32)
    for i in range(num_classes):
        hue = i / num_classes
        # Simple HSV -> RGB with S=1, V=1
        h = hue * 6.0
        c = 1.0
        x = 1.0 - abs(h % 2 - 1)
        if h < 1:
            rgb = (c, x, 0)
        elif h < 2:
            rgb = (x, c, 0)
        elif h < 3:
            rgb = (0, c, x)
        elif h < 4:
            rgb = (0, x, c)
        elif h < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
        colors[i] = rgb
    return colors  # (num_classes, 3) in [0, 1]


def labels_to_feats_color(labels, num_classes):
    """
    Encode labels as color-coded 6-channel features.
    Channels: [R, G, B, class_norm, class_norm, 1.0]
    Where class_norm = label / (num_classes - 1), mapped to [-1, 1].
    """
    palette = generate_class_colors(num_classes)  # (C, 3)
    palette_t = torch.from_numpy(palette).to(labels.device)
    
    rgb = palette_t[labels]  # (N, 3) in [0, 1]
    
    # Normalized label as additional signal (redundancy helps reconstruction)
    label_norm = labels.float() / (num_classes - 1)  # [0, 1]
    
    feats = torch.zeros(labels.shape[0], 6, device=labels.device)
    feats[:, 0:3] = rgb
    feats[:, 3] = label_norm
    feats[:, 4] = label_norm
    feats[:, 5] = 1.0  # alpha = 1 (fully opaque, like real PBR data)
    
    # Map [0, 1] -> [-1, 1] to match texture VAE expected input range
    feats = feats * 2 - 1
    return feats


def labels_to_feats_onehot_packed(labels, num_classes):
    """
    Pack one-hot into 6 channels using bit-like encoding.
    Each channel carries multiple classes' info via careful normalization.
    This gives the VAE maximum separability.
    """
    one_hot = F.one_hot(labels, num_classes).float()  # (N, num_classes)
    
    # Strategy: Split classes into 6 groups, sum within group
    # This creates a soft-encoding where each channel "responds" to a subset
    group_size = (num_classes + 5) // 6
    feats = torch.zeros(labels.shape[0], 6, device=labels.device)
    for i in range(6):
        start = i * group_size
        end = min(start + group_size, num_classes)
        if start < num_classes:
            # Weight classes within group by position for disambiguation
            weights = torch.arange(1, end - start + 1, device=labels.device).float()
            weights = weights / weights.sum()
            feats[:, i] = (one_hot[:, start:end] * weights).sum(dim=-1)
    
    # Map [0, 1] -> [-1, 1]
    feats = feats * 2 - 1
    return feats


def labels_to_feats_normalized(labels, num_classes):
    """
    Simple approach: encode each label as a normalized scalar repeated 6x.
    Advantage: smooth in value space.
    Disadvantage: nearby classes are close in feature space.
    """
    label_norm = labels.float() / (num_classes - 1)  # [0, 1]
    feats = label_norm.unsqueeze(-1).expand(-1, 6)
    feats = feats * 2 - 1  # [-1, 1]
    return feats


# ============================================================
# Decoding strategies (6ch features -> labels)
# ============================================================

def feats_to_labels_color(feats, num_classes):
    """
    Decode color-coded features back to labels via nearest-neighbor
    in the class color palette.
    """
    # Map [-1, 1] -> [0, 1]
    feats = (feats + 1) / 2
    
    palette = generate_class_colors(num_classes)
    palette_t = torch.from_numpy(palette).to(feats.device)  # (C, 3)
    
    rgb_recon = feats[:, 0:3]  # (N, 3)
    
    # Also use the redundant label_norm channels
    label_norm_recon = feats[:, 3:5].mean(dim=-1)  # (N,)
    label_from_norm = (label_norm_recon * (num_classes - 1)).round().long().clamp(0, num_classes - 1)
    
    # Nearest neighbor in RGB space
    dists = torch.cdist(rgb_recon.unsqueeze(0), palette_t.unsqueeze(0)).squeeze(0)  # (N, C)
    label_from_color = dists.argmin(dim=-1)
    
    # Combine: if both agree, great; otherwise trust the one with lower distance
    # For simplicity, use a weighted vote
    return label_from_color, label_from_norm


def feats_to_labels_onehot_packed(feats, num_classes):
    """Decode packed one-hot features back to labels."""
    feats = (feats + 1) / 2  # [-1, 1] -> [0, 1]
    
    group_size = (num_classes + 5) // 6
    
    # For each voxel, find which channel has highest response
    # and within that group, estimate the class
    best_channel = feats.argmax(dim=-1)  # (N,)
    
    # Also try: construct a score for each class
    scores = torch.zeros(feats.shape[0], num_classes, device=feats.device)
    for i in range(6):
        start = i * group_size
        end = min(start + group_size, num_classes)
        if start < num_classes:
            n_in_group = end - start
            weights = torch.arange(1, n_in_group + 1, device=feats.device).float()
            weights = weights / weights.sum()
            # Each class in the group contributes proportionally
            for j, w in enumerate(weights):
                scores[:, start + j] = feats[:, i] * w
    
    return scores.argmax(dim=-1)


def feats_to_labels_normalized(feats, num_classes):
    """Decode normalized features back to labels."""
    feats = (feats + 1) / 2  # [-1, 1] -> [0, 1]
    label_norm = feats.mean(dim=-1)  # average across 6 channels
    labels = (label_norm * (num_classes - 1)).round().long().clamp(0, num_classes - 1)
    return labels


# ============================================================
# Main test logic
# ============================================================

@torch.no_grad()
def test_reconstruction(coords, labels, encoder, decoder, num_classes, 
                        encoding='color', device='cuda'):
    """
    Test encode → decode reconstruction accuracy.
    
    Returns dict with metrics.
    """
    coords_t = torch.tensor(coords, dtype=torch.int32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    
    # Encode labels to 6ch features
    if encoding == 'color':
        feats = labels_to_feats_color(labels_t, num_classes)
    elif encoding == 'onehot_packed':
        feats = labels_to_feats_onehot_packed(labels_t, num_classes)
    elif encoding == 'normalized':
        feats = labels_to_feats_normalized(labels_t, num_classes)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    
    # Build SparseTensor
    batch_coords = torch.cat([
        torch.zeros(coords_t.shape[0], 1, dtype=torch.int32, device=device),
        coords_t
    ], dim=-1)
    
    x = sp.SparseTensor(feats.float(), batch_coords)
    
    # VAE encode
    z = encoder(x)
    
    # VAE decode: spatial cache from encoder is preserved in z's SparseTensor
    # so decoder can upsample without explicit guide_subs
    recon = decoder(z)
    
    # Get reconstructed features (decoder outputs in same range as input: [-1, 1])
    recon_feats = recon.feats  # (N', 6)
    recon_coords = recon.coords[:, 1:]  # remove batch dim, (N', 3)
    
    # Map reconstructed voxels back to original coords for comparison
    # Since VAE may change topology slightly, find matching coords
    # Build a lookup from coord -> original label
    RES = 2048  # Must be > max coordinate value
    coord_key = coords_t[:, 0].long() * RES * RES + coords_t[:, 1].long() * RES + coords_t[:, 2].long()
    recon_coord_key = recon_coords[:, 0].long() * RES * RES + recon_coords[:, 1].long() * RES + recon_coords[:, 2].long()
    
    # Find intersection
    coord_to_label = {}
    for i in range(coords_t.shape[0]):
        coord_to_label[coord_key[i].item()] = labels_t[i].item()
    
    matched_recon_feats = []
    matched_gt_labels = []
    for i in range(recon_coords.shape[0]):
        key = recon_coord_key[i].item()
        if key in coord_to_label:
            matched_recon_feats.append(recon_feats[i])
            matched_gt_labels.append(coord_to_label[key])
    
    if len(matched_recon_feats) == 0:
        return {
            'accuracy': 0.0,
            'accuracy_norm': 0.0,
            'coverage': 0.0,
            'num_input_voxels': coords_t.shape[0],
            'num_output_voxels': recon_coords.shape[0],
            'num_matched': 0,
        }
    
    matched_recon_feats = torch.stack(matched_recon_feats)
    matched_gt_labels = torch.tensor(matched_gt_labels, device=device)
    
    # Decode features back to labels
    if encoding == 'color':
        pred_color, pred_norm = feats_to_labels_color(matched_recon_feats, num_classes)
        acc_color = (pred_color == matched_gt_labels).float().mean().item()
        acc_norm = (pred_norm == matched_gt_labels).float().mean().item()
        # Best of both
        acc_best = max(acc_color, acc_norm)
        # Combined vote
        agree_mask = pred_color == pred_norm
        pred_combined = pred_color.clone()
        pred_combined[~agree_mask] = pred_norm[~agree_mask]  # fallback to norm where disagree
        acc_combined = (pred_combined == matched_gt_labels).float().mean().item()
    elif encoding == 'onehot_packed':
        pred = feats_to_labels_onehot_packed(matched_recon_feats, num_classes)
        acc_color = (pred == matched_gt_labels).float().mean().item()
        acc_norm = acc_color
        acc_best = acc_color
        acc_combined = acc_color
    elif encoding == 'normalized':
        pred = feats_to_labels_normalized(matched_recon_feats, num_classes)
        acc_color = (pred == matched_gt_labels).float().mean().item()
        acc_norm = acc_color
        acc_best = acc_color
        acc_combined = acc_color
    
    coverage = len(matched_recon_feats) / coords_t.shape[0]
    
    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = matched_gt_labels == c
        if mask.sum() > 0:
            if encoding == 'color':
                per_class_acc[c] = (pred_color[mask] == matched_gt_labels[mask]).float().mean().item()
            elif encoding == 'normalized':
                per_class_acc[c] = (pred[mask] == matched_gt_labels[mask]).float().mean().item()
            else:
                per_class_acc[c] = (pred[mask] == matched_gt_labels[mask]).float().mean().item()
    
    return {
        'accuracy_color': acc_color,
        'accuracy_norm': acc_norm,
        'accuracy_best': acc_best,
        'accuracy_combined': acc_combined,
        'coverage': coverage,
        'num_input_voxels': coords_t.shape[0],
        'num_output_voxels': recon_coords.shape[0],
        'num_matched': len(matched_recon_feats),
        'per_class_acc': per_class_acc,
    }


def generate_synthetic_test_data(num_voxels=50000, num_classes=20, resolution=256, seed=42):
    """
    Generate synthetic voxel segmentation data for testing.
    Creates spatially coherent regions (like a real segmented object).
    """
    rng = np.random.RandomState(seed)
    
    # Generate a sphere-like point cloud
    n_pts = num_voxels
    # Random points on surface of a sphere + some inside
    phi = rng.uniform(0, 2 * np.pi, n_pts)
    cos_theta = rng.uniform(-1, 1, n_pts)
    theta = np.arccos(cos_theta)
    r = rng.uniform(0.3, 1.0, n_pts)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Quantize to voxel grid
    coords = np.stack([x, y, z], axis=-1)  # (N, 3) in [-1, 1]
    coords = ((coords + 1) / 2 * (resolution - 1)).astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)
    
    # Remove duplicates
    coords_unique = np.unique(coords, axis=0)
    n_pts = coords_unique.shape[0]
    
    # Assign labels based on spatial regions (simulate real parts)
    # Use simple spatial partitioning for coherent regions
    num_actual_classes = min(num_classes, 10)  # Use a subset
    labels = np.zeros(n_pts, dtype=np.int64)
    
    # Assign by angular sectors + height bands
    x_norm = coords_unique[:, 0] / (resolution - 1) * 2 - 1
    y_norm = coords_unique[:, 1] / (resolution - 1) * 2 - 1
    z_norm = coords_unique[:, 2] / (resolution - 1) * 2 - 1
    
    angles = np.arctan2(y_norm, x_norm)  # [-pi, pi]
    heights = z_norm  # [-1, 1]
    
    # 4 angular sectors x height bands
    angle_bins = np.digitize(angles, np.linspace(-np.pi, np.pi, 5)) - 1  # 0-3
    height_bins = np.digitize(heights, np.linspace(-1, 1, 4)) - 1  # 0-2
    
    labels = (angle_bins * 3 + height_bins) % num_actual_classes
    
    return coords_unique, labels


def main():
    parser = argparse.ArgumentParser(description='Test TRELLIS.2 Texture VAE for part-id reconstruction')
    parser.add_argument('--seg_data', type=str, default=None, help='Single seg_labels npz file')
    parser.add_argument('--seg_dir', type=str, default=None, help='Directory of seg_labels npz files')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic test data (no data files needed)')
    parser.add_argument('--num_voxels', type=int, default=50000, help='Number of voxels for synthetic data')
    parser.add_argument('--resolution', type=int, default=256, help='Voxel grid resolution for synthetic data')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=50, help='Max samples to test in batch mode')
    parser.add_argument('--encoding', type=str, default='all', 
                        choices=['color', 'onehot_packed', 'normalized', 'all'],
                        help='Label encoding strategy')
    parser.add_argument('--encoder_path', type=str, 
                        default='/path/to/trellis2_ckpts/tex_enc',
                        help='Texture VAE encoder path')
    parser.add_argument('--decoder_path', type=str,
                        default='/path/to/trellis2_ckpts/tex_dec',
                        help='Texture VAE decoder path')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None, help='Save detailed results')
    args = parser.parse_args()
    
    assert args.seg_data or args.seg_dir or args.synthetic, \
        "Must specify --seg_data, --seg_dir, or --synthetic"
    
    # Load models
    print(f"Loading Texture VAE encoder from {args.encoder_path}...")
    encoder = models.from_pretrained(args.encoder_path).eval().to(args.device)
    print(f"Loading Texture VAE decoder from {args.decoder_path}...")
    decoder = models.from_pretrained(args.decoder_path).eval().to(args.device)
    print("Models loaded.")
    
    # Gather test data
    encodings = ['color', 'onehot_packed', 'normalized'] if args.encoding == 'all' else [args.encoding]
    all_results = {enc: [] for enc in encodings}
    
    if args.synthetic:
        # Generate synthetic test cases with varying parameters
        test_cases = []
        for seed in range(min(5, args.max_samples)):
            for n_voxels in [10000, 30000, 50000]:
                coords, labels = generate_synthetic_test_data(
                    num_voxels=n_voxels, 
                    num_classes=args.num_classes,
                    resolution=args.resolution,
                    seed=seed * 100 + n_voxels
                )
                test_cases.append((f"synthetic_s{seed}_n{n_voxels}", coords, labels))
        
        print(f"Testing {len(test_cases)} synthetic samples...")
        for name, coords, labels in tqdm(test_cases, desc="Testing"):
            for encoding in encodings:
                try:
                    result = test_reconstruction(
                        coords, labels, encoder, decoder,
                        args.num_classes, encoding, args.device
                    )
                    result['file'] = name
                    all_results[encoding].append(result)
                except Exception as e:
                    print(f"  Error with {name} [{encoding}]: {e}")
                    torch.cuda.empty_cache()
                    continue
    else:
        if args.seg_data:
            files = [args.seg_data]
        else:
            files = sorted([
                os.path.join(args.seg_dir, f)
                for f in os.listdir(args.seg_dir)
                if f.endswith('.npz')
            ])[:args.max_samples]

        print(f"Testing {len(files)} samples...")
        for fpath in tqdm(files, desc="Testing"):
            data = np.load(fpath)
            coords = data['coords']  # (N, 3) int
            labels = data['labels']  # (N,) int

            if coords.shape[0] == 0:
                continue
            if coords.shape[0] > 500000:
                print(f"  Skipping {os.path.basename(fpath)}: too large ({coords.shape[0]} voxels)")
                continue

            for encoding in encodings:
                try:
                    result = test_reconstruction(
                        coords, labels, encoder, decoder,
                        args.num_classes, encoding, args.device
                    )
                    result['file'] = os.path.basename(fpath)
                    all_results[encoding].append(result)
                except Exception as e:
                    print(f"  Error with {os.path.basename(fpath)} [{encoding}]: {e}")
                    torch.cuda.empty_cache()
                    continue
    
    # Print summary
    print("\n" + "=" * 70)
    print("RECONSTRUCTION RESULTS SUMMARY")
    print("=" * 70)
    
    for encoding in encodings:
        results = all_results[encoding]
        if not results:
            print(f"\n[{encoding}] No results.")
            continue
        
        avg_acc_color = np.mean([r['accuracy_color'] for r in results])
        avg_acc_norm = np.mean([r['accuracy_norm'] for r in results])
        avg_acc_best = np.mean([r['accuracy_best'] for r in results])
        avg_coverage = np.mean([r['coverage'] for r in results])
        
        print(f"\n[Encoding: {encoding}]")
        print(f"  Samples tested: {len(results)}")
        print(f"  Avg accuracy (color-based):  {avg_acc_color:.4f} ({avg_acc_color*100:.2f}%)")
        print(f"  Avg accuracy (norm-based):   {avg_acc_norm:.4f} ({avg_acc_norm*100:.2f}%)")
        print(f"  Avg accuracy (best):         {avg_acc_best:.4f} ({avg_acc_best*100:.2f}%)")
        print(f"  Avg voxel coverage:          {avg_coverage:.4f} ({avg_coverage*100:.2f}%)")
        
        # Per-sample details
        if len(results) <= 10:
            print(f"  Per-sample details:")
            for r in results:
                print(f"    {r['file']}: acc={r['accuracy_best']:.4f}, "
                      f"coverage={r['coverage']:.4f}, "
                      f"voxels={r['num_input_voxels']}→{r['num_output_voxels']}")
    
    # Save results if requested
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        import json
        for encoding in encodings:
            results = all_results[encoding]
            save_data = []
            for r in results:
                r_save = {k: v for k, v in r.items() if k != 'per_class_acc'}
                r_save['per_class_acc'] = {str(k): v for k, v in r.get('per_class_acc', {}).items()}
                save_data.append(r_save)
            with open(os.path.join(args.output_dir, f'results_{encoding}.json'), 'w') as f:
                json.dump(save_data, f, indent=2)
        print(f"\nDetailed results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
