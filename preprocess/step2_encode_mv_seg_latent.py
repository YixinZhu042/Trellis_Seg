"""
Step 2: Encode 6-view seg maps → mv_seg_latent using Texture VAE.

For each view independently:
  1. seg_map (H, W) int labels → 6ch color features [-1, 1]
  2. Foreground pixels (label > 0) → 3D coords via box_surface_image_ids
  3. Quantize to 512³ grid, deduplicate (average features)
  4. Build SparseTensor → Texture VAE encode → latent tokens (32ch)
Merge 6 views' latent tokens → deduplicate on latent grid → save.

Input:
    {data_dir}/seg_views/{instance}.npz  (seg_maps: (6, H, W) int)

Output:
    {data_dir}/mv_seg_latent/{instance}.npz
        coords: (M, 3) uint8   — latent grid coordinates [0, 31]
        feats:  (M, 32) float32 — VAE latent features

Usage:
    python preprocess/step2_encode_mv_seg_latent.py \
        --data_dir ./data \
        --num_classes 20 \
        --num_samples 5
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

import trellis2.modules.sparse as sp

RESOLUTION = 512  # input voxel grid resolution (VAE downsamples 16x → 32³ latent)


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


def box_surface_image_ids(h, w, view_idx):
    """Map (h, w) pixels to 3D coords on unit cube surface. Returns (h*w, 3) in [0,1]."""
    img_ids = torch.zeros(h, w, 3)
    if view_idx == 0:  # front (z=1)
        img_ids[..., 0] = torch.arange(w)[None, :] / (w - 1)
        img_ids[..., 1] = (h - torch.arange(1, h+1)[:, None]) / (h - 1)
        img_ids[..., 2] = 1
    elif view_idx == 1:  # right (x=1)
        img_ids[..., 0] = 1
        img_ids[..., 1] = (h - torch.arange(1, h+1)[:, None]) / (h - 1)
        img_ids[..., 2] = (w - torch.arange(1, w+1)[None, :]) / (w - 1)
    elif view_idx == 2:  # back (z=0)
        img_ids[..., 0] = (w - torch.arange(1, w+1)[None, :]) / (w - 1)
        img_ids[..., 1] = (h - torch.arange(1, h+1)[:, None]) / (h - 1)
        img_ids[..., 2] = 0
    elif view_idx == 3:  # left (x=0)
        img_ids[..., 0] = 0
        img_ids[..., 1] = (h - torch.arange(1, h+1)[:, None]) / (h - 1)
        img_ids[..., 2] = torch.arange(w)[None, :] / (w - 1)
    elif view_idx == 4:  # top (y=1)
        img_ids[..., 0] = torch.arange(w)[None, :] / (w - 1)
        img_ids[..., 1] = 1
        img_ids[..., 2] = torch.arange(h)[:, None] / (h - 1)
    elif view_idx == 5:  # bottom (y=0)
        img_ids[..., 0] = torch.arange(w)[None, :] / (w - 1)
        img_ids[..., 1] = 0
        img_ids[..., 2] = (h - torch.arange(1, h+1)[:, None]) / (h - 1)
    return rearrange(img_ids, 'h w c -> (h w) c')


def voxelize_foreground(seg_map, view_idx, num_classes):
    """Extract foreground pixels → one-hot features + quantized 512³ coords."""
    H, W = seg_map.shape

    flat_labels = seg_map.reshape(-1)
    mask = flat_labels > 0  # 0 = image background; mesh pixels have labels >= 1
    if mask.sum() == 0:
        return None, None

    fg_labels = flat_labels[mask] - 1  # remove the render-only +1 offset exactly once
    if not np.issubdtype(fg_labels.dtype, np.integer) or np.any(fg_labels < 0) or np.any(fg_labels >= num_classes):
        raise ValueError(f'Rendered labels must be integers in [1, {num_classes}]')

    # One-hot encode (num_classes channels)
    feats = np.zeros((mask.sum(), num_classes), dtype=np.float32)
    feats[np.arange(mask.sum()), fg_labels] = 1.0

    positions = box_surface_image_ids(H, W, view_idx)
    valid_pos = positions[mask]
    coords_int = (valid_pos * (RESOLUTION - 1)).round().int().clamp(0, RESOLUTION - 1)

    # Deduplicate on grid (average features for overlapping pixels)
    linear = (coords_int[:, 0].long() * RESOLUTION * RESOLUTION +
              coords_int[:, 1].long() * RESOLUTION +
              coords_int[:, 2].long())
    unique_lin, inverse = torch.unique(linear, return_inverse=True)

    feats_t = torch.tensor(feats)
    n = unique_lin.shape[0]
    unique_feats = torch.zeros(n, feats_t.shape[1])
    count = torch.zeros(n, 1)
    unique_feats.scatter_add_(0, inverse.unsqueeze(1).expand_as(feats_t), feats_t)
    count.scatter_add_(0, inverse.unsqueeze(1), torch.ones(len(feats_t), 1))
    unique_feats = unique_feats / count

    unique_coords = torch.stack([
        unique_lin // (RESOLUTION * RESOLUTION),
        (unique_lin % (RESOLUTION * RESOLUTION)) // RESOLUTION,
        unique_lin % RESOLUTION
    ], dim=1).int()

    return unique_coords, unique_feats


@torch.no_grad()
def encode_one_view(seg_map, view_idx, encoder, num_classes, device):
    """Voxelize one view's foreground + VAE encode → latent tokens."""
    coords, feats = voxelize_foreground(seg_map, view_idx, num_classes)
    if coords is None:
        return None, None

    batch_coords = torch.cat([
        torch.zeros(coords.shape[0], 1, dtype=torch.int32),
        coords
    ], dim=-1).to(device)
    x = sp.SparseTensor(feats.to(device), batch_coords)

    z = encoder(x)
    return z.coords[:, 1:].cpu(), z.feats.cpu()


def deduplicate_latent(coords, feats):
    """Merge overlapping latent tokens (from different views) by averaging."""
    res = coords.max().item() + 1
    linear = coords[:, 0].long() * res * res + coords[:, 1].long() * res + coords[:, 2].long()

    unique_lin, inverse = torch.unique(linear, return_inverse=True)

    n = unique_lin.shape[0]
    unique_feats = torch.zeros(n, feats.shape[1])
    count = torch.zeros(n, 1)
    unique_feats.scatter_add_(0, inverse.unsqueeze(1).expand_as(feats), feats)
    count.scatter_add_(0, inverse.unsqueeze(1), torch.ones(feats.shape[0], 1))
    unique_feats = unique_feats / count

    unique_coords = torch.stack([
        unique_lin // (res * res),
        (unique_lin % (res * res)) // res,
        unique_lin % res
    ], dim=1).int()

    return unique_coords, unique_feats


SEG_VAE_ENC_PATH = './checkpoints/seg_vae/encoder_step0001000.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/path/to/data')
    parser.add_argument('--encoder_path', type=str, default=SEG_VAE_ENC_PATH)
    parser.add_argument('--encoder_config', type=str, default='configs/seg_vae_train.json')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = os.path.join(args.data_dir, 'mv_seg_latent')
    os.makedirs(output_dir, exist_ok=True)

    if args.world_size > 1:
        args.device = f'cuda:{args.rank % torch.cuda.device_count()}'

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
    print(f"Seg VAE encoder loaded (in_channels={enc_cfg['args']['in_channels']}, latent grid: {RESOLUTION//16}³)")

    seg_views_dir = os.path.join(args.data_dir, 'seg_views')
    files = sorted([f for f in os.listdir(seg_views_dir) if f.endswith('.npz')])
    if args.num_samples:
        files = files[:args.num_samples]

    files = files[args.rank::args.world_size]
    todo_files = [f for f in files if not os.path.exists(os.path.join(output_dir, f))]
    print(f"[Rank {args.rank}] Total: {len(files)}, existing: {len(files)-len(todo_files)}, to encode: {len(todo_files)}")

    for fname in tqdm(todo_files, desc="Step2 mv_seg VAE encode"):
        instance = fname[:-4]
        out_path = os.path.join(output_dir, f'{instance}.npz')

        data = np.load(os.path.join(seg_views_dir, fname))
        seg_maps = data['seg_maps']  # (6, H, W)

        all_coords = []
        all_feats = []

        for v in range(6):
            try:
                coords, feats = encode_one_view(
                    seg_maps[v], v, encoder, args.num_classes, args.device
                )
                if coords is not None:
                    all_coords.append(coords)
                    all_feats.append(feats)
            except Exception as e:
                print(f"  View {v} error for {instance}: {e}")
                torch.cuda.empty_cache()

        if not all_coords:
            continue

        final_coords = torch.cat(all_coords)
        final_feats = torch.cat(all_feats)

        np.savez_compressed(out_path,
            coords=final_coords.numpy().astype(np.uint8),
            feats=final_feats.numpy().astype(np.float32)
        )

    print(f"Done! Output: {output_dir}")


if __name__ == '__main__':
    main()
