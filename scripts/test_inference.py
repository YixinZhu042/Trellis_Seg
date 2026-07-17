"""
Test inference script: sample N instances from training data, run denoising, decode, and save results.

Verifies the full pipeline:
  1. Data loading (seg_latent, shape_latent, mv_seg_latent, cond_image)
  2. Normalization consistency
  3. Denoiser forward pass
  4. Texture VAE decoding
  5. Label extraction + visualization

Usage:
    python scripts/test_inference.py \
        --config configs/slat_flow_seg_dit_1_3B_512_bf16.json \
        --data_dir /path/to/data \
        --ckpt /path/to/denoiser_ema.pt \
        --num_samples 4 \
        --output_dir ./test_output
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from trellis2.modules.sparse.basic import SparseTensor
import trellis2.models as trellis2_models
from models import get_model

torch.set_grad_enabled(False)

TEX_DEC_PATH = '/path/to/trellis2_ckpts/tex_dec'


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


def decode_feats_to_labels(feats_6ch, num_classes):
    feats_01 = (feats_6ch + 1) / 2
    rgb = feats_01[:, 0:3]
    palette = torch.tensor(generate_class_colors(num_classes), device=feats_6ch.device)
    dists = torch.cdist(rgb.float(), palette.float())
    return dists.argmin(dim=-1)


def lookup_feats_at_coords(target_coords, source_coords, source_feats):
    N = target_coords.shape[0]
    C = source_feats.shape[-1]
    matched = torch.zeros(N, C, device=source_feats.device, dtype=source_feats.dtype)

    res = max(target_coords.max().item(), source_coords.max().item()) + 1
    src_lin = (source_coords[:, 0].long() * res * res +
               source_coords[:, 1].long() * res +
               source_coords[:, 2].long())
    tgt_lin = (target_coords[:, 0].long() * res * res +
               target_coords[:, 1].long() * res +
               target_coords[:, 2].long())

    src_hash = {}
    for i in range(source_coords.shape[0]):
        idx = src_lin[i].item()
        if idx in src_hash:
            src_hash[idx] = (src_hash[idx][0] + source_feats[i], src_hash[idx][1] + 1)
        else:
            src_hash[idx] = (source_feats[i].clone(), 1)

    for i in range(N):
        idx = tgt_lin[i].item()
        if idx in src_hash:
            feat, cnt = src_hash[idx]
            matched[i] = feat / cnt

    return matched


@torch.no_grad()
def encode_shape_for_cache(mesh_path, encoder, resolution, device):
    """Encode mesh → shape_slat with spatial cache (needed for tex_decoder)."""
    import trimesh
    import o_voxel

    mesh = trimesh.load(mesh_path, force='mesh')
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()

    # Normalize to [-0.5, 0.5]
    v_min = vertices.min(dim=0)[0]
    v_max = vertices.max(dim=0)[0]
    center = (v_min + v_max) / 2
    scale = 0.99999 / (v_max - v_min).max()
    vertices = (vertices - center) * scale

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
    )

    vert_st = SparseTensor(
        feats=dual_vertices * resolution - voxel_indices,
        coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
    ).to(device)
    inter_st = vert_st.replace(intersected).to(device)

    shape_slat = encoder(vert_st, inter_st)
    return shape_slat


@torch.no_grad()
def sample_flow(denoiser, cond_features, concat_cond, noise, steps=50,
                mv_cond=None, mv_coords=None, sigma_min=1e-5):
    """Euler sampling using TRELLIS.2's FlowEulerSampler (correct t=1→0 direction)."""
    from trellis2.pipelines.samplers import FlowEulerSampler

    sampler = FlowEulerSampler(sigma_min=sigma_min)
    result = sampler.sample(
        denoiser,
        noise=noise,
        cond=cond_features,
        concat_cond=concat_cond,
        mv_cond=mv_cond,
        mv_coords=mv_coords,
        steps=steps,
        verbose=False,
    )
    return result.samples


def load_sample(data_dir, instance, config, device, num_classes=20):
    """Load one training sample and build inference inputs.

    Architecture (actual):
      - concat_cond = shape_latent (32ch) aligned to seg_latent coords
      - mv_seg_latent → cross-attention tokens (not concat)
      - in_channels = 64 (32 noise + 32 shape concat)
    """
    dataset_args = config['dataset']['args']

    # seg_latent (ground truth)
    seg_data = np.load(os.path.join(data_dir, 'seg_latent', f'{instance}.npz'))
    seg_coords = torch.tensor(seg_data['coords'], dtype=torch.int32, device=device)
    seg_feats = torch.tensor(seg_data['feats'], dtype=torch.float32, device=device)

    # shape_latent
    shape_data = np.load(os.path.join(data_dir, 'shape_latent', f'{instance}.npz'))
    shape_coords = torch.tensor(shape_data['coords'], dtype=torch.int32, device=device)
    shape_feats = torch.tensor(shape_data['feats'], dtype=torch.float32, device=device)

    # mv_seg_latent (for cross-attention, not concat)
    mv_data = np.load(os.path.join(data_dir, 'mv_seg_latent', f'{instance}.npz'))
    mv_coords = torch.tensor(mv_data['coords'], dtype=torch.int32, device=device)
    mv_feats = torch.tensor(mv_data['feats'], dtype=torch.float32, device=device)

    # Normalize seg_latent
    seg_norm = dataset_args.get('normalization')
    if seg_norm:
        seg_mean = torch.tensor(seg_norm['mean'], device=device).reshape(1, -1)
        seg_std = torch.tensor(seg_norm['std'], device=device).reshape(1, -1)
        seg_feats_normed = (seg_feats - seg_mean) / seg_std
    else:
        seg_feats_normed = seg_feats

    # Normalize shape_latent
    shape_norm = dataset_args.get('shape_cond_normalization')
    if shape_norm:
        s_mean = torch.tensor(shape_norm['mean'], device=device).reshape(1, -1)
        s_std = torch.tensor(shape_norm['std'], device=device).reshape(1, -1)
        shape_feats_normed = (shape_feats - s_mean) / s_std
    else:
        shape_feats_normed = shape_feats

    # Normalize mv_seg_latent
    mv_norm = dataset_args.get('mv_cond_normalization')
    if mv_norm:
        mv_mean = torch.tensor(mv_norm['mean'], device=device).reshape(1, -1)
        mv_std = torch.tensor(mv_norm['std'], device=device).reshape(1, -1)
        mv_feats_normed = (mv_feats - mv_mean) / mv_std
    else:
        mv_feats_normed = mv_feats

    # Build concat_cond = shape_latent (32ch) aligned to seg_latent coords
    shape_at_seg = lookup_feats_at_coords(seg_coords, shape_coords, shape_feats_normed)

    batch_coords = torch.cat([
        torch.zeros(seg_coords.shape[0], 1, dtype=torch.int32, device=device),
        seg_coords
    ], dim=-1)

    concat_cond = SparseTensor(coords=batch_coords, feats=shape_at_seg)
    concat_cond._shape = torch.Size([1, 32])

    # noise on seg_latent coords
    noise = SparseTensor(
        coords=batch_coords,
        feats=torch.randn(seg_coords.shape[0], 32, device=device)
    )
    noise._shape = torch.Size([1, 32])

    # GT as SparseTensor (for comparison)
    gt = SparseTensor(coords=batch_coords, feats=seg_feats_normed)
    gt._shape = torch.Size([1, 32])

    # mv_cond for cross-attention: list of (Li, 32) tensors (batch=1)
    mv_cond_list = [mv_feats_normed]  # (L_mv, 32)
    mv_coords_list = [mv_coords.float()]  # (L_mv, 3)

    # Cond image: use colorized front seg view (matches training fallback)
    cond_image = None
    seg_views_path = os.path.join(data_dir, 'seg_views', f'{instance}.npz')
    if os.path.exists(seg_views_path):
        seg_views_data = np.load(seg_views_path)
        front_seg = seg_views_data['seg_maps'][0]  # (H, W) int
        # Colorize with palette
        palette_rgb = generate_class_colors(num_classes)  # (C, 3) float [0,1]
        h, w = front_seg.shape
        colored = palette_rgb[front_seg.clip(0, num_classes - 1).flatten()]  # (H*W, 3)
        colored = (colored.reshape(h, w, 3) * 255).astype(np.uint8)
        cond_image = Image.fromarray(colored)

    return {
        'concat_cond': concat_cond,
        'noise': noise,
        'gt': gt,
        'mv_cond': mv_cond_list,
        'mv_coords': mv_coords_list,
        'seg_coords': seg_coords,
        'seg_feats_raw': seg_feats,
        'cond_image': cond_image,
        'num_tokens': seg_coords.shape[0],
        'num_mv_tokens': mv_feats.shape[0],
    }


def visualize_labels_3d(coords, labels, num_classes, output_path):
    """Save a simple 3-axis projection visualization."""
    palette = (generate_class_colors(num_classes) * 255).astype(np.uint8)
    coords_np = coords.cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig_size = 256
    projections = []
    # XY, XZ, YZ projections
    axes_pairs = [(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]

    for ax0, ax1, name in axes_pairs:
        img = np.ones((fig_size, fig_size, 3), dtype=np.uint8) * 240
        c0 = coords_np[:, ax0].astype(np.float32)
        c1 = coords_np[:, ax1].astype(np.float32)

        if c0.max() > c0.min():
            c0 = (c0 - c0.min()) / (c0.max() - c0.min()) * (fig_size - 10) + 5
        else:
            c0 = np.full_like(c0, fig_size // 2)
        if c1.max() > c1.min():
            c1 = (c1 - c1.min()) / (c1.max() - c1.min()) * (fig_size - 10) + 5
        else:
            c1 = np.full_like(c1, fig_size // 2)

        for i in range(len(labels_np)):
            x = int(c0[i])
            y = fig_size - 1 - int(c1[i])
            x = max(0, min(fig_size - 1, x))
            y = max(0, min(fig_size - 1, y))
            color = palette[labels_np[i] % num_classes]
            img[y, x] = color

        projections.append(img)

    combined = np.concatenate(projections, axis=1)
    Image.fromarray(combined).save(output_path)


def export_seg_glb(mesh_path, voxel_coords, voxel_labels, num_classes, output_path,
                   resolution=512):
    """Export mesh as GLB with per-face colors based on segmentation labels.

    Samples the decoded voxel label grid onto mesh face centers using
    nearest-neighbor lookup.
    """
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.load(mesh_path, force='mesh')

    # Normalize mesh to [-0.5, 0.5] (same as shape encoding)
    vertices = mesh.vertices.copy()
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    center = (v_min + v_max) / 2
    scale = 0.99999 / (v_max - v_min).max()
    vertices = (vertices - center) * scale

    coords_np = voxel_coords.cpu().numpy().astype(np.float32)
    labels_np = voxel_labels.cpu().numpy()

    # Face centers in normalized mesh space [-0.5, 0.5] → voxel space [0, resolution-1]
    # This matches the aabb=[-0.5, 0.5] → grid_size=resolution mapping used by o_voxel
    face_centers = np.mean(vertices[mesh.faces], axis=1)  # (F, 3)
    face_voxel = (face_centers + 0.5) * (resolution - 1)

    # KD-tree nearest-neighbor lookup
    tree = cKDTree(coords_np)
    _, nn_idx = tree.query(face_voxel)
    face_labels = labels_np[nn_idx]

    # Assign colors from palette
    palette = (generate_class_colors(num_classes) * 255).astype(np.uint8)

    # Build mesh with unique vertices per face (no shared verts) so face colors export exactly
    n_faces = len(mesh.faces)
    new_verts = mesh.vertices[mesh.faces.flatten()]  # (F*3, 3)
    new_faces = np.arange(n_faces * 3).reshape(n_faces, 3)

    colored_mesh = trimesh.Trimesh(
        vertices=new_verts, faces=new_faces, process=False
    )
    # Assign vertex colors (3 verts per face get the same color)
    vert_colors = np.zeros((n_faces * 3, 4), dtype=np.uint8)
    for i in range(n_faces):
        c = palette[face_labels[i] % num_classes]
        vert_colors[i*3:i*3+3] = [c[0], c[1], c[2], 255]
    colored_mesh.visual.vertex_colors = vert_colors
    colored_mesh.export(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/slat_flow_seg_dit_1_3B_512_bf16.json')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None, help='Denoiser checkpoint (skip denoising if not provided)')
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='./test_output')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--decode', action='store_true',
                        help='Decode latent via Texture VAE (requires --mesh_dir)')
    parser.add_argument('--mesh_dir', type=str, default=None,
                        help='Mesh directory for decode (structure: {type_id}/{instance}.glb)')
    parser.add_argument('--arrow_dir', type=str,
                        default='/path/to/PartNeXt',
                        help='PartNeXt arrow dir for id→type_id mapping')
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    config = json.load(open(args.config))
    resolution = config['dataset']['args'].get('resolution', 512)
    num_classes = args.num_classes
    print(f'Config: {args.config}')
    print(f'Data: {args.data_dir}')
    print(f'Output: {args.output_dir}')
    print(f'Samples: {args.num_samples}, Steps: {args.steps}, Resolution: {resolution}')
    print()

    # Collect available instances (must have all 3 latent files)
    seg_dir = os.path.join(args.data_dir, 'seg_latent')
    shape_dir = os.path.join(args.data_dir, 'shape_latent')
    mv_dir = os.path.join(args.data_dir, 'mv_seg_latent')

    seg_ids = set(f[:-4] for f in os.listdir(seg_dir) if f.endswith('.npz'))
    shape_ids = set(f[:-4] for f in os.listdir(shape_dir) if f.endswith('.npz'))
    mv_ids = set(f[:-4] for f in os.listdir(mv_dir) if f.endswith('.npz'))

    valid_ids = sorted(seg_ids & shape_ids & mv_ids)
    print(f'Available instances: {len(valid_ids)} (seg={len(seg_ids)}, shape={len(shape_ids)}, mv={len(mv_ids)})')

    if len(valid_ids) == 0:
        print('ERROR: No valid instances found with all 3 latent types.')
        return

    # Sample N
    rng = np.random.default_rng(42)
    selected = rng.choice(valid_ids, size=min(args.num_samples, len(valid_ids)), replace=False)
    print(f'Selected {len(selected)} samples: {[s[:8] + "..." for s in selected]}')
    print()

    # Load denoiser if checkpoint provided
    denoiser = None
    if args.ckpt:
        print(f'Loading denoiser: {args.ckpt}')
        model_cfg = config['models']['denoiser']
        denoiser = get_model(model_cfg['name'], **model_cfg['args'])

        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(args.ckpt, device='cpu')
        else:
            state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=True)

        # Handle shape mismatch (e.g. loading texture DIT with in_channels=64 into 96ch model)
        model_sd = denoiser.state_dict()
        skipped = []
        for k in list(state_dict.keys()):
            if k not in model_sd:
                skipped.append(f'{k} (not in model)')
                del state_dict[k]
            elif state_dict[k].shape != model_sd[k].shape:
                skipped.append(f'{k} ({list(state_dict[k].shape)} vs {list(model_sd[k].shape)})')
                del state_dict[k]
        if skipped:
            print(f'  Skipped {len(skipped)} params (shape mismatch):')
            for s in skipped[:5]:
                print(f'    {s}')
            if len(skipped) > 5:
                print(f'    ... and {len(skipped)-5} more')
        denoiser.load_state_dict(state_dict, strict=False)
        denoiser = denoiser.to(device).to(torch.bfloat16).eval()
        print(f'  Loaded {len(state_dict)}/{len(model_sd)} params. Total: {sum(p.numel() for p in denoiser.parameters()) / 1e9:.2f}B')
        del state_dict

    # Load DINOv3 if denoiser is available and cond images exist
    extractor = None
    if denoiser is not None:
        image_cond_cfg = config['trainer']['args'].get('image_cond_model')
        if image_cond_cfg:
            from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
            print(f'Loading DINOv3...')
            extractor = DinoV3FeatureExtractor(**image_cond_cfg['args'])
            extractor.cuda()

    # Load tex decoder + shape encoder if --decode
    tex_decoder = None
    shape_encoder = None
    id_to_type = {}
    if args.decode:
        if not args.mesh_dir:
            print('ERROR: --decode requires --mesh_dir for spatial cache construction')
            return
        print(f'Loading Texture VAE decoder...')
        tex_decoder = trellis2_models.from_pretrained(TEX_DEC_PATH).eval().to(device)
        print(f'Loading Shape VAE encoder (for spatial cache)...')
        SHAPE_ENC_PATH = '/path/to/trellis2_ckpts/shape_enc'
        shape_encoder = trellis2_models.from_pretrained(SHAPE_ENC_PATH).eval().to(device)
        # Build id→type mapping for mesh lookup
        import pyarrow.ipc as ipc
        from glob import glob
        for af in sorted(glob(os.path.join(args.arrow_dir, 'data-*.arrow'))):
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
        print(f'  Loaded {len(id_to_type)} id→type mappings')

    print()
    print('=' * 60)

    # Process each sample
    results_summary = []
    for idx, instance in enumerate(selected):
        print(f'\n[{idx+1}/{len(selected)}] Instance: {instance}')

        sample = load_sample(args.data_dir, instance, config, device, num_classes)
        print(f'  Tokens: {sample["num_tokens"]}, MV tokens: {sample["num_mv_tokens"]}')
        print(f'  concat_cond (shape): {sample["concat_cond"].feats.shape}')
        print(f'  Cond image: {"yes" if sample["cond_image"] else "no"}')

        # Get cond features
        if extractor and sample['cond_image']:
            img = sample['cond_image'].convert('RGB')
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            cond_features = extractor([img])
        else:
            cond_features = torch.zeros(1, 1029, 1024, device=device)

        # --- Test 1: GT decode (verify data is valid) ---
        seg_norm = config['dataset']['args'].get('normalization')
        gt_feats_raw = sample['seg_feats_raw']
        print(f'  GT seg_latent: mean={gt_feats_raw.mean():.3f}, std={gt_feats_raw.std():.3f}')

        # --- Test 2: Denoise (if ckpt available) ---
        pred_feats = None
        if denoiser is not None:
            print(f'  Denoising ({args.steps} steps)...')
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                seg_slat = sample_flow(
                    denoiser, cond_features, sample['concat_cond'], sample['noise'],
                    steps=args.steps,
                    mv_cond=sample['mv_cond'],
                    mv_coords=sample['mv_coords'],
                )
            pred_feats_normed = seg_slat.feats.float()

            # Denormalize
            if seg_norm:
                s_mean = torch.tensor(seg_norm['mean'], device=device).reshape(1, -1)
                s_std = torch.tensor(seg_norm['std'], device=device).reshape(1, -1)
                pred_feats = pred_feats_normed * s_std + s_mean
            else:
                pred_feats = pred_feats_normed

            # Compare with GT
            mse = ((pred_feats - gt_feats_raw) ** 2).mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                pred_feats.reshape(1, -1), gt_feats_raw.reshape(1, -1)
            ).item()
            print(f'  Pred vs GT: MSE={mse:.4f}, CosSim={cos_sim:.4f}')
            results_summary.append({
                'instance': instance,
                'tokens': sample['num_tokens'],
                'mse': mse,
                'cos_sim': cos_sim,
            })
        else:
            # No ckpt: just use GT as "pred" to test decode pipeline
            pred_feats = gt_feats_raw
            results_summary.append({
                'instance': instance,
                'tokens': sample['num_tokens'],
                'mse': 0.0,
                'cos_sim': 1.0,
            })

        # --- Test 3: Decode via Texture VAE ---
        if tex_decoder is not None and pred_feats is not None:
            # Build spatial cache by encoding shape from mesh
            type_id = id_to_type.get(instance)
            mesh_path = os.path.join(args.mesh_dir, type_id, f'{instance}.glb') if type_id else None
            if mesh_path and os.path.exists(mesh_path):
                print(f'  Encoding shape for spatial cache...')
                shape_slat = encode_shape_for_cache(mesh_path, shape_encoder, resolution, device)
                shape_coords_dec = shape_slat.coords[:, 1:]  # (M, 3)

                # Map pred_feats (on seg_latent coords) to shape_slat coords
                pred_at_shape = lookup_feats_at_coords(
                    shape_coords_dec, sample['seg_coords'], pred_feats
                )
                gt_at_shape = lookup_feats_at_coords(
                    shape_coords_dec, sample['seg_coords'], gt_feats_raw
                )

                dec_dtype = next(tex_decoder.parameters()).dtype

                print(f'  Decoding predicted latent ({shape_slat.feats.shape[0]} tokens)...')
                pred_st = shape_slat.replace(pred_at_shape.to(dec_dtype))
                seg_voxel = tex_decoder(pred_st)
                decoded_feats = seg_voxel.feats.float() * 2 - 1  # → [-1, 1]
                decoded_coords = seg_voxel.coords[:, 1:]

                labels = decode_feats_to_labels(decoded_feats, args.num_classes)
                unique_labels = labels.unique()
                print(f'  Decoded voxels: {decoded_feats.shape[0]}, unique labels: {len(unique_labels)}')

                vis_path = os.path.join(args.output_dir, f'{instance}_decoded.png')
                visualize_labels_3d(decoded_coords, labels, args.num_classes, vis_path)
                print(f'  Saved: {vis_path}')

                # Export GLB with per-face part colors
                glb_path = os.path.join(args.output_dir, f'{instance}_pred.glb')
                export_seg_glb(mesh_path, decoded_coords, labels, args.num_classes, glb_path, resolution)
                print(f'  Saved: {glb_path}')

                # --- Test 4: Decode GT latent for comparison ---
                print(f'  Decoding GT latent...')
                gt_st = shape_slat.replace(gt_at_shape.to(dec_dtype))
                gt_voxel = tex_decoder(gt_st)
                gt_decoded = gt_voxel.feats.float() * 2 - 1
                gt_labels = decode_feats_to_labels(gt_decoded, args.num_classes)
                gt_unique = gt_labels.unique()
                print(f'  GT decoded: {gt_decoded.shape[0]} voxels, {len(gt_unique)} unique labels')

                vis_path = os.path.join(args.output_dir, f'{instance}_gt.png')
                visualize_labels_3d(gt_voxel.coords[:, 1:], gt_labels, args.num_classes, vis_path)
                print(f'  Saved: {vis_path}')

                gt_glb_path = os.path.join(args.output_dir, f'{instance}_gt.glb')
                export_seg_glb(mesh_path, gt_voxel.coords[:, 1:], gt_labels, args.num_classes, gt_glb_path, resolution)
                print(f'  Saved: {gt_glb_path}')

                del shape_slat
            else:
                print(f'  Skip decode: mesh not found for {instance}')

        # Save cond image if available
        if sample['cond_image']:
            cond_path = os.path.join(args.output_dir, f'{instance}_cond.png')
            sample['cond_image'].convert('RGB').save(cond_path)

        # Save raw latent
        raw_path = os.path.join(args.output_dir, f'{instance}_latent.npz')
        np.savez_compressed(raw_path,
                           coords=sample['seg_coords'].cpu().numpy(),
                           feats=pred_feats.cpu().numpy())

        torch.cuda.empty_cache()

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'{"Instance":<40} {"Tokens":>7} {"MSE":>8} {"CosSim":>8}')
    print('-' * 65)
    for r in results_summary:
        print(f'{r["instance"][:38]:<40} {r["tokens"]:>7} {r["mse"]:>8.4f} {r["cos_sim"]:>8.4f}')

    if denoiser is not None:
        avg_mse = np.mean([r['mse'] for r in results_summary])
        avg_cos = np.mean([r['cos_sim'] for r in results_summary])
        print('-' * 65)
        print(f'{"AVERAGE":<40} {"":>7} {avg_mse:>8.4f} {avg_cos:>8.4f}')

    print(f'\nOutputs saved to: {args.output_dir}')
    print('Done!')


if __name__ == '__main__':
    main()
