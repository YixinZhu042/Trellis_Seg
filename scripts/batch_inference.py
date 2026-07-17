"""
Batch inference: fully online from mesh.

Correct decode strategy:
  Training: DIT works on seg_latent coords (from Seg VAE encoder).
  So inference must also use Seg VAE encoder's coords/cache.

  Flow:
    1. Voxelize mesh (same as step4) → high-res coords
    2. Seg VAE encoder(coords + dummy one-hot) → z with correct latent coords + spatial cache
    3. Shape VAE encoder(mesh) → shape_slat → lookup shape feats at z's latent coords → concat_cond
    4. DIT sampling on z's coords → seg_slat feats
    5. z.replace(seg_slat_feats) → Seg VAE decoder → 20ch logits → labels

Usage:
    python scripts/batch_inference.py --num_samples 10 --output_dir ./batch_results
"""
import os
import sys
import json
import argparse
import glob
import numpy as np
import torch
import trimesh
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

TRELLIS2_ROOT = os.path.join(PROJECT_DIR, '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.modules.sparse.basic import SparseTensor
import trellis2.models as trellis2_models
import trellis2.modules.sparse as sp
from models import get_model

torch.set_grad_enabled(False)

# ============================================================
# Paths
# ============================================================
DATA_DIR = '/path/to/data'
MESH_DIR = '/path/to/PartNeXt_mesh/glbs'
CONFIG_PATH = os.path.join(PROJECT_DIR, 'configs/slat_flow_seg_dit_1_3B_512_bf16.json')
DIT_CKPT = './output/seg_dit/ckpts/denoiser_ema0.9999_step0036000.pt'
SHAPE_ENC_PATH = '/path/to/trellis2_ckpts/shape_enc'
SEG_VAE_ENC_CKPT = './checkpoints/seg_vae/encoder_step0001000.pt'
SEG_VAE_DEC_CKPT = './checkpoints/seg_vae/decoder_step0001000.pt'
SEG_VAE_CONFIG = os.path.join(PROJECT_DIR, 'configs/seg_vae_train.json')

NUM_CLASSES = 20

# ============================================================
# Color palette
# ============================================================
def generate_class_colors(num_classes):
    BASE_PALETTE = np.array([
        [0.900, 0.098, 0.294], [0.235, 0.706, 0.294], [0.263, 0.388, 0.847],
        [0.957, 0.831, 0.118], [0.522, 0.078, 0.694], [0.957, 0.522, 0.114],
        [0.275, 0.941, 0.941], [0.941, 0.196, 0.902], [0.824, 0.961, 0.235],
        [0.980, 0.745, 0.831], [0.000, 0.502, 0.502], [0.902, 0.745, 1.000],
        [0.667, 0.431, 0.157], [1.000, 0.980, 0.784], [0.502, 0.000, 0.000],
        [0.667, 1.000, 0.765], [0.502, 0.502, 0.000], [1.000, 0.843, 0.706],
        [0.000, 0.000, 0.502], [0.502, 0.502, 0.502],
    ], dtype=np.float32)
    if num_classes <= 20:
        return BASE_PALETTE[:num_classes]
    colors = np.zeros((num_classes, 3), dtype=np.float32)
    colors[:20] = BASE_PALETTE
    for i in range(20, num_classes):
        colors[i] = BASE_PALETTE[i % 20] * 0.6
    return colors


# ============================================================
# Voxelize mesh — SAME as step4_encode_seg_latent.py
# ============================================================
def voxelize_mesh_like_step4(mesh, resolution=512, max_voxels=30000):
    """
    Voxelize mesh surface → coords (N, 3).
    Matches step4's voxelize_with_labels() exactly (same normalization, same seeded subsample).
    """
    num_samples = min(len(mesh.faces) * 10, 100000)
    points, _ = mesh.sample(num_samples, return_index=True)

    # Normalize (same as step4)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = (bbox_max - bbox_min).max()
    if bbox_size < 1e-8:
        return None
    center = (bbox_min + bbox_max) / 2
    points_norm = (points - center) / bbox_size * (resolution - 3) + resolution / 2
    coords = points_norm.astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)

    # Deduplicate
    coord_hash = (coords[:, 0].astype(np.int64) * resolution * resolution +
                  coords[:, 1].astype(np.int64) * resolution +
                  coords[:, 2].astype(np.int64))
    unique_hash = np.unique(coord_hash)
    unique_coords = np.stack([
        unique_hash // (resolution * resolution),
        (unique_hash % (resolution * resolution)) // resolution,
        unique_hash % resolution
    ], axis=1).astype(np.int32)

    # Subsample (same seed logic as step4)
    if len(unique_coords) > max_voxels:
        rng = np.random.default_rng(seed=len(unique_coords))
        idx = rng.choice(len(unique_coords), max_voxels, replace=False)
        unique_coords = unique_coords[idx]

    return unique_coords


# ============================================================
# Build Seg VAE cache: encoder with dummy feats on real coords
# ============================================================
def build_seg_vae_latent(seg_vae_encoder, voxel_coords, num_classes, device):
    """
    Run Seg VAE encoder on voxel_coords with dummy one-hot input.
    Returns z: SparseTensor on latent grid with correct spatial cache.
    The feats are meaningless — only coords and cache matter.
    """
    N = len(voxel_coords)
    dummy_feats = torch.zeros(N, num_classes, dtype=torch.float32, device=device)
    dummy_feats[:, 0] = 1.0  # valid one-hot, content doesn't matter

    batch_coords = torch.cat([
        torch.zeros(N, 1, dtype=torch.int32, device=device),
        torch.tensor(voxel_coords, dtype=torch.int32, device=device)
    ], dim=-1)

    x = sp.SparseTensor(dummy_feats, batch_coords)
    z = seg_vae_encoder(x, sample_posterior=False)
    return z


# ============================================================
# Shape encoding
# ============================================================
def encode_shape(mesh, encoder, resolution=512, device='cuda'):
    import o_voxel
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices.cpu(), faces.cpu(),
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )
    vert_st = SparseTensor(
        feats=dual_vertices * resolution - voxel_indices,
        coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
    ).to(device)
    inter_st = vert_st.replace(intersected).to(device)
    return encoder(vert_st, inter_st)


# ============================================================
# Coordinate matching
# ============================================================
def lookup_feats_at_coords(target_coords, source_coords, source_feats):
    N = target_coords.shape[0]
    C = source_feats.shape[-1]
    matched = torch.zeros(N, C, device=source_feats.device, dtype=source_feats.dtype)
    if source_coords.shape[0] == 0 or N == 0:
        return matched
    res = max(target_coords.max().item(), source_coords.max().item()) + 1
    src_lin = (source_coords[:, 0].long() * res * res +
               source_coords[:, 1].long() * res +
               source_coords[:, 2].long())
    tgt_lin = (target_coords[:, 0].long() * res * res +
               target_coords[:, 1].long() * res +
               target_coords[:, 2].long())
    uniq_lin, inverse = torch.unique(src_lin, return_inverse=True)
    sums = torch.zeros(uniq_lin.shape[0], C, device=source_feats.device, dtype=source_feats.dtype)
    sums.index_add_(0, inverse, source_feats)
    counts = torch.zeros(uniq_lin.shape[0], device=source_feats.device, dtype=source_feats.dtype)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=source_feats.dtype))
    uniq_feats = sums / counts.unsqueeze(-1)
    pos = torch.searchsorted(uniq_lin, tgt_lin).clamp(max=uniq_lin.shape[0] - 1)
    hit = uniq_lin[pos] == tgt_lin
    matched[hit] = uniq_feats[pos[hit]]
    return matched


# ============================================================
# Bake labels on mesh
# ============================================================
def bake_labels_on_mesh(mesh, voxel_coords, voxel_labels, num_classes, resolution=512):
    from scipy.spatial import cKDTree
    palette = generate_class_colors(num_classes)
    face_centers = mesh.triangles_center
    # Use same normalization as step4 for face centers → voxel space
    verts = mesh.vertices
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_size = (bbox_max - bbox_min).max()
    center = (bbox_min + bbox_max) / 2
    face_voxel = ((face_centers - center) / bbox_size * (resolution - 3) + resolution / 2).astype(np.float32)

    coords_np = voxel_coords.cpu().numpy().astype(np.float32)
    labels_np = voxel_labels.cpu().numpy()
    tree = cKDTree(coords_np)
    _, nn_idx = tree.query(face_voxel)
    face_labels = labels_np[nn_idx]

    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    rgb = palette[face_labels % num_classes]
    face_colors[:, :3] = (rgb * 255).astype(np.uint8)
    face_colors[:, 3] = 255

    # Split vertices so each face has unique verts (prevents color blending at shared vertices)
    verts_split = mesh.vertices[mesh.faces].reshape(-1, 3)
    faces_split = np.arange(len(verts_split)).reshape(-1, 3)
    vert_colors = np.repeat(face_colors, 3, axis=0)
    colored = trimesh.Trimesh(vertices=verts_split, faces=faces_split,
                              vertex_colors=vert_colors, process=False)
    return colored, face_labels


# ============================================================
# Find mesh
# ============================================================
def find_mesh_path(instance):
    results = glob.glob(os.path.join(MESH_DIR, '**', f'{instance}.glb'), recursive=True)
    return results[0] if results else None


# ============================================================
# Flow sampling
# ============================================================
def sample_flow(denoiser, cond_features, concat_cond, noise, steps=50, sigma_min=1e-5,
                mv_cond=None, mv_coords=None):
    from trellis2.pipelines.samplers import FlowEulerSampler
    sampler = FlowEulerSampler(sigma_min=sigma_min)
    result = sampler.sample(
        denoiser, noise=noise, cond=cond_features,
        concat_cond=concat_cond,
        mv_cond=mv_cond, mv_coords=mv_coords,
        steps=steps, verbose=False,
    )
    return result.samples


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./batch_results')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dit_ckpt', type=str, default=DIT_CKPT)
    parser.add_argument('--config', type=str, default=CONFIG_PATH)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = json.load(open(args.config))
    dataset_cfg = config['dataset']['args']
    sigma_min = config['trainer']['args'].get('sigma_min', 1e-5)

    # Normalization
    seg_norm = dataset_cfg.get('normalization')
    shape_norm = dataset_cfg.get('shape_cond_normalization')
    mv_norm = dataset_cfg.get('mv_cond_normalization')
    if seg_norm:
        seg_mean = torch.tensor(seg_norm['mean'], device=device).reshape(1, -1)
        seg_std = torch.tensor(seg_norm['std'], device=device).reshape(1, -1).clamp_min(1e-6)
    if shape_norm:
        shape_mean = torch.tensor(shape_norm['mean'], device=device).reshape(1, -1)
        shape_std = torch.tensor(shape_norm['std'], device=device).reshape(1, -1).clamp_min(1e-6)
    if mv_norm:
        mv_mean = torch.tensor(mv_norm['mean'], device=device).reshape(1, -1)
        mv_std = torch.tensor(mv_norm['std'], device=device).reshape(1, -1).clamp_min(1e-6)

    # Instances
    seg_latent_dir = os.path.join(args.data_dir, 'seg_latent')
    all_instances = sorted([f[:-4] for f in os.listdir(seg_latent_dir) if f.endswith('.npz')])
    selected = np.random.choice(all_instances, min(args.num_samples, len(all_instances)), replace=False)
    print(f'Selected {len(selected)} samples from {len(all_instances)} total')

    # ----------------------------------------------------------
    # Load models
    # ----------------------------------------------------------
    print('Loading Shape VAE encoder...')
    shape_encoder = trellis2_models.from_pretrained(SHAPE_ENC_PATH).eval().to(device)

    print('Loading Seg VAE encoder + decoder...')
    seg_vae_cfg = json.load(open(SEG_VAE_CONFIG))
    enc_args = seg_vae_cfg['models']['encoder']['args']
    dec_args = seg_vae_cfg['models']['decoder']['args']

    seg_vae_encoder = trellis2_models.SparseUnetVaeEncoder(**enc_args).eval().to(device)
    enc_state = torch.load(SEG_VAE_ENC_CKPT, map_location=device, weights_only=True)
    seg_vae_encoder.load_state_dict(enc_state)
    del enc_state

    seg_vae_decoder = trellis2_models.SparseUnetVaeDecoder(**dec_args).eval().to(device)
    dec_state = torch.load(SEG_VAE_DEC_CKPT, map_location=device, weights_only=True)
    seg_vae_decoder.load_state_dict(dec_state)
    del dec_state

    print('Loading DIT denoiser...')
    model_cfg = config['models']['denoiser']
    denoiser = get_model(model_cfg['name'], **model_cfg['args'])
    dit_state = torch.load(args.dit_ckpt, map_location='cpu', weights_only=True)
    denoiser.load_state_dict(dit_state)
    denoiser = denoiser.to(device).to(torch.bfloat16).eval()
    del dit_state

    torch.cuda.empty_cache()
    print(f'All models loaded.\n')

    # ----------------------------------------------------------
    # Per-sample inference
    # ----------------------------------------------------------
    for idx, instance in enumerate(tqdm(selected, desc='Inference')):
        try:
            out_path = os.path.join(args.output_dir, f'{instance}.glb')
            if os.path.exists(out_path):
                continue

            # --- Find and load mesh ---
            mesh_path = find_mesh_path(instance)
            if mesh_path is None:
                print(f'  [{idx+1}] {instance}: mesh not found, skipping')
                continue
            mesh = trimesh.load(mesh_path, force='mesh')
            # NOTE: do NOT normalize mesh here — voxelize_mesh_like_step4 does its own normalization
            # But we need a normalized mesh for bake_labels_on_mesh and export
            verts = mesh.vertices
            center_mesh = (verts.max(0) + verts.min(0)) / 2
            scale_mesh = 0.99999 / (verts.max(0) - verts.min(0)).max()
            verts_norm = (verts - center_mesh) * scale_mesh
            mesh_norm = trimesh.Trimesh(vertices=verts_norm, faces=mesh.faces, process=False)

            # ==========================================================
            # STEP A: Voxelize mesh → Seg VAE encoder → z (latent coords + cache)
            # ==========================================================
            voxel_coords = voxelize_mesh_like_step4(mesh, args.resolution)
            if voxel_coords is None:
                print(f'  [{idx+1}] {instance}: voxelization failed, skipping')
                continue
            seg_vae_z = build_seg_vae_latent(seg_vae_encoder, voxel_coords, NUM_CLASSES, device)
            seg_latent_coords = seg_vae_z.coords[:, 1:].int()  # latent grid coords
            n_seg_tokens = seg_vae_z.feats.shape[0]

            # ==========================================================
            # STEP B: Shape VAE encoder → shape_slat → lookup at seg_vae latent coords
            # ==========================================================
            shape_slat = encode_shape(mesh_norm, shape_encoder, args.resolution, device)
            shape_latent_coords = shape_slat.coords[:, 1:].int()

            # Lookup shape feats at seg_vae_z's latent coords
            shape_at_seg = lookup_feats_at_coords(
                seg_latent_coords, shape_latent_coords, shape_slat.feats.float()
            )
            if shape_norm:
                shape_at_seg = (shape_at_seg - shape_mean) / shape_std

            # Build concat_cond on seg_vae_z coords (with seg_vae cache!)
            concat_cond = seg_vae_z.replace(shape_at_seg)
            concat_cond._shape = torch.Size([1, 32])

            # ==========================================================
            # STEP C: mv_seg condition
            # ==========================================================
            mv_data = np.load(os.path.join(args.data_dir, 'mv_seg_latent', f'{instance}.npz'))
            mv_coords = torch.tensor(mv_data['coords'], dtype=torch.float32, device=device)
            mv_feats = torch.tensor(mv_data['feats'], dtype=torch.float32, device=device)
            if mv_norm:
                mv_feats = (mv_feats - mv_mean) / mv_std
            mv_cond_feats = [mv_feats]
            mv_cond_coords = [mv_coords]

            # ==========================================================
            # STEP D: Cross-attention condition (zeros — DINOv3 removed)
            # ==========================================================
            cond_features = torch.zeros(1, 1029, 1024, device=device)

            # ==========================================================
            # STEP E: Create noise on seg_vae_z coords (inherits seg_vae cache!)
            # ==========================================================
            noise = seg_vae_z.replace(torch.randn(n_seg_tokens, 32, device=device))
            noise._shape = torch.Size([1, 32])

            # ==========================================================
            # STEP F: DIT sampling
            # ==========================================================
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                seg_slat = sample_flow(
                    denoiser, cond_features, concat_cond, noise,
                    steps=args.steps, sigma_min=sigma_min,
                    mv_cond=mv_cond_feats, mv_coords=mv_cond_coords,
                )

            # Denormalize
            if seg_norm:
                seg_slat = seg_slat.replace(seg_slat.feats.float() * seg_std + seg_mean)

            # ==========================================================
            # STEP G: Decode via Seg VAE decoder (cache from seg_vae_z!)
            # ==========================================================
            dec_dtype = next(seg_vae_decoder.parameters()).dtype
            z_decode = seg_vae_z.replace(seg_slat.feats.to(dec_dtype))
            z_decode._shape = torch.Size([1, 32])
            logits = seg_vae_decoder(z_decode)
            pred_labels = logits.feats.float().argmax(dim=-1)
            pred_coords = logits.coords[:, 1:]

            # Bake on mesh
            colored_mesh, face_labels = bake_labels_on_mesh(
                mesh, pred_coords, pred_labels, NUM_CLASSES, args.resolution
            )
            colored_mesh.export(out_path)

            # ==========================================================
            # GT: decode ground-truth seg_latent with same cache
            # ==========================================================
            seg_data = np.load(os.path.join(args.data_dir, 'seg_latent', f'{instance}.npz'))
            gt_seg_coords = torch.tensor(seg_data['coords'], dtype=torch.int32, device=device)
            gt_seg_feats = torch.tensor(seg_data['feats'], dtype=torch.float32, device=device)
            # GT feats are already on seg_vae's latent coords (produced by same encoder)
            # Just lookup at our seg_vae_z latent coords
            gt_at_z = lookup_feats_at_coords(seg_latent_coords, gt_seg_coords, gt_seg_feats)
            gt_z = seg_vae_z.replace(gt_at_z.to(dec_dtype))
            gt_z._shape = torch.Size([1, 32])
            gt_logits = seg_vae_decoder(gt_z)
            gt_labels = gt_logits.feats.float().argmax(dim=-1)
            gt_coords = gt_logits.coords[:, 1:]
            gt_mesh, _ = bake_labels_on_mesh(mesh, gt_coords, gt_labels, NUM_CLASSES, args.resolution)
            gt_mesh.export(os.path.join(args.output_dir, f'{instance}_gt.glb'))

            # Stats
            # Check coord overlap between our voxelization and the saved seg_latent
            res32 = seg_latent_coords.max().item() + 1
            our_lin = (seg_latent_coords[:, 0].long() * res32 * res32 +
                       seg_latent_coords[:, 1].long() * res32 +
                       seg_latent_coords[:, 2].long())
            gt_lin = (gt_seg_coords[:, 0].long() * res32 * res32 +
                      gt_seg_coords[:, 1].long() * res32 +
                      gt_seg_coords[:, 2].long())
            our_set = set(our_lin.cpu().numpy().tolist())
            gt_set = set(gt_lin.cpu().numpy().tolist())
            overlap = len(our_set & gt_set)
            coord_iou = overlap / max(len(our_set | gt_set), 1)

            cos_sim = torch.nn.functional.cosine_similarity(
                seg_slat.feats.float(), gt_at_z, dim=-1
            )
            valid_mask = gt_at_z.abs().sum(-1) > 1e-6
            cos_valid = cos_sim[valid_mask].mean().item() if valid_mask.any() else 0.0

            n_parts = len(np.unique(face_labels))
            print(f'  [{idx+1}] {instance}: {n_seg_tokens} seg tokens, '
                  f'coord_iou={coord_iou:.3f} ({overlap}/{len(gt_set)} gt matched), '
                  f'cos_sim={cos_valid:.4f}, {n_parts} parts')

        except Exception as e:
            import traceback
            print(f'  [{idx+1}] {instance}: ERROR - {e}')
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

    print(f'\nDone! Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
