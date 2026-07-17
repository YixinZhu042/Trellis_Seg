"""
Inference script for segmentation generation — aligned with TRELLIS texturing pipeline.

Input: mesh (.glb/.obj) + 6-view seg maps (or mv_seg_latent)
Output: mesh with per-face part colors baked as albedo texture (.glb)

Pipeline (mirrors trellis2_texturing):
  1. Preprocess mesh → normalize to [-0.5, 0.5]
  2. Encode shape → shape_slat (Shape VAE, same as TRELLIS)
  3. Load/encode mv_seg condition → mv_seg_slat
  4. Build concat_cond = [shape_slat(32), mv_seg_at_shape(32)] = 64ch
  5. Create noise on shape_slat coords (32ch)
  6. DIT sampling with CFG → seg_slat (32ch)
  7. Decode seg_slat via Texture VAE → 6ch voxels
  8. Sample voxels onto mesh faces → per-face part labels
  9. Export mesh with part colors as GLB

Usage:
    python inference.py \
        --config configs/slat_flow_seg_dit_1_3B_512_bf16.json \
        --ckpt ./experiments/seg_dit_100samples/ckpts/denoiser_ema0.9999_step0005000.pt \
        --mesh /path/to/input.glb \
        --mv_seg_latent /path/to/mv_seg_latent/instance.npz \
        --cond_image /path/to/render_cond/instance.png \
        --output ./result.glb \
        --steps 50
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import trimesh
from PIL import Image

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.modules.sparse.basic import SparseTensor
import trellis2.models as trellis2_models
import trellis2.modules.sparse as sp
from models import get_model

torch.set_grad_enabled(False)

# Model paths
SHAPE_ENC_PATH = '/path/to/trellis2_ckpts/shape_enc'
TEX_ENC_PATH = '/path/to/trellis2_ckpts/tex_enc'
TEX_DEC_PATH = '/path/to/trellis2_ckpts/tex_dec'


# ============================================================
# Color palette for part labels
# ============================================================

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
    """Decode 6ch features ([-1,1]) → integer labels via nearest color."""
    feats_01 = (feats_6ch + 1) / 2
    rgb = feats_01[:, 0:3]
    palette = torch.tensor(generate_class_colors(num_classes), device=feats_6ch.device)
    dists = torch.cdist(rgb.float(), palette.float())
    return dists.argmin(dim=-1)


# ============================================================
# Coordinate matching
# ============================================================

def lookup_feats_at_coords(target_coords, source_coords, source_feats):
    """Look up source features at target coordinates. Zeros for unmatched.

    Vectorized: duplicate source coords are averaged, then matched to targets via
    a sorted-search on linear indices (no per-voxel Python loop).
    """
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

    # Average features of source voxels sharing a linear index.
    uniq_lin, inverse = torch.unique(src_lin, return_inverse=True)
    sums = torch.zeros(uniq_lin.shape[0], C, device=source_feats.device, dtype=source_feats.dtype)
    sums.index_add_(0, inverse, source_feats)
    counts = torch.zeros(uniq_lin.shape[0], device=source_feats.device, dtype=source_feats.dtype)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=source_feats.dtype))
    uniq_feats = sums / counts.unsqueeze(-1)

    # Match each target linear index against the sorted unique source indices.
    pos = torch.searchsorted(uniq_lin, tgt_lin)
    pos = pos.clamp(max=uniq_lin.shape[0] - 1)
    hit = uniq_lin[pos] == tgt_lin
    matched[hit] = uniq_feats[pos[hit]]

    return matched


# ============================================================
# Shape encoding (same as TRELLIS texturing pipeline)
# ============================================================

def encode_shape(mesh, encoder, resolution=512, device='cuda'):
    """
    Encode mesh → shape_slat using o_voxel + Shape VAE encoder.
    Mirrors trellis2_texturing.encode_shape_slat().
    Returns SparseTensor with spatial cache intact.
    """
    import o_voxel
    
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices.cpu(), faces.cpu(),
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


# ============================================================
# Flow matching sampling
# ============================================================

@torch.no_grad()
def sample_flow(denoiser, cond_features, concat_cond, noise, steps=50, sigma_min=1e-5,
                mv_cond=None, mv_coords=None):
    """
    Euler sampling for flow matching, using TRELLIS.2's official FlowEulerSampler.

    IMPORTANT: the training convention (trellis2 FlowMatchingTrainer) is:
        x_t = (1-t)*x_0 + (sigma_min + (1-sigma_min)*t)*noise    # t=1 -> noise, t=0 -> data
        v   = (1-sigma_min)*noise - x_0
    so sampling must integrate t from 1 -> 0 with x_{prev} = x_t - (t-t_prev)*v.
    The previous hand-rolled loop ran t from 0 upward AND used x + dt*v (wrong
    direction + wrong sign), which produced pure garbage regardless of the ckpt.

    mv_cond / mv_coords: optional multi-view seg tokens (cross-attention). Passed
    through as extra kwargs to the denoiser's forward.
    """
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
        verbose=True,
    )
    return result.samples


# ============================================================
# Mesh postprocessing: bake part labels as vertex/face colors
# ============================================================

def bake_labels_on_mesh(mesh, seg_voxel, num_classes, resolution=512):
    """
    Sample seg voxel grid onto mesh faces → per-face part color.
    
    Args:
        mesh: trimesh.Trimesh (normalized to [-0.5, 0.5])
        seg_voxel: SparseTensor with feats (N, 6) in [0, 1]
        num_classes: int
        resolution: voxelization resolution
    
    Returns:
        trimesh.Trimesh with face colors
    """
    # Decode voxel feats → labels
    labels = decode_feats_to_labels(seg_voxel.feats.float() * 2 - 1, num_classes)
    coords = seg_voxel.coords[:, 1:]  # (N, 3) — in high-res space [0, resolution-1]
    
    # Sample face centers
    face_centers = mesh.triangles_center  # (F, 3)
    face_centers_t = torch.from_numpy(face_centers).float().to(labels.device)
    
    # Convert face centers to voxel coordinates (same space as decoded voxels)
    # mesh is in [-0.5, 0.5], decoded voxels are in [0, resolution-1]
    face_voxel_coords = ((face_centers_t + 0.5) * resolution).long().clamp(0, resolution - 1)
    
    # Build lookup from voxel coords → label.
    # Base must cover BOTH the decoded voxel coords and the clamped face_voxel_coords
    # (which can reach resolution-1 > coords.max()), otherwise the linear hash is not
    # unique and faces collide onto wrong voxel keys → mislabeled parts.
    res = max(coords.max().item(), face_voxel_coords.max().item()) + 1
    coord_lin = coords[:, 0].long() * res * res + coords[:, 1].long() * res + coords[:, 2].long()
    face_lin = face_voxel_coords[:, 0] * res * res + face_voxel_coords[:, 1] * res + face_voxel_coords[:, 2]
    
    # Hash map for O(1) lookup
    label_map = {}
    for i in range(coords.shape[0]):
        label_map[coord_lin[i].item()] = labels[i].item()
    
    # For faces that don't hit a voxel exactly, find nearest voxel
    # Build a KD-tree for fallback
    coords_np = coords.cpu().numpy().astype(np.float32)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords_np)
    
    # Assign face labels
    palette = generate_class_colors(num_classes)
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    face_voxel_np = face_voxel_coords.cpu().numpy().astype(np.float32)
    
    # First pass: exact lookup
    miss_indices = []
    for f_idx in range(len(mesh.faces)):
        key = face_lin[f_idx].item()
        if key in label_map:
            label = label_map[key]
            c = palette[label % num_classes]
            face_colors[f_idx] = [int(c[0]*255), int(c[1]*255), int(c[2]*255), 255]
        else:
            miss_indices.append(f_idx)
    
    # Second pass: nearest-neighbor for misses
    if miss_indices:
        miss_coords = face_voxel_np[miss_indices]
        _, nn_idx = tree.query(miss_coords)
        for i, f_idx in enumerate(miss_indices):
            label = labels[nn_idx[i]].item()
            c = palette[label % num_classes]
            face_colors[f_idx] = [int(c[0]*255), int(c[1]*255), int(c[2]*255), 255]
    
    print(f'  Exact hits: {len(mesh.faces) - len(miss_indices)}/{len(mesh.faces)}, '
          f'NN fallback: {len(miss_indices)}')
    
    # Apply face colors
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = face_colors
    return colored_mesh


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Seg generation: mesh in → colored mesh out')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Denoiser checkpoint path')
    parser.add_argument('--mesh', type=str, required=True,
                        help='Input mesh (.glb/.obj)')
    parser.add_argument('--mv_seg_latent', type=str, default=None,
                        help='Pre-encoded mv_seg_latent .npz (optional)')
    parser.add_argument('--cond_image', type=str, default=None,
                        help='Condition image for DINOv3 (optional)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .glb path')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = args.device
    config = json.load(open(args.config))
    
    # --------------------------------------------------------
    # 1. Load mesh & preprocess (same as TRELLIS texturing)
    # --------------------------------------------------------
    print(f'Loading mesh: {args.mesh}')
    mesh = trimesh.load(args.mesh, force='mesh')
    
    # Normalize to [-0.5, 0.5].
    # NOTE: do NOT apply the Y/Z axis swap from TRELLIS's preprocess_mesh here.
    # The training preprocessing (preprocess/step3_encode_shape_latent.py) normalizes
    # WITHOUT swapping axes, so the DiT was trained in the un-swapped frame. Swapping
    # here rotated the shape encoding into a frame the model never saw (only ~29% of
    # voxel coords matched the training grid), collapsing the generated segmentation.
    vertices = mesh.vertices
    center = (vertices.max(0) + vertices.min(0)) / 2
    scale = 0.99999 / (vertices.max(0) - vertices.min(0)).max()
    vertices = (vertices - center) * scale
    mesh_normalized = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    print(f'  Vertices: {len(vertices)}, Faces: {len(mesh.faces)}')
    
    # --------------------------------------------------------
    # 2. Encode shape → shape_slat (with spatial cache)
    # --------------------------------------------------------
    print('Encoding shape...')
    shape_encoder = trellis2_models.from_pretrained(SHAPE_ENC_PATH).eval().to(device)
    shape_slat = encode_shape(mesh_normalized, shape_encoder, args.resolution, device)
    del shape_encoder
    torch.cuda.empty_cache()
    print(f'  shape_slat: {shape_slat.feats.shape[0]} tokens × {shape_slat.feats.shape[1]}ch')
    
    # --------------------------------------------------------
    # 3. Load mv_seg condition (as cross-attention tokens, NOT concat)
    # --------------------------------------------------------
    mv_as_cross_attn = config['dataset']['args'].get('mv_as_cross_attn', False)
    mv_cond_feats = None
    mv_cond_coords = None
    if args.mv_seg_latent:
        print(f'Loading mv_seg_latent: {args.mv_seg_latent}')
        mv_data = np.load(args.mv_seg_latent)
        mv_coords = torch.tensor(mv_data['coords'], dtype=torch.float32, device=device)
        mv_feats = torch.tensor(mv_data['feats'], dtype=torch.float32, device=device)

        mv_norm = config['dataset']['args'].get('mv_cond_normalization')
        if mv_norm is not None:
            mv_mean = torch.tensor(mv_norm['mean'], device=device).reshape(1, -1)
            mv_std = torch.tensor(mv_norm['std'], device=device).reshape(1, -1).clamp_min(1e-6)
            mv_feats = (mv_feats - mv_mean) / mv_std

        if mv_as_cross_attn:
            # Rescale mv coords to the main latent grid's positional space [0, res/16-1].
            latent_grid = max(args.resolution // 16, 1)              # 512//16 = 32
            enc_res = config['dataset']['args'].get('mv_enc_resolution', 64)
            enc_grid = max(enc_res // 16, 1)                          # 64//16 = 4
            scale = (latent_grid - 1) / max(enc_grid - 1, 1)
            mv_cond_feats = [mv_feats]                                # batch of 1
            mv_cond_coords = [mv_coords * scale]
    elif not mv_as_cross_attn:
        # Legacy concat path fallback
        mv_coords = shape_slat.coords[:, 1:].int()
        mv_feats = torch.zeros(shape_slat.feats.shape[0], 32, device=device)

    # --------------------------------------------------------
    # 4. Build concat_cond on shape_slat coords (shape geometry only, 32ch)
    # --------------------------------------------------------
    shape_norm = config['dataset']['args'].get('shape_cond_normalization')
    shape_feats_normed = shape_slat.feats.float()
    if shape_norm is not None:
        s_mean = torch.tensor(shape_norm['mean'], device=device).reshape(1, -1)
        s_std = torch.tensor(shape_norm['std'], device=device).reshape(1, -1).clamp_min(1e-6)
        shape_feats_normed = (shape_feats_normed - s_mean) / s_std

    if mv_as_cross_attn:
        # in_channels = 64: [noise(32) | shape(32)]
        concat_cond = shape_slat.replace(shape_feats_normed)
        concat_cond._shape = torch.Size([1, 32])
    else:
        # Legacy: concat_cond = [shape(32), mv_seg(32)] = 64ch
        shape_coords = shape_slat.coords[:, 1:].int()
        mv_at_shape = lookup_feats_at_coords(shape_coords, mv_coords, mv_feats)
        concat_feats = torch.cat([shape_feats_normed, mv_at_shape], dim=-1)
        concat_cond = shape_slat.replace(concat_feats)
        concat_cond._shape = torch.Size([1, 64])
    
    # --------------------------------------------------------
    # 5. Create noise (32ch) on shape_slat coords
    # --------------------------------------------------------
    noise = shape_slat.replace(torch.randn(shape_slat.feats.shape[0], 32, device=device))
    noise._shape = torch.Size([1, 32])
    
    # --------------------------------------------------------
    # 6. Load denoiser & sample
    # --------------------------------------------------------
    print('Loading denoiser...')
    model_cfg = config['models']['denoiser']
    denoiser = get_model(model_cfg['name'], **model_cfg['args'])
    state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    denoiser.load_state_dict(state_dict)
    denoiser = denoiser.to(device).to(torch.bfloat16).eval()
    
    # DINOv3 condition
    if args.cond_image:
        from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
        image_cond_cfg = config['trainer']['args']['image_cond_model']
        print(f'Loading DINOv3...')
        extractor = DinoV3FeatureExtractor(**image_cond_cfg['args'])
        extractor.cuda()
        # Match training preprocessing (datasets/seg_dataset._load_cond_image):
        # crop to alpha bbox, resize, composite on BLACK (image * alpha), not white.
        img = Image.open(args.cond_image)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        image_size = config['dataset']['args'].get('image_size', 512)
        alpha = np.array(img.getchannel(3))
        nz = alpha.nonzero()
        if len(nz[0]) > 0:
            bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_bbox = [int(center[0] - hsize), int(center[1] - hsize),
                        int(center[0] + hsize), int(center[1] + hsize)]
            img = img.crop(aug_bbox)
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        alpha_t = torch.tensor(np.array(img.getchannel(3))).float() / 255.0
        rgb = torch.tensor(np.array(img.convert('RGB'))).permute(2, 0, 1).float() / 255.0
        rgb = rgb * alpha_t.unsqueeze(0)  # black background, matches training
        cond_features = extractor(rgb.unsqueeze(0).cuda())
    else:
        cond_features = torch.zeros(1, 1029, 1024, device=device)

    print(f'Sampling: {args.steps} steps (no CFG)')
    sigma_min = config['trainer']['args'].get('sigma_min', 1e-5)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        seg_slat = sample_flow(
            denoiser, cond_features, concat_cond, noise,
            steps=args.steps, sigma_min=sigma_min,
            mv_cond=mv_cond_feats, mv_coords=mv_cond_coords,
        )
    
    del denoiser
    torch.cuda.empty_cache()
    
    # Denormalize seg_slat (inverse of training normalization; clamp std to match).
    seg_norm = config['dataset']['args'].get('normalization')
    if seg_norm is not None:
        s_mean = torch.tensor(seg_norm['mean'], device=device).reshape(1, -1)
        s_std = torch.tensor(seg_norm['std'], device=device).reshape(1, -1).clamp_min(1e-6)
        seg_slat = seg_slat.replace(seg_slat.feats.float() * s_std + s_mean)
    
    # --------------------------------------------------------
    # 7. Decode seg_slat → 6ch voxels via Texture VAE
    #    (seg_slat inherits spatial cache from shape_slat → decoder works directly)
    # --------------------------------------------------------
    print('Decoding seg latent...')
    tex_decoder = trellis2_models.from_pretrained(TEX_DEC_PATH).eval().to(device)
    dec_dtype = next(tex_decoder.parameters()).dtype
    seg_slat_cast = seg_slat.replace(seg_slat.feats.to(dec_dtype))
    seg_voxel = tex_decoder(seg_slat_cast) * 0.5 + 0.5  # → [0, 1]
    del tex_decoder
    torch.cuda.empty_cache()
    print(f'  Decoded voxels: {seg_voxel.feats.shape}')
    
    # --------------------------------------------------------
    # 8. Bake labels onto mesh
    # --------------------------------------------------------
    print('Baking part labels on mesh...')
    colored_mesh = bake_labels_on_mesh(
        mesh_normalized, seg_voxel, args.num_classes, args.resolution
    )
    
    # --------------------------------------------------------
    # 9. Export GLB
    # --------------------------------------------------------
    # No axis swap on input (see note in step 1), so no un-swap needed here.
    # mesh_normalized is a uniformly scaled/translated copy of the input mesh.
    colored_mesh.export(args.output)
    print(f'\nExported: {args.output}')
    print(f'  Faces: {len(colored_mesh.faces)}')
    
    # Also save raw latent
    raw_path = args.output.replace('.glb', '_latent.npz')
    np.savez_compressed(raw_path,
                       coords=seg_slat.coords[:, 1:].cpu().numpy(),
                       feats=seg_slat.feats.float().cpu().numpy())
    print(f'  Raw latent: {raw_path}')
    print('\nDone!')


if __name__ == '__main__':
    main()
