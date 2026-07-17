"""
In-the-wild inference: mesh + 6-view seg maps → colored segmentation mesh.

Usage:
    python scripts/inference_wild.py \
        --mesh /path/to/input.glb \
        --seg_maps /path/to/seg_maps.npz \
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

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
TRELLIS2_ROOT = os.path.join(PROJECT_DIR, '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.modules.sparse.basic import SparseTensor
import trellis2.models as trellis2_models
import trellis2.modules.sparse as sp
from models import get_model
from datasets.seg_dataset import colorize_seg_map, box_surface_image_ids
from trellis2.pipelines.samplers import FlowEulerSampler
from scipy.spatial import cKDTree

torch.set_grad_enabled(False)

# Default paths
CONFIG_PATH = os.path.join(PROJECT_DIR, 'configs/slat_flow_seg_dit_1_3B_512_bf16.json')
SHAPE_ENC_PATH = '/path/to/trellis2_ckpts/shape_enc'
SEG_VAE_ENC_CKPT = './checkpoints/seg_vae/encoder_step0001000.pt'
SEG_VAE_DEC_CKPT = './checkpoints/seg_vae/decoder_step0001000.pt'
DIT_CKPT = './output/seg_dit/ckpts/denoiser_ema0.9999_step0036000.pt'
SEG_VAE_CONFIG = os.path.join(PROJECT_DIR, 'configs/seg_vae_train.json')

NUM_CLASSES = 20
RESOLUTION = 512


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


def main():
    parser = argparse.ArgumentParser(description='In-the-wild 3D segmentation inference')
    parser.add_argument('--mesh', type=str, required=True, help='Input mesh (.glb/.obj)')
    parser.add_argument('--seg_maps', type=str, required=True,
                        help='seg_maps.npz with seg_maps (6,H,W) int32, 0=background')
    parser.add_argument('--output', type=str, required=True, help='Output .glb path')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--config', type=str, default=CONFIG_PATH)
    parser.add_argument('--dit_ckpt', type=str, default=DIT_CKPT)
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    config = json.load(open(args.config))
    dc = config['dataset']['args']
    sigma_min = config['trainer']['args'].get('sigma_min', 1e-5)

    seg_norm = dc.get('normalization')
    shape_norm = dc.get('shape_cond_normalization')
    mv_norm = dc.get('mv_cond_normalization')
    if seg_norm:
        seg_mean = torch.tensor(seg_norm['mean'], device=device).reshape(1, -1)
        seg_std = torch.tensor(seg_norm['std'], device=device).reshape(1, -1).clamp_min(1e-6)
    if shape_norm:
        shape_mean = torch.tensor(shape_norm['mean'], device=device).reshape(1, -1)
        shape_std = torch.tensor(shape_norm['std'], device=device).reshape(1, -1).clamp_min(1e-6)
    if mv_norm:
        mv_mean = torch.tensor(mv_norm['mean'], device=device).reshape(1, -1)
        mv_std = torch.tensor(mv_norm['std'], device=device).reshape(1, -1).clamp_min(1e-6)

    # ================================================================
    # Load models
    # ================================================================
    print('Loading models...')
    shape_enc = trellis2_models.from_pretrained(SHAPE_ENC_PATH).eval().to(device)

    svc = json.load(open(SEG_VAE_CONFIG))
    seg_enc = trellis2_models.SparseUnetVaeEncoder(**svc['models']['encoder']['args']).eval().to(device)
    seg_enc.load_state_dict(torch.load(SEG_VAE_ENC_CKPT, map_location=device, weights_only=True))
    seg_dec = trellis2_models.SparseUnetVaeDecoder(**svc['models']['decoder']['args']).eval().to(device)
    seg_dec.load_state_dict(torch.load(SEG_VAE_DEC_CKPT, map_location=device, weights_only=True))

    mc = config['models']['denoiser']
    dit = get_model(mc['name'], **mc['args'])
    dit.load_state_dict(torch.load(args.dit_ckpt, map_location='cpu', weights_only=True))
    dit = dit.to(device).to(torch.bfloat16).eval()

    torch.cuda.empty_cache()
    print('All models loaded.')

    # ================================================================
    # Load mesh
    # ================================================================
    print(f'Loading mesh: {args.mesh}')
    scene = trimesh.load(args.mesh)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    v = mesh.vertices
    c0 = (v.max(0) + v.min(0)) / 2
    s0 = 0.99999 / (v.max(0) - v.min(0)).max()
    mesh_norm = trimesh.Trimesh(vertices=(v - c0) * s0, faces=mesh.faces, process=False)
    print(f'  {len(mesh.vertices)} verts, {len(mesh.faces)} faces')

    # ================================================================
    # Step 1: Voxelize → Seg VAE encoder → spatial cache
    # ================================================================
    print('Building seg VAE spatial cache...')
    pts, _ = mesh.sample(min(len(mesh.faces) * 10, 100000), return_index=True)
    bmi = pts.min(0); bma = pts.max(0)
    bsz = (bma - bmi).max(); bct = (bmi + bma) / 2
    pn = (pts - bct) / bsz * (RESOLUTION - 3) + RESOLUTION / 2
    cr = np.clip(pn.astype(np.int32), 0, RESOLUTION - 1)
    ch = cr[:, 0].astype(np.int64) * RESOLUTION * RESOLUTION + cr[:, 1].astype(np.int64) * RESOLUTION + cr[:, 2].astype(np.int64)
    uh = np.unique(ch)
    uc = np.stack([uh // (RESOLUTION * RESOLUTION), (uh % (RESOLUTION * RESOLUTION)) // RESOLUTION, uh % RESOLUTION], axis=1).astype(np.int32)
    if len(uc) > 30000:
        rng = np.random.default_rng(seed=len(uc))
        uc = uc[rng.choice(len(uc), 30000, replace=False)]

    N = len(uc)
    df = torch.zeros(N, NUM_CLASSES, device=device)
    df[:, 0] = 1.0
    bc = torch.cat([torch.zeros(N, 1, dtype=torch.int32, device=device),
                    torch.tensor(uc, dtype=torch.int32, device=device)], dim=-1)
    seg_z = seg_enc(sp.SparseTensor(df, bc), sample_posterior=False)
    slc = seg_z.coords[:, 1:].int()
    ntok = seg_z.feats.shape[0]
    print(f'  {N} voxels → {ntok} latent tokens')

    # ================================================================
    # Step 2: Encode mv_seg_latent from 6-view seg maps
    # ================================================================
    print('Encoding mv_seg_latent...')
    seg_data = np.load(args.seg_maps)
    sms = seg_data['seg_maps']  # (6, H, W) int32

    bg_count = (sms == 0).sum()
    fg_count = (sms > 0).sum()
    if bg_count == 0:
        print(f'  WARNING: seg_maps has no background (label=0). All {fg_count} pixels are foreground.')
        print(f'  This likely means background was not masked. Results may be degraded.')

    alc, alf = [], []
    for vi in range(6):
        sm = sms[vi]
        fl = sm.reshape(-1)
        mk = fl > 0
        if mk.sum() == 0:
            continue
        fg = np.clip(fl[mk], 0, NUM_CLASSES - 1)
        ft = np.zeros((mk.sum(), NUM_CLASSES), dtype=np.float32)
        ft[np.arange(mk.sum()), fg] = 1.0
        pos = box_surface_image_ids(sm.shape[0], sm.shape[1], vi)
        vp = pos[mk]
        ci = (vp * (RESOLUTION - 1)).round().int().clamp(0, RESOLUTION - 1)
        lin = ci[:, 0].long() * RESOLUTION * RESOLUTION + ci[:, 1].long() * RESOLUTION + ci[:, 2].long()
        ul, inv = torch.unique(lin, return_inverse=True)
        n = ul.shape[0]
        ftt = torch.tensor(ft)
        uf = torch.zeros(n, NUM_CLASSES)
        cnt = torch.zeros(n, 1)
        uf.scatter_add_(0, inv.unsqueeze(1).expand_as(ftt), ftt)
        cnt.scatter_add_(0, inv.unsqueeze(1), torch.ones(len(ftt), 1))
        uf = uf / cnt
        ucc = torch.stack([ul // (RESOLUTION * RESOLUTION),
                           (ul % (RESOLUTION * RESOLUTION)) // RESOLUTION,
                           ul % RESOLUTION], dim=1).int()
        bcc = torch.cat([torch.zeros(n, 1, dtype=torch.int32), ucc], dim=-1).to(device)
        zv = seg_enc(sp.SparseTensor(uf.to(device), bcc), sample_posterior=False)
        alc.append(zv.coords[:, 1:].cpu())
        alf.append(zv.feats.cpu())

    ac = torch.cat(alc)
    af = torch.cat(alf)
    r = ac.max().item() + 1
    la = ac[:, 0].long() * r * r + ac[:, 1].long() * r + ac[:, 2].long()
    ul3, inv3 = torch.unique(la, return_inverse=True)
    nm = ul3.shape[0]
    mf = torch.zeros(nm, af.shape[1])
    mcc = torch.zeros(nm, 1)
    mf.scatter_add_(0, inv3.unsqueeze(1).expand_as(af), af)
    mcc.scatter_add_(0, inv3.unsqueeze(1), torch.ones(len(af), 1))
    mf = mf / mcc
    mvc = torch.stack([ul3 // (r * r), (ul3 % (r * r)) // r, ul3 % r], dim=1).float().to(device)
    mf = mf.to(device)
    if mv_norm:
        mf = (mf - mv_mean) / mv_std
    print(f'  {nm} mv tokens')

    # ================================================================
    # Step 3: Shape encoder
    # ================================================================
    print('Encoding shape...')
    import o_voxel
    vt = torch.from_numpy(mesh_norm.vertices).float()
    ft2 = torch.from_numpy(mesh_norm.faces).long()
    vi2, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        vt.cpu(), ft2.cpu(), grid_size=RESOLUTION,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2)
    vst = SparseTensor(
        feats=(dv * RESOLUTION - vi2).to(device),
        coords=torch.cat([torch.zeros_like(vi2[:, 0:1]), vi2], dim=-1).to(device))
    shape_slat = shape_enc(vst, vst.replace(inter.to(device)))

    sas = lookup_feats_at_coords(slc, shape_slat.coords[:, 1:].int(), shape_slat.feats.float())
    if shape_norm:
        sas = (sas - shape_mean) / shape_std
    concat_cond = seg_z.replace(sas)
    concat_cond._shape = torch.Size([1, 32])
    print(f'  {shape_slat.feats.shape[0]} shape tokens')

    # ================================================================
    # Step 4: Cross-attention condition (zeros — DINOv3 removed)
    # ================================================================
    print('Setting zero cross-attention condition (DINOv3 removed)...')
    cond_features = torch.zeros(1, 1029, 1024, device=device)

    # ================================================================
    # Step 5: DIT sampling
    # ================================================================
    print(f'Sampling ({args.steps} steps)...')
    noise = seg_z.replace(torch.randn(ntok, 32, device=device))
    noise._shape = torch.Size([1, 32])
    sampler = FlowEulerSampler(sigma_min=sigma_min)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        result = sampler.sample(
            dit, noise=noise, cond=cond_features, concat_cond=concat_cond,
            mv_cond=[mf], mv_coords=[mvc],
            steps=args.steps, verbose=True)
    seg_slat = result.samples
    if seg_norm:
        seg_slat = seg_slat.replace(seg_slat.feats.float() * seg_std + seg_mean)

    # ================================================================
    # Step 6: Decode
    # ================================================================
    print('Decoding...')
    dd = next(seg_dec.parameters()).dtype
    zd = seg_z.replace(seg_slat.feats.to(dd))
    zd._shape = torch.Size([1, 32])
    logits = seg_dec(zd)
    pred_labels = logits.feats.float().argmax(dim=-1)
    pred_coords = logits.coords[:, 1:]

    # ================================================================
    # Step 7: Bake on mesh + export
    # ================================================================
    print('Baking labels on mesh...')
    palette = generate_class_colors(NUM_CLASSES)
    face_centers = mesh.triangles_center
    bm = mesh.vertices.min(0); bx = mesh.vertices.max(0)
    bs = (bx - bm).max(); bc2 = (bm + bx) / 2
    face_voxel = ((face_centers - bc2) / bs * (RESOLUTION - 3) + RESOLUTION / 2).astype(np.float32)

    cn = pred_coords.cpu().numpy().astype(np.float32)
    ln = pred_labels.cpu().numpy()
    tree = cKDTree(cn)
    _, nn = tree.query(face_voxel)
    face_labels = ln[nn]

    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    face_colors[:, :3] = (palette[face_labels % NUM_CLASSES] * 255).astype(np.uint8)
    face_colors[:, 3] = 255

    verts_split = mesh.vertices[mesh.faces].reshape(-1, 3)
    faces_split = np.arange(len(verts_split)).reshape(-1, 3)
    vert_colors = np.repeat(face_colors, 3, axis=0)
    colored = trimesh.Trimesh(vertices=verts_split, faces=faces_split,
                              vertex_colors=vert_colors, process=False)
    colored.export(args.output)

    n_parts = len(np.unique(face_labels))
    print(f'\nDone! Saved: {args.output}')
    print(f'  {n_parts} unique labels, distribution: {np.bincount(face_labels, minlength=NUM_CLASSES).tolist()}')


if __name__ == '__main__':
    main()
