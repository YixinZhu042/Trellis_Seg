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

from scipy.spatial import cKDTree
from utils.seg_projection import (
    collect_evidence, consistency_metrics, refine_projection,
    validate_seg_maps,
)

# Default paths
CONFIG_PATH = os.path.join(PROJECT_DIR, 'configs/slat_flow_seg_dit_1_3B_512_bf16.json')
SHAPE_ENC_PATH = '/path/to/trellis2_ckpts/shape_enc'
SEG_VAE_ENC_CKPT = './checkpoints/seg_vae/encoder_step0001000.pt'
SEG_VAE_DEC_CKPT = './checkpoints/seg_vae/decoder_step0001000.pt'
DIT_CKPT = './output/seg_dit/ckpts/denoiser_ema0.9999_step0050000.pt'
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


def lookup_feats_at_coords(target_coords, source_coords, source_feats, return_mask=False):
    N = target_coords.shape[0]
    C = source_feats.shape[-1]
    matched = torch.zeros(N, C, device=source_feats.device, dtype=source_feats.dtype)
    if source_coords.shape[0] == 0 or N == 0:
        mask = torch.zeros(N, device=source_feats.device, dtype=torch.bool)
        return (matched, mask) if return_mask else matched
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
    return (matched, hit) if return_mask else matched


def build_face_adjacency(num_faces, face_adjacency):
    adjacency = [[] for _ in range(num_faces)]
    for left, right in face_adjacency:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return adjacency


def remove_small_face_components(face_labels, face_adjacency, min_faces=100, iterations=2):
    """Merge isolated same-label face components into their dominant neighbor."""
    if min_faces <= 0 or len(face_adjacency) == 0:
        return face_labels

    adjacency = build_face_adjacency(len(face_labels), face_adjacency)
    labels = face_labels.copy()
    for _ in range(iterations):
        visited = np.zeros(len(labels), dtype=bool)
        updated = labels.copy()
        changed = 0
        for start in range(len(labels)):
            if visited[start]:
                continue
            label = labels[start]
            component = []
            boundary_labels = []
            stack = [start]
            visited[start] = True
            while stack:
                face = stack.pop()
                component.append(face)
                for neighbor in adjacency[face]:
                    if labels[neighbor] == label:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                    else:
                        boundary_labels.append(labels[neighbor])

            if len(component) < min_faces and boundary_labels:
                values, counts = np.unique(boundary_labels, return_counts=True)
                replacement = values[counts.argmax()]
                updated[component] = replacement
                changed += len(component)
        labels = updated
        if changed == 0:
            break

    return labels


def smooth_face_labels(face_labels, face_adjacency, iterations=5):
    """Majority-vote smoothing on mesh faces to shrink ragged boundaries."""
    if iterations <= 0 or len(face_adjacency) == 0:
        return face_labels

    adjacency = build_face_adjacency(len(face_labels), face_adjacency)
    labels = face_labels.copy()
    for _ in range(iterations):
        updated = labels.copy()
        changed = 0
        for face, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            candidates = np.array([labels[face], *[labels[n] for n in neighbors]])
            values, counts = np.unique(candidates, return_counts=True)
            replacement = values[counts.argmax()]
            if replacement != labels[face] and counts.max() >= 3:
                updated[face] = replacement
                changed += 1
        labels = updated
        if changed == 0:
            break
    return labels


def postprocess_face_labels(face_labels, face_adjacency, *, seg_prior_mode='off',
                           postprocess_after_prior=False, min_component_faces=0,
                           smooth_iterations=2):
    """Apply cleanup after optional prior fusion; preserve legacy prior-only behavior."""
    if min_component_faces < 0 or smooth_iterations < 0:
        raise ValueError('Postprocessing counts must be non-negative')
    labels = face_labels.copy()
    if seg_prior_mode != 'off' and not postprocess_after_prior:
        return labels
    if min_component_faces > 0:
        labels = remove_small_face_components(labels, face_adjacency, min_component_faces)
    if smooth_iterations > 0:
        labels = smooth_face_labels(labels, face_adjacency, smooth_iterations)
    return labels


def project_seg_maps_to_faces(mesh, seg_maps, device='cuda'):
    """Rasterize the mesh from all six cameras and vote partition IDs onto visible faces."""
    evidence = collect_evidence(mesh, seg_maps, device=device)
    return evidence.votes.argmax(1).astype(np.int32), evidence.visible


def erode_face_partition_interiors(prior_labels, prior_valid, face_adjacency, rings=3):
    """Remove several mesh-adjacency rings around projected partition boundaries."""
    interior = prior_valid.copy()
    if rings <= 0 or len(face_adjacency) == 0:
        return interior

    left = face_adjacency[:, 0]
    right = face_adjacency[:, 1]
    for _ in range(rings):
        keep = interior.copy()
        different = prior_labels[left] != prior_labels[right]
        remove_left = interior[left] & (~interior[right] | different)
        remove_right = interior[right] & (~interior[left] | different)
        keep[left[remove_left]] = False
        keep[right[remove_right]] = False
        interior = keep
    return interior


def remap_prior_parts_to_pred_labels(prior_labels, prior_valid, pred_labels, apply_mask=None):
    """Map arbitrary 2D partitions to model IDs, optionally modifying interiors only."""
    if apply_mask is None:
        apply_mask = prior_valid
    remapped = pred_labels.copy()
    for part_id in np.unique(prior_labels[apply_mask]):
        mask = apply_mask & (prior_labels == part_id)
        if not mask.any():
            continue
        values, counts = np.unique(pred_labels[mask], return_counts=True)
        remapped[mask] = values[counts.argmax()]
    return remapped


def relabel_predicted_components_with_prior(
    pred_labels, prior_labels, prior_valid, face_adjacency, min_prior_agreement=0.6
):
    """Relabel whole model-predicted islands using 2D evidence without copying 2D edges."""
    if len(face_adjacency) == 0:
        return pred_labels, 0, 0

    part_to_model = {}
    for part_id in np.unique(prior_labels[prior_valid]):
        mask = prior_valid & (prior_labels == part_id)
        values, counts = np.unique(pred_labels[mask], return_counts=True)
        part_to_model[part_id] = values[counts.argmax()]

    adjacency = build_face_adjacency(len(pred_labels), face_adjacency)
    labels = pred_labels.copy()
    visited = np.zeros(len(labels), dtype=bool)
    changed_components = 0
    changed_faces = 0
    for start in range(len(labels)):
        if visited[start]:
            continue
        source_label = labels[start]
        component = []
        stack = [start]
        visited[start] = True
        while stack:
            face = stack.pop()
            component.append(face)
            for neighbor in adjacency[face]:
                if not visited[neighbor] and labels[neighbor] == source_label:
                    visited[neighbor] = True
                    stack.append(neighbor)

        component = np.asarray(component, dtype=np.int64)
        visible = component[prior_valid[component]]
        if visible.size == 0:
            continue
        parts, counts = np.unique(prior_labels[visible], return_counts=True)
        best_index = counts.argmax()
        agreement = counts[best_index] / visible.size
        target_label = part_to_model[parts[best_index]]
        if target_label != source_label and agreement >= min_prior_agreement:
            labels[component] = target_label
            changed_components += 1
            changed_faces += len(component)

    return labels, changed_components, changed_faces


def main():
    parser = argparse.ArgumentParser(description='In-the-wild 3D segmentation inference')
    parser.add_argument('--mesh', type=str, required=True, help='Input mesh (.glb/.obj)')
    parser.add_argument('--seg_maps', type=str, required=True,
                        help='seg_maps.npz with seg_maps (6,H,W) int32, 0=background')
    parser.add_argument('--output', type=str, required=True, help='Output .glb path')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seg_prior_mode', type=str, default='consistency',
                        choices=['off', 'components', 'interior', 'override', 'raw_ids', 'consistency'],
                        help='consistency preserves input part IDs and maximizes visible 2D agreement')
    parser.add_argument('--prior_component_agreement', type=float, default=0.5,
                        help='Minimum projected-part agreement required to relabel a model island')
    parser.add_argument('--prior_erode_rings', type=int, default=1,
                        help='Mesh adjacency rings excluded around 2D partition boundaries in interior mode')
    parser.add_argument('--projection_subdivide', type=int, default=None,
                        help='Boundary subdivision rounds (default: 5 in consistency mode; 0 preserves original face count)')
    parser.add_argument('--projection_cleanup', choices=['graph', 'ties'], default='graph',
                        help='graph reduces fragmentation within input-fidelity budgets; ties maximizes pixel accuracy')
    parser.add_argument('--projection_accuracy_budget', type=float, default=.003)
    parser.add_argument('--projection_miou_budget', type=float, default=.01)
    parser.add_argument('--part_iou_refine', choices=['on', 'off'], default='on',
                        help='Improve small-part input IoU within existing accuracy and boundary budgets')
    parser.add_argument('--hidden_prior_confidence', type=float, default=.8,
                        help='Minimum input-view agreement to retain a model part on unseen faces')
    parser.add_argument('--hidden_smooth', type=float, default=.25,
                        help='Boundary penalty for unseen faces only; 0 disables hidden cleanup')
    parser.add_argument('--bake_knn', type=int, default=8,
                        help='Average decoder logits from K nearest voxels when baking labels onto mesh faces')
    parser.add_argument('--smooth_iterations', type=int, default=2,
                        help='Mesh face majority-vote smoothing iterations after baking')
    parser.add_argument('--min_component_faces', type=int, default=0,
                        help='Merge predicted face-label components smaller than this size; set 0 to disable')
    parser.add_argument('--postprocess_after_prior', action='store_true',
                        help='Apply component cleanup and smoothing after 2D prior fusion')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--config', type=str, default=CONFIG_PATH)
    parser.add_argument('--dit_ckpt', type=str, default=DIT_CKPT)
    parser.add_argument('--seg_encoder', default=SEG_VAE_ENC_CKPT)
    parser.add_argument('--seg_decoder', default=SEG_VAE_DEC_CKPT)
    args = parser.parse_args()
    if args.projection_subdivide is None:
        args.projection_subdivide = 5 if args.seg_prior_mode == 'consistency' else 0
    if not 0 <= args.projection_subdivide <= 5:
        parser.error('--projection_subdivide must be between 0 and 5')
    if args.projection_subdivide and args.seg_prior_mode != 'consistency':
        parser.error('--projection_subdivide requires --seg_prior_mode consistency')
    if not 0 <= args.projection_accuracy_budget <= 1 or not 0 <= args.projection_miou_budget <= 1:
        parser.error('Projection fidelity budgets must be between 0 and 1')
    if not 0 <= args.hidden_prior_confidence <= 1:
        parser.error('--hidden_prior_confidence must be between 0 and 1')
    if not np.isfinite(args.hidden_smooth) or args.hidden_smooth < 0:
        parser.error('--hidden_smooth must be finite and nonnegative')

    # Validate inputs before loading the large model checkpoints.
    with np.load(args.seg_maps) as seg_data:
        sms = validate_seg_maps(seg_data['seg_maps'], max_parts=NUM_CLASSES - 1)
    from trellis2.modules.sparse.basic import SparseTensor
    import trellis2.models as trellis2_models
    import trellis2.modules.sparse as sp
    from models import get_model
    from datasets.seg_dataset import box_surface_image_ids, external_seg_ids_to_model_ids
    from trellis2.pipelines.samplers import FlowEulerSampler
    torch.set_grad_enabled(False)
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
    seg_enc = trellis2_models.SparseUnetVaeEncoder(**svc['models']['encoder']['args'])
    seg_enc.load_state_dict(torch.load(args.seg_encoder, map_location='cpu', weights_only=True))
    seg_enc.convert_to_fp16()
    seg_enc = seg_enc.eval().to(device)
    seg_dec = trellis2_models.SparseUnetVaeDecoder(**svc['models']['decoder']['args'])
    seg_dec.load_state_dict(torch.load(args.seg_decoder, map_location='cpu', weights_only=True))
    seg_dec.convert_to_fp16()
    seg_dec = seg_dec.eval().to(device)

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
    pts, _ = trimesh.sample.sample_surface(
        mesh, min(len(mesh.faces) * 10, 100000), seed=args.seed)
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
        # External/API maps already use the model's part IDs: background=0,
        # parts=1..19. Do not apply the +1 render-file offset used in training.
        fg = external_seg_ids_to_model_ids(fl[mk], NUM_CLASSES)
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
    # No dedup across views — match training (direct concat)
    mvc = ac.float().to(device)
    mf = af.to(device)
    if mv_norm:
        mf = (mf - mv_mean) / mv_std
    nm = mf.shape[0]
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
    pred_logits = logits.feats.float()
    pred_coords = logits.coords[:, 1:]

    # ================================================================
    # Step 7: Bake on mesh + export
    # ================================================================
    print('Baking labels on mesh...')
    palette = generate_class_colors(NUM_CLASSES)
    face_centers = mesh.triangles_center
    # Decode coordinates belong to the sampled-point frame from Step 1.
    # Recomputing this from the full mesh shifts nearest-voxel queries.
    face_voxel = ((face_centers - bct) / bsz * (RESOLUTION - 3) + RESOLUTION / 2).astype(np.float32)

    cn = pred_coords.cpu().numpy().astype(np.float32)
    logits_np = pred_logits.cpu().numpy()
    tree = cKDTree(cn)
    k = max(1, min(args.bake_knn, len(cn)))
    distances, nn = tree.query(face_voxel, k=k)
    if k == 1:
        face_labels = logits_np[nn].argmax(axis=-1)
    else:
        weights = 1.0 / np.maximum(distances, 1e-3)
        face_logits = (logits_np[nn] * weights[..., None]).sum(axis=1)
        face_logits /= weights.sum(axis=1, keepdims=True)
        face_labels = face_logits.argmax(axis=-1)
    print(f'  Baked labels using inverse-distance weighted {k}-NN logits')
    np.savez_compressed(os.path.splitext(args.output)[0] + '_model_labels.npz', face_labels=face_labels)

    consistency_report = None
    evidence = None
    if args.seg_prior_mode == 'consistency':
        mesh, face_labels, parent_faces, evidence, _, consistency_report = refine_projection(
            mesh, face_labels, sms, device=device, subdivide=args.projection_subdivide,
            graph_cleanup=args.projection_cleanup == 'graph',
            accuracy_budget=args.projection_accuracy_budget, miou_budget=args.projection_miou_budget,
            hidden_confidence=args.hidden_prior_confidence, hidden_smooth=args.hidden_smooth,
            part_iou_refine=args.part_iou_refine == 'on')
        observed_labels = face_labels[evidence.visible].copy()
        print(f'  Fused input part IDs on {evidence.visible.sum()}/{len(face_labels)} visible faces')
    elif args.seg_prior_mode != 'off':
        prior_labels, prior_valid = project_seg_maps_to_faces(mesh, sms, device)
        if args.seg_prior_mode == 'raw_ids':
            face_labels[prior_valid] = prior_labels[prior_valid]
            print(f'  Applied raw 2D part IDs to {prior_valid.sum()}/{len(face_labels)} faces')
        elif args.seg_prior_mode == 'components':
            face_labels, changed_components, changed_faces = relabel_predicted_components_with_prior(
                face_labels, prior_labels, prior_valid, mesh.face_adjacency,
                args.prior_component_agreement,
            )
            print(f'  Relabeled {changed_components} model islands ({changed_faces} faces) using 2D evidence; '
                  'kept all model-predicted boundaries')
        elif args.seg_prior_mode == 'interior':
            interior = erode_face_partition_interiors(
                prior_labels, prior_valid, mesh.face_adjacency, args.prior_erode_rings
            )
            face_labels = remap_prior_parts_to_pred_labels(
                prior_labels, prior_valid, face_labels, apply_mask=interior
            )
            print(f'  Corrected {interior.sum()}/{len(face_labels)} interior faces using 2D partitions; '
                  f'preserved {prior_valid.sum() - interior.sum()} projected boundary faces')
        else:
            face_labels = remap_prior_parts_to_pred_labels(prior_labels, prior_valid, face_labels)
            print(f'  Applied visible-pixel 2D partitions to {prior_valid.sum()}/{len(face_labels)} faces; IDs remapped from model predictions')

    face_labels = postprocess_face_labels(
        face_labels, mesh.face_adjacency,
        seg_prior_mode=args.seg_prior_mode,
        postprocess_after_prior=args.postprocess_after_prior,
        min_component_faces=args.min_component_faces,
        smooth_iterations=args.smooth_iterations,
    )
    if evidence is not None:
        # Optional cleanup may fill unobserved regions but cannot erase 2D evidence.
        face_labels[evidence.visible] = observed_labels
        consistency_report['refined'] = consistency_metrics(face_labels, sms, evidence)
        np.savez_compressed(os.path.splitext(args.output)[0] + '_labels.npz',
                            face_labels=face_labels, parent_faces=parent_faces)
        score = consistency_report['refined']
        print(f"  2D agreement: {score['accuracy']:.4%}; mIoU: {score['miou']:.4%}; "
              f"fixed-mesh ceiling: {score['fixed_mesh_accuracy_ceiling']:.4%}")
    if args.seg_prior_mode == 'off' or args.postprocess_after_prior:
        print(f'  Postprocessing: min_component_faces={args.min_component_faces}, '
              f'smooth_iterations={args.smooth_iterations}')
    else:
        print('  Skipped component removal and smoothing to preserve the selected boundaries')

    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    face_colors[:, :3] = (palette[face_labels % NUM_CLASSES] * 255).astype(np.uint8)
    face_colors[:, 3] = 255

    verts_split = mesh.vertices[mesh.faces].reshape(-1, 3)
    faces_split = np.arange(len(verts_split)).reshape(-1, 3)
    vert_colors = np.repeat(face_colors, 3, axis=0)
    colored = trimesh.Trimesh(vertices=verts_split, faces=faces_split,
                              vertex_colors=vert_colors, process=False)
    colored.export(args.output)
    if consistency_report is not None:
        consistency_report['model'] = {
            'dit_ckpt': args.dit_ckpt, 'seg_encoder': args.seg_encoder,
            'seg_decoder': args.seg_decoder, 'seed': args.seed, 'steps': args.steps,
            'mesh': os.path.abspath(args.mesh), 'seg_maps': os.path.abspath(args.seg_maps),
        }
        # Account for export precision in the final consistency report.
        from scripts.evaluate_seg_quality import load_mesh, recover_face_labels_from_colors
        exported = load_mesh(args.output)
        exported_labels = recover_face_labels_from_colors(args.output, NUM_CLASSES, len(face_labels))
        if not np.array_equal(exported_labels, face_labels):
            raise ValueError('GLB label/color roundtrip failed')
        exported_evidence = collect_evidence(exported, sms, device=device)
        consistency_report['exported'] = consistency_metrics(exported_labels, sms, exported_evidence)
        with open(os.path.splitext(args.output)[0] + '_consistency.json', 'w') as f:
            json.dump(consistency_report, f, indent=2, allow_nan=False)

    n_parts = len(np.unique(face_labels))
    print(f'\nDone! Saved: {args.output}')
    print(f'  {n_parts} unique labels, distribution: {np.bincount(face_labels, minlength=NUM_CLASSES).tolist()}')


if __name__ == '__main__':
    main()
