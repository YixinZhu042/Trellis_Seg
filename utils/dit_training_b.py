"""Train-only label renaming and reproducible object oversampling for DiT B.

No latent channel permutations, source-cache mutation, or validation resampling.
The existing train-only normalization stays fixed when forking a trained DiT.
"""
import numpy as np

from vae_v2.augmentation import label_permutation, permute_sample

POLICY = 'online_synchronized_nonzero_v1'
BINS = ((1, 20), (21, 32), (33, 63))


def validate_options(train):
    aug = train.get('label_augmentation', {})
    if aug:
        if set(aug) != {'policy', 'probability', 'normalization'}:
            raise ValueError('Invalid label_augmentation keys')
        if aug['policy'] != POLICY or aug['normalization'] != 'preserve_base_train_statistics':
            raise ValueError('Unsupported augmentation/normalization policy')
        if not np.isfinite(aug['probability']) or not 0 <= aug['probability'] <= 1:
            raise ValueError('Invalid augmentation probability')
    sampling = train.get('object_sampling', {})
    if sampling:
        if set(sampling) != {'policy', 'bin_probabilities'} or sampling['policy'] != 'part_count_stratified_v1':
            raise ValueError('Invalid object sampling policy')
        p = np.asarray(sampling['bin_probabilities'], dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or (p <= 0).any() or not np.isclose(p.sum(), 1):
            raise ValueError('Expected three positive bin probabilities summing to one')
    fraction = train.get('surface_macro_fraction', 0.)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid surface macro fraction')


def object_order(rows, seed, epoch, sampling=None):
    """A fixed-size global draw epoch; DDP ranks stride this SAME sequence.

Within each bin, shuffled cycles avoid replacement until that bin is exhausted.
Counts use largest-remainder rounding. This changes the training objective;
there is deliberately no inverse object-probability correction.
"""
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Sampling accepts nonempty training rows only')
    if not sampling:
        return np.random.default_rng(seed + epoch).permutation(len(rows))
    validate_options({'object_sampling': sampling})
    if any(not isinstance(r['parts'], (int, np.integer)) or not 1 <= r['parts'] <= 63 for r in rows):
        raise ValueError('Invalid annotated part count')
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0x4F424A]))
    groups = [np.array([i for i, r in enumerate(rows) if lo <= r['parts'] <= hi]) for lo, hi in BINS]
    if any(not len(g) for g in groups):
        raise ValueError('Every configured sampling bin must contain training objects')
    exact = np.asarray(sampling['bin_probabilities']) * len(rows)
    counts = np.floor(exact).astype(int)
    counts[np.argsort(-(exact-counts), kind='stable')[:len(rows)-counts.sum()]] += 1
    draws = []
    for group, count in zip(groups, counts):
        if count:
            draws.append(np.concatenate([rng.permutation(group) for _ in range((count+len(group)-1)//len(group))])[:count])
    return rng.permutation(np.concatenate(draws))


def training_permutation(seed, sample_index, augmentation, split='train'):
    if split != 'train':
        raise ValueError('Label augmentation is training-only')
    if not augmentation:
        return None
    validate_options({'label_augmentation': augmentation})
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_index, 0x444954]))
    if rng.random() >= augmentation['probability']:
        return None
    return label_permutation(64, seed, sample_index)


def selected_row_index(rows, order, offset, sample_index, smoke_ids=False):
    """Explicit smoke visits chosen geometries without requiring all three bins."""
    return sample_index % len(rows) if smoke_ids else int(order[offset])


def synchronized_targets(sample, mapping):
    """One bijection for soft voxels, hard supervision AND all six face rasters."""
    from utils.seg_projection import render_labels
    altered = permute_sample(sample, mapping)
    maps = render_labels(altered['face_labels'] + 1, sample['face_raster'])
    if maps.shape[0] != 6:
        raise ValueError('Expected six condition views')
    return altered, maps


def reencode_labels(vae, sample, data, mapping, device):
    import torch
    from vae_v2.data import voxel_tensors
    from preprocess.step2_encode_mv_seg_latent import encode_one_view
    from utils.dit_vae_v2 import validate_latents

    if vae.training or any(p.requires_grad for p in vae.parameters()):
        raise ValueError('Online augmentation requires a frozen eval-mode VAE')
    altered, maps = synchronized_targets(sample, mapping)
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        z, _, _ = vae.encode(voxel_tensors(altered, device), sample_posterior=False)
        coords = z.coords[:, 1:].cpu().numpy()
        if not np.array_equal(coords, data['latent_coords']):
            raise ValueError('Permutation changed target geometry/order')
        result = dict(data, latent_feats=z.feats.float().cpu().numpy())
        del z
        all_coords, all_feats, offsets = [], [], [0]
        for view in range(6):
            coords, feats = encode_one_view(maps[view], view, vae.encoder, 64, device)
            if coords is not None:
                all_coords.append(coords.numpy().astype(np.int32))
                all_feats.append(feats.float().numpy())
                offsets.append(offsets[-1] + len(coords))
            else:
                offsets.append(offsets[-1])
        if not all_feats:
            raise ValueError('No visible condition tokens')
        if not np.array_equal(np.asarray(offsets), data['view_offsets']) or not np.array_equal(np.concatenate(all_coords), data['mv_coords']):
            raise ValueError('Permutation changed six-view geometry/order')
        result['mv_feats'] = np.concatenate(all_feats)
    validate_latents(result, len(data['latent_coords']))
    return result, altered


def checked_fork_state(checkpoint, cfg, manifest, vae_sha256, base_digest):
    """Never mix an old-space model or step-5000 optimizer into the best-step fork."""
    import torch
    from vae_v2.config import digest
    if 'model' not in checkpoint:  # Backwards-compatible legacy flat state dict.
        if cfg.get('initialization'):
            raise ValueError('Audited B initialization requires a metadata checkpoint')
        return checkpoint, {'mode': 'legacy_weights_only'}
    policy = cfg.get('initialization')
    if not policy or policy.get('mode') != 'fork_weights_only':
        raise ValueError('Metadata checkpoint requires explicit fork_weights_only')
    meta = checkpoint['metadata']
    if (checkpoint['step'] != policy['expected_step'] or checkpoint['kind'] != policy['expected_kind']
            or meta.get('smoke_only') or meta['vae_sha256'] != vae_sha256
            or meta['cache_digest'] != digest(manifest) or meta['base_digest'] != base_digest):
        raise ValueError('Fork source step/kind/data/VAE/model differs')
    if not all(torch.is_tensor(v) for v in checkpoint['model'].values()):
        raise ValueError('Invalid model state')
    return checkpoint['model'], {'mode': 'fork_weights_only', 'source_step': checkpoint['step'],
                                 'source_kind': checkpoint['kind'], 'optimizer_reset': True, 'ema_reset': True}
