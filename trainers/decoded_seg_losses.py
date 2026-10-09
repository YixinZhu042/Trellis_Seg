"""Differentiable task losses for DiT x0 predictions through a frozen decoder.

All labels use the original 0..19 model namespace. Zero is ignored for part
supervision (unannotated surface), while all 20 channels remain in softmax.
Reprojection sums the original visible pixel CE exactly using face hit counts.
Geometry/visibility are constants; gradients flow through logits, never argmax.
"""
import torch
import torch.nn.functional as F


def recover_clean_latent(noisy, velocity, t, sigma_min=1e-5):
    return (1 - sigma_min) * noisy - (sigma_min + (1 - sigma_min) * t) * velocity


def face_logits_from_voxels(logits, neighbors, weights, chunk_size=16384):
    chunks = []
    for start in range(0, len(neighbors), chunk_size):
        index = neighbors[start:start + chunk_size]
        weight = weights[start:start + chunk_size]
        chunks.append((logits[index] * weight[..., None]).sum(1))
    return torch.cat(chunks)


def balanced_voxel_ce(logits, target):
    valid = target > 0
    if not valid.any():
        return logits.sum() * 0
    counts = torch.bincount(target[valid], minlength=logits.shape[1]).float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = (counts[present].mean() / counts[present]).sqrt().clamp(.25, 5.)
    return F.cross_entropy(logits[valid].float(), target[valid], weight=weights)


def surface_boundary_loss(face_logits, face_labels, adjacency):
    """Balanced CE for same/different part relations on true surface neighbors.

    Same-label edges should have large probability overlap; true boundary edges
    should have small overlap. This does not encourage smoothing actual GT edges.
    """
    if adjacency.numel() == 0:
        return face_logits.sum() * 0
    left, right = adjacency.unbind(1)
    valid = (face_labels[left] > 0) & (face_labels[right] > 0)
    if not valid.any():
        return face_logits.sum() * 0
    left, right = left[valid], right[valid]
    probability = face_logits.float().softmax(1)
    agreement = (probability[left] * probability[right]).sum(1).clamp(1e-6, 1 - 1e-6)
    boundary = face_labels[left] != face_labels[right]
    terms = []
    if boundary.any():
        terms.append(-torch.log1p(-agreement[boundary]).mean())
    if (~boundary).any():
        terms.append(-agreement[~boundary].log().mean())
    return torch.stack(terms).mean()


def projected_pixel_ce(face_logits, face_labels, face_pixels):
    valid = (face_labels > 0) & (face_pixels > 0)
    if not valid.any():
        return face_logits.sum() * 0
    losses = F.cross_entropy(face_logits[valid].float(), face_labels[valid], reduction='none')
    counts = face_pixels[valid].float()
    return (losses * counts).sum() / counts.sum()


def surface_segmentation_ce(face_logits, face_labels, face_area):
    valid = (face_labels > 0) & (face_area > 0)
    if not valid.any():
        return face_logits.sum() * 0
    losses = F.cross_entropy(face_logits[valid].float(), face_labels[valid], reduction='none')
    return (losses * face_area[valid]).sum() / face_area[valid].sum()


def decoded_task_losses(voxel_logits, sample):
    face_logits = face_logits_from_voxels(voxel_logits, sample['face_knn'], sample['face_knn_weights'])
    return {
        'segmentation': .5 * balanced_voxel_ce(voxel_logits, sample['voxel_labels']) +
                        .5 * surface_segmentation_ce(face_logits, sample['face_labels'], sample['face_area']),
        'boundary': surface_boundary_loss(face_logits, sample['face_labels'], sample['adjacency']),
        'reprojection': projected_pixel_ce(face_logits, sample['face_labels'], sample['face_pixels']),
    }, face_logits
