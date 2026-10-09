"""Task losses: ignore unannotated channel 0; preserve original part IDs."""
import torch
import torch.nn.functional as F


def kl_weight(step, train):
    progress = max(0, step - train['deterministic_steps'])
    return train['kl_weight'] * min(1., progress / max(1, train['kl_warmup_steps']))


def masked_soft_target(target):
    target = target.clone()
    target[:, 0] = 0
    mass = target.sum(-1)
    return target / mass[:, None].clamp_min(1e-12), mass > 0


def macro_dice(probability, truth, importance=None):
    if importance is None:
        importance = torch.ones(len(truth), device=truth.device)
    weight = importance / importance.sum().clamp_min(1e-12)
    mass = (truth * weight[:, None]).sum(0)
    present = mass > 0
    present[0] = False
    if not present.any():
        return probability.sum() * 0
    intersection = (probability * truth * weight[:, None]).sum(0)
    denominator = ((probability + truth) * weight[:, None]).sum(0)
    return (1 - (2 * intersection + 1e-8) / (denominator + 1e-8))[present].mean()


def balanced_soft_ce(logits, target):
    truth, valid = masked_soft_target(target)
    if not valid.any():
        return logits.sum() * 0
    truth, scores = truth[valid], logits[valid].float()
    counts = truth.sum(0)
    present = counts > 0
    weight = torch.zeros_like(counts)
    weight[present] = (counts[present].mean() / counts[present]).sqrt().clamp(.25, 5)
    weighted = truth * weight
    return -(weighted * scores.log_softmax(-1)).sum() / weighted.sum().clamp_min(1e-12)


def weighted_ce(logits, labels, importance):
    valid = (labels > 0) & (importance > 0)
    if not valid.any():
        return logits.sum() * 0
    loss = F.cross_entropy(logits[valid].float(), labels[valid], reduction='none')
    return (loss * importance[valid]).sum() / importance[valid].sum()


def relation_losses(face_logprob, pair_index, same, geometry_weight):
    zero = face_logprob.sum() * 0
    if not len(pair_index):
        return zero, zero
    prob = face_logprob.float().exp()
    left, right = prob[pair_index[:, 0]], prob[pair_index[:, 1]]
    middle = (left + right) / 2
    js = .5 * ((left * (left.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(1) +
               (right * (right.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(1))
    w = geometry_weight * same
    consistency = (js * w).sum() / w.sum().clamp_min(1e-12)
    overlap = (left * right).sum(1).clamp(1e-6, 1-1e-6)
    # Never impose smoothness on different-GT pairs, including close disconnected shells.
    boundary = -torch.log1p(-overlap[~same]).mean() if (~same).any() else zero
    return consistency, boundary


def macro_surface_ce(logits, labels, importance):
    """Monte Carlo equal-part CE; importance = p_equal_part / p_draw.

    Unlike area/p_draw, this retains the desired small-part contribution.
    Do NOT normalize by a random batch sum or invert object oversampling.
    """
    if len(labels) == 0:
        return logits.sum() * 0
    if (labels <= 0).any() or not torch.isfinite(importance).all() or (importance < 0).any():
        raise ValueError('Macro CE expects annotated, finite importance samples')
    return (F.cross_entropy(logits.float(), labels, reduction='none') * importance).mean()


def reconstruction_losses(output, sample, query, weights, beta, balanced=True):
    logits = output['voxel_logits'].float()
    num_classes = logits.shape[-1]
    target = sample['voxel_target'].float()
    if target.shape != logits.shape:
        raise ValueError('Prediction and target class counts/voxel shapes differ')
    truth, valid = masked_soft_target(target)
    voxel_ce = balanced_soft_ce(logits, target) if balanced else (
        -(truth[valid] * logits[valid].log_softmax(-1)).sum(-1).mean() if valid.any() else logits.sum()*0)
    terms = {'voxel_ce': voxel_ce,
             'voxel_dice': macro_dice(logits[valid].softmax(-1), truth[valid])}
    zero = logits.sum() * 0
    for key in ('point_ce', 'surface_ce', 'surface_dice', 'reprojection', 'consistency', 'boundary'):
        terms[key] = zero
    if query is not None:
        n = query['supervised_count']
        labels = query['query_labels'][:n]
        importance = query['surface_importance']
        face = output['face_logprob'][:n]
        points = output['point_logits'][:n]
        part_mass = torch.bincount(labels, weights=importance, minlength=num_classes)
        balance = (part_mass[part_mass>0].mean() / part_mass.clamp_min(1e-12)).sqrt().clamp(.25, 5)
        balanced_importance = importance * balance[labels]
        terms['point_ce'] = weighted_ce(points.reshape(-1, num_classes), labels[:, None].expand(-1, points.shape[1]).reshape(-1),
                                        balanced_importance[:, None].expand(-1, points.shape[1]).reshape(-1))
        terms['surface_ce'] = weighted_ce(face, labels, .5 * importance + .5 * balanced_importance)
        terms['surface_dice'] = macro_dice(face.exp(), F.one_hot(labels, num_classes).float(), importance)
        terms['reprojection'] = weighted_ce(face, labels, query['pixel_importance'])
        terms['consistency'], terms['boundary'] = relation_losses(output['face_logprob'], query['pair_index'],
                                                                  query['pair_same'], query['pair_geometry_weight'])
    terms['kl'] = .5 * (output['mean'].float().square() + output['logvar'].float().exp() - 1 - output['logvar'].float()).mean()
    loss = sum(weights[k] * terms[k] for k in weights) + beta * terms['kl']
    terms['loss'] = loss
    return loss, terms
