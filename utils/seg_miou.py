"""Part-balanced input fidelity on a fixed mesh, using original input IDs.

Unlike pixel-majority voting, mIoU gives a tiny part the same importance as a
large part. Candidate updates are checked against exact IoU, an accuracy floor,
and a surface boundary-length cap. Only conflicting observed faces can change.
"""
import numpy as np
from scipy.sparse import coo_matrix

from .seg_graph import surface_graph, fragmentation


def iou_statistics(labels, seg_maps, evidence):
    votes = evidence.votes
    total = votes.sum(1)
    ground_truth = np.bincount(seg_maps.ravel(), minlength=votes.shape[1]).astype(float)
    ground_truth[0] = 0
    predicted = np.bincount(labels, weights=total, minlength=votes.shape[1])
    true_positive = np.bincount(labels, weights=votes[np.arange(len(labels)), labels],
                                minlength=votes.shape[1])
    return ground_truth, predicted, true_positive


def per_part_iou_ceiling(seg_maps, evidence):
    """Independent binary optimum for each part on this raster/triangulation.

    This is an upper bound, not a jointly achievable multiclass solution: two
    parts may require the same face. Missing mesh pixels remain in the GT count.
    The optimal binary subset is a prefix sorted by TP/FP contribution.
    """
    total = evidence.votes.sum(1)
    parts, counts = np.unique(seg_maps[seg_maps > 0], return_counts=True)
    result = {}
    for part, count in zip(parts, counts):
        true = evidence.votes[:, part]
        selected = np.flatnonzero(true > 0)
        false = total[selected] - true[selected]
        priority = true[selected] / np.maximum(false, 1e-30)
        order = np.argsort(-priority, kind='stable')
        ratios = np.cumsum(true[selected][order]) / (count + np.cumsum(false[order]))
        result[int(part)] = float(ratios.max()) if len(ratios) else 0.
    return result


def _coordinate_ascent(labels, votes, gt, predicted, tp, offsets, neighbors, edge_weights,
                       order, pixel_floor, boundary_limit, iterations):
    labels = labels.copy()
    predicted, tp = predicted.copy(), tp.copy()
    total = votes.sum(1)
    correct = 0.
    boundary = 0.
    for face in range(len(labels)):
        correct += votes[face, labels[face]]
        for edge in range(offsets[face], offsets[face + 1]):
            if labels[face] != labels[neighbors[edge]]:
                boundary += edge_weights[edge] * .5
    accepted = 0
    for iteration in range(iterations):
        changed = 0
        for j in range(len(order)):
            face = order[j] if iteration % 2 == 0 else order[len(order) - j - 1]
            old = labels[face]
            if total[face] == 0 or votes[face].max() == total[face]:
                continue
            old_union = gt[old] + predicted[old] - tp[old]
            removed_tp = tp[old] - votes[face, old]
            removed_union = old_union - total[face] + votes[face, old]
            old_delta = removed_tp / max(removed_union, 1.) - tp[old] / max(old_union, 1.)
            best, best_gain, best_edge = old, 1e-12, 0.
            for part in range(1, votes.shape[1]):
                if part == old or votes[face, part] == 0:
                    continue
                new_correct = correct + votes[face, part] - votes[face, old]
                if new_correct < pixel_floor - 1e-8:
                    continue
                union = gt[part] + predicted[part] - tp[part]
                added_tp = tp[part] + votes[face, part]
                added_union = union + total[face] - votes[face, part]
                gain = old_delta + added_tp / max(added_union, 1.) - tp[part] / max(union, 1.)
                if gain <= best_gain:
                    continue
                edge_delta = 0.
                for edge in range(offsets[face], offsets[face + 1]):
                    neighbor = labels[neighbors[edge]]
                    edge_delta += edge_weights[edge] * (int(part != neighbor) - int(old != neighbor))
                if boundary + edge_delta > boundary_limit + 1e-8:
                    continue
                best, best_gain, best_edge = part, gain, edge_delta
            if best != old:
                predicted[old] -= total[face]
                predicted[best] += total[face]
                tp[old] -= votes[face, old]
                tp[best] += votes[face, best]
                correct += votes[face, best] - votes[face, old]
                boundary += best_edge
                labels[face] = best
                changed += 1
        accepted += changed
        if changed == 0:
            break
    return labels, accepted


_optimizer = None


def refine_part_iou(mesh, labels, seg_maps, evidence, accuracy_budget=.001,
                    boundary_budget=.01, iterations=5):
    """Exact-IoU improving moves with <=0.1pp accuracy loss by default.

    Boundary budget is fractional growth from the supplied segmentation. The
    result is accepted only if each input part IoU loses at most 0.002 and no
    hidden or unanimous observed labels change. No 3D annotations are used.
    """
    from .seg_projection import consistency_metrics
    if not 0 <= accuracy_budget <= 1 or not 0 <= boundary_budget <= 1:
        raise ValueError('Budgets must be between 0 and 1')
    if iterations < 1:
        raise ValueError('iterations must be positive')
    gt, predicted, tp = iou_statistics(labels, seg_maps, evidence)
    adjacency, weights = surface_graph(mesh, max(evidence.face_ids.shape[1:]))
    left, right = adjacency.T
    graph = coo_matrix((np.r_[weights, weights], (np.r_[left, right], np.r_[right, left])),
                       shape=(len(labels), len(labels))).tocsr()
    before = consistency_metrics(labels, seg_maps, evidence)
    before_fragment = fragmentation(mesh, labels, adjacency, weights)
    pixels = int(gt.sum())
    # Rarer supported parts first, so a large region cannot consume the entire
    # accuracy budget before a small region gets considered.
    supported = np.where(evidence.votes > 0, gt[None, :], np.inf)
    supported[:, 0] = np.inf
    order = np.argsort(supported.min(1), kind='stable')
    global _optimizer
    if _optimizer is None:
        from numba import njit
        _optimizer = njit(cache=True)(_coordinate_ascent)
    candidate, moves = _optimizer(
        labels, evidence.votes, gt, predicted, tp, graph.indptr, graph.indices, graph.data,
        order, (before['accuracy'] - accuracy_budget) * pixels,
        before_fragment['boundary_length_pixels'] * (1 + boundary_budget), iterations)
    after = consistency_metrics(candidate, seg_maps, evidence)
    proposed = after
    # Do not trade a large regression on one part for mean-IoU improvement.
    accepted = (after['miou'] >= before['miou'] - 1e-12 and
                after['accuracy'] >= before['accuracy'] - accuracy_budget - 1e-12 and
                all(after['part_iou'][p] >= iou - .002 for p, iou in before['part_iou'].items()))
    if not accepted:
        candidate, after = labels.copy(), before
    return candidate, {'accuracy_budget': accuracy_budget, 'boundary_budget': boundary_budget,
                        'proposed_moves': int(moves), 'accepted': bool(accepted),
                        'max_part_iou_drop': .002, 'proposed_scores': proposed,
                        'changed_faces': int(np.count_nonzero(candidate != labels)),
                        'before': before, 'after': after,
                        'fragmentation_before': before_fragment,
                        'fragmentation_after': fragmentation(mesh, candidate, adjacency, weights),
                        'independent_part_iou_ceiling': per_part_iou_ceiling(seg_maps, evidence)}
