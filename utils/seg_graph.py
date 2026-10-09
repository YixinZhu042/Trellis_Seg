"""Surface Potts optimization with an explicit input-agreement budget.

Alpha expansion uses SciPy integer max-flow; each accepted move is verified
against the original floating point energy. Unanimous observed labels are hard
constraints. All reported projection metrics still use the original input maps.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order, connected_components


def surface_graph(mesh, resolution):
    import trimesh
    vertices, inverse = np.unique(np.asarray(mesh.vertices), axis=0, return_inverse=True)
    welded = trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
    adjacency, edges = welded.face_adjacency, welded.face_adjacency_edges
    scale = resolution / max(float(np.max(mesh.extents)), 1e-12)
    length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) * scale
    normal = np.einsum('ij,ij->i', welded.face_normals[adjacency[:, 0]], welded.face_normals[adjacency[:, 1]])
    weights = length * np.maximum(normal, 0.) ** 2
    return adjacency, weights


def energy(labels, unary, adjacency, weights):
    return float(unary[np.arange(len(labels)), labels].sum() +
                 weights[labels[adjacency[:, 0]] != labels[adjacency[:, 1]]].sum())


def alpha_expansion(labels, unary, adjacency, weights, iterations=3):
    labels = labels.copy()
    n = len(labels)
    left, right = adjacency.T
    rows = np.arange(n)
    value = energy(labels, unary, adjacency, weights)
    for _ in range(iterations):
        changed = False
        for alpha in range(1, unary.shape[1]):
            e00 = weights * (labels[left] != labels[right])
            e01 = weights * (labels[left] != alpha)
            e10 = weights * (alpha != labels[right])
            edge = (e01 + e10 - e00) / 2
            d0 = unary[rows, labels].copy()
            d1 = unary[:, alpha].copy()
            np.add.at(d1, left, e10 - e00 - edge)
            np.add.at(d1, right, e01 - e00 - edge)
            shift = np.minimum(d0, d1)
            d0 -= shift
            d1 -= shift
            # SciPy's flow output is int32 even for int64 input. Bound every
            # capacity by the feasible all-keep cut and scale that cut below
            # 2**28, preventing both individual and total-flow overflow.
            cut_bound = float(d0.sum()) + 1.
            quantization = min(1024., 2**28 / cut_bound)
            capacities = np.rint(np.clip(np.r_[d1, d0, edge, edge], 0, cut_bound)
                                   * quantization).astype(np.int64)
            graph = coo_matrix((capacities, (np.r_[np.full(n, n), rows, left, right],
                                             np.r_[rows, np.full(n, n + 1), right, left])),
                               shape=(n + 2, n + 2)).tocsr()
            flow = maximum_flow(graph, n, n + 1)
            residual = graph - flow.flow
            residual.data = (residual.data > 0).astype(np.int64)
            residual.eliminate_zeros()
            source_side = breadth_first_order(residual, n, directed=True, return_predecessors=False)
            switch = np.ones(n + 2, dtype=bool)
            switch[source_side] = False
            proposal = labels.copy()
            proposal[switch[:n]] = alpha
            new_value = energy(proposal, unary, adjacency, weights)
            if new_value < value - 1e-7:
                labels, value, changed = proposal, new_value, True
        if not changed:
            break
    return labels


def fragmentation(mesh, labels, adjacency, weights):
    same = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
    l, r = adjacency[same].T
    graph = coo_matrix((np.ones(2 * len(l)), (np.r_[l, r], np.r_[r, l])),
                       shape=(len(labels), len(labels))).tocsr()
    count, component = connected_components(graph, directed=False)
    areas = np.bincount(component, weights=mesh.area_faces)
    return {'components': int(count),
            'tiny_component_area_fraction': float(areas[areas < mesh.area * 1e-4].sum() / mesh.area),
            'boundary_length_pixels': float(weights[~same].sum())}


def optimize_conflicts(mesh, labels, evidence, strength):
    adjacency, weights = surface_graph(mesh, max(evidence.face_ids.shape[1:]))
    votes = evidence.votes.astype(float)
    unary = votes.max(1, keepdims=True) - votes
    # No absent labels on observed faces and no background labels anywhere.
    forbidden = max(float(votes.sum() + weights.sum() * strength + 1), 1e6)
    unary[:, 0] = forbidden
    visible = evidence.visible
    unary[visible] = np.where(votes[visible] > 0, unary[visible], forbidden)
    # Hidden surfaces retain their propagated/model prior; regularization can
    # connect them to a consistent neighboring patch when evidence warrants it.
    scale = max(evidence.face_ids.shape[1:]) / max(float(np.max(mesh.extents)), 1e-12)
    prior = mesh.area_faces * scale ** 2 * 0.05
    unary[~visible] = prior[~visible, None]
    unary[~visible, labels[~visible]] = 0
    unary[:, 0] = forbidden
    result = alpha_expansion(labels, unary, adjacency, strength * weights)
    return result, {'before': fragmentation(mesh, labels, adjacency, weights),
                    'after': fragmentation(mesh, result, adjacency, weights),
                    'strength': float(strength),
                    'changed_faces': int(np.count_nonzero(labels != result))}


def optimize_with_budget(mesh, labels, seg_maps, evidence, max_accuracy_drop=.003, max_miou_drop=.01):
    """Choose the lowest-boundary candidate under both input-fidelity limits.

    Candidates use only geometry, the fixed model prediction and input masks.
    GT annotations are never consulted. Strict 0 budgets retain exact fidelity.
    """
    from .seg_projection import consistency_metrics
    before = consistency_metrics(labels, seg_maps, evidence)
    adjacency, weights = surface_graph(mesh, max(evidence.face_ids.shape[1:]))
    baseline_frag = fragmentation(mesh, labels, adjacency, weights)
    best = labels.copy()
    best_frag = baseline_frag
    chosen = 0.
    trials = []
    for strength in [.1, .2, .4, .8]:
        candidate, detail = optimize_conflicts(mesh, labels, evidence, strength)
        score = consistency_metrics(candidate, seg_maps, evidence)
        accepted = (score['accuracy'] >= before['accuracy'] - max_accuracy_drop - 1e-12 and
                    score['miou'] >= before['miou'] - max_miou_drop - 1e-12)
        # Avoid erasing even a tiny input part already present in the solution.
        accepted = accepted and set(np.unique(labels)).issubset(np.unique(candidate))
        trials.append({'strength': strength, 'accuracy': score['accuracy'], 'miou': score['miou'],
                       'within_budget': bool(accepted), 'fragmentation': detail['after']})
        if accepted and detail['after']['boundary_length_pixels'] < best_frag['boundary_length_pixels']:
            best, best_frag, chosen = candidate, detail['after'], strength
    return best, {'before': baseline_frag, 'after': best_frag, 'selected_strength': chosen,
                   'max_accuracy_drop': max_accuracy_drop, 'max_miou_drop': max_miou_drop,
                   'trials': trials}


def smooth_hidden_surfaces(mesh, labels, evidence, strength=.25, prior_weight=.5):
    """Denoise hidden model boundaries while locking every observed face.

    Area-weighted model priors keep large hidden parts stable. The perimeter
    penalty removes small islands without depending on triangle density.
    """
    adjacency, weights = surface_graph(mesh, max(evidence.face_ids.shape[1:]))
    scale = max(evidence.face_ids.shape[1:]) / max(float(np.max(mesh.extents)), 1e-12)
    penalty = mesh.area_faces * scale ** 2 * prior_weight
    unary = np.broadcast_to(penalty[:, None], evidence.votes.shape).copy()
    unary[np.arange(len(labels)), labels] = 0
    forbidden = max(float(penalty.sum() + strength * weights.sum() + 1), 1e6)
    unary[evidence.visible] = forbidden
    unary[np.flatnonzero(evidence.visible), labels[evidence.visible]] = 0
    unary[:, 0] = forbidden
    result = alpha_expansion(labels, unary, adjacency, weights * strength)
    if not np.array_equal(result[evidence.visible], labels[evidence.visible]):
        raise RuntimeError('Hidden graph optimization modified an observed label')
    return result, {'strength': strength, 'prior_weight': prior_weight,
                    'changed_faces': int(np.count_nonzero(result != labels)),
                    'before': fragmentation(mesh, labels, adjacency, weights),
                    'after': fragmentation(mesh, result, adjacency, weights)}
