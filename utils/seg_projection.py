"""Visibility-aware fusion of the six canonical segmentation views.

Part IDs in this module are the input IDs: 0 is background, 1..K are parts.
Model labels have a separate namespace and must never be copied into this one.
The initial objective is the number of agreeing foreground pixels across all
views. Each visible face takes a maximum-vote label, attaining the exact optimum
for a fixed mesh with one label per face. Optional graph cleanup reduces boundary
fragmentation within explicitly reported accuracy/mIoU budgets; unanimous input
evidence remains locked.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


VIEW_NAMES = ('front', 'right', 'back', 'left', 'top', 'bottom')
VIEW_CONFIGS = (
    (2, 1, 1, 0, False), (0, 1, 1, 2, True),
    (2, -1, 1, 0, True), (0, -1, 1, 2, False),
    (1, 1, 2, 0, False), (1, -1, 2, 0, False),
)


def validate_seg_maps(seg_maps, max_parts=None):
    seg_maps = np.asarray(seg_maps)
    if seg_maps.ndim != 3 or seg_maps.shape[0] != 6 or min(seg_maps.shape[1:]) < 1:
        raise ValueError('seg_maps must have shape (6, H, W)')
    if not np.issubdtype(seg_maps.dtype, np.integer) or np.any(seg_maps < 0):
        raise ValueError('seg_maps must contain nonnegative integer part IDs')
    if not np.any(seg_maps > 0):
        raise ValueError('seg_maps has no foreground parts')
    if max_parts is not None and seg_maps.max() > max_parts:
        raise ValueError(f'seg_maps contains IDs above the supported maximum {max_parts}')
    return seg_maps


def normalized_vertices(mesh):
    # Match render_views.py, including its float32 normalization.
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if not len(vertices) or not len(mesh.faces) or not np.isfinite(vertices).all():
        raise ValueError('Expected a nonempty mesh with finite vertices')
    lower, upper = vertices.min(0), vertices.max(0)
    extent = float((upper - lower).max())
    if extent <= 0:
        raise ValueError('Mesh has zero extent')
    return (vertices - (lower + upper) / 2) / extent * 2


def _rasterize_cpu_impl(vertices, triangles, height, width):
    """Pixel-center orthographic z buffer, unlike the legacy integer-vertex renderer."""
    result = np.full((6, height, width), -1, dtype=np.int32)
    for view in range(6):
        axis, sign, up, right, flip = VIEW_CONFIGS[view]
        px = (vertices[:, right] * (-1 if flip else 1) + 1) * width / 2 - 0.5
        py = (1 - vertices[:, up]) * height / 2 - 0.5
        depth = vertices[:, axis] * sign
        z_buffer = np.full((height, width), -np.inf)
        for face in range(len(triangles)):
            a, b, c = triangles[face]
            x0, x1, x2 = px[a], px[b], px[c]
            y0, y1, y2 = py[a], py[b], py[c]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denominator) < 1e-12:
                continue
            xmin = max(0, int(np.ceil(min(x0, x1, x2))))
            xmax = min(width - 1, int(np.floor(max(x0, x1, x2))))
            ymin = max(0, int(np.ceil(min(y0, y1, y2))))
            ymax = min(height - 1, int(np.floor(max(y0, y1, y2))))
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
                    w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
                    w2 = 1 - w0 - w1
                    if min(w0, w1, w2) < -1e-9:
                        continue
                    z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
                    if z > z_buffer[y, x]:
                        z_buffer[y, x] = z
                        result[view, y, x] = face
    return result


_cpu_rasterizer = None


def rasterize_face_ids(mesh, image_shape, device='cpu'):
    """Return (6,H,W) original face indices, -1 at background, in saved-PNG order.

    CPU and CUDA use the same cameras, pixel centers, and z test. Exact shared-edge
    and coplanar tie ownership can differ between rasterizers.
    """
    height, width = map(int, image_shape)
    if height < 1 or width < 1:
        raise ValueError('Raster dimensions must be positive')
    vertices = normalized_vertices(mesh)
    triangles = np.asarray(mesh.faces, dtype=np.int32)
    if str(device) == 'cpu':
        global _cpu_rasterizer
        if _cpu_rasterizer is None:
            from numba import njit
            _cpu_rasterizer = njit(cache=True)(_rasterize_cpu_impl)
        return _cpu_rasterizer(vertices, triangles, height, width)
    import torch
    import nvdiffrast.torch as dr
    with torch.no_grad(), torch.cuda.device(device):
        v = torch.as_tensor(vertices, device=device)
        clips = []
        for axis, sign, up, right, flip in VIEW_CONFIGS:
            x = v[:, right] * (-1 if flip else 1)
            clips.append(torch.stack([x, v[:, up], -v[:, axis] * sign, torch.ones_like(x)], -1))
        raster, _ = dr.rasterize(
            dr.RasterizeCudaContext(device=device), torch.stack(clips).contiguous(),
            torch.as_tensor(triangles, device=device), resolution=[height, width])
        return (raster[..., 3].int() - 1).flip(1).cpu().numpy()


@dataclass
class ProjectionEvidence:
    face_ids: np.ndarray
    votes: np.ndarray
    visible: np.ndarray
    confidence: np.ndarray


def collect_evidence(mesh, seg_maps, device='cpu', face_ids=None):
    seg_maps = validate_seg_maps(seg_maps)
    if face_ids is None:
        face_ids = rasterize_face_ids(mesh, seg_maps.shape[1:], device)
    face_ids = np.asarray(face_ids)
    if (face_ids.shape != seg_maps.shape or not np.issubdtype(face_ids.dtype, np.integer)
            or face_ids.min() < -1 or face_ids.max() >= len(mesh.faces)):
        raise ValueError('Invalid face-index raster for this mesh and segmentation')
    num_labels = int(seg_maps.max()) + 1
    valid = (face_ids >= 0) & (seg_maps > 0)
    linear = face_ids[valid].astype(np.int64) * num_labels + seg_maps[valid]
    votes = np.bincount(linear, minlength=len(mesh.faces) * num_labels).reshape(-1, num_labels)
    total = votes.sum(1)
    confidence = votes.max(1) / np.maximum(total, 1)
    return ProjectionEvidence(face_ids, votes, total > 0, confidence)


def _model_overlap(model_labels, votes):
    model_ids, inverse = np.unique(model_labels, return_inverse=True)
    overlap = np.zeros((len(model_ids), votes.shape[1]), dtype=np.int64)
    np.add.at(overlap, inverse, votes)
    return model_ids, inverse, overlap


def align_model_labels(model_labels, votes):
    """One GLOBAL one-to-one permutation for fair baseline evaluation across views.

    Unmatched model parts become 0 (an error on foreground), never many-to-one
    matches. A per-view assignment could hide cross-view label inconsistency.
    """
    ids, inverse, overlap = _model_overlap(model_labels, votes)
    mapping = np.zeros(len(ids), dtype=np.int32)
    parts = np.flatnonzero(votes.sum(0))
    if len(parts):
        rows, cols = linear_sum_assignment(-overlap[:, parts])
        positive = overlap[rows, parts[cols]] > 0
        mapping[rows[positive]] = parts[cols[positive]]
    return mapping[inverse], {int(i): int(p) for i, p in zip(ids, mapping)}


def welded_face_adjacency(mesh):
    """Reconnect exactly coincident vertices across material/UV seams, without
    moving vertices or changing face order. Do not bridge separate nearby shells.
    """
    import trimesh
    vertices, inverse = np.unique(np.asarray(mesh.vertices), axis=0, return_inverse=True)
    welded = trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
    return welded.face_adjacency


def fuse_labels(mesh, model_labels, evidence):
    """Maximize visible agreement and extend labels through hidden surface paths."""
    model_labels = np.asarray(model_labels)
    if (model_labels.shape != (len(mesh.faces),) or
            not np.issubdtype(model_labels.dtype, np.integer)):
        raise ValueError('Expected one integer model label per mesh face')
    votes, visible = evidence.votes, evidence.visible
    if not visible.any():
        raise ValueError('No segmentation foreground overlaps the mesh; check cameras/registration')
    ids, inverse, overlap = _model_overlap(model_labels, votes)
    # Only used without evidence or for exact ties. A model part may split into
    # several input parts on visible faces; no one-to-one bottleneck in fusion.
    dominant = int(votes.sum(0).argmax())
    mapping = overlap.argmax(1).astype(np.int32)
    mapping[overlap.sum(1) == 0] = dominant
    labels = mapping[inverse]
    best = votes.argmax(1).astype(np.int32)
    tied = votes[np.arange(len(labels)), labels] == votes.max(1)
    labels[visible & ~tied] = best[visible & ~tied]

    unseen = ~visible
    propagated = np.zeros(len(labels), dtype=bool)
    if unseen.any():
        adjacency = welded_face_adjacency(mesh)
        if len(adjacency):
            left, right = adjacency.T
            centers = np.asarray(mesh.triangles_center)
            extent = max(float(np.max(mesh.extents)), 1e-12)
            weights = np.maximum(np.linalg.norm(centers[left] - centers[right], axis=1) / extent, 1e-9)
            weights *= np.where(model_labels[left] == model_labels[right], 1., 8.)
            graph = coo_matrix((np.r_[weights, weights],
                                (np.r_[left, right], np.r_[right, left])),
                               shape=(len(labels), len(labels))).tocsr()
            _, _, sources = dijkstra(graph, directed=False, indices=np.flatnonzero(visible),
                                     min_only=True, return_predecessors=True)
            propagated = unseen & (sources >= 0)
            labels[propagated] = labels[sources[propagated]]
    diagnostics = {
        'visible_faces': int(visible.sum()), 'unobserved_faces': int(unseen.sum()),
        'propagated_faces': int(propagated.sum()),
        'fallback_faces': int((unseen & ~propagated).sum()),
        'model_to_input_fallback': {int(i): int(p) for i, p in zip(ids, mapping)},
    }
    return labels, diagnostics


def subdivide_conflicting_faces(mesh, model_labels, evidence, min_edge_pixels=4., max_faces=500000):
    """Conforming midpoint refinement where a face receives multiple input IDs.

    Splits all edges of selected faces and the same edges on their neighbors.
    Green triangles on neighboring faces avoid T junctions. Surface coordinates
    stay on the original triangles. Returns (mesh, inherited model labels,
    parent face indices); identity indices indicate no refinement.
    """
    import trimesh
    votes = evidence.votes
    conflict = votes.sum(1) > votes.max(1)
    v = normalized_vertices(mesh).astype(np.float64)
    triangles = v[mesh.faces]
    longest = np.maximum.reduce([
        np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1),
    ]) * max(evidence.face_ids.shape[1:]) / 2
    candidates = np.flatnonzero(conflict & (longest > min_edge_pixels))
    # Each red face can add at most 3 children and 3 neighbor children.
    capacity = max(0, (max_faces - len(mesh.faces)) // 6)
    if len(candidates) > capacity:
        error = votes[candidates].sum(1) - votes[candidates].max(1)
        candidates = candidates[np.argsort(-error, kind='stable')[:capacity]]
    if not len(candidates):
        return mesh, np.asarray(model_labels), np.arange(len(mesh.faces))

    vertices, inverse = np.unique(np.asarray(mesh.vertices), axis=0, return_inverse=True)
    faces = inverse[mesh.faces]
    chosen = faces[candidates]
    selected_edges = np.sort(np.concatenate([chosen[:, [0, 1]], chosen[:, [1, 2]], chosen[:, [2, 0]]]), axis=1)
    selected_edges = np.unique(selected_edges, axis=0)
    # GLB stores float32 vertices. Subdividing almost-collinear triangles whose
    # altitude is below that precision can flip their winding on export. Protect
    # all their edges (including neighbor-induced splits), leaving them intact.
    tri = vertices[faces]
    edge_max = np.maximum.reduce([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                                  np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                                  np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)])
    altitude = 2 * mesh.area_faces / np.maximum(edge_max, 1e-30)
    precision = np.spacing(np.abs(tri).astype(np.float32)).max(axis=(1, 2))
    unsafe = faces[altitude < 32 * precision]
    if len(unsafe):
        protected = np.sort(np.concatenate([unsafe[:, [0, 1]], unsafe[:, [1, 2]], unsafe[:, [2, 0]]]), axis=1)
        protected_hashes = set(map(tuple, protected))
        selected_edges = np.asarray([e for e in selected_edges if tuple(e) not in protected_hashes], dtype=np.int64).reshape(-1, 2)
    if not len(selected_edges):
        return mesh, np.asarray(model_labels), np.arange(len(mesh.faces))
    midpoint_ids = {tuple(e): len(vertices) + i for i, e in enumerate(selected_edges)}
    vertices = np.concatenate([vertices, vertices[selected_edges].mean(1)])
    new_faces, parents = [], []
    for parent, (a, b, c) in enumerate(faces):
        ab = midpoint_ids.get(tuple(sorted((a, b))))
        bc = midpoint_ids.get(tuple(sorted((b, c))))
        ca = midpoint_ids.get(tuple(sorted((c, a))))
        count = sum(x is not None for x in (ab, bc, ca))
        if count == 0:
            children = [(a, b, c)]
        elif count == 3:
            children = [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        elif count == 1:
            if ab is not None:
                children = [(a, ab, c), (ab, b, c)]
            elif bc is not None:
                children = [(b, bc, a), (bc, c, a)]
            else:
                children = [(c, ca, b), (ca, a, b)]
        else:
            # Rotate so the two split edges meet at b, preserving winding.
            if ab is None:
                a, b, c, ab, bc = b, c, a, bc, ca
            elif bc is None:
                a, b, c, ab, bc = c, a, b, ca, ab
            children = [(a, ab, c), (ab, bc, c), (ab, b, bc)]
        new_faces.extend(children)
        parents.extend([parent] * len(children))
    parents = np.asarray(parents, dtype=np.int64)
    refined = trimesh.Trimesh(vertices=vertices, faces=new_faces, process=False)
    return refined, np.asarray(model_labels)[parents], parents


def _regularize_impl(labels, votes, offsets, neighbors, weights, iterations):
    labels = labels.copy()
    for iteration in range(iterations):
        changed = 0
        for j in range(len(labels)):
            face = j if iteration % 2 == 0 else len(labels) - j - 1
            total = votes[face].sum()
            if total == 0 or votes[face].max() == total:
                continue  # unanimous input evidence is never smoothed away
            current = labels[face]
            best_label = current
            best_cost = 0.
            for edge in range(offsets[face], offsets[face + 1]):
                if labels[neighbors[edge]] != current:
                    best_cost += weights[edge]
            for label in range(1, votes.shape[1]):
                if votes[face, label] != votes[face].max() or label == current:
                    continue
                cost = 0.
                for edge in range(offsets[face], offsets[face + 1]):
                    if labels[neighbors[edge]] != label:
                        cost += weights[edge]
                if cost < best_cost - 1e-10:
                    best_cost, best_label = cost, label
            if best_label != current:
                labels[face] = best_label
                changed += 1
        if changed == 0:
            break
    return labels


_regularizer = None


def smooth_vote_ties(mesh, labels, evidence):
    """Reduce boundary length only among equally supported input labels.

    The Potts weight is shared edge length in image pixels, so it does not grow
    just because a surface was triangulated more densely. Every selected label
    retains maximum pixel support; smoothing cannot sacrifice input agreement.
    """
    import trimesh
    vertices, inverse = np.unique(np.asarray(mesh.vertices), axis=0, return_inverse=True)
    welded = trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
    edges = welded.face_adjacency_edges
    adjacency = welded.face_adjacency
    if not len(edges):
        return labels.copy()
    left, right = adjacency.T
    edge_length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    edge_length *= max(evidence.face_ids.shape[1:]) / max(float(np.max(mesh.extents)), 1e-12)
    # Sharp geometric creases are plausible boundaries and receive less penalty.
    cosine = np.einsum('ij,ij->i', mesh.face_normals[left], mesh.face_normals[right])
    weights = edge_length * np.maximum(cosine, 0.) ** 2
    graph = coo_matrix((np.r_[weights, weights], (np.r_[left, right], np.r_[right, left])),
                       shape=(len(labels), len(labels))).tocsr()
    global _regularizer
    if _regularizer is None:
        from numba import njit
        _regularizer = njit(cache=True)(_regularize_impl)
    return _regularizer(labels, evidence.votes, graph.indptr, graph.indices, graph.data, 10)


def restore_confident_hidden(labels, model_labels, evidence, confidence=.8):
    """Keep reliable model parts on unobserved faces without changing pixel scores.

    Confidence is the global projected-pixel agreement for that model part.
    It is computed from input masks only, not GT or hidden surface annotations.
    A threshold of 1 disables all but perfectly agreeing model-part mappings.
    """
    if not 0 <= confidence <= 1:
        raise ValueError('Hidden prior confidence must be between 0 and 1')
    ids, inverse, overlap = _model_overlap(model_labels, evidence.votes)
    total = overlap.sum(1)
    agreement = overlap.max(1) / np.maximum(total, 1)
    mapping = overlap.argmax(1)
    eligible = ~evidence.visible & (agreement[inverse] >= confidence) & (total[inverse] > 0)
    result = labels.copy()
    result[eligible] = mapping[inverse[eligible]]
    return result, {'confidence': confidence, 'eligible_faces': int(eligible.sum()),
                    'changed_faces': int(np.count_nonzero(result != labels))}


def refine_projection(mesh, model_labels, seg_maps, device='cpu', subdivide=0,
                      graph_cleanup=False, accuracy_budget=.003, miou_budget=.01,
                      hidden_confidence=.8, hidden_smooth=.25, part_iou_refine=True):
    """Shared inference/frozen-prediction path, including optional boundary refinement."""
    if not 0 <= subdivide <= 5:
        raise ValueError('subdivide must be between 0 and 5')
    if not np.isfinite(hidden_smooth) or hidden_smooth < 0:
        raise ValueError('hidden_smooth must be finite and nonnegative')
    evidence = collect_evidence(mesh, seg_maps, device=device)
    baseline, mapping = align_model_labels(model_labels, evidence.votes)
    baseline_metrics = consistency_metrics(baseline, seg_maps, evidence)
    parents = np.arange(len(mesh.faces))
    history = []
    for step in range(subdivide):
        finer, inherited, step_parents = subdivide_conflicting_faces(mesh, model_labels, evidence)
        if len(finer.faces) == len(mesh.faces):
            break
        baseline = baseline[step_parents]
        parents = parents[step_parents]
        mesh, model_labels = finer, inherited
        evidence = collect_evidence(mesh, seg_maps, device=device)
        history.append({'iteration': step + 1, 'faces': len(mesh.faces),
                        'accuracy_ceiling': float(evidence.votes.max(1).sum() / (seg_maps > 0).sum())})
    labels, diagnostics = fuse_labels(mesh, model_labels, evidence)
    before_regularization = consistency_metrics(labels, seg_maps, evidence)
    clean = smooth_vote_ties(mesh, labels, evidence)
    diagnostics['smoothed_tie_faces'] = int(np.count_nonzero(clean != labels))
    labels = clean
    graph_report = None
    if graph_cleanup:
        from .seg_graph import optimize_with_budget
        labels, graph_report = optimize_with_budget(
            mesh, labels, seg_maps, evidence, accuracy_budget, miou_budget)
    labels, hidden_report = restore_confident_hidden(labels, model_labels, evidence, hidden_confidence)
    if hidden_smooth > 0 and (~evidence.visible).any():
        from .seg_graph import smooth_hidden_surfaces
        labels, hidden_report['cleanup'] = smooth_hidden_surfaces(mesh, labels, evidence, hidden_smooth)
    part_iou_report = None
    if graph_cleanup and part_iou_refine:
        from .seg_miou import refine_part_iou
        current = consistency_metrics(labels, seg_maps, evidence)
        # Retain the overall accuracy floor promised by graph cleanup, even
        # when the optional part-balanced pass follows regularization.
        remaining_budget = max(0., accuracy_budget - current['gap_to_ceiling'])
        labels, part_iou_report = refine_part_iou(
            mesh, labels, seg_maps, evidence, accuracy_budget=min(.001, remaining_budget))
    report = {
        'baseline_global_mapping': mapping, 'baseline': baseline_metrics,
        'refined': consistency_metrics(labels, seg_maps, evidence),
        'before_regularization': before_regularization,
        'diagnostics': diagnostics, 'refinement': history,
        'original_mesh_accuracy_ceiling': baseline_metrics['fixed_mesh_accuracy_ceiling'],
        'graph': graph_report,
        'hidden_prior': hidden_report,
        'part_iou_refinement': part_iou_report,
    }
    return mesh, labels, parents, evidence, baseline, report


def render_labels(face_labels, face_ids):
    result = np.zeros(face_ids.shape, dtype=np.int32)
    valid = face_ids >= 0
    result[valid] = np.asarray(face_labels)[face_ids[valid]]
    return result


def _pixel_metrics(target, prediction, mask, part_ids):
    count = int(mask.sum())
    if count == 0:
        return {'pixels': 0, 'accuracy': None, 'miou': None, 'part_iou': {}}
    gt, pred = target[mask], prediction[mask]
    ious = {}
    for part in part_ids:
        union = np.count_nonzero((gt == part) | (pred == part))
        if union:
            ious[int(part)] = float(np.count_nonzero((gt == part) & (pred == part)) / union)
    return {'pixels': count, 'accuracy': float(np.mean(gt == pred)),
            'miou': float(np.mean(list(ious.values()))) if ious else None, 'part_iou': ious}


def consistency_metrics(face_labels, seg_maps, evidence):
    """Fixed input IDs; missing mesh coverage counts as disagreement in the primary score."""
    pred = render_labels(face_labels, evidence.face_ids)
    foreground = seg_maps > 0
    hit = evidence.face_ids >= 0
    overlap = hit & foreground
    part_ids = np.unique(seg_maps[foreground])
    result = _pixel_metrics(seg_maps, pred, foreground, part_ids)
    result['overlap'] = _pixel_metrics(seg_maps, pred, overlap, part_ids)
    result['foreground_coverage'] = float(overlap.sum() / foreground.sum())
    result['silhouette_iou'] = float(overlap.sum() / np.count_nonzero(hit | foreground))
    result['mesh_pixels_without_input_label'] = int(np.count_nonzero(hit & ~foreground))
    best_correct = int(evidence.votes.max(1).sum())
    result['fixed_mesh_accuracy_ceiling'] = best_correct / int(foreground.sum())
    result['overlap_accuracy_ceiling'] = best_correct / max(int(overlap.sum()), 1)
    result['gap_to_ceiling'] = result['fixed_mesh_accuracy_ceiling'] - result['accuracy']
    result['views'] = {
        name: _pixel_metrics(seg_maps[i], pred[i], foreground[i], part_ids)
        for i, name in enumerate(VIEW_NAMES)
    }
    return result
