"""Geometry-only sampling and neighborhoods; never use GT to choose anchors.

Labels are attached to the already selected samples solely as supervision.
Disconnected shells never exchange query features. Opposing normals are filtered
where possible; fallbacks are explicitly marked, never silently counted as valid.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

FORMAT_VERSION = 2
BARYCENTRIC = np.array([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]], dtype=np.float32)


def geometry_components(vertices, faces, include_nonmanifold=False):
    # Exact welding avoids joining spatially close but disconnected parts.
    _, inverse = np.unique(vertices, axis=0, return_inverse=True)
    welded_faces = inverse[faces]
    if include_nonmanifold:
        # trimesh.face_adjacency omits edges with >2 incident faces. On meshes
        # containing duplicate triangles this can isolate almost every face.
        # Connect all incident faces by a deterministic star per exact edge:
        # O(edge incidences), not O(degree squared). Never proximity-weld shells,
        # never reorder/delete mesh faces or use labels to infer connectivity.
        edges = np.sort(welded_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        incident = np.repeat(np.arange(len(faces)), 3)
        valid = edges[:, 0] != edges[:, 1]
        edges, incident = edges[valid], incident[valid]
        if len(edges):
            _, edge_group, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
            order = np.argsort(edge_group, kind='stable')
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            representative = np.repeat(incident[order[starts]], counts)
            neighbor = incident[order]
            pairs = np.column_stack([representative, neighbor])
            adjacency = np.unique(np.sort(pairs[representative != neighbor], axis=1), axis=0)
        else:
            adjacency = np.empty((0, 2), np.int64)
    else:
        welded = trimesh.Trimesh(vertices=np.unique(vertices, axis=0), faces=welded_faces, process=False)
        adjacency = welded.face_adjacency.astype(np.int64)
    edge = np.concatenate([adjacency, adjacency[:, ::-1]])
    graph = coo_matrix((np.ones(len(edge)), (edge[:, 0], edge[:, 1])), shape=(len(faces), len(faces)))
    _, component = connected_components(graph, directed=False)
    return adjacency, component.astype(np.int32)


def build_geometry(vertices, faces, settings, seed):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Expected triangle geometry')
    if not len(faces) or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid mesh')
    extent = float(np.ptp(vertices, axis=0).max())
    if extent <= 0:
        raise ValueError('Zero mesh extent')
    center = (vertices.min(0) + vertices.max(0)) / 2
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    area = length / 2
    if area.sum() <= 0:
        raise ValueError('No positive-area faces')
    normals = cross / np.maximum(length[:, None], 1e-12)
    adjacency, component = geometry_components(vertices, faces)
    rng = np.random.default_rng(seed)
    area_faces = rng.choice(len(faces), size=settings['area_samples'], p=(area.astype(np.float64) / area.sum(dtype=np.float64)))
    uv = rng.random((len(area_faces), 2))
    sqrt_u = np.sqrt(uv[:, 0])
    bary = np.column_stack([1 - sqrt_u, sqrt_u * (1 - uv[:, 1]), sqrt_u * uv[:, 1]])
    # One centroid per face supplements area sampling for very small geometry.
    points = np.concatenate([tri.mean(1), np.einsum('ni,nij->nj', bary, tri[area_faces])]).astype(np.float32)
    point_face = np.concatenate([np.arange(len(faces)), area_faces])
    coords = np.clip(((points - center) / extent * 509 + 256).astype(np.int32), 0, 511)
    hashes = coords[:, 0].astype(np.int64) * 512**2 + coords[:, 1] * 512 + coords[:, 2]
    unique, inverse = np.unique(hashes, return_inverse=True)
    kept = np.arange(len(unique))
    repair = None
    if len(kept) > settings['max_voxels']:
        # Guarantee an anchor for each connected component, using geometry only.
        _, first = np.unique(component[point_face], return_index=True)
        required = np.unique(inverse[first])
        if len(required) > settings['max_voxels']:
            legacy_components = int(component.max()) + 1
            adjacency, component = geometry_components(vertices, faces, include_nonmanifold=True)
            _, first = np.unique(component[point_face], return_index=True)
            required = np.unique(inverse[first])
            repair = {'topology_repair': np.array(1, np.int32),
                      'legacy_component_count': np.array(legacy_components, np.int32)}
        if len(required) > settings['max_voxels']:
            raise ValueError('More components than the voxel budget, even with nonmanifold edge connectivity')
        remainder = np.setdiff1d(kept, required)
        kept = np.sort(np.r_[required, rng.choice(remainder, settings['max_voxels'] - len(required), replace=False)])
    remap = np.full(len(unique), -1, np.int32)
    remap[kept] = np.arange(len(kept))
    valid = remap[inverse] >= 0
    hashes = unique[kept]
    result = {
        'coords': np.column_stack([hashes // 512**2, hashes // 512 % 512, hashes % 512]).astype(np.int32),
        'anchor_points': points[valid], 'anchor_faces': point_face[valid].astype(np.int32),
        'anchor_voxel': remap[inverse[valid]], 'face_normals': normals.astype(np.float32),
        'face_area': area.astype(np.float32), 'face_components': component,
        'adjacency': adjacency.astype(np.int32), 'mesh_vertices': vertices, 'mesh_faces': faces.astype(np.int32),
        'center': center.astype(np.float32), 'extent': np.array(extent, np.float32),
    }
    if repair is not None:
        result.update(repair)
    return result


def attach_targets(geometry, labels, num_classes=20):
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer) or labels.shape != (len(geometry['mesh_faces']),):
        raise ValueError('Expected one integer label per face')
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f'GT labels outside 0..{num_classes-1}')
    counts = np.bincount(geometry['anchor_voxel'].astype(np.int64) * num_classes + labels[geometry['anchor_faces']],
                         minlength=len(geometry['coords']) * num_classes).reshape(-1, num_classes).astype(np.float32)
    # Encode soft distributions, preserving conflicting labels rather than hard majority.
    probability = counts / counts.sum(1, keepdims=True)
    annotated = counts.copy()
    annotated[:, 0] = 0
    result = dict(geometry, face_labels=labels.astype(np.int32), voxel_target=probability,
                  voxel_labels=annotated.argmax(1).astype(np.int32))
    return result


def near_surface_pairs(sample, radius_voxels):
    """Close cross-component faces provide separation supervision without fake edges."""
    points = sample['anchor_points']
    af = sample['anchor_faces']
    component = sample['face_components']
    distance, neighbors = cKDTree(points).query(points, k=min(8, len(points)))
    if neighbors.ndim == 1:
        return np.empty((0, 2), np.int32)
    left = np.broadcast_to(af[:, None], neighbors.shape)
    right = af[neighbors]
    valid = (component[left] != component[right]) & (distance <= radius_voxels * float(sample['extent']) / 509)
    pairs = np.column_stack([left[valid], right[valid]])
    return np.unique(np.sort(pairs, axis=1), axis=0).astype(np.int32)


class SurfaceQueries:
    def __init__(self, sample, settings):
        self.sample, self.settings = sample, settings
        self.triangles = sample['mesh_vertices'][sample['mesh_faces']]
        self.trees = {}
        anchors_component = sample['face_components'][sample['anchor_faces']]
        for comp in np.unique(anchors_component):
            ids = np.flatnonzero(anchors_component == comp)
            self.trees[int(comp)] = (ids, cKDTree(sample['anchor_points'][ids]))

    def build(self, face_ids):
        face_ids = np.asarray(face_ids, dtype=np.int64)
        sample, cfg = self.sample, self.settings
        points = np.einsum('qi,fij->fqj', BARYCENTRIC, self.triangles[face_ids]).reshape(-1, 3)
        query_faces = np.repeat(face_ids, len(BARYCENTRIC))
        normals = sample['face_normals'][query_faces]
        components = sample['face_components'][query_faces]
        k = cfg['neighbors']
        indices = np.zeros((len(points), k), np.int64)
        mask = np.zeros((len(points), k), bool)
        relative = np.zeros((len(points), k, 6), np.float32)
        prior = np.zeros((len(points), k), np.float32)
        fallback = np.zeros(len(points), bool)
        scale = 509 / float(sample['extent'])
        for comp in np.unique(components):
            qi = np.flatnonzero(components == comp)
            if int(comp) not in self.trees:
                raise ValueError(f'Geometry component {comp} has no retained anchor')
            anchors, tree = self.trees[int(comp)]
            distance, local = tree.query(points[qi], k=min(cfg['candidates'], len(anchors)))
            if local.ndim == 1:
                distance, local = distance[:, None], local[:, None]
            ai = anchors[local]
            offsets = (sample['anchor_points'][ai] - points[qi, None]) * scale
            dot = (sample['face_normals'][sample['anchor_faces'][ai]] * normals[qi, None]).sum(-1)
            normal_offset = (offsets * normals[qi, None]).sum(-1)
            permitted = dot >= cfg['normal_min_dot']
            missing = ~permitted.any(1)
            # Same-component fallback remains explicit in the query features/report.
            permitted[missing, 0] = True
            fallback[qi] = missing
            cost = np.log1p(distance * scale) + 2 * (1 - dot) + abs(normal_offset)
            cost = np.where(permitted, cost, np.inf)
            take = np.argsort(cost, axis=1, kind='stable')[:, :k]
            n = take.shape[1]
            sel = np.take_along_axis(ai, take, 1)
            valid = np.take_along_axis(permitted, take, 1)
            indices[qi, :n] = sample['anchor_voxel'][sel]
            mask[qi, :n] = valid
            chosen_offset = np.take_along_axis(offsets, take[..., None], 1)
            chosen_dot = np.take_along_axis(dot, take, 1)
            chosen_normal_offset = np.take_along_axis(normal_offset, take, 1)
            relative[qi, :n] = np.concatenate([chosen_offset, chosen_dot[..., None],
                                             chosen_normal_offset[..., None],
                                             np.log1p(np.linalg.norm(chosen_offset, axis=-1))[..., None]], -1)
            prior[qi, :n] = -np.take_along_axis(cost, take, 1)
        prior[~mask] = -1e4
        return {'neighbor_index': indices, 'neighbor_mask': mask, 'neighbor_geometry': relative,
                'neighbor_prior': prior, 'query_normal': normals.astype(np.float32),
                'query_position': ((points - sample['center']) / float(sample['extent'])).astype(np.float32),
                'query_fallback': fallback.astype(np.float32), 'query_face_ids': query_faces,
                'face_ids': face_ids}


def face_sampling_probability(labels, area, balanced_fraction):
    valid = (labels > 0) & (area > 0)
    if not valid.any():
        raise ValueError('No annotated surface')
    p_area = np.where(valid, area, 0).astype(np.float64)
    p_area /= p_area.sum()
    per_part = np.bincount(labels, weights=np.where(valid, area, 0))
    parts = np.count_nonzero(per_part[1:])
    p_balanced = np.where(valid, area / np.maximum(per_part[labels], 1e-30) / parts, 0)
    p = (1-balanced_fraction) * p_area + balanced_fraction * p_balanced
    return p / p.sum()


def training_queries(sample, settings, train, rng):
    p = face_sampling_probability(sample['face_labels'], sample['face_area'], train['balanced_fraction'])
    supervised = rng.choice(len(p), train['query_faces'], replace=True, p=p)
    edges = np.concatenate([sample['adjacency'], sample['near_pairs']])
    if len(edges):
        labels = sample['face_labels']
        edges = edges[(labels[edges] > 0).all(1)]
    chosen = []
    if len(edges):
        different = sample['face_labels'][edges[:, 0]] != sample['face_labels'][edges[:, 1]]
        for flag in (False, True):
            subset = edges[different == flag]
            if len(subset):
                chosen.append(subset[rng.choice(len(subset), min(len(subset), max(1, train['pair_count']//2)), replace=False)])
    pairs = np.concatenate(chosen) if chosen else np.empty((0, 2), np.int32)
    ids = np.concatenate([supervised, pairs.ravel()])
    query = SurfaceQueries(sample, settings).build(ids)
    query['supervised_count'] = len(supervised)
    query['surface_importance'] = (sample['face_area'][supervised] / p[supervised]).astype(np.float32)
    # Equal total mass per GT part, area-weighted within that part. Division by
    # the ACTUAL draw probability makes mean(CE * weight) estimate macro CE.
    macro = face_sampling_probability(sample['face_labels'], sample['face_area'], 1.)
    query['surface_macro_importance'] = (macro[supervised] / p[supervised]).astype(np.float32)
    query['pixel_importance'] = (sample['face_pixels'][supervised] / p[supervised]).astype(np.float32)
    query['pair_index'] = np.arange(len(supervised), len(ids)).reshape(-1, 2)
    query['pair_same'] = (sample['face_labels'][pairs[:, 0]] == sample['face_labels'][pairs[:, 1]])
    query['pair_geometry_weight'] = np.maximum(0, (sample['face_normals'][pairs[:, 0]] * sample['face_normals'][pairs[:, 1]]).sum(1)).astype(np.float32)
    query['query_labels'] = sample['face_labels'][ids]
    return query
