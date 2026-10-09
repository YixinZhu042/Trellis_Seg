"""
Evaluate Trellis SegPart prediction quality on 3D part-segmentation benchmarks.

Reads predicted colored GLB files produced by scripts/batch_inference.py or
scripts/inference_wild.py, recovers per-face labels from vertex colors, loads
per-face GT labels, and reports face-level metrics.

Supported GT sources (`--dataset`):
  * `partnext`      : PartNeXt arrow annotations (label 0 = unlabeled, excluded).
  * `partobjaverse` : PartObjaverse-Tiny / SAMPart3D benchmark, 200 objects with
                      per-face semantic and instance GT (label 0 IS a valid part,
                      so it is included by default).

Metric suite:
  * Published protocol (comparable to SAMPart3D / PartSLIP / PartNet tables)
      - `miou_ca`              : class-agnostic mIoU. For each GT part take the
                                 max IoU over all predictions and average over GT
                                 parts. No one-to-one constraint, so several GT
                                 parts may claim the same prediction -- this is
                                 exactly the SAMPart3D §4.2 / PartSLIP formula.
      - `ap50`                 : per-object mAP at IoU=0.5 (object treated as its
                                 own category), greedy highest-IoU-first
                                 assignment, each prediction consumed once.
                                 Caveat: the model emits no per-part confidence,
                                 so the predicted part's area share is used as a
                                 proxy. Ranking-sensitive AP is therefore only
                                 approximately comparable across papers.
  * Stricter internal diagnostics
      - `face_acc`, `miou_h`   : Hungarian one-to-one label matching. Unlike
                                 `miou_ca` this penalises many-to-one collapse,
                                 so `miou_h <= miou_ca` always.
      - `miou_ca_pred`         : pred -> best GT, penalises over-segmentation.
      - `ap25` / `ap75`        : same as `ap50` at other IoU thresholds.
      - `prec50/rec50/f1_50`   : threshold-based matching counts.
  * Clustering agreement (label-permutation invariant): `ari`, `nmi`
  * Boundary quality and spatial coherence:
      `boundary_precision/recall/f1`, component statistics.

All region metrics are computed both face-count weighted and surface-area
weighted (`*_area` suffix); area weighting is the geometrically faithful one for
meshes with non-uniform triangulation.

Examples:
    # PartNeXt (training-set diagnostics)
    python scripts/evaluate_seg_quality.py \
        --pred_dir ./batch_results_demo \
        --output_dir ./eval/seg_quality

    # PartObjaverse-Tiny (held-out benchmark, publishable protocol)
    python scripts/evaluate_seg_quality.py \
        --dataset partobjaverse \
        --pred_dir ./results_partobjaverse \
        --output_dir ./eval/partobjaverse
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque
from glob import glob

import numpy as np
import trimesh
from tqdm import tqdm

try:
    import pyarrow.ipc as ipc
except ImportError:
    ipc = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except ImportError:
    adjusted_rand_score = None
    normalized_mutual_info_score = None


DEFAULT_ARROW_DIR = '/path/to/PartNeXt'
DEFAULT_MESH_DIR = '/path/to/PartNeXt_mesh/glbs'
DEFAULT_PARTOBJAVERSE_ROOT = '/path/to/PartObjaverse-Tiny'


BASE_PALETTE = np.array([
    [0.900, 0.098, 0.294], [0.235, 0.706, 0.294], [0.263, 0.388, 0.847],
    [0.957, 0.831, 0.118], [0.522, 0.078, 0.694], [0.957, 0.522, 0.114],
    [0.275, 0.941, 0.941], [0.941, 0.196, 0.902], [0.824, 0.961, 0.235],
    [0.980, 0.745, 0.831], [0.000, 0.502, 0.502], [0.902, 0.745, 1.000],
    [0.667, 0.431, 0.157], [1.000, 0.980, 0.784], [0.502, 0.000, 0.000],
    [0.667, 1.000, 0.765], [0.502, 0.502, 0.000], [1.000, 0.843, 0.706],
    [0.000, 0.000, 0.502], [0.502, 0.502, 0.502],
], dtype=np.float32)


def generate_class_colors(num_classes):
    if num_classes <= len(BASE_PALETTE):
        return BASE_PALETTE[:num_classes]
    colors = np.zeros((num_classes, 3), dtype=np.float32)
    colors[:len(BASE_PALETTE)] = BASE_PALETTE
    for i in range(len(BASE_PALETTE), num_classes):
        colors[i] = BASE_PALETTE[i % len(BASE_PALETTE)] * 0.6
    return colors


def _as_face_list(face_indices):
    if isinstance(face_indices, int):
        return (face_indices,)
    return face_indices


def parse_face_labels(masks_json, num_faces, hierarchy_json=None, max_parts=19):
    masks = json.loads(masks_json) if isinstance(masks_json, str) else masks_json
    num_raw_parts = len(masks)

    if num_raw_parts <= max_parts or hierarchy_json is None:
        face_labels = np.full(num_faces, 0, dtype=np.int32)
        part_id = 1
        for part_key in sorted(masks.keys(), key=int):
            for _, face_indices in masks[part_key].items():
                for fi in _as_face_list(face_indices):
                    if 0 <= fi < num_faces:
                        face_labels[fi] = part_id
            part_id += 1
        return face_labels, part_id - 1

    hierarchy = json.loads(hierarchy_json) if isinstance(hierarchy_json, str) else hierarchy_json
    all_node_ids = []

    def collect_nodes(node):
        all_node_ids.append(node['nodeId'])
        for child in node.get('children', []):
            collect_nodes(child)

    roots = hierarchy if isinstance(hierarchy, list) else [hierarchy]
    for root in roots:
        collect_nodes(root)

    def get_nodes_at_depth(node, target_depth, current_depth=0):
        if current_depth == target_depth or not node.get('children'):
            return [node]
        nodes = []
        for child in node['children']:
            nodes.extend(get_nodes_at_depth(child, target_depth, current_depth + 1))
        return nodes

    def get_descendant_ids(node):
        ids = [node['nodeId']]
        for child in node.get('children', []):
            ids.extend(get_descendant_ids(child))
        return ids

    best_groups = None
    for depth in range(1, 10):
        groups = []
        for root in roots:
            groups.extend(get_nodes_at_depth(root, depth))
        if len(groups) <= max_parts:
            best_groups = groups
            if len(groups) >= 2:
                break
        else:
            if best_groups is None:
                best_groups = groups
            break
    if best_groups is None:
        best_groups = roots

    face_labels = np.full(num_faces, 0, dtype=np.int32)
    part_id = 1
    mask_keys = sorted(masks.keys(), key=int)
    for group_node in best_groups:
        descendant_ids = set(get_descendant_ids(group_node))
        for idx, mask_key in enumerate(mask_keys):
            if idx < len(all_node_ids) and all_node_ids[idx] in descendant_ids:
                for _, face_indices in masks[mask_key].items():
                    for fi in _as_face_list(face_indices):
                        if 0 <= fi < num_faces:
                            face_labels[fi] = part_id
        part_id += 1
    return face_labels, part_id - 1


def load_mesh(path):
    scene = trimesh.load(path, force='scene')
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_geometry()
    else:
        mesh = scene
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f'not a Trimesh: {path}')
    return mesh


def load_partnext_records(arrow_dir, target_ids=None, max_arrow_files=None):
    if ipc is None:
        raise ImportError('pyarrow is required to read PartNeXt arrow annotations')
    target_ids = set(target_ids) if target_ids else None
    records = {}
    arrow_files = sorted(glob(os.path.join(arrow_dir, 'data-*.arrow')))
    if max_arrow_files is not None:
        arrow_files = arrow_files[:max_arrow_files]

    for arrow_file in tqdm(arrow_files, desc='Loading PartNeXt arrows'):
        with open(arrow_file, 'rb') as f:
            reader = ipc.open_stream(f)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                names = set(batch.schema.names)
                for i in range(batch.num_rows):
                    model_id = batch.column('model_id')[i].as_py()
                    if target_ids is not None and model_id not in target_ids:
                        continue
                    records[model_id] = {
                        'model_id': model_id,
                        'type_id': batch.column('type_id')[i].as_py(),
                        'masks': batch.column('masks')[i].as_py(),
                        'hierarchy': batch.column('hierarchyList')[i].as_py() if 'hierarchyList' in names else None,
                    }
                if target_ids is not None and target_ids.issubset(records.keys()):
                    return records
    return records


def flatten_partobjaverse_semantic(semantic_json):
    """PartObjaverse-Tiny_semantic.json is nested as {category: {uid: [labels]}}.

    Returns (uid -> label list, uid -> category).
    """
    labels_by_uid = {}
    category_by_uid = {}
    for category, entries in semantic_json.items():
        if not isinstance(entries, dict):
            raise ValueError(f'unexpected semantic.json layout for category {category!r}')
        for uid, labels in entries.items():
            labels_by_uid[uid] = labels
            category_by_uid[uid] = category
    return labels_by_uid, category_by_uid


def load_partobjaverse_records(root, task='semantic', target_ids=None):
    """Build GT records for the PartObjaverse-Tiny benchmark (SAMPart3D).

    Layout (after unzipping the HF dataset):
        {root}/PartObjaverse-Tiny_semantic.json
        {root}/PartObjaverse-Tiny_mesh/{uid}.glb
        {root}/PartObjaverse-Tiny_semantic_gt/{uid}.npy   per-face semantic ids
        {root}/PartObjaverse-Tiny_instance_gt/{uid}.npy   per-face instance ids

    NOTE: label 0 here is a *valid* part (the first entry of the label set), which
    is the opposite convention from PartNeXt where 0 means unlabeled.
    """
    semantic_path = os.path.join(root, 'PartObjaverse-Tiny_semantic.json')
    if not os.path.exists(semantic_path):
        raise FileNotFoundError(f'PartObjaverse-Tiny_semantic.json not found under {root}')
    with open(semantic_path, 'r') as f:
        semantic_json = json.load(f)
    labels_by_uid, category_by_uid = flatten_partobjaverse_semantic(semantic_json)

    gt_dirname = {
        'semantic': 'PartObjaverse-Tiny_semantic_gt',
        'instance': 'PartObjaverse-Tiny_instance_gt',
    }[task]
    gt_dir = os.path.join(root, gt_dirname)
    mesh_dir = os.path.join(root, 'PartObjaverse-Tiny_mesh')
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f'{gt_dir} not found (did you unzip {gt_dirname}.zip?)')
    if not os.path.isdir(mesh_dir):
        raise FileNotFoundError(f'{mesh_dir} not found (did you unzip PartObjaverse-Tiny_mesh.zip?)')

    target_ids = set(target_ids) if target_ids else None
    records = {}
    for uid, labels in labels_by_uid.items():
        if target_ids is not None and uid not in target_ids:
            continue
        gt_path = os.path.join(gt_dir, f'{uid}.npy')
        mesh_path = os.path.join(mesh_dir, f'{uid}.glb')
        if not (os.path.exists(gt_path) and os.path.exists(mesh_path)):
            continue
        records[uid] = {
            'model_id': uid,
            'type_id': category_by_uid[uid],
            'gt_path': gt_path,
            'mesh_path': mesh_path,
            'label_names': labels,
            'source': 'partobjaverse',
        }
    return records


def resolve_gt_partobjaverse(instance, record, mesh_dir, num_classes):
    """Return (mesh, gt_labels, num_parts). Labels are used verbatim."""
    mesh = load_mesh(record['mesh_path'])
    gt_labels = np.load(record['gt_path']).astype(np.int32).reshape(-1)
    if len(gt_labels) != len(mesh.faces):
        raise ValueError(
            f'GT/mesh face mismatch for {instance}: gt={len(gt_labels)}, faces={len(mesh.faces)}')
    return mesh, gt_labels, int(len(np.unique(gt_labels)))


def resolve_gt_partnext(instance, record, mesh_dir, num_classes):
    mesh_path = os.path.join(mesh_dir, record['type_id'], f'{instance}.glb')
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f'GT mesh not found: {mesh_path}')
    mesh = load_mesh(mesh_path)
    gt_labels, num_parts = parse_face_labels(
        record['masks'], len(mesh.faces),
        hierarchy_json=record.get('hierarchy'),
        max_parts=num_classes - 1,
    )
    return mesh, gt_labels, int(num_parts)


def normalize_instance_id(path, strip_suffixes):
    name = os.path.splitext(os.path.basename(path))[0]
    for suffix in strip_suffixes:
        if suffix and name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def collect_prediction_files(pred_dir, pred_glob, recursive, strip_suffixes):
    pattern = os.path.join(pred_dir, '**', pred_glob) if recursive else os.path.join(pred_dir, pred_glob)
    files = sorted(glob(pattern, recursive=recursive))
    results = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.endswith('_gt') or stem.endswith('_latent'):
            continue
        instance = normalize_instance_id(path, strip_suffixes)
        if instance:
            results.append((instance, path))
    return results


def recover_face_labels_from_colors(pred_path, num_classes, expected_faces=None):
    mesh = load_mesh(pred_path)
    if expected_faces is not None and len(mesh.faces) != expected_faces:
        raise ValueError(f'face count mismatch: pred={len(mesh.faces)}, gt={expected_faces}')

    colors = None
    if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) == len(mesh.vertices):
        vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32)
        colors = vertex_colors[mesh.faces].mean(axis=1)
    elif hasattr(mesh.visual, 'face_colors') and len(mesh.visual.face_colors) == len(mesh.faces):
        colors = np.asarray(mesh.visual.face_colors)[:, :3].astype(np.float32)

    if colors is None:
        raise ValueError(f'no vertex/face colors found: {pred_path}')

    palette = (generate_class_colors(num_classes) * 255).astype(np.uint8).astype(np.float32)
    dists = ((colors[:, None, :] - palette[None, :, :]) ** 2).sum(axis=-1)
    return dists.argmin(axis=1).astype(np.int32)


def build_adjacency(num_faces, face_adjacency):
    adjacency = [[] for _ in range(num_faces)]
    for left, right in face_adjacency:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    return adjacency


def _confusion(gt, pred, weights):
    """Weighted confusion matrix plus the label values indexing its axes."""
    gt_values = np.unique(gt)
    pred_values = np.unique(pred)
    gt_index = {v: i for i, v in enumerate(gt_values)}
    pred_index = {v: i for i, v in enumerate(pred_values)}
    conf = np.zeros((len(gt_values), len(pred_values)), dtype=np.float64)
    gt_rows = np.array([gt_index[v] for v in gt], dtype=np.int64)
    pred_cols = np.array([pred_index[v] for v in pred], dtype=np.int64)
    np.add.at(conf, (gt_rows, pred_cols), weights)
    return conf, gt_values, pred_values


def _iou_matrix(conf):
    """IoU between every GT part (rows) and predicted part (cols)."""
    gt_area = conf.sum(axis=1, keepdims=True)
    pred_area = conf.sum(axis=0, keepdims=True)
    union = gt_area + pred_area - conf
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0, conf / union, 0.0)
    return iou


def _greedy_pairs(conf):
    """One-to-one matching maximising total overlap (Hungarian when available)."""
    if conf.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if linear_sum_assignment is not None:
        return linear_sum_assignment(-conf)
    pairs = []
    used_rows, used_cols = set(), set()
    flat = [(idx[0], idx[1], conf[idx]) for idx in np.ndindex(conf.shape)]
    for r, c, _ in sorted(flat, key=lambda x: x[2], reverse=True):
        if r not in used_rows and c not in used_cols:
            pairs.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
    return (np.array([p[0] for p in pairs], dtype=np.int64),
            np.array([p[1] for p in pairs], dtype=np.int64))


def _average_precision(scores, is_true_positive, num_gt):
    """VOC-style all-point-interpolated AP."""
    if num_gt == 0:
        return math.nan
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.asarray(is_true_positive, dtype=np.float64)[order]
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # monotone envelope of precision
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    ap = float(np.sum(np.diff(recall) * precision))
    return ap


def _nan_metrics(suffix=''):
    keys = ['face_acc', 'miou_h', 'mean_iou_matched', 'miou_ca', 'miou_ca_pred',
            'ap25', 'ap50', 'ap75', 'prec50', 'rec50', 'f1_50']
    out = {f'{k}{suffix}': math.nan for k in keys}
    out.update({
        f'num_gt_parts{suffix}': 0,
        f'num_pred_parts{suffix}': 0,
        f'part_count_abs_err{suffix}': math.nan,
    })
    return out


def region_metrics(gt_labels, pred_labels, valid_mask, weights=None, suffix='',
                   ap_thresholds=(0.25, 0.5, 0.75)):
    """Label-aware + class-agnostic + instance-level part segmentation metrics.

    `weights` gives a per-face weight (e.g. triangle area). When None every face
    counts as 1, which reproduces plain face-count statistics.
    """
    gt = gt_labels[valid_mask].astype(np.int64)
    pred = pred_labels[valid_mask].astype(np.int64)
    if gt.size == 0:
        return _nan_metrics(suffix)

    if weights is None:
        w = np.ones(gt.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)[valid_mask]
        if w.sum() <= 0:
            w = np.ones(gt.size, dtype=np.float64)

    conf, gt_values, pred_values = _confusion(gt, pred, w)
    iou = _iou_matrix(conf)
    total = conf.sum()

    # ---- Hungarian (label-aware, one-to-one, strict) --------------------
    row_ind, col_ind = _greedy_pairs(conf)
    ious_by_gt = np.zeros(len(gt_values), dtype=np.float64)
    matched_weight = 0.0
    for r, c in zip(row_ind, col_ind):
        ious_by_gt[r] = iou[r, c]
        matched_weight += conf[r, c]
    face_acc = float(matched_weight / total) if total > 0 else math.nan
    miou_h = float(ious_by_gt.mean())

    # ---- class-agnostic mIoU (PartNet / SAMPart3D protocol) -------------
    miou_ca = float(iou.max(axis=1).mean()) if iou.size else math.nan
    miou_ca_pred = float(iou.max(axis=0).mean()) if iou.size else math.nan

    # ---- instance-level AP ---------------------------------------------
    # proxy confidence: predicted part's share of the total weighted area,
    # larger parts are the more confident detections.
    pred_area = conf.sum(axis=0)
    confidences = pred_area / max(total, 1e-12)
    ap_scores = {}
    for thr in ap_thresholds:
        masked = np.where(iou >= thr, iou, 0.0)
        r_ind, c_ind = _greedy_pairs(masked)
        matched_cols = {}
        for r, c in zip(r_ind, c_ind):
            if iou[r, c] >= thr:
                matched_cols[int(c)] = int(r)
        scores, tps = [], []
        for c in range(len(pred_values)):
            scores.append(confidences[c])
            tps.append(1.0 if c in matched_cols else 0.0)
        ap_scores[thr] = _average_precision(scores, tps, len(gt_values))
        if abs(thr - 0.5) < 1e-9:
            num_tp = len(matched_cols)
            precision50 = num_tp / max(len(pred_values), 1)
            recall50 = num_tp / max(len(gt_values), 1)
            f1_50 = 2 * precision50 * recall50 / max(precision50 + recall50, 1e-12)

    out = {
        f'face_acc{suffix}': face_acc,
        f'miou_h{suffix}': miou_h,
        f'mean_iou_matched{suffix}': miou_h,
        f'miou_ca{suffix}': miou_ca,
        f'miou_ca_pred{suffix}': miou_ca_pred,
        f'ap25{suffix}': float(ap_scores.get(0.25, math.nan)),
        f'ap50{suffix}': float(ap_scores.get(0.5, math.nan)),
        f'ap75{suffix}': float(ap_scores.get(0.75, math.nan)),
        f'prec50{suffix}': float(precision50),
        f'rec50{suffix}': float(recall50),
        f'f1_50{suffix}': float(f1_50),
        f'num_gt_parts{suffix}': int(len(gt_values)),
        f'num_pred_parts{suffix}': int(len(pred_values)),
        f'part_count_abs_err{suffix}': int(abs(len(pred_values) - len(gt_values))),
    }
    return out


def clustering_metrics(gt_labels, pred_labels, valid_mask):
    """Permutation-invariant agreement between the two partitions."""
    gt = gt_labels[valid_mask].astype(np.int64)
    pred = pred_labels[valid_mask].astype(np.int64)
    if gt.size == 0 or adjusted_rand_score is None:
        return {'ari': math.nan, 'nmi': math.nan}
    return {
        'ari': float(adjusted_rand_score(gt, pred)),
        'nmi': float(normalized_mutual_info_score(gt, pred)),
    }


def hungarian_metrics(gt_labels, pred_labels, valid_mask):
    """Backwards-compatible alias returning only the face-count metrics."""
    return region_metrics(gt_labels, pred_labels, valid_mask)


def expand_faces(face_set, adjacency, rings):
    if rings <= 0 or not face_set:
        return set(face_set)
    expanded = set(face_set)
    frontier = set(face_set)
    for _ in range(rings):
        nxt = set()
        for face in frontier:
            for neighbor in adjacency[face]:
                if neighbor not in expanded:
                    expanded.add(neighbor)
                    nxt.add(neighbor)
        frontier = nxt
        if not frontier:
            break
    return expanded


def boundary_metrics(gt_labels, pred_labels, valid_mask, face_adjacency, tolerance_rings=1):
    """Boundary agreement over edge-adjacent face pairs.

    NOTE: this is only meaningful when part boundaries actually run through the
    mesh interior. Some benchmarks (e.g. PartObjaverse-Tiny) ship each part as a
    topologically separate component, so no adjacent face pair ever crosses a
    part boundary and the GT boundary set is empty. In that case the metric is
    undefined and NaN is returned rather than a misleading 0.
    """
    adjacency = build_adjacency(len(gt_labels), face_adjacency)
    gt_boundary_faces = set()
    pred_boundary_faces = set()
    valid_edges = 0
    for a, b in face_adjacency:
        a = int(a)
        b = int(b)
        if not (valid_mask[a] and valid_mask[b]):
            continue
        valid_edges += 1
        if gt_labels[a] != gt_labels[b]:
            gt_boundary_faces.add(a)
            gt_boundary_faces.add(b)
        if pred_labels[a] != pred_labels[b]:
            pred_boundary_faces.add(a)
            pred_boundary_faces.add(b)

    undefined = {'boundary_precision': math.nan, 'boundary_recall': math.nan,
                 'boundary_f1': math.nan}
    if valid_edges == 0 or not gt_boundary_faces:
        return undefined

    gt_for_precision = expand_faces(gt_boundary_faces, adjacency, tolerance_rings)
    pred_for_recall = expand_faces(pred_boundary_faces, adjacency, tolerance_rings)

    if not pred_boundary_faces:
        return {'boundary_precision': math.nan, 'boundary_recall': 0.0, 'boundary_f1': 0.0}

    precision = len(pred_boundary_faces & gt_for_precision) / len(pred_boundary_faces)
    recall = len(gt_boundary_faces & pred_for_recall) / len(gt_boundary_faces)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        'boundary_precision': float(precision),
        'boundary_recall': float(recall),
        'boundary_f1': float(f1),
    }


def component_metrics(pred_labels, valid_mask, face_adjacency, small_component_faces=20):
    adjacency = build_adjacency(len(pred_labels), face_adjacency)
    visited = np.zeros(len(pred_labels), dtype=bool)
    components_by_label = defaultdict(int)
    face_count_by_label = defaultdict(int)
    largest_by_label = defaultdict(int)
    small_faces = 0
    num_components = 0

    for start in range(len(pred_labels)):
        if visited[start] or not valid_mask[start]:
            continue
        label = int(pred_labels[start])
        queue = deque([start])
        visited[start] = True
        size = 0
        while queue:
            face = queue.popleft()
            size += 1
            for neighbor in adjacency[face]:
                if visited[neighbor] or not valid_mask[neighbor]:
                    continue
                if int(pred_labels[neighbor]) == label:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components_by_label[label] += 1
        face_count_by_label[label] += size
        largest_by_label[label] = max(largest_by_label[label], size)
        num_components += 1
        if size < small_component_faces:
            small_faces += size

    valid_faces = int(valid_mask.sum())
    labels = list(components_by_label.keys())
    components_per_label = float(np.mean([components_by_label[l] for l in labels])) if labels else math.nan
    largest_ratios = [largest_by_label[l] / max(face_count_by_label[l], 1) for l in labels]
    mean_largest_component_ratio = float(np.mean(largest_ratios)) if largest_ratios else math.nan

    return {
        'num_pred_components': int(num_components),
        'components_per_label': components_per_label,
        'small_component_ratio': float(small_faces / max(valid_faces, 1)),
        'mean_largest_component_ratio': mean_largest_component_ratio,
    }


def evaluate_one(instance, pred_path, record, mesh_dir, num_classes, include_gt_zero,
                 boundary_tolerance_rings, small_component_faces):
    if record.get('source') == 'partobjaverse':
        mesh, gt_labels, num_parts = resolve_gt_partobjaverse(instance, record, mesh_dir, num_classes)
    else:
        mesh, gt_labels, num_parts = resolve_gt_partnext(instance, record, mesh_dir, num_classes)

    pred_labels = recover_face_labels_from_colors(pred_path, num_classes, expected_faces=len(mesh.faces))
    valid_mask = gt_labels >= 0 if include_gt_zero else gt_labels > 0

    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)

    metrics = {
        'instance': instance,
        'type_id': record['type_id'],
        'pred_file': pred_path,
        'num_faces': int(len(mesh.faces)),
        'valid_faces': int(valid_mask.sum()),
        'gt_num_parts_raw': int(num_parts),
    }
    metrics.update(region_metrics(gt_labels, pred_labels, valid_mask))
    metrics.update(region_metrics(gt_labels, pred_labels, valid_mask,
                                  weights=face_areas, suffix='_area'))
    metrics.update(clustering_metrics(gt_labels, pred_labels, valid_mask))
    metrics.update(boundary_metrics(gt_labels, pred_labels, valid_mask, mesh.face_adjacency, boundary_tolerance_rings))
    metrics.update(component_metrics(pred_labels, valid_mask, mesh.face_adjacency, small_component_faces))
    return metrics


KEY_METRICS = [
    # --- comparable to published SAMPart3D / PartSLIP / PartNet tables ---
    'miou_ca', 'miou_ca_area', 'ap50', 'ap50_area',
    # --- stricter internal diagnostics ---
    'miou_h', 'miou_h_area', 'face_acc', 'face_acc_area',
    'miou_ca_pred', 'ap25', 'ap75',
    'prec50', 'rec50', 'f1_50',
    'ari', 'nmi',
    'boundary_f1', 'boundary_precision', 'boundary_recall',
    'num_gt_parts', 'num_pred_parts', 'part_count_abs_err',
    'small_component_ratio', 'mean_largest_component_ratio', 'components_per_label',
]


def summarize(rows):
    numeric_keys = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {'num_faces', 'valid_faces'}:
                numeric_keys.append(key)
    numeric_keys = sorted(set(numeric_keys))

    summary = {'num_samples': len(rows)}
    for key in numeric_keys:
        values = np.array([row.get(key, np.nan) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            summary[f'{key}_mean'] = float(values.mean())
            summary[f'{key}_std'] = float(values.std())
            summary[f'{key}_median'] = float(np.median(values))
    return summary


def summarize_by_group(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(group_key, 'unknown')].append(row)
    return {str(name): summarize(items) for name, items in sorted(groups.items())}


def format_table(summary, by_parts=None):
    lines = []
    lines.append(f"Samples evaluated: {summary.get('num_samples', 0)}")
    lines.append('')
    lines.append(f"{'Metric':<32}{'Mean':>10}{'Std':>10}{'Median':>10}")
    lines.append('-' * 62)
    for key in KEY_METRICS:
        mean = summary.get(f'{key}_mean')
        if mean is None:
            continue
        std = summary.get(f'{key}_std', float('nan'))
        med = summary.get(f'{key}_median', float('nan'))
        lines.append(f'{key:<32}{mean:>10.4f}{std:>10.4f}{med:>10.4f}')

    if by_parts:
        lines.append('')
        lines.append('Breakdown by GT part count')
        lines.append(f"{'#GT parts':<12}{'N':>6}{'mIoU_CA':>10}{'AP50':>10}{'mIoU_H':>10}{'ARI':>10}")
        lines.append('-' * 58)
        for name, sub in by_parts.items():
            lines.append(
                f"{name:<12}{sub.get('num_samples', 0):>6}"
                f"{sub.get('miou_ca_mean', float('nan')):>10.4f}"
                f"{sub.get('ap50_mean', float('nan')):>10.4f}"
                f"{sub.get('miou_h_mean', float('nan')):>10.4f}"
                f"{sub.get('ari_mean', float('nan')):>10.4f}"
            )
    return '\n'.join(lines)


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trellis SegPart face-level quality')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predicted colored GLB files')
    parser.add_argument('--output_dir', type=str, default='./eval/seg_quality')
    parser.add_argument('--dataset', type=str, default='partnext',
                        choices=['partnext', 'partobjaverse'],
                        help='GT source. partobjaverse = PartObjaverse-Tiny (SAMPart3D benchmark)')
    parser.add_argument('--partobjaverse_root', type=str, default=DEFAULT_PARTOBJAVERSE_ROOT,
                        help='Root of the unzipped PartObjaverse-Tiny dataset')
    parser.add_argument('--partobjaverse_task', type=str, default='semantic',
                        choices=['semantic', 'instance'],
                        help='Which GT to evaluate against: semantic_gt or instance_gt')
    parser.add_argument('--arrow_dir', type=str, default=DEFAULT_ARROW_DIR)
    parser.add_argument('--mesh_dir', type=str, default=DEFAULT_MESH_DIR)
    parser.add_argument('--pred_glob', type=str, default='*.glb')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--instances', type=str, default=None,
                        help='Comma-separated model ids or a text file with one model id per line')
    parser.add_argument('--strip_suffixes', type=str, default='_pred,_result,_e2e,_colored',
                        help='Comma-separated filename suffixes stripped to recover model_id')
    parser.add_argument('--include_gt_zero', action='store_true', default=None,
                        help='Include GT label 0. Default: excluded for partnext (0=unlabeled), '
                             'included for partobjaverse (0 is a valid part)')
    parser.add_argument('--boundary_tolerance_rings', type=int, default=1)
    parser.add_argument('--small_component_faces', type=int, default=20)
    parser.add_argument('--max_arrow_files', type=int, default=None)
    args = parser.parse_args()

    # Label-0 convention differs between the two benchmarks: PartNeXt uses 0 for
    # unlabeled faces, PartObjaverse-Tiny uses 0 as the first real part.
    if args.include_gt_zero is None:
        args.include_gt_zero = (args.dataset == 'partobjaverse')
        print(f'[INFO] include_gt_zero defaulted to {args.include_gt_zero} for dataset={args.dataset}')

    strip_suffixes = [s for s in args.strip_suffixes.split(',') if s]
    pred_items = collect_prediction_files(args.pred_dir, args.pred_glob, args.recursive, strip_suffixes)

    if args.instances:
        if os.path.exists(args.instances):
            with open(args.instances, 'r') as f:
                wanted = {line.strip() for line in f if line.strip()}
        else:
            wanted = {x.strip() for x in args.instances.split(',') if x.strip()}
        pred_items = [(inst, path) for inst, path in pred_items if inst in wanted]

    if args.max_samples is not None:
        pred_items = pred_items[:args.max_samples]
    if not pred_items:
        raise RuntimeError(f'No prediction files found in {args.pred_dir}')

    target_ids = [inst for inst, _ in pred_items]
    if args.dataset == 'partobjaverse':
        records = load_partobjaverse_records(
            args.partobjaverse_root, task=args.partobjaverse_task, target_ids=target_ids)
    else:
        records = load_partnext_records(
            args.arrow_dir, target_ids=target_ids, max_arrow_files=args.max_arrow_files)

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    failures = []
    for instance, pred_path in tqdm(pred_items, desc='Evaluating'):
        try:
            if instance not in records:
                raise KeyError(f'annotation not found for {instance}')
            row = evaluate_one(
                instance, pred_path, records[instance], args.mesh_dir, args.num_classes,
                args.include_gt_zero, args.boundary_tolerance_rings, args.small_component_faces,
            )
            rows.append(row)
        except Exception as exc:
            failures.append({'instance': instance, 'pred_file': pred_path, 'error': str(exc)})
            print(f'[WARN] {instance}: {exc}')

    summary = summarize(rows)
    for row in rows:
        row['gt_parts_bucket'] = str(row.get('num_gt_parts', 'na'))
    by_parts = summarize_by_group(rows, 'gt_parts_bucket') if rows else {}
    by_type = summarize_by_group(rows, 'type_id') if rows else {}
    table = format_table(summary, by_parts)
    output = {
        'summary': summary,
        'by_gt_part_count': by_parts,
        'by_type_id': by_type,
        'samples': rows,
        'failures': failures,
    }

    json_path = os.path.join(args.output_dir, 'metrics.json')
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    summary_path = os.path.join(args.output_dir, 'summary.txt')

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    write_csv(csv_path, rows)
    with open(summary_path, 'w') as f:
        f.write(table + '\n')
        f.write('\nFull summary (JSON)\n')
        f.write(json.dumps(summary, indent=2))
        if failures:
            f.write('\n\nFailures:\n')
            for item in failures:
                f.write(f"{item['instance']}: {item['error']}\n")

    print('\n' + table)
    print(f'\nWrote: {json_path}')
    print(f'Wrote: {csv_path}')
    print(f'Wrote: {summary_path}')
    if failures:
        print(f'Failures: {len(failures)}')


if __name__ == '__main__':
    main()
