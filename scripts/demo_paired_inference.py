"""Paired, geometry-only legacy20 / released64 demo inference.

Public API: Runner(plan, device).run(row), or run_pair(plan, row, device).
CLI: python scripts/demo_paired_inference.py --plan PLAN.json --worker 0
Workers are launched externally; this module never creates workers or shape caches.

The JSON plan supplies absolute legacy_config, legacy_vae_config, legacy_encoder,
legacy_decoder, legacy_dit, new_vae, new_dit, output_dir and prediction_dir paths;
new_model_args and new_normalization; samples and workers. Defaults are seed=42,
steps=20 and max_tokens=16384. Each sample has id, geometry_path, condition_path,
shape_path, title, kind and source_name, with optional archive_path/provenance.
Geometry is the full build_geometry result plus face_raster (-1 background),
without targets. Conditions contain render_maps, integer (6,512,512), 0 background,
1 unknown mesh and 2..20 part IDs plus one. Shape NPZ contains native coords/feats
(features is also accepted). Arrays and original face order are never resampled.

Outputs: objects/ID/{old.glb,new.glb,inputcondition.png,comparison.png},
metrics/ID.json, and prediction_dir/ID.npz. Metrics compare projected predictions
only with the supplied input condition, using its original IDs; they are not GT
metrics. source_geometry_hash is the SHA256 of the complete geometry NPZ file.
All hashes, including initial Gaussian bytes, are recorded in the metrics.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent / "TRELLIS.2") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "TRELLIS.2"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dit_vae_v2 import align_shape, file_sha256, identity
from vae_v2.config import atomic_json, digest, read_config

VIEWS = ("front", "right", "back", "left", "top", "bottom")
MODEL_PATHS = ("legacy_config", "legacy_vae_config", "legacy_encoder",
               "legacy_decoder", "legacy_dit", "new_vae", "new_dit")
GEOMETRY_KEYS = ("coords", "anchor_points", "anchor_faces", "anchor_voxel",
                 "face_normals", "face_area", "face_components", "adjacency",
                 "mesh_vertices", "mesh_faces", "center", "extent", "face_raster")


def array_hash(array):
    """SHA256 of contiguous array bytes, without a dtype/shape prefix."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _integer_array(value, name, ndim, width=None, low=0, high=None):
    a = np.asarray(value)
    if (a.ndim != ndim or not np.issubdtype(a.dtype, np.integer)
            or (width is not None and a.shape[-1] != width)
            or (a.size and (a.min() < low or (high is not None and a.max() > high)))):
        raise ValueError(f"Invalid {name}")
    return a


def _stats(stats):
    result = {}
    for key in ("latent_feats", "shape_feats", "mv_feats"):
        value = stats[key]
        mean, std = np.asarray(value["mean"]), np.asarray(value["std"])
        if (mean.shape != (32,) or std.shape != (32,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError(f"Invalid normalization for {key}")
        result[key] = {"mean": mean.tolist(), "std": std.tolist()}
    return result


def legacy_normalization(config):
    args = config["dataset"]["args"]
    return _stats({key: args[source] for key, source in (
        ("latent_feats", "normalization"), ("shape_feats", "shape_cond_normalization"),
        ("mv_feats", "mv_cond_normalization"))})


def _model_args(args):
    return dict(args, dtype="float32", use_checkpoint=False)


def validate_plan(plan):
    for key in (*MODEL_PATHS, "output_dir", "prediction_dir"):
        path = Path(plan[key])
        if not path.is_absolute():
            raise ValueError(f"{key} must be an absolute path")
        if key in MODEL_PATHS and not path.is_file():
            raise ValueError(f"Missing {key}: {path}")
    for key, default in (("seed", 42), ("steps", 20), ("max_tokens", 16384), ("workers", 1)):
        value = plan.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid {key}")
    if plan.get("seed", 42) != 42 or plan.get("steps", 20) != 20:
        raise ValueError("The paired demo requires seed=42 and steps=20")
    if plan.get("max_tokens", 16384) > 16384:
        raise ValueError("max_tokens must not exceed 16384")
    for key in ("postprocessing", "label_matching", "prior_matching", "smoothing"):
        if plan.get(key, False):
            raise ValueError(f"{key} is not supported")
    _stats(plan["new_normalization"])
    if not isinstance(plan["new_model_args"], dict):
        raise ValueError("new_model_args must be a dictionary")
    ids = [row["id"] for row in plan["samples"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    for uid in ids:
        if not isinstance(uid, str) or not uid or uid in (".", "..") or "/" in uid or "\\" in uid:
            raise ValueError("Sample IDs must be single path components")


def projection_agreement(face_labels, face_raster, render_maps):
    """CPU exact-ID agreement on (render_maps > 1) & (face_raster >= 0).

    mIoU averages the nonzero IDs present in the scored input condition, matching
    the repository's classification convention. Predicting unknown or an absent
    ID is an error, never a reason to drop a pixel or perform label matching.
    """
    labels = _integer_array(face_labels, "face labels", 1, high=63)
    raster = _integer_array(face_raster, "face raster", np.ndim(render_maps),
                            low=-1, high=len(labels)-1)
    maps = _integer_array(render_maps, "render maps", raster.ndim, high=64)
    if raster.shape != maps.shape:
        raise ValueError("Raster and condition shapes differ")
    valid = (maps > 1) & (raster >= 0)
    expected = maps[valid].astype(np.int64) - 1
    predicted = labels[raster[valid]]
    part_iou = {}
    for label in np.unique(expected):
        a, b = expected == label, predicted == label
        part_iou[str(int(label))] = float(np.count_nonzero(a & b) / np.count_nonzero(a | b))
    accuracy = float(np.mean(expected == predicted)) if len(expected) else None
    return {"pixels": int(valid.sum()), "pixel_accuracy": accuracy, "accuracy": accuracy,
            "miou": float(np.mean(list(part_iou.values()))) if part_iou else None,
            "part_iou": part_iou, "condition_ids": np.unique(expected).tolist(),
            "prediction_ids": np.unique(predicted).tolist(),
            "miou_policy": "IDs present in valid input-condition pixels",
            "condition_pixels_without_face": int(((maps > 1) & (raster < 0)).sum())}


def load_inputs(row):
    """Load only supplied caches; never reconstruct geometry from an archive."""
    sources = {}
    for key in ("geometry_path", "condition_path", "shape_path"):
        path = Path(row[key])
        if not path.is_absolute():
            raise ValueError(f"{key} must be absolute")
        before = identity(path)
        sources[key] = dict(before, sha256=file_sha256(path))
        if identity(path) != before:
            raise ValueError(f"{key} changed while hashing")
    if row.get("geometry_sha256", sources["geometry_path"]["sha256"]) != sources["geometry_path"]["sha256"]:
        raise ValueError("Geometry file hash differs from the plan")
    with np.load(row["geometry_path"], allow_pickle=False) as data:
        forbidden = {"face_labels", "voxel_target", "voxel_labels", "latent_feats", "gt_labels"} & set(data.files)
        if forbidden:
            raise ValueError(f"Geometry cache contains supervision: {sorted(forbidden)}")
        geometry = {key: data[key] for key in GEOMETRY_KEYS}
        for key in ("topology_repair", "legacy_component_count"):
            if key in data:
                geometry[key] = data[key]
        if "metadata" in data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("format_version", 2) != 2:
                raise ValueError("Expected geometry format version 2")
    with np.load(row["condition_path"], allow_pickle=False) as data:
        maps = data["render_maps"]
    _integer_array(maps, "render_maps", 3, high=20)
    if maps.shape != (6, 512, 512) or not (maps > 0).any():
        raise ValueError("Expected nonempty six-view 512x512 legacy20 render_maps")
    vertices = np.asarray(geometry["mesh_vertices"])
    faces = _integer_array(geometry["mesh_faces"], "mesh faces", 2, 3, high=len(vertices)-1)
    coords = _integer_array(geometry["coords"], "geometry coords", 2, 3, high=511)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(faces) or not len(coords)
            or not np.isfinite(vertices).all() or len(np.unique(coords, axis=0)) != len(coords)):
        raise ValueError("Invalid or empty geometry")
    extent, center = float(geometry["extent"]), geometry["center"]
    if not np.isfinite(extent) or extent <= 0 or center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Invalid geometry center/extent")
    if not np.isclose(np.ptp(vertices, axis=0).max(), extent, rtol=1e-5):
        raise ValueError("Geometry extent does not match original vertices")
    raster = _integer_array(geometry["face_raster"], "face_raster", 3, low=-1, high=len(faces)-1)
    if raster.shape != maps.shape:
        raise ValueError("Condition/raster shape mismatch")
    af = _integer_array(geometry["anchor_faces"], "anchor_faces", 1, high=len(faces)-1)
    av = _integer_array(geometry["anchor_voxel"], "anchor_voxel", 1, high=len(coords)-1)
    components = _integer_array(geometry["face_components"], "face_components", 1)
    if (not len(af) or len(af) != len(av) or components.shape != (len(faces),)
            or not np.array_equal(np.unique(components), np.unique(components[af]))):
        raise ValueError("Invalid anchor/component correspondence")
    for key, shape in (("anchor_points", (len(af), 3)), ("face_normals", (len(faces), 3)),
                       ("face_area", (len(faces),))):
        if geometry[key].shape != shape or not np.isfinite(geometry[key]).all():
            raise ValueError(f"Invalid {key}")
    with np.load(row["shape_path"], allow_pickle=False) as data:
        shape_coords = data["coords"]
        shape_feats = data["feats"] if "feats" in data else data["features"]
    for key, source in sources.items():
        if identity(row[key]) != {k: source[k] for k in ("path", "bytes", "mtime_ns")}:
            raise ValueError(f"{key} changed while loading")
    maps.setflags(write=False)
    return geometry, maps, shape_coords, shape_feats, sources


def encode_views(maps, encoder, num_classes, device):
    import torch
    from preprocess.step2_encode_mv_seg_latent import encode_one_view

    coords, feats, offsets = [], [], [0]
    with torch.no_grad(), torch.autocast(torch.device(device).type, enabled=False):
        for view in range(6):
            c, f = encode_one_view(maps[view], view, encoder, num_classes, device)
            if c is not None:
                coords.append(c.numpy().astype(np.int32))
                feats.append(f.float().numpy())
                offsets.append(offsets[-1] + len(c))
            else:
                offsets.append(offsets[-1])
    if not feats:
        raise ValueError("No multi-view condition tokens")
    # Keep view order and duplicate positions across views, as in preprocessing.
    result = {"mv_coords": np.concatenate(coords), "mv_feats": np.concatenate(feats),
              "view_offsets": np.asarray(offsets, np.int32)}
    _integer_array(result["mv_coords"], "mv coords", 2, 3, high=31)
    if result["mv_feats"].shape != (len(result["mv_coords"]), 32) or not np.isfinite(result["mv_feats"]).all():
        raise ValueError("Invalid encoded condition features")
    return result


def normalize_conditions(shape_feats, shape_valid, mv_feats, stats, device):
    import torch

    result = {}
    for key, values in (("shape_feats", shape_feats), ("mv_feats", mv_feats)):
        x = torch.as_tensor(values, dtype=torch.float32, device=device)
        mean = torch.tensor(stats[key]["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(stats[key]["std"], dtype=torch.float32, device=device)
        x = (x - mean) / std
        if key == "shape_feats":
            x = x.masked_fill(~torch.as_tensor(shape_valid, dtype=torch.bool, device=device)[:, None], 0)
        result[key] = x
    return result


def paired_noise(count, seed, device):
    import torch

    values = [torch.randn((count, 32), dtype=torch.float32, device=device,
                          generator=torch.Generator(device=device).manual_seed(seed)) for _ in range(2)]
    hashes = [array_hash(x.cpu().numpy()) for x in values]
    if hashes[0] != hashes[1] or not torch.equal(*values):
        raise ValueError("Paired initial Gaussian bytes differ")
    return values[0], values[1], {"old": hashes[0], "new": hashes[1], "identical": True,
                                  "dtype": "float32", "shape": [count, 32]}


def generate_latent(model, template, initial_noise, shape_feats, shape_valid, mv, stats, steps, device):
    import torch

    normalized = normalize_conditions(shape_feats, shape_valid, mv["mv_feats"], stats, device)
    cond = {"cond": torch.zeros(1, 1029, model.cond_channels, dtype=torch.float32, device=device),
            "concat_cond": template.replace(normalized["shape_feats"]),
            "mv_cond": [normalized["mv_feats"]],
            "mv_coords": [torch.as_tensor(mv["mv_coords"], dtype=torch.float32, device=device)]}
    x = initial_noise.clone()
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([(1 - i / steps) * 1000], dtype=torch.float32, device=device)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                                enabled=torch.device(device).type == "cuda"):
                velocity = model(template.replace(x), t, **cond).feats.float()
            if velocity.shape != x.shape or not torch.isfinite(velocity).all():
                raise FloatingPointError("Invalid DiT velocity")
            x = x - velocity / steps
    if not torch.isfinite(x).all():
        raise FloatingPointError("Nonfinite generated latent")
    return x * torch.tensor(stats["latent_feats"]["std"], device=device) + torch.tensor(
        stats["latent_feats"]["mean"], device=device)


def check_templates(old, new, voxel_coords, max_tokens):
    import torch

    if not torch.equal(old.coords, new.coords):
        raise ValueError("Old/new target latent coordinate order differs")
    coords = old.coords[:, 1:].cpu().numpy()
    _integer_array(coords, "latent coords", 2, 3, high=31)
    if not len(coords) or len(coords) > max_tokens:
        raise ValueError("Geometry exceeds latent token budget; no truncation is allowed")
    if not np.array_equal(np.unique(voxel_coords // 16, axis=0), np.unique(coords, axis=0)):
        raise ValueError("Geometry/latent downsampling correspondence differs")
    if len(np.unique(coords, axis=0)) != len(coords) or (old.coords[:, 0] != 0).any():
        raise ValueError("Duplicate or batched target coordinates")
    if old.feats.shape != (len(coords), 32) or new.feats.shape != old.feats.shape:
        raise ValueError("Expected matching latent32 templates")
    return coords


def bake_legacy_faces(geometry, voxel_coords, logits, chunk_faces=1024):
    """Eight nearest voxel logits, inverse-distance averaged at face centroids."""
    from scipy.spatial import cKDTree

    coords, scores = np.asarray(voxel_coords), np.asarray(logits)
    if scores.shape != (len(coords), 20) or not len(coords) or not np.isfinite(scores).all():
        raise ValueError("Invalid legacy decoder logits")
    tree = cKDTree(coords)
    faces, vertices = geometry["mesh_faces"], geometry["mesh_vertices"]
    labels = np.empty(len(faces), np.int32)
    for start in range(0, len(faces), chunk_faces):
        stop = min(start + chunk_faces, len(faces))
        centroids = vertices[faces[start:stop]].mean(1)
        query = ((centroids - geometry["center"]) / float(geometry["extent"]) * 509 + 256)
        distance, neighbors = tree.query(query, k=min(8, len(coords)))
        if neighbors.ndim == 1:
            distance, neighbors = distance[:, None], neighbors[:, None]
        weights = 1 / np.maximum(distance, 1e-8)
        weights /= weights.sum(1, keepdims=True)
        labels[start:stop] = (scores[neighbors] * weights[..., None]).sum(1).argmax(1)
    return labels


def decode_new_faces(vae, template, raw, geometry, device):
    import torch
    from vae_v2.data import to_device
    from vae_v2.geometry import SurfaceQueries

    with torch.no_grad():
        features, logits, coords = vae.decode_voxels(template.replace(raw))
        if not np.array_equal(coords[:, 1:].cpu().numpy(), geometry["coords"]):
            raise ValueError("New decoder order differs from geometry anchor_voxel indices")
        if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite new decoder output")
        builder = SurfaceQueries(geometry, vae.cfg["geometry"])
        labels = np.empty(len(geometry["mesh_faces"]), np.int32)
        fallback = 0
        for start in range(0, len(labels), 1024):
            ids = np.arange(start, min(start + 1024, len(labels)))
            query = builder.build(ids)
            if (not np.array_equal(query["face_ids"], ids)
                    or not np.array_equal(query["query_face_ids"], np.repeat(ids, 4))):
                raise ValueError("Surface queries changed original face order or four-point sampling")
            fallback += int(query["query_fallback"].sum())
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                                enabled=torch.device(device).type == "cuda"):
                _, scores = vae.decode_surface(features, logits, to_device(query, device))
            if scores.shape != (len(ids), 64) or not torch.isfinite(scores).all():
                raise FloatingPointError("Invalid new surface scores")
            labels[ids] = scores.argmax(1).cpu().numpy()
    return labels, fallback / (4 * len(labels))


def export_pair(out, geometry, maps, old_faces, new_faces):
    import trimesh
    from PIL import Image, ImageDraw
    from vae_v2.visuals import export_labels, render_palette

    out.mkdir(parents=True, exist_ok=True)
    center, extent = geometry["center"], float(geometry["extent"])
    mesh = trimesh.Trimesh(vertices=(geometry["mesh_vertices"] - center) / extent,
                           faces=geometry["mesh_faces"], process=False)
    for name, labels in (("old", old_faces), ("new", new_faces)):
        export_labels(mesh, labels, out / f"{name}.glb", 64)
    palette = render_palette(64)
    condition = Image.new("RGB", (1536, 1024), "white")
    comparison = Image.new("RGB", (1536, 840), "white")
    draw = ImageDraw.Draw(comparison)
    raster = geometry["face_raster"]
    hit = raster >= 0
    for row, (name, labels) in enumerate((("Input condition", None), ("Legacy20", old_faces), ("Released64", new_faces))):
        rendered = maps
        if labels is not None:
            rendered = np.zeros(raster.shape, np.int32)
            rendered[hit] = labels[raster[hit]] + 1
        for view, view_name in enumerate(VIEWS):
            im = Image.fromarray(palette[rendered[view]])
            if labels is None:
                condition.paste(im, ((view % 3) * 512, (view // 3) * 512))
            draw.text((view * 256 + 4, row * 280 + 4), f"{name}: {view_name}", fill="black")
            comparison.paste(im.resize((256, 256), Image.Resampling.NEAREST), (view * 256, row * 280 + 24))
    condition.save(out / "inputcondition.png")
    comparison.save(out / "comparison.png")
    matrix = np.eye(4)
    matrix[:3, :3] /= extent
    matrix[:3, 3] = -center / extent
    return {"formula": "web_vertices = (source_vertices - center) / extent",
            "center": center.tolist(), "extent": extent, "source_to_web": matrix.tolist(),
            "web_max_extent": 1.0, "face_order": "original geometry NPZ order", "palette_classes": 64}


class Runner:
    """Load one frozen model pair per external worker; run its rows sequentially."""

    def __init__(self, plan, device="cuda:0"):
        import torch

        validate_plan(plan)
        self.plan = copy.deepcopy(plan)
        self.device = torch.device(device)
        self.plan_digest = digest(plan)
        self.seed, self.steps = plan.get("seed", 42), plan.get("steps", 20)
        self.max_tokens = plan.get("max_tokens", 16384)
        legacy = read_config(plan["legacy_config"])
        self.old_stats = legacy_normalization(legacy)
        self.new_stats = _stats(plan["new_normalization"])
        old_args = _model_args(legacy["models"]["denoiser"]["args"])
        new_args = _model_args(plan["new_model_args"])
        if old_args != new_args:
            raise ValueError("Old/new DiT architectures differ")
        if any(old_args[k] != v for k, v in (("resolution", 32), ("in_channels", 64), ("out_channels", 32))):
            raise ValueError("Expected SegSLatFlowModel at resolution32 with 64->32 channels")
        self.hashes = {}
        for key in MODEL_PATHS:
            print(f"paired: hashing {key}", flush=True)
            before = identity(plan[key])
            self.hashes[key] = dict(before, sha256=file_sha256(plan[key]))
            if identity(plan[key]) != before:
                raise ValueError(f"{key} changed while hashing")
        print("paired: loading FP32 VAE/DiT pair", flush=True)
        # Runtime imports deliberately happen only after validating the plan.
        from vae_v2.model import SegVaeV2
        from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
        from models.seg_flow_model import SegSLatFlowModel

        vae_settings = read_config(plan["legacy_vae_config"])["models"]
        for name, cls, channel_key in (("encoder", SparseUnetVaeEncoder, "in_channels"),
                                       ("decoder", SparseUnetVaeDecoder, "out_channels")):
            print(f"paired: loading legacy {name}", flush=True)
            args = dict(vae_settings[name]["args"], use_fp16=False)
            if args[channel_key] != 20 or args["latent_channels"] != 32:
                raise ValueError("Expected legacy20 VAE with latent32")
            module = cls(**args)
            module.load_state_dict(self._load("legacy_" + name), strict=True)
            setattr(self, "old_" + name, self._freeze(module))
        print("paired: loading new VAE", flush=True)
        state = self._load("new_vae")
        cfg = copy.deepcopy(state["config"])
        if cfg.get("num_classes") != 64 or cfg["model"]["latent_channels"] != 32 or cfg["model"]["mode"] != "surface":
            raise ValueError("Expected released64 VAE with surface head and latent32")
        # Use the supplied backbone file, not historical machine paths in weights.
        # SegVaeV2 overrides its input/output channel counts from num_classes.
        cfg["vae_config"] = plan["legacy_vae_config"]
        cfg["model"].update(initialize_backbone=False, gradient_checkpointing=False)
        vae = SegVaeV2(cfg, initialize=False)
        vae.load_state_dict(state["model"], strict=True)
        self.new_vae = self._freeze(vae)
        del state, vae, module
        for name, key, args in (("old_dit", "legacy_dit", old_args), ("new_dit", "new_dit", new_args)):
            print(f"paired: loading {name}", flush=True)
            state = self._load(key)
            if key == "new_dit":
                expected = state.get("metadata", {}).get("vae_sha256")
                if expected and expected != self.hashes["new_vae"]["sha256"]:
                    raise ValueError("New DiT checkpoint was paired with another VAE")
                weights = state["model"]
            else:
                weights = state
            model = SegSLatFlowModel(**args)
            model.load_state_dict(weights, strict=True)
            model = self._freeze(model)
            model.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            setattr(self, name, model)
            del state, weights, model
        self.runtime = {"dit_model": "SegSLatFlowModel", "dit_args_digest": digest(old_args),
                        "vae_weights": "float32", "dit_weights": "float32", "vae_compute": "float32",
                        "dit_autocast": "bfloat16" if self.device.type == "cuda" else None,
                        "old_normalization_digest": digest(self.old_stats),
                        "new_normalization_digest": digest(self.new_stats)}
        print("paired: model pair ready", flush=True)

    def _load(self, key):
        import torch

        state = torch.load(self.plan[key], map_location="cpu", weights_only=True, mmap=True)
        expected = {k: self.hashes[key][k] for k in ("path", "bytes", "mtime_ns")}
        if identity(self.plan[key]) != expected:
            raise ValueError(f"{key} changed after hashing")
        if state.get("smoke_only", False) or state.get("metadata", {}).get("smoke_only", False):
            raise ValueError("Smoke checkpoints are not eligible for paired demos")
        return state

    def _freeze(self, model):
        return model.float().to(self.device).eval().requires_grad_(False)

    def _old_template(self, coords):
        import torch
        from trellis2.modules.sparse.basic import SparseTensor

        coords = torch.as_tensor(coords, dtype=torch.int32, device=self.device)
        features = torch.zeros(len(coords), 20, device=self.device)
        features[:, 0] = 1
        batch_coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1)
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            template = self.old_encoder(SparseTensor(feats=features, coords=batch_coords), sample_posterior=False)
        # Retain down/up-sampling correspondence, discard channel-specific conv maps.
        template._spatial_cache = {
            scale: {k: v for k, v in cache.items() if not k.startswith("SubMConv3d_neighbor_cache_")}
            for scale, cache in template._spatial_cache.items()}
        return template

    def run(self, row):
        import torch

        expected = next((r for r in self.plan["samples"] if r["id"] == row["id"]), None)
        if expected != row:
            raise ValueError("Row does not match the frozen plan")
        start = time.monotonic()
        geometry, maps, shape_coords, shape_feats, sources = load_inputs(row)
        print(f"paired: {row['id']} {len(geometry['mesh_faces'])} faces", flush=True)
        with torch.no_grad():
            print(f"paired: {row['id']} geometry templates", flush=True)
            old_template = self._old_template(geometry["coords"])
            new_template = self.new_vae.geometry_template(torch.as_tensor(geometry["coords"], device=self.device))
            coords = check_templates(old_template, new_template, geometry["coords"], self.max_tokens)
            aligned, valid = align_shape(coords, shape_coords, shape_feats)
            if not valid.any():
                raise ValueError("Native shape cache has no exact target-coordinate matches")
            print(f"paired: {row['id']} six-view conditions", flush=True)
            old_mv = encode_views(maps, self.old_encoder, 20, self.device)
            new_mv = encode_views(maps, self.new_vae.encoder, 64, self.device)
            if (not np.array_equal(old_mv["mv_coords"], new_mv["mv_coords"])
                    or not np.array_equal(old_mv["view_offsets"], new_mv["view_offsets"])):
                raise ValueError("Old/new encoded condition coordinates differ")
            old_noise, new_noise, noise_hashes = paired_noise(len(coords), self.seed, self.device)
            print(f"paired: {row['id']} old DiT, {len(coords)} tokens, identical noise verified", flush=True)
            raw = generate_latent(self.old_dit, old_template, old_noise, aligned, valid,
                                  old_mv, self.old_stats, self.steps, self.device)
            with torch.autocast(self.device.type, enabled=False):
                decoded = self.old_decoder(old_template.replace(raw.float()))
            decoded_coords = decoded.coords[:, 1:].cpu().numpy()
            if not np.array_equal(decoded_coords, geometry["coords"]):
                raise ValueError("Legacy decoder coordinate order differs from geometry")
            old_faces = bake_legacy_faces(geometry, decoded_coords, decoded.feats.float().cpu().numpy())
            del raw, decoded, old_template, old_mv, old_noise
            print(f"paired: {row['id']} new DiT", flush=True)
            raw = generate_latent(self.new_dit, new_template, new_noise, aligned, valid,
                                  new_mv, self.new_stats, self.steps, self.device)
            print(f"paired: {row['id']} new surface decode", flush=True)
            new_faces, fallback = decode_new_faces(self.new_vae, new_template, raw, geometry, self.device)
            del raw, new_template, new_mv, new_noise
        root = Path(self.plan["output_dir"])
        print(f"paired: {row['id']} export", flush=True)
        transform = export_pair(root / "objects" / row["id"], geometry, maps, old_faces, new_faces)
        prediction = Path(self.plan["prediction_dir"]) / f"{row['id']}.npz"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        geometry_hash = sources["geometry_path"]["sha256"]
        temporary = prediction.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, old_faces=old_faces, new_faces=new_faces,
                                source_geometry_hash=np.array(geometry_hash),
                                plan_digest=np.array(self.plan_digest), seed=np.array(self.seed),
                                steps=np.array(self.steps), initial_noise_sha256=np.array(noise_hashes["old"]))
        temporary.replace(prediction)
        record = {"id": row["id"], "title": row["title"], "kind": row["kind"],
                  "source_name": row["source_name"], "source": copy.deepcopy(row),
                  "face_count": len(old_faces), "seed": self.seed, "steps": self.steps,
                  "postprocessing": False, "label_matching": False, "prior_matching": False,
                  "smoothing": False, "gt_initialization": False,
                  "plan_digest": self.plan_digest, "source_geometry_hash": geometry_hash,
                  "input_hashes": sources, "model_hashes": self.hashes,
                  "condition_sha256": array_hash(maps), "initial_noise": noise_hashes,
                  "target_coords_sha256": array_hash(coords), "target_coords_identical": True,
                  "target_tokens": len(coords), "shape_valid_fraction": float(valid.mean()),
                  "runtime": self.runtime, "web_transform": transform,
                  "generation": "standard Gaussian; 20 uniform Euler steps from t=1 to t=0; CFG=1",
                  "metrics_reference": "supplied input condition only; no external GT",
                  "old": {"condition_agreement": projection_agreement(old_faces, geometry["face_raster"], maps),
                          "unique_ids": np.unique(old_faces).tolist(), "num_classes": 20,
                          "decode": "8NN inverse-distance voxel logits at face centroids"},
                  "new": {"condition_agreement": projection_agreement(new_faces, geometry["face_raster"], maps),
                          "unique_ids": np.unique(new_faces).tolist(), "num_classes": 64,
                          "decode": "geometry-only surface head, 4 points/face, 1024 faces/chunk",
                          "query_fallback_fraction": fallback},
                  "artifacts": {name: f"objects/{row['id']}/{name}" for name in
                                ("old.glb", "new.glb", "inputcondition.png", "comparison.png")},
                  "prediction_path": str(prediction), "seconds": time.monotonic() - start}
        (root / "metrics").mkdir(parents=True, exist_ok=True)
        atomic_json(root / "metrics" / f"{row['id']}.json", record)
        print(f"paired: {row['id']} complete in {record['seconds']:.1f}s", flush=True)
        return record


def run_pair(plan, row, device="cuda:0"):
    """Convenience for one object; reuse Runner for multiple objects per worker."""
    return Runner(plan, device).run(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    workers = plan.get("workers", 1)
    if not 0 <= args.worker < workers:
        raise ValueError("Invalid worker index")
    rows = plan["samples"][args.worker::workers]
    if not rows:
        return 0
    import torch
    torch.set_num_threads(2)
    runner = Runner(plan, args.device)
    failures = []
    for index, row in enumerate(rows):
        print(json.dumps({"worker": args.worker, "phase": "inference", "id": row["id"],
                          "done": index, "total": len(rows)}), flush=True)
        try:
            runner.run(row)
        except Exception as exc:
            failures.append(row["id"])
            print(json.dumps({"worker": args.worker, "id": row["id"], "error": str(exc)}), file=sys.stderr, flush=True)
    print(json.dumps({"worker": args.worker, "phase": "complete", "total": len(rows),
                      "done": len(rows)-len(failures), "failures": failures}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
