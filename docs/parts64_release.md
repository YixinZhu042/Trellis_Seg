# Parts64 release — October 9, 2026

This release pairs the geometry-aware segmentation VAE v2 with the B + small-part-optimization segmentation DiT. It generates part labels on known mesh geometry from shape features and six-view segmentation conditions. The public scripts are research pipelines that require audited, versioned caches and experiment metadata. There is no turnkey v2 mesh/photo-to-segmentation command; `scripts/inference_wild.py` belongs to the historical 20-class family.

## Exact checkpoints and companion files

Model repository: [GilgameshYX/Trellis_segPart](https://huggingface.co/GilgameshYX/Trellis_segPart/tree/a5e92a02738ea0ef741053a863a9cf53b007cf9b/releases/parts64_20261009). Pin revision **`a5e92a02738ea0ef741053a863a9cf53b007cf9b`**. All paths in the table are relative to **`releases/parts64_20261009/`** in that repository.

| File | Role |
| --- | --- |
| `vae/regular_step009500.pt` | Validation-selected VAE regular step 9,500; encoder, decoder, and learned surface head |
| `dit/ema_step005000.pt` | Validation-selected B + small-part DiT EMA step 5,000 |
| `configs/vae_v2.json` | Portable VAE architecture/runtime settings |
| `configs/vae_backbone.json` | Matching sparse VAE backbone architecture |
| `configs/dit_model.json` | `SegSLatFlowModel` constructor arguments |
| `normalization.json` | Training-only target, six-view, and shape mean/std |
| `release.json` | Checkpoint identities, SHA-256 checksums, and provenance |
| `evaluation_summary.json` | Reported protocol and numerical evaluation summary |

```bash
hf download GilgameshYX/Trellis_segPart \
  --revision a5e92a02738ea0ef741053a863a9cf53b007cf9b \
  --include 'releases/parts64_20261009/*' \
  --local-dir ./trellis_segpart_models
```

These are the full paths to the weights: `releases/parts64_20261009/vae/regular_step009500.pt` and `releases/parts64_20261009/dit/ema_step005000.pt`. The prefix is plural **`releases/`**. The checkpoint dictionaries contain `model` and provenance metadata, without optimizer states. VAE training ended at step 10,000; the frozen VAE used to encode the DiT data and obtain these results is step **9,500**. A later training step is not a compatible substitute without re-evaluation.

Current weights are distributed through Hugging Face, not Git. Raw datasets, prepared caches, runtime caches, training logs, optimizer/RNG states, and generated prediction galleries are not shipped with the code release. The existing GitHub demo is retained separately and does not establish parts64 performance. Follow the project's existing MIT license and applicable upstream model and dataset terms.

## Compatibility contract

- There are **64 label channels including 0**. Label 0 is unknown/unannotated; nonzero part IDs range from 1 to 63. IDs are local to each object and have no fixed cross-object semantic meaning.
- The segmentation latent has **32 learned feature channels**, on sparse coordinates in a 32³ grid derived from 512³ geometry. Permuting those 32 channels does not rename part IDs.
- The DiT takes 32 noisy segmentation channels plus 32 aligned shape channels and predicts 32 velocity channels. Six-view segmentation features enter cross-attention. The evaluated code supplies a zero tensor of shape **`(1, 1029, 1024)`** for image-placeholder tokens; it does not extract real DINO image features.
- Rendered segmentation maps use 0 for image background and `part_id + 1` on mesh pixels. The view encoder removes this render-only offset once. Unknown mesh label 0 and background are distinct cases.
- Target and view latents use the same frozen VAE's posterior mean (`sample_posterior=False`). Shape features are joined by exact 32³ coordinates; unmatched positions remain masked zero, including after normalization. Six-view tokens retain their view order and coordinates.
- Decoding requires the geometry-only spatial template and the learned surface head. Sparse latent features alone are insufficient to reconstruct the decoder's spatial mappings.

Use the complete pair, companion configs, matching geometry/condition caches, and supplied normalization together. Do not mix in root-level legacy `encoder.pt`/`decoder.pt`, old 20-class caches, earlier VAE latents, legacy normalization, or the legacy inference scripts. Strict loading is intentional; skipping missing keys would omit parts of the model.

## Environment and portable loading

Use a compatible TRELLIS.2 CUDA environment with PyTorch, `flex_gemm`, the attention backend required by TRELLIS.2, NumPy, SciPy, and trimesh. Make the current repository and the installed TRELLIS.2 importable. Several modules use a sibling `../TRELLIS.2` checkout by default; configure your own checkout or Python path. This is not an `AutoModel.from_pretrained` package or a CPU-only quick start.

The DiT is approximately 1.3B parameters (30 blocks, hidden width 1536). The loading example retains FP32 parameters and uses the evaluated BF16 compute setting; available GPU memory depends on token and mesh sizes. Do not treat the recorded research hardware as a minimum-memory guarantee.

For example, after setting the paths for your machine:

```bash
export TRELLIS2_ROOT=/path/to/TRELLIS.2
export PYTHONPATH="$PWD:$TRELLIS2_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PARTS64_RELEASE_DIR="$PWD/trellis_segpart_models/releases/parts64_20261009"
```

Run the following Python blocks from the repository root in the same Python session or script. The first block only loads the released models. It follows the companion-config contract in the [Hugging Face model card source](hf_readme_parts64_20261009.md). Config paths are resolved relative to the downloaded **release directory**, not the Git checkout or the checkpoint's historical training paths.

```python
import json
import os
from pathlib import Path

import torch

from vae_v2.model import SegVaeV2
from models.seg_flow_model import SegSLatFlowModel

release_dir = Path(os.environ["PARTS64_RELEASE_DIR"]).expanduser().resolve()
device = torch.device("cuda:0")

vae_cfg = json.loads((release_dir / "configs/vae_v2.json").read_text())
vae_cfg["vae_config"] = str((release_dir / vae_cfg["vae_config"]).resolve())
vae_cfg["model"]["initialize_backbone"] = False
vae_cfg["model"]["gradient_checkpointing"] = False
vae = SegVaeV2(vae_cfg, initialize=False)
state = torch.load(
    release_dir / "vae/regular_step009500.pt",
    map_location="cpu", weights_only=True, mmap=True,
)
vae.load_state_dict(state["model"], strict=True)
del state
vae = vae.to(device).eval().requires_grad_(False)

dit_cfg = json.loads((release_dir / "configs/dit_model.json").read_text())
assert dit_cfg["name"] == "SegSLatFlowModel"
dit_args = dict(dit_cfg["args"], use_checkpoint=False)
model = SegSLatFlowModel(**dit_args)
state = torch.load(
    release_dir / "dit/ema_step005000.pt",
    map_location="cpu", weights_only=True, mmap=True,
)
model.load_state_dict(state["model"], strict=True)
del state
model = model.to(device).eval().requires_grad_(False)
model.dtype = torch.bfloat16

normalization = json.loads((release_dir / "normalization.json").read_text())
assert all(normalization[k]["split"] == "train"
           for k in ("latent_feats", "mv_feats", "shape_feats"))
```

Do not construct the VAE from `state["config"]`: checkpoint metadata contains historical training paths. Do not pass the portable runtime config through the training-only config loader, which also expects dataset and initialization paths. Load the full `model` entry into `SegVaeV2(..., initialize=False)` as above. The VAE sparse backbone remains FP32; the DiT compute dtype and surface-head autocast follow the evaluation implementation.

## Generation from prepared inputs

Provide your own geometry NPZ and condition NPZ prepared under the parts64 contract. They are not included in the weight download. The example consumes these fields:

| Input | Required fields |
| --- | --- |
| Geometry cache | `coords`, `mesh_vertices`, `mesh_faces`, `anchor_points`, `anchor_faces`, `anchor_voxel`, `face_normals`, `face_components`, `center`, `extent` |
| Condition cache | `latent_coords`, `shape_feats`, `shape_valid`, `mv_coords`, `mv_feats`, `view_offsets` |

Geometry `coords` are the retained 512³ voxel coordinates in their prepared order. `latent_coords` are the corresponding 32³ coordinates; shape features must have exactly that row order and 32 channels. `shape_valid` is a Boolean mask. Six-view features have 32 channels and matching `mv_coords`; `view_offsets` has seven entries marking six consecutive views. `SurfaceQueries` uses geometry, connected components, and normals to form four queries per face. Preserve mesh face order and the prepared coordinate system.

The research data preparation is implemented in `scripts/prepare_seg_vae_v2.py` and `scripts/prepare_dit_vae_v2.py`. Geometry-only preparation is available in `vae_v2.geometry.build_geometry`, but obtaining consistent external six-view conditions and native shape features still requires integration work. The example below assumes the input files and provenance have already been audited against the selected VAE and normalization. It illustrates the generation/decoding core; it does not replace the full evaluator's manifest, checksum, and selection checks.

```python
import numpy as np

from scripts.evaluate_dit_vae_v2 import generate_latent
from utils.dit_vae_v2 import unnormalize
from vae_v2.data import to_device
from vae_v2.geometry import SurfaceQueries


@torch.inference_mode(False)
@torch.no_grad()
def segment_prepared(geometry_npz, conditions_npz, seed):
    geometry_keys = (
        "coords", "mesh_vertices", "mesh_faces", "anchor_points",
        "anchor_faces", "anchor_voxel", "face_normals", "face_components",
        "center", "extent",
    )
    condition_keys = (
        "latent_coords", "shape_feats", "shape_valid",
        "mv_coords", "mv_feats", "view_offsets",
    )
    with np.load(geometry_npz, allow_pickle=False) as archive:
        geometry = {k: archive[k] for k in geometry_keys}
    with np.load(conditions_npz, allow_pickle=False) as archive:
        data = {k: archive[k] for k in condition_keys}

    # Geometry only: no face labels or target latent values are loaded.
    template = vae.geometry_template(
        torch.as_tensor(geometry["coords"], dtype=torch.int32, device=device)
    )
    if not np.array_equal(template.coords[:, 1:].cpu().numpy(),
                          data["latent_coords"]):
        raise ValueError("Geometry template and condition coordinates differ")

    # Normalizes shape/views, keeps absent shape features zero, creates
    # (1, 1029, 1024) zero image tokens, and starts from Gaussian noise.
    # The loop uses BF16 autocast: 20 Euler steps, CFG=1.
    generated = generate_latent(
        model, template, data, normalization, seed, 20, device
    )
    features, logits, coords = vae.decode_voxels(
        template.replace(unnormalize(generated, normalization))
    )
    if not np.array_equal(coords[:, 1:].cpu().numpy(), geometry["coords"]):
        raise ValueError("Decoded voxel order differs from geometry")

    query_builder = SurfaceQueries(geometry, vae_cfg["geometry"])
    face_labels = []
    face_count = len(geometry["mesh_faces"])
    for start in range(0, face_count, 1024):
        ids = np.arange(start, min(start + 1024, face_count))
        query = to_device(query_builder.build(ids), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, scores = vae.decode_surface(features, logits, query)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite surface probabilities")
        face_labels.append(scores.argmax(1).cpu().numpy())

    return np.concatenate(face_labels), logits.argmax(1).cpu().numpy()


# Set these to your audited files. The dataset-specific seed is supplied
# explicitly; the reported protocol uses 20260930 + int(instance[:8], 16).
predicted_faces, predicted_voxels = segment_prepared(
    os.environ["PARTS64_GEOMETRY_NPZ"],
    os.environ["PARTS64_CONDITIONS_NPZ"],
    seed=int(os.environ["PARTS64_SAMPLE_SEED"]),
)
```

This uses evaluation mode with gradients disabled and ordinary tensors (`inference_mode(False)`). `geometry_template` internally encodes dummy unknown labels with `sample_posterior=False` solely to create spatial mappings; its feature values do not initialize generation. The sampler starts from Gaussian noise, integrates predicted velocity, and unnormalizes the resulting segmentation latent before decoding. Target GT latent values are used only in the separate VAE reconstruction reference. Output face labels follow the input mesh's face order; no matching, smoothing, or topology cleanup is applied.

## Training methodology and split membership

The corrected PartNeXt preparation parses mesh-local GLB annotations and mask IDs, and retains soft voxel targets when multiple surface parts occupy a voxel. There are **22,633 training, 128 validation, and 128 test objects**. Twenty objects with annotations that cannot be reliably associated with geometry remain quarantined with reasons. The nested holdout policy (`nested_holdout_sha256_v1`, seed `20260929`) selects 128 objects within each original usable validation/test pool; the other usable objects enter training. The old 512-object test list is therefore not the current holdout.

**The finalized `manifest.json` sample `split` defines current membership.** In the VAE geometry caches, NPZ `metadata.split` retains the preparation-time split, and manifest `preparation_split` records that provenance. Do not reconstruct current membership from NPZ metadata or filenames. The DiT cache is newly encoded using the finalized membership and seals its own matching metadata. Training and evaluation validate identities, membership, checksums/audits, and normalization; manually relabeling a manifest is not a valid migration.

The VAE uses a sparse encoder/decoder with 64 label channels, latent32, and a learned geometry-aware four-point surface head. Losses combine voxel and surface CE/Dice, point supervision, reprojection, same/different-part relations, and KL regularization. The first corrected parts64 run trained for 5,000 steps with semantic input/output layers reinitialized from the historical backbone setup. A continuation restored model, optimizer, EMA, and RNG state and ran steps 5,001–10,000 with a random bijection of all IDs 1–63 for each training sample, keeping 0 fixed. Geometry is unchanged and all supervision is renamed consistently. Validation uses original IDs, and regular step 9,500 was selected. See [loss explanations](seg_vae_v2_losses_explained.md), [parts64 preparation](seg_vae_v2_parts64.md), and [permutation continuation](seg_vae_v2_parts64_permute.md).

The first DiT run in this latent space froze VAE step 9,500, regenerated target and six-view posterior-mean latents, and computed new statistics using training tokens only. Native shape features were reused via exact coordinate joins with explicit missing-feature masks. It adapted existing segmentation DiT weights with a new optimizer; validation selected regular step 4,500. The [frozen-VAE DiT notes](dit_vae_v2_parts64.md) document this baseline. B then continued **weights only** from that selected DiT for **5,000 additional optimizer steps**, starting a new optimizer and EMA (decay 0.999). It was not an optimizer-state resume or random initialization.

The [B + small-part configuration](../configs/dit_vae_v2_parts64_b_smallparts.json) combines:

- 50% original labels and 50% synchronized permutations of all nonzero IDs. One mapping covers the 3D target, all six views, and auxiliary labels; the same frozen VAE re-encodes the permuted target and views online. Latent channels are never permuted.
- Object draw proportions of **70% / 15% / 15%** for **1–20 / 21–32 / 33–63** part groups, using training objects only.
- Surface-query sampling of **25% by area / 75% equally by part**, and surface CE combining **50% area-weighted / 50% equal-part** objectives with importance correction.
- LR `1e-5`, 500-step warmup, global batch 24 in the recorded run, and the same flow and frozen-decoder auxiliary supervision. The original corrected training-only normalization remains fixed during B.

This is a combined experiment involving additional training, augmentation, sampling, loss balance, and EMA changes. There is no matched single-factor ablation attributing the improvement to one change. See the [B training notes](dit_b_smallparts.md).

## Adapting the research scripts

The repository includes preparation, training, evaluation, and contract tests under `scripts/`, `vae_v2/`, `utils/`, and `tests/`. Experiment configs describe the researchers' audited pipelines. **Configure paths and environment before running them.** Publication configs replace machine-specific locations with `/path/to/...` placeholders, including `storage_root: /path/to/trellisseg_run`. Set your own storage root and its cache/checkpoint subdirectories, dataset/mesh/annotation paths, TRELLIS.2 and initialization/config paths, VAE selection metadata, output/log directories, Python environment, GPU visibility, and distributed topology. These placeholders are not usable defaults. Config `_base` entries resolve relative to the config file, while other research paths are resolved by the corresponding config loader.

`vae_v2/config.py` accepts `TRELLISSEG_STORAGE_ROOT` as a fallback **only when `storage_root` is absent from the resolved JSON config**. An explicit JSON value, including an inherited `/path/to/trellisseg_run`, wins. Edit your configs accordingly; setting an environment variable does not rewrite explicit JSON paths.

The portable Hugging Face configs above are for loading the released models. Full training/evaluation commands additionally need source manifests, sealed cache plans and audits, checkpoint selection records, and matching provenance. Downloading weights alone does not recreate these inputs or permit an exact optimizer resume.

The native research cache provenance uses **exact path/identity matches**, file sizes and modification times, content checksums, and configuration/manifest digests. Existing trained-run records can therefore reject copied files even when their tensor contents match. Replacing an old path in a JSON file is insufficient: relocating trained caches or checkpoints requires a deliberate contract migration and re-audit of the dependent plans, manifests, selection records, and checkpoint provenance. Keep checks enabled; use fresh preparation or a reviewed migration rather than assuming that path substitution preserves a trained run's identity. The portable companion-config loader above avoids historical machine paths for model loading, but it does not migrate native training caches.

| Purpose | Entry points |
| --- | --- |
| Corrected geometry/labels and split sealing | `scripts/prepare_seg_vae_v2.py`, `scripts/audit_seg_vae_v2_cache.py`, `scripts/finalize_seg_vae_v2_split.py` |
| VAE training and reconstruction evaluation | `scripts/train_seg_vae_v2.py`, `scripts/evaluate_seg_vae_v2.py`, `scripts/test_seg_vae_v2_parts64.py` |
| Frozen-VAE target/view encoding and deep cache audit | `scripts/prepare_dit_vae_v2.py` (`--plan`, `--worker`, `--merge`) |
| DiT training, including B | `scripts/train_dit_vae_v2.py` |
| Frozen-checkpoint generation evaluation | `scripts/evaluate_dit_vae_v2.py` (`--prepare`, `--worker`, `--merge`) |
| Paired evaluation comparison | `scripts/compare_dit_runs.py` |

After preparing compatible data and adapting configs, these checks validate inputs without launching training:

```bash
python scripts/train_seg_vae_v2.py --config /path/to/your_vae_config.json --check_only
python scripts/train_dit_vae_v2.py --config /path/to/your_dit_config.json --check-only
```

The shell/remote launchers are optional, site-specific orchestration. They require a separately configured Remote API SDK/service, credentials, registered nodes, and matching paths. Configure the shared remote defaults with:

| Environment variable | Your setting |
| --- | --- |
| `TRELLIS_REMOTE_API_ROOT` | Path to your installed Remote API SDK |
| `TRELLIS_PYTHON` | Python executable in the environment available on the worker nodes |
| `TRELLIS_REMOTE_NODES` | Exactly three comma-separated host names or addresses |
| `TRELLIS_REMOTE_BOARD` | Your authorized Remote API board identifier |

These settings configure orchestration, not the dataset/storage paths in JSON. Do not assume the researchers' node addresses or storage mounts exist on your machine. Adapt the direct entry points to your scheduler instead if appropriate. Recorded 24-GPU training settings and fixed cache cohort checks are reproduction constraints, not a claim that arbitrary datasets or smaller hardware work unchanged. Historical notes may refer to logs or result galleries that are not distributed here.

### Tests

Run the CPU unit/contract suite from the repository root with the required Python dependencies installed and TRELLIS.2 importable. Some test imports require TRELLIS.2 even though the tests run on CPU; this is not a dependency-free suite. If TRELLIS.2 is not a sibling checkout, include its absolute path:

```bash
export TRELLIS2_ROOT=/path/to/TRELLIS.2
export PYTHONPATH="$PWD:$TRELLIS2_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests
```

The test that checks an external full-data manifest is skipped when that manifest is unavailable. Passing unit tests does not establish CUDA backend compatibility, full-cache readiness, or reproduction of the published metrics; those require the configured runtime and audited data.

## Evaluation and limits

Both checkpoints were selected on the **128-object validation set**. The published comparison uses **128 fixed test objects** with the same VAE, normalization, cached conditions, and per-object seed for both DiTs. The previous paired model is `dit_vae_v2_parts64/regular_step004500.pt`; the new model is B EMA step 5,000.

Generation begins from Gaussian noise and uses **20 Euler steps, CFG=1 (no guidance amplification)**. There is one fixed seed per object: `20260930 + int(instance[:8], 16)`. Evaluation uses original IDs, no Hungarian matching, no smoothing, and no other postprocessing. Inputs are **GT-derived six-view segmentation conditions plus native shape features on known meshes**. These are not end-to-end results from external 2D segmentation, actual photographs, or newly generated mesh geometry.

| Metric | Previous paired DiT | B EMA step 5,000 |
| --- | ---: | ---: |
| Surface mIoU | 83.58% | **85.05%** |
| Surface area accuracy | 93.86% | **94.95%** |
| Projection mIoU | 88.51% | **89.72%** |
| Hidden-surface mIoU | 71.67% | **73.48%** |
| Small-part recall | 57.03% | **60.61%** |
| Boundary F1 | 30.71% | **31.93%** |

Metrics are computed per object and averaged over defined values. Small parts occupy less than 1% of an object's annotated surface area; objects without small parts do not contribute to that recall average. Hidden surfaces have no visible pixel coverage in the six views. The same VAE's GT-encoded reconstruction reference reaches **95.96% surface mIoU** and **91.25% small-part recall**. Those are reconstruction results, not generation scores or strict per-object upper bounds.

The 33–63-part group contains only eight test objects: surface mIoU rises from 34.74% to 40.60%, while small-part recall remains low at 31.13%. A 60-part guitar recovers its main body but still has 44 parts with zero recall. Across the full cohort, 34 objects improve by more than one percentage point, six regress by more than one point, and one regresses by more than five points. A single seed does not measure generation variance.

These test objects have already informed development diagnostics, and historical initialization may have seen some objects. Treat the results as **development regression evaluation, not a new independent blind-test generalization result**. Current split disjointness and validation-only checkpoint selection do not remove that historical exposure. The [October 9 report](dit_b_smallparts_test_20261009.md) and downloaded `evaluation_summary.json` document the reported comparison; runtime galleries referenced by experiment notes are not part of the code package.
