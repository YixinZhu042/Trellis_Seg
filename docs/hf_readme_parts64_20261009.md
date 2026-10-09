---
datasets:
- AuWang/PartNeXt
base_model:
- microsoft/TRELLIS.2-4B
pipeline_tag: image-to-3d
tags:
- 3d
- segmentation
- part-segmentation
- vae
- diffusion-transformer
---

# Trellis SegPart

<p align="center">
  <img src="trellis_seg.png" width="100%">
</p>

**Trellis Seg is native 3D segmentation guided by a 2D prior, built upon TRELLIS.2. For more details, please refer to our [code](https://github.com/YixinZhu042/Trellis_Seg).**

## News

- **2026-10-09:** Recommended paired release: 64-channel segmentation VAE v2 and the B + small-part-optimization DiT. The pair was evaluated on 128 fixed test objects. See the compatibility and evaluation notes below.
- **2026-07-16:** Segmentation DiT checkpoints became available.
- **2026-07-15:** Segmentation VAE checkpoints became available.

## Recommended checkpoints: parts64 v2

Use these two checkpoints **together**, with the matching v2 code and normalization statistics.

| Component | Repository path | Selection |
| --- | --- | --- |
| Segmentation VAE v2 | `releases/parts64_20261009/vae/regular_step009500.pt` | Validation-selected regular step 9,500 from the label-permutation continuation |
| Segmentation DiT 1.3B | `releases/parts64_20261009/dit/ema_step005000.pt` | Validation-selected EMA step 5,000 from B + small-part optimization |
| VAE architecture/runtime configuration | `releases/parts64_20261009/configs/vae_v2.json` | Matching geometry-aware surface decoder configuration |
| VAE backbone architecture | `releases/parts64_20261009/configs/vae_backbone.json` | Encoder/decoder structure used to construct VAE v2 |
| DiT architecture | `releases/parts64_20261009/configs/dit_model.json` | `SegSLatFlowModel`, 64 input / 32 output channels |
| Latent normalization | `releases/parts64_20261009/normalization.json` | Training-only target, six-view, and shape mean/std |
| Release manifest | `releases/parts64_20261009/release.json` | Checkpoint identities, SHA-256 checksums, and evaluation provenance |

"Recommended" means selected on validation, not necessarily the last VAE training step. VAE training reached step 10,000, but **step 9,500** is the frozen VAE used to encode the DiT training data and to produce the reported results. Do not substitute VAE step 10,000 without re-evaluating compatibility.

The `.pt` checkpoints contain a `model` state dictionary and provenance metadata. They do not contain optimizer state. VAE v2's `model` includes the **encoder, decoder, and learned surface head**; legacy `encoder.pt` and `decoder.pt` are not substitutes for the complete v2 model.

### Compatibility and label conventions

- **64 label channels include channel 0**, reserved for unknown/unannotated labels. There are at most **63 nonzero part IDs**.
- Part IDs are **object-local identifiers**, not a shared semantic vocabulary. The same physical part can have different IDs in different objects.
- The VAE latent has 32 channels. These are learned latent features, **not** label channels; do not permute latent channels to rename part IDs.
- The DiT receives 32 noisy segmentation channels concatenated with 32 shape channels and predicts a 32-channel velocity. Six-view segmentation latents enter cross-attention.
- The evaluated implementation supplies zero image-placeholder tokens; it does **not** use real DINO image features.
- In rendered segmentation maps, image background is 0 and mesh pixels use `part_id + 1`. The view encoder removes this render-only offset exactly once. Unannotated mesh label 0 and image background are distinct cases.
- Use the v2 geometry-aware surface decoder and its spatial template. Replacing only the checkpoint paths in a legacy 20-class inference script is **not sufficient**.
- Do not mix these weights with the legacy 20-class caches, legacy latent normalization, or earlier segmentation VAE checkpoints. The published normalization is part of the model contract.

### Download

```bash
hf download GilgameshYX/Trellis_segPart \
  --include 'releases/parts64_20261009/*' \
  --local-dir ./trellis_segpart_models
```

Loading requires the v2 implementation, including `vae_v2.model.SegVaeV2`, `vae_v2.geometry.SurfaceQueries`, and `models.seg_flow_model.SegSLatFlowModel`, in addition to TRELLIS.2 and its CUDA dependencies. This is not an `AutoModel.from_pretrained` package.

After downloading, resolve configuration paths relative to the downloaded release. In particular, set the VAE configuration's `vae_config` to the downloaded `configs/vae_backbone.json` before constructing `SegVaeV2(config, initialize=False)`. Load each checkpoint's `model` entry with strict state-dictionary loading. Construct the DiT from the released `dit_model.json` arguments. Historical paths embedded in checkpoint provenance are not paths to use on another machine.

For generation, start from Gaussian noise, normalize shape and six-view conditions with the supplied statistics, integrate the DiT velocity, then unnormalize the segmentation latent before VAE decoding. Do not initialize generation from the target GT latent. Cached GT latent decoding is used only for the separately reported VAE reconstruction reference.

## What changed

### VAE v2

- Corrected mesh-local annotation parsing and 64-channel labels.
- Soft voxel targets retain information where multiple surface parts occupy a voxel.
- Geometry-aware four-point surface decoding with a learned surface head.
- Surface, voxel, boundary/consistency, and reprojection supervision.
- Training-time permutations of all nonzero part IDs, keeping 0 fixed.

### DiT B + small-part optimization

- Initialized from the previously validation-selected segmentation DiT, not from random weights and not directly from the original TRELLIS checkpoint.
- Frozen VAE v2 throughout training; 5,000 additional optimizer steps.
- 50% original labels and 50% synchronized label permutations. One permutation is applied to the 3D target, all six conditioning maps, and auxiliary supervision; target and conditioning latents are then re-encoded by the same frozen VAE.
- Object sampling proportions: **70% / 15% / 15%** for **1–20 / 21–32 / 33–63** part groups.
- Surface queries mix 25% area sampling and 75% equal-part sampling. Surface CE mixes 50% area-weighted and 50% equal-part objectives.
- Matching training-only normalization is preserved. Optimizer and EMA were restarted; EMA decay is 0.999.

These changes were evaluated as a **combined experiment**. There is no matched single-factor ablation establishing that any one change alone caused the improvements.

## Evaluation: 128 fixed test objects

Protocol: **20 Euler steps, CFG=1**, one fixed seed per object, original IDs, no label matching, no smoothing or other postprocessing. The two DiTs use the same VAE, cached inputs, normalization, and seed per object. Checkpoint selection uses only the 128-object validation set.

Inputs are **GT-derived six-view segmentation conditions plus native shape features on a known mesh**. This is not an end-to-end result from an external 2D segmentation API or real photographs, and the model does not generate new mesh geometry.

| Metric | Previous paired DiT | B EMA step 5,000 | Change |
| --- | ---: | ---: | ---: |
| Surface mIoU | 83.58% | **85.05%** | +1.46 pp |
| Surface area accuracy | 93.86% | **94.95%** | +1.09 pp |
| Projection mIoU | 88.51% | **89.72%** | +1.21 pp |
| Hidden-surface mIoU | 71.67% | **73.48%** | +1.81 pp |
| Small-part recall | 57.03% | **60.61%** | +3.59 pp |
| Boundary F1 | 30.71% | **31.93%** | +1.22 pp |

Metrics are computed per object and averaged over defined values. A small part occupies less than 1% of an object's annotated surface area; objects without such parts do not contribute to small-part recall.

| Part count | Test objects | Previous surface mIoU | B surface mIoU |
| --- | ---: | ---: | ---: |
| 1–20 | 108 | 89.20% | 90.13% |
| 21–32 | 12 | 65.57% | 68.87% |
| 33–63 | 8 | 34.74% | 40.60% |

The same VAE's GT-encoded reconstruction reference achieves **95.96% surface mIoU** and **91.25% small-part recall**. This is a reconstruction reference, not the DiT generation score or a strict per-object upper bound.

### Known limitations

- Very small and high-count parts remain difficult. The 33–63-part test group has only eight objects; its small-part recall improved only from 29.33% to 31.13%.
- A 60-part guitar's body recall improved from 0.35% to 90.56%, but **44/60 parts still have zero recall**, and its surface mIoU is only 8.77%. Recovering a large body does not solve fine-part segmentation.
- Across 128 test objects, 34 improved by more than 1 pp, six regressed by more than 1 pp, and one regressed by more than 5 pp. Not all objects improve.
- These objects have previously been used for development diagnostics, and historical initialization may have seen some objects. Treat this as **development regression evaluation**, not a new independent blind-test generalization result.
- Only one fixed generation seed per object was evaluated. These results do not characterize seed variance or justify selecting inference settings on the test set.

## Legacy checkpoints

The root-level `encoder.pt`, `decoder.pt`, `config.json`, and older `denoiser_ema0.9999_step*.pt` files are retained for legacy workflows. They belong to the earlier **20-class** model family and are **not compatible** with the recommended parts64 v2 pair or its latent caches.

The release does not change the repository's visibility or grant a new license. Please retain and review the applicable project, upstream model, and dataset terms for your use.
