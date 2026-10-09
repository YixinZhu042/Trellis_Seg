# Controlled demo reruns: original meshes and three characters

The [interactive comparison](https://yixinzhu042.github.io/Trellis_Seg/demo/) contains all 15 original demo geometries plus the three static textured GLBs supplied in `character_glb`. Its [public manifest](../demo/comparison.json) records exact model and asset hashes, same-noise hashes, face counts, and per-object condition-agreement metrics. The original numbered GLBs are unchanged, and the [previous HTML](../demo/legacy.html) is preserved.

## Models and inputs

- Old: 20-channel segmentation VAE encoder/decoder step 1,000 and segmentation DiT EMA0.9999 step 50,000.
- New: 64-channel segmentation VAE v2 regular step 9,500 and B + small-part DiT EMA step 5,000, the pair released on Hugging Face.
- Both reruns use FP32 parameters; DiT forwards use BF16 autocast. Each uses its own trained latent normalization.
- Shared: same geometry-only v2 sparse sampling, exact latent-coordinate order, native shape features, six-view label pixels, and bitwise-identical initial Gaussian noise. Seed 42, 20 uniform Euler steps, CFG=1.
- Legacy faces use 8-nearest-voxel inverse-distance logits at the face centroid. V2 uses its trained geometry-aware four-point surface head. No smoothing, label matching, projected-label overwrite, or output cleanup.
- Original face order/count is preserved. Export applies the same bounding-box centering and uniform scale to both web GLBs; it does not simplify the mesh.

These are controlled model-pair reruns, not exact historical replay: geometry sampling, runtime precision, baking, and seed protocol are fixed here rather than reconstructed from incomplete historical run metadata. The archived outputs are separately downloadable and are not used as GT or model input.

## Source provenance

The controller (demo 00) is exactly the 50,000-face geometry in the archived `0.glb`; its six-view input is the historical controller segmentation map. The other 14 demo GLBs were matched by complete file SHA-256 to the original dataset exports. Their old segmentation conditions are reused unchanged, rather than replacing them with corrected parts64 ground truth. Some legacy labels may be incorrect.

| Character | Supplied filename | Original faces | Shared input parts |
| --- | --- | ---: | ---: |
| Seated monk | `5d2730529f18ae35ca26bd320a0e0f14_moved.glb` | 490,563 | 8 |
| Armored bear | `after_moved (1).glb` | 500,000 | 6 |
| Fantasy warrior | `after_moved (2).glb` | 500,000 | 6 |

The characters have textures but no part annotations. Six orthographic views were rendered once per character, segmented in one shared 3×2 image request, and quantized into a shared discrete palette with the exact foreground mask. External 2D conditions may have imperfect silhouettes, inconsistent part assignments, or merged anatomy/clothing. No claim of 3D ground-truth accuracy is made.

## Results and limitations

All scores below compare projected labels with the shared **2D input condition** on observed foreground pixels. mIoU averages IDs present in the scored input. Extra output IDs are errors on those pixels; labels on unobserved surfaces cannot be judged by this metric. These are neither the 128-object test metrics nor a new unbiased benchmark.

| Cohort / character | Old pixel agreement | New pixel agreement | Old mIoU | New mIoU |
| --- | ---: | ---: | ---: | ---: |
| Original demos (15-object mean) | 99.14% | 99.23% | 93.53% | 92.53% |
| Seated monk | 85.07% | 88.96% | 45.72% | 65.15% |
| Armored bear | 85.82% | 88.00% | 45.05% | 60.33% |
| Fantasy warrior | 86.62% | 90.15% | 70.21% | 76.02% |

The 15 original demos are not uniformly better: four improve by more than 1 pp, three regress by more than 1 pp. Demo 08 (articulated lamp) drops 18.57 pp and demo 10 drops 7.99 pp. The original-cohort mean mIoU decreases about 1 pp despite a small increase in pixel agreement. Every original object is retained in the page.

The new character outputs recover some detail missing from the old model, but also produce small fragments using IDs absent from the 2D conditions. The new outputs use 45/44/53 distinct IDs for the 8/6/6-part inputs; the surface area carrying absent IDs is approximately 0.45%/1.47%/2.33%. No postprocessing hides these fragments. A higher 2D agreement score must not be described as a complete solution to small-part quality or consistent 3D semantics.

## Reproduction and verification

`scripts/demo_paired_inference.py` accepts a JSON plan with checkpoint/config paths, normalization, a fixed sample list, and paths to geometry/condition/shape NPZ files. Geometry contains the full `vae_v2.geometry.build_geometry` output plus the original-face raster; it must not contain target labels or target latent features. Condition maps use 0 for image background, 1 for unannotated mesh, and `part_id+1` for parts, within the old model's 20-channel capacity. The script checks coordinate identity and shared noise before generating both results.

```bash
python scripts/demo_paired_inference.py --plan /path/to/plan.json --worker 0
python scripts/package_demo_comparison.py --result /path/to/completed/results --git-repo /path/to/Trellis_Seg
```

The prepared inputs and numerical predictions remain outside Git. The selected web assets comprise 36 full-face-count GLBs, 36 input/comparison PNGs, and three character source previews (roughly 204 MB). No checkpoint or API credential is included.

All 18 paired workers and all three nodes exited successfully. Exported GLBs were reloaded to verify triangle order, geometry, and exact face colors. Thirteen paired-inference CPU tests cover identical noise, input-label offsets, missing-shape masks, and large-mesh chunk ordering. The browser page supports camera synchronization, keyboard navigation, mobile layout, model-load errors, and lazy selected-pair loading.

For local viewing, serve the repository over HTTP (`python -m http.server 8000`) and open `/demo/`. Direct `file://` access cannot load the JSON manifest. Interactive rendering uses the pinned Google model-viewer 4.0.0 CDN; static comparisons and GLB downloads remain available if the renderer cannot load.
