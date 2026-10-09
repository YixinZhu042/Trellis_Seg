# Geometry-only demo postprocessing — 2026-10-09

This note documents the optional cleanup applied to the released parts64 new-model outputs for the 15 original web demos and three added character meshes. The interactive page preserves each raw GLB and exposes raw and postprocessed downloads side by side.

## What the transform does

The transform operates only on mesh topology, face areas, face normals, and the raw predicted face labels:

1. Build face adjacency from shared mesh edges.
2. Find connected components separately within each predicted label.
3. Mark a component as small when its surface area is below 0.05% of the object's total surface area.
4. Measure which labels border that component. Shared-boundary support is weighted by the areas of the adjacent faces and their normal agreement. Sharp folds therefore have little influence.
5. Merge the component only if one neighboring label has at least 80% of weighted boundary support.
6. Repeat component cleanup for at most two passes, then perform one synchronous, high-confidence boundary vote. Synchronous means every decision reads the same pre-vote labels, avoiding order-dependent cascades.

The transform does **not** read ground truth, use the six-view condition, match or permute label IDs, or overwrite 3D predictions with labels projected from 2D. The 2D condition is used only after transformation to report input-agreement metrics and, during development, to select one global parameter setting.

Using surface area rather than a fixed face count matters because the meshes have very different tessellation densities. Ten tiny triangles on a dense character do not represent the same physical region as ten large triangles on a coarse mesh.

## Parameter selection

Six global candidates were scanned. A candidate was eligible only when:

- no object's input-condition mIoU decreased by more than 1 percentage point;
- mean pixel agreement decreased by no more than 0.1 percentage point; and
- mean changed surface area was at most 2%.

Among eligible candidates, selection first maximized mean input-condition mIoU, then favored fewer connected components and less changed area. The selected setting was:

```json
{
  "name": "area_0p05_boundary",
  "component_area_fraction": 0.0005,
  "component_dominance": 0.8,
  "smooth": true
}
```

This selection used all 18 displayed examples. They are a development cohort, not a held-out benchmark. Consequently, the observed absence of mIoU regressions on these examples is not evidence that every unseen object will improve.

## Aggregate results

All scores below are exact-ID agreement with the supplied six-view 2D condition, not independent 3D ground-truth accuracy.

| Cohort | Objects | Raw mIoU | Post mIoU | Raw accuracy | Post accuracy | Components before | Components after | Mean changed area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original demos | 15 | 92.5321% | 93.0797% | 99.2322% | 99.3269% | 4,195 | 3,127 | 0.6063% |
| Characters | 3 | 67.1656% | 67.3595% | 89.0390% | 89.3200% | 15,310 | 4,058 | 1.1251% |
| All displayed examples | 18 | — | — | — | — | 19,505 | 7,185 | 0.6928% |

Fourteen objects improved in input-condition mIoU, four were unchanged, and none regressed on this development cohort. Total connected components decreased by 63.2%. The largest mIoU changes were demo 09 (+1.999 pp), demo 02 (+1.688 pp), demo 08 (+1.176 pp), and demo 04 (+1.112 pp).

The character results were:

| Object | Raw mIoU | Post mIoU | Components before | Components after |
| --- | ---: | ---: | ---: | ---: |
| Character 1 — seated monk | 65.15% | 65.51% | 3,084 | 732 |
| Character 2 — armored bear | 60.33% | 60.37% | 6,146 | 1,433 |
| Character 3 — fantasy warrior | 76.02% | 76.20% | 6,080 | 1,893 |

Per-object development results:

| Object | Raw mIoU | Post mIoU | Change (pp) | Components | Changed area |
| --- | ---: | ---: | ---: | ---: | ---: |
| demo00 | 76.20% | 76.36% | +0.163 | 204 → 46 | 0.109% |
| demo01 | 100.00% | 100.00% | 0.000 | 2 → 2 | 0.000% |
| demo02 | 84.64% | 86.32% | +1.688 | 74 → 34 | 0.146% |
| demo03 | 83.65% | 84.02% | +0.369 | 506 → 203 | 1.319% |
| demo04 | 95.53% | 96.64% | +1.112 | 275 → 191 | 0.564% |
| demo05 | 99.85% | 99.85% | 0.000 | 6 → 5 | 0.003% |
| demo06 | 99.18% | 99.58% | +0.396 | 212 → 63 | 0.814% |
| demo07 | 91.87% | 92.27% | +0.408 | 291 → 200 | 0.821% |
| demo08 | 76.92% | 78.09% | +1.176 | 166 → 100 | 0.460% |
| demo09 | 90.67% | 92.67% | +1.999 | 29 → 18 | 0.483% |
| demo10 | 91.82% | 91.88% | +0.059 | 2,275 → 2,202 | 0.396% |
| demo11 | 100.00% | 100.00% | 0.000 | 13 → 4 | 0.344% |
| demo12 | 98.95% | 99.01% | +0.061 | 75 → 37 | 0.977% |
| demo13 | 100.00% | 100.00% | 0.000 | 42 → 13 | 0.948% |
| demo14 | 98.72% | 99.50% | +0.784 | 25 → 9 | 1.708% |
| character1 | 65.15% | 65.51% | +0.361 | 3,084 → 732 | 0.547% |
| character2 | 60.33% | 60.37% | +0.046 | 6,146 → 1,433 | 1.938% |
| character3 | 76.02% | 76.20% | +0.175 | 6,080 → 1,893 | 0.890% |

## Artifacts and reproducibility

The public manifest is `demo/postprocess.json`. Each sample directory contains:

- `new.glb`: unchanged parts64 raw output;
- `post.glb`: geometry-only postprocessed output; and
- `post_comparison.png`: input condition, raw output, and postprocessed output rendered with the same view layout.

The implementation is in `scripts/postprocess_demo.py`; packaging is in `scripts/package_demo_postprocess.py`; focused unit tests are in `tests/test_demo_postprocess.py`. Numeric scan reports and runtime predictions are experiment artifacts and are not intended to replace the public manifest.

## Limitations

- Cleanup can remove genuine tiny parts if their geometry and boundary context resemble an isolated error.
- A conservative boundary rule cannot fix a large region carrying the wrong label.
- Geometry-only cleanup cannot infer semantic meaning or recover a missing part.
- Input-condition agreement can reward errors present in the condition itself.
- The selected setting should be evaluated on a genuinely held-out cohort before being treated as a general default.
