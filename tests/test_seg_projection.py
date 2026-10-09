"""CPU regression tests for projection geometry and input-label fidelity."""
import itertools
import unittest

import numpy as np
import trimesh

from utils.seg_projection import (
    align_model_labels, collect_evidence, consistency_metrics, fuse_labels,
    rasterize_face_ids, subdivide_conflicting_faces, validate_seg_maps, welded_face_adjacency,
    refine_projection, smooth_vote_ties,
    restore_confident_hidden,
)


class ProjectionTests(unittest.TestCase):
    def test_hidden_model_prior_leaves_all_observed_labels_unchanged(self):
        mesh = trimesh.creation.box()
        r = np.full((6, 2, 2), -1, np.int32)
        s = np.zeros_like(r)
        r[0] = [[0, 0], [1, 1]]
        s[0] = [[3, 3], [9, 9]]
        e = collect_evidence(mesh, s, face_ids=r)
        model = np.zeros(len(mesh.faces), np.int32)
        model[1:6] = 5
        labels = np.full(len(mesh.faces), 3, np.int32)
        labels[1] = 9
        restored, diagnostics = restore_confident_hidden(labels, model, e)
        np.testing.assert_array_equal(restored[e.visible], labels[e.visible])
        np.testing.assert_array_equal(restored[2:6], np.full(4, 9))
        self.assertEqual(diagnostics['changed_faces'], 4)
        self.assertEqual(consistency_metrics(labels, s, e), consistency_metrics(restored, s, e))

    def test_cameras_and_nearest_surface(self):
        # Six sides have analytically known normals; checks front/back, mirrored
        # right/left and the intentionally identical top/bottom screen up axes.
        mesh = trimesh.creation.box(extents=[2, 1, 1.5])
        raster = rasterize_face_ids(mesh, (37, 41))
        expected = np.array([[0, 0, 1], [1, 0, 0], [0, 0, -1],
                             [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        for view, normal in enumerate(expected):
            ids = raster[view][raster[view] >= 0]
            self.assertGreater(len(ids), 100)
            np.testing.assert_allclose(mesh.face_normals[np.unique(ids)],
                                       np.broadcast_to(normal, (len(np.unique(ids)), 3)))

    def test_screen_orientation_with_asymmetric_triangle(self):
        # Positive x/y vertex must lie at the upper right in front, and upper
        # left in back. Test geometry rather than roundtripping our own labels.
        mesh = trimesh.Trimesh(vertices=[[-1, -1, 0], [1, -1, 0], [0.8, 1, 0]],
                               faces=[[0, 1, 2]], process=False)
        r = rasterize_face_ids(mesh, (40, 40))
        self.assertEqual(r[0, 5, 32], 0)
        self.assertEqual(r[0, 5, 7], -1)
        self.assertEqual(r[2, 5, 7], 0)
        self.assertEqual(r[2, 5, 32], -1)

    def test_occluded_face_receives_no_votes(self):
        outer = trimesh.creation.box()
        inner = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        mesh = trimesh.util.concatenate([outer, inner])
        r = rasterize_face_ids(mesh, (32, 32))
        self.assertLess(r.max(), len(outer.faces))
        s = np.where(r >= 0, 7, 0).astype(np.int32)
        e = collect_evidence(mesh, s, face_ids=r)
        labels, diag = fuse_labels(mesh, np.zeros(len(mesh.faces), np.int32), e)
        self.assertEqual(e.visible[len(outer.faces):].sum(), 0)
        self.assertTrue(np.all(labels == 7))
        self.assertEqual(diag['fallback_faces'], len(inner.faces))

    def test_collapsed_model_parts_split_and_hidden_ids_do_not_collide(self):
        mesh = trimesh.creation.box()
        r = np.full((6, 2, 2), -1, np.int32)
        s = np.zeros_like(r)
        r[0] = [[0, 0], [1, 1]]
        s[0] = [[3, 3], [9, 9]]
        model = np.full(len(mesh.faces), 3, np.int32)
        e = collect_evidence(mesh, s, face_ids=r)
        baseline, _ = align_model_labels(model, e.votes)
        labels, _ = fuse_labels(mesh, model, e)
        self.assertEqual(labels[0], 3)
        self.assertEqual(labels[1], 9)
        self.assertTrue(set(labels).issubset({3, 9}))
        self.assertEqual(consistency_metrics(baseline, s, e)['accuracy'], 0.5)
        self.assertEqual(consistency_metrics(labels, s, e)['accuracy'], 1.)

    def test_conflicting_views_attain_bruteforce_ceiling(self):
        mesh = trimesh.creation.box()
        r = np.full((6, 2, 2), -1, np.int32)
        s = np.zeros_like(r)
        r[0] = [[0, 0], [1, 1]]
        r[1] = [[0, 1], [2, 2]]
        s[0] = [[1, 1], [2, 1]]
        s[1] = [[2, 2], [3, 1]]
        e = collect_evidence(mesh, s, face_ids=r)
        labels, _ = fuse_labels(mesh, np.zeros(len(mesh.faces), np.int32), e)
        best = max(sum(e.votes[f, p] for f, p in enumerate(parts))
                   for parts in itertools.product([1, 2, 3], repeat=3)) / 8
        m = consistency_metrics(labels, s, e)
        self.assertEqual(m['accuracy'], best)
        self.assertEqual(m['gap_to_ceiling'], 0.)
        self.assertLess(best, 1.)

    def test_missing_coverage_is_an_error_and_empty_views_are_reported(self):
        mesh = trimesh.creation.box()
        r = np.full((6, 2, 2), -1, np.int32)
        s = np.zeros_like(r)
        r[0, 0, 0] = 0
        s[0, 0] = 1
        e = collect_evidence(mesh, s, face_ids=r)
        metrics = consistency_metrics(np.ones(len(mesh.faces), np.int32), s, e)
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(metrics['overlap']['accuracy'], 1.)
        self.assertIsNone(metrics['views']['back']['accuracy'])

    def test_welding_reconnects_split_vertices_without_reordering(self):
        mesh = trimesh.creation.box()
        split = trimesh.Trimesh(vertices=mesh.triangles.reshape(-1, 3),
                               faces=np.arange(36).reshape(-1, 3), process=False)
        before = split.triangles.copy()
        self.assertEqual(len(split.face_adjacency), 0)
        self.assertEqual(len(welded_face_adjacency(split)), len(mesh.face_adjacency))
        np.testing.assert_array_equal(before, split.triangles)

    def test_invalid_maps_fail_early(self):
        for maps in [np.zeros((6, 2, 2), np.int32), np.ones((5, 2, 2), np.int32),
                     np.full((6, 2, 2), -1), np.ones((6, 2, 2), np.float32)]:
            with self.assertRaises(ValueError):
                validate_seg_maps(maps)
        with self.assertRaises(ValueError):
            validate_seg_maps(np.full((6, 2, 2), 20), max_parts=19)

    def test_adaptive_subdivision_preserves_surface_and_seams(self):
        mesh = trimesh.creation.box()
        r = rasterize_face_ids(mesh, (48, 48))
        s = np.zeros_like(r)
        hit = r >= 0
        s[hit] = 1
        # A thin stripe crossing original triangle interiors demands refinement.
        s[:, :, 20:28] = np.where(hit[:, :, 20:28], 2, 0)
        evidence = collect_evidence(mesh, s, face_ids=r)
        model = np.arange(len(mesh.faces), dtype=np.int32)
        finer, inherited, parents = subdivide_conflicting_faces(mesh, model, evidence)
        self.assertGreater(len(finer.faces), len(mesh.faces))
        self.assertTrue(finer.is_watertight)
        self.assertTrue(finer.is_winding_consistent)
        np.testing.assert_allclose(finer.area, mesh.area)
        np.testing.assert_allclose(finer.volume, mesh.volume)
        areas = np.bincount(parents, weights=finer.area_faces, minlength=len(mesh.faces))
        np.testing.assert_allclose(areas, mesh.area_faces)
        np.testing.assert_array_equal(inherited, model[parents])
        refined_evidence = collect_evidence(finer, s)
        # One round still leaves the narrow stripe a minority on each child;
        # further refinement must resolve it rather than erase the small part.
        for _ in range(2):
            finer, inherited, _ = subdivide_conflicting_faces(finer, inherited, refined_evidence)
            refined_evidence = collect_evidence(finer, s)
        self.assertGreater(refined_evidence.votes.max(1).sum(), evidence.votes.max(1).sum())

    def test_subdivision_leaves_consistent_faces_and_respects_budget(self):
        mesh = trimesh.creation.box()
        r = rasterize_face_ids(mesh, (32, 32))
        s = np.where(r >= 0, 1, 0).astype(np.int32)
        e = collect_evidence(mesh, s, face_ids=r)
        labels = np.zeros(len(mesh.faces), np.int32)
        result, _, parents = subdivide_conflicting_faces(mesh, labels, e)
        self.assertIs(result, mesh)
        np.testing.assert_array_equal(parents, np.arange(len(mesh.faces)))
        s[:, :, :16] = np.where(r[:, :, :16] >= 0, 2, 0)
        e = collect_evidence(mesh, s, face_ids=r)
        result, _, _ = subdivide_conflicting_faces(mesh, labels, e, max_faces=len(mesh.faces))
        self.assertIs(result, mesh)

    def test_tie_smoothing_never_loses_input_evidence(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        raster = rasterize_face_ids(mesh, (32, 32))
        rng = np.random.default_rng(71)
        seg = rng.integers(1, 4, size=raster.shape, dtype=np.int32)
        seg[raster < 0] = 0
        evidence = collect_evidence(mesh, seg, face_ids=raster)
        labels, _ = fuse_labels(mesh, rng.integers(0, 5, len(mesh.faces)), evidence)
        cleaned = smooth_vote_ties(mesh, labels, evidence)
        before = consistency_metrics(labels, seg, evidence)
        after = consistency_metrics(cleaned, seg, evidence)
        self.assertEqual(before['accuracy'], after['accuracy'])
        self.assertEqual(after['gap_to_ceiling'], 0.)

    def test_thin_faces_are_protected_from_export_precision(self):
        # Altitude below float32 resolution; even a neighbor must not split its
        # long shared edge and introduce flipped child faces on GLB export.
        mesh = trimesh.Trimesh(vertices=[[1, 2, 0], [2, 2, 0], [1.5, 2 + 1e-8, 0],
                                        [1.5, 3, 0]], faces=[[0, 1, 2], [1, 0, 3]], process=False)
        r = np.full((6, 8, 8), -1, np.int32)
        r[0] = 1
        seg = np.ones_like(r)
        seg[0, :, :4] = 2
        seg[r < 0] = 0
        e = collect_evidence(mesh, seg, face_ids=r)
        refined, _, parents = subdivide_conflicting_faces(mesh, np.array([0, 1]), e)
        self.assertEqual(np.count_nonzero(parents == 0), 1)
        np.testing.assert_array_equal(refined.triangles[parents == 0], mesh.triangles[:1])

    def test_shared_refinement_path_and_export(self):
        import tempfile
        from pathlib import Path
        from scripts.refine_seg_projection import export_labels
        from scripts.audit_projection_results import closest_points
        mesh = trimesh.creation.box()
        r = rasterize_face_ids(mesh, (32, 32))
        seg = np.where(r >= 0, 1, 0).astype(np.int32)
        seg[:, :, :16] = np.where(r[:, :, :16] >= 0, 2, 0)
        refined, labels, parents, evidence, _, report = refine_projection(
            mesh, np.zeros(len(mesh.faces), np.int32), seg, subdivide=2)
        self.assertGreater(report['refined']['accuracy'], report['baseline']['accuracy'])
        self.assertEqual(report['refined']['gap_to_ceiling'], 0.)
        with tempfile.TemporaryDirectory() as directory:
            export_labels(refined, labels, Path(directory) / 'test.glb')
        # Analytic closest point and degenerate triangle regression.
        triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], float)
        points = np.array([[.2, .3, 1], [1, 1, 1]])
        np.testing.assert_allclose(closest_points(triangles, points), [[.2, .3, 0], [0, 0, 0]])


if __name__ == '__main__':
    unittest.main()
