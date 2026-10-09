import itertools
import unittest

import numpy as np
import trimesh

from utils.seg_graph import alpha_expansion, energy, optimize_with_budget, smooth_hidden_surfaces
from utils.seg_projection import collect_evidence, consistency_metrics, fuse_labels, rasterize_face_ids


class GraphTests(unittest.TestCase):
    def test_hidden_cleanup_locks_every_observed_label(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        r = rasterize_face_ids(mesh, (32, 32))
        seg = np.where(r >= 0, 1, 0).astype(np.int32)
        # Only one input view supplies evidence; the far side stays unobserved.
        seg[1:] = 0
        e = collect_evidence(mesh, seg, face_ids=r)
        labels = np.ones(len(mesh.faces), np.int32)
        labels[np.flatnonzero(~e.visible)[::3]] = 2
        # Part 2 is absent from the single observed view but may exist unseen.
        e.votes = np.pad(e.votes, ((0, 0), (0, 1)))
        cleaned, _ = smooth_hidden_surfaces(mesh, labels, e)
        np.testing.assert_array_equal(cleaned[e.visible], labels[e.visible])
        self.assertEqual(consistency_metrics(labels, seg, e), consistency_metrics(cleaned, seg, e))

    def test_large_forbidden_cost_does_not_overflow_flow(self):
        unary = np.array([[1e9, 0, 1e9], [1e9, 10, 0], [1e9, 0, 1e9]])
        adjacency = np.array([[0, 1], [1, 2]])
        weights = np.array([20., 20.])
        labels = alpha_expansion(np.array([1, 2, 1]), unary, adjacency, weights)
        np.testing.assert_array_equal(labels, [1, 1, 1])

    def test_binary_cut_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(42)
        adjacency = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]])
        for _ in range(20):
            unary = np.column_stack([np.full(5, 1e6), rng.random((5, 2)) * 3])
            weights = rng.random(5)
            initial = rng.integers(1, 3, 5)
            labels = alpha_expansion(initial, unary, adjacency, weights)
            optimum = min(energy(np.array(p), unary, adjacency, weights)
                          for p in itertools.product([1, 2], repeat=5))
            self.assertAlmostEqual(energy(labels, unary, adjacency, weights), optimum)

    def test_multilabel_moves_reduce_energy(self):
        rng = np.random.default_rng(70)
        adjacency = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]])
        for _ in range(10):
            unary = np.column_stack([np.full(5, 1e6), rng.random((5, 4)) * 3])
            weights = rng.random(5)
            initial = rng.integers(1, 5, 5)
            labels = alpha_expansion(initial, unary, adjacency, weights)
            self.assertLessEqual(energy(labels, unary, adjacency, weights),
                                 energy(initial, unary, adjacency, weights))

    def test_fidelity_budget_and_unanimous_faces(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        raster = rasterize_face_ids(mesh, (32, 32))
        seg = np.where(raster >= 0, 1, 0).astype(np.int32)
        seg[0, :, 16:] = np.where(raster[0, :, 16:] >= 0, 2, 0)
        evidence = collect_evidence(mesh, seg, face_ids=raster)
        initial, _ = fuse_labels(mesh, np.ones(len(mesh.faces), np.int32), evidence)
        labels, diagnostic = optimize_with_budget(mesh, initial, seg, evidence, 0., 0.)
        before = consistency_metrics(initial, seg, evidence)
        after = consistency_metrics(labels, seg, evidence)
        self.assertGreaterEqual(after['accuracy'], before['accuracy'])
        self.assertGreaterEqual(after['miou'], before['miou'])
        unanimous = (evidence.votes.sum(1) == evidence.votes.max(1)) & evidence.visible
        np.testing.assert_array_equal(labels[unanimous], initial[unanimous])
        self.assertTrue(set(initial).issubset(labels))


if __name__ == '__main__':
    unittest.main()
