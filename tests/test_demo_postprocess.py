import unittest

import numpy as np

from scripts.postprocess_demo import apply_candidate, cleanup_components, smooth_isolated_boundaries


class DemoPostprocessTests(unittest.TestCase):
    def fixture(self):
        # Four coplanar triangles in a chain; face 1 is a tiny label-2 island.
        return {
            "adjacency": np.array([[0, 1], [1, 2], [2, 3]], np.int32),
            "face_area": np.array([1.0, 0.001, 1.0, 1.0]),
            "face_normals": np.tile([0.0, 0.0, 1.0], (4, 1)),
        }

    def test_small_component_merges_by_area_without_condition(self):
        geometry = self.fixture()
        labels = np.array([1, 2, 1, 1])
        output, changed, history = cleanup_components(
            labels,
            geometry["adjacency"],
            geometry["face_area"],
            geometry["face_normals"],
            0.001,
            0.7,
        )
        np.testing.assert_array_equal(output, [1, 1, 1, 1])
        self.assertTrue(changed[1])
        self.assertTrue(history)

    def test_large_or_ambiguous_components_remain(self):
        geometry = self.fixture()
        labels = np.array([1, 2, 3, 3])
        output, _, _ = cleanup_components(
            labels,
            geometry["adjacency"],
            geometry["face_area"],
            geometry["face_normals"],
            0.001,
            0.8,
        )
        self.assertEqual(output[1], 2)  # Equal support cannot select a dominant target.
        output, _, _ = cleanup_components(
            np.array([1, 2, 2, 1]),
            geometry["adjacency"],
            np.ones(4),
            geometry["face_normals"],
            0.1,
            0.7,
        )
        np.testing.assert_array_equal(output, [1, 2, 2, 1])

    def test_sharp_crease_does_not_drive_smoothing(self):
        geometry = self.fixture()
        geometry["face_normals"][1] = [0, 0, -1]
        output, changed = smooth_isolated_boundaries(
            np.array([1, 2, 1, 1]),
            geometry["adjacency"],
            geometry["face_area"],
            geometry["face_normals"],
        )
        self.assertFalse(changed[1])
        self.assertEqual(output[1], 2)

    def test_candidate_reports_exact_changes(self):
        geometry = self.fixture()
        raw = np.array([1, 2, 1, 1])
        candidate = {
            "component_area_fraction": 0.001,
            "component_dominance": 0.7,
            "smooth": False,
        }
        output, report = apply_candidate(raw, geometry, candidate)
        np.testing.assert_array_equal(output, [1, 1, 1, 1])
        self.assertEqual(report["changed_faces"], 1)
        self.assertAlmostEqual(report["changed_area_fraction"], 0.001 / 3.001)


if __name__ == "__main__":
    unittest.main()
