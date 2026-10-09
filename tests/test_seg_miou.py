import itertools
import unittest

import numpy as np
import trimesh

from utils.seg_miou import iou_statistics, per_part_iou_ceiling, refine_part_iou
from utils.seg_projection import collect_evidence, consistency_metrics, rasterize_face_ids, refine_projection


class PartIouTests(unittest.TestCase):
    def make_example(self):
        mesh = trimesh.creation.box()
        raster = np.full((6, 3, 3), -1, np.int32)
        seg = np.zeros_like(raster)
        raster[0].flat[:8] = [0, 0, 0, 1, 1, 1, 2, 2]
        seg[0].flat[:8] = [1, 1, 2, 1, 1, 2, 1, 1]
        raster[1].flat[:4] = [0, 1, 2, 3]
        seg[1].flat[:4] = [2, 1, 2, 1]
        # An uncovered input pixel must remain in the IoU denominator.
        seg[2, 0, 0] = 2
        evidence = collect_evidence(mesh, seg, face_ids=raster)
        return mesh, seg, evidence

    def test_statistics_match_direct_rendered_metric(self):
        mesh, seg, evidence = self.make_example()
        labels = np.ones(len(mesh.faces), np.int32)
        labels[0] = 2
        gt, pred, tp = iou_statistics(labels, seg, evidence)
        direct = consistency_metrics(labels, seg, evidence)
        for part, expected in direct['part_iou'].items():
            self.assertAlmostEqual(tp[part] / (gt[part] + pred[part] - tp[part]), expected)

    def test_binary_iou_ceiling_matches_exhaustive_subsets(self):
        mesh, seg, evidence = self.make_example()
        bound = per_part_iou_ceiling(seg, evidence)
        totals = evidence.votes.sum(1)
        for part in [1, 2]:
            best = 0.
            gt = np.count_nonzero(seg == part)
            for subset in itertools.product([False, True], repeat=4):
                selected = np.flatnonzero(subset)
                tp = evidence.votes[selected, part].sum()
                fp = totals[selected].sum() - tp
                best = max(best, tp / (gt + fp))
            self.assertAlmostEqual(bound[part], best)

    def test_small_part_improves_with_budget_and_unchanged_hidden_faces(self):
        mesh, seg, evidence = self.make_example()
        labels = np.ones(len(mesh.faces), np.int32)
        labels[0] = 2
        labels[-1] = 2
        result, report = refine_part_iou(mesh, labels, seg, evidence, .1, 1.)
        before = consistency_metrics(labels, seg, evidence)
        after = consistency_metrics(result, seg, evidence)
        self.assertGreaterEqual(after['miou'], before['miou'])
        self.assertGreaterEqual(after['accuracy'], before['accuracy'] - .1)
        np.testing.assert_array_equal(result[~evidence.visible], labels[~evidence.visible])
        unanimous = evidence.visible & (evidence.votes.sum(1) == evidence.votes.max(1))
        np.testing.assert_array_equal(result[unanimous], labels[unanimous])
        self.assertLessEqual(report['fragmentation_after']['boundary_length_pixels'],
                             report['fragmentation_before']['boundary_length_pixels'] * 2 + 1e-8)

    def test_unanimous_input_and_zero_budget_stay_exact(self):
        mesh = trimesh.creation.box()
        raster = np.full((6, 2, 2), -1, np.int32)
        raster[0] = [[0, 0], [1, 1]]
        seg = np.zeros_like(raster)
        seg[0] = [[1, 1], [2, 2]]
        evidence = collect_evidence(mesh, seg, face_ids=raster)
        labels = np.ones(len(mesh.faces), np.int32)
        labels[1] = 2
        result, report = refine_part_iou(mesh, labels, seg, evidence, 0., 0.)
        np.testing.assert_array_equal(result, labels)
        self.assertEqual(report['after']['miou'], 1.)
        self.assertEqual(report['after']['accuracy'], 1.)

    def test_shared_pipeline_keeps_total_accuracy_budget(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        raster = rasterize_face_ids(mesh, (24, 24))
        seg = np.where(raster >= 0, 1, 0).astype(np.int32)
        seg[0, :, 12:] = np.where(raster[0, :, 12:] >= 0, 2, 0)
        seg[1, 10:14, 10:14] = np.where(raster[1, 10:14, 10:14] >= 0, 3, 0)
        _, labels, _, evidence, _, report = refine_projection(
            mesh, np.ones(len(mesh.faces), np.int32), seg,
            graph_cleanup=True, accuracy_budget=.01, part_iou_refine=True)
        self.assertIsNotNone(report['part_iou_refinement'])
        self.assertLessEqual(report['refined']['gap_to_ceiling'], .01 + 1e-12)
        self.assertEqual(report['refined'], consistency_metrics(labels, seg, evidence))
        _, _, _, _, _, strict = refine_projection(
            mesh, np.ones(len(mesh.faces), np.int32), seg, graph_cleanup=False)
        self.assertIsNone(strict['part_iou_refinement'])
        self.assertEqual(strict['refined']['gap_to_ceiling'], 0.)


if __name__ == '__main__':
    unittest.main()
