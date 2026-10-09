import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

from scripts.compare_dit_decoded_experiment import paired_comparison
from scripts.evaluate_dit_decoded import merge_results


def row(uid, seed, score):
    metrics = {'accuracy': score, 'miou': score, 'small_part_recall': None}
    return {'instance': uid, 'seed': seed, 'voxel': dict(metrics),
            'surface': dict(metrics), 'projection': dict(metrics), 'boundary_f1': None}


class DecodedComparisonTests(unittest.TestCase):
    def test_independent_worker_shards_merge_without_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank, uid in enumerate(['a', 'b']):
                sample = row(uid, 42, .5 + .1 * rank)
                oracle = dict(sample)
                oracle.pop('seed')
                (root / f'metrics_rank{rank:04d}.json').write_text(json.dumps({
                    'checkpoint': '/checkpoint.pt', 'split': 'validation', 'seeds': [42],
                    'rank': rank, 'world_size': 2, 'samples': [sample], 'oracle': [oracle],
                }))
            args = SimpleNamespace(output_dir=root, checkpoint=Path('/checkpoint.pt'),
                                   split='validation', seeds=[42], no_save_predictions=True,
                                   independent_worker=True)
            merge_results(args, 0, 2, [{'instance': 'a'}, {'instance': 'b'}])
            merged = json.loads((root / 'metrics.json').read_text())
            self.assertEqual(len(merged['samples']), 2)
            self.assertEqual(len(merged['oracle']), 2)
            self.assertAlmostEqual(merged['summary']['projection']['miou'], .55)

    def test_averages_seeds_then_bootstraps_objects(self):
        before = [row('a', 1, .5), row('a', 2, .5), row('b', 1, .5), row('b', 2, .5)]
        after = [row('a', 1, .8), row('a', 2, .6), row('b', 1, .6), row('b', 2, .8)]
        report = paired_comparison(before, after)['projection.miou']
        self.assertEqual(report['objects'], 2)
        self.assertEqual(report['improved'], 2)
        self.assertAlmostEqual(report['mean_delta'], .2)
        self.assertAlmostEqual(report['ci95'][0], .2)
        self.assertAlmostEqual(report['ci95'][1], .2)

    def test_requires_identical_complete_pairs(self):
        with self.assertRaises(ValueError):
            paired_comparison([row('a', 1, .5)], [row('a', 2, .6)])
        with self.assertRaises(ValueError):
            paired_comparison([row('a', 1, .5), row('a', 1, .6)], [row('a', 1, .7)])

    def test_undefined_metrics_not_silently_zeroed(self):
        report = paired_comparison([row('a', 1, .5)], [row('a', 1, .6)])
        self.assertIsNone(report['boundary_f1']['mean_delta'])
        self.assertEqual(report['boundary_f1']['objects'], 0)
        bad = row('a', 1, .6)
        bad['boundary_f1'] = .5
        with self.assertRaises(ValueError):
            paired_comparison([row('a', 1, .5)], [bad])


if __name__ == '__main__':
    unittest.main()
