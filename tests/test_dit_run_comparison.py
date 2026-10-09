"""Synthetic contracts and CPU artifact checks; no model inference or network."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import compare_dit_runs as c


def uid(i):
    return f'{i:08x}' + '0' * 24


def scores(value=.5, parts=2):
    result = {group: {'miou': value, 'accuracy': value, 'small_part_recall': None}
              for group in ('surface', 'projection', 'voxel', 'hidden_surface')}
    result.update({key: None for key in c.METRICS if '.' not in key})
    result['parts'] = [{'label': i+1, 'area_fraction': 1/parts, 'iou': value, 'recall': value}
                       for i in range(parts)]
    return result


def source(i, role='test', parts=2):
    return {'instance': uid(i), 'role': role, 'split': c.ROLES[role], 'type_id': 'guitar',
            'path': f'samples/{uid(i)}.npz', 'bytes': 100, 'mtime_ns': 200, 'sha256': 'cache-sha',
            'plan_digest': 'cache-plan', 'tokens': 2, 'mv_tokens': 2, 'shape_coverage': 1.,
            'shape_source': {'path': 'shape.npz', 'bytes': 50, 'mtime_ns': 100},
            'source_path': 'geometry.npz', 'source_bytes': 100, 'source_mtime_ns': 300, 'parts': parts}


def fixture():
    base = {'models': {'denoiser': {'args': {'channels': 32}}}}
    plan = {'cache_digest': 'cache', 'vae_checkpoint': {'path': 'vae', 'bytes': 1, 'mtime_ns': 2},
            'vae_sha256': 'vae-sha', 'source_manifest_digest': 'source-manifest',
            'normalization': {'latent_feats': {'mean': [0.], 'std': [1.]}},
            'base_config': base, 'base_digest': c.digest(base), 'vae_config': {'num_classes': 64},
            'config': {'seed': 20260930, 'cache_dir': 'cache', 'max_tokens': 16384},
            'checkpoint': {'path': 'old.pt', 'bytes': 10, 'mtime_ns': 20},
            'selection': {'split': 'validation', 'test_used_for_selection': False,
                          'ids': [uid(i) for i in range(1000, 1128)], 'checkpoint': 'old.pt', 'sampling_steps': 20},
            'seed_policy': c.SEED_POLICY, 'sampling_steps': 20, 'condition': c.CONDITION,
            'generation': 'Gaussian Euler', 'oracle': 'cached mean', 'label_matching': False,
            'postprocessing': False, 'test_used_for_selection': False, 'prediction_dir': 'old-preds',
            'samples': [source(i, parts=(2, 21, 33)[i % 3]) for i in range(128)] +
                       [source(500, 'train_diagnostic'), source(1000, 'validation_parity')]}
    new = copy.deepcopy(plan)
    new['checkpoint']['path'] = new['selection']['checkpoint'] = 'new.pt'
    new['prediction_dir'] = 'new-preds'
    new['samples'].append(source(1001, 'validation_diagnostic'))
    sweep = copy.deepcopy(plan)
    sweep.update(split='validation', test_used=False,
                 seed_policy='20260930 + int(instance[:8],16); identical x1 across settings',
                 prediction_dir='sweep-preds', settings=[{'name': c.BASELINE, 'steps': 20,
                                                        'cfg': 1., 'condition': 'correct', 'eligible': True}])
    sweep['samples'] = [{k: v for k, v in source(i, 'validation_diagnostic').items() if k != 'role'}
                        for i in range(1000, 1128)]
    return plan, new, sweep


def record(row, plan, value=.5):
    return {**{k: row[k] for k in ('instance', 'role', 'split', 'type_id', 'parts', 'shape_coverage')},
            'plan_digest': c.digest(plan), 'seed': c.expected_seed(plan, row['instance']),
            'dit': scores(value, row['parts']), 'oracle': scores(.9, row['parts'])}


def merged(plan, value=.5, diagnostic=.99):
    result = {k: plan[k] for k in ('checkpoint', 'vae_checkpoint', 'cache_digest', 'sampling_steps',
                                   'condition', 'prediction_dir', 'test_used_for_selection')}
    for section, role in (('samples', 'test'), ('diagnostics', 'train_diagnostic'),
                          ('validation_diagnostics', 'validation_diagnostic')):
        result[section] = [record(r, plan, value if role == 'test' else diagnostic)
                           for r in plan['samples'] if r['role'] == role]
    result['ids'] = [r['instance'] for r in result['samples']]
    # These intentionally wrong precomputed summaries must never drive aggregates.
    result['summary'] = {'surface': {'miou': -999}}
    return result


def sweep_record(plan, i=1001):
    row = next(r for r in plan['samples'] if r['instance'] == uid(i))
    return {'instance': row['instance'], 'split': 'validation', 'parts': row['parts'],
            'seed': c.expected_seed(plan, row['instance']), 'plan_digest': c.digest(plan),
            'setting': plan['settings'][0], 'nfe': 20, 'metrics': scores(.2, row['parts'])}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.old, self.new, self.sweep = fixture()

    def test_test_aggregation_excludes_all_diagnostics_and_parity(self):
        old, new = merged(self.old, .5, 0), merged(self.new, .52, 1)
        old.pop('validation_diagnostics')  # Real old schema omits this key.
        pairs = c.prepare_pairs(self.old, self.new, old, new)
        result = c.build_report(pairs, self.old, self.new)
        self.assertEqual(result['test_count'], 128)
        self.assertEqual(len(pairs), 130)
        self.assertAlmostEqual(result['metrics']['surface.miou']['delta'], .02)
        self.assertEqual(result['metrics']['surface.miou']['before'], .5)
        self.assertEqual(result['improved_over_1pp'], 128)
        self.assertEqual(result['improved_over_5pp'], 0)
        self.assertEqual(result['training_diagnostics']['count'], 1)
        self.assertEqual(result['validation_diagnostics']['unpaired_ids'], [uid(1001)])
        self.assertNotIn(uid(1000), [p['instance'] for p in pairs])
        self.assertEqual(sum(g['count'] for g in result['groups_by_parts']), 128)
        for group in result['groups_by_parts']:
            self.assertAlmostEqual(group['metrics']['surface.miou']['delta'], .02)
            self.assertIsNone(group['metrics']['surface.small_part_recall']['delta'])

    def test_baseline_pairs_validation_with_role_ignored_and_no_old_oracle(self):
        baseline = {uid(1001): sweep_record(self.sweep)}
        pairs = c.prepare_pairs(self.old, self.new, merged(self.old), merged(self.new), self.sweep, baseline)
        pair = next(p for p in pairs if p['role'] == 'validation_diagnostic')
        self.assertEqual(pair['before_source'], 'validation_baseline')
        self.assertEqual(pair['before']['surface']['miou'], .2)
        self.assertIsNone(pair['oracle_before'])
        self.assertIsNotNone(pair['oracle_after'])
        self.assertTrue(pair['paired'])

    def test_reject_frozen_contract_counterfactuals(self):
        mutations = [lambda p: p.update(cache_digest='wrong'),
                     lambda p: p['vae_checkpoint'].update(bytes=9),
                     lambda p: p['normalization']['latent_feats'].update(mean=[1.]),
                     lambda p: p['base_config'].update(changed=True),
                     lambda p: p.update(base_digest='wrong'),
                     lambda p: p['config'].update(seed=1),
                     lambda p: p['config'].update(cache_dir='other'),
                     lambda p: p.update(cfg=2), lambda p: p.update(sampling_steps=50),
                     lambda p: p.update(condition='shuffled'), lambda p: p.update(seed_policy='random'),
                     lambda p: p.update(label_matching=True), lambda p: p.update(postprocessing=True),
                     lambda p: p['selection'].update(test_used_for_selection=True),
                     lambda p: p['selection'].update(ids=p['selection']['ids'][:-1]+[uid(0)])]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                new = copy.deepcopy(self.new)
                mutate(new)
                with self.assertRaises(ValueError):
                    c.validate_plans(self.old, new)

    def test_reject_cohort_source_and_cache_counterfactuals(self):
        for key in c.CACHE_FIELDS:
            with self.subTest(key=key):
                new = copy.deepcopy(self.new)
                new['samples'][0][key] = 'changed'
                with self.assertRaises(ValueError):
                    c.validate_plans(self.old, new)
        for mutate in (lambda p: p['samples'].append(copy.deepcopy(p['samples'][0])),
                       lambda p: p['samples'].pop(0),
                       lambda p: p['samples'].pop(128)):
            new = copy.deepcopy(self.new)
            mutate(new)
            with self.assertRaises(ValueError):
                c.validate_plans(self.old, new)

    def test_reject_report_duplicates_seeds_parts_and_roles(self):
        mutations = [lambda r: r['ids'].__setitem__(0, r['ids'][1]),
                     lambda r: r['samples'].append(copy.deepcopy(r['samples'][0])),
                     lambda r: r['samples'][0].update(seed=7),
                     lambda r: r['samples'][0].update(plan_digest='bad'),
                     lambda r: r['samples'][0].update(role='train_diagnostic'),
                     lambda r: r['samples'][0]['dit']['parts'][0].update(area_fraction=.123),
                     lambda r: r['samples'][0]['oracle']['parts'][0].update(label=63),
                     lambda r: r['samples'][0]['dit']['surface'].update(miou=float('nan'))]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                report = merged(self.new)
                mutate(report)
                with self.assertRaises(ValueError):
                    c.prepare_pairs(self.old, self.new, merged(self.old), report)

    def test_reject_baseline_checkpoint_setting_seed_and_coordinates_contract(self):
        mutations = [lambda p: p['checkpoint'].update(path='different.pt'),
                     lambda p: p['settings'][0].update(cfg=2),
                     lambda p: p['settings'][0].update(steps=50),
                     lambda p: p['settings'][0].update(condition='no_mv'),
                     lambda p: p['config'].update(seed=42),
                     lambda p: p['samples'][1].update(sha256='different-coordinate-cache')]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                sweep = copy.deepcopy(self.sweep)
                mutate(sweep)
                with self.assertRaises(ValueError):
                    c.validate_plans(self.old, self.new, sweep)
        for key, value in [('seed', 7), ('plan_digest', 'bad'), ('nfe', 40)]:
            baseline = sweep_record(self.sweep)
            baseline[key] = value
            with self.assertRaises(ValueError):
                c.prepare_pairs(self.old, self.new, merged(self.old), merged(self.new),
                                self.sweep, {uid(1001): baseline})

    def test_none_metrics_common_support_and_strict_thresholds(self):
        pairs = c.prepare_pairs(self.old, self.new, merged(self.old), merged(self.new))
        tests = [p for p in pairs if p['role'] == 'test']
        tests[0]['before']['surface']['miou'] = None
        tests[1]['after']['surface']['miou'] = None
        result = c.aggregate(tests)
        self.assertEqual(result['surface.miou']['paired_count'], 126)
        self.assertEqual(result['surface.miou']['after_count'], 127)
        self.assertIsNone(result['hidden_surface.small_part_recall']['delta'])
        ds = [.51-.5, -.01, .05, -.05, .0101, -.0101, .0501, -.0501, None]
        self.assertEqual(c.changes([{'surface_miou_delta': d} for d in ds]),
                         {'improved_over_1pp': 3, 'improved_over_5pp': 1,
                          'regressed_over_1pp': 3, 'regressed_over_5pp': 1})


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.old, self.new, self.sweep = fixture()
        self.row = self.new['samples'][0]
        self.geometry = {'face_labels': np.array([1, 2]), 'face_area': np.array([1., 1.]),
                         'voxel_labels': np.array([1, 2]), 'coords': np.array([[0, 0, 0], [16, 0, 0]]),
                         'face_raster': np.tile(np.array([[-1, 0], [1, 0]]), (6, 1, 1))}

    def pair(self):
        return c.make_pair(self.row, record(self.old['samples'][0], self.old), record(self.row, self.new),
                           'before', self.old, self.new)

    def test_npz_plan_seed_shapes_and_coords_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = copy.deepcopy(self.new)
            plan['prediction_dir'] = tmp
            path = Path(tmp) / f'{self.row["instance"]}.npz'
            arrays = {'face_labels': np.array([2, 1]), 'voxel_labels': np.array([1, 2]),
                      'oracle_faces': np.array([1, 2]), 'oracle_voxels': np.array([1, 2]),
                      'seed': np.array(c.expected_seed(plan, self.row['instance'])),
                      'plan_digest': np.array(c.digest(plan))}
            np.savez(path, **arrays)
            self.assertEqual(c.load_prediction(plan, self.row, self.geometry)['face_labels'].tolist(), [2, 1])
            for key, value in [('plan_digest', np.array('other')), ('seed', np.array(7)),
                               ('face_labels', np.array([1.])), ('oracle_faces', np.array([64, 1])),
                               ('coords', np.array([[1, 0, 0], [16, 0, 0]]))]:
                np.savez(path, **dict(arrays, **{key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    c.load_prediction(plan, self.row, self.geometry)
            # Sweep predictions have no oracle and live in the named subdirectory.
            (Path(tmp) / c.BASELINE).mkdir()
            sweep_arrays = {k: v for k, v in arrays.items() if not k.startswith('oracle')}
            np.savez(Path(tmp) / c.BASELINE / path.name, **sweep_arrays)
            self.assertNotIn('oracle_faces', c.load_prediction(plan, self.row, self.geometry, sweep=True))

    def test_four_rows_render_raw_labels_and_exact_gt_crop(self):
        from vae_v2.visuals import render_palette
        pair = self.pair()
        old = {'face_labels': np.array([2, 1])}
        new = {'face_labels': np.array([0, 2]), 'oracle_faces': np.array([1, 1])}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'panel.png'
            c.render_panel(path, pair, self.geometry, old, new)
            palette = render_palette(64)
            with Image.open(path) as image:
                self.assertEqual(image.size, (1536, 1160))
                for row, label in enumerate([1, 2, 0, 1]):
                    self.assertEqual(image.getpixel((200, 40+row*280+24+30)), tuple(palette[label+1]))
            folder = Path(tmp) / 'objects' / self.row['instance']
            folder.mkdir(parents=True)
            tiles = c.gt_tiles(self.geometry)
            gt = Image.new('RGB', (1536, 890), 'white')
            for i, tile in enumerate(tiles):
                gt.paste(tile, (i*256, 74))
            target = folder / f'{self.row["instance"]}_comparison.png'
            gt.save(target)
            c.verify_gt_views(tmp, self.row['instance'], tiles)
            gt.putpixel((200, 100), (1, 2, 3))
            gt.save(target)
            with self.assertRaisesRegex(ValueError, 'GT views'):
                c.verify_gt_views(tmp, self.row['instance'], tiles)

    def test_gt_part_geometry_and_correct_relative_glb_links(self):
        pair = self.pair()
        c.verify_parts(pair, self.geometry)
        bad = dict(self.geometry, face_area=np.array([2., 1.]))
        with self.assertRaisesRegex(ValueError, 'part labels/area'):
            c.verify_parts(pair, bad)
        links = c.artifact_links(pair, '/runs/old', '/runs/new')
        self.assertEqual(links['old DiT GLB'], f'../old/objects/{self.row["instance"]}/{self.row["instance"]}_dit.glb')
        self.assertEqual(links['new DiT GLB'], f'objects/{self.row["instance"]}/{self.row["instance"]}_dit.glb')
        pair['before_source'] = 'validation_baseline'
        self.assertEqual(c.artifact_links(pair, '/runs/old', '/runs/new', '/sweep')['old DiT GLB (validation baseline)'],
                         f'../../sweep/objects/{self.row["instance"]}/{c.BASELINE}.glb')

    def test_exact_oracle_face_and_voxel_equality_required(self):
        old = {'oracle_faces': np.array([1, 2]), 'oracle_voxels': np.array([1, 2])}
        c.verify_oracles(self.row['instance'], old, copy.deepcopy(old))
        for key in old:
            new = copy.deepcopy(old)
            new[key][0] = 2
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                c.verify_oracles(self.row['instance'], old, new)

    def test_html_none_values_separate_diagnostics_and_paths(self):
        pairs = c.prepare_pairs(self.old, self.new, merged(self.old), merged(self.new))
        for pair in pairs:
            pair['links'] = c.artifact_links(pair, '/runs/old', '/runs/new')
        document = c.html_document(c.build_report(pairs, self.old, self.new))
        for text in ('Guitars', 'Historical training diagnostics', 'Largest test regressions',
                     'All 128 test objects', 'Validation diagnostics', 'N/A', 'Developmental regression',
                     'One fixed generation seed', '_dit.glb', '_vae.glb'):
            self.assertIn(text, document)

    def test_unmerged_run_refused_before_output_or_geometry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'load_geometry') as loader:
            with self.assertRaises(ValueError):
                c.compare_runs(Path(tmp) / 'old', Path(tmp) / 'new')
            loader.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])


class GeometrySplitTests(unittest.TestCase):
    def test_manifest_split_and_preparation_split_are_distinct(self):
        row={'instance':'a'*32,'split':'train','source_bytes':100,'source_mtime_ns':200}
        source={'instance':'a'*32,'split':'train','preparation_split':'test','bytes':100,'mtime_ns':200}
        meta={'instance':'a'*32,'split':'test','num_classes':64}
        c.verify_geometry_metadata(meta,row,source)
        with self.assertRaisesRegex(ValueError,'Current source manifest split'):
            c.verify_geometry_metadata(meta,dict(row,split='test'),source)
        with self.assertRaisesRegex(ValueError,'preparation split'):
            c.verify_geometry_metadata(dict(meta,split='train'),row,source)
        with self.assertRaisesRegex(ValueError,'file identity'):
            c.verify_geometry_metadata(meta,row,dict(source,bytes=999))


if __name__ == '__main__':
    unittest.main()
