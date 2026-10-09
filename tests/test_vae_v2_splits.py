"""Split changes must preserve cache identity and exclude only accounted failures."""
import copy
import unittest
from vae_v2.splits import assign_splits, seal_split_manifest, validate_split_manifest, POLICY


class SplitTests(unittest.TestCase):
    def fixtures(self):
        source = [{'instance': f'{split}_{i}', 'split': split, 'source': f'{split}/{i}',
                   'source_bytes': 100, 'source_mtime_ns': 1}
                  for split in ('train', 'validation', 'test') for i in range(5)]
        bad = ['train_1', 'validation_2']
        rows = [dict(r, path=r['instance']+'.npz', bytes=10, mtime_ns=2, reused=True)
                for r in source if r['instance'] not in bad]
        failures = [{'instance': uid, 'error': 'Missing matching geometry'} for uid in bad]
        spec = {'policy': POLICY, 'seed': 20260929, 'validation': 2, 'test': 2, 'quarantine_count': 2}
        return source, rows, failures, spec

    def test_deterministic_disjoint_nested_holdouts_and_remainder_train(self):
        source, rows, failures, spec = self.fixtures()
        a = assign_splits(source, [r['instance'] for r in failures], spec)
        b = assign_splits(list(reversed(source)), [r['instance'] for r in failures], spec)
        self.assertEqual(a, b)
        self.assertEqual(sum(s=='validation' for s in a.values()), 2)
        self.assertEqual(sum(s=='test' for s in a.values()), 2)
        self.assertEqual(sum(s=='train' for s in a.values()), 9)
        self.assertTrue(all(uid.startswith(s+'_') for uid,s in a.items() if s!='train'))

    def test_seal_preserves_old_split_and_does_not_mutate_input(self):
        source, rows, failures, spec = self.fixtures()
        original = copy.deepcopy(rows)
        m = seal_split_manifest({}, source, rows, failures, spec, {})
        validate_split_manifest(m, source, spec)
        self.assertEqual(rows, original)
        for row in m['samples']:
            self.assertEqual(row['bytes'], 10)
            self.assertEqual(row['mtime_ns'], 2)
            self.assertEqual(row['preparation_split'], next(s['split'] for s in source if s['instance']==row['instance']))
            self.assertNotIn('reused', row)
        self.assertEqual(len(m['excluded']), 2)

    def test_membership_tampering_and_unaccounted_objects_rejected(self):
        source, rows, failures, spec = self.fixtures()
        with self.assertRaises(ValueError):
            seal_split_manifest({}, source, rows[:-1], failures, spec, {})
        m = seal_split_manifest({}, source, rows, failures, spec, {})
        held = next(r for r in m['samples'] if r['split']=='test')
        held['split']='train'
        with self.assertRaises(ValueError):
            validate_split_manifest(m, source, spec)

    def test_changed_exclusion_provenance_and_insufficient_holdouts_rejected(self):
        source, rows, failures, spec = self.fixtures()
        m = seal_split_manifest({}, source, rows, failures, spec, {})
        m['excluded'][0]['source_bytes']+=1
        with self.assertRaises(ValueError):
            validate_split_manifest(m, source, spec)
        spec['test']=6
        with self.assertRaises(ValueError):
            assign_splits(source, [r['instance'] for r in failures], spec)


if __name__ == '__main__':
    unittest.main()
