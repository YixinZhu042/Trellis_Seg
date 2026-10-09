import copy
import unittest
from scripts.test_seg_vae_v2_parts64 import select_rows, CASES
from vae_v2.config import digest


class TestGalleryTests(unittest.TestCase):
    def fixture(self):
        rows=[{'instance':f'{s}_{i:03d}','split':s} for s in ('test','validation') for i in range(128)]
        rows += [{'instance':u,'split':'train'} for u in CASES]
        manifest={'samples':rows}
        selected={'split':'validation','test_used_for_selection':False,'smoke_only':False,
                  'ids':[r['instance'] for r in rows if r['split']=='validation'],
                  'manifest_digest':digest(manifest)}
        return manifest,selected

    def test_all_tests_and_separate_training_diagnostics(self):
        m,s=self.fixture()
        rows=select_rows(m,s)
        self.assertEqual(len(rows),131)
        self.assertEqual(sum(r['role']=='test' for r in rows),128)
        self.assertTrue(all(r['split']=='train' for r in rows if r['role']=='diagnostic'))
        self.assertNotIn('role',m['samples'][0])

    def test_wrong_selection_rejected(self):
        m,s=self.fixture()
        for change in ({'test_used_for_selection':True},{'smoke_only':True},
                       {'manifest_digest':'wrong'},{'ids':s['ids'][:-1]}, {'split':'test'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                select_rows(m,dict(s,**change))

    def test_changed_test_count_rejected(self):
        m,s=self.fixture();m=copy.deepcopy(m)
        m['samples']=m['samples'][1:]
        s['manifest_digest']=digest(m)
        with self.assertRaises(ValueError):select_rows(m,s)


if __name__=='__main__':unittest.main()
