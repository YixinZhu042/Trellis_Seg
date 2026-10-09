import copy
import unittest
from scripts.compare_seg_vae_v2_parts64 import paired_records


class PairedComparisonTests(unittest.TestCase):
    def fixture(self):
        def row(uid, role, score):
            return {'instance':uid,'role':role,'split':'test' if role=='test' else 'train',
                    'type_id':'chair','parts':[{'label':1,'area_fraction':1.}],
                    'surface':{'miou':score,'accuracy':score}}
        a={'manifest_digest':'same','ids':['a'],'samples':[row('a','test',.8)],
           'diagnostics':[row('b','diagnostic',.9)]}
        b=copy.deepcopy(a);b['samples'][0]['surface']['miou']=.7
        return a,b

    def test_matched_identity_and_signed_delta(self):
        a,b=self.fixture();pairs=paired_records(a,b)
        self.assertEqual(len(pairs),2)
        self.assertAlmostEqual(pairs[0]['surface_miou_delta'],-.1)
        self.assertEqual(pairs[1]['role'],'diagnostic')

    def test_changed_manifest_and_cohort_rejected(self):
        a,b=self.fixture()
        for patch in ({'manifest_digest':'different'},{'ids':['a','a']},{'samples':[]}):
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                paired_records(a,dict(b,**patch))

    def test_changed_gt_or_role_rejected(self):
        a,b=self.fixture();b['samples'][0]['parts'][0]['area_fraction']=.9
        with self.assertRaises(ValueError):paired_records(a,b)
        a,b=self.fixture();b['samples'][0]['role']='diagnostic'
        with self.assertRaises(ValueError):paired_records(a,b)


if __name__=='__main__':unittest.main()
