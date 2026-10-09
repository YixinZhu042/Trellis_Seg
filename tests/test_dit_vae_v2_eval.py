import copy
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.evaluate_dit_vae_v2 import select_cohort,normalize_conditions,generate_latent,DIAGNOSTICS


class EvalTests(unittest.TestCase):
    def fixture(self):
        val=[f'{i:032x}' for i in range(128)]
        tests=[f'{i:032x}' for i in range(128,256)]
        m={'samples':[{'instance':u,'split':s} for s,ids in [('validation',val),('test',tests),('train',DIAGNOSTICS)] for u in ids]}
        selection={'split':'validation','test_used_for_selection':False,'ids':val,'checkpoint':'ckpt',
                   'step':4500,'kind':'regular','sampling_steps':20,'summary':{'surface':{'miou':.8}}}
        report=dict(selection,samples=[{'instance':u,'surface':{'miou':.7 if i==7 else .8}} for i,u in enumerate(val)])
        return m,selection,report

    def test_fixed_128_tests_train_diagnostics_and_separate_validation_parity(self):
        m,s,r=self.fixture();rows=select_cohort(m,s,r)
        self.assertEqual(sum(x['role']=='test' for x in rows),128)
        self.assertEqual(sum(x['role']=='train_diagnostic' for x in rows),3)
        self.assertEqual(sum(x['role']=='validation_parity' for x in rows),3)
        self.assertTrue(all(x['split']=='validation' for x in rows if x['role']=='validation_parity'))
        self.assertEqual(len(rows),len({x['instance'] for x in rows}))

    def test_invalid_selection_and_reference_rejected(self):
        m,s,r=self.fixture()
        for changed in [dict(s,test_used_for_selection=True),dict(s,ids=s['ids'][:-1]),dict(s,step=5000)]:
            with self.assertRaises(ValueError):select_cohort(m,changed,r)
        bad=copy.deepcopy(r);bad['samples'][0]['instance']='wrong'
        with self.assertRaises(ValueError):select_cohort(m,s,bad)

    def test_validation_diagnostics_never_enter_test(self):
        m,s,r=self.fixture();uid=s['ids'][5]
        rows=select_cohort(m,s,r,[uid])
        self.assertEqual(sum(x['role']=='test' for x in rows),128)
        self.assertEqual(next(x for x in rows if x['instance']==uid)['role'],'validation_diagnostic')
        with self.assertRaises(ValueError):select_cohort(m,s,r,[uid,uid])
        with self.assertRaises(ValueError):select_cohort(m,s,r,[f'{128:032x}'])

    def test_generation_does_not_depend_on_gt_latent_values(self):
        class Template:
            device='cpu'
            def replace(self,x):return SimpleNamespace(feats=x,device='cpu')
        class Model:
            def __call__(self,x,t,**conditions):return SimpleNamespace(feats=torch.ones_like(x.feats))
        data={'latent_coords':np.zeros((3,3),np.int32),'latent_feats':None,
              'shape_feats':np.ones((3,32),np.float32),'shape_valid':np.array([True,False,True]),
              'mv_feats':np.ones((2,32),np.float32),'mv_coords':np.zeros((2,3),np.int32)}
        stats={k:{'mean':[2.]*32,'std':[2.]*32} for k in ('shape_feats','mv_feats')}
        norm=normalize_conditions(data,stats,'cpu')
        torch.testing.assert_close(norm['shape_feats'][1],torch.zeros(32))
        with patch('torch.autocast',return_value=nullcontext()):
            x=generate_latent(Model(),Template(),data,stats,42,20,'cpu')
            data['latent_feats']=np.full((3,32),np.nan)
            y=generate_latent(Model(),Template(),data,stats,42,20,'cpu')
        torch.testing.assert_close(x,y)
        expected=torch.randn((3,32),generator=torch.Generator().manual_seed(42))-1
        torch.testing.assert_close(x,expected)


if __name__=='__main__':unittest.main()
