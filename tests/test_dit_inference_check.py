import copy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from utils.dit_inference_check import experiments,shuffled_view_features,guided_velocity,generate,metric_difference


class InferenceCheckTests(unittest.TestCase):
    def fixture(self):
        return {'latent_coords':np.zeros((3,3),np.int32),'latent_feats':None,
                'shape_feats':np.ones((3,32),np.float32),'shape_valid':np.array([True,False,True]),
                'mv_feats':np.repeat(np.arange(12,dtype=np.float32)[:,None],32,axis=1),
                'mv_coords':np.arange(36,dtype=np.int32).reshape(12,3)%32,
                'view_offsets':np.arange(0,13,2,dtype=np.int32)}

    def test_experiments_fixed_and_ablations_ineligible(self):
        settings=experiments()
        self.assertEqual(len(settings),10)
        self.assertEqual(len({s['name'] for s in settings}),10)
        self.assertEqual({(s['steps'],s['cfg']) for s in settings if s['eligible']},
                         {(n,c) for n in (20,50) for c in (1.,1.5,2.,3.)})
        self.assertTrue(all(not s['eligible'] for s in settings if s['condition']!='correct'))

    def test_shuffle_preserves_each_view_not_spatial_mapping(self):
        data=self.fixture();before=copy.deepcopy(data)
        features,order=shuffled_view_features(data,5)
        np.testing.assert_array_equal(features,data['mv_feats'][order])
        self.assertFalse(np.array_equal(features,data['mv_feats']))
        for start,end in zip(data['view_offsets'][:-1],data['view_offsets'][1:]):
            np.testing.assert_array_equal(np.sort(order[start:end]),np.arange(start,end))
        for k in ('shape_feats','mv_feats','mv_coords','view_offsets'):
            np.testing.assert_array_equal(data[k],before[k])
        np.testing.assert_array_equal(shuffled_view_features(data,5)[1],order)

    def test_cfg_endpoints_and_direction(self):
        c=torch.tensor([3.]);u=torch.tensor([1.])
        self.assertIs(guided_velocity(c,u,1),c)
        self.assertIs(guided_velocity(c,u,0),u)
        torch.testing.assert_close(guided_velocity(c,u,2),torch.tensor([5.]))
        with self.assertRaises(ValueError):guided_velocity(c,u,float('nan'))

    def test_same_noise_shape_retained_and_no_gt_dependency(self):
        class Template:
            device='cpu'
            def replace(self,x):return SimpleNamespace(feats=x,device='cpu')
        class Model:
            def __init__(self):self.first=None;self.calls=0
            def __call__(self,x,t,**cond):
                if self.first is None:self.first=x.feats.clone()
                self.calls+=1
                self.assert_shape(cond['concat_cond'].feats)
                value=1. if cond['mv_cond'][0] is None else 3.
                return SimpleNamespace(feats=torch.full_like(x.feats,value))
            def assert_shape(self,x):
                torch.testing.assert_close(x,torch.tensor([[1.]*32,[0.]*32,[1.]*32]))
        data=self.fixture();stats={k:{'mean':[0.]*32,'std':[1.]*32} for k in ('shape_feats','mv_feats')}
        initial=torch.randn((3,32),generator=torch.Generator().manual_seed(42))
        for mode,scale,expected,nfe in [('correct',1.,3.,4),('correct',2.,5.,8),('no_mv',1.,1.,4),('shuffle_within_view',1.,3.,4)]:
            model=Model()
            setting={'condition':mode,'steps':4,'cfg':scale}
            x,calls=generate(model,Template(),data,stats,42,setting,'cpu')
            torch.testing.assert_close(model.first,initial)
            torch.testing.assert_close(x,initial-expected)
            self.assertEqual((calls,model.calls),(nfe,nfe))
            changed=dict(data,latent_feats=np.full((3,32),np.nan))
            y,_=generate(Model(),Template(),changed,stats,42,setting,'cpu')
            torch.testing.assert_close(x,y)

    def test_parity_none_handling(self):
        a={k:{'accuracy':.8,'miou':.7,'small_part_recall':None} for k in ('surface','projection','voxel','hidden_surface')}
        self.assertEqual(metric_difference(a,a),0)
        b=copy.deepcopy(a);b['surface']['miou']=.71
        self.assertAlmostEqual(metric_difference(a,b),.01)
        b['surface']['small_part_recall']=0
        with self.assertRaises(ValueError):metric_difference(a,b)


if __name__=='__main__':unittest.main()
