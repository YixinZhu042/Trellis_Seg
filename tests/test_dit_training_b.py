import copy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from utils.dit_training_b import (object_order,training_permutation,synchronized_targets,
                                  checked_fork_state,validate_options,selected_row_index,POLICY)
from utils.dit_vae_v2 import config
from vae_v2.config import digest
from vae_v2.geometry import face_sampling_probability
from vae_v2.losses import macro_surface_ce

ROOT=Path(__file__).resolve().parents[1]
AUG={'policy':POLICY,'probability':.5,'normalization':'preserve_base_train_statistics'}
SAMPLING={'policy':'part_count_stratified_v1','bin_probabilities':[.7,.15,.15]}


class TrainingBTests(unittest.TestCase):
    def test_explicit_smoke_single_bin_does_not_need_production_order(self):
        rows=[{'split':'train','parts':1},{'split':'train','parts':5}]
        self.assertEqual(selected_row_index(rows,None,0,3,True),1)
        self.assertEqual(selected_row_index(rows,np.array([1,0]),0,3,False),1)

    def test_config_keeps_base_statistics_and_best_regular_source(self):
        cfg,_=config(ROOT/'configs/dit_vae_v2_parts64_b_smallparts.json')
        old,_=config(ROOT/'configs/dit_vae_v2_parts64.json')
        self.assertEqual(cfg['cache_dir'],old['cache_dir'])
        self.assertEqual(cfg['seed'],old['seed'])
        self.assertTrue(cfg['dit_init'].endswith('regular_step004500.pt'))
        self.assertNotEqual(cfg['output_dir'],old['output_dir'])
        self.assertEqual(cfg['train']['ema_rate'],.999)
        self.assertEqual(cfg['train']['aux_weights'],old['train']['aux_weights'])

    def test_sampling_bin_counts_global_ddp_and_resume(self):
        rows=[{'split':'train','parts':p} for p,n in [(10,1700),(25,200),(50,100)] for _ in range(n)]
        a=object_order(rows,7,3,SAMPLING)
        self.assertEqual(len(a),len(rows))
        self.assertEqual([sum(lo<=rows[i]['parts']<=hi for i in a) for lo,hi in [(1,20),(21,32),(33,63)]],[1400,300,300])
        np.testing.assert_array_equal(a,object_order(rows,7,3,SAMPLING))
        self.assertFalse(np.array_equal(a,object_order(rows,7,4,SAMPLING)))
        # Rank-striding and resuming a global sample index reproduce the sequence.
        interleaved=np.empty_like(a)
        for rank in range(24):interleaved[rank::24]=a[rank::24]
        np.testing.assert_array_equal(interleaved,a)
        np.testing.assert_array_equal(object_order(rows,7,3,SAMPLING)[1000:],a[1000:])
        # Largest bin is never exhausted; it should not repeat in this draw epoch.
        low=[i for i in a if rows[i]['parts']<=20]
        self.assertEqual(len(low),len(set(low)))
        with self.assertRaises(ValueError):object_order(rows+[{'split':'test','parts':10}],7,0,SAMPLING)
        with self.assertRaises(ValueError):object_order(rows[:20],7,0,SAMPLING)

    def test_augmentation_half_stateless_all_channels(self):
        draws=[training_permutation(7,i,AUG) for i in range(2000)]
        self.assertTrue(850<sum(p is not None for p in draws)<1150)
        seen=set()
        for i,p in enumerate(draws):
            if p is None:self.assertIsNone(training_permutation(7,i,AUG))
            else:
                np.testing.assert_array_equal(p,training_permutation(7,i,AUG))
                self.assertEqual(p[0],0)
                np.testing.assert_array_equal(np.sort(p),np.arange(64))
                seen.add(int(p[1]))
        self.assertEqual(seen,set(range(1,64)))
        with self.assertRaises(ValueError):training_permutation(7,1,AUG,'validation')

    def test_same_bijection_all_views_targets_unknown_and_background(self):
        sample={'voxel_target':np.eye(64,dtype=np.float32)[[0,1,7,60]],
                'voxel_labels':np.array([0,1,7,60]),'face_labels':np.array([0,1,7,60]),
                'face_raster':np.tile(np.array([[-1,0,1,2,3]],np.int32),(6,1,1))}
        before=copy.deepcopy(sample);p=np.arange(64);p[[1,7,60]]=[7,60,1]
        altered,maps=synchronized_targets(sample,p)
        for v in range(6):np.testing.assert_array_equal(maps[v],[[0,1,8,61,2]])
        np.testing.assert_array_equal(altered['voxel_target'][:,p],sample['voxel_target'])
        np.testing.assert_array_equal(altered['face_labels'],p[sample['face_labels']])
        for k in sample:np.testing.assert_array_equal(sample[k],before[k])

    def test_macro_ce_gives_tiny_and_large_parts_equal_expected_weight(self):
        labels=np.array([1,1,60]);area=np.array([99.,.9,.1])
        draw=face_sampling_probability(labels,area,.75)
        macro=face_sampling_probability(labels,area,1.)
        self.assertAlmostEqual(float(macro[:2].sum()),.5)
        self.assertAlmostEqual(float(macro[2]),.5)
        # All faces with known probabilities: exact expectation of MC estimator.
        logits=torch.zeros(3,64,requires_grad=True)
        scores=logits+torch.tensor([[0.]*64,[0.]*64,[0.]*64])
        with torch.no_grad():scores[2,60]=-4
        y=torch.tensor(labels)
        ce=F.cross_entropy(scores,y,reduction='none')
        estimate=sum(float(draw[i])*macro_surface_ce(scores[i:i+1],y[i:i+1],torch.tensor([macro[i]/draw[i]])) for i in range(3))
        expected=.5*(ce[0]*99/99.9+ce[1]*.9/99.9)+.5*ce[2]
        torch.testing.assert_close(estimate,expected.to(estimate.dtype))
        self.assertGreater(float(estimate),float((ce*torch.tensor(area/area.sum())).sum()))
        estimate.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_macro_invariant_to_non_involutive_id_renaming(self):
        x=torch.randn(12,64,requires_grad=True);y=torch.tensor([1,7,60]*4)
        weight=torch.linspace(.1,2,12);p=np.arange(64);p[[1,7,60]]=[7,60,1]
        original=macro_surface_ce(x,y,weight)
        renamed=macro_surface_ce(x[:,np.argsort(p)],torch.tensor(p[y.numpy()]),weight)
        torch.testing.assert_close(original,renamed)
        a=torch.autograd.grad(original,x,retain_graph=True)[0]
        b=torch.autograd.grad(renamed,x)[0];torch.testing.assert_close(a,b)

    def test_metadata_fork_rejects_different_vae_cache_step_or_smoke(self):
        cfg={'initialization':{'mode':'fork_weights_only','expected_step':4500,'expected_kind':'regular'}}
        manifest={'data':'same'}
        checkpoint={'model':{'x':torch.ones(2)},'step':4500,'kind':'regular',
                    'metadata':{'cache_digest':digest(manifest),'vae_sha256':'v','base_digest':'b','smoke_only':False}}
        state,report=checked_fork_state(checkpoint,cfg,manifest,'v','b')
        self.assertIs(state,checkpoint['model']);self.assertTrue(report['optimizer_reset'])
        for key,value in [('vae_sha256','other'),('cache_digest','other'),('smoke_only',True),('base_digest','other')]:
            bad=copy.deepcopy(checkpoint);bad['metadata'][key]=value
            with self.assertRaises(ValueError):checked_fork_state(bad,cfg,manifest,'v','b')
        with self.assertRaises(ValueError):checked_fork_state(dict(checkpoint,step=5000),cfg,manifest,'v','b')
        with self.assertRaises(ValueError):checked_fork_state({'x':torch.ones(2)},cfg,manifest,'v','b')

    def test_invalid_hyperparameters(self):
        for change in [dict(AUG,probability=float('nan')),dict(AUG,normalization='new_stats')]:
            with self.assertRaises(ValueError):validate_options({'label_augmentation':change})
        for p in [[1,0,0],[.5,.5,.5],[float('nan'),.1,.1]]:
            with self.assertRaises(ValueError):validate_options({'object_sampling':dict(SAMPLING,bin_probabilities=p)})


if __name__=='__main__':unittest.main()
