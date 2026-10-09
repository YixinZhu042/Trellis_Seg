import copy
import unittest
from pathlib import Path
import numpy as np
import torch

from vae_v2.augmentation import label_permutation, permute_sample
from vae_v2.config import load_config, permutation_resume_compatible, resume_metadata_compatible
from vae_v2.data import cache_contract, to_device
from vae_v2.geometry import build_geometry, attach_targets, training_queries
from vae_v2.losses import reconstruction_losses

ROOT=Path(__file__).resolve().parents[1]


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(ROOT/'configs/seg_vae_v2_parts64.json')
        self.new=load_config(ROOT/'configs/seg_vae_v2_parts64_permute.json')

    def test_bijection_all_channels_zero_fixed_and_stateless_resume(self):
        p=label_permutation(64,20260923,120000)
        np.testing.assert_array_equal(np.sort(p),np.arange(64))
        self.assertEqual(p[0],0)
        np.testing.assert_array_equal(p,label_permutation(64,20260923,120000))
        self.assertFalse(np.array_equal(p,label_permutation(64,20260923,120001)))
        seen=set()
        for index in range(1000):seen.update(label_permutation(64,7,index)[1:3].tolist())
        self.assertEqual(seen,set(range(1,64)))

    def test_non_involutive_mapping_and_no_mutation(self):
        p=np.array([0,2,3,1])  # three-cycle detects forward/inverse indexing bugs
        soft=np.array([[.1,.2,.3,.4],[0,.5,.5,0]],np.float32)
        s={'voxel_target':soft,'voxel_labels':np.array([3,1]),
           'face_labels':np.array([0,1,2,3]),'coords':np.arange(6).reshape(2,3)}
        before=copy.deepcopy(s);a=permute_sample(s,p)
        np.testing.assert_array_equal(a['voxel_target'][:,p],soft)
        np.testing.assert_array_equal(a['face_labels'],p[s['face_labels']])
        np.testing.assert_array_equal(a['voxel_labels'],p[s['voxel_labels']])
        self.assertIs(a['coords'],s['coords'])
        restored=permute_sample(a,np.argsort(p))
        for k in s:
            np.testing.assert_array_equal(s[k],before[k]);np.testing.assert_array_equal(restored[k],s[k])
        for bad in ([1,0,2,3],[0,2,2,3],[0.,1.,2.,3.]):
            with self.assertRaises(ValueError):permute_sample(s,bad)

    def test_all_losses_invariant_under_consistent_renaming(self):
        gcfg=dict(self.cfg['geometry'],area_samples=100,max_voxels=500)
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.001],[0,1,.001],[1,0,.001]],np.float32)
        s=attach_targets(build_geometry(v,np.array([[0,1,2],[3,4,5]]),gcfg,1),np.array([1,2]),64)
        s['face_pixels']=np.array([100,50]);s['near_pairs']=np.array([[0,1]])
        p=label_permutation(64,7,120000);a=permute_sample(s,p)
        tcfg=dict(self.cfg['train'],query_faces=16,pair_count=8)
        q=training_queries(s,gcfg,tcfg,np.random.default_rng(10))
        aq=training_queries(a,gcfg,tcfg,np.random.default_rng(10))
        for key in q:
            if key=='query_labels':np.testing.assert_array_equal(aq[key],p[q[key]])
            else:np.testing.assert_allclose(aq[key],q[key])
        logits=torch.randn(len(s['coords']),64,requires_grad=True)
        point=torch.randn(len(q['query_labels']),4,64,requires_grad=True)
        face=torch.logsumexp(point.log_softmax(-1),dim=1)-np.log(4)
        output={'voxel_logits':logits,'point_logits':point,'face_logprob':face,
                'mean':torch.randn(3,32),'logvar':torch.randn(3,32)}
        swapped={k:(x[...,np.argsort(p)] if k in ('voxel_logits','point_logits','face_logprob') else x) for k,x in output.items()}
        loss,terms=reconstruction_losses(output,to_device({'voxel_target':s['voxel_target']},'cpu'),to_device(q,'cpu'),self.cfg['loss'],1e-5)
        aloss,aterms=reconstruction_losses(swapped,to_device({'voxel_target':a['voxel_target']},'cpu'),to_device(aq,'cpu'),self.cfg['loss'],1e-5)
        for k in terms:torch.testing.assert_close(terms[k],aterms[k],rtol=1e-5,atol=1e-6)
        loss.backward(retain_graph=True);original=logits.grad.clone();logits.grad.zero_()
        aloss.backward();torch.testing.assert_close(logits.grad,original,rtol=1e-5,atol=1e-6)

    def test_only_explicit_resume_fork_allows_augmentation(self):
        old={'config':self.cfg,'manifest_digest':'same','world_size':24,'validation_ids':['v']}
        new=dict(old,config=self.new)
        self.assertFalse(resume_metadata_compatible(old,new))
        self.assertTrue(permutation_resume_compatible(old,new))
        self.assertTrue(resume_metadata_compatible(new,new))
        self.assertFalse(permutation_resume_compatible(new,old))
        for key,value in [('world_size',8),('manifest_digest','different'),('validation_ids',['x'])]:
            self.assertFalse(permutation_resume_compatible(old,dict(new,**{key:value})))
        for section,key,value in [('train','lr',1e-3),('model','latent_channels',64),('loss','surface_ce',1.0)]:
            bad=copy.deepcopy(new);bad['config'][section][key]=value
            self.assertFalse(permutation_resume_compatible(old,bad))
        self.assertEqual(cache_contract(self.cfg),cache_contract(self.new))


if __name__=='__main__':unittest.main()
