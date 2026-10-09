import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from utils.dit_vae_v2 import (align_shape,moments,normalization,validate_latents,
                              normalize_tensors,unnormalize,capacity)
from vae_v2.config import digest


class DitVaeV2Tests(unittest.TestCase):
    def fixture(self):
        return {'latent_coords':np.array([[1,2,3],[4,5,6]],np.int32),
                'latent_feats':np.ones((2,32),np.float32),'mv_coords':np.array([[31,0,0],[0,31,0]],np.int32),
                'mv_feats':np.ones((2,32),np.float32),'shape_feats':np.r_[np.ones((1,32)),np.zeros((1,32))].astype(np.float32),
                'shape_valid':np.array([True,False]),'view_offsets':np.array([0,1,2,2,2,2,2])}

    def test_exact_shape_join_and_missing(self):
        target=np.array([[1,2,3],[4,5,6],[0,0,0]])
        coords=np.array([[4,5,6],[1,2,3]])
        feats=np.r_[np.ones((1,32))*4,np.ones((1,32))*7]
        aligned,valid=align_shape(target,coords,feats)
        np.testing.assert_array_equal(valid,[True,True,False])
        np.testing.assert_array_equal(aligned[:,0],[7,4,0])
        with self.assertRaises(ValueError):align_shape(target,np.array([[1,2,3],[1,2,3]]),feats)
        with self.assertRaises(ValueError):align_shape(target.astype(float),coords,feats)

    def test_normalization_uses_only_training_tokens(self):
        x=np.repeat(np.array([[1],[3]],float),32,axis=1)
        data=[{'split':'train','moments':{k:moments(x) for k in ('latent_feats','shape_feats','mv_feats')}},
              {'split':'test','moments':{k:moments(x*10000) for k in ('latent_feats','shape_feats','mv_feats')}},
              {'split':'validation','moments':{k:moments(x*20000) for k in ('latent_feats','shape_feats','mv_feats')}}]
        norm=normalization(data)
        for v in norm.values():
            np.testing.assert_allclose(v['mean'],2);np.testing.assert_allclose(v['std'],1)
            self.assertEqual(v['tokens'],2)
        fixture=self.fixture();normalized=normalize_tensors(fixture,norm,'cpu')
        torch.testing.assert_close(normalized['shape_feats'][1],torch.zeros(32))
        torch.testing.assert_close(unnormalize(normalized['latent_feats'],norm),torch.tensor(fixture['latent_feats']))

    def test_cache_validation_no_truncation_or_nonfinite(self):
        data=self.fixture();validate_latents(data,2)
        with self.assertRaises(ValueError):validate_latents(data,1)
        for key,mutate in [('latent_coords',lambda x:x.__setitem__((0,0),32)),
                           ('latent_feats',lambda x:x.__setitem__((0,0),np.nan)),
                           ('view_offsets',lambda x:x.__setitem__(-1,3)),
                           ('shape_feats',lambda x:x.__setitem__((1,0),1))]:
            bad=copy.deepcopy(data);mutate(bad[key])
            with self.subTest(key=key),self.assertRaises(ValueError):validate_latents(bad,100)

    def test_capacity_extension_never_alters_original_plan(self):
        plan={'config':{'max_tokens':8192}}
        with tempfile.TemporaryDirectory() as tmp:
            cfg={'cache_dir':tmp,'max_tokens':16384}
            record={'plan_digest':digest(plan),'original_max_tokens':8192,'max_tokens':16384,'resampling':False}
            p=Path(tmp)/'capacity.json';p.write_text(json.dumps(record))
            self.assertEqual(capacity(cfg,plan),16384)
            self.assertEqual(plan['config']['max_tokens'],8192)
            record['resampling']=True;p.write_text(json.dumps(record))
            with self.assertRaises(ValueError):capacity(cfg,plan)


if __name__=='__main__':unittest.main()
