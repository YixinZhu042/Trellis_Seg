"""VAE v2 contracts that can be exercised without TRELLIS/CUDA."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import torch
from vae_v2.config import ROOT, load_config, STORAGE_ROOT, storage_path, default_run_dir, resume_metadata_compatible
from vae_v2.geometry import build_geometry, attach_targets, SurfaceQueries, near_surface_pairs, face_sampling_probability, training_queries
from vae_v2.data import to_device, validate_sample, training_order, load_manifest, cache_contract
from vae_v2.losses import balanced_soft_ce, kl_weight, relation_losses, reconstruction_losses
from vae_v2.surface import SurfaceHead
from vae_v2.evaluation import classification, surface_diagnostics, selection_score
from vae_v2.checkpoint import stage_checkpoint


class VaeV2Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(ROOT/'configs/seg_vae_v2.json')
        self.settings=dict(self.cfg['geometry'],area_samples=200,max_voxels=1000)
        # Close, disconnected, opposing sheets: exactly the Euclidean KNN failure case.
        self.vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.001],[0,1,.001],[1,0,.001]],np.float32)
        self.faces=np.array([[0,1,2],[3,4,5]])

    def test_geometry_independent_of_gt_and_shells_do_not_mix(self):
        geometry=build_geometry(self.vertices,self.faces,self.settings,42)
        a=attach_targets(geometry,np.array([1,2]));b=attach_targets(geometry,np.array([8,3]))
        qa=SurfaceQueries(a,self.settings).build([0,1]);qb=SurfaceQueries(b,self.settings).build([0,1])
        for key in qa:np.testing.assert_array_equal(qa[key],qb[key])
        # Position-sensitive anchor selection keeps the sheets separate despite voxel collisions.
        self.assertTrue(qa['neighbor_mask'].any(1).all())
        self.assertTrue((qa['neighbor_geometry'][qa['neighbor_mask'],3]>.99).all())

    def test_soft_voxel_targets_preserve_collisions(self):
        g=build_geometry(self.vertices,self.faces,self.settings,42)
        # Controlled collision to verify no majority-vote information loss.
        g['anchor_voxel']=np.zeros_like(g['anchor_voxel']);g['coords']=g['coords'][:1]
        s=attach_targets(g,np.array([1,2]));self.assertGreater(s['voxel_target'][0,1],0)
        self.assertGreater(s['voxel_target'][0,2],0)
        np.testing.assert_allclose(s['voxel_target'].sum(1),1)
        with self.assertRaises(ValueError):attach_targets(g,np.array([1,20]))

    def test_multipoint_head_normalized_and_backward(self):
        g=build_geometry(self.vertices,self.faces,self.settings,42)
        query=to_device(SurfaceQueries(g,self.settings).build([0,1]),'cpu')
        features=torch.randn(len(g['coords']),8,requires_grad=True)
        logits=torch.randn(len(g['coords']),20,requires_grad=True)
        head=SurfaceHead(8,16);points,face=head(features,logits,query)
        self.assertEqual(points.shape,(2,4,20))
        torch.testing.assert_close(face.exp().sum(-1),torch.ones(2))
        (-face[:,1].mean()).backward()
        self.assertGreater(float(logits.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_invalid_neighbors_cannot_change_prediction(self):
        g=build_geometry(self.vertices,self.faces,self.settings,42)
        q=to_device(SurfaceQueries(g,self.settings).build([0]),'cpu')
        q['neighbor_mask'][:,1:]=False
        features=torch.randn(len(g['coords'])+1,8)
        logits=torch.randn(len(g['coords'])+1,20)
        q['neighbor_index'][:,1:]=len(g['coords'])
        head=SurfaceHead(8,8)
        a=head(features,logits,q)[1]
        logits[-1]=10000;features[-1]=10000
        torch.testing.assert_close(head(features,logits,q)[1],a)

    def test_zero_label_ignored_and_small_part_weighted(self):
        labels=torch.tensor([0,1,1,1,2])
        truth=torch.nn.functional.one_hot(labels,20).float()
        logits=torch.zeros(5,20,requires_grad=True)
        loss=balanced_soft_ce(logits,truth);loss.backward()
        self.assertEqual(float(logits.grad[0].abs().sum()),0)
        self.assertGreater(float(logits.grad[4].abs().sum()),float(logits.grad[1].abs().sum()))

    def test_no_smoothing_across_true_boundary(self):
        pairs=torch.tensor([[0,1]])
        logprob=torch.tensor([[.99,.01],[.01,.99]]).log()
        consistency,boundary=relation_losses(logprob,pairs,torch.tensor([False]),torch.ones(1))
        self.assertEqual(float(consistency),0)
        self.assertLess(float(boundary),.1)
        _,collapsed=relation_losses(torch.tensor([[.99,.01],[.99,.01]]).log(),pairs,torch.tensor([False]),torch.ones(1))
        self.assertGreater(float(collapsed),float(boundary))

    def test_near_disconnected_pairs_and_balanced_sampling(self):
        g=build_geometry(self.vertices,self.faces,self.settings,42)
        p=near_surface_pairs(g,8)
        self.assertTrue(any(np.array_equal(x,[0,1]) for x in p))
        probability=face_sampling_probability(np.array([0,1,2]),np.array([100.,99.,1.]),1.)
        np.testing.assert_allclose(probability,[0,.5,.5])

    def test_schedule_and_ablation_configs(self):
        t=self.cfg['train']
        self.assertEqual(kl_weight(t['deterministic_steps'],t),0)
        self.assertAlmostEqual(kl_weight(t['deterministic_steps']+t['kl_warmup_steps'],t),t['kl_weight'])
        for name in ['voxel_baseline','loss_only','latent64','stride8','no_kl']:
            load_config(ROOT/f'configs/seg_vae_v2_{name}.json')

    def test_training_losses_all_terms_finite_with_missing_boundaries(self):
        geometry=build_geometry(self.vertices,self.faces,self.settings,42)
        s=attach_targets(geometry,np.array([1,2]))
        s['face_pixels']=np.array([100,0]);s['near_pairs']=near_surface_pairs(s,8)
        cfg=dict(self.cfg['train'],query_faces=16,pair_count=8)
        query=to_device(training_queries(s,self.settings,cfg,np.random.default_rng(1)),'cpu')
        logits=torch.randn(len(s['coords']),20,requires_grad=True)
        head=SurfaceHead(8,8)
        point,face=head(torch.randn(len(s['coords']),8),logits,query)
        out={'voxel_logits':logits,'point_logits':point,'face_logprob':face,
             'mean':torch.zeros(3,32,requires_grad=True),'logvar':torch.zeros(3,32,requires_grad=True)}
        loss,terms=reconstruction_losses(out,{'voxel_target':torch.tensor(s['voxel_target'])},query,self.cfg['loss'],1e-5)
        self.assertTrue(all(torch.isfinite(t) for t in terms.values()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_only_unannotated_voxels_have_zero_reconstruction_gradient(self):
        logits=torch.randn(7,20,requires_grad=True)
        truth=torch.zeros(7,20);truth[:,0]=1
        loss=balanced_soft_ce(logits,truth);loss.backward()
        self.assertEqual(float(loss),0)
        self.assertEqual(float(logits.grad.abs().sum()),0)

    def test_same_surface_disagreement_is_penalized(self):
        pairs=torch.tensor([[0,1]])
        same=torch.tensor([True]);weight=torch.ones(1)
        disagree=torch.tensor([[.99,.01],[.01,.99]]).log()
        agree=torch.tensor([[.99,.01],[.99,.01]]).log()
        a,_=relation_losses(disagree,pairs,same,weight)
        b,_=relation_losses(agree,pairs,same,weight)
        self.assertGreater(float(a),float(b))

    def test_metrics_ignore_zero_and_missing_boundaries(self):
        result=classification(np.array([0,1,1]),np.array([9,1,1]),np.ones(3))
        self.assertEqual(result['miou'],1.)
        self.assertIsNone(result['small_part_recall'])
        sample={'face_labels':np.array([1,1]),'face_area':np.ones(2),
                'face_pixels':np.ones(2),'adjacency':np.array([[0,1]]),'near_pairs':np.empty((0,2),int)}
        correct=surface_diagnostics(sample,np.array([1,1]))
        incorrect=surface_diagnostics(sample,np.array([1,2]))
        self.assertIsNone(correct['boundary_f1'])
        self.assertEqual(correct['false_internal_boundary_rate'],0)
        self.assertEqual(incorrect['false_internal_boundary_rate'],1)

    def test_isolated_error_area_and_selection_order(self):
        sample={'face_labels':np.array([1,1,1]),'face_area':np.array([1.,1.,.0001]),
                'face_pixels':np.ones(3),'adjacency':np.array([[0,1],[1,2]]),'near_pairs':np.empty((0,2),int)}
        result=surface_diagnostics(sample,np.array([1,1,2]))
        self.assertAlmostEqual(result['isolated_error_area_fraction'],.0001/2.0001)
        a={'surface':{'miou':.8,'small_part_recall':.9},'isolated_error_area_fraction':.01}
        b=copy.deepcopy(a);b['surface']['miou']=.81;b['surface']['small_part_recall']=.5
        self.assertGreater(selection_score(b),selection_score(a))

    def test_resume_sample_order_is_deterministic_and_split_local(self):
        rows=list(range(23))
        def stream(first,last):
            result=[]
            for step in range(first,last):
                for micro in range(2):
                    for rank in range(3):
                        epoch,offset=divmod(((step-1)*2+micro)*3+rank,len(rows))
                        result.append(int(training_order(rows,31,epoch)[offset]))
            return result
        self.assertEqual(stream(1,8),stream(1,4)+stream(4,8))
        self.assertEqual(set(stream(1,8)[:23]),set(rows))

    def test_checkpoint_staging_keeps_source_and_reuses_complete_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.pt'
            torch.save({'weight':torch.arange(10)},source)
            staged=stage_checkpoint(source,root/'cache')
            self.assertEqual(staged.read_bytes(),source.read_bytes())
            before=staged.stat().st_mtime_ns
            self.assertEqual(stage_checkpoint(source,root/'cache'),staged)
            self.assertEqual(staged.stat().st_mtime_ns,before)

    def test_storage_defaults_and_checkpoint_relocation(self):
        self.assertTrue(Path(self.cfg['prepared_dir']).is_relative_to(STORAGE_ROOT))
        self.assertTrue(default_run_dir('seg_vae_v2_latent64.json',self.cfg).is_relative_to(STORAGE_ROOT))
        with self.assertRaises(ValueError):storage_path(ROOT/'experiments/vae_v2',self.cfg)
        with self.assertRaises(ValueError):storage_path('/tmp/seg_vae_v2_resume',self.cfg)
        old={'config':copy.deepcopy(self.cfg),'manifest_digest':'same','world_size':2}
        new=copy.deepcopy(old)
        old['config'].pop('storage_root')
        old['config']['prepared_dir']=str(ROOT/'data/seg_vae_v2')
        new['config']['train']['steps']+=1
        self.assertTrue(resume_metadata_compatible(old,new))
        new['manifest_digest']='changed'
        self.assertFalse(resume_metadata_compatible(old,new))
        new['manifest_digest']='same';new['config']['train']['lr']*=2
        self.assertFalse(resume_metadata_compatible(old,new))

    def test_manifest_rejects_wrong_split_and_changed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=copy.deepcopy(self.cfg);cfg['prepared_dir']=str(root)
            source=root/'source';source.write_bytes(b'geometry')
            sample=root/'sample.npz';sample.write_bytes(b'prepared')
            ss,st=source.stat(),sample.stat()
            row={'instance':'object','split':'train','source':str(source),
                 'source_bytes':ss.st_size,'source_mtime_ns':ss.st_mtime_ns}
            entry=dict(row,path=sample.name,bytes=st.st_size,mtime_ns=st.st_mtime_ns)
            report={'contract':cache_contract(cfg),'failures':[],'samples':[entry]}
            (root/'manifest.json').write_text(json.dumps(report))
            with patch('vae_v2.data.source_rows',return_value=[row]):
                self.assertEqual(load_manifest(cfg)['samples'][0]['instance'],'object')
                self.assertEqual(load_manifest(cfg,verify_workers=4),load_manifest(cfg))
                report['samples'][0]['split']='test'
                (root/'manifest.json').write_text(json.dumps(report))
                with self.assertRaises(ValueError):load_manifest(cfg)
                report['samples'][0]['split']='train'
                (root/'manifest.json').write_text(json.dumps(report))
                sample.write_bytes(b'mutated-data')
                with self.assertRaises(ValueError):load_manifest(cfg)
                with self.assertRaises(ValueError):load_manifest(cfg,verify_workers=4)


if __name__=='__main__':unittest.main()
