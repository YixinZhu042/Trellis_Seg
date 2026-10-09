"""Corrected raw GT + 64 label channels, independently of latent width."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import struct
import numpy as np
import torch
import trimesh
from vae_v2.annotations import canonical_masks,partition_masks,local_labels,parse_raw_mesh
from vae_v2.config import ROOT,load_config,resume_metadata_compatible
from vae_v2.geometry import attach_targets,build_geometry,training_queries
from vae_v2.data import to_device,validate_sample
from vae_v2.surface import SurfaceHead
from vae_v2.losses import reconstruction_losses
from vae_v2.visuals import class_colors,export_labels
from vae_v2.initialization import load_backbone


class Parts64Tests(unittest.TestCase):
    def rewrite_glb(self,path,modify):
        blob=path.read_bytes();size,kind=struct.unpack_from('<II',blob,12)
        header=json.loads(blob[20:20+size]);modify(header)
        payload=json.dumps(header).encode();payload+=b' '*(-len(payload)%4)
        tail=blob[20+size:]
        path.write_bytes(struct.pack('<4sII',b'glTF',2,20+len(payload)+len(tail))+
                         struct.pack('<II',len(payload),kind)+payload+tail)

    def test_scene_order_compatibility_requires_complete_count_match(self):
        scene=trimesh.Scene();scene.add_geometry(trimesh.creation.box(),node_name='box')
        scene.add_geometry(trimesh.creation.icosphere(subdivisions=0),node_name='sphere')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.glb';scene.export(str(path))
            def reverse_scene(g):
                root=g['scenes'][g.get('scene',0)]['nodes'][0]
                g['nodes'][root]['children'].reverse()
            self.rewrite_glb(path,reverse_scene)
            record={'masks':{'0':{'0':list(range(20))},'1':{'1':list(range(12))}},
                    'mesh_face_num':{'0':20,'1':12},'hierarchy':None}
            mesh,labels,audit=parse_raw_mesh(path,record,63)
            self.assertEqual(audit['compatibility_mapping'],'scene_dfs_primitive')
            self.assertEqual((labels==1).sum(),20);self.assertEqual((labels==2).sum(),12)

    def test_unannotated_line_primitive_is_omitted_but_annotated_line_fails(self):
        scene=trimesh.Scene();scene.add_geometry(trimesh.creation.box(),node_name='line')
        scene.add_geometry(trimesh.creation.icosphere(subdivisions=0),node_name='sphere')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'line.glb';scene.export(str(path))
            self.rewrite_glb(path,lambda g:g['meshes'][0]['primitives'][0].update(mode=1))
            record={'masks':{'0':{'0':list(range(20))}},'mesh_face_num':{'0':20},'hierarchy':None}
            mesh,labels,audit=parse_raw_mesh(path,record,63)
            self.assertEqual(len(labels),20);self.assertEqual(len(audit['omitted_nontriangle_primitives']),1)
            record={'masks':{'0':{'0':list(range(12))},'1':{'1':list(range(20))}},
                    'mesh_face_num':{'0':12,'1':20},'hierarchy':None}
            with self.assertRaisesRegex(ValueError,'No count-verified triangle mapping'):
                parse_raw_mesh(path,record,63)

    def test_duplicate_hierarchy_reference_not_used_when_no_merge_required(self):
        masks=canonical_masks({'0':{'0':[0]},'1':{'0':[1]}})
        hierarchy={'nodeId':0,'children':[{'nodeId':2,'maskId':0},{'nodeId':3,'maskId':0},{'nodeId':4,'maskId':1}]}
        mapping,audit=partition_masks(masks,hierarchy,63)
        self.assertEqual(mapping,{'0':1,'1':2})
        self.assertEqual(audit['hierarchy_duplicate_references_ignored_without_merging'],['0'])
        with self.assertRaises(ValueError):partition_masks(masks,hierarchy,1)

    def test_dense_binary_annotations_without_face_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'dense.glb'
            trimesh.Scene(trimesh.creation.box()).export(str(path))
            record={'masks':{'0':{'0':[1]*6+[0]*6},'1':{'0':[0]*6+[1]*6}},
                    'mesh_face_num':None,'hierarchy':None}
            mesh,labels,audit=parse_raw_mesh(path,record,63)
            np.testing.assert_array_equal(labels,np.array([1]*6+[2]*6))
            self.assertEqual(audit['raw_mask_encoding'],'dense_binary_face_vector')
            self.assertEqual(audit['annotated_faces'],12)

    def test_sparse_annotation_mesh_ids_with_nontriangle_geometry(self):
        scene=trimesh.Scene();scene.add_geometry(trimesh.creation.box(),node_name='line')
        scene.add_geometry(trimesh.creation.icosphere(subdivisions=0),node_name='sphere')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sparse_ids.glb';scene.export(str(path))
            self.rewrite_glb(path,lambda g:g['meshes'][0]['primitives'][0].update(mode=1))
            record={'masks':{'0':{'1':list(range(20))}},'mesh_face_num':{'1':20},'hierarchy':None}
            mesh,labels,audit=parse_raw_mesh(path,record,63)
            self.assertEqual(len(labels),20);self.assertTrue((labels==1).all())
            self.assertEqual(audit['mesh_blocks'][0]['mesh_id'],1)

    def test_maskid_not_nodeid_or_traversal_index(self):
        masks=canonical_masks({'0':{'0':[0]},'3':{'0':[1]},'9':{'0':[2]},'15':{'0':[3]}})
        hierarchy={'nodeId':100,'children':[
            {'nodeId':800,'name':'left','children':[{'nodeId':999,'maskId':9},{'nodeId':123,'maskId':0}]},
            {'nodeId':13,'name':'right','children':[{'nodeId':444,'maskId':15},{'nodeId':5,'maskId':3}]}]}
        mapping,audit=partition_masks(masks,hierarchy,2)
        self.assertEqual(mapping['0'],mapping['9']);self.assertEqual(mapping['3'],mapping['15'])
        self.assertNotEqual(mapping['0'],mapping['3']);self.assertEqual(audit['parts'],2)
        result=local_labels(masks,{'0':4},mapping)
        self.assertTrue((result['0']>0).all())

    def test_over63_merges_complete_subtrees_with_no_loss(self):
        masks=canonical_masks({str(i):{'0':[i]} for i in range(70)})
        h={'nodeId':9999,'children':[{'nodeId':1000+j,'children':[
            {'nodeId':5000+i,'maskId':i} for i in range(j*10,(j+1)*10)]} for j in range(7)]}
        mapping,audit=partition_masks(masks,h,63)
        self.assertLessEqual(audit['parts'],63);self.assertEqual(set(mapping),set(masks))
        labels=local_labels(masks,{'0':70},mapping)['0']
        self.assertTrue((labels>0).all())
        with self.assertRaises(ValueError):partition_masks(masks,None,63)

    def test_submesh_indices_must_be_mapped_before_concatenation(self):
        a=trimesh.creation.box();b=trimesh.creation.icosphere(subdivisions=0)
        scene=trimesh.Scene();scene.add_geometry(a,geom_name='box',node_name='box',transform=trimesh.transformations.translation_matrix([3,0,0]))
        scene.add_geometry(b,geom_name='ball',node_name='ball',transform=trimesh.transformations.translation_matrix([-3,0,0]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'two.glb';scene.export(str(path))
            rec={'masks':{'8':{'0':list(range(12))},'29':{'1':list(range(20))}},
                 'mesh_face_num':{'0':12,'1':20},'hierarchy':None}
            mesh,labels,audit=parse_raw_mesh(path,rec,63)
            self.assertEqual(len(labels),32);self.assertEqual(audit['unannotated_faces'],0)
            centers=mesh.triangles_center
            self.assertTrue((labels[centers[:,0]>0]==1).all())
            self.assertTrue((labels[centers[:,0]<0]==2).all())
            rec['mesh_face_num']['0']=11
            with self.assertRaises(ValueError):parse_raw_mesh(path,rec,63)

    def test_out_of_range_overlapping_and_missing_masks_fail(self):
        with self.assertRaises(ValueError):
            local_labels(canonical_masks({'0':{'0':[0]},'1':{'0':[0]}}),{'0':1},{'0':1,'1':2})
        with self.assertRaises(ValueError):
            local_labels(canonical_masks({'0':{'0':[2]}}),{'0':2},{'0':1})
        with self.assertRaises(ValueError):
            local_labels(canonical_masks({'0':{'1':[0]}}),{'0':2},{'0':1})

    def test_channel63_forward_backward_and_serialized_contract(self):
        cfg=load_config(ROOT/'configs/seg_vae_v2_parts64.json')
        self.assertEqual(cfg['num_classes'],64);self.assertEqual(cfg['model']['latent_channels'],32)
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.01],[0,1,.01],[1,0,.01]],np.float32)
        f=np.array([[0,1,2],[3,4,5]])
        settings=dict(cfg['geometry'],area_samples=100,max_voxels=500)
        s=attach_targets(build_geometry(v,f,settings,42),np.array([1,63]),64)
        s['face_pixels']=np.array([100,100]);s['near_pairs']=np.array([[0,1]])
        validate_sample(s,64)
        with self.assertRaises(ValueError):validate_sample(s,20)
        with self.assertRaises(ValueError):attach_targets(s,np.array([1,64]),64)
        q=to_device(training_queries(s,settings,dict(cfg['train'],query_faces=16,pair_count=8),np.random.default_rng(2)),'cpu')
        logits=torch.randn(len(s['coords']),64,requires_grad=True)
        head=SurfaceHead(8,8,64)
        point,face=head(torch.randn(len(s['coords']),8),logits,q)
        out={'voxel_logits':logits,'point_logits':point,'face_logprob':face,
             'mean':torch.zeros(2,32,requires_grad=True),'logvar':torch.zeros(2,32,requires_grad=True)}
        loss,terms=reconstruction_losses(out,{'voxel_target':torch.tensor(s['voxel_target'])},q,cfg['loss'],1e-5)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all());self.assertGreater(float(logits.grad[:,63].abs().sum()),0)

    def test_unique_palette_and_glb_roundtrip_high_ids(self):
        self.assertEqual(len(np.unique(class_colors(64),axis=0)),64)
        mesh=trimesh.creation.box();labels=np.array([0,1,20,32,63,5]*2)
        with tempfile.TemporaryDirectory() as directory:
            export_labels(mesh,labels,Path(directory)/'labels.glb',64)

    def test_interface_reset_and_backbone_preserved(self):
        class Encoder(torch.nn.Module):
            def __init__(self,classes):
                super().__init__();self.input_layer=torch.nn.Linear(classes,4);self.body=torch.nn.Linear(4,4)
        old=Encoder(20);new=Encoder(64)
        initial=new.input_layer.weight.detach().clone()
        reset=load_backbone(new,old.state_dict(),'encoder',32,64)
        self.assertEqual(set(reset),{'input_layer.weight','input_layer.bias'})
        torch.testing.assert_close(new.body.weight,old.body.weight)
        torch.testing.assert_close(new.input_layer.weight,initial)
        bad=old.state_dict();bad.pop('body.weight')
        with self.assertRaises(ValueError):load_backbone(new,bad,'encoder',32,64)

    def test_old_optimizer_metadata_cannot_resume_new_labels(self):
        old={'config':load_config(ROOT/'configs/seg_vae_v2.json'),'manifest_digest':'old'}
        new={'config':load_config(ROOT/'configs/seg_vae_v2_parts64.json'),'manifest_digest':'new'}
        self.assertFalse(resume_metadata_compatible(old,new))


if __name__=='__main__':unittest.main()
