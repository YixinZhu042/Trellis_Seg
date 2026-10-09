import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from preprocess.step4_encode_seg_latent import parse_face_labels
from scripts.audit_prepared_dit_supervision import audit, REQUIRED
from scripts.prepare_dit_supervision import check, merge


class TrainingDataContractTests(unittest.TestCase):
    def test_preprocessing_imports_do_not_disable_autograd(self):
        self.assertTrue(torch.is_grad_enabled())

    def test_flat_hierarchy_can_exceed_cap_and_must_not_be_clipped(self):
        # A flat 16-part hierarchy cannot be made <=15 by the current hierarchy
        # grouping routine. Callers must reject it instead of clipping IDs.
        masks = {str(i): {'mesh': i} for i in range(16)}
        hierarchy = [{'nodeId': i, 'children': []} for i in range(16)]
        _, parts19 = parse_face_labels(masks, 16, hierarchy, max_parts=19)
        labels15, parts15 = parse_face_labels(masks, 16, hierarchy, max_parts=15)
        self.assertEqual(parts19, 16)
        self.assertEqual(parts15, 16)
        self.assertGreater(parts15, 15)
        self.assertEqual(labels15.max(), 16)

    def test_deep_audit_accepts_valid_file_and_catches_mapping_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'x.npz'
            data = dict(coords=np.array([[1,1,1],[2,2,2]],np.int32),voxel_labels=np.array([1,2]),
                        latent_coords=np.array([[0,0,0]],np.uint8),latent_feats=np.ones((1,32),np.float32),
                        shape_feats=np.ones((1,32),np.float32),shape_valid=np.array([True]),
                        mv_coords=np.array([[0,0,0]],np.uint8),
                        mv_feats=np.ones((1,32),np.float32),face_labels=np.array([1,2]),face_area=np.ones(2,np.float32),
                        face_knn=np.array([[0,1],[1,0]],np.int32),face_knn_weights=np.full((2,2),.5,np.float32),
                        adjacency=np.array([[0,1]],np.int32),boundary=np.array([1],np.float32),
                        face_pixels=np.array([1,1]),face_raster=np.full((6,512,512),-1,np.int32),
                        seg_maps=np.zeros((6,512,512),np.int32),sample_center=np.zeros(3),sample_extent=np.array(1.),
                        mesh_vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]]),mesh_faces=np.array([[0,1,2],[0,2,3]]))
            data['face_raster'][0,0,:2]=[0,1]
            data['seg_maps'][0,0,:2]=[2,3]
            self.assertEqual(set(data), REQUIRED)
            np.savez(path, **data)
            self.assertEqual(audit(path, 'train')['errors'], [])
            data['face_pixels'] = np.array([0,0])
            np.savez(path, **data)
            self.assertIn('face pixel counts mismatch', audit(path, 'train')['errors'])

    def test_merge_rejects_incomplete_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'selection.json').write_text(json.dumps({'samples':[{'model_id':'a','split':'train'},{'model_id':'b','split':'test'}]}))
            (root/'prep_rank0.json').write_text(json.dumps(
                {'rank':0,'world_size':1,'samples':[{'instance':'a','split':'train'}],'failures':[]}))
            with self.assertRaises(RuntimeError):
                merge(SimpleNamespace(output_dir=root))
            report=json.loads((root/'prepared_manifest.json').read_text())
            self.assertEqual(report['missing'],['b'])

    def test_merge_rejects_missing_rank_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'selection.json').write_text(json.dumps({'samples':[{'model_id':'a','split':'train'}]}))
            (root/'prep_rank0.json').write_text(json.dumps(
                {'rank':0,'world_size':2,'samples':[{'instance':'a','split':'train'}],'failures':[]}))
            with self.assertRaises(RuntimeError):
                merge(SimpleNamespace(output_dir=root))
            report=json.loads((root/'prepared_manifest.json').read_text())
            self.assertEqual(report['missing_ranks'],[1])

    def test_check_requires_matching_deep_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);samples=root/'samples';samples.mkdir()
            selection={'format_version':2,'samples':[{'model_id':'a','split':'train','mesh_path':'unused'}],
                       'split_counts':{'train':1,'validation':0,'test':0},'excluded_over_max_tokens':[]}
            (root/'selection.json').write_text(json.dumps(selection))
            prepared={'expected':1,'prepared':1,'failures':[],'missing':[],'extra':[],'wrong_split':[],
                      'orphan_files':[],'missing_ranks':[],'extra_ranks':[],'duplicate_ranks':[]}
            (root/'prepared_manifest.json').write_text(json.dumps(prepared))
            sample=samples/'a.npz';np.savez(sample,x=np.array([1]))
            stat=sample.stat();state=[('a',stat.st_size,stat.st_mtime_ns)]
            audit_report={'objects':1,'errors_count':0,'split_counts':selection['split_counts'],'size_bytes':stat.st_size,
                          'selection_sha256':hashlib.sha256((root/'selection.json').read_bytes()).hexdigest(),
                          'prepared_manifest_sha256':hashlib.sha256((root/'prepared_manifest.json').read_bytes()).hexdigest(),
                          'file_state_sha256':hashlib.sha256(json.dumps(state,separators=(',',':')).encode()).hexdigest()}
            (root/'prepared_audit.json').write_text(json.dumps(audit_report))
            args=SimpleNamespace(output_dir=root,data_dir=root,require_prepared=True)
            check(args)
            np.savez(sample,x=np.array([1,2]))
            with self.assertRaises(ValueError):
                check(args)


if __name__=='__main__':unittest.main()
