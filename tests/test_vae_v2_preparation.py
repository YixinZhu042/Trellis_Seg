"""Concurrency, resume and sealing tests for the CPU preparation pipeline."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.prepare_seg_vae_v2 import prepare_one, parallel_results, merge_prepared, output_lock
from vae_v2.config import load_config, ROOT
from vae_v2.data import cache_contract


class PreparationTests(unittest.TestCase):
    def fixture(self, root):
        cfg=load_config(ROOT/'configs/seg_vae_v2.json')
        cfg['geometry']=dict(cfg['geometry'],area_samples=100,max_voxels=1000)
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        rows=[]
        for i in range(4):
            p=root/f'source{i}.npz'
            np.savez_compressed(p,mesh_vertices=vertices,mesh_faces=np.array([[0,1,2]]),
                                face_labels=np.array([1]),face_pixels=np.array([200]))
            stat=p.stat()
            rows.append({'instance':f'obj{i}','split':'train' if i<2 else 'validation',
                         'type_id':'example','source':str(p),'source_bytes':stat.st_size,
                         'source_mtime_ns':stat.st_mtime_ns})
        return cfg,rows

    def test_process_pool_matches_serial_and_reuses_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,rows=self.fixture(root)
            serial=root/'serial';parallel=root/'parallel'
            for output in (serial,parallel):(output/'samples').mkdir(parents=True)
            one=[prepare_one(row,cfg,serial) for row in rows]
            results=list(parallel_results((('prepare',row,cfg,parallel) for row in rows),2))
            self.assertTrue(all(error is None for _,error in results))
            for row in rows:
                left=serial/'samples'/f'{row["instance"]}.npz'
                right=parallel/'samples'/left.name
                with np.load(left) as a,np.load(right) as b:
                    self.assertEqual(a.files,b.files)
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                before=(right.stat().st_mtime_ns,hashlib.sha256(right.read_bytes()).hexdigest())
                reused=prepare_one(row,cfg,parallel)
                self.assertTrue(reused['reused'])
                self.assertEqual(before,(right.stat().st_mtime_ns,hashlib.sha256(right.read_bytes()).hexdigest()))

    def test_parallel_merge_and_corrupt_sample_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);cfg,rows=self.fixture(output)
            (output/'samples').mkdir()
            samples=[prepare_one(row,cfg,output) for row in rows]
            report={'contract':cache_contract(cfg),'rank':0,'world_size':1,'samples':samples,'failures':[],'status':'complete'}
            (output/'prep_rank0000.json').write_text(json.dumps(report))
            merge_prepared(rows,cfg,output,1,2)
            sealed=json.loads((output/'manifest.json').read_text())
            self.assertEqual(len(sealed['samples']),4)
            self.assertTrue(all('reused' not in x for x in sealed['samples']))
            (output/samples[0]['path']).write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):merge_prepared(rows,cfg,output,1,2)

    def test_same_rank_or_merge_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            with output_lock(output,0):
                with self.assertRaises(RuntimeError):
                    with output_lock(output,0):pass
                with output_lock(output,1):pass
                with self.assertRaises(RuntimeError):
                    with output_lock(output,1,merge=True):pass

    def test_stop_does_not_submit_more_work(self):
        self.assertEqual(list(parallel_results([None],1,lambda:True)),[])
        self.assertEqual(list(parallel_results([None],2,lambda:True)),[])


if __name__=='__main__':unittest.main()
