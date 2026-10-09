"""Budget overflow on duplicate/nonmanifold meshes without mutating GT geometry."""
import unittest
from unittest.mock import patch
import numpy as np
from vae_v2.geometry import geometry_components, build_geometry, attach_targets, SurfaceQueries
from vae_v2.config import load_config, ROOT


class TopologyTests(unittest.TestCase):
    def settings(self, budget=30000):
        return dict(load_config(ROOT/'configs/seg_vae_v2.json')['geometry'], area_samples=200, max_voxels=budget)

    def test_duplicate_triangles_share_connectivity_but_keep_face_ids(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
        f=np.array([[0,1,2],[1,3,2]]*3)
        original=f.copy()
        a,c=geometry_components(v,f,include_nonmanifold=True)
        np.testing.assert_array_equal(f,original)
        self.assertEqual(len(c),6)
        self.assertEqual(len(np.unique(c)),1)
        self.assertTrue((a[:,0]!=a[:,1]).all())

    def test_nonmanifold_fan_connected_without_connecting_nearby_shell(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1],
                    [0,0,.001],[1,0,.001],[0,1,.001]],np.float32)
        f=np.array([[0,1,2],[1,0,3],[0,1,4],[5,6,7]])
        _,c=geometry_components(v,f,include_nonmanifold=True)
        self.assertEqual(c[0],c[1]);self.assertEqual(c[0],c[2])
        self.assertNotEqual(c[0],c[3])

    def test_regular_mesh_arrays_unchanged(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
        f=np.array([[0,1,2],[1,3,2]])
        native=geometry_components
        a=build_geometry(v,f,self.settings(40),17)
        def legacy(vertices,faces,include_nonmanifold=False):
            self.assertFalse(include_nonmanifold)
            return native(vertices,faces)
        with patch('vae_v2.geometry.geometry_components',side_effect=legacy):
            b=build_geometry(v,f,self.settings(40),17)
        self.assertNotIn('topology_repair',a)
        for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_overflow_falls_back_without_deleting_faces_or_changing_gt(self):
        # A connected strip, each face triplicated: all its edges have >2 faces.
        v=np.array([[x,y,0] for x in range(7) for y in (0,1)],np.float32)
        strip=[]
        for i in range(6):strip.extend([[2*i,2*i+2,2*i+1],[2*i+2,2*i+3,2*i+1]])
        f=np.repeat(np.array(strip),3,axis=0)
        labels=np.arange(len(f))%2+1
        s=attach_targets(build_geometry(v,f,self.settings(4),42),labels)
        self.assertEqual(int(s['topology_repair']),1)
        self.assertEqual(len(np.unique(s['face_components'])),1)
        np.testing.assert_array_equal(s['mesh_faces'],f)
        np.testing.assert_array_equal(s['mesh_vertices'],v)
        np.testing.assert_array_equal(s['face_labels'],labels)
        self.assertLessEqual(len(s['coords']),4)
        q=SurfaceQueries(s,self.settings(4)).build(np.arange(len(f)))
        self.assertTrue(q['neighbor_mask'].any(1).all())

    def test_real_disconnected_overflow_still_fails(self):
        v=np.array([[x,y,0] for x in (0,2,4) for y in (0,1,2)],np.float32)
        # Noncollinear, separate triangles.
        v[2,0]+=.5;v[5,0]+=.5;v[8,0]+=.5
        f=np.array([[0,1,2],[3,4,5],[6,7,8]])
        with self.assertRaisesRegex(ValueError,'even with nonmanifold'):
            build_geometry(v,f,self.settings(2),42)


if __name__=='__main__':unittest.main()
