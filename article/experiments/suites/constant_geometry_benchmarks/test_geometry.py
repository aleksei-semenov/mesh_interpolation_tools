"""Run: python -m unittest -v"""
import unittest
import numpy as np
from geometry import area, mesh, validate, overlap_matrix, sample_matrix, centroids, errors

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.p,self.t=mesh(3)
        self.x=np.linspace(0,1,6); self.y=np.linspace(0,1,8)
        self.a=validate(self.p,self.t,self.x,self.y)
        self.at=np.outer(np.diff(self.y),np.diff(self.x)).ravel()
    def test_known_clip(self):
        p=np.array([[0.,0.],[1.,0.],[0.,1.]])
        o=overlap_matrix(p,np.array([[0,1,2]]),np.array([0.,.5,1.]),np.array([0.,1.]))
        np.testing.assert_allclose(o.toarray()[:,0],[.375,.125])
    def test_overlap_partition(self):
        o=overlap_matrix(self.p,self.t,self.x,self.y)
        np.testing.assert_allclose(np.asarray(o.sum(axis=0)).ravel(),self.a,atol=1e-14)
        np.testing.assert_allclose(np.asarray(o.sum(axis=1)).ravel(),self.at,atol=1e-14)
    def test_deposition_and_normalizations(self):
        for k in (0,1,3):
            b=sample_matrix(self.p,self.t,self.a,self.x,self.y,k)
            np.testing.assert_allclose(np.asarray(b.sum(axis=0)).ravel(),self.a,atol=1e-14)
            q=np.arange(len(self.a),dtype=float)
            w=np.asarray(b.sum(axis=1)).ravel(); s=b@q; hit=w>0
            np.testing.assert_allclose(s.sum(),self.a@q,atol=1e-12)
            np.testing.assert_allclose(self.at@(s/self.at),self.a@q,atol=1e-12)
            np.testing.assert_allclose(w[hit]@(s[hit]/w[hit]),self.a@q,atol=1e-12)
            np.testing.assert_array_equal(w[hit]/w[hit],1.)
    def test_centroid_mean_and_count(self):
        t=self.p[self.t[0]]
        for k in range(4):
            c=centroids(t,k)
            self.assertEqual(c.shape,(4**k,2))
            np.testing.assert_allclose(c.mean(axis=0),t.mean(axis=0),atol=1e-14)
    def test_undefined_error(self):
        self.assertTrue(np.isnan(errors(np.array([np.nan]),np.array([1.]),np.array([1.]))).all())
    def test_invalid_connectivity(self):
        t=self.t.copy(); t[0,0]=-1
        with self.assertRaises(ValueError): validate(self.p,t,self.x,self.y)

if __name__=='__main__': unittest.main()
