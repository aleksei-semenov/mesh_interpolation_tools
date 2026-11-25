import unittest
import numpy as np
from geometry import mesh,area
from bench import (quad_nodes,gauss_triangle,field,reconstruct,evaluate,intersection_polygons,
                   overlap_integral,deposit,disk_integral_triangle,disk_rectangle,metrics)

class BenchmarkTests(unittest.TestCase):
    def test_affine_and_quadratic_rules(self):
        t=np.array([[0.,0.],[1.,0.],[0.,1.]])
        f=lambda x,y: x*x+y*y+x*y+2*x
        exact=gauss_triangle(t,f,12)
        pts=quad_nodes(t[None,:,:],2)[0]
        self.assertAlmostEqual(np.mean(f(pts[:,0],pts[:,1])),exact,places=12)
        p=quad_nodes(t[None,:,:],1)[0]; self.assertGreater(abs(np.mean(f(p[:,0],p[:,1]))-exact),.01)
        lin=lambda x,y:1+2*x-3*y
        self.assertAlmostEqual(float(lin(*p[0])),gauss_triangle(t,lin),places=12)

    def test_reconstruction_and_clipped_overlap(self):
        nodes,tri=mesh(5);vals=np.array([gauss_triangle(t,lambda x,y:field('quadratic',x,y)) for t in nodes[tri]])
        rec=reconstruct(nodes,tri,vals,'quadratic')
        for i,t in enumerate(nodes[tri]):
            pts=quad_nodes(t[None,:,:],2)[0]
            np.testing.assert_allclose(evaluate(pts,i,vals,*rec),field('quadratic',pts[:,0],pts[:,1]),atol=2e-12)
        x=np.linspace(0,1,5);y=np.linspace(0,1,6);polys,cover,parent=intersection_polygons(nodes,tri,x,y)
        A=np.outer(np.diff(y),np.diff(x)).ravel();np.testing.assert_allclose(cover,A,atol=1e-12)
        exact=overlap_integral(polys,len(A),vals,rec)
        from bench import target_average
        np.testing.assert_allclose(exact/A,target_average('quadratic',x,y),atol=2e-12)
        self.assertAlmostEqual(np.sum(exact),np.dot(vals,[area(t) for t in nodes[tri]]),places=12)

    def test_each_quadrature_node_assigned(self):
        nodes=np.array([[0.,0.],[1.,0.],[0.,1.]])
        tri=np.array([[0,1,2]]);vals=np.ones(1);rec=reconstruct(nodes,tri,vals,'constant')
        S,W,d=deposit(nodes,tri,np.array([0,.4,1]),np.array([0,1]),0,vals,rec,2)
        np.testing.assert_allclose(W,[1/3,1/6]);np.testing.assert_allclose(S,W);self.assertEqual(d,0)

    def test_disk_reference(self):
        nodes,tri=mesh(5);total=sum(disk_integral_triangle(t) for t in nodes[tri]);self.assertAlmostEqual(total,np.pi*.2**2,places=8)
        edges=np.linspace(0,1,8);summed=sum(disk_rectangle(edges[i],edges[i+1],edges[j],edges[j+1]) for i in range(7) for j in range(7))
        self.assertAlmostEqual(summed,np.pi*.2**2,places=8)

    def test_undefined_error(self):
        self.assertTrue(np.isnan(metrics(np.array([np.nan,1]),np.ones(2),np.ones(2))[1]))

if __name__=='__main__':unittest.main()
