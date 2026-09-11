"""Focused algebra, nuisance, admission and diagnostic-isolation checks."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy import sparse
import rbf

class RBFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y,x=np.mgrid[-150:152:2,-150:152:2];x=x.ravel();y=y.ravel();r=np.hypot(x,y)
        cls.data=SimpleNamespace(x=x,ygrid=y,D=np.tile(r<=90,(3,1)),O=np.tile((r>90)&(r<=150),(3,1)))
        cls.b=rbf.Basis(cls.data,0)
    def test_background_projected_normal_operator_and_gradient(self):
        s=self.b.systems[0];rng=np.random.default_rng(48);c=rng.normal(size=self.b.n);v=rng.normal(size=self.b.n)
        Ac=s.A@c;explicit=s.A.T@(Ac-rbf.base.mm(s.Q,rbf.base.mm(s.Q.T,Ac)))/s.n+self.b.lam*(self.b.penalty@c)
        np.testing.assert_allclose(s.product(c),explicit,atol=1e-16,rtol=1e-10)
        f=lambda z:.5*np.dot(z,s.product(z))
        self.assertAlmostEqual((f(c+1e-5*v)-f(c-1e-5*v))/2e-5,np.dot(v,s.product(c)),places=8)
    def test_overlap_integral_and_scale_invariance(self):
        a=np.ones(self.b.n);r=self.b.concentration(a)['raw_fit_diagnostic'];s=self.b.concentration(7*a)['raw_fit_diagnostic']
        self.assertGreater(np.dot(a,self.b.H@a),np.dot(a*a,self.b.H.diagonal()))
        self.assertAlmostEqual(r['effective_area_arcsec2'],r['direct_effective_area_arcsec2'],places=8)
        self.assertAlmostEqual(r['effective_area_arcsec2'],s['effective_area_arcsec2'],places=8)
        self.assertAlmostEqual(r['coefficient_effective_number'],s['coefficient_effective_number'],places=8)
        self.assertFalse(self.b.concentration(np.zeros_like(a))['available'])
    def test_plane_not_feedback_and_revocation(self):
        plane=13+7*self.data.x/90-4*self.data.ygrid/90;self.b.reset()
        model,a,r=self.b.infer(plane,1.)
        self.assertFalse(r['admitted']);self.assertEqual(np.max(abs(model)),0.)
        np.testing.assert_allclose(r['background'],[13,7,-4],atol=1e-10)
    def test_multilobed_admission_and_diagnostic_independence(self):
        x=self.data.x;y=self.data.ygrid
        source=np.where(self.data.D[0],20*np.exp(-((x-12)**2+(y+5)**2)/100)+15*np.exp(-((x+14)**2+(y-15)**2)/80),0)
        m=source+13+7*x/90-4*y/90;self.b.reset()
        with patch.object(rbf.base,'fit_source',side_effect=AssertionError('Gaussian must not control RBF')):
            z,a,r=self.b.infer(m,1.)
        self.assertTrue(r['admitted']);self.assertTrue(np.all(a>=0));self.assertEqual(np.max(abs(z[~self.data.D[0]])),0.)
        self.b.reset()
        with patch.object(self.b,'concentration',return_value={'unused':'arbitrary'}):zz,aa,rr=self.b.infer(m,1.)
        np.testing.assert_array_equal(z,zz);np.testing.assert_array_equal(a,aa);self.assertEqual(r['admitted'],rr['admitted'])
        self.assertLess(np.linalg.norm((z-source)[self.data.D[0]])/np.linalg.norm(source),.04)

if __name__=='__main__':unittest.main()
