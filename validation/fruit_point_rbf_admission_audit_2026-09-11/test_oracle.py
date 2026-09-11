import unittest
import numpy as np
import audit
class OracleTests(unittest.TestCase):
    def test_unbiased_amplitude_and_covariance_weighting(self):
        x=np.array([[1.,2.,-1.],[-1.,-2.,1.],[2.,1.,2.],[-2.,-1.,-2.],[1.,-1.,3.],[-1.,1.,-3.]])
        p=np.array([2.,3.,4.]);w,sigma,C,Cs,cond=audit.covariance_detector(x,p)
        self.assertAlmostEqual(np.dot(w,p),1.,places=12)
        np.testing.assert_allclose(Cs@w,p*sigma**2,rtol=1e-12)
        self.assertGreater(np.max(abs(C-np.diag(np.diag(C)))),0.)
        self.assertLess(cond,1e8)
        f=np.array([.3,-.2,.4]);self.assertAlmostEqual(np.dot(w,f+2*p)-np.dot(w,f),2.,places=12)
    def test_singular_covariance_is_unavailable(self):
        with self.assertRaises(ValueError):audit.covariance_detector(np.zeros((20,3)),np.ones(3))
if __name__=='__main__':unittest.main()
