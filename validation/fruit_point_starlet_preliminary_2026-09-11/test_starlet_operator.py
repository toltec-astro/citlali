import unittest
import numpy as np
import starlet as s

class StarletAlgebra(unittest.TestCase):
    def test_sum_and_exact_masked_adjoint(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(81,83));v=rng.normal(size=(5,81,83))
        v*=rng.random(v.shape)>.35
        np.testing.assert_allclose(s.analysis(x).sum(axis=0),x,atol=1e-14)
        np.testing.assert_allclose(np.sum(s.analysis(x)*v),np.sum(x*s.adjoint(v)),rtol=1e-12,atol=1e-12)

    def test_complete_mask_excludes_missing_influence(self):
        rng=np.random.default_rng(92);S=np.ones((101,103),bool);S[51,53]=False
        a=rng.normal(size=S.shape);b=a.copy();b[~S]=1e8
        valid=s.eligible(S)
        np.testing.assert_allclose(s.analysis(a)[valid],s.analysis(b)[valid],atol=1e-12)
        yy,xx=np.indices(S.shape)
        for j,r in enumerate([2,6,14,30,30]):
            expected=(yy>=r)&(yy<101-r)&(xx>=r)&(xx<103-r)&~((abs(yy-51)<=r)&(abs(xx-53)<=r))
            np.testing.assert_array_equal(valid[j],expected)

    def test_positive_source_has_signed_details(self):
        y,x=np.mgrid[-50:51,-50:51];z=np.exp(-(x*x+y*y)/20)
        w=s.analysis(z)
        self.assertTrue(np.all(w[:4].min(axis=(1,2))<0));self.assertTrue(np.all(w[:4].max(axis=(1,2))>0))

    def test_background_excluded_and_selected_gradient(self):
        y,x=np.mgrid[-70:71,-70:71];r=np.hypot(x,y);S=np.ones(x.size,bool);D=(r.ravel()<30);O=(r.ravel()>45)&(r.ravel()<60)
        est=s.Estimator(x.ravel(),y.ravel(),x.shape,S,D,O,np.ones(x.size))
        z=(7+2*x/90-3*y/90).ravel();res,beta=est.residual(z)
        np.testing.assert_allclose(beta,[7,2,-3],atol=1e-13);np.testing.assert_allclose(res,0,atol=1e-13)
        rng=np.random.default_rng(93);v=rng.normal(size=x.shape);direction=rng.normal(size=x.shape)
        weights=rng.random((5,)+x.shape)*(rng.random((5,)+x.shape)>.7)
        f=lambda v:.5*np.sum(weights*s.analysis(v)**2)
        numeric=(f(v+1e-5*direction)-f(v-1e-5*direction))/2e-5
        analytic=np.sum(s.adjoint(weights*s.analysis(v))*direction)
        np.testing.assert_allclose(numeric,analytic,rtol=1e-7,atol=1e-7)

if __name__=='__main__':unittest.main()
