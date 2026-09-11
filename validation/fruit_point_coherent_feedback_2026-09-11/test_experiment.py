import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
import experiment as e

class Invariants(unittest.TestCase):
    def test_gaussian_jacobian_and_free_recovery(self):
        yy,xx=np.mgrid[-60:61:2,-60:61:2];x=xx.ravel();y=yy.ravel()
        p=np.array([123.,17.,-11.,np.log(15.),np.log(8.),.4,3.,4.,-2.])
        J=approx_derivative(lambda v:e.gaussian(v,x,y,True),p)
        np.testing.assert_allclose(e.mm(np.ones((30,12)),np.ones((12,5))),np.full((30,5),12.))
        np.testing.assert_allclose(e.model_jac(p,x,y),J,rtol=1e-5,atol=2e-6)
        z=e.gaussian(p,x,y,True);f=e.fit_source(z,x,y,np.ones(len(x),bool))
        np.testing.assert_allclose(f['centroid'],p[1:3],atol=1e-5)
        np.testing.assert_allclose(f['peak'],123,atol=1e-5)
        np.testing.assert_allclose(f['widths'],[15,8],atol=1e-5)
        self.assertFalse(f['boundary_rejection'])
        # The astronomical component excludes all three fitted background terms.
        np.testing.assert_allclose(e.gaussian(f['parameters'],x,y),e.gaussian(p,x,y),atol=1e-5)
    def test_support_formula_and_empty(self):
        q=np.r_[0.,np.arange(1,101),.01,.02]
        sn,ss=e.support(q);v=np.sort(q[q>0]);Q=v[(int(.75*len(v))+len(v))//2]
        np.testing.assert_array_equal(sn,(q>0)&(q>=.01*Q));np.testing.assert_array_equal(ss,(q>0)&(q>=.1*Q))
        self.assertFalse(e.support(np.zeros(3))[0].any())
    def test_replacement_centering_and_recorded_apply(self):
        d=object.__new__(e.Data);rng=np.random.default_rng(3);parent=rng.normal(size=(30,12))
        d.valid=np.ones_like(parent,bool);d.ar=np.zeros(12,int);d.pixel=np.broadcast_to(np.arange(30)[:,None],parent.shape)
        mask=np.ones_like(parent);d.groups=[(0,0,0,30,np.arange(12),mask,mask.sum(0),e.mm(mask.T,mask)-1)]
        m=np.zeros((3,30));m[0]=np.linspace(-.1,.1,30)**2
        z,st,_=d.clean(parent,m);A=st['c00_nw00_basis'];lam=st['c00_nw00_location'];rho=parent-d.project(m)
        expected=rho-lam-e.mm(e.mm(rho-lam,A),A.T)+d.project(m)
        np.testing.assert_allclose(z,expected,atol=1e-12)
        np.testing.assert_allclose((z-d.project(m)).mean(0),0,atol=1e-12)
        _,st0,_=d.clean(parent,np.zeros_like(m));self.assertGreater(np.max(abs(lam-st0['c00_nw00_location'])),1e-4)
        # No previous cleaned result or accumulated previous model enters this second call.
        z2,_,_=d.clean(parent,m);np.testing.assert_array_equal(z,z2)
    def test_required_rank_fails(self):
        d=object.__new__(e.Data);parent=np.ones((20,8));d.valid=np.ones_like(parent,bool);d.ar=np.zeros(8,int)
        d.pixel=np.zeros(parent.shape,int);mask=np.ones_like(parent);d.groups=[(0,0,0,20,np.arange(8),mask,mask.sum(0),e.mm(mask.T,mask)-1)]
        with self.assertRaisesRegex(ValueError,'nonpositive'):d.clean(parent,np.zeros((3,1)))
if __name__=='__main__':unittest.main()
