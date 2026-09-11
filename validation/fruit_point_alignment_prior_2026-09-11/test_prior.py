import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize._numdiff import approx_derivative
import experiment as e


class PriorTests(unittest.TestCase):
    def geometry(self):
        yy,xx=np.mgrid[-90:91:2,-90:91:2];x=xx.ravel();y=yy.ravel();r=np.hypot(x,y)
        return SimpleNamespace(x=x,ygrid=y,D=np.tile(r<=70,(3,1)),O=np.tile((r>70)&(r<=90),(3,1)))

    def test_joint_and_offset_jacobians(self):
        x=np.linspace(-20,30,83);y=np.sin(x)*20
        q=np.array([17.,-11.,100.,np.log(12),np.log(8),.4,3.,2.,1.,70.,np.log(15),np.log(9),.2,-2.,1.,3.])
        for a in range(2):
            J=approx_derivative(lambda v:e.b.gaussian(e.unpack_pair(v,a),x,y,True),q)
            np.testing.assert_allclose(e.pair_jac(q,a,x,y),J,rtol=2e-5,atol=3e-6)
        q2=np.r_[1.5,-.6,q[2:9]];c=np.array([17.,-11.])
        J=approx_derivative(lambda v:e.b.gaussian(e.unpack_offset(v,c),x,y,True),q2)
        np.testing.assert_allclose(e.offset_jac(q2,c,x,y),J,rtol=2e-5,atol=3e-6)

    def test_free_recovery_with_nonzero_offset_and_distinct_shapes(self):
        d=self.geometry();maps=[];truth=[]
        for a,(amp,wx,wy,angle) in enumerate([(130.,12.,8.,.4),(80.,15.,10.,.1),(55.,20.,12.,.7)]):
            p=np.array([amp,17.,-11.,np.log(wx),np.log(wy),angle,3+a,2.,-1.])
            if a==2:p[1:3]+=[1.2,-.8]
            truth.append(p);maps.append(e.b.gaussian(p,d.x,d.ygrid,True))
        maps=np.array(maps);free=e.b.describe(d,maps);fits,info=e.prior_fits(d,maps,free)
        np.testing.assert_allclose(info['a2000_offset'],[1.2,-.8],atol=1e-5)
        self.assertFalse(info['a2000_prior_tension'])
        for f,p in zip(fits,truth):
            np.testing.assert_allclose(f['centroid'],p[1:3],atol=1e-5)
            self.assertAlmostEqual(f['peak'],p[0],places=4)
            np.testing.assert_allclose(f['widths'],np.exp(p[3:5]),atol=1e-5)
            np.testing.assert_allclose(e.b.gaussian(f['parameters'],d.x,d.ygrid),e.b.gaussian(p,d.x,d.ygrid),atol=1e-5)

    def test_own_signal_anchor_and_tension_rejections(self):
        d=self.geometry();maps=np.tile(np.sin(d.x)+np.cos(d.ygrid),(3,1))
        p=np.array([100.,17.,-11.,np.log(12),np.log(8),.4,1000.,22.,-13.])
        fit=dict(parameters=p,peak=100.,boundary_rejection=False)
        fits=[dict(fit),dict(fit),dict(fit,peak=0.)]
        info=dict(a2000_prior_tension=False)
        m,r,_=e.prior_inference(d,maps,fits,info);self.assertFalse(m[2].any())
        # Independent backgrounds are never rejoined even when very large.
        np.testing.assert_allclose(m[0,d.D[0]],e.b.gaussian(p,d.x[d.D[0]],d.ygrid[d.D[0]]))
        fits=[dict(fit,peak=0.),dict(fit,peak=0.),dict(fit)]
        m,r,_=e.prior_inference(d,maps,fits,info);self.assertFalse(m[2].any());self.assertEqual(r[2]['reason'],'no_reference_pair_anchor')
        m,r,_=e.prior_inference(d,maps,[fit]*3,dict(a2000_prior_tension=True));self.assertFalse(m[2].any());self.assertEqual(r[2]['reason'],'a2000_offset_boundary_tension')

if __name__=='__main__':unittest.main()
