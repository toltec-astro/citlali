"""Artificial operator/coordinate tests, not empirical candidate screening."""
from pathlib import Path
import importlib.util
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import candidate as c

spec=importlib.util.spec_from_file_location('frozen_starlet',Path(__file__).resolve().parents[1]/'fruit_point_starlet_preliminary_2026-09-11/starlet.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class CentralDomain(unittest.TestCase):
    def setUp(self):
        yy,xx=np.mgrid[-70:71,-70:71];self.x=2*xx.ravel();self.y=2*yy.ravel();r=np.hypot(self.x,self.y)
        self.args=(self.x,self.y,xx.shape,np.ones(xx.size,bool),r<=90,(r>90)&(r<=120),100+self.x/4)
        self.noise=np.random.default_rng(713).normal(size=xx.size)

    def make(self,cls=c.Estimator):
        e=cls(*self.args);record=e.calibrate(self.noise);self.assertTrue(record['available']);return e

    def test_domain_change_preserves_noise_background_and_computational_support(self):
        f=self.make();g=self.make();g.D=g.D&(np.hypot(self.x,self.y)<=60)
        for key in ['S','O','valid','stratum','sigmas']:np.testing.assert_array_equal(getattr(f,key),getattr(g,key))
        self.assertEqual(f.q_split,g.q_split);self.assertLess(g.D.sum(),f.D.sum())
        np.testing.assert_array_equal(f.residual(self.noise)[0],g.residual(self.noise)[0])

    def test_zero_MAD_repair_preserves_unscaled_objective(self):
        e=self.make();z=np.where(e.D,100*np.exp(-((self.x-7)**2+(self.y+3)**2)/40),0)
        observed=[]
        def fake(fun,x,**kw):
            value,grad=fun(x);observed.append(value);self.assertTrue(np.isfinite(grad).all())
            return SimpleNamespace(x=x,success=False,message='unit-test probe',nit=0,nfev=1)
        with patch('candidate.minimize',fake):_,rec,omega=e.infer(z)
        self.assertEqual(rec['scale_source'],'fixed_calibration_null_outer_MAD')
        self.assertGreater(rec['solver_scale'],0)
        W=c.analysis(z.reshape(e.shape)).reshape(5,-1);sigma=e.sigmas[:,e.stratum]
        np.testing.assert_allclose(observed[0],.5*np.sum(np.where(omega,(W/sigma)**2,0)),rtol=1e-12)

    def test_nonzero_MAD_matches_original_objective_and_gradient(self):
        e=self.make();f=self.make(old.Estimator);z=self.noise+np.where(e.D,100*np.exp(-(self.x**2+self.y**2)/40),0)
        probes=[]
        def fake(fun,x,**kw):
            probes.append(fun(x));return SimpleNamespace(x=x,success=False,message='unit-test probe',nit=0,nfev=1)
        with patch('candidate.minimize',fake):u,rec,omega=e.infer(z)
        with patch.object(old,'minimize',fake):v,prior,previous=f.infer(z)
        np.testing.assert_array_equal(omega,previous);np.testing.assert_array_equal(u,v)
        np.testing.assert_array_equal(probes[0][1],probes[1][1]);self.assertEqual(probes[0][0],probes[1][0])

if __name__=='__main__':unittest.main()
