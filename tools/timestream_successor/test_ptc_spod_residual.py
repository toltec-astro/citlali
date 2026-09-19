import unittest
from types import SimpleNamespace
import numpy as np
from scipy import linalg
from ptc_spod import fourier, supported_windows
from ptc_spod_residual import apply_complete_fourier, projected_power, matched_residual, temporal_cohort


class Residual(unittest.TestCase):
    def test_complete_frozen_projection_matches_time_operation(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(400,8));u=linalg.qr(rng.normal(size=(8,2)),mode='economic')[0]
        y=x-linalg.blas.dgemm(1,linalg.blas.dgemm(1,x,u),u,trans_b=1);win=np.array([[0,200],[200,400]])
        _,q=fourier(x,win,.01);_,out=fourier(y,win,.01)
        for a,b in zip(q,out):np.testing.assert_allclose(apply_complete_fourier(a,u),b,rtol=1e-12,atol=1e-14)

    def test_frozen_noise_control_contains_cleaner_induced_correlations(self):
        rng=np.random.default_rng(7);q=(rng.normal(size=(10000,5))+1j*rng.normal(size=(10000,5)))/np.sqrt(2)
        u=np.ones((5,1))/np.sqrt(5);y=apply_complete_fourier(q,u)
        covariance=linalg.blas.zgemm(1/len(y),y,y,trans_a=2)
        expected=np.eye(5)-np.ones((5,5))/5
        np.testing.assert_allclose(covariance,expected,atol=.025,rtol=0)
        self.assertLess(covariance[0,1].real,-.15)  # created by cleaning independent input

    def test_zero_power_has_no_invented_pattern(self):
        x=np.zeros((2400,4));valid=np.ones_like(x,bool)
        win,_=supported_windows(valid,[(0,2400)],200,100);_,q=fourier(x,win,.01)
        seg=[dict(first=0,past_last=2400,full_basis=np.ones((4,1))/2,fit_members=list(range(4)))]
        r=matched_residual(SimpleNamespace(valid=valid,values=x),np.arange(4),win,q,x,seg,.01)
        self.assertEqual(r['state'],'available')
        self.assertTrue(all(f['state']=='unavailable-no-identifiable-input-pattern' for fold in r['separate_pattern_folds'] for f in fold['frequency']))

    def test_phase_pattern_projection_keeps_relative_detector_phase(self):
        u=np.array([1,1j])/np.sqrt(2);q=np.array([2*u,3j*u])
        self.assertAlmostEqual(projected_power(q,u),6.5)

    def test_shared_tone_survives_wrong_frozen_cleaner_and_is_detected(self):
        rng=np.random.default_rng(55);n=4000;d=6;t=np.arange(n)*.01
        pattern=np.ones(d)/np.sqrt(d);other=np.array([1,-1,0,0,0,0])/np.sqrt(2)
        x=4*np.sin(2*np.pi*11*t)[:,None]*pattern+rng.normal(size=(n,d))*.1
        u=other[:,None];y=x-linalg.blas.dgemm(1,linalg.blas.dgemm(1,x,u),u,trans_b=1)
        valid=np.ones_like(x,bool);win,_=supported_windows(valid,[(0,n)],200,100);_,q=fourier(x,win,.01)
        seg=[dict(first=0,past_last=n,full_basis=u,fit_members=list(range(d)))]
        result=matched_residual(SimpleNamespace(valid=valid,values=x),np.arange(d),win,q,y,seg,.01)
        self.assertEqual(result['state'],'available')
        f=min(result['control_frequency'],key=lambda r:abs(r['frequency_hz']-11))
        self.assertGreater(f['output_leading_eigenvalue'],max(r['leading_eigenvalue'] for r in f['frozen_operator_phase_controls']))
        self.assertLess(result['maximum_Fourier_operator_absolute_error'],1e-12)

    def test_missing_full_fit_member_is_not_a_false_complete_noise_control(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(2400,4));valid=np.ones_like(x,bool);valid[:,3]=False
        cols=np.arange(3);win,_=supported_windows(valid[:,cols],[(0,2400)],200,100);_,q=fourier(x[:,cols],win,.01)
        seg=[dict(first=0,past_last=2400,full_basis=np.ones((4,1))/2,fit_members=list(range(4)))]
        r=matched_residual(SimpleNamespace(valid=valid,values=x),cols,win,q,x[:,:3],seg,.01)
        self.assertEqual(r['control_state'],'unavailable-insufficient-complete-fit-windows')

    def test_temporal_selection_is_fixed_and_uses_no_data_values(self):
        valid=np.ones((2000,20),bool);valid[::50,19]=False
        r=temporal_cohort(valid,[(0,2000)],.01)
        self.assertNotIn(19,r['selected_columns']);self.assertEqual(len(r['selected_columns']),18)
        self.assertEqual(r['candidates'][0]['windows'],0)
        self.assertGreater(r['candidates'][1]['Fourier_seconds'],0)


if __name__=='__main__':unittest.main()
