import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy import signal
import yaml
from ptc_spod import (Inputs, digest, read_cal, supported_windows, fourier,
                      decompose, overlap, disjoint_halves, orthonormal_span)


class SpodTests(unittest.TestCase):
    def test_complex_csd_density_matches_scipy_and_phase(self):
        rng=np.random.default_rng(6); y=rng.normal(size=(2048,4));dt=.01
        windows,_=supported_windows(np.ones_like(y,bool),[(0,len(y))],256,128)
        f,q=fourier(y,windows,dt)
        c=np.einsum('kfd,kfe->fde',q,q.conj())/len(q)
        f0,c0=signal.csd(y[:,1],y[:,0],fs=1/dt,nperseg=256,noverlap=128)
        np.testing.assert_allclose(f,f0);np.testing.assert_allclose(c[:,0,1],c0,rtol=1e-12,atol=1e-15)
        for i in (1,28,120):
            ev,u=decompose(q[:,i],keep=4)
            np.testing.assert_allclose((u*ev)@u.conj().T,c[i],rtol=1e-11,atol=1e-15)

    def test_stable_tonal_pattern_changes_strength_and_global_phase(self):
        fs=128;n=256;t=np.arange(n)/fs;shape=np.exp(1j*np.array([0,.4,1.,2.]))*np.array([1,2,.8,1.4])
        blocks=[np.real(np.exp(2j*np.pi*11*t+1j*p)[:,None]*shape)*a
                for p,a in zip(np.arange(12)*.7,np.linspace(1,5,12))]
        y=np.concatenate(blocks);windows=np.array([(i*n,(i+1)*n) for i in range(12)])
        f,q=fourier(y,windows,1/fs);i=np.argmin(abs(f-11))
        e,u=decompose(q[:6,i]);e2,v=decompose(q[6:,i])
        self.assertGreater(e2[0],e[0]*2);self.assertGreater(overlap(u,v,1)['mean_cos2'],.9999999)
        self.assertGreater(overlap(u,orthonormal_span(shape[:,None]),1)['mean_cos2'],.9999999)

    def test_changing_subspace_with_fixed_detector_gains(self):
        rng=np.random.default_rng(4);a=np.array([1,1,1,1])/2;b=np.array([1,1,-1,-1])/2
        z=rng.normal(size=80)+1j*rng.normal(size=80)
        _,u=decompose(z[:,None]*a);_,v=decompose(z[:,None]*b)
        self.assertLess(overlap(u,v,1)['mean_cos2'],1e-12)

    def test_rotation_and_order_invariance(self):
        rng=np.random.default_rng(9);a=orthonormal_span(rng.normal(size=(10,3))+1j*rng.normal(size=(10,3)))
        r=orthonormal_span(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))
        self.assertAlmostEqual(overlap(a,a@r,3)['mean_cos2'],1)

    def test_masks_noise_no_fill_or_gap_join(self):
        rng=np.random.default_rng(123);y=rng.normal(size=(8192,8));v=np.ones_like(y,bool)
        v[400:440,0]=False;v[2500:2550,4]=False;y[~v]=np.nan
        w,_=supported_windows(v,[(0,2000),(2000,8192)],128,64)
        self.assertTrue(all(not(a<2000<b) for a,b in w))
        self.assertTrue(all(v[a:b].all() for a,b in w))
        _,q=fourier(y,w,1/128);ev,_=decompose(q[:,20])
        self.assertLess(ev[0]/ev.sum(),.24)
        a,b=disjoint_halves(w,np.arange(len(w)));ids=np.sort(np.r_[a,b])
        self.assertTrue(all(w[j,0]>=w[i,1]+128 for i,j in zip(ids[:-1],ids[1:])))

    def test_unexpected_nonfinite_is_failure_not_drop(self):
        y=np.ones((128,2));y[30,0]=np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            fourier(y,np.array([[0,64],[64,128]]),.01)
        v=np.isfinite(y);w,_=supported_windows(v,[(0,128)],64,32)
        self.assertEqual(len(w),2)
        self.assertTrue(np.isfinite(fourier(y,w,.01)[1]).all())

    def test_insufficient_support_and_rank_limit(self):
        w,_=supported_windows(np.zeros((128,3),bool),[(0,128)],64,32)
        self.assertEqual(w.shape,(0,2))
        with self.assertRaisesRegex(ValueError,'insufficient'):fourier(np.zeros((128,3)),w,.01)
        with self.assertRaises(ValueError):decompose(np.ones((1,10)))
        rng=np.random.default_rng(9);ev,u=decompose(rng.normal(size=(3,10))+1j*rng.normal(size=(3,10)))
        self.assertEqual(len(ev),3);self.assertEqual(u.shape,(10,3))
        np.testing.assert_allclose(u.conj().T@u,np.eye(3),atol=1e-14)
        self.assertIsNone(overlap(u,u,4))

    def test_frozen_response_keeps_zero_support_unavailable(self):
        from compare_ptc_spod_rank import frozen_response
        basis=np.array([[1.],[1.],[1.]])/np.sqrt(3)
        mask=np.ones((600,3),bool);mask[:10]=False;mask[270:290,0]=False
        r=frozen_response(basis,mask)
        self.assertGreaterEqual(r['energy_fraction'],0)
        self.assertLessEqual(r['energy_fraction'],1)
        self.assertAlmostEqual(r['template_amplitude'],r['energy_fraction'],places=12)

    def test_diagonal_power_concentration_is_not_coherence(self):
        rng=np.random.default_rng(1909)
        q=(rng.normal(size=(2000,6))+1j*rng.normal(size=(2000,6)))*[20,1,1,1,1,1]
        e,u=decompose(q)
        en,_=decompose(q*np.exp(1j*rng.uniform(-np.pi,np.pi,q.shape)))
        self.assertGreater(e[0]/e.sum(),.95)
        self.assertLess(1/np.sum(abs(u[:,0])**4),1.1)
        self.assertLess(abs(e[0]/e.sum()-en[0]/en.sum()),.001)

    def fixture(self,p):
        c=p/'donor-continuity/cal';c.mkdir(parents=True)
        profile=p/'profile.json';profile.write_text(json.dumps(dict(factor=1)))
        cfg=p/'input.json';cfg.write_text(json.dumps(dict(network=12,observation=152390,filter_plan=dict(path=str(profile),sha256=digest(profile)),detectors=[dict(channel=3)])))
        t=np.arange(20)*.01;t.tofile(p/'native-time.f64')
        d=dict(channel=3,detector=0,occurrence='occurrence:row=6')
        l=dict(configuration_sha256=digest(cfg),network=12,observation=152390,explicit_array_lowpass=dict(factor=1),native_time_sha256=digest(p/'native-time.f64'),rows=20,native_integration_seconds=.01,native_runs=[[0,20]],detectors=[d])
        (p/'learning-receipt.yaml').write_text(yaml.safe_dump(l))
        causes=np.zeros(20,dtype='<u2');causes[4:6]=1;slots=np.flatnonzero(causes==0).astype('<i8');values=slots.astype('<f8')
        files={}
        for name,a in [('channel-3-causes.u16',causes),('channel-3-slot.i64',slots),('channel-3-value.f64',values)]:a.tofile(c/name);files[name]=digest(c/name)
        slots.tofile(p/'donor-continuity/3-rows.i64')
        r=dict(schema='citlali-cal-output-v1',unit='mJy/nominal-beam',detectors=[dict(channel=3,selected_row='apt:row=6',available=len(slots),files=files)])
        (c/'receipt.yaml').write_text(yaml.safe_dump(r));return cfg

    def test_reader_keeps_invalid_separate_and_detects_changed_identity(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);cfg=self.fixture(p);ledger=Inputs();data=read_cal(p,cfg,ledger)
            self.assertFalse(data.valid[4,0]);self.assertTrue(np.isnan(data.values[4,0]));self.assertTrue(ledger.unchanged())
            a=p/'donor-continuity/cal/channel-3-value.f64';a.write_bytes(a.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError,'input changed'):ledger.unchanged()
            with self.assertRaisesRegex(ValueError,'checksum'):read_cal(p,cfg,Inputs())


if __name__=='__main__':unittest.main()
