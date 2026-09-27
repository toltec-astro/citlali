import numpy as np
from point_rtc_diagnostics import spectrum,coherence


def test_one_sided_hann_density_has_correct_sine_variance_and_resolution():
    dt=.01;n=400;t=np.arange(n)*dt
    psd,_=spectrum(np.sqrt(2)*np.sin(2*np.pi*11*t),dt)
    f=np.fft.rfftfreq(n,dt)
    assert f[np.argmax(psd)]==11
    np.testing.assert_allclose(psd.sum()/(n*dt),1,rtol=1e-6)
    assert f[1]-f[0]==.25


def test_pair_coherence_uses_only_common_finite_windows():
    rng=np.random.default_rng(7);z=rng.normal(size=20)+1j*rng.normal(size=20)
    a=np.column_stack((z,2j*z,np.full(20,np.nan)));a[:4,1]=np.nan
    q=coherence(a)
    assert q['n']==1 and q['common_windows']['median']==16
    np.testing.assert_allclose(q['median'],1,atol=1e-14)
