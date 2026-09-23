import numpy as np
import pytest
from dataclasses import replace
from scipy.signal import fftconvolve
from point_additive import (NoiseRequest,learn,pair_covariance,spectrum_from_pairs,variance,draw_noise,
                            aggregate,make_context,experiment,projected_ranges,influence)
from validate_point_empirical import fixture
from inspect_point_source import gaussian,measure


def test_mask_corrected_pairs_against_direct_signed_products():
    z=np.arange(99,dtype=float).reshape(9,11)-49;m=np.ones(z.shape,bool);m[2:5,4:7]=False
    cv,n=pair_covariance(z,m,3,2)
    for dy,dx in [(0,0),(2,-3),(-1,2)]:
        products=[]
        for y,x in np.argwhere(m):
            j,i=y-dy,x-dx
            if 0<=j<9 and 0<=i<11 and m[j,i]:products.append(z[y,x]*z[j,i])
        assert n[dy+3,dx+3]==len(products)
        assert cv[dy+3,dx+3]==pytest.approx(np.mean(products))


def test_signed_anisotropic_covariance_propagation_generation_and_padding():
    k=np.array([[0.,0.,0.],[.9,1.,-.6],[0.,.25,0.]]);k/=np.linalg.norm(k)
    cv=fftconvolve(k,k[::-1,::-1]);psd,repair=spectrum_from_pairs(cv,2,13)
    assert psd.min()>=0 and psd.mean()==pytest.approx(1)
    rho=np.fft.ifft2(psd).real
    assert rho[0,2]<0 and rho[1,0]!=pytest.approx(rho[0,1])
    q=np.zeros((13,13));q[6,4:9]=[.2,.4,1,.4,.2];s=np.linspace(1,2,169).reshape(13,13)
    theory=variance(psd,s,q)
    samples=np.array([np.sum(q*draw_noise(psd,s,[913,i])) for i in range(1400)])
    assert samples.var(ddof=1)==pytest.approx(theory,rel=.11)
    p2,_=spectrum_from_pairs(cv,2,13,domain_factor=2)
    assert variance(p2,s,q)==pytest.approx(theory,rel=1e-12)
    # Independently assemble the full crop covariance, including distant corners.
    coords=np.indices(q.shape).reshape(2,-1).T;dy=coords[:,0,None]-coords[None,:,0];dx=coords[:,1,None]-coords[None,:,1]
    a=(q*s).ravel();direct=np.sum(a[:,None]*rho[dy%psd.shape[0],dx%psd.shape[1]]*a[None,:])
    assert theory==pytest.approx(direct,rel=1e-12)


def test_amplitude_is_second_moment_no_clipping_and_extrapolation_flag():
    h,x,y,p=fixture(81);s=np.hypot(x,y)<75;q=np.where(s,100.,0.);q[38:43,38:43]=200.
    rng=np.random.default_rng(12);d=rng.normal(size=x.shape)+19;d[10,40]=40
    m=learn(d,q,s,h,p,NoiseRequest(lag_pixels=12,bins=1))
    assert m['summary']['mean_removed']>18
    assert not m['valid'][40,40] and np.isfinite(m['scale'][40,40])
    bg=m['background_mask'];expected=np.sum((d[bg]-d[bg].mean())**2)/(bg.sum()-1)
    assert m['summary']['bins'][0]['second_moment_about_global_mean']==pytest.approx(expected)
    assert m['summary']['histogram']['excess_kurtosis']>1


def test_failures_withhold_but_constraints_stay_in_scatter():
    trials=[dict(status='success',amplitude=a,constraint_active=(i==0),fixed_noise_amplitude=a-3) for i,a in enumerate([1.,2.,3.,4.])]
    r=aggregate(trials,2.,1.);assert r['candidate_refit_scatter']==pytest.approx(np.std([1,2,3,4],ddof=1));assert r['bound_limited']==1
    trials[1]=dict(status='failed',fixed_noise_amplitude=-1.)
    r=aggregate(trials,2.,1.);assert r['candidate_refit_scatter'] is None and r['partial_survivor_scatter'] is not None


def test_original_support_and_actual_estimator_reference():
    h,x,y,p=fixture(81);h['COMMENT']='First original FITS comment';h['COMMENT']='Second original FITS comment';p[6]=7;support=np.hypot(x,y)<75;q=np.where(support,100.,0.)
    d=gaussian(p,x,y)+np.random.default_rng(41).normal(size=x.shape)
    model=learn(d,q,support,h,p,NoiseRequest(lag_pixels=12,bins=1))
    c=make_context(d,q,support,h,6.,p,model)
    assert c['A0']==pytest.approx(200.,abs=1e-7)
    assert c['complete_amplitude_support']
    w=influence(c);assert abs(w.sum())<1e-7
    ranges=projected_ranges(c,c['q']);assert ranges[-1]['signed_partial_variance']==pytest.approx(variance(c['psd'],c['scale'],c['q']))
    s=support.copy();s[40,40]=False
    with pytest.raises(ValueError,match='incomplete'):make_context(d,q,s,h,6.,p,model)
    a,ta=experiment(c,4,91,1);b,tb=experiment(c,4,91,1)
    assert [r['parameters'] for r in ta]==[r['parameters'] for r in tb]


def test_all_failed_sensitivity_and_null_report_are_preserved():
    from run_point_additive import sensitivity_range,report_row
    failed=[dict(status='failed',fixed_noise_amplitude=v) for v in [-1.,1.,-.5,.5]]
    s=aggregate(failed,200.,1.)
    r=[dict(name='base',experiment=s),dict(name='mask4',status='unavailable')]
    assert sensitivity_range(r) is None
    s.update(array='a1100',unchanged_measured_amplitude=200.,sensitivity_scatter_range=None,amplitude_supported=True)
    row=report_row(s)
    assert row.count('unavailable')==3 and '4 / 4 / 0' in row
    unavailable=report_row(dict(array='a2000',status='unavailable',cause='unsupported model',unchanged_measured_amplitude=600.))
    assert 'unsupported model' in unavailable


def test_one_trial_records_failure(monkeypatch):
    import point_additive as m
    def failed(*args,**kwargs):raise ValueError('deliberate fit failure')
    monkeypatch.setattr(m,'measure',failed)
    c=dict(psd=np.ones((16,16)),scale=np.ones((5,5)),support=np.ones((5,5),bool),baseline=np.zeros((5,5)),q=np.ones((5,5))/25,header={},beam=5.)
    r=m.one_trial((c,(9,1100,0),0))
    assert r['status']=='failed' and r['cause']=='deliberate fit failure' and np.isfinite(r['fixed_noise_amplitude'])
