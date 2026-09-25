from types import SimpleNamespace
import numpy as np
import pytest
import point_expected as expected
from inspect_point_source import gaussian
from point_additive import variance
from test_point_corr_admission import fixture


def setup(p=None):
    n,data,f,fit,b,psd=fixture(correlated=True)
    p=np.array([1000.,2.,-3.,3.2,4.1,.3,120.] if p is None else p)
    f.disk=n.disk
    return n,gaussian(p,n.x,n.y),f,psd,p


def test_strict_quality_boundaries_and_upper_nuisance_bounds():
    p=np.array([100.,20.,0.,1.5,2.,0.,10.]);active=np.zeros(7,int)
    assert expected.geometry_rejection(p,active)
    p[3]=np.nextafter(1.5,np.inf)
    assert expected.geometry_rejection(p,active) is None
    active[3]=-1; assert expected.geometry_rejection(p,active)
    active[:]=0;p[1]=np.nextafter(20.,np.inf)
    assert expected.geometry_rejection(p,active)
    p[1]=20;p[2]=0;p[3]=20;active[3]=1
    assert expected.geometry_rejection(p,active) is None
    p[0]=0;assert expected.geometry_rejection(p,active)
    assert expected.snr_rejection(100.,10.)
    assert expected.snr_rejection(np.nextafter(100.,np.inf),10.) is None
    assert expected.SIGMA_FLOOR*np.sqrt(8*np.log(2))==pytest.approx(3.53223006755)


def test_all_parameter_jacobian_and_local_refit_sensitivity():
    n,data,f,psd,p=setup();roi=n.disk;x=n.x[roi];y=n.y[roi]
    J=expected.jacobian(p,x,y)
    for k in range(7):
        delta=1e-5;hi=p.copy();lo=p.copy();hi[k]+=delta;lo[k]-=delta
        assert np.allclose((gaussian(hi,x,y)-gaussian(lo,x,y))/(2*delta),J[:,k],rtol=2e-6,atol=2e-7)
    q,info=expected.amplitude_sensitivity(J)
    assert info['rank']==7
    assert np.allclose(J.T@q,np.eye(7)[0],atol=1e-11)
    base=expected.measure_expected(data,np.ones(data.shape,bool),f,n.scale,psd)
    assert base['accepted'] and base['optimizer_calls']==1
    perturb=np.cos(n.x/7)*np.sin(n.y/9)
    e=.001
    hi=expected.measure_expected(data+e*perturb,np.ones(data.shape,bool),f,n.scale,psd)
    lo=expected.measure_expected(data-e*perturb,np.ones(data.shape,bool),f,n.scale,psd)
    empirical=(hi['accepted_parameters'][0]-lo['accepted_parameters'][0])/(2*e)
    assert empirical==pytest.approx(np.sum(q*perturb[roi]),rel=2e-4,abs=1e-5)


def test_full_covariance_small_contraction_and_circular_identifiability():
    n,data,f,psd,p=setup([800.,0.,0.,3.,3.,0.,120.])
    use=(abs(n.x)<=4)&(abs(n.y)<=4)
    q,info=expected.amplitude_sensitivity(expected.jacobian(p,n.x[use],n.y[use]))
    assert info['rank']==6 and q is not None
    weights=np.zeros(data.shape);weights[use]=q
    loc=np.argwhere(use);rho=np.fft.ifft2(psd).real
    direct=sum(q[i]*q[j]*rho[(r-s)%100,(c-d)%100] for i,(r,c) in enumerate(loc) for j,(s,d) in enumerate(loc))
    assert variance(psd,n.scale,weights)==pytest.approx(direct,rel=1e-12)
    fit=expected.measure_expected(data,np.ones(data.shape,bool),f,n.scale,psd)
    assert fit['accepted'] and fit['sigma_A']>0
    J=np.ones((30,7));q,info=expected.amplitude_sensitivity(J)
    assert q is None and info['rank']==1


def test_constant_map_rejected_empty_and_state_cleared():
    n,data,f,psd,p=setup();support=np.ones(data.shape,bool)
    accepted=expected.measure_expected(data,support,f,n.scale,psd)
    assert expected.candidate(data,support,f,accepted)['selected'].any()
    constant=np.full(data.shape,19.)
    rejected=expected.measure_expected(constant,support,f,n.scale,psd)
    assert rejected['complete'] and not rejected['accepted'] and rejected['optimizer_calls']==1
    assert rejected['accepted_parameters'] is None and rejected['pointing_correction'] is None
    assert accepted['pointing_correction'] is None and accepted['accepted_centroid_tangent_arcsec'] is not None
    assert rejected['accepted_centroid_tangent_arcsec'] is None
    empty=expected.candidate(constant,support,f,rejected)
    assert not empty['selected'].any()
    assert np.all(empty['candidate'][empty['valid']]==0)
    assert np.isnan(empty['candidate'][~empty['valid']]).all()
    assert np.all(constant==19.)


def test_budget_rejection_nonfinite_error_and_input_unavailability(monkeypatch):
    n,data,f,psd,p=setup();support=np.ones(data.shape,bool)
    raw=SimpleNamespace(x=p,fun=np.zeros(n.disk.sum()),jac=expected.jacobian(p,n.x[n.disk],n.y[n.disk]),
        cost=0.,optimality=1.,active_mask=np.zeros(7,int),success=False,status=0,message='budget exhausted',nfev=3000,njev=1)
    calls=[]
    def solve(*args,**kwargs):calls.append(kwargs);return raw
    monkeypatch.setattr(expected,'solve_gaussian',solve)
    failed=expected.measure_expected(data,support,f,n.scale,psd)
    assert failed['complete'] and not failed['accepted'] and failed['diagnostic']['nfev']==3000
    assert calls==[dict(sigma_floor=1.5)]
    raw.fun[0]=np.nan
    assert not expected.measure_expected(data,support,f,n.scale,psd)['complete']
    raw.fun[0]=0;raw.success=True
    scale=n.scale.copy();scale[20,20]=np.nan
    assert not expected.measure_expected(data,support,f,scale,psd)['complete']
    bad=data.copy();bad[20,20]=np.nan
    assert not expected.measure_expected(bad,support,f,n.scale,psd)['complete']
    def unexpected(*a,**k):raise RuntimeError('unexpected failure')
    monkeypatch.setattr(expected,'solve_gaussian',unexpected)
    with pytest.raises(RuntimeError,match='unexpected'):
        expected.measure_expected(data,support,f,n.scale,psd)


def test_acceptance_uses_amplitude_above_constant_and_common_empty_supplement():
    n,data,f,psd,p=setup();support=np.ones(data.shape,bool)
    a,b,fields=expected.infer_expected(data,support,f,n,n.scale,psd)
    shift=expected.measure_expected(data+10000,support,f,n.scale,psd)
    assert shift['SNR_A']==pytest.approx(a['SNR_A'],rel=1e-7)
    assert np.array_equal(b['candidate'][b['selected']],(data-a['accepted_parameters'][6])[b['selected']])
    a,b,fields=expected.infer_expected(np.full(data.shape,19.),support,f,n,n.scale,psd)
    assert not a['accepted'] and not b['selected'].any() and fields is None


def test_failed_statistical_or_numerical_screen_never_publishes_candidate(tmp_path):
    from run_point_expected import screen_eligibility,publish
    from point_fruit_diagnostics import load
    n,data,f,psd,p=setup();support=np.ones(data.shape,bool)
    prepared=expected.infer_expected(data,support,f,n,n.scale,psd)
    c=dict(header=f.header,k0=n.k0)
    screens={'a1100':{'CORR':dict(comparison_usable=False)}}
    cause=screen_eligibility('CORR','a1100',screens)
    assert cause and screen_eligibility('BASE-POINT','a1100',screens) is None
    product=publish(tmp_path/'failed',c,data,prepared,'CORR',3.,blocked_cause=cause)
    images,_=load(product['path']);assert 'CANDIDATE' not in images and 'ADMITTED' not in images
    prepared[2]['policies']['CORR']['numerical_failures']=1
    product=publish(tmp_path/'error',c,data,prepared,'CORR',3.)
    images,_=load(product['path']);assert 'CANDIDATE' not in images
