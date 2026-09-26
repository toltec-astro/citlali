"""Focused frozen-method bootstrap, amplitude normalization and scope tests."""
import numpy as np
import pytest
from point_additive import learn,NoiseRequest
from point_expected import measure_expected
from test_point_expected import setup
from point_empirical import coordinates
from point_fruit_diagnostics import load
from point_empirical_io import write_fits
from point_empirical_io import exact_header
from run_independent_point import cleanup_cycle


def test_single_attempt_bootstrap_matches_precomputed_guard(monkeypatch):
    import point_expected as e
    n,data,f,psd,p=setup();v=np.ones(data.shape,bool)
    expected=measure_expected(data,v,f,n.scale,psd)
    calls=[];old=e.solve_gaussian
    def solve(*a,**k):calls.append(k);return old(*a,**k)
    monkeypatch.setattr(e,'solve_gaussian',solve)
    init=[]
    def covariance(parameters):init.append(parameters);return n.scale,psd
    actual=measure_expected(data,v,f,None,None,initialize_covariance=covariance)
    assert actual==expected and len(init)==len(calls)==1
    assert calls==[dict(sigma_floor=1.5)]


def test_bootstrap_failure_no_refit_or_accepted_source():
    n,data,f,psd,p=setup();calls=[]
    def bad(parameters):calls.append(parameters);raise ValueError('unsupported background')
    r=measure_expected(data,np.ones(data.shape,bool),f,None,None,initialize_covariance=bad)
    assert r['optimizer_calls']==1 and len(calls)==1 and not r['complete']
    assert r['accepted_parameters'] is None


def test_amplitude_only_matches_old_full_learner_with_missing_coverage():
    h=dict(CRPIX1=61.,CRPIX2=61.,CDELT1=-2/3600,CDELT2=2/3600)
    x,y=coordinates((121,121),h);rng=np.random.default_rng(781);coverage=1000/(1+(x*x+y*y)/3000);v=x*x+y*y<110**2
    signal=10+3*rng.normal(size=x.shape)+400*np.exp(-(x*x+y*y)/32);signal[~v]=np.nan
    p=np.array([400.,0,0,4,4,0,10]);req=NoiseRequest(mask_fwhm=3.)
    full=learn(signal,coverage,v,h,p,req,stamp_size=49);amp=learn(signal,coverage,v,h,p,req,stamp_size=49,amplitude_only=True)
    assert np.array_equal(full['valid'],amp['valid'])
    assert np.allclose(full['scale'],amp['scale'],rtol=1e-14,atol=1e-14,equal_nan=True)
    assert amp['summary']['spatial_covariance_learned'] is False


def test_temporary_cleanup_preserves_shared_injection_and_compact_records(tmp_path):
    p=tmp_path/'run/cycle/native/products';d=p/'network-11-segment-0';d.mkdir(parents=True)
    for name in ['cleaned.f64','operand.f64','model-support.u8','causes.u8','injection.f64','injected-parent.f64','basis.f64','mean.f64','receipt.yaml']:(d/name).write_bytes(b'123')
    (p/'operator-pixel.i32').write_bytes(b'123');(p/'total.fits').write_bytes(b'map')
    cleanup_cycle(tmp_path,p,keep_injected=True)
    assert not(d/'cleaned.f64').exists() and not(d/'causes.u8').exists()
    assert all((d/n).exists() for n in ['injection.f64','injected-parent.f64','basis.f64','mean.f64','receipt.yaml'])
    assert (p/'total.fits').exists()


def test_failed_native_attempt_is_counted_and_cannot_be_retried(tmp_path,monkeypatch):
    import json
    import point_processing as run
    from point_fruit_trajectory import bound
    (tmp_path/'freeze').mkdir();(tmp_path/'freeze/budget.json').write_text(json.dumps(dict(native_cells=1,peak_ceiling_bytes=1000000,free_space_reserve_bytes=0)))
    own=tmp_path/'owner.json';own.write_text(json.dumps(dict(policy='BASE-POINT')))
    req=tmp_path/'request.json';req.write_text(json.dumps(dict(owner_binding=bound(own),cycle_index=1,branch='REF')))
    calls=[]
    def fail(*args,**kwargs):calls.append(args);raise RuntimeError('failed native call')
    monkeypatch.setattr(run,'invoke',fail)
    with pytest.raises(RuntimeError,match='failed native'):run.call(tmp_path,tmp_path/'binary',req,tmp_path/'native')
    rows=[json.loads(x) for x in (tmp_path/'execution-attempts.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['kind']=='feedback' and run._RESERVED==0
    with pytest.raises(ValueError,match='duplicate logical'):run.call(tmp_path,tmp_path/'binary',req,tmp_path/'native')
    assert len(calls)==1
    rows=[dict(kind='feedback',policy='OTHER',branch='OTHER',cycle=k) for k in range(150)]
    (tmp_path/'execution-attempts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(ValueError,match='hard application attempt limit'):run.call(tmp_path,tmp_path/'binary',req,tmp_path/'native')
    assert len(calls)==1


def test_M0_control_uses_REF_support_and_preserves_signed_injection(tmp_path):
    from run_independent_point import initial_control
    from types import SimpleNamespace
    h=dict(CRPIX1=21.,CRPIX2=21.,CDELT1=-2/3600,CDELT2=2/3600,CRVAL1=0.,CRVAL2=0.,RADESYS='FK5',EQUINOX=2000.)
    x,y=coordinates((41,41),h);z=-5*np.exp(-(x*x+y*y)/32);z[np.hypot(x,y)>36]=0
    valid=np.ones(z.shape,dtype='u1');valid[0,0]=0;hits=valid*5
    ref=np.full(z.shape,12.);actual=ref+z
    for name,a in [('ref',ref),('actual',actual)]:write_fits(tmp_path/(name+'.fits'),exact_header(h),[('PRIMARY',a,'mJy/nominal beam'),('HITS',hits,'1'),('NORMALIZATION',hits,'1'),('SCI_SUPPORT',valid,'1')])
    write_fits(tmp_path/'raster.fits',exact_header(h),[('INJECTION',z,'mJy/nominal beam')])
    c=dict(fixed=SimpleNamespace(disk=np.hypot(x,y)<=40))
    r=initial_control(tmp_path/'ref.fits',tmp_path/'actual.fits',tmp_path/'raster.fits',c,tmp_path)
    assert abs(r['full']['g']-1)<1e-14 and r['full']['eta']<1e-14
    q,_=load(tmp_path/'response.fits');assert q['VALID'][0,0]==0 and np.isnan(q['CONTROL'][0,0])
    assert q['CONTROL'][20,20]==-5
