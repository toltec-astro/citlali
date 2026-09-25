"""Bounded integration controls: shared fit, measured operand, stop and seed expiry."""
import numpy as np
import pytest
from test_point_expected import setup
from point_guarded_feedback import infer,publish,continuation_reason,ideal_comparison,POLICIES
from point_response_cd import seed_candidate
from point_fruit_diagnostics import load
from run_point_guarded_feedback import request
import point_expected


def fixture():
    n,data,f,psd,p=setup();f.parameters=p;f.x=n.x;f.y=n.y;f.major=max(p[3:5])*np.sqrt(8*np.log(2));f.radius=np.hypot(n.x-p[1],n.y-p[2]);f.background=n.disk&(f.radius>3*f.major);f.edges=np.arange(0,42,2)
    unit=np.exp(-.5*((n.x-p[1])**2+(n.y-p[2])**2)/9);f.weights=unit/np.sum(unit**2)
    return dict(fixed=f,header=f.header,neighborhood=n,scale=n.scale,psd=psd,k0=n.k0),data,np.ones(data.shape,bool)


def test_one_fit_shared_by_measurement_and_three_policies(tmp_path,monkeypatch):
    c,data,support=fixture();calls=[];solve=point_expected.solve_gaussian
    def wrapped(*a,**k):calls.append(1);return solve(*a,**k)
    monkeypatch.setattr(point_expected,'solve_gaussian',wrapped)
    m,prepared=infer(c,data,support)
    assert m['guard']['accepted'] and m['fit']['parameters']==m['guard']['accepted_parameters']
    for pol in POLICIES:
        p,s=publish(tmp_path/pol,c,data,prepared,pol,4.);im,_=load(p);use=im['ADMITTED'].astype(bool)
        np.testing.assert_array_equal(im['CANDIDATE'][use],(data-m['fit']['parameters'][6])[use])
        assert continuation_reason(m['guard'],s,im,c['header'])==(None,None)
    assert len(calls)==1
    for raw,relative,bg in zip(m['raw_apertures'],m['apertures'],m['aperture_background_terms']):
        assert raw['value_mJy_per_beam_arcsec2']-bg==pytest.approx(relative['value_mJy_per_beam_arcsec2'])


def test_rejected_map_persists_empty_and_computational_error_stays_distinct(tmp_path):
    c,data,support=fixture();m,prepared=infer(c,np.ones_like(data)*120,support)
    for pol in POLICIES:
        p,s=publish(tmp_path/pol,c,np.ones_like(data)*120,prepared,pol,4.);im,_=load(p)
        assert not im['ADMITTED'].any() and np.all(im['CANDIDATE'][im['VALID'].astype(bool)]==0)
        assert continuation_reason(m['guard'],s,im,c['header'])[0]=='source_retention_miss'
        assert m['fit']['status']=='failed' and 'parameters' not in m['fit']
    support[20,20]=False;m,prepared=infer(c,data,support)
    p,s=publish(tmp_path/'bad',c,data,prepared,'CORR',4.)
    assert continuation_reason(m['guard'],s,load(p)[0],c['header'])[0]=='computational_unavailable'
    with pytest.raises(ValueError,match='threshold'):publish(tmp_path/'missing',c,data,prepared,'CORR',None)


def test_seed_exception_expires_and_guard_cannot_be_bypassed(tmp_path):
    c,data,support=fixture();m,prepared=infer(c,data,support);p,_=publish(tmp_path/'initial',c,data,prepared,'BASE-POINT',3.)
    original,_=load(p);raster=np.zeros(data.shape);raster[10,10]=.1
    seed=seed_candidate(original,'D-seed-shoulder',raster);assert seed['ADMITTED'][10,10]
    # Later replacement receives only the actual map and normal fixed policy.
    p,s=publish(tmp_path/'later',c,data,prepared,'BASE-POINT',3.);replaced,_=load(p)
    assert not replaced['ADMITTED'][10,10] and replaced['CANDIDATE'][10,10]==0
    failed=dict(m['guard'],accepted=False,status='failed',cause='insufficient source',message='POINT fit failed',accepted_parameters=None)
    assert continuation_reason(failed,s,seed,c['header'])[0]=='source_retention_miss'


def test_ideal_measurement_noop_and_known_addition_do_not_refit_stored_maps():
    c,data,support=fixture();base,_=infer(c,data,support);known=data*.01
    actual,_=infer(c,data+known,support)
    result=ideal_comparison(c,base,actual,actual)
    assert all(r['absolute_discrepancy']==0 for r in result['comparisons'])
    no=ideal_comparison(c,base,base,base)
    assert all(r['ratio'] is None for r in no['comparisons'])


def test_request_bound_and_branch_input_isolation(tmp_path):
    f=tmp_path/'bound';f.write_text('{}')
    p=dict(original_owner=dict(parents={'a1100':dict(map_request={},parent_plan={})},injections={'a1100':{'C-core-plus':{'sha256':'injection'}}}))
    prior=dict(ready=True,cycle_index=0,candidate={'sha256':'own-prior'},state_path=str(f))
    for pol in POLICIES:
        for b in ('REF','C-core-plus','D-seed-shoulder'):
            q=request(p,f,'a1100',pol,b,1,prior);assert ('injection' in q)==(b=='C-core-plus');assert q['candidate']==prior['candidate']
    for b,k in [('REF',0),('REF',11),('D-seed-low',1)]:
        with pytest.raises(ValueError):request(p,f,'a1100','BASE-POINT',b,k,prior)
    with pytest.raises(ValueError,match='predecessor'):request(p,f,'a1100','CORR','REF',1,dict(prior,ready=False))
