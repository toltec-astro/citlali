"""Focused map-domain tests; no native reduction or source-response validation."""
import numpy as np
import pytest
from types import SimpleNamespace
from point_additive import NoiseRequest,learn,variance,linear_weights
from point_empirical import coordinates,source_patch
from point_post_fruit import (frozen_masks,context_for,operators,saved_measure,measurement_values,
    ensemble,joint_summary,trial,dense_variance,larger_same_covariance,blank_remeasure)
from inspect_point_source import gaussian
from validate_point_empirical import fixture
from point_admission import fit_diagnostic


def context():
    h,x,y,p=fixture(129);p[6]=12
    support=np.hypot(x,y)<120;cov=np.where(support,1000.,0.)
    d=gaussian(p,x,y)+np.random.default_rng(821).normal(0,8,x.shape)
    m=learn(d,cov,support,h,p,NoiseRequest(lag_pixels=16,bins=1),stamp_size=49)
    data,v,_,sh,_=source_patch(d,support,cov,h);xx,yy=coordinates(data.shape,sh)
    q,_=linear_weights(sh,data.shape,p)
    f=SimpleNamespace(beam=6.,parameters=p,weights=q,major=max(p[3:5])*np.sqrt(8*np.log(2)),radius=np.hypot(xx-p[1],yy-p[2]))
    c=context_for(d,cov,support,h,f,p,m)
    return c,f,h,p,d,cov,support


def test_fixed_cycle_masks_union_and_no_value_driven_selection():
    c,f,h,p,d,cov,s=context()
    images=[dict(PRIMARY=d,SCI_SUPPORT=s,NORMALIZATION=cov) for _ in range(5)]
    p2=p.copy();p2[1]+=4;p2[3]*=1.3
    common,mask,_=frozen_masks(images,h,[p,p2],4.)
    common2,larger,_=frozen_masks(images,h,[p,p2],5.)
    assert np.all(larger[mask]) and np.array_equal(common,common2)
    images[-1]=dict(PRIMARY=d*500-200,SCI_SUPPORT=s,NORMALIZATION=cov)
    a,b,_=frozen_masks(images,h,[p,p2],4.)
    assert np.array_equal(a,common) and np.array_equal(b,mask)
    missing=s.copy();missing[60,60]=False;images[-1]['SCI_SUPPORT']=missing
    a,_,_=frozen_masks(images,h,[p,p2],4.);assert not a[60,60]
    req=NoiseRequest(lag_pixels=16,bins=1)
    model=learn(d,cov,common,h,p,req,source_mask=mask)
    assert np.array_equal(model['background_mask'],common&~mask)


def test_original_apertures_and_trial_background_cancel_constant():
    c,*_=context();r=saved_measure(c['baseline'],c);r2=saved_measure(c['baseline']+31,c)
    assert np.allclose(measurement_values(r),measurement_values(r2),rtol=1e-8,atol=1e-5)
    p=np.array(r2['fit']['parameters']);assert p[6]==pytest.approx(43,abs=1e-7)
    for i,a in enumerate(r['apertures']):
        assert a['radius_arcsec']==pytest.approx([1,1.5,2][i]*c['major'])
        assert a['value_mJy_per_beam_arcsec2']==pytest.approx(np.sum(c['weights'][i+1]*(c['baseline']-p[6]+31)))


def test_same_covariance_larger_domain_dense_and_modulated_operator():
    c,*_=context();large=larger_same_covariance(c['psd'])
    assert large.mean()==pytest.approx(1)
    c['scale']*=np.linspace(.7,1.3,c['scale'].shape[1])[None,:]
    for q in c['weights']:
        v=variance(c['psd'],c['scale'],q)
        assert v==pytest.approx(dense_variance(c['psd'],c['scale'],q),rel=1e-12)
        assert v==pytest.approx(variance(large,c['scale'],q),rel=1e-12)


def test_different_worker_scheduling_and_coupled_background_cross_term():
    c,*_=context()
    a=ensemble(c,8,42,'a1100','base',1);b=ensemble(c,8,42,'a1100','north',2)
    assert [r['values'] for r in a]==[r['values'] for r in b]
    assert [r['seed'] for r in a]==[r['seed'] for r in b]
    bg=np.array([r['fit']['parameters'][6] for r in a]);raw=np.array([r['linear_noise'] for r in a])
    for i,w in enumerate(c['weights']):
        vals=np.array([r['values'][i+1] for r in a]);s=w.sum()
        exact=raw[:,i].var(ddof=1)+s*s*bg.var(ddof=1)-2*s*np.cov(raw[:,i],bg,ddof=1)[0,1]
        assert vals.var(ddof=1)==pytest.approx(exact,rel=1e-10)
    result=joint_summary(a,c['noiseless']);assert result['covariance'].shape==(5,5)
    assert result['measurements'][0]['scatter']==pytest.approx(np.std([r['values'][0] for r in a],ddof=1))


def test_failure_preserved_primary_withheld_independent_linear_survives(monkeypatch):
    import point_post_fruit as m
    c,*_=context()
    monkeypatch.setattr(m,'fit_diagnostic',lambda *a:dict(status='failed',cause='intentional'))
    r=trial((c,(2,3,4),{},0));assert r['status']=='failed'
    assert np.isfinite(r['linear_noise']).all() and r['values']==[None]*5
    agg=joint_summary([r,r],c['noiseless']);assert not agg['primary_available']
    assert all(a['scatter'] is None for a in agg['measurements'])


def test_missing_measurement_support_unavailable():
    c,*_=context();c['support'][20,20]=False
    r=saved_measure(c['baseline'],c)
    assert r['fit']['status']=='unavailable' and measurement_values(r)==[None]*5


def test_new_blank_mask_filters_locations_and_cycle_values_recomputed():
    c,f,h,p,d,cov,s=context();model=dict(background_mask=s.copy(),scale=np.ones(s.shape),valid=s,psd=c['psd'])
    prior=dict(trials=[dict(row=64,column=64),dict(row=90,column=64)])
    model['background_mask'][64,64]=False
    a=blank_remeasure(d,cov,s,model,c,prior);b=blank_remeasure(d+10,cov,s,model,c,prior)
    assert not a['locations'][0]['available'] and a['locations'][1]['available']
    va=np.array(a['locations'][1]['values']);vb=np.array(b['locations'][1]['values'])
    assert np.allclose(vb-va,10*c['weights'].sum(axis=(1,2)))
