import numpy as np
import pytest
from point_admission import anchor_pixel,admit,evaluate,fit_diagnostic,summarize_trials,CASES
from validate_point_empirical import fixture
from inspect_point_source import gaussian
from point_additive import linear_weights


def setup():
    h,x,y,p=fixture(41)
    support=np.ones(x.shape,bool);scale=np.ones(x.shape);valid=support.copy()
    p=np.array([100.,6.,-4.,3.,4.,.2,17.])
    return h,x,y,p,support,scale,valid


def test_anchor_uses_measured_centroid_signed_grid_and_half_tie():
    h,x,y,p,s,n,v=setup()
    r,c=anchor_pixel(p[1:3],h)
    assert x[r,c]==6 and y[r,c]==-4
    # One half pixel from zero-based index20, tie toward index21 even for CDELT<0.
    center=[.5*h['CDELT1']*3600,.5*h['CDELT2']*3600]
    assert anchor_pixel(center,h)==(21,21)


def test_eight_connected_anchor_island_no_global_peak_fallback():
    h,x,y,p,s,n,v=setup();r,c=anchor_pixel(p[1:3],h)
    data=np.full(x.shape,p[6]);data[r,c]=p[6]+4;data[r+1,c+1]=p[6]+5
    data[r-5,c-5]=p[6]+500
    a=admit(data,s,n,v,h,p,3,1)
    assert a['summary']['pixels']==2 and a['summary']['unassociated_above_threshold_pixels']==1
    data[r,c]=p[6]+2
    a=admit(data,s,n,v,h,p,3,1)
    assert a['summary']['status']=='anchor_below_threshold' and not a['selected'].any()
    p[1]=100
    assert admit(data,s,n,v,h,p,3,1)['summary']['status']=='anchor_outside_domain'


def test_invalid_anchor_no_extrapolation_and_valid_zero_exclusions():
    h,x,y,p,s,n,v=setup();data=gaussian(p,x,y);r,c=anchor_pixel(p[1:3],h)
    for mode in ['mask','nan','zero','negative','map']:
        scale=n.copy();valid=v.copy();support=s.copy()
        if mode=='mask':valid[r,c]=False
        elif mode=='map':support[r,c]=False
        else:scale[r,c]={'nan':np.nan,'zero':0,'negative':-1}[mode]
        a=admit(data,support,scale,valid,h,p,3,1)
        assert a['summary']['status']=='anchor_invalid' and np.isnan(a['candidate'][r,c])
        assert np.all(a['candidate'][a['valid']]==0)
    a=admit(data,s,n,v,h,p,3,1)
    assert np.isnan(a['candidate'][~a['domain']]).all()


def test_background_excluded_scale_equivalence_and_nonmutation():
    h,x,y,p,s,n,v=setup();data=gaussian(p,x,y);before=[a.copy() for a in [data,s,n,v,p]]
    a=admit(data,s,n,v,h,p,3,2);b=admit(data,s,n,v,h,p,6,1)
    assert np.array_equal(a['selected'],b['selected'])
    assert np.array_equal(a['candidate'],b['candidate'],equal_nan=True)
    shifted=p.copy();shifted[6]+=100
    d=admit(data+100,s,n,v,h,shifted,3,2)
    assert np.allclose(a['candidate'],d['candidate'],equal_nan=True)
    assert np.all(a['candidate'][a['selected']]>0)
    for original,old in zip([data,s,n,v,p],before):assert np.array_equal(original,old)


def test_boundary_and_adjacent_unsupported_contact():
    h,x,y,p,s,n,v=setup();p[1:3]=0;data=np.full(x.shape,p[6]+4)
    a=admit(data,s,n,v,h,p,3,1)
    assert a['summary']['boundary_contact']
    v[20,22]=False
    a=admit(data,s,n,v,h,p,3,1)
    assert a['summary']['unsupported_contact'] and a['summary']['unsupported_domain_pixels']==1
    f=fit_diagnostic(a['candidate'],a['valid'],h,6)
    assert f['status']=='unavailable' and 'incomplete' in f['cause']


def test_candidate_fits_and_fixed_projection_additivity():
    h,x,y,p,s,n,v=setup();data=gaussian(p,x,y);q,_=linear_weights(h,x.shape,p)
    reference,outputs=evaluate(data,s,n,v,h,p,6,q)
    assert reference['fit']['parameters'][0]==pytest.approx(100,abs=1e-7)
    assert reference['fit']['parameters'][6]==pytest.approx(0,abs=1e-7)
    masks=[outputs[k]['selected'] for k in ['scale_stress','restrictive','reference','permissive']]
    assert all(np.all(~a|b) for a,b in zip(masks,masks[1:]))
    for r in outputs.values():
        assert r['summary']['fit']['status']=='success'
        assert r['summary']['fixed_projection']+r['summary']['excluded_fixed_projection']==pytest.approx(reference['fixed_projection'])
    # No circular/Gaussian replacement: one admitted perturbation is copied exactly.
    data[18,18]+=2
    a=admit(data,s,n,v,h,p,3,1)
    assert a['candidate'][18,18]==data[18,18]-p[6]


def test_trial_failures_and_constraints_are_not_dropped(monkeypatch):
    import point_admission as module
    h,x,y,p,s,n,v=setup();q,_=linear_weights(h,x.shape,p)
    ref,outputs=evaluate(gaussian(p,x,y),s,n,v,h,p,6,q)
    cases={name:r['summary'] for name,r in outputs.items()}
    for row in cases.values():row['fit']['constrained']=True
    records=[dict(raw_fit={'status':'success'},cases=cases),dict(raw_fit={'status':'failed'},cases={})]
    summary=summarize_trials(records)
    assert summary['reference']['constrained']==1
    assert summary['reference']['raw_fit_failures']==1 and summary['reference']['partial_distribution_only']
    def fail(*args):raise ValueError('deliberate failure')
    monkeypatch.setattr(module,'measure',fail)
    assert fit_diagnostic(gaussian(p,x,y),s,h,6)=={'status':'failed','cause':'deliberate failure'}
