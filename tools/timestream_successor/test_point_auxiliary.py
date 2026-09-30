import copy
from types import SimpleNamespace
import numpy as np
import pytest
from point_auxiliary import geometry,weights,sample,projection,embed,slices,covariance_spectrum
from point_additive import linear_weights,variance
from point_empirical import coordinates
from inspect_point_source import gaussian
from point_expected import candidate,measure_expected
from point_guarded_feedback import measurement


def header(half=20,step=2.):
    return dict(CRPIX1=half+1,CRPIX2=half+1,CDELT1=-step/3600,CDELT2=step/3600,
                CRVAL1=120.,CRVAL2=20.,RADESYS='FK5',EQUINOX=2000.)


def test_contained_estimator_exact_and_no_feedback_change():
    h=header();p=np.array([220.,4.,7.,3.,2.,.3,-2.]);x,y=coordinates((41,41),h)
    g=geometry(h,x.shape,p);q,roi=weights(g);old,oldroi=linear_weights(h,x.shape,p)
    assert np.array_equal(q,old) and np.array_equal(roi,oldroi)
    s=gaussian(p,x,y);v=np.ones(s.shape,bool);payload=sample(s,v,h,g)
    assert projection(dict(auxiliary=g),payload,p[6])[0]==pytest.approx(p[0],rel=1e-15)
    f=SimpleNamespace(header=h,scale=np.ones(s.shape),noise_valid=v)
    guard=dict(accepted=True,accepted_parameters=p)
    before=candidate(s,v,f,guard);f.weights=q;after=candidate(s,v,f,guard)
    for name in ('candidate','valid','selected','domain','z'):
        assert np.array_equal(before[name],after[name],equal_nan=True)
    assert not after['valid'][np.hypot(x,y)>40].any()


def test_enlarged_offcenter_rotated_template_coordinates_and_missing_support():
    h=header();p=[220.,5.,8.,6.,2.,.7,-2.];g=geometry(h,(41,41),p);q,roi=weights(g)
    assert min(q.shape)>41
    fullh=header(70);x,y=coordinates((141,141),fullh);s=gaussian(p,x,y);v=np.ones(s.shape,bool)
    payload=sample(s,v,fullh,g)
    assert payload['available'] and projection(dict(auxiliary=g),payload,p[6])[0]==pytest.approx(220.,rel=1e-14)
    sl=slices(s.shape,fullh,g);gx,gy=coordinates(q.shape,g['header'])
    assert np.array_equal(x[sl],gx) and np.array_equal(y[sl],gy)
    ij=np.argwhere(roi)[0];start=np.array([sl[0].start,sl[1].start]);missing=tuple(ij+start)
    v[missing]=False
    bad=sample(s,v,fullh,g);assert not bad['available'] and bad['missing_pixels']==1
    assert projection(dict(auxiliary=g),bad,p[6])[0] is None
    assert np.array_equal(weights(g)[0],q) # No renormalization.
    v[missing]=True;s[missing]=np.nan
    assert sample(s,v,fullh,g)['nonfinite_pixels']==1
    assert not sample(s[:40,:40],v[:40,:40],fullh,g)['available']
    wrong=copy.deepcopy(g);wrong['shape']=[41,41];wrong['header']=h
    with pytest.raises(ValueError,match='clips'):weights(wrong)


def test_operator_embedding_preserves_noise_variance_and_constant_nuisance():
    h=header();g=geometry(h,(41,41),[220.,5.,8.,6.,2.,.7,-2.]);q,_=weights(g)
    rng=np.random.default_rng(7);op=rng.normal(size=(41,41));op2=embed(op,h,g)
    assert op2.sum()==pytest.approx(op.sum())
    assert variance(np.ones((128,128)),np.ones(op.shape),op)==pytest.approx(variance(np.ones((128,128)),np.ones(op2.shape),op2),rel=1e-14)
    constant=np.ones((41,41))/41**2
    net=q-q.sum()*embed(constant,h,g)
    assert abs(net.sum())<1e-14


def test_guard_fit_domain_and_parameters_independent_of_auxiliary():
    h=header();x,y=coordinates((41,41),h);p=[220.,4.7,7.5,4.6,2.1,.2,-2.]
    s=gaussian(p,x,y);v=np.ones(s.shape,bool);f=SimpleNamespace(header=h,beam=5.)
    before=measure_expected(s,v,f,np.ones(s.shape),np.ones((100,100)))
    g=geometry(h,s.shape,p);f.weights=weights(g)[0]
    after=measure_expected(s,v,f,np.ones(s.shape),np.ones((100,100)))
    assert before==after and after['accepted'] and after['fit_pixels']==int((np.hypot(x,y)<=40).sum())


def test_unavailable_auxiliary_does_not_suppress_independent_fit():
    h=header();x,y=coordinates((41,41),h);p=np.array([220.,4.7,7.5,4.6,2.1,.2,-2.])
    g=geometry(h,x.shape,p);r=np.hypot(x-p[1],y-p[2]);major=max(p[3:5])*np.sqrt(8*np.log(2))
    f=SimpleNamespace(x=x,y=y,parameters=p,major=major,beam=5.,weights=weights(g)[0],background=(np.hypot(x,y)<=40)&(r>3*major),radius=r,edges=np.arange(0,42,2))
    guard=dict(accepted=True,accepted_parameters=p.tolist(),status='accepted',cause=None,diagnostic=dict(active_mask=[0]*7))
    m=measurement(dict(fixed=f,auxiliary=g),gaussian(p,x,y),np.ones(x.shape,bool),guard)
    assert m['fit']['status']=='success' and m['frozen_template_projection'] is None
    assert not m['auxiliary_support']['available']


def test_auxiliary_covariance_domain_uses_same_empirical_pairs():
    from point_additive import spectrum_from_pairs
    empirical=np.zeros((65,65));empirical[32,32]=1.;empirical[32,31]=empirical[32,33]=.1
    old,_=spectrum_from_pairs(empirical,32,49)
    model=dict(psd=old,empirical=empirical,summary=dict(request=dict(lag_pixels=32,taper_power=1.,domain_factor=1)))
    g=geometry(header(),(41,41),[220.,5.,8.,12.,2.,.7,-2.])
    new=covariance_spectrum(model,g);expected,_=spectrum_from_pairs(empirical,32,max(g['shape']))
    assert min(new.shape)>2*max(g['shape'])
    assert np.array_equal(new,expected) and np.array_equal(model['psd'],old)
