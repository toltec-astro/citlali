"""Deterministic checks of diagnostic high-coverage metadata vs usability."""
from types import SimpleNamespace
import numpy as np
import pytest
from point_additive import learn,NoiseRequest
from point_admission import admit
from point_corr_admission import Neighborhood,select
from point_fruit_trajectory import candidate_guard
from point_empirical import coordinates,patch
from point_empirical_io import write_fits
from point_jinc_campaign import exact_header
from point_independent import initialize,context


def fixture():
    h=dict(CRPIX1=61.,CRPIX2=61.,CDELT1=-2/3600,CDELT2=2/3600,CRVAL1=150.,CRVAL2=40.,RADESYS='FK5',EQUINOX=2000.,CTYPE1='RA---TAN',CTYPE2='DEC--TAN')
    x,y=coordinates((121,121),h);p=np.array([1000.,0,0,4,4.2,.2,3.])
    signal=3+1000*np.exp(-(x*x+y*y)/32)+np.sin(.7*x)*np.cos(.37*y)
    cov=114+(np.arange(x.size).reshape(x.shape)%343).astype(float);valid=x*x+y*y<110**2
    for rc,value in [((60,61),458.),((60,62),475.),((61,62),1e9),((60,66),np.nextafter(456.,np.inf))]:cov[rc]=value
    return h,p,signal,cov,valid


def models():
    h,p,s,c,v=fixture()
    old=learn(s,c,v,h,p,NoiseRequest(mask_fwhm=3.),stamp_size=49)
    new=learn(s,c,v,h,p,NoiseRequest(mask_fwhm=3.,high_coverage_usable=True),stamp_size=49)
    return h,p,s,c,v,old,new


def test_high_tail_values_unchanged_and_no_new_upper_cutoff():
    h,p,s,c,v,old,new=models();high=new['coverage_extrapolated']
    assert new['summary']['coverage_transfer']['training_coverage_range']==[114.,456.]
    assert all(new['valid'][rc] for rc in [(60,61),(60,62),(61,62),(60,66)])
    assert new['summary']['coverage_transfer']['maximum_applied_coverage']==1e9
    assert np.array_equal(old['scale'],new['scale'],equal_nan=True)
    assert np.array_equal(old['psd'],new['psd'])
    assert np.array_equal(old['empirical'],new['empirical'],equal_nan=True)
    assert np.array_equal(old['valid'][~high],new['valid'][~high])
    assert np.isfinite(new['scale'][high]).all() and (new['scale'][high]>0).all()
    assert np.ptp(new['scale'][high])==0


def base_patch(h,p,s,v,m):
    ph={**h,'CRPIX1':21.,'CRPIX2':21.}
    return admit(patch(s,60,60,20),patch(v,60,60,20),patch(m['scale'],60,60,20),patch(m['valid'],60,60,20),ph,p,3.,1.),ph


def candidate_planes(b):return dict(CANDIDATE=b['candidate'],ADMITTED=b['selected'],VALID=b['valid'])


def test_contact_and_supplemental_footprints_use_real_usability():
    h,p,s,c,v,old,new=models();before,ph=base_patch(h,p,s,v,old);after,_=base_patch(h,p,s,v,new)
    with pytest.raises(ValueError,match='unsupported contact'):candidate_guard(candidate_planes(before),ph)
    candidate_guard(candidate_planes(after),ph)
    assert np.array_equal(before['selected'][before['valid']],after['selected'][before['valid']])
    n0=Neighborhood(ph,(41,41),10.,patch(old['scale'],60,60,20),old['psd'],before['selected'])
    n1=Neighborhood(ph,(41,41),10.,patch(new['scale'],60,60,20),new['psd'],after['selected'])
    f0,f1=n0.scores(before),n1.scores(after)
    for pol in ('CORR','DIAG'):
        avail=f0['policies'][pol]['available'];assert (f1['policies'][pol]['available']&~avail).any()
        assert np.array_equal(f0['policies'][pol]['score'][avail],f1['policies'][pol]['score'][avail])
        result=select(after,f1,after['selected'],pol,5.)
        assert np.all(result['valid'][after['valid']]);assert np.array_equal(result['valid'],after['valid'])


def test_independent_defects_and_lower_coverage_remain_invalid():
    h,p,s,c,v=fixture();v[60,61]=False;s[60,62]=np.nan;c[61,62]=np.inf;c[62,62]=50.
    m=learn(s,c,v,h,p,NoiseRequest(high_coverage_usable=True),stamp_size=49)
    assert all(not m['valid'][rc] for rc in [(60,61),(60,62),(61,62),(62,62)])
    h,p,s,c,v,old,new=models();b,ph=base_patch(h,p,s,v,new)
    b['valid'][20,21]=False;b['candidate'][20,21]=np.nan;b['selected'][20,21]=False
    with pytest.raises(ValueError,match='unsupported contact'):candidate_guard(candidate_planes(b),ph)
    for bad in (np.nan,0.,-1.):
        scale=patch(new['scale'],60,60,20);scale[20,26]=bad
        n=Neighborhood(ph,(41,41),10.,scale,new['psd'],b['selected']);assert not n.scores(b)['policies']['CORR']['available'][20,29]
    with pytest.raises(ValueError,match='invalid covariance'):Neighborhood(ph,(41,41),10.,scale,-new['psd'],b['selected'])
    x,y=coordinates((41,41),ph);disk=np.hypot(x,y)<=40;selected=np.zeros(disk.shape,bool);selected[20,40]=True
    with pytest.raises(ValueError,match='boundary contact'):candidate_guard(dict(CANDIDATE=np.where(disk,selected.astype(float),np.nan),VALID=disk,ADMITTED=selected),ph)


def test_saved_reloaded_new_operands_keep_extrapolation_separate(tmp_path):
    h,p,s,c,v=fixture();path=tmp_path/'map.fits'
    write_fits(path,exact_header(h),[('PRIMARY',s,'mJy/nominal beam'),('NORMALIZATION',c,'1'),('SCI_SUPPORT',v.astype('u1'),'1')],companion='SYNTHETIC-UNIT')
    first,g=initialize(path,tmp_path/'freeze');assert g['accepted']
    second=context(tmp_path/'freeze')
    for key in ('noise_valid','scale','base_scale','noise_high_extrapolated','covariance_high_extrapolated','k0'):
        assert np.array_equal(first[key],second[key],equal_nan=True)
    high=second['noise_high_extrapolated'];assert high.any() and second['noise_valid'][high].all()
    from point_expected import candidate
    from point_guarded_feedback import publish
    b=candidate(second['original'],second['support'],second['fixed'],g);fields=second['neighborhood'].scores(b)
    from point_fruit_diagnostics import load
    for pol in ('BASE-POINT','CORR','DIAG'):
        cp,_=publish(tmp_path/pol,second,second['original'],(g,b,fields),pol,3. if pol=='BASE-POINT' else 5.)
        saved,_=load(cp);diag,_=load(cp.parent/'admission.fits')
        assert saved['VALID'][high].all() and np.array_equal(diag['NOISE_EXTRAP'],high)
        candidate_guard(saved,second['header'])
