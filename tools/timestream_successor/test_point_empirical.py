import dataclasses
import numpy as np
import pytest
from astropy.io import fits
from point_empirical import Request,stats,background,source_experiment,fixed_amplitude,aperture_measurements,coordinates,gaussian
from point_empirical_io import threshold,write_fits
from validate_point_empirical import fixture


def test_signed_fixed_projection_preserves_constant_and_requires_complete_footprint():
    t=np.array([[1.,.5],[.25,0.]]);roi=t>0;valid=np.ones(t.shape,bool)
    data=-3*t+7
    r=fixed_amplitude(data,valid,t,roi)
    assert r['amplitude']==pytest.approx(-3+7*t.sum()/(t*t).sum())
    valid[0,1]=False
    assert fixed_amplitude(data,valid,t,roi) is None


def test_raw_aperture_has_no_implicit_background_subtraction():
    x,y=np.meshgrid(np.arange(-10,11)*2,np.arange(-10,11)*2);data=np.ones(x.shape)*9
    result=aperture_measurements(data,np.ones(x.shape,bool),x,y,[0,0],4,5,4)
    for r in result:
        assert r['value_mJy_per_beam_arcsec2']==pytest.approx(9*r['aperture_pixels']*4)
    support=np.ones(x.shape,bool);support[10,10]=False
    assert not aperture_measurements(data,support,x,y,[0,0],4,5,4)[0]['available']


def test_background_statistics_and_domain_no_extrapolation():
    h,x,y,p=fixture(81);rng=np.random.default_rng(333);d=rng.normal(11,3,x.shape)
    q=np.where(x<0,10.,20.);s=np.ones(x.shape,bool);q[40,40]=10000
    bg=background(d,q,s,h,[0,0],6,Request(coverage_bins=2))
    assert not bg['valid'][40,40]
    for r in bg['records']:
        assert r['median']==pytest.approx(11,abs=.25)
        assert r['std']==pytest.approx(3,abs=.20)
        assert r['robust_scale']==pytest.approx(3,abs=.20)
    assert bg['background_mask'][40,40]==0
    assert len(bg['sensitivity'])==3 and len(bg['spatial'])==4


def test_blank_selection_ignores_signal_and_never_uses_source_or_missing_pixels():
    h,x,y,p=fixture();s=np.hypot(x,y)<120;q=np.where(s,1000.,0.);d=gaussian(p,x,y)
    request=Request(maximum_blanks=8);major=max(p[3:5])*np.sqrt(8*np.log(2));bg=background(d,q,s,h,p[1:3],major,request)
    first=source_experiment(d,q,s,h,6,p,bg['background_mask'],request)
    rng=np.random.default_rng(44);changed=d+rng.normal(0,2,d.shape)
    second=source_experiment(changed,q,s,h,6,p,bg['background_mask'],request)
    assert [(t['row'],t['column']) for t in first['trials']]==[(t['row'],t['column']) for t in second['trials']]
    assert all(t['fit_support_fraction']==pytest.approx(1) for t in first['trials'])
    from point_empirical import patch,source_patch
    _,_,_,sh,half=source_patch(d,s,q,h);xx,yy=coordinates((2*half+1,2*half+1),sh)
    roi=np.hypot(xx-p[1],yy-p[2])<=3*major
    for t in first['trials']:
        assert t['fixed'] is not None
        assert np.all(patch(bg['background_mask'],t['row'],t['column'],half)[roi])
    assert first['refit']['failed']==0
    assert max(abs(t['amplitude_error']) for t in first['refit_trials'])<1e-6


def test_no_blank_support_is_explicitly_unavailable():
    h,x,y,p=fixture(51);s=np.hypot(x,y)<43;q=np.where(s,1000.,0.);d=gaussian(p,x,y)
    bg=background(d,q,s,h,p[1:3],7,Request())
    r=source_experiment(d,q,s,h,6,p,bg['background_mask'],Request())
    assert not r['refit']['scatter_available']
    assert not r['refit_trials']
    assert r['refit']['error']['std'] is None


def test_support_threshold_and_companion_fits(tmp_path):
    q=np.array([[0.,1.],[2.,8.]])
    s,r=threshold(q,.5);assert r==8 and s.sum()==1
    h,_,_,_=fixture();path=tmp_path/'companion.fits'
    write_fits(path,h,[('SCALE',q,'mJy/beam'),('VALID',s.astype('u1'),'1')])
    with fits.open(path,checksum=True) as f:
        assert all(x.verify_checksum()==1 and x.verify_datasum()==1 for x in f)
        assert f[0].header['EXTNAME']=='SCALE'
    with pytest.raises(OSError):write_fits(path,h,[('SCALE',q,'mJy/beam')])


def test_request_rejects_partial_blank_footprints():
    with pytest.raises(ValueError):Request(minimum_stamp_fraction=.98)
    with pytest.raises(ValueError):Request(maximum_blanks=3)
    assert stats([])['std'] is None


def test_refit_bounds_are_included_and_failures_are_not_silently_dropped(monkeypatch):
    import point_empirical as module
    h,x,y,p=fixture();s=np.hypot(x,y)<120;q=np.where(s,1000.,0.);d=gaussian(p,x,y)
    request=Request(maximum_blanks=6);bg=background(d,q,s,h,p[1:3],7,request)
    monkeypatch.setattr(module,'measure',lambda *args:({'fit_constraint_active':True},p.copy(),x,y))
    r=source_experiment(d,q,s,h,6,p,bg['background_mask'],request)
    assert r['refit']['constrained']==len(r['refit_trials'])
    assert r['refit']['error']['n']==len(r['refit_trials'])
    assert r['refit']['scatter_available']
    def fail(*args):raise ValueError('test nonconvergence')
    monkeypatch.setattr(module,'measure',fail)
    r=source_experiment(d,q,s,h,6,p,bg['background_mask'],request)
    assert r['refit']['failed']==len(r['refit_trials'])
    assert not r['refit']['scatter_available']
    assert all(t['cause']=='test nonconvergence' for t in r['refit_trials'])


def test_split_accounting_reuses_each_retained_occurrence_once(tmp_path):
    import yaml
    from point_empirical_io import split_maps
    shape=(51,51);h,x,y,p=fixture(51);root=tmp_path
    (root/'bundle.yaml').write_text('coverage_cut: 0.5\n')
    pix=np.array([25*51+24,25*51+25,25*51+26],dtype='<i4')
    np.concatenate([pix,pix]).tofile(root/'operator-pixel.i32')
    domain=[]
    for i,values in enumerate([[2.,4.,6.],[6.,8.,10.]]):
        stem=f'segment-{i}'
        np.array(values,dtype='<f8').tofile(root/(stem+'-cleaned.f64'))
        np.zeros(3,dtype='u1').tofile(root/(stem+'-causes.u8'))
        pub=dict(identity=f'test{i}',source_PTC_root=str(root),source_segment=dict(network=0,scan=i,scheduled_times=1,output_detectors=3,files={stem+'-cleaned.f64':'test'}))
        (root/f'pub{i}.yaml').write_text(yaml.safe_dump(pub));domain.append(dict(publication_file=f'pub{i}.yaml',first_input_row=i*3,input_rows=3))
    counts=np.zeros(shape,dtype='i8');counts.flat[pix]=2
    numerator=np.zeros(shape);numerator.flat[pix]=[8,12,16]
    full=dict(PRIMARY=np.where(counts>0,numerator/2,np.nan),HITS=counts,NUMERATOR=numerator,SCI_SUPPORT=counts>0)
    result=split_maps(root,{'ordered_column_domain':domain},full,h,6,p)
    assert np.array_equal(result['counts'].sum(axis=0),counts)
    assert np.array_equal(result['numerators'].sum(axis=0),numerator)
    np.testing.assert_array_equal(result['difference'].ravel()[pix],[-4.,-4.,-4.])
    assert result['report']['coverage_A_fraction_quantiles']==[.5]*5
