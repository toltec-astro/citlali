import copy,json,subprocess,sys
from pathlib import Path
import numpy as np,pytest,yaml
sys.path.insert(0,str(Path(__file__).parent))
import campaign as c
from point_jinc_interval import enclose_cache

def request():
    return dict(schema='oof-a2000-jinc-diagnostic-v1',observation=152385,subobservation=0,scan=1,array='a2000',frame='FK5',mode='cache',spec=dict(rows=35,columns=35,phases=1,pixel_matrix_rad=[-np.pi/180/3600,0.,0.,np.pi/180/3600],scale_rad=(2./1000)/45,a=1.1,b=3.17,c=2.,rmax=1.5))
def native(tmp,q,name):
    p=tmp/(name+'.json');c.save_json(p,q);out=tmp/name
    result=subprocess.run([str(c.HELPER),str(p),str(out)],capture_output=True,text=True)
    return result,out
@pytest.mark.parametrize('field,value',[('scan',2),('observation',152420),('array','a1100'),('frame','AltAz')])
def test_native_rejects_foreign_scope(tmp_path,field,value):
    q=request();q[field]=value;p,out=native(tmp_path,q,'bad');assert p.returncode and 'acquisition/frame differs' in p.stderr;assert not out.exists()
def test_native_rejects_unselected_kernel(tmp_path):
    q=request();q['spec']['phases']=16;p,out=native(tmp_path,q,'bad');assert p.returncode and 'selected parameters differ' in p.stderr

def test_selected_cache_certificate_and_constant_field(tmp_path):
    q=request();p,out=native(tmp_path,q,'cache');assert p.returncode==0,p.stderr
    cache=np.fromfile(out/'cache.f64',dtype='<f8');r=c.yread(out/'receipt.yaml');assert r['half_width']==14
    e=enclose_cache(cache,14,[-1/3600,1/3600],'2.0','3.17',phases=1,rmax='1.5')
    assert e['lower'].shape==(1,1,29,29) and np.all(e['lower']<=e['upper'])
    assert np.all(np.maximum(abs(cache.reshape(e['errors'].shape)-e['lower']),abs(cache.reshape(e['errors'].shape)-e['upper']))<=e['errors']+1e-15)
    errors=tmp_path/'errors.f64';e['errors'].astype('<f8').tofile(errors)
    geom=tmp_path/'geometry.f64';np.array([[18.,18.,1.,60.],[18.49,18.49,1.,60.],[.5,.5,1.,60.]],dtype='<f8').tofile(geom)
    values=tmp_path/'values.f64';np.array([7.,7.,7.],dtype='<f8').tofile(values)
    q.update(mode='accumulate',profile=c.PROFILE,cache_sha256=c.digest(out/'cache.f64'),coefficient_errors=c.bound(errors),samples=3,geometry_payload=c.bound(geom),value_payload=c.bound(values),workers=1,frozen_peak=148.)
    p,out=native(tmp_path,q,'apply');assert p.returncode==0,p.stderr
    valid=np.fromfile(out/'causes.u8',dtype='u1')==0;m=np.fromfile(out/'m.f64',dtype='<f8');err=np.fromfile(out/'error-4.f64',dtype='<f8');assert valid.any()
    assert np.all(abs(m[valid]-7)<=err[valid]+1e-14)
    assert np.any(np.fromfile(out/'C.f64',dtype='<f8')[valid]<0)
    # Exercise the actual writer and read back the five roles plus local validity.
    header=tmp_path/'header.json';c.save_json(header,dict(NAXIS1=35,NAXIS2=35,RADESYS='FK5',CRPIX1=18.,CRPIX2=18.,CDELT1=-1/3600,CDELT2=1/3600))
    state=dict(identity='obs:152385_0_1',request=c.bound(tmp_path/'apply.json'),header=c.bound(header))
    c.publish(state,out)
    with c.fits.open(out/'map.fits',checksum=True) as f:
        assert len(f)==6
        assert f[0].header['OBSID']=='obs:152385_0_1'
        assert np.array_equal(f['LOCAL_VALID'].data.astype(bool).ravel(),valid)
        assert np.isnan(f[0].data.ravel()[~valid]).all()
        assert np.array_equal(f[0].data.ravel()[valid],m[valid])
    assert c.read(out/'BUNDLE.json')['base_roles']==['N','C','Q','N/C with LOCAL_VALID','coefficient-squared time']

    bad=copy.deepcopy(q);bad['value_payload']['sha256']='0'*64;p,_=native(tmp_path,bad,'bad-hash');assert p.returncode and 'input hash mismatch' in p.stderr
    bad=copy.deepcopy(q);bad['profile']='POINT-152389-JINC-NUMERICAL-ADEQUACY-v1';p,_=native(tmp_path,bad,'bad-profile');assert p.returncode and 'unknown profile' in p.stderr

def test_occurrence_order_and_missingness():
    v=np.array([[1.,0.],[3.,4.]]);ca=np.array([[0,1],[0,0]],dtype='u1');xy=np.array([[1.,2.],[3.,4.],[5.,6.]])
    assert np.array_equal(c.selected_values(v,ca,xy,np.array([7],dtype='u1')),[1,3,4])
    with pytest.raises(ValueError):c.selected_values(v,ca,xy,np.array([3],dtype='u1'))
    with pytest.raises(ValueError):c.selected_values(v,ca,xy[:-1],np.array([7],dtype='u1'))

def test_replay_detects_wrong_coordinate_join():
    h=dict(CDELT1=-1.,CDELT2=1.,CRPIX1=2.,CRPIX2=2.,NAXIS1=3,NAXIS2=3)
    xy=np.array([[0.,0.],[1.,0.]]);z=np.array([5.,7.]);m=np.full((3,3),np.nan);m[1,1]=5;m[1,0]=7;n=np.zeros_like(m);n[1,0:2]=1
    assert c.replay(z,xy,h,m,n,np.array([4,3]))['max_signal_difference']==0
    with pytest.raises(ValueError):c.replay(z,xy[::-1],h,m,n,np.array([4,3]))

def test_freeze_hashes_are_enforced(tmp_path):
    file=tmp_path/'saved';file.write_text('parent');record=dict(preserved=[c.bound(file)])
    c.save_json(tmp_path/'COMPLETED-SCRATCH-REMOVALS.json',record);a=c.SavedInputs(tmp_path);a.get(file)
    file.write_text('changed')
    with pytest.raises(ValueError):a.get(file)

def test_positive_normalization_outside_signal_support_is_not_zero_sky():
    h=dict(CDELT1=-1.,CDELT2=1.,CRPIX1=2.,CRPIX2=2.,NAXIS1=3,NAXIS2=3)
    xy=np.array([[0.,0.],[1.,0.]]);z=np.array([5.,7.]);m=np.full((3,3),np.nan);m[1,1]=5
    n=np.zeros((3,3));n[1,0:2]=1;num=np.zeros_like(n);num[1,0:2]=[7,5]
    r=c.replay(z,xy,h,m,n,np.array([4,3]),num,np.isfinite(m))
    assert r['published_signal_pixels']==1 and r['positive_normalization_pixels']==2
    num[1,0]=8
    with pytest.raises(ValueError):c.replay(z,xy,h,m,n,np.array([4,3]),num,np.isfinite(m))

def test_gate_inventory_and_completion_required(tmp_path):
    log=tmp_path/'log';log.write_text('passed')
    g=dict(source_revision='revision',complete=True,gates=[dict(name=n,exit_code=0,log=c.bound(log)) for n in c.MANDATORY_GATES])
    c.validate_gates(g,'revision')
    for field,value in [('gates',[]),('gates',g['gates'][:-1]),('complete',False),('source_revision','other')]:
        bad=copy.deepcopy(g);bad[field]=value
        with pytest.raises(ValueError):c.validate_gates(bad,'revision')
    g['gates'][2]['exit_code']=1
    with pytest.raises(ValueError):c.validate_gates(g,'revision')

def test_execution_identity_rejects_rebuild_or_source_move(tmp_path,monkeypatch):
    binary=tmp_path/'helper';binary.write_text('binary');p=dict(source_revision='revision',helper=c.bound(binary))
    monkeypatch.setattr(c,'HELPER',binary);monkeypatch.setattr(c,'revision',lambda:'revision');c.execution_identity(p)
    binary.write_text('rebuild')
    with pytest.raises(ValueError):c.execution_identity(p)
    monkeypatch.setattr(c,'revision',lambda:'other')
    with pytest.raises(ValueError):c.execution_identity(p)
