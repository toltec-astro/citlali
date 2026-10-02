import json,sys,subprocess
from pathlib import Path
import numpy as np,pytest
sys.path.insert(0,str(Path(__file__).parent))
import altaz_campaign as a

def request():
    return dict(schema='oof-a2000-altaz-jinc-diagnostic-v1',observation=152385,subobservation=0,scan=1,array='a2000',frame=a.FRAME,mode='cache',spec=dict(rows=35,columns=35,phases=1,pixel_matrix_rad=[-np.pi/180/3600,0.,0.,np.pi/180/3600],scale_rad=(2./1000)/45,a=1.1,b=3.17,c=2.,rmax=1.5))
def invoke(tmp,q,name):
    p=tmp/(name+'.json');a.save_json(p,q);out=tmp/name
    return subprocess.run([str(a.HELPER),str(p),str(out)],capture_output=True,text=True),out
@pytest.mark.parametrize('key,value',[('frame','FK5'),('scan',2),('array','a1100'),('observation',152420)])
def test_altaz_explicit_scope(tmp_path,key,value):
    q=request();q[key]=value;r,out=invoke(tmp_path,q,'bad');assert r.returncode and not out.exists()
def test_altaz_and_fk5_use_identical_native_kernel(tmp_path):
    q=request();r,alt=invoke(tmp_path,q,'alt');assert r.returncode==0,r.stderr
    q.update(schema='oof-a2000-jinc-diagnostic-v1',frame='FK5');r,fk=invoke(tmp_path,q,'fk');assert r.returncode==0,r.stderr
    assert (alt/'cache.f64').read_bytes()==(fk/'cache.f64').read_bytes()
    assert a.yread(alt/'receipt.yaml')['frame']==a.FRAME

def test_altaz_fits_roundtrip_and_certificate(tmp_path):
    from point_jinc_interval import enclose_cache
    q=request();r,out=invoke(tmp_path,q,'cache');assert r.returncode==0
    cache=np.fromfile(out/'cache.f64',dtype='<f8');e=enclose_cache(cache,14,[-1/3600,1/3600],'2.0','3.17',phases=1,rmax='1.5')
    err=tmp_path/'errors';e['errors'].astype('<f8').tofile(err)
    geom=tmp_path/'geometry';np.array([[18.,18.,1.,60.]],dtype='<f8').tofile(geom)
    vals=tmp_path/'values';np.array([7.],dtype='<f8').tofile(vals)
    q.update(mode='accumulate',profile=a.PROFILE,cache_sha256=a.digest(out/'cache.f64'),coefficient_errors=a.bound(err),samples=1,geometry_payload=a.bound(geom),value_payload=a.bound(vals),workers=1,frozen_peak=148.)
    r,out=invoke(tmp_path,q,'apply');assert r.returncode==0,r.stderr
    header=tmp_path/'header.json';a.save_json(header,dict(NAXIS1=35,NAXIS2=35,CTYPE1='AZOFFSET',CTYPE2='ELOFFSET',CUNIT1='arcsec',CUNIT2='arcsec',CRVAL1=0.,CRVAL2=0.,CRPIX1=18.,CRPIX2=18.,CDELT1=-1.,CDELT2=1.,ASTFRAME=a.FRAME))
    a.publish(dict(identity='obs:152385_0_1',request=a.bound(tmp_path/'apply.json'),header=a.bound(header)),out)
    with a.fits.open(out/'map.fits',checksum=True) as f:
        h=f[0].header;assert h['CTYPE1']=='AZOFFSET' and h['CUNIT1']=='arcsec' and h['CRVAL1']==0 and 'RADESYS' not in h
        v=f['LOCAL_VALID'].data.astype(bool);assert np.allclose(f[0].data[v],7)
        assert (19-h['CRPIX1'])*h['CDELT1']==-1
    assert a.read(out/'BUNDLE.json')['frame']==a.FRAME
