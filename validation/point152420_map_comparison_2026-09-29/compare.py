"""Bounded saved-map descriptive comparison; never mutates maps or production."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/point152420-map-comparison-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/point152420-map-comparison-cache')
import sys,json,hashlib,platform,time
from pathlib import Path
import numpy as np
from astropy.io import fits
import astropy,scipy
from scipy.ndimage import map_coordinates
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from inspect_point_source import solve_gaussian,gaussian
BASE=Path('/Users/gwilson/work_toltec/local_data/citlali-validation')
OLD=BASE/'v2/science/pointings/reduced/redu00/152420/raw/toltec_commissioning_a1100_pointing_152420_citlali.fits'
NEW=BASE/'successor-rtc-point-transfer-runtime-repair-20260929/B-REF/M0/products/a1100/map.fits'
GUARD=BASE/'successor-rtc-point-transfer-runtime-repair-20260929/B-REF/freeze/a1100/guard.json'
FACTOR=np.sqrt(8*np.log(2))
def bound(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def load(p,old):
 with fits.open(p,memmap=False,checksum=True) as f:
  im=f['signal_I'] if old else f[0];h=im.header.copy()
  s=np.asarray(im.data,dtype=float).squeeze().copy()
  v=np.asarray(f['coverage_bool_I' if old else 'SCI_SUPPORT'].data).squeeze().astype(bool)
  assert s.ndim==2 and v.shape==s.shape
  units=[h.get('CUNIT1'),h.get('CUNIT2')]
  if old:
   assert h['CTYPE1']=='AZOFFSET' and h['CTYPE2']=='ELOFFSET'
   # Inspect and record units; legacy producer emits arcsecond coordinates.
   assert all(str(u).lower() in ('arcsec','arcsecond','arcseconds') for u in units),units
   mult=1.
  else:
   assert h['CTYPE1']=='RA---TAN' and h['CTYPE2']=='DEC--TAN' and h['RADESYS']=='FK5' and h['EQUINOX']==2000
   assert all(str(u).lower() in ('deg','degree','degrees') for u in units),units
   mult=3600.
  assert not any(k in h for k in ('CD1_1','PC1_2','PC2_1')),'unexpected WCS mixing'
  yy,xx=np.indices(s.shape);x=(xx+1-h['CRPIX1'])*h['CDELT1']*mult;y=(yy+1-h['CRPIX2'])*h['CDELT2']*mult
  checks=[dict(hdu=q.name,checksum=q.verify_checksum(),datasum=q.verify_datasum()) for q in f if q.data is not None]
  assert all(q['checksum']!=0 and q['datasum']!=0 for q in checks)
  return dict(signal=s,support=v,x=x,y=y,header={k:h.get(k) for k in ('CTYPE1','CTYPE2','CUNIT1','CUNIT2','CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','BUNIT')},checksums=checks)
def fit(d,radius=40.,center=(0.,0.),initial=None):
 s,v,x,y=(d[k] for k in ('signal','support','x','y'))
 roi=np.hypot(x-center[0],y-center[1])<=radius
 assert roi.sum()>=30 and np.all(v[roi]&np.isfinite(s[roi])),'incomplete fit support'
 if initial is None:
  central=np.hypot(x,y)<=20;ix=np.argmax(np.where(central&v,s,-np.inf));bg=float(np.median(s[roi]))
  initial=[max(0.,s.flat[ix]-bg),x.flat[ix],y.flat[ix],5/FACTOR,5/FACTOR,0.,bg]
 f=solve_gaussian(s,roi,x,y,initial,sigma_floor=1.5)
 assert f.success and np.isfinite(f.x).all() and not np.any(f.active_mask),(f.message,f.active_mask)
 p=f.x;widths=np.sort(p[3:5]*FACTOR)[::-1];offset=np.hypot(*p[1:3])
 angle=p[5]+(np.pi/2 if p[4]>p[3] else 0)
 rec=dict(parameters=p.tolist(),fwhm_major_minor_arcsec=widths.tolist(),axis_ratio=float(widths[0]/widths[1]),centroid_radial_offset_arcsec=float(offset),aperture_outer_extent_arcsec=float(offset+3*widths[0]),constant=float(p[6]),amplitude=float(p[0]),fit_radius_arcsec=radius,fit_center_arcsec=list(center),fit_pixels=int(roi.sum()),residual_rms=float(np.sqrt(np.mean(f.fun**2))),residual_rms_over_peak=float(np.sqrt(np.mean(f.fun**2))/p[0]),cost=float(f.cost),nfev=f.nfev,active_mask=f.active_mask.tolist(),major_angle_native_radians=float(angle),converged=bool(f.success))
 return rec

def bin2(d):
 ny,nx=d['signal'].shape;ny-=ny%2;nx-=nx%2
 def block(a):return a[:ny,:nx].reshape(ny//2,2,nx//2,2)
 v=block(d['support']).all(axis=(1,3))&np.isfinite(block(d['signal'])).all(axis=(1,3))
 s=block(d['signal']).mean(axis=(1,3));s[~v]=np.nan
 return dict(signal=s,support=v,x=block(d['x']).mean(axis=(1,3)),y=block(d['y']).mean(axis=(1,3)))
def self_check():
 yy,xx=np.indices((81,81));x=xx-40.;y=yy-40.;truth=[220.,4.7,7.5,4.6,2.1,.47,-2.]
 d=dict(signal=gaussian(truth,x,y),support=np.ones(x.shape,bool),x=x,y=y)
 r=fit(d);assert np.allclose(r['fwhm_major_minor_arcsec'],np.sort(np.array(truth[3:5])*FACTOR)[::-1],rtol=1e-6)
 rot=dict(d,x=-y,y=x);rr=fit(rot);assert np.allclose(r['fwhm_major_minor_arcsec'],rr['fwhm_major_minor_arcsec'],rtol=1e-6)
 b=bin2(d);assert b['x'][0,0]==-39.5 and b['y'][0,0]==-39.5 and b['signal'][0,0]==np.mean(d['signal'][:2,:2])
 d['support'][40,40]=False
 try:fit(d)
 except AssertionError:pass
 else:raise AssertionError('invalid fit support was accepted')
 return dict(rotated_gaussian_recovery=True,orthogonal_coordinate_invariance=True,invalid_support_rejected=True,block_center_and_mean_checked=True)
def plot(data,records,out):
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(3,3,figsize=(13,11),layout='constrained')
 for col,(key,title) in enumerate([('old_native','Earlier V2 · 1″ pixels'),('old_bin2','Earlier V2 · averaged to 2″'),('new_native','Current M0 · 2″ pixels')]):
  d=data[key];r=records[key];p=np.array(r['parameters']);s,x,y=(d[k] for k in ('signal','x','y'));normalized=(s-p[6])/p[0];model=gaussian(p,x,y)
  dx=x[0,1]-x[0,0];dy=y[1,0]-y[0,0];extent=[x[0,0]-dx/2,x[0,-1]+dx/2,y[0,0]-dy/2,y[-1,0]+dy/2]
  for row,a in enumerate((normalized,(s-model)/p[0])):
   ax=axs[row,col];im=ax.imshow(np.where(d['support'],a,np.nan),origin='lower',extent=extent,cmap='RdBu_r',vmin=-.3 if row else -.15,vmax=.3 if row else 1)
   if row==0:ax.contour(x,y,(model-p[6])/p[0],levels=[.5],colors='black',linewidths=1.2)
   ax.plot(*p[1:3],'x',color='black',ms=6);ax.plot(0,0,'+',color='gray');ax.add_patch(plt.Circle((0,0),40,fill=False,color='gray',ls='--',lw=.8))
   ax.set(xlim=(40,-40),ylim=(-40,40),aspect='equal',xlabel='native x offset (arcsec)',ylabel='native y offset (arcsec)')
   fig.colorbar(im,ax=ax,shrink=.75,label='fraction of fitted peak')
  a,b=r['fwhm_major_minor_arcsec'];axs[0,col].set_title(f'{title}\nFWHM {a:.2f}″ × {b:.2f}″')
  axs[1,col].set_title('Data − fitted Gaussian and constant')
  ax=axs[2,col];t=np.linspace(-18,18,181);theta=r['major_angle_native_radians']
  for angle,label,color,w in [(theta,'major axis','#1967a1',a),(theta+np.pi/2,'minor axis','#b44424',b)]:
   px=p[1]+t*np.cos(angle);py=p[2]+t*np.sin(angle)
   coord=np.array([(py-y[0,0])/dy,(px-x[0,0])/dx]);vals=map_coordinates(normalized,coord,order=1,mode='constant',cval=np.nan)
   valid=map_coordinates(d['support'].astype(float),coord,order=1,mode='constant',cval=0)>=1-1e-12;vals[~valid]=np.nan
   ax.plot(t,vals,color=color,label=label);ax.plot(t,np.exp(-.5*(t/(w/FACTOR))**2),color=color,ls='--',alpha=.6)
  ax.axhline(.5,color='gray',lw=.7);ax.set(xlabel='distance from fitted center (arcsec)',ylabel='fraction of fitted peak',ylim=(-.25,1.2),title='Native-map cuts; dashed = Gaussian');ax.legend(fontsize=8)
 fig.suptitle('POINT 152420 · a1100 · same free elliptical Gaussian fit inside 40″\nNative Alt-Az (left/middle) and FK5 tangent (right); orientations differ. Widths are rotation invariant.',fontsize=13)
 fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)

def main():
 began=time.perf_counter();out=HERE/'results';out.mkdir(exist_ok=False)
 inputs=[bound(p) for p in (OLD,NEW,GUARD)];tests=self_check()
 old=load(OLD,True);new=load(NEW,False);data={'old_native':old,'old_bin2':bin2(old),'new_native':new}
 records={k:fit(v) for k,v in data.items()}
 for key in ('old_native','new_native'):
  p=records[key]['parameters'];records[key+'_core15']=fit(data[key],15.,p[1:3],p)
 saved=np.array(json.loads(GUARD.read_text())['accepted_parameters'])
 error=float(np.max(np.abs(np.array(records['new_native']['parameters'])-saved)))
 assert error<1e-7,('current saved fit not reproduced',error)
 plot(data,records,out)
 assert [bound(p) for p in (OLD,NEW,GUARD)]==inputs,'input changed'
 report=dict(schema='point152420-saved-map-comparison-v1',inputs=inputs,script=bound(Path(__file__)),solver=bound(REPO/'tools/timestream_successor/inspect_point_source.py'),self_checks=tests,current_saved_parameter_max_abs_difference=error,inputs_unchanged=True,environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__,backend='Agg'),maps={k:dict(header=v['header'],fits_checksums=v['checksums']) for k,v in [('old',old),('new',new)]},fits=records,seconds=time.perf_counter()-began,limitations=['Descriptive fits only; no noise-calibrated uncertainty or recovered PSF claim.','Different detector populations, reductions, weights, and native map frames; no single upstream cause identified.','No guessed Alt-Az-to-FK5 rotation, reprojection, or pixelwise subtraction. Widths and radial offsets are rotation invariant.','2x2 averaging is a sampling sensitivity check, not a rerun with the new mapmaker.','Core15 fits are diagnostic; never substitute for original campaign guard.'])
 (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:{z:v[z] for z in ('fwhm_major_minor_arcsec','centroid_radial_offset_arcsec','aperture_outer_extent_arcsec','residual_rms_over_peak')} for k,v in records.items()},indent=2))
if __name__=='__main__':main()
