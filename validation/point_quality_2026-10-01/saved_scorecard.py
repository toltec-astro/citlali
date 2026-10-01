"""Saved POINT quality diagnostics; native frames stay distinct, no map rescaling."""
import os
os.environ.update(MPLBACKEND='Agg', MPLCONFIGDIR='/private/tmp/point-quality-mpl', XDG_CACHE_HOME='/private/tmp/point-quality-cache')
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,importlib.util,time
import numpy as np
from scipy.ndimage import gaussian_filter,minimum_filter,maximum_filter,label
from astropy.io import fits
D=Path(__file__).resolve().parent; W=D.parents[1];L=Path('/Users/gwilson/work_toltec/local_data')
sys.path.insert(0,str(W/'tools/timestream_successor'))
from inspect_point_source import gaussian
spec=importlib.util.spec_from_file_location('saved_compare',W/'validation/point152420_map_comparison_2026-09-29/compare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def finite_stats(values):
 a=np.asarray(values);a=a[np.isfinite(a)]
 if not len(a):return dict(n=0,available=False)
 med=float(np.median(a));mad=float(1.4826*np.median(abs(a-med)))
 return dict(n=len(a),available=True,median=med,std=float(np.std(a)),robust_scale=mad,
             quantiles=np.quantile(a,[.01,.1,.5,.9,.99]).tolist(),positive_5mad=int(np.sum(a>med+5*mad)),negative_5mad=int(np.sum(a<med-5*mad)))

def metrics(d):
 s,v,x,y=(d[k] for k in ('signal','support','x','y'));v=v&np.isfinite(s);radius=np.hypot(x,y);area=abs((x[0,1]-x[0,0])*(y[1,0]-y[0,0]));pixel=np.sqrt(area)
 try:
  fitted=m.fit(d);p=np.array(fitted['parameters']);fitted.update(status='converged',centroid_native_arcsec=p[1:3].tolist(),gaussian_angular_integral_mJy_arcsec2_per_beam=float(2*np.pi*p[0]*p[3]*p[4]))
 except (AssertionError,ValueError,RuntimeError) as e:fitted=dict(status='unavailable',reason=str(e));p=None
 bg=(radius>=60)&(radius<100);background=finite_stats(s[bg&v]);scales={}
 for fwhm in (0.,6.,12.):
  if fwhm:
   sigma=fwhm/np.sqrt(8*np.log(2))/pixel;guard=int(np.ceil(4*sigma));valid=minimum_filter(v.astype('u1'),size=2*guard+1,mode='constant',cval=0).astype(bool)
   z=gaussian_filter(np.where(v,s,0.),sigma=sigma,truncate=4,mode='constant',cval=0.)
  else:z=s;valid=v
  roi=bg&valid;summary=finite_stats(z[roi]);summary['supported_background_pixels']=int(roi.sum());summary['expected_background_pixels']=int(bg.sum());scales[str(fwhm)]=summary
 corr=[]
 for lag in (2.,6.,12.):
  step=max(1,round(lag/pixel))
  for axis in (0,1):
   aa=[slice(None),slice(None)];bb=aa.copy();aa[axis]=slice(None,-step);bb[axis]=slice(step,None);aa=tuple(aa);bb=tuple(bb);mask=(bg&v)[aa]&(bg&v)[bb]
   a=s[aa][mask];b=s[bb][mask]
   corr.append(dict(axis='native-y' if axis==0 else 'native-x',requested_lag_arcsec=lag,actual_lag_arcsec=step*pixel,pairs=len(a),correlation=float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else None))
 apertures=[]
 if p is not None:
  r=np.hypot(x-p[1],y-p[2])
  for rad in (10.,20.,40.):
   roi=r<=rad;good=roi&v;full=bool(np.all(v[roi]));apertures.append(dict(radius_arcsec=rad,pixels=int(roi.sum()),supported=int(good.sum()),complete=full,integral_minus_fit_constant=float(np.sum(s[roi]-p[6])*area) if full else None))
 residual=s-gaussian(p,x,y) if p is not None else s
 profiles=[]
 for lo,hi in [(0,10),(10,20),(20,40),(40,60),(60,80),(80,100)]:
  roi=(radius>=lo)&(radius<hi);profiles.append(dict(inner=lo,outer=hi,expected_pixels=int(roi.sum()),supported=int((roi&v).sum()),signal=finite_stats(s[roi&v]),gaussian_residual=finite_stats(residual[roi&v])))
 artifacts=dict(available=False)
 if background['available'] and background['robust_scale']>0:
  good=bg&minimum_filter(v.astype('u1'),size=max(3,2*round(5/pixel)+1),mode='constant',cval=0).astype(bool);z=(s-background['median'])/background['robust_scale'];size=max(3,2*round(5/pixel)+1)
  peaks=good&(z>5)&(z==maximum_filter(np.where(v,z,-np.inf),size=size));troughs=good&(z< -5)&(z==minimum_filter(np.where(v,z,np.inf),size=size));artifacts=dict(available=True,positive_local_extrema=int(peaks.sum()),negative_local_extrema=int(troughs.sum()),rule='5 descriptive background MAD; 5 arcsec isolation; not significance or source catalog')
 return dict(fit=fitted,background=background,spatial_scales_FWHM_arcsec=scales,native_axis_correlations=corr,apertures=apertures,radial_profiles=profiles,artifact_diagnostic=artifacts,support=dict(total_pixels=int(v.size),science_pixels=int(v.sum()),science_area_arcsec2=float(v.sum()*area),fit_expected=int((radius<=40).sum()),fit_supported=int(((radius<=40)&v).sum()),background_expected=int(bg.sum()),background_supported=int((bg&v).sum())),uncertainty='Descriptive spatial scatter only; no independent pixels, no paired-test significance; endpoint conditional uncertainty requires its own compatible estimator/operator binding.')

def common_pair(a,b):
 assert a['signal'].shape==b['signal'].shape and np.array_equal(a['x'],b['x']) and np.array_equal(a['y'],b['y']), 'incompatible grids'
 va=a['support']&np.isfinite(a['signal']);vb=b['support']&np.isfinite(b['signal']);common=va&vb
 bg=(np.hypot(a['x'],a['y'])>=60)&(np.hypot(a['x'],a['y'])<100)
 rows=[];pixel=abs(a['x'][0,1]-a['x'][0,0])
 # All scale summaries use the same conservative common domain.
 guard=int(np.ceil(4*12/np.sqrt(8*np.log(2))/pixel));fixed=bg&minimum_filter(common.astype('u1'),size=2*guard+1,mode='constant',cval=0).astype(bool)
 for fwhm in (0.,6.,12.):
  z=[]
  for d,v in ((a,va),(b,vb)):
   z.append(gaussian_filter(np.where(v,d['signal'],0.),sigma=fwhm/np.sqrt(8*np.log(2))/pixel,truncate=4,mode='constant',cval=0.) if fwhm else d['signal'])
  rows.append(dict(fwhm_arcsec=fwhm,M0=finite_stats(z[0][fixed]),T5=finite_stats(z[1][fixed]),paired_delta=finite_stats((z[1]-z[0])[fixed])))
 return dict(common_pixels=int(common.sum()),lost_from_M0=int((va&~vb).sum()),gained_after_M0=int((vb&~va).sum()),M0_original_pixels=int(va.sum()),T5_original_pixels=int(vb.sum()),background_common_pixels=int((bg&common).sum()),same_domain_all_scales_pixels=int(fixed.sum()),scales=rows,qualification='Same-grid conditional M0/T5 diagnostic, not independent noise realizations or R/S sensitivity improvement. Fixed12arcsec kernel-safe domain for every scale; small domain limits interpretation.')

def inputs():
 r=L/'citlali-validation';out=[]
 for obs in (152420,152430):
  path=r/f'v2/science/pointings/reduced/redu00/{obs}/raw/toltec_commissioning_a1100_pointing_{obs}_citlali.fits';out.append((obs,'R-saved-V2',path,True))
  root=(r/'successor-rtc-point-transfer-completion-20260930/B-REF') if obs==152420 else (W/'validation/rtc_two_stage_2026-09-28/compact/B-REF-rep1')
  for stage,rel in [('M0','M0/products/a1100/map.fits'),('T5','run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits')]:
   p=root/rel
   if not p.exists():p=Path(str(p)+'.gz')
   out.append((obs,'B-executed-'+stage,p,False))
 out.append((152420,'historical-redu09',L/'2025-C1-COM-01/pointings/reduced/redu09/152420/raw/toltec_commissioning_a1100_pointing_152420_citlali.fits',True))
 return out

def main():
 started=time.monotonic();result=[];plots={}
 for obs,role,path,old in inputs():
  d=m.load(path,old);rec=dict(observation=obs,role=role,input=m.bound(path),native_frame=d['header'],**metrics(d));result.append(rec);plots[(obs,role)]=(d,rec)
  print(obs,role,rec['fit'].get('amplitude'),rec['fit'].get('fwhm_major_minor_arcsec'),flush=True)
 import matplotlib.pyplot as plt
 for obs in (152420,152430):
  fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
  for col,role in enumerate(('R-saved-V2','B-executed-M0','B-executed-T5')):
   d,rec=plots[(obs,role)];x,y,s,v=(d[k] for k in ('x','y','signal','support'));dx=x[0,1]-x[0,0];dy=y[1,0]-y[0,0];extent=[x[0,0]-dx/2,x[0,-1]+dx/2,y[0,0]-dy/2,y[-1,0]+dy/2]
   for row,limit in enumerate((60,100)):
    im=axs[row,col].imshow(np.where(v,s,np.nan),origin='lower',extent=extent,cmap='RdBu_r',vmin=-80,vmax=650 if row==0 else 80);axs[row,col].set(xlim=(limit,-limit),ylim=(-limit,limit),title=role+' · '+d['header']['CTYPE1'],xlabel='native x arcsec',ylabel='native y arcsec');axs[row,col].plot(0,0,'k+',ms=5)
    for rad in ([40] if row==0 else [60,100]):axs[row,col].add_patch(plt.Circle((0,0),rad,fill=False,color='black',ls=':',lw=.6))
    fig.colorbar(im,ax=axs[row,col],label='mJy / nominal beam',shrink=.7)
  fig.suptitle(f'POINT {obs}: original supported maps, common display scales\nNative frames NOT registered; R includes network0; B networks1–4; no fitted rescaling')
  fig.savefig(D/f'results/point-{obs}-saved.png',dpi=130);plt.close(fig)
 report=dict(schema='point-quality-saved-v1',records=result,within_successor_common_support={str(obs):common_pair(plots[(obs,'B-executed-M0')][0],plots[(obs,'B-executed-T5')][0]) for obs in (152420,152430)},recipe=dict(fit='free7parameter Gaussian+constant inside40arcsec around commanded origin; 1.5arcsec sigma floor',background='fixed60–100arcsec commanded-origin annulus, no final-map selection',scales='native and6/12arcsec Gaussian FWHM; require complete4sigma measured kernel support; zero padding excluded from reported domain',apertures='10/20/40arcsec around freely fitted centroid, complete support, fitted constant removed; brightness angular integral not flux density',cross_frame='no pixel differences or common-pixel claim across AltAz and FK5; same angular region does not prove identical measured sky support'),evaluation_wall_seconds=time.monotonic()-started,new_observation_starts=0)
 (D/'results/saved-scorecard.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
