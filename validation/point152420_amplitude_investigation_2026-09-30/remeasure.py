"""Saved-product angle/environment audit and matched native-frame profiles.
No observation trajectory, altered input, registration, or production change.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-amplitude-mpl',XDG_CACHE_HOME='/private/tmp/point-amplitude-cache')
from pathlib import Path
import sys,json,hashlib,importlib.util,platform
import numpy as np,scipy,astropy
from scipy.optimize import least_squares
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W/'tools/timestream_successor'))
from inspect_point_source import gaussian,solve_gaussian
from point_reference import context
from point_fruit_diagnostics import patch_map
from point_expected import measure_expected,candidate
spec=importlib.util.spec_from_file_location('comparison',W/'validation/point152420_map_comparison_2026-09-29/compare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=W/'validation/point152420_amplitude_investigation_2026-09-30';OUT=D/'results';R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930');P=Path('/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/pointings/reduced');F=np.sqrt(8*np.log(2))
def read(p):return json.loads(p.read_text())
def old_solve(s,roi,x,y,initial):
 return least_squares(lambda p:gaussian(p,x[roi],y[roi])-s[roi],initial,bounds=([0,-20,-20,1.5,1.5,-np.pi,-np.inf],[np.inf,20,20,20,20,np.pi,np.inf]),x_scale='jac',max_nfev=3000,method='trf',jac='2-point',loss='linear',f_scale=1.,tr_solver='exact',tr_options={},diff_step=None,verbose=0,ftol=1e-12,xtol=1e-12,gtol=1e-12)
def record(f):
 p=f.x;fw=np.sort(p[3:5]*F)[::-1]
 return dict(parameters=p.tolist(),peak=float(p[0]),constant=float(p[6]),centroid=p[1:3].tolist(),fwhm=fw.tolist(),angular_integral=float(2*np.pi*p[0]*p[3]*p[4]),cost=float(f.cost),rms=float(np.sqrt(np.mean(f.fun**2))),active=f.active_mask.tolist(),success=bool(f.success),nfev=f.nfev)
def profiles(d,p):
 s,v,x,y=(d[k] for k in ('signal','support','x','y'));good=v&np.isfinite(s);r=np.hypot(x-p[1],y-p[2]);ann=(r>=45)&(r<60);bg=float(np.median(s[ann&good]));area=abs((x[0,1]-x[0,0])*(y[1,0]-y[0,0]));rows=[]
 for lo,hi in zip((0,4,8,12,20,30,40,50),(4,8,12,20,30,40,50,60)):
  use=(r>=lo)&(r<hi);ok=use&good;disk=r<hi;full=bool(np.all(good[disk]));rows.append(dict(inner=lo,outer=hi,pixels=int(use.sum()),supported=int(ok.sum()),median=float(np.median(s[ok])),mean=float(np.mean(s[ok])),annulus_subtracted_mean=float(np.mean(s[ok])-bg),aperture_full_support=full,aperture_integral_minus_annulus=float(np.sum(s[disk]-bg)*area) if full else None,aperture_integral_minus_fit_constant=float(np.sum(s[disk]-p[6])*area) if full else None))
 return dict(center='same convention for every map: its newly fitted centroid in its own native frame, no rotation/registration',centroid=p[1:3].tolist(),independent_background=dict(region=[45,60],median=bg,pixels=int(ann.sum()),supported=int((ann&good).sum())),rows=rows)
rows=[];plot_data={};c=context(R/'B-REF/freeze/a1100');hist=read(R/'B-REF/run/BASE-POINT/REF/a1100/SUMMARY.json')['histories']
paths=[(f'B-{h["cycle_index"]}',Path(h['total']['path']),False,h) for h in hist]
paths += [('redu09',P/'redu09/152420/raw/toltec_commissioning_a1100_pointing_152420_citlali.fits',True,None),('earlier-V2',m.OLD,True,None)]
# Other reductions are contextual jobs until their own provenance establishes a relationship.
paths += [(p.parents[2].name,p,True,None) for p in sorted(P.glob('redu*/152420/raw/*a1100*.fits')) if p.parents[2].name!='redu09']
paths += [('Q-core-minus-T5',R/'Q-C-core-minus/run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits',False,None)]
for label,path,old,histrow in paths:
 d=m.load(path,old);s,v,x,y=(d[k] for k in ('signal','support','x','y'));roi=np.hypot(x,y)<=40;assert np.all(v[roi]&np.isfinite(s[roi]));ix=np.argmax(np.where(np.hypot(x,y)<=20,s,-np.inf));bg=float(np.median(s[roi]));initial=[max(0.,s.flat[ix]-bg),x.flat[ix],y.flat[ix],5/F,5/F,0.,bg]
 original=old_solve(s,roi,x,y,initial);f=solve_gaussian(s,roi,x,y,initial,sigma_floor=1.5);assert f.success
 row=dict(label=label,input=m.bound(path),frame=d['header'],legacy_solver_current_environment=record(original),repaired=record(f),profiles=profiles(d,f.x))
 if histrow:
  g=histrow['measurement']['guard'];row['executed']=dict(parameters=g['accepted_parameters'],cost=g['diagnostic']['cost'],active=g['diagnostic']['active_mask']);row['legacy_reproduction_parameter_maxabs']=float(np.max(abs(original.x-np.array(g['accepted_parameters']))))
  patch,valid,_,_=patch_map(path);guard=measure_expected(patch,valid,c['fixed'],c['scale'],c['psd']);assert guard['complete']
  old_candidate=candidate(patch,valid,c['fixed'],g);new_candidate=candidate(patch,valid,c['fixed'],guard);common=old_candidate['valid']&new_candidate['valid'];delta=new_candidate['candidate'][common]-old_candidate['candidate'][common]
  row['feedback_diagnostic']=dict(accepted=guard['accepted'],parameters=guard['accepted_parameters'],old_selected=int(old_candidate['selected'].sum()),new_selected=int(new_candidate['selected'].sum()),membership_changed=int(np.sum(old_candidate['selected']!=new_candidate['selected'])),candidate_maxabs_change=float(np.max(abs(delta))),candidate_l2_change=float(np.linalg.norm(delta)),constant_change=float(guard['accepted_parameters'][6]-g['accepted_parameters'][6]),scope='same original saved map and frozen noise context; changes prove dependency, not a repaired executed trajectory')
 rows.append(row)
 if label in ('B-0','B-5','redu09'):plot_data[label]=(d,f.x)
 print(label,round(f.x[0],5),np.round(np.sort(f.x[3:5]*F)[::-1],4),round(f.x[6],4),'cost',round(f.cost,3),'legacy',round(original.x[0],5),'feedback',row.get('feedback_diagnostic',{}).get('membership_changed'),flush=True)
recs={z['label']:z['repaired'] for z in rows};a,b=recs['B-5'],recs['redu09'];ratios=dict(peak=b['peak']/a['peak'],difference=b['peak']-a['peak'],gaussian_area=np.prod(b['fwhm'])/np.prod(a['fwhm']),angular_integral=b['angular_integral']/a['angular_integral'])
result=dict(schema='point152420-saved-remeasurement-v1',environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__,platform=platform.platform()),fitter=m.bound(W/'tools/timestream_successor/inspect_point_source.py'),rows=rows,redu09_over_B_T5=ratios,limitations=['Unweighted same Gaussian+constant inside40arcsec; interior fit is not global-optimality proof.','Original executed trajectories are unchanged. Fixed-context feedback diagnostics do not re-execute adaptive feedback.','Angular integrals are brightness times arcsec squared, not absolute flux.','Other redu directories are separate contextual jobs, not presumed iterations of redu09.'])
(OUT/'saved-measurements.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
for j,(label,(d,p)) in enumerate(plot_data.items()):
 s,v,x,y=(d[k] for k in ('signal','support','x','y'));dx=x[0,1]-x[0,0];dy=y[1,0]-y[0,0];ext=[x[0,0]-dx/2,x[0,-1]+dx/2,y[0,0]-dy/2,y[-1,0]+dy/2]
 im=axs[0,j].imshow(np.where(v,s,np.nan),origin='lower',extent=ext,cmap='RdBu_r',vmin=-60,vmax=650);axs[0,j].set(xlim=(60,-60),ylim=(-60,60),title=label,xlabel='native x (arcsec)',ylabel='native y (arcsec)');fig.colorbar(im,ax=axs[0,j],label='mJy / nominal beam',shrink=.7)
 rr=next(q for q in rows if q['label']==label);z=rr['profiles'];rads=[q['outer'] for q in z['rows']];axs[1,0].plot(rads,[q['annulus_subtracted_mean'] for q in z['rows']],marker='o',label=label);axs[1,1].plot(rads,[q['aperture_integral_minus_annulus'] for q in z['rows']],marker='o',label=label);axs[1,2].plot(rads,[q['aperture_integral_minus_fit_constant'] for q in z['rows']],marker='o',label=label)
for ax,title in zip(axs[1],('Annular mean minus 45–60″ median','Curve of growth minus 45–60″ median','Curve of growth minus fitted constant')):ax.set(title=title,xlabel='outer radius from own fitted centroid (arcsec)');ax.legend();ax.grid(alpha=.2)
axs[1,0].set_ylabel('mJy / nominal beam');axs[1,1].set_ylabel('mJy arcsec² / nominal beam');axs[1,2].set_ylabel('mJy arcsec² / nominal beam')
fig.suptitle('Saved maps remeasured with free periodic angle · native frames not registered\nFixed angular bins and background annulus; missing aperture support is unavailable')
fig.savefig(OUT/'saved-profiles.png',dpi=150);plt.close(fig)
print('ratios',ratios)
