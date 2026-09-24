"""Compact numerical tables and headless scientific figures for the fixed matrix."""
import argparse,time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from point_jinc_full import read,ARRAYS,key
from point_jinc_campaign import read_map,fixed_for
from point_fruit_diagnostics import load
from point_fruit_trajectory import manifest,bound
from point_additive_diagnostics import save_json
from point_empirical import source_patch,coordinates
from inspect_forward_map import require

COLORS={'MAP':'#31558a','JINC':'#b7472a'}
def fmt(v):return 'unavailable' if v is None else f'{v:.5g}'
def extent(h,shape):
 x,y=coordinates(shape,h);dx=h['CDELT1']*3600;dy=h['CDELT2']*3600
 return [x[0,0]-dx/2,x[0,-1]+dx/2,y[0,0]-dy/2,y[-1,0]+dy/2]
def show(ax,v,h,title,lo,hi,cmap='RdBu_r'):
 cm=plt.get_cmap(cmap).copy();cm.set_bad('#d9d9d9');im=ax.imshow(v,origin='lower',extent=extent(h,v.shape),cmap=cm,vmin=lo,vmax=hi,interpolation='nearest');ax.set_title(title,fontsize=10);ax.set_xlabel('East offset (arcsec)');ax.set_ylabel('North offset (arcsec)');return im

def report(out):
 start=time.perf_counter();p=read(out/'PREPARED.json');s=read(out/'SCIENCE.json');n=read(out/'NOISE.json');g=read(out/'GRIDS.json');dest=out/'report';dest.mkdir();figures=[];plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.dpi':140})
 lines=['# POINT 152389: fixed map-only JINC comparison','',
 'Same processed samples and original adaptive ordinary-MAP histories. All new signal use is gated by POINT-152389-JINC-NUMERICAL-ADEQUACY-v1. Peaks remain mJy/nominal beam; nominal beams are 5/7.5/10 arcsec. No calibration, feedback or weighting change is inferred.','',
 '## Execution coverage','',f"30 primary maps,18 logical injection references,54 phase maps; {g['actual_native_accumulations']} actual JINC accumulations including rejoin controls. {n['trial_count']} T10 source-plus-noise trials. M0 has no refit ensemble. Failed/constrained outcomes remain in the tables and complete-ensemble records.",'',
 '## Processed endpoint source measurements','', '| Array | Operator | Peak above constant | East | North | Major FWHM | Minor FWHM | Constant | Fit/constraint |','|---|---|---:|---:|---:|---:|---:|---:|---|']
 for a in ARRAYS:
  for op in ('MAP','JINC'):
   m=read(out/'science'/(key(a,'REF',10)+'-'+op+'-measurement.json'));f=m['fit'];par=f.get('parameters');vals=[None]*6 if par is None else [par[0],par[1],par[2],*sorted(np.array(par[3:5])*np.sqrt(8*np.log(2)),reverse=True),par[6]]
   lines.append('| '+a+' | '+op+' | '+' | '.join(map(fmt,vals))+f" | {f['status']}/{f.get('constrained','—')} |")
 lines+=['','Centroids and widths are arcseconds; ellipse tensors and angle-degeneracy/constraint details remain in the measurement JSON. All apertures retain their original centers/radii and four-square-arcsecond pixel area.','', '## Response to the archived perturbations','', '| Array | Perturbation | Operator | Original full-domain g | eta | rho | Lost original pixels | Paired-common g |','|---|---|---|---:|---:|---:|---:|---:|']
 for a in ARRAYS:
  for r in s['arrays'][a]['responses']:
   q=r['original_benchmark'];pc=r['paired_common'];lines.append(f"| {a} | {r['branch']} | {r['operator']} | {fmt(q.get('g'))} | {fmt(q.get('eta'))} | {fmt(q.get('rho'))} | {r['support']['lost_original']} | {fmt(pc.get('g'))} |")
 lines+=['','Original domains are the authenticated full-map C+D COMMON masks. Native and paired-common metrics are separate records. rho is orthogonal response distortion. H, Delta, Delta−H and numerical bounds are preserved in response FITS. These compare finished adaptive histories, not a fixed cleaner or JINC feedback.','', '## Endpoint additive uncertainty and compact-response reference','', '| Array | Operator | Measurement | Output scatter | Compact slope | Input-referenced scatter | Complete nonlinear ensemble |','|---|---|---|---:|---:|---:|---|']
 for r in n['results']:
  for q in r.get('response_referenced',[]):lines.append(f"| {r['array']} | {r['operator']} | {q['measurement']} | {fmt(q['output_scatter'])} | {fmt(q['slope'])} | {fmt(q['input_referenced_scatter'])} | {r['uncertainty']['primary_available']} |")
  if not r.get('response_referenced'):lines.append(f"| {r['array']} | {r['operator']} | {r['status']} | — | — | — | {r.get('cause','')} |")
 lines+=['','Peak/template scatter has units mJy/nominal beam; aperture scatter has units mJy/nominal beam × arcsec². All input-referenced scatters have the units of the tested compact input amplitude, mJy/nominal beam. Quantiles describe the conditional simulation distribution; scatter is not divided by sqrt(256). Independent fixed-background linear results are saved separately.','',
 'The background covariance is learned separately for each map on the same frozen common physical support and dilated A+B masks. Membership counts are empirical sampling covariates, never inverse variance. Source-depth extrapolation, finite-domain/PSD treatment, regional diagnostics and larger-mask analytic sensitivity remain explicit. a2000 retains its spatial-model qualification; a1100 retains the inherited network-3/scan-5 condition. Gaussian baseline simulations omit non-Gaussian processed-source structure. There is no physical JINC−MAP covariance or significance estimate.','',
 '## Nine-phase fixture','', '| Array | Operator | Peak min/max | Maximum centroid error (arcsec) | Negative-area min/max | Successful / constrained fits |','|---|---|---:|---:|---:|---|']
 for a in ARRAYS:
  for op in ('MAP','JINC'):
   rows=[r for r in s['arrays'][a]['phases'] if r['operator']==op];good=[r for r in rows if r['measurement']['fit']['status']=='success'];peaks=[r['measurement']['fit']['parameters'][0] for r in good];err=[np.linalg.norm(r['centroid_error_arcsec']) for r in good];negative=[r['negative_brightness_area'] for r in rows]
   lines.append(f"| {a} | {op} | {fmt(min(peaks)) if peaks else '—'} / {fmt(max(peaks)) if peaks else '—'} | {fmt(max(err)) if err else '—'} | {fmt(min(negative))} / {fmt(max(negative))} | {len(good)} / {sum(r['measurement']['fit'].get('constrained',False) for r in good)} |")
 lines+=['','The saved compact normalization/shape and its declared taper were sampled directly at native continuous coordinates. This fixture does not replace the original raster-projected injection references. Signed-area integrals retain original brightness-area units; no flux correction.','', '## Numerical support and runtime','', '| Array | Primary JINC accepted pixels | Negative-C accepted | Largest signal bound | Largest bound/tolerance |','|---|---:|---:|---:|---:|']
 for a in ARRAYS:
  rows=[r for r in g['JINC'] if r['array']==a and r['role']=='total'];lines.append(f"| {a} | {min(r['accepted_pixels'] for r in rows)}–{max(r['accepted_pixels'] for r in rows)} | {max(r['negative_C_accepted'] for r in rows)} | {fmt(max(r['max_error_m'] for r in rows))} | {fmt(max(r['max_error_fraction'] for r in rows))} |")
 runtime=dict(preparation_wall_seconds=p['wall_seconds'],cache_enclosure_seconds_by_array={a:p['arrays'][a]['cache']['enclosure_seconds'] for a in ARRAYS},grid_campaign_wall_seconds=g['wall_seconds'],native_sum_cpu_seconds=sum(r['native']['cpu_seconds'] for r in g['JINC']),native_cache_sum_seconds=sum(r['native']['cache_seconds'] for r in g['JINC']),native_loading_sum_seconds=sum(r['native']['loaded_seconds']-r['native']['cache_seconds'] for r in g['JINC']),native_accumulate_certify_sum_seconds=sum(r['native']['accumulated_seconds']-r['native']['loaded_seconds'] for r in g['JINC']),source_response_phase_wall_seconds=s['wall_seconds'],noise_wall_seconds=n['wall_seconds'],trial_cpu_sum_seconds=sum(r.get('trial_cpu_seconds',0) for r in n['results']),sampled_grid_aggregate_rss_bytes=read(out/'GRID_RESOURCES.json')['sampled_aggregate_rss_max'])
 save_json(dest/'RUNTIME.json',runtime)
 lines+=['',f"Grid campaign wall: {runtime['grid_campaign_wall_seconds']:.1f}s; summed native CPU: {runtime['native_sum_cpu_seconds']:.1f}s. Source/response/phase diagnostics: {runtime['source_response_phase_wall_seconds']:.1f}s; covariance/ensemble work: {runtime['noise_wall_seconds']:.1f}s. Sampled aggregate grid RSS peak: {runtime['sampled_grid_aggregate_rss_bytes']/2**30:.2f}GiB. Startup, loading and covariance/refit details are in RUNTIME.json and individual receipts. Native processes use one accumulation worker each; at most12 independent jobs and one library thread per job.",'',
 'The gridding profile certifies the selected exact-real discrete operator on fixed numerical inputs. It does not establish six-digit astrophysical accuracy. Diagnostic maps preserve all unavailable causes; negative normalization is permitted when certified. Q and coefficient-squared time have no uncertainty/exposure interpretation.','', '## Diagnostic figures','']
 for a in ARRAYS:
  peak=p['arrays'][a]['peak']['P_mJy_per_nominal_beam'];fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
  for i,op in enumerate(('MAP','JINC')):
   for j,k in enumerate((0,10)):
    v,h=read_map(out,key(a,'REF',k),operator=op);sp,su,_,sh,_=source_patch(v['PRIMARY'],v['SCI_SUPPORT'].astype(bool),v['HITS'],h);im=show(axes[i,j],sp,sh,f'{op} '+('M0' if k==0 else 'T10'),-.04*peak,peak);fig.colorbar(im,ax=axes[i,j],label='mJy/nominal beam',shrink=.8)
   v,h=read_map(out,key(a,'REF',10),operator=op);im=show(axes[i,2],v['PRIMARY'],h,f'{op} T10 · full support',-.04*peak,.04*peak);fig.colorbar(im,ax=axes[i,2],label='mJy/nominal beam',shrink=.8)
  fig.suptitle(f'{a}: same processed samples · fixed nominal beam · gray = unavailable',fontsize=13);path=dest/(a+'-maps.png');fig.savefig(path);plt.close(fig);figures.append(path)
  fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained');raster={}
  for op in ('MAP','JINC'):
   d,h=load(out/'science'/a/('C-shoulder-'+op)/'response.fits');raster[op]=(d,h)
  vmax=max(np.nanmax(abs(d[n])) for d,h in raster.values() for n in ('H','DELTA'))
  rmax=max(np.nanmax(abs(d['RESIDUAL'])) for d,h in raster.values())
  for i,op in enumerate(('MAP','JINC')):
   d,h=raster[op]
   for j,name in enumerate(('H','DELTA','RESIDUAL')):
    sp,_,_,sh,_=source_patch(d[name],d['NATIVE'].astype(bool),np.ones(d[name].shape),h);v=rmax if name=='RESIDUAL' else vmax;im=show(axes[i,j],sp,sh,f'{op} '+{'H':'H: original injection reference','DELTA':'Delta: completed-history increment','RESIDUAL':'Delta − H'}[name],-v,v);fig.colorbar(im,ax=axes[i,j],label='mJy/nominal beam',shrink=.8)
  fig.suptitle(f'{a}: shoulder benchmark · shared scales across operators',fontsize=13);path=dest/(a+'-shoulder.png');fig.savefig(path);plt.close(fig);figures.append(path)
  fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
  for i,op in enumerate(('MAP','JINC')):
   d=out/'noise'/a/op/'T10';rec=read(d/'BACKGROUND.json')
   if rec['status']!='available':
    for ax in axes[i]:ax.text(.5,.5,rec.get('cause','unavailable'),ha='center',va='center',wrap=True,transform=ax.transAxes);ax.set_axis_off()
    continue
   z=np.load(d/'model.npz');v,h=read_map(out,key(a,'REF',10),operator=op);bg=z['background_mask'];mean=rec['model']['mean_removed'];scale=z['scale'];norm=np.divide(v['PRIMARY']-mean,scale,out=np.full(scale.shape,np.nan),where=scale>0);im=show(axes[i,0],np.where(bg,norm,np.nan),h,op+' · training residual / fitted scale',-3,3);fig.colorbar(im,ax=axes[i,0],shrink=.8)
   cor=z['empirical'];L=(len(cor)-1)//2;im=axes[i,1].imshow(cor,origin='lower',extent=[-2*L-1,2*L+1,-2*L-1,2*L+1],vmin=-.4,vmax=1,cmap='RdBu_r');axes[i,1].set(title=op+' · empirical directional correlation',xlabel='East lag (arcsec)',ylabel='North lag (arcsec)');fig.colorbar(im,ax=axes[i,1],shrink=.8)
   hist=rec['model']['histogram'];edges=np.array(hist['edges']);density=np.array(hist['counts'])/(np.sum(hist['counts'])*np.diff(edges));axes[i,2].stairs(density,edges,color=COLORS[op]);xx=np.linspace(-5,5,200);axes[i,2].plot(xx,np.exp(-xx**2/2)/np.sqrt(2*np.pi),color='gray',ls='--',label='unit Gaussian');axes[i,2].set(xlabel='Centered residual / fitted scale',ylabel='Density',title=op+f" · excess kurtosis {hist['excess_kurtosis']:.2f}");axes[i,2].legend(fontsize=8)
  fig.suptitle(a+': empirical T10 background · transferred covariance remains qualified',fontsize=13);path=dest/(a+'-noise.png');fig.savefig(path);plt.close(fig);figures.append(path)
 fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
 for i,a in enumerate(ARRAYS):
  for op in ('MAP','JINC'):
   rows=[r for r in s['arrays'][a]['phases'] if r['operator']==op and r['measurement']['fit']['status']=='success'];idx=[r['phase'] for r in rows];peaks=[r['measurement']['fit']['parameters'][0]/r['normalization'] for r in rows];err=[np.linalg.norm(r['centroid_error_arcsec']) for r in rows];axes[i,0].plot(idx,peaks,'o-',label=op,color=COLORS[op]);axes[i,1].plot(idx,err,'o-',label=op,color=COLORS[op])
  axes[i,0].set(title=a+' · fitted peak / input normalization',ylabel='Ratio',xlabel='Phase index (x varies fastest)');axes[i,1].set(title=a+' · centroid error',ylabel='Arcsec',xlabel='Phase index (x varies fastest)')
  for ax in axes[i]:ax.legend();ax.grid(alpha=.2);ax.set_xticks(range(9))
 fig.suptitle('Continuous profile: nine offsets, identical sampled values for MAP and JINC',fontsize=13);path=dest/'phases.png';fig.savefig(path);plt.close(fig);figures.append(path)
 for path in figures:lines+=['!['+path.stem+']('+str(path)+')','']
 lines+=['## Detailed reproducible records','',f"[All numerical diagnostics]({out/'SCIENCE.json'}) · [Noise and complete-ensemble results]({out/'NOISE.json'}) · [Map certificates]({out/'GRIDS.json'}) · [Prospective profile and seeds]({out/'freeze/prospective.json'})",'', 'The finite experiment stops here. No source model, FRUIT admission, feedback history, map weights, kernel settings, optical beam or production default has changed.']
 (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n');save_json(dest/'FIGURES.json',dict(figures=[bound(p) for p in figures],report=bound(dest/'RESULTS.md'),wall_seconds=time.perf_counter()-start));manifest(dest)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.output)
