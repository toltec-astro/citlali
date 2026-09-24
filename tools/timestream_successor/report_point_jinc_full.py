"""Compact numerical tables and headless scientific figures for the fixed matrix."""
import argparse,time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,ListedColormap
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
 cm=plt.get_cmap(cmap).copy();cm.set_bad('#d9d9d9');im=ax.imshow(v,origin='lower',extent=extent(h,v.shape),cmap=cm,norm=TwoSlopeNorm(vmin=lo,vcenter=0.,vmax=hi),interpolation='nearest');ax.set_title(title,fontsize=10);ax.set_xlabel('East offset (arcsec)');ax.set_ylabel('North offset (arcsec)');return im

def finite_max(arrays):
 values=[float(abs(v[np.isfinite(v)]).max()) for v in arrays if np.isfinite(v).any()]
 return max(values+[np.finfo(float).eps]) if values else 1.

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
 lines+=['','Centroids and widths are arcseconds; ellipse tensors, axis ratios and constraint details remain in the measurement JSON. Fitted angle becomes weakly determined as the axis ratio approaches circularity. All apertures retain their original centers/radii and four-square-arcsecond pixel area.','', '## Response to the archived perturbations','', '| Array | Perturbation | Operator | Original full-domain g | eta | rho | Lost original pixels | Paired-common g |','|---|---|---|---:|---:|---:|---:|---:|']
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
  vmax=finite_max([d[n] for d,h in raster.values() for n in ('H','DELTA')])
  rmax=finite_max([d['RESIDUAL'] for d,h in raster.values()])
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
   cor=z['empirical'];L=(len(cor)-1)//2;im=axes[i,1].imshow(cor,origin='lower',extent=[(-L-.5)*h['CDELT1']*3600,(L+.5)*h['CDELT1']*3600,(-L-.5)*h['CDELT2']*3600,(L+.5)*h['CDELT2']*3600],norm=TwoSlopeNorm(vmin=-.4,vcenter=0.,vmax=1),cmap='RdBu_r');axes[i,1].set(title=op+' · empirical directional correlation',xlabel='East lag (arcsec)',ylabel='North lag (arcsec)');fig.colorbar(im,ax=axes[i,1],shrink=.8)
   hist=rec['model']['histogram'];edges=np.array(hist['edges']);density=np.array(hist['counts'])/(np.sum(hist['counts'])*np.diff(edges));axes[i,2].stairs(density,edges,color=COLORS[op]);xx=np.linspace(-5,5,200);axes[i,2].plot(xx,np.exp(-xx**2/2)/np.sqrt(2*np.pi),color='gray',ls='--',label='unit Gaussian');axes[i,2].set(xlabel='Centered residual / fitted scale',ylabel='Density',title=op+f" · excess kurtosis {hist['excess_kurtosis']:.2f}");axes[i,2].legend(fontsize=8)
  fig.suptitle(a+': empirical T10 background · transferred covariance remains qualified',fontsize=13);path=dest/(a+'-noise.png');fig.savefig(path);plt.close(fig);figures.append(path)
 fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
 for i,a in enumerate(ARRAYS):
  for op in ('MAP','JINC'):
   rows=[r for r in s['arrays'][a]['phases'] if r['operator']==op and r['measurement']['fit']['status']=='success'];idx=[r['phase'] for r in rows];peaks=[r['measurement']['fit']['parameters'][0]/r['normalization'] for r in rows];err=[np.linalg.norm(r['centroid_error_arcsec']) for r in rows];axes[i,0].plot(idx,peaks,'o-',label=op,color=COLORS[op]);axes[i,1].plot(idx,err,'o-',label=op,color=COLORS[op])
   widths=[sorted(np.array(r['measurement']['fit']['parameters'][3:5])*np.sqrt(8*np.log(2)),reverse=True) for r in rows]
   if widths:
    axes[i,2].plot(idx,np.array(widths)[:,0],'o-',color=COLORS[op],label=op+' major');axes[i,2].plot(idx,np.array(widths)[:,1],'x--',color=COLORS[op],label=op+' minor')
  axes[i,0].set(title=a+' · fitted peak / input normalization',ylabel='Ratio',xlabel='Phase index (x varies fastest)');axes[i,1].set(title=a+' · centroid error',ylabel='Arcsec',xlabel='Phase index (x varies fastest)')
  axes[i,2].set(title=a+' · core shape',ylabel='FWHM (arcsec)',xlabel='Phase index (x varies fastest)')
  for ax in axes[i]:ax.legend();ax.grid(alpha=.2);ax.set_xticks(range(9))
 fig.suptitle('Continuous profile: nine offsets, identical sampled values for MAP and JINC',fontsize=13);path=dest/'phases.png';fig.savefig(path);plt.close(fig);figures.append(path)
 figures.extend(summary_figures(out,dest,p,s,n))
 for path in figures:lines+=['!['+path.stem+']('+str(path)+')','']
 lines+=['## Detailed reproducible records','',f"[All numerical diagnostics]({out/'SCIENCE.json'}) · [Noise and complete-ensemble results]({out/'NOISE.json'}) · [Map certificates]({out/'GRIDS.json'}) · [Prospective profile and seeds]({out/'freeze/prospective.json'})",'', 'The finite experiment stops here. No source model, FRUIT admission, feedback history, map weights, kernel settings, optical beam or production default has changed.']
 (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n');save_json(dest/'FIGURES.json',dict(figures=[bound(p) for p in figures],report=bound(dest/'RESULTS.md'),wall_seconds=time.perf_counter()-start));manifest(dest)

def summary_figures(out,dest,p,s,n):
 paths=[]
 fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
 for i,a in enumerate(ARRAYS):
  for op in ('MAP','JINC'):
   for branch,style in [('C-core-plus','-'),('C-core-minus','--')]:
    row=next(r for r in s['arrays'][a]['responses'] if r['operator']==op and r['branch']==branch);pr=row['profiles']
    for j,name in enumerate(('H','delta','residual')):
     axes[i,j].plot([r['radius_arcsec'] for r in pr],[np.nan if r[name] is None else r[name] for r in pr],style,color=COLORS[op],label=op+(' +' if branch.endswith('plus') else ' −'))
  for j,name in enumerate(('H: injection reference','Delta: history increment','Delta − H')):
   axes[i,j].set(title=a+' · '+name,xlabel='Radius from frozen M0 center (arcsec)',ylabel='mJy/nominal beam',xlim=(0,60));axes[i,j].axhline(0,color='gray',lw=.5);axes[i,j].legend(fontsize=8)
 fig.suptitle('Compact ± benchmarks: signed profiles on native supported annuli',fontsize=13);path=dest/'compact-response.png';fig.savefig(path);plt.close(fig);paths.append(path)
 fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained');names=['Peak','Template','Aperture1','Aperture1.5','Aperture2']
 for i,a in enumerate(ARRAYS):
  for op in ('MAP','JINC'):
   row=next(r for r in n['results'] if r['array']==a and r['operator']==op)
   if not row.get('response_referenced'):continue
   ratios=row['response_referenced'];observed=[r['observed'] for r in row['uncertainty']['measurements']];peak=p['arrays'][a]['peak']['P_mJy_per_nominal_beam']
   rel=[100*r['output_scatter']/abs(o) if r['output_scatter'] is not None and o else np.nan for r,o in zip(ratios,observed)];resp=[r['slope']*peak/o if r['slope'] is not None and o else np.nan for r,o in zip(ratios,observed)];sc=[r['input_referenced_scatter'] if r['input_referenced_scatter'] is not None else np.nan for r in ratios]
   for j,vals in enumerate((rel,resp,sc)):axes[i,j].plot(range(5),vals,'o-',color=COLORS[op],label=op)
  for j,(title,unit) in enumerate([('raw relative scatter','Percent of observed measurement'),('compact fractional response','Slope × original peak / observed'),('response-referenced scatter','mJy/nominal beam of tested input')]):
   axes[i,j].set(title=a+' · '+title,ylabel=unit);axes[i,j].set_xticks(range(5),names,rotation=35,ha='right');axes[i,j].legend();axes[i,j].grid(alpha=.2)
 fig.suptitle('Conditional additive precision: transferred covariance + Gaussian baseline',fontsize=13);path=dest/'response-referenced-scatter.png';fig.savefig(path);plt.close(fig);paths.append(path)
 fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
 for i,a in enumerate(ARRAYS):
  d=out/'grids'/key(a,'REF',10)/'total';v,h=read_map(out,key(a,'REF',10));c,_=load(d/'certificate.fits');ordinary,_=read_map(out,key(a,'REF',10),operator='MAP');js=c['SCI_SUPPORT'].astype(bool);ms=ordinary['SCI_SUPPORT'].astype(bool)
  state=np.where(ms&js,1,np.where(js,2,np.where(ms,3,0)));im=axes[i,0].imshow(state,origin='lower',extent=extent(h,state.shape),cmap=ListedColormap(['#d9d9d9','#4a6f9c','#cf8c46','#b84040']),vmin=-.5,vmax=3.5);bar=fig.colorbar(im,ax=axes[i,0],ticks=range(4),shrink=.7);bar.ax.set_yticklabels(['Neither','Common','JINC only','MAP only']);axes[i,0].set_title(a+' · native signal support')
  bit=c['CAUSES'].astype('u1');failure=(bit!=0)&(bit!=16);diag=np.where(bit==16,0,np.where(failure,3,np.where(v['NORMALIZATION']<0,2,1)));im=axes[i,1].imshow(diag,origin='lower',extent=extent(h,diag.shape),cmap=ListedColormap(['#d9d9d9','#4a6f9c','#cf8c46','#b84040']),vmin=-.5,vmax=3.5);bar=fig.colorbar(im,ax=axes[i,1],ticks=range(4),shrink=.7);bar.ax.set_yticklabels(['No members','C positive','C negative','Bound failure']);axes[i,1].set_title(a+f' · failures {failure.sum()}')
  fraction=np.divide(c['ERROR_M'],c['TOLERANCE_LOWER'],out=np.full(js.shape,np.nan),where=js);positive=js&(fraction>0);axes[i,2].hist(np.log10(fraction[positive]),bins=35,color='#4a6f9c');axes[i,2].axvline(0,color='#b84040',ls='--',label='acceptance ceiling');axes[i,2].set(title=a+' · componentwise certificate',xlabel='log10(bound / allowed tolerance)',ylabel='Accepted pixels');axes[i,2].legend()
  for ax in axes[i,:2]:ax.set_xlabel('East offset (arcsec)');ax.set_ylabel('North offset (arcsec)')
 fig.suptitle('REF T10 support: arithmetic validity does not imply background precision',fontsize=13);path=dest/'numerical-support.png';fig.savefig(path);plt.close(fig);paths.append(path)
 return paths

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.output)
