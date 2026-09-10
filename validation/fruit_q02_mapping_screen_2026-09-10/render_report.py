"""Render declared measurements from immutable saved products; no input access."""
import os
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/q02-mapping-report-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/q02-mapping-report-cache')
from pathlib import Path
import json,math
import numpy as np
import matplotlib.pyplot as plt
from core import write_json,require
HERE=Path(__file__).resolve().parent;A=HERE/'attempt_03';R=HERE/'report';R.mkdir(exist_ok=False)
t1=json.loads((A/'T1_SUMMARY.json').read_text());t2=json.loads((A/'T2_SUMMARY.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
# All cases, displayed without invented real-data intervals.
fig,axes=plt.subplots(1,2,figsize=(11,5.7),sharey=True,layout='constrained')
rows=[]
for i,r in enumerate(t2):
 c=r['comparisons']['full'];native=c['native'];common=c['common'];D=c['support']['D'];un=native['U'];nn=native['N4U'];uc=common['U'];nc=common['N4U']
 complete=un['rings']['O']['available'] and nn['rings']['O']['available']
 ratio=math.sqrt(nc['rings']['O']['power']/uc['rings']['O']['power'])
 axes[0].scatter(ratio,i,marker='o' if complete else 'D',s=60,facecolors='#227f83' if complete else 'none',edgecolors='#227f83');axes[0].text(ratio+.016,i,f'{ratio:.3f}',va='center')
 su=un['source']['amplitude'];sn=nn['source']['amplitude'];native_positive=su is not None and sn is not None and su>0 and sn>0
 cu=uc['source']['amplitude'];cn=nc['source']['amplitude'];aratio=sn/su if native_positive else cn/cu if cu is not None and cn is not None and cu>0 and cn>0 else None
 if aratio is not None:
  axes[1].scatter(aratio,i,marker='o' if native_positive else 'D',s=60,facecolors='#a05033' if native_positive else 'none',edgecolors='#a05033');axes[1].text(aratio+.016,i,f'{aratio:.3f}',va='center')
 else:axes[1].text(.47,i,'Both amplitudes negative; ratio omitted',va='center',fontsize=9)
 rows.append(dict(obsnum=r['obsnum'],array=r['array'],full_region_RMS_N_over_U=ratio if complete else None,common_region_RMS_N_over_U=ratio,native_amplitude_U=su,native_amplitude_N4U=sn,common_amplitude_U=cu,common_amplitude_N4U=cn,positive_native_amplitude_fractional_change=sn/su-1 if native_positive else None,required_D=D['required'],missing_D_U=D['required']-D['S_sci']['U'],missing_D_N4U=D['required']-D['S_sci']['N4U']))
labels=[f"{r['obsnum']}  {r['array']}" for r in t2];axes[0].set_yticks(range(6),labels);axes[0].invert_yaxis()
for ax,title in zip(axes,['Background RMS: N4U / U','Fixed-template amplitude: N4U / U']):
 ax.set_title(title);ax.axvline(1,color='#666',ls='--',lw=1);ax.set_xlim(.4,1.12);ax.grid(axis='x',alpha=.2)
axes[1].axvspan(.98,1.02,color='#a05033',alpha=.08)
fig.suptitle('Quieter background does not establish preserved source recovery',fontsize=14)
fig.supxlabel('Filled: complete prescribed region. Open diamond: common support only; full result unavailable.\nPoint estimates only. No real-data confidence intervals; signed moments are not fitted source parameters.',fontsize=9)
fig.savefig(R/'discovery_comparison.png');plt.close(fig)
# Synthetic mechanisms with real decision semantics in accompanying tables.
fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
ratios=[math.sqrt(r['metric_means'][0][0]/r['metric_means'][1][0]) for r in t1]
for i,(r,v) in enumerate(zip(t1,ratios)):
 ax.scatter(v,i,c='#227f83' if r['conditional_disposition']=='pass' else '#a05033',s=55);ax.text(v+.03,i,f"{v:.3f}  {r['conditional_disposition']}",va='center')
ax.set_yticks(range(9),[r['case'] for r in t1]);ax.invert_yaxis();ax.set_xlim(.9,2.65);ax.axvline(1.05,color='#555',ls='--',lw=1);ax.set_xlabel('RMS U / N4U (point estimate; reference line at 1.05)');ax.set_title('Uniform sensitivity depends on the declared noise regime');ax.grid(axis='x',alpha=.2)
fig.supxlabel('1,024 paired trials per case. Decisions also use source, morphology, excursion and support criteria.\nTwo passes and seven failures are conditional on these synthetic laws.',fontsize=9)
fig.savefig(R/'synthetic_comparison.png');plt.close(fig)
# Fixed-domain map displays; scaling affects rendering only, never metrics/support.
for obs in [123424,152389]:
 fig,axes=plt.subplots(3,3,figsize=(12,10.5),layout='constrained')
 for ai,name in enumerate(['a1100','a1400','a2000']):
  pref=f'T2_{obs}_{name}';p=np.load(A/(pref+'_population.npz'));shape=tuple(p['shape']);x=p['x'];y=p['y'];extent=[x.min()-1,x.max()+1,y.min()-1,y.max()+1]
  u=np.load(A/(pref+'_full_U.npz'))['map'];n=np.load(A/(pref+'_full_N4U.npz'))['map'];display=p['D'];both=np.r_[u[display&np.isfinite(u)],n[display&np.isfinite(n)]];lim=float(np.quantile(abs(both),.995));difference=u-n;dl=float(np.quantile(abs(difference[display&np.isfinite(difference)]),.995));Robs=28.59742792952955 if obs==123424 else 27.38359774550894
  for col,(val,label,bound) in enumerate([(u,'Uniform',lim),(n,'N4U',lim),(difference,'Uniform − N4U',dl)]):
   ax=axes[ai,col];cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#c2c2c2');im=ax.imshow(val.reshape(shape),origin='lower',extent=extent,cmap=cmap,vmin=-bound,vmax=bound,interpolation='nearest');ax.set_xlim(-3*Robs,3*Robs);ax.set_ylim(-3*Robs,3*Robs);ax.add_patch(plt.Circle((0,0),Robs,fill=False,color='#444',ls=':',lw=.8));ax.set_title(f'{name} · {label}');ax.set_xlabel('Legacy x [arcsec]');ax.set_ylabel('Legacy y [arcsec]');fig.colorbar(im,ax=ax,shrink=.8,label='delivered mJy/beam')
 fig.suptitle(f'{obs}: identical evaluation occurrences, fixed coefficient generations',fontsize=14)
 fig.supxlabel('Circle: fixed training/source guard. Grey: unavailable native support.\nU/N4U share each row’s colour scale; difference has its own. 99.5% display scaling only. No physical-recovery claim.',fontsize=9)
 fig.savefig(R/f'maps_{obs}.png');plt.close(fig)
# Required coefficient-mass diagnostics derived from saved exact maps, never new weights.
inf=[]
for r in t2:
 prefix=f"T2_{r['obsnum']}_{r['array']}";p=np.load(A/(prefix+'_population.npz'))
 for scope in ['full','window0','window1','window2']:
  for arm in ['U','N4U']:
   z=np.load(A/(prefix+f'_{scope}_{arm}.npz'));q=z['Q'];q2=z['Q2'];cnt=z['count'];good=np.isfinite(q)&np.isfinite(q2)&(q2>0)
   concentration=np.divide(q*q,q2,out=np.full(q.shape,np.nan),where=good)
   fcount=np.divide(z['fallback_count'],cnt,out=np.full(q.shape,np.nan),where=cnt>0)
   fq=np.divide(z['fallback_Q'],q,out=np.full(q.shape,np.nan),where=q>0)
   np.savez_compressed(R/(prefix+f'_{scope}_{arm}_influence.npz'),concentration=concentration,fallback_count_share=fcount,fallback_coefficient_share=fq)
   for region in ['C','O','D']:
    mask=p[region];den=q[mask].sum();de=z['detector_Q'].sum()
    inf.append(dict(obsnum=r['obsnum'],array=r['array'],scope=scope,policy=arm,region=region,count=int(cnt[mask].sum()),sum_gamma=float(den),sum_gamma_squared=float(q2[mask].sum()),max_pixel_detector_share=float(np.nanmax(z['max_detector_share'][mask])),max_pixel_fallback_share=float(np.nanmax(fq[mask])),region_concentration=float(den**2/q2[mask].sum()),meaning='coefficient concentration only, not independent measurements or exposure'))
write_json(R/'DERIVED_MEASUREMENTS.json',dict(discovery=rows,influence=inf))
# A compact, complete owner-readable table set; detailed JSON preserves all bounds/metrics.
lines=['# Complete screen tables','', 'These tables summarize immutable attempt 03. All nine synthetic cases and all six discovery array cases are retained. No omitted support or unavailable metric becomes a pass.','', '## Synthetic results','', '| Case | RMS U / N4U | Conditional disposition | Demonstrated failures | Inconclusive criteria |','| --- | ---: | --- | --- | --- |']
for r,v in zip(t1,ratios):
 failed=', '.join(k for k,c in r['checks'].items() if c['disposition']=='failure') or 'none';inc=', '.join(k for k,c in r['checks'].items() if c['disposition']=='inconclusive') or 'none'
 lines.append(f"| {r['case']} | {v:.6f} | {r['conditional_disposition']} | {failed} | {inc} |")
lines+=['','Decisions use the registered 242 scalar tails at 0.05/256 each, with approximate Student-t continuous bounds and exact binomial bounds. The ratio column is descriptive. Four independent-Gaussian U analytic checks passed. Each full per-case JSON retains every numerical bound and margin.','','## Full and common discovery regions','','| Observation / array | Full O RMS N4U / U | Common O RMS N4U / U | Native template amplitude U → N4U (mJy/beam) | Missing D pixels U / N4U |','| --- | ---: | ---: | --- | ---: |']
f=lambda x:'unavailable' if x is None else f'{x:.6g}'
for r in rows:lines.append(f"| {r['obsnum']} {r['array']} | {f(r['full_region_RMS_N_over_U'])} | {r['common_region_RMS_N_over_U']:.6f} | {f(r['native_amplitude_U'])} → {f(r['native_amplitude_N4U'])} | {r['missing_D_U']} / {r['missing_D_N4U']} |")
lines+=['','O is the fixed R–3R annulus; D is the full radius-3R comparison disk. The two a1400 common-only values cannot satisfy full-region requirements. No value in this table is a confidence interval, physical flux recovery or detector-noise estimate.','','## All full-map alerts','','| Observation / array | Triggered alerts | Unavailable alerts |','| --- | --- | --- |']
for r in t2:
 alerts=r['comparisons']['full']['full_map_alerts'];triggered=', '.join(k for k,v in alerts.items() if v is True and k!='complete_D') or 'none';unavailable=', '.join(k for k,v in alerts.items() if v is None) or 'none'
 lines.append(f"| {r['obsnum']} {r['array']} | {triggered} | {unavailable} |")
lines+=['','## Window support and native source moments','','| Observation / array | Window | Missing D pixels U / N4U | U amplitude; x/y widths | N4U amplitude; x/y widths |','| --- | ---: | ---: | --- | --- |']
for r in t2:
 for j in range(3):
  c=r['comparisons'][f'window{j}'];d=c['support']['D'];u=c['native']['U']['source'];n=c['native']['N4U']['source']
  lines.append(f"| {r['obsnum']} {r['array']} | {j} | {d['required']-d['S_sci']['U']} / {d['required']-d['S_sci']['N4U']} | {f(u['amplitude'])}; {f(u['width_x'])} / {f(u['width_y'])} | {f(n['amplitude'])}; {f(n['width_x'])} / {f(n['width_y'])} |")
lines+=['','All window centroids, ring means/powers/correlations, native/common supports, temporal differences and causes are retained in each case summary and map products. Widths use signed moments with no clipping; unavailable widths must not be replaced by fitted or absolute-valued widths.','','## Fallback and cost','','| Observation / array | Fallback groups | Full-array fallback coefficient fraction | Central fallback coefficient fraction | U mapping seconds | N4U mapping seconds |','| --- | ---: | ---: | ---: | ---: | ---: |']
for r in t2:
 inf=r['maps']['full_N4U']['influence'];lines.append(f"| {r['obsnum']} {r['array']} | {len(r['fallback_groups'])} | {inf['all']['fallback_Q_share']:.8g} | {inf['C']['fallback_Q_share']:.8g} | {r['mapping_seconds']['U']:.6f} | {r['mapping_seconds']['N4U']:.6f} |")
lines+=['','Coefficient estimation is separately timed in each generation lock. Mapping times include required accumulation/influence work and signed checks, excluding serialization; timing order was fixed U then N4U, so these are descriptive cost measurements, not a speed ranking. T1 weight/mapping costs are recorded in its case JSON.','']
(R/'FULL_RESULT_TABLES.md').write_text('\n'.join(lines))
print(json.dumps(dict(figures=4,discovery_rows=len(rows),influence_rows=len(inf) if isinstance(inf,list) else 144,report_directory=str(R))))
