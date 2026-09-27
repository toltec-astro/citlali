"""Saved-only fixed-notch tables/figures; no original or native-array reads."""
import argparse,json
from pathlib import Path
import numpy as np
from point_additive_diagnostics import save_json
from point_response_cd import vector_measurement,MEASUREMENT_NAMES
from report_point_rank import SIGMA,equivalent

def read(p):return json.loads(p.read_text())
def fmt(v):return 'unavailable' if v is None else f'{v:.6g}'
def report(root,old,dest):
 dest.mkdir(parents=True,exist_ok=True);d=root/'report';bands=read(d/'matched-population-summary.json');maps=read(d/'final-map-comparison.json');native=read(d/'native-comparison.json');timing=read(d/'timing.json');responses={'B':read(old/'report/A/source-responses.json'),'C':read(d/'C/source-responses.json')};support={f:read(d/f/'population-support.json') for f in ('B','C')}
 lines=['# Fixed 11 Hz saved-data comparison','','B response is inherited from the prior A→B exact equivalence, reconfirmed by the new B1 gate. C is independently remeasured. Shared-data method sensitivities; conditional background-transfer uncertainties are not paired significance or absolute calibration truth.','', '| Quantity | B-control | C-fixed11 |','|---|---:|---:|']
 def add(name,b,c):lines.append(f'| {name} | {b} | {c} |')
 def band(f,nw,pop,stage,metric='line_power'):return next(r for r in bands if r['family']==f and r['network']==nw and r['population']==pop and r['stage']==stage and r['metric']==metric and r['segment']=='all' and r['domain']=='all-stage-matched')
 def med(f,key):return float(np.median([r[key] for r in timing if r['family']==f]))
 add('Treatment, selected/full nw11; nw12','No added notch','331/486; 151/491')
 for nw in (11,12):
  for stage in ('RTC_native','T5_residual'):add(f'nw{nw} treated {stage} median 9.75–12.25 Hz power',*(fmt(band(f,nw,'treated',stage)['detector_mean']['median']) for f in ('B','C')))
 for branch in ('C-core-plus','C-core-minus','C-shoulder'):
  vals=[]
  for f in ('B','C'):
   r=next(r for r in responses[f] if r['cycle']==5 and r['branch']==branch);q={x['name']:x for x in r['measurement']['comparisons']};vals.append(f"{q['peak']['ratio']:.6f} / {q['aperture_2']['ratio']:.6f}")
  add(branch+' T5 peak / aperture2 increment gain',*vals)
 add('T5 peak conditional sigma (mJy/beam)',*(fmt(r['conditional_sigmas']['peak']) for r in maps['measurements']))
 for key in ('HITS','SCI_SUPPORT'):add(key+' total',fmt(maps['full_support'][key]['B_sum']),fmt(maps['full_support'][key]['C_sum']))
 add('Command exit median (s)',*(f'{med(f,"exit"):.3f}' for f in ('B','C')))
 lines+=['','Integrated power units: (mJy/nominal beam)². Native x is converted with its accepted sample CAL multiplier for comparison; native RTC remains raw Δf/f. All-stage spectra use identical B∩C physical windows and fixed B M0 source exclusion. Full support losses below precede intersection.','', '## Timing: all repeats retained','','| Arm | Repeat | Usable T5 | Analysis | Exit | CPU | RTC wall bounds | Sampled tree RSS GB |','|---|---:|---:|---:|---:|---:|---|---:|']
 for r in timing:lines.append(f"| {r['family']} | {r['rep']} | {r['ready']:.3f} | {r['analysis']:.3f} | {r['exit']:.3f} | {r['CPU']:.3f} | {r['RTC_wall_occupancy_bounds'][0]:.3f}–{r['RTC_wall_occupancy_bounds'][1]:.3f} | {r['sampled_tree_RSS']/1e9:.3f} |")
 lines+=['','| Arm | Metric | Median | Minimum | Maximum |','|---|---|---:|---:|---:|']
 for f in ('B','C'):
  for key in ('ready','analysis','exit','CPU','sampled_tree_RSS'):
   v=[r[key] for r in timing if r['family']==f];lines.append(f'| {f} | {key} | {np.median(v):.6g} | {min(v):.6g} | {max(v):.6g} |')
 lines+=['','RTC bounds use overlapping worker clocks, never worker sums as latency. Process CPU bounds and operation intervals are in timing.json. Required synchronous output/noise work is included. RTC captures and response extraction are untimed diagnostics; no overhead subtraction. Fresh processes, first use retained, OS caches not flushed. One workload, two network workers, one feedback worker and numerical thread.','', '## Matched contamination','','| Arm / network / population | Stage | Spectral/full detectors | Median / p95 power | Median excess | Median / p95 prominence | Prominent windows % | Median persistence |','|---|---|---:|---|---:|---|---:|---:|']
 for f in ('B','C'):
  for nw in (11,12):
   for pop in ('full','treated','untreated'):
    for stage in ('raw_native','RTC_native','CAL','PTC_M0','T5_input','T5_residual','T5_total'):
     b=band(f,nw,pop,stage);e=band(f,nw,pop,stage,'line_excess');p=band(f,nw,pop,stage,'line_prominence');v=b['detector_mean'];pv=p['detector_mean'];lines.append(f"| {f}/{nw}/{pop} | {stage} | {b['spectral_detectors']}/{b['full_detector_denominator']} | {fmt(v['median'])} / {fmt(v['p95'])} | {fmt(e['detector_mean']['median'])} | {fmt(pv['median'])} / {fmt(pv['p95'])} | {fmt(100*p['prominent_gt5_window_fraction'])} | {fmt(p['persistence']['median'])} |")
 lines+=['','Window and detector distributions, maxima, prevalence, persistence, per-processing-segment rows and absent-estimate denominators are retained in matched-population-summary.json. The two networks each have one physical native run; processing scan segments do not create new RTC domains. Prominence >5 is the inherited descriptive statistic, not a residual allowance or production threshold.','', '## Coherence','','| Arm / network | Population / stage | Pairs | Median | p95 |','|---|---|---:|---:|---:|']
 for f in ('B','C'):
  for file,label in [('population-coherence.json','full'),('treated-coherence.json','treated')]:
   for r in read(d/f'{f}-matched'/file):lines.append(f"| {f}/{r['network']} | {label}/{r['stage']} | {r['n']} | {fmt(r['median'])} | {fmt(r['p95'])} |")
 lines+=['','Same nearest-11-Hz bin, all eligible pairs with at least three common complete windows; no electronics attribution.','', '## Full-domain support before matching','','| Network / population | Native cells | B / C available | Lost / gained | B / C scheduled map centers | Lost scheduled centers |','|---|---:|---|---|---|---:|']
 for r in native:
  a=r['available'];m=r['map_center'];lines.append(f"| {r['network']}/{r['population']} | {r['native_cells']} | {a['B']} / {a['C']} | {a['lost']} / {a['gained']} | {m['scheduled_B']} / {m['scheduled_C']} | {m['scheduled_lost']} |")
 lines+=['','| Arm / network | Events / candidates / admitted jumps | Replacement / donor-influenced cells | Unrepaired influence | Source-near original / map-center scheduled cells |','|---|---|---|---:|---|']
 for f in ('B','C'):
  for r in support[f]:
   q=r['source_crossing_support']['inside_source_exclusion'];lines.append(f"| {f}/{r['network']} | {r['event_count']} / {r['candidate_count']} / {r['admitted_jump_groups']} | {r['replacement_centers']} / {r['replacement_influence']} | {r['unrepaired_influence']} | {q['scheduled_original_cells']} / {q['map_centers']} |")
 lines+=['','Raw x/r and all untreated filtered x/r, state, center, speed and cause arrays are byte-identical. Treated x/r share the fixed filter without coordinate mixing. Cause distributions and raw-scale adjacent-difference diagnostics within 0.25 s of unavailable boundaries versus away are retained for every detector. No donor replacement is exercised; one physical run per network limits real gap/replacement conclusions.','', '## Final T5 physical source differences','','| Measurement | B | C | C−B | B / C conditional sigma |','|---|---:|---:|---:|---|']
 b,c=maps['measurements']
 for name,v in b['values'].items():
  key=SIGMA.get(name);bs=b['conditional_sigmas'].get(key);cs=c['conditional_sigmas'].get(key);lines.append(f"| {name} | {fmt(v)} | {fmt(c['values'][name])} | {fmt(c['values'][name]-v)} | {fmt(bs)} / {fmt(cs)} |")
 lines+=['','Width-axis uncertainty keys sx/sy describe fitted sigma axes; they are not already FWHM-major/minor errors. No unsupported direct width discrepancy/scale ratio is claimed.','', '## Source response: each sign and morphology','','| Arm / branch / stage | Spatial gain / distortion | Peak / template / aperture2 increment gain |','|---|---|---|']
 effects=[]
 for f in ('B','C'):
  for r in responses[f]:
   q={x['name']:x for x in r['measurement']['comparisons']};lines.append(f"| {f}/{r['branch']}/T{r['cycle']} | {r['spatial']['g']:.6f} / {r['spatial']['eta']:.6f} | {q['peak']['ratio']:.6f} / {q['fixed_template']['ratio']:.6f} / {q['aperture_2']['ratio']:.6f} |")
   if r['cycle']==0:lines.append(f"| {f}/{r['branch']}/conditional RTC | {r['RTC_spatial']['g']:.6f} / {r['RTC_spatial']['eta']:.6f} | Fixed own-REF M0 projection; not an independent RTC map |")
   kinds=[('T'+str(r['cycle']),r['measurement'])]+([('conditional RTC',r['conditional_RTC_only_measurement'])] if r['cycle']==0 else [])
   for stage,m in kinds:
    for x in m['comparisons']:
     scale=r['endpoint_conditional_sigmas'].get(SIGMA.get(x['name'])) if stage=='T5' else None
     eq,why=equivalent(scale,x['ratio'],r['spatial']['eta'],r['branch']) if stage=='T5' and x['name'] in ('peak','fixed_template','aperture_1','aperture_1p5','aperture_2') else (None,'No scalar normalization for this quantity/stage')
     effects.append(dict(family=f,branch=r['branch'],stage=stage,**x,conditional_sigma=scale,discrepancy_over_sigma=x['absolute_discrepancy']/scale if scale else None,response_normalized_sigma=eq,normalization_scope=why))
 lines+=['','All physical actual−ideal discrepancies (peak, centroid, widths, template, each aperture, background), conditional scales, and allowed compact response-normalized scales are in effects.json. Raw-aperture and fitted-background terms remain in each source-responses.json. No scalar shoulder correction or science-map rescaling.','', '## Background and support','','| Arm | Common background pixels | Mean | Std | Robust scale | Full hits | Full science pixels |','|---|---:|---:|---:|---:|---:|---:|']
 for r in maps['backgrounds']:
  b=r['background'];lines.append(f"| {r['family']} | {r['background_pixels']} | {fmt(b['mean'])} | {fmt(b['std'])} | {fmt(b['robust_scale'])} | {r['hits']} | {r['full_science_pixels']} |")
 lines+=['','Background uses existing four-FWHM union exclusion on common supported full-map pixels, ordinary mean removal and existing 32-pixel / 200-pair covariance. Conditional endpoint models keep their original own-arm masks. No new uncertainty ensemble.','', '![Matched spectra](spectra.png)','','![T5 maps and difference](maps.png)','','![Background correlations](background.png)','','![Signed source residuals](response.png)']
 lines+=['','## Physical response discrepancies','','Actual minus ideal addition; peak/template in mJy/beam, centroid/width in arcsec, aperture2 in mJy/beam arcsec². RTC rows are conditional fixed-M0 projections.','', '| Arm / branch / stage | Peak | East / north | Major / minor FWHM | Template | Aperture2 |','|---|---:|---|---|---:|---:|']
 for f in ('B','C'):
  for r in responses[f]:
   kinds=[('T'+str(r['cycle']),r['measurement'])]+([('conditional RTC',r['conditional_RTC_only_measurement'])] if r['cycle']==0 else [])
   for stage,m in kinds:
    q={x['name']:x['absolute_discrepancy'] for x in m['comparisons']};lines.append(f"| {f}/{r['branch']}/{stage} | {fmt(q['peak'])} | {fmt(q['east'])} / {fmt(q['north'])} | {fmt(q['FWHM_major'])} / {fmt(q['FWHM_minor'])} | {fmt(q['fixed_template'])} | {fmt(q['aperture_2'])} |")
 lines+=['','## Shoulder aperture accounting at T5','','| Arm / construction | Raw aperture2 | Fitted-background term | Net aperture2 |','|---|---:|---:|---:|']
 for f in ('B','C'):
  r=next(r for r in responses[f] if r['cycle']==5 and r['branch']=='C-shoulder')
  for key in ('reference','actual','ideal_map_addition'):
   m=r['measurement'][key];lines.append(f"| {f}/{key} | {fmt(m['raw_apertures'][2]['value_mJy_per_beam_arcsec2'])} | {fmt(m['aperture_background_terms'][2])} | {fmt(m['apertures'][2]['value_mJy_per_beam_arcsec2'])} |")
 lines+=['','## Combined response and boundary illustrations','','![Fixed filter response](filter.png)','','![Native boundary and existing event traces](boundaries.png)','','Traces use the first selected detector at each physical end and the first three B events in selected detectors per network. Invalid filter support remains missing. They illustrate existing events rather than validating their completeness. Full population near-boundary/away adjacent-difference records are in population-support.json.']
 (dest/'METRICS.md').write_text('\n'.join(lines)+'\n');save_json(dest/'effects.json',effects);save_json(dest/'report-facts.json',dict(timing=timing,map_comparison=maps,native=native,resources=read(d/'resources.json')))
 figures(d,old,dest,support)

def figures(d,old,dest,support):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for col,nw in enumerate((11,12)):
  for f,style in [('B','-'),('C','--')]:
   z=np.load(d/f'{f}-matched/population-spectra.npz')
   for row,stages in enumerate([('raw','RTC'),('CAL','PTC_M0','T5_residual')]):
    for stage in stages:
     if row==0:a=z[f'n{nw}_{stage}'];freq=z['native_frequency']
     else:
      sr=next(r for r in read(d/f'{f}-matched/population-support.json') if r['network']==nw)['segments'];v=[(z[f'n{nw}_s{s["segment"]}_{stage}'],np.array(s['windows_by_full_population'])) for s in sr];cnt=sum(c for a,c in v);a=np.divide(sum(np.nan_to_num(a)*c for a,c in v),cnt,out=np.full_like(v[0][0],np.nan),where=cnt>0);freq=z['output_frequency']
     axs[row,col].semilogy(freq[1:],np.nanmedian(a,axis=1)[1:],ls=style,label=f'{f} {stage}')
  for row in (0,1):axs[row,col].set(title=f'Network {nw} · '+('native' if row==0 else 'all-stage matched'),xlabel='Hz',ylabel='Median PSD (mJy²/beam²/Hz)');axs[row,col].axvspan(9.75,12.25,color='gray',alpha=.12);axs[row,col].legend(fontsize=7);axs[row,col].grid(alpha=.2)
 fig.savefig(dest/'spectra.png',dpi=150);plt.close(fig)
 z=np.load(d/'final-map-comparison.npz');fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained');lim=np.nanmax([z['B_PRIMARY'],z['C_PRIMARY']]);mask=z['common'].astype(bool);dl=np.quantile(abs(z['difference'][mask]),.99)
 for ax,key,title in zip(axs,('B_PRIMARY','C_PRIMARY','difference'),('B T5','C T5','C − B (own scale)')):
  l=dl if key=='difference' else lim;im=ax.imshow(np.where(mask,z[key],np.nan),origin='lower',cmap='RdBu_r',vmin=-l,vmax=l);ax.set(title=title,xlabel='East-grid column',ylabel='North-grid row');fig.colorbar(im,ax=ax,label='mJy/beam',shrink=.8)
 fig.savefig(dest/'maps.png',dpi=150);plt.close(fig)
 m=read(d/'final-map-comparison.json');fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
 for r in m['backgrounds']:
  for ax,key in zip(axs[:2],('correlation_east','correlation_north')):ax.plot(r['lag_arcsec'],r[key],label=r['family']);ax.set(xlabel='Lag (arcsec)',ylabel='Correlation',title=key);ax.legend()
  vals=z[r['family']+'_PRIMARY'][z['background'].astype(bool)];axs[2].hist(vals,bins=np.linspace(-80,80,81),histtype='step',density=True,label=r['family'])
 axs[2].set(title='Common source-excluded background',xlabel='mJy/beam',ylabel='Density');axs[2].legend();fig.savefig(dest/'background.png',dpi=150);plt.close(fig)
 zz={'B':np.load(old/'report/A/response-maps.npz'),'C':np.load(d/'C/response-maps.npz')};branches=('C-core-plus','C-core-minus','C-shoulder');fig,axs=plt.subplots(2,3,figsize=(11,7),layout='constrained')
 for j,b in enumerate(branches):
  limit=max(np.max(abs(zz[f][f'{b}_T5_residual'])) for f in ('B','C'))
  for i,f in enumerate(('B','C')):
   im=axs[i,j].imshow(zz[f][f'{b}_T5_residual'],origin='lower',extent=(41,-41,-41,41),cmap='RdBu_r',vmin=-limit,vmax=limit);axs[i,j].set(title=f'{f} {b}',xlabel='East (arcsec)',ylabel='North (arcsec)')
  fig.colorbar(im,ax=axs[:,j].tolist(),label='Actual − REF − ideal (mJy/beam)',shrink=.8)
 fig.savefig(dest/'response.png',dpi=150);plt.close(fig)
 detail=read(d/'boundary-filter.json');z=np.load(d/'boundary-filter.npz');fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for nw in (11,12):
  for key,style in [('lowpass',':'),('combined','-')]:axs[0].semilogy(z[f'n{nw}_frequency'],np.maximum(z[f'n{nw}_{key}'],1e-12),ls=style,label=f'{nw} {key}')
  axs[1].plot(z[f'n{nw}_frequency'],z[f'n{nw}_combined'],label=str(nw))
 axs[0].set(xlabel='Hz',ylabel='Amplitude',title='Realized notch × antialias response');axs[1].set(xlabel='Hz',ylabel='Amplitude',xlim=(9,13),title='Fixed target response');axs[0].legend();axs[1].legend();fig.savefig(dest/'filter.png',dpi=150);plt.close(fig)
 ex=detail['examples'];fig,axs=plt.subplots(5,2,figsize=(12,13),layout='constrained')
 for ax,r in zip(axs.T.ravel(),ex):
  key=r['key'];t=z[key+'_time'];ax.plot(t,z[key+'_raw'],color='gray',alpha=.4,lw=.6,label='raw');ax.plot(t,z[key+'_B'],lw=.8,label='B');ax.plot(t,z[key+'_C'],lw=.8,label='C');ax.set(title=f"nw{r['network']} detector{r['detector']} {r['name']}",xlabel='Seconds from trace start',ylabel='mJy/beam');ax.legend(fontsize=7)
 for ax in axs.T.ravel()[len(ex):]:ax.axis('off')
 fig.savefig(dest/'boundaries.png',dpi=140);plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.root,a.old,a.output)
