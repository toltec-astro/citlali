"""Saved-array B/C audit; no scientific execution or new noise model."""
import json,sys,gzip
from pathlib import Path
from collections import defaultdict
import numpy as np
REPO=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_rtc_diagnostics import read,yread,load_network,distribution
from point_additive_diagnostics import save_json
from point_fruit_diagnostics import load
from point_rtc_response import map_path
from point_empirical import coordinates,stats
from point_post_fruit import frozen_masks
from point_additive import pair_covariance
from point_response_cd import vector_measurement,MEASUREMENT_NAMES
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-fixed11-20260927');D=ROOT/'report';selection=read(REPO/'validation/rtc_fixed11_2026-09-27/APPLICABILITY.json')
selected={n['network']:np.array([d['disposition']=='selected' for d in n['detectors']]) for n in selection['networks']}
# Full-domain native loss and strict untreated identity precede matched spectra.
audit=[]
for nw in (11,12):
 b=load_network(ROOT,'B',nw);c=load_network(ROOT,'C',nw);sel=selected[nw];assert b['channels']==c['channels'];assert np.array_equal(b['times'],c['times'])
 assert b['raw'].tobytes()==c['raw'].tobytes() and b['raw_r'].tobytes()==c['raw_r'].tobytes()
 for label,ids in [('full',np.ones(len(sel),bool)),('treated',sel),('untreated',~sel)]:
  fields={}
  for key in ('filtered','filtered_r','state','center','speed','causes'):
   x=b[key][:,ids];y=c[key][:,ids];same=(x==y)|(np.isnan(x)&np.isnan(y));fields[key]=dict(changed_cells=int((~same).sum()),byte_exact=x.tobytes()==y.tobytes())
  if label=='untreated':assert all(x['byte_exact'] for x in fields.values()),fields
  masks={f:((a['state']&1)!=0)[:,ids] for f,a in [('B',b),('C',c)]}
  row=dict(network=nw,population=label,detectors=int(ids.sum()),native_cells=int(b['shape'][0]*ids.sum()),fields=fields,raw_x_r_identical=True)
  for name,mb,mc in [('available',masks['B'],masks['C']),('map_center',b['center'][:,ids].astype(bool),c['center'][:,ids].astype(bool))]:
   row[name]=dict(B=int(mb.sum()),C=int(mc.sum()),lost=int((mb&~mc).sum()),gained=int((~mb&mc).sum()),scheduled_B=int(mb[::2].sum()),scheduled_C=int(mc[::2].sum()),scheduled_lost=int((mb&~mc)[::2].sum()))
  audit.append(row)
save_json(D/'native-comparison.json',audit)
# Same accepted estimator, group by full/treated/untreated, include processing segment.
summary=[]
for f in ('B','C'):
 records=json.loads(gzip.decompress((D/f'{f}-matched/window-bands.json.gz').read_bytes()))
 groups=defaultdict(list)
 for r in records:
  if r['value'] is None:continue
  for label in ('full','treated' if selected[r['network']][r['detector']] else 'untreated'):
   for seg in ('all',r['segment']):groups[(r['network'],label,r['domain'],r['stage'],r['metric'],seg)].append(r)
 for (nw,label,domain,stage,metric,seg),rr in groups.items():
  by=defaultdict(list)
  for r in rr:by[r['detector']].append(r['value'])
  vals=np.array([r['value'] for r in rr]);den=len(selected[nw]) if label=='full' else int((selected[nw] if label=='treated' else ~selected[nw]).sum())
  summary.append(dict(family=f,network=nw,population=label,domain=domain,stage=stage,metric=metric,segment=seg,full_detector_denominator=den,spectral_detectors=len(by),unassessed_detectors=den-len(by),window_count=len(rr),detector_mean=distribution([np.mean(v) for v in by.values()]),window_population=distribution(vals),prominent_gt5_window_fraction=float(np.mean(vals>5)) if metric=='line_prominence' else None,prominent_detector_fraction=float(np.mean([np.any(np.array(v)>5) for v in by.values()])) if metric=='line_prominence' else None,persistence=distribution([np.mean(np.array(v)>5) for v in by.values()]) if metric=='line_prominence' else None))
save_json(D/'matched-population-summary.json',summary)
# Existing full-map measurements and pair-count corrected correlation on a fixed common background.
images={};headers={};sums={};models={}
for f in ('B','C'):
 ref=ROOT/f'{f}-REF-rep1';images[f],headers[f]=load(map_path(ref,5));sums[f]=read(ref/'run/BASE-POINT/REF/a2000/SUMMARY.json');models[f]=dict(np.load(ref/'run/BASE-POINT/REF/a2000/endpoint/model.npz'))
assert images['B']['PRIMARY'].shape==images['C']['PRIMARY'].shape
for k in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2'):assert headers['B'][k]==headers['C'][k]
pars=[sums[f]['histories'][5]['measurement']['guard']['accepted_parameters'] for f in ('B','C')]
common,source,maskinfo=frozen_masks(list(images.values()),headers['B'],pars,factor=4.)
bg=common&~source;rows=[];maps={};corr={};measurements=[]
from report_point_rank import SIGMA
for f in ('B','C'):
 im=images[f];signal=im['PRIMARY'];vals=signal[bg];z=signal-float(np.mean(vals));cov,count=pair_covariance(z,bg,32,200);rho=cov/cov[32,32];corr[f]=rho
 rows.append(dict(family=f,background=stats(vals),background_pixels=int(bg.sum()),full_science_pixels=int(im['SCI_SUPPORT'].sum()),hits=int(im['HITS'].sum()),normalization=float(im['NORMALIZATION'].sum()),normalization_distribution=distribution(im['NORMALIZATION'][im['SCI_SUPPORT'].astype(bool)]),correlation_east=rho[32,:],correlation_north=rho[:,32],pair_counts=count,lag_arcsec=list(range(-64,65,2)),endpoint_model_summary=read(ROOT/f'{f}-REF-rep1/run/BASE-POINT/REF/a2000/endpoint/model.json')))
 for name in ('PRIMARY','SCI_SUPPORT','HITS','NORMALIZATION'):maps[f'{f}_{name}']=im[name]
 v=vector_measurement(sums[f]['histories'][5]['measurement']);sig={r['name']:r['conditional_sigma'] for r in sums[f]['endpoint_uncertainty']}
 measurements.append(dict(family=f,values=dict(zip(MEASUREMENT_NAMES,v)),conditional_sigmas=sig))
delta=images['C']['PRIMARY']-images['B']['PRIMARY'];maps['difference']=delta;maps['common']=common;maps['background']=bg;maps['source']=source
loss={name:dict(B_sum=float(images['B'][name].sum()),C_sum=float(images['C'][name].sum()),changed_pixels=int((images['B'][name]!=images['C'][name]).sum()),decreased_pixels=int((images['C'][name]<images['B'][name]).sum()),increased_pixels=int((images['C'][name]>images['B'][name]).sum())) for name in ('HITS','NORMALIZATION','SCI_SUPPORT')}
save_json(D/'final-map-comparison.json',dict(measurements=measurements,full_support=loss,background_mask=maskinfo,backgrounds=rows,background_difference=stats(delta[bg]),source_difference=stats(delta[common&source]),scope='Shared-data method sensitivity, not independent realizations. Existing four-FWHM union and 32-pixel/200-pair covariance; ordinary mean removed, no clipping or gradient subtraction. No new uncertainty learning.'))
np.savez_compressed(D/'final-map-comparison.npz',**maps)
print('extra extraction complete')
