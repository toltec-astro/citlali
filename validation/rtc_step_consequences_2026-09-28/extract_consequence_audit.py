"""Saved-native correctness, decision reasons and MAP exposure; no new fit/run."""
from pathlib import Path
import sys,json,re,collections
import numpy as np,yaml
from astropy.io import fits
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_additive_diagnostics import save_json
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928');D=R/'report';REF=R/'N14-REF-rep2'
def y(p):return yaml.load(p.read_text(),Loader=yaml.CSafeLoader)
def read(p):return json.loads(p.read_text())
def arr(p,t,shape=None):
 a=np.fromfile(p,t);return a.reshape(shape) if shape else a
checks=[];attempts=[];reasons=[]
for out in sorted(R.glob('N14-*')):
 if read(out/'EXECUTION.json')['status']!='completed':continue
 for nw in (1,2,3,4):
  p=out/f'upstream/products/network{nw}';rec=y(p/'receipt.yaml');apply=y(p/'donor-continuity/apply-receipt.yaml');pr=y(p/'source-probe/receipt.yaml');n=pr['native_rows'];d=pr['detectors']
  assert rec['explicit_array_lowpass']['array']==0 and rec['explicit_array_lowpass']['factor']==1
  mismatches=0;crossings=0;replacements=0
  for a in apply['realized_detectors']:
   assert a['factor']==1 and a['phase_native_rows']==0 and len(a['lowpass_FIR'])==67 and not a['finite_notch_FIR']
   half=len(a['lowpass_FIR'])//2;planned=np.zeros(n,bool)
   for lo,hi in a.get('admitted_runs') or []:
    crossings+=not any(x<=lo and hi<=z for x,z in rec['native_runs'])
    if hi-lo>2*half:planned[lo+half:hi-half]=True
   state=arr(p/f"donor-continuity/{a['channel']}-state.u8",'u1');mismatches+=int(np.count_nonzero(planned!=((state&1)!=0)));replacements+=int(np.count_nonzero(state&(4|16)))
  assert mismatches==crossings==replacements==0
  ev=y(p/'events.yaml');st=y(p/'step-evidence.yaml');de=y(p/'event-decisions.yaml');assert len(ev)==len(st)==len(de)
  assert all(a['event']==b['event']==c['event'] for a,b,c in zip(ev,st,de))
  checks.append(dict(run=out.name,network=nw,physical_runs=rec['native_runs'],planned_vs_realized_mismatches=mismatches,admitted_run_physical_gap_crossings=crossings,replacement_or_influence_cells=replacements,events=len(ev),candidate_count=rec['candidate_count'],admitted_groups=sum(e['jump_admitted'] for e in ev),array='a1100',factor=1,phase=0,taps=67))
  cs=[c for e in st for c in e['coordinates']]
  attempts.append(dict(run=out.name,network=nw,coordinate_records=len(cs),**{k:dict(true=sum(c.get(k) is True for c in cs),false=sum(c.get(k) is False for c in cs),absent=sum(k not in c for c in cs)) for k in ('refit_requested','remeasure_requested','reassessed','admitted')}))
  if out==REF:
   reasons.append(dict(network=nw,classes=dict(collections.Counter(e['class'] for e in de)),coordinate_admission=dict(collections.Counter('joint' if all(c['admitted'] for c in e['coordinates']) else 'x-only' if e['coordinates'][0]['admitted'] else 'r-only' if e['coordinates'][1]['admitted'] else 'neither' for e in st)),coordinate_causes={k:dict(collections.Counter(str(c.get(k,'absent')) for c in cs)) for k in ('amplitude_cause','consistency_cause','admission_cause','transition_request_cause')},unresolved_groups=sum(e['class']=='unresolved-extent' for e in de),unresolved_detectors=len({e['channel'] for e in de if e['class']=='unresolved-extent'}),unresolved_not_recovered=sum(e['class']=='unresolved-extent' and not e['recovered'] for e in de)))
save_json(D/'correctness.json',dict(pass_=True,checks=checks,scope='all16 successful network applications independently eroded complete67-tap plan support; actual x-state equal; runs stay within physical runs; no donor synthesis'))
save_json(D/'internal-reassessments.json',dict(scope='coordinate requests in existing one support-overlap reassessment, not new observational parents',records=attempts));save_json(D/'decision-reasons.json',reasons)
pop=read(D/'N14-REF-rep2/population.json');packed=np.load(D/'N14-REF-rep2/population-masks.npz');sources={n['network']:np.unpackbits(packed[f"n{n['network']}_source"],axis=1)[:,:n['shape'][0]].T.astype(bool) for n in pop};maprows=[]
for k in (0,5):
 product=REF/'M0/products/a1100' if k==0 else REF/'run/BASE-POINT/REF/a1100/cycle-05/native/products';groups=[]
 if k==0:
  for c in y(product/'plan.yaml')['ordered_column_domain']:
   nw,sid=map(int,re.search(r'network-(\d+)-segment-(\d+)',c['publication_file']).groups());ptc=REF/f'upstream/products/network{nw}/donor-continuity/ptc';s=next(s for s in y(ptc/'receipt.yaml')['segments'] if s['scan']==sid);groups.append((s,ptc/f'segment-{sid}-causes.u8',c['first_input_row']))
 else:groups=[(q['source_segment'],product/f"network-{q['source_segment']['network']}-segment-{q['source_segment']['scan']}/causes.u8",q['MAP_first_input_row']) for q in y(product/'receipt.yaml')['segments']]
 pixels=arr(product/'operator-pixel.i32','<i4');by={nw:collections.Counter() for nw in (1,2,3,4)}
 with fits.open(product/('map.fits' if k==0 else 'total.fits')) as h:
  support=np.asarray(h['SCI_SUPPORT'].data).ravel().astype(bool);totalhits=int(h['HITS'].data.sum());norm=float(h['NORMALIZATION'].data.sum())
 for s,causes,first in groups:
  nw=s['network'];ptc=REF/f'upstream/products/network{nw}/donor-continuity/ptc';rows=arr(ptc/f"segment-{s['scan']}-slots.i64",'<i8');ids=np.asarray(s['output_detector_indices']);keep=arr(causes,'u1',(s['scheduled_times'],s['output_detectors']))==0
  assert int(keep.sum())==s['retained'];src=sources[nw][rows[:,None],ids][keep];pix=pixels[first:first+len(src)];hit=pix>=0
  by[nw].update(dict(PTC_retained_contributions=len(src),MAP_contributions=int(hit.sum()),MAP_source_context_contributions=int((src&hit).sum()),MAP_background_contributions=int((~src&hit).sum()),MAP_science_supported_contributions=int(support[pix[hit]].sum())))
 assert sum(v['MAP_contributions'] for v in by.values())==totalhits
 maprows.append(dict(cycle=k,networks=[dict(network=nw,**v) for nw,v in by.items()],MAP_contributions=totalhits,normalization_sum=norm,weighting='existing uniform gamma; contribution count is exposure accounting, not an independent-information count'))
save_json(D/'MAP-exposure.json',maprows)
print('verified',len(checks),'network applications; MAP',[(q['cycle'],q['MAP_contributions']) for q in maprows])
