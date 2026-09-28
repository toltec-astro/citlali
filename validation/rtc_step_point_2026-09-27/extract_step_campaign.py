"""Extract compact Learn/Consider/Apply facts and predeclared windows; no science rerun."""
from pathlib import Path
import sys,json,gzip,collections,hashlib
import numpy as np,yaml
from astropy.io import fits
W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_additive_diagnostics import save_json
from point_rtc_diagnostics import distribution
from point_fruit_trajectory import bound
from inspect_forward_map import require
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-point-20260927')
def y(p):return yaml.load(p.read_text(),Loader=yaml.CSafeLoader)
def read(p):return json.loads(p.read_text())
def arr(p,dtype,shape=None):
 a=np.fromfile(p,dtype);return a.reshape(shape) if shape else a

def mark(mask,d,ranges):
 for lo,hi in ranges or []:
  if lo>=0 and hi>lo:mask[max(0,lo):min(len(mask),hi),d]=True

def ranges(a):
 x=np.flatnonzero(np.diff(np.r_[False,a,False].astype('i1')));return x.reshape(-1,2).tolist()

def extract(name):
 out=ROOT/name;dest=ROOT/'report'/name;dest.mkdir(parents=True,exist_ok=True)
 obs=read(out/'freeze/prospective.json')['observation'];summary=read(out/'run/BASE-POINT/REF/a2000/SUMMARY.json') if (out/'run/BASE-POINT/REF/a2000/SUMMARY.json').exists() else None
 params=summary['histories'][0]['measurement']['guard']['accepted_parameters'] if summary else None
 eventtable=[];networks=[];windows=[];packed={};maskpack={};timing=[]
 for nw in (11,12):
  base=out/f'upstream/products/network{nw}';r=y(base/'receipt.yaml');probe=base/'source-probe';pr=y(probe/'receipt.yaml');n,d=pr['native_rows'],pr['detectors'];shape=n,d;dt=r['native_integration_seconds'];times=arr(probe/'assigned-time.f64','<f8')
  raw=arr(probe/'original-x.f64','<f8',shape);rawr=arr(probe/'original-r.f64','<f8',shape);vx=arr(probe/'input-valid.u8','u1',shape).astype(bool);vr=arr(probe/'input-r-valid.u8','u1',shape).astype(bool);mapping=arr(probe/'mapping-available.u8','u1',shape).astype(bool)
  lon=arr(probe/'longitude-deg.f64','<f8',shape)*3600;lat=arr(probe/'latitude-deg.f64','<f8',shape)*3600;mult=arr(probe/'cal-multiplier.f64','<f8',shape)
  events=y(base/'events.yaml');steps=y(base/'step-evidence.yaml');dec=y(base/'event-decisions.yaml');support=y(base/'support-decisions.yaml');jump=y(base/'jump-exclusions.yaml');apply=y(base/'donor-continuity/apply-receipt.yaml')
  dist=np.hypot(lon-params[1],lat-params[2]) if params else np.full(shape,np.nan);source=(dist<=4*max(params[3:5])*np.sqrt(8*np.log(2)))&mapping if params else np.zeros(shape,bool)
  masks={k:np.zeros(shape,bool) for k in ('outside_science','transition','fitting_guard','jump_scan','unresolved_extent','operation_unavailable','transient','outside_resolved_domain','donor_contaminated','planned_filtered','RTC_x','RTC_r','map_center','replacement','replacement_influence','unrepaired_influence','speed','CAL_eligible')}
  filtered=np.full(shape,np.nan);filteredr=np.full(shape,np.nan);states=np.zeros(shape,'u1');causes=np.zeros(shape,'u1')
  masks['outside_science'][:]=True
  for sc in y(base/'processing-scans.yaml')['scans']:
   for lo,hi in sc['science_native']:masks['outside_science'][lo:hi,:]=False
  for a in jump.get('detectors') or []:mark(masks['jump_scan'],a['channel'],a['paired_exclusions'])
  for a in support.get('transient_exclusions') or []:mark(masks['transient'],a['channel'],a['excluded'])
  for a in support['detectors']:
   j=a['channel'];masks['outside_resolved_domain'][:,j]=True
   for lo,hi in a.get('stable_domains') or []:masks['outside_resolved_domain'][lo:hi,j]=False
   mark(masks['donor_contaminated'],j,a.get('donor_contaminated'))
  for a in apply['realized_detectors']:
   j=a['channel'];q=base/'donor-continuity';pair=arr(q/f'{j}-filtered.f64','<f8',(n,2));filtered[:,j]=pair[:,0];filteredr[:,j]=pair[:,1];state=arr(q/f'{j}-state.u8','u1');states[:,j]=state;causes[:,j]=arr(q/f'{j}-causes.u8','u1')
   masks['RTC_x'][:,j]=(state&1)!=0;masks['RTC_r'][:,j]=(state&2)!=0;masks['map_center'][:,j]=arr(q/f'{j}-map-center.u8','u1')!=0
   masks['replacement'][:,j]=(state&4)!=0;masks['replacement_influence'][:,j]=(state&16)!=0;masks['unrepaired_influence'][:,j]=(state&32)!=0;masks['speed'][:,j]=arr(q/f'{j}-speed-evidence.u8','u1')!=0
   half=len(a['lowpass_FIR'])//2;require(not a['finite_notch_FIR'],'parked notch active')
   for lo,hi in a.get('admitted_runs') or []:
    if hi-lo>2*half:masks['planned_filtered'][lo+half:hi-half,j]=True
  require(np.array_equal(masks['planned_filtered'],masks['RTC_x']),'plan/realized finite-support mismatch')
  require(not masks['replacement'].any() and not masks['replacement_influence'].any(),'unexpected natural donor treatment')
  for e,st,de in zip(events,steps,dec):
   require(e['event']==st['event']==de['event'],'event relation mismatch');j=e['detector'];row=e['seed_earlier_row'];mark(masks['fitting_guard'],j,[e['trial_exclusion']]);mark(masks['operation_unavailable'],j,[de['operation_unavailable']])
   if de['class']=='unresolved-extent':mark(masks['unresolved_extent'],j,[de['operation_unavailable']])
   for c in st['coordinates']:
    if c['admitted']:mark(masks['transition'],j,[c['admitted_affected']])
   ev=dict(observation=obs,network=nw,event=e['event'],detector=j,channel=e['channel'],occurrence=r['detectors'][j]['occurrence'],seed_native_row=row,seed_unix=float(times[row]),source_distance_arcsec=float(dist[row,j]) if np.isfinite(dist[row,j]) else None,geometric_source_crossing=bool(source[row,j]),original=e,chain=st,consider=de,applied=dict(excluded_at_seed=bool(not masks['RTC_x'][row,j]),map_center_at_seed=bool(masks['map_center'][row,j]),donor_replacement=False))
   eventtable.append(ev)
  # Deterministic representative selection uses ORIGINAL evidence only.
  ne=[e for e in eventtable if e['network']==nw];chosen=[]
  def take(items,label):
   for e in items:
    if e['event'] not in [q[0]['event'] for q in chosen]:chosen.append((e,label));break
  admitted=[e for e in ne if e['original']['jump_admitted']]
  score=lambda e:max([abs(q['offset_sigma']) for q in e['chain']['coordinates'] if np.isfinite(q['offset_sigma'])] or [0])
  if admitted:
   ordered=sorted(admitted,key=lambda e:(score(e),e['event']));take([ordered[len(ordered)//2]],'accepted median original offset/sigma');take(list(reversed(ordered)),'accepted largest original offset/sigma')
   neighbors=sorted(admitted,key=lambda e:(min([abs(e['seed_native_row']-f['seed_native_row']) for f in ne if e['detector']==f['detector'] and e['event']!=f['event']] or [n]),e['event']));take(neighbors,'closest original neighboring candidate')
  take(sorted(ne,key=lambda e:(float('inf') if e['source_distance_arcsec'] is None else e['source_distance_arcsec'],e['event'])),'nearest original geometric source context')
  for cl in ('unresolved-extent','isolated-admission-unavailable','no-resolved-excursion'):take([e for e in ne if e['consider']['class']==cl],cl)
  for e in sorted(ne,key=lambda e:e['event']):
   if len(chosen)>=6:break
   take([e],'original event identity fill')
  if obs==152420:
   # Four immutable-evidence choices per network plus two retained discovery
   # windows per network: twelve total event windows, not twelve plus discovery.
   oldchosen=chosen;chosen=[]
   wanted=('accepted median original offset/sigma','accepted largest original offset/sigma','unresolved-extent','nearest original geometric source context') if nw==11 else ('accepted median original offset/sigma','closest original neighboring candidate','isolated-admission-unavailable','nearest original geometric source context')
   for wanted_label in wanted:
    matches=[q for q in oldchosen if q[1]==wanted_label]
    if matches and matches[0][0]['event'] not in [q[0]['event'] for q in chosen]:chosen.append(matches[0])
   for q in oldchosen:
    if len(chosen)>=4:break
    if q[0]['event'] not in [x[0]['event'] for x in chosen]:chosen.append(q)
  chosen=chosen[:(4 if obs==152420 else 6)];netwindows=[]
  for e,label in chosen:
   row=e['seed_native_row'];h=int(round(2.5/r['cadence_interval_seconds']));netwindows.append(dict(kind='event',event=e['event'],detector=e['detector'],first=max(0,row-h),past_last=min(n,row+h+1),center=row,rule=label))
  if obs==152420:
   discovery=read(W/'validation/rtc_step_point_2026-09-27/CANDIDATE_DISCOVERY.json')
   for drow in [z for z in discovery['records'] if z['obsnum']==obs and z['network']==nw]:
    j=drow['review_tones'][0];center=int(np.argmin(abs(times-drow['center_unix'])));half=int(round(1.5/r['cadence_interval_seconds']))
    close=[e for e in ne if e['detector']==j and abs(e['seed_native_row']-center)<=int(round(.25/r['cadence_interval_seconds']))]
    event=min(close,key=lambda e:abs(e['seed_native_row']-center))['event'] if close else None
    netwindows.append(dict(kind='event',event=event,detector=j,first=max(0,center-half),past_last=min(n,center+half+1),center=center,rule='retained original-IQ discovery maximum, first ranked tone; acquisition receiver-time anchor, nearest ALIGN sample; not selected on cleaned result',discovery_key=drow['excerpt_key'],discovery_peers=drow['review_tones'][1:],anchor_residual_seconds=float(times[center]-drow['center_unix'])))
  # Six original time/coverage windows: three time fractions x two identity quantiles.
  for ti,fraction in enumerate((.2,.5,.8)):
   center=int(fraction*(n-1));half=int(round(1.5/r['cadence_interval_seconds']))
   for di,frac in enumerate((.25,.75)):
    target=int(frac*(d-1));poss=sorted(range(d),key=lambda j:(abs(j-target),j));selected=None
    for j in poss:
     lo=max(0,center-half);hi=min(n,center+half+1)
     if vx[lo:hi,j].all() and mapping[lo:hi,j].all() and not any(e['detector']==j and lo<=e['seed_native_row']<hi for e in ne):selected=j;break
    if selected is not None:netwindows.append(dict(kind='unflagged',event=None,detector=selected,first=lo,past_last=hi,center=center,rule=f'time fraction{fraction}, detector quantile{frac}; first covered candidate-free identity; no cleaned selection'))
  # Persist selection before inspecting or extracting cleaned stage traces.
  save_json(dest/f'nw{nw}-window-selection.json',netwindows)
  for wi,q in enumerate(netwindows):
   j=q['detector'];lo=q['first'];hi=q['past_last'];key=f'n{nw}_w{wi}';q.update(network=nw,key=key,channel=pr['channels'][j],geometric_source_cells=int(source[lo:hi,j].sum()),mapping_available_cells=int(mapping[lo:hi,j].sum()))
   fields=dict(time=times[lo:hi],native_row=np.arange(lo,hi),raw_x=raw[lo:hi,j],raw_r=rawr[lo:hi,j],valid_x=vx[lo:hi,j],valid_r=vr[lo:hi,j],RTC_x=filtered[lo:hi,j],RTC_r=filteredr[lo:hi,j],CAL_multiplier=mult[lo:hi,j],source=source[lo:hi,j],distance=dist[lo:hi,j],state=states[lo:hi,j],cause=causes[lo:hi,j])
   for k in ('transition','fitting_guard','jump_scan','operation_unavailable','planned_filtered','map_center','speed'):fields[k]=masks[k][lo:hi,j]
   for k in ('CAL','M0','T5_residual','T5_total'):fields[k]=np.full(hi-lo,np.nan)
   peers=sorted(set(q.get('discovery_peers',[c['strongest_peer_channel'] for c in events[q['event']]['coordinates']] if q['event'] is not None else [max(0,j-1),min(d-1,j+1)])));q['peer_channels']=peers
   fields['peer_x']=raw[lo:hi,peers];fields['peer_r']=rawr[lo:hi,peers]
   q['original_x_pre_post_median']=[float(np.median(raw[lo:q['center'],j])),float(np.median(raw[q['center']+1:hi,j]))];q['original_r_pre_post_median']=[float(np.median(rawr[lo:q['center'],j])),float(np.median(rawr[q['center']+1:hi,j]))]
   q['original_x_difference_scale']=float(1.4826*np.median(abs(np.diff(raw[lo:hi,j])-np.median(np.diff(raw[lo:hi,j])))))
   q['adjudication']='pending original/peer visual review; production class unchanged'
   q['_fields']=fields
  ptc=base/'donor-continuity/ptc';ptcr=y(ptc/'receipt.yaml');segments=[]
  for s in ptcr['segments']:
   sid=s['scan'];nt=s['scheduled_times'];nd=s['detectors'];no=s['output_detectors'];ids=np.asarray(s['detector_indices']);cols=np.asarray(s['fit_columns_in_output']);stem=f'segment-{sid}';rows=arr(ptc/f'{stem}-slots.i64','<i8')*2
   cal=arr(ptc/f'{stem}-input.f64','<f8',(nt,nd));eligible=arr(ptc/f'{stem}-eligible.u8','u1',(nt,nd)).astype(bool);clean=arr(ptc/f'{stem}-cleaned.f64','<f8',(nt,no))[:,cols];masks['CAL_eligible'][rows[:,None],ids]=eligible
   last=out/f'run/BASE-POINT/REF/a2000/cycle-05/native/products/network-{nw}-segment-{sid}'
   late=arr(last/'cleaned.f64','<f8',(nt,no))[:,cols] if last.exists() else np.full((nt,nd),np.nan)
   operand=arr(last/'operand.f64','<f8',(nt,nd)) if last.exists() else np.full((nt,nd),np.nan)
   for q in netwindows:
    col=np.flatnonzero(ids==q['detector']);sel=(rows>=q['first'])&(rows<q['past_last'])
    if len(col) and sel.any():
     j=col[0];where=rows[sel]-q['first'];good=eligible[sel,j]
     for k,v in [('CAL',cal),('M0',clean),('T5_residual',late),('T5_total',late+operand)]:q['_fields'][k][where]=np.where(good,v[sel,j],np.nan)
   segments.append(dict(scan=sid,scheduled_times=nt,fit_detectors=nd,eligible=int(eligible.sum()),retained=s['retained']))
  for q in netwindows:
   fields=q.pop('_fields')
   for k,v in fields.items():packed[q['key']+'_'+k]=v
   q['stage_finite_counts']={k:int(np.isfinite(fields[k]).sum()) for k in ('raw_x','RTC_x','CAL','M0','T5_residual','T5_total')};windows.append(q)
  # Overlapping gross masks and sequential union attribution (not a counterfactual).
  cause_order=['outside_science','transition','jump_scan','transient','unresolved_extent','operation_unavailable','outside_resolved_domain','donor_contaminated'];excluded=~vx;ledger=[]
  for cause in cause_order:
   m=masks[cause];new=m&~excluded
   ledger.append(dict(cause=cause,gross_cells=int(m.sum()),gross_input_valid_cells=int((m&vx).sum()),incremental_preceding_eligible_cells=int(new.sum()),incremental_seconds=float(new.sum()*dt),source_crossing_incremental_cells=int((new&source).sum())))
   excluded|=m
  prefilter=~excluded;filterloss=prefilter&~masks['RTC_x'];unexpected=masks['RTC_x']&excluded
  require(not unexpected.any(),'realized support crosses declared excluded union')
  ledger.append(dict(cause='complete_filter_support_expansion',gross_cells=int((~masks['RTC_x']).sum()),gross_input_valid_cells=int((vx&~masks['RTC_x']).sum()),incremental_preceding_eligible_cells=int(filterloss.sum()),incremental_seconds=float(filterloss.sum()*dt),source_crossing_incremental_cells=int((filterloss&source).sum())))
  all_excluded=excluded|filterloss;require(np.array_equal(all_excluded,~masks['RTC_x']),'union accounting mismatch')
  physical_counts=collections.Counter(e['consider']['class'] for e in ne);admitted_det=sorted({e['detector'] for e in admitted});seedtimes=sorted(e['seed_unix'] for e in admitted);clusters=[]
  for t in seedtimes:
   if not clusters or t-clusters[-1][-1]>.05:clusters.append([t])
   else:clusters[-1].append(t)
  total_valid=int(vx.sum());burden=np.bincount([e['detector'] for e in admitted],minlength=d)
  net=dict(network=nw,observation=obs,shape=[n,d],integration_seconds=dt,cadence_seconds=r['cadence_interval_seconds'],physical_runs=r['native_runs'],input_valid_x_cells=total_valid,input_valid_r_cells=int(vr.sum()),paired_valid_cells=int((vx&vr).sum()),input_valid_detector_seconds=total_valid*dt,candidate_count=r['candidate_count'],event_groups=len(ne),admitted_jump_groups=len(admitted),admitted_detector_count=len(admitted_det),admitted_detector_fraction=len(admitted_det)/d,admitted_groups_per_detector_second=len(admitted)/(total_valid*dt),admitted_groups_per_detector=distribution(burden),descriptive_50ms_shared_clusters=[dict(first_unix=q[0],last_unix=q[-1],instances=len(q)) for q in clusters],cluster_caveat='temporal descriptive grouping only; physical event identity remains unresolved',classes=dict(physical_counts),exposure_ledger=ledger,fitting_guard_cells=int(masks['fitting_guard'].sum()),fitting_guards_are_not_output_exclusions=True,mask_intersections={a+' & '+b:int((masks[a]&masks[b]&vx).sum()) for a in cause_order for b in cause_order if a<b},RTC_retained_x_cells=int(masks['RTC_x'].sum()),RTC_retained_r_cells=int(masks['RTC_r'].sum()),RTC_scheduled_x_cells=int(masks['RTC_x'][::2].sum()),RTC_map_center_cells=int(masks['map_center'][::2].sum()),CAL_eligible_scheduled_cells=int(masks['CAL_eligible'].sum()),mapping_unavailable_cells=int((~mapping).sum()),donor_replacement_cells=int(masks['replacement'].sum()),donor_influence_cells=int(masks['replacement_influence'].sum()),plan_apply_equal=True,finite_support_crossings=0,segments=segments,source_crossing=dict(input_valid_scheduled=int((source&vx)[::2].sum()),RTC_scheduled=int((source&masks['RTC_x'])[::2].sum()),CAL_eligible=int((source&masks['CAL_eligible']).sum())),union_loss_cells=int((all_excluded&vx).sum()),short_stable_runs=sum(0<hi-lo<=len(a['lowpass_FIR'])-1 for a in apply['realized_detectors'] for lo,hi in (a.get('admitted_runs') or [])),unassessed='only selected review windows adjudicated; full flags are not completeness truth')
  networks.append(net)
  for k,v in dict(masks,valid_x=vx,valid_r=vr,source=source).items():maskpack[f'n{nw}_{k}']=np.packbits(v.T,axis=1)
  for stagekey in ('ingress_seconds','Learn_seconds','Consider_evidence_seconds','transient_seconds','spectral_seconds','total_seconds'):timing.append(dict(network=nw,stage=stagekey,seconds=r.get(stagekey)))
  timing.append(dict(network=nw,stage='Apply_operation_seconds',seconds=apply.get('Apply_operation_seconds')))
 np.savez_compressed(dest/'windows.npz',**packed);np.savez_compressed(dest/'population-masks.npz',**maskpack)
 prior=read(dest/'windows.json') if (dest/'windows.json').exists() else []
 for q in windows:
  same=[v for v in prior if all(v[k]==q[k] for k in ('key','first','past_last','detector','event'))]
  if same:q['adjudication']=same[0]['adjudication']
 save_json(dest/'windows.json',windows)
 # YAML preserves explicit unavailable floating evidence; JSON report omits nonfinite metrics elsewhere.
 (dest/'event-chain.yaml').write_text(yaml.safe_dump(eventtable,sort_keys=False));save_json(dest/'population.json',networks);save_json(dest/'timing.json',timing)
 maps={}
 for k in (0,5):
  p=out/'M0/products/a2000/map.fits' if k==0 else out/'run/BASE-POINT/REF/a2000/cycle-05/native/products/total.fits'
  if not p.exists():continue
  with fits.open(p,memmap=False) as hdus:
   ims={h.name:np.asarray(h.data) for h in hdus if h.data is not None};maps[str(k)]=dict(MAP_contributions=int(ims['HITS'].sum()),normalization_sum=float(ims['NORMALIZATION'].sum()),supported_pixels=int(ims['SCI_SUPPORT'].sum()))
 save_json(dest/'map-support.json',maps)
 print(name,'extracted',len(eventtable),'events',len(windows),'windows',networks[0]['admitted_jump_groups'],networks[1]['admitted_jump_groups'])
if __name__=='__main__':extract(sys.argv[1])
