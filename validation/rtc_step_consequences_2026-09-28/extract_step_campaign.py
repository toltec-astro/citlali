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
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928')
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
 obs=read(out/'freeze/prospective.json')['observation'];summary=read(out/'run/BASE-POINT/REF/a1100/SUMMARY.json') if (out/'run/BASE-POINT/REF/a1100/SUMMARY.json').exists() else None
 params=summary['histories'][0]['measurement']['guard']['accepted_parameters'] if summary else None
 eventtable=[];networks=[];windows=[];packed={};maskpack={};timing=[]
 for nw in (1,2,3,4):
  base=out/f'upstream/products/network{nw}';r=y(base/'receipt.yaml');probe=base/'source-probe';pr=y(probe/'receipt.yaml');n,d=pr['native_rows'],pr['detectors'];shape=n,d;factor=r['explicit_array_lowpass']['factor'];require(factor==1,'frozen a1100 cadence changed');dt=r['native_integration_seconds'];times=arr(probe/'assigned-time.f64','<f8')
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
   ev=dict(observation=obs,network=nw,event=e['event'],detector=j,channel=e['channel'],occurrence=r['detectors'][j]['occurrence'],seed_native_row=row,seed_unix=float(times[row]),selection_nominal_distance_arcsec=float(np.hypot(lon[row,j],lat[row,j])) if mapping[row,j] else None,source_distance_arcsec=float(dist[row,j]) if np.isfinite(dist[row,j]) else None,geometric_source_crossing=bool(source[row,j]) if params else None,original=e,chain=st,consider=de,applied=dict(excluded_at_seed=bool(not masks['RTC_x'][row,j]),map_center_at_seed=bool(masks['map_center'][row,j]),donor_replacement=False))
   eventtable.append(ev)
  # Frozen original/evidence-only rules, no cleaned trace enters ranking.
  ne=[e for e in eventtable if e['network']==nw];admitted=[e for e in ne if e['original']['jump_admitted']];chosen=[]
  def take(items,label):
   for e in items:
    if e['event'] not in [q[0]['event'] for q in chosen]:chosen.append((e,label));break
  score=lambda e:max([abs(q['offset_sigma']) for q in e['chain']['coordinates'] if np.isfinite(q['offset_sigma'])] or [0])
  ordered=sorted(admitted,key=lambda e:(score(e),e['event']))
  if ordered:
   take([ordered[len(ordered)//2]],'accepted median original offset/sigma')
   take(list(reversed(ordered)),'accepted largest original offset/sigma')
  ronly=[e for e in admitted if not e['chain']['coordinates'][0]['admitted'] and e['chain']['coordinates'][1]['admitted']]
  take(sorted(ronly,key=lambda e:(-score(e),e['event'])),'r-only admission with largest original offset/sigma')
  # Fourth example varies prospectively across the four networks, not outcomes.
  if nw==1:take(sorted(ne,key=lambda e:(float('inf') if e['selection_nominal_distance_arcsec'] is None else e['selection_nominal_distance_arcsec'],e['event'])),'nearest original AST pointing to nominal (0,0); independent of fitted source')
  if nw==2:take(sorted(admitted,key=lambda e:(min([abs(e['seed_native_row']-f['seed_native_row']) for f in ne if e['detector']==f['detector'] and e['event']!=f['event']] or [n]),e['event'])),'closest repeated original candidate')
  if nw==3:take([e for e in ne if e['consider']['class']=='unresolved-extent'],'unresolved original extent')
  if nw==4:take([e for e in ne if not e['original']['jump_admitted']],'first non-admitted original group')
  for e in sorted(ne,key=lambda e:e['event']):
   if len(chosen)>=4:break
   take([e],'original event identity fill')
  netwindows=[]
  for e,label in chosen[:4]:
   row=e['seed_native_row'];h=int(round(2.5/r['cadence_interval_seconds']));netwindows.append(dict(kind='event',event=e['event'],detector=e['detector'],first=max(0,row-h),past_last=min(n,row+h+1),center=row,rule=label))
  # Two high-contrast candidate-free windows, in separate halves. Robust
  # pre/post support is necessary; this ranking is NOT a step truth label.
  dtrow=r['cadence_interval_seconds'];h=max(8,int(round(.5/dtrow)));gap=max(2,int(round(.08/dtrow)))
  for half_start,half_end in ((h+gap,n//2),(n//2,n-h-gap)):
   best=None
   for row in range(half_start,half_end,max(1,h//2)):
    if not any(lo<=row-gap-h and row+gap+h<=hi for lo,hi in r['native_runs']):continue
    pre=raw[row-gap-h:row-gap];post=raw[row+gap:row+gap+h]
    valid=vx[row-gap-h:row+gap+h].all(axis=0)&mapping[row-gap-h:row+gap+h].all(axis=0)
    medpre=np.median(pre,axis=0);medpost=np.median(post,axis=0)
    scale=1.4826*np.median(np.abs(np.diff(np.concatenate((pre,post)),axis=0)-np.median(np.diff(np.concatenate((pre,post)),axis=0),axis=0)),axis=0)
    score0=np.divide(abs(medpost-medpre),scale,out=np.zeros(d),where=scale>0);score0[~valid]=0
    for e in ne:
     if abs(e['seed_native_row']-row)<h+gap:score0[e['detector']]=0
    j=int(np.argmax(score0));candidate=(float(score0[j]),-row,-j)
    if best is None or candidate>best[0]:best=(candidate,row,j)
   if best and best[0][0]>0:
    _,row,j=best;wh=int(round(1.5/dtrow));netwindows.append(dict(kind='unflagged',event=None,detector=j,first=max(0,row-wh),past_last=min(n,row+wh+1),center=row,rule='largest supported original x pre/post contrast in separate time half; candidate-free central 0.58s; not truth',original_selection_score=best[0][0]))
  # Two quiet coverage controls, spread in physical time/identity.
  for fraction,frac in ((.25,.25),(.75,.75)):
   center=int(fraction*(n-1));half=int(round(1.5/dtrow));target=int(frac*(d-1));lo=max(0,center-half);hi=min(n,center+half+1)
   for j in sorted(range(d),key=lambda j:(abs(j-target),j)):
    if any(a<=lo and hi<=b for a,b in r['native_runs']) and vx[lo:hi,j].all() and mapping[lo:hi,j].all() and not any(e['detector']==j and lo<=e['seed_native_row']<hi for e in ne):
     netwindows.append(dict(kind='unflagged',event=None,detector=j,first=lo,past_last=hi,center=center,rule=f'time fraction{fraction}, detector quantile{frac}; first covered candidate-free identity; no cleaned selection'));break
  # Persist selection before inspecting or extracting cleaned stage traces.
  selection_path=dest/f'nw{nw}-window-selection.json'
  if selection_path.exists():require(read(selection_path)==netwindows,'frozen selection changed')
  else:save_json(selection_path,netwindows)
  for wi,q in enumerate(netwindows):
   j=q['detector'];lo=q['first'];hi=q['past_last'];key=f'n{nw}_w{wi}';q.update(network=nw,key=key,channel=pr['channels'][j],source_context_available=params is not None,geometric_source_cells=int(source[lo:hi,j].sum()) if params else None,mapping_available_cells=int(mapping[lo:hi,j].sum()))
   q['physical_run_boundaries']=[v for run in r['native_runs'] for v in run if lo<v<hi];q['processing_boundaries']=sorted({v for sc in y(base/'processing-scans.yaml')['scans'] for run in sc['science_native'] for v in run if lo<v<hi})
   fields=dict(time=times[lo:hi],native_row=np.arange(lo,hi),raw_x=raw[lo:hi,j],raw_r=rawr[lo:hi,j],valid_x=vx[lo:hi,j],valid_r=vr[lo:hi,j],RTC_x=filtered[lo:hi,j],RTC_r=filteredr[lo:hi,j],CAL_multiplier=mult[lo:hi,j],source=source[lo:hi,j],distance=dist[lo:hi,j],state=states[lo:hi,j],cause=causes[lo:hi,j])
   for k in ('transition','fitting_guard','jump_scan','operation_unavailable','planned_filtered','map_center','speed'):fields[k]=masks[k][lo:hi,j]
   for k in ('CAL','M0','T5_residual','T5_total'):fields[k]=np.full(hi-lo,np.nan)
   peers=sorted(set(q.get('discovery_peers',[c['strongest_peer_channel'] for c in events[q['event']]['coordinates']] if q['event'] is not None else [max(0,j-1),min(d-1,j+1)])));q['peer_channels']=peers
   fields['peer_x']=np.where(vx[lo:hi,peers],raw[lo:hi,peers],np.nan);fields['peer_r']=np.where(vr[lo:hi,peers],rawr[lo:hi,peers],np.nan)
   q['original_x_pre_post_median']=[float(np.nanmedian(np.where(vx[lo:q['center'],j],raw[lo:q['center'],j],np.nan))),float(np.nanmedian(np.where(vx[q['center']+1:hi,j],raw[q['center']+1:hi,j],np.nan)))];q['original_r_pre_post_median']=[float(np.nanmedian(np.where(vr[lo:q['center'],j],rawr[lo:q['center'],j],np.nan))),float(np.nanmedian(np.where(vr[q['center']+1:hi,j],rawr[q['center']+1:hi,j],np.nan)))]
   q['original_x_difference_scale']=float(1.4826*np.median(abs(np.diff(raw[lo:hi,j])-np.median(np.diff(raw[lo:hi,j])))))
   q['adjudication']='pending original/peer visual review; production class unchanged'
   q['_fields']=fields
  ptc=base/'donor-continuity/ptc';ptcr=y(ptc/'receipt.yaml');segments=[]
  for s in ptcr['segments']:
   sid=s['scan'];nt=s['scheduled_times'];nd=s['detectors'];no=s['output_detectors'];ids=np.asarray(s['detector_indices']);cols=np.asarray(s['fit_columns_in_output']);stem=f'segment-{sid}';rows=arr(ptc/f'{stem}-slots.i64','<i8')*factor
   cal=arr(ptc/f'{stem}-input.f64','<f8',(nt,nd));eligible=arr(ptc/f'{stem}-eligible.u8','u1',(nt,nd)).astype(bool);clean=arr(ptc/f'{stem}-cleaned.f64','<f8',(nt,no))[:,cols];masks['CAL_eligible'][rows[:,None],ids]=eligible
   last=out/f'run/BASE-POINT/REF/a1100/cycle-05/native/products/network-{nw}-segment-{sid}'
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
   ledger.append(dict(cause=cause,gross_cells=int(m.sum()),gross_input_valid_cells=int((m&vx).sum()),incremental_preceding_eligible_cells=int(new.sum()),incremental_seconds=float(new.sum()*dt),source_crossing_incremental_cells=int((new&source).sum()) if params else None))
   excluded|=m
  prefilter=~excluded;filterloss=prefilter&~masks['RTC_x'];unexpected=masks['RTC_x']&excluded
  require(not unexpected.any(),'realized support crosses declared excluded union')
  ledger.append(dict(cause='complete_filter_support_expansion',gross_cells=int((~masks['RTC_x']).sum()),gross_input_valid_cells=int((vx&~masks['RTC_x']).sum()),incremental_preceding_eligible_cells=int(filterloss.sum()),incremental_seconds=float(filterloss.sum()*dt),source_crossing_incremental_cells=int((filterloss&source).sum()) if params else None))
  all_excluded=excluded|filterloss;require(np.array_equal(all_excluded,~masks['RTC_x']),'union accounting mismatch')
  physical_counts=collections.Counter(e['consider']['class'] for e in ne);admitted_det=sorted({e['detector'] for e in admitted});seedtimes=sorted(e['seed_unix'] for e in admitted);clusters=[]
  for t in seedtimes:
   if not clusters or t-clusters[-1][-1]>.05:clusters.append([t])
   else:clusters[-1].append(t)
  total_valid=int(vx.sum());burden=np.bincount([e['detector'] for e in admitted],minlength=d)
  net=dict(network=nw,observation=obs,shape=[n,d],integration_seconds=dt,cadence_seconds=r['cadence_interval_seconds'],physical_runs=r['native_runs'],input_valid_x_cells=total_valid,input_valid_r_cells=int(vr.sum()),paired_valid_cells=int((vx&vr).sum()),input_valid_detector_seconds=total_valid*dt,candidate_count=r['candidate_count'],event_groups=len(ne),admitted_jump_groups=len(admitted),admitted_detector_count=len(admitted_det),admitted_detector_fraction=len(admitted_det)/d,admitted_groups_per_detector_second=len(admitted)/(total_valid*dt),admitted_groups_per_detector=distribution(burden),descriptive_50ms_shared_clusters=[dict(first_unix=q[0],last_unix=q[-1],instances=len(q)) for q in clusters],cluster_caveat='temporal descriptive grouping only; physical event identity remains unresolved',classes=dict(physical_counts),exposure_ledger=ledger,fitting_guard_cells=int(masks['fitting_guard'].sum()),fitting_guards_are_not_output_exclusions=True,mask_intersections={a+' & '+b:int((masks[a]&masks[b]&vx).sum()) for a in cause_order for b in cause_order if a<b},RTC_retained_x_cells=int(masks['RTC_x'].sum()),RTC_retained_r_cells=int(masks['RTC_r'].sum()),RTC_scheduled_x_cells=int(masks['RTC_x'][::factor].sum()),RTC_map_center_cells=int(masks['map_center'][::factor].sum()),CAL_eligible_scheduled_cells=int(masks['CAL_eligible'].sum()),mapping_unavailable_cells=int((~mapping).sum()),donor_replacement_cells=int(masks['replacement'].sum()),donor_influence_cells=int(masks['replacement_influence'].sum()),plan_apply_equal=True,finite_support_crossings=0,segments=segments,source_crossing=dict(available=params is not None,basis='own accepted M0 geometry' if params else 'unavailable: no accepted M0 geometry; packed false source mask is not zero physical crossings',input_valid_scheduled=int((source&vx)[::factor].sum()) if params else None,RTC_scheduled=int((source&masks['RTC_x'])[::factor].sum()) if params else None,CAL_eligible=int((source&masks['CAL_eligible']).sum()) if params else None),union_loss_cells=int((all_excluded&vx).sum()),short_stable_runs=sum(0<hi-lo<=len(a['lowpass_FIR'])-1 for a in apply['realized_detectors'] for lo,hi in (a.get('admitted_runs') or [])),unassessed='only selected review windows adjudicated; full flags are not completeness truth')
  networks.append(net)
  maskpack[f'n{nw}_time']=times-times[0]
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
  p=out/'M0/products/a1100/map.fits' if k==0 else out/'run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits'
  if not p.exists():continue
  with fits.open(p,memmap=False) as hdus:
   ims={h.name:np.asarray(h.data) for h in hdus if h.data is not None};maps[str(k)]=dict(MAP_contributions=int(ims['HITS'].sum()),normalization_sum=float(ims['NORMALIZATION'].sum()),supported_pixels=int(ims['SCI_SUPPORT'].sum()))
 save_json(dest/'map-support.json',maps)
 print(name,'extracted',len(eventtable),'events',len(windows),'windows',[q['admitted_jump_groups'] for q in networks])
if __name__=='__main__':extract(sys.argv[1])
