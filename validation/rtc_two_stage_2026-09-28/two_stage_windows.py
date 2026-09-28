"""Twelve bounded M0-audit excerpts; no candidate selection or science change."""
from two_stage_population import *
def main():
 pop=read(D/'population.json');selected=[];packed={};facts=[]
 selected=read(W/'validation/rtc_retained_residuals_2026-09-28/compact/report/new-window-selection.json')
 for q in selected:q.pop('T5',None)
 # Four final new windows (16 total new beyond the earlier 32): one per network,
 # largest fully supported background T5 mean departure not already excerpted.
 for nw in (1,2,3,4):
  used={q['detector'] for q in selected if q['network']==nw}
  rr=[q for q in pop if q['network']==nw and q['detector'] not in used and 'max_half_second_mean' in q['stages']['T5']]
  q=max(rr,key=lambda q:(abs(q['stages']['T5']['max_half_second_mean']),-q['detector']))
  row=q['stages']['T5']['max_half_second_mean_row'];lo,hi=q['native_interval']
  selected.append(dict(key=f'n{nw}_t5',network=nw,detector=q['detector'],channel=q['channel'],physical_run=q['physical_run'],first=max(lo,row-183),past_last=min(hi,row+184),center=row,selection='largest absolute fully supported 61-row background T5 mean departure on a detector not in previous 12 windows'))
 # Freeze selection before original/peer inspection, without any candidate outcomes.
 save_json(D/'new-window-selection.json',selected)
 for nw in (1,2,3,4):
  base,r,pr,stages,pix,off,distance,lon,lat=network(nw);probe=base/'source-probe';shape=(pr['native_rows'],pr['detectors']);raw=arr(probe/'original-x.f64','<f8',shape);rawr=arr(probe/'original-r.f64','<f8',shape);vx=arr(probe/'input-valid.u8','u1',shape)>0;vr=arr(probe/'input-r-valid.u8','u1',shape)>0;times=arr(probe/'assigned-time.f64','<f8');events=y(base/'events.yaml');dec=y(base/'event-decisions.yaml');steps=y(base/'step-evidence.yaml')
  for q in [q for q in selected if q['network']==nw]:
   lo,hi,j=q['first'],q['past_last'],q['detector'];idx=slice(lo,hi);peers=[max(0,j-1),min(pr['detectors']-1,j+1)];channel=pr['channels'][j]
   rtc=arr(base/f'donor-continuity/{channel}-filtered.f64','<f8',(shape[0],2));state=arr(base/f'donor-continuity/{channel}-state.u8','u1');mult=arr(probe/'cal-multiplier.f64','<f8',shape)
   fields=dict(row=np.arange(lo,hi),time=times[idx]-times[0],raw_x=np.where(vx[idx,j],raw[idx,j],np.nan),raw_r=np.where(vr[idx,j],rawr[idx,j],np.nan),peer_x=np.where(vx[idx,peers],raw[idx,peers],np.nan),peer_r=np.where(vr[idx,peers],rawr[idx,peers],np.nan),CAL=stages['CAL'][idx,j],M0=stages['M0'][idx,j],T5=stages['T5'][idx,j],RTC_x=np.where(state[idx]&1,rtc[idx,0],np.nan),RTC_r=np.where(state[idx]&2,rtc[idx,1],np.nan),multiplier=mult[idx,j],longitude_arcsec=lon[idx,j],latitude_arcsec=lat[idx,j],distance_arcsec=distance[idx,j],background=off[idx,j],map_pixel=pix[idx,j])
   for k,v in fields.items():packed[q['key']+'_'+k]=v
   local=[dict(event=e,consider=d,step=s) for e,d,s in zip(events,dec,steps) if e['detector']==j and e['trial_exclusion'][0]<hi and lo<e['trial_exclusion'][1]]
   trace=[]
   for e,ss,dd in zip(events,steps,dec):
    if e['detector']!=j:continue
    cs=[c for c in ss['candidates'] if lo<=c['earlier_row']<hi];left,right=e['trial_exclusion'];overlap=left<hi and lo<right
    if cs or overlap:trace.append(dict(event=e['event'],seed=e['seed_earlier_row'],raw_candidates_in_window=cs,trial_overlap=overlap,consider_class=dd['class'],jump_admitted=e['jump_admitted']))
   facts.append(dict(**q,learn_trace=trace,peer_channels=peers,MAP_contributions=int((pix[idx,j]>=0).sum()),source_contributions=int(((pix[idx,j]>=0)&~off[idx,j]).sum()),min_distance_arcsec=float(distance[idx,j].min()),max_distance_arcsec=float(distance[idx,j].max()),overlapping_event_trials=local,stages={k:summarize(stages[k][idx,j],off[idx,j],lo,hi) for k in ('CAL','M0','T5')},raw_x_mad=mad(fields['raw_x'][np.isfinite(fields['raw_x'])]),raw_r_mad=mad(fields['raw_r'][np.isfinite(fields['raw_r'])]),interpretation='baseline-targeted diagnostic, not by itself a proven non-astronomical class; neighboring detector indices are context, not guaranteed physical neighbors'))
 save_json(D/'new-windows.json',facts);np.savez_compressed(D/'new-windows.npz',**packed);print('saved',len(facts),'new windows')
if __name__=='__main__':main()
