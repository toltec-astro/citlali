"""Read-only quantitative context for the one proposed paired-excursion class."""
import sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor/validation/rtc_two_stage_2026-09-28')))
from two_stage_population import *
annotations=[dict(key='n1_new0',network=1,detector=451,before=[.4,.9],during=[1.2,1.45],after=[2.65,2.75]),dict(key='n2_new1',network=2,detector=88,before=[36.8,37.0],during=[37.2,37.45],after=[37.7,38.0]),dict(key='n2_t5',network=2,detector=309,before=[13.2,13.6],during=[13.9,14.3],after=[14.65,14.85])]
records=[]
for nw in (1,2):
 base,receipt,pr,stages,pix,off,distance,lon,lat=network(nw);shape=(pr['native_rows'],pr['detectors']);probe=base/'source-probe';x=arr(probe/'original-x.f64','<f8',shape);r=arr(probe/'original-r.f64','<f8',shape);v=(arr(probe/'input-valid.u8','u1',shape)>0)&(arr(probe/'input-r-valid.u8','u1',shape)>0);t=arr(probe/'assigned-time.f64','<f8');t-=t[0]
 for q in [a for a in annotations if a['network']==nw]:
  j=q['detector'];masks={k:(t>=q[k][0])&(t<q[k][1]) for k in ('before','during','after')};ok=v&off
  contrasts={};cohorts=[]
  for c,z in [('x',x),('r',r)]:
   with warnings.catch_warnings():
    warnings.filterwarnings('ignore',message='All-NaN slice encountered',category=RuntimeWarning)
    before=np.nanmedian(np.where(ok[masks['before']],z[masks['before']],np.nan),axis=0);during=np.nanmedian(np.where(ok[masks['during']],z[masks['during']],np.nan),axis=0);after=np.nanmedian(np.where(ok[masks['after']],z[masks['after']],np.nan),axis=0)
   deltas=during-before;recovery=after-before;valid=np.array([ok[masks[k]].all(axis=0) for k in masks]).all(axis=0);peers=valid.copy();peers[j]=False
   # Same units within network; median and robust spread describe cohort, not event uncertainty.
   centre=float(np.median(deltas[peers]));spread=mad(deltas[peers]);contrasts[c]=dict(before=float(before[j]),during=float(during[j]),after=float(after[j]),delta=float(deltas[j]),recovery_delta=float(recovery[j]),cohort_n=int(peers.sum()),cohort_median_delta=centre,cohort_MAD=spread,target_to_cohort_MAD=float(abs(deltas[j]-centre)/spread),cohort_same_sign_at_least_half_target=int((np.sign(deltas[peers])==np.sign(deltas[j])) .__and__(abs(deltas[peers])>=.5*abs(deltas[j])).sum()))
  idx=masks['during'];records.append(dict(**q,channel=pr['channels'][j],occurrence=receipt['detectors'][j]['occurrence'],physical_run=0,native_interval=receipt['native_runs'][0],all_three_intervals_paired_valid=bool(v[np.logical_or.reduce(list(masks.values())),j].all()),all_three_intervals_background=bool(off[np.logical_or.reduce(list(masks.values())),j].all()),during_min_source_distance=float(distance[idx,j].min()),contrasts=contrasts,interpretation='Descriptive original-coordinate contrast and all-network off-source cohort; MAD is cohort scale, not uncertainty on the target step. Selected from baseline windows, no Q data.'))
save_json(D/'candidate-evidence.json',dict(annotations=records,scope='three displayed paired opposite-coordinate event examples; no complete classifier/catalogue claim'))
for q in records:print(json.dumps(q))
