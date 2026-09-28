import sys
from pathlib import Path
sys.path.insert(0,'/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor/validation/rtc_two_stage_2026-09-28')
from two_stage_population import *
windows=read(D/'new-windows.json');records=[]
for nw in (1,2):
 base,receipt,pr,stages,pix,off,distance,lon,lat=network(nw);shape=(pr['native_rows'],pr['detectors']);probe=base/'source-probe';x=arr(probe/'original-x.f64','<f8',shape);r=arr(probe/'original-r.f64','<f8',shape);v=(arr(probe/'input-valid.u8','u1',shape)>0)&(arr(probe/'input-r-valid.u8','u1',shape)>0)
 for q in [q for q in windows if q['key'] in ('n1_new0','n2_new1','n2_t5') and q['network']==nw]:
  idx=slice(q['first'],q['past_last']);j=q['detector'];ok=(v[idx]&off[idx]).all(axis=0);ok[j]=False;ids=np.flatnonzero(ok);cs={};ratios={}
  for c,z in [('x',x),('r',r)]:
   a=z[idx,j]-z[idx,j].mean();b=z[idx][:,ids];b=b-b.mean(axis=0);cs[c]=a@b/np.sqrt((a*a).sum()*(b*b).sum(axis=0));ratios[c]=np.std(b,axis=0)/np.std(a)
  shared=(abs(cs['x'])>=.5)&(abs(cs['r'])>=.5)
  records.append(dict(key=q['key'],network=nw,detector=j,cohort_n=len(ids),paired_coherent_count=int(shared.sum()),paired_coherent=[dict(detector=int(d),x_corr=float(cs['x'][i]),r_corr=float(cs['r'][i]),x_std_ratio=float(ratios['x'][i]),r_std_ratio=float(ratios['r'][i])) for i,d in enumerate(ids) if shared[i]],rule='whole saved window, original paired valid and outside 4FWHM throughout, absolute correlation >=0.5 in both coordinates; descriptive coherence, not causal proof'))
save_json(D/'candidate-cohort.json',records)
for q in records:print(q)
