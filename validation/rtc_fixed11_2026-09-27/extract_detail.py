"""Fixed filter response and saved event/boundary illustrations; no new processing."""
from pathlib import Path
import sys,numpy as np
REPO=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_rtc_diagnostics import read,yread,load_network,distribution
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-fixed11-20260927');D=ROOT/'report';S=read(REPO/'validation/rtc_fixed11_2026-09-27/APPLICABILITY.json');design=read(REPO/'validation/rtc_fixed11_2026-09-27/DESIGN.json');data={};records=[];filters=[]
for nw in (11,12):
 b=load_network(ROOT,'B',nw);c=load_network(ROOT,'C',nw);sel=[d['detector'] for n in S['networks'] if n['network']==nw for d in n['detectors'] if d['disposition']=='selected'];ev=yread(b['base']/'events.yaml');chosen=[e for e in ev if e['detector'] in sel][:3]
 examples=[dict(name='start',detector=sel[0],lo=0,hi=1000),dict(name='end',detector=sel[0],lo=b['shape'][0]-1000,hi=b['shape'][0])]+[dict(name=f"event-{e['event']}",detector=e['detector'],lo=max(0,e['seed_earlier_row']-488),hi=min(b['shape'][0],e['seed_earlier_row']+488),event=e) for e in chosen]
 for ex in examples:
  j=ex['detector'];sl=slice(ex['lo'],ex['hi']);key=f"n{nw}_{ex['name']}";data[key+'_time']=b['times'][sl]-b['times'][ex['lo']];data[key+'_raw']=b['raw'][sl,j]*b['mult'][sl,j]
  for f,a in [('B',b),('C',c)]:
   good=((a['state'][sl,j]&1)!=0)&a['mapping'][sl,j];data[key+'_'+f]=np.where(good,a['filtered'][sl,j]*a['mult'][sl,j],np.nan)
  common=np.isfinite(data[key+'_B'])&np.isfinite(data[key+'_C']);ex.update(network=nw,key=key,matched_retained_cells=int(common.sum()),C_minus_B=distribution((data[key+'_C']-data[key+'_B'])[common]));records.append(ex)
 h=np.array(design['finite_notch']);lp=np.array(b['receipt']['donor_continuity']['realized_detectors'][0]['lowpass_FIR']);dt=b['dt'];freq=np.linspace(0,.5/dt,8193)
 def gain(h):return abs(np.sum(np.exp(-2j*np.pi*freq[:,None]*dt*np.arange(len(h)))*h,axis=1))
 nh=gain(h);lh=gain(lp);data[f'n{nw}_frequency']=freq;data[f'n{nw}_notch']=nh;data[f'n{nw}_combined']=nh*lh;data[f'n{nw}_lowpass']=lh
 filters.append(dict(network=nw,dt=dt,notch_taps=len(h),lowpass_taps=len(lp),notch_half_support_seconds=(len(h)//2)*dt,combined_half_support_seconds=(len(h)//2+len(lp)//2)*dt,amplitudes={str(f):dict(notch=float(abs(np.sum(np.exp(-2j*np.pi*f*dt*np.arange(len(h)))*h))),combined=float(abs(np.sum(np.exp(-2j*np.pi*f*dt*np.arange(len(h)))*h))*abs(np.sum(np.exp(-2j*np.pi*f*dt*np.arange(len(lp)))*lp)))) for f in (0,1,3,5,10,11,11.006255844577186,12,15,30,40)}))
checks=[]
for out in sorted(ROOT.glob('*-*')):
 if not (out/'LAUNCH.json').exists():continue
 p=read(out/'freeze/prospective.json');assert p['fixed11_applicability']==bound(REPO/'validation/rtc_fixed11_2026-09-27/APPLICABILITY.json');assert p['source_revision']=='b3d997456a8b9762d9fc323e37827f96b4ec7b37';s=read(out/'run/BASE-POINT/REF/a2000/SUMMARY.json');assert s['actual_feedback_applications']==5
 checks.append(dict(run=out.name,source_revision=p['source_revision'],applicability=p['fixed11_applicability'],feedback_applications=5))
np.savez_compressed(D/'boundary-filter.npz',**data);save_json(D/'boundary-filter.json',dict(examples=records,filter_response=filters,illustration_selection='first selected detector at native ends; first three prior B events in selected detectors per network; fixed trace domains, no policy or membership change',run_bindings=checks))
