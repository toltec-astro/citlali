"""Selected POINT saved-array response; existing source estimators and REF MAP owners."""
from pathlib import Path
import sys,json,collections
import numpy as np
W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(W/'tools/timestream_successor'))
import point_rtc_response as response
from point_rtc_diagnostics import yread,arr,read
from point_additive_diagnostics import save_json
from inspect_forward_map import require
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-point-20260927');DEST=ROOT/'report/response';REF=ROOT/'S-REF-rep1'
def load_network(root,family,nw):
 base=root/f'{family}-REF-rep1/upstream/products/network{nw}';probe=base/'source-probe';pr=yread(probe/'receipt.yaml');n,d=pr['native_rows'],pr['detectors'];shape=n,d
 a=dict(shape=shape,channels=pr['channels'],probe_receipt=pr,mult=arr(probe/'cal-multiplier.f64','<f8',shape),valid=arr(probe/'input-valid.u8','u1',shape).astype(bool),mapping=arr(probe/'mapping-available.u8','u1',shape).astype(bool),filtered=np.full(shape,np.nan),state=np.zeros(shape,'u1'),center=np.zeros(shape,'u1'))
 for j,c in enumerate(a['channels']):
  q=base/'donor-continuity';a['filtered'][:,j]=arr(q/f'{c}-filtered.f64','<f8',(n,2))[:,0];a['state'][:,j]=arr(q/f'{c}-state.u8','u1');a['center'][:,j]=arr(q/f'{c}-map-center.u8','u1')
 return a
response.load_network=load_network
oldload=response.load
def load(path):
 data,header=oldload(path)
 if Path(path).parent.name=='shapes' and 'INJECTION' not in data:data['INJECTION']=data['PRIMARY']
 return data,header
response.load=load

def match_events(a,b):
 # Exact detector/native seed first; remaining one-to-one overlapping original
 # trial intervals only. Ambiguous overlaps stay unresolved, never force-match.
 pairs=[];left=set(range(len(a)));right=set(range(len(b)))
 for i,e in enumerate(a):
  matches=[j for j in right if b[j]['detector']==e['detector'] and b[j]['seed_earlier_row']==e['seed_earlier_row']]
  if len(matches)==1:j=matches[0];pairs.append((i,j,'same detector/native seed'));left.remove(i);right.remove(j)
 ambiguous=[]
 for i in sorted(left.copy()):
  e=a[i];lo,hi=e['trial_exclusion'];matches=[j for j in right if b[j]['detector']==e['detector'] and max(lo,b[j]['trial_exclusion'][0])<min(hi,b[j]['trial_exclusion'][1])]
  if len(matches)==1:
   j=matches[0];reverse=[k for k in left if a[k]['detector']==b[j]['detector'] and max(a[k]['trial_exclusion'][0],b[j]['trial_exclusion'][0])<min(a[k]['trial_exclusion'][1],b[j]['trial_exclusion'][1])]
   if len(reverse)==1:pairs.append((i,j,'unique overlapping original physical trial cells'));left.remove(i);right.remove(j)
   else:ambiguous.append(dict(reference_event=i,injected_events=matches))
  elif matches:ambiguous.append(dict(reference_event=i,injected_events=matches))
 return pairs,sorted(left),sorted(right),ambiguous

def active():
 records=[];packed={};refwindows=read(ROOT/'report/S-REF-rep1/windows.json')
 for branch in response.BRANCHES:
  out=ROOT/('S-'+branch)
  for nw in (11,12):
   rb=REF/f'upstream/products/network{nw}';bb=out/f'upstream/products/network{nw}';re=yread(rb/'events.yaml');be=yread(bb/'events.yaml');rd=yread(rb/'event-decisions.yaml');bd=yread(bb/'event-decisions.yaml');rc=yread(rb/'step-evidence.yaml');bc=yread(bb/'step-evidence.yaml');a=load_network(ROOT,'S',nw);n,d=a['shape'];times=arr(rb/'source-probe/assigned-time.f64','<f8');pairs,missing,new,ambiguous=match_events(re,be);matches=[]
   lost=0;gained=0;centerlost=0;centergained=0;eventmasks={}
   for j,ch in enumerate(a['channels']):
    q=bb/'donor-continuity';state=arr(q/f'{ch}-state.u8','u1');center=arr(q/f'{ch}-map-center.u8','u1').astype(bool);rv=(a['state'][:,j]&1)!=0;bv=(state&1)!=0;lost+=int((rv&~bv).sum());gained+=int((bv&~rv).sum());centerlost+=int((a['center'][:,j].astype(bool)&~center).sum());centergained+=int((center&~a['center'][:,j].astype(bool)).sum())
   for i,j,method in pairs:
    matches.append(dict(reference_event=i,injected_event=j,detector=re[i]['detector'],physical_seed_shift_seconds=float(times[be[j]['seed_earlier_row']]-times[re[i]['seed_earlier_row']]),matching=method,original_class=rd[i]['class'],injected_class=bd[j]['class'],admission_changed=re[i]['jump_admitted']!=be[j]['jump_admitted'],original_transition=[c['admitted_affected'] for c in rc[i]['coordinates']],injected_transition=[c['admitted_affected'] for c in bc[j]['coordinates']],original_support=rd[i]['operation_unavailable'],injected_support=bd[j]['operation_unavailable']))
   for q in refwindows:
    if q['network']!=nw:continue
    lo,hi=q['first'],q['past_last'];ch=q['channel'];stem=branch+'_'+q['key'];probe=bb/'source-probe';increment=arr(probe/'increment-x.f64','<f8',(n,d));packed[stem+'_injected_x']=arr(probe/'original-x.f64','<f8',(n,d))[lo:hi,ch]+increment[lo:hi,ch];packed[stem+'_increment_x']=increment[lo:hi,ch];packed[stem+'_filtered']=arr(bb/f'donor-continuity/{ch}-filtered.f64','<f8',(n,2))[lo:hi];packed[stem+'_state']=arr(bb/f'donor-continuity/{ch}-state.u8','u1')[lo:hi]
   record=dict(branch=branch,network=nw,reference_events=len(re),injected_events=len(be),matches=matches,disappeared=[dict(event=i,original=re[i],consider=rd[i],chain=rc[i]) for i in missing],appeared=[dict(event=j,original=be[j],consider=bd[j],chain=bc[j]) for j in new],ambiguous=ambiguous,RTC_lost_cells=lost,RTC_gained_cells=gained,map_center_lost_cells=centerlost,map_center_gained_cells=centergained,shifted_matches=sum(m['physical_seed_shift_seconds']!=0 for m in matches),decision_changes=sum(m['original_class']!=m['injected_class'] or m['admission_changed'] or m['original_transition']!=m['injected_transition'] or m['original_support']!=m['injected_support'] for m in matches),method='independent full active Learn/Consider/Apply in each branch; branch-local numeric IDs are not global identities; physical native time matched from same original acquisition')
   records.append(record)
 np.savez_compressed(DEST/'active-response-window-overlays.npz',**packed)
 # YAML retains explicit unavailable fit coefficients in event chains.
 import yaml
 (DEST/'active-selection.yaml').write_text(yaml.safe_dump(records,sort_keys=False))
 save_json(DEST/'active-selection-summary.json',[{k:v for k,v in r.items() if k not in ('matches','appeared','disappeared','ambiguous')}|dict(appeared=len(r['appeared']),disappeared=len(r['disappeared']),ambiguous=len(r['ambiguous'])) for r in records])

if __name__=='__main__':
 response.response(ROOT,'S',DEST)
 records=read(DEST/'source-responses.json')
 for r in records:
  r['spatial_scope']='conditional common supported pixels; ideal unchanged on complete unmodified REF support, excluded comparison regions enumerated below'
  r['unsupported_regions']=dict(RTC_reference_contributions_unavailable=r['RTC_delta_accounting']['lost_contributions'],RTC_science_pixels_unavailable=r['RTC_delta_accounting']['lost_science_pixels'],ideal_reference_contributions_unmapped=r['ideal_accounting']['lost_contributions'],actual_science_support_changed_pixels=r['support']['SCI_SUPPORT']['changed'])
  # Preserve loss/gain counts instead of treating absent outputs as zero response.
  refmap,_=load(response.map_path(REF,r['cycle']));branchmap,_=load(response.map_path(ROOT/('S-'+r['branch']),r['cycle']));rv=refmap['SCI_SUPPORT'].astype(bool);bv=branchmap['SCI_SUPPORT'].astype(bool)
  r['unsupported_regions'].update(actual_lost_science_pixels=int((rv&~bv).sum()),actual_gained_science_pixels=int((bv&~rv).sum()),reference_MAP_contributions=int(refmap['HITS'].sum()),injected_MAP_contributions=int(branchmap['HITS'].sum()),reference_supported_comparison_fraction=float((rv&bv).sum()/rv.sum()))
 save_json(DEST/'source-responses.json',records);active()
