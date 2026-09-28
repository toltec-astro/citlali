"""Scalar explanation of all active changes; no new exemplar windows or fits."""
from pathlib import Path
import json,yaml,numpy as np
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928')
def y(p):return yaml.load(p.read_text(),Loader=yaml.CSafeLoader)
def arr(p,shape):return np.memmap(p,dtype='<f8',mode='r',shape=shape)
a=y(R/'report/response/active-selection.yaml');records=[]
for q in a:
 changed=[m for m in q['matches'] if m['original_support']!=m['injected_support'] or m['admission_changed'] or m['original_transition']!=m['injected_transition'] or m['original_class']!=m['injected_class']]
 if not (q['appeared'] or changed):continue
 nw=q['network'];b=R/('N14-'+q['branch'])/f'upstream/products/network{nw}';ref=R/'N14-REF-rep2'/f'upstream/products/network{nw}'
 pr=y(b/'source-probe/receipt.yaml');shape=(pr['native_rows'],pr['detectors']);x=arr(b/'source-probe/original-x.f64',shape);inc=arr(b/'source-probe/increment-x.f64',shape);rc=y(ref/'step-evidence.yaml');be=y(b/'events.yaml');re=y(ref/'events.yaml')
 for new,m in [(True,e) for e in q['appeared']]+[(False,e) for e in changed]:
  ev=m['original'] if new else be[m['injected_event']];ch=ev['channel'];seed=ev['seed_earlier_row'];support=ev['coordinates'][0]['fit_support'];rows=np.concatenate([np.arange(*s['rows']) for s in support]);rows=rows[(rows>=0)&(rows<shape[0])]
  r=dict(branch=q['branch'],network=nw,channel=ch,new_event=new,seed=seed,original_difference=float(x[seed+1,ch]-x[seed,ch]),injection_difference=float(inc[seed+1,ch]-inc[seed,ch]),injection_max_abs_on_event_fit_support=float(np.max(abs(inc[rows,ch]))),injection_max_abs_entire_detector=float(np.max(abs(inc[:,ch]))),injection_nonzero_entire_detector=int(np.count_nonzero(inc[:,ch])),original_fit_offset=ev['coordinates'][0]['offset'],original_fit_scale=ev['coordinates'][0]['scale'],source_probe_original_x_hash=pr['original_x_sha256'])
  if new:
   candidate=m['chain']['candidates'][0];block=candidate['noise_block'];r['branch_candidate']=candidate;r['branch_sigma_delta']=m['chain']['coordinates'][0]['sigma_delta']
   candidates=[c for e in rc if e['detector']==ch for c in e['candidates'] if c['coordinate']=='x' and c['noise_block']==block]
   if candidates:
    c=candidates[0];sigma=abs(c['centered_difference'])/c['absolute_score'];center=c['difference']-c['centered_difference'];r['reference_same_block_candidate']=c;r['reference_sigma_inferred_from_saved_score']=sigma;r['reference_difference_center']=center;r['reference_seed_score_from_saved_noise']=abs(r['original_difference']-center)/sigma
   r['admission']='x amplitude and short-fit consistency pass, finite transition available, support-overlap refit/reassessment measured; r not seeded; original peers support an existing disturbance'
  else:
   r['matched_change']=m;r['reference_original_fit_offset']=re[m['reference_event']]['coordinates'][0]['offset']
  records.append(r)
(R/'report/response/active-change-explanation.json').write_text(json.dumps(dict(scope='scalar audit of all changed decisions; no extra review windows; original native values and stored fit/noise evidence, no new scientific fitting',records=records),indent=2,allow_nan=False)+'\n')
print(json.dumps(records,indent=2))
