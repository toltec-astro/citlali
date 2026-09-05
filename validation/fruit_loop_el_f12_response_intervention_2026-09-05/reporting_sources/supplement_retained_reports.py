"""Descriptive accounting summaries; no selector, protection, or ranking changes."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import sys
import numpy as np
from netCDF4 import Dataset

REPO=Path('/Users/gwilson/.codex/worktrees/4c31/citlali-refactor')
ROOT=Path('/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f12-response-aware-intervention-r0.1')
sys.path.insert(0,str(REPO))
from tools.fruit_loops.analyze_response_intervention import ARRAYS, iteration_dirs, product_path, image, support

out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
coefficients=[];reasons=[];iterations=[]
for arm in ('H','Half','Hold'):
 for injection in ('uninjected','injected'):
  tree=iteration_dirs(ROOT/arm/injection/'reduced',123424)
  ht=iteration_dirs(ROOT/'H'/injection/'reduced',123424)
  for k,directory in sorted(tree.items()):
   nc=next(directory.rglob('*_fruit_response.nc'))
   with Dataset(nc) as f:
    iterations.append(dict(arm=arm,injection=injection,iteration=k,science_occurrences=int(f.science_occurrences),
      spool_records=int(f.spool_records),spool_record_bytes=int(f.spool_record_bytes),ledger=str(nc)))
    for index,array in enumerate(ARRAYS):
     g=f.groups[f'array_{index}']
     a=product_path(directory,123424,array);h=product_path(ht[k],123424,array)
     av=image(a,'signal_I');hv=image(h,'signal_I');aw=image(a,'weight_I');hw=image(h,'weight_I')
     am=support(av,aw);hm=support(hv,hw);common=am&hm
     af=image(a,'weight_formal_I');hf=image(h,'weight_formal_I')
     assert common.any() and np.isfinite(af).all() and np.isfinite(hf).all()
     diff=af-hf;valid=common&(hf>0)&(af>0);ratio=af[valid]/hf[valid]
     coefficients.append(dict(arm=arm,injection=injection,iteration=k,array=array,
      ledger_pixel_terms=int(np.asarray(g['occurrence_pixel_count'][...]).sum()),
      max_unique_detector_count=int(np.asarray(g['unique_detector_count'][...]).max()),
      signed_C_sum=float(np.asarray(g['C'][...]).sum()),quadratic_Q_sum=float(np.asarray(g['Q'][...]).sum()),
      H_support_pixels=int(hm.sum()),candidate_support_pixels=int(am.sum()),
      lost_H_pixels=int((hm&~am).sum()),gained_pixels=int((am&~hm).sum()),
      formal_changed_pixels=int((af!=hf).sum()),formal_difference_rms_common=float(np.sqrt(np.mean(diff[common]**2))),
      formal_difference_max_abs=float(np.max(np.abs(diff))),formal_ratio_positive_common_pixels=int(valid.sum()),
      formal_ratio_min=float(ratio.min()) if ratio.size else None,formal_ratio_max=float(ratio.max()) if ratio.size else None,
      signal_changed_pixels=int((av!=hv).sum()),kernel_changed_pixels=int((image(a,'kernel_I')!=image(h,'kernel_I')).sum()),
      map=str(a),H_map=str(h)))
   # Every reason stays visible, including unrelated flags that retain precedence.
   buckets=defaultdict(lambda:{'records':0,'newly_flagged_samples':0,'proposed_samples':0,'matched_records':0,'applied_records':0})
   for path in directory.glob('learning*.csv'):
    with path.open(newline='') as stream:
     for row in csv.DictReader(stream):
      if int(row.get('iter',row.get('iteration','-1')))!=k:continue
      key=tuple(row.get(x,'') for x in ('record_type','producer','reason','application_stage','array'))
      b=buckets[key];b['records']+=1
      for field in ('newly_flagged_samples','proposed_samples','matched_records'):
       if row.get(field): b[field]+=int(float(row[field]))
      if row.get('applied') in ('1','true'):b['applied_records']+=1
   for key,value in sorted(buckets.items()):
    reasons.append(dict(arm=arm,injection=injection,iteration=k,
       **dict(zip(('record_type','producer','reason','application_stage','array'),key)),**value))
for name,rows in [('FORMAL_COEFFICIENT_AND_SUPPORT_R0.1.csv',coefficients),
                  ('REASON_SPECIFIC_RECORDS_R0.1.csv',reasons),('OCCURRENCE_TOTALS_R0.1.csv',iterations)]:
 with (out/name).open('x',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({'coefficient_rows':len(coefficients),'reason_rows':len(reasons),'ledger_rows':len(iterations)}))
