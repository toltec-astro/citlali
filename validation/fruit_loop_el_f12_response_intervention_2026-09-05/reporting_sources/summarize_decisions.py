"""Mechanical reporting of the registered state and recorded resource clocks."""
from pathlib import Path
import csv
import json
import re
import shlex
import sys
from netCDF4 import Dataset

r=Path('/Users/gwilson/.codex/worktrees/4c31/citlali-refactor')
sys.path.insert(0,str(r))
from tools.fruit_loops.analyze_response_intervention import iteration_dirs, file_record, scalar_text
root=Path('/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f12-response-aware-intervention-r0.1')
out=Path(sys.argv[1])
out.mkdir(parents=True,exist_ok=True)
census_rows=[]
applications=[]
joint_rows=[]
resources=[]
for arm in ('H0','H','Half','Hold'):
 for injection in ('uninjected','injected'):
  case=root/arm/injection
  receipt=case/'EXECUTION_RECEIPT.json'
  if not receipt.exists():
   resources.append({'arm':arm,'injection':injection,'status':'not_run'})
   continue
  result=json.loads(receipt.read_text())
  repair_identity=None
  if arm=='H' and injection=='uninjected':
   proof=case/'SCALAR_DIMENSION_REPAIR_R0.2.json'
   repair=json.loads(proof.read_text())
   assert repair['status']=='pass' and repair['original_failed_receipt']==file_record(receipt)
   result=repair['execution_receipt_after_comparison_repair']
   repair_identity=file_record(proof)
  text=(case/'reduction.log').read_text()
  profile=[float(value) for value in re.findall(r'profile stage=map.output context=outputting raw obs files elapsed_s=([0-9.]+)',text)]
  iteration_times={int(k):float(value) for k,value in re.findall(r'profile stage=reduction.iteration context=fruit_iter=(\d+) elapsed_s=([0-9.]+)',text)}
  resources.append({'arm':arm,'injection':injection,'status':result['status'],
   'resource_receipt':result.get('resources'),
   'elapsed_with_gates_and_retention_seconds':result['elapsed_including_gates_and_retention_seconds'],
   'map_output_including_accounting_seconds':sum(profile),'map_output_iteration_seconds':profile,
   'native_iteration_seconds':iteration_times,
   'timing_scope':'inclusive existing map-output profile; includes census, accounting and receipt output, plus ordinary map output; spool append cost is in timestream and total runtime',
   'temporary_spool_original_bytes':sum(x['original']['size_bytes'] for x in result.get('compressed_spools',[])),
   'retained_compressed_spool_bytes':sum(x['lossless_retained']['size_bytes'] for x in result.get('compressed_spools',[])),
   'receipt_identity':file_record(receipt),'comparison_repair_identity':repair_identity})
  if arm=='H0' or result['status']!='pass': continue
  for iteration,directory in sorted(iteration_dirs(case/'reduced',123424).items()):
   path=directory/'citlali_restart_checkpoint.nc'
   with Dataset(path) as f: lines=iter(scalar_text(f['fruit_response_state'][...]).splitlines())
   assert next(lines)=='SCI-FRUIT-EL-F12-STATE-R0.1+CAP-001'
   assert next(lines)==f'{arm} {iteration}'
   assigned={}
   for _ in range(int(next(lines))):
    row=shlex.split(next(lines));assigned[tuple(row[:4])]=[int(x) for x in row[4:]]
   for _ in range(int(next(lines))):
    row=shlex.split(next(lines));key=tuple(row[:4]);score=row[6:]
    assert len(row)==14
    source,first=assigned.get(key,[-1,-1])
    census_rows.append(dict(arm=arm,injection=injection,iteration=iteration,observation=key[0],array=int(key[1]),uid=int(key[2]),scan=int(key[3]),
     disposition=row[4],selected=int(row[5]),source_iteration=source,first_application=first,
     score_available=int(score[0]) if row[4]=='eligible' else None,
     support_risk=int(score[1]) if row[4]=='eligible' else None,
     response_ratio=float(score[2]) if row[4]=='eligible' else None,
     map_scale_mjy_beam=float(score[3]) if row[4]=='eligible' else None,
     footprint_pixels=int(score[4]) if row[4]=='eligible' else None,
     conditioned_pixels=int(score[5]) if row[4]=='eligible' else None,
     lost_support_pixels=int(score[6]) if row[4]=='eligible' else None,
     gained_support_pixels=int(score[7]) if row[4]=='eligible' else None,
     checkpoint=str(path)))
   for _ in range(int(next(lines))):
    row=shlex.split(next(lines)); assert len(row)==9
    joint_rows.append(dict(arm=arm,injection=injection,iteration=iteration,array=int(row[0]),
     score_available=int(row[1]),support_risk=int(row[2]),response_ratio=float(row[3]),
     map_scale_mjy_beam=float(row[4]),footprint_pixels=int(row[5]),conditioned_pixels=int(row[6]),
     lost_support_pixels=int(row[7]),gained_support_pixels=int(row[8]),checkpoint=str(path)))
   for _ in range(int(next(lines))):
    row=shlex.split(next(lines));key=tuple(row[:4]);assert len(row)==16
    stages=[row[4:10],row[10:16]]
    permitted=any(int(stage[0])==4 for stage in stages)
    entry=dict(arm=arm,injection=injection,iteration=iteration,observation=key[0],array=int(key[1]),uid=int(key[2]),scan=int(key[3]),
      source_iteration=assigned[key][0],first_application=assigned[key][1],combined_permission=int(permitted),
      multiplier_if_otherwise_admitted=.5 if arm=='Half' and permitted else 1.,checkpoint=str(path))
    for label,stage in zip(('pre_rtc','pre_ptc'),stages):
     entry.update({f'{label}_{field}':kind(value) for field,kind,value in zip(
      ('status','complete_proposed_fraction','cap','suppressed','independent_reason','ordinary_new_samples'),(int,float,float,int,int,int),stage)})
    applications.append(entry)
   assert list(lines)==[]
for name,rows in [('CANDIDATE_CENSUS_R0.1.csv',census_rows),('APPLICATION_RECEIPTS_R0.1.csv',applications),('JOINT_SCORES_R0.1.csv',joint_rows)]:
 if rows:
  with (out/name).open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (out/'RESOURCE_SUMMARY_R0.1.json').open('x') as f: f.write(json.dumps(resources,indent=2,allow_nan=False)+'\n')
print(json.dumps({'census_records':len(census_rows),'joint_records':len(joint_rows),'application_records':len(applications),'trajectories':len(resources)}))
