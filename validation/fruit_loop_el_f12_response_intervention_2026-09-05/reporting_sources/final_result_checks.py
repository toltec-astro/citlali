"""Final retained-product checks and descriptive resource comparisons."""
from pathlib import Path
import json
import sys
import time
from collections import Counter
REPO=Path('/Users/gwilson/.codex/worktrees/4c31/citlali-refactor')
ROOT=Path('/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f12-response-aware-intervention-r0.1')
sys.path.insert(0,str(REPO))
from tools.fruit_loops.analyze_response_intervention import file_record, write_json, iteration_dirs, product_path, require_map_identity
from tools.fruit_loops.run_response_intervention import ORDER, tree_bytes
out=ROOT/'analysis/REPORT_R0.1'
checks=[]
for arm in ('Half','Hold'):
 for injection in ('uninjected','injected'):
  candidate=iteration_dirs(ROOT/arm/injection/'reduced',123424)
  historical=iteration_dirs(ROOT/'H'/injection/'reduced',123424)
  for k in range(7):
   for array in ('a1100','a1400'):
    checks.append(require_map_identity(product_path(historical[k],123424,array),product_path(candidate[k],123424,array)))
write_json(out/'UNAFFECTED_ARRAY_IDENTITY_R0.1.json',{'status':'pass','map_files_checked':len(checks),
  'bitwise_planes_checked':4*len(checks),'checks':checks})
resources=json.loads((out/'RESOURCE_SUMMARY_R0.1.json').read_text())
by_case={(r['arm'],r['injection']):r for r in resources}
comparisons=[]
for arm in ('Half','Hold'):
 for injection in ('uninjected','injected'):
  a=by_case[(arm,injection)]['resource_receipt']
  for baseline in ('H0','H'):
   b=by_case[(baseline,injection)]['resource_receipt']
   comparisons.append(dict(arm=arm,injection=injection,reference=baseline,
    native_wall_ratio=a['wall_seconds']/b['wall_seconds'],
    native_cpu_ratio=(a['user_seconds']+a['system_seconds'])/(b['user_seconds']+b['system_seconds']),
    native_peak_rss_ratio=a['peak_rss_bytes']/b['peak_rss_bytes']))
write_json(out/'RESOURCE_COMPARISONS_R0.1.json',comparisons)
records=[]
for name in ORDER:
 path=ROOT/name/'EXECUTION_RECEIPT.json';receipt=json.loads(path.read_text());proof=None
 if name=='H/uninjected':
  p=ROOT/name/'SCALAR_DIMENSION_REPAIR_R0.2.json';proof=file_record(p)
  repair=json.loads(p.read_text());assert repair['original_failed_receipt']==file_record(path)
  receipt=repair['execution_receipt_after_comparison_repair']
 assert receipt['status']=='pass'
 records.append(dict(trajectory=name,status='pass',receipt=file_record(path),comparison_repair=proof,
   resources=receipt['resources'],compatibility=receipt['compatibility']))
archives=[file_record(REPO/name) for name in ('SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review.tar.gz','SCI-FRUIT-v0.1-empirical-lane-gate-0-r0.1-owner-review.tar.gz')]
assert [x['sha256'] for x in archives]==['5f11836908aa6aeb4f51690209a32dc6e8d4cee4e6b9c223903c6c57033b9b22','761ba278a53e32ad1d5d3977230cc4f1f90f257056b547508342797f584167ed']
first=json.loads((ROOT/'H0/uninjected/STARTED.json').read_text())['unix_time']
retained,spool=tree_bytes(ROOT)
write_json(out/'EXECUTION_COMPLETION_R0.1.json',dict(status='complete',primary_trajectories=8,primary_passes=56,
  conditional_restarts=0,diagnostic_replacements=0,attempt_count=8,completed=records,
  conditional_disposition='Neither candidate met protections and prioritized endpoints; no restart condition triggered.',
  archives=archives,retained_bytes_at_check=retained,temporary_spool_bytes_at_check=spool,
  aggregate_wall_seconds_at_check=time.time()-first,unchanged_arrays='56 maps / 224 science-plane comparisons bitwise',
  native_binary=file_record(ROOT/'setup/citlali-el-f12-r0.1')))
print(json.dumps({'map_checks':len(checks),'retained_GiB':retained/(1<<30),'aggregate_hours':(time.time()-first)/3600}))
