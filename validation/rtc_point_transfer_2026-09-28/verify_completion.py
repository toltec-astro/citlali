"""Read-only completed-campaign binding/accounting audit; no fits or reductions.
Run with the local venv Python and --output-directory pointing to a writable directory.
"""
import argparse, collections, datetime, hashlib, json, re
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-directory', type=Path, required=True)
args=parser.parse_args()
r=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930')
ROLES=tuple(f'{family}-{branch}' for family in ('B','Q') for branch in ('REF','C-core-plus','C-core-minus','C-shoulder'))
PRIOR=[r.parent/name for name in ('successor-rtc-retained-residuals-20260928','successor-rtc-two-stage-20260928','successor-rtc-point-transfer-20260928','successor-rtc-point-transfer-build-repair-20260929','successor-rtc-point-transfer-runtime-repair-20260929')]
def bound(p):
 p=Path(p); h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return dict(path=str(p),sha256=h.hexdigest())
def verified(b):
 assert bound(b['path'])['sha256']==b['sha256'],b['path']
def tree_bytes(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
out=args.output_directory;out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
status=read(r/'STATUS.json');assert status['state']=='COMPLETE_PENDING_OWNER_REVIEW' and status['exit_code']==0
assert status['completed_roles']==list(ROLES)
manifest=read(r/'report/MANIFEST.json')
for b in manifest:verified(b)
accounts=read(r/'ACCOUNTING.json');assert accounts['accounting_complete'] and not accounts['unknown_evidence']
assert accounts['charged_attempts']==dict(parent=8,trajectory=8,feedback=40,endpoint=8)
for b in accounts['native_receipts']:verified(b['binding'])
assert accounts['completed_native_RTC_Apply_receipts']==32
role_results=[];constrained=[]
for role in ROLES:
 p=r/role;ex=read(p/'EXECUTION.json');la=read(p/'LAUNCH.json');s=read(p/'run/BASE-POINT/REF/a1100/SUMMARY.json')
 assert ex['status']==la['status']=='completed' and la['exit_code']==0
 assert s['completed_cycle']==5 and s['actual_feedback_applications']==5 and s['endpoint_status']=='completed' and not s['m10_applied']
 assert [h['cycle_index'] for h in s['histories']]==list(range(6))
 for h in s['histories']:
  for k in ('total','receipt','candidate'):verified(h[k])
  g=h['measurement']['guard'];assert g['accepted']
  if any(g['diagnostic']['active_mask']):constrained.append(dict(role=role,cycle=h['cycle_index'],active_mask=g['diagnostic']['active_mask']))
  if h['cycle_index']:
   req=read(p/f"run/BASE-POINT/REF/a1100/cycle-{h['cycle_index']:02d}/request.json")
   assert req['candidate']['sha256']==s['histories'][h['cycle_index']-1]['candidate']['sha256']
   assert req['cycle_index']==h['cycle_index'];verified(req['previous_state'])
 for f in p.rglob('run.log'):assert not re.search(r'\[(error|critical)\]',f.read_text(),re.I),f
 removal=read(p/'COMPLETED-SCRATCH-REMOVALS.json');assert all(not Path(z['path']).exists() for z in removal['files'])
 geometry=read(p/'freeze/measurement-geometry.json');assert geometry==read(r/'B-REF/freeze/measurement-geometry.json')
 for b in ('C-core-plus','C-core-minus','C-shoulder'):
  assert bound(p/f'freeze/shapes/{b}.fits')['sha256']==bound(r/f'B-REF/freeze/shapes/{b}.fits')['sha256']
 spans=collections.defaultdict(float)
 for e in map(json.loads,(p/'timing.jsonl').read_text().splitlines()):
  if e['event']=='span':spans[e['name']]+=e['seconds']
 role_results.append(dict(role=role,raw_to_final_seconds=la['exit_seconds'],initial_upstream_map_guard_seconds=spans['raw_parent_PTC_MAP_initial_guard_noise'],five_cycles_endpoint_seconds=spans['trajectory_including_endpoint'],peak_sampled_RSS_bytes=la['aggregate_tree_sampled_RSS'],retained_bytes=tree_bytes(p),scratch_removed_bytes=removal['removed_bytes']))
result=dict(source_revision=status['source_revision'],campaign_status=bound(r/'STATUS.json'),manifest=bound(r/'report/MANIFEST.json'),verified_compact_bindings=len(manifest),accounting=accounts['charged_attempts'],RTC_Apply=32,unexpected_error_logs=0,all_five_applications_verified=True,fixed_B_geometry_and_rasters_verified=True,constrained_executed_fits=constrained,roles=role_results,prior_roots_bytes=sum(tree_bytes(x) for x in PRIOR),campaign_bytes=tree_bytes(r),run_wall_seconds=(datetime.datetime.fromisoformat(status['completed_utc'])-datetime.datetime.fromisoformat(status['started_utc'])).total_seconds(),replacement_starts_used=0,remaining_replacement_starts=2)
(out/'execution-verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
