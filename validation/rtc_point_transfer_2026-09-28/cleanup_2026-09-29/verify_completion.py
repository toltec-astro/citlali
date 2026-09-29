from pathlib import Path
import json,gzip,hashlib,subprocess
from datetime import datetime,timezone
W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');D=W/'validation/rtc_point_transfer_2026-09-28/cleanup_2026-09-29';R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-scratch-cleanup-20260929')
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(D/'OWNER_APPROVAL.json');r=read(R/'CLEANUP.json');s=read(R/'STATUS.json');v=read(D/'REVIEW.json');l=read(R/'LAUNCH.json')
assert s['state'] in ('COMPLETE_PENDING_OWNER_REVIEW','COMPLETE_VERIFIED') and s['phase']=='complete' and s['exit_status']==0 and not s['science_launched']
assert r['applied'] and r['files']==a['files']==s['deleted_files']==153062 and r['bytes']==a['bytes']==s['deleted_bytes']==16042260125
assert r['approval_sha256']==sha(D/'OWNER_APPROVAL.json')==v['approval_sha256'] and sha(D/'cleanup_native.py')==v['script_sha256']
assert v['verdict']=='pass' and l['reviewed_candidate']==r['reviewed_candidate']==v['candidate_sha']
assert r['manifest_bindings']==a['plans']
expected={};samples=[]
for p in a['plans']:
 m=Path(p['manifest']);assert sha(m)==p['manifest_sha256'];j=read(m)
 assert len(j['files'])==p['files'] and j['bytes']==p['bytes']
 for q in j['files']:
  assert q['path'] not in expected;expected[q['path']]=q['bytes']
 for q in (j['files'][0],j['files'][len(j['files'])//2],j['files'][-1]):
  assert not Path(q['path']).exists();samples.append(q['path'])
seen={}
for line in (R/'DELETED.jsonl').read_text().splitlines():
 q=json.loads(line);assert q['path'] not in seen;assert expected[q['path']]==q['bytes'];seen[q['path']]=q['bytes']
assert seen==expected
reports=[]
for spec in a['plans']:
 for phase in ('before','after'):
  p=R/(Path(spec['evidence_directory']).name+'-'+phase+'-isolation.json');q=read(p)
  assert q['pass_'] and q['positive_denial_control'] and all(x['returncode']==0 and not x['stderr'] for x in q['commands']);reports.append(dict(file=p.name,artifacts=q['exact_artifacts'],figures=q['figure_count'],sha256=sha(p)))
assert sum(q['artifacts'] for q in reports if '-after-' in q['file'])==r['post_cleanup_exact_report_artifacts']==17
assert r['retained_shared_bytes']==sum(q['bytes'] for q in r['remaining'])==1856251212
assert r['remaining_cap_headroom']==30000000000-r['retained_shared_bytes']==28143748788
assert r['protected_files_unchanged']==7347 and r['non_native_files_unchanged']==2620 and r['exhaustive_post_inventory_per_campaign']==1
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
old=read(D.parent/'STORAGE_GATE.json');newlower=r['retained_shared_bytes']+7*old['minimum_dense_retained_bytes_per_completed_role']+old['single_trajectory_reservation_bytes']
result=dict(pass_=True,review_time_utc=datetime.now(timezone.utc).isoformat(),reviewed_procedure=v['candidate_sha'],assessment_checkout=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),counts_exact=True,journal_paths_and_bytes_match_approved_manifests=True,journal_files=len(seen),journal_bytes=sum(seen.values()),journal_sha256=sha(R/'DELETED.jsonl'),representative_targets_absent=samples,runner_preservation_checks_pass=True,report_receipts=reports,shared_retained_bytes=r['retained_shared_bytes'],cap_headroom_bytes=r['remaining_cap_headroom'],updated_partial_eighth_role_reservation_lower_bound=newlower,full_eight_role_budget_proven=False,no_additional_exhaustive_native_scan=True,no_rerun_or_science_launched=True,monitoring_during_batch=0,completion_status_reads=1,token_credit_usage='unavailable',timing={'runner_monotonic_seconds':r['seconds'],'timestamp_span_seconds':(datetime.fromisoformat(s.get('completed_at',s['updated_at']))-datetime.fromisoformat(s['started_at'])).total_seconds(),'interpretation':'Different clock spans retained as reported; no performance/savings inference.'})
Path('/private/tmp/cleanup-completion-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('pass_','journal_files','journal_bytes','shared_retained_bytes','cap_headroom_bytes','updated_partial_eighth_role_reservation_lower_bound','full_eight_role_budget_proven')},indent=2))
