from pathlib import Path
import json,re,yaml,hashlib
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-two-stage-20260928');W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');D=W/'validation/rtc_two_stage_2026-09-28'
def read(p):return json.loads(p.read_text())
def y(p):return yaml.load(p.read_text(),Loader=yaml.CSafeLoader)
names=['B-REF-rep1','Q-REF','Q-C-core-plus','Q-C-core-minus','Q-C-shoulder'];attempts=[json.loads(s) for s in (R/'attempts.jsonl').read_text().splitlines()];runs=[];errors=[];apply=0;ptc_initial=0;ptc_feedback=0;feedback=0;fixed=[]
for name in names:
 p=R/name;e=read(p/'EXECUTION.json');l=read(p/'LAUNCH.json');assert e['status']=='completed' and e['completed_cycle']==5 and l['status']=='completed'
 for log in p.rglob('*.log'):
  for line in log.read_text(errors='replace').splitlines():
   if re.search(r'\[(error|critical)\]',line,re.I):errors.append(dict(path=str(log.relative_to(R)),line=line))
 for nw in range(1,5):
  a=y(p/f'upstream/products/network{nw}/receipt.yaml');apply+=int(a['Apply_performed']);ptc_initial+=len(y(p/f'upstream/products/network{nw}/donor-continuity/ptc/receipt.yaml')['segments'])
 for cycle in range(1,6):
  a=y(p/f'run/BASE-POINT/REF/a1100/cycle-{cycle:02d}/native/products/receipt.yaml');assert a['outcome']=='success';feedback+=1;ptc_feedback+=len(a['segments'])
 b=read(p/'freeze/budget.json');runs.append(dict(name=name,role='B' if name.startswith('B-') else 'Q',source=e['identity']['source_revision'],binary=e['identity']['binary'],seconds=l['exit_seconds'],peak_sampled_tree_RSS=l['aggregate_tree_sampled_RSS'],cpu_seconds=l['cpu_seconds'],readiness=l['readiness'],shared_retained_before_launch=b['campaign_retained_bytes'],single_trajectory_reservation=b['single_trajectory_reservation_bytes'],conservative_peak_reservation=b['campaign_retained_bytes']+b['single_trajectory_reservation_bytes']))
 if name.startswith('Q-'):
  for nw in range(1,5):
   obs=read(p/'inputs/observation.json');cfg=read(Path(next(q for q in obs['networks'] if read(Path(q['input']['path']))['network']==nw)['input']['path']));fixed.append(cfg['rtc_whole_run_quarantine']['sha256'])
assert len(set(fixed))==1 and set(fixed)=={'a8edbfecdc297e06fd8625fbd1b435ff1b43bae60de852043cac0fcda995f1c5'}
counts={k:sum(a['kind']==k for a in attempts) for k in ('parent','trajectory','feedback','endpoint')};assert counts==dict(parent=5,trajectory=5,feedback=25,endpoint=5);assert not errors
rec=dict(original_parents=counts['parent'],charged_trajectories=counts['trajectory'],feedback_attempts=counts['feedback'],feedback_completed=feedback,endpoints=counts['endpoint'],Q_trajectories=4,replacements=0,failed_launches=0,initial_network_Apply_completions=apply,initial_PTC_segment_solves=ptc_initial,feedback_PTC_segment_solves=ptc_feedback,unexpected_error_messages=errors,runs=runs,scope='fresh local original-input active RTC per trajectory; one cold M0 and fully resolved own-basis warm ALS per feedback; segment solves are not additional trajectory launches; Stage1 validation-only calls are not observations',prior_failed_campaign='da94a43795cde31c80b0f642ddde258d65905d3a retained separately, unchanged',cleanup_applied=False)
(R/'EXECUTION_ACCOUNTING.json').write_text(json.dumps(rec,indent=2)+'\n')
# Bring the already sealed historical response controls into the self-contained report.
old=W/'validation/rtc_step_consequences_2026-09-28/compact/report/response/source-responses.json';dest=R/'report/sealed-B-source-responses.json';dest.write_bytes(old.read_bytes());(R/'report/sealed-B-control-binding.json').write_text(json.dumps(dict(source=str(old),sha256=hashlib.sha256(old.read_bytes()).hexdigest(),compatibility='Stage1 bindings and new exact full B replay; no new B injection trajectories'),indent=2)+'\n')
print(json.dumps({k:v for k,v in rec.items() if k!='runs'},indent=2))
