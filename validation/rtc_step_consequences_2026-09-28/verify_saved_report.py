"""OS-denied original/scratch access; exact saved-only report regeneration."""
import json,os,subprocess,tempfile,time,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT='/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928';start=time.perf_counter();protected=json.loads((D/'PROTECTED_INPUTS.json').read_text())
# Coalesce prior-campaign trees; thousands of literal clauses exceed Seatbelt's
# jump serialization limit. Original dependencies retain individual denial.
prior=[str(D.parent/x) for x in ('rtc_cost_response_2026-09-27','rtc_fixed11_2026-09-27','rtc_step_point_2026-09-27')]
literals=[r['path'] for r in protected if not any(Path(r['path']).is_relative_to(q) for q in prior)]
policy='(version 1) (allow default) '+' '.join('(deny file-read* (subpath '+json.dumps(q)+'))' for q in [ROOT,*prior])+' '+' '.join('(deny file-read* (literal '+json.dumps(q)+'))' for q in literals)

env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/rtc-consequences-mpl',XDG_CACHE_HOME='/private/tmp/rtc-consequences-cache',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='step-saved-only-',dir='/private/tmp') as tmp:
 p=Path(tmp);sb=p/'deny.sb';sb.write_text(policy);out=p/'generated'
 denial=subprocess.run(['/usr/bin/sandbox-exec','-f',str(sb),'/bin/cat',ROOT+'/attempts.jsonl'],capture_output=True,text=True)
 assert denial.returncode and 'Operation not permitted' in denial.stderr,(denial.returncode,denial.stderr[:2000])
 cmds=[['/Users/gwilson/tolteca/bin/python',str(D/'report_step_campaign.py'),'--root',str(D/'compact'),'--output',str(out)]]
 for name in ('N14-REF-rep2',):cmds.append(['/Users/gwilson/tolteca/bin/python',str(D/'report_step_windows.py'),str(D/'compact/report'/name),str(out/name)])
 results=[]
 for cmd in cmds:
  r=subprocess.run(['/usr/bin/sandbox-exec','-f',str(sb),*cmd],env=env,capture_output=True,text=True);assert r.returncode==0,r.stderr;results.append(dict(script=Path(cmd[1]).name,returncode=r.returncode,stderr=r.stderr))
 expected=sorted(x.relative_to(D/'generated') for x in (D/'generated').rglob('*') if x.is_file());actual=sorted(x.relative_to(out) for x in out.rglob('*') if x.is_file());assert expected==actual
 for rel in expected:assert (D/'generated'/rel).read_bytes()==(out/rel).read_bytes(),str(rel)
record=dict(pass_=True,seconds=time.perf_counter()-start,denied_original_run_tree=ROOT,denied_protected_files=len(protected),positive_denial_control=True,exact_artifacts=len(expected),figure_count=sum(p.suffix=='.png' for p in expected),commands=results,scope='saved evidence summaries/plots only; no scientific fit, original input, learning, reduction or uncertainty ensemble')
(D/'REPORT_ISOLATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
