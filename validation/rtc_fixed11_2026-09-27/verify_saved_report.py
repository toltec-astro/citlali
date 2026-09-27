"""Demonstrate reporting with OS-denied original inputs and run-array tree."""
import json,os,subprocess,tempfile,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];RECORD=Path(__file__).parent;COMPACT=RECORD/'compact'
ROOT='/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-fixed11-20260927'
start=time.perf_counter();protected=json.loads((RECORD/'PROTECTED_INPUTS.json').read_text())
policy='(version 1) (allow default) (deny file-read* (subpath '+json.dumps(ROOT)+')) '+ ' '.join('(deny file-read* (literal '+json.dumps(r['path'])+'))' for r in protected)
env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/fixed11-mpl',XDG_CACHE_HOME='/private/tmp/fixed11-cache',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='rtc-report-check-',dir='/private/tmp') as tmp:
 tmp=Path(tmp);policy_path=tmp/'read-boundary.sb';policy_path.write_text(policy)
 # Positive denial control demonstrates the policy really excludes the run tree.
 deny=subprocess.run(['/usr/bin/sandbox-exec','-f',str(policy_path),'/bin/cat',ROOT+'/COMPARISON.json'],capture_output=True,text=True)
 assert deny.returncode!=0 and 'Operation not permitted' in deny.stderr,deny.stderr
 cmds=[['/Users/gwilson/tolteca/bin/python',str(REPO/'tools/timestream_successor/report_point_fixed11.py'),'--root',str(COMPACT),'--old',str(REPO/'validation/rtc_cost_response_2026-09-27/compact'),'--output',str(tmp/'report')]]
 results=[]
 for cmd in cmds:
  r=subprocess.run(['/usr/bin/sandbox-exec','-f',str(policy_path),*cmd],env=env,capture_output=True,text=True);assert r.returncode==0,r.stderr
  results.append(dict(command=cmd,returncode=r.returncode,stderr=r.stderr))
 for name in ('METRICS.md','effects.json','report-facts.json'):
  assert (tmp/'report'/name).read_bytes()==(RECORD/name).read_bytes(),name
 figures=sorted(p.name for p in (tmp/'report').glob('*.png'));assert len(figures)==6
 for name in figures:
  assert (tmp/'report'/name).read_bytes()==(RECORD/name).read_bytes(),name
record=dict(pass_=True,seconds=time.perf_counter()-start,denied_original_run_tree=ROOT,denied_protected_files=len(protected),positive_denial_control=True,machine_readable_science_records_exact=True,figure_count=6,commands=results,scope='saved map/measurement reporting only; no native processing or extra uncertainty endpoints; native RTC population/source diagnostics extracted and preserved before cleanup')
(RECORD/'REPORT_ISOLATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='commands'},indent=2))
