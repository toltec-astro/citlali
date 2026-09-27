"""Demonstrate reporting with OS-denied original inputs and run-array tree."""
import json,os,subprocess,tempfile,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];RECORD=Path(__file__).parent;COMPACT=RECORD/'compact'
ROOT='/Users/gwilson/work_toltec/local_data/citlali-validation/successor-point-speed-pca-20260927'
start=time.perf_counter();protected=json.loads((RECORD/'PROTECTED_INPUTS.json').read_text())
policy='(version 1) (allow default) (deny file-read* (subpath '+json.dumps(ROOT)+')) '+ ' '.join('(deny file-read* (literal '+json.dumps(r['path'])+'))' for r in protected)
env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-speed-mpl',XDG_CACHE_HOME='/private/tmp/point-speed-cache',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='point-speed-report-check-',dir='/private/tmp') as tmp:
 tmp=Path(tmp);policy_path=tmp/'read-boundary.sb';policy_path.write_text(policy)
 # Positive denial control demonstrates the policy really excludes the run tree.
 deny=subprocess.run(['/usr/bin/sandbox-exec','-f',str(policy_path),'/bin/cat',ROOT+'/PHASE-A.json'],capture_output=True,text=True)
 assert deny.returncode!=0 and 'Operation not permitted' in deny.stderr,deny.stderr
 cmds=[['/Users/gwilson/tolteca/bin/python',str(REPO/'tools/timestream_successor/report_point_speed_science.py'),'--root',str(COMPACT),'--retained',str(REPO/'validation/ptc_rank_response_2026-09-26/compact'),'--report-dir',str(tmp/'report')],['/Users/gwilson/tolteca/bin/python',str(REPO/'tools/timestream_successor/plot_point_speed.py'),'--root',str(COMPACT),'--figure-dir',str(tmp/'figures')]]
 results=[]
 for cmd in cmds:
  r=subprocess.run(['/usr/bin/sandbox-exec','-f',str(policy_path),*cmd],env=env,capture_output=True,text=True);assert r.returncode==0,r.stderr
  results.append(dict(command=cmd,returncode=r.returncode,stderr=r.stderr))
 for name in ('science-measurements','science-responses','science-plateaus','method-timing'):
  actual=json.loads((tmp/'report'/f'{name}.json').read_text());expected=json.loads((COMPACT/'report'/f'{name}.json').read_text());assert actual==expected,name
 figures=sorted(p.name for p in (tmp/'figures').glob('*.png'));assert len(figures)==4
record=dict(pass_=True,seconds=time.perf_counter()-start,denied_original_run_tree=ROOT,denied_protected_files=len(protected),positive_denial_control=True,machine_readable_science_records_exact=True,figure_count=4,commands=results,scope='saved map/measurement reporting only; no native processing or extra uncertainty endpoints; native basis diagnostics extracted and preserved before cleanup')
(RECORD/'REPORT_ISOLATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='commands'},indent=2))
