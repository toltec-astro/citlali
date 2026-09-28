"""One explicit authorized replay; candidate is blocked pending population audit."""
from pathlib import Path
import argparse,sys,json,subprocess
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_step_experiment import prepare_retained_baseline,consume_retained
from point_speed_experiment import launch
from inspect_forward_map import require,digest
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-retained-residuals-20260928')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('branch',choices=['B-REF']);p.add_argument('--review',required=True,type=Path);a=p.parse_args()
 ROOT.mkdir(parents=True,exist_ok=True);out=prepare_retained_baseline(ROOT,Path('/Users/gwilson/work_toltec/local_data'))
 old=REPO/'validation/rtc_step_consequences_2026-09-28'
 paths={r['path'] for r in json.loads((old/'PROTECTED_INPUTS.json').read_text())}
 paths.update(r['path'] for r in json.loads((out/'freeze/dependencies.json').read_text())['inputs'])
 paths.update(str(REPO/p) for p in subprocess.check_output(['git','ls-files',str(old.relative_to(REPO))],cwd=REPO,text=True).splitlines())
 (ROOT/'PROTECTED_INPUTS.json').write_text(json.dumps([dict(path=p,sha256=digest(Path(p)),bytes=Path(p).stat().st_size) for p in sorted(paths)],indent=2)+'\n')
 r=launch(ROOT,out,REPO/'build/bin/citlali',a.review,'local cache not flushed; original inputs and fresh scientific state; no timing-repeat inference',attempt_consumer=consume_retained,child_mode='rtc-once')
 print(json.dumps(dict(run=out.name,status=r['status'],seconds=r['exit_seconds'])))
 require(r['status']=='completed','B replay failed; no replacement; comparison blocked')
