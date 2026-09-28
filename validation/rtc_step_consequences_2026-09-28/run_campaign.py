"""One explicit, budgeted branch; no implicit continuation or replacement."""
from pathlib import Path
import argparse,sys,json
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_step_experiment import prepare_consequences,consume_consequences
from point_speed_experiment import launch
from inspect_forward_map import require,digest
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('branch',choices=['REF','C-core-plus','C-core-minus','C-shoulder']);p.add_argument('--review',required=True,type=Path);a=p.parse_args()
 ROOT.mkdir(parents=True,exist_ok=True)
 out=prepare_consequences(ROOT,Path('/Users/gwilson/work_toltec/local_data'),a.branch,None if a.branch=='REF' else ROOT/'N14-REF-rep1')
 # Preserve original and earlier compact identities before entering science.
 protection=ROOT/'PROTECTED_INPUTS.json'
 if not protection.exists():
  import subprocess
  old=REPO/'validation/rtc_step_point_2026-09-27/PROTECTED_INPUTS.json'
  paths={r['path'] for r in json.loads(old.read_text())}
  paths.update(r['path'] for r in json.loads((out/'freeze/dependencies.json').read_text())['inputs'])
  paths.update(str(REPO/p) for p in subprocess.check_output(['git','ls-files','validation/rtc_step_point_2026-09-27'],cwd=REPO,text=True).splitlines())
  rows=[dict(path=p,sha256=digest(Path(p)),bytes=Path(p).stat().st_size) for p in sorted(paths)]
  protection.write_text(json.dumps(rows,indent=2)+'\n')
 r=launch(ROOT,out,REPO/'build/bin/citlali',a.review,'local cache not flushed; dependency verification may warm inputs; fresh scientific state',attempt_consumer=consume_consequences,child_mode='rtc-once')
 print(json.dumps(dict(run=out.name,status=r['status'],seconds=r['exit_seconds'])))
 require(r['status']=='completed','branch failed; inspect truthful guard/processing evidence before any further launch')
