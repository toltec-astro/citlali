"""Nonobservational exact-source gates and real-input preparation. Never launches science."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-quality-mpl',XDG_CACHE_HOME='/private/tmp/point-quality-cache',PYTHONDONTWRITEBYTECODE='1')
import sys,subprocess,json
from pathlib import Path
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W/'tools/timestream_successor'))
import point_quality_experiment as q
from point_fruit_trajectory import bound
from point_additive_diagnostics import save_json
from inspect_forward_map import require
rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip();dest=q.ROOT/'preflight'/rev
require(not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'clean preflight source')
dest.mkdir(parents=True);rows=[]
def run(name,command):
 with (dest/(name+'.log')).open('w') as f:r=subprocess.run([str(x) for x in command],cwd=W,stdout=f,stderr=subprocess.STDOUT)
 rows.append(dict(name=name,exit_code=r.returncode,log=bound(dest/(name+'.log'))));save_json(dest/'GATES.json',dict(source_revision=rev,gates=rows,new_observation_starts=0))
 require(r.returncode==0,'failed gate '+name);print(name+' passed',flush=True)
run('configure',['cmake','-S','.', '-B','build','-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0'])
run('build',['cmake','--build','build','--target','citlali_cli','citlali_point_frozen_apply','citlali_rtc_point_select','citlali_ptc_cost','-j','8'])
require(rev[:9] in subprocess.check_output([str(W/'build/bin/citlali'),'--version'],text=True),'exact native source version')
run('ctest',['cmake','--build','build','--target','check','-j','8'])
run('baseline-tools',[sys.executable,'-m','unittest','discover','-s','tools/baseline','-p','test_*.py'])
os.environ['POINT_FROZEN_APPLY_BINARY']=str(W/'build/bin/citlali_point_frozen_apply')
run('point-tools',[sys.executable,'-m','pytest','-q',*sorted((W/'tools/timestream_successor').glob('test_point_*.py'))])
run('config',[sys.executable,'tools/config/run_config_preflight.py','--require-all'])
run('native-five-cycle-contract',[sys.executable,'tests/test_point_fruit_contract_cli.py','--binary',W/'build/bin/citlali'])
run('transfer-fixtures',[sys.executable,W/'validation/rtc_point_transfer_2026-09-28/test_transfer_preparation.py'])
run('quality-fixtures',[sys.executable,Path(__file__).with_name('test_quality.py')])
# Exercise actual newB preparation including canonical original-input hashes,
# without any native parent/trajectory/feedback or attempt consumption.
q.ROOT=dest/'real-input-preparation';q.ROOT.mkdir();save_json(q.ROOT/'FREEZE.json',dict(preflight_only=True,source_revision=rev))
out=q.prepare_role('B-REF');p=q.read(out/'freeze/prospective.json')
require(p['source_revision']==rev and p['observation']==152420 and p['feedback_count']==5 and p['ptc']['rank']==10,'prepared B identity')
require(not (q.ROOT/'attempts.jsonl').exists(),'preflight charged observation work')
save_json(dest/'INPUT_PREPARATION.json',dict(prospective=bound(out/'freeze/prospective.json'),dependencies=bound(out/'freeze/dependencies.json'),generated_inputs=bound(out/'freeze/generated-inputs.json'),new_observation_starts=0))
save_json(dest/'PASSED.json',dict(source_revision=rev,binary=bound(W/'build/bin/citlali'),gates=bound(dest/'GATES.json'),inputs=bound(dest/'INPUT_PREPARATION.json'),status='pass',new_observation_starts=0));print('all gates and actual input preparation passed; no observation starts',flush=True)
