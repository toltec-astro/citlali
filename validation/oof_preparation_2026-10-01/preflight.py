"""Exact-source engineering gates, saved checks and metadata only. No reductions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/oof-mpl',XDG_CACHE_HOME='/private/tmp/oof-cache',PYTHONDONTWRITEBYTECODE='1')
import sys,subprocess,json,copy,shutil
from pathlib import Path
import yaml
W=Path(__file__).resolve().parents[2];D=Path(__file__).parent;sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_pipeline_inputs import prepare
from point_fruit_trajectory import bound
from point_additive_diagnostics import save_json
from point_storage import tree_bytes
from inspect_forward_map import require
from run_successor_observation import run as native
import point_quality_experiment as previous
L=Path.home()/'work_toltec/local_data';ROOT=L/'citlali-validation/successor-oof-cold-m0-20261001'
rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip();dest=ROOT/'preflight'/rev
require(not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'clean committed preflight source required')
dest.mkdir(parents=True,exist_ok=False);gates=[]
def gate(name,command):
    with (dest/(name+'.log')).open('w') as f:p=subprocess.run([str(x) for x in command],cwd=W,stdout=f,stderr=subprocess.STDOUT)
    gates.append(dict(name=name,exit_code=p.returncode,log=bound(dest/(name+'.log'))));save_json(dest/'GATES.json',dict(source_revision=rev,gates=gates,observation_starts=0))
    require(p.returncode==0,'failed gate '+name);print(name+' passed',flush=True)
gate('configure',['cmake','-S','.','-B','build','-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0'])
gate('build',['cmake','--build','build','--target','citlali_cli','citlali_point_frozen_apply','citlali_rtc_point_select','citlali_ptc_cost','-j','8'])
binary=W/'build/bin/citlali';require(rev[:9] in subprocess.check_output([str(binary),'--version'],text=True),'native version mismatch')
gate('ctest',['cmake','--build','build','--target','check','-j','8'])
gate('baseline-tools',[sys.executable,'-m','unittest','discover','-s','tools/baseline','-p','test_*.py'])
os.environ['POINT_FROZEN_APPLY_BINARY']=str(W/'build/bin/citlali_point_frozen_apply')
gate('point-tools',[sys.executable,'-m','pytest','-q',*sorted((W/'tools/timestream_successor').glob('test_point_*.py'))])
gate('oof-tools',[sys.executable,'-m','pytest','-q','tools/timestream_successor/test_oof_preparation.py'])
gate('config',[sys.executable,'tools/config/run_config_preflight.py','--require-all'])
gate('native-point-contract',[sys.executable,'tests/test_point_fruit_contract_cli.py','--binary',binary])
gate('lmtoof-saved-interface',[sys.executable,D/'saved_checks.py',dest/'saved-interface'])
prepared=[]
for obs in (152385,152386,152387):
    out=dest/str(obs);prepare(out,D/f'inputs/reference_{obs}.json',L)
    require(native(binary,out/'inputs/observation.json',out/'native-preparation',2,terminal='prepare')==0,'native OOF metadata preparation failed')
    receipt=out/'native-preparation/products/PREPARED.yaml';r=yaml.safe_load(receipt.read_text());log=(out/'native-preparation/run.log').read_text()
    require('[error]' not in log and r['mode']=='OOF' and r['scan']==1 and not r['reduction_started'] and not r['detector_signal_processed'],'native preparation contract')
    budget=json.loads((out/'freeze/budget.json').read_text());prepared.append(dict(observation=obs,receipt=bound(receipt),budget=budget,selected_detectors=sum(len(n['detectors']) for n in r['networks']),matched_detectors=sum(d['matched'] for n in r['networks'] for d in n['detectors'])))
# Negative actual-input metadata checks: deliberately malformed scope and detector identities.
base=dest/'152385/inputs/observation.json';original=json.loads(base.read_text());negatives=[]
for kind in ('foreign-scan','duplicate-channel','foreign-APT','POINT-source-probe'):
    folder=dest/'negative'/kind;folder.mkdir(parents=True);request=copy.deepcopy(original)
    if kind=='foreign-scan':request['scan']=2
    else:
        path=Path(request['networks'][0]['input']['path']);n=json.loads(path.read_text())
        if kind=='duplicate-channel':n['detectors'][1]['channel']=n['detectors'][0]['channel']
        elif kind=='POINT-source-probe':n['rtc_source_probe']={'authority':'owner-RTC-point-transfer-2026-09-28'}
        else:
            foreign=json.loads((dest/'152386/inputs/network1-input.json').read_text());n['manifest']=foreign['manifest']
        changed=folder/'network.json';save_json(changed,n);request['networks'][0]['input']=bound(changed)
    path=folder/'observation.json';save_json(path,request);code=native(binary,path,folder/'native',2,terminal='prepare')
    require(code!=0 and not list(folder.rglob('PREPARED.yaml')),'negative identity accepted')
    log=(folder/'native/run.log').read_text();expected={'foreign-scan':'OOF observation identity','duplicate-channel':'duplicate/foreign detector','foreign-APT':'shared_input_mismatch','POINT-source-probe':'POINT-only experiment'}[kind]
    require(expected in log,'negative failed for unrelated reason: '+kind)
    negatives.append(dict(kind=kind,exit_code=code,expected=expected,log=bound(folder/'native/run.log')))
prior=[dict(path=str(p),bytes=tree_bytes(p)) for p in [*previous.PRIOR,previous.ROOT]]
retained=sum(p['bytes'] for p in prior);cells=max(p['budget']['native_cells'] for p in prepared)
active=304*cells+2_700_000_000;peak=retained+5*2_700_000_000+active+1_100_000_000
save_json(dest/'PREPARATION.json',dict(source_revision=rev,prepared=prepared,negative_cases=negatives,storage=dict(prior=prior,prior_bytes=retained,max_native_cells=cells,active_S_bound_bytes=active,R_active_bound_status='unavailable until reference executor/population binding',six_role_S_based_lower_reservation_bytes=peak,ceiling_bytes=30_000_000_000,minimum_release_or_added_allowance_bytes=max(0,peak-30_000_000_000),free_reserve_bytes=2_000_000_000,current_free_bytes=shutil.disk_usage(ROOT).free,roles_serial=True,network_workers=2,library_threads=1),observation_starts=0,inversion_runs=0,launch_ready=False))
require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==rev and not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'source changed during validation')
save_json(dest/'PASSED.json',dict(status='engineering-and-interface-checks-passed',source_revision=rev,binary=bound(binary),gates=bound(dest/'GATES.json'),preparation=bound(dest/'PREPARATION.json'),saved_interface=bound(dest/'saved-interface/SAVED_INTERFACE.json'),launch_ready=False,observation_starts=0,inversion_runs=0))
print('all engineering preparation gates passed; launch prerequisites remain; no observation or inversion starts',flush=True)
