"""Exact source build, fixtures, saved-sample replay/cache; zero observation Apply."""
from altaz_campaign import *
rev=revision();dest=ROOT/'preflight'/rev;dest.mkdir(parents=True,exist_ok=False);gates=[]
def gate(name,cmd):
    log=dest/(name+'.log')
    with log.open('w') as f:p=subprocess.run([str(x) for x in cmd],cwd=W,stdout=f,stderr=subprocess.STDOUT)
    gates.append(dict(name=name,exit_code=p.returncode,log=bound(log)))
    save_json(ROOT/'GATES.json',dict(source_revision=rev,gates=gates,complete=False,observational_accumulations=0))
    require(p.returncode==0,'failed gate '+name);print(name+' passed',flush=True)
gate('configure',['cmake','-S','.','-B','build','-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0'])
gate('build',['cmake','--build','build','--target','citlali_cli','citlali_jinc_diagnostic','-j','8'])
require(rev[:9] in subprocess.check_output([str(W/'build/bin/citlali'),'--version'],text=True),'CLI source differs')
gate('ctest',['cmake','--build','build','--target','check','-j','8'])
gate('baseline',[sys.executable,'-m','unittest','discover','-s','tools/baseline','-p','test_*.py'])
os.environ['POINT_FROZEN_APPLY_BINARY']=str(W/'build/bin/citlali_point_frozen_apply')
gate('point-tools',[sys.executable,'-m','pytest','-q',*sorted((W/'tools/timestream_successor').glob('test_point_*.py'))])
gate('oof-tools',[sys.executable,'-m','pytest','-q','tools/timestream_successor/test_oof_preparation.py',W/'validation/oof_jinc_2026-10-02/test_oof_jinc_campaign.py',D/'test_altaz.py'])
gate('config',[sys.executable,'tools/config/run_config_preflight.py','--require-all'])
gate('native-point-contract',[sys.executable,'tests/test_point_fruit_contract_cli.py','--binary',W/'build/bin/citlali'])
gate('saved-preparation',[sys.executable,D/'altaz_campaign.py','prepare'])
require(revision()==rev,'source changed during gates')
save_json(ROOT/'GATES.json',dict(source_revision=rev,gates=gates,complete=True,observational_accumulations=0))
print('Engineering gates and three native AltAz coordinate checks passed; independent review/freeze remain. No observational JINC accumulation.')
