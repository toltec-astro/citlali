#!/usr/bin/env python3
"""One foreground, sequential batch; no retries, cleanup, adoption or agent calls."""
import argparse,contextlib,datetime,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_transfer_experiment import *
from point_speed_experiment import launch
from point_rtc_diagnostics import yread
PYTHON=Path('/Users/gwilson/tolteca/bin/python');BINARY=REPO/'build/bin/citlali';SELECTOR=REPO/'build/bin/citlali_rtc_point_select'
STATE={}
class ProcessFailure(RuntimeError):pass
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def status(state,**fields):
    STATE.update(state=state,updated_utc=now(),**fields)
    p=ROOT/'STATUS.json';tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(STATE,indent=2)+'\n');os.replace(tmp,p)

def command(name,args):
    path=ROOT/'logs'/f'{name}.log';status('RUNNING',phase=name,current_log=str(path))
    with path.open('x') as f:
        child=subprocess.Popen([str(x) for x in args],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        status('RUNNING',child_pid=child.pid,child_command=[str(x) for x in args])
        try:code=child.wait()
        except BaseException:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    STATE.setdefault('gates',[]).append(dict(name=name,exit_code=code,log=str(path)))
    if code:raise ProcessFailure(f'{name} process failed with exit code {code}; see {path}')

def accounting():
    attempts=[json.loads(s) for s in (ROOT/'attempts.jsonl').read_text().splitlines()] if (ROOT/'attempts.jsonl').exists() else []
    counts={k:sum(r['kind']==k for r in attempts) for k in LIMITS};native=0;receipts=[]
    for role in ROLES:
        for n in range(1,5):
            p=ROOT/role/f'upstream/products/network{n}/receipt.yaml'
            if p.exists():
                r=yread(p);native+=bool(r.get('Apply_performed'));receipts.append(dict(role=role,network=n,Apply_performed=r.get('Apply_performed'),binding=bound(p)))
    return dict(charged_attempts=counts,completed_native_RTC_Apply_receipts=native,native_receipts=receipts,qualification='Missing failed/interrupted receipt is unknown application completion, never proof of no work. Parent/trajectory slot is charged before numerical child starts.')

def verify_role(role,revision):
    out=ROOT/role;require(read(out/'EXECUTION.json')['status']=='completed','role endpoint failed')
    p=read(out/'freeze/prospective.json');require(p['source_revision']==revision and p['cycles']==5,'source/cycle freeze changed')
    launchrec=read(out/'LAUNCH.json');require(launchrec['exit_code']==0,'role native child failed')
    for f in out.rglob('run.log'):
        require(not re.search(r'\[(?:error|critical)\]',f.read_text(),re.I),'unexpected scientific error-level log: '+str(f))
    for nw in range(1,5):
        r=yread(out/f'upstream/products/network{nw}/receipt.yaml');require(r['Apply_performed'] and r['original_pair_unchanged'],'native RTC endpoint/original pair failed')
        require(revision.startswith(r['source_revision']) or r['source_revision'].startswith(revision),'native source mismatch')
    require(tree_bytes(out)<=ROLE_CAP,'completed role exceeds frozen retention cap; stop and preserve')
    counts=accounting();save_json(ROOT/'ACCOUNTING.json',counts)
    require(sum(tree_bytes(x) for x in PRIOR)+tree_bytes(ROOT)<=CAP,'shared actual cap')

def main():
    global STATE
    a=argparse.ArgumentParser();a.add_argument('--review',type=Path,required=True);args=a.parse_args()
    review=read(args.review);revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require(review['source_revision']==revision and review['verdict']=='pass','exact-source independent preparation review required')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'clean exact source required')
    require(Path(sys.executable).resolve()==PYTHON.resolve(),'required local venv')
    ROOT.mkdir(parents=True,exist_ok=True)
    allowed={'STATUS.json','PREFLIGHT_REVIEW.json','READY.json','preflight-inputs','STORAGE_PREFLIGHT.json','HISTORICAL_CHECKS.json'}
    require(all(x.name in allowed for x in ROOT.iterdir()),'existing incompatible campaign files; no implicit resume')
    with (ROOT/'LAUNCHED.json').open('x') as f:json.dump(dict(source_revision=revision,started_utc=now(),review=bound(args.review)),f)
    (ROOT/'logs').mkdir();STATE=dict(campaign='rtc-point-transfer-20260928',source_revision=revision,pid=os.getpid(),started_utc=now(),completed_roles=[],current_role=None,status_file=str(ROOT/'STATUS.json'),resume=str(D/'RESUME.md'),summary=str(ROOT/'report/metrics.json'),model_monitoring_calls_after_handoff=0,token_credit_usage='unavailable; no measured savings claim')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
    os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-transfer-mpl',XDG_CACHE_HOME='/private/tmp/point-transfer-cache',PYTHONDONTWRITEBYTECODE='1')
    try:
        status('RUNNING',phase='storage and frozen source preflight')
        save_json(ROOT/'STORAGE_AT_LAUNCH.json',capacity())
        save_json(ROOT/'FREEZE.json',dict(source_revision=revision,policy=bound(D/'SELECTOR_POLICY.md'),target=bound(D/'TARGET.json'),inputs=[bound(x) for x in sorted((D/'inputs').iterdir()) if x.is_file()],roles=ROLES,limits=LIMITS,review=bound(args.review),environment='local AppleClang/Homebrew fallback; supplemental development evidence, not accepted Unity Spack reproduction'))
        command('build',['cmake','--build','build','--target','citlali_cli','citlali_rtc_point_select','-j','8'])
        require(revision[:9] in subprocess.check_output([str(BINARY),'--version'],text=True),'built native source mismatch')
        save_json(ROOT/'BINARY.json',dict(source_revision=revision,binary=bound(BINARY),selector=bound(SELECTOR)))
        command('ctest',['cmake','--build','build','--target','check','-j','8'])
        command('baseline-tools',[PYTHON,'-m','unittest','discover','-s','tools/baseline','-p','test_*.py'])
        command('point-tools',[PYTHON,'-m','pytest','-q',*sorted((REPO/'tools/timestream_successor').glob('test_point_*.py'))])
        command('config',[PYTHON,'tools/config/run_config_preflight.py','--require-all'])
        command('native-five-cycle-requests',[PYTHON,'tests/test_point_fruit_contract_cli.py','--binary',BINARY])
        command('historical-checks',[PYTHON,D/'historical_checks.py','--output',ROOT/'HISTORICAL_AT_LAUNCH.json'])
        command('runner-fixtures',[PYTHON,D/'test_transfer_preparation.py'])
        for role in ROLES:
            status('RUNNING',phase='prepare role',current_role=role,child_pid=None)
            require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==revision,'bound source moved')
            require(digest(BINARY)==read(ROOT/'BINARY.json')['binary']['sha256'] and digest(SELECTOR)==read(ROOT/'BINARY.json')['selector']['sha256'],'bound binary changed')
            out=prepare_role(role)
            with (ROOT/'logs'/f'{role}-launch.log').open('x') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                status('RUNNING',phase='numerical trajectory',current_log=str(out/'outer.log'))
                r=launch(ROOT,out,BINARY,args.review,'unflushed local cache; own original inputs and active RTC; no timing-repeat inference',attempt_consumer=consume,child_mode='rtc-once')
            if r['status']!='completed':
                er=read(out/'EXECUTION.json') if (out/'EXECUTION.json').exists() else {}
                if er.get('status')=='source_rejected':raise RuntimeError(role+' scientific source gate rejected; no retry')
                raise ProcessFailure(role+' numerical process failed; no retry; '+str(er.get('reason')))
            verify_role(role,revision);STATE['completed_roles'].append(role)
            if role=='B-REF':
                (ROOT/'selection').mkdir()
                for nw in range(1,5):command(f'original-selection-network{nw}',[SELECTOR,out/f'upstream/products/network{nw}',ROOT/f'selection/network{nw}.yaml'])
                with (ROOT/'logs/B-selection-summary.log').open('x') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                    from point_transfer_report import select_summary
                    n=select_summary()
                if not n:
                    from point_transfer_report import no_applicable_report
                    no_applicable_report()
                    require(tree_bytes(ROOT/'selection')<=REPORT_CAP,'no-case compact allowance')
                    status('COMPLETE_NO_APPLICABLE_CASE',phase='complete; seven remaining roles skipped',current_role=None,exit_code=0,completed_utc=now(),summary=str(ROOT/'selection/SELECTION.json'));return 0
            if role=='Q-REF':
                # Fail immediately if offline B and in-pass Q membership differ;
                # compare before any injected Q is allowed to start.
                wanted={(q['network'],q['detector'],q['physical_run']) for q in read(ROOT/'selection/SELECTION.json')['selected']};actual=set()
                for nw in range(1,5):
                    for q in yread(out/f'upstream/products/network{nw}/source-probe/run-selection.yaml')['population']:
                        if q['selected']:actual.add((nw,q['detector'],q['physical_run']))
                require(actual==wanted,'saved and active REF selector mismatch')
        status('RUNNING',phase='compact saved-product reports',current_role=None)
        with (ROOT/'logs/report.log').open('x') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            from point_transfer_report import compare
            compare()
        require(tree_bytes(ROOT/'report')+tree_bytes(ROOT/'selection')<=REPORT_CAP,'compact evidence allowance exceeded')
        require(sum(tree_bytes(x) for x in PRIOR)+tree_bytes(ROOT)<=CAP,'final shared cap')
        counts=accounting();require(counts['charged_attempts']==LIMITS and counts['completed_native_RTC_Apply_receipts']==32,'final exact execution accounting')
        save_json(ROOT/'ACCOUNTING.json',counts)
        status('COMPLETE_PENDING_OWNER_REVIEW',phase='complete',current_role=None,exit_code=0,completed_utc=now());return 0
    except KeyboardInterrupt:
        # Foreground Ctrl-C normally reaches the whole terminal process group.
        # Explicitly terminate any remaining descendants before publishing stop.
        import psutil
        children=psutil.Process().children(recursive=True)
        for p in children:
            try:p.terminate()
            except psutil.Error:pass
        _,live=psutil.wait_procs(children,timeout=5)
        for p in live:
            try:p.kill()
            except psutil.Error:pass
        save_json(ROOT/'ACCOUNTING.json',accounting());status('INTERRUPTED',exit_code=130,reason='Owner interruption; all outputs retained; no automatic resume');return 130
    except Exception as exc:
        traceback.print_exc(file=(ROOT/'failure.log').open('w'))
        save_json(ROOT/'ACCOUNTING.json',accounting());status('PROCESS_FAILED' if isinstance(exc,ProcessFailure) else 'REQUIRED_GATE_FAILED',exit_code=1,reason=str(exc),failure_log=str(ROOT/'failure.log'));return 1
if __name__=='__main__':raise SystemExit(main())
