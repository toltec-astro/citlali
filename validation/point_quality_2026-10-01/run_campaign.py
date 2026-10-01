#!/usr/bin/env python3
"""Foreground owner launch: three serial roles, no retries or observation monitoring."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-quality-mpl',XDG_CACHE_HOME='/private/tmp/point-quality-cache',PYTHONDONTWRITEBYTECODE='1')
import argparse,contextlib,datetime,json,subprocess,sys,traceback,re
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_quality_experiment import *
from point_fruit_trajectory import verified
from point_transfer_experiment import compact_completed_role
from point_speed_experiment import launch
from point_rtc_diagnostics import yread
from inspect_forward_map import digest
BINARY=REPO/'build/bin/citlali';STATE={}
def status(state,**fields):
    STATE.update(state=state,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
    p=ROOT/'STATUS.json';tmp=p.with_suffix('.pending');save_json(tmp,STATE);os.replace(tmp,p)
def accounting():
    rows=[json.loads(x) for x in (ROOT/'attempts.jsonl').read_text().splitlines()] if (ROOT/'attempts.jsonl').exists() else []
    counts={k:sum(x['kind']==k for x in rows) for k in LIMITS}
    receipts=[]
    for role in ROLES:
        for nw in range(1,5):
            p=ROOT/role/f'upstream/products/network{nw}/receipt.yaml'
            if (ROOT/role).exists():receipts.append(dict(role=role,network=nw,receipt=bound(p) if p.exists() else None,Apply_performed=yread(p).get('Apply_performed') if p.exists() else None))
    return dict(charged_attempts=counts,receipts=receipts,full_roles_charged=sum(r['kind']=='parent' and r['identity']['run']=='B-REF' for r in rows),compact_roles_charged=sum(r['kind']=='parent' and r['identity']['run']!='B-REF' for r in rows),subset_roles_charged=0,replacements=0,maximum_new_starts=14,frozen_batch_maximum=3,qualification='Charge before child start. Missing/partial receipts mean unknown work, not no work. Unused slots parked, not automatic authorization for extra roles.')
def verify_role(out,revision):
    require(read(out/'EXECUTION.json')['status']=='completed' and read(out/'EXECUTION.json')['completed_cycle']==5,'role endpoint failed')
    require(read(out/'freeze/prospective.json')['source_revision']==revision,'role source mismatch')
    for f in out.rglob('run.log'):require(not re.search(r'\[(?:error|critical)\]',f.read_text(),re.I),'unexpected error-level log '+str(f))
    for nw in range(1,5):
        r=yread(out/f'upstream/products/network{nw}/receipt.yaml');decision=yread(out/f'upstream/products/network{nw}/donor-continuity/reassessment-decision.yaml')
        require(r['Apply_performed'] and r['original_pair_unchanged'] and r['PTC']['iteration_limit']==500,'native B contract failed')
        require(decision['conditioned_grid_policy']=='rtc-conditioned-padded-2s-fixed-4s-v1','diagnostic policy changed')
        require(revision.startswith(r['source_revision']) or r['source_revision'].startswith(revision),'native revision changed')
    applications=[json.loads(x) for x in (out/'execution-attempts.jsonl').read_text().splitlines()]
    require(len(applications)==5 and {r['cycle'] for r in applications}==set(range(1,6)),'not exactly five feedback applications')
    require(tree_bytes(out)<=ROLE_CAP,'role retention exceeded; stop and preserve')
    require(sum(tree_bytes(p) for p in PRIOR)+tree_bytes(ROOT)<=CAP,'actual shared ceiling exceeded')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--review',type=Path,required=True);args=parser.parse_args()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'clean frozen source required')
    require(Path(sys.executable).resolve()==Path('/Users/gwilson/tolteca/bin/python').resolve(),'local venv required')
    freeze=read(ROOT/'FREEZE.json');rv=read(args.review)
    require(freeze['source_revision']==revision==rv['source_revision'] and rv['verdict']=='pass','exact independent source review required')
    require(tuple(freeze['roles'])==ROLES and freeze['limits']==LIMITS,'frozen roles changed')
    for b in freeze['bindings']:verified(b)
    require(verified(freeze['binary'])==BINARY and revision[:9] in subprocess.check_output([str(BINARY),'--version'],text=True),'exact native version required')
    require(not any((ROOT/r).exists() for r in ROLES),'existing role; no implicit resume/retry')
    with (ROOT/'LAUNCHED.json').open('x') as f:json.dump(dict(source_revision=revision,pid=os.getpid(),review=bound(args.review)),f)
    STATE.update(source_revision=revision,pid=os.getpid(),completed_roles=[],status_file=str(ROOT/'STATUS.json'),resume=str(D/'RESUME.md'))
    try:
        status('RUNNING',phase='storage admission');save_json(ROOT/'STORAGE_AT_LAUNCH.json',capacity());(ROOT/'logs').mkdir()
        for name in ROLES:
            require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==revision,'source moved during batch')
            require(digest(BINARY)==freeze['binary']['sha256'],'native binary moved during batch')
            status('RUNNING',phase='prepare',current_role=name);out=prepare_role(name)
            status('RUNNING',phase='numerical trajectory',current_log=str(out/'outer.log'))
            with (ROOT/'logs'/f'{name}.log').open('x') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                rec=launch(ROOT,out,BINARY,args.review,'OS cache unflushed; fresh originals/RTC/PTC/feedback; not repeated timing',attempt_consumer=consume,child_mode='rtc-once')
            require(rec['status']=='completed',name+' failed; no automatic retry')
            compact_completed_role(out,root=ROOT,roles=ROLES,authority=D/'OWNER_DIRECTIVE.md')
            verify_role(out,revision);STATE['completed_roles'].append(name);save_json(ROOT/'ACCOUNTING.json',accounting())
        status('RUNNING',phase='saved compact-response extraction',current_role=None)
        from point_rtc_response import response
        from point_transfer_report import response_network
        with (ROOT/'logs/response.log').open('x') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            response(ROOT,'B',ROOT/'report/response-B',reference_name='B-REF',branches=('C-core-plus','C-core-minus'),network_loader=response_network)
        require(tree_bytes(ROOT/'report')<=1_000_000_000,'report cap exceeded')
        require(accounting()['charged_attempts']==LIMITS,'final accounting mismatch')
        save_json(ROOT/'ACCOUNTING.json',accounting());status('COMPLETE_PENDING_OWNER_REVIEW',phase='completed',exit_code=0);return 0
    except BaseException as exc:
        # On interruption stop descendants before terminal status; never auto-resume.
        import psutil
        children=psutil.Process().children(recursive=True)
        for p in children:
            try:p.terminate()
            except psutil.Error:pass
        _,live=psutil.wait_procs(children,timeout=5)
        for p in live:
            try:p.kill()
            except psutil.Error:pass
        try:save_json(ROOT/'ACCOUNTING.json',accounting())
        except Exception as e:STATE['accounting_error']=str(e)
        (ROOT/'failure.log').write_text(traceback.format_exc());status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',phase='stopped',reason=str(exc),exit_code=1);return 1
if __name__=='__main__':raise SystemExit(main())
