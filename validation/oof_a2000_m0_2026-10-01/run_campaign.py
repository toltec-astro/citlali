#!/usr/bin/env python3
"""Owner foreground, three S cold-M0 starts; no retry, automatic resume or R starts."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/oof2-mpl',XDG_CACHE_HOME='/private/tmp/oof2-cache',PYTHONDONTWRITEBYTECODE='1')
import argparse,json,subprocess,sys,shutil,time,traceback,re
from pathlib import Path
W=Path(__file__).resolve().parents[2];D=Path(__file__).parent;sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound,verified
from inspect_forward_map import require,digest,read_yaml
from point_storage import tree_bytes
from point_pipeline_inputs import prepare
from point_speed_experiment import launch
import point_quality_experiment as previous
L=Path.home()/'work_toltec/local_data';ROOT=L/'citlali-validation/successor-oof-a2000-m0-20261001'
PRIOR=[*previous.PRIOR,previous.ROOT,ROOT.with_name('successor-oof-cold-m0-20261001')]
ROLES=('S-152385','S-152386','S-152387');CAP=30_000_000_000;ROLE_CAP=2_000_000_000;META=200_000_000
BINARY=W/'build/bin/citlali'
def read(p):return json.loads(Path(p).read_text())
def capacity(cells,root=ROOT,prior=PRIOR):
    inventory=[dict(path=str(p),bytes=tree_bytes(p)) for p in prior];retained=sum(p['bytes'] for p in inventory)
    active=304*cells+2_700_000_000
    metadata=tree_bytes(root)-sum(tree_bytes(root/r) for r in ROLES)
    require(metadata<=META,'campaign metadata reservation exceeded')
    peak=retained+2*ROLE_CAP+active+META;final=retained+3*ROLE_CAP+META
    require(max(peak,final)<=CAP,'three-role reservation exceeds shared30GB ceiling')
    require(shutil.disk_usage(root).free>3*ROLE_CAP+active+2_000_000_000,'whole-batch free-space reserve')
    return dict(prior=inventory,prior_bytes=retained,max_native_cells=cells,active_bytes=active,role_retained_cap_bytes=ROLE_CAP,metadata_bytes=metadata,metadata_reserve_bytes=META,peak_bound_bytes=peak,final_bound_bytes=final,shared_ceiling_bytes=CAP,free_reserve_bytes=2_000_000_000,network_workers=2,numeric_threads=1,serial_roles=True)
def consume(root,kind,identity):
    require(kind in ('parent','trajectory') and identity.get('run') in ROLES,'outside three-S batch')
    require(kind!='trajectory' or identity==dict(run=identity['run'],branch='REF',cycle=0),'OOF cold M0 only')
    path=root/'attempts.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    require(not any(r['kind']==kind and r['identity']['run']==identity['run'] for r in rows),'role already charged; no retry')
    require(sum(r['kind']==kind for r in rows)<3,'three S start ceiling')
    with path.open('a') as f:f.write(json.dumps(dict(kind=kind,identity=identity,epoch=time.time()))+'\n');f.flush()
def verify_role(out,rev):
    e=read(out/'EXECUTION.json');require(e['status']=='completed' and e['completed_cycle']==0 and e['feedback_applications']==0,'cold M0 endpoint missing')
    require(not (out/'execution-attempts.jsonl').exists(),'unexpected feedback applications')
    for p in out.rglob('run.log'):require(not re.search(r'\[(?:error|critical)\]',p.read_text(),re.I),'unexpected error-level message')
    for nw in (11,12):
        r=read_yaml(out/f'upstream/products/network{nw}/receipt.yaml');ptc=read_yaml(out/f'upstream/products/network{nw}/donor-continuity/ptc/receipt.yaml')
        require(r['source_revision'].startswith(rev[:9]) and r['original_pair_unchanged'],'native source/pair identity')
        require(ptc['rank']==3 and ptc['failed_fits']==0 and all(x['converged'] for x in ptc['segments']),'PTC rank/convergence differs')
    require((out/'M0/products/a2000/map.fits').is_file(),'missing full native map')
def compact(out):
    # Same exited-child scratch policy as POINT, narrowed to cold-M0 duplicates.
    require(out.parent==ROOT and out.name in ROLES,'foreign cleanup root')
    require(read(out/'LAUNCH.json')['status']=='completed' and read(out/'EXECUTION.json')['status']=='completed','active or failed role cleanup forbidden')
    targets=[]
    for nw in (11,12):
        donor=out/f'upstream/products/network{nw}/donor-continuity'
        targets+=list(donor.glob('*-conditioned.f64'))+list((donor/'ptc').glob('segment-*-centered.f64'))
    kept=[bound(p) for p in sorted(out.rglob('*')) if p.is_file() and p not in targets]
    removed=[]
    for p in targets:
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(out.resolve()),'unsafe scratch path')
        removed.append(dict(**bound(p),bytes=p.stat().st_size))
    manifest=out/'COMPLETED-SCRATCH-REMOVALS.json';require(not manifest.exists(),'no repeated cleanup')
    save_json(manifest,dict(authority=bound(D/'OWNER_DIRECTIVE.md'),removed=removed,preserved=kept,reason='conditioned/centered duplicates after successful exited M0; filtered RTC, CAL, PTC masks/inputs/cleaned/bases/receipts, AST/operator and full MAP retained',resumable_checkpoint=False))
    for b in removed:verified(b).unlink()
    for b in kept:verified(b)
    require(tree_bytes(out)<=ROLE_CAP,'retained role exceeds2GB; stop before next start')
def prepare_role(name,freeze):
    out=ROOT/name;prepare(out,D/f'inputs/reference_{name[2:]}.json',L)
    budget=read(out/'freeze/budget.json');require(budget['native_cells']<=freeze['storage']['max_native_cells'],'native dimensions changed')
    retained=sum(tree_bytes(p) for p in PRIOR)+tree_bytes(ROOT)-tree_bytes(out)
    active=freeze['storage']['active_bytes'];require(retained+active+META<=CAP,'next-role shared reservation')
    require(shutil.disk_usage(ROOT).free>active+2_000_000_000,'next-role free reserve')
    budget.update(campaign_root=str(ROOT),campaign_retained_bytes=retained,peak_ceiling_bytes=min(CAP,retained+active),metadata_reserve_bytes=100_000_000)
    save_json(out/'freeze/budget.json',budget);p=read(out/'freeze/prospective.json');p['budget']=bound(out/'freeze/budget.json');save_json(out/'freeze/prospective.json',p)
    return out

def prelaunch(review):
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'clean frozen source required')
    require(Path(sys.executable).resolve()==Path('/Users/gwilson/tolteca/bin/python').resolve(),'local venv required')
    f=read(ROOT/'FREEZE.json');rv=read(review)
    require(f['source_revision']==rev==rv['source_revision'] and rv['verdict']=='pass','exact-source freeze/review required')
    require(tuple(f['roles'])==ROLES and f['new_R_starts']==0 and f['new_S_starts']==3,'matrix changed')
    for b in f['bindings']:verified(b)
    require(verified(f['binary'])==BINARY and rev[:9] in subprocess.check_output([str(BINARY),'--version'],text=True),'native version differs')
    require(not (ROOT/'LAUNCHED.json').exists() and not any((ROOT/r).exists() for r in ROLES),'batch already attempted; no retry')
    storage=capacity(f['storage']['max_native_cells']);return rev,f,storage

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    rev,f,storage=prelaunch(a.review)
    if a.check_only:print(json.dumps(dict(status='launch-preflight-passed',observation_starts=0,storage=storage),indent=2));return 0
    with (ROOT/'LAUNCHED.json').open('x') as out:json.dump(dict(source_revision=rev,pid=os.getpid(),review=bound(a.review)),out)
    state=dict(source_revision=rev,pid=os.getpid(),completed_roles=[])
    def status(s,**kw):
        state.update(state=s,**kw);save_json(ROOT/'STATUS.pending',state);os.replace(ROOT/'STATUS.pending',ROOT/'STATUS.json')
    try:
        save_json(ROOT/'STORAGE_AT_LAUNCH.json',storage);status('RUNNING')
        for name in ROLES:
            require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==rev and not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'source changed during batch')
            require(digest(BINARY)==f['binary']['sha256'],'binary changed during batch')
            status('RUNNING',current_role=name);out=prepare_role(name,f)
            rec=launch(ROOT,out,BINARY,a.review,'fresh independent acquisition; OS cache unflushed',attempt_consumer=consume,child_mode='forward')
            require(rec['status']=='completed',name+' failed; no retry')
            verify_role(out,rev);compact(out);state['completed_roles'].append(name)
            require(sum(tree_bytes(p) for p in PRIOR)+tree_bytes(ROOT)<=CAP,'actual shared ceiling')
        status('COMPLETE_PENDING_OWNER_REVIEW',current_role=None,exit_code=0);print('Batch completed. Tell Codex done for saved-product assessment.');return 0
    except BaseException as exc:
        import psutil
        children=psutil.Process().children(recursive=True)
        for child in children:
            try:child.terminate()
            except psutil.Error:pass
        _,alive=psutil.wait_procs(children,timeout=5)
        for child in alive:
            try:child.kill()
            except psutil.Error:pass
        (ROOT/'failure.log').write_text(traceback.format_exc());status('FAILED',reason=str(exc),exit_code=1);return 1
if __name__=='__main__':raise SystemExit(main())
