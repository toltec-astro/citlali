"""Bounded RTC experiment orchestration. Existing native/pipeline owners do science."""
import gzip,json,os,shutil,subprocess,time
from pathlib import Path
from inspect_forward_map import require,digest
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound,verified
from point_processing import read
from point_pipeline_inputs import prepare
from point_storage import tree_bytes,RunStorage
from point_speed_experiment import launch,once as speed_once,compare_prefix
from point_rank_experiment import REFERENCE,OLD

REPO=Path(__file__).resolve().parents[2]
RECORD=REPO/'validation/rtc_cost_response_2026-09-27'
PRIOR=REPO/'validation/point_speed_pca_2026-09-27/compact/B-warm-rank10-N10-rep1'
LIMITS=dict(parent=15,trajectory=15,feedback=75,endpoint=15,rtc_only=4)
ABSOLUTE=dict(parent=17,trajectory=17,feedback=85,endpoint=17,rtc_only=4)
CEILING=20_000_000_000
BRANCHES=('C-core-plus','C-core-minus','C-shoulder')


def consume(root,kind,identity):
    require(kind in LIMITS,'unknown RTC operation')
    path=root/'attempts.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    limits=LIMITS;replacement=root/'DEFECT-REPLACEMENTS.json'
    if replacement.exists():
        repair=read(replacement)
        require(repair.get('implementation_defect') and repair.get('corrected_source') and
                0<repair.get('maximum_replacements',0)<=2,'replacement needs corrected implementation defect')
        limits={k:min(ABSOLUTE[k],v+repair['maximum_replacements']*(5 if k=='feedback' else 0 if k=='rtc_only' else 1)) for k,v in LIMITS.items()}
    require(sum(r['kind']==kind for r in rows)<limits[kind],'RTC absolute attempt ceiling')
    require(not any(r['kind']==kind and r['identity']==identity for r in rows),'duplicate RTC attempt')
    with path.open('a') as f:
        f.write(json.dumps(dict(kind=kind,identity=identity,epoch=time.time()))+'\n');f.flush()


def prepare_run(root,name,input_root,family,branch='REF',noise_ref=None):
    out=root/name;prepare(out,REFERENCE,input_root)
    p=read(out/'freeze/prospective.json')
    p.update(schema='citlali-point-speed-pca-v1',profile='POINT-SPEED-PCA-v1',
             campaign_scope='rtc-cost-response-v1',rtc_family=family,signal_branch=branch,
             cycles=5,feedback_count=5,experimental_method='warm-observed-als',trajectory_identity=name)
    p['scientific_recipe']['PTC']['method']='observed-entry ALS; own preceding converged basis initializer; fully resolved current residual'
    geometry=REPO/'validation/ptc_rank_response_2026-09-26/compact/operands/measurement-geometry.json'
    shutil.copyfile(geometry,out/'freeze/measurement-geometry.json')
    shapes=out/'freeze/shapes';shapes.mkdir()
    for b in BRANCHES:
        dest=shapes/f'{b}.fits';dest.write_bytes(gzip.decompress((OLD/f'freeze/shapes/{b}.fits.gz').read_bytes()))
        require(digest(dest)==read(OLD/'SHAPES.json')['rasters'][b]['sha256'],'retained physical sky changed')
    p['experiment_operands']=[bound(out/'freeze/measurement-geometry.json')]+[bound(f) for f in sorted(shapes.iterdir())]
    if branch!='REF':
        require(noise_ref is not None,'response requires own unperturbed REF')
        p['noise_reference']=dict(root=str(noise_ref),binding=bound(noise_ref/'freeze/prospective.json'),
            files=[bound(x) for x in sorted((noise_ref/'freeze/a2000').iterdir()) if x.is_file()])
        observation=read(out/'inputs/observation.json')
        for entry in observation['networks']:
            path=Path(entry['input']['path']);cfg=read(path)
            cfg['rtc_source_probe']=dict(schema='point-rtc-source-probe-v1',authority='owner-RTC-cost-response-2026-09-27',
                branch=branch,raster=bound(shapes/f'{branch}.fits'))
            save_json(path,cfg);entry['input']=bound(path)
        save_json(out/'inputs/observation.json',observation)
        save_json(out/'freeze/generated-inputs.json',[bound(x) for x in sorted((out/'inputs').iterdir()) if x.is_file()])
        p['generated_inputs']=bound(out/'freeze/generated-inputs.json')
    budget=read(out/'freeze/budget.json')
    budget.update(campaign_root=str(root),campaign_retained_bytes=tree_bytes(root)-tree_bytes(out),
                  peak_ceiling_bytes=CEILING,metadata_reserve_bytes=100_000_000,maximum_branch_workers=1)
    require(shutil.disk_usage(root).free>budget['shared_upstream_allowance']+budget['free_space_reserve_bytes'],'RTC peak free space')
    save_json(out/'freeze/budget.json',budget);p['budget']=bound(out/'freeze/budget.json')
    save_json(out/'freeze/prospective.json',p)
    return out


def once(out,binary,entry_ns):
    p=read(out/'freeze/prospective.json')
    require(p['campaign_scope'] in ('rtc-cost-response-v1','rtc-fixed11-v1','rtc-step-point-v1','rtc-step-consequences-v1') and p['feedback_count']==5 and
            p['cycles']==5 and p['ptc']==dict(method='observed-als',rank=10) and
            p['experimental_method']=='warm-observed-als','fixed RTC endpoint differs')
    code=speed_once(out,binary,entry_ns)
    try:RunStorage(out,read(out/'freeze/budget.json')).reconcile()
    except Exception as exc:
        rec=read(out/'EXECUTION.json');rec.update(status='storage_failed',reason=str(exc));save_json(out/'EXECUTION.json',rec);return 1
    return code


def run_family(root,binary,review,input_root,selection):
    root.mkdir(parents=True,exist_ok=True);spec=read(selection);family=spec['family']
    require(family in ('A','B','C') and spec.get('endpoint')=='warm-rank10-N5','undeclared RTC family')
    if family!='A':
        require(read(root/'FAMILY-A.json')['status']=='completed' and spec.get('measured_rationale'),'candidate lacks completed baseline/rationale')
        require(spec.get('changes') and len(spec['changes'])<=(2 if family=='B' else 1),'too many RTC changes')
    dest=root/f'FAMILY-{family}.json'
    resumed=None
    if dest.exists():
        repair=root/'COMPARISON_REPAIR.json'
        require(family=='A' and repair.is_file(),'family already attempted; no implicit retry')
        resumed=read(dest);fix=read(repair)
        require(resumed['status']=='failed' and resumed.get('reason')=='warmN5 numerical regression failed' and
                resumed['cells']==[dict(name='A-REF-rep1',branch='REF',rep=1,status='completed')] and
                fix['no_science_reexecution'] and fix['verdict']=='sealed-reference byte recovery',
                'only explicit completed-first-run comparison repair can resume')
        save_json(root/'FAMILY-A-original-failed.json',resumed)
    record=dict(status='running',selection=bound(selection),cells=[],peak_disk_ceiling_bytes=CEILING,
        resource_policy=dict(workloads=1,network_workers=2,feedback_workers=1,numerical_threads=1),
        cache='not flushed; first-use retained; preflight hashing can warm files')
    if resumed:record['comparison_repair']=bound(root/'COMPARISON_REPAIR.json')
    save_json(dest,record)
    try:
        ref=None
        for rep in (1,2,3):
            if resumed and rep==1:
                out=root/'A-REF-rep1';result=read(out/'LAUNCH.json')
            else:
                out=prepare_run(root,f'{family}-REF-rep{rep}',input_root,family)
                result=launch(root,out,binary,review,'first-use recorded; OS cache not flushed; original input/fresh state',attempt_consumer=consume,child_mode='rtc-once')
            record['cells'].append(dict(name=out.name,branch='REF',rep=rep,status=result['status']));save_json(dest,record)
            require(result['status']=='completed','required REF failed; no scientific replacement')
            reference=PRIOR if family=='A' and rep==1 else root/'A-REF-rep1' if family=='B' else ref
            if reference:
                if reference==PRIOR:
                    from compare_point_pipeline import compare
                    sealed=read_hashes()
                    comparison=compare(out,reference,'a2000',read(REFERENCE)['reference_tolerances'],cycles=5,compare_endpoint=False,retained_hashes=sealed,retained_products=RECORD/'prior-prefix')
                else:comparison=compare_prefix(out,reference,5)
                if resumed and rep==1:shutil.copyfile(out/'REGRESSION.json',out/'REGRESSION-original-failed.json')
                if reference!=PRIOR:require(comparison['endpoint_exact'],'own N5 endpoint uncertainty regression failed')
                save_json(out/'REGRESSION.json',comparison)
                require(comparison['pass_'],'warmN5 numerical regression failed')
            if family=='B':compare_rtc_exact(out,root/'A-REF-rep1')
            if rep==1:ref=out
        if family in ('A','C'):
            capture_reference(root,ref,binary,family)
            for branch in BRANCHES:
                out=prepare_run(root,f'{family}-{branch}',input_root,family,branch,ref)
                result=launch(root,out,binary,review,'response control; fresh originals with pre-RTC injection; not baseline timing',attempt_consumer=consume,child_mode='rtc-once')
                record['cells'].append(dict(name=out.name,branch=branch,status=result['status']));save_json(dest,record)
                require(result['status']=='completed','required response branch failed')
        record['status']='completed';save_json(dest,record);return 0
    except Exception as exc:
        record.update(status='failed',reason=str(exc));save_json(dest,record);raise


def read_hashes():
    rows=json.loads(gzip.decompress((REPO/'validation/point_speed_pca_2026-09-27/HASH_ONLY.json.gz').read_bytes()))['files']
    prefix=PRIOR.name+'/'
    return {r['path'][len(prefix):]:r['sha256'] for r in rows if r['path'].startswith(prefix)}


def capture_reference(root,ref,binary,family,*,attempt_consumer=None):
    """One separately counted RTC-only capture, outside REF timings."""
    from run_successor_observation import run
    dest=root/f'{family}-RTC-diagnostic';require(not dest.exists(),'no diagnostic retry')
    budget=read(ref/'freeze/budget.json')
    budget.update(campaign_retained_bytes=tree_bytes(root),peak_ceiling_bytes=CEILING)
    dest.mkdir();obs=read(ref/'inputs/observation.json')
    for e in obs['networks']:
        cfg=read(Path(e['input']['path']))
        cfg['rtc_source_probe']=dict(schema='point-rtc-source-probe-v1',authority='owner-RTC-cost-response-2026-09-27',branch='REF')
        path=dest/f"network{e['network']}.json";save_json(path,cfg);e['input']=bound(path)
    path=dest/'observation.json';save_json(path,obs)
    def invoke():
        (attempt_consumer or consume)(root,'rtc_only',dict(family=family,purpose='original-input native coordinate/calibration capture'))
        return run(binary,path,dest/'native',2,'rtc-only')
    try:code=accounted_capture(dest,budget,invoke)
    except Exception as exc:
        save_json(dest/'DIAGNOSTIC.json',dict(status='failed',reason=str(exc)))
        raise
    save_json(dest/'DIAGNOSTIC.json',dict(return_code=code,status='completed' if code==0 else 'failed',
        scope='extra RTC-only capture; no PTC or feedback; excluded from normal REF timing; all collection costs retained in RUN.json'))
    require(code==0,'RTC-only capture failed')
    checks=[]
    for nw in (11,12):
        a=dest/f'native/products/network{nw}/donor-continuity';b=ref/f'upstream/products/network{nw}/donor-continuity'
        for f in sorted(a.glob('*.f64')):
            same=digest(f)==digest(b/f.name);checks.append(dict(network=nw,file=f.name,exact=same));require(same,'diagnostic capture changed RTC output')
        for f in sorted([*a.glob('*.u8'),*a.glob('*.i64')]):
            same=digest(f)==digest(b/f.name);checks.append(dict(network=nw,file=f.name,exact=same));require(same,'diagnostic capture changed RTC support')
    save_json(dest/'EQUIVALENCE.json',checks)


def accounted_capture(dest,budget,invoke):
    """Apply the trajectory storage contract to an extra RTC-only execution."""
    storage=RunStorage(dest,budget);allowance=budget['shared_upstream_allowance']
    storage.reserve(allowance)
    try:return invoke()
    finally:
        storage.finish(dest/'native',allowance)
        storage.reconcile()


def compare_rtc_exact(actual,reference):
    from point_rtc_diagnostics import yread
    checks=[]
    for nw in (11,12):
        a=actual/f'upstream/products/network{nw}';b=reference/f'upstream/products/network{nw}'
        paths=sorted(p.relative_to(a) for p in a.rglob('*') if p.is_file() and p.suffix in ('.f64','.u8','.i64','.u16','.bits'))
        require(paths==sorted(p.relative_to(b) for p in b.rglob('*') if p.is_file() and p.suffix in ('.f64','.u8','.i64','.u16','.bits')),'native payload inventory differs')
        for rel in paths:
            same=digest(a/rel)==digest(b/rel);checks.append(dict(network=nw,path=str(rel),exact=same))
        aa=yread(a/'receipt.yaml');bb=yread(b/'receipt.yaml')
        # Exclude version/identity and measured elapsed time, never scientific values.
        for key in ('candidate_count','event_count','admitted_jump_groups','transient_excluded_pair_cells'):
            checks.append(dict(network=nw,field=key,exact=aa[key]==bb[key]))
        for key in ('realized_detectors','speed_support_comparison','speed_support_authority'):
            checks.append(dict(network=nw,field=key,exact=aa['donor_continuity'][key]==bb['donor_continuity'][key]))
        for file in ('events.yaml','event-decisions.yaml','support-decisions.yaml'):
            checks.append(dict(network=nw,field=file,exact=(a/file).read_bytes()==(b/file).read_bytes()))
    result=dict(pass_=all(x['exact'] for x in checks),checks=checks,scope='all native RTC/CAL/PTC binary payloads, realized filters/support and event decisions; source/version identities and timings differ')
    save_json(actual/'NATIVE_REGRESSION.json',result);require(result['pass_'],'B scientific equivalence failed')
