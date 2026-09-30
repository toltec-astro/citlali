"""Bounded independent POINT transfer preparation and accounting; no science here."""
import json,shutil,time
from pathlib import Path
from inspect_forward_map import require,digest
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound,verified
from point_storage import tree_bytes
REPO=Path(__file__).resolve().parents[2]
D=REPO/'validation/rtc_point_transfer_2026-09-28'
FAILED_ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-20260928')
FAILED_SCIENCE_ROOT=FAILED_ROOT.with_name('successor-rtc-point-transfer-build-repair-20260929')
FAILED_RUNTIME_ROOT=FAILED_ROOT.with_name('successor-rtc-point-transfer-runtime-repair-20260929')
ROOT=FAILED_ROOT.with_name('successor-rtc-point-transfer-completion-20260930')
PRIOR=[ROOT.parent/'successor-rtc-retained-residuals-20260928',ROOT.parent/'successor-rtc-two-stage-20260928',FAILED_ROOT,FAILED_SCIENCE_ROOT,FAILED_RUNTIME_ROOT]
BRANCHES=('REF','C-core-plus','C-core-minus','C-shoulder')
ROLES=tuple(f'{f}-{b}' for f in ('B','Q') for b in BRANCHES)
LIMITS=dict(parent=8,trajectory=8,feedback=40,endpoint=8)
CAP=30_000_000_000;ROLE_CAP=2_700_000_000;CAMPAIGN_METADATA=100_000_000;REPORT_CAP=1_000_000_000
CELLS=15_912_679;ACTIVE=304*CELLS+2_000_000_000

def read(p):return json.loads(Path(p).read_text())
def consume(root,kind,identity):
    require(kind in LIMITS,'foreign transfer operation; no RTC-only or automatic replacement')
    require(not any(root.glob('DEFECT-REPLACEMENT*')),'replacement requires diagnosed owner-resumed repair and a separately frozen batch')
    require(identity.get('run') in ROLES,'foreign transfer role')
    path=root/'attempts.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    require(sum(r['kind']==kind for r in rows)<LIMITS[kind],'transfer absolute attempt ceiling')
    require(not any(r['kind']==kind and r['identity']==identity for r in rows),'duplicate transfer attempt')
    if kind=='feedback':require(identity.get('cycle') in range(1,6) and identity.get('branch')=='REF','five native feedback applications only')
    with path.open('a') as f:f.write(json.dumps(dict(kind=kind,identity=identity,epoch=time.time()))+'\n');f.flush()

def capacity(root=ROOT):
    prior=[dict(path=str(p),bytes=tree_bytes(p)) for p in PRIOR]
    retained=sum(r['bytes'] for r in prior)
    peak=retained+7*ROLE_CAP+ACTIVE+CAMPAIGN_METADATA
    final=retained+8*ROLE_CAP+REPORT_CAP+CAMPAIGN_METADATA
    require(peak<=CAP and final<=CAP,'complete eight-role retention/reservation bound exceeds shared ceiling')
    require(shutil.disk_usage(root).free>CAP-retained+2_000_000_000,'complete campaign free-space reserve')
    return dict(prior=prior,prior_bytes=retained,native_cells=CELLS,ceiling_bytes=CAP,
        completed_role_maximum_bytes=ROLE_CAP,active_reservation_bytes=ACTIVE,
        campaign_metadata_maximum_bytes=CAMPAIGN_METADATA,compact_report_maximum_bytes=REPORT_CAP,
        peak_bound_bytes=peak,final_bound_bytes=final,free_space_reserve_bytes=2_000_000_000,
        qualification='Enforced admission budget, not a prediction or promise that every role will fit. Existing native reservation checks run before every stage. A completed role exceeding its cap ends the batch and preserves files; never delete to continue. No-applicable early exit is not assumed.')

def prepare_role(name):
    from point_pipeline_inputs import prepare
    require(name in ROLES,'foreign transfer role');family,branch=name.split('-',1)
    for previous in ROLES[:ROLES.index(name)]:require(read(ROOT/previous/'EXECUTION.json')['status']=='completed','earlier role incomplete')
    if name!='B-REF':require(read(ROOT/'selection/SELECTION.json')['newly_retained_runs']>0,'no applicable case stops seven roles')
    out=ROOT/name;prepare(out,D/'inputs/reference.json',Path('/Users/gwilson/work_toltec/local_data'))
    p=read(out/'freeze/prospective.json');p.update(schema='citlali-point-speed-pca-v1',profile='POINT-SPEED-PCA-v1',campaign_scope='rtc-point-transfer-v1',rtc_family=family,signal_branch=branch,cycles=5,feedback_count=5,experimental_method='warm-observed-als',trajectory_identity=name)
    p['scientific_recipe']['PTC']['method']='observed-entry ALS; own preceding converged basis initializer; fully resolved current residual'
    p['experiment_operands']=[bound(D/'SELECTOR_POLICY.md'),bound(D/'TARGET.json'),bound(D/'COMPLETION_AUTHORITY.md'),bound(D/'RECOVERY_ACCOUNTING.json')]
    if name!='B-REF':
        base=ROOT/'B-REF';require(read(base/'SHAPES.json')['available'],'own B prescribed injection support unavailable')
        for f in ('freeze/measurement-geometry.json','SHAPES.json'):shutil.copyfile(base/f,out/f)
        shutil.copytree(base/'freeze/shapes',out/'freeze/shapes')
        p['experiment_operands'] += [bound(out/'freeze/measurement-geometry.json'),bound(out/'SHAPES.json'),bound(ROOT/'selection/SELECTION.json')]+[bound(x) for x in sorted((out/'freeze/shapes').iterdir())]
    if branch!='REF':
        ref=ROOT/(family+'-REF');require(read(ref/'EXECUTION.json')['status']=='completed','own noise reference missing')
        p['noise_reference']=dict(root=str(ref),binding=bound(ref/'freeze/prospective.json'),files=[bound(x) for x in sorted((ref/'freeze/a1100').iterdir()) if x.is_file()])
    obs=read(out/'inputs/observation.json')
    for e in obs['networks']:
        path=Path(e['input']['path']);cfg=read(path)
        cfg['point_transfer_runtime_profile']='point-transfer-runtime-2026-09-29-v1'
        cfg['rtc_source_probe']=dict(schema='point-rtc-source-probe-v1',authority='owner-RTC-point-transfer-2026-09-28',branch=branch)
        if branch!='REF':cfg['rtc_source_probe']['raster']=bound(out/f'freeze/shapes/{branch}.fits')
        if family=='Q':cfg['rtc_point_run_selection']='rtc-point-physical-run-quarantine-v1'
        save_json(path,cfg);e['input']=bound(path)
    save_json(out/'inputs/observation.json',obs);save_json(out/'freeze/generated-inputs.json',[bound(x) for x in sorted((out/'inputs').iterdir()) if x.is_file()]);p['generated_inputs']=bound(out/'freeze/generated-inputs.json')
    budget=read(out/'freeze/budget.json');require(budget['native_cells']==CELLS,'bound native size changed')
    retained=sum(tree_bytes(x) for x in PRIOR)+tree_bytes(ROOT)-tree_bytes(out)
    require(retained+ACTIVE+CAMPAIGN_METADATA<=CAP,'next role shared capacity reservation')
    require(shutil.disk_usage(ROOT).free>ACTIVE+2_000_000_000,'next role free space')
    # Existing RunStorage enforces actual stages against both global and active ceilings.
    budget.update(campaign_root=str(ROOT),campaign_retained_bytes=retained,peak_ceiling_bytes=min(CAP,retained+ACTIVE),metadata_reserve_bytes=100_000_000,maximum_branch_workers=1,single_trajectory_reservation_bytes=ACTIVE)
    save_json(out/'freeze/budget.json',budget);p['budget']=bound(out/'freeze/budget.json');save_json(out/'freeze/prospective.json',p)
    return out


def compact_completed_role(out):
    """Release explicitly dead payloads only after the sole numerical child exits.

    M0/T5 native comparison inputs, source probes, filtered RTC, maps, receipts,
    fit/feedback products and warm bases remain. Removed derivatives are logged
    by exact digest; this compacted role is evidence, not a replay checkpoint.
    """
    require(out.parent==ROOT and out.name in ROLES,'foreign cleanup role')
    require(read(out/'freeze/prospective.json')['campaign_scope']=='rtc-point-transfer-v1','foreign cleanup scope')
    launch=read(out/'LAUNCH.json');execution=read(out/'EXECUTION.json')
    require(launch['status']=='completed' and launch['exit_code']==0 and execution['status']=='completed' and execution['completed_cycle']==5,
            'cleanup requires completed exited five-cycle child')
    targets=[];run=out/'run/BASE-POINT/REF/a1100'
    for k in range(1,6):
        products=run/f'cycle-{k:02d}/native/products'
        targets.append(products/'output-rows.u64')
        if k<5:targets.append(products/'operator-pixel.i32')
        for segment in products.glob('network-*-segment-*'):
            names=('operand.f64','model-support.u8') if k==5 else ('cleaned.f64','operand.f64','model-support.u8','causes.u8','injection.f64')
            targets.extend(segment/name for name in names)
    targets.append(out/'M0/products/a1100/output-rows.u64')
    for nw in range(1,5):
        donor=out/f'upstream/products/network{nw}/donor-continuity'
        targets.extend(donor.glob('*-conditioned.f64'))
        targets.extend((donor/'ptc').glob('segment-*-centered.f64'))
    files=[]
    for p in targets:
        require(not p.is_symlink() and p.resolve().is_relative_to(out.resolve()),'unsafe completed-role scratch path')
        if p.exists():
            require(p.is_file(),'scratch payload is not a file')
            files.append(dict(**bound(p),bytes=p.stat().st_size))
    manifest=out/'COMPLETED-SCRATCH-REMOVALS.json'
    require(not manifest.exists(),'completed cleanup already recorded')
    before=tree_bytes(out)
    save_json(manifest,dict(schema='point-transfer-completed-scratch-v1',source_revision=read(out/'freeze/prospective.json')['source_revision'],
        authority=bound(D/'COMPLETION_AUTHORITY.md'),exited_launch=bound(out/'LAUNCH.json'),completed_execution=bound(out/'EXECUTION.json'),
        reason='completed sole child; intermediate projection unused by M0/T5 response; conditioned/centered duplicates no longer consumed',
        removed_bytes=sum(r['bytes'] for r in files),before_bytes=before,files=files,
        preserved='M0/T5 operator and causes; T5 cleaned residual; original PTC input/eligibility/slots/cleaned; source-probe arrays; filtered RTC; all maps, receipts, bases, fits and logs',
        resumable_checkpoint=False))
    for r in files:
        p=Path(r['path']);require(p.stat().st_size==r['bytes'],'scratch changed after exited-child inventory');p.unlink()
    return dict(manifest=bound(manifest),removed_bytes=sum(r['bytes'] for r in files),after_bytes=tree_bytes(out))


def report_bytes(root):
    """Count each compact evidence directory once, including B's early export."""
    return tree_bytes(root/'report')+tree_bytes(root/'selection')
