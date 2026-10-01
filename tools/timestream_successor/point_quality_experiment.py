"""October POINT accounting/preparation; the existing native B route owns science."""
import json,shutil,time
from pathlib import Path
from inspect_forward_map import require
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound
from point_storage import tree_bytes
from point_transfer_experiment import ROOT as OLD_ROOT,PRIOR as OLD_PRIOR,CAP,ACTIVE,ROLE_CAP,CELLS,read
REPO=Path(__file__).resolve().parents[2]
D=REPO/'validation/point_quality_2026-10-01'
ROOT=OLD_ROOT.with_name('successor-point-quality-20261001')
PRIOR=[*OLD_PRIOR,OLD_ROOT,OLD_ROOT.with_name('successor-point152420-amplitude-20260930')]
ROLES=('B-REF','B-C-core-plus','B-C-core-minus')
LIMITS=dict(parent=3,trajectory=3,feedback=15,endpoint=3)

def consume(root,kind,identity):
    require(kind in LIMITS and identity.get('run') in ROLES,'outside frozen three-role quality batch')
    if kind=='feedback':require(identity.get('cycle') in range(1,6) and identity.get('branch')=='REF','five feedback applications only')
    if kind=='trajectory':require(identity.get('cycle')==0 and identity.get('branch')=='REF','one cold trajectory per role')
    if kind=='endpoint':require(identity.get('cycle')==5,'T5 endpoint only')
    path=root/'attempts.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    require(sum(r['kind']==kind for r in rows)<LIMITS[kind],'quality operation ceiling')
    require(not any(r['kind']==kind and r['identity']==identity for r in rows),'duplicate attempt; no automatic retry')
    with path.open('a') as f:f.write(json.dumps(dict(kind=kind,identity=identity,epoch=time.time(),milestone='point-quality-2026-10-01'))+'\n');f.flush()

def capacity(root=ROOT):
    prior=[dict(path=str(p),bytes=tree_bytes(p)) for p in PRIOR]
    retained=sum(x['bytes'] for x in prior)
    peak=retained+2*ROLE_CAP+ACTIVE+1_100_000_000
    final=retained+3*ROLE_CAP+1_100_000_000
    require(max(peak,final)<=CAP,'all-role shared reservation exceeds30GB')
    require(shutil.disk_usage(root if root.exists() else root.parent).free>3*ROLE_CAP+ACTIVE+2_000_000_000,'free-space reserve')
    return dict(prior=prior,prior_bytes=retained,peak_bound_bytes=peak,final_bound_bytes=final,ceiling_bytes=CAP,active_reservation_bytes=ACTIVE,role_cap_bytes=ROLE_CAP,report_metadata_reserve_bytes=1_100_000_000,free_reserve_bytes=2_000_000_000,scope='Same shared predecessor campaign roots, retired amplitude preparation, and this campaign. Immutable input archive excluded; tracked compact evidence included conservatively through report reserve.')

def prepare_role(name):
    from point_pipeline_inputs import prepare
    require(name in ROLES,'unfrozen role')
    for prev in ROLES[:ROLES.index(name)]:require(read(ROOT/prev/'EXECUTION.json')['status']=='completed','previous role incomplete')
    branch=name.split('-',1)[1];out=ROOT/name
    prepare(out,REPO/'validation/rtc_point_transfer_2026-09-28/inputs/reference.json',Path('/Users/gwilson/work_toltec/local_data'))
    p=read(out/'freeze/prospective.json')
    p.update(schema='citlali-point-speed-pca-v1',profile='POINT-SPEED-PCA-v1',campaign_scope='rtc-point-transfer-v1',accounting_scope='point-map-quality-v1',rtc_family='B',signal_branch=branch,cycles=5,feedback_count=5,experimental_method='warm-observed-als',trajectory_identity=name)
    p['scientific_recipe']['PTC']['method']='observed-entry ALS; own preceding converged basis initializer; fully resolved current residual'
    p['experiment_operands']=[bound(D/'OWNER_DIRECTIVE.md'),bound(D/'MATRIX.json'),bound(ROOT/'FREEZE.json')]
    if branch!='REF':
        base=ROOT/'B-REF';require(read(base/'SHAPES.json')['available'],'compact support unavailable; no replacement shape')
        for f in ('freeze/measurement-geometry.json','SHAPES.json'):shutil.copyfile(base/f,out/f)
        shutil.copytree(base/'freeze/shapes',out/'freeze/shapes')
        p['experiment_operands'] += [bound(out/'freeze/measurement-geometry.json'),bound(out/'SHAPES.json')]+[bound(x) for x in sorted((out/'freeze/shapes').iterdir())]
        p['noise_reference']=dict(root=str(base),binding=bound(base/'freeze/prospective.json'),files=[bound(x) for x in sorted((base/'freeze/a1100').iterdir()) if x.is_file()])
    obs=read(out/'inputs/observation.json')
    for e in obs['networks']:
        path=Path(e['input']['path']);cfg=read(path)
        cfg['point_transfer_runtime_profile']='point-transfer-runtime-2026-09-29-v1'
        cfg['rtc_source_probe']=dict(schema='point-rtc-source-probe-v1',authority='owner-RTC-point-transfer-2026-09-28',branch=branch)
        if branch!='REF':cfg['rtc_source_probe']['raster']=bound(out/f'freeze/shapes/{branch}.fits')
        save_json(path,cfg);e['input']=bound(path)
    save_json(out/'inputs/observation.json',obs);save_json(out/'freeze/generated-inputs.json',[bound(x) for x in sorted((out/'inputs').iterdir()) if x.is_file()]);p['generated_inputs']=bound(out/'freeze/generated-inputs.json')
    budget=read(out/'freeze/budget.json');require(budget['native_cells']==CELLS,'native size changed')
    retained=sum(tree_bytes(x) for x in PRIOR)+tree_bytes(ROOT)-tree_bytes(out)
    require(retained+ACTIVE+1_100_000_000<=CAP,'next-role reservation exceeds shared ceiling')
    require(shutil.disk_usage(ROOT).free>ACTIVE+2_000_000_000,'next-role free space')
    budget.update(campaign_root=str(ROOT),campaign_retained_bytes=retained,peak_ceiling_bytes=min(CAP,retained+ACTIVE),metadata_reserve_bytes=100_000_000,maximum_branch_workers=1,single_trajectory_reservation_bytes=ACTIVE)
    save_json(out/'freeze/budget.json',budget);p['budget']=bound(out/'freeze/budget.json');save_json(out/'freeze/prospective.json',p)
    return out
