"""Bounded POINT step diagnostic campaign. Existing native owners do all science."""
import json,time
from pathlib import Path
from inspect_forward_map import require,digest
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound
from point_processing import read
from point_rtc_experiment import prepare_run as rtc_prepare
LIMITS=dict(parent=5,trajectory=5,feedback=25,endpoint=5,rtc_only=2)

def consume(root,kind,identity,*,limits=LIMITS):
    require(kind in limits,'unknown step campaign operation')
    p=root/'attempts.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    limits=dict(limits);repair=root/'DEFECT-REPLACEMENT.json'
    if repair.exists():
        r=read(repair)
        require(r.get('implementation_defect') and r.get('corrected_source') and r.get('maximum_replacements')==1,'one documented corrected implementation defect only')
        for k in limits:limits[k]+=5 if k=='feedback' else 0 if k=='rtc_only' else 1
    require(sum(r['kind']==kind for r in rows)<limits[kind],'step campaign absolute ceiling')
    require(not any(r['kind']==kind and r['identity']==identity for r in rows),'duplicate step attempt')
    with p.open('a') as f:f.write(json.dumps(dict(kind=kind,identity=identity,epoch=time.time()))+'\n');f.flush()

def prepare_baseline(root,input_root):
    out=rtc_prepare(root,'B-REF-rep1',input_root,'B')
    p=read(out/'freeze/prospective.json');p['campaign_scope']='rtc-step-point-v1'
    obs=read(out/'inputs/observation.json')
    for e in obs['networks']:
        q=Path(e['input']['path']);cfg=read(q)
        cfg['rtc_source_probe']=dict(schema='point-rtc-source-probe-v1',authority='owner-RTC-cost-response-2026-09-27',branch='REF')
        save_json(q,cfg);e['input']=bound(q)
    save_json(out/'inputs/observation.json',obs)
    save_json(out/'freeze/generated-inputs.json',[bound(x) for x in sorted((out/'inputs').iterdir()) if x.is_file()])
    p['generated_inputs']=bound(out/'freeze/generated-inputs.json');save_json(out/'freeze/prospective.json',p)
    return out

def compare_baseline(out):
    """Sealed B science bytes, discrete plans, MAP/measurement and endpoint comparator."""
    import gzip
    from compare_point_pipeline import compare
    from point_rtc_diagnostics import yread
    from point_rank_experiment import REFERENCE
    repo=Path(__file__).resolve().parents[2];record=repo/'validation/rtc_cost_response_2026-09-27'
    ref=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-cost-response-20260927/B-REF-rep1')
    result=compare(out,ref,'a2000',read(REFERENCE)['reference_tolerances'],cycles=5,compare_endpoint=True)
    save_json(out/'REGRESSION.json',result)
    require(result['pass_'] and result['endpoint_exact'],'B numerical/endpoint regression')
    plan=json.loads(gzip.decompress((record/'NATIVE_REMOVALS.json.gz').read_bytes()))['files'];checks=[]
    for nw in (11,12):
        prefix=str(ref/f'upstream/products/network{nw}')+'/'
        old={r['path'][len(prefix):]:r['sha256'] for r in plan if r['path'].startswith(prefix) and Path(r['path']).suffix in ('.f64','.u8','.i64','.u16','.bits')}
        actual=out/f'upstream/products/network{nw}'
        paths=sorted(str(p.relative_to(actual)) for p in actual.rglob('*') if p.is_file() and p.suffix in ('.f64','.u8','.i64','.u16','.bits') and 'source-probe' not in p.parts)
        require(paths==sorted(old),'B native science inventory differs')
        checks += [dict(network=nw,path=rel,exact=digest(actual/rel)==old[rel]) for rel in paths]
        for name in ('events.yaml','event-decisions.yaml','support-decisions.yaml'):
            checks.append(dict(network=nw,field=name,exact=(actual/name).read_bytes()==(ref/f'upstream/products/network{nw}'/name).read_bytes()))
        a=yread(actual/'receipt.yaml');b=yread(ref/f'upstream/products/network{nw}/receipt.yaml')
        for key in ('candidate_count','event_count','admitted_jump_groups','transient_excluded_pair_cells'):
            checks.append(dict(network=nw,field=key,exact=a[key]==b[key]))
        for key in ('realized_detectors','speed_support_comparison','speed_support_authority'):
            checks.append(dict(network=nw,field=key,exact=a['donor_continuity'][key]==b['donor_continuity'][key]))
        aa=yread(actual/'jump-exclusions.yaml');bb=yread(ref/f'upstream/products/network{nw}/jump-exclusions.yaml')
        for q in (aa,bb):q.pop('processing_generation',None)
        checks.append(dict(network=nw,field='jump plan excluding run-specific generation identity',exact=aa==bb))
    save_json(out/'NATIVE_REGRESSION.json',dict(pass_=all(q['exact'] for q in checks),checks=checks,prior_removal_binding=bound(record/'NATIVE_REMOVALS.json.gz')))
    require(all(q['exact'] for q in checks),'B exact discrete/native regression')

def consume_consequences(root,kind,identity):
    return consume(root,kind,identity,limits=dict(parent=4,trajectory=4,feedback=20,endpoint=4,rtc_only=1))

def prepare_selected(root,input_root,branch='REF',noise_ref=None):
    return _prepare_selected(root,input_root,branch,noise_ref,False)

def prepare_consequences(root,input_root,branch='REF',noise_ref=None):
    return _prepare_selected(root,input_root,branch,noise_ref,True)

def _prepare_selected(root,input_root,branch,noise_ref,consequences):
    import shutil
    from point_pipeline_inputs import prepare
    from point_storage import tree_bytes
    from point_fruit_trajectory import verified
    repo=Path(__file__).resolve().parents[2];record=repo/('validation/rtc_step_consequences_2026-09-28' if consequences else 'validation/rtc_step_point_2026-09-27')
    observation=152430 if consequences else 152420;array='a1100' if consequences else 'a2000';family='N14' if consequences else 'S'
    scope='rtc-step-consequences-v1' if consequences else 'rtc-step-point-v1'
    selection=read(record/'SELECTION.json');require(selection['selected']==[observation,0,2],'frozen selected POINT changed')
    require(branch in ('REF','C-core-plus','C-core-minus','C-shoulder'),'undeclared step response')
    if consequences:require(selection['array']=='a1100' and selection['networks']==[1,2,3,4],'foreign four-network selection')
    refname=family+('-REF-rep2' if consequences and (root/'DEFECT-REPLACEMENT.json').exists() else '-REF-rep1')
    out=root/(refname if branch=='REF' else family+'-'+branch)
    prepare(out,verified(selection['reference']),input_root)
    p=read(out/'freeze/prospective.json');p.update(schema='citlali-point-speed-pca-v1',profile='POINT-SPEED-PCA-v1',campaign_scope=scope,rtc_family=family,signal_branch=branch,cycles=5,feedback_count=5,experimental_method='warm-observed-als',trajectory_identity=out.name,step_selection=bound(record/'SELECTION.json'),experiment_operands=[bound(record/'SELECTION.json')])
    p['scientific_recipe']['PTC']['method']='observed-entry ALS; own preceding converged basis initializer; fully resolved current residual'
    if branch!='REF':
        require(noise_ref is not None,'own selected REF required')
        rp=read(noise_ref/'freeze/prospective.json')
        require(rp['observation']==observation and rp['array']==array and rp['networks']==p['networks'] and rp['signal_branch']=='REF' and rp['rtc_family']==family,'foreign noise/reference')
        require(read(noise_ref/'EXECUTION.json')['status']=='completed','selected reference not completed')
        require(read(noise_ref/'SHAPES.json').get('available',False),'own prescribed injection support unavailable')
        shutil.copyfile(noise_ref/'freeze/measurement-geometry.json',out/'freeze/measurement-geometry.json')
        shutil.copytree(noise_ref/'freeze/shapes',out/'freeze/shapes')
        shutil.copyfile(noise_ref/'SHAPES.json',out/'SHAPES.json')
        p['experiment_operands'] += [bound(out/'freeze/measurement-geometry.json'),bound(out/'SHAPES.json')]+[bound(x) for x in sorted((out/'freeze/shapes').iterdir())]
        p['noise_reference']=dict(root=str(noise_ref),binding=bound(noise_ref/'freeze/prospective.json'),files=[bound(x) for x in sorted((noise_ref/f'freeze/{array}').iterdir()) if x.is_file()])
    obs=read(out/'inputs/observation.json')
    for e in obs['networks']:
        q=Path(e['input']['path']);cfg=read(q)
        cfg['rtc_source_probe']=dict(schema='point-rtc-source-probe-v1',authority='owner-RTC-step-consequences-2026-09-28' if consequences else 'owner-RTC-step-POINT-2026-09-27',branch=branch)
        if branch!='REF':cfg['rtc_source_probe']['raster']=bound(out/f'freeze/shapes/{branch}.fits')
        save_json(q,cfg);e['input']=bound(q)
    save_json(out/'inputs/observation.json',obs)
    save_json(out/'freeze/generated-inputs.json',[bound(x) for x in sorted((out/'inputs').iterdir()) if x.is_file()]);p['generated_inputs']=bound(out/'freeze/generated-inputs.json')
    budget=read(out/'freeze/budget.json');budget.update(campaign_root=str(root),campaign_retained_bytes=tree_bytes(root)-tree_bytes(out),peak_ceiling_bytes=30_000_000_000 if consequences else 20_000_000_000,metadata_reserve_bytes=100_000_000,maximum_branch_workers=1)
    if consequences:
        require(shutil.disk_usage(root).free>budget['peak_ceiling_bytes']+budget['free_space_reserve_bytes'],'four-network campaign free-space bound')
    save_json(out/'freeze/budget.json',budget);p['budget']=bound(out/'freeze/budget.json');save_json(out/'freeze/prospective.json',p)
    return out

def freeze_own_geometry(out,c):
    """Bind deterministic operands from this POINT's own M0, before any injection."""
    import numpy as np
    from point_response_cd import shapes
    from point_empirical_io import write_fits
    from point_empirical_io import exact_header
    p=read(out/'freeze/prospective.json')
    require((p['campaign_scope']=='rtc-step-point-v1' and p['observation']==152420 or p['campaign_scope']=='rtc-point-transfer-v1' and p['observation']==152420 and p['array']=='a1100' and p['networks']==[1,2,3,4] or p['campaign_scope']=='rtc-step-consequences-v1' and p['observation']==152430 and p['array']=='a1100' and p['networks']==[1,2,3,4]) and p['signal_branch']=='REF','own selected M0 only')
    require(not (out/'freeze/measurement-geometry.json').exists(),'geometry already frozen')
    f=c['fixed'];geometry=dict(header=c['header'],parameters=c['parameters'].tolist(),major=c['major'],weights=np.asarray(f.weights).tolist())
    if 'auxiliary' in c:geometry['auxiliary']=c['auxiliary']
    save_json(out/'freeze/measurement-geometry.json',geometry)
    from point_empirical import coordinates
    x,y=coordinates(c['support'].shape,c['header'])
    if not np.all(c['support'][np.hypot(x,y)<36]):
        save_json(out/'SHAPES.json',dict(observation=p['observation'],array=p['array'],networks=p['networks'],available=False,reason='prescribed injection overlaps unavailable support; reference trajectory still valid'))
        return
    rasters,metadata=shapes(c['parameters'],c['major'],c['header'],c['support'])
    dest=out/'freeze/shapes';dest.mkdir()
    for branch,raster in rasters.items():
        path=dest/f'{branch}.fits';write_fits(path,exact_header(c['header']),[('PRIMARY',raster,'mJy/nominal beam'),('VALID',c['support'].astype('u1'),'1')]);metadata[branch].update(bound(path))
    save_json(out/'SHAPES.json',dict(observation=p['observation'],array=p['array'],networks=p['networks'],available=True,reference_M0=bound(out/f"M0/products/{p['array']}/map.fits"),geometry=bound(out/'freeze/measurement-geometry.json'),rasters=metadata,recipe='existing point_response_cd.shapes; own unmodified M0 parameters/support; no relocation or normalization change'))


RETAINED_LIMITS=dict(parent=5,trajectory=5,feedback=25,endpoint=5)
def consume_retained(root,kind,identity):
    """New owner budget: no replacement, RTC-only or duplicate run identities."""
    require(kind in RETAINED_LIMITS,'operation outside retained-residual budget')
    require(not (root/'DEFECT-REPLACEMENT.json').exists() and not (root/'DEFECT-REPLACEMENTS.json').exists(),'no replacement allowance')
    path=root/'attempts.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    require(sum(r['kind']==kind for r in rows)<RETAINED_LIMITS[kind],'retained-residual absolute ceiling')
    require(not any(r['kind']==kind and r['identity']==identity for r in rows),'duplicate retained-residual attempt')
    with path.open('a') as f:f.write(json.dumps(dict(kind=kind,identity=identity,epoch=time.time()))+'\n');f.flush()


def prepare_retained_baseline(root,input_root):
    """One unchanged B replay. Original route, frozen old physical operands."""
    import gzip,shutil
    from point_pipeline_inputs import prepare
    from point_storage import tree_bytes
    repo=Path(__file__).resolve().parents[2];old=repo/'validation/rtc_step_consequences_2026-09-28';ref=old/'compact/N14-REF-rep2'
    require(not any(root.glob('*/LAUNCH.json')),'single B replay only')
    out=root/'B-REF-rep1';prepare(out,old/'inputs/reference.json',input_root)
    p=read(out/'freeze/prospective.json')
    require((p['observation'],p['subobservation'],p['scan'],p['array'],p['networks'])==(152430,0,2,'a1100',[1,2,3,4]),'foreign retained-residual target')
    p.update(schema='citlali-point-speed-pca-v1',profile='POINT-SPEED-PCA-v1',campaign_scope='rtc-retained-residuals-v1',rtc_family='B',signal_branch='REF',cycles=5,feedback_count=5,experimental_method='warm-observed-als',trajectory_identity=out.name)
    p['scientific_recipe']['PTC']['method']='observed-entry ALS; own preceding converged basis initializer; fully resolved current residual'
    shutil.copyfile(ref/'freeze/measurement-geometry.json',out/'freeze/measurement-geometry.json');shutil.copyfile(ref/'SHAPES.json',out/'SHAPES.json');shapes=out/'freeze/shapes';shapes.mkdir()
    for branch in ('C-core-plus','C-core-minus','C-shoulder'):
        dest=shapes/f'{branch}.fits';dest.write_bytes(gzip.decompress((ref/f'freeze/shapes/{branch}.fits.gz').read_bytes()))
        require(digest(dest)==read(ref/'SHAPES.json')['rasters'][branch]['sha256'],'prior raster identity changed')
    p['experiment_operands']=[bound(out/'freeze/measurement-geometry.json'),bound(out/'SHAPES.json')]+[bound(x) for x in sorted(shapes.iterdir())]
    obs=read(out/'inputs/observation.json')
    for entry in obs['networks']:
        q=Path(entry['input']['path']);cfg=read(q)
        # Preserve the accepted exact acquisition/array probe authority and gate.
        cfg['rtc_source_probe']=dict(schema='point-rtc-source-probe-v1',authority='owner-RTC-step-consequences-2026-09-28',branch='REF')
        save_json(q,cfg);entry['input']=bound(q)
    save_json(out/'inputs/observation.json',obs);save_json(out/'freeze/generated-inputs.json',[bound(x) for x in sorted((out/'inputs').iterdir()) if x.is_file()]);p['generated_inputs']=bound(out/'freeze/generated-inputs.json')
    budget=read(out/'freeze/budget.json');retained=tree_bytes(root)-tree_bytes(out);single=304*budget['native_cells']+2_000_000_000
    require(retained+single<=30_000_000_000,'next trajectory exceeds campaign peak ceiling')
    require(shutil.disk_usage(root).free>single+budget['free_space_reserve_bytes'],'next trajectory exceeds free-space reserve')
    budget.update(campaign_root=str(root),campaign_retained_bytes=retained,peak_ceiling_bytes=30_000_000_000,metadata_reserve_bytes=100_000_000,maximum_branch_workers=1,single_trajectory_reservation_bytes=single)
    save_json(out/'freeze/budget.json',budget);p['budget']=bound(out/'freeze/budget.json');save_json(out/'freeze/prospective.json',p)
    return out
