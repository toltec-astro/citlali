"""One fixed-applicability B/C comparison; existing pipeline owns all science."""
import gzip,json,shutil,time
from pathlib import Path
from inspect_forward_map import require,digest
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound,verified
from point_processing import read
from point_rtc_experiment import prepare_run as rtc_prepare,BRANCHES,compare_rtc_exact,capture_reference
from point_speed_experiment import launch
from compare_point_pipeline import compare
from point_rank_experiment import REFERENCE
REPO=Path(__file__).resolve().parents[2]
RECORD=REPO/'validation/rtc_fixed11_2026-09-27'
OLD_RECORD=REPO/'validation/rtc_cost_response_2026-09-27'
OLD_ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-cost-response-20260927')
LIMITS=dict(parent=9,trajectory=9,feedback=45,endpoint=9,rtc_only=2)
ORDER=[(f,i) for i in (1,2,3) for f in ('B','C')]

def consume(root,kind,identity):
    require(kind in LIMITS,'unknown fixed11 operation');p=root/'attempts.jsonl'
    rows=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    limits=dict(LIMITS);replacement=root/'DEFECT-REPLACEMENT.json'
    if replacement.exists():
        r=read(replacement);require(r.get('implementation_defect') and r.get('corrected_source') and r.get('maximum_replacements')==1,'one documented corrected implementation-defect replacement only')
        for k in limits:limits[k]+=5 if k=='feedback' else 0 if k=='rtc_only' else 1
    require(sum(r['kind']==kind for r in rows)<limits[kind],'fixed11 absolute attempt ceiling')
    require(not any(r['kind']==kind and r['identity']==identity for r in rows),'duplicate fixed11 attempt')
    with p.open('a') as f:f.write(json.dumps(dict(kind=kind,identity=identity,epoch=time.time()))+'\n');f.flush()

def configure(cfg,selection,design,family):
    require(family in ('B','C'),'unknown arm')
    nw=cfg['network'];net=next(n for n in selection['networks'] if n['network']==nw)
    require(len(net['detectors'])==len(cfg['detectors']),'full population differs')
    for key,b in net['input_bindings'].items():require(cfg[key]['sha256']==b['sha256'],'fixed input differs '+key)
    for entry,d in zip(cfg['detectors'],net['detectors']):
        require(entry['channel']==d['channel'],'fixed detector order differs')
        entry['filter']='w1-t3' if family=='C' and d['disposition']=='selected' else 'lowpass'
    cfg['fixed11']=dict(arm='B-control' if family=='B' else 'C-fixed11',applicability=bound(RECORD/'APPLICABILITY.json'))
    if family=='C':
        cfg['finite_notch_identity']=design['finite_notch_identity'];cfg['finite_notch']=design['finite_notch']
    return cfg

def prepare_run(root,family,rep,input_root,branch='REF',noise_ref=None):
    out=rtc_prepare(root,f'{family}-REF-rep{rep}' if branch=='REF' else f'{family}-{branch}',input_root,family,branch,noise_ref)
    p=read(out/'freeze/prospective.json');p['campaign_scope']='rtc-fixed11-v1'
    selection=read(RECORD/'APPLICABILITY.json');design=read(verified(selection['design']))
    p['fixed11_applicability']=bound(RECORD/'APPLICABILITY.json');p['experiment_operands'] += [p['fixed11_applicability'],selection['design']]
    obs=read(out/'inputs/observation.json')
    for e in obs['networks']:
        path=Path(e['input']['path']);save_json(path,configure(read(path),selection,design,family));e['input']=bound(path)
    save_json(out/'inputs/observation.json',obs)
    save_json(out/'freeze/generated-inputs.json',[bound(x) for x in sorted((out/'inputs').iterdir()) if x.is_file()]);p['generated_inputs']=bound(out/'freeze/generated-inputs.json');save_json(out/'freeze/prospective.json',p)
    return out

def compare_control(out):
    """Exact prior B binary digests; old metadata/FITS remain sealed and intact."""
    reference=OLD_ROOT/'B-REF-rep1';oldmeta={r['source']:r['source_sha256'] for r in read(OLD_RECORD/'COMPACT_MANIFEST.json')['files']}
    oldmeta.update({r['path']:r['sha256'] for r in json.loads(gzip.decompress((OLD_RECORD/'HASH_ONLY.json.gz').read_bytes()))['files']})
    def verify_meta(rel):
        p=reference/rel;require(digest(p)==oldmeta['B-REF-rep1/'+str(rel)],'prior reference changed '+str(rel))
    route=Path('run/BASE-POINT/REF/a2000')
    refs=[Path('M0/products/a2000/map.fits'),route/'SUMMARY.json']
    refs += [Path(f'freeze/a2000/{n}.npz') for n in ('context','C0','BASE')]
    for k in range(6):
        folder=route/('initial' if k==0 else f'cycle-{k:02d}');refs.append(folder/'candidate.fits')
        if k:refs.append(folder/'native/products/total.fits')
    for rel in refs:verify_meta(rel)
    result=compare(out,reference,'a2000',read(REFERENCE)['reference_tolerances'],cycles=5,compare_endpoint=True)
    save_json(out/'REGRESSION.json',result);require(result['pass_'] and result['endpoint_exact'],'new B differs from completed B')
    plan=json.loads(gzip.decompress((OLD_RECORD/'NATIVE_REMOVALS.json.gz').read_bytes()))['files'];checks=[]
    for nw in (11,12):
        prefix=str(reference/f'upstream/products/network{nw}')+'/'
        old={r['path'][len(prefix):]:r['sha256'] for r in plan if r['path'].startswith(prefix) and Path(r['path']).suffix in ('.f64','.u8','.i64','.u16','.bits')}
        actual=out/f'upstream/products/network{nw}';paths=sorted(str(p.relative_to(actual)) for p in actual.rglob('*') if p.is_file() and p.suffix in ('.f64','.u8','.i64','.u16','.bits'))
        require(paths==sorted(old),'B native inventory differs')
        checks += [dict(network=nw,path=rel,exact=digest(actual/rel)==old[rel]) for rel in paths]
        from point_rtc_diagnostics import yread
        rel=Path(f'upstream/products/network{nw}')
        for name in ('receipt.yaml','events.yaml','event-decisions.yaml','support-decisions.yaml'):verify_meta(rel/name)
        a=yread(actual/'receipt.yaml');b=yread(reference/rel/'receipt.yaml')
        for key in ('candidate_count','event_count','admitted_jump_groups','transient_excluded_pair_cells'):checks.append(dict(network=nw,field=key,exact=a[key]==b[key]))
        for key in ('realized_detectors','speed_support_comparison','speed_support_authority'):checks.append(dict(network=nw,field=key,exact=a['donor_continuity'][key]==b['donor_continuity'][key]))
        for name in ('events.yaml','event-decisions.yaml','support-decisions.yaml'):checks.append(dict(network=nw,field=name,exact=(actual/name).read_bytes()==(reference/rel/name).read_bytes()))
    rec=dict(pass_=all(x['exact'] for x in checks),checks=checks,prior_plan=bound(OLD_RECORD/'NATIVE_REMOVALS.json.gz'),prior_B_equivalence=bound(OLD_RECORD/'compact/B-REF-rep1/NATIVE_REGRESSION.json'),inherited_response=bound(OLD_RECORD/'compact/report/A/source-responses.json'))
    save_json(out/'NATIVE_REGRESSION.json',rec);require(rec['pass_'],'new B exact native control failed')

def run(root,binary,review,input_root):
    root.mkdir(parents=True,exist_ok=True);dest=root/'COMPARISON.json';require(not dest.exists(),'comparison already attempted; no implicit retry')
    selection=bound(RECORD/'APPLICABILITY.json');rv=read(review);require(rv.get('applicability_sha256')==selection['sha256'],'independent preflight must bind frozen membership')
    record=dict(status='running',applicability=selection,order=[f'{f}{i}' for f,i in ORDER],runs=[],limits=LIMITS,resource_policy=dict(workloads=1,network_workers=2,feedback_workers=1,numerical_threads=1),peak_disk_ceiling_bytes=20_000_000_000);save_json(dest,record)
    try:
        for family,rep in ORDER:
            out=prepare_run(root,family,rep,input_root)
            result=launch(root,out,binary,review,'first-use retained; OS caches not flushed; original inputs/fresh scientific state',attempt_consumer=consume,child_mode='rtc-once')
            record['runs'].append(dict(name=out.name,status=result['status']));save_json(dest,record);require(result['status']=='completed','required trajectory failed; no scientific rescue')
            if family=='B' and rep==1:compare_control(out)
            elif rep>1:
                ref=root/f'{family}-REF-rep1';comparison=compare(out,ref,'a2000',read(REFERENCE)['reference_tolerances'],cycles=5,compare_endpoint=True)
                save_json(out/'REGRESSION.json',comparison);require(comparison['pass_'] and comparison['endpoint_exact'],'own-arm deterministic repeat differs');compare_rtc_exact(out,ref)
        for family in ('B','C'):capture_reference(root,root/f'{family}-REF-rep1',binary,family,attempt_consumer=consume)
        for branch in BRANCHES:
            out=prepare_run(root,'C',1,input_root,branch,root/'C-REF-rep1')
            result=launch(root,out,binary,review,'untimed C pre-RTC x-only response; same fixed applicability, other learning active',attempt_consumer=consume,child_mode='rtc-once')
            record['runs'].append(dict(name=out.name,status=result['status']));save_json(dest,record);require(result['status']=='completed','required source-response failed')
        record['status']='completed';save_json(dest,record);return 0
    except Exception as exc:
        record.update(status='failed',reason=str(exc));save_json(dest,record);raise
