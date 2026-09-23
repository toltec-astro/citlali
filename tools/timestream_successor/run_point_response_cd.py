#!/usr/bin/env python3
"""Freeze or execute the owner-prescribed POINT152389 C+D finite experiment."""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from inspect_forward_map import require, digest, read_yaml
from point_additive_diagnostics import save_json, verify_companion
from point_fruit_trajectory import (bound, verified, manifest, verify_manifest, FixedMeasurements,
                                    candidate_guard, load, selection_difference)
from point_fruit_diagnostics import patch_map, compare_zero, learned_state
from point_response_cd import BRANCHES, shapes, infer, seed_candidate, save_candidate, processing_guard
from point_empirical_io import write_fits
from run_point_fruit_ten import original_parents, ready_state, ARRAYS
from run_successor_observation import run as invoke

BASE=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs')
FIRST=BASE/'successor-point-152389-fruit-one-cycle-20260923'
TEN=BASE/'successor-point-152389-fruit-ten-cycles-20260923'
AB=BASE/'successor-point-152389-post-fruit-ab-20260923'
EXPECTED={FIRST/'cycle':'b98267bac1cecfd9000ed11446e71ba3472ea2b072443fee59e86d54dea7e006',
          FIRST/'diagnostics-r2':'15f7469c14aa8324258727ff67cf74f7bd3f4174f9cf3669c036555f2386fa77',
          TEN:'8f699bf69ded92e98fde671e49daee059f0141128c20c88ac5047f8d2dd6b320',
          AB:'8f8d21469ba82e8a5357bcd13eaaff07a51f41d0c257eac782776356615ebebc'}


def json_read(p):return json.loads(Path(p).read_text())


def source():
    repo=Path(__file__).resolve().parents[2]
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),'clean source required')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()


def recover(verify=True):
    if verify:
        for root,expected in EXPECTED.items():verify_manifest(root,expected)
    provenance=json_read(FIRST/'cycle/provenance.json');req=provenance['request']
    roots={n:Path(req[n]) for n in ('maps','admission','additive')}
    hashes={a:digest(roots['maps']/a/'map.fits') for a in ARRAYS}
    require(hashes==provenance['maps'],'original M0 changed')
    if verify:
        for key in ('admission','additive'):
            require(verify_companion(roots[key],hashes)['manifest_sha256']==provenance[key+'_manifest'],key+' changed')
    verified(provenance['measurement'])
    contexts={a:FixedMeasurements(*(roots[n]/a for n in ('maps','admission','additive'))) for a in ARRAYS}
    return roots,contexts


def original_cycle(array,k,roots):
    if k==0:
        return dict(total=bound(roots['maps']/array/'map.fits'),candidate=bound(roots['admission']/array/'candidate-reference.fits'))
    if k==1:
        m=json_read(FIRST/f'diagnostics-r2/{array}/measurements.json')['measurements']['T1']
        return dict(total=bound(FIRST/f'cycle/reference/{array}/products/total.fits'),
                    candidate=bound(FIRST/f'diagnostics-r2/{array}/m1.fits'),measurement=m)
    m=json_read(TEN/f'run/{array}/cycle-{k:02d}/diagnostics/measurements.json')
    verified(m['total']);verified(m['candidate'])
    return {n:m[n] for n in ('total','candidate','measurement')}


def freeze(out):
    started=time.perf_counter();revision=source()
    require(not out.exists(),'fresh campaign directory required')
    roots,contexts=recover();out.mkdir(parents=True);f=out/'freeze';f.mkdir()
    owner=dict(identity='POINT-152389-CD-OWNER-2026-09-23',outside_disk_no_feedback=True,
        direct_identical_operand_rejoin=True,gain=1,nonzero_cycles=10,
        schema='point-cd-prospective-v1',source_revision=revision,branches=list(BRANCHES),
        run_order=['REF','C-core-plus','C-core-minus','C-shoulder','D-seed-low','D-seed-high','D-seed-shoulder','D-threshold-5'],
        workers=3,numerical_threads=1,randomness='none; deterministic finite interventions',
        ratio_floor_native_units=1e-8,reference_tolerances=dict(signal_atol=1e-9,signal_rtol=1e-11,numerator_atol=1e-7,
            fit_atol_rtol=1e-8,deterministic_measurements_candidates='exact',support_normalization='exact'),
        accounting_atol=1e-8,parents={},injections={},seeds={},original_cycles={},geometry={},
        immutable_roots={k:str(v) for k,v in roots.items()},
        ancestors=[dict(root=str(p),manifest=bound(p/'FILES_SHA256SUMS')) for p in EXPECTED],
        directive=bound(Path(__file__).resolve().parents[2]/'validation/point_152389_response_robustness_cd_2026-09-23/OWNER_DIRECTIVE.md'))
    for a,fixed in contexts.items():
        d=f/a;d.mkdir();owner['parents'][a]=original_parents(FIRST,a)
        owner['original_cycles'][a]=[original_cycle(a,k,roots) for k in range(11)]
        for c in owner['original_cycles'][a]:
            s,v,_,_=patch_map(verified(c['total']));fresh=fixed.measure(s,v)
            if 'measurement' in c:
                old=c['measurement']
                require(np.array_equal(fresh['fit']['parameters'],old['fit']['parameters']) and
                    fresh['frozen_template_projection']==old['frozen_template_projection'] and fresh['apertures']==old['apertures'],
                    'saved measurement definitions changed')
            c['measurement']=fresh
        p=owner['original_cycles'][a][10]['measurement']['fit']['parameters']
        original,h=load(roots['admission']/a/'candidate-reference.fits');candidate_guard(original,h)
        raster,info=shapes(p,fixed.major,h,original['VALID'].astype(bool))
        owner['geometry'][a]=info;owner['injections'][a]={};owner['seeds'][a]={}
        for name,values in raster.items():
            path=d/(name+'.fits')
            write_fits(path,h,[('INJECTION',values,'mJy/nominal beam'),('VALID',original['VALID'],'1')],companion='POINT-CD-INPUT')
            owner['injections'][a][name]=bound(path)
        for branch in BRANCHES[4:7]:
            c=seed_candidate(original,branch,raster['C-shoulder']);candidate_guard(c,h)
            path=d/(branch+'.fits')
            write_fits(path,h,[(n,c[n],'mJy/nominal beam' if n=='CANDIDATE' else '1') for n in ('CANDIDATE','ADMITTED','VALID','RATIO')],companion='POINT-CD-SEED')
            owner['seeds'][a][branch]=bound(path)
        fit,c=infer(fixed,fixed.original,fixed.support,5.)
        require(c is not None and c['summary']['status']=='retained','threshold5 initial candidate unavailable')
        path=d/'D-threshold-5.fits';save_candidate(path,h,c);candidate_guard(load(path)[0],h)
        owner['seeds'][a]['D-threshold-5']=bound(path)
    owner['freeze_seconds']=time.perf_counter()-started
    save_json(f/'prospective.json',owner);manifest(f)
    save_json(out/'FREEZE.json',dict(prospective=bound(f/'prospective.json'),manifest=bound(f/'FILES_SHA256SUMS')))
    print(json.dumps(dict(frozen=str(out),seconds=owner['freeze_seconds'])),flush=True)


def campaign_request(owner_path,owner,array,branch,k,candidate,previous=None):
    require(branch in BRANCHES and 0<=k<=10,'campaign branch/cycle bound')
    require(k>0 or branch=='REF' or branch.startswith('C-'),'D must reuse M0')
    request=dict(schema='citlali-point-fruit-cd-v1',method='point-152389-response-robustness-cd-v1',
        array=array,branch=branch,cycle_index=k,mode='zero' if k==0 else 'reference',
        candidate=bound(candidate),owner_binding=bound(owner_path),**owner['parents'][array])
    if k>0:request['previous_state']=bound(previous)
    if branch.startswith('C-'):request['injection']=owner['injections'][array][branch]
    return request


def replay_check(original,products,measurement,candidate,beam):
    result=compare_zero(verified(original['total']),products/'total.fits',beam)
    expected=original['measurement']
    require(np.array_equal(measurement['fit']['parameters'],expected['fit']['parameters']), 'deterministic REF fit differs')
    require(measurement['frozen_template_projection']==expected['frozen_template_projection'] and
        measurement['apertures']==expected['apertures'],'deterministic REF measurement differs')
    old,_=load(verified(original['candidate']));new,_=load(candidate)
    for name in ('CANDIDATE','ADMITTED','VALID','RATIO'):
        require(np.array_equal(old[name],new[name],equal_nan=True),'deterministic REF candidate differs: '+name)
    result.update(measurements_candidates_exact=True)
    return result


def execute_branch(out,binary,owner_path,owner,roots,fixed,array,branch):
    started=time.perf_counter();root=out/'run'/branch/array;root.mkdir(parents=True,exist_ok=False)
    history=[];prior=None;parents=owner['parents'][array];threshold=5. if branch=='D-threshold-5' else 3.
    if branch.startswith('D-'):
        c=owner['seeds'][array][branch];verified(c)
        initial=owner['original_cycles'][array][0]
        prior=ready_state(array,0,verified(c),verified(initial['total']),verified(parents['parent_plan']),parents,True)
        prior.update(branch=branch,owner_sha256=digest(owner_path),measurement=initial['measurement'],reused_M0=True)
        save_json(root/'state-00.json',prior);prior['state_path']=str(root/'state-00.json');history.append(prior)
    for k in range(1 if prior else 0,11):
        if prior is not None and not prior['ready']:break
        cycle=root/f'cycle-{k:02d}';cycle.mkdir();tic=time.perf_counter()
        candidate=verified(prior['candidate']) if prior else verified(owner['original_cycles'][array][0]['candidate'])
        request=campaign_request(owner_path,owner,array,branch,k,candidate,Path(prior['state_path']) if prior else None)
        save_json(cycle/'request.json',request)
        rec=dict(array=array,branch=branch,cycle_index=k,ready=False,request=bound(cycle/'request.json'),threshold=threshold)
        try:
            rc=invoke(binary,cycle/'request.json',cycle/'native',1,'point-fruit')
            require(rc==0,'native cycle failed; inspect preserved log')
            products=cycle/'native/products';receipt=read_yaml(products/'receipt.yaml')
            signal,support,_,h=patch_map(products/'total.fits');dt=time.perf_counter()
            measurement=fixed.measure(signal,support);fit,c=infer(fixed,signal,support,threshold)
            rec.update(measurement=measurement,total=bound(products/'total.fits'),receipt=bound(products/'receipt.yaml'))
            if c is not None:
                path=cycle/f'm{k}.fits';save_candidate(path,h,c)
                rec.update(candidate=bound(path),candidate_summary=c['summary'])
            save_json(cycle/'cycle.json',rec)  # preserve failed measurements before continuation guards
            rec['guards']=processing_guard(products,receipt)
            require(c is not None and c['summary']['status']=='retained','unusable fit/candidate')
            candidate_guard(load(path)[0],h)
            rec['PTC_segments']=learned_state(products,receipt)
            if branch=='REF':rec['replay']=replay_check(owner['original_cycles'][array][k],products,measurement,path,fixed.beam)
            if prior:
                old,_=load(verified(prior['candidate']));new,_=load(path)
                rec['candidate_turnover']=selection_difference(new,old)
                restore,rv,_,_=patch_map(products/'restored.fits');valid=rv&old['VALID'].astype(bool)
                err=float(np.max(abs((restore-old['CANDIDATE'])[valid])))
                require(err<1e-8,'applied operand remap mismatch')
                rec['guards']['remapped_operand_max_abs']=err
            state=ready_state(array,k,path,products/'total.fits',products/'receipt.yaml',parents,True)
            state.update(branch=branch,owner_sha256=digest(owner_path))
            save_json(cycle/'state.json',state)
            rec.update(ready=True,state_path=str(cycle/'state.json'),diagnostic_seconds=time.perf_counter()-dt)
        except (ValueError,OSError,KeyError,RuntimeError) as exc:
            rec.update(ready=False,stop_reason=str(exc))
        rec['wall_seconds']=time.perf_counter()-tic;save_json(cycle/'cycle.json',rec)
        history.append(rec);prior=rec
        print(json.dumps(dict(array=array,branch=branch,cycle=k,ready=rec['ready'],stop=rec.get('stop_reason'))),flush=True)
    summary=dict(array=array,branch=branch,completed_cycle=history[-1]['cycle_index'] if history else None,
        ready=bool(history and history[-1]['ready']),stop_reason=history[-1].get('stop_reason') if history else 'no state',
        histories=history,wall_seconds=time.perf_counter()-started,m10_applied=False)
    save_json(root/'SUMMARY.json',summary);manifest(root)
    return summary


def execute(out,binary,phase):
    start=time.perf_counter();revision=source();freeze_ref=json_read(out/'FREEZE.json')
    owner_path=verified(freeze_ref['prospective']);verify_manifest(out/'freeze',freeze_ref['manifest']['sha256'])
    owner=json_read(owner_path)
    short=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
    version=subprocess.check_output([str(binary),'--version'],text=True)
    require(short in version,'binary/source mismatch')
    roots,contexts=recover();verify_seconds=time.perf_counter()-start
    for a in ARRAYS:require(original_parents(FIRST,a)==owner['parents'][a],'frozen upstream ancestry changed')
    branches={'REF':BRANCHES[:1],'C':BRANCHES[1:4],'D':BRANCHES[4:]}[phase]
    if phase!='REF':
        require((out/'PHASE-REF.json').exists(),'REF profile missing')
    if phase=='D':require((out/'PHASE-C.json').exists(),'C execution has priority')
    selected=[];skipped=[]
    for branch in branches:
        for a in ARRAYS:
            refpath=out/f'run/REF/{a}/SUMMARY.json'
            if phase!='REF' and not (refpath.exists() and json_read(refpath)['ready'] and json_read(refpath)['completed_cycle']==10):
                skipped.append(dict(array=a,branch=branch,status='not-run',reason='REF replay unavailable'));continue
            selected.append((a,branch))
    summaries=[]
    # Branch order is fixed; independent arrays use at most three native processes.
    for branch in branches:
        tasks=[a for a,b in selected if b==branch]
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(lambda a:execute_branch(out,binary,owner_path,owner,roots,contexts[a],a,branch),tasks))
        summaries.extend({k:s[k] for k in ('array','branch','completed_cycle','ready','stop_reason','wall_seconds')} for s in results)
    verification=time.perf_counter()
    for p,sha in EXPECTED.items():verify_manifest(p,sha)
    for a in ARRAYS:original_parents(FIRST,a)
    verify_manifest(out/'freeze',freeze_ref['manifest']['sha256'])
    save_json(out/f'PHASE-{phase}.json',dict(phase=phase,source_revision=revision,binary=bound(binary),version=version,
        prospective=bound(owner_path),summaries=summaries,not_run=skipped,workers=3,numerical_threads=1,
        input_verification_seconds=verify_seconds,final_verification_seconds=time.perf_counter()-verification,
        wall_seconds=time.perf_counter()-start,m10_applied=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','REF','C','D'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path)
    args=p.parse_args()
    if args.action=='freeze':freeze(args.output.resolve())
    else:
        require(args.binary is not None,'native binary required');execute(args.output.resolve(),args.binary,args.action)
