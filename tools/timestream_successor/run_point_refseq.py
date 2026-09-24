#!/usr/bin/env python3
"""Owner-bounded a2000 stage-matched REF-basis control; no observation Learn."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from point_frozen_shoulder import *
from point_fruit_trajectory import (bound,verified,manifest,verify_manifest,FixedMeasurements,
                                  candidate_guard,selection_difference)
from point_fruit_diagnostics import patch_map,compare_zero
from point_response_cd import save_candidate
from point_admission import admit
from run_point_response_cd import source,replay_check
from run_point_fruit_ten import ready_state
from run_successor_observation import run as invoke

FROZEN=CD.parent/'successor-point-152389-frozen-ptc-shoulder-20260924'
FROZEN_HASH='254894c94b4cc5b2fccbe33bd02cb248740fe6f8ce0eadb69d6bbc0b1e7140b7'
BRANCHES=('REFSEQ-REF','REFSEQ-SHOULDER')


def contexts(auth,owner):
    ctx=saved_context(auth,owner,'a2000')
    # Restore fixed definitions from authenticated originals; constructor would
    # refit an old map. Only newly produced branch maps may be fitted here.
    f=FixedMeasurements.__new__(FixedMeasurements)
    f.header=ctx['header'];f.beam=10.;f.parameters=ctx['p'];f.x=ctx['x'];f.y=ctx['y']
    f.disk=np.hypot(f.x,f.y)<=40;f.radius=ctx['radius'];f.major=ctx['major']
    f.background=f.disk&(f.radius>3*f.major);f.edges=np.arange(0,42,2)
    f.weights=ctx['q'];f.scale=ctx['scale'];f.noise_valid=ctx['noise_valid']
    return ctx,f


def freeze(out):
    t=time.perf_counter();revision=source();require(not out.exists(),'fresh bundle required')
    auth,old=ancestors();auth.manifest(FROZEN,FROZEN_HASH)
    for root in (CD,FROZEN):
        for n in ('ASSESSMENT.md','REVIEW.md','FINAL_REVIEW.json','COMPLETION.json','REGENERATION.md'):auth.check(root/n)
    _,fixed=contexts(auth,old)
    sums={b:auth.json(CD/f'run/{name}/a2000/SUMMARY.json') for b,name in zip(BRANCHES,('REF','C-shoulder'))}
    out.mkdir();f=out/'freeze';f.mkdir()
    w=dict(schema='point-a2000-refseq-prospective-v1',identity='POINT-152389-A2000-REFSEQ-OWNER-2026-09-24',
        source_revision=revision,outside_disk_no_feedback=True,direct_identical_operand_rejoin=True,gain=1,nonzero_cycles=10,
        parents={'a2000':old['parents']['a2000']},injections={'a2000':old['injections']['a2000']},
        branches=list(BRANCHES),run_order=list(BRANCHES),basis_schedule=[],old_stages=sums,
        injection_operands={},injected_parents={},workers=3,library_threads=1,available_logical_cpus=os.cpu_count(),
        tolerances=dict(TOL,deterministic_measurements_candidates='exact',support_membership_operator='exact'),
        endpoints=[0,10],threshold=3,feedback_radius_arcsec=40,
        method='original native observed-entry rank10; fresh current means/application; externally supplied stage-matched REF bases',
        source_model='ordinary source-relative measured-map brightness, connected threshold3; complete replacement',
        source_fit='unchanged seven-parameter Gaussian plus constant; no injection initialization',
        comparison_domain='original C+D full COMMON with finite KNOWN, fixed per stage; no headline shrinking',
        isolation_proof='refseq branch calls fruit_apply_reference_basis -> prepare/ptc_apply only; does not enter FruitLearned::learn; no arbitrary-call interception claimed',
        figures=['all-stage response trajectories','matched-scale T10 maps/residuals','candidate history/masks','M0/T10 avoided-subspace and history accounting'],
        directive=bound(Path(__file__).resolve().parents[2]/'validation/point_152389_a2000_ref_basis_sequence_2026-09-24/OWNER_DIRECTIVE.md'))
    repo=Path(__file__).resolve().parents[2];w['ptc_source_equivalence']={}
    for path in ('src/citlali/core/pipeline/timestream_ptc_numerics.cpp','include/citlali/core/pipeline/timestream_ptc_numerics.h'):
        require((repo/path).read_bytes()==subprocess.check_output(['git','show','26af74ac5e59d88a99da1ed4047a7a71a327fca0:'+path],cwd=repo),'numerical owner changed')
        w['ptc_source_equivalence'][path]=digest(repo/path)
    for k in range(11):
        schedule={};receipts={}
        for branch in BRANCHES:
            c=sums[branch]['histories'][k];require(c['ready'] and c['cycle_index']==k,'original stage incomplete')
            p=Path(c['receipt']['path']).parent;r=auth.yaml(p/'receipt.yaml');receipts[branch]=r
            for path,h in auth.yaml(p/'input-manifest.yaml').items():auth.bind(path,h);auth.check(path)
            for path in p.rglob('*'):
                if path.is_file():auth.check(path)
            for n in ('total','candidate','receipt','request'):auth.check(c[n]['path'])
            req=auth.json(c['request']['path'])
            if k:require(req['candidate']==sums[branch]['histories'][k-1]['candidate'],'original operand indexing')
        rr=receipts[BRANCHES[0]];hr=receipts[BRANCHES[1]]
        require(len(rr['segments'])==len(hr['segments'])==12,'required12-segment a2000 grouping')
        for rs,hs in zip(rr['segments'],hr['segments']):
            s=rs['source_segment'];require(s==hs['source_segment'],'paired original grouping changed')
            name=f"network-{s['network']}-segment-{s['segment_index']}" if 'segment_index' in s else None
            stem=next(n.removesuffix('-input.f64') for n in s['files'] if n.endswith('-input.f64'));name=f"network-{s['network']}-{stem}"
            root=Path(rs['source_root']);pr=auth.yaml(root/'receipt.yaml')
            require(rs['converged'] and rs['rank']==10 and s['basis_rows']==s['detectors'] and s['basis_columns']==10,'reference basis unavailable')
            bp=Path(sums[BRANCHES[0]]['histories'][k]['receipt']['path']).parent/name/'basis.f64'
            schedule[name]=dict(stage=k,basis=auth.ref(bp),rank=10,relative_rank_tolerance=pr['relative_rank_tolerance'],
                parent_input_sha256=auth.ref(root/(stem+'-input.f64'))['sha256'],eligible_sha256=auth.ref(root/(stem+'-eligible.u8'))['sha256'],
                source_receipt=auth.ref(Path(sums[BRANCHES[0]]['histories'][k]['receipt']['path'])),source_segment=s)
            if k==0:
                nt,nd=s['scheduled_times'],s['detectors'];ip=Path(sums[BRANCHES[1]]['histories'][k]['receipt']['path']).parent/name/'injection.f64'
                u=np.fromfile(auth.check(ip),dtype='<f8').reshape(nt,nd);y=np.fromfile(auth.check(root/(stem+'-input.f64')),dtype='<f8').reshape(nt,nd)
                mask=np.fromfile(auth.check(root/(stem+'-eligible.u8')),dtype='u1').reshape(nt,nd).astype(bool)
                require(np.isfinite(u).all() and np.all(u[~mask]==0),'injection invalid-entry policy')
                yy=y.copy();yy[mask]+=u[mask];e=np.abs((yy[mask]-y[mask])-u[mask]);require(np.all(e<=8*np.finfo(float).eps*np.maximum(1,np.maximum(abs(y[mask]),abs(u[mask])))),'injection identity')
                dest=f/(name+'-injected-parent.f64');yy.tofile(dest)
                w['injection_operands'][name]=auth.ref(ip);w['injected_parents'][name]=bound(dest)
            else:
                ip=Path(sums[BRANCHES[1]]['histories'][k]['receipt']['path']).parent/name/'injection.f64'
                require(auth.ref(ip)['sha256']==w['injection_operands'][name]['sha256'],'injection changed with stage')
        w['basis_schedule'].append(schedule)
        for suffix in ('json','fits'):auth.check(CD/f'report/a2000/C-shoulder/cycle-{k:02d}.{suffix}')
    for k in (0,10):
        for name in ('components.fits','result.json'):auth.check(FROZEN/f'pairs/a2000/{k:02d}/{name}')
        pair=next(p for p in auth.json(FROZEN/'freeze/prospective.json')['pairs'] if p['array']=='a2000' and p['cycle']==k)
        for s in pair['segments']:
            req=auth.yaml(s['request']['path'])
            for ref in req['inputs'].values():auth.check(ref['path'])
            auth.check(s['request']['path'])
    w['original_initial_grid_control']=old['original_cycles']['a2000'][0]['candidate'];auth.check(w['original_initial_grid_control']['path'])
    w['identities']=auth.consumed;w['freeze_seconds']=time.perf_counter()-t
    save_json(f/'prospective.json',w);manifest(f);save_json(out/'FREEZE.json',dict(prospective=bound(f/'prospective.json'),manifest=bound(f/'FILES_SHA256SUMS')))
    print(json.dumps(dict(frozen=True,stages=11,bases=sum(map(len,w['basis_schedule'])),authenticated=len(w['identities']),seconds=w['freeze_seconds'])),flush=True)


def auth_work(w):
    auth=Auth()
    for p,h in w['identities'].items():auth.bind(p,h);auth.check(p)
    return auth


def stage_request(w,path,branch,k,prior):
    require(branch in BRANCHES and 0<=k<=10,'bounded branch/stage')
    require((k==0 and prior is None) or (k>0 and prior['ready'] and prior['cycle_index']==k-1),'predecessor required')
    r=dict(schema='citlali-point-fruit-refseq-v1',method='point-152389-a2000-ref-basis-sequence-v1',
        array='a2000',branch=branch,cycle_index=k,mode='zero' if k==0 else 'reference',
        candidate=w['original_initial_grid_control'] if k==0 else prior['candidate'],owner_binding=bound(path),**w['parents']['a2000'])
    if k:r['previous_state']=bound(prior['state_path'])
    if branch==BRANCHES[1]:r['injection']=w['injections']['a2000']['C-shoulder']
    return r


def exact_file(a,b,name):require(Path(a).read_bytes()==Path(b).read_bytes(),name+' differs')


def support_guard(p,r,old,w,k,reference):
    op=Path(old['receipt']['path']).parent;orr=read_yaml(op/'receipt.yaml');checks={}
    data,h=load(p/'total.fits');expected,eh=load(op/'total.fits');v=data['SCI_SUPPORT'].astype(bool)
    for key in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','RADESYS','EQUINOX']:require(h[key]==eh[key],'WCS transition')
    for key in ('HITS','NORMALIZATION','SCI_SUPPORT','NORM_SUPPORT'):require(np.array_equal(data[key],expected[key]),key+' transition')
    for name in ('operator-pixel.i32','output-rows.u64'):exact_file(p/name,op/name,name)
    residual,_=load(p/'residual.fits');restore,_=load(p/'restored.fits')
    checks['remove_rejoin']=close(data['PRIMARY'][v],(residual['PRIMARY']+restore['PRIMARY'])[v],'remove/rejoin')
    require(len(r['segments'])==len(orr['segments'])==12 and r['learning_performed'] is False,'stage/learning state')
    checks['segments']=[]
    for new,oldseg in zip(r['segments'],orr['segments']):
        s=oldseg['source_segment'];require(s==new['source_segment'],'segment identity');stem=next(n.removesuffix('-input.f64') for n in s['files'] if n.endswith('-input.f64'));name=f"network-{s['network']}-{stem}"
        require(new['basis_source']==w['basis_schedule'][k][name],'imported basis/stage identity')
        require('converged' not in new and new['learning_performed'] is False and new['basis_available'],'false learning lineage')
        require(new['input_eligibility_changed']==0 and new['projection_support_counts'][0]==0,'input/model transition')
        require(new['application_failed_times']==oldseg['application_failed_times'] and new['application_outcome']=='success','new failed Apply')
        exact_file(p/name/'causes.u8',op/name/'causes.u8','output causes')
        exact_file(p/name/'basis.f64',verified(w['basis_schedule'][k][name]['basis']),'basis serialization')
        require(new['retained']==oldseg['retained'] and new['MAP_first_input_row']==oldseg['MAP_first_input_row'] and new['MAP_admitted']==oldseg['MAP_admitted'],'per-network MAP contributions')
        q=dict(segment=name,causes_operator_basis_exact=True)
        if reference:
            for field in ('mean.f64','cleaned.f64','operand.f64'):
                a=np.fromfile(p/name/field,dtype='<f8');b=np.fromfile(op/name/field,dtype='<f8');q[field]=close(a,b,field,1e-9)
                require(np.array_equal(a,b),'deterministic '+field+' replay differs')
            exact_file(p/name/'model-support.u8',op/name/'model-support.u8','model support')
        checks['segments'].append(q)
    return checks


def trajectory_complete(history):
    # m10 is terminal/unapplied; its optional fit/candidate cannot erase T10.
    return (len(history)==11 and all(r['numerical_passed'] for r in history)
            and all(r['ready'] for r in history[:-1]))


def check_native_version(version,revision):
    # Logger preambles and dependency versions are separate provenance.
    versions=[line for line in version.splitlines() if line.startswith('v')]
    native=versions[0] if len(versions)==1 else ''
    require(revision[:9] in native and 'dirty' not in native,'native/source mismatch')


def execute(out,binary,branch):
    start=time.perf_counter();cpu=time.process_time();revision=source();fr=json.loads((out/'FREEZE.json').read_text());path=verified(fr['prospective']);w=json.loads(path.read_text());verify_manifest(out/'freeze',fr['manifest']['sha256'])
    require(branch in BRANCHES,'foreign branch');version=subprocess.check_output([str(binary),'--version'],text=True).strip();check_native_version(version,revision)
    for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):require(os.environ.get(n)=='1','numerical thread setting')
    if branch==BRANCHES[1]:
        control=json.loads((out/'run'/BRANCHES[0]/'SUMMARY.json').read_text());require(control['complete'] and all(p.get('replay',{}).get('measurements_candidates_exact',False) for p in control['stages']),'full REF replay must gate shoulder')
    auth=auth_work(w);before=time.perf_counter()-start;owner=auth.json(CD/'freeze/prospective.json');ctx,fixed=contexts(auth,owner)
    root=out/'run'/branch;root.mkdir(parents=True);history=[];prior=None
    for k in range(11):
        dest=root/f'cycle-{k:02d}';dest.mkdir();t=time.perf_counter();rec=dict(branch=branch,cycle_index=k,ready=False,map_completed=False,numerical_passed=False,candidate_available=False)
        req=stage_request(w,path,branch,k,prior);save_json(dest/'request.json',req);rec['request']=bound(dest/'request.json')
        try:
            require(invoke(binary,dest/'request.json',dest/'native',1,'point-fruit')==0,'native stage failed; retained log')
            p=dest/'native/products';receipt=read_yaml(p/'receipt.yaml');rec.update(map_completed=True,total=bound(p/'total.fits'),receipt=bound(p/'receipt.yaml'))
            rec['support_checks']=support_guard(p,receipt,w['old_stages'][BRANCHES[0]]['histories'][k],w,k,branch==BRANCHES[0]);rec['numerical_passed']=True
            tfit=time.perf_counter();signal,support,_,header=patch_map(p/'total.fits');measurement=fixed.measure(signal,support);rec['measurement']=measurement
            fit=measurement['fit'];require(fit['status']=='success' and not fit.get('constrained',False),'required branch fit unavailable/constrained')
            candidate=admit(signal,support,fixed.scale,fixed.noise_valid,fixed.header,np.asarray(fit['parameters']),3.,1.)
            candpath=dest/f'm{k}.fits';save_candidate(candpath,header,candidate);rec.update(candidate=bound(candpath),candidate_available=True,candidate_summary=candidate['summary'],fit_admission_seconds=time.perf_counter()-tfit)
            require(candidate['summary']['status']=='retained','candidate unavailable');candidate_guard(load(candpath)[0],header)
            if branch==BRANCHES[0]:rec['replay']=replay_check(w['old_stages'][branch]['histories'][k],p,measurement,candpath,10.)
            oldcand,_=load(w['old_stages'][branch]['histories'][k]['candidate']['path']);newcand,_=load(candpath)
            rec['candidate_vs_old']=selection_difference(newcand,oldcand)
            if prior:rec['candidate_vs_previous']=selection_difference(newcand,load(verified(prior['candidate']))[0])
            state=ready_state('a2000',k,candpath,p/'total.fits',p/'receipt.yaml',w['parents']['a2000'],True);state.update(branch=branch,owner_sha256=digest(path))
            save_json(dest/'state.json',state);rec.update(ready=True,state_path=str(dest/'state.json'))
        except (ValueError,KeyError,OSError,RuntimeError,AssertionError) as exc:
            rec.update(stop_reason=str(exc))
            if rec['numerical_passed'] and 'measurement' not in rec:
                rec['measurement']=dict(fit=dict(status='unavailable',reason=str(exc)))
        rec['wall_seconds']=time.perf_counter()-t;save_json(dest/'stage.json',rec);history.append(rec);prior=rec
        print(json.dumps(dict(branch=branch,stage=k,ready=rec['ready'],reason=rec.get('stop_reason'),seconds=rec['wall_seconds'])),flush=True)
        if not rec['ready']:break
    after=auth.verify_all();verify_manifest(out/'freeze',fr['manifest']['sha256'])
    summary=dict(branch=branch,source_revision=revision,binary=bound(binary),version=version,stages=history,complete=trajectory_complete(history),
        terminal_candidate_ready=bool(history and history[-1]['cycle_index']==10 and history[-1]['ready']),
        maps_completed=sum(r['map_completed'] for r in history),candidates_available=sum(r['candidate_available'] for r in history),
        workers=3,library_threads=1,available_logical_cpus=os.cpu_count(),wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,
        loading_verification_seconds=before,final_verification_seconds=after,parent_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_peak_rss_units='bytes macOS; KiB Linux',aggregate_memory='not measured',
        m10_applied=False,observation_learning_calls=0,zero_learning_evidence=w['isolation_proof'])
    save_json(root/'SUMMARY.json',summary);manifest(root)
    require(summary['complete'],'branch stopped; dependent execution forbidden')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze',*BRANCHES]);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path);a=p.parse_args()
    if a.action=='freeze':freeze(a.output)
    else:require(a.binary is not None,'binary required');execute(a.output,a.binary,a.action)
