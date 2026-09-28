"""Shared native invocation and trajectory sequencing; scientific owners remain separate."""
import argparse,json,os,shutil,subprocess,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import psutil
from point_reference import initialize,context,endpoint
from point_guarded_feedback import infer_base, POLICIES,BRANCHES,infer,publish,measurement,continuation_reason,ideal_comparison
from point_expected import candidate
from point_response_cd import shapes,seed_candidate,processing_guard,injection_reference,response_metrics,direct_raster
from point_fruit_trajectory import bound,verified,selection_difference
from point_fruit_diagnostics import load,patch_map,learned_state
from point_empirical_io import write_fits
from point_empirical_io import exact_header
from point_empirical import source_patch
from point_additive_diagnostics import save_json
from point_fruit_trajectory import ready_state
from run_successor_observation import run as invoke
from inspect_forward_map import require,read_yaml,digest
from point_timing import timed,span,ready,emit

REPO=Path(__file__).resolve().parents[2]
_LOCK=threading.Lock();_RESERVED=0

def read(p):return json.loads(Path(p).read_text())


def size(root):
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:pass
    return total


@timed("native_invocation_including_reservation_hashing_publication")
def call(out,binary,request,dest,terminal='point-fruit',workers=1,allowance=None):
    global _RESERVED
    from point_storage import RunStorage
    budget=read(out/'freeze/budget.json');allowance=allowance or 48*budget['native_cells']
    with _LOCK:
        storage=RunStorage(out,budget);storage.reserve(allowance,_RESERVED)
        if terminal=='point-fruit':record_attempt(out,request)
        _RESERVED+=allowance
    try:return invoke(binary,request,dest,workers,terminal,ptc=read(out/'freeze/prospective.json').get('ptc') if terminal=='ptc' else None)
    finally:
        with _LOCK:
            _RESERVED-=allowance
            accounting=RunStorage(out,budget).finish(dest,allowance)
            with (out/'storage.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),stage=str(dest.relative_to(out)),accounting=accounting,reserved=_RESERVED,ceiling=budget['peak_ceiling_bytes']))+'\n')


def record_attempt(out,request):
    """Called under the reservation lock before invocation; failures still count."""
    q=read(request);owner_record=read(verified(q['owner_binding']));k=q['cycle_index']
    if speed_experiment(out):
        experiment_consume(out,'feedback' if k else 'trajectory',dict(run=out.name,branch=q['branch'],cycle=k))
    elif rank_experiment(out):
        from point_rank_experiment import consume
        consume(Path(read(out/'freeze/budget.json')['campaign_root']),'feedback' if k else 'trajectory',dict(rank=expected_rank(out),branch=q['branch'],cycle=k))
    kind='feedback' if k else 'injected_initial'
    row=dict(kind=kind,policy=owner_record['policy'],branch=q['branch'],cycle=k,request=bound(request))
    path=out/'execution-attempts.jsonl';previous=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    require(not any((r['kind'],r['policy'],r['branch'],r['cycle'])==(kind,row['policy'],row['branch'],k) for r in previous),'duplicate logical application attempt')
    require(sum(r['kind']==kind for r in previous)<(10 if reference(out) else 40 if rank_experiment(out) and k else 4 if rank_experiment(out) else 150 if k else 3),'hard application attempt limit')
    with path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()


def parents(out):return read(out/'freeze/parents.json')

def speed_experiment(out):
    p=out/'freeze/prospective.json'
    return p.exists() and read(p).get('schema')=='citlali-point-speed-pca-v1'

def rtc_experiment(out):
    p=out/'freeze/prospective.json'
    return p.exists() and read(p).get('campaign_scope') in ('rtc-cost-response-v1','rtc-fixed11-v1','rtc-step-point-v1','rtc-step-consequences-v1','rtc-retained-residuals-v1')

def experiment_consume(out,kind,identity):
    if read(out/'freeze/prospective.json').get('campaign_scope')=='rtc-retained-residuals-v1':
        from point_step_experiment import consume_retained as consume
    elif read(out/'freeze/prospective.json').get('campaign_scope')=='rtc-step-consequences-v1':
        from point_step_experiment import consume_consequences as consume
    elif read(out/'freeze/prospective.json').get('campaign_scope')=='rtc-step-point-v1':
        from point_step_experiment import consume
    elif read(out/'freeze/prospective.json').get('campaign_scope')=='rtc-fixed11-v1':
        from point_fixed11_experiment import consume
    elif rtc_experiment(out):
        from point_rtc_experiment import consume
    else:
        from point_speed_experiment import consume
    return consume(Path(read(out/'freeze/budget.json')['campaign_root']),kind,identity)

def feedback_count(out):
    return read(out/'freeze/prospective.json').get('feedback_count',10) if speed_experiment(out) else 10

def rank_experiment(out):
    p=out/'freeze/prospective.json'
    return p.exists() and read(p).get('schema') in ('citlali-point-ptc-rank-v1','citlali-point-speed-pca-v1')

def expected_rank(out):return read(out/'freeze/prospective.json').get('ptc',{}).get('rank',10)

def base_route(out):return reference(out) or rank_experiment(out)

def reference(out):
    p=out/'freeze/prospective.json'
    return p.exists() and read(p).get('schema')=='citlali-bright-point-reference-v1'

def selected_array(out):return read(out/'freeze/prospective.json')['array']


@timed("raw_parent_PTC_MAP_initial_guard_noise")
def upstream(out,binary):
    require(not (out/'UPSTREAM.json').exists() and not (out/'upstream').exists(),'no upstream retry')
    t=time.perf_counter();rc=call(out,binary,out/'inputs/observation.json',out/'upstream','ptc',read(out/'freeze/budget.json').get('network_workers',2),read(out/'freeze/budget.json')['shared_upstream_allowance'])
    if rc:save_json(out/'UPSTREAM.json',dict(status='failed',return_code=rc,seconds=time.perf_counter()-t));return
    networks=[]
    for nw in read(out/'inputs/observation.json')['networks']:
        nw=nw['network']
        root=out/f'upstream/products/network{nw}';pr=read_yaml(root/'donor-continuity/ptc/receipt.yaml')
        require(pr['failed_fits']==0 and all(s['converged'] and s.get('application_failed_times',0)==0 for s in pr['segments']),'new observation initial PTC failure')
        networks.append(dict(root=str(root),input=bound(out/f'inputs/network{nw}-input.json'),receipt_sha256=digest(root/'receipt.yaml'),ptc_receipt_sha256=digest(root/'donor-continuity/ptc/receipt.yaml')))
    q=out/'inputs/map-request.json';save_json(q,dict(schema='citlali-map-saved-ptc-v1',owner_binding=bound(out/'inputs/map-owner.json'),pixel_size_arcsec=2.,coverage_cut=.5,networks=networks))
    require(call(out,binary,q,out/'M0','map')==0,'ordinary initial MAP failed')
    p=dict(map_request=bound(q),parent_plan=bound(out/f'M0/products/{selected_array(out)}/plan.yaml'));save_json(out/'freeze/parents.json',p)
    if rank_experiment(out) and not speed_experiment(out):
        from point_rank_experiment import consume
        consume(Path(read(out/'freeze/budget.json')['campaign_root']),'noise',dict(rank=expected_rank(out)))
    noise_reference=read(out/'freeze/prospective.json').get('noise_reference')
    if rtc_experiment(out) and noise_reference:
        # Same RTC variant's unperturbed M0 noise/guard conditioning only.
        # Native parent and cold M0 fit remain branch-owned.
        ref=Path(noise_reference['root']);verified(noise_reference['binding'])
        require(read(ref/'freeze/prospective.json')['rtc_family']==read(out/'freeze/prospective.json')['rtc_family'],'foreign RTC noise variant')
        for b in noise_reference['files']:verified(b)
        shutil.copytree(ref/f'freeze/{selected_array(out)}',out/f'freeze/{selected_array(out)}')
        c=context(out/f'freeze/{selected_array(out)}');signal,valid,_,_=patch_map(out/f'M0/products/{selected_array(out)}/map.fits')
        injected_measurement,_=infer_base(c,signal,valid);g=injected_measurement['guard']
        if not g['accepted']:
            from point_response_cd import save_candidate
            save_candidate(out/f'freeze/{selected_array(out)}/candidate.fits',exact_header(c['header']),candidate(signal,valid,c['fixed'],g))
            save_json(out/f'freeze/{selected_array(out)}/injected-guard.json',g)
    else:
        c,g=initialize(out/f'M0/products/{selected_array(out)}/map.fits',out/f'freeze/{selected_array(out)}',beam={'a1100':5.,'a1400':7.5,'a2000':10.}[selected_array(out)])
    save_json(out/'UPSTREAM.json',dict(status='accepted' if g['accepted'] else 'source_or_covariance_failed',guard=g,seconds=time.perf_counter()-t,initial_physical_inputs=1,parent=p))
    if c is not None:
        if speed_experiment(out):
            if read(out/'freeze/prospective.json').get('campaign_scope') in ('rtc-step-point-v1','rtc-step-consequences-v1') and not (out/'freeze/measurement-geometry.json').exists():
                from point_step_experiment import freeze_own_geometry
                freeze_own_geometry(out,c)
            shutil.copyfile(out/'freeze/measurement-geometry.json',out/f'freeze/{selected_array(out)}/measurement-geometry.json')
            c=context(out/f'freeze/{selected_array(out)}')
        m=injected_measurement if noise_reference else measurement(c,c['original'],c['support'],g);save_json(out/'freeze/M0-measurement.json',m)
        save_json(out/'freeze/realized.json',[bound(p) for p in sorted((out/f'freeze/{selected_array(out)}').iterdir()) if p.is_file()]+[bound(out/'freeze/M0-measurement.json'),bound(out/'freeze/parents.json'),bound(out/'inputs/map-request.json')])


def owner(out,pol,branch,initial=None):
    if base_route(out):
        p=read(out/'freeze/prospective.json')
        o=dict(identity='POINT-SPEED-PCA-v1' if speed_experiment(out) else 'POINT-PTC-RANK-v1' if rank_experiment(out) else 'BRIGHT-POINT-BASE-v1',observation=p['observation'],outside_disk_no_feedback=True,direct_identical_operand_rejoin=True,gain=1,nonzero_cycles=feedback_count(out),policy='BASE-POINT',source_method='POINT-EXPECTED-v1',threshold=3.,prospective=bound(out/'freeze/prospective.json'),parents={selected_array(out):parents(out)},injections={})
        if rank_experiment(out) and not (p.get('campaign_scope') in ('rtc-step-point-v1','rtc-step-consequences-v1') and not (out/'freeze/shapes').exists()):
            o['injections']={selected_array(out):{b:bound(out/f'freeze/shapes/{b}.fits') for b in BRANCHES[1:4]}}
            if branch.startswith('C-') and initial:
                o['injected_parents']={};o['injection_operands']={}
                for s in read_yaml(initial/'receipt.yaml')['segments']:
                    old=s['source_segment'];stem=next(n[:-10] for n in old['files'] if n.endswith('-input.f64'));name=f"network-{old['network']}-{stem}"
                    o['injected_parents'][name]=bound(initial/name/'injected-parent.f64');o['injection_operands'][name]=bound(initial/name/'injection.f64')
        return o
    p=read(out/'freeze/prospective.json');o=dict(identity='POINT-152430-NO-UPPER-COVERAGE-VETO-2026-09-26',outside_disk_no_feedback=True,direct_identical_operand_rejoin=True,gain=1,nonzero_cycles=10,policy=pol,source_method='POINT-EXPECTED-v1',threshold=3. if pol=='BASE-POINT' else p['thresholds'][pol],prospective=bound(out/'freeze/prospective.json'),thresholds=bound(out/'freeze/thresholds.json'),parents={'a2000':parents(out)},injections={'a2000':{b:bound(out/f'freeze/shapes/{b}.fits') for b in BRANCHES[1:4]}} if (out/'SHAPES.json').exists() else {})
    if branch.startswith('C-') and initial:
        o['injected_parents']={};o['injection_operands']={}
        for s in read_yaml(initial/'receipt.yaml')['segments']:
            old=s['source_segment'];stem=next(n[:-10] for n in old['files'] if n.endswith('-input.f64'));name=f"network-{old['network']}-{stem}"
            o['injected_parents'][name]=bound(initial/name/'injected-parent.f64');o['injection_operands'][name]=bound(initial/name/'injection.f64')
    return o


def request(out,op,pol,branch,k,cp,previous=None):
    if rank_experiment(out):
        require(pol=='BASE-POINT' and branch in BRANCHES[:4] and 0<=k<=feedback_count(out),'bounded rank matrix')
        q=dict(schema='citlali-point-speed-pca-v1' if speed_experiment(out) else 'citlali-point-ptc-rank-v1',method='point-speed-pca-v1' if speed_experiment(out) else 'point-ptc-rank-v1',mode='zero' if k==0 else 'reference',array=selected_array(out),branch=branch,cycle_index=k,candidate=bound(cp),owner_binding=bound(op),**parents(out))
        if k:q['previous_state']=bound(previous)
        if branch.startswith('C-'):q['injection']=bound(out/f'freeze/shapes/{branch}.fits')
        return q
    if reference(out):
        require(pol=='BASE-POINT' and branch=='REF' and 1<=k<=10,'frozen bright POINT route')
        return dict(schema='citlali-bright-point-reference-v1',method='bright-point-base-v1',mode='reference',array=selected_array(out),branch='REF',cycle_index=k,candidate=bound(cp),owner_binding=bound(op),previous_state=bound(previous),**parents(out))
    require(pol in POLICIES and branch in BRANCHES and 0<=k<=10,'matrix outside bound')
    q=dict(schema='citlali-point-fruit-independent-v1',method='point-152430-no-upper-coverage-veto-v1',mode='zero' if k==0 else 'reference',array='a2000',branch=branch,cycle_index=k,candidate=bound(cp),owner_binding=bound(op),**parents(out))
    if k:q['previous_state']=bound(previous)
    if branch.startswith('C-'):q['injection']=bound(out/f'freeze/shapes/{branch}.fits')
    return q


@timed("cycle_cleanup")
def cleanup_cycle(out,products,keep_projection=False,keep_injected=False):
    # Only exact native producer payload names, never broad directory removal.
    targets=[]
    names=['cleaned.f64','operand.f64','model-support.u8']
    if not keep_projection:names+=['causes.u8','injection.f64']
    for d in products.glob('network-*-segment-*'):
        for n in names:
            if (d/n).is_file() and not(keep_injected and n=='injection.f64'):targets.append(d/n)
    if not keep_projection:
        targets += [products/n for n in ['operator-pixel.i32','output-rows.u64'] if (products/n).is_file()]
    with _LOCK:
        with (out/'temporary-removals.jsonl').open('a') as log:
            for p in targets:
                require(p.resolve().is_relative_to(out.resolve()) and not p.is_symlink(),'unsafe temporary target');ref=bound(p);ref['bytes']=p.stat().st_size;log.write(json.dumps(ref)+'\n');log.flush();p.unlink()


@timed("trajectory_including_endpoint")
def execute_branch(out,binary,pol,branch):
    array=selected_array(out);infer_stage=infer_base if base_route(out) else infer
    start=time.perf_counter();c=context(out/f'freeze/{array}');dest=out/'run'/pol/branch/array;dest.mkdir(parents=True);init=dest/'initial';init.mkdir()
    physical=out/'initial'/branch/'native/products' if branch.startswith('C-') else None
    if physical:
        r=read(out/'initial'/branch/'INITIAL.json');require(r['available'],r.get('reason','initial input unavailable'));total=verified(r['total']);receipt=verified(r['receipt']);m=r['measurement']
    else:
        total=out/f'M0/products/{array}/map.fits';receipt=out/'M0/products/attempt.yaml';m=read(out/'freeze/M0-measurement.json')
    s,v,_,_=patch_map(total);g=m['guard'];b=candidate(s,v,c['fixed'],g);fields=c['neighborhood'].scores(b) if not base_route(out) and g['accepted'] and b['selected'].any() else None
    o=owner(out,pol,branch,physical);tau=o['threshold'];cp,summary=publish(init,c,s,(g,b,fields),pol,tau);cause,reason=continuation_reason(g,summary,load(cp)[0],c['header'])
    if branch=='D-seed-shoulder' and cause is None:
        own=read(out/'run'/pol/'REF/a2000/SUMMARY.json')['histories'][0];orig,_=load(verified(own['candidate']));current,_=load(cp)
        require(all(np.array_equal(orig[n],current[n],equal_nan=True) for n in ('CANDIDATE','VALID','ADMITTED','RATIO')),'seed not own REF candidate')
        z,_=load(out/'freeze/shapes/C-shoulder.fits');seed=seed_candidate(orig,branch,z['INJECTION']);cp=init/'seed.fits';write_fits(cp,exact_header(c['header']),[(n,seed[n],'mJy/nominal beam' if n=='CANDIDATE' else '1') for n in ('CANDIDATE','ADMITTED','VALID','RATIO')],companion='POINT-INDEPENDENT-SEED');cause,reason=continuation_reason(g,summary,seed,c['header'])
    o['seeds']={array:{branch:bound(cp)}};op=dest/'owner.json';save_json(op,o)
    prior=ready_state(array,0,cp,total,receipt,parents(out),cause is None,reason);prior.update(branch=branch,policy=pol,owner_sha256=digest(op),measurement=m,admission=summary,stop_kind=cause);sp=init/'state.json';save_json(sp,prior);prior['state_path']=str(sp)
    if physical:prior['response']=r['response']
    history=[prior]
    ready(out,0,total,m,sp)
    for k in range(1,feedback_count(out)+1):
        if not prior['ready']:break
        d=dest/f'cycle-{k:02d}';d.mkdir();q=d/'request.json';save_json(q,request(out,op,pol,branch,k,verified(prior['candidate']),Path(prior['state_path'])))
        rec=dict(array=array,policy=pol,branch=branch,cycle_index=k,ready=False);tic=time.perf_counter()
        try:
            require(call(out,binary,q,d/'native')==0,'native cycle failed')
            products=d/'native/products';rr=read_yaml(products/'receipt.yaml');s,v,_,_=patch_map(products/'total.fits');m,prepared=infer_stage(c,s,v);cp,summary=publish(d,c,s,prepared,pol,tau)
            rec.update(total=bound(products/'total.fits'),receipt=bound(products/'receipt.yaml'),candidate=bound(cp),measurement=m,admission=summary)
            rec['guards']=processing_guard(products,rr,allow_historical_failure=False,expected_rank=expected_rank(out));rec['PTC_segments']=learned_state(products,rr)
            old,_=load(verified(prior['candidate']));new,_=load(cp);rec['candidate_turnover']=selection_difference(new,old)
            restored,rv,_,_=patch_map(products/'restored.fits');use=rv&old['VALID'].astype(bool);err=float(np.max(abs((restored-old['CANDIDATE'])[use])));require(err<1e-8,'operand remap differs');rec['guards']['operand_remap_max_abs']=err
            if branch.startswith('C-'):
                ref=read(out/'run'/pol/'REF/a2000/SUMMARY.json')['histories'][k];refprod=Path(ref['total']['path']).parent;z,zh=load(out/f'freeze/shapes/{branch}.fits');known,kv,kh,audit=injection_reference(refprod,products,z['INJECTION'],zh)
                refdata,_=load(ref['total']['path']);actual,ah=load(products/'total.fits');common=kv&actual['SCI_SUPPORT'].astype(bool);delta=actual['PRIMARY']-refdata['PRIMARY'];ks,vs,_,_,_=source_patch(known,kv,refdata['NORMALIZATION'],kh);ds,_,_,_,_=source_patch(delta,common,refdata['NORMALIZATION'],kh)
                rec['response']=dict(audit=audit,full=response_metrics(delta,known,common),source_disk=response_metrics(ds,ks,c['fixed'].disk&v&vs))
                write_fits(d/'response.fits',kh,[('DELTA',delta,'mJy/nominal beam'),('CONTROL',known,'mJy/nominal beam'),('RESIDUAL',delta-known,'mJy/nominal beam'),('VALID',common.astype('u1'),'1')],companion='POINT-INDEPENDENT-RESPONSE')
                if k in ((1,3,5,10) if speed_experiment(out) else (10,)):
                    if rank_experiment(out) and not speed_experiment(out):
                        from point_rank_experiment import consume
                        consume(Path(read(out/'freeze/budget.json')['campaign_root']),'ideal',dict(rank=expected_rank(out),branch=branch))
                    rs,rsv,_,_=patch_map(ref['total']['path']);ideal,_=infer_stage(c,rs+ks,rsv&vs);rec['ideal']=ideal_comparison(c,ref['measurement'],m,ideal)
            cause,reason=continuation_reason(prepared[0],summary,new,c['header']);rec.update(ready=cause is None,stop_kind=cause,stop_reason=reason)
            state=ready_state(array,k,cp,products/'total.fits',products/'receipt.yaml',parents(out),cause is None,reason);state.update(branch=branch,policy=pol,owner_sha256=digest(op));sp=d/'state.json';save_json(sp,state);rec['state_path']=str(sp)
            if not (rtc_experiment(out) and k==5):
                cleanup_cycle(out,products,keep_projection=branch=='REF' and not reference(out))
        except (ValueError,RuntimeError,KeyError,OSError) as e:rec.update(ready=False,stop_kind='computation_or_processing',stop_reason=str(e))
        rec['wall_seconds']=time.perf_counter()-tic;save_json(d/'cycle.json',rec);history.append(rec);prior=rec
        if 'measurement' in rec:
            if speed_experiment(out) and rec.get('stop_kind')=='computation_or_processing':
                sp=d/'state.json';save_json(sp,dict(cycle_index=k,ready=False,stop_kind=rec.get('stop_kind','processing_failed'),stop_reason=rec.get('stop_reason'),measurement=m));rec['state_path']=str(sp)
            ready(out,k,products/'total.fits',m,sp)
        print(json.dumps(dict(policy=pol,branch=branch,cycle=k,ready=rec['ready'],reason=rec.get('stop_reason'))),flush=True)
    result=dict(array=array,policy=pol,branch=branch,ready=history[-1]['ready'],completed_cycle=history[-1]['cycle_index'],stop_reason=history[-1].get('stop_reason'),histories=history,actual_feedback_applications=len(history)-1,wall_seconds=time.perf_counter()-start,m10_applied=False)
    save_json(dest/'SUMMARY.json',result)
    if branch=='REF' and result['ready'] and result['completed_cycle']==feedback_count(out):
        if speed_experiment(out):
            experiment_consume(out,'endpoint',dict(run=out.name,cycle=result['completed_cycle']))
        elif rank_experiment(out):
            from point_rank_experiment import consume
            consume(Path(read(out/'freeze/budget.json')['campaign_root']),'endpoint',dict(rank=expected_rank(out)))
        save_json(dest/'ENDPOINT-ATTEMPT.json',dict(status='started',diagnostic_only=True,maximum_attempts=1))
        try:
            result['endpoint_uncertainty']=endpoint(verified(history[-1]['total']),c,history[-1]['measurement'],dest/'endpoint')
            result['endpoint_status']='completed'
        except (ValueError,RuntimeError,KeyError,OSError,np.linalg.LinAlgError) as exc:
            result['endpoint_status']='unavailable';result['endpoint_reason']=str(exc)
        save_json(dest/'ENDPOINT-ATTEMPT.json',dict(status=result['endpoint_status'],reason=result.get('endpoint_reason'),diagnostic_only=True,maximum_attempts=1))
        save_json(dest/'SUMMARY.json',result)
        emit('analysis_complete',cycle=result['completed_cycle'],status=result['endpoint_status'])
    return result

