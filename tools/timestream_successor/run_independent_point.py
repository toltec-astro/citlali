"""Bounded ordinary-MAP POINT152430 a2000 frozen-method validation command."""
import argparse,json,os,shutil,subprocess,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import psutil
from point_independent import initialize,context,endpoint
from point_guarded_feedback import POLICIES,BRANCHES,infer,publish,measurement,continuation_reason,ideal_comparison
from point_expected import candidate
from point_response_cd import shapes,seed_candidate,processing_guard,injection_reference,response_metrics,direct_raster
from point_fruit_trajectory import bound,verified,selection_difference
from point_fruit_diagnostics import load,patch_map,learned_state
from point_empirical_io import write_fits
from point_jinc_campaign import exact_header
from point_empirical import source_patch
from point_additive_diagnostics import save_json
from run_point_fruit_ten import ready_state
from run_successor_observation import run as invoke
from inspect_forward_map import require,read_yaml,digest

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

def check(out,binary,review):
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'clean committed source required')
    r=read(review);require(r['disposition']=='pass' and r['source_revision']==rev and r['prospective_sha256']==digest(out/'freeze/prospective.json'),'exact source/prospective preflight review required')
    require(digest(binary)==r['binary_sha256'],'reviewed binary differs')
    require(rev[:9] in subprocess.check_output([str(binary),'--version'],text=True),'native/source revision mismatch')
    require(all(os.environ.get(n)=='1' for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']),'explicit one-thread numeric libraries required')
    prospective=read(out/'freeze/prospective.json')
    verified(prospective['dependencies']);verified(prospective['budget']);verified(prospective['generated_inputs'])
    for ref in [*read(out/'freeze/dependencies.json')['inputs'],*read(verified(prospective['generated_inputs']))]:verified(ref)
    for name in ('realized.json','shapes-manifest.json'):
        manifest=out/'freeze'/name
        if manifest.exists():
            for ref in read(manifest):verified(ref)


def call(out,binary,request,dest,terminal='point-fruit',workers=1,allowance=None):
    global _RESERVED
    budget=read(out/'freeze/budget.json');allowance=allowance or 48*budget['native_cells']
    with _LOCK:
        current=size(out);require(current+_RESERVED+allowance<=budget['peak_ceiling_bytes'],'reserved campaign storage ceiling')
        require(shutil.disk_usage(out).free>_RESERVED+allowance+budget['free_space_reserve_bytes'],'free-space reserve')
        if terminal=='point-fruit':record_attempt(out,request)
        _RESERVED+=allowance
    try:return invoke(binary,request,dest,workers,terminal)
    finally:
        with _LOCK:
            _RESERVED-=allowance
            with (out/'storage.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),stage=str(dest.relative_to(out)),bytes=size(out),reserved=_RESERVED,ceiling=budget['peak_ceiling_bytes']))+'\n')


def record_attempt(out,request):
    """Called under the reservation lock before invocation; failures still count."""
    q=read(request);owner_record=read(verified(q['owner_binding']));k=q['cycle_index']
    kind='feedback' if k else 'injected_initial'
    row=dict(kind=kind,policy=owner_record['policy'],branch=q['branch'],cycle=k,request=bound(request))
    path=out/'execution-attempts.jsonl';previous=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    require(not any((r['kind'],r['policy'],r['branch'],r['cycle'])==(kind,row['policy'],row['branch'],k) for r in previous),'duplicate logical application attempt')
    require(sum(r['kind']==kind for r in previous)<(150 if k else 3),'hard application attempt limit')
    with path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()


def parents(out):return read(out/'freeze/parents.json')

def upstream(out,binary):
    require(not (out/'UPSTREAM.json').exists() and not (out/'upstream').exists(),'no upstream retry')
    t=time.perf_counter();rc=call(out,binary,out/'inputs/observation.json',out/'upstream','ptc',2,read(out/'freeze/budget.json')['shared_upstream_allowance'])
    if rc:save_json(out/'UPSTREAM.json',dict(status='failed',return_code=rc,seconds=time.perf_counter()-t));return
    networks=[]
    for nw in (11,12):
        root=out/f'upstream/products/network{nw}';pr=read_yaml(root/'donor-continuity/ptc/receipt.yaml')
        require(pr['failed_fits']==0 and all(s['converged'] and s.get('application_failed_times',0)==0 for s in pr['segments']),'new observation initial PTC failure')
        networks.append(dict(root=str(root),input=bound(out/f'inputs/network{nw}-input.json'),receipt_sha256=digest(root/'receipt.yaml'),ptc_receipt_sha256=digest(root/'donor-continuity/ptc/receipt.yaml')))
    q=out/'inputs/map-request.json';save_json(q,dict(schema='citlali-map-saved-ptc-v1',owner_binding=bound(out/'inputs/map-owner.json'),pixel_size_arcsec=2.,coverage_cut=.5,networks=networks))
    require(call(out,binary,q,out/'M0','map')==0,'ordinary initial MAP failed')
    p=dict(map_request=bound(q),parent_plan=bound(out/'M0/products/a2000/plan.yaml'));save_json(out/'freeze/parents.json',p)
    c,g=initialize(out/'M0/products/a2000/map.fits',out/'freeze/a2000')
    save_json(out/'UPSTREAM.json',dict(status='accepted' if g['accepted'] else 'source_or_covariance_failed',guard=g,seconds=time.perf_counter()-t,initial_physical_inputs=1,parent=p))
    if c is not None:
        m=measurement(c,c['original'],c['support'],g);save_json(out/'freeze/M0-measurement.json',m)
        save_json(out/'freeze/realized.json',[bound(p) for p in sorted((out/'freeze/a2000').iterdir()) if p.is_file()]+[bound(out/'freeze/M0-measurement.json'),bound(out/'freeze/parents.json'),bound(out/'inputs/map-request.json')])


def owner(out,pol,branch,initial=None):
    p=read(out/'freeze/prospective.json');o=dict(identity='POINT-152430-NO-UPPER-COVERAGE-VETO-2026-09-26',outside_disk_no_feedback=True,direct_identical_operand_rejoin=True,gain=1,nonzero_cycles=10,policy=pol,source_method='POINT-EXPECTED-v1',threshold=3. if pol=='BASE-POINT' else p['thresholds'][pol],prospective=bound(out/'freeze/prospective.json'),thresholds=bound(out/'freeze/thresholds.json'),parents={'a2000':parents(out)},injections={'a2000':{b:bound(out/f'freeze/shapes/{b}.fits') for b in BRANCHES[1:4]}} if (out/'SHAPES.json').exists() else {})
    if branch.startswith('C-') and initial:
        o['injected_parents']={};o['injection_operands']={}
        for s in read_yaml(initial/'receipt.yaml')['segments']:
            old=s['source_segment'];stem=next(n[:-10] for n in old['files'] if n.endswith('-input.f64'));name=f"network-{old['network']}-{stem}"
            o['injected_parents'][name]=bound(initial/name/'injected-parent.f64');o['injection_operands'][name]=bound(initial/name/'injection.f64')
    return o


def request(out,op,pol,branch,k,cp,previous=None):
    require(pol in POLICIES and branch in BRANCHES and 0<=k<=10,'matrix outside bound')
    q=dict(schema='citlali-point-fruit-independent-v1',method='point-152430-no-upper-coverage-veto-v1',mode='zero' if k==0 else 'reference',array='a2000',branch=branch,cycle_index=k,candidate=bound(cp),owner_binding=bound(op),**parents(out))
    if k:q['previous_state']=bound(previous)
    if branch.startswith('C-'):q['injection']=bound(out/f'freeze/shapes/{branch}.fits')
    return q


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


def initial_injections(out,binary):
    c=context(out/'freeze/a2000');base=read(out/'run/BASE-POINT/REF/a2000/SUMMARY.json')
    require(base['ready'] and base['completed_cycle']==10,'BASE reference failure blocks shapes')
    d=out/'freeze/shapes';d.mkdir();ref0=load(verified(base['histories'][0]['candidate']))[0]
    rasters,info=shapes(base['histories'][-1]['measurement']['fit']['parameters'],c['major'],c['header'],ref0['VALID'].astype(bool))
    for b,z in rasters.items():write_fits(d/(b+'.fits'),exact_header(c['header']),[('INJECTION',z,'mJy/nominal beam'),('VALID',ref0['VALID'],'1')],companion='POINT-INDEPENDENT-INJECTION')
    save_json(out/'SHAPES.json',dict(source=base['histories'][-1]['total'],M0_major=c['major'],shapes=info,rasters={b:bound(d/(b+'.fits')) for b in rasters},frozen_before_other_policy_endpoints=True))
    save_json(out/'freeze/shapes-manifest.json',[bound(out/'SHAPES.json'),*[bound(p) for p in sorted(d.glob('*.fits'))]])
    for branch in BRANCHES[1:4]:
        try:initial_injection(out,binary,branch,base,c)
        except (ValueError,RuntimeError,KeyError,OSError) as exc:
            save_json(out/'initial'/branch/'INITIAL.json',dict(available=False,reason=str(exc)))

def initial_control(reference,injected,raster,c,dest):
    """M0 control for the aligned containing-pixel, uniform-gamma MAP.

    No fit or PTC: identical grid makes normalized gridding of P(I) equal I
    at each supported output pixel. T1–T10 additionally audit native streams.
    """
    ref,h=load(reference);actual,ah=load(injected);z,zh=load(raster)
    require(all(h[k]==ah[k] for k in ('CRPIX1','CRPIX2','CDELT1','CDELT2','CRVAL1','CRVAL2')),'M0 control grid mismatch')
    require(np.array_equal(ref['HITS'],ref['NORMALIZATION']),'M0 nonuniform coefficient')
    known=direct_raster(z['INJECTION'],zh,ref['PRIMARY'].shape,h)
    common=ref['SCI_SUPPORT'].astype(bool)&actual['SCI_SUPPORT'].astype(bool)
    delta=actual['PRIMARY']-ref['PRIMARY'];known[~ref['SCI_SUPPORT'].astype(bool)]=np.nan
    ks,vs,_,_,_=source_patch(known,ref['SCI_SUPPORT'].astype(bool),ref['NORMALIZATION'],h)
    ds,dv,_,_,_=source_patch(delta,common,ref['NORMALIZATION'],h)
    write_fits(dest/'response.fits',h,[('DELTA',delta,'mJy/nominal beam'),('CONTROL',known,'mJy/nominal beam'),('RESIDUAL',delta-known,'mJy/nominal beam'),('VALID',common.astype('u1'),'1')],companion='POINT-INDEPENDENT-M0-CONTROL')
    return dict(audit=dict(definition='aligned containing-pixel ordinary MAP of injection on original M0 support; uniform gamma=1, no PTC/centering',REF_counts_normalization_exact=True,native_stream_audit='performed separately for every T1–T10 control'),full=response_metrics(delta,known,common),source_disk=response_metrics(ds,ks,c['fixed'].disk&dv&vs))


def initial_injection(out,binary,branch,base,c):
        dest=out/'initial'/branch;dest.mkdir(parents=True);op=dest/'owner.json';save_json(op,owner(out,'BASE-POINT',branch));q=dest/'request.json';save_json(q,request(out,op,'BASE-POINT',branch,0,verified(base['histories'][0]['candidate'])))
        require(call(out,binary,q,dest/'native')==0,'injected M0 native failure')
        products=dest/'native/products';receipt=read_yaml(products/'receipt.yaml');guards=processing_guard(products,receipt,allow_historical_failure=False)
        s,v,_,_=patch_map(products/'total.fits');m,_=infer(c,s,v);response=initial_control(out/'M0/products/a2000/map.fits',products/'total.fits',out/f'freeze/shapes/{branch}.fits',c,dest);save_json(dest/'INITIAL.json',dict(available=True,total=bound(products/'total.fits'),receipt=bound(products/'receipt.yaml'),measurement=m,guards=guards,response=response))
        cleanup_cycle(out,products,keep_projection=False,keep_injected=True)


def execute_branch(out,binary,pol,branch):
    start=time.perf_counter();c=context(out/'freeze/a2000');dest=out/'run'/pol/branch/'a2000';dest.mkdir(parents=True);init=dest/'initial';init.mkdir()
    physical=out/'initial'/branch/'native/products' if branch.startswith('C-') else None
    if physical:
        r=read(out/'initial'/branch/'INITIAL.json');require(r['available'],r.get('reason','initial input unavailable'));total=verified(r['total']);receipt=verified(r['receipt']);m=r['measurement']
    else:
        total=out/'M0/products/a2000/map.fits';receipt=out/'M0/products/attempt.yaml';m=read(out/'freeze/M0-measurement.json')
    s,v,_,_=patch_map(total);g=m['guard'];b=candidate(s,v,c['fixed'],g);fields=c['neighborhood'].scores(b) if g['accepted'] and b['selected'].any() else None
    o=owner(out,pol,branch,physical);tau=o['threshold'];cp,summary=publish(init,c,s,(g,b,fields),pol,tau);cause,reason=continuation_reason(g,summary,load(cp)[0],c['header'])
    if branch=='D-seed-shoulder' and cause is None:
        own=read(out/'run'/pol/'REF/a2000/SUMMARY.json')['histories'][0];orig,_=load(verified(own['candidate']));current,_=load(cp)
        require(all(np.array_equal(orig[n],current[n],equal_nan=True) for n in ('CANDIDATE','VALID','ADMITTED','RATIO')),'seed not own REF candidate')
        z,_=load(out/'freeze/shapes/C-shoulder.fits');seed=seed_candidate(orig,branch,z['INJECTION']);cp=init/'seed.fits';write_fits(cp,exact_header(c['header']),[(n,seed[n],'mJy/nominal beam' if n=='CANDIDATE' else '1') for n in ('CANDIDATE','ADMITTED','VALID','RATIO')],companion='POINT-INDEPENDENT-SEED');cause,reason=continuation_reason(g,summary,seed,c['header'])
    o['seeds']={'a2000':{branch:bound(cp)}};op=dest/'owner.json';save_json(op,o)
    prior=ready_state('a2000',0,cp,total,receipt,parents(out),cause is None,reason);prior.update(branch=branch,policy=pol,owner_sha256=digest(op),measurement=m,admission=summary,stop_kind=cause);sp=init/'state.json';save_json(sp,prior);prior['state_path']=str(sp)
    if physical:prior['response']=r['response']
    history=[prior]
    for k in range(1,11):
        if not prior['ready']:break
        d=dest/f'cycle-{k:02d}';d.mkdir();q=d/'request.json';save_json(q,request(out,op,pol,branch,k,verified(prior['candidate']),Path(prior['state_path'])))
        rec=dict(array='a2000',policy=pol,branch=branch,cycle_index=k,ready=False);tic=time.perf_counter()
        try:
            require(call(out,binary,q,d/'native')==0,'native cycle failed')
            products=d/'native/products';rr=read_yaml(products/'receipt.yaml');s,v,_,_=patch_map(products/'total.fits');m,prepared=infer(c,s,v);cp,summary=publish(d,c,s,prepared,pol,tau)
            rec.update(total=bound(products/'total.fits'),receipt=bound(products/'receipt.yaml'),candidate=bound(cp),measurement=m,admission=summary)
            rec['guards']=processing_guard(products,rr,allow_historical_failure=False);rec['PTC_segments']=learned_state(products,rr)
            old,_=load(verified(prior['candidate']));new,_=load(cp);rec['candidate_turnover']=selection_difference(new,old)
            restored,rv,_,_=patch_map(products/'restored.fits');use=rv&old['VALID'].astype(bool);err=float(np.max(abs((restored-old['CANDIDATE'])[use])));require(err<1e-8,'operand remap differs');rec['guards']['operand_remap_max_abs']=err
            if branch.startswith('C-'):
                ref=read(out/'run'/pol/'REF/a2000/SUMMARY.json')['histories'][k];refprod=Path(ref['total']['path']).parent;z,zh=load(out/f'freeze/shapes/{branch}.fits');known,kv,kh,audit=injection_reference(refprod,products,z['INJECTION'],zh)
                refdata,_=load(ref['total']['path']);actual,ah=load(products/'total.fits');common=kv&actual['SCI_SUPPORT'].astype(bool);delta=actual['PRIMARY']-refdata['PRIMARY'];ks,vs,_,_,_=source_patch(known,kv,refdata['NORMALIZATION'],kh);ds,_,_,_,_=source_patch(delta,common,refdata['NORMALIZATION'],kh)
                rec['response']=dict(audit=audit,full=response_metrics(delta,known,common),source_disk=response_metrics(ds,ks,c['fixed'].disk&v&vs))
                write_fits(d/'response.fits',kh,[('DELTA',delta,'mJy/nominal beam'),('CONTROL',known,'mJy/nominal beam'),('RESIDUAL',delta-known,'mJy/nominal beam'),('VALID',common.astype('u1'),'1')],companion='POINT-INDEPENDENT-RESPONSE')
                if k==10:
                    rs,rsv,_,_=patch_map(ref['total']['path']);ideal,_=infer(c,rs+ks,rsv&vs);rec['ideal']=ideal_comparison(c,ref['measurement'],m,ideal)
            cause,reason=continuation_reason(prepared[0],summary,new,c['header']);rec.update(ready=cause is None,stop_kind=cause,stop_reason=reason)
            state=ready_state('a2000',k,cp,products/'total.fits',products/'receipt.yaml',parents(out),cause is None,reason);state.update(branch=branch,policy=pol,owner_sha256=digest(op));sp=d/'state.json';save_json(sp,state);rec['state_path']=str(sp)
            cleanup_cycle(out,products,keep_projection=branch=='REF')
        except (ValueError,RuntimeError,KeyError,OSError) as e:rec.update(stop_kind='computation_or_processing',stop_reason=str(e))
        rec['wall_seconds']=time.perf_counter()-tic;save_json(d/'cycle.json',rec);history.append(rec);prior=rec
        print(json.dumps(dict(policy=pol,branch=branch,cycle=k,ready=rec['ready'],reason=rec.get('stop_reason'))),flush=True)
    result=dict(array='a2000',policy=pol,branch=branch,ready=history[-1]['ready'],completed_cycle=history[-1]['cycle_index'],stop_reason=history[-1].get('stop_reason'),histories=history,actual_feedback_applications=len(history)-1,wall_seconds=time.perf_counter()-start,m10_applied=False)
    save_json(dest/'SUMMARY.json',result)
    if branch=='REF' and result['ready'] and result['completed_cycle']==10:
        save_json(dest/'ENDPOINT-ATTEMPT.json',dict(status='started',diagnostic_only=True,maximum_attempts=1))
        try:
            result['endpoint_uncertainty']=endpoint(verified(history[-1]['total']),c,history[-1]['measurement'],dest/'endpoint')
            result['endpoint_status']='completed'
        except (ValueError,RuntimeError,KeyError,OSError,np.linalg.LinAlgError) as exc:
            result['endpoint_status']='unavailable';result['endpoint_reason']=str(exc)
        save_json(dest/'ENDPOINT-ATTEMPT.json',dict(status=result['endpoint_status'],reason=result.get('endpoint_reason'),diagnostic_only=True,maximum_attempts=1))
        save_json(dest/'SUMMARY.json',result)
    return result


def execute(out,binary,phase):
    require(read(out/'UPSTREAM.json')['status']=='accepted','initial source guard unavailable')
    record=out/('PHASE-'+phase+'.json');require(not record.exists(),'no implicit phase retry');start=time.perf_counter();tasks=[]
    if phase=='base':tasks=[('BASE-POINT','REF')];workers=1
    else:
        profile=read(out/'PHASE-base.json');require(profile['results'][0]['ready'] and profile['results'][0]['completed_cycle']==10,'failed baseline reference')
        if phase=='shapes':
            save_json(record,dict(status='running'))
            try:
                initial_injections(out,binary);save_json(record,dict(status='completed',seconds=time.perf_counter()-start))
            except (ValueError,RuntimeError,KeyError,OSError) as exc:save_json(record,dict(status='failed',reason=str(exc)))
            return
        require((out/'SHAPES.json').exists(),'shapes must be frozen before other policy endpoints or branches')
        if phase=='references':tasks=[('CORR','REF'),('DIAG','REF')]
        elif phase=='responses':
            for pol in POLICIES:
                ref=out/'run'/pol/'REF/a2000/SUMMARY.json'
                if ref.exists() and read(ref)['ready'] and read(ref)['completed_cycle']==10:tasks.extend((pol,b) for b in BRANCHES[1:])
        budget=read(out/'freeze/budget.json');rss=profile['peak_family_rss_bytes'];workers=min(len(tasks),12,max(1,os.cpu_count()-2),max(1,int(psutil.virtual_memory().available//max(2*rss,1))))
    save_json(record,dict(status='running',tasks=tasks,workers=workers))
    stop=threading.Event();samples=[]
    def monitor():
        p=psutil.Process()
        while not stop.is_set():
            total=0
            for child in [p,*p.children(recursive=True)]:
                try:total+=child.memory_info().rss
                except psutil.Error:pass
            samples.append(total);stop.wait(.5)
    t=threading.Thread(target=monitor,daemon=True);t.start();results=[]
    try:
        with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
            futures={pool.submit(execute_branch,out,binary,*q):q for q in tasks}
            for f in as_completed(futures):
                try:r=f.result();results.append({k:r[k] for k in ('policy','branch','ready','completed_cycle','stop_reason','actual_feedback_applications','wall_seconds')})
                except Exception as e:results.append(dict(policy=futures[f][0],branch=futures[f][1],ready=False,stop_reason=str(e)))
    finally:stop.set();t.join()
    save_json(record,dict(status='completed',results=results,workers=workers,numeric_threads=1,seconds=time.perf_counter()-start,peak_family_rss_bytes=max(samples,default=0),final_bytes=size(out)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['upstream','base','shapes','references','responses']);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();check(a.output,a.binary,a.review)
    if a.stage=='upstream':
        try:upstream(a.output,a.binary)
        except (ValueError,RuntimeError,KeyError,OSError) as exc:
            if not (a.output/'UPSTREAM.json').exists():save_json(a.output/'UPSTREAM.json',dict(status='failed',reason=str(exc)))
            raise
    else:execute(a.output,a.binary,a.stage)
