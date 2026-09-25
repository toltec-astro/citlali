"""Freeze and execute the authorized 45-cell guarded POINT comparison only."""
import argparse,json,os,platform,shutil,subprocess,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import psutil,scipy,astropy
from point_frozen_shoulder import Auth,CD,CD_HASH
from run_point_corr_admission import read,context,ARRAYS,REPO,SNAPSHOTS
from run_point_response_cd import source,AB
from point_guarded_feedback import POLICIES,BRANCHES,infer,publish,continuation_reason
from point_response_cd import processing_guard,seed_candidate,injection_reference
from point_expected import METHOD
from point_fruit_trajectory import bound,verified,manifest,verify_manifest,selection_difference
from point_fruit_diagnostics import load,patch_map,learned_state
from run_point_fruit_ten import ready_state
from point_empirical_io import write_fits
from point_jinc_campaign import exact_header
from point_additive_diagnostics import save_json
from run_successor_observation import run as invoke
from inspect_forward_map import require,digest,read_yaml

V=REPO/'validation/point_152389_guarded_feedback_comparison_2026-09-25'
NATIVE_SCHEMA='citlali-point-fruit-guarded-v1'


def authenticated(out):
    p=read(out/'freeze/prospective.json')
    for name,sha in p['consumed_inputs'].items():require(digest(Path(name))==sha,'changed consumed input '+name)
    verify_manifest(out/'freeze');verify_manifest(out/'checks')
    return p


def ctx(p,a):return context(Path(p['correlation_root']),a)


def prepare(out,binary):
    start=time.perf_counter();rev=source();require(not out.exists(),'fresh output required')
    version=subprocess.check_output([str(binary),'--version'],text=True);require(rev[:9] in version,'exact native/source mismatch')
    auth=Auth()
    def ancestor(name):
        r=read(REPO/'validation'/name/'COMPLETION.json');root=Path(r['product_root']);auth.manifest(root,r['root_manifest']['sha256']);return root
    guardroot=ancestor('point_152389_expected_source_guard_2026-09-25')
    corr=ancestor('point_152389_correlation_admission_2026-09-24');auth.manifest(CD,CD_HASH)
    gp=auth.json(guardroot/'freeze/prospective.json');cp=auth.json(corr/'freeze/prospective.json');owner=cp['original_owner']
    for ancestor in owner['ancestors']:auth.manifest(ancestor['root'],ancestor['manifest']['sha256'])
    for path,sha in cp['consumed_inputs'].items():auth.bind(path,sha)
    for path,sha in gp['consumed_inputs'].items():auth.bind(path,sha)
    guardchecks=auth.json(guardroot/'checks/RESULTS.json');gates=auth.json(guardroot/'validation/screens.json')['policies'];thresholds=auth.json(guardroot/'calibration/thresholds.json')['thresholds']
    for n in ('FINAL_REVIEW.md','CLOSURE_REVIEW.md','CALIBRATION.json','VALIDATION.json','report/SUMMARY.json','screen/RESULTS.json'):auth.check(guardroot/n)
    out.mkdir();fr=out/'freeze';fr.mkdir();checks=out/'checks';checks.mkdir();shutil.copy2(V/'OWNER_DIRECTIVE.md',out/'OWNER_DIRECTIVE.md')
    p=dict(identity='POINT-152389-GUARDED-PROSPECTIVE-2026-09-25',source_revision=rev,native_source_revision=rev,binary=bound(binary),version=version,
        directive=bound(out/'OWNER_DIRECTIVE.md'),guard_root=str(guardroot),correlation_root=str(corr),historical_root=str(CD),original_owner=owner,
        thresholds=thresholds,threshold_file=auth.ref(guardroot/'calibration/thresholds.json'),launch_gate=auth.ref(guardroot/'validation/screens.json'),source_checks=auth.ref(guardroot/'checks/RESULTS.json'),
        policies=POLICIES,branches=BRANCHES,arrays=ARRAYS,cycles=10,logical_trajectories=45,maximum_feedback_applications=450,
        maximum_ideal_additions=27,maximum_endpoint_analyses=9,no_new_noise_fields=True,m10_unapplied=True,
        initial_REF_workers=3,target_workers=min(12,max(1,os.cpu_count()-2)),native_workers=1,numeric_threads=1,
        environment=dict(platform=platform.platform(),cpus=os.cpu_count(),memory_bytes=psutil.virtual_memory().total,available_bytes=psutil.virtual_memory().available,
            disk_free_bytes=shutil.disk_usage(out).free,python_executable=os.sys.executable,numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__),
        tolerances=owner['reference_tolerances'],accounting_atol=1e-8,ratio_floor_native_units=1e-8,
        methods=dict(source=METHOD,guard_unchanged=True,measurements='same guard fit plus original M0 template/apertures and raw profiles; signed maps never source-fitted',
            covariance='full frozen C0 for guard and CORR; DIAG diagonal only for supplement; endpoint fits never feed back',
            recurrence='immutable branch parent, complete replacement, direct subtract/fresh rank10 native PTC/identical restore/ordinary MAP',
            M0='12 authenticated original C+D physical-input M0 products shared before policy selection; no initial PTC/MAP rerun',
            reuse='M0 only; no new feedback numerical reuse unless separately proven exact'),
        originals={},M0={},injected_parents={},injection_operands={},source_code={})
    # Freeze unchanged scientific owners against the sealed guard code revision.
    for n in ('point_expected.py','inspect_point_source.py','point_corr_admission.py','point_additive.py'):
        path=REPO/'tools/timestream_successor'/n;old=subprocess.check_output(['git','show',gp['source_revision']+':tools/timestream_successor/'+n],cwd=REPO)
        require(path.read_bytes()==old,'frozen scientific owner changed: '+n);p['source_code'][n]=bound(path)
    control_rows=[];initial_rows={};interventions=[]
    for a in ARRAYS:
        c=context(corr,a);p['M0'][a]={};p['injected_parents'][a]={};p['injection_operands'][a]={};p['originals'][a]={}
        for n in ('inputs','geometry'):auth.check(cp['arrays'][a][n]['path'])
        for n in ('C0','C0_metadata','K0'):auth.check(cp['arrays'][a][n]['path'])
        for n in ('frozen-domains.npz','frozen-domains.json','M0/model.npz','M0/model.json','M0/comparison.json'):
            auth.check(AB/'companion'/a/n)
        for n in ('ASSESSMENT.md','FINAL_REVIEW.md'):auth.check(corr/n)
        auth.check(corr/f'run/BASE/REF/{a}/SUMMARY.json')
        require(guardchecks['repair'][a]['usable'] and guardchecks['repair'][a]['source_retention_check_pass'],'unusable source checks '+a)
        for pol in ('CORR','DIAG'):
            require(gates[a][pol]['comparison_usable'] and gates[a][pol]['threshold']==thresholds[a][pol],'unusable frozen policy '+a+pol)
        # Exact integrated replay of existing15guard fits and45saved policy masks.
        for saved in [r for r in guardchecks['bright'] if r['array']==a]:
            k=saved['stage'];signal,support,_,_=patch_map(auth.check(saved['total']['path']));measurement,prepared=infer(c,signal,support)
            require(prepared[0]==saved['result']['guard'],'integrated guarded fit differs')
            if k==0:ref_control=(signal.copy(),support.copy(),measurement)
            for pol in POLICIES:
                path,summary=publish(checks/f'{a}-{k:02d}-{pol}',c,signal,prepared,pol,3. if pol=='BASE-POINT' else thresholds[a][pol])
                old,_=auth.fits(guardroot/'screen'/f'{a}-{k:02d}-{pol}'/'selection.fits');new,_=load(path)
                for n in ('CANDIDATE','ADMITTED','VALID'):require(np.array_equal(old[n],new[n],equal_nan=True),'guarded policy replay differs '+n)
                control_rows.append(dict(array=a,cycle=k,policy=pol,exact=True,guard_accepted=prepared[0]['accepted'],candidate=bound(path)))
        for branch in BRANCHES:
            summary=auth.json(CD/'run'/branch/a/'SUMMARY.json');p['originals'][a][branch]=auth.ref(CD/'run'/branch/a/'SUMMARY.json')
            for state in summary['histories']:
                for key in ('total','candidate','receipt'):
                    if state.get(key):auth.check(state[key]['path'])
            if branch=='D-seed-shoulder':continue
            initial=summary['histories'][0];prod=Path(initial['total']['path']).parent
            # Reuse full native M0 input/output records, not a mask-equality proxy.
            prefix=str(prod.resolve())+os.sep
            for name in list(auth.expected):
                if name.startswith(prefix):auth.check(name)
            for name,sha in auth.yaml(prod/'input-manifest.yaml').items():auth.bind(name,sha);auth.check(name)
            receipt=auth.yaml(prod/'receipt.yaml');processing_guard(prod,receipt)
            p['M0'][a][branch]={n:initial[n] for n in ('total','receipt')}
            signal,support,_,_=patch_map(prod/'total.fits')
            # REF source equals existing checked map numerically; bind a single new
            # guard/measurement for each physical input, shared by policies.
            if branch=='REF':
                require(np.array_equal(signal,ref_control[0],equal_nan=True) and np.array_equal(support,ref_control[1]),'shared REF M0 numerical identity differs')
                measurement=ref_control[2]
            else:measurement,_=infer(c,signal,support)
            initial_rows[a,branch]=measurement
            p['M0'][a][branch]['measurement']=measurement
            if branch.startswith('C-'):
                auth.check(owner['injections'][a][branch]['path']);p['injected_parents'][a][branch]={};p['injection_operands'][a][branch]={}
                dest=fr/a/branch;dest.mkdir(parents=True)
                raster,header=auth.fits(owner['injections'][a][branch]['path'])
                refprod=Path(p['M0'][a]['REF']['total']['path']).parent
                _,_,_,audit=injection_reference(refprod,prod,raster['INJECTION'],header)
                for row in receipt['segments']:
                    seg=row['source_segment'];stem=next(n[:-10] for n in seg['files'] if n.endswith('-input.f64'));name=f"network-{seg['network']}-{stem}"
                    ypath=Path(row['source_root'])/(stem+'-input.f64');epath=Path(row['source_root'])/(stem+'-eligible.u8');ip=prod/name/'injection.f64'
                    for f in (ypath,epath,ip):auth.check(f)
                    y=np.fromfile(ypath,dtype='<f8');e=np.fromfile(epath,dtype='u1').astype(bool);pi=np.fromfile(ip,dtype='<f8')
                    require(y.shape==pi.shape==e.shape and np.all(pi[~e]==0) and np.isfinite(pi).all(),'native intervention shape/invalidity')
                    yi=y.copy();yi[e]+=pi[e];require(np.isfinite(yi[e]).all(),'nonfinite injected parent');fp=dest/(name+'-parent.f64');yi.tofile(fp)
                    p['injected_parents'][a][branch][name]=bound(fp);p['injection_operands'][a][branch][name]=auth.ref(ip)
                    interventions.append(dict(array=a,branch=branch,segment=name,parent=auth.ref(ypath),eligibility=auth.ref(epath),injection=auth.ref(ip),
                        injected_parent=bound(fp),invalid_unchanged=bool(np.array_equal(yi[~e].view('u8'),y[~e].view('u8'))),
                        arithmetic='copy y; add archived native P(I) exactly once on eligible entries',projection=audit))
            # Preserve original response domains/results for historical comparison.
            if branch.startswith('C-'):
                for k in range(11):auth.check(CD/'report'/a/branch/f'cycle-{k:02d}.fits')
        auth.check(CD/'report'/a/'results.json')
    save_json(checks/'RESULTS.json',dict(controls=control_rows,physical_M0_measurements={a:{b:initial_rows[a,b] for b in BRANCHES[:4]} for a in ARRAYS},interventions=interventions,
        initial_PTC_MAP_calculations=0,new_calibration_fields=0,guard_control_replays=15,logical_policy_controls=45))
    p['consumed_inputs']=auth.consumed;p['authentication_seconds']=auth.seconds;p['preparation_seconds']=time.perf_counter()-start
    save_json(fr/'prospective.json',p);manifest(fr);manifest(checks);auth.verify_all()
    save_json(out/'PREPARED.json',dict(source_revision=rev,prospective=bound(fr/'prospective.json'),checks=bound(checks/'RESULTS.json'),consumed_files=len(auth.consumed),wall_seconds=time.perf_counter()-start))
    print(json.dumps(read(out/'PREPARED.json')),flush=True)


def request(p,owner_path,a,pol,branch,k,prior):
    require(pol in POLICIES and branch in BRANCHES and 1<=k<=10,'guarded matrix bound')
    require(prior['ready'] and prior['cycle_index']==k-1,'stopped or foreign predecessor')
    r=dict(schema=NATIVE_SCHEMA,method='point-152389-guarded-feedback-v1',mode='reference',array=a,branch=branch,cycle_index=k,
        candidate=prior['candidate'],owner_binding=bound(owner_path),previous_state=bound(Path(prior['state_path'])),**p['original_owner']['parents'][a])
    if branch.startswith('C-'):r['injection']=p['original_owner']['injections'][a][branch]
    return r


def execute_branch(out,p,a,pol,branch):
    start=time.perf_counter();c=ctx(p,a);dest=out/'run'/pol/branch/a;dest.mkdir(parents=True)
    initial=p['M0'][a][branch if branch.startswith('C-') else 'REF'];init=dest/'initial';init.mkdir();signal,support,_,_=patch_map(verified(initial['total']))
    measurement=initial['measurement'];g=measurement['guard']
    from point_expected import candidate
    base=candidate(signal,support,c['fixed'],g);fields=c['neighborhood'].scores(base) if g['accepted'] and base['selected'].any() else None
    tau=3. if pol=='BASE-POINT' else p['thresholds'][a][pol];path,summary=publish(init,c,signal,(g,base,fields),pol,tau)
    cause,reason=continuation_reason(g,summary,load(path)[0],c['header'])
    if branch=='D-seed-shoulder' and cause is None:
        ownref=read(out/'run'/pol/'REF'/a/'SUMMARY.json')['histories'][0];old,_=load(verified(ownref['candidate']));current,_=load(path)
        for n in ('CANDIDATE','ADMITTED','VALID','RATIO'):require(np.array_equal(current[n],old[n],equal_nan=True),'seed not own REF m0')
        shoulder,_=load(verified(p['original_owner']['injections'][a]['C-shoulder']));seeded=seed_candidate(old,branch,shoulder['INJECTION']);path=init/'seed.fits'
        write_fits(path,exact_header(c['header']),[(n,seeded[n],'mJy/nominal beam' if n=='CANDIDATE' else '1') for n in ('CANDIDATE','ADMITTED','VALID','RATIO')],companion='POINT-GUARDED-SEED')
        cause,reason=continuation_reason(g,summary,seeded,c['header'])
    owner=dict(identity='POINT-152389-GUARDED-OWNER-2026-09-25',outside_disk_no_feedback=True,direct_identical_operand_rejoin=True,gain=1,nonzero_cycles=10,
        policy=pol,source_method=METHOD,threshold=tau,prospective=bound(out/'freeze/prospective.json'),source_checks=p['source_checks'],launch_gate=p['launch_gate'],thresholds=p['threshold_file'],
        parents=p['original_owner']['parents'],injections=p['original_owner']['injections'],seeds={a:{branch:bound(path)}},
        injected_parents=p['injected_parents'][a].get(branch,{}),injection_operands=p['injection_operands'][a].get(branch,{}))
    op=dest/'owner.json';save_json(op,owner)
    prior=ready_state(a,0,path,verified(initial['total']),verified(initial['receipt']),owner['parents'][a],cause is None,reason)
    prior.update(branch=branch,policy=pol,owner_sha256=bound(op)['sha256'],measurement=measurement,reused_M0=True,admission=summary,stop_kind=cause)
    sp=init/'state.json';save_json(sp,prior);prior['state_path']=str(sp);history=[prior]
    for k in range(1,11):
        if not prior['ready']:break
        d=dest/f'cycle-{k:02d}';d.mkdir();tic=time.perf_counter();rec=dict(array=a,policy=pol,branch=branch,cycle_index=k,ready=False)
        q=request(p,op,a,pol,branch,k,prior);save_json(d/'request.json',q)
        try:
            require(invoke(verified(p['binary']),d/'request.json',d/'native',1,'point-fruit')==0,'native cycle failed')
            products=d/'native/products';receipt=read_yaml(products/'receipt.yaml');signal,support,_,_=patch_map(products/'total.fits');dt=time.perf_counter()
            rec.update(total=bound(products/'total.fits'),receipt=bound(products/'receipt.yaml'),request=bound(d/'request.json'))
            measurement,prepared=infer(c,signal,support);path,summary=publish(d,c,signal,prepared,pol,tau)
            rec.update(measurement=measurement,admission=summary,candidate=bound(path));save_json(d/'cycle.json',rec)
            rec['guards']=processing_guard(products,receipt);rec['PTC_segments']=learned_state(products,receipt)
            old,_=load(verified(prior['candidate']));new,_=load(path);rec['candidate_turnover']=selection_difference(new,old)
            restore,rv,_,_=patch_map(products/'restored.fits');use=rv&old['VALID'].astype(bool);err=float(np.max(abs((restore-old['CANDIDATE'])[use])));require(err<1e-8,'applied operand remap mismatch');rec['guards']['operand_remap_max_abs']=err
            if branch.startswith('C-'):
                for row in receipt['segments']:
                    seg=row['source_segment'];stem=next(n[:-10] for n in seg['files'] if n.endswith('-input.f64'));name=f"network-{seg['network']}-{stem}"
                    require(row['files']['injection.f64']==owner['injection_operands'][name]['sha256'],'injection changed or accumulated')
            cause,reason=continuation_reason(prepared[0],summary,new,c['header'])
            state=ready_state(a,k,path,products/'total.fits',products/'receipt.yaml',owner['parents'][a],cause is None,reason);state.update(branch=branch,policy=pol,owner_sha256=bound(op)['sha256'])
            sp=d/'state.json';save_json(sp,state);rec.update(ready=cause is None,stop_kind=cause,stop_reason=reason,state_path=str(sp),diagnostic_seconds=time.perf_counter()-dt)
        except (ValueError,RuntimeError,KeyError,OSError) as exc:rec.update(stop_kind='computational_or_processing_blocker',stop_reason=str(exc))
        rec['wall_seconds']=time.perf_counter()-tic;save_json(d/'cycle.json',rec);history.append(rec);prior=rec
        print(json.dumps(dict(array=a,policy=pol,branch=branch,cycle=k,ready=rec['ready'],reason=rec.get('stop_reason'))),flush=True)
    result=dict(array=a,policy=pol,branch=branch,ready=history[-1]['ready'],completed_cycle=history[-1]['cycle_index'],stop_reason=history[-1].get('stop_reason'),histories=history,wall_seconds=time.perf_counter()-start,m10_applied=False,actual_feedback_applications=len(history)-1)
    save_json(dest/'SUMMARY.json',result);manifest(dest);return result


def execute(out,phase,review):
    revision=source();p=authenticated(out);rv=read(review)
    require(rv['disposition']=='pass' and rv['source_revision']==revision and rv['prospective_sha256']==digest(out/'freeze/prospective.json'),'exact source/preflight review required')
    require(all(os.environ.get(n)=='1' for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'one numeric thread required')
    require(not (out/f'FEEDBACK-{phase}.json').exists(),'no phase retry')
    policies=('BASE-POINT',) if phase=='profile' else ('CORR','DIAG') if phase=='REF' else POLICIES
    branches=('REF',) if phase in ('profile','REF') else BRANCHES[1:3] if phase=='compact' else BRANCHES[3:]
    if phase!='profile':require((out/'FEEDBACK-profile.json').exists(),'initial REF resource profile required')
    if phase in ('compact','shoulder'):require((out/'FEEDBACK-REF.json').exists(),'REF phase required')
    if phase=='shoulder':require((out/'FEEDBACK-compact.json').exists(),'compact pair precedes shoulder/seed')
    tasks=[];blocked=[]
    for b in branches:
        for pol in policies:
            for a in ARRAYS:
                ref=out/'run'/pol/'REF'/a/'SUMMARY.json'
                if b!='REF' and (not ref.exists() or not read(ref)['ready'] or read(ref)['completed_cycle']!=10):blocked.append(dict(array=a,policy=pol,branch=b,status='blocked',cause='own policy REF incomplete'))
                else:tasks.append((a,pol,b))
    workers=min(3 if phase=='profile' else p['target_workers'],max(1,len(tasks)))
    if phase!='profile':
        resource_check=read(out/'RESOURCE_DISPOSITION.json');require(resource_check['proceed'],'resource disposition unavailable');workers=min(workers,resource_check['workers'])
    started=time.perf_counter();stop=threading.Event();rss=[]
    def monitor():
        parent=psutil.Process()
        while not stop.is_set():
            total=0
            for child in [parent,*parent.children(recursive=True)]:
                try:total+=child.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            rss.append(total);stop.wait(.5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();results=[]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fs={pool.submit(execute_branch,out,p,*t):t for t in tasks}
            for f in as_completed(fs):
                try:r=f.result();results.append({k:r[k] for k in ('array','policy','branch','ready','completed_cycle','stop_reason','wall_seconds','actual_feedback_applications')})
                except Exception as exc:
                    a,pol,b=fs[f];blocked.append(dict(array=a,policy=pol,branch=b,status='execution_exception',cause=type(exc).__name__+': '+str(exc)))
    finally:stop.set();watcher.join()
    auth_start=time.perf_counter();authenticated(out)
    save_json(out/f'FEEDBACK-{phase}.json',dict(source_revision=revision,native_source_revision=p['native_source_revision'],native_binary=p['binary'],review=bound(review),results=results,blocked=blocked,
        workers=workers,threads_per_native=1,requested_numeric_threads=1,actual_library_threads='Apple Accelerate count not observable',sampled_aggregate_RSS_bytes=max(rss,default=0),rss_sampling_seconds=.5,
        wall_seconds=time.perf_counter()-started,final_authentication_seconds=time.perf_counter()-auth_start,actual_feedback_applications=sum(r['actual_feedback_applications'] for r in results),new_M0_calculations=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','profile','REF','compact','shoulder']);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--binary',type=Path);ap.add_argument('--review',type=Path);a=ap.parse_args()
    prepare(a.output,a.binary) if a.stage=='prepare' else execute(a.output,a.stage,a.review)
