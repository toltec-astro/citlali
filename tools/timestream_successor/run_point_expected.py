"""Bounded POINT-EXPECTED-v1 checks and replay of existing admission fields."""
import argparse,json,resource,shutil,time,os,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from point_expected import infer_expected, METHOD
from point_admission import fit_diagnostic
from point_corr_admission import select,maximum,threshold,binomial_upper,trial_seed,POLICIES,base_from_fit
from point_additive import draw_noise
from run_point_corr_admission import context,read,source,ARRAYS,SNAPSHOTS,REPO
from point_fruit_trajectory import bound,verified,manifest,verify_manifest
from point_fruit_diagnostics import patch_map,load
from point_empirical_io import write_fits
from point_additive_diagnostics import save_json
from point_jinc_campaign import exact_header
from inspect_forward_map import require,digest

V=REPO/'validation/point_152389_expected_source_guard_2026-09-25'


def inputs(out):
    p=read(out/'freeze/prospective.json')
    for path,sha in p['consumed_inputs'].items():require(digest(Path(path))==sha,'changed consumed input: '+path)
    return p


def ctx(out,a):
    p=read(out/'freeze/prospective.json')
    return context(Path(p['parent_root']),a)


def outcome(c,data,tau=None,source_mask=None):
    """Share one fit between all policies; rejection is a known empty scene."""
    guard,base,fields=infer_expected(data,c['support'],c['fixed'],c['neighborhood'],c['scale'],c['psd'])
    valid=guard['complete'];false=base['selected'] if source_mask is None else base['selected']&~source_mask
    rec=dict(guard=guard,numerically_valid=valid,BASE_POINT_pixels=int(base['selected'].sum()) if valid else None,
        BASE_POINT_false_pixels=int(false.sum()) if valid else None,
        BASE_POINT_false_brightness_area=float(np.sum(base['candidate'][false])*4) if valid else None,
        policies={})
    for pol in POLICIES:
        nbad=fields['policies'][pol]['numerical_failures'] if fields else 0
        known=valid and nbad==0
        r=dict(max_score=maximum(base,fields,pol) if known else None,maximum_known=known,numerical_failures=nbad)
        t=None if tau is None else tau[pol]
        if known and t is not None:
            selected=select(base,fields,c['k0'],pol,t)
            r.update(added_pixels=int(selected['added'].sum()),extra_brightness_area=float(np.sum(selected['candidate'][selected['added']])*4),total_pixels=int(selected['selected'].sum()),thresholded_available=True)
        else:r.update(added_pixels=None,extra_brightness_area=None,total_pixels=None,thresholded_available=False)
        rec['policies'][pol]=r
    return rec,(guard,base,fields)


def compare(guard,base,oldfit,oldcandidate,data):
    r=dict(old_status=oldfit['status'],new_status=guard['status'],historical=oldfit,
           new_candidate_pixels=int(base['selected'].sum()),historical_candidate_pixels=int(oldcandidate['ADMITTED'].sum()),
           added_vs_historical=int(np.sum(base['selected']&~oldcandidate['ADMITTED'].astype(bool))),
           removed_vs_historical=int(np.sum(oldcandidate['ADMITTED'].astype(bool)&~base['selected'])))
    if guard['diagnostic'] is not None and oldfit['status']=='success':
        p=np.array(guard['diagnostic']['parameters']);q=np.array(oldfit['parameters'])
        r.update(peak_delta=float(p[0]-q[0]),centroid_delta_arcsec=(p[1:3]-q[1:3]).tolist(),
                 sigmas_delta_arcsec=(np.sort(p[3:5])-np.sort(q[3:5])).tolist(),constant_delta=float(p[6]-q[6]))
        valid=base['valid']&oldcandidate['VALID'].astype(bool)
        r['measured_brightness_operand_max_delta']=float(np.max(abs((data-p[6]-(data-q[6]))[valid])))
        r['candidate_brightness_max_delta']=float(np.max(abs((base['candidate']-oldcandidate['CANDIDATE'])[valid])))
    return r


def publish(dest,c,data,prepared,pol,tau):
    dest.mkdir(parents=True);guard,base,fields=prepared
    result=base if pol=='BASE-POINT' else select(base,fields,c['k0'],pol,tau) if tau is not None and guard['complete'] else None
    planes=[('MEASURED',data,'mJy/nominal beam'),('BASE_POINT',base['selected'].astype('u1'),'1'),
        ('K0',c['k0'].astype('u1'),'1'),('VALID',base['valid'].astype('u1'),'1'),('BRIGHTNESS',base['brightness'],'mJy/nominal beam')]
    if result is not None:
        planes.extend([('CANDIDATE',result['candidate'],'mJy/nominal beam'),('ADMITTED',result['selected'].astype('u1'),'1')])
        if pol!='BASE-POINT':planes.extend([('PROPOSED',result['proposed_extra'].astype('u1'),'1'),('ADDED',result['added'].astype('u1'),'1')])
    if fields is not None:
        planes.extend([('NEFF_W',fields['neff_weights'],'1'),('EVIDENCE',fields['evidence'],'mJy/nominal beam')])
        for name in POLICIES:
            f=fields['policies'][name];planes.extend([(name+'_SCORE',f['score'],'1'),(name+'_VAR',f['variance'],'(mJy/nominal beam)^2'),(name+'_VALID',f['available'].astype('u1'),'1')])
    write_fits(dest/'selection.fits',exact_header(c['header']),planes,companion='POINT-EXPECTED-UNAPPLIED')
    save_json(dest/'selection.json',dict(method=METHOD,guard=guard,policy=pol,threshold=tau,applied=False,
        candidate_summary=None if result is None else result['summary'],source_mask_definition='BASE-POINT threshold-3 source-connected mask; empty on source rejection',
        screen_status='unavailable' if result is None else 'complete' if guard['complete'] else 'computationally unavailable with outward empty product'))
    return bound(dest/'selection.fits')


def prepare(out):
    rev=source();require(not out.exists(),'fresh output required');out.mkdir();dest=out/'freeze';dest.mkdir();start=time.perf_counter();consumed={}
    def bind(path,sha):
        path=Path(path);require(digest(path)==sha,'changed bound original: '+str(path));consumed[str(path)]=sha;return path
    def ancestor(name):
        comp=read(REPO/'validation'/name/'COMPLETION.json');root=Path(comp['product_root']);mf=bind(root/'FILES_SHA256SUMS',comp['root_manifest']['sha256'])
        entries={n:s for s,n in (line.split('  ',1) for line in mf.read_text().splitlines())}
        return root,entries
    parent,idx=ancestor('point_152389_correlation_admission_2026-09-24')
    original=read(bind(parent/'freeze/prospective.json',idx['freeze/prospective.json']))
    for a in ARRAYS:
        for key in ('inputs','geometry'):
            ref=original['arrays'][a][key];bind(ref['path'],ref['sha256'])
        for k in SNAPSHOTS:
            state=original['original_owner']['original_cycles'][a][k]
            for key in ('total','candidate'):bind(state[key]['path'],state[key]['sha256'])
        for name in (f'run/BASE/REF/{a}/SUMMARY.json',f'calibration/{a}-trials.json',f'validation/{a}-trials.json'):
            bind(parent/name,idx[name])
    diag,didx=ancestor('point_152389_blank_fit_diagnosis_2026-09-25')
    diagnosed=read(bind(diag/'CASES.json',didx['CASES.json']))
    shutil.copy2(V/'OWNER_DIRECTIVE.md',out/'OWNER_DIRECTIVE.md')
    prospective=dict(method=METHOD,source_revision=rev,parent_root=str(parent),original_prospective=bound(parent/'freeze/prospective.json'),
        directive=bound(out/'OWNER_DIRECTIVE.md'),consumed_inputs=consumed,workers=12,numeric_threads=1,
        original_C0_K0_means_unchanged=True,validation_status='replayed previously examined fields, not untouched holdout',
        sigma_floor_arcsec=1.5,SNR_strict_min=10,centroid_inclusive_radius_arcsec=20,
        amplitude_uncertainty='W=I, analytic full seven-column J, SVD amplitude row of J+, q^T frozen full C0 q via existing FFT contraction; all nuisance parameters retained even at bounds',
        SVD_rank_tolerance='eps*max(J.shape)*largest singular value',identifiability_tolerance='100*eps*max(J.shape) on amplitude null projection',
        expected_origin='original AST tangent-plane origin (0,0), sky/grid origin retained per array header',
        no_new_trajectories=True,independent_new_fields=0,calibration_reused_fields=3072,validation_reused_fields=3072,stress_reuses=768)
    save_json(dest/'prospective.json',prospective);manifest(dest)
    checks=out/'checks';checks.mkdir();cases=[];bright=[];controls={};repair={}
    for a in ARRAYS:
        c=context(parent,a)
        for row in [r for r in diagnosed if r['array']==a]:
            noise=draw_noise(c['psd'],c['scale'],int(row['seed']));data=np.where(c['support'],c[row['scene']]+noise,np.nan)
            oldfit=fit_diagnostic(data,c['support'],c['fixed'].header,c['fixed'].beam)
            require(oldfit==row['original']['outcome'],'historical diagnosis replay changed')
            result,prepared=outcome(c,data)
            cases.append(dict(array=a,trial=row['trial'],scene=row['scene'],role=row['role'],old=oldfit,new=result,historical_exact=True))
            publish(checks/f"{a}-case-{row['scene']}-{row['trial']:04d}",c,data,prepared,'BASE-POINT',3.)
        for k in SNAPSHOTS:
            state=original['original_owner']['original_cycles'][a][k];data,support,_,_=patch_map(verified(state['total']))
            oldfit=fit_diagnostic(data,support,c['fixed'].header,c['fixed'].beam)
            require(oldfit==state['measurement']['fit'],'historical saved-map fit replay changed')
            old,_=load(verified(state['candidate']));historical=base_from_fit(data,support,c['fixed'],oldfit)
            for key,field in [('CANDIDATE','candidate'),('ADMITTED','selected'),('VALID','valid'),('RATIO','z')]:
                require(np.array_equal(old[key],historical[field],equal_nan=True),'historical candidate changed')
            result,prepared=outcome(c,data);comparison=compare(prepared[0],prepared[1],oldfit,old,data)
            record=dict(array=a,stage=k,result=result,comparison=comparison,total=state['total'],historical_candidate=state['candidate'],historical_exact=True)
            record['product']=publish(checks/f'{a}-REF-{k:02d}',c,data,prepared,'BASE-POINT',3.);bright.append(record)
        controls[a]={}
        for name in ('blank','core'):
            data=np.where(c['support'],c[name],np.nan);result,prepared=outcome(c,data)
            controls[a][name]=result;publish(checks/f'{a}-noiseless-{name}',c,data,prepared,'BASE-POINT',3.)
        usable=original['arrays'][a]['covariance_preflight']['pass'] and all(r['result']['guard']['complete'] for r in bright if r['array']==a)
        passed=usable and all(r['result']['guard']['accepted'] for r in bright if r['array']==a) and controls[a]['core']['guard']['accepted'] and controls[a]['blank']['guard']['complete'] and not controls[a]['blank']['guard']['accepted']
        repair[a]=dict(usable=bool(usable),source_retention_check_pass=bool(passed))
        print(json.dumps(dict(array=a,repair=repair[a],core_SNR=controls[a]['core']['guard']['SNR_A'])),flush=True)
    save_json(checks/'RESULTS.json',dict(cases=cases,bright=bright,noiseless=controls,repair=repair,source_revision=rev));manifest(checks)
    inputs(out);save_json(out/'PREPARED.json',dict(source_revision=rev,prospective=bound(dest/'prospective.json'),checks=bound(checks/'RESULTS.json'),wall_seconds=time.perf_counter()-start,consumed_files=len(consumed)))


_CONTEXT=None


def initialize(out,a):
    global _CONTEXT
    _CONTEXT=ctx(Path(out),a)


def safe_outcome(c,data,tau,mask):
    try:return outcome(c,data,tau,mask)[0]
    except Exception as exc:
        # Explicit computational error, never a completed rejected source.
        return dict(numerically_valid=False,guard=dict(status='unavailable',accepted=False,complete=False,cause=str(exc),diagnostic=None,sigma_A=None,SNR_A=None),
            error_type=type(exc).__name__,traceback=traceback.format_exc(),BASE_POINT_pixels=None,BASE_POINT_false_pixels=None,BASE_POINT_false_brightness_area=None,
            policies={pol:dict(max_score=None,maximum_known=False,numerical_failures=1,added_pixels=None,extra_brightness_area=None,total_pixels=None,thresholded_available=False) for pol in POLICIES})


def trial(payload):
    a,phase,i,tau=payload;c=_CONTEXT;start=time.perf_counter();cpu=time.process_time();seed=trial_seed(a,phase,i)
    noise=draw_noise(c['psd'],c['scale'],seed)
    rec=dict(array=a,phase=phase,trial=i,seed=str(seed),scenes={})
    for name in ('blank','core'):
        data=np.where(c['support'],c[name]+noise,np.nan)
        rec['scenes'][name]=safe_outcome(c,data,tau,c['k0'] if name=='core' else None)
    rec['maxima_known']={pol:all(s['policies'][pol]['maximum_known'] for s in rec['scenes'].values()) for pol in POLICIES}
    rec['maxima']={pol:max([s['policies'][pol]['max_score'] for s in rec['scenes'].values() if s['policies'][pol]['max_score'] is not None],default=None) if rec['maxima_known'][pol] else None for pol in POLICIES}
    if phase=='validation' and i<256:
        rec['stress']={name:safe_outcome(c,np.where(c['support'],c[name]+1.25*noise,np.nan),tau,c['k0'] if name=='core' else None) for name in ('blank','core')}
    rec.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rec


def simulate(out,phase,review):
    rev=source();p=inputs(out);rv=read(review);verify_manifest(out/'freeze');verify_manifest(out/'checks')
    require(rv['disposition']=='pass' and rv['source_revision']==rev and rv['prospective_sha256']==digest(out/'freeze/prospective.json'),'exact-source/input independent review required')
    require(all(os.environ.get(n)=='1' for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'one numerical thread required')
    dest=out/phase;dest.mkdir();start=time.perf_counter();allrows={};taus=read(out/'calibration/thresholds.json')['thresholds'] if phase=='validation' else None
    if phase=='validation':verify_manifest(out/'calibration')
    repairs=read(out/'checks/RESULTS.json')['repair'];unrun={}
    for a in ARRAYS:
        if not repairs[a]['usable']:
            unrun[a]='input/covariance or computational check unavailable';continue
        rows=[]
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=initialize,initargs=(str(out),a)) as pool:
            for row in pool.map(trial,[(a,phase,i,None if taus is None else taus[a]) for i in range(1024)],chunksize=8):rows.append(row)
        save_json(dest/(a+'-trials.json'),rows);allrows[a]=rows
        print(json.dumps(dict(phase=phase,array=a,trials=len(rows),accepted_core=sum(r['scenes']['core']['guard']['accepted'] for r in rows),computational_errors=sum(not s['numerically_valid'] for r in rows for s in r['scenes'].values()))),flush=True)
    result=dict(source_revision=rev,review=bound(review),prospective=bound(out/'freeze/prospective.json'),wall_seconds=time.perf_counter()-start,
        reused_fields=sum(len(r) for r in allrows.values()),independent_new_fields=0,validation_status=p['validation_status'],
        workers=p['workers'],requested_numeric_threads=1,actual_library_threads='Apple Accelerate internal count not observable',
        CPU_seconds=sum(r['cpu_seconds'] for rows in allrows.values() for r in rows),
        maximum_worker_RSS_bytes=max((r['peak_RSS_bytes'] for rows in allrows.values() for r in rows),default=0),unrun=unrun)
    if phase=='calibration':
        ts={a:{pol:threshold([r['maxima'][pol] for r in rows]) if all(r['maxima_known'][pol] for r in rows) else None for pol in POLICIES} for a,rows in allrows.items()}
        save_json(dest/'thresholds.json',dict(thresholds=ts,order_statistic_one_based=1015,frozen_before_validation=True,empty_maximum_serialization='null represents minus infinity only when maximum_known is true'))
        result['thresholds']=bound(dest/'thresholds.json')
    else:
        screens={}
        for a,rows in allrows.items():
            c=ctx(out,a);controls={name:outcome(c,np.where(c['support'],c[name],np.nan),taus[a],c['k0'] if name=='core' else None)[0] for name in ('blank','core')};screens[a]={}
            for pol in POLICIES:
                tau=taus[a][pol];complete=all(r['maxima_known'][pol] for r in rows)
                k=sum(r['maxima'][pol] is not None and r['maxima'][pol]>tau for r in rows) if complete and tau is not None else None
                u=binomial_upper(k) if k is not None else None
                final=sum(any(s['policies'][pol]['added_pixels']>0 for s in r['scenes'].values()) for r in rows) if u is not None else None
                clean=all(s['numerically_valid'] and s['policies'][pol]['added_pixels']==0 for s in controls.values()) if tau is not None else None
                screens[a][pol]=dict(threshold=tau,complete_outcomes=complete,preconnectivity_exceedances=k,upper95=u,
                    final_supplemental_false_events=final,noiseless_no_supplement=clean,
                    statistical_screen_pass=bool(u is not None and u<=.02 and clean),source_retention_check_pass=repairs[a]['source_retention_check_pass'],
                    comparison_usable=bool(u is not None and u<=.02 and clean and repairs[a]['source_retention_check_pass']),
                    accepted_core=sum(r['scenes']['core']['guard']['accepted'] for r in rows),core_misses=sum(r['scenes']['core']['guard']['complete'] and not r['scenes']['core']['guard']['accepted'] for r in rows),
                    no_feedback_authorized=True)
        save_json(dest/'screens.json',dict(policies=screens))
    inputs(out);manifest(dest);save_json(out/(phase.upper()+'.json'),result)


def screen(out):
    rev=source();p=inputs(out);verify_manifest(out/'calibration');verify_manifest(out/'validation')
    ts=read(out/'calibration/thresholds.json')['thresholds'];checks=read(out/'checks/RESULTS.json');dest=out/'screen';dest.mkdir();rows=[]
    for saved in checks['bright']:
        a=saved['array'];k=saved['stage'];c=ctx(out,a);data,support,_,_=patch_map(verified(saved['total']))
        # Reuse the already-executed new guard on exactly this authenticated map.
        from point_expected import candidate
        g=saved['result']['guard'];b=candidate(data,support,c['fixed'],g);fields=c['neighborhood'].scores(b) if g['accepted'] and b['selected'].any() else None
        for pol in ('BASE-POINT',*POLICIES):
            tau=3. if pol=='BASE-POINT' else ts.get(a,{}).get(pol)
            product=publish(dest/f'{a}-{k:02d}-{pol}',c,data,(g,b,fields),pol,tau)
            rows.append(dict(array=a,stage=k,policy=pol,product=product,threshold=tau,guard_reused=True,source_accepted=g['accepted'],source_retention_check_pass=checks['repair'][a]['source_retention_check_pass']))
    save_json(dest/'RESULTS.json',dict(source_revision=rev,rows=rows,new_fits=0,applied=False));manifest(dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','calibration','validation','screen']);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--review',type=Path);args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='screen':screen(args.output)
    else:simulate(args.output,args.stage,args.review)
