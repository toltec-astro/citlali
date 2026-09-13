"""All saved-pair projections, then exactly six authorized one-step branches."""
import datetime
import os
import platform
import resource
import signal
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from common import b,candidate,read,HERE,ROOT,REPAIR,PRIOR,SAVED,SOLUTIONS,OLD,OUT,initialize,state_rows,models,source_core,evaluation,verify_freeze,array_hash
from diagnostics import image_problem,projection_groups,select_states,subspace_distance,energy

def guard(start):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    elapsed=time.monotonic()-start
    if elapsed>3600 or rss>8*2**30 or size>4*2**30:
        raise RuntimeError('registered time/memory/output limit')
    return dict(wall_seconds=elapsed,peak_rss_bytes=rss,output_bytes=size)

def screen(data,estimators,states,pairs,start):
    dest=OUT/'screen';dest.mkdir(exist_ok=False)
    records=[]
    for row in states:
        case,k=row['case'],row['pass_index']
        name=f'{case}_pass{k:02d}'
        n,t,present=models(data,row,pairs)
        with np.load(SAVED/case/'C'/f'pass{k:02d}_maps.npz') as z:
            total=z['total'];original_selected=z['selected']
        core=np.array([source_core(data,e,row['measurement'][a]) for a,e in enumerate(estimators)])
        pn=data.project(n);pt=data.project(t);delta=pt-pn
        projected_difference=data.project(t-n)
        np.testing.assert_allclose(delta,projected_difference,rtol=1e-12,atol=1e-12)
        del projected_difference,pt
        crossings=data.project(core.astype(float))
        waveforms={}
        with np.load(SAVED/case/'C'/f'pass{k:02d}_state.npz') as previous_state:
            for a,e in enumerate(estimators):
                rec=dict(case=case,pass_index=k,array=b.ARRAYS[a],available=present[a])
                if present[a]:
                    pair=pairs[case,k,b.ARRAYS[a]]
                    img=image_problem(data,e,total[a],n[a],t[a],row['measurement'][a],pair)
                    np.testing.assert_array_equal(img.pop('selected'),original_selected[a])
                    projected,groups,wave=projection_groups(data,a,delta,pn,crossings,previous_state)
                    np.testing.assert_allclose(projected['difference_energy'],float(np.sum(data.Q[a]*(t[a]-n[a])**2)),rtol=1e-12,atol=1e-10)
                    rec.update(image=img,projection=projected,groups=groups,projector_linearity_verified=True)
                    waveforms.update(wave)
                else:
                    rec['reason']='saved_tighter_solution_unavailable; no substitute'
                records.append(rec)
        np.savez_compressed(dest/(name+'_waveforms.npz'),**waveforms)
        np.savez_compressed(dest/(name+'_difference.npz'),difference=t-n,pair_available=present,source_core=core)
        print('SCREEN',name,'pairs',sum(present),flush=True)
        guard(start)
    assert len(records)==144 and sum(r['available'] for r in records)==141
    b.write(dest/'PAIRS.json',records)
    selected,ranking=select_states(records)
    bindings=[]
    for state in selected:
        case,k=state['case'],state['pass_index']
        paths=[SAVED/case/'C'/f'pass{k:02d}_maps.npz',SAVED/case/'C'/f'pass{k:02d}_state.npz',SAVED/case/'C'/'START.json']
        paths += [SOLUTIONS/'stopping'/(pairs[case,k,array]['name']+'.npz') for array in b.ARRAYS]
        bindings.append(dict(**state,files=[dict(path=str(p),sha256=b.digest(p)) for p in paths]))
    b.write(OUT/'SELECTED_STATES.json',dict(states=bindings,ranking=[dict(source_core_difference_energy=score,case=key[0],pass_index=key[1]) for score,key in ranking],
        selected_before_cleaning=True,cleaning_calls_at_selection=0,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('SELECTED',selected,flush=True)
    return selected,records

def parent_for(data,case):
    if case=='real123424':
        return data.y,None,{}
    state,seed=case.rsplit('_',1)
    with np.load(OLD/'geometry.npz') as z:
        scales=z['scales']
    noise=b.nuisance(data,scales,int(seed))
    with np.load(SAVED/(state+'_truth.npz')) as z:
        truth=z['truth']
    parent=noise+data.project(truth)
    if state=='T':
        parent[:,data.ar==1] += -.2*noise[:,data.ar==1]
    return parent,truth,read(PRIOR/'CASES.json')['states'][state]

def branch(data,estimators,parent,model,truth,definition,dest):
    dest.mkdir(exist_ok=False)
    start=time.monotonic()
    tod,state,rank=data.clean(parent,model)
    clean_seconds=time.monotonic()-start
    t=time.monotonic();maps=data.grid(tod);map_seconds=time.monotonic()-t
    del tod
    t=time.monotonic();measure=evaluation.measure(data,maps,estimators)
    old_measure=[m['original'] for m in measure]
    scores=None if truth is None else evaluation.old_evaluation.score_truth(data,maps,old_measure,truth,definition)
    evaluation_seconds=time.monotonic()-t
    np.savez_compressed(dest/'MAPS.npz',total=maps,applied_model=model)
    np.savez_compressed(dest/'LEARNED_STATE.npz',**state)
    result=dict(measurements=measure,truth_scores=scores,minimum_application_rank_ratio=rank,
        clean_seconds=clean_seconds,map_seconds=map_seconds,evaluation_seconds=evaluation_seconds,
        wall_seconds=time.monotonic()-start,model_sha256_float64=array_hash(model),new_feedback_inference=False)
    b.write(dest/'RESULT.json',result)
    return maps,state,result

def replay(data,estimators,states,pairs,selected,start):
    calls=0;results=[]
    lookup={(r['case'],r['pass_index']):r for r in states}
    for i,s in enumerate(selected):
        case,k=s['case'],s['pass_index']
        row=lookup[case,k];n,t,present=models(data,row,pairs)
        assert all(present)
        parent,truth,definition=parent_for(data,case)
        expected=read(SAVED/case/'C'/'START.json')['parent_sha256_float64_C']
        assert array_hash(parent)==expected
        dest=OUT/'replay'/f'{case}_after_pass{k:02d}';dest.mkdir(parents=True,exist_ok=False)
        b.write(dest/'START.json',dict(**s,parent_sha256_float64=expected,
            selected_states_sha256=b.digest(OUT/'SELECTED_STATES.json'),preceding_state_is_shared=True,
            nominal_model_diagnostic=True,tight_model_diagnostic=True,relearned_rank=5))
        outputs={}
        for label in (['nominal','tight'] if i%2==0 else ['tight','nominal']):
            guard(start)
            calls+=1
            assert calls<=6
            b.write(OUT/'CLEANING_CALLS.json',dict(started=calls,case=case,branch=label))
            print('REPLAY',calls,case,label,flush=True)
            outputs[label]=branch(data,estimators,parent,n if label=='nominal' else t,truth,definition,dest/label)
            assert array_hash(parent)==expected
        nm,ns,nr=outputs['nominal'];tm,ts,tr=outputs['tight']
        groups=[]
        for c,g,lo,hi,cc,mask,counts,den in data.groups:
            prefix=f'c{c:02d}_nw{g:02d}'
            np.testing.assert_array_equal(ns[prefix+'_columns'],ts[prefix+'_columns'])
            groups.append(dict(group=prefix,array=b.ARRAYS[data.ar[cc[0]]],
                mean_change_L2=float(np.linalg.norm(ts[prefix+'_location']-ns[prefix+'_location'])),
                **subspace_distance(ns[prefix+'_basis'],ts[prefix+'_basis'])))
        array_results=[]
        projection_difference=data.project(t)-data.project(n)
        restored_delta=data.grid(projection_difference)
        for a in range(3):
            nn,tt=nr['measurements'][a],tr['measurements'][a]
            nf,tf=nn['original'].get('fit'),tt['original'].get('fit')
            raw=nf is not None and tf is not None and nf['peak']>0
            peak=tf['peak']/nf['peak']-1 if raw else None
            center=float(np.linalg.norm(np.array(tf['centroid'])-nf['centroid'])) if raw else None
            d=tm[a]-nm[a]
            pp=dict(array=b.ARRAYS[a],raw_fits_available=raw,
                peak_relative_change=peak,centroid_change_arcsec=center,
                both_peak_usable=nn['judgments']['peak_response_usable'] and tt['judgments']['peak_response_usable'],
                both_centroid_usable=nn['judgments']['centroid_usable'] and tt['judgments']['centroid_usable'],
                within_numerical_peak_allocation=abs(peak)<=.005 if peak is not None else None,
                within_numerical_centroid_allocation=center<=.1 if center is not None else None,
                within_operational_peak_scale=abs(peak)<=.05 if peak is not None else None,
                within_operational_centroid_scale=center<=1 if center is not None else None,
                signed_total_difference_energy=energy(d[data.D[a]]),
                restored_model_difference_energy=energy(restored_delta[a,data.D[a]]),
                residual_processing_difference_energy=energy((d-restored_delta[a])[data.D[a]]),
                exterior_difference_rms=float(np.sqrt(np.mean(d[data.O[a]]**2))),
                nominal=nn,tight=tt,nominal_truth=None if nr['truth_scores'] is None else nr['truth_scores'][a],
                tight_truth=None if tr['truth_scores'] is None else tr['truth_scores'][a])
            array_results.append(pp)
        np.savez_compressed(dest/'PAIRED_DIFFERENCES.npz',next_total=tm-nm,restoration=restored_delta,
                            residual_processing=tm-nm-restored_delta,projected_feedback=projection_difference)
        result=dict(**s,parent_unchanged=True,arrays=array_results,subspace_groups=groups,
                    timing={label:outputs[label][2] for label in outputs})
        b.write(dest/'COMPARISON.json',result)
        results.append(result)
        b.write(OUT/'REPLAY_RESULTS.json',results)
        print('RESULT',case,[(r['array'],r['peak_relative_change'],r['centroid_change_arcsec']) for r in array_results],flush=True)
        del outputs,parent,projection_difference
    assert calls==6
    return results

def main():
    verify_freeze();OUT.mkdir(exist_ok=False);start=time.monotonic()
    def alarm(*_):
        raise RuntimeError('one-hour sensitivity budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(3600)
    b.write(OUT/'START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),
        source_freeze_sha256=b.digest(HERE/'FREEZE.json'),python=platform.python_version(),numpy=np.__version__,scipy=b.scipy.__version__))
    try:
        with threadpool_limits(limits=4):
            b.write(OUT/'THREAD_POOLS.json',threadpool_info())
            data,estimators=initialize()
            states=state_rows()
            pairs={(r['case'],r['pass_index'],r['array']):r for r in read(SOLUTIONS/'stopping/RECEIPTS.json')}
            selected,screen_records=screen(data,estimators,states,pairs,start)
            verify_freeze()
            replay(data,estimators,states,pairs,selected,start)
            verify_freeze()
            b.write(OUT/'COMPLETE.json',dict(saved_pair_records=144,available_pairs=141,
                unavailable_pairs=3,one_step_branches=6,new_feedback_optimizations=0,**guard(start)))
    except BaseException as error:
        b.write(OUT/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)
        b.write(OUT/'PRODUCT_MANIFEST.json',dict(root=str(OUT),files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=b.digest(p))
            for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='PRODUCT_MANIFEST.json']))

if __name__=='__main__':
    main()
