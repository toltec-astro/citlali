"""Matched residual evidence and a conditional frozen-operator noise control.

The phase control retains observed diagonal power (including contamination),
removes input detector coherence and applies the exact saved broadband operator.
It is a descriptive conditional null, not a noise estimate or line efficiency.
"""
import numpy as np
from scipy import linalg
from ptc_spod import decompose, fourier, disjoint_halves, overlap, require, supported_windows


def apply_complete_fourier(q,basis):
    """Right detector projection on a window with complete fit membership.

    Window demeaning removes the fixed per-detector centering constants. This
    cannot be used when an application mask changes inside that window.
    """
    return q-linalg.blas.zgemm(1,linalg.blas.zgemm(1,q,basis),basis,trans_b=1)


def projected_power(q,u):
    return float(np.mean(abs(linalg.blas.zgemm(1,q,u[:,None].conj()))**2))


def matched_residual(data,columns,windows,q,cleaned,segments,dt):
    selected=[];owners=[]
    for wi,(a,b) in enumerate(windows):
        seg=next((s for s in segments if s['first']<=a and b<=s['past_last']),None)
        if seg is not None and np.isfinite(cleaned[a:b]).all():selected.append(wi);owners.append(seg)
    result=dict(state='unavailable-fewer-than-two-matched-windows',window_indices=selected,
        meaning='absolute matched CAL/PTC spectra; total bin power is not isolated pickup',
        control='independent detector/realization phases, observed diagonal power, then actual frozen PTC operator; conditional descriptive control, not a physical noise estimate')
    if len(selected)<2:return result
    freq,qout=fourier(cleaned,windows[selected],dt);qin=q[selected]
    bins=sorted({int(np.argmin(abs(freq-h))) for h in (10,10.75,11,11.25,12)})
    # Learn the pattern on disjoint CAL windows; evaluate on matched-output
    # windows in the other interleaved half. Neither half shares samples.
    halves=disjoint_halves(windows,np.arange(len(windows)))
    lookup={w:i for i,w in enumerate(selected)};folds=[]
    rng=np.random.default_rng(20260920)
    for fold,(train,test) in enumerate((halves,halves[::-1])):
        evaluation=[lookup[int(i)] for i in test if int(i) in lookup]
        entry=dict(fold=fold,pattern_training_window_indices=train.tolist(),evaluation_window_indices=[selected[i] for i in evaluation])
        if len(train)<2 or len(evaluation)<2:
            folds.append(dict(entry,state='unavailable-separate-training-or-evaluation-support'));continue
        summaries=[]
        for f in bins:
            _,u=decompose(q[train,f])
            if not u.shape[1]:
                summaries.append(dict(bin=int(f),frequency_hz=float(freq[f]),state='unavailable-no-identifiable-input-pattern'));continue
            pattern=u[:,0]
            before=qin[evaluation,f];after=qout[evaluation,f]
            ev,v=decompose(after)
            summaries.append(dict(bin=int(f),frequency_hz=float(freq[f]),
                input_trace_psd=float(np.mean(np.sum(abs(before)**2,axis=1))),
                output_trace_psd=float(np.mean(np.sum(abs(after)**2,axis=1))),
                input_pattern_psd=projected_power(before,pattern),output_pattern_psd=projected_power(after,pattern),
                output_leading_eigenvalue=float(ev[0]),output_leading_fraction=float(ev[0]/ev.sum()) if ev.sum()>0 else None,
                output_leading_participation=float(1/np.sum(abs(v[:,0])**4)) if v.shape[1] else None,
                output_pattern_overlap=overlap(u,v,1)))
        folds.append(dict(entry,state='available',frequency=summaries))
    # A complete selected cohort need not cover the entire fit. Use the full
    # saved operator and full fit population, then gather the diagnostic cohort.
    # If any other fit member is invalid here, retain the measured residual but
    # explicitly exclude this window from this simple control (no mask model).
    controlled=[];before_full=[];after_null=[[] for _ in range(8)];actual=[];max_error=0.
    for k,(wi,seg) in enumerate(zip(selected,owners)):
        a,b=windows[wi];members=seg['fit_members'];basis=seg['full_basis']
        if not data.valid[a:b,members].all():continue
        _,full=fourier(data.values[:,members],np.array([[a,b]]),dt)
        full=full[0,bins];gather=[members.index(int(c)) for c in columns]
        predicted=apply_complete_fourier(full,basis)[:,gather]
        observed=qout[k,bins]
        error=float(np.max(abs(predicted-observed)));scale=max(float(np.max(abs(observed))),float(np.max(abs(full))),1.)
        require(error<=2e-8*scale,'saved PTC Fourier operator does not reproduce matched output')
        max_error=max(max_error,error)
        controlled.append(wi);before_full.append(q[wi,bins]);actual.append(observed)
        for r in range(8):
            randomized=full*np.exp(1j*rng.uniform(-np.pi,np.pi,full.shape))
            after_null[r].append(apply_complete_fourier(randomized,basis)[:,gather])
    controls=[]
    if len(controlled)>=2:
        observed=np.asarray(actual);scrambled=np.asarray(after_null)
        before=np.asarray(before_full)
        for z,f in enumerate(bins):
            ev,u=decompose(observed[:,z]);null=[]
            for trial in scrambled:
                ne,nv=decompose(trial[:,z]);null.append(dict(trace_psd=float(ne.sum()),leading_eigenvalue=float(ne[0]),leading_fraction=float(ne[0]/ne.sum()) if ne.sum()>0 else None))
            controls.append(dict(bin=int(f),frequency_hz=float(freq[f]),input_trace_psd=float(np.mean(np.sum(abs(before[:,z])**2,axis=1))),
                output_trace_psd=float(ev.sum()),output_leading_eigenvalue=float(ev[0]),output_leading_fraction=float(ev[0]/ev.sum()) if ev.sum()>0 else None,
                frozen_operator_phase_controls=null))
    # Relate the output to the separately learned pattern on the SAME subset
    # used by the frozen-operator control, not to its own best-fitting mode.
    control_lookup={wi:i for i,wi in enumerate(controlled)}
    if len(controlled)>=2:
        for fold in folds:
            if fold['state']!='available':continue
            ci=[control_lookup[wi] for wi in fold['evaluation_window_indices'] if wi in control_lookup]
            fold['controlled_evaluation_window_indices']=[controlled[i] for i in ci]
            fold['controlled_pattern']=[]
            if len(ci)<2:continue
            train=fold['pattern_training_window_indices']
            for z,f in enumerate(bins):
                _,u=decompose(q[train,f])
                if not u.shape[1]:continue
                pattern=u[:,0]
                fold['controlled_pattern'].append(dict(bin=int(f),frequency_hz=float(freq[f]),
                    input_pattern_psd=projected_power(before[ci,z],pattern),
                    output_pattern_psd=projected_power(observed[ci,z],pattern),
                    frozen_operator_phase_control_psd=[projected_power(trial[ci,z],pattern) for trial in scrambled]))
    result.update(state='available',frequency_hz=freq[bins].tolist(),separate_pattern_folds=folds,
        control_window_indices=controlled,control_excluded_windows=len(selected)-len(controlled),
        control_exclusion_reason='one or more full PTC fit members unavailable within window; no complete-operator substitution',
        maximum_Fourier_operator_absolute_error=max_error,control_frequency=controls,
        control_state='available' if len(controlled)>=2 else 'unavailable-insufficient-complete-fit-windows')
    return result


def temporal_cohort(valid,runs,dt):
    """One optional fixed supplement, selected only by temporal coverage.

    Evaluate two predetermined fractions of the nonzero-support population,
    ranked by total eligible duration (ties preserve detector order). Choose
    the largest cohort supplying more complete 8-second time than the full set.
    This is descriptive selection, not a new validity threshold.
    """
    cols=np.flatnonzero(valid.any(axis=0));rank=cols[np.argsort(-valid[:,cols].sum(axis=0),kind='stable')]
    candidates=[];selected=None
    for fraction in (1.,.9,.75):
        count=max(2,int(np.ceil(len(cols)*fraction)));cohort=np.sort(rank[:count])
        windows,stretches=supported_windows(valid[:,cohort],runs,round(8/dt),round(8/dt)//2)
        used=np.zeros(len(valid),bool)
        for a,b in windows:used[a:b]=True
        candidates.append(dict(fraction=fraction,detectors=len(cohort),common_seconds=sum(b-a for a,b in stretches)*dt,
            Fourier_seconds=int(used.sum())*dt,windows=len(windows),columns=cohort.tolist()))
        if fraction<1 and selected is None and used.sum()*dt>candidates[0]['Fourier_seconds'] and len(windows)>=2:selected=cohort
    return dict(selection='fixed duration ranking; compare full, 90%, 75%; choose largest with more 8-second support; no spectra used',
                candidates=candidates,selected_columns=None if selected is None else selected.tolist())
