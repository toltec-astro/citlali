#!/usr/bin/env python3
"""Bounded, advisory SPOD of saved CAL products; see --help and work-order record."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
for _thread_var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_var]='1'
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import linalg, signal
from threadpoolctl import threadpool_limits, threadpool_info
from ptc_spod import (Inputs, read_cal, require, digest, supported_windows, fourier,
                      decompose, overlap, disjoint_halves, orthonormal_span)


class Costs:
    def __init__(self):self.seconds=defaultdict(float)
    @contextmanager
    def stage(self,name):
        t=time.perf_counter()
        yield
        self.seconds[name]+=time.perf_counter()-t


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def write_json(p,x):p.write_text(json.dumps(x,default=plain,indent=2,allow_nan=False)+'\n')


def load_ptc(data,columns,ledger):
    if data.ptc is None:return None,[],[]
    root=data.root/'donor-continuity/ptc';record=data.ptc
    cleaned=np.full((len(data.times),len(columns)),np.nan);bases=[];segments=[]
    for i,g in enumerate(record['segments']):
        if not g['converged']:continue
        files=g['files'];prefix=f'segment-{i}'
        def read(s,dtype):return ledger.binary(root/(prefix+s),dtype,files[prefix+s])
        slots=read('-slots.i64','<i8');n,d=g['scheduled_times'],g['detectors']
        require(len(slots)==n and np.all(np.diff(slots)>0),'PTC slot ordering')
        vals=read('-input.f64','<f8').reshape(n,d);mask=read('-eligible.u8','u1').reshape(n,d).astype(bool)
        members=np.array(g['detector_indices']);calvals=data.values[np.ix_(slots,members)]
        require(np.array_equal(mask,data.valid[np.ix_(slots,members)]),'PTC/CAL mask mismatch')
        require(np.array_equal(vals[mask],calvals[mask]),'PTC/CAL numerical mismatch')
        if not set(columns).issubset(members):continue
        col=[g['detector_indices'].index(int(c)) for c in columns]
        basis=read('-basis.f64','<f8').reshape(g['basis_rows'],g['basis_columns'])
        basis=orthonormal_span(basis[col]);bases.append(basis)
        out=read('-cleaned.f64','<f8').reshape(n,g['output_detectors'])
        causes=read('-causes.u8','u1').reshape(out.shape)
        fullcols=[g['output_detector_indices'].index(int(c)) for c in columns]
        c=out[:,fullcols].copy();c[causes[:,fullcols]!=0]=np.nan
        cleaned[slots]=c
        segments.append(dict(index=i,first=int(slots[0]),past_last=int(slots[-1]+1),
                             actual_native_interval=g['native_interval'],basis_index=len(bases)-1))
    return cleaned,bases,segments


def context(data,ledger):
    """Descriptive TEL context, not replacement AST motion or admission."""
    import netCDF4
    ref=data.config['telescope'];p=ledger.bind(ref['path'],ref['sha256'])
    with netCDF4.Dataset(p) as nc:
        def v(name):return np.asarray(nc.variables['Data.TelescopeBackend.'+name][:],float)
        t,ra,dec,el=(v(k) for k in ('TelTime','SourceRaAct','SourceDecAct','TelElAct'))
    require(np.isfinite(np.c_[t,ra,dec,el]).all() and np.all(np.diff(t)>0),'TEL context invalid')
    # Fixed tangent approximation only for descriptive context, exact source
    # digest retained. Median pooling suppresses sample derivative jitter.
    east=(np.unwrap(ra)-data.cal['AST_center_ra_rad'])*np.cos(data.cal['AST_center_dec_rad'])*206264.806247
    north=(dec-data.cal['AST_center_dec_rad'])*206264.806247
    ve=np.gradient(east,t);vn=np.gradient(north,t)
    return t,dict(east_arcsec=east,north_arcsec=north,elevation_deg=np.degrees(el),
                  east_velocity_arcsec_s=ve,north_velocity_arcsec_s=vn,speed_arcsec_s=np.hypot(ve,vn))


def summarize_context(ctx,lo,hi,support=None):
    t,fields=ctx;mask=(t>=lo)&(t<hi)
    if support is not None:
        admitted=np.zeros(len(t),bool)
        for a,b in support:admitted|=(t>=a)&(t<b)
        mask&=admitted
    if not mask.any():return None
    d={k:float(np.median(v[mask])) for k,v in fields.items()}
    angle=np.arctan2(fields['north_velocity_arcsec_s'][mask],fields['east_velocity_arcsec_s'][mask])
    d['direction_quadrant_fractions']=[float(np.mean((angle>=a)&(angle<b))) for a,b in zip(np.linspace(-np.pi,np.pi,5)[:-1],np.linspace(-np.pi,np.pi,5)[1:])]
    return d


def eigen_summary(q,probe_bins,cost):
    nf=q.shape[1];fractions=np.zeros((nf,3));power=np.zeros(nf);eigen=np.zeros((nf,min(10,q.shape[0],q.shape[2])))
    modes={};participation=np.zeros(nf);maximum_loading_fraction=np.zeros(nf);dominant_detector=np.zeros(nf,int)
    with cost.stage('eigensolving'):
        for f in range(nf):
            ev,u=decompose(q[:,f]);power[f]=ev.sum();eigen[f,:min(10,len(ev))]=ev[:10]
            fractions[f]=[ev[:k].sum()/ev.sum() if ev.sum()>0 else 0 for k in (1,3,10)]
            if u.shape[1]:
                loading=abs(u[:,0])**2
                participation[f]=1/np.sum(loading**2)
                maximum_loading_fraction[f]=loading.max();dominant_detector[f]=loading.argmax()
            if f in probe_bins:modes[f]=u
    return dict(realizations=len(q),estimator_rank_limit=min(q.shape[0],q.shape[2]),trace_psd=power,
                leading_eigenvalues=eigen,leading_fractions_1_3_10=fractions,leading_participation_detectors=participation,leading_maximum_loading_fraction=maximum_loading_fraction,leading_dominant_cohort_position=dominant_detector),modes


def profile(data,columns,seconds,pool,cleaned,bases,segments,ctx,cost,out):
    nfft=round(seconds/data.dt);hop=nfft//2
    with cost.stage('spectral_preparation'):
        windows,stretches=supported_windows(data.valid[:,columns],data.runs,nfft,hop)
        info=dict(requested_fft_seconds=seconds,fft_samples=nfft,actual_fft_seconds=nfft*data.dt,
                  bin_spacing_hz=1/(nfft*data.dt),Hann_ENBW_hz=1.5/(nfft*data.dt),hop_samples=hop,
                  pool_seconds=pool,PTC_fit_interval_seconds=10 if data.ptc else None,windows=windows,common_stretches=stretches,
                  common_seconds=sum(b-a for a,b in stretches)*data.dt)
        used=np.zeros(len(data.times),bool)
        for a,b in windows:used[a:b]=True
        info['unique_Fourier_support_seconds']=int(used.sum())*data.dt
        if len(windows)<2:return dict(**info,state='unavailable-fewer-than-two-complete-windows')
        freq,q=fourier(data.values[:,columns],windows,data.dt)
        centers=(data.times[windows[:,0]]+data.times[windows[:,1]-1])/2
        groups=np.floor((centers-data.times[0])/pool).astype(int)
        probes=sorted(set(int(np.argmin(abs(freq-h))) for h in (.125,.25,.5,1,3,8,10,10.75,11,11.25,12,15.75,16.75,20,29,40,50) if freq[1]*.99<=h<=freq[-1]))
    pooled,ref=eigen_summary(q,probes,cost);local=[];local_modes=[]
    with cost.stage('comparisons'):
        a,b=disjoint_halves(windows,np.arange(len(windows)))
        pooled_control={}
        if min(len(a),len(b))>=2:
            for f in probes:
                _,u=decompose(q[a,f]);_,v=decompose(q[b,f])
                pooled_control[str(f)]={str(k):overlap(u,v,k) for k in (1,3,10)}
        info['separated_pooled_half_indices']=[a,b]
        rng=np.random.default_rng(20260919)
        null={}
        for f in probes:
            rows=[]
            for repeat in range(8):
                randomized=q[:,f]*np.exp(1j*rng.uniform(-np.pi,np.pi,q[:,f].shape))
                ev,u=decompose(randomized)
                rows.append([float(ev[:k].sum()/ev.sum()) for k in (1,3,10)])
            null[str(f)]=rows
        info['phase_scramble_null_fractions_1_3_10']=null
        info['null_description']='8 deterministic independent detector/realization phases; exact diagonal PSD retained; coherence destroyed; diagnostic control, not a production significance threshold'
    # Persist only selected-frequency vectors; never a detector x detector x
    # frequency x time CSD cube. Every actual window is recorded above.
    arrays={'frequency_hz':freq,**{f'pooled_mode_{f}':ref[f] for f in probes}}
    for group in range(int(groups.max())+1):
        ids=np.flatnonzero(groups==group)
        entry=dict(pool=int(group),relative_interval_seconds=[group*pool,(group+1)*pool],window_indices=ids,
                   context=summarize_context(ctx,data.times[0]+group*pool,data.times[0]+(group+1)*pool,support=[(data.times[windows[i,0]],data.times[windows[i,1]-1]+data.dt) for i in ids]))
        if len(ids)<2:
            local.append(dict(**entry,state='unavailable-fewer-than-two-windows'));local_modes.append(None);continue
        est,modes=eigen_summary(q[ids],probes,cost);entry.update(est)
        with cost.stage('comparisons'):
            entry['to_pooled']={str(f):{str(k):overlap(modes[f],ref[f],k) for k in (1,3,10)} for f in probes}
            a,b=disjoint_halves(windows,ids);entry['separated_half_indices']=[a,b];entry['split_half']={}
            if min(len(a),len(b))>=2:
                for f in probes:
                    _,u=decompose(q[a,f]);_,v=decompose(q[b,f])
                    entry['split_half'][str(f)]={str(k):overlap(u,v,k) for k in (1,3,10)}
            band=np.flatnonzero((freq>=10)&(freq<=12.5))
            if len(band):entry['peak_10_12p5_hz']=float(freq[band[np.argmax(entry['trace_psd'][band])]])
        arrays.update({f'pool_{group}_mode_{f}':modes[f] for f in probes})
        local.append(entry);local_modes.append(modes)
    with cost.stage('comparisons'):
        comparisons=[]
        for i,mi in enumerate(local_modes):
            if mi is None:continue
            for j in range(i+1,len(local_modes)):
                mj=local_modes[j]
                if mj is not None:
                    comparisons.append(dict(pools=[i,j],by_frequency={str(f):{str(k):overlap(mi[f],mj[f],k) for k in (1,3,10)} for f in probes}))
        ptc_compare=[]
        if cleaned is not None:
            # Baseline-output spectra only use windows within one saved fit;
            # discontinuities between independently centered fits are excluded.
            chosen=[];proj=[]
            for wi,(a,b) in enumerate(windows):
                seg=next((s for s in segments if s['first']<=a and b<=s['past_last']),None)
                if seg is None or not np.isfinite(cleaned[a:b]).all():continue
                chosen.append(wi);proj.append(seg['basis_index'])
            if len(chosen)>=2:
                _,qc=fourier(cleaned,windows[chosen],data.dt)
                before=np.mean(np.abs(q[chosen])**2,axis=(0,2));after=np.mean(np.abs(qc)**2,axis=(0,2))
                fractions=[]
                for i,bi in zip(chosen,proj):
                    projected=linalg.blas.zgemm(1,q[i],bases[bi])
                    fractions.append(np.sum(abs(projected)**2,axis=1)/np.sum(abs(q[i])**2,axis=1))
                captures=[]
                for group,modes in enumerate(local_modes):
                    if modes is None:continue
                    bs=sorted({proj[ii] for ii,wi in enumerate(chosen) if groups[wi]==group})
                    for bi in bs:
                        by_frequency={}
                        for f in probes:
                            by_frequency[str(f)]={}
                            for k in (1,3,10):
                                if modes[f].shape[1]<k:continue
                                sv=linalg.svdvals(linalg.blas.zgemm(1,modes[f][:,:k],bases[bi],trans_a=2))
                                sv=np.clip(sv,0,1)
                                by_frequency[str(f)][str(k)]=dict(mean_cos2=float(np.sum(sv**2)/k),angles_deg=np.degrees(np.arccos(sv)).tolist())
                        captures.append(dict(pool=group,basis_index=bi,by_frequency=by_frequency))
                ptc_compare=dict(local_SPOD_to_full_PTC_span=captures,windows=chosen,window_basis_indices=proj,mean_spectral_power_fraction_in_actual_PTC_span=np.mean(fractions,axis=0),
                                 matched_cleaned_to_CAL_power_ratio=after/before,
                                 comparison='same CAL windows, detectors and identity metric; actual saved ALS rank10; FFT demeaning on both')
        info.update(state='available',frequency_hz=freq,probe_bins=probes,pooled=pooled,local=local,
                    pooled_separated_control=pooled_control,pairwise_pool_comparisons=comparisons,PTC=ptc_compare,
                    pooled_cross_frequency_comparisons=[dict(bins=[i,j],ranks={str(k):overlap(ref[i],ref[j],k) for k in (1,3,10)}) for z,i in enumerate(probes) for j in probes[z+1:]])
    with cost.stage('publication'):
        np.savez_compressed(out/f'fft{seconds:g}-modes.npz',**arrays)
    return info


def original_spectra_context(data,ledger):
    # Existing accepted D2 evidence. Different support and units: descriptive
    # original-parent context, not a matched before/after attenuation metric.
    root=data.root;meta=ledger.yaml(root/'original-spectra.yaml')
    spectra=ledger.binary(root/'original-psd.f64','<f8');result=[];offset=0
    for d in meta:
        n=d['bins'];p=spectra[offset:offset+n];offset+=n
        if d['coordinate']!=0 or not d['available'] or n<2:continue
        hz=np.arange(n)/(488*data.learning['cadence_interval_seconds'])
        # Derive FFT size from actual admitted window lengths, not a nominal
        # 4 seconds assumption. This producer records 488 samples here.
        lengths={b-a for a,b in d['windows']}
        require(lengths=={488},'upstream comparison needs explicit uniform 488-sample D2 windows')
        band=(hz>=10)&(hz<=12.5);side=(hz>=8)&(hz<=10)
        result.append(dict(channel=d['channel'],windows=len(d['windows']),peak_10_12p5_hz=float(hz[band][np.argmax(p[band])]),
                           max_10_12p5_over_median_8_10=float(p[band].max()/np.median(p[side]))))
    require(offset==len(spectra),'original spectrum layout mismatch')
    return dict(stage='original native x, accepted D2 4-second evidence',unit='original x squared/Hz',
                support='own producer-valid support; not matched CAL windows',rows=result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='existing connected output directory')
    p.add_argument('--config',type=Path,required=True,help='exact producing configuration')
    p.add_argument('--output',type=Path,required=True,help='new diagnostic directory')
    p.add_argument('--fft-seconds',type=float,nargs='+',default=[2,8])
    p.add_argument('--pool-seconds',type=float,default=120)
    args=p.parse_args();require(all(x>0 for x in args.fft_seconds) and args.pool_seconds>max(args.fft_seconds),'time settings')
    args.output.mkdir(parents=True,exist_ok=False)
    wall=time.perf_counter();cpu=time.process_time();ledger=Inputs();cost=Costs()
    with threadpool_limits(limits=1):
        with cost.stage('input_preparation'):
            data=read_cal(args.input,args.config,ledger)
            columns=np.flatnonzero(data.valid.any(axis=0))
            require(len(columns)>=2,'fewer than two usable detectors')
            cleaned,bases,segments=load_ptc(data,columns,ledger)
            ctx=context(data,ledger)
            upstream=original_spectra_context(data,ledger)
        result=dict(schema='citlali-ptc-spod-diagnostic-v1',scientific_qualification=False,production_changes=False,
                    observation=data.learning['observation'],network=data.learning['network'],unit=data.cal['unit'],
                    input_VAL_generation=data.cal['output_VAL_generation'],input_RTC_attempt=data.cal['input_RTC_attempt'],CAL_plan=data.cal['CAL_plan'],
                    dt_seconds=data.dt,sampling_hz=1/data.dt,nyquist_hz=.5/data.dt,original_parent=data.learning['original_parent'],
                    input_root=str(args.input.resolve()),config=str(args.config.resolve()),
                    requested_detectors=len(data.identities),cohort_columns=columns,cohort_identities=[data.identities[i] for i in columns],
                    cohort_selection='all and only detectors with nonzero CAL support, fixed throughout; no spectral ranking',
                    excluded_zero_support_channels=[data.identities[i]['channel'] for i in range(len(data.identities)) if i not in columns],
                    eligible_fraction_by_detector=data.valid.mean(axis=0),native_runs=data.learning['native_runs'],
                    filter_history={str(d['channel']):d['filter'] for d in data.config['detectors']},
                    lowpass=data.learning['explicit_array_lowpass'],metric='identity Euclidean, fixed CAL units, no variance normalization',
                    estimator='complex Welch CSD; periodic Hann; arithmetic segment mean; dt/sum(w^2); doubled interior one-sided bins; no padding',
                    uncertainty='interleaved realizations separated by >=one full window; no shared samples, not proof of atmospheric independence',
                    scan_context='TEL actual RA/Dec tangent approximation for description only, no replacement AST speed authority',
                    input_classification='development signal; upstream complete response and total covariance unavailable',
                    original_spectra_context=upstream,profiles=[])
        print(f'prepared network {result["network"]}: {len(columns)}/{len(data.identities)} detectors',flush=True)
        for seconds in args.fft_seconds:
            result['profiles'].append(profile(data,columns,seconds,args.pool_seconds,cleaned,bases,segments,ctx,cost,args.output))
            print(f'completed {seconds:g}s Fourier profile',flush=True)
        with cost.stage('input_reverification'):ledger.unchanged()
        result['inputs_sha256']=ledger.files
        result['thread_libraries']=threadpool_info()
        result['thread_limits']={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
        result['thread_limit_observability']='environment caps plus threadpoolctl where supported; empty library list means backend thread count not introspectable'
    with cost.stage('publication'):
        from plot_ptc_spod import plots
        plots(result,args.output)
    result['cost_seconds']=dict(cost.seconds);result['wall_seconds']=time.perf_counter()-wall
    result['cpu_seconds']=time.process_time()-cpu
    result['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    result['inputs_unchanged']=True
    result['tool_sources_sha256']={p.name:digest(p) for p in Path(__file__).parent.glob('*ptc_spod.py')}
    write_json(args.output/'result.json',result)
    print(json.dumps({k:result[k] for k in ('network','wall_seconds','cpu_seconds','process_peak_rss_bytes','cost_seconds')}),flush=True)


if __name__=='__main__':main()
