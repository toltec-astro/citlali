"""Extend the empirical companion's independent-map fixture to covariance inference.

True noise is a sum of independent spatial moving-average fields, implemented
with direct kernels, not the fitted spectrum representation. Every outer map
learns its own model; the inner ensemble is deliberately modest.
"""
import argparse,json,time,subprocess,resource
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import chi2
from validate_point_empirical import fixture
from inspect_point_source import gaussian,measure
from point_empirical import source_patch,coordinates
from point_additive import NoiseRequest,learn,make_context,linear_weights,variance,experiment
from point_additive_diagnostics import save_json
from inspect_forward_map import digest

CASES=['white','anisotropic_signed','varying_incomplete','long_range','spatially_misspecified','non_gaussian_wings']


def kernel(kind):
    if kind=='white':return np.ones((1,1))
    n=40 if kind=='long' else 8;y,x=np.mgrid[-n:n+1,-n:n+1]
    if kind=='long':k=np.exp(-.5*((x/12)**2+(y/7)**2))
    else:
        u=.8*x+.6*y;v=-.6*x+.8*y
        k=np.exp(-.5*((u/2.3)**2+(v/.7)**2))-.42*np.exp(-.5*(((u-4)/2.3)**2+(v/.7)**2))
        if kind=='rotated':k=k.T
    return k/np.linalg.norm(k)


def scene(case):
    h,x,y,p=fixture(129);p[6]=12.;support=np.hypot(x,y)<120;coverage=np.where(support,1000.,0.)
    if case=='varying_incomplete':
        support &=~((x>45)&(x<58)&(y<65));support &=~((x< -70)&(y>0))
        coverage=np.where(support,1000*(1+.45*x/120),0.)
    source=gaussian(p,x,y)
    if case=='non_gaussian_wings':source+=50/(1+((x-p[1])**2+(y-p[2])**2)/8**2)**2
    mean=np.where(support,source,np.nan);desc,reference,_,_=measure(mean,support,h,6.)
    if case=='white':components=[(np.ones(x.shape)*8,kernel('white'))]
    elif case=='long_range':components=[(np.ones(x.shape)*5,kernel('white')),(np.ones(x.shape)*7,kernel('long'))]
    elif case=='spatially_misspecified':
        # Different independent correlation fields on disjoint regions; the
        # broad transition avoids inventing an abrupt scalar-depth discontinuity.
        alpha=(1+np.tanh((x-12)/8))/2
        components=[(8*np.sqrt(alpha),kernel('signed')),(8*np.sqrt(1-alpha),kernel('rotated'))]
    else:
        scale=8*(1+.35*x/120) if case=='varying_incomplete' else np.ones(x.shape)*8
        components=[(scale,kernel('signed'))]
    return h,x,y,p,support,coverage,mean,reference,components


def outer(payload):
    case,index,inner,seed=payload;start=time.perf_counter();cpu=time.process_time()
    h,x,y,p,support,coverage,mean,reference,components=scene(case)
    rng=np.random.default_rng([seed,CASES.index(case),index]);noise=np.zeros(x.shape)
    for scale,k in components:
        border=max(k.shape)+2
        w=rng.normal(size=(x.shape[0]+2*border,x.shape[1]+2*border))
        field=fftconvolve(w,k,mode='same')[border:-border,border:-border]
        noise+=field*scale
    data=np.where(support,mean+noise,np.nan);desc,observed,_,_=measure(data,support,h,6.)
    model=learn(data,coverage,support,h,observed,NoiseRequest(seed=seed))
    c=make_context(data,coverage,support,h,6.,observed,model)
    _,_,_,sh,half=source_patch(data,support,coverage,h);q,roi=linear_weights(sh,c['baseline'].shape,reference)
    c['q']=q
    result,trials=experiment(c,inner,seed+100,index+1000*CASES.index(case),1)
    row=col=64;ns=noise[row-half:row+half+1,col-half:col+half+1]
    known=0.
    for scale,k in components:
        s=scale[row-half:row+half+1,col-half:col+half+1]
        known+=np.sum(fftconvolve(q*s,k,mode='full')**2)
    regional=model['regions'];second=[r['normalized_second_moment'] for r in regional if r['normalized_second_moment'] is not None]
    # Correlation contrast directly compares two disjoint hemispheres using the
    # same normalization; it is evidence of misspecification, not a calibrated test.
    from point_additive import pair_covariance
    lags=[]
    for region in [x<0,x>=0]:
        cv,ct=pair_covariance(model['z'],model['background_mask']&region,8,50);lags.append(cv/cv[8,8])
    return dict(case=case,outer=index,actual_fit_amplitude=float(observed[0]),noiseless_fit_amplitude=float(reference[0]),
        physical_source_peak=250. if case=='non_gaussian_wings' else 200.,
        actual_fixed_noise_amplitude=float(np.sum(q*ns)),known_fixed_variance=float(known),
        estimated_fixed_variance=variance(model['psd'],c['scale'],q),inner=result,trials=trials,
        amplitude_supported=c['complete_amplitude_support'],
        region_correlation_contrast_RMS=float(np.sqrt(np.nanmean((lags[0]-lags[1])**2))),
        PSD_negative_mass=model['summary']['spectrum_fit']['negative_spectral_mass_fraction'],
        background_mean=model['summary']['mean_removed'],
        wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)


def compare(actual,estimated_variances):
    actual=np.asarray(actual);v=np.asarray(estimated_variances);n=len(actual)
    empirical=float(actual.var(ddof=1));pred=float(v.mean());ratio=pred/empirical
    # Prespecified approximation allowance is separate from MC sampling.
    se=float(np.sqrt(2/(n-1)+v.var(ddof=1)/(n*pred**2)))
    return dict(empirical_variance=empirical,mean_estimated_variance=pred,variance_ratio=ratio,
        approximate_log_ratio_MC_SE=se,model_approximation_allowance_fraction=.20,
        sampling_check=bool(abs(np.log(ratio))<=3*se+np.log(1.20)),
        actual_scatter=float(np.sqrt(empirical)),RMS_estimated_scatter=float(np.sqrt(pred)))


def validate(args):
    if args.outer<4 or args.inner<4:raise ValueError('at least four outer and inner maps for pilot')
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cases=args.cases or CASES;payload=[(c,i,args.inner,args.seed) for c in cases for i in range(args.outer)]
    if args.workers==1:records=list(map(outer,payload))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:records=list(pool.map(outer,payload,chunksize=1))
    results=[]
    for case in cases:
        rr=[r for r in records if r['case']==case];failed=sum(r['inner']['failures'] for r in rr)
        fixed=compare([r['actual_fixed_noise_amplitude'] for r in rr],[r['estimated_fixed_variance'] for r in rr])
        fit=compare([r['actual_fit_amplitude'] for r in rr],[r['inner']['candidate_refit_scatter']**2 for r in rr]) if not failed else None
        ref=rr[0]['noiseless_fit_amplitude'];amps=np.array([r['actual_fit_amplitude'] for r in rr])
        true_fixed=compare([r['actual_fixed_noise_amplitude'] for r in rr],[r['known_fixed_variance'] for r in rr])
        model_challenge=case in ['spatially_misspecified','long_range','non_gaussian_wings']
        result=dict(case=case,outer_count=len(rr),inner_count=args.inner,fixed=fixed,fit=fit,known_generator_check=true_fixed,
            known_reference_amplitude=ref,noiseless_model_mismatch=ref-rr[0]['physical_source_peak'],
            actual_noise_bias_about_noiseless_fit=float(amps.mean()-ref),mean_predicted_MC_bias=float(np.mean([r['inner']['mean_shift_from_A0'] for r in rr])),
            failures=failed,bound_limited=sum(r['inner']['bound_limited'] for r in rr),
            unsupported_amplitude_realizations=sum(not r['amplitude_supported'] for r in rr),
            mean_region_correlation_contrast_RMS=float(np.mean([r['region_correlation_contrast_RMS'] for r in rr])),
            mean_PSD_negative_mass=float(np.mean([r['PSD_negative_mass'] for r in rr])),
            challenge_case=model_challenge,
            disposition='diagnostic challenge; disagreements retained without tuning' if model_challenge else 'sampling-calibration check',
            sampling_checks_passed=bool(not failed and fixed['sampling_check'] and fit['sampling_check']))
        results.append(result);print(case,json.dumps(result),flush=True)
    report=dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*point*.py')},seed=args.seed,workers=args.workers,
        cases=results,wall_seconds=time.perf_counter()-start,worker_cpu_seconds=sum(r['cpu_seconds'] for r in records),
        peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        scope='independent synthetic model inference from each map background; own source fit and modest inner ensembles; kernels independent of fitted PSD; no PTC/cleaning validation',
        ordinary_cases_pass=all(r['sampling_checks_passed'] for r in results if not r['challenge_case']),
        MC_precision_actual_SD=1/np.sqrt(2*(args.outer-1)),
        finite_window='single estimated background mean; offset12 included; long_range case exposes finite-footprint/mean effects',
        optional_frozen_MAP='not performed by this driver; separately assess existing boundary')
    save_json(out/'validation.json',report);save_json(out/'realizations.json',records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for j,key in enumerate(['fixed','fit']):
        axs[0].plot(np.arange(len(results))+.12*j,[r[key]['variance_ratio'] if r[key] else np.nan for r in results],'o',label=key)
    axs[0].axhline(1,color='black',lw=.8);axs[0].legend();axs[0].set(ylabel='Mean estimated / actual variance',title='Inference across independent observations')
    axs[1].bar(np.arange(len(results))-.15,[r['noiseless_model_mismatch'] for r in results],width=.3,label='Noiseless model mismatch')
    axs[1].bar(np.arange(len(results))+.15,[r['actual_noise_bias_about_noiseless_fit'] for r in results],width=.3,label='Noise-related mean shift');axs[1].legend(fontsize=8);axs[1].set(title='Different quantities',ylabel='Amplitude (mJy/nominal beam)')
    axs[2].bar(range(len(results)),[r['mean_region_correlation_contrast_RMS'] for r in results]);axs[2].set(title='Spatial-correlation challenge diagnostic',ylabel='Disjoint hemisphere correlation contrast RMS')
    for ax in axs:ax.set_xticks(range(len(results)),[r['case'] for r in results],rotation=40,ha='right')
    fig.savefig(out/'validation.png',dpi=150);plt.close(fig)
    manifest=out/'FILES_SHA256SUMS';manifest.write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p!=manifest))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--outer',type=int,default=48);p.add_argument('--inner',type=int,default=8);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=723152389);p.add_argument('--cases',nargs='+',choices=CASES)
    result=validate(p.parse_args())
    if not result['ordinary_cases_pass']:raise SystemExit(1)
