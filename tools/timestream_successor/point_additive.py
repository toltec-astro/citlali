"""Compact conditional map-noise model; never a MAP weight or SCI-NOI product.

Arrays use (north-row, east-column) grid indexing; lags are pixel displacements.
Models are dimensionless periodic spectra with mean one; only padded interior
crops are measurements. Scale has the unchanged map's mJy/nominal-beam units.
"""
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len
from scipy.stats import chi2, kurtosis, skew
from scipy.linalg import lstsq
from point_empirical import coordinates, stats, source_patch, patch
from inspect_point_source import gaussian, measure

METHOD='point-full-observation-conditional-additive-v1'

@dataclass(frozen=True)
class NoiseRequest:
    mask_fwhm: float=3.
    lag_pixels: int=32
    taper_power: float=1.
    bins: int=5
    minimum_pixels: int=100
    minimum_pairs: int=200
    region: str='all'
    domain_factor: int=1
    seed: int=23152389
    def __post_init__(self):
        if self.mask_fwhm<=0 or self.lag_pixels<2 or self.taper_power<=0 or self.bins<1 or self.minimum_pixels<2 or self.minimum_pairs<2 or self.domain_factor<1:
            raise ValueError('invalid additive model request')
        if self.region not in ['all','north','south','east','west']:raise ValueError('invalid training region')


def pair_covariance(z,mask,lag,minimum_pairs):
    """Mask-corrected empirical second products, NOT an uncorrected periodogram.

    The caller removes the declared mean. Pair correction does not undo the
    finite-window bias of estimating that mean; validation/sensitivity expose it.
    """
    values=np.where(mask,z,0.)
    numer=fftconvolve(values,values[::-1,::-1],mode='full')
    counts=np.rint(fftconvolve(mask.astype(float),mask[::-1,::-1].astype(float),mode='full'))
    cy,cx=np.array(z.shape)-1
    if lag>min(cy,cx):raise ValueError('lag exceeds training image dimensions')
    sl=np.s_[cy-lag:cy+lag+1,cx-lag:cx+lag+1]
    count=counts[sl];cov=np.full(count.shape,np.nan)
    np.divide(numer[sl],count,out=cov,where=count>=minimum_pairs)
    return cov,count


def spectrum_from_pairs(cov,lag,stamp_size,power=1.,domain_factor=1):
    c0=float(cov[lag,lag])
    if not np.isfinite(c0) or c0<=0:raise ValueError('invalid zero-lag second moment')
    grid=next_fast_len(max(2*stamp_size+2,2*lag+3))*domain_factor
    # Strictly > 2 * maximum retained lag and > 2 * crop span. No wraparound
    # can alias the retained empirical lag window into the measurement patch.
    d=np.arange(-lag,lag+1);yy,xx=np.meshgrid(d,d,indexing='ij')
    taper=((1+np.cos(np.pi*xx/(lag+1)))/2 * (1+np.cos(np.pi*yy/(lag+1)))/2)**power
    target=np.where(np.isfinite(cov),cov/c0,0.)*taper
    embedded=np.zeros((grid,grid));embedded[yy%grid,xx%grid]=target
    raw=np.fft.fft2(embedded).real
    neg=float(np.sum(np.maximum(-raw,0))/np.sum(abs(raw)))
    psd=np.maximum(raw,0.)
    psd/=psd.mean()
    rho=np.fft.ifft2(psd).real
    residual=rho[yy%grid,xx%grid]-np.where(np.isfinite(cov),cov/c0,0.)
    return psd,dict(domain_pixels=grid,negative_spectral_mass_fraction=neg,
        pair_to_model_rms=float(np.sqrt(np.mean(residual[np.isfinite(cov)]**2))),
        zero_lag_before_normalization=c0,
        maximum_lag_pixels=lag,taper_power=power,
        representation='real inversion-symmetric nonnegative 2D discrete spectrum; mean=1; cosine-product lag taper then nonnegative spectral projection',
        unsupported_lag_cells=int(np.sum(~np.isfinite(cov))),
        missing_lags='zero only in the computational model; unconstrained physical correlations are unknown')


def learn(signal,coverage,support,header,parameters,request=NoiseRequest(),stamp_size=41,*,source_mask=None):
    x,y=coordinates(signal.shape,header);p=np.asarray(parameters)
    major=max(p[3:5])*np.sqrt(8*np.log(2));r=np.hypot(x-p[1],y-p[2])
    good=support & np.isfinite(signal) & np.isfinite(coverage) & (coverage>0)
    source=r<=request.mask_fwhm*major if source_mask is None else np.asarray(source_mask,dtype=bool)
    if source.shape!=signal.shape:raise ValueError('source mask shape differs from map')
    bg=good & ~source
    if request.region!='all':
        direction={'north':y>=p[2],'south':y<p[2],'east':x>=p[1],'west':x<p[1]}[request.region];bg &=direction
    if bg.sum()<max(request.minimum_pixels,request.minimum_pairs):raise ValueError('insufficient training background')
    mean=float(np.mean(signal[bg]));residual=signal-mean
    edges=np.unique(np.quantile(coverage[bg],np.linspace(0,1,request.bins+1)))
    if len(edges)==1:edges=np.array([edges[0],np.nextafter(edges[0],np.inf)])
    else:edges[-1]=np.nextafter(edges[-1],np.inf)
    index=np.clip(np.searchsorted(edges,coverage,side='right')-1,0,len(edges)-2)
    scale=np.full(signal.shape,np.nan);valid=np.zeros(signal.shape,bool);records=[]
    for b in range(len(edges)-1):
        train=bg & (index==b);n=int(train.sum());s=stats(signal[train])
        moment=float(np.sum(residual[train]**2)/(n-1)) if n>=request.minimum_pixels else None
        use=good & (index==b)
        if moment is not None and moment>0:
            scale[use]=np.sqrt(moment)
            valid[use]=(coverage[use]>=edges[0])&(coverage[use]<edges[-1])
        records.append(dict(bin=b,coverage_min=float(edges[b]),coverage_max_exclusive=float(edges[b+1]),
            n=n,second_moment_about_global_mean=moment,**{k:v for k,v in s.items() if k!='n'}))
    if not np.isfinite(scale[bg]).all():raise ValueError('a training coverage bin is unsupported')
    z=np.divide(residual,scale,out=np.zeros(signal.shape),where=np.isfinite(scale))
    empirical,pairs=pair_covariance(z,bg,request.lag_pixels,request.minimum_pairs)
    psd,fit=spectrum_from_pairs(empirical,request.lag_pixels,stamp_size,request.taper_power,request.domain_factor)
    c0=fit['zero_lag_before_normalization'];scale*=np.sqrt(c0)
    regions=[]
    for label,m in [('NW',(x<p[1])&(y>=p[2])),('NE',(x>=p[1])&(y>=p[2])),('SW',(x<p[1])&(y<p[2])),('SE',(x>=p[1])&(y<p[2])),
                    ('low_coverage',coverage<np.median(coverage[bg])),('high_coverage',coverage>=np.median(coverage[bg]))]:
        mask=bg&m;small=min(8,request.lag_pixels)
        cv,ct=pair_covariance(z,mask,small,max(20,request.minimum_pairs//4))
        central=cv[small,small];cor=cv/central if np.isfinite(central) and central>0 else cv
        regions.append(dict(name=label,n=int(mask.sum()),normalized_second_moment=float(central) if np.isfinite(central) else None,
            covariance=np.where(np.isfinite(cv),cv,np.nan),pairs=ct,correlation=cor,
            histogram=stats(z[mask]),bins=[stats(z[mask&(index==b)]) for b in range(len(edges)-1)]))
    values=z[bg];hist,hedges=np.histogram(values,bins=40)
    summary=dict(method=METHOD,request=asdict(request),mask_radius_arcsec=float(request.mask_fwhm*major) if source_mask is None else None,
        source_mask_policy='parameter-centered circle' if source_mask is None else 'explicit externally frozen source mask',
        mean_removed=mean,mean_policy='single ordinary mean of selected background; no gradient, per-bin mean or source-fit subtraction',
        mean_estimation_limitation='finite background mean removal biases unconstrained long modes; not corrected by pair-count normalization',
        background_pixels=int(bg.sum()),coverage_range=[float(edges[0]),float(edges[-1])],bins=records,
        histogram=dict(counts=hist.tolist(),edges=hedges.tolist(),skew=float(skew(values)),excess_kurtosis=float(kurtosis(values)),
                       fraction_abs_over_3=float(np.mean(abs(values)>3)),robust_scale=stats(values)['robust_scale']),spectrum_fit=fit,
        validity='coverage-bin interpolation within observed range; endpoint continuation outside range is computational only and flagged invalid',
        assumptions='Gaussian generator; separable amplitude modulation times shared stationary normalized anisotropic covariance')
    return dict(scale=scale,valid=valid,source_mask=source,background_mask=bg,z=z,psd=psd,
                empirical=empirical/c0,pairs=pairs,regions=regions,summary=summary)


def linear_weights(header,shape,parameters):
    x,y=coordinates(shape,header);p=np.array(parameters);major=max(p[3:5])*np.sqrt(8*np.log(2))
    unit=p.copy();unit[0]=1.;unit[6]=0.;t=gaussian(unit,x,y)
    roi=np.hypot(x-p[1],y-p[2])<=3*major
    if np.any(roi & (np.hypot(x,y)>40)):raise ValueError('fixed-template footprint exceeds source stamp')
    q=np.where(roi,t,0);q/=np.sum(q*q)
    return q,roi


def variance(psd,scale,weights):
    if scale.shape!=weights.shape or np.any(~np.isfinite(scale[weights!=0])):raise ValueError('invalid amplitude on estimator support')
    q=np.where(weights!=0,weights*np.nan_to_num(scale),0.)
    if max(q.shape)>min(psd.shape):raise ValueError('spectrum domain too small')
    return float(np.sum(abs(np.fft.fft2(q,s=psd.shape))**2*psd)/psd.size)


def draw_noise(psd,scale,seed):
    rng=np.random.default_rng(seed);w=rng.normal(size=psd.shape)
    field=np.fft.ifft2(np.fft.fft2(w)*np.sqrt(psd)).real
    start=(np.array(psd.shape)-np.array(scale.shape))//2
    return field[start[0]:start[0]+scale.shape[0],start[1]:start[1]+scale.shape[1]]*scale


def make_context(signal,coverage,support,header,beam,parameters,model):
    data,valid,depth,h,half=source_patch(signal,support,coverage,header)
    row=int(round(header['CRPIX2']-1));col=int(round(header['CRPIX1']-1))
    scale=patch(model['scale'],row,col,half);amplitude_valid=patch(model['valid'],row,col,half)
    x,y=coordinates(data.shape,h);roi=np.hypot(x,y)<=40
    if np.any(roi & ~valid):raise ValueError('original full-observation fit footprint incomplete')
    if np.any(roi & ~np.isfinite(scale)):raise ValueError('no noise amplitude computable on original fit footprint')
    # Missing nonmeasurement pixels stay unavailable, even in representative realizations.
    baseline=np.where(valid,gaussian(parameters,x,y),np.nan)
    desc,p0,_,_=measure(baseline,valid,h,beam)
    q,fixedroi=linear_weights(h,data.shape,parameters)
    if np.any(fixedroi&~valid):raise ValueError('original fixed-template footprint incomplete')
    return dict(baseline=baseline,support=valid,scale=np.where(np.isfinite(scale),scale,0.),header=h.copy(),beam=beam,q=q,
        psd=model['psd'],roi=roi,amplitude_valid=amplitude_valid,A0=float(p0[0]),reference_parameters=p0.tolist(),
        source_fixed=float(np.sum(q*np.nan_to_num(data))),source_coverage=depth,
        complete_amplitude_support=bool(np.all(amplitude_valid[roi])),
        outside_amplitude_range_fit_pixels=int(np.sum(roi&~amplitude_valid)))


def one_trial(payload):
    context,seed,trial=payload;t=time.perf_counter();cpu=time.process_time()
    noise=draw_noise(context['psd'],context['scale'],seed);generated=time.perf_counter()-t
    data=np.where(context['support'],context['baseline']+noise,np.nan)
    fixed=float(np.sum(context['q']*noise));start=time.perf_counter()
    result=dict(trial=trial,seed=list(seed),fixed_noise_amplitude=fixed)
    try:
        desc,p,_,_=measure(data,context['support'],context['header'],context['beam'])
        lower=np.array([0,-20,-20,.5,.5,-np.pi,-np.inf]);upper=np.array([np.inf,20,20,20,20,np.pi,np.inf])
        result.update(status='success',parameters=p.tolist(),amplitude=float(p[0]),
            constraint_active=desc['fit_constraint_active'],near_lower=np.isclose(p,lower,rtol=0,atol=1e-7).tolist(),
            near_upper=np.isclose(p,upper,rtol=0,atol=1e-7).tolist())
    except (ValueError,RuntimeError) as e:result.update(status='failed',cause=str(e))
    result.update(generation_wall_seconds=generated,fitting_wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
    return result


def aggregate(trials,A0,analytic_variance):
    good=[r for r in trials if r['status']=='success'];amps=np.array([r['amplitude'] for r in good])
    fixed=np.array([r['fixed_noise_amplitude'] for r in trials]);B=len(trials)
    sd=float(amps.std(ddof=1)) if len(amps)>1 else None
    shift=float(amps.mean()-A0) if len(amps) else None
    fvar=float(fixed.var(ddof=1));lo,hi=chi2.ppf([.005,.995],B-1)/(B-1)
    return dict(trial_count=B,successes=len(good),failures=B-len(good),bound_limited=sum(r.get('constraint_active',False) for r in good),
        candidate_refit_scatter=sd if len(good)==B else None,partial_survivor_scatter=sd,
        primary_withheld_for_trial_failure=len(good)!=B,mean_shift_from_A0=shift,A0=A0,
        fixed_analytic_scatter=float(np.sqrt(analytic_variance)),fixed_simulated_scatter=float(np.sqrt(fvar)),
        fixed_variance_ratio=fvar/analytic_variance,fixed_variance_ratio_MC99_interval=[float(lo),float(hi)],
        fixed_analytic_simulation_check=bool(lo<=fvar/analytic_variance<=hi),
        approximate_MC_relative_SE_of_scatter=1/np.sqrt(2*(B-1)),MC_SE_of_mean=float(sd/np.sqrt(B)) if sd else None,
        MC_scope='sampling precision conditional on fitted noise model; no model-estimation uncertainty or observed-source confidence interval',
        Gaussian_tail_probabilities_authorized=False)


def experiment(context,count,seed,array_id,workers=1):
    if count<2:raise ValueError('at least two trials required')
    payload=[(context,(seed,array_id,b),b) for b in range(count)]
    start=time.perf_counter()
    if workers==1:records=list(map(one_trial,payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:records=list(pool.map(one_trial,payload,chunksize=8))
    summary=aggregate(records,context['A0'],variance(context['psd'],context['scale'],context['q']))
    summary['wall_seconds']=time.perf_counter()-start
    summary['worker_cpu_seconds']=sum(r['cpu_seconds'] for r in records)
    summary['generation_worker_wall_seconds']=sum(r['generation_wall_seconds'] for r in records)
    summary['fitting_worker_wall_seconds']=sum(r['fitting_wall_seconds'] for r in records)
    return summary,records


def influence(context):
    """Optional local linearization only, never substituted for the refitter."""
    x,y=coordinates(context['baseline'].shape,context['header']);roi=context['roi']&context['support']
    p=np.asarray(context['reference_parameters']);columns=[]
    for i in range(7):
        step=1e-5*max(1.,abs(p[i]));a=p.copy();b=p.copy();a[i]+=step;b[i]-=step
        columns.append((gaussian(a,x[roi],y[roi])-gaussian(b,x[roi],y[roi]))/(2*step))
    J=np.column_stack(columns);w=np.zeros(roi.shape);w[roi]=lstsq(J.T,np.eye(7)[0],lapack_driver='gelsd')[0]
    return w


def projected_ranges(context,weights):
    a=weights*context['scale'];ac=fftconvolve(a,a[::-1,::-1],mode='full')
    yy,xx=np.indices(ac.shape);dy=yy-(a.shape[0]-1);dx=xx-(a.shape[1]-1)
    rho=np.fft.ifft2(context['psd']).real;terms=ac*rho[dy%rho.shape[0],dx%rho.shape[1]]
    return [dict(max_lag_pixels=k,signed_partial_variance=float(terms[np.hypot(dx,dy)<=k].sum())) for k in [0,2,4,8,16,24,32,40,64]]


def broad_mode_response(context):
    x,y=coordinates(context['baseline'].shape,context['header']);out=[]
    modes=[('constant',np.ones(x.shape)),('east_gradient',x/40),('north_gradient',y/40)]
    for period in [80,160,320]:
        for axis,c in [('east',x),('north',y)]:
            for fn in ['sin','cos']:modes.append((f'{axis}_{fn}_{period}arcsec',getattr(np,fn)(2*np.pi*c/period)))
    for name,mode in modes:
        fits=[]
        for sign in [-1,1]:
            desc,p,_,_=measure(context['baseline']+sign*mode,context['support'],context['header'],context['beam']);fits.append(p[0])
        out.append(dict(mode=name,unit_peak_map_perturbation=1.,fit_amplitude_response_per_unit=float((fits[1]-fits[0])/2),fixed_response_per_unit=float(np.sum(context['q']*mode))))
    return out
