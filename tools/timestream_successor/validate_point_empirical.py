"""Independent-realization map-level validation of the empirical POINT companion.

No timestream, PTC relearning, source attenuation or FRUIT validation is implied.
"""
import argparse,json,time,subprocess,platform
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits
from point_empirical import Request,coordinates,background,source_experiment,source_patch,gaussian,measure,stats
from point_empirical_io import dump
from inspect_forward_map import digest


def fixture(size=129):
    h=fits.Header(dict(CRPIX1=(size+1)/2,CRPIX2=(size+1)/2,CDELT1=-2/3600,CDELT2=2/3600,
        CRVAL1=176.7,CRVAL2=39.9,CTYPE1='RA---TAN',CTYPE2='DEC--TAN',RADESYS='FK5',EQUINOX=2000.))
    x,y=coordinates((size,size),h);p=np.array([200.,2.,-3.,3.,2.3,.35,0.])
    return h,x,y,p


def noise_kernel(correlated):
    if not correlated:return np.ones((1,1))
    x,y=np.meshgrid(np.arange(-5,6),np.arange(-5,6));k=np.exp(-(x*x+y*y)/(2*1.2**2))
    return k/np.sqrt(np.sum(k*k))


def validate(output,realizations=48,seed=94152389):
    if realizations<24:raise ValueError('validation requires at least 24 independent realizations')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    rng=np.random.default_rng(seed);h,x,y,p=fixture();beam=6.;sigma=8.;shape=x.shape
    records=[];cases=[]
    for name,correlated,offset,incomplete,wings in [
        ('white',False,0.,False,False),('correlated',True,0.,False,False),
        ('offset',False,12.,False,False),('incomplete',True,12.,True,False),
        ('non_gaussian_wings',True,12.,False,True)]:
        support=np.hypot(x,y)<120
        if incomplete:
            support &= ~((x>45)&(x<55)&(y<65))
            support &= ~((x< -70)&(y>0))
        coverage=np.where(support,1000.,0.)
        source=gaussian(p,x,y)
        if wings:source+=50/(1+((x-p[1])**2+(y-p[2])**2)/8**2)**2
        noiseless=np.where(support,source+offset,np.nan)
        reference,truth_fit,_,_=measure(noiseless,support,h,beam)
        major=max(truth_fit[3:5])*np.sqrt(8*np.log(2));request=Request(maximum_blanks=12,seed=seed+len(cases))
        bgmask=background(noiseless,coverage,support,h,truth_fit[1:3],major,request)['background_mask']
        kernel=noise_kernel(correlated);case=[]
        for i in range(realizations):
            # Independent padded realization; exact stationary kernel away from
            # padding. Correlated and white cases both have pixel sigma=8.
            noise=fftconvolve(rng.normal(size=(shape[0]+20,shape[1]+20)),kernel,mode='same')[10:-10,10:-10]*sigma
            signal=np.where(support,source+offset+noise,np.nan)
            desc,fit,_,_=measure(signal,support,h,beam)
            # Freeze the noiseless measurement template for the linear comparison;
            # the nonlinear fit is rerun on each independent map above.
            exp=source_experiment(signal,coverage,support,h,beam,truth_fit,bgmask,request)
            bg=background(signal,coverage,support,h,truth_fit[1:3],major,request)
            record=dict(case=name,realization=i,fixed=exp['fixed_template']['source']['amplitude'],
                fixed_sigma=exp['fixed_template']['blank']['std'],fixed_blank_mean=exp['fixed_template']['blank']['mean'],
                fit=fit[0],fit_sigma=exp['refit']['error']['std'],fit_bias=exp['refit']['error']['mean'],
                refit_failures=exp['refit']['failed'],refit_constrained=exp['refit']['constrained'],fit_scatter_available=exp['refit']['scatter_available'],
                apertures=[a['source']['value_mJy_per_beam_arcsec2'] for a in exp['apertures']],
                aperture_sigmas=[a['blank']['std'] for a in exp['apertures']],
                background_sigma=bg['records'][0]['std'],background_median=bg['records'][0]['median'])
            case.append(record);records.append(record)
        comparisons={}
        for namekey,valkey,estkey in [('fixed','fixed','fixed_sigma'),('fit','fit','fit_sigma')]:
            actual=stats([r[valkey] for r in case]);est=np.asarray([r[estkey] for r in case],float)
            ratio=float(np.sqrt(np.mean(est**2))/actual['std'])
            comparisons[namekey]=dict(actual_scatter=actual['std'],rms_empirical_scatter=float(np.sqrt(np.mean(est**2))),ratio=ratio,
                actual_mean=actual['mean'],ratio_check=bool(.60<ratio<1.55))
        for a in range(3):
            actual=stats([r['apertures'][a] for r in case]);est=np.asarray([r['aperture_sigmas'][a] for r in case],float)
            ratio=float(np.sqrt(np.mean(est**2))/actual['std'])
            comparisons[f'aperture_{a}']=dict(actual_scatter=actual['std'],rms_empirical_scatter=float(np.sqrt(np.mean(est**2))),ratio=ratio,
                actual_mean=actual['mean'],ratio_check=bool(.60<ratio<1.55))
        d,v,q,sh,half=source_patch(noiseless,support,coverage,h);sx,sy=coordinates(d.shape,sh)
        unit=truth_fit.copy();unit[0]=1;unit[6]=0;t=gaussian(unit,sx,sy);roi=np.hypot(sx-truth_fit[1],sy-truth_fit[2])<=3*major
        weights=np.where(roi,t,0);weights/=np.sum(weights**2)
        theory=float(sigma*np.sqrt(np.sum(fftconvolve(weights,kernel,mode='full')**2)))
        analytical=dict(fixed_sigma=theory,measured_ratio=comparisons['fixed']['actual_scatter']/theory,
            offset_fixed_contribution=float(offset*weights.sum()))
        summary=dict(case=name,pixel_sigma=sigma,background_offset=offset,realizations=realizations,
            noiseless_fitted_amplitude=float(truth_fit[0]),physical_peak_above_background=250. if wings else 200.,
            model_mismatch_peak_difference=float(truth_fit[0]-(250. if wings else 200.)),
            actual_fit_bias_about_noiseless_fit=float(np.mean([r['fit'] for r in case])-truth_fit[0]),
            empirical_refit_bias=float(np.mean([r['fit_bias'] for r in case])),
            mean_background_sigma=float(np.mean([r['background_sigma'] for r in case])),
            mean_background_median=float(np.mean([r['background_median'] for r in case])),
            comparisons=comparisons,analytic=analytical,
            refit_failures=sum(r['refit_failures'] for r in case),constrained_refits=sum(r['refit_constrained'] for r in case))
        summary['passed']=all(c['ratio_check'] for c in comparisons.values()) and .7<analytical['measured_ratio']<1.3 and not summary['refit_failures']
        cases.append(summary);print(name,json.dumps(comparisons),flush=True)
    report=dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
        code_sha256={name:digest(Path(__file__).parent/name) for name in ['validate_point_empirical.py','point_empirical.py','inspect_point_source.py']},
        python=platform.python_version(),seed=seed,realizations=realizations,cases=cases,passed=all(c['passed'] for c in cases),
        wall_seconds=time.perf_counter()-start,scope='independent map realizations only; no PTC/full-chain recovery claim',
        bound_policy='all finite bound-limited outputs retained in scatter and counted; failed refits withhold aggregate uncertainty',
        comparison_definition='RMS of within-map blank/refit standard deviations versus actual SD of measurements across independent maps; no division by sqrt(blank count)',
        approximate_relative_MC_error_on_actual_SD=1/np.sqrt(2*(realizations-1)))
    dump(output/'validation.json',report);dump(output/'realizations.json',records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for j,key in enumerate(['fixed','fit','aperture_0','aperture_1','aperture_2']):
        axes[0].plot(np.arange(len(cases))+(j-2)*.08,[c['comparisons'][key]['ratio'] for c in cases],'o',label=key)
    axes[0].axhline(1,color='black',lw=.8);axes[0].set_xticks(range(len(cases)),[c['case'] for c in cases],rotation=20)
    axes[0].set(ylabel='Empirical scatter / independent-realization scatter',title='Map-level uncertainty calibration');axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(len(cases))-.15,[c['actual_fit_bias_about_noiseless_fit'] for c in cases],width=.3,label='Noise-related fit bias')
    axes[1].bar(np.arange(len(cases))+.15,[c['model_mismatch_peak_difference'] for c in cases],width=.3,label='Noiseless fit minus physical peak')
    axes[1].set_xticks(range(len(cases)),[c['case'] for c in cases],rotation=20);axes[1].set(ylabel='mJy / nominal beam',title='Fit scatter does not measure model mismatch');axes[1].legend(fontsize=8)
    fig.savefig(output/'validation.png',dpi=160);plt.close(fig)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--realizations',type=int,default=48);p.add_argument('--seed',type=int,default=94152389)
    a=p.parse_args();r=validate(a.output,a.realizations,a.seed)
    if not r['passed']:raise SystemExit(1)
