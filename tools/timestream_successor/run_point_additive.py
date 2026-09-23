"""Full-observation conditional additive-noise companion for saved POINT 152389."""
import argparse,json,os,platform,resource,subprocess,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import scipy,astropy
from point_empirical_io import load_parent,write_fits
from inspect_forward_map import digest,require,read_yaml
from point_empirical import patch,coordinates
from point_additive import NoiseRequest,learn,make_context,variance,experiment,influence,projected_ranges,broad_mode_response,draw_noise,aggregate,METHOD
from point_additive_diagnostics import save_json,verify_companion,regional_summary,populations,blank_checks,draw


def sensitivity_range(records):
    values=[r.get('matched_budget_experiment',r.get('experiment',{})).get('candidate_refit_scatter') for r in records]
    values=[v for v in values if v is not None]
    return [min(values),max(values)] if values else None


def number(value):
    return 'unavailable' if value is None else f'{value:.3f}'


def report_row(s):
    if s.get('status')=='unavailable':
        return f"| {s['array']} | {number(s['unchanged_measured_amplitude'])} | unavailable: {s['cause']} | — | — | — | — | — |"
    span=s['sensitivity_scatter_range'];span_text='unavailable' if span is None else f'{span[0]:.3f}–{span[1]:.3f}'
    return f"| {s['array']} | {s['unchanged_measured_amplitude']:.3f} | {number(s['candidate_refit_scatter'])} | {number(s['mean_shift_from_A0'])} | {number(s['fixed_analytic_scatter'])} / {number(s['fixed_simulated_scatter'])} | {s['trial_count']} / {s['failures']} / {s['bound_limited']} | {span_text} | {s['amplitude_supported']} |"


def run(args):
    start=time.perf_counter();cpu=time.process_time();out=args.output.resolve()
    require(not out.exists() and all(not out.is_relative_to(r.resolve()) for r in [args.maps,args.empirical,args.pairings,args.measurement.parent]),'new separate output directory required')
    require(args.workers>=1 and args.trials>=2 and args.sensitivity_trials>=2,'invalid trial/worker budget')
    measurement=json.loads(args.measurement.read_text());require(measurement['schema']=='citlali-point-source-description-v1' and measurement['observation']==152389,'foreign source measurement')
    saved={r['array']:r for r in measurement['arrays']};require(set(saved)=={'a1100','a1400','a2000'},'three saved arrays required')
    hashes={a:read_yaml(args.maps/a/'bundle.yaml')['files']['map.fits'] for a in saved}
    t=time.perf_counter();prior=verify_companion(args.empirical,hashes);pairing=verify_companion(args.pairings,hashes)
    companion_verification_seconds=time.perf_counter()-t
    provenance=dict(method=METHOD,profile=args.profile,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*point*.py')},
        measurement_sha256=digest(args.measurement),measurement_path=str(args.measurement.resolve()),maps=str(args.maps.resolve()),
        empirical=prior,pairings=pairing,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__),
        workers=args.workers,threading={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
        scientific_authority='owner work order CODEX_POINT_152389_FULL_OBSERVATION_ADDITIVE_UNCERTAINTY.md; saved-descriptive-v1 reproduced exactly',
        method_boundary='separate model-based additive companion; not SCI-NOI sign randomization, a formal POINT error method, attenuation/calibration uncertainty, or FRUIT admission')
    out.mkdir(parents=True);save_json(out/'provenance.json',provenance)
    request=NoiseRequest(seed=args.seed);save_json(out/'request.json',dict(noise=asdict(request),trials=args.trials,sensitivity_trials=args.sensitivity_trials,workers=args.workers))
    arrays=args.arrays or list(saved);summaries=[];profiles=[]
    for array in arrays:
        ar=out/array;ar.mkdir();profile={};t=time.perf_counter()
        data,h,bundle,plan,desc,p,verification=load_parent(args.maps/array,saved[array]);save_json(ar/'parent-verification.json',verification)
        profile['loading_verification_wall_seconds']=time.perf_counter()-t
        sig=data['PRIMARY'];cov=data['NORMALIZATION'];support=data['SCI_SUPPORT'].astype(bool);beam=bundle['nominal_beam_fwhm_arcsec'];t=time.perf_counter()
        try:
            model=learn(sig,cov,support,h,p,request);context=make_context(sig,cov,support,h,beam,p,model)
        except ValueError as exc:
            # Scientific model/footprint unavailability is per array. Parent
            # verification above remains fatal; corrupt input is not tolerated.
            unavailable=dict(array=array,status='unavailable',cause=str(exc),unchanged_measured_amplitude=float(p[0]))
            save_json(ar/'summary.json',unavailable);summaries.append(unavailable)
            profile.update(array=array,covariance_inference_wall_seconds=time.perf_counter()-t)
            profiles.append(profile);continue
        prior_measure=json.loads((args.empirical/array/'measurements.json').read_text())
        require(np.allclose(p,prior_measure['source_parameters'],rtol=1e-10,atol=1e-10),'saved seven-parameter binding differs')
        require(np.isclose(context['source_fixed'],prior_measure['fixed_template']['source']['amplitude'],rtol=1e-10,atol=1e-10),'fixed-template saved measurement differs')
        profile['covariance_inference_wall_seconds']=time.perf_counter()-t
        t=time.perf_counter();summary,trials=experiment(context,args.trials,args.seed,int(array[1:]),args.workers)
        profile['base_experiment_wall_seconds']=time.perf_counter()-t
        save_json(ar/'trials.json',trials);save_json(ar/'base-experiment.json',summary)
        profile.update({k:summary[k] for k in ['worker_cpu_seconds','generation_worker_wall_seconds','fitting_worker_wall_seconds']})
        variants=[('base',request)]+[(name,replace(request,**change)) for name,change in [
            ('mask4',dict(mask_fwhm=4.)),('mask5',dict(mask_fwhm=5.)),('lag16',dict(lag_pixels=16)),('lag48',dict(lag_pixels=48)),
            ('taper0.7',dict(taper_power=.7)),('taper1.4',dict(taper_power=1.4)),('north',dict(region='north')),('south',dict(region='south'))]]
        sensitivities=[];t=time.perf_counter()
        for name,req in variants:
            if name=='base':m,c,exp,records=model,context,summary,trials
            else:
                try:
                    m=learn(sig,cov,support,h,p,req);c=make_context(sig,cov,support,h,beam,p,m)
                    exp,records=experiment(c,args.sensitivity_trials,args.seed,int(array[1:]),args.workers)
                except ValueError as e:sensitivities.append(dict(name=name,status='unavailable',cause=str(e)));continue
            matched_count=min(args.trials,args.sensitivity_trials)
            matched=aggregate(records[:matched_count],c['A0'],variance(c['psd'],c['scale'],c['q']))
            base_matched=aggregate(trials[:matched_count],context['A0'],variance(context['psd'],context['scale'],context['q']))
            paired=None
            if not matched['failures'] and not base_matched['failures']:
                a=np.array([r['amplitude'] for r in records[:matched_count]])
                b=np.array([r['amplitude'] for r in trials[:matched_count]])
                indices=np.random.default_rng([args.seed,int(array[1:]),9001]).integers(0,matched_count,(1000,matched_count))
                ratios=a[indices].std(axis=1,ddof=1)/b[indices].std(axis=1,ddof=1)
                paired=dict(scatter_ratio=matched['candidate_refit_scatter']/base_matched['candidate_refit_scatter'],
                    paired_bootstrap_MC95_ratio_interval=np.quantile(ratios,[.025,.975]),
                    interpretation='finite Monte Carlo comparison with common trial streams only; not a model or observational confidence interval')
            rec=dict(name=name,matched_budget_experiment=matched,matched_base_scatter=base_matched['candidate_refit_scatter'],paired_MC_comparison=paired,request=asdict(req),amplitude_supported=c['complete_amplitude_support'],outside_amplitude_range_fit_pixels=c['outside_amplitude_range_fit_pixels'],model=m['summary'],experiment=exp)
            sensitivities.append(rec)
            if name!='base':
                save_json(ar/f'trials-{name}.json',records)
                np.savez_compressed(ar/f'model-{name}.npz',psd=m['psd'],scale=m['scale'],amplitude_valid=m['valid'],background_mask=m['background_mask'],empirical=m['empirical'],pairs=m['pairs'])
        profile['sensitivity_wall_seconds']=time.perf_counter()-t
        # Double the periodic domain; normalization and projected variance must
        # remain stable. This is an internal numerical check, not source validity.
        m2=learn(sig,cov,support,h,p,replace(request,domain_factor=2));c2=make_context(sig,cov,support,h,beam,p,m2)
        w=influence(context);local=variance(model['psd'],context['scale'],w)
        regions=regional_summary(model,context)
        t=time.perf_counter();blanks=blank_checks(model,context,prior_measure,h,args.seed,int(array[1:]),args.workers,inner=min(32,args.sensitivity_trials));profile['blank_comparison_wall_seconds']=time.perf_counter()-t
        bgq=cov[model['background_mask']];fitq=context['source_coverage'][context['roi']]
        depth=dict(background_quantiles=np.quantile(bgq,[0,.1,.5,.9,1]),source_fit_quantiles=np.quantile(fitq,[0,.1,.5,.9,1]),
            source_fit_coverage_percentiles=np.searchsorted(np.sort(bgq),fitq)/len(bgq),
            complete_amplitude_support=context['complete_amplitude_support'],outside_range_pixels=context['outside_amplitude_range_fit_pixels'])
        pop=populations(args.maps/array,plan,model,context,h,p)
        summary.update(array=array,unchanged_measured_amplitude=float(p[0]),source_fixed_amplitude=context['source_fixed'],units='mJy/nominal beam; peak amplitude above fitted constant, not integrated flux',
            local_linear_refit_scatter=float(np.sqrt(local)),fixed_diagonal_only_scatter=float(np.sqrt(np.sum((context['q']*context['scale'])**2))),
            double_domain_fixed_variance_ratio=variance(m2['psd'],c2['scale'],c2['q'])/variance(model['psd'],context['scale'],context['q']),
            amplitude_supported=context['complete_amplitude_support'],outside_amplitude_range_fit_pixels=context['outside_amplitude_range_fit_pixels'],
            applicability='candidate conditional model calculation; observational applicability assessed separately from numerical checks',
            fixed_lag_projection=projected_ranges(context,context['q']),local_linear_refit_lag_projection=projected_ranges(context,w),
            broad_mode_response=broad_mode_response(context),
            long_mode_limit='model beyond trained lags is unconstrained; listed mode responses have no inferred physical mode amplitudes',
            sensitivity_scatter_range=sensitivity_range(sensitivities),
            sensitivity_scatter_unavailable_cause='no variant has a complete successful trial ensemble' if sensitivity_range(sensitivities) is None else None)
        save_json(ar/'summary.json',summary);save_json(ar/'trials.json',trials);save_json(ar/'sensitivity.json',sensitivities);save_json(ar/'model.json',model['summary']);save_json(ar/'regions.json',regions)
        save_json(ar/'source-depth.json',depth);save_json(ar/'contributions.json',pop);save_json(ar/'blank-comparison.json',blanks)
        save_json(ar/'regeneration.json',dict(header={k:v for k,v in context['header'].items() if k not in ['COMMENT','HISTORY','']},nominal_beam=beam,source_parameters=p,reference_parameters=context['reference_parameters'],A0=context['A0'],seed=args.seed,array_id=int(array[1:]),trial_seed='[seed, array_id, zero-based trial]',noise_model='model.npz',trial_count=args.trials))
        np.savez_compressed(ar/'model.npz',psd=model['psd'],scale=model['scale'],amplitude_valid=model['valid'],background_mask=model['background_mask'],empirical=model['empirical'],pairs=model['pairs'],source_scale=context['scale'],baseline=context['baseline'],source_support=context['support'],weights=context['q'])
        write_fits(ar/'noise-amplitude.fits',h,[('PIXEL_NOISE_SCALE',model['scale'],'mJy/beam'),('AMPLITUDE_VALID',model['valid'].astype('u1'),'boolean'),('SOURCE_MASK',model['source_mask'].astype('u1'),'boolean'),('BACKGROUND_MASK',model['background_mask'].astype('u1'),'boolean')],companion='POINT-ADDITIVE')
        for i in range(3):
            noise=np.where(context['support'],draw_noise(model['psd'],context['scale'],[args.seed,int(array[1:]),i]),np.nan)
            write_fits(ar/f'realization-{i:03d}.fits',context['header'],[('SOURCE_PLUS_NOISE',context['baseline']+noise,'mJy/beam'),('ADDITIVE_NOISE',noise,'mJy/beam')],companion='POINT-ADDITIVE')
        t=time.perf_counter();draw(ar,array,sig,h,model,context,trials,sensitivities,regions);profile['plotting_wall_seconds']=time.perf_counter()-t
        profile['array']=array;profiles.append(profile);summaries.append(summary)
        print(array,json.dumps({k:summary[k] for k in ['candidate_refit_scatter','mean_shift_from_A0','fixed_analytic_scatter','fixed_simulated_scatter','amplitude_supported','sensitivity_scatter_range']}),flush=True)
    require(all(digest(args.maps/a/'map.fits')==hashes[a] for a in saved),'parent MAP changed')
    require(digest(args.measurement)==provenance['measurement_sha256'],'original measurement changed')
    save_json(out/'summary.json',summaries)
    runtime=dict(source_revision=provenance['source_revision'],wall_seconds=time.perf_counter()-start,parent_process_cpu_seconds=time.process_time()-cpu,
        peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        memory_scope='macOS bytes; per-process high-water marks, not aggregate simultaneous memory',companion_verification_wall_seconds=companion_verification_seconds,profiles=profiles,workers=args.workers,command=vars(args),parent_maps_unchanged=hashes)
    runtime['command']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()};save_json(out/'RUN.json',runtime)
    report=['# POINT 152389: full-observation conditional additive uncertainty','',
        'These are model-based candidate additive scatters for the unchanged saved seven-parameter fit. Background representativeness must be assessed separately; passing propagation tests is not observational applicability. No attenuation, calibration, full-chain, Gaussian significance or FRUIT claim.','',
        '| Array | Measured peak | Candidate refit SD | MC mean shift from A0 | Fixed analytic / simulated SD | Trials / failures / bounds | Model-sensitivity SD range | Amplitude-depth support |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for s in summaries:
        report.append(report_row(s))
    report+=['','All amplitudes/scatters are mJy/nominal beam. The fit footprint is a40″ RADIUS. SD uses ddof1 and is not divided by sqrt(trials). MC mean shift is not applied to the published amplitude. Failed trials withhold primary aggregate; bound-limited finite trials remain included.','',
        f'Base MC relative precision on SD is approximately {1/np.sqrt(2*(args.trials-1)):.1%}; this excludes model-learning uncertainty. Sensitivity trials number{args.sensitivity_trials} per variant with shared deterministic seeds; ranges compare equal-budget prefixes including the base. Paired Monte Carlo ratio intervals are recorded separately. Model-sensitivity ranges are not confidence intervals.','',
        'Five coverage bins use second moments about one ordinary training-background mean; no clipping, per-bin mean or gradient removal. Correlation uses observed pair counts in both directions; a tapered lag model is projected to a nonnegative spectrum. Projection magnitude, missing pairs, tails, quadrant/coverage differences and depth extrapolation are recorded. Gaussian draws are cropped from a padded domain then amplitude-modulated. Long modes remain unknown.','']
    for a in [s['array'] for s in summaries if s.get('status')!='unavailable']:report +=[f"## {a}",'',f"[Noise model]({out/a/'noise-model.png'}) · [Sampling and sensitivity]({out/a/'measurement-and-sensitivity.png'}) · [Full numerical result]({out/a/'summary.json'})",'',f"[Background regions]({out/a/'regions.json'}) · [Depth support]({out/a/'source-depth.json'}) · [Network/chunk contributions]({out/a/'contributions.json'}) · [Existing blank comparisons]({out/a/'blank-comparison.json'})",'']
    report += [f'Companion wall time {runtime["wall_seconds"]:.3f} s. No RTC/PTC or native execution. Runtime profiles, parent hashes, seeds, model arrays, representative realizations and every trial are retained. Original maps and measurements unchanged.','',
        'Numerical reference: [SciPy chi-square distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html) supplies the stated finite Gaussian Monte Carlo variance comparison. The model is not an uncorrected masked periodogram.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n');manifest=out/'FILES_SHA256SUMS';manifest.write_text(''.join(f'{digest(f)}  {f.relative_to(out)}\n' for f in sorted(out.rglob('*')) if f.is_file() and f!=manifest))
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['maps','measurement','empirical','pairings','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',choices=['saved-descriptive-v1'],required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--trials',type=int,default=256);p.add_argument('--sensitivity-trials',type=int,default=64);p.add_argument('--seed',type=int,default=23152389)
    p.add_argument('--arrays',nargs='+',choices=['a1100','a1400','a2000']);run(p.parse_args())
if __name__=='__main__':main()
