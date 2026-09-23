#!/usr/bin/env python3
"""Saved POINT152389 M0/T1/T3/T5/T10 noise comparison and T10 additive uncertainty."""
import argparse
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import scipy
import astropy
from point_additive import NoiseRequest, learn, variance, draw_noise
from point_additive_diagnostics import save_json, verify_companion, regional_summary
from point_fruit_trajectory import FixedMeasurements, bound, verified, verify_manifest, manifest
from point_fruit_diagnostics import load, patch_map
from point_empirical import stats
from point_empirical_io import write_fits
from point_post_fruit import (frozen_masks, measurement_values, operators, context_for, ensemble,
    joint_summary, propagation_checks, paired_comparison, contribution_maps, applicability, blank_remeasure, verify_cycle_contributions)
from run_point_fruit_ten import original_parents
from inspect_forward_map import require, read_yaml, digest

ARRAYS=('a1100','a1400','a2000')
CYCLES=(0,1,3,5,10)
REPO=Path(__file__).resolve().parents[2]
COMPLETION=REPO/'validation/point_152389_fruit_ten_cycles_2026-09-23/completion.json'
OWNER=REPO/'validation/point_152389_post_fruit_ab_2026-09-23/OWNER_DIRECTIVE.md'


class ModelUnavailable(ValueError):
    """Scientific support failure; parent/provenance failures remain fatal."""


def learned(*args, **kwargs):
    try:
        return learn(*args, **kwargs)
    except ValueError as exc:
        raise ModelUnavailable(str(exc)) from exc


def contextual(*args, **kwargs):
    try:
        return context_for(*args, **kwargs)
    except ValueError as exc:
        raise ModelUnavailable(str(exc)) from exc


def larger_comparison(signal, coverage, common, support, header, fixed, parameters, request, source_mask, weights):
    try:
        m=learned(signal,coverage,common,header,fixed.parameters,replace(request,mask_fwhm=5.),stamp_size=49,source_mask=source_mask)
        c=contextual(signal,coverage,support,header,fixed,parameters,m)
        return dict(status='available',background=stats(signal[m['background_mask']]),
                    scatters=[np.sqrt(variance(m['psd'],c['scale'],w)) for w in weights])
    except ModelUnavailable as exc:
        return dict(status='unavailable',cause=str(exc),background=None,scatters=[None]*4)


def read(path):
    return json.loads(Path(path).read_text())


def recover(ten):
    accepted=read(COMPLETION)
    require(ten.resolve()==Path(accepted['product_root']), 'foreign ten-cycle reference')
    verify_manifest(ten/'run',accepted['run_manifest_sha256'])
    verify_manifest(ten/'report',accepted['report_manifest_sha256'])
    p=read(ten/'run/provenance.json'); original=read(verified(p['original_provenance']))
    first=verified(p['first_cycle_manifest']).parent.parent
    verify_manifest(first/'cycle',p['first_cycle_manifest']['sha256'])
    verify_manifest(first/'diagnostics-r2',p['corrected_manifest']['sha256'])
    req=original['request']; maps,additive,admission=(Path(req[n]) for n in ('maps','additive','admission'))
    for key,root in [('additive',additive),('admission',admission)]:
        require(verify_companion(root,original['maps'])['manifest_sha256']==original[key+'_manifest'],'changed '+key)
    verified(original['measurement'])
    ap=read(additive/'provenance.json'); empirical=Path(ap['empirical']['path'])
    verify_manifest(empirical,ap['empirical']['manifest_sha256'])
    bindings=[bound(COMPLETION),bound(OWNER),bound(ten/'run/FILES_SHA256SUMS'),bound(ten/'report/FILES_SHA256SUMS'),
              p['original_provenance'],p['first_cycle_manifest'],p['corrected_manifest'],original['measurement'],
              bound(additive/'FILES_SHA256SUMS'),bound(admission/'FILES_SHA256SUMS'),bound(empirical/'FILES_SHA256SUMS')]
    for a in ARRAYS:
        parents=original_parents(first,a)
        bindings.extend(parents.values())
        require(digest(maps/a/'map.fits')==original['maps'][a],'changed M0')
        bundle=read_yaml(maps/a/'bundle.yaml')
        for name,sha in bundle['files'].items():
            verified(dict(path=str(maps/a/name),sha256=sha))
        # T10 consumes m9; saved m10 is bound but remains unapplied.
        r=read(ten/f'run/{a}/cycle-10/request.json'); state=read(ten/f'run/{a}/cycle-10/state.json')
        previous=read(verified(r['previous_state']))
        require(r['candidate']==previous['candidate'] and previous['cycle_index']==9 and state['cycle_index']==10,'T10 lineage differs')
        verified(r['candidate']); verified(state['candidate']);verified(state['total']);verified(state['receipt'])
        bindings.extend([r['candidate'],state['candidate'],state['total'],state['receipt']])
    return dict(first=first,maps=maps,additive=additive,admission=admission,empirical=empirical,bindings=bindings)


def saved_cycles(ten, parents, array, fixed):
    records=[]
    for k in CYCLES:
        if k==0:
            path=parents['maps']/array/'map.fits'
            ref=read(parents['first']/f'diagnostics-r2/{array}/measurements.json')['measurements']['M0']
        elif k==1:
            path=parents['first']/f'cycle/reference/{array}/products/total.fits'
            ref=read(parents['first']/f'diagnostics-r2/{array}/measurements.json')['measurements']['T1']
        else:
            path=ten/f'run/{array}/cycle-{k:02d}/native/products/total.fits'
            ref=read(ten/f'run/{array}/cycle-{k:02d}/diagnostics/measurements.json')['measurement']
        data,h=load(path);s,v,_,_=patch_map(path);actual=fixed.measure(s,v)
        require(actual['fit']['status']=='success','saved source fit unavailable')
        av,rv=np.asarray(measurement_values(actual)),np.asarray(measurement_values(ref))
        require(np.allclose(av,rv,rtol=1e-10,atol=1e-10),'bound peak/template/aperture reproduction failure')
        require(np.allclose(actual['fit']['parameters'],ref['fit']['parameters'],rtol=1e-10,atol=1e-10),'seven parameter reproduction failure')
        if records:
            base=records[0]
            for key in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','RADESYS','EQUINOX']:
                require(h[key]==base['header'][key],'cycle WCS differs')
            for key in ['HITS','NORMALIZATION','SCI_SUPPORT','NORM_SUPPORT']:
                require(np.array_equal(data[key],base['data'][key]),'cycle support/depth differs')
        records.append(dict(cycle=k,path=bound(path),data=data,header=h,measurement=actual,
                            reproduction_max_absolute_error=float(np.max(abs(av-rv)))))
    return records


def save_model(root,model,c,header):
    np.savez_compressed(root/'model.npz',psd=model['psd'],scale=model['scale'],amplitude_valid=model['valid'],
        source_mask=model['source_mask'],background_mask=model['background_mask'],empirical=model['empirical'],pairs=model['pairs'],
        source_scale=c['scale'],baseline=c['baseline'],source_support=c['support'],weights=c['weights'],measurement_domain=c['measurement_domain'])
    save_json(root/'model.json',model['summary'])
    write_fits(root/'noise-amplitude.fits',header,[('PIXEL_NOISE_SCALE',model['scale'],'mJy/beam'),
        ('AMPLITUDE_VALID',model['valid'].astype('u1'),'1'),('SOURCE_MASK',model['source_mask'].astype('u1'),'1'),
        ('BACKGROUND_MASK',model['background_mask'].astype('u1'),'1')],companion='POINT-ADDITIVE-AB')


def analyze_array(array,ten,parents,out,workers,seed):
    ar=out/array;ar.mkdir();profile={};t=time.perf_counter()
    fixed=FixedMeasurements(parents['maps']/array,parents['admission']/array,parents['additive']/array)
    cycles=saved_cycles(ten,parents,array,fixed)
    profile['loading_measurement_reproduction_seconds']=time.perf_counter()-t
    images=[r['data'] for r in cycles];h=cycles[0]['header'];geometries=[r['measurement']['fit']['parameters'] for r in cycles]
    common,source,geometry=frozen_masks(images,h,geometries,4.)
    _,large,largegeometry=frozen_masks(images,h,geometries,5.)
    save_json(ar/'frozen-domains.json',dict(base=geometry,larger=largegeometry,
        measurement='40 arcsec radius AST centered fit; original M0 template and apertures; trial-refitted constant',
        units='FK5 J2000 tangent offsets arcsec, map mJy/nominal beam',nominal_beam_arcsec=fixed.beam))
    np.savez_compressed(ar/'frozen-domains.npz',common_support=common,source_mask=source,larger_source_mask=large)
    save_json(ar/'measurement-reproduction.json',[dict(cycle=r['cycle'],map=r['path'],measurement=r['measurement'],
         max_absolute_error=r['reproduction_max_absolute_error'],rtol=1e-10,atol=1e-10) for r in cycles])
    t=time.perf_counter()
    plan=read_yaml(parents['maps']/array/'plan.yaml')
    contributions=contribution_maps(parents['maps']/array,plan,common.shape,images[0]['HITS'])
    contribution_checks={r['cycle']:verify_cycle_contributions(Path(r['path']['path']).parent,contributions) for r in cycles if r['cycle']>0}
    save_json(ar/'network-contribution-verification.json',contribution_checks)
    np.savez_compressed(ar/'network-contributions.npz',**contributions)
    profile['contribution_diagnostics_seconds']=time.perf_counter()-t
    prior=read(parents['empirical']/array/'measurements.json');rows=[];models=[]
    request=NoiseRequest(mask_fwhm=4.,seed=seed)
    base=context=None
    for record in cycles:
        k=record['cycle'];root=ar/('M0' if k==0 else f'T{k}');root.mkdir()
        d=record['data'];p=record['measurement']['fit']['parameters'];t=time.perf_counter()
        try:
            m=learned(d['PRIMARY'],d['NORMALIZATION'],common,h,fixed.parameters,request,stamp_size=49,source_mask=source)
            c=contextual(d['PRIMARY'],d['NORMALIZATION'],d['SCI_SUPPORT'].astype(bool),h,fixed,p,m)
        except ModelUnavailable as exc:
            row=dict(cycle=k,status='unavailable',cause=str(exc),inference_seconds=time.perf_counter()-t,diagnostics_seconds=0.)
            save_json(root/'comparison.json',row);rows.append(row)
            continue
        inference=time.perf_counter()-t;t=time.perf_counter()
        ap=applicability(d['PRIMARY'],d['NORMALIZATION'],h,fixed.parameters,m,contributions,c['weights'])
        blanks=blank_remeasure(d['PRIMARY'],d['NORMALIZATION'],common,m,c,prior)
        b=d['PRIMARY'][m['background_mask']]
        larger=larger_comparison(d['PRIMARY'],d['NORMALIZATION'],common,d['SCI_SUPPORT'].astype(bool),h,fixed,p,request,large,c['weights'])
        row=dict(cycle=k,background=stats(b),second_moment_about_zero=float(np.mean(b*b)),
            second_moment_about_training_mean=float(np.mean((b-b.mean())**2)),
            source_measurement=measurement_values(record['measurement']),
            fixed_background_scatter=[np.sqrt(variance(m['psd'],c['scale'],w)) for w in c['weights']],
            diagonal_scatter=[np.sqrt(np.sum((w*c['scale'])**2)) for w in c['weights']],
            larger_mask_background=larger['background'],larger_mask_fixed_background_scatter=larger['scatters'],
            larger_mask_status=larger,
            model=m['summary'],applicability=ap,analytic_scope='fixed background only; true source background refitted in T10 trials',
            inference_seconds=inference,diagnostics_seconds=time.perf_counter()-t)
        save_model(root,m,c,h);save_json(root/'comparison.json',row);save_json(root/'blank-measurements.json',blanks)
        save_json(root/'regions.json',regional_summary(m,c));models.append(m);rows.append(row)
        if k==10:base,context=m,c
    profile['cycle_inference_seconds']=sum(r['inference_seconds'] for r in rows)
    profile['cycle_diagnostics_seconds']=sum(r['diagnostics_seconds'] for r in rows)
    save_json(ar/'cycle-comparison.json',rows)
    if context is None:
        return dict(array=array,status='unavailable',cause='T10 model or measurement domain unavailable; see preserved cycle comparisons',cycle_comparison=rows,profile=profile)
    observed=measurement_values(cycles[-1]['measurement']);t=time.perf_counter()
    records=ensemble(context,256,seed,array,'base',workers)
    summary=joint_summary(records,context['noiseless'],observed)
    save_json(ar/'trials.json',records);save_json(ar/'source-uncertainty.json',summary)
    save_json(ar/'noiseless-measurement.json',context['noiseless_record'])
    profile['base_ensemble_wall_seconds']=time.perf_counter()-t
    allrecords=list(records);variants=[];t=time.perf_counter()
    definitions=[('mask5',replace(request,mask_fwhm=5.),large),('north',replace(request,region='north'),source),
                 ('south',replace(request,region='south'),source),('lag48',replace(request,lag_pixels=48),source)]
    for name,req,mask in definitions:
        root=ar/name;root.mkdir()
        try:
            m=learned(d['PRIMARY'],d['NORMALIZATION'],common,h,fixed.parameters,req,stamp_size=49,source_mask=mask)
            c=contextual(d['PRIMARY'],d['NORMALIZATION'],d['SCI_SUPPORT'].astype(bool),h,fixed,p,m)
            trials=ensemble(c,64,seed,array,name,workers);allrecords.extend(trials)
            comparison=paired_comparison(records[:64],trials,c['noiseless'],[seed,int(array[1:]),1001])
            a=applicability(d['PRIMARY'],d['NORMALIZATION'],h,fixed.parameters,m,contributions,c['weights'])
            rec=dict(name=name,status='calculated',comparison=comparison,applicability=a,
                source_depth_supported=c['complete_amplitude_support'],outside_range_pixels=c['outside_amplitude_range_fit_pixels'],
                measurement_domain_complete=bool(np.all(c['support'][c['measurement_domain']])),
                regional_support=req.region,mixture_support='see overlap by coverage in applicability; not a validated joint transfer',model=m['summary'])
            save_model(root,m,c,h);save_json(root/'trials.json',trials)
        except ValueError as exc:
            rec=dict(name=name,status='unavailable',cause=str(exc),request=asdict(req))
        save_json(root/'result.json',rec);variants.append(rec)
    profile['variant_ensemble_wall_seconds']=time.perf_counter()-t;t=time.perf_counter()
    checks=propagation_checks(context,records,seed,array)
    save_json(ar/'propagation-checks.json',checks);profile['propagation_checks_seconds']=time.perf_counter()-t
    # Demonstrate the cross term for background coupling from the SAME trials.
    successes=[r for r in records if r['status']=='success'];bg=np.array([r['fit']['parameters'][6] for r in successes])
    linear=np.array([r['linear_noise'] for r in successes]);coupling=[]
    for i,w in enumerate(context['weights']) if len(successes)>1 else []:
        raw=linear[:,i];a=w.sum();cross=np.cov(raw,bg,ddof=1)[0,1]
        predicted=raw.var(ddof=1)+a*a*bg.var(ddof=1)-2*a*cross
        measured=np.array([r['values'][i+1] for r in successes]).var(ddof=1)
        coupling.append(dict(name=summary['measurements'][i+1]['name'],raw_variance=raw.var(ddof=1),
            background_variance=bg.var(ddof=1),weight_sum=a,covariance_raw_background=cross,
            cross_term=-2*a*cross,total_variance_identity=predicted,actual_variance=measured,
            primary_available=summary['primary_available']))
        require(np.isclose(predicted,measured,rtol=1e-10,atol=1e-10),'background cross covariance mismatch')
    save_json(ar/'background-coupling.json',coupling)
    save_json(ar/'cycle-comparison.json',rows);save_json(ar/'sensitivity.json',variants)
    save_json(ar/'regeneration.json',dict(seed=seed,array=array,cycle=10,model='base',
        stream='[seed, integer array wavelength, 10, 0, trial]; variants use same stream with explicit model identity',
        header={k:v for k,v in context['header'].items() if k not in ['COMMENT','HISTORY','']},
        nominal_beam=fixed.beam,original_parameters=fixed.parameters,T10_parameters=p,model_path='T10/model.npz',
        noiseless=context['noiseless'],trial_count=256,sensitivity_count=64,measurement_names=summary['order']))
    for i in range(3):
        noise=np.where(context['support'],draw_noise(base['psd'],context['scale'],[seed,int(array[1:]),10,0,i]),np.nan)
        write_fits(ar/f'realization-{i:03d}.fits',context['header'],[('SOURCE_PLUS_NOISE',context['baseline']+noise,'mJy/beam'),('NOISE',noise,'mJy/beam')],companion='POINT-ADDITIVE-AB')
    for key in ('cpu_seconds','generation_wall_seconds','fitting_wall_seconds'):
        profile['summed_worker_'+key]=sum(r[key] for r in allrecords)
    t=time.perf_counter()
    from point_post_fruit_report import figures
    if len(models)==len(cycles):
        figures(ar,array,cycles,models,rows,summary,variants,checks,context)
    profile['plotting_seconds']=time.perf_counter()-t
    print(array,'scatters',[r['scatter'] for r in summary['measurements']],flush=True)
    return dict(array=array,source=summary,cycle_comparison=rows,sensitivity=variants,propagation_checks=checks,profile=profile)


def run(args):
    start=time.perf_counter();cpu=time.process_time()
    require(not args.output.exists(),'fresh separate output required')
    require(args.workers>=1,'positive worker count required')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'clean committed source required')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    t=time.perf_counter();parents=recover(args.ten);verifytime=time.perf_counter()-t
    args.output.mkdir(parents=True)
    save_json(args.output/'provenance.json',dict(source_revision=revision,inputs=parents['bindings'],
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*point*.py')},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__),
        inherited_condition='a1100 network3 scan5 iteration-limit-not-converged; no rescue',T10_uses='m9',m10_applied=False))
    save_json(args.output/'request.json',dict(command={k:str(v) for k,v in vars(args).items()},cycles=CYCLES,
        base_trials=256,variant_trials=64,variants=['mask5','north','south','lag48'],
        fixed_mask='four-major-FWHM union of selected saved fits; larger five-major-FWHM union',
        covariance=asdict(NoiseRequest(mask_fwhm=4.,seed=args.seed))))
    results=[]
    for a in ARRAYS:
        try:
            results.append(analyze_array(a,args.ten,parents,args.output,args.workers,args.seed))
        except ModelUnavailable as exc:
            unavailable=dict(array=a,status='unavailable',cause=str(exc),profile={})
            save_json(args.output/a/'unavailable.json',unavailable)
            results.append(unavailable)
    t=time.perf_counter();after=recover(args.ten)
    require(after['bindings']==parents['bindings'],'input references changed during companion')
    for ref in parents['bindings']:verified(ref)
    reverifytime=time.perf_counter()-t
    runtime=dict(wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,
        child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        memory_scope='macOS per-process high-water marks; child maximum, not simultaneous aggregate',
        workers=args.workers,library_threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
        before_verification_seconds=verifytime,after_verification_seconds=reverifytime,inputs_unchanged=True,
        source_revision=revision,profiles={r['array']:r['profile'] for r in results},scope='map companion only; no native science execution')
    save_json(args.output/'summary.json',results);save_json(args.output/'RUN.json',runtime)
    from point_post_fruit_report import report
    report(args.output,results,runtime)
    manifest(args.output)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ten',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--seed',type=int,default=923152389)
    run(parser.parse_args())
