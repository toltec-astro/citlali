"""Run the four-control, 32-trial bright-POINT admission prototype on saved inputs."""
import argparse,json,os,platform,resource,subprocess,time
from pathlib import Path
import numpy as np
from astropy.io import fits
import scipy,astropy
from inspect_forward_map import digest,read_yaml,require
from point_empirical_io import load_parent,write_fits
from point_empirical import source_patch,patch
from point_additive import draw_noise
from point_additive_diagnostics import verify_companion,save_json
from point_admission import METHOD,CASES,evaluate,fit_diagnostic,summarize_trials
from point_admission_report import figures,report


def run(args):
    start=time.perf_counter();cpu=time.process_time();out=args.output.resolve()
    require(not out.exists(),'new output directory required')
    for parent in [args.maps,args.measurement.parent,args.additive]:
        require(not out.is_relative_to(parent.resolve()),'output must be separate from immutable inputs')
    saved=json.loads(args.measurement.read_text());require(saved['schema']=='citlali-point-source-description-v1' and saved['observation']==152389,'foreign measurement')
    descriptions={r['array']:r for r in saved['arrays']};require(set(descriptions)=={'a1100','a1400','a2000'},'three arrays required')
    hashes={a:read_yaml(args.maps/a/'bundle.yaml')['files']['map.fits'] for a in descriptions}
    t=time.perf_counter();prior=verify_companion(args.additive,hashes)
    prov=json.loads((args.additive/'provenance.json').read_text())
    require(prov['measurement_sha256']==digest(args.measurement) and prov['profile']=='saved-descriptive-v1','foreign additive source method')
    require(prior['manifest_sha256']=='d650deddab4886d9ad386183a6425e5309643cab8effb82fd645106dde3f38c8','requires reviewed Sept23 additive companion-r2')
    for name in ['point_additive.py','point_empirical.py','inspect_point_source.py']:
        require(digest(Path(__file__).parent/name)==prov['code_sha256'][name],'generator or saved method changed')
    verification_seconds=time.perf_counter()-t
    out.mkdir(parents=True)
    request=dict(method=METHOD,profile=args.profile,cases=[dict(name=n,threshold=t,noise_scale_multiplier=m) for n,t,m in CASES],trials=32,trial_indices=list(range(32)),anchor_convention='zero-based pixel coordinate; floor(pixel+0.5), exact half ties toward increasing index',domain='40 arcsec radius about AST center',sign='positive source-associated emission',candidate='M minus own fitted constant on anchor component; zero on valid exclusions; NaN on invalid/outside domain',connectivity=8)
    save_json(out/'request.json',request)
    provenance=dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),method=METHOD,
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*point*.py')},measurement_path=str(args.measurement.resolve()),measurement_sha256=digest(args.measurement),maps=str(args.maps.resolve()),additive=prior,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__),threading={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})
    save_json(out/'provenance.json',provenance);summaries={};profiles=[]
    for array,desc in descriptions.items():
        ar=out/array;ar.mkdir();t=time.perf_counter()
        full,h,bundle,plan,description,parameters,verified=load_parent(args.maps/array,desc)
        save_json(ar/'parent-verification.json',verified)
        signal,support,coverage,header,half=source_patch(full['PRIMARY'],full['SCI_SUPPORT'].astype(bool),full['NORMALIZATION'],h)
        reg=json.loads((args.additive/array/'regeneration.json').read_text());model=np.load(args.additive/array/'model.npz')
        require(np.allclose(parameters,reg['source_parameters'],rtol=1e-10,atol=1e-10),'noise/source parameter mismatch')
        row=int(round(h['CRPIX2']-1));column=int(round(h['CRPIX1']-1))
        scale=patch(model['scale'],row,column,half);valid=patch(model['amplitude_valid'],row,column,half).astype(bool)
        require(np.array_equal(support,model['source_support']) and np.array_equal(np.where(np.isfinite(scale),scale,0.),model['source_scale']),'saved source support/scale mismatch')
        q=model['weights'];beam=bundle['nominal_beam_fwhm_arcsec'];profile=dict(array=array,loading_verification_seconds=time.perf_counter()-t)
        t=time.perf_counter();reference,outputs=evaluate(signal,support,scale,valid,header,parameters,beam,q)
        records=dict(array=array,saved_parameters=parameters,reference=reference,cases={n:r['summary'] for n,r in outputs.items()})
        profile['actual_measurements_seconds']=time.perf_counter()-t
        ref=outputs['reference'];units='mJy/nominal beam'
        write_fits(ar/'source-relative.fits',header,[('SOURCE_RELATIVE',ref['brightness'],units),('PIXEL_SCALE',scale,units),('DOMAIN',ref['domain'].astype('u1'),'1'),('MAP_VALID',support.astype('u1'),'1'),('NOISE_VALID',valid.astype('u1'),'1'),('JOINT_VALID',ref['valid'].astype('u1'),'1')],companion='POINT-ADMISSION')
        for name,result in outputs.items():
            write_fits(ar/f'candidate-{name}.fits',header,[('CANDIDATE',result['candidate'],units),('ADMITTED',result['selected'].astype('u1'),'1'),('VALID',result['valid'].astype('u1'),'1'),('EXCLUDED',result['excluded'],units),('RATIO',result['z'],'1')],companion='POINT-ADMISSION')
        t=time.perf_counter();trials=[];cubes={n:[] for n,_,_ in CASES};masks={n:[] for n,_,_ in CASES}
        old_trials=json.loads((args.additive/array/'trials.json').read_text())
        for index in range(32):
            seed=[reg['seed'],reg['array_id'],index]
            noise=draw_noise(model['psd'],model['source_scale'],seed)
            realization=np.where(support,model['baseline']+noise,np.nan)
            raw=fit_diagnostic(realization,support,header,beam)
            trial=dict(trial=index,seed=seed,raw_fit=raw,cases={})
            if raw['status']=='success':
                require(np.allclose(raw['parameters'],old_trials[index]['parameters'],rtol=1e-10,atol=1e-10),'existing deterministic realization/fit differs')
                trial['reference'],cases=evaluate(realization,support,scale,valid,header,raw['parameters'],beam,q)
                trial['cases']={n:r['summary'] for n,r in cases.items()}
            else:cases={}
            for name,_,_ in CASES:
                cubes[name].append(cases[name]['candidate'] if name in cases else np.full(signal.shape,np.nan))
                masks[name].append(cases[name]['selected'] if name in cases else np.zeros(signal.shape,bool))
            trials.append(trial)
        profile['conditional_trials_seconds']=time.perf_counter()-t
        # NPZ preserves the trial axis explicitly without implying a third celestial FITS axis.
        np.savez_compressed(ar/'trial-candidates.npz',**{n:np.asarray(v) for n,v in cubes.items()},**{n+'_admitted':np.asarray(v) for n,v in masks.items()},valid=ref['valid'],domain=ref['domain'])
        save_json(ar/'trials.json',trials);records['simulation_summary']=summarize_trials(trials)
        save_json(ar/'measurements.json',records);summaries[array]=records
        t=time.perf_counter();figures(ar/'candidates.png',array,signal,parameters,header,outputs)
        profile['plotting_seconds']=time.perf_counter()-t;profiles.append(profile)
    unchanged={a:digest(args.maps/a/'map.fits') for a in descriptions};require(unchanged==hashes,'parent MAP mutation')
    require(digest(args.measurement)==provenance['measurement_sha256'],'measurement mutation')
    verify_companion(args.additive,hashes)
    runtime=dict(source_revision=provenance['source_revision'],wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,workers=1,profiles=profiles,prior_verification_seconds=verification_seconds,
        command={k:str(v) for k,v in vars(args).items()},parent_maps_unchanged=unchanged)
    save_json(out/'RUN.json',runtime);save_json(out/'summary.json',summaries);report(out,summaries,runtime)
    manifest=out/'FILES_SHA256SUMS';manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file() and p!=manifest))
    return runtime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['maps','measurement','additive','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile',choices=['saved-descriptive-v1'],required=True)
    print(json.dumps(run(parser.parse_args()),indent=2))


if __name__=='__main__':main()
