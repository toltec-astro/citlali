#!/usr/bin/env python3
"""One authorized POINT152389 cycle: verify, zero control, first cycle, m1, stop."""
import argparse,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from inspect_forward_map import digest,read_yaml,require
from point_empirical_io import load_parent
from point_additive_diagnostics import verify_companion,save_json
from point_fruit_diagnostics import compare_zero,diagnose
from run_successor_observation import run as invoke

ARRAYS=('a1100','a1400','a2000')


def run(args):
    started=time.perf_counter();repo=Path(__file__).resolve().parents[2]
    require(not args.output.exists(),'fresh output required; never automatically iterate or overwrite')
    require(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()=='','committed clean source required')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    version=subprocess.check_output([str(args.binary),'--version'],text=True,stderr=subprocess.STDOUT)
    short=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=repo,text=True).strip();require(short in version,'binary/source revision mismatch')
    args.output.mkdir(parents=True)
    def bound(p):return dict(path=str(p.resolve()),sha256=digest(p))
    hashes={a:digest(args.maps/a/'map.fits') for a in ARRAYS};saved=json.loads(args.measurement.read_text());desc={r['array']:r for r in saved['arrays']}
    ad=verify_companion(args.admission,hashes);noise=verify_companion(args.additive,hashes)
    require(ad['manifest_sha256']=='42e656d6fedefc1a47599333983bc5a7db31acb42e78be0ec3449a80fe577f71','foreign admission')
    require(noise['manifest_sha256']=='d650deddab4886d9ad386183a6425e5309643cab8effb82fd645106dde3f38c8','foreign selection scale')
    for a in ARRAYS:load_parent(args.maps/a,desc[a])
    provenance=dict(source_revision=revision,binary=bound(args.binary),version=version,request={k:str(v) for k,v in vars(args).items()},maps=hashes,admission_manifest=ad['manifest_sha256'],additive_manifest=noise['manifest_sha256'],owner=bound(args.owner),measurement=bound(args.measurement),input_verification_seconds=time.perf_counter()-started)
    save_json(args.output/'provenance.json',provenance)
    def execute(mode,a):
        request=dict(schema='citlali-point-fruit-cycle-v1',method='point-152389-direct-one-cycle-v1',mode=mode,array=a,map_request=bound(args.map_request),parent_plan=bound(args.maps/a/'plan.yaml'),candidate=bound(args.admission/a/'candidate-reference.fits'),owner_binding=bound(args.owner))
        p=args.output/f'{mode}-{a}.json';save_json(p,request)
        rc=invoke(args.binary,p,args.output/mode/a,1,'point-fruit');require(rc==0,f'{mode} {a} failed: see preserved run.log')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda a:execute('zero',a),ARRAYS))
    checks={}
    for a in ARRAYS:
        root=args.output/'zero'/a/'products';receipt=read_yaml(root/'receipt.yaml')
        for s in receipt['segments']:
            require(s['input_eligibility_changed']==s['output_cause_changed']==0,'zero PTC support changed')
            require(s['zero_replay_maximum_cleaned_difference']<=1e-8,'zero PTC numerical mismatch')
        checks[a]=compare_zero(args.maps/a/'map.fits',root/'total.fits',desc[a]['nominal_calibration_beam_fwhm_arcsec'])
    save_json(args.output/'zero-control.json',checks)
    print('All three fresh-PTC zero controls passed; executing the one nonzero cycle.',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda a:execute('reference',a),ARRAYS))
    reports={}
    for a in ARRAYS:reports[a]=diagnose(a,args.maps/a,args.admission/a,args.additive/a,args.output/'zero'/a/'products',args.output/'reference'/a/'products',args.output/f'diagnostics-{a}')
    require({a:digest(args.maps/a/'map.fits') for a in ARRAYS}==hashes,'original maps changed')
    verify_companion(args.admission,hashes);verify_companion(args.additive,hashes)
    runtime=dict(wall_seconds=time.perf_counter()-started,array_workers=args.workers,numerical_threads_per_worker=1,stop_reason='authorized single nonzero feedback cycle completed; not convergence',m1_applied=False)
    save_json(args.output/'RUN.json',runtime);save_json(args.output/'summary.json',reports)
    manifest=args.output/'FILES_SHA256SUMS';manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(args.output)}\n' for p in sorted(args.output.rglob('*')) if p.is_file() and p!=manifest))
    return runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['binary','maps','map-request','measurement','admission','additive','owner','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3)
    a=p.parse_args();print(json.dumps(run(a),indent=2))

if __name__=='__main__':main()
