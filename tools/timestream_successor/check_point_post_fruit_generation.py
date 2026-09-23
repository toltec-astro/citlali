#!/usr/bin/env python3
"""Bounded linear-only follow-up to a recorded post-FRUIT sampling-check failure.

Preserves the original result. No source fits, model learning or native runs.
"""
import argparse,json,time,resource,subprocess
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from point_additive import variance,draw_noise
from point_post_fruit import larger_same_covariance,dense_variance
from point_fruit_trajectory import verify_manifest,bound,verified,manifest
from point_additive_diagnostics import save_json
from inspect_forward_map import require


def check(args):
    started=time.perf_counter();cpu=time.process_time()
    require(not args.output.exists(),'fresh follow-up output required')
    require(args.draws==4096,'prespecified follow-up budget is4096, not tune-until-pass')
    verify_manifest(args.companion)
    old=json.loads((args.companion/args.array/'propagation-checks.json').read_text())
    require(any(not r['sampling_check'] or not r['larger_sampling_check'] for r in old['operators']),
            'follow-up requires a recorded sampling-check failure')
    inputs=[bound(args.companion/'FILES_SHA256SUMS'),bound(args.companion/args.array/'T10/model.npz')]
    saved=np.load(verified(inputs[1]));psd=saved['psd'];scale=saved['source_scale'];weights=saved['weights']
    larger=larger_same_covariance(psd);rows=[];trials=[]
    # Independent new key, fixed before execution. Grid identity is explicit;
    # coupling is not assumed between two differently sized white-noise arrays.
    lo,hi=chi2.ppf([.0005,.9995],args.draws-1)/(args.draws-1)
    for grid,model in [(0,psd),(1,larger)]:
        samples=[];seeds=[]
        for b in range(args.draws):
            seed=[args.seed,int(args.array[1:]),10,4096,grid,b]
            noise=draw_noise(model,scale,seed)
            samples.append(np.einsum('ijk,jk->i',weights,noise));seeds.append(seed)
        samples=np.asarray(samples)
        for i,w in enumerate(weights):
            theory=variance(model,scale,w);dense=dense_variance(model,scale,w)
            ratio=samples[:,i].var(ddof=1)/theory
            rows.append(dict(grid='base' if grid==0 else 'double',operator=i,analytic_variance=theory,
                dense_variance=dense,variance_ratio=ratio,sampling_check=bool(lo<=ratio<=hi),
                MC999_ratio_interval=[lo,hi],draws=args.draws,observed_mean=float(samples[:,i].mean()),
                mean_MC_SE=float(np.sqrt(theory/args.draws))))
        trials.append(dict(grid=grid,seeds=seeds,linear_noise_measurements=samples))
    for ref in inputs:verified(ref)
    args.output.mkdir(parents=True)
    result=dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
        inputs=inputs,array=args.array,seed=args.seed,original_checks=old,checks=rows,
        all_sampling_checks_pass=all(r['sampling_check'] for r in rows),
        scope='linear generator diagnosis only; original256 failed flag retained; no refits/relearning; does not validate observational representativeness',
        wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
        peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,workers=1,
        command={k:str(v) for k,v in vars(args).items()})
    save_json(args.output/'CHECK.json',result);save_json(args.output/'trials.json',trials);manifest(args.output)
    print(json.dumps(rows),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--companion',type=Path,required=True);p.add_argument('--array',choices=['a1100','a1400','a2000'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--draws',type=int,default=4096);p.add_argument('--seed',type=int,default=9234096)
    check(p.parse_args())
