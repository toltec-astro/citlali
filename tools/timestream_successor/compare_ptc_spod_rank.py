#!/usr/bin/env python3
"""Nine bounded existing-solver replays, no upstream rerun or selected winner.

First/middle/last saved segments; baseline ALS10, ALS5 and pairwise10.
Synthetic response is conditional on each published frozen basis. It excludes
upstream response, beam/pointing fidelity and data-dependent relearning.
"""
import argparse
import json
import os
for _thread_var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_var]='1'
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
from ptc_spod import Inputs,require,digest,supported_windows,fourier


def frozen_response(basis,mask):
    n,d=mask.shape
    template=np.exp(-.5*((np.arange(n)[:,None]-280-np.arange(d)[None,:]*.05)/2)**2)
    output=np.full_like(template,np.nan)
    for pattern in np.unique(mask,axis=0):
        if not pattern.any():continue
        times=np.flatnonzero(np.all(mask==pattern,axis=1));b=basis[pattern]
        u,s,v=linalg.svd(b,full_matrices=False)
        require(len(s)==basis.shape[1] and s[-1]**2>s[0]**2*1e-10,'frozen response rank failure')
        x=template[np.ix_(times,pattern)]
        proj=linalg.blas.dgemm(1,x,u)
        out=x-linalg.blas.dgemm(1,proj,u,trans_b=1)
        output[np.ix_(times,pattern)]=out
    x=template[mask];y=output[mask]
    return dict(template_amplitude=float(np.dot(x,y)/np.dot(x,x)),energy_fraction=float(np.dot(y,y)/np.dot(x,x)),
                positive_peak_fraction=float(y.max()/x.max()),negative_peak_fraction=float(y.min()/x.max()),
                scope='existing cost-probe CAL-grid Gaussian: width2 samples, stagger .05 sample/detector; frozen basis; no location subtraction; no upstream response or relearning')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ptc',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);ledger=Inputs();receipt=ledger.yaml(a.ptc/'receipt.yaml')
    learning=ledger.yaml(a.ptc.parent.parent/'learning-receipt.yaml',stop='spectra')
    ledger.bind(a.ptc.parent/'cal/receipt.yaml',receipt['source_CAL_receipt_sha256'])
    dt=learning['native_integration_seconds']*learning['explicit_array_lowpass']['factor']
    require(all(g['network']==learning['network'] for g in receipt['segments']),'network mismatch')
    binary=ledger.bind(a.binary);rows=[];start=time.perf_counter()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    with threadpool_limits(limits=1):
        for index in (0,len(receipt['segments'])//2,len(receipt['segments'])-1):
            g=receipt['segments'][index];n,d=g['scheduled_times'],g['detectors'];prefix=f'segment-{index}'
            def read(s,dt):return ledger.binary(a.ptc/(prefix+s),dt,g['files'][prefix+s])
            values=read('-input.f64','<f8').reshape(n,d);mask=read('-eligible.u8','u1').reshape(n,d).astype(bool)
            reference=read('-cleaned.f64','<f8').reshape(n,g['output_detectors'])[:,g['fit_columns_in_output']]
            for method,rank in [('observed-als',10),('observed-als',5),('pairwise-covariance',10)]:
                out=a.output/f'segment-{index}-{method}-{rank}';cmd=[str(binary),str(a.ptc),str(index),str(rank),method,'0.00001',str(out)]
                begun=time.perf_counter();run=subprocess.run(cmd,env=env,capture_output=True,text=True)
                subprocess_seconds=time.perf_counter()-begun
                require(run.returncode==0,run.stderr);r=ledger.yaml(out/'result.yaml')
                require(r['converged'] and r['retained']==g['eligible'],'trial did not retain identical successful support')
                y=np.fromfile(out/'cleaned.f64','<f8').reshape(n,d);basis=np.fromfile(out/'basis.f64','<f8').reshape(d,rank)
                exact=np.array_equal(y[mask],reference[mask]) if method=='observed-als' and rank==10 else None
                if exact is not None:require(exact,'baseline failed exact saved-output replay')
                common=mask.all(axis=1);z=y[common];z=z-z.mean(axis=0)
                cov=linalg.blas.dgemm(1,z,z,trans_a=1)/(len(z)-1);var=np.diag(cov)
                corr=cov/np.sqrt(var[:,None]*var[None,:]);upper=np.triu_indices(d,1)
                windows,_=supported_windows(mask,[(0,n)],round(2/dt),round(2/dt)//2)
                before=values.copy();before[~mask]=np.nan;after=y.copy();after[~mask]=np.nan
                spectral=None
                if len(windows)>=2:
                    f,q=fourier(before,windows,dt);_,v=fourier(after,windows,dt)
                    ratio=np.sum(abs(v)**2,axis=(0,2))/np.sum(abs(q)**2,axis=(0,2))
                    spectral={str(h):float(ratio[abs(f-h).argmin()]) for h in (.5,11,16,20)}
                rows.append(dict(segment=index,native_interval=g['native_interval'],method=method,rank=rank,command=cmd,
                                 subprocess_wall_seconds=subprocess_seconds,solver_fit_seconds=r['fit_seconds'],
                                 total_probe_wall_seconds=r['wall_seconds'],peak_rss_bytes=r['peak_rss_bytes'],threads=r['threads'],build_identity=r['build_identity'],
                                 same_eligible_samples=g['eligible'],baseline_byte_equal=exact,complete_comparison_rows=int(common.sum()),
                                 mean_abs_pair_correlation=float(np.mean(abs(corr[upper]))),median_abs_pair_correlation=float(np.median(abs(corr[upper]))),
                                 same_window_power_ratio=spectral,frozen_CAL_grid_response=frozen_response(basis,mask)))
    ledger.unchanged()
    (a.output/'comparison.json').write_text(json.dumps(dict(rows=rows,inputs_sha256=ledger.files,inputs_unchanged=True,
        wall_seconds=time.perf_counter()-start,rank_selection=False,relearned_source_injection=False,
        warning='training residuals and conditional toy response cannot qualify a science cleaner'),indent=2)+'\n')
    print('PASS: 3 identical baseline replays; six bounded alternatives; no scientific selection')


if __name__=='__main__':main()
