#!/usr/bin/env python3
"""Original inputs -> ordinary M0 -> optional frozen bright-POINT BASE T10.

Preflight resolves only; report reads only. There is no implicit resume or rerun.
"""
import time as _launch_time
_PROCESS_ENTRY_NS = _launch_time.monotonic_ns()
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from inspect_forward_map import require, digest
from point_additive_diagnostics import save_json
from point_fruit_trajectory import verified, bound
from point_processing import upstream, execute_branch, read, selected_array
from point_pipeline_inputs import prepare

REPO = Path(__file__).resolve().parents[2]


def validate_environment(out, binary):
    from point_timing import emit
    emit('process_start',kind='git_revision')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    emit('process_start',kind='git_status')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(), 'clean committed candidate required')
    emit('process_start',kind='validation_version')
    version = subprocess.check_output([str(binary),'--version'],text=True)
    require(revision[:9] in version, 'candidate/native revision mismatch')
    require(binary.resolve().is_relative_to(REPO/'build'), 'use this durable candidate build')
    p=read(out/'freeze/prospective.json')
    require(p['source_revision']==revision,'preflight source revision differs')
    for ref in read(verified(p['dependencies']))['inputs']+read(verified(p['generated_inputs'])):verified(ref)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        require(os.environ.get(name)=='1','explicit one-thread numeric environment required: '+name)
    return dict(source_revision=revision,binary=bound(binary),version=version,python=__import__('sys').executable)


def execute(mode,out,binary):
    identity=validate_environment(out,binary)
    attempt=out/'EXECUTION.json'
    require(not attempt.exists(),'execution already attempted; no implicit retry')
    start=time.perf_counter();record=dict(status='running',mode=mode,identity=identity,limits=dict(raw_to_M0=1,trajectories=1 if mode=='feedback' else 0,feedback_applications=10 if mode=='feedback' else 0,endpoint_analyses=1 if mode=='feedback' else 0))
    save_json(attempt,record)
    try:
        upstream(out,binary)
        initial=read(out/'UPSTREAM.json')
        if initial['status']=='failed':raise RuntimeError('upstream processing failed; see UPSTREAM.json')
        if initial['status']!='accepted':
            record.update(status='source_rejected' if initial['guard']['complete'] else 'processing_failed',reason=initial['guard']['message'])
        elif mode=='feedback':
            r=execute_branch(out,binary,'BASE-POINT','REF')
            record.update(status='completed' if r['ready'] and r['completed_cycle']==10 and r.get('endpoint_status')=='completed' else 'source_rejected' if r.get('stop_reason') and r['histories'][-1].get('stop_kind')=='source_retention_miss' else 'processing_failed',completed_cycle=r['completed_cycle'],reason=r.get('stop_reason',r.get('endpoint_reason')))
        else:record.update(status='completed',completed_cycle=0)
    except Exception as exc:
        record.update(status='processing_failed',reason=str(exc))
    record['seconds']=time.perf_counter()-start;save_json(attempt,record)
    return 0 if record['status'] in ('completed','source_rejected') else 1


def inspect_saved(out):
    """Read relocated compact records. Never inspect raw inputs or invoke processing."""
    execution=read(out/'EXECUTION.json');p=read(out/'freeze/prospective.json');a=p['array']
    initial=read(out/'UPSTREAM.json');summary=out/f'run/BASE-POINT/REF/{a}/SUMMARY.json'
    require((out/f'M0/products/{a}/map.fits').is_file() or (out/f'M0/products/{a}/map.fits.gz').is_file(),'required saved M0 absent; reporting never regenerates')
    result=dict(schema='citlali-bright-point-report-v1',observation=p['observation'],array=a,status=execution['status'],execution=execution,initial=initial,units=dict(peak='mJy/nominal beam',centroid_sigma='arcsec',aperture='mJy/nominal beam arcsec^2'),limitations=['bright POINT development reference only','conditional covariance transfer, not full adaptive uncertainty','existing injection controls bypass RTC/CAL','shoulder response limitation remains','local evidence does not qualify all modes or production'])
    if execution['mode']=='feedback' and initial['status']=='accepted':
        r=read(summary);result.update(histories=r['histories'],endpoint_uncertainty=r.get('endpoint_uncertainty'),completed_cycle=r['completed_cycle'],m10_applied=r['m10_applied'])
        for row in r['histories']:
            k=row['cycle_index'];local=out/f'run/BASE-POINT/REF/{a}'/('initial' if k==0 else f'cycle-{k:02d}')/'candidate.fits'
            if 'candidate' in row:require(local.exists() or Path(str(local)+'.gz').exists(),'required saved candidate absent; reporting never regenerates')
            if k and 'total' in row:
                total=local.parent/'native/products/total.fits';require(total.exists() or Path(str(total)+'.gz').exists(),'required saved total absent; reporting never regenerates')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','forward','feedback','report','rank-preflight','rank-run','rank-report','speed-once','speed-run','speed-candidates','rtc-run','rtc-once'])
    p.add_argument('--output',type=Path,required=True,help='explicit run root, fresh for preflight')
    p.add_argument('--config',type=Path);p.add_argument('--input-root',type=Path);p.add_argument('--binary',type=Path)
    p.add_argument('--review',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='rtc-run':
        from point_rtc_experiment import run_family
        require(a.binary and a.review and a.input_root and a.config,'RTC run requires binary, review, original root and exact family config')
        return run_family(out,a.binary.resolve(),a.review.resolve(),a.input_root.resolve(),a.config.resolve())
    if a.mode=='rtc-once':
        from point_rtc_experiment import once
        return once(out,a.binary.resolve(),_PROCESS_ENTRY_NS)
    if a.mode=='speed-candidates':
        from point_speed_experiment import candidates
        require(a.binary and a.review and a.input_root,'candidate run requires binary, review and original input root')
        return candidates(out,a.binary.resolve(),a.review.resolve(),a.input_root.resolve())
    if a.mode=='speed-run':
        from point_speed_experiment import phase_a
        require(a.binary and a.review and a.input_root,'speed run requires binary, review and original input root')
        phase_a(out,a.binary.resolve(),a.review.resolve(),a.input_root.resolve());return 0
    if a.mode=='speed-once':
        from point_speed_experiment import once
        return once(out,a.binary.resolve(),_PROCESS_ENTRY_NS)
    if a.mode.startswith('rank-'):
        import point_rank_experiment as experiment
        if a.mode=='rank-preflight':
            require(a.input_root is not None,'input-root required');experiment.preflight(out,a.input_root.resolve());return 0
        if a.mode=='rank-run':
            require(a.binary is not None and a.review is not None,'binary and independent review required');return experiment.run(out,a.binary.resolve(),a.review.resolve())
        from report_point_rank import report
        report(out);return 0
    if a.mode=='report':print(json.dumps(inspect_saved(out),indent=2,allow_nan=False));return 0
    require(a.config is not None and a.input_root is not None and a.binary is not None,'config, input-root and binary required')
    require(a.binary.is_file() and os.access(a.binary,os.X_OK),'missing executable')
    if a.mode=='preflight' or not out.exists():prepare(out,a.config.resolve(),a.input_root.resolve())
    else:
        r=read(out/'freeze/prospective.json');require(r['reference']['sha256']==digest(a.config) and r['input_root']==str(a.input_root.resolve()),'prepared configuration/root differs')
    if a.mode=='preflight':
        save_json(out/'ENVIRONMENT.json',dict(binary=bound(a.binary),version=subprocess.check_output([str(a.binary),'--version'],text=True),python=__import__('sys').executable));return 0
    return execute(a.mode,out,a.binary.resolve())

if __name__=='__main__':raise SystemExit(main())
