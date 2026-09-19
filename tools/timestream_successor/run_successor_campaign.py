#!/usr/bin/env python3
"""Owner-run fixed 1/2/4 campaign; observations use the ordinary successor CLI."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import yaml
from run_successor_observation import run
from ptc_spod import digest


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def compare(left,right,out,networks):
    out.mkdir();result=[]
    for nw in networks:
        command=[sys.executable,str(Path(__file__).with_name('compare_successor_outputs.py')),
                 str(left/'products'/f'network{nw}'),str(right/'products'/f'network{nw}'),'--output',str(out/f'network{nw}')]
        with (out/f'network{nw}.log').open('w') as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        result.append(dict(network=nw,exit_code=code))
    write(out/'SUMMARY.json',result)
    return all(r['exit_code']==0 for r in result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('binary','input','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--repeat',action='store_true',help='repeat 1 and fastest valid after equivalence checks')
    p.add_argument('--diagnostics',action='store_true');p.add_argument('--budget-seconds',type=float,default=24000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);campaign_start=time.monotonic()
    request=json.loads(a.input.read_text());ids=[r['network'] for r in request['networks']]
    record=dict(source_input_sha256=digest(a.input),host=platform.node(),platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else 'not-exposed',
        allocation={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
        cache='OS file cache not flushed; fixed order 1,2,4; later runs may benefit from warm files',
        cost_scope='fresh processes; build, verification and diagnostic outside timed reductions',runs=[],comparisons=[],diagnostics=[])
    write(a.output/'CAMPAIGN.json',record)
    for w in (1,2,4):
        dest=a.output/f'workers-{w}'
        code=run(a.binary,a.input,dest,w)
        row=json.loads((dest/'RUN.json').read_text());row.pop('samples');record['runs'].append(row)
        write(a.output/'CAMPAIGN.json',record)
        # A true execution failure is a preserved blocker, not a benchmark success.
        if code not in (0,2):raise RuntimeError(f'worker {w} execution failure; products and logs retained')
    for w in (2,4):
        started=time.monotonic();ok=compare(a.output/'workers-1',a.output/f'workers-{w}',a.output/f'equivalence-1-{w}',ids)
        record['comparisons'].append(dict(workers=w,pass_equivalence=ok,seconds=time.monotonic()-started))
        write(a.output/'CAMPAIGN.json',record)
        if not ok:raise RuntimeError('scheduling equivalence failed; no automatic normalization or tolerance widening')
    if a.repeat:
        fastest=min(record['runs'],key=lambda r:r['wall_seconds'])['workers']
        for number,w in enumerate(dict.fromkeys((1,fastest))):
            estimate=next(r['wall_seconds'] for r in record['runs'] if r['workers']==w)
            if a.budget_seconds-(time.monotonic()-campaign_start)<2*estimate+3600:
                record.setdefault('repeat_unavailable',[]).append(dict(workers=w,reason='time reserve for diagnostics and verification'));continue
            # Do not start an optional repeat with insufficient disk for a measured run.
            required=max(r['output_bytes'] for r in record['runs'])*1.2
            if shutil.disk_usage(a.output).free<required:
                record.setdefault('repeat_unavailable',[]).append(dict(workers=w,reason='insufficient measured output-space headroom'));continue
            dest=a.output/f'repeat-{number}-workers-{w}';code=run(a.binary,a.input,dest,w)
            if code not in (0,2):raise RuntimeError('optional repeat failed')
            row=json.loads((dest/'RUN.json').read_text());row.pop('samples');row['repeat']=True;record['runs'].append(row)
            ok=compare(a.output/'workers-1',dest,a.output/f'equivalence-repeat-{number}',ids)
            if not ok:raise RuntimeError('repeat equivalence failed')
            write(a.output/'CAMPAIGN.json',record)
    if a.diagnostics:
        for entry in request['networks']:
            nw=entry['network'];dest=a.output/f'diagnostic-network{nw}';start=time.monotonic()
            cmd=[sys.executable,str(Path(__file__).with_name('run_ptc_spod.py')),'--input',str(a.output/'workers-1/products'/f'network{nw}'),
                 '--config',entry['input']['path'],'--output',str(dest)]
            with (a.output/f'diagnostic-network{nw}.log').open('w') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            record['diagnostics'].append(dict(network=nw,exit_code=code,seconds=time.monotonic()-start,output=str(dest)))
            write(a.output/'CAMPAIGN.json',record)
        if any(r['exit_code'] for r in record['diagnostics']):raise RuntimeError('diagnostic failures explicitly recorded; remaining networks were assessed')
    record['state']='completed-equivalent-with-partial-results' if any(r['exit_code'] for r in record['runs']) else 'completed-equivalent';write(a.output/'CAMPAIGN.json',record)


if __name__=='__main__':main()
