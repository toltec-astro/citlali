#!/usr/bin/env python3
"""Measure one fresh ordinary development invocation with explicit thread caps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import psutil
import yaml
from ptc_spod import digest


def run(binary,input_path,output,workers,terminal='ptc'):
    output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             VECLIB_MAXIMUM_THREADS='1',BLIS_NUM_THREADS='1',OMP_DYNAMIC='FALSE')
    request=dict(schema='citlali-development-v1',input=dict(path=str(input_path.resolve()),sha256=digest(input_path)),
                 output=str(output/'products'),terminal=terminal,network_workers=workers)
    config=output/'development.yaml';config.write_text(yaml.safe_dump(request))
    result=dict(command=[str(binary.resolve()),str(config)],executable_sha256=digest(binary),
                version=subprocess.check_output([str(binary),'--version'],text=True,stderr=subprocess.STDOUT),
                input_sha256=digest(input_path),workers=workers,thread_environment={k:env[k] for k in env if k.endswith('NUM_THREADS') or k in ('VECLIB_MAXIMUM_THREADS','OMP_DYNAMIC')},
                measurement='fresh process; whole-invocation CPU and sampled RSS; no concurrent diagnostic',samples=[])
    started=time.monotonic()
    with (output/'run.log').open('w') as log:
        child=subprocess.Popen(result['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                cpu=process.cpu_times()
                result['samples'].append(dict(seconds=time.monotonic()-started,rss_bytes=process.memory_info().rss,
                    user_seconds=cpu.user,system_seconds=cpu.system,os_threads=process.num_threads()))
            except (psutil.NoSuchProcess,psutil.ZombieProcess):break
            time.sleep(.5)
        child.wait()
    result.update(exit_code=child.returncode,wall_seconds=time.monotonic()-started,
        polled_peak_rss_bytes=max((x['rss_bytes'] for x in result['samples']),default=0),
        output_bytes=sum(p.stat().st_size for p in (output/'products').rglob('*') if p.is_file()))
    (output/'RUN.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','version')},indent=2),flush=True)
    return child.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[1,2,4],default=1);p.add_argument('--terminal',choices=['rtc-only','cal','ptc'],default='ptc')
    a=p.parse_args();raise SystemExit(run(a.binary,a.input,a.output.resolve(),a.workers,a.terminal))
