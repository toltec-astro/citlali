"""Optional monotonic instrumentation; no scientific state or policy."""
import functools,json,os,time
from contextlib import contextmanager
from pathlib import Path

def emit(event,**fields):
    path=os.environ.get('CITLALI_POINT_TIMING')
    if path:
        with open(path,'a') as f:
            f.write(json.dumps(dict(event=event,monotonic_ns=time.monotonic_ns(),pid=os.getpid(),**fields))+'\n')
            f.flush()

@contextmanager
def span(name,**fields):
    start=time.monotonic_ns()
    try:yield
    finally:emit('span',name=name,start_ns=start,seconds=(time.monotonic_ns()-start)/1e9,scope='inclusive wall; nested spans are not additive',**fields)

def timed(name):
    def decorate(fn):
        @functools.wraps(fn)
        def call(*args,**kwargs):
            with span(name):return fn(*args,**kwargs)
        return call
    return decorate

def ready(out,k,map_path,measurement,state_path):
    if not os.environ.get('CITLALI_POINT_TIMING'):return
    # Caller-visible file milestone after required synchronous publications.
    row=dict(cycle=k,map=str(map_path),measurement=measurement,state=str(state_path))
    p=out/f'READY-{k:02d}.json'
    with p.open('w') as f:json.dump(row,f);f.flush()
    emit('ready',cycle=k,ready_file=str(p),source_accepted=measurement['guard']['accepted'])
