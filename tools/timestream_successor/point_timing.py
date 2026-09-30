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
    prospective=out/'freeze/prospective.json'
    if prospective.exists() and json.loads(prospective.read_text()).get('campaign_scope')=='rtc-point-transfer-v1':
        from point_source_evolution import export
        export(out,k,map_path,measurement)
    if not os.environ.get('CITLALI_POINT_TIMING'):return
    # Caller-visible file milestone after required synchronous publications.
    state=json.loads(Path(state_path).read_text())
    status='accepted' if state.get('ready') else state.get('stop_kind',state.get('status','processing_failed'))
    row=dict(cycle=k,map=str(map_path),measurement=measurement,state=str(state_path),status=status)
    p=out/f'READY-{k:02d}.json'
    pending=p.with_suffix('.pending')
    with pending.open('w') as f:json.dump(row,f);f.flush()
    os.replace(pending,p)
    emit('ready',cycle=k,ready_file=str(p),status=status,source_accepted=measurement['guard']['accepted'])
