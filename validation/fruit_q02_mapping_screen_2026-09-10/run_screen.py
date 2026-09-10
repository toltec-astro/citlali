#!/usr/bin/env python3
"""Run only the owner-approved Q02 r0.6 local campaign; no Citlali or Unity."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/q02-mapping-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/q02-mapping-cache')
import argparse,datetime,json,platform,resource,signal,sys,time,traceback
import numpy as np
import scipy,netCDF4
from threadpoolctl import threadpool_limits,threadpool_info
from core import REPO,OLD,legacy,digest,write_json,require,tail_registry
from test_screen import run_checks
from synthetic import run_t1
from discovery import run_t2
from verify_products import verify
Q=REPO/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review'

class Run:
    def __init__(self,out,settings):self.out,self.settings,self.start=out,settings,time.monotonic()
    def check(self,stage=None):
        elapsed=time.monotonic()-self.start;size=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());peak=legacy.peak_bytes()
        if elapsed>self.settings['wall_time_limit_seconds'] or size>self.settings['output_limit_bytes'] or peak>self.settings['aggregate_memory_limit_bytes']:
            raise legacy.BoundExceeded(f'Bound exceeded: {elapsed=} {size=} {peak=}')
        if stage:
            record=dict(seconds=elapsed,stage=stage,peak_bytes=peak,all_new_output_bytes=size)
            with (self.out/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);args=parser.parse_args()
    require(Path(args.attempt).name==args.attempt,'Attempt must be a directory name')
    def forbid_children(event, arguments):
        if event in {'subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.system'}:
            raise RuntimeError('Child-process launch prohibited in this single-process campaign')
    sys.addaudithook(forbid_children)
    out=HERE/args.attempt;out.mkdir(exist_ok=False)
    settings=json.loads((HERE/'settings.json').read_text());run=Run(out,settings)
    sources=sorted(HERE.glob('*.py'))+[HERE/'settings.json',OLD,Q/'r0.6/MAPPING_SCREEN_PROTOCOL.md',Q/'r0.6/README.md',Q/'r0.6/REVIEW_MANIFEST.json',Q/'MAPPING_SCREEN_APPROVAL_2026-09-10.md',Q/'r0.3/FIRST_SCREEN_BINDINGS.md',Q/'r0.4/EXECUTION_PROPOSAL.md',Q/'r0.5/WEIGHT_PRECISION_PROTOCOL.md']
    source_rows=[dict(repository_path=p.relative_to(REPO).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sources]
    write_json(out/'RUN_START.json',dict(started_utc=datetime.datetime.now(datetime.UTC).isoformat(),decision=settings['decision'],proposal_commit=settings['proposal_commit'],source_rows=source_rows,environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,netCDF4=netCDF4.__version__,platform=list(os.uname()),pid=os.getpid()),process_bound='No child-process calls in harness; Python audit hook rejects subprocess/fork/spawn/system; ru_maxrss measures the sole process',thread_limits={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},inputs_opened=False))
    def timeout(*_):raise legacy.BoundExceeded('Campaign wall-clock deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(settings['wall_time_limit_seconds'])
    try:
        require(digest(Q/'r0.6/MAPPING_SCREEN_PROTOCOL.md')==settings['protocol_sha256'],'Protocol identity changed')
        require(digest(Q/'MAPPING_SCREEN_APPROVAL_2026-09-10.md')==settings['approval_sha256'],'Approval identity changed')
        with threadpool_limits(limits=1):
            write_json(out/'DETERMINISTIC_VERIFICATION.json',run_checks());write_json(out/'TAIL_REGISTRY.json',dict(probability=.05/256,count=242,tails=tail_registry()))
            write_json(out/'THREAD_POOLS.json',threadpool_info());run.check('Deterministic verification passed; no input or stochastic access yet')
            inputs=[]
            for item in settings['inputs']:
                require(Path(item['path']).stat().st_size==item['bytes'] and digest(item['path'])==item['sha256'],'Input identity mismatch before signal access')
                inputs.append(item)
            write_json(out/'INPUT_IDENTITIES.json',inputs)
            t1=run_t1(run,settings);t2=run_t2(run,settings);product_check=verify(out,settings,run)
            run.check('Campaign and independent product verification complete')
        # Reverify sources so the immutable attempt has a single implementation identity.
        for row in source_rows:require(digest(REPO/row['repository_path'])==row['sha256'],'Source changed during campaign')
        write_json(out/'COMPLETION.json',dict(status='COMPLETE',completed_utc=datetime.datetime.now(datetime.UTC).isoformat(),wall_seconds=time.monotonic()-run.start,peak_aggregate_memory_bytes=legacy.peak_bytes(),processes=1,T1_stochastic_maps=36864,T2_maps=72,deterministic_checks=15,product_verification=product_check['status'],scientific_disposition='See all cases and alerts; no automatic policy qualification',external_inputs='both hashes unchanged',reserved_129081='not opened'))
    except BaseException as error:
        write_json(out/'FAILURE.json',dict(status='INCOMPLETE_INTERPRETATION_BLOCKED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-run.start,peak_aggregate_memory_bytes=legacy.peak_bytes(),retention='This attempt is preserved; method/input/population/scope changes need review'))
        raise
    finally:
        signal.alarm(0)
        rows=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['PRODUCT_MANIFEST.json','PRODUCT_MANIFEST.json.sha256']]
        write_json(out/'PRODUCT_MANIFEST.json',dict(products=rows,payload_bytes=sum(r['bytes'] for r in rows),external_inputs_modified=False))
        (out/'PRODUCT_MANIFEST.json.sha256').write_text(digest(out/'PRODUCT_MANIFEST.json')+'  PRODUCT_MANIFEST.json\n')
if __name__=='__main__':main()
