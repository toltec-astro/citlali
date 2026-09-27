"""Bounded serial POINT latency experiment; shared pipeline owns all science."""
import gzip,json,os,platform,resource,shutil,subprocess,sys,time
from pathlib import Path
import psutil
from inspect_forward_map import require,digest
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound,verified
from point_processing import read,upstream,execute_branch,feedback_count
from point_pipeline_inputs import prepare
from point_rank_experiment import REFERENCE,OLD
from point_timing import emit,span

REPO=Path(__file__).resolve().parents[2]
RETAINED=REPO/'validation/ptc_rank_response_2026-09-26/compact'
LIMITS=dict(parent=23,trajectory=32,feedback=258,endpoint=32)
ABSOLUTE=dict(parent=26,trajectory=34,feedback=278,endpoint=34)


def consume(root,kind,identity):
    p=root/'attempts.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    require(kind in LIMITS,'unknown experiment operation')
    limits=LIMITS
    repair=root/'DEFECT-REPLACEMENTS.json'
    if repair.exists():
        r=read(repair);require(r.get('implementation_defect') and 0<r.get('maximum_replacements',0)<=2,'invalid defect replacement disposition');limits=ABSOLUTE
    require(sum(r['kind']==kind for r in rows)<limits[kind],'experiment attempt ceiling')
    require(not any(r['kind']==kind and r['identity']==identity for r in rows),'duplicate attempt forbidden')
    with p.open('a') as f:f.write(json.dumps(dict(kind=kind,identity=identity,epoch=time.time()))+'\n');f.flush()


def prepare_run(root,name,rank,N,input_root,method='observed-als'):
    require(rank in (10,15) and N in (0,1,3,5,10),'outside predeclared experiment')
    out=root/name;prepare(out,REFERENCE,input_root)
    p=read(out/'freeze/prospective.json');p.update(schema='citlali-point-speed-pca-v1',profile='POINT-SPEED-PCA-v1',feedback_count=N,experimental_method=method,trajectory_identity=name)
    p['ptc']['rank']=rank;p['scientific_recipe']['PTC']['rank']=rank
    budget=read(out/'freeze/budget.json');cells=budget['native_cells']
    budget.update(campaign_root=str(root),peak_ceiling_bytes=(26*256+9*16+320*16+48)*cells+2_000_000_000,maximum_branch_workers=1)
    require(shutil.disk_usage(root).free>budget['peak_ceiling_bytes']+budget['free_space_reserve_bytes'],'insufficient bounded campaign space')
    save_json(out/'freeze/budget.json',budget);p['budget']=bound(out/'freeze/budget.json')
    # Identical physical measurement operands, not transplanted noise or guard state.
    geometry=RETAINED/'operands/measurement-geometry.json';shutil.copyfile(geometry,out/'freeze/measurement-geometry.json')
    d=out/'freeze/shapes';d.mkdir()
    for branch in ('C-core-plus','C-core-minus','C-shoulder'):
        src=OLD/f'freeze/shapes/{branch}.fits.gz';dest=d/f'{branch}.fits';dest.write_bytes(gzip.decompress(src.read_bytes()))
        require(digest(dest)==read(OLD/'SHAPES.json')['rasters'][branch]['sha256'],'injection identity')
    p['experiment_operands']=[bound(out/'freeze/measurement-geometry.json')]+[bound(x) for x in sorted(d.iterdir())]
    save_json(out/'freeze/prospective.json',p)
    return out


def once(out,binary,entry_ns):
    """Exactly one fresh original-input trajectory inside the timed child."""
    from point_pipeline import validate_environment
    emit('python_entry',entry_ns=entry_ns)
    require(not (out/'EXECUTION.json').exists(),'no implicit resume')
    p=read(out/'freeze/prospective.json');N=feedback_count(out)
    rec=dict(status='running',rank=p['ptc']['rank'],method=p['experimental_method'],feedback_count=N)
    save_json(out/'EXECUTION.json',rec)
    try:
        with span('environment_input_verification'):
            rec['identity']=validate_environment(out,binary)
            for ref in p['experiment_operands']:verified(ref)
        upstream(out,binary)
        require(read(out/'UPSTREAM.json')['status']=='accepted','M0 processing/source gate failed')
        result=execute_branch(out,binary,'BASE-POINT','REF')
        rec.update(status='completed' if result['ready'] and result['completed_cycle']==N and result.get('endpoint_status')=='completed' else 'processing_failed',completed_cycle=result['completed_cycle'],reason=result.get('stop_reason',result.get('endpoint_reason')))
    except Exception as exc:rec.update(status='processing_failed',reason=str(exc))
    save_json(out/'EXECUTION.json',rec);emit('finalized',status=rec['status'])
    return 0 if rec['status']=='completed' else 1


def launch(root,out,binary,review,cache):
    p=read(out/'freeze/prospective.json');rv=read(review)
    require(rv['source_revision']==p['source_revision'] and rv['verdict']=='pass','exact-source independent preflight required')
    require(not (out/'LAUNCH.json').exists(),'no launch retry')
    identity=dict(run=out.name);consume(root,'parent',identity);consume(root,'trajectory',dict(**identity,branch='REF',cycle=0))
    env=dict(os.environ,CITLALI_POINT_TIMING=str(out/'timing.jsonl'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-speed-mpl',XDG_CACHE_HOME='/private/tmp/point-speed-cache')
    command=[sys.executable,str(REPO/'tools/timestream_successor/point_pipeline.py'),'speed-once','--output',str(out),'--binary',str(binary)]
    rec=dict(command=command,cache=cache,policy=dict(independent_workloads=1,network_workers=2,feedback_segment_workers=1,numeric_threads=1),preflight_review=bound(review),scope='POINT152430 a2000 networks11/12 on local Mac; not full instrument or Unity')
    save_json(out/'LAUNCH.json',dict(**rec,status='starting'))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic_ns();samples=[];seen={};ready_seen={}
    with (out/'outer.log').open('w') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            rss=0;cpu=0
            try:processes=[psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]
            except psutil.Error:processes=[]
            for proc in processes:
                try:
                    rss+=proc.memory_info().rss;c=proc.cpu_times();cpu+=c.user+c.system;seen[proc.pid]=max(seen.get(proc.pid,0),c.user+c.system)
                except psutil.Error:pass
            now=time.monotonic_ns();samples.append(dict(seconds=(now-started)/1e9,rss_bytes=rss,live_cpu_seconds=cpu))
            for f in out.glob('READY-*.json'):ready_seen.setdefault(f.name,(now-started)/1e9)
            time.sleep(.05)
        child.wait()
    ended=time.monotonic_ns();after=resource.getrusage(resource.RUSAGE_CHILDREN)
    events=[json.loads(s) for s in (out/'timing.jsonl').read_text().splitlines()] if (out/'timing.jsonl').exists() else []
    readiness=[dict(cycle=e['cycle'],published_seconds=(e['monotonic_ns']-started)/1e9,caller_observed_seconds=ready_seen.get(Path(e['ready_file']).name)) for e in events if e['event']=='ready']
    complete=[(e['monotonic_ns']-started)/1e9 for e in events if e['event']=='analysis_complete']
    rec['process_startups']=dict(outer_python=1,git_validation=2,version_queries=p['feedback_count']+3,native_processing=p['feedback_count']+2,total=8+2*p['feedback_count'],scope='explicit subprocess calls including validation, version and native processing; excludes preparation/comparison outside launch')
    rec.update(status='completed' if child.returncode==0 else 'failed',exit_code=child.returncode,start_monotonic_ns=started,exit_seconds=(ended-started)/1e9,analysis_complete_seconds=complete[-1] if complete else None,readiness=readiness,aggregate_tree_sampled_RSS=max((s['rss_bytes'] for s in samples),default=0),sampling_interval_seconds=.05,sampled_process_count=len(seen),cpu_seconds=dict(user=after.ru_utime-before.ru_utime,system=after.ru_stime-before.ru_stime),samples=samples)
    save_json(out/'LAUNCH.json',rec)
    return rec


def compare_prefix(out,reference,N):
    from compare_point_pipeline import compare
    return compare(out,reference,'a2000',read(REFERENCE)['reference_tolerances'],cycles=N,compare_endpoint=N==10)


def _phase_a(root,binary,review,input_root):
    require(not (root/'PHASE-A.json').exists(),'Phase A already started')
    root.mkdir(parents=True,exist_ok=True)
    record=dict(status='running',runs=[]);save_json(root/'PHASE-A.json',record)
    for rank in (10,15):
        for rep,N in [(1,10),(2,10),(3,10),(1,0),(1,1),(1,3),(1,5)]:
            name=f'A-rank{rank}-N{N}-rep{rep}';out=prepare_run(root,name,rank,N,input_root)
            result=launch(root,out,binary,review,'first measured use; cache not flushed; preflight hashing may warm inputs' if not record['runs'] else 'warm OS cache permitted; fresh scientific state')
            row=dict(name=name,rank=rank,N=N,rep=rep,status=result['status']);record['runs'].append(row);save_json(root/'PHASE-A.json',record)
            require(result['status']=='completed','Phase A native/source/endpoint failure; dependent work blocked')
            comparator=RETAINED/f'rank-{rank}' if N==10 else root/f'A-rank{rank}-N10-rep1'
            comparison=compare_prefix(out,comparator,N);save_json(out/'REGRESSION.json',comparison)
            require(comparison['pass_'],'unchanged science regression failed; stop before alternative methods')
            actual=[json.loads(s) for s in (out/'execution-attempts.jsonl').read_text().splitlines()] if (out/'execution-attempts.jsonl').exists() else []
            require(len(actual)==N and all(r['cycle']<=N for r in actual),'hidden later feedback work')
    record['status']='completed';save_json(root/'PHASE-A.json',record)

def phase_a(root,binary,review,input_root):
    require(not (root/'PHASE-A.json').exists(),'Phase A already started')
    root.mkdir(parents=True,exist_ok=True)
    try:
        machine=dict(platform=platform.platform(),machine=platform.machine(),cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),logical_cores=psutil.cpu_count(),physical_cores=psutil.cpu_count(logical=False),memory_total_bytes=psutil.virtual_memory().total,memory_available_bytes=psutil.virtual_memory().available,disk_free_bytes=shutil.disk_usage(root).free,python=sys.version,build_type='Release',compiler='AppleClang/Homebrew; exact build cache retained',library_threads=1,network_workers=2,feedback_workers=1)
        save_json(root/'MACHINE.json',machine)
        _phase_a(root,binary,review,input_root)
    except Exception as exc:
        record=read(root/'PHASE-A.json') if (root/'PHASE-A.json').exists() else {}
        record.update(status='blocked_gate',reason=str(exc));save_json(root/'PHASE-A.json',record)
        raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['phase-a']);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);a=p.parse_args()
    phase_a(a.output.resolve(),a.binary.resolve(),a.review.resolve(),a.input_root.resolve())
