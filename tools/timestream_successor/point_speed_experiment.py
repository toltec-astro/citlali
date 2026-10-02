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
from point_timing import emit,span,ready

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
    require(method in ('observed-als','warm-observed-als','masked-covariance-count-v1'),'unapproved method')
    out=root/name;prepare(out,REFERENCE,input_root)
    p=read(out/'freeze/prospective.json');p.update(schema='citlali-point-speed-pca-v1',profile='POINT-SPEED-PCA-v1',feedback_count=N,experimental_method=method,trajectory_identity=name)
    p['ptc']['rank']=rank;p['scientific_recipe']['PTC']['rank']=rank
    p['ptc']['method']='masked-covariance-count-v1' if method=='masked-covariance-count-v1' else 'observed-als'
    p['scientific_recipe']['PTC']['experimental_method']=method
    p['scientific_recipe']['PTC']['method']=('binary pair-overlap count covariance; positive algebraic eigenspace; validity-aware coefficients' if method=='masked-covariance-count-v1' else 'observed-entry ALS; previous own converged basis initializer; full current-residual solve' if method=='warm-observed-als' else p['scientific_recipe']['PTC']['method'])
    budget=read(out/'freeze/budget.json');cells=budget['native_cells']
    from point_storage import tree_bytes
    budget.update(campaign_root=str(root),campaign_retained_bytes=tree_bytes(root)-tree_bytes(out),peak_ceiling_bytes=(26*256+9*16+320*16+48)*cells+2_000_000_000,maximum_branch_workers=1)
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
        initial=read(out/'UPSTREAM.json')
        if initial['status']!='accepted':
            guard=initial.get('guard',{})
            status='source_rejected' if guard.get('complete') and not guard.get('accepted') else 'processing_failed'
            if guard and (out/f"M0/products/{p['array']}/map.fits").exists():
                state=out/'M0-state.json';save_json(state,dict(cycle_index=0,ready=False,status=status,guard=guard))
                ready(out,0,out/f"M0/products/{p['array']}/map.fits",dict(guard=guard),state)
            rec.update(status=status,reason=guard.get('message','upstream processing failed'))
            save_json(out/'EXECUTION.json',rec);emit('finalized',status=status);return 1
        result=execute_branch(out,binary,'BASE-POINT','REF')
        rec.update(status='completed' if result['ready'] and result['completed_cycle']==N and result.get('endpoint_status')=='completed' else 'processing_failed',completed_cycle=result['completed_cycle'],reason=result.get('stop_reason',result.get('endpoint_reason')))
    except Exception as exc:rec.update(status='processing_failed',reason=str(exc))
    save_json(out/'EXECUTION.json',rec);emit('finalized',status=rec['status'])
    return 0 if rec['status']=='completed' else 1


def launch(root,out,binary,review,cache,*,attempt_consumer=None,child_mode='speed-once'):
    p=read(out/'freeze/prospective.json');rv=read(review)
    require(rv['source_revision']==p['source_revision'] and rv['verdict']=='pass','exact-source independent preflight required')
    require(not (out/'LAUNCH.json').exists(),'no launch retry')
    identity=dict(run=out.name);counter=attempt_consumer or consume;counter(root,'parent',identity);counter(root,'trajectory',dict(**identity,branch='REF',cycle=0))
    env=dict(os.environ,CITLALI_POINT_TIMING=str(out/'timing.jsonl'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-speed-mpl',XDG_CACHE_HOME='/private/tmp/point-speed-cache')
    command=[sys.executable,str(REPO/'tools/timestream_successor/point_pipeline.py'),child_mode,'--output',str(out),'--binary',str(binary)]
    if child_mode=='forward':command+=['--config',p['reference']['path'],'--input-root',p['input_root']]
    rec=dict(command=command,cache=cache,policy=dict(independent_workloads=1,network_workers=read(out/'freeze/budget.json')['network_workers'],feedback_segment_workers=1,numeric_threads=1),preflight_review=bound(review),scope=f"{p.get('mode','POINT')}{p['observation']}/{p['subobservation']}/{p['scan']} {p['array']} networks{p['networks']} partial-array subset on local Mac; not full instrument or Unity")
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
    readiness=[dict(cycle=e['cycle'],status=e['status'],published_seconds=(e['monotonic_ns']-started)/1e9,caller_observed_seconds=ready_seen.get(Path(e['ready_file']).name)) for e in events if e['event']=='ready']
    complete=[(e['monotonic_ns']-started)/1e9 for e in events if e['event']=='analysis_complete']
    rec['process_startup_attempts']={'outer_python':1,**{kind:sum(e['event']=='process_start' and e['kind']==kind for e in events) for kind in ('git_revision','git_status','validation_version','native_version','native_processing')}}
    rec['process_startup_attempts']['scope']='instrumented explicit attempts, including failures; no inference from planned N'
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



def support_audit(out,reference,branch='REF'):
    from astropy.io import fits
    from compare_point_pipeline import available
    import numpy as np
    rows=[]
    for k in range(11):
        actual=out/f'run/BASE-POINT/{branch}/a2000/cycle-{k:02d}/native/products/total.fits' if k else out/'M0/products/a2000/map.fits' if branch=='REF' else out/f'initial/{branch}/native/products/total.fits'
        control=reference/f'run/BASE-POINT/REF/a2000/cycle-{k:02d}/native/products/total.fits' if k else reference/'M0/products/a2000/map.fits'
        with fits.open(available(actual),memmap=False) as a,fits.open(available(control),memmap=False) as b:
            fields={name:bool(np.array_equal(a[name].data,b[name].data,equal_nan=True)) for name in ('HITS','NORMALIZATION','SCI_SUPPORT','NORM_SUPPORT')}
            wcs=all(a[0].header.get(key)==b[0].header.get(key) for key in ('CRPIX1','CRPIX2','CDELT1','CDELT2','CRVAL1','CRVAL2','RADESYS','EQUINOX'))
            rows.append(dict(cycle=k,fields=fields,wcs_equal=wcs,actual_science_pixels=int(a['SCI_SUPPORT'].data.sum()),reference_science_pixels=int(b['SCI_SUPPORT'].data.sum())))
            require(all(fields.values()) and wcs,'method changed complete map support or normalization')
    save_json(out/f'SUPPORT-{branch}.json',rows);return rows


def checkpoint_analysis(root,out):
    from point_reference import context,endpoint
    summary=read(out/'run/BASE-POINT/REF/a2000/SUMMARY.json');c=context(out/'freeze/a2000');rows=[]
    for k in (1,3,5):
        consume(root,'endpoint',dict(run=out.name,cycle=k,offline=True))
        row=summary['histories'][k];dest=out/f'checkpoint-{k:02d}';started=time.perf_counter()
        sigma=endpoint(verified(row['total']),c,row['measurement'],dest)
        rows.append(dict(cycle=k,seconds=time.perf_counter()-started,uncertainties=sigma,scope='offline saved REF map companion after timing; not an early-terminating invocation; never fed back'))
    save_json(out/'CHECKPOINTS.json',rows)


def failure_details(out):
    """Inspect actual native failures; a crash is not an unsupported covariance."""
    reasons=[]
    paths=list(out.glob('upstream/products/network*/donor-continuity/ptc/receipt.yaml'))+list(out.glob('run/BASE-POINT/*/a2000/cycle-*/native/products/**/receipt.yaml'))
    from inspect_forward_map import read_yaml
    for p in paths:
        record=read_yaml(p)
        for s in record.get('segments',[record]):
            if s.get('converged') is False:reasons.append(s.get('stopping_reason','unknown'))
    reasons=sorted(set(reasons))
    execution=read(out/'EXECUTION.json') if (out/'EXECUTION.json').exists() else {}
    if execution.get('status')=='source_rejected':return dict(status='source_rejected',native_reasons=reasons,reason=execution.get('reason'))
    return dict(status='unsupported' if reasons and all('unsupported' in x or 'too-few-positive' in x for x in reasons) else 'solver_failed' if reasons else 'processing_failed',native_reasons=reasons)


def run_candidate_family(root,binary,review,input_root,phase,method,rank):
    from point_rank_experiment import run_injection
    tag='warm' if method=='warm-observed-als' else 'covariance';cells=[];representative=None
    control=root/f'A-rank{rank}-N10-rep1'
    for rep in (1,2,3):
        name=f'{phase}-{tag}-rank{rank}-N10-rep{rep}';out=prepare_run(root,name,rank,10,input_root,method)
        launched=launch(root,out,binary,review,'warm OS cache allowed; fresh process/raw parent/scientific state')
        row=dict(name=name,method=method,rank=rank,branch='REF',rep=rep,status=launched['status']);cells.append(row)
        if launched['status']=='completed':
            try:support_audit(out,control)
            except Exception as exc:row.update(status='support_failed',reason=str(exc))
        else:row.update(failure_details(out))
        if rep==1:
            if row['status']!='completed':
                for other in (2,3):cells.append(dict(method=method,rank=rank,branch='REF',rep=other,status='blocked',reason='predetermined first REF invalid; no repeat to rescue scientific failure'))
                break
            representative=out
        save_json(root/f'{phase}-{tag}-rank{rank}-progress.json',cells)
    if representative is not None:
        # These are controls, never raw-to-result latency. Own first REF supplies
        # immutable original-derived parent and method/rank-local noise package.
        for branch in ('C-core-plus','C-core-minus','C-shoulder'):
            started=time.perf_counter();row=dict(name=representative.name,method=method,rank=rank,branch=branch,status='running')
            try:
                r=run_injection(root,binary,rank,branch,out=representative)
                row.update(status='completed' if r['ready'] and r['completed_cycle']==10 else 'processing_failed',completed_cycle=r['completed_cycle'],reason=r.get('stop_reason'))
                if row['status']=='completed':support_audit(representative,control,branch)
            except Exception as exc:row.update(status='processing_failed',reason=str(exc))
            row['control_seconds']=time.perf_counter()-started;cells.append(row);save_json(root/f'{phase}-{tag}-rank{rank}-progress.json',cells)
        checkpoint_analysis(root,representative)
    else:
        for branch in ('C-core-plus','C-core-minus','C-shoulder'):cells.append(dict(method=method,rank=rank,branch=branch,status='blocked',reason='predetermined first REF invalid'))
    save_json(root/f'{phase}-{tag}-rank{rank}-progress.json',cells)
    return cells


def candidates(root,binary,review,input_root):
    require(read(root/'PHASE-A.json')['status']=='completed' and (root/'report/PHASE-A.md').is_file(),'Phase A report/regression gate absent')
    require(not (root/'PHASE-B.json').exists(),'no implicit candidate campaign retry')
    record=dict(status='running',matrix=[],phase_a_report=bound(root/'report/PHASE-A.md'));save_json(root/'PHASE-B.json',record)
    for method in ('warm-observed-als','masked-covariance-count-v1'):
        try:record['matrix']+=run_candidate_family(root,binary,review,input_root,'B',method,10)
        except Exception as exc:record['matrix'].append(dict(method=method,rank=10,status='processing_failed',reason=str(exc)))
        save_json(root/'PHASE-B.json',record)
    record['status']='completed' if len(record['matrix'])==12 and all(r['status']=='completed' for r in record['matrix']) else 'completed_with_failures'
    save_json(root/'PHASE-B.json',record)
    return 0 if record['status']=='completed' else 1


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['phase-a']);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);a=p.parse_args()
    phase_a(a.output.resolve(),a.binary.resolve(),a.review.resolve(),a.input_root.resolve())
