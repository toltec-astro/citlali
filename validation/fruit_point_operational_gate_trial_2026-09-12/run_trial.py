"""One predeclared operational gate trial, with current-residual PTC relearning."""
import datetime
import hashlib
import os
import platform
import resource
import signal
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
import candidate
import evaluation
from common import b, read, verify_freeze, HERE, PREVIOUS, OLD, OUT

class TrialLimit(RuntimeError):
    """A campaign limit must escape the per-trajectory failure handler."""

def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def make_estimators(data):
    null = np.load(OLD/'null20260911/R/pass00_maps.npz')['total']
    estimators, records = [], []
    for a in range(3):
        e = candidate.Estimator(data.x, data.ygrid, data.shape, data.S[a], data.D[a], data.O[a], data.Q[a])
        cal = e.calibrate(null[a])
        if not cal['available']:
            raise ValueError('fixed calibration unavailable')
        e.D = e.D & (np.hypot(data.x, data.ygrid) <= 60)
        estimators.append(e)
        records.append(cal)
    return estimators, records

def truths(data):
    empty = np.zeros((3,data.npix))
    def gauss(peak, center, widths):
        p = np.array([peak,*center,*np.log(widths),np.deg2rad(25),0,0,0])
        return np.where(data.D, b.gaussian(p,data.x,data.ygrid)[None,:], 0.)
    H = gauss(100, (17,-11), (12,8))
    states = {'N':empty, 'B':empty, 'H':H,
              'H-shift':gauss(100,(20,-9),(12,8)),
              'D':gauss(90,(17,-11),np.array([12,8])/np.sqrt(.9)),
              'C':np.load('/private/tmp/sci-fruit-point-starlet-preliminary-20260911-r0.1/coma4_truth.npz')['truth'],
              'T':H*np.array([1,.8,1])[:,None],
              'E':gauss(100,(58,-11),(12,8))}
    return states

class Trial:
    def __init__(self):
        self.start = time.monotonic()
        self.cleaning_calls = 0
        self.receipts = []
    def guard(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
        size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        elapsed = time.monotonic()-self.start
        if elapsed>3600 or rss>8*2**30 or size>4*2**30:
            raise TrialLimit('trial time/RSS/output resource limit')
        return dict(elapsed_seconds=elapsed, peak_rss_bytes=rss, output_bytes=size)

    def trajectory(self,data,parent,name,arm,estimators,truth,definition):
        dest = OUT/name/arm
        dest.mkdir(parents=True, exist_ok=False)
        model = np.zeros((3,data.npix))
        rows = []
        start = time.monotonic()
        eval_total = 0.
        core_total = 0.
        parent_hash = array_hash(parent)
        b.write(dest/'START.json',dict(parent_sha256_float64_C=parent_hash, parent_shape=parent.shape,
                                      arm=arm, case=name, maximum_passes=7))
        try:
            for k in range(7):
                self.guard()
                if time.monotonic()-start > 600:
                    raise ValueError('ten minute trajectory budget')
                t = time.monotonic()
                self.cleaning_calls += 1
                b.write(OUT/'CLEANING_CALLS.json',dict(started=self.cleaning_calls,last_case=name,last_arm=arm,last_pass=k))
                tod,state,rank_guard = data.clean(parent,model)
                clean_time = time.monotonic()-t
                t = time.monotonic()
                maps = data.grid(tod)
                del tod
                map_time = time.monotonic()-t
                core_total += clean_time+map_time
                map_core = core_total
                map_wall = time.monotonic()-start
                t = time.monotonic()
                measurements = evaluation.measure(data,maps,estimators)
                scores = None if truth is None else evaluation.score_truth(data,maps,measurements,truth,definition)
                eval_time = time.monotonic()-t
                eval_total += eval_time
                t = time.monotonic()
                if arm == 'P':
                    next_model, decisions = b.inference(data,maps,'P',None)
                    for d in decisions:
                        d.update(available=True,admitted=bool(d['admitted_pixels']))
                    selected = np.zeros((3,5,data.npix),bool)
                    trial = next_model.copy()
                else:
                    next_model, decisions, selected, trial = [], [], [], []
                    for a,e in enumerate(estimators):
                        u,d,w = e.infer(maps[a])
                        next_model.append(u)
                        decisions.append(d)
                        selected.append(np.zeros((5,data.npix),bool) if w is None else w)
                        trial.append(e.last_trial.copy())
                    next_model,selected,trial = map(np.array,[next_model,selected,trial])
                infer_time = time.monotonic()-t
                core_total += infer_time
                valid = all(d['available'] for d in decisions)
                t = time.monotonic()
                np.savez_compressed(dest/f'pass{k:02d}_maps.npz',total=maps,applied_model=model,
                                    next_model=next_model,solver_trial=trial,selected=selected)
                np.savez_compressed(dest/f'pass{k:02d}_state.npz',**state)
                output_time = time.monotonic()-t
                row = dict(case=name,arm=arm,pass_index=k,cleaning_passes=k+1,
                           clean_seconds=clean_time,map_seconds=map_time,inference_seconds=infer_time,
                           evaluation_seconds=eval_time,output_seconds=output_time,
                           cumulative_core_map_seconds=map_core,cumulative_map_wall_seconds=map_wall,
                           cumulative_core_seconds=core_total,cumulative_wall_seconds=time.monotonic()-start,
                           cumulative_evaluation_seconds=eval_total,measurement=measurements,truth_score=scores,
                           decisions=decisions,next_model_available=valid,application_rank_guard=rank_guard,
                           model_change_rms=float(np.sqrt(np.mean((next_model-model)**2))),**self.guard())
                b.write(dest/f'pass{k:02d}.json',row)
                rows.append(row)
                self.receipts.append(row)
                print(name,arm,k,round(row['cumulative_wall_seconds'],2),
                      [d['reason'] for d in decisions],flush=True)
                if not valid:
                    raise ValueError('required feedback solve unavailable; failed iterate not applied')
                model = next_model
            assert array_hash(parent) == parent_hash
            b.write(dest/'COMPLETE.json',dict(passes=7,parent_unchanged=True,
                                            wall_seconds=time.monotonic()-start,core_seconds=core_total))
        except Exception as error:
            if array_hash(parent) != parent_hash:
                raise RuntimeError('immutable parent changed') from error
            b.write(dest/'FAILURE.json',dict(reason=str(error),traceback=traceback.format_exc(),
                                           parent_unchanged=True,retained_passes=len(rows),
                                           wall_seconds=time.monotonic()-start))
            print('FAILED',name,arm,str(error),flush=True)
            if isinstance(error, TrialLimit):
                raise

def main():
    frozen = verify_freeze()
    OUT.mkdir(exist_ok=False)
    run = Trial()
    def alarm(*_):
        raise TrialLimit('one hour trial budget')
    signal.signal(signal.SIGALRM,alarm)
    signal.alarm(3600)
    b.write(OUT/'START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                freeze_sha256=b.digest(HERE/'FREEZE.json'),pid=os.getpid(),
                                python=platform.python_version(),numpy=np.__version__,scipy=b.scipy.__version__))
    try:
        with threadpool_limits(limits=4):
            b.write(OUT/'THREAD_POOLS.json',dict(actual=threadpool_info(),environment={
                k:os.getenv(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}))
            data = b.Data(read(PREVIOUS/'INPUT_PREFLIGHT.json'))
            old = np.load(OLD/'geometry.npz')
            for key,value in [('x',data.x),('y',data.ygrid),('D',data.D),('S',data.S),('O',data.O),
                              ('Q',data.Q),('valid',data.valid),('uid',data.uid),('edges',data.edges)]:
                np.testing.assert_array_equal(value,old[key])
            estimators,calibration = make_estimators(data)
            scene_truth = truths(data)
            definitions = read(HERE/'CASES.json')['states']
            np.savez_compressed(OUT/'geometry.npz',x=data.x,y=data.ygrid,shape=data.shape,D=data.D,S=data.S,
                                O=data.O,Q=data.Q,central=np.array([e.D for e in estimators]),
                                uid=data.uid,array=data.ar,network=data.nw,edges=data.edges,valid=data.valid)
            for name,truth in scene_truth.items():
                np.savez_compressed(OUT/(name+'_truth.npz'),truth=truth)
            b.write(OUT/'SETUP.json',dict(calibration=calibration,groups=len(data.groups),
                                         input_shape=data.y.shape,setup_seconds=time.monotonic()-run.start,
                                         definitions=definitions,source_truth_hashes={
                                             n:array_hash(t) for n,t in scene_truth.items()}))
            run.trajectory(data,data.y,'real123424','P',estimators,None,{})
            run.trajectory(data,data.y,'real123424','C',estimators,None,{})
            for i,seed in enumerate([20260911,20260912]):
                noise = b.nuisance(data,old['scales'],seed)
                noise_hash = array_hash(noise)
                for j,(name,truth) in enumerate(scene_truth.items()):
                    parent = noise+data.project(truth)
                    if name=='B':
                        plane=np.tile(13+7*data.x/90-4*data.ygrid/90,(3,1))
                        parent += data.project(plane)
                    if name=='T':
                        # The source truth is already scaled; scale only the
                        # nuisance part here so the entire affected stream is .8 H.
                        parent[:,data.ar==1] += -.2*noise[:,data.ar==1]
                    case = f'{name}_{seed}'
                    for arm in (['P','C'] if (i+j)%2==0 else ['C','P']):
                        run.trajectory(data,parent,case,arm,estimators,truth,definitions[name])
                    del parent
                assert array_hash(noise)==noise_hash
                del noise
            verify_freeze()
            b.write(OUT/'COMPLETE.json',dict(state='trial_finished_see_trajectory_statuses',
                                           cleaning_calls=run.cleaning_calls,maximum_cleaning_calls=238,
                                           trajectories=34,**run.guard()))
    except BaseException as err:
        b.write(OUT/'FAILURE.json',dict(error=str(err),traceback=traceback.format_exc(),
                                     cleaning_calls=run.cleaning_calls))
        raise
    finally:
        signal.alarm(0)
        b.write(OUT/'RECEIPTS.json',run.receipts)
        b.write(OUT/'PRODUCT_MANIFEST.json',dict(root=str(OUT),files=[
            dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=b.digest(p))
            for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='PRODUCT_MANIFEST.json']))

if __name__=='__main__':
    main()
