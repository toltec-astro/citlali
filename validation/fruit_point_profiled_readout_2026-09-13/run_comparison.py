"""Execute only the registered terminal saved-map readout problems."""
import datetime,platform,resource,subprocess,time,traceback
import numpy as np
import scipy
from threadpoolctl import threadpool_limits,threadpool_info
from common import *
from profiled import fit_source,linear_profile,fit_record,diagnostics

def main():
    assert not (HERE/'START.json').exists()
    verify_freeze();assert preserve()==read(HERE/'PRESERVATION_START.json')
    start=time.perf_counter();rows=[]
    write(HERE/'START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        freeze_sha256=digest(HERE/'FREEZE.json'),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    try:
        with threadpool_limits(limits=4):
            write(HERE/'THREAD_POOLS.json',threadpool_info())
            problems=read(HERE/'PROBLEMS.json')
            with np.load(OUT/'geometry.npz') as z:x,y,central=z['x'],z['y'],z['central']
            for problem in problems:
                with np.load(problem['map_path']) as z:total=z['total'][problem['array_index']]
                mask=central[problem['array_index']]&(np.hypot(x,y)<=problem['radius'])&np.isfinite(total)
                assert int(mask.sum())==problem['pixels']
                linear=None
                if problem['primary']:
                    tick=time.perf_counter();old=np.array(problem['original_fit']['parameters'])
                    v=linear_profile(old[1:6],x[mask],y[mask],total[mask]);c=v['coefficients']
                    params=np.r_[c[0],old[1:6],c[1:]];sse=float(np.sum(v['residual']**2))
                    linear=dict(fit=fit_record(params,sse,int(mask.sum()),1,1,1),
                        diagnostics=diagnostics(v,1.),seconds=time.perf_counter()-tick)
                # Normal route gets ONLY the saved map and allowed domain/coordinate metadata.
                normal=fit_source(total,x,y,mask)
                row=dict(id=problem['id'],case=problem['case'],arm=problem['arm'],array=problem['array'],
                    radius=problem['radius'],primary=problem['primary'],pixels=problem['pixels'],
                    fixed_original_geometry=linear,repaired=normal)
                rows.append(row)
                write(HERE/'PARTIAL_RESULTS.json',rows)
                if len(rows)%12==0:print('completed',len(rows),'of',len(problems),flush=True)
            assert len(rows)==84
            write(HERE/'RESULTS.json',rows)
            (HERE/'RESULTS.json.sha256').write_text(digest(HERE/'RESULTS.json')+'  RESULTS.json\n')
            verify_freeze();assert preserve()==read(HERE/'PRESERVATION_START.json')
            complete=dict(problems=84,primary=48,safeguards=36,normal_starts=sum(r['repaired']['starts_attempted'] for r in rows),
                residual_evaluations=sum(r['repaired']['residual_evaluations'] for r in rows),
                linear_subsolves_normal=sum(r['repaired']['linear_subsolves'] for r in rows),linear_diagnostic_subsolves=48,
                normal_fitter_seconds=sum(r['repaired']['seconds'] for r in rows),
                diagnostic_linear_seconds=sum(r['fixed_original_geometry']['seconds'] for r in rows if r['primary']),
                wall_seconds=time.perf_counter()-start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
                results_sha256=digest(HERE/'RESULTS.json'),PTC_calls=0,feedback_fits=0,reserved_observations=0)
            write(HERE/'COMPLETE.json',complete)
            print(complete)
    except BaseException as err:
        write(HERE/'FAILURE.json',dict(error=str(err),traceback=traceback.format_exc(),completed=len(rows)))
        raise
if __name__=='__main__':main()
