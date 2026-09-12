"""Execute the two saved-map stages; this file cannot start a PTC replay."""
import datetime
import os
import platform
import resource
import signal
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from common import b, candidate, read, setup, verify_freeze, HERE, SAVED, OUT
import reevaluate
import stopping

def no_timestream(*args, **kwargs):
    raise RuntimeError('pre-PTC reads and cleaning are forbidden in saved-map stages')

def resources(start):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    elapsed = time.monotonic()-start
    if rss > 8*2**30 or size > 4*2**30 or elapsed > 3600:
        raise RuntimeError('saved-stage resource limit')
    return dict(wall_seconds=elapsed, peak_rss_bytes=rss, output_bytes=size)

def model_readout(data, e, model):
    total = float(np.sum(model[e.D]))
    rec = dict(peak=float(np.max(model[e.D])), sum=total,
               centroid=[float(np.sum(model[e.D]*x[e.D])/total) for x in [data.x, data.ygrid]] if total > 0 else None)
    try:
        fit = b.fit_source(model, data.x, data.ygrid, e.D)
        rec.update(fit=fit, fit_available=bool(fit['peak'] > 0 and not fit['boundary_rejection']
                                              and np.isfinite(fit['parameters']).all()))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        rec.update(fit_available=False, fit_error=str(error))
    return rec

def compare_models(data, e, loose, tight):
    a, bfit = model_readout(data, e, loose), model_readout(data, e, tight)
    valid = a['peak'] > 0 and bfit['peak'] > 0 and a['centroid'] is not None and bfit['centroid'] is not None
    d = dict(declared=a, tight=bfit, finite_positive_readouts=valid)
    if not valid:
        d.update(passed=False, reason='model_readout_unavailable')
        return d
    peak = abs(a['peak']/bfit['peak']-1)
    center = float(np.linalg.norm(np.array(a['centroid'])-bfit['centroid']))
    passed = peak <= .005 and center <= .1
    d.update(sampled_peak_relative_difference=peak, brightness_centroid_difference_arcsec=center,
             model_relative_L2=float(np.linalg.norm(loose-tight)/max(np.linalg.norm(tight), 1e-30)),
             brightness_relative_difference=abs(a['sum']/bfit['sum']-1))
    if a['fit_available'] and bfit['fit_available']:
        fp = abs(a['fit']['peak']/bfit['fit']['peak']-1)
        fc = float(np.linalg.norm(np.array(a['fit']['centroid'])-bfit['fit']['centroid']))
        d.update(fitted_peak_relative_difference=fp, fitted_centroid_difference_arcsec=fc,
                 fitted_readout_comparison_available=True)
        passed = passed and fp <= .005 and fc <= .1
    else:
        d.update(fitted_readout_comparison_available=False, readout_limitation='Gaussian parameter limit or unavailable readout')
    d['passed'] = bool(passed)
    return d

def stage_a(data, estimators, old_rows, start):
    dest = OUT/'evaluator'
    dest.mkdir(exist_ok=False)
    rows = []
    for r in old_rows:
        k = r['pass_index']
        with np.load(SAVED/r['case']/r['arm']/f'pass{k:02d}_maps.npz') as z:
            measurements = reevaluate.measure(data, z['total'], estimators, r['measurement'])
        row = dict(case=r['case'], arm=r['arm'], pass_index=k, measurements=measurements,
                   original_truth_score=r['truth_score'], original_cumulative_wall_seconds=r['cumulative_wall_seconds'])
        rows.append(row)
        if len(rows) % 30 == 0:
            print('A maps', len(rows), 'of', len(old_rows), flush=True)
        resources(start)
    b.write(dest/'MEASUREMENTS.json', rows)
    return rows

def stage_b(data, estimators, old_rows, start):
    dest = OUT/'stopping'
    dest.mkdir(exist_ok=False)
    rows = []
    for r in old_rows:
        if r['arm'] != 'C':
            continue
        k = r['pass_index']
        with np.load(SAVED/r['case']/'C'/f'pass{k:02d}_maps.npz') as z:
            maps, old_selected = z['total'], z['selected']
        for a, e in enumerate(estimators):
            resources(start)
            name = f"{r['case']}_pass{k:02d}_{b.ARRAYS[a]}"
            p = stopping.problem(e, maps[a])
            np.testing.assert_array_equal(p['omega'], old_selected[a])
            old = r['decisions'][a]
            np.testing.assert_array_equal(p['background'], old['background'])
            assert p['scale'] == old['solver_scale']
            row = dict(case=r['case'], pass_index=k, array=b.ARRAYS[a], name=name,
                       selected_per_band=p['omega'].sum(axis=1), background=p['background'],
                       scale=p['scale'], gradient_scale=p['gradient_scale'],
                       prior_decision=old, fixed_problem_checks=True)
            if not p['omega'].any():
                row.update(empty=True, qualified=True, reason='empty_support_exact_zero', wall_seconds=0.)
                np.savez_compressed(dest/(name+'.npz'), declared=np.zeros_like(maps[a]), tight=np.zeros_like(maps[a]))
            else:
                row['empty'] = False
                solved = stopping.solve_path(p)
                models = {}
                for label, snapshot in {**solved['snapshots'], 'final':solved['final']}.items():
                    model = np.zeros_like(maps[a])
                    model[e.D] = snapshot.pop('v')*p['scale']
                    assert np.isfinite(model).all() and np.all(model >= 0) and not np.any(model[~e.D])
                    models[label] = model
                row.update(solved)
                comparison = compare_models(data, e, models['declared'], models['tight']) if all(q in models for q in ['declared', 'tight']) else None
                row['comparison'] = comparison
                row['qualified'] = bool(solved['operational_stop_available'] and comparison is not None and comparison['passed'])
                np.savez_compressed(dest/(name+'.npz'), **models)
                print('B', name, 'qualified', row['qualified'],
                      'stops', {n: v['iterations'] for n, v in solved['snapshots'].items()},
                      'seconds', round(solved['wall_seconds'], 2), flush=True)
            b.write(dest/(name+'.json'), row)
            rows.append(row)
            b.write(dest/'RECEIPTS.json', rows)
    assert len(rows) == 144 and sum(not r['empty'] for r in rows) == 60
    return rows

def main():
    verify_freeze()
    OUT.mkdir(exist_ok=False)
    start = time.monotonic()
    b.Data.__init__ = no_timestream
    b.Data.clean = no_timestream
    def alarm(*_):
        raise RuntimeError('one-hour saved-stage budget')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(3600)
    b.write(OUT/'START.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_freeze_sha256=b.digest(HERE/'FREEZE.json'), pid=os.getpid(), cleaning_calls=0,
        pre_PTC_read_guard=True, python=platform.python_version(), numpy=np.__version__, scipy=b.scipy.__version__))
    try:
        with threadpool_limits(limits=4):
            b.write(OUT/'THREAD_POOLS.json', threadpool_info())
            data, estimators = setup()
            rows = read(SAVED/'RECEIPTS.json')
            assert len(rows) == 167
            stage_a(data, estimators, rows, start)
            solved = stage_b(data, estimators, rows, start)
            verify_freeze()
            b.write(OUT/'COMPLETE.json', dict(saved_maps=167, numerical_problems=len(solved),
                numerical_qualification_passed=all(r['qualified'] for r in solved),
                cleaning_calls=0, **resources(start)))
    except BaseException as error:
        b.write(OUT/'FAILURE.json', dict(error=str(error), traceback=traceback.format_exc(), cleaning_calls=0))
        raise
    finally:
        signal.alarm(0)
        b.write(OUT/'PRODUCT_MANIFEST.json', dict(root=str(OUT), files=[
            dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=b.digest(p))
            for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'PRODUCT_MANIFEST.json']))

if __name__ == '__main__':
    main()
