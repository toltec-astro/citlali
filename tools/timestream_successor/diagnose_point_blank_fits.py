"""Read-only diagnosis of fixed archived POINT null trials; no admission updates."""
import argparse, collections, json, platform, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
import scipy
import inspect_point_source as point
from point_admission import fit_diagnostic
from point_additive import draw_noise
from point_corr_admission import trial_seed
from run_point_corr_admission import context, read, ARRAYS, REPO, source
from point_fruit_trajectory import bound, manifest, verify_manifest
from point_additive_diagnostics import save_json
from inspect_forward_map import digest, require


def capture(data, c, budget=None):
    """Intercept the existing fitter, preserving its objective and all settings.

    The optional 6000-evaluation probe is diagnostic only. Its result never
    enters a candidate, threshold or archived trial record.
    """
    original = point.least_squares
    captured = {}
    trace = []
    calls = 0
    nonfinite = 0
    best = np.inf
    start = time.perf_counter()

    def instrument(fun, initial, **kwargs):
        nonlocal calls, nonfinite, best
        if budget is not None:
            kwargs['max_nfev'] = budget
        captured['initial'] = list(initial)
        captured['settings'] = kwargs

        def residual(p):
            nonlocal calls, nonfinite, best
            r = fun(p)
            calls += 1
            nonfinite += int(not np.isfinite(r).all())
            cost = float(.5*np.sum(r*r))
            best = min(best, cost)
            if calls == 1 or calls % 250 == 0:
                trace.append(dict(residual_call=calls, cost=cost, best_cost=best, parameters=p.tolist()))
            return r

        fit = original(residual, initial, **kwargs)
        singular = np.linalg.svd(fit.jac, compute_uv=False)
        norms = np.sqrt(np.sum(fit.jac**2, axis=0))
        normalized = np.divide(fit.jac, norms, out=np.zeros_like(fit.jac), where=norms>0)
        scaled_singular = np.linalg.svd(normalized, compute_uv=False)
        captured.update(status=int(fit.status), message=fit.message, success=bool(fit.success),
                        nfev=fit.nfev, njev=fit.njev, parameters=fit.x.tolist(), cost=float(fit.cost),
                        optimality=float(fit.optimality), active_mask=fit.active_mask.tolist(),
                        finite_parameters=bool(np.isfinite(fit.x).all()), finite_residuals=bool(np.isfinite(fit.fun).all()),
                        jacobian_singular_values=singular.tolist(), normalized_jacobian_singular_values=scaled_singular.tolist(),
                        widths_sigma_arcsec=fit.x[3:5].tolist(), centroid_radius_arcsec=float(np.hypot(*fit.x[1:3])),
                        constant_only_cost=float(.5*np.sum((data[c['fixed'].disk]-np.mean(data[c['fixed'].disk]))**2)))
        return fit

    point.least_squares = instrument
    try:
        outcome = fit_diagnostic(data, c['support'], c['fixed'].header, c['fixed'].beam)
    finally:
        point.least_squares = original
    captured.update(outcome=outcome, residual_calls_including_numeric_Jacobian=calls,
                    nonfinite_residual_calls=nonfinite, trace=trace,
                    trace_scope='Every 250 residual calls, including finite-difference Jacobian calls; not accepted-iteration history.',
                    wall_seconds=time.perf_counter()-start)
    return captured


def run(parent, out):
    revision = source()
    require(not out.exists(), 'fresh output required')
    out.mkdir(parents=True)
    start = time.perf_counter()
    completion = read(REPO/'validation/point_152389_correlation_admission_2026-09-24/COMPLETION.json')
    require(str(parent) == completion['product_root'], 'wrong parent bundle')
    index = parent/'FILES_SHA256SUMS'
    require(digest(index) == completion['root_manifest']['sha256'], 'parent root manifest identity changed')
    entries = {name: sha for sha, name in (line.split('  ', 1) for line in index.read_text().splitlines())}
    consumed = {}

    def authenticate(relative):
        f = parent/relative
        require(relative in entries and digest(f) == entries[relative], 'changed consumed parent: '+relative)
        consumed[str(f)] = entries[relative]
        return f

    p = read(authenticate('freeze/prospective.json'))
    for a in ARRAYS:
        for key in ('inputs', 'geometry'):
            authenticate(str(Path(p['arrays'][a][key]['path']).relative_to(parent)))
    census = {}
    selection = []
    contexts = {}
    rows_by_array = {}
    for a in ARRAYS:
        contexts[a] = context(parent, a)
        census[a] = {}
        for phase in ('calibration', 'validation'):
            rows = read(authenticate(f'{phase}/{a}-trials.json'))
            if phase == 'calibration': rows_by_array[a] = rows
            for name in ('scenes', 'stress'):
                if name == 'stress' and phase == 'calibration': continue
                census[a][phase+'-'+name] = {}
                for scene in ('blank', 'core'):
                    selected = [r[name][scene] for r in rows if name in r]
                    census[a][phase+'-'+name][scene] = dict(count=len(selected),
                        causes=dict(collections.Counter(s['fit'].get('cause', 'success') for s in selected)),
                        finite_constrained=sum(s['fit'].get('constrained', False) for s in selected),
                        candidate_outcomes=dict(collections.Counter(s['candidate_status'] for s in selected)))
        rows = rows_by_array[a]
        failed = [r['trial'] for r in rows if not r['scenes']['blank']['numerically_valid']][:3]
        require(len(failed) == 3, 'three fixed failed cases required')
        selection.extend(dict(array=a, trial=i, scene='blank', role='failed') for i in failed)
        for constrained in (False, True):
            i = next(r['trial'] for r in rows if r['scenes']['blank']['fit']['status']=='success' and r['scenes']['blank']['fit']['constrained']==constrained)
            selection.append(dict(array=a, trial=i, scene='blank', role='constrained-control' if constrained else 'unconstrained-control'))
        selection.append(dict(array=a, trial=failed[0], scene='core', role='paired-core-control'))
    save_json(out/'PLAN.json', dict(source_revision=revision, parent_manifest=bound(index), consumed_inputs=consumed,
        selection=selection, budget_probes='Exactly first failed calibration trial per array; original start, max_nfev6000 only.',
        stop='18 exact-setting replays plus3 budget-only probes; no new noise draws, model changes, candidate or threshold publication.',
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform())))
    results = []
    for sel in selection:
        a = sel['array']; c = contexts[a]; row = rows_by_array[a][sel['trial']]
        seed = trial_seed(a, 'calibration', sel['trial'])
        require(seed == int(row['seed']), 'seed identity differs')
        noise = draw_noise(c['psd'], c['scale'], seed)
        data = np.where(c['support'], c[sel['scene']]+noise, np.nan)
        require(np.isfinite(data[c['fixed'].disk]).all(), 'incomplete/nonfinite fit domain')
        archived = row['scenes'][sel['scene']]['fit']
        replay = capture(data, c)
        require(replay['outcome'] == archived, 'original fit replay differs')
        record = dict(**sel, seed=str(seed), fit_pixels=int(c['fixed'].disk.sum()),
                      exact_archived_outcome=True, original=replay)
        if sel['role']=='failed' and sel['trial']==next(s['trial'] for s in selection if s['array']==a and s['role']=='failed'):
            record['budget_probe'] = capture(data, c, budget=6000)
        results.append(record)
        save_json(out/'CASES.json', results)
        print(json.dumps({**sel, 'status':replay.get('status'), 'nfev':replay.get('nfev'), 'probe_status':record.get('budget_probe',{}).get('status')}), flush=True)
    for name, sha in consumed.items(): require(digest(Path(name)) == sha, 'consumed original changed')
    require(digest(index) == completion['root_manifest']['sha256'], 'parent seal changed')
    save_json(out/'CENSUS.json', census)
    save_json(out/'RUN.json', dict(source_revision=revision, wall_seconds=time.perf_counter()-start,
        process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        replay_fits=len(results), budget_only_fits=sum('budget_probe' in r for r in results),
        independent_new_draws=0, consumed_originals_unchanged=True, consumed_files=len(consumed),
        parent_manifest=bound(index), command=sys.argv))
    manifest(out)
    verify_manifest(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.output)
