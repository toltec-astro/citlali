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


def report(parent, out):
    """Summarize recorded fits; no optimizer calls or new field realizations."""
    import matplotlib.pyplot as plt
    revision = source()
    verify_manifest(out)
    rows = read(out/'CASES.json')
    census = read(out/'CENSUS.json')
    dest = out/'report'
    dest.mkdir()
    failed = [r for r in rows if r['role']=='failed']
    require(len(failed)==9 and all(r['original']['status']==0 and r['original']['nfev']==3000 for r in failed), 'diagnostic conclusion no longer matches')
    require(all(r['original']['nonfinite_residual_calls']==0 for r in rows), 'unexpected nonfinite objective')
    factor = np.sqrt(8*np.log(2))
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout='constrained')
    summaries = []
    for i, a in enumerate(ARRAYS):
        first = next(r for r in failed if r['array']==a)
        base, probe = first['original'], first['budget_probe']
        for record, label, color in ((base, 'Original budget 3000', '#315b89'), (probe, 'Probe budget 6000', '#bc5b32')):
            trace = record['trace']
            x = np.array([t['residual_call'] for t in trace])
            best = np.array([t['best_cost'] for t in trace])
            params = np.array([t['parameters'] for t in trace])
            axes[i,0].plot(x, (best-record['cost'])/record['constant_only_cost'], label=label, color=color)
            axes[i,1].plot(x, params[:,0], label=label, color=color)
            axes[i,2].plot(x, np.min(params[:,3:5],axis=1)*factor, label=label, color=color)
        axes[i,0].set(yscale='symlog', ylabel='Cost above own terminal / constant-only cost', title=f'{a}: first failure, trial {first["trial"]}')
        axes[i,0].set_yscale('symlog', linthresh=1e-9)
        axes[i,1].set(ylabel='Fitted amplitude (mJy/nominal beam)', title='Noise-peak amplitude')
        axes[i,2].axhline(2, color='black', ls='--', label='2 arcsec pixel spacing')
        axes[i,2].axhline(.5*factor, color='gray', ls=':', label='Inherited width lower bound')
        axes[i,2].set(ylabel='Minor FWHM (arcsec)', title='Narrowing noise feature')
        for ax in axes[i]: ax.set_xlabel('Residual calls, including numerical Jacobian')
        summaries.append(dict(array=a, trial=first['trial'], original_nfev=base['nfev'], probe_nfev=probe['nfev'],
            original_amplitude=base['parameters'][0], probe_amplitude=probe['parameters'][0],
            relative_cost_decrease=(base['cost']-probe['cost'])/base['cost'],
            original_minor_FWHM_arcsec=min(base['parameters'][3:5])*factor,
            probe_minor_FWHM_arcsec=min(probe['parameters'][3:5])*factor))
    axes[0,0].legend(fontsize=8); axes[0,2].legend(fontsize=8)
    fig.suptitle('Existing blank-fit diagnosis: all three budget probes still hit their limit\nEvaluation traces include finite-difference calls; they are not accepted-iteration histories')
    fig.savefig(dest/'blank-fit-traces.png', dpi=150, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    lines = ['# POINT 152389: bounded blank-fit diagnosis', '',
        '**The sampled failures exhaust the function-evaluation budget while fitting narrow noise peaks.** All nine exact-setting failed replays return optimizer status0, “The maximum number of function evaluations is exceeded,” at nfev=3,000. All 1,257 fit pixels, terminal parameters and residual evaluations are finite. There is no sampled NaN, missing-coverage or solver-exception failure.', '',
        'All 18 archived outcomes reproduce exactly. The three isolated 6,000-evaluation probes still fail. They use the original start, data, objective, bounds and tolerances; only the budget differs. These are diagnostic fits, not replacement calibration outcomes.', '',
        '| Array | Probe trial | Amplitude at 3000 | Amplitude at 6000 | Relative cost decrease | Minor FWHM at 3000 /6000 (arcsec) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['array']} | {r['trial']} | {r['original_amplitude']:.5f} | {r['probe_amplitude']:.5f} | {r['relative_cost_decrease']:.5g} | {r['original_minor_FWHM_arcsec']:.4f} / {r['probe_minor_FWHM_arcsec']:.4f} |")
    lines.extend(['', 'Amplitudes are mJy/nominal beam. Relative cost decrease is `(cost3000-cost6000)/cost3000`, not a fractional amplitude error or detection statistic.', '',
        'The nine failed fits have minor FWHM below the 2-arcsecond pixel spacing. Several approach the inherited sigma lower bound of 0.5 arcsec (FWHM 1.1774 arcsec). The two larger amplitude drifts accompany narrowing widths and little objective improvement. This is consistent with weakly determined subpixel noise peaks and slow progress along coupled amplitude/shape directions. It is a diagnosis of the selected cases, not proof of a unique cause for every archived failure.', '',
        'The paired core controls converge in 15/17/15 evaluations for a1100/a1400/a2000. The successful blank controls converge in 30–562 evaluations but can also fit narrow or bound-limited noise structures. In particular, one successful a1100 control sits extremely near the width bound without an active-bound flag. Neither numerical success nor that flag is a scientific source-presence test.', '',
        '## Why blank scenes enter the positive-source fitter', '',
        'The inherited initializer searches for the largest pixel within 20 arcsec, subtracts the median within 40 arcsec, then fits a positive elliptical Gaussian plus constant. Its pre-fit absence check is only whether that maximum exceeds the median. This is not a noise-aware association or significance test. No archived blank outcome satisfies that absence check; noise peaks enter the seven-parameter optimizer. The strict 1e-12 tolerances and broad shape freedom are inherited descriptive-fit choices. Their individual causal contributions have not been isolated, and this task does not select replacements.', '',
        '## Scope and preserved failures', '',
        '| Array | Calibration blank failures /1024 | Validation blank failures /1024 | Stress blank failures /256 |',
        '|---|---:|---:|---:|'])
    for a in ARRAYS:
        counts = [sum(v for k,v in census[a][phase]['blank']['causes'].items() if k!='success') for phase in ('calibration-scenes','validation-scenes','validation-stress')]
        lines.append(f'| {a} | {counts[0]} | {counts[1]} | {counts[2]} |')
    lines.extend(['',
        'The full archived census reports only the generic nonconvergence cause for failures; optimizer termination details were not originally saved. Evaluation-limit exhaustion is established for these nine replayed failures, not retrospectively for all failures. The sample is deterministic (first three failed calibration IDs per array), small and not randomly representative.', '',
        'No NaN/input repair, automatic source-absence relabeling, larger-budget adoption, threshold from surviving draws, source-model change or feedback rerun is justified by this diagnosis. All six launch gates remain unavailable. No source fitter, admission rule, map, original trial record or sealed parent was changed.', '',
        '**Recommended next action:** separately define and review the source-association/no-candidate behavior for these known-bright-source null controls, keeping optimizer failure distinct from absence. Preserve detailed optimizer diagnostics in any later implementation. A blanket budget increase is not supported by the three probes. No rule or repair is selected here.', '',
        f"Runtime: {read(out/'RUN.json')['wall_seconds']:.3f}s for 18 replays and 3 budget probes; zero new independent noise fields, zero upstream reductions. Ten focused correlation-admission tests pass in the existing local Python environment. This does not claim Unity/Spack validation.", '',
        f"![Fixed diagnostic traces]({dest/'blank-fit-traces.png'})", '',
        f"[Every recorded case and optimizer result]({out/'CASES.json'})", f"[Full archived outcome census]({out/'CENSUS.json'})", f"[Fixed selection, input hashes and environment]({out/'PLAN.json'})"])
    (dest/'ASSESSMENT.md').write_text('\n'.join(lines)+'\n')
    save_json(dest/'SUMMARY.json', dict(source_revision=revision, probes=summaries, sampled_failures=9,
        exact_outcome_replays=18, new_independent_fields=0, covariance_or_fit_setting_changes_adopted=False))
    manifest(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    report(args.parent, args.output) if args.report_only else run(args.parent, args.output)
