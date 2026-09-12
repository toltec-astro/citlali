"""Independent saved-product arithmetic and preservation checks; no solves/PTC."""
import numpy as np
from common import b, candidate, read, setup, verify_freeze, HERE, OUT, SAVED
from prepare import preservation
import stopping

def main():
    frozen = verify_freeze()
    preserved = preservation()
    assert preserved == read(HERE/'PRESERVATION_START.json')
    product_manifest = read(OUT/'PRODUCT_MANIFEST.json')
    for r in product_manifest['files']:
        assert b.digest(OUT/r['path']) == r['sha256'], r['path']
    data, estimators = setup()
    rows = read(OUT/'stopping/RECEIPTS.json')
    original_lookup = {(r['case'],r['pass_index']):r for r in read(SAVED/'RECEIPTS.json') if r['arm']=='C'}
    old_objectives = 0
    checkpoints = 0
    constraints = 0
    finite_differences = []
    for row in rows:
        a = b.ARRAYS.index(row['array'])
        e = estimators[a]
        with np.load(SAVED/row['case']/'C'/f"pass{row['pass_index']:02d}_maps.npz") as z:
            total, original_trial = z['total'][a], z['solver_trial'][a]
            selected = z['selected'][a]
        Y, beta = e.residual(total)
        sigma = e.sigmas[:,e.stratum]
        W = candidate.analysis(Y.reshape(e.shape)).reshape(5,-1)
        omega = e.valid & e.D & (abs(W) >= 5*sigma)
        np.testing.assert_array_equal(omega,selected)
        scale = candidate.mad(Y[e.O])
        scale = e.normalization_fallback if scale == 0 else scale
        assert row['scale'] == scale
        np.testing.assert_array_equal(beta,row['background'])
        target = candidate.analysis((Y/scale).reshape(e.shape))
        weight = np.where(omega,(scale/sigma)**2,0).reshape((5,)+e.shape)
        g0 = candidate.adjoint(-weight*target).ravel()[e.D]
        gscale = max(float(np.max(abs(g0))),1e-12)
        def check(model, receipt):
            nonlocal checkpoints
            diff = candidate.analysis((model/scale).reshape(e.shape))-target
            f = .5*float(np.sum(weight*diff**2))
            g = candidate.adjoint(weight*diff).ravel()[e.D]
            v = model[e.D]/scale
            # Independent active-set construction, same signed KKT convention.
            pg = g.copy()
            pg[(v <= 0) & (g > 0)] = 0
            relative = float(np.max(abs(pg))/gscale)
            np.testing.assert_allclose(f,receipt['objective'],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(relative,receipt['relative_projected_gradient'],rtol=1e-8,atol=1e-12)
            checkpoints += 1
        with np.load(OUT/'stopping'/(row['name']+'.npz')) as z:
            for label in z.files:
                u = z[label]
                assert np.isfinite(u).all() and np.all(u >= 0) and not np.any(u[~e.D])
                constraints += 1
                if not row['empty']:
                    rec = row['final'] if label == 'final' else row['snapshots'][label]
                    assert rec['finite'] and rec['feasible']
                    check(u,rec)
                    if label in ['declared','tight']:
                        assert rec['relative_projected_gradient'] <= (1e-4 if label == 'declared' else 1e-6)
                else:
                    assert not np.any(u)
            if row.get('comparison'):
                loose,tight = z['declared'],z['tight']
                comparison = row['comparison']
                peak = abs(float(loose.max()/tight.max())-1)
                centers = [np.array([np.sum(u*data.x),np.sum(u*data.ygrid)])/u.sum() for u in [loose,tight]]
                shift = float(np.linalg.norm(centers[0]-centers[1]))
                np.testing.assert_allclose(peak,comparison['sampled_peak_relative_difference'],rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(shift,comparison['brightness_centroid_difference_arcsec'],rtol=1e-10,atol=1e-12)
                expected = peak <= .005 and shift <= .1
                if comparison['fitted_readout_comparison_available']:
                    l,t = comparison['declared']['fit'],comparison['tight']['fit']
                    fp = abs(l['peak']/t['peak']-1)
                    fc = float(np.linalg.norm(np.array(l['centroid'])-t['centroid']))
                    assert fp == comparison['fitted_peak_relative_difference']
                    assert fc == comparison['fitted_centroid_difference_arcsec']
                    expected = expected and fp <= .005 and fc <= .1
                assert comparison['passed'] == expected
        if not row['empty']:
            check(original_trial,original_lookup[row['case'],row['pass_index']]['decisions'][a])
            old_objectives += 1
            declared = row['snapshots'].get('declared')
            available = bool(declared and declared['iterations'] <= 3000 and declared['function_evaluations'] <= 30000)
            assert available == row['operational_stop_available']
            assert row['qualified'] == bool(available and row['comparison'] and row['comparison']['passed'])
            if row['case']=='H_20260911':
                p = stopping.problem(e,total)
                v = np.maximum(original_trial[e.D]/scale,.1)
                direction = np.sin(np.arange(len(v))+1.)
                direction /= np.linalg.norm(direction)
                eps = 1e-3
                f,g = p['objective'](v)
                fd = (p['objective'](v+eps*direction)[0]-p['objective'](v-eps*direction)[0])/(2*eps)
                # Elementwise reduction avoids spurious Accelerate BLAS status
                # warnings; finite values are still required explicitly.
                analytic = float(np.sum(g*direction))
                assert np.isfinite(analytic) and np.isfinite(fd)
                error = abs(fd-analytic)/max(abs(fd),abs(analytic),1.)
                assert error < 1e-7
                finite_differences.append(dict(array=row['array'],relative_error=error))
    evaluator = read(HERE/'EVALUATOR_EVIDENCE.json')
    decision = read(HERE/'DECISION_EVIDENCE.json')
    assert decision['stopping_qualification_passed'] == all(r['qualified'] for r in rows)
    assert decision['operational_rerun_admitted'] == (evaluator['passed'] and all(r['qualified'] for r in rows))
    result = dict(frozen_files=len(frozen['files']), preserved=preserved,
        new_external_payloads=len(product_manifest['files']), selected_problem_checks=len(rows),
        original_objective_gradient_checks=old_objectives, checkpoint_objective_gradient_checks=checkpoints-old_objectives,
        finite_positive_supported_models=constraints, finite_difference_checks=finite_differences,
        original_measurement_records_preserved=evaluator['common_gate']['original_records_preserved'],
        matched_evaluator_bootstraps=sum(r['equal'] for r in evaluator['bootstrap_checks']),
        verification_cleaning_calls=0, verification_optimizer_runs=0)
    b.write(HERE/'VERIFICATION.json',result)
    print({k:v for k,v in result.items() if k!='preserved'})

if __name__=='__main__':
    main()
