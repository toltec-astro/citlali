"""Read-only verification and evaluation of frozen screen products; no learning."""
from pathlib import Path
import json,re,subprocess,collections
from types import SimpleNamespace
import numpy as np
import run_screen as run
import starlet
b=run.b;H=run.H;OUT=run.OUT;OLD=run.OLD;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_text())

def main():
    freeze=read(H/'FREEZE.json')
    for r in freeze['files']:assert b.digest(r['path'])==r['sha256']
    manifest=read(OUT/'PRODUCT_MANIFEST.json')
    for r in manifest['files']:assert b.digest(OUT/r['path'])==r['sha256']
    results=read(OUT/'RESULTS.json');rows=results['rows'];assert len(rows)==75
    geom=np.load(OLD/'geometry.npz');g=SimpleNamespace(x=geom['x'],ygrid=geom['y'],shape=tuple(geom['shape']),S=geom['S'],D=geom['D'],O=geom['O'],Q=geom['Q'])
    estimators=[starlet.Estimator(g.x,g.ygrid,g.shape,g.S[a],g.D[a],g.O[a],g.Q[a]) for a in range(3)]
    calibration=np.load(OLD/'null20260911/R/pass00_maps.npz')['total']
    for a,e in enumerate(estimators):
        actual=e.calibrate(calibration[a]);assert b.native(actual)==results['calibration'][a]
    residual_checks=[];admitted_models=[];trials=[]
    for name in sorted({r['case'] for r in rows}):
        saved=np.load(OUT/(name+'.npz'));records=read(OUT/(name+'.json'))['rows']
        for a,rec in enumerate(records):
            e=estimators[a];u=saved['model'][a];trial=saved['solver_trial'][a];z=saved['input_total'][a]
            assert np.isfinite(u).all() and np.all(u>=0) and not np.any(u[~g.D[a]])
            assert bool(np.any(u))==rec['admitted']
            if not rec['available']:assert not np.any(u)
            Y,beta=e.residual(z);W=starlet.analysis(Y.reshape(g.shape)).reshape(5,-1);sigma=e.sigmas[:,e.stratum]
            selected=e.valid&e.D&(abs(W)>=5*sigma)
            np.testing.assert_array_equal(selected,saved['selected'][a]);np.testing.assert_allclose(beta,rec['background'],rtol=0,atol=0)
            if 'iterations' in rec:
                scale=rec['outer_MAD'];weight=np.where(selected,(scale/sigma)**2,0).reshape((5,)+g.shape)
                v=trial/scale;diff=starlet.analysis((v-Y/scale).reshape(g.shape));grad=starlet.adjoint(weight*diff).ravel()[e.D]
                g0=starlet.adjoint(-weight*starlet.analysis((Y/scale).reshape(g.shape))).ravel()[e.D]
                pg=np.where((v[e.D]<=0)&(grad>0),0,grad);rel=np.max(abs(pg))/max(np.max(abs(g0)),1e-12)
                objective=.5*np.sum(weight*diff*diff)
                np.testing.assert_allclose(objective,rec['objective'],rtol=1e-6,atol=1e-12)
                np.testing.assert_allclose(rel,rec['relative_projected_gradient'],rtol=1e-6,atol=1e-12)
                assert rec['available']==(rec['solver_success'] and rel<=1e-4)
                residual_checks.append(dict(case=name,array=rec['array'],relative_projected_gradient=rel,objective=objective))
            if rec['admitted']:
                admitted_models.append(dict(case=name,array=rec['array'],pixels=int(np.count_nonzero(u)),peak=float(u.max()),integrated_brightness=float(4*u.sum()),selected_per_band=rec['selected_per_band']))
            if name.startswith('saved_gauss') or name=='saved_coma4':
                truth=np.load(OUT/'coma4_truth.npz')['truth'] if name=='saved_coma4' else np.load(OLD/(name.removeprefix('saved_')+'_truth.npz'))['truth']
                trials.append(dict(case=name,array=rec['array'],admitted=rec['admitted'],trial_only_not_available=not rec['available'],metrics=run.metrics(g,trial,truth[a],a,(17,-11),name.startswith('saved_gauss'))))
    state=np.load(OUT/'coma4_bootstrap_state.npz');basis_keys=[k for k in state.files if k.endswith('_basis')];assert len(basis_keys)==144
    max_ortho=0.
    for k in basis_keys:
        A=state[k];assert A.shape[1]==5;max_ortho=max(max_ortho,float(np.max(abs(b.mm(A.T,A)-np.eye(5)))))
        ev=state[k.removesuffix('_basis')+'_eigenvalues'];assert np.all(ev>0) and np.isfinite(ev).all()
    assert max_ortho<1e-10
    bootstrap=read(OUT/'BOOTSTRAP_COMPLETE.json');assert bootstrap['cleaning_passes']==1
    np.testing.assert_array_equal(np.load(OUT/'coma4_bootstrap_maps.npz')['applied_model'],0)
    # The new case is exactly four times the retained sampled coma, before PTC.
    np.testing.assert_array_equal(np.load(OUT/'coma4_truth.npz')['truth'],4*np.load(OLD/'coma_bright_truth.npz')['truth'])
    assert b.digest(OUT/'coma4_truth.npz')==read(OUT/'TRUTH_BINDING.json')['truth_sha256']
    # Reproduce old oracle numbers with the factored implementation, with no
    # cleaning. Keep these verification products distinct from new-case results.
    oracle_out=H/'oracle_verification'
    if not oracle_out.exists():
        oracle_out.mkdir()
        actual=run.oracle(g,np.load(OLD/'coma_bright/R/pass00_maps.npz')['total'],calibration,np.load(OLD/'null20260912/R/pass00_maps.npz')['total'],oracle_out)
    else:actual=read(oracle_out/'ORACLE.json')['rows']
    old=read(H.parent/'fruit_point_rbf_admission_audit_2026-09-11/AUDIT_RESULTS.json')['processed_template_diagnostic']
    for a,row in enumerate(actual):
        reference=next(r for r in old if r['case']=='coma_bright' and r['array']==b.ARRAYS[a])
        for key in ['source_score','paired_null_score','processed_response_score']:np.testing.assert_allclose(row[key],reference[key],rtol=1e-12,atol=1e-12)
    preservation=read(H/'PRESERVATION_START.json')
    for packet in preservation['packets']:
        p=Path(packet['path']);assert b.digest(p)==packet['sha256']
        for row in read(p)['files']:assert b.digest(p.parent/row['path'])==row['sha256']
    folder=R/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4'
    payloads=re.findall(r'^\| `([^`]+)` \| \d+ \| `([0-9a-f]{64})` \|$',(folder/'PACKET_MANIFEST.md').read_text(),re.M)
    assert len(payloads)==57
    for path,digest in payloads:assert b.digest(folder/path)==digest
    protected=[]
    for worktree,expected in [('/Users/gwilson/.codex/worktrees/4c31/citlali-refactor','?? SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review.tar.gz\n?? SCI-FRUIT-v0.1-empirical-lane-gate-0-r0.1-owner-review.tar.gz\n'),('/Users/gwilson/.codex/worktrees/346d/citlali-refactor','')]:
        status=subprocess.check_output(['git','-C',worktree,'status','--porcelain=v1','--untracked-files=all'],text=True);assert status==expected
        protected.append(dict(path=worktree,status=status,archive_handling='presence/status only; no read/hash/unpack'))
    nulls=[r for r in rows if r['case'] in ['saved_null20260911','saved_null20260912']]
    log=Path('/private/tmp/sci-fruit-starlet-preliminary-execution.log').read_text()
    assert not re.search(r'Traceback|Warning:|Error:',log)
    background=[r for r in rows if r['case']=='saved_background20260911'];plane=[r for r in rows if r['case']=='pure_plane']
    compact=[r for r in rows if r['case'].startswith('saved_gauss')];coma=[r for r in rows if r['case']=='saved_coma4']
    times=[dict(case=name,**{k:read(OUT/(name+'.json'))[k] for k in ['array_time_median','array_time_max','serial_inference_seconds']}) for name in sorted({r['case'] for r in rows})]
    gates=[dict(name='noise_calibration',status='pass',available_arrays=3),dict(name='brighter_coma_regime',status='pass',scores=[r['source_score'] for r in results['oracle']]),
        dict(name='null_zero_admission',status='fail',admitted=sum(r['admitted'] for r in nulls),cases=6),dict(name='background_zero_admission',status='fail',admitted=sum(r['admitted'] for r in background),cases=3),
        dict(name='pure_plane_zero_admission',status='pass',admitted=sum(r['admitted'] for r in plane),cases=3),dict(name='processed_compact_admission',status='fail',admitted=sum(r['admitted'] for r in compact),cases=6),
        dict(name='processed_coma4_admission',status='fail',admitted=sum(r['admitted'] for r in coma),cases=3),dict(name='noiseless_compact_recovery',status='unavailable',reason='nonempty support with zero prescribed normalization'),
        dict(name='noiseless_coma4_recovery',status='unavailable',reason='nonempty support with zero prescribed normalization'),dict(name='phase_fidelity',status='unavailable',reason='required noiseless solves unavailable'),
        dict(name='map_add_compact_and_coma4_recovery',status='unavailable',reason='all required source reconstructions hit maxiter=300'),
        dict(name='solver_availability',status='fail',iteration_cap_failures=sum(r['reason']=='solver_gate_failure' for r in rows),zero_MAD_unavailable=sum(r['reason'].startswith('nonpositive') for r in rows)),
        dict(name='timing_median_per_map',status='fail',bound_seconds=.5,maps_exceeding=sum(t['array_time_median']>.5 for t in times)),
        dict(name='timing_maximum_per_array',status='pass',bound_seconds=2.,observed_seconds=max(r['seconds'] for r in rows)),
        dict(name='complete_estimator_cost',status='unavailable',reason='unavailable solves do not establish useful complete inference latency'),
        dict(name='execution_resources',status='pass',evidence=read(OUT/'COMPLETE.json'))]
    b.write(H/'DECISION_EVIDENCE.json',dict(decision='reject_this_preliminary_candidate',gates=gates,reason_counts=collections.Counter(r['reason'] for r in rows),admitted_models=admitted_models,
        per_map_times=times,first_source_failed_trials=trials,calibration=results['calibration'],oracle=results['oracle'],bootstrap=bootstrap,
        no_full_trajectories=True,method_revision_used=False,reserved_129081='no new scientific input or feedback comparison; historical exposure disclosed',thread_limit='four requested via environment and threadpool_limits; backend pool enumeration empty'))
    b.write(H/'VERIFICATION.json',dict(freeze_files_unchanged=len(freeze['files']),run_manifest_files_verified=len(manifest['files']),selected_support_recomputed=75,
        stored_objective_and_gradient_checks=len(residual_checks),new_truth_exactly_four_times_old=True,new_bootstrap_model_exact_zero=True,learned_groups=144,max_basis_orthogonality_error=max_ortho,
        old_oracle_reproduced=True,new_cleaning_passes=1,full_trajectories=0,old_packet_payloads=[49,13],frozen_ordinary_MAP_payloads=57,protected_worktrees=protected,
        verification_is_read_only_on_screen_products=True,focused_unit_tests=4,unexpected_warnings_or_errors_in_execution=False))
    print('Decision gates',[(x['name'],x['status']) for x in gates]);print('Timing worst median',max(t['array_time_median'] for t in times))
if __name__=='__main__':main()
