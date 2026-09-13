"""Verify fixed problems, saved recurrence and relearning without a new PTC call."""
import shutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from common import b,read,HERE,OUT,SAVED,REPAIR,PREVIOUS,OLD,verify_freeze
from prepare import preserve
from run_trial import make_estimators,truths,array_hash
import nominal
import evaluation

def forbidden(*args,**kwargs):
    raise RuntimeError('verification cannot clean or optimize')

def main():
    verify_freeze()
    assert preserve()==read(HERE/'PRESERVATION_START.json')
    manifest=read(OUT/'PRODUCT_MANIFEST.json')
    for row in manifest['files']:
        assert b.digest(OUT/row['path'])==row['sha256'],row['path']
    rows=read(OUT/'RECEIPTS.json')
    b.Data.clean=forbidden
    nominal.stopping.solve_path=forbidden
    with threadpool_limits(limits=4):
        data=b.Data(read(PREVIOUS/'INPUT_PREFLIGHT.json'))
        estimators,_=make_estimators(data)
        scene=truths(data);definitions=read(HERE/'CASES.json')['states']
        for name,truth in scene.items():
            np.testing.assert_array_equal(truth,np.load(SAVED/(name+'_truth.npz'))['truth'])
            np.testing.assert_array_equal(truth,np.load(OUT/(name+'_truth.npz'))['truth'])
        parent_checks=0;recur=0;control_arrays=0;control_states=0;bootstrap=0;problem_checks=0;truth_checks=0
        for case in ['real123424']+[f'{n}_{seed}' for seed in [20260911,20260912] for n in scene]:
            starts=[read(OUT/case/arm/'START.json') for arm in ['P','C']]
            assert starts[0]['parent_sha256_float64_C']==starts[1]['parent_sha256_float64_C']
            assert starts[0]['parent_sha256_float64_C']==read(SAVED/case/'P'/'START.json')['parent_sha256_float64_C'];parent_checks+=1
            for arm in ['P','C']:
                previous=np.zeros((3,data.npix))
                rr=[r for r in rows if r['case']==case and r['arm']==arm]
                for k,r in enumerate(rr):
                    assert r['pass_index']==k
                    with np.load(OUT/case/arm/f'pass{k:02d}_maps.npz') as z:
                        total,model=z['total'],z['next_model']
                        np.testing.assert_array_equal(z['applied_model'],previous);recur+=1
                        assert np.isfinite(total[data.S]).all() and np.isnan(total[~data.S]).all()
                        if arm=='P':
                            with np.load(SAVED/case/arm/f'pass{k:02d}_maps.npz') as old:
                                for key in old.files:
                                    np.testing.assert_array_equal(z[key],old[key]);control_arrays+=1
                            with np.load(OUT/case/arm/f'pass{k:02d}_state.npz') as ss,np.load(SAVED/case/arm/f'pass{k:02d}_state.npz') as old:
                                for key in old.files:
                                    np.testing.assert_array_equal(ss[key],old[key]);control_states+=1
                        else:
                            for a,e in enumerate(estimators):
                                p=nominal.stopping.problem(e,total[a])
                                np.testing.assert_array_equal(p['omega'],z['selected'][a])
                                np.testing.assert_array_equal(p['background'],r['decisions'][a]['background'])
                                assert p['scale']==r['decisions'][a]['solver_scale']
                                assert np.isfinite(model[a]).all() and np.all(model[a]>=0) and not np.any(model[a,~e.D])
                                d=r['decisions'][a]
                                if not p['omega'].any():
                                    assert d['available'] and not model[a].any()
                                elif d['available']:
                                    f,g=p['objective'](model[a,e.D]/p['scale'])
                                    c=nominal.stopping.completion(model[a,e.D]/p['scale'],f,g,p['gradient_scale'])
                                    assert c['finite'] and c['feasible'] and c['relative_projected_gradient']<=1.000001e-4
                                    assert d['iterations']<=3000 and d['function_evaluations']<=30000
                                    assert d['solver_options']==dict(maxiter=3000,maxfun=30000,ftol=0.,gtol=0.,maxcor=10,maxls=20)
                                    np.testing.assert_allclose(c['objective'],d['objective'],rtol=1e-10,atol=1e-9)
                                else: assert not model[a].any()
                                problem_checks+=1
                                if k==0:
                                    saved=Path('/private/tmp/sci-fruit-point-bounded-repair-20260912-r0.1/stopping')/f'{case}_pass00_{b.ARRAYS[a]}.npz'
                                    with np.load(saved) as old:
                                        np.testing.assert_array_equal(model[a],old['declared']);bootstrap+=1
                        if case!='real123424':
                            name=case.rsplit('_',1)[0]
                            scores=evaluation.score_truth(data,total,r['measurement'],scene[name],definitions[name])
                            assert b.native(scores)==r['truth_score'];truth_checks+=3
                        if r['next_model_available']:previous=model.copy()
                        else:assert k==len(rr)-1
        # Three registered source/real terminal states: verify means and eigen identities
        # from each actual residual, not another optimization, cleaning or eigensolve.
        learned=[]
        seed_noise=None
        for case in ['real123424','H_20260911','T_20260911']:
            if case=='real123424':parent=data.y
            else:
                if seed_noise is None:seed_noise=b.nuisance(data,np.load(OLD/'geometry.npz')['scales'],20260911)
                name=case.rsplit('_',1)[0]
                parent=seed_noise+data.project(scene[name])
                if name=='T':parent[:,data.ar==1] += -.2*seed_noise[:,data.ar==1]
            assert array_hash(parent)==read(OUT/case/'C'/'START.json')['parent_sha256_float64_C']
            for arm in ['P','C']:
                if not (OUT/case/arm/'COMPLETE.json').exists():continue
                with np.load(OUT/case/arm/'pass06_maps.npz') as z:projected=data.project(z['applied_model'])
                with np.load(OUT/case/arm/'pass06_state.npz') as state:
                    for c,g,lo,hi,cc,mask,counts,den in data.groups:
                        prefix=f'c{c:02d}_nw{g:02d}'
                        np.testing.assert_array_equal(cc,state[prefix+'_columns'])
                        raw=parent[lo:hi,cc]-projected[lo:hi,cc]
                        mean=np.sum(np.where(mask!=0,raw,0),axis=0)/counts
                        np.testing.assert_allclose(mean,state[prefix+'_location'],rtol=1e-12,atol=1e-10)
                        centered=np.where(mask!=0,raw-mean,0)
                        cov=np.divide(b.mm(centered.T,centered),den,out=np.zeros_like(den),where=den>0)
                        A,ev=state[prefix+'_basis'],state[prefix+'_eigenvalues']
                        error=float(np.sqrt(np.sum((b.mm(cov,A)-A*ev)**2)/max(np.sum(cov**2),1e-30)))
                        assert error<1e-10
                        np.testing.assert_allclose(b.mm(A.T,A),np.eye(5),rtol=1e-10,atol=1e-10)
                        learned.append(dict(case=case,arm=arm,group=prefix,eigen_residual=error))
                del projected
        assert read(OUT/'CLEANING_CALLS.json')['started']==len(rows)<=238
        assert (HERE/'OWNER_DIRECTION.txt').read_bytes()==Path('/Users/gwilson/.codex/attachments/e5b83d03-d81b-440b-817a-bf8bd096173b/pasted-text.txt').read_bytes()
        b.write(HERE/'LEARNED_STATE_CHECKS.json',learned)
        result=dict(frozen_files=len(read(HERE/'FREEZE.json')['files']),external_payloads=len(manifest['files']),
            matched_parent_cases=parent_checks,recurrence_checks=recur,repeated_P_map_array_checks=control_arrays,
            repeated_P_learned_array_checks=control_states,saved_nominal_bootstrap_model_identities=bootstrap,
            fixed_candidate_problem_checks=problem_checks,external_truth_checks=truth_checks,
            relearned_covariance_eigen_checks=len(learned),maximum_eigen_residual=max(r['eigen_residual'] for r in learned),
            cleaning_calls=len(rows),verification_cleaning_calls=0,verification_optimization_calls=0,
            prior_evidence_preserved=True,protected_archives_preserved=True,unexpected_runtime_errors=False)
        b.write(HERE/'VERIFICATION.json',result)
        for name in ['START.json','COMPLETE.json','PRODUCT_MANIFEST.json','THREAD_POOLS.json','CLEANING_CALLS.json']:
            if (OUT/name).exists():shutil.copyfile(OUT/name,HERE/('RUN_'+name))
        print(result)
if __name__=='__main__':main()
