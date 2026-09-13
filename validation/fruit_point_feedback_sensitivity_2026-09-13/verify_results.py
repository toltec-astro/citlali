"""Verify saved products and learned identities without another clean/eigensolve."""
import numpy as np
from common import b,read,HERE,OUT,SAVED,SOLUTIONS,initialize,verify_freeze,models,state_rows,array_hash,forbidden
from prepare import preserve
from run_experiment import parent_for
from diagnostics import select_states,energy

def main():
    verify_freeze()
    assert preserve()==read(HERE/'PRESERVATION_START.json')
    manifest=read(OUT/'PRODUCT_MANIFEST.json')
    for row in manifest['files']:
        assert b.digest(OUT/row['path'])==row['sha256'],row['path']
    b.Data.clean=forbidden
    data,estimators=initialize()
    screen=read(OUT/'screen/PAIRS.json')
    selection,_=select_states(screen)
    saved_selection=read(OUT/'SELECTED_STATES.json')
    assert saved_selection['cleaning_calls_at_selection']==0
    for a,c in zip(selection,saved_selection['states']):
        assert all(a[k]==c[k] for k in a)
        for p in c['files']:
            assert b.digest(p['path'])==p['sha256']
    wave_checks=0
    for r in screen:
        if not r['available']:continue
        with np.load(OUT/'screen'/f"{r['case']}_pass{r['pass_index']:02d}_waveforms.npz") as z:
            for g in r['groups']:
                wave=z[g['group']]
                np.testing.assert_array_equal(wave[0],np.arange(g['lo'],g['hi']))
                np.testing.assert_allclose(np.sum(wave[1]*wave[2]**2),g['coherent_time_mean_energy'],rtol=1e-12,atol=1e-9)
                wave_checks+=1
    pairs={(r['case'],r['pass_index'],r['array']):r for r in read(SOLUTIONS/'stopping/RECEIPTS.json')}
    states={(r['case'],r['pass_index']):r for r in state_rows()}
    learned=[];model_checks=0;map_checks=0;comparisons=0
    for selected in selection:
        case,k=selected['case'],selected['pass_index']
        parent,truth,definition=parent_for(data,case)
        expected=read(SAVED/case/'C'/'START.json')['parent_sha256_float64_C']
        assert array_hash(parent)==expected
        n,t,present=models(data,states[case,k],pairs);assert all(present)
        dest=OUT/'replay'/f'{case}_after_pass{k:02d}'
        saved={};maps={}
        for label,model in [('nominal',n),('tight',t)]:
            with np.load(dest/label/'MAPS.npz') as z:
                np.testing.assert_array_equal(z['applied_model'],model);model_checks+=1
                maps[label]=z['total']
            assert np.isfinite(maps[label][data.S]).all()
            assert np.isnan(maps[label][~data.S]).all();map_checks+=1
            state=np.load(dest/label/'LEARNED_STATE.npz');saved[label]=state
            projected=data.project(model)
            for c,g,lo,hi,cc,mask,counts,den in data.groups:
                prefix=f'c{c:02d}_nw{g:02d}'
                np.testing.assert_array_equal(cc,state[prefix+'_columns'])
                raw=parent[lo:hi,cc]-projected[lo:hi,cc]
                mean=np.sum(np.where(mask!=0,raw,0),axis=0)/counts
                np.testing.assert_allclose(mean,state[prefix+'_location'],rtol=1e-12,atol=1e-10)
                centered=np.where(mask!=0,raw-mean,0)
                covariance=np.divide(b.mm(centered.T,centered),den,out=np.zeros_like(den),where=den>0)
                A=state[prefix+'_basis'];ev=state[prefix+'_eigenvalues']
                error=np.sqrt(energy(b.mm(covariance,A)-A*ev)/max(energy(covariance),1e-30))
                assert error<1e-10
                np.testing.assert_allclose(b.mm(A.T,A),np.eye(5),rtol=1e-10,atol=1e-10)
                learned.append(dict(case=case,branch=label,group=prefix,eigen_residual=error))
            del projected
            result=read(dest/label/'RESULT.json')
            if truth is not None:
                measured=[m['original'] for m in result['measurements']]
                from common import evaluation
                scores=evaluation.old_evaluation.score_truth(data,maps[label],measured,truth,definition)
                assert b.native(scores)==result['truth_scores']
        comparison=read(dest/'COMPARISON.json')
        for g in comparison['subspace_groups']:
            A=saved['nominal'][g['group']+'_basis'];B=saved['tight'][g['group']+'_basis']
            independent=np.sqrt(energy(A-b.mm(B,b.mm(B.T,A)))/5)
            np.testing.assert_allclose(independent,g['normalized_projector_distance'],rtol=1e-7,atol=1e-7)
            comparisons+=1
        with np.load(dest/'PAIRED_DIFFERENCES.npz') as z:
            np.testing.assert_array_equal(z['next_total'],maps['tight']-maps['nominal'])
            projected=data.project(t)-data.project(n)
            np.testing.assert_array_equal(z['projected_feedback'],projected)
            np.testing.assert_array_equal(z['restoration'],data.grid(projected))
            np.testing.assert_array_equal(z['residual_processing'],z['next_total']-z['restoration'])
        for a,r in enumerate(comparison['arrays']):
            nf,tf=r['nominal']['original']['fit'],r['tight']['original']['fit']
            assert r['peak_relative_change']==tf['peak']/nf['peak']-1
            assert r['centroid_change_arcsec']==float(np.linalg.norm(np.array(tf['centroid'])-nf['centroid']))
        assert array_hash(parent)==expected
        for z in saved.values():z.close()
        del parent
    assert len(learned)==864 and comparisons==432
    assert read(OUT/'CLEANING_CALLS.json')['started']==6
    b.write(HERE/'LEARNED_STATE_CHECKS.json',learned)
    result=dict(frozen_files=len(read(HERE/'FREEZE.json')['files']),external_payloads=len(manifest['files']),
        same_parent_pairs=3,saved_model_identity_checks=model_checks,total_map_support_checks=map_checks,
        group_waveform_checks=wave_checks,relearned_covariance_eigen_checks=len(learned),
        invariant_subspace_checks=comparisons,maximum_eigen_residual=max(r['eigen_residual'] for r in learned),
        truth_score_checks=12,cleaning_calls_in_experiment=6,verification_cleaning_calls=0,
        new_feedback_optimization_calls=0,prior_evidence_preserved=True,protected_archive_states_unchanged=True)
    b.write(HERE/'VERIFICATION.json',result)
    print(result)

if __name__=='__main__':main()
