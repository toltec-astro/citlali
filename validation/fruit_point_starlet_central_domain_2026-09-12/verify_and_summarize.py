"""Verify stored calculations and summarize the two declared questions."""
from pathlib import Path
import json,re,ast,collections,subprocess
import numpy as np
import candidate
import run_comparison as run
H=run.H;OUT=run.OUT;b=run.b;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_text())

def main():
    freeze=read(H/'FREEZE.json')
    for r in freeze['files']:assert b.digest(r['path'])==r['sha256']
    manifest=read(OUT/'PRODUCT_MANIFEST.json')
    for r in manifest['files']:assert b.digest(OUT/r['path'])==r['sha256']
    result=read(OUT/'RESULTS.json');rows=result['rows'];assert len(rows)==174
    g=run.load_geometry();null=np.load(run.OLD/'saved_null20260911.npz')['input_total'];estimators={arm:[run.make_estimator(g,a,arm,null)[0] for a in range(3)] for arm in ['F','C']}
    gradient_checks=0;trial_metrics=[];phase_checks=[]
    for name in sorted({r['case'] for r in rows}):
        tp=OUT/(name+'_truth.npz');truth=np.load(tp)['truth'] if tp.exists() else None
        for arm in ['F','C']:
            p=np.load(OUT/f'{name}_{arm}.npz');records=read(OUT/f'{name}_{arm}.json')['rows']
            for a,rec in enumerate(records):
                e=estimators[arm][a];u=p['model'][a];trial=p['solver_trial'][a];z=p['input_total'][a]
                assert np.isfinite(u).all() and np.all(u>=0) and not np.any(u[~e.D]);assert rec['admitted']==bool(np.any(u))
                if not rec['available']:assert not np.any(u)
                Y,beta=e.residual(z);W=candidate.analysis(Y.reshape(g.shape)).reshape(5,-1);sigma=e.sigmas[:,e.stratum];sel=e.valid&e.D&(abs(W)>=5*sigma)
                np.testing.assert_array_equal(sel,p['selected'][a]);np.testing.assert_array_equal(beta,rec['background'])
                if 'iterations' in rec:
                    s=rec['solver_scale'];weights=np.where(sel,(s/sigma)**2,0).reshape((5,)+g.shape);v=trial/s
                    diff=candidate.analysis((v-Y/s).reshape(g.shape));grad=candidate.adjoint(weights*diff).ravel()[e.D]
                    g0=candidate.adjoint(-weights*candidate.analysis((Y/s).reshape(g.shape))).ravel()[e.D]
                    pg=np.where((v[e.D]<=0)&(grad>0),0,grad);rel=np.max(abs(pg))/max(np.max(abs(g0)),1e-12)
                    objective=.5*np.sum(weights*diff*diff)
                    np.testing.assert_allclose(objective,rec['objective'],rtol=1e-6,atol=1e-12);np.testing.assert_allclose(rel,rec['relative_projected_gradient'],rtol=1e-6,atol=1e-12)
                    assert rec['available']==(rec['solver_success'] and rel<=1e-4);gradient_checks+=1
                if truth is not None and np.any(trial):
                    trial_metrics.append(dict(case=name,array=rec['array'],arm=arm,available=rec['available'],diagnostic_only=not rec['available'],
                        metrics=run.previous.metrics(g,trial,truth[a],a,rec['phase'],rec['compact'])))
    # Finite normalization is now supplied for every old zero-MAD nonempty case.
    assert not any(r['reason']=='normalization_unavailable' for r in rows)
    assert len(result['matched_controls'])==51 and all(r['bitwise_identical'] for r in result['matched_controls'])
    for arm in ['F','C']:
        for shape in ['compact','coma4p0']:
            for a in b.ARRAYS:
                pair=[next(r for r in rows if r['case']==f'noiseless_phase{k}_{shape}' and r['array']==a and r['arm']==arm) for k in [0,1]]
                available=all(r['available'] for r in pair)
                phase_checks.append(dict(arm=arm,array=a,shape=shape,available=available,
                    brightness_error_change=None if not available else abs(pair[1]['model_metrics']['brightness_error']-pair[0]['model_metrics']['brightness_error']),
                    image_error_change=None if not available else abs(pair[1]['model_metrics']['relative_L2']-pair[0]['model_metrics']['relative_L2'])))
    audit=read(OUT/'SPATIAL_AUDIT.json');spatial=[]
    for group in ['nulls','background','combined']:
        sub=[r for r in audit['rings'] if group=='combined' or (('null' in r['case'])==(group=='nulls'))]
        for region in ['central','outer']:
            rr=[r for r in sub if (r['outer_arcsec']<=60)==(region=='central')];n=sum(r['eligible'] for r in rr);k=sum(r['selected'] for r in rr)
            spatial.append(dict(group=group,region=region,eligible=n,selected=k,rate=k/n,per_10000=10000*k/n))
    loc=audit['selected_locations'];assert len(loc)==52 and sum(r['inside_central'] for r in loc)==0
    group_tests=[('nulls',lambda r:r['case'] in ['saved_null20260911','saved_null20260912'],6),('background',lambda r:r['case']=='saved_background20260911',3),('plane',lambda r:r['case']=='pure_plane',3),
        ('processed_compact',lambda r:r['case'].startswith('saved_gauss'),6),('processed_coma4',lambda r:r['case']=='saved_coma4',3)]
    gates=[]
    for arm in ['F','C']:
        rr=[r for r in rows if r['arm']==arm]
        for name,predicate,count in group_tests:
            selected=[r for r in rr if predicate(r)];assert len(selected)==count
            nulltest=name in ['nulls','background','plane'];ok=all(r['available'] and (not r['admitted'] if nulltest else r['admitted']) for r in selected)
            gates.append(dict(arm=arm,test=name,count=count,available=sum(r['available'] for r in selected),admitted=sum(r['admitted'] for r in selected),status='pass' if ok else 'fail'))
        for kind in ['noiseless','estimator_only_map_addition']:
            selected=[r for r in rr if r['kind']==kind and (r['compact'] or r['coma4'])]
            outcomes=[]
            for r in selected:
                if not r['available']:outcomes.append(dict(case=r['case'],array=r['array'],status='unavailable',reason=r['reason']));continue
                m=r['model_metrics'];pure=kind=='noiseless'
                if r['compact']:
                    ok=('amplitude_bias' in m and not m['fit_boundary'] and abs(m['amplitude_bias'])<=(.05 if pure else .15) and max(abs(np.array(m['width_bias'])))<=(.05 if pure else .15) and m['gaussian_centroid_error']<=(.4 if pure else 1) and (pure or m['relative_L2']<=.25))
                else:ok=abs(m['brightness_error'])<=(.1 if pure else .2) and m['relative_L2']<=(.15 if pure else .25) and m['centroid_error'] is not None and m['centroid_error']<=(.4 if pure else 1) and abs(m['wing_error'])<=(.2 if pure else .3)
                outcomes.append(dict(case=r['case'],array=r['array'],status='pass' if ok else 'fail'))
            gates.append(dict(arm=arm,test=kind+'_recovery',status='pass' if all(r['status']=='pass' for r in outcomes) else 'not_passed',outcomes=outcomes))
    timings=[]
    for name in sorted({r['case'] for r in rows}):
        for arm in ['F','C']:
            rec=read(OUT/f'{name}_{arm}.json');timings.append(dict(case=name,arm=arm,median=rec['median_array_seconds'],maximum=rec['max_array_seconds'],serial=rec['serial_seconds']))
    packets=[]
    for record in read(H/'PRESERVATION_START.json')['packets']:
        p=Path(record['path']);assert b.digest(p)==record['sha256']
        for row in read(p)['files']:assert b.digest(p.parent/row['path'])==row['sha256']
        packets.append(record['payloads'])
    folder=R/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4'
    payloads=re.findall(r'^\| `([^`]+)` \| \d+ \| `([0-9a-f]{64})` \|$',(folder/'PACKET_MANIFEST.md').read_text(),re.M);assert len(payloads)==57
    for name,digest in payloads:assert b.digest(folder/name)==digest
    protected=[]
    for worktree,expected in [('/Users/gwilson/.codex/worktrees/4c31/citlali-refactor','?? SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review.tar.gz\n?? SCI-FRUIT-v0.1-empirical-lane-gate-0-r0.1-owner-review.tar.gz\n'),('/Users/gwilson/.codex/worktrees/346d/citlali-refactor','')]:
        status=subprocess.check_output(['git','-C',worktree,'status','--porcelain=v1','--untracked-files=all'],text=True);assert status==expected;protected.append(dict(path=worktree,status=status,archives='presence/status only; no read/hash/unpack'))
    tree=ast.parse((H/'run_comparison.py').read_text())
    forbidden=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['Data','clean','nuisance','trajectory']];assert not forbidden
    oldtree=ast.parse((run.PREV/'starlet.py').read_text());newtree=ast.parse((H/'candidate.py').read_text())
    for name in ['smooth','analysis','adjoint','eligible','mad']:
        find=lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        assert find(oldtree)==find(newtree)
    log=Path('/private/tmp/sci-fruit-starlet-central-domain-execution.log').read_text();assert not re.search(r'Traceback|Warning:|Error:',log)
    evidence=dict(spatial_conclusion='edge hypothesis supported on retained null/background cases',candidate_conclusion='central null exclusion works; fixed 60-arcsec reconstruction candidate fails readiness and offset-coma support',
        spatial_rates=spatial,selected_radius_range=[min(r['radius'] for r in loc),max(r['radius'] for r in loc)],selected_low_Q_stratum=sum(r['stratum']==0 for r in loc),selected_total=len(loc),false_models=audit['false_models'],
        gates=gates,trial_metrics=trial_metrics,trial_warning='Unfinished solver iterates are diagnostic only, not available feedback or convergence evidence',phase_checks=phase_checks,truth_clipping=read(OUT/'TRUTH_CLIPPING.json'),
        reason_counts={arm:collections.Counter(r['reason'] for r in rows if r['arm']==arm) for arm in ['F','C']},timings=timings,resources=read(OUT/'COMPLETE.json'))
    b.write(H/'DECISION_EVIDENCE.json',evidence)
    b.write(H/'VERIFICATION.json',dict(frozen_source_input_files=len(freeze['files']),run_payloads=len(manifest['files']),recomputed_selected_supports=174,recomputed_objectives_gradients=gradient_checks,
        positive_MAD_control_matches=51,transform_and_masks_unchanged=True,zero_MAD_normalization_available=True,no_new_PTC_or_parent_calls=True,new_cleaning_passes=0,full_trajectories=0,
        preserved_previous_packet_payloads=packets,frozen_ordinary_MAP_payloads=57,protected_worktrees=protected,unexpected_execution_warnings_or_errors=False,unit_tests=3))
    print('Verified',gradient_checks,'solver calculations; gates',[(x['arm'],x['test'],x['status']) for x in gates])
if __name__=='__main__':main()
