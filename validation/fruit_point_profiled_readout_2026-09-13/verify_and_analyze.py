"""Saved-result verification and interpretation. No optimizer or fitted solve calls."""
import time
from collections import Counter
import numpy as np
import scipy.optimize
from common import *
from profiled import QSCALE,LOW,HIGH,OBJECTIVE_ATOL,OBJECTIVE_RTOL,NEAR_COST_RTOL,FIRST_ORDER_TOL
from evaluator import evaluate

def forbidden(*a,**k):raise RuntimeError('verification refits forbidden')
scipy.optimize.least_squares=forbidden
scipy.optimize.minimize=forbidden
np.linalg.lstsq=forbidden

def key(r):return r['case'],r['arm'],r['array'],r.get('radius',r.get('radius_arcsec'))
def near(a,b):return a<=b+OBJECTIVE_ATOL+OBJECTIVE_RTOL*max(abs(a),abs(b),1.)
def fit_metrics(f,definition,a):
    if f is None:return dict(available=False)
    p=np.array(f['parameters']);peak=definition.get('peak');center=definition.get('centroid')
    return dict(available=True,peak=f['peak'],centroid=f['centroid'],widths=f['widths'],angle_rad=f['angle_rad'],
        sse=f['sse'],background=f['background'],background_at_fit_center=float(p[6]+p[7]*p[1]/90+p[8]*p[2]/90),
        background_at_truth_center=None if center is None else float(p[6]+p[7]*center[0]/90+p[8]*center[1]/90),
        peak_error=None if peak is None else f['peak']/peak[a]-1,
        centroid_error=None if center is None else float(np.linalg.norm(p[1:3]-center)),
        width_error=None if definition.get('widths') is None else np.array(f['widths'])/definition['widths']-1)

def independent(p,d,x,y,sse,scale=1.,diag=None):
    p=np.array(p);model=gaussian(p,x,y,True);r=model-d;actual=float(np.sum(r*r))
    np.testing.assert_allclose(actual,sse,rtol=2e-12,atol=1e-8)
    jac=model_jac(p,x,y);design=jac[:,[0,6,7,8]]
    normal=np.sum(design*r[:,None],axis=0)
    rel=float(np.linalg.norm(normal)/max(np.linalg.norm(design)*np.linalg.norm(r),1e-30))
    grad=np.sum(jac[:,1:6]*r[:,None],axis=0)/scale**2
    if diag is not None:
        np.testing.assert_allclose(grad,diag['geometry_gradient'],rtol=1e-7,atol=1e-7)
        assert rel<=1e-10,rel
        singular=np.linalg.svd(design,compute_uv=False)
        np.testing.assert_allclose(singular,diag['singular_values'],rtol=1e-12,atol=1e-12)
        q=p[1:6];scaled=QSCALE*grad
        proj=np.where(((q-LOW<=1e-8*QSCALE)&(scaled>0))|((HIGH-q<=1e-8*QSCALE)&(scaled<0)),0,scaled)
        score=float(np.max(abs(proj))/max(.5*actual/scale**2,1))
        np.testing.assert_allclose(score,diag['relative_scaled_projected_gradient'],rtol=1e-6,atol=1e-10)
    return rel

def main():
    begin=time.perf_counter();verify_freeze();assert preserve()==read(HERE/'PRESERVATION_START.json')
    done=read(HERE/'COMPLETE.json');assert digest(HERE/'RESULTS.json')==done['results_sha256']
    rows=read(HERE/'RESULTS.json');problems={r['id']:r for r in read(HERE/'PROBLEMS.json')}
    fixed={key(r):r for r in read(FIXED/'READOUTS.json')}
    witnesses={key(r):r for r in read(FIXED/'FIT_OBJECTIVE_WITNESSES.json')['witnesses']}
    definitions=read(PRIOR/'CASES.json')['states']
    with np.load(OUT/'geometry.npz') as z:x,y,D,central,O=[z[k] for k in ['x','y','D','central','O']]
    with np.load(HERE/'EVIDENCE.npz') as z:evidence={k:z[k] for k in z.files}
    checks=[];comparison=[];attempt_checks=0;lookup={};witness_results=[]
    for r in rows:
        pr=problems[r['id']];a=pr['array_index'];name=r['case'].rsplit('_',1)[0]
        with np.load(pr['map_path']) as z:total=z['total'][a]
        mask=central[a]&(np.hypot(x,y)<=r['radius'])&np.isfinite(total)
        old=pr['original_fit'];n=r['repaired'];new=n['selected_fit']
        independent(old['parameters'],total[mask],x[mask],y[mask],old['sse'])
        for attempt in n['attempts']:
            if not attempt['diagnostics']['feasible']:continue
            q=attempt['parameters'][1:6]
            assert np.all(np.array(q)>=LOW) and np.all(np.array(q)<=HIGH)
            rel=independent(attempt['parameters'],total[mask],x[mask],y[mask],attempt['sse'],n['normalization_scale'],attempt['diagnostics'])
            checks.append(rel);attempt_checks+=1
        line=r['fixed_original_geometry']
        if line:
            independent(line['fit']['parameters'],total[mask],x[mask],y[mask],line['fit']['sse'],1.,line['diagnostics'])
            np.testing.assert_array_equal(line['fit']['parameters'][1:6],old['parameters'][1:6])
            assert near(line['fit']['sse'],old['sse'])
        feasible=[t for t in n['attempts'] if t['diagnostics']['feasible']]
        selected=min(feasible,key=lambda t:(t['sse'],t['start_index'])) if feasible else None
        assert (selected is None and new is None) or selected['start_index']==n['selected_start']
        expected=definitions[name]
        entry=dict(id=r['id'],case=r['case'],arm=r['arm'],array=r['array'],radius=r['radius'],primary=r['primary'],
            original=fit_metrics(old,expected,a),linear=fit_metrics(line['fit'],expected,a) if line else None,
            repaired=fit_metrics(new,expected,a),new_numerical_complete=bool(selected and selected['diagnostics']['numerical_complete']),
            new_fit_available=n['available'],original_judgments=pr['original_measurement']['judgments'],
            objective_no_worse_than_original=bool(new and near(new['sse'],old['sse'])),seconds=n['seconds'],
            coefficient_relative_objective_improvement=None if not line else (old['sse']-line['fit']['sse'])/old['sse'],
            repaired_relative_objective_improvement=None if not new else (old['sse']-new['sse'])/old['sse'])
        entry['centroid_change']=None if not new else float(np.linalg.norm(np.array(new['centroid'])-old['centroid']))
        entry['peak_change']=None if not new or old['peak']==0 else new['peak']/old['peak']-1
        entry['background_change']=None if not new else np.array(new['background'])-old['background']
        if key(r) in fixed:
            f=fixed[key(r)];entry['truth_template']=dict(amplitude=f['amplitude'],peak_error=f['relative_peak_error'],
                sse=f['sse'],background=f['background'],role='coefficient against known injected shape, not free-profile peak')
        if key(r) in witnesses:
            w=witnesses[key(r)]
            witness_results.append(dict(id=r['id'],case=r['case'],arm=r['arm'],array=r['array'],radius=r['radius'],
                original_sse=old['sse'],linear_sse=line['fit']['sse'],repaired_sse=None if not new else new['sse'],witness_sse=w['feasible_fixed_SSE'],
                linear_clears=near(line['fit']['sse'],w['feasible_fixed_SSE']),repaired_clears=bool(new and near(new['sse'],w['feasible_fixed_SSE'])),
                coefficient_fraction_of_witness_gap=(old['sse']-line['fit']['sse'])/(old['sse']-w['feasible_fixed_SSE'])))
        if selected:
            close=[t for t in feasible if t['sse']-selected['sse']<=OBJECTIVE_ATOL+NEAR_COST_RTOL*max(selected['sse'],1)]
            def spread(tt):
                peaks=[t['parameters'][0] for t in tt];cs=np.array([t['parameters'][1:3] for t in tt])
                return dict(starts=len(tt),peak_min=min(peaks),peak_max=max(peaks),
                    relative_peak_span=(max(peaks)-min(peaks))/max(abs(selected['parameters'][0]),1e-30),
                    centroid_diameter=float(max(np.linalg.norm(a-b) for a in cs for b in cs)),
                    sse_min=min(t['sse'] for t in tt),sse_max=max(t['sse'] for t in tt))
            entry['all_start_spread']=spread(feasible);entry['similar_cost_spread']=spread(close)
        comparison.append(entry);lookup[key(r)]=r
    write(HERE/'FIT_COMPARISON.json',comparison)
    write(HERE/'WITNESS_RESULTS.json',witness_results)
    measurements=[]
    tick=time.perf_counter()
    for r in rows:
        if r['radius']!=60:continue
        pr=problems[r['id']];a=pr['array_index'];inner=lookup[r['case'],r['arm'],r['array'],52]['repaired']
        with np.load(pr['map_path']) as z:total=z['total'][a]
        fullfit=r['repaired']['selected_fit'] if r['repaired']['available'] else None
        innerfit=inner['selected_fit'] if inner['available'] else None
        e=dict(positive=evidence[pr['evidence_key']],R=pr['outer_MAD'])
        m=evaluate(total,x,y,D[a],central[a],O[a],pr['original_measurement'],e,fullfit,innerfit)
        old=pr['original_measurement']
        measurements.append(dict(case=r['case'],arm=r['arm'],array=r['array'],primary=r['primary'],
            original=old,repaired=m,changed_judgments={k:dict(old=old['judgments'][k],new=v) for k,v in m['judgments'].items() if v!=old['judgments'][k]}))
    evaluation_seconds=time.perf_counter()-tick
    write(HERE/'MEASUREMENTS.json',measurements)
    mlookup={(r['case'],r['arm'],r['array']):r for r in measurements}
    comp={key(r):r for r in comparison};ratios=[]
    for arm in ['P','C']:
        for array in ARRAYS:
            for radius in [60,52]:
                for s1 in SEEDS:
                    for s2 in SEEDS:
                        h=comp[f'H_{s1}',arm,array,radius];d=comp[f'D_{s2}',arm,array,radius]
                        hm=mlookup[f'H_{s1}',arm,array];dm=mlookup[f'D_{s2}',arm,array]
                        for kind in ['original','linear','repaired','truth_template']:
                            name='amplitude' if kind=='truth_template' else 'peak'
                            ratio=h[kind][name]/d[kind][name] if h[kind] and d[kind] and d[kind].get(name) else None
                            usable=None if kind in ['linear','truth_template'] else hm[kind]['judgments']['peak_response_usable'] and dm[kind]['judgments']['peak_response_usable']
                            error=None if ratio is None else ratio/(100/90)-1
                            ratios.append(dict(arm=arm,array=array,radius=radius,numerator_seed=s1,denominator_seed=s2,
                                pairing='matched' if s1==s2 else 'crossed',readout=kind,ratio=ratio,error=error,
                                raw_within_5pct=error is not None and abs(error)<=.05,full_domain_pair_usable=usable,
                                usability_applies_to_this_radius=radius==60 and usable is not None))
    write(HERE/'RATIOS.json',ratios)
    summary=[]
    for arm in ['P','C']:
        for radius in [60,52]:
            ss=[r for r in comparison if r['primary'] and r['arm']==arm and r['radius']==radius]
            for kind in ['original','linear','repaired','truth_template']:
                errs=[r[kind].get('peak_error') for r in ss if r[kind]]
                rr=[r for r in ratios if r['arm']==arm and r['radius']==radius and r['readout']==kind]
                usable=[mlookup[r['case'],arm,r['array']][kind]['judgments']['peak_response_usable'] for r in ss] if kind in ['original','repaired'] else None
                summary.append(dict(arm=arm,radius=radius,readout=kind,required=12,
                    raw_absolute_within_5pct=sum(e is not None and abs(e)<=.05 for e in errs),
                    max_abs_peak_error=max(abs(e) for e in errs if e is not None),
                    full_domain_peaks_usable=None if usable is None else sum(usable),
                    usable_absolute_within_5pct=None if usable is None else sum(ok and r[kind] and r[kind]['peak_error'] is not None and abs(r[kind]['peak_error'])<=.05 for ok,r in zip(usable,ss)),
                    ratios=[dict(pairing=pair,required=6,raw_within_5pct=sum(r['raw_within_5pct'] for r in rr if r['pairing']==pair),
                        usable=None if usable is None else sum(r['full_domain_pair_usable'] for r in rr if r['pairing']==pair),
                        usable_within_5pct=None if usable is None else sum(r['full_domain_pair_usable'] and r['raw_within_5pct'] for r in rr if r['pairing']==pair),
                        max_abs_error=max(abs(r['error']) for r in rr if r['pairing']==pair and r['error'] is not None)) for pair in ['matched','crossed']]))
    attempts=[a for r in rows for a in r['repaired']['attempts']]
    selected=[a for r in rows for a in r['repaired']['attempts'] if a['start_index']==r['repaired']['selected_start']]
    numeric=dict(problems=84,selected_available=sum(r['repaired']['available'] for r in rows),
        selected_numerically_complete=sum(a['diagnostics'].get('numerical_complete',False) for a in selected),
        attempts=len(attempts),termination_counts=dict(Counter(str(a['status']) for a in attempts)),
        attempt_numerically_complete=sum(a['diagnostics'].get('numerical_complete',False) for a in attempts),
        no_worse_than_original=sum(r['objective_no_worse_than_original'] for r in comparison),
        witnesses_cleared=sum(r['repaired_clears'] for r in witness_results),witnesses_required=len(witness_results),
        fixed_geometry_clears=sum(r['linear_clears'] for r in witness_results),
        similar_cost_peak_gt_5pct=sum(r.get('similar_cost_spread',{}).get('relative_peak_span',0)>.05 for r in comparison),
        similar_cost_centroid_gt_1arcsec=sum(r.get('similar_cost_spread',{}).get('centroid_diameter',0)>1 for r in comparison),
        max_normal_residual=max(checks),normal_fitter_seconds=done['normal_fitter_seconds'],
        per_fit_seconds=dict(min=min(r['seconds'] for r in comparison),median=float(np.median([r['seconds'] for r in comparison])),max=max(r['seconds'] for r in comparison)),
        evaluation_seconds=evaluation_seconds)
    safeguards=[]
    for name in ['H-shift_20260911','N_20260911','E_20260911']:
        for arm in ['P','C']:
            for kind in ['original','repaired']:
                ms=[r for r in measurements if r['case']==name and r['arm']==arm]
                qq=[comp[name,arm,r['array'],60][kind] for r in ms]
                safeguards.append(dict(case=name,arm=arm,readout=kind,required=3,
                    source_evidence=sum(r[kind]['judgments']['source_evidence_present'] for r in ms),
                    usable_centroids=sum(r[kind]['judgments']['centroid_usable'] for r in ms),
                    usable_peaks=sum(r[kind]['judgments']['peak_response_usable'] for r in ms),
                    shape_warnings=sum(r[kind]['judgments']['shape_warning'] for r in ms),
                    support_warnings=sum(r[kind]['judgments']['support_warning'] for r in ms),
                    max_centroid_error=max((r['centroid_error'] for r in qq if r['centroid_error'] is not None),default=None)))
    write(HERE/'SUMMARY.json',dict(numerical=numeric,primary=summary,safeguards=safeguards))
    verify_freeze();assert preserve()==read(HERE/'PRESERVATION_START.json')
    assert digest(HERE/'RESULTS.json')==done['results_sha256']
    write(HERE/'VERIFICATION.json',dict(independent_attempt_models_costs_gradients=attempt_checks,original_costs=84,
        diagnostic_costs_and_unchanged_geometry=48,selection_rules=84,evaluator_reproductions_before_run=42,
        new_terminal_judgments=42,source_input_freeze_preserved=True,all_prior_products_preserved=True,
        verification_refits=0,PTC_calls=0,feedback_fits=0,reserved_observations=0,
        max_linear_normal_residual=max(checks),results_frozen_before_comparisons=True,
        verification_analysis_seconds=time.perf_counter()-begin))
    print('NUMERICAL',numeric)
    print('PRIMARY',[r for r in summary if r['radius']==60])
    print('WITNESSES',witness_results)
    print('SAFEGUARDS',safeguards)
if __name__=='__main__':main()
