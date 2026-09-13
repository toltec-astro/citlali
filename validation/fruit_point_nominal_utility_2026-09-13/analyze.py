"""Apply the frozen POINT questions, retaining unavailable/raw values separately."""
import numpy as np
from common import b, read, HERE, OUT, PREVIOUS

SEEDS=[20260911,20260912]
NAMES=['N','B','H','H-shift','D','C','T','E']
CASES=['real123424']+[f'{name}_{seed}' for seed in SEEDS for name in NAMES]

def analyze():
    rows=read(OUT/'RECEIPTS.json')
    terminal={}; trajectories=[]
    for case in CASES:
        for arm in ['P','C']:
            d=OUT/case/arm
            rr=[r for r in rows if r['case']==case and r['arm']==arm]
            complete=(d/'COMPLETE.json').exists()
            if complete:
                assert len(rr)==7 and rr[-1]['pass_index']==6 and all(r['next_model_available'] for r in rr)
                terminal[case,arm]=rr[-1]
            receipt=read(d/'COMPLETE.json') if complete else read(d/'FAILURE.json') if (d/'FAILURE.json').exists() else None
            trajectories.append(dict(case=case,arm=arm,complete=complete,retained_passes=len(rr),receipt=receipt,
                wall_seconds=receipt['wall_seconds'] if receipt else None,
                components_seconds={k:sum(r[k] for r in rr) for k in ['clean_seconds','map_seconds','inference_seconds','evaluation_seconds','output_seconds']}))
    def m(case,arm,a):
        r=terminal.get((case,arm))
        return r['measurement'][a] if r else None
    def usable(measure,kind):
        return measure is not None and measure['judgments'][kind]
    def peak(measure):
        return measure['original'].get('fit',{}).get('peak') if measure else None
    scores=[]
    def add(use,case,arm,a,passed,**details):
        scores.append(dict(use=use,case=case,arm=arm,array=b.ARRAYS[a],pass_gate=bool(passed),**details))
    for arm in ['P','C']:
        for seed in SEEDS:
            for name in ['H','H-shift','D']:
                case=f'{name}_{seed}';r=terminal.get((case,arm))
                for a in range(3):
                    error=r['truth_score'][a].get('centroid_error_arcsec') if r else None
                    ok=usable(m(case,arm,a),'centroid_usable')
                    add('U3_pointing',case,arm,a,ok and error is not None and error<=1,available=ok,error_arcsec=error)
            for name,use in [('C','U1_startup'),('E','G3_boundary')]:
                case=f'{name}_{seed}'
                for a in range(3):
                    v=m(case,arm,a);j=v['judgments'] if v else {}
                    ok=(j.get('source_evidence_present') and (j.get('shape_warning') or j.get('support_warning'))) if name=='C' else (j.get('support_warning') and not j.get('centroid_usable'))
                    add(use,case,arm,a,ok,judgments=j)
            for n,d,expected,use in [(f'H_{seed}',f'D_{seed}',[1/.9]*3,'U2_gain'),(f'D_{seed}',f'H_{seed}',[.9]*3,'U4_degradation'),(f'T_{seed}',f'H_{seed}',[1,.8,1],'U5_health')]:
                for a in range(3):
                    nm,dm=m(n,arm,a),m(d,arm,a)
                    ok=usable(nm,'peak_response_usable') and usable(dm,'peak_response_usable')
                    npk,dpk=peak(nm),peak(dm)
                    ratio=npk/dpk if npk is not None and dpk not in [None,0] else None
                    error=ratio/expected[a]-1 if ratio is not None else None
                    accurate=error is not None and abs(error)<=.05
                    if use=='U4_degradation':accurate=accurate and ratio<.95
                    if use=='U5_health' and a==1:accurate=accurate and ratio<.9
                    add(use,n+'/'+d,arm,a,ok and accurate,available=ok,raw_ratio=ratio,expected=expected[a],raw_relative_error=error,
                        numerator_usable=usable(nm,'peak_response_usable'),denominator_usable=usable(dm,'peak_response_usable'),
                        raw_accuracy_pass=bool(accurate),affected_withheld_warning=bool(use=='U5_health' and a==1 and nm and not usable(nm,'peak_response_usable')))
        for a in range(3):
            nm,dm=m('H_20260912',arm,a),m('H_20260911',arm,a)
            ok=usable(nm,'peak_response_usable') and usable(dm,'peak_response_usable')
            npk,dpk=peak(nm),peak(dm)
            ratio=npk/dpk if npk is not None and dpk not in [None,0] else None
            add('U2_U4_unchanged','H_seed2/seed1',arm,a,ok and ratio is not None and .95<=ratio<=1.05,available=ok,raw_ratio=ratio,expected=1.,raw_relative_error=ratio-1 if ratio is not None else None)
    nulls=[];availability=[];readouts=[]
    for arm in ['P','C']:
        for seed in SEEDS:
            for name in ['N','B']:
                rr=[r for r in rows if r['case']==f'{name}_{seed}' and r['arm']==arm]
                nulls.append(dict(case=f'{name}_{seed}',arm=arm,complete=(f'{name}_{seed}',arm) in terminal,required_array_passes=21,
                    retained_array_passes=len(rr)*3,feedback_admissions=sum(d['admitted'] for r in rr for d in r['decisions']),
                    source_reports=sum(v['judgments']['source_evidence_present'] for r in rr for v in r['measurement']),
                    usable_centroids=sum(v['judgments']['centroid_usable'] for r in rr for v in r['measurement']),
                    usable_peaks=sum(v['judgments']['peak_response_usable'] for r in rr for v in r['measurement'])))
        for label,names in [('required_pointing',['H','H-shift','D']),('peak_pair_members',['H','D','T']),('all_compact_source_cases',['H','H-shift','D','T'])]:
            mm=[m(f'{name}_{seed}',arm,a) for name in names for seed in SEEDS for a in range(3)]
            availability.append(dict(arm=arm,population=label,required=len(mm),completed_measurements=sum(v is not None for v in mm),
                centroids=sum(usable(v,'centroid_usable') for v in mm),peaks=sum(usable(v,'peak_response_usable') for v in mm),
                shape_warnings=sum(bool(v and v['judgments']['shape_warning']) for v in mm),support_warnings=sum(bool(v and v['judgments']['support_warning']) for v in mm)))
        for case in CASES:
            r=terminal.get((case,arm))
            for a in range(3):
                v=m(case,arm,a)
                readouts.append(dict(case=case,arm=arm,array=b.ARRAYS[a],complete=r is not None,measurement=v,
                                     truth_score=r['truth_score'][a] if r and r['truth_score'] else None))
    costs=[];leakage=[]
    lookup={(t['case'],t['arm']):t for t in trajectories}
    for case in CASES:
        p,c=lookup[case,'P'],lookup[case,'C']
        if p['complete'] and c['complete']:
            ratio=c['wall_seconds']/p['wall_seconds']
            costs.append(dict(case=case,P_wall_seconds=p['wall_seconds'],C_wall_seconds=c['wall_seconds'],wall_ratio_C_over_P=ratio,pass_gate=ratio<=2,
                P_components=p['components_seconds'],C_components=c['components_seconds']))
            for a in range(3):
                ratio=m(case,'C',a)['original']['exterior_rms']/m(case,'P',a)['original']['exterior_rms']
                leakage.append(dict(case=case,array=b.ARRAYS[a],exterior_RMS_ratio=ratio,historical_1p10_comparison=ratio<=1.1,role='diagnostic_only_in_this_protocol'))
    displacement=[];means=[]
    for arm in ['P','C']:
        for seed in SEEDS:
            for a in range(3):
                h,s=m(f'H_{seed}',arm,a),m(f'H-shift_{seed}',arm,a)
                ok=usable(h,'centroid_usable') and usable(s,'centroid_usable')
                delta=np.array(s['original']['fit']['centroid'])-h['original']['fit']['centroid'] if s and h and 'fit' in s['original'] and 'fit' in h['original'] else None
                displacement.append(dict(arm=arm,seed=seed,array=b.ARRAYS[a],available=ok,raw_displacement=delta,raw_error_arcsec=float(np.linalg.norm(delta-[3,2])) if delta is not None else None))
        for name in ['H','H-shift','D']:
            for a in range(3):
                rr=[terminal.get((f'{name}_{seed}',arm)) for seed in SEEDS]
                ok=all(r and r['measurement'][a]['judgments']['centroid_usable'] for r in rr)
                vectors=np.array([r['truth_score'][a]['centroid_error_vector_arcsec'] for r in rr]) if all(rr) else None
                means.append(dict(arm=arm,case=name,array=b.ARRAYS[a],available=bool(ok),raw_mean_error_vector=vectors.mean(0) if vectors is not None else None,meaning='Two reused finite realizations; no qualified ensemble bias or uncertainty'))
    summary=[]
    for arm in ['P','C']:
        for use in sorted({s['use'] for s in scores}):
            ss=[s for s in scores if s['use']==use and s['arm']==arm]
            summary.append(dict(arm=arm,use=use,passed=sum(s['pass_gate'] for s in ss),required=len(ss),available=sum(s.get('available',False) for s in ss)))
    decisions=[dict(case=r['case'],pass_index=r['pass_index'],array=b.ARRAYS[a],**d) for r in rows if r['arm']=='C' for a,d in enumerate(r['decisions'])]
    prior=read(PREVIOUS/'DECISION_EVIDENCE.json')
    result=dict(summary=summary,trajectories=trajectories,use_scores=scores,nulls=nulls,availability=availability,terminal_readouts=readouts,
        costs=costs,exterior_leakage=leakage,displacement=displacement,two_seed_means=means,
        candidate_solves=dict(calls=len(decisions),nonempty=sum(sum(d.get('selected_per_band',[]))>0 for d in decisions),
            admitted=sum(d['admitted'] for d in decisions),unavailable=[d for d in decisions if not d['available']],
            max_iterations=max(d.get('iterations',0) for d in decisions),max_relative_PG=max(d.get('relative_projected_gradient',0) for d in decisions)),
        OG=dict(binding=read(PREVIOUS/'OG_BENCHMARK.json'),pointing=prior.get('OG_native_pointing_sequence'),runtime=prior.get('OG_runtime')),
        campaign=read(OUT/'COMPLETE.json') if (OUT/'COMPLETE.json').exists() else read(OUT/'FAILURE.json'),
        limitations='Finite development cases with reused noise/calibration; no numerical image-stability qualification, intervals, false-alarm rates, held-out replication or production claim.')
    b.write(HERE/'DECISION_EVIDENCE.json',result)
    print('Summary:',summary)
    print('Availability:',availability)
    print('Costs:',[(r['case'],round(r['wall_ratio_C_over_P'],3)) for r in costs])
    print('Solves:',result['candidate_solves'])
    return result
if __name__=='__main__': analyze()
