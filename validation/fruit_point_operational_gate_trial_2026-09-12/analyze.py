"""Predeclared finite-case scores; missing terminal results never become passes."""
import numpy as np
from common import b, read, HERE, OUT, PREVIOUS

def main():
    rows=read(OUT/'RECEIPTS.json')
    lookup={(r['case'],r['arm'],r['pass_index']):r for r in rows}
    cases=['real123424']+[f'{s}_{seed}' for seed in [20260911,20260912]
                         for s in ['N','B','H','H-shift','D','C','T','E']]
    terminal={}
    trajectories=[]
    for case in cases:
        for arm in ['P','C']:
            d=OUT/case/arm
            complete=(d/'COMPLETE.json').exists()
            rr=[r for r in rows if r['case']==case and r['arm']==arm]
            if complete:terminal[case,arm]=lookup[case,arm,6]
            trajectories.append(dict(case=case,arm=arm,complete=complete,retained_passes=len(rr),
                failure=read(d/'FAILURE.json') if (d/'FAILURE.json').exists() else None,
                final_retained_wall_seconds=rr[-1]['cumulative_wall_seconds'] if rr else None,
                terminal_wall_seconds=rr[-1]['cumulative_wall_seconds'] if complete else None))
    scores=[]
    def add(use,case,arm,a,passed,**details):
        scores.append(dict(use=use,case=case,arm=arm,array=b.ARRAYS[a],pass_gate=bool(passed),**details))
    for arm in ['P','C']:
        for seed in [20260911,20260912]:
            for name in ['H','H-shift','D']:
                case=f'{name}_{seed}';r=terminal.get((case,arm))
                for a in range(3):
                    error=None if r is None else r['truth_score'][a].get('centroid_error_arcsec')
                    ok=r is not None and r['measurement'][a]['measurement_available']
                    add('U3',case,arm,a,ok and error is not None and error<=1,
                        measurement_available=ok,error_arcsec=error)
            for name,use in [('C','U1'),('E','G3')]:
                case=f'{name}_{seed}';r=terminal.get((case,arm))
                for a in range(3):
                    m=None if r is None else r['measurement'][a]
                    ok=(m is not None and m['source_present'] and
                        (m['distortion_warning'] or m['boundary_warning'])) if use=='U1' else (
                        m is not None and m['boundary_warning'] and not m['measurement_available'])
                    add(use,case,arm,a,ok,measurement=m)
            for numerator,denominator,expected,use in [
                    (f'H_{seed}',f'D_{seed}',[1/.9]*3,'U2_gain'),
                    (f'D_{seed}',f'H_{seed}',[.9]*3,'U4_loss'),
                    (f'T_{seed}',f'H_{seed}',[1,.8,1],'U5_health')]:
                n,d=terminal.get((numerator,arm)),terminal.get((denominator,arm))
                for a in range(3):
                    ok=n is not None and d is not None
                    ok=ok and n['measurement'][a]['measurement_available'] and d['measurement'][a]['measurement_available']
                    ratio=n['measurement'][a]['fit']['peak']/d['measurement'][a]['fit']['peak'] if ok else None
                    error=ratio/expected[a]-1 if ratio is not None else None
                    passed=ok and abs(error)<=.05
                    if use=='U4_loss':passed=passed and ratio<.95
                    if use=='U5_health' and a==1:
                        affected_unavailable=n is not None and not n['measurement'][a]['measurement_available']
                        passed=(passed and ratio<.9) or affected_unavailable
                    else:affected_unavailable=False
                    add(use,numerator+'/'+denominator,arm,a,passed,ratio=ratio,expected=expected[a],
                        relative_error=error,measurement_available=ok,affected_unavailable=affected_unavailable)
        n,d=terminal.get(('H_20260912',arm)),terminal.get(('H_20260911',arm))
        for a in range(3):
            ok=n is not None and d is not None
            ok=ok and n['measurement'][a]['measurement_available'] and d['measurement'][a]['measurement_available']
            ratio=n['measurement'][a]['fit']['peak']/d['measurement'][a]['fit']['peak'] if ok else None
            add('U2_U4_unchanged','H_seed2/seed1',arm,a,ok and abs(ratio-1)<=.05,ratio=ratio)
    nulls=[]
    for arm in ['P','C']:
        for seed in [20260911,20260912]:
            for s in ['N','B']:
                rr=[r for r in rows if r['case']==f'{s}_{seed}' and r['arm']==arm]
                nulls.append(dict(case=f'{s}_{seed}',arm=arm,
                    completed=(f'{s}_{seed}',arm) in terminal,
                    examined_array_passes=3*len(rr),
                    admitted_array_passes=sum(d['admitted'] for r in rr for d in r['decisions']),
                    positive_source_reports=sum(m['source_present'] for r in rr for m in r['measurement'])))
    leakage=[]
    cost=[]
    for case in cases:
        p,c=terminal.get((case,'P')),terminal.get((case,'C'))
        if p and c:
            for a in range(3):
                ratio=c['measurement'][a]['exterior_rms']/p['measurement'][a]['exterior_rms']
                leakage.append(dict(case=case,array=b.ARRAYS[a],ratio=ratio,pass_gate=ratio<=1.10))
            ratio=c['cumulative_wall_seconds']/p['cumulative_wall_seconds']
            cost.append(dict(case=case,wall_ratio_C_over_P=ratio,pass_gate=ratio<=2,
                             P_wall=p['cumulative_wall_seconds'],C_wall=c['cumulative_wall_seconds']))
    displacements=[]
    means=[]
    for arm in ['P','C']:
        for seed in [20260911,20260912]:
            h,s=terminal.get((f'H_{seed}',arm)),terminal.get((f'H-shift_{seed}',arm))
            for a in range(3):
                available=h is not None and s is not None
                available=available and h['measurement'][a]['measurement_available'] and s['measurement'][a]['measurement_available']
                delta=(np.array(s['measurement'][a]['fit']['centroid'])-h['measurement'][a]['fit']['centroid']) if available else None
                displacements.append(dict(arm=arm,seed=seed,array=b.ARRAYS[a],available=available,
                    measured=delta,error_arcsec=float(np.linalg.norm(delta-[3,2])) if available else None))
        for name in ['H','H-shift','D']:
            rr=[terminal.get((f'{name}_{seed}',arm)) for seed in [20260911,20260912]]
            for a in range(3):
                available=all(r is not None and r['measurement'][a]['measurement_available'] for r in rr)
                vectors=np.array([r['truth_score'][a]['centroid_error_vector_arcsec'] for r in rr]) if available else None
                means.append(dict(case=name,arm=arm,array=b.ARRAYS[a],available=available,
                    mean_error_vector=vectors.mean(axis=0) if available else None,
                    coordinate_sample_scatter=vectors.std(axis=0,ddof=1) if available else None,
                    meaning='Two reused finite realizations; not an ensemble-bias or uncertainty qualification'))
    summary=[]
    for arm in ['P','C']:
        for use in sorted({r['use'] for r in scores}):
            rr=[r for r in scores if r['arm']==arm and r['use']==use]
            summary.append(dict(arm=arm,use=use,passed=sum(r['pass_gate'] for r in rr),required=len(rr)))
    og=read(PREVIOUS/'DECISION_EVIDENCE.json').get('OG')
    result=dict(summary=summary,trajectories=trajectories,use_scores=scores,nulls=nulls,
                exterior_leakage=leakage,cost=cost,OG=og,displacements=displacements,two_seed_error_summaries=means,
                campaign=read(OUT/'COMPLETE.json') if (OUT/'COMPLETE.json').exists() else read(OUT/'FAILURE.json'),
                uncertainty='Finite development errors; two reused nuisance seeds do not qualify intervals, ensemble bias, or false-alarm rates.')
    b.write(HERE/'DECISION_EVIDENCE.json',result)
    print('Gate counts',summary)
    print('Completed trajectories',sum(t['complete'] for t in trajectories),'/',len(trajectories))
    print('Nulls',nulls)
    print('Cost',cost)

if __name__=='__main__':main()
