"""Compare the two readouts without fitting or assigning new usability labels."""
import numpy as np
from common import HERE,ARRAYS,SEEDS,read,write


def main():
    rows=read(HERE/'READOUTS.json')
    lookup={(r['case'],r['arm'],r['array'],r['radius_arcsec']):r for r in rows}
    ratios=[];seed_changes=[];domains=[];summary=[]
    def value(r,kind):return r['free_fit']['peak'] if kind=='free' else r.get('amplitude')
    for arm in ['P','C']:
        for radius in [60,52]:
            rr=[r for r in rows if r['arm']==arm and r['radius_arcsec']==radius]
            for kind in ['free','fixed']:
                errors=[value(r,kind)/r['expected_peak']-1 for r in rr if value(r,kind) is not None]
                summary.append(dict(arm=arm,radius_arcsec=radius,readout=kind,quantity='absolute_peak',required=12,
                    numerical_results=len(errors),within_5pct=sum(abs(x)<=.05 for x in errors),maximum_abs_error=max(abs(x) for x in errors),
                    min_error=min(errors),max_error=max(errors),median_abs_error=float(np.median(abs(np.array(errors)))),
                    original_usable_count=sum(r['original_peak_usable'] for r in rr),
                    original_usable_and_within_5pct=sum(r['original_peak_usable'] and value(r,kind) is not None and abs(value(r,kind)/r['expected_peak']-1)<=.05 for r in rr),
                    meaning='Original labels carried for context only; neither the fixed readout nor the inner free fit has a new usability policy.'))
                for array in ARRAYS:
                    for i in SEEDS:
                        for j in SEEDS:
                            h=lookup[f'H_{i}',arm,array,radius];d=lookup[f'D_{j}',arm,array,radius]
                            a,b=value(h,kind),value(d,kind)
                            ratio=a/b if a is not None and b not in [None,0] else None
                            error=ratio*.9-1 if ratio is not None else None
                            ratios.append(dict(arm=arm,radius_arcsec=radius,array=array,readout=kind,numerator_seed=i,denominator_seed=j,
                                pairing='matched' if i==j else 'crossed',ratio=ratio,relative_error=error,
                                original_pair_usable=h['original_peak_usable'] and d['original_peak_usable']))
                    for state in ['H','D']:
                        a,b=[lookup[f'{state}_{seed}',arm,array,radius] for seed in SEEDS]
                        av,bv=value(a,kind),value(b,kind)
                        seed_changes.append(dict(arm=arm,radius_arcsec=radius,array=array,readout=kind,state=state,
                            relative_change=bv/av-1 if av not in [None,0] and bv is not None else None,
                            original_pair_usable=a['original_peak_usable'] and b['original_peak_usable']))
                for pairing in ['matched','crossed']:
                    rr2=[r for r in ratios if r['arm']==arm and r['radius_arcsec']==radius and r['readout']==kind and r['pairing']==pairing]
                    errs=[r['relative_error'] for r in rr2 if r['relative_error'] is not None]
                    summary.append(dict(arm=arm,radius_arcsec=radius,readout=kind,quantity=pairing+'_H_over_D',required=6,
                        numerical_results=len(errs),within_5pct=sum(abs(e)<=.05 for e in errs),maximum_abs_error=max(abs(e) for e in errs),
                        min_error=min(errs),max_error=max(errs),median_abs_error=float(np.median(abs(np.array(errs)))),
                        original_usable_count=sum(r['original_pair_usable'] for r in rr2)))
        for array in ARRAYS:
            for state in ['H','D']:
                for seed in SEEDS:
                    a,b=[lookup[f'{state}_{seed}',arm,array,radius] for radius in [60,52]]
                    domains.append(dict(arm=arm,array=array,state=state,seed=seed,original_peak_usable=a['original_peak_usable'],
                        free_peak_relative_change=value(b,'free')/value(a,'free')-1,
                        fixed_peak_relative_change=value(b,'fixed')/value(a,'fixed')-1,
                        fixed_background_difference=np.array(b['background'])-a['background'],
                        free_background_difference=np.array(b['free_fit']['background'])-a['free_fit']['background'],
                        fixed_background_at_truth_change=b['fitted_background_at_truth_center']-a['fitted_background_at_truth_center'],
                        free_background_at_truth_change=b['free_background_at_truth_center']-a['free_background_at_truth_center']))
    result=dict(summary=summary,ratios=ratios,seed_changes=seed_changes,domain_changes=domains,
        numerical_rank_range=[min(r['rank'] for r in rows),max(r['rank'] for r in rows)],
        design_condition_range=[min(r['condition_number'] for r in rows),max(r['condition_number'] for r in rows)],
        maximum_fixed_domain_change=max(abs(r['fixed_peak_relative_change']) for r in domains),
        maximum_free_domain_change=max(abs(r['free_peak_relative_change']) for r in domains),
        free_vs_fixed_objective=[dict(case=r['case'],arm=r['arm'],array=r['array'],radius_arcsec=r['radius_arcsec'],
            free_SSE_minus_fixed_SSE=r['free_SSE_minus_fixed_SSE'],relative_to_free=r['free_SSE_minus_fixed_SSE']/r['free_fit']['sse']) for r in rows])
    # The saved 60-domain comparator must reproduce the prior diagnostic exactly.
    previous=read(HERE.parent/'fruit_point_peak_failure_diagnosis_2026-09-13/RATIOS.json')
    old={(r['arm'],r['array'],r['numerator_seed'],r['denominator_seed']):r for r in previous if r['family']=='H/D' and r['pass_index']==6}
    for r in ratios:
        if r['readout']=='free' and r['radius_arcsec']==60:
            o=old[r['arm'],r['array'],r['numerator_seed'],r['denominator_seed']]
            np.testing.assert_allclose(r['relative_error'],o['relative_error'],rtol=1e-12,atol=1e-15)
            assert r['original_pair_usable']==o['available']
    write(HERE/'COMPARISON.json',result)
    for r in summary:print(r['arm'],r['radius_arcsec'],r['readout'],r['quantity'],str(r['within_5pct'])+'/'+str(r['required']),'max%',round(100*r['maximum_abs_error'],5))
    print('Domain maxima',result['maximum_free_domain_change'],result['maximum_fixed_domain_change'])
    print('Rank and conditioning',result['numerical_rank_range'],result['design_condition_range'])
    print('Free minus fixed SSE range',min(r['relative_to_free'] for r in result['free_vs_fixed_objective']),max(r['relative_to_free'] for r in result['free_vs_fixed_objective']))
if __name__=='__main__':main()
