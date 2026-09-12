"""Score the frozen saved-map rules; no labels or models are changed here."""
import numpy as np
from common import b, read, HERE, SAVED, OUT, PRIOR

def evaluator_evidence():
    rows = read(OUT/'evaluator/MEASUREMENTS.json')
    original = read(SAVED/'RECEIPTS.json')
    lookup = {(r['case'],r['arm'],r['pass_index']):r for r in rows}
    unchanged = all(m['original'] == o for r, old in zip(rows, original)
                    for m, o in zip(r['measurements'],old['measurement']))
    bootstrap = []
    for case in sorted({r['case'] for r in rows}):
        a, c = lookup[case,'P',0], lookup[case,'C',0]
        bootstrap.append(dict(case=case, equal=a['measurements'] == c['measurements']))
    records = []
    for r in rows:
        for a, m in enumerate(r['measurements']):
            rec = dict(case=r['case'], arm=r['arm'], pass_index=r['pass_index'], array=b.ARRAYS[a],
                       **m['judgments'], original_available=m['original']['measurement_available'],
                       fit=m['original'].get('fit'), shape_error=m['original'].get('shape_error'),
                       empirical_peak_score=m['original'].get('empirical_peak_score'),
                       domain_probe=m['fixed_inner_domain_probe'],
                       truth_score=None if r['original_truth_score'] is None else r['original_truth_score'][a])
            records.append(rec)
    terminal = [r for r in records if r['pass_index'] == 6]
    compact = [r for r in terminal if r['case'].split('_')[0] in ['H','H-shift','D']]
    nulls = [r for r in records if r['case'].split('_')[0] in ['N','B']]
    edges = [r for r in records if r['case'].split('_')[0] == 'E']
    gross = [r for r in records if r['case'].split('_')[0] == 'C']
    common_gate = dict(original_records_preserved=unchanged, matched_bootstraps=all(r['equal'] for r in bootstrap),
        no_null_usable_measurements=not any(r['centroid_usable'] or r['peak_response_usable'] for r in nulls),
        no_boundary_usable_corrections=not any(r['centroid_usable'] for r in edges),
        gross_warnings_retained=all(r['shape_warning'] or r['support_warning'] or r['limitations'] for r in gross),
        reference_compact_available_at_least_ten=sum(r['arm']=='P' and r['centroid_usable'] for r in compact) >= 10,
        admitted_compact_centroids_within_one_arcsec=all(r['truth_score']['centroid_error_arcsec'] <= 1 for r in compact if r['centroid_usable']))
    pairs = []
    for arm in ['P','C']:
        for seed in [20260911,20260912]:
            definitions = [('U2_gain',f'H_{seed}',f'D_{seed}',[1/.9]*3),
                           ('U4_loss',f'D_{seed}',f'H_{seed}',[.9]*3),
                           ('U5_health',f'T_{seed}',f'H_{seed}',[1,.8,1])]
            if seed == 20260912:
                definitions.append(('unchanged_H','H_20260912','H_20260911',[1]*3))
            for use, num, den, expected in definitions:
                n, d = lookup.get((num,arm,6)), lookup.get((den,arm,6))
                for a in range(3):
                    nm = None if n is None else n['measurements'][a]
                    dm = None if d is None else d['measurements'][a]
                    raw = nm is not None and dm is not None and 'fit' in nm['original'] and 'fit' in dm['original']
                    ratio = nm['original']['fit']['peak']/dm['original']['fit']['peak'] if raw else None
                    available = bool(raw and nm['judgments']['peak_response_usable'] and dm['judgments']['peak_response_usable'])
                    error = ratio/expected[a]-1 if ratio is not None else None
                    passed = available and abs(error) <= .05
                    if use == 'U4_loss':
                        passed = passed and ratio < .95
                    if use == 'U5_health' and a == 1:
                        passed = passed and ratio < .9
                    pairs.append(dict(use=use, arm=arm, numerator=num, denominator=den, array=b.ARRAYS[a],
                        available=available, raw_ratio=ratio, expected=expected[a], relative_error=error, pass_accuracy=bool(passed),
                        newly_unavailable=bool(nm is not None and dm is not None and dm['judgments']['peak_response_usable'] and not nm['judgments']['peak_response_usable']),
                        already_unavailable=bool(nm is not None and dm is not None and not dm['judgments']['peak_response_usable'] and not nm['judgments']['peak_response_usable'])))
    summaries = []
    for arm in ['P','C']:
        rr = [r for r in compact if r['arm'] == arm]
        nn = [r for r in nulls if r['arm'] == arm]
        ee = [r for r in edges if r['arm'] == arm]
        summaries.append(dict(arm=arm, compact_terminal_rows=len(rr), required_compact_terminal_rows=18,
            original_compact_available=sum(r['original_available'] for r in rr),
            compact_centroid_usable=sum(r['centroid_usable'] for r in rr),
            compact_peak_usable=sum(r['peak_response_usable'] for r in rr),
            retained_shape_warnings=sum(r['shape_warning'] for r in rr),
            null_array_passes=len(nn), null_source_evidence=sum(r['source_evidence_present'] for r in nn),
            null_usable=sum(r['centroid_usable'] or r['peak_response_usable'] for r in nn),
            edge_array_passes=len(ee), edge_warnings=sum(r['support_warning'] for r in ee),
            edge_usable_centroids=sum(r['centroid_usable'] for r in ee)))
    result = dict(common_gate=common_gate, passed=all(bool(v) for v in common_gate.values()),
        saved_map_count=len(rows), array_map_count=len(records), bootstrap_checks=bootstrap,
        summaries=summaries, terminal_records=terminal, compact_terminal=compact, peak_pairs=pairs,
        all_pass_judgment_counts={arm:{key:sum(r['arm']==arm and r[key] for r in records)
            for key in ['source_evidence_present','centroid_usable','peak_response_usable','shape_warning','support_warning']} for arm in ['P','C']},
        records_withheld_for_centroid=[dict(case=r['case'],array=r['array'],limitations=r['limitations'],
            centroid_probe_difference=r['domain_probe'].get('centroid_difference_arcsec'),truth_score=r['truth_score'])
            for r in compact if not r['centroid_usable']],
        interpretation='Prospective data-only rule evaluated on reused saved maps; original scores remain authoritative for the old registration. No calibrated uncertainty or independent validation.')
    b.write(HERE/'EVALUATOR_EVIDENCE.json',result)
    return result

def numerical_evidence():
    rows = read(OUT/'stopping/RECEIPTS.json')
    nonempty = [r for r in rows if not r['empty']]
    comparisons = [r for r in nonempty if r['comparison'] is not None]
    condensed = []
    for r in nonempty:
        condensed.append({k:r.get(k) for k in ['name','case','array','pass_index','qualified','operational_stop_available',
            'snapshots','termination','wall_seconds','comparison','final']})
    summary = dict(problems=len(rows), empty=len(rows)-len(nonempty), nonempty=len(nonempty),
        operational_stop_available=sum(r['operational_stop_available'] for r in nonempty),
        declared_checkpoint_found=sum('declared' in r['snapshots'] for r in nonempty),
        tight_checkpoint_found=sum('tight' in r['snapshots'] for r in nonempty),
        comparable=len(comparisons), stability_passed=sum(r['comparison']['passed'] for r in comparisons),
        qualified_nonempty=sum(r['qualified'] for r in nonempty),
        fitted_comparisons=sum(r['comparison']['fitted_readout_comparison_available'] for r in comparisons),
        fitted_comparison_limits=sum(not r['comparison']['fitted_readout_comparison_available'] for r in comparisons),
        solve_seconds=sum(r['wall_seconds'] for r in nonempty))
    maxima = {}
    for key in ['sampled_peak_relative_difference','brightness_centroid_difference_arcsec',
                'fitted_peak_relative_difference','fitted_centroid_difference_arcsec','model_relative_L2']:
        rr = [r for r in comparisons if key in r['comparison']]
        worst = max(rr,key=lambda r:r['comparison'][key]) if rr else None
        maxima[key] = None if worst is None else dict(value=worst['comparison'][key],name=worst['name'])
    groups = []
    for name, states in [('real',['real123424']), ('compact_mild_response_loss',['H','H-shift','D','T']),
                         ('gross_coma',['C']), ('near_boundary',['E'])]:
        selected = [r for r in nonempty if r['case'].split('_')[0] in states]
        compared = [r for r in selected if r['comparison'] is not None]
        fitted = [r for r in compared if r['comparison']['fitted_readout_comparison_available']]
        group = dict(name=name, problems=len(selected), tight_comparisons=len(compared),
                     qualified=sum(r['qualified'] for r in selected), fitted_comparisons=len(fitted),
                     both_fitted_allocations_pass=sum(r['comparison']['fitted_peak_relative_difference']<=.005
                         and r['comparison']['fitted_centroid_difference_arcsec']<=.1 for r in fitted))
        for key in ['sampled_peak_relative_difference','brightness_centroid_difference_arcsec',
                    'fitted_peak_relative_difference','fitted_centroid_difference_arcsec']:
            values = [r['comparison'][key] for r in compared if key in r['comparison']]
            group[key+'_range'] = [min(values),max(values)] if values else None
        groups.append(group)
    result = dict(summary=summary, maxima=maxima, groups=groups, problems=condensed,
        passed=bool(len(rows)==144 and len(nonempty)==60 and all(r['qualified'] for r in rows)),
        interpretation='Same saved problems and optimizer paths; operational cap unchanged. These are numerical comparisons, not new FRUIT trajectories.')
    b.write(HERE/'STOPPING_EVIDENCE.json',result)
    return result

def main():
    a = evaluator_evidence()
    n = numerical_evidence()
    result = dict(evaluator_saved_stage_passed=a['passed'], stopping_qualification_passed=n['passed'],
        operational_rerun_admitted=bool(a['passed'] and n['passed']),
        disposition=('saved_stages_passed_operational_comparison_pending' if a['passed'] and n['passed'] else
                     'retain_evaluator_reject_completion_only_repair' if a['passed'] else 'saved_stage_repair_not_accepted'),
        candidate_policy_recommended=False, independent_pointing_replication=False,
        new_cleaning_calls=0, new_operational_trajectories=0,
        saved_stage_campaign=read(OUT/'COMPLETE.json') if (OUT/'COMPLETE.json').exists() else read(OUT/'FAILURE.json'),
        prior_OG=read(PRIOR/'DIAGNOSTIC_EVIDENCE.json')['OG'])
    b.write(HERE/'DECISION_EVIDENCE.json',result)
    print('A:',a['passed'],a['summaries'],a['common_gate'])
    print('B:',n['passed'],n['summary'],n['maxima'])
    print('Operational rerun admitted:',result['operational_rerun_admitted'])

if __name__ == '__main__':
    main()
