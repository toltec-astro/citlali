"""Compact descriptive evidence; original qualification and gates remain fixed."""
import numpy as np
from common import b,read,HERE,OUT,SOLUTIONS

def main():
    rows=read(OUT/'screen/PAIRS.json');available=[r for r in rows if r['available']]
    nonempty=[r for r in available if not r['image']['empty']]
    selected=read(OUT/'SELECTED_STATES.json')
    lookup={(r['case'],r['pass_index'],r['array']):r for r in rows}
    replay=read(OUT/'REPLAY_RESULTS.json')
    case_summary=[];measurements=[];selected_screen=[]
    for r in replay:
        for a in r['arrays']:
            groups=[g for g in r['subspace_groups'] if g['array']==a['array']]
            n,t=a['nominal']['original'].get('fit'),a['tight']['original'].get('fit')
            width=None if n is None or t is None else np.array(t['widths'])/n['widths']-1
            screen=lookup[r['case'],r['pass_index'],a['array']]
            sg=screen['groups'];energy=screen['projection']['difference_energy']
            summary=dict(case=r['case'],pass_index=r['pass_index'],role=r['role'],array=a['array'],
                peak_relative_change=a['peak_relative_change'],centroid_change_arcsec=a['centroid_change_arcsec'],
                both_peak_usable=a['both_peak_usable'],both_centroid_usable=a['both_centroid_usable'],
                width_relative_change=width,
                nominal_peak=None if n is None else n['peak'],tight_peak=None if t is None else t['peak'],
                nominal_centroid=None if n is None else n['centroid'],tight_centroid=None if t is None else t['centroid'],
                nominal_truth=a['nominal_truth'],tight_truth=a['tight_truth'],
                nominal_judgments=a['nominal']['judgments'],tight_judgments=a['tight']['judgments'],
                nominal_domain_probe=a['nominal']['fixed_inner_domain_probe'],tight_domain_probe=a['tight']['fixed_inner_domain_probe'],
                nominal_shape_error=a['nominal']['original'].get('shape_error'),tight_shape_error=a['tight']['original'].get('shape_error'),
                exterior_difference_rms=a['exterior_difference_rms'],
                maximum_subspace_angle_degrees=max(g['maximum_principal_angle_degrees'] for g in groups),
                median_subspace_angle_degrees=float(np.median([g['maximum_principal_angle_degrees'] for g in groups])),
                maximum_projector_distance=max(g['normalized_projector_distance'] for g in groups),
                subspace_groups_over_one_degree=sum(g['maximum_principal_angle_degrees']>1 for g in groups),
                centered_projected_energy=sum(g['centered_difference_energy'] for g in sg),
                coherent_group_energy_fraction=sum(g['coherent_time_mean_energy'] for g in sg)/energy if energy else None,
                top_group_energy_fraction=max(g['difference_energy'] for g in sg)/energy if energy else None,
                strongest_subspace_group=max(groups,key=lambda g:g['maximum_principal_angle_degrees']),
                next_map_difference_energy=a['signed_total_difference_energy'],
                restoration_difference_energy=a['restored_model_difference_energy'],
                residual_processing_difference_energy=a['residual_processing_difference_energy'])
            measurements.append(summary)
            selected_screen.append({k:v for k,v in screen.items() if k!='groups'})
        case_summary.append(dict(case=r['case'],pass_index=r['pass_index'],
            timing={label:{k:rec[k] for k in ['clean_seconds','map_seconds','evaluation_seconds','wall_seconds']} for label,rec in r['timing'].items()}))
    compact=[r for r in nonempty if r['case'].split('_')[0] in ['H','H-shift','D','T']]
    all_screen=[]
    for r in nonempty:
        i=r['image'];p=r['projection']
        all_screen.append(dict(case=r['case'],pass_index=r['pass_index'],array=r['array'],
            **{k:i[k] for k in ['nominal_maximum','tight_maximum','relative_image_L2','core_energy_fraction','rim_energy_fraction','low_coverage_energy_fraction','top_one_percent_pixel_energy_fraction','change_components','scale_bands','selected_change_relative_to_target','objective_nominal','objective_tight','objective_zero','remaining_objective_fraction_removed','objective_reduction_relative_to_zero']},projection=p))
    summary=dict(available_pairs=len(available),empty_pairs=sum(r['image']['empty'] for r in available),
        nonempty_pairs=len(nonempty),unavailable_pairs=len(rows)-len(available),
        pair_group_records=sum(len(r.get('groups',[])) for r in rows),
        replay_states=len(replay),replay_branches=read(OUT/'CLEANING_CALLS.json')['started'],
        next_map_array_comparisons=len(measurements),
        raw_peak_changes_within_0p5_percent=sum(abs(r['peak_relative_change'])<=.005 for r in measurements),
        raw_centroid_changes_within_0p1_arcsec=sum(r['centroid_change_arcsec']<=.1 for r in measurements),
        maximum_raw_peak_relative_change=max(abs(r['peak_relative_change']) for r in measurements),
        maximum_raw_centroid_change_arcsec=max(r['centroid_change_arcsec'] for r in measurements),
        joint_usable_peak_comparisons=sum(r['both_peak_usable'] for r in measurements),
        joint_usable_centroid_comparisons=sum(r['both_centroid_usable'] for r in measurements),
        compact_pairs_with_over_half_difference_energy_in_rim=sum(r['image']['rim_energy_fraction']>.5 for r in compact),
        compact_pair_count=len(compact),
        original_qualification_remains_failed=True,new_candidate_admitted=False,
        full_trajectory_stability_established=False)
    result=dict(summary=summary,selected=selected,selected_screen=selected_screen,measurements=measurements,
                all_nonempty_screen=all_screen,cases=case_summary,campaign=read(OUT/'COMPLETE.json'),
                unavailable=[r for r in rows if not r['available']],
                disposition='diagnostic_complete_mixed_central_response_with_consequential_structure_and_subspace_changes',
                recommendation='Propose one explicit image-selection constraint targeting unsupported fine-scale feedback; preserve free source amplitude, centroid and shape. Exact formulation and execution require a separate owner decision.',
                limitations='No exact null space established; remaining numerical progress and conditioning are not excluded. No revised gate, new estimator or trajectory qualification.',
                descriptive_scales='0.5%/0.1 arcsec numerical and 5%/1 arcsec operational; no retrospective replacement of acceptance rules.')
    b.write(HERE/'DECISION_EVIDENCE.json',result)
    print(summary)
    print([(r['case'],r['array'],r['peak_relative_change'],r['centroid_change_arcsec'],r['width_relative_change']) for r in measurements])

if __name__=='__main__':main()
