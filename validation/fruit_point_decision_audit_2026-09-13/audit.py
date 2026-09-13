"""Decision arithmetic on saved JSON; no numerical/reduction imports or map reads."""
from pathlib import Path
import hashlib,json,math,subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
READOUT=HERE.parent/'fruit_point_profiled_readout_2026-09-13'
UTILITY=HERE.parent/'fruit_point_nominal_utility_2026-09-13'
THRESHOLD=HERE.parent/'fruit_point_threshold_sensitivity_2026-09-13'
ARRAYS=['a1100','a1400','a2000'];SEEDS=[20260911,20260912]

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def yes(v):return 'yes' if v else 'no'
def signed(v):return f'{v:+.4f}'

def main():
    paths=[READOUT/n for n in ['FIT_COMPARISON.json','MEASUREMENTS.json','RATIOS.json','PRESERVATION_START.json','RESULT_MANIFEST.json']]
    paths += [UTILITY/'DECISION_EVIDENCE.json',UTILITY/'PROTOCOL.md',UTILITY/'RESULT_MANIFEST.json',THRESHOLD/'RESULT_MANIFEST.json']
    for parent in [READOUT,UTILITY]:
        index={r['path']:r['sha256'] for r in read(parent/'RESULT_MANIFEST.json')['files']}
        for p in paths:
            if p.parent==parent and p.name in index:assert digest(p)==index[p.name],str(p)
    protected=read(READOUT/'PRESERVATION_START.json')['protected_worktrees']
    for w in protected:assert subprocess.check_output(['git','-C',w['path'],'status','--short'],text=True)==w['status']
    binding=dict(starting_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=[dict(path=str(p),sha256=digest(p)) for p in paths],protected_worktrees=protected,
        script_sha256=digest(HERE/'audit.py'),scope_sha256=digest(HERE/'SCOPE.md'))
    write(HERE/'INPUT_BINDINGS.json',binding)
    c=read(READOUT/'FIT_COMPARISON.json');m=read(READOUT/'MEASUREMENTS.json');previous_ratios=read(READOUT/'RATIOS.json')
    source={(r['case'],r['arm'],r['array'],r['radius']):r for r in c}
    measurements={(r['case'],r['arm'],r['array']):r['repaired'] for r in m}
    evidence=read(UTILITY/'DECISION_EVIDENCE.json');costs={r['case']:r for r in evidence['costs']}
    traj={(r['case'],r['arm']):r for r in evidence['trajectories']}
    def cost(case):
        v=costs[case];p,c=traj[case,'P'],traj[case,'C']
        for t in [p,c]:
            assert t['complete'] and t['retained_passes']==7 and t['receipt']['passes']==7 and t['receipt']['parent_unchanged']
            assert t['wall_seconds']==t['receipt']['wall_seconds']
        assert v['P_wall_seconds']==p['wall_seconds'] and v['C_wall_seconds']==c['wall_seconds']
        ratio=c['wall_seconds']/p['wall_seconds']
        assert ratio==v['wall_ratio_C_over_P'] and (ratio<=2)==v['pass_gate']
        return dict(case=case,P_seconds=p['wall_seconds'],C_seconds=c['wall_seconds'],ratio=ratio,
            existing_cost_pass=ratio<=2,retained_passes=7,role='full all-array trajectory; no apportioning or retiming')
    peaks=[]
    for array in ARRAYS:
        for seed in SEEDS:
            for state in ['H','D']:
                case=f'{state}_{seed}'
                for arm in ['P','C']:
                    s=source[case,arm,array,60];inner=source[case,arm,array,52];f=s['repaired'];v=measurements[case,arm,array]
                    assert f['peak']==v['original']['fit']['peak']
                    peaks.append(dict(case=case,array=array,arm=arm,peak=f['peak'],absolute_error_pct=100*f['peak_error'],
                        fit_available=s['new_fit_available'],peak_usable=v['judgments']['peak_response_usable'],
                        numerical_complete=s['new_numerical_complete'],inner_numerical_complete=inner['new_numerical_complete'],
                        judgments=v['judgments'],peak_score=v['original']['empirical_peak_score'],
                        inner_probe=v['fixed_inner_domain_probe'],centroid_error_arcsec=f['centroid_error'],
                        matched_trajectory_cost=cost(case)))
    lookup={(r['case'],r['arm'],r['array']):r for r in peaks}
    rows=[]
    for array in ARRAYS:
        for arm in ['P','C']:
            pairings=[('H/H','repeat','H_20260912','H_20260911',1.),('D/D','repeat','D_20260912','D_20260911',1.)]
            pairings += [('D/H','matched',f'D_{s}',f'H_{s}',.9) for s in SEEDS]
            pairings += [('D/H','crossed',f'D_{d}',f'H_{h}',.9) for d,h in [(SEEDS[0],SEEDS[1]),(SEEDS[1],SEEDS[0])]]
            for family,pairing,numerator,denominator,target in pairings:
                n,d=[lookup[name,arm,array] for name in [numerator,denominator]]
                ratio=n['peak']/d['peak'];error=ratio/target-1
                numeric=abs(error)<=.05
                degradation=ratio<.95 if family=='D/H' else None
                usable=n['peak_usable'] and d['peak_usable']
                numerical=n['numerical_complete'] and d['numerical_complete']
                both_costs=n['matched_trajectory_cost']['existing_cost_pass'] and d['matched_trajectory_cost']['existing_cost_pass']
                raw=numeric and (degradation is not False)
                row=dict(id=f'{array}_{arm}_{numerator}_over_{denominator}',array=array,arm=arm,family=family,pairing=pairing,
                    numerator=numerator,denominator=denominator,expected_ratio=target,ratio=ratio,
                    signed_error_pct=100*error,observed_fractional_change_pct=100*(ratio-1),
                    raw_within_5pct=numeric,degradation_below_0p95=degradation,raw_requirement_pass=raw,
                    usable=usable,usable_requirement_pass=usable and raw,numerical_complete=numerical,
                    inner_numerical_complete=n['inner_numerical_complete'] and d['inner_numerical_complete'],
                    involves_incomplete_fit=not numerical,both_trajectory_costs_pass=both_costs,
                    cost_eligible_usable_pass=bool(usable and raw and both_costs),
                    cost_and_numerically_complete_usable_pass=bool(usable and raw and both_costs and numerical),
                    members=[n,d])
                if family=='D/H':
                    old=next(x for x in previous_ratios if x['radius']==60 and x['readout']=='repaired' and x['array']==array and x['arm']==arm and x['numerator_seed']==int(denominator.rsplit('_',1)[1]) and x['denominator_seed']==int(numerator.rsplit('_',1)[1]))
                    assert math.isclose(ratio*old['ratio'],1.,rel_tol=2e-15,abs_tol=2e-15)
                    assert math.isclose(error,1/(1+old['error'])-1,rel_tol=2e-13,abs_tol=2e-15)
                rows.append(row)
    assert len(rows)==36 and len(peaks)==24
    summary=[]
    for array in ARRAYS:
        for arm in ['P','C']:
            for family,pairing in [('H/H','repeat'),('D/D','repeat'),('D/H','matched'),('D/H','crossed')]:
                rr=[r for r in rows if r['array']==array and r['arm']==arm and r['family']==family and r['pairing']==pairing]
                summary.append(dict(array=array,arm=arm,family=family,pairing=pairing,required=len(rr),
                    raw_passes=sum(r['raw_requirement_pass'] for r in rr),usable=sum(r['usable'] for r in rr),
                    usable_passes=sum(r['usable_requirement_pass'] for r in rr),numerically_complete=sum(r['numerical_complete'] for r in rr),
                    cost_eligible_usable_passes=sum(r['cost_eligible_usable_pass'] for r in rr),
                    cost_and_numerically_complete_usable_passes=sum(r['cost_and_numerically_complete_usable_pass'] for r in rr),
                    signed_errors_pct=[r['signed_error_pct'] for r in rr]))
    # Inventory every apparent C-only required-output gain without choosing a realization.
    gains=[]
    for candidate in rows:
        if candidate['arm']!='C':continue
        reference=next(r for r in rows if r['arm']=='P' and r['array']==candidate['array'] and r['numerator']==candidate['numerator'] and r['denominator']==candidate['denominator'])
        if candidate['usable_requirement_pass'] and not reference['usable_requirement_pass']:
            gains.append(dict(kind='new_usable_ratio_success',candidate=candidate,reference=reference))
    for candidate in peaks:
        if candidate['arm']!='C':continue
        reference=lookup[candidate['case'],'P',candidate['array']]
        if candidate['peak_usable'] and abs(candidate['absolute_error_pct'])<=5 and not(reference['peak_usable'] and abs(reference['absolute_error_pct'])<=5):
            gains.append(dict(kind='new_usable_absolute_peak_success',candidate=candidate,reference=reference))
    write(HERE/'PEAKS.json',peaks);write(HERE/'REPEATABILITY_AND_RESPONSE.json',rows);write(HERE/'SUMMARY.json',summary)
    write(HERE/'APPARENT_BENEFITS.json',gains)
    write(HERE/'TRAJECTORY_COSTS.json',[cost(f'{state}_{seed}') for state in ['H','D'] for seed in SEEDS])
    # Retain all historical context, without replacing it with repaired results.
    write(HERE/'PRESERVED_CONTEXT.json',dict(prior_nulls=evidence['nulls'],prior_use_summary=evidence['summary'],
        role='historical registered results, not repaired-readout claims; no image-only veto'))
    for p in binding['files']:assert digest(Path(p['path']))==p['sha256']
    for w in protected:assert subprocess.check_output(['git','-C',w['path'],'status','--short'],text=True)==w['status']
    write(HERE/'VERIFICATION.json',dict(source_hashes_preserved=True,protected_archive_statuses_preserved=True,
        archive_contents_read=False,primary_peaks=24,repeat_and_response_rows=36,inverse_response_identities=24,
        matched_full_trajectory_receipt_pairs=4,saved_judgments_retained=True,numerical_flags_retained=True,
        new_tolerances=0,amplitude_corrections=0,subset_selected_by_noise=False,
        maps_read=0,source_fits=0,FRUIT_runs=0,PTC_calls=0,reserved_observations_opened=0,
        standalone_image_vetoes_applied=False,independent_replication_started=False))
    print('SUMMARY',json.dumps(summary,indent=2))
    print('C-ONLY USABLE SUCCESSES',[(g['kind'],g['candidate'].get('case',g['candidate'].get('numerator')),g['candidate']['array']) for g in gains])
if __name__=='__main__':main()
