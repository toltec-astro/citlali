"""Compare a fixed onset-support trial with sealed control, preserving all misses."""
import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path

import numpy as np
import report_rtc_jump_losses as plots


def cells(ranges):
    return {i for a,b in ranges for i in range(a,b)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('control_diagnosis',type=Path)
    p.add_argument('trial_truth',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--source',default='uncommitted pilot')
    a=p.parse_args()
    if any(root.resolve() in a.output.resolve().parents for root in [a.control_diagnosis,a.trial_truth]):
        p.error('output must be outside the input evidence directories')
    a.output.mkdir(parents=True,exist_ok=False)
    old=json.loads((a.control_diagnosis/'truth-final-01/replays.json').read_text())
    new=json.loads((a.trial_truth/'replays.json').read_text())
    cm=json.loads((a.control_diagnosis/'truth-final-01/truth-analysis.json').read_text())
    tm=json.loads((a.trial_truth/'truth-analysis.json').read_text())
    assert len(old)==len(new)==20 and tm['status']=='PASS'
    audited=Counter()
    for before,after in zip(old,new,strict=True):
        # Every independent input and upstream Learn result remains identical.
        for key in ['background_identity','trial','jump_injected','pulse_injected','truth_cells','truth_begin','truth_end',
                    'truth_duration_seconds','injected_offset','injected_excursion','sigma_delta_unmodified',
                    'candidate_edges_at_truth','all_candidate_edges','all_candidates','samples','known_location_diagnostic_only']:
            assert before[key]==after[key],(after['trial'],key)
        assert after['transition_policy']=='rtc-jump-transition-onset-2026-09-12-v2'
        assert after['hard_event_accepted'] is after['apply_authorized'] is False
        sample=np.array(after['samples']);audited['sample_values_preserved']+=len(sample)*2
        for bg,tg in zip(before['groups'],after['groups'],strict=True):
            for key in ['event','forced_location','origin','time_scale','trial_exclusion','original_exclusions','truth_exclusions','candidate_members']:
                assert bg[key]==tg[key],key
            # Independent set closure: no assumed chronological traversal.
            component=set(range(*tg['transition_seed_edges']))
            while True:
                updated=set(component)
                for m in tg['candidate_members']:
                    edge=set(range(m['earlier'],m['later']+1))
                    if edge & component:updated |= edge
                if updated==component:break
                component=updated
            assert component==set(range(*tg['transition_onset_edges']))
            # Reconstruct all candidate guards in the established search context.
            seed=tg['transition_seed_edges'];middle=sample[seed[0],1]+(sample[seed[1]-1,1]-sample[seed[0],1])/2
            first=int(np.searchsorted(sample[:,1],middle-2.05,side='left'))
            members={(m['earlier'],m['later'],m['coordinate']) for m in tg['candidate_members']}
            expected=set()
            for c in after['all_candidates']:
                lo,hi=c['earlier'],c['later']
                # Supplied-location probes preserve the existing diagnostic's
                # full candidate-mask list; production bounds use local lookup.
                if not tg['forced_location']:
                    if lo<first-1 or sample[lo,1]>middle+2.05:continue
                    if (lo,hi,c['coordinate']) in members and set(range(lo,hi+1)) <= component:continue
                center=sample[lo,1]+(sample[hi,1]-sample[lo,1])/2
                begin=min(lo,int(np.searchsorted(sample[:,3],center-.05,side='right')))
                end=max(hi+1,int(np.searchsorted(sample[:,2],center+.05,side='left')))
                expected.update(range(begin,end))
            assert expected==cells(tg['transition_neighbor_exclusions']),(after['background_identity'],after['trial'],tg['forced_location'])
            audited['onset_and_neighbor_mask_audits']+=1
            for bc,tc in zip(bg['coordinates'],tg['coordinates'],strict=True):
                assert bc['coordinate']==tc['coordinate']
                for version in ['original','truth_fit']:
                    for key in ['primary_available','primary_support_cause','primary_pre_scale_fit','primary_cubic','primary_offset',
                                'short_available','short_cause','short_pre_scale_fit','short_offset','primary_rows','short_rows',
                                'consistency_cause','sigma_delta_comparison_tolerance','recovery',
                                'matched_uninjected_primary','matched_uninjected_short']:
                        assert bc[version][key]==tc[version][key],(after['trial'],version,key)
                audited['upstream_coordinate_chains_preserved']+=1
    rows=[];counts=Counter()
    key=lambda r:(r['background'],r['trial'],r['event'],r['coordinate'],r['version'])
    prior={key(r):r for r in cm['records']}
    for r in tm['records']:
        if r['version']!='current':continue
        b=prior[key(r)]
        out={k:r[k] for k in ['background','background_identity','coordinate','jump_injected','pulse_injected','forced_location','candidate_detected','injected_offset']}
        out['trial_name']=r['trial']
        for label,m in [('control',b),('trial',r)]:
            out[label]={k:m[k] for k in ['end_to_end_retained','diagnostic_retained','first_decisive_stage',
                'primary_counts','short_counts','offset_fit_available','fitted_offset','raw_fit_minus_injected',
                'recovered_injection_error','actual_transition']}
        rows.append(out)
        if r['jump_injected']:
            counts['jump_coordinate_cases']+=1;counts['detected']+=r['candidate_detected']
            counts['control_retained']+=b['end_to_end_retained'];counts['trial_retained']+=r['end_to_end_retained']
            counts['gained']+=r['end_to_end_retained'] and not b['end_to_end_retained']
            counts['lost']+=b['end_to_end_retained'] and not r['end_to_end_retained']
        elif r['background']==0:
            counts['synthetic_control_coordinates']+=1
            counts['synthetic_control_retained']+=r['end_to_end_retained']
    assert counts['jump_coordinate_cases']==24 and counts['detected']==18
    result=dict(source=a.source,counts=dict(counts),audits=dict(audited),records=rows,
        scope='Fixed20-case trial. Missing candidates remain in denominators. Known-location probes are downstream diagnostics only. Real backgrounds are not assumed event-free.',
        admission='No accepted physical events, flags, corrections or Apply authority.')
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output/'comparison.tsv').open('w',newline='') as stream:
        flat=[]
        for r in rows:
            q={k:v for k,v in r.items() if k not in ['control','trial']}
            for label in ['control','trial']:
                for k,v in r[label].items():q[label+'_'+k]=json.dumps(v) if isinstance(v,(dict,list)) else v
            flat.append(q)
        w=csv.DictWriter(stream,fieldnames=list(flat[0]),delimiter='\t');w.writeheader();w.writerows(flat)
    plots.NAMES=('Control reassessment','Onset trial','Known-support diagnostic')
    selected=[('152385-04','sharp_step','sharp.png'),('152385-04','sharp_plus_neighbor_spike','sharp-neighbor.png'),
              ('152430-08','sharp_plus_neighbor_spike','complex-background.png'),('synthetic','sharp_step','synthetic.png')]
    for identity,trial,name in selected:
        i=next(i for i,d in enumerate(new) if d['background_identity'].startswith(identity) and d['trial']==trial)
        d=copy.deepcopy(new[i]);bg=old[i]['groups'][0];tg=d['groups'][0]
        tg['original_exclusions']=bg['inferred_reassessment_exclusions']
        for c in range(2):tg['coordinates'][c]['original']=bg['coordinates'][c]['current']
        plots.detail(d,a.output/name)
    fmt=lambda v:'unavailable' if v is None else f'{v:.5g}'
    lines=['# Onset-support trial', '',
        f'**The fixed trial retains {counts["trial_retained"]}/24 injected-jump coordinate measurements, compared with {counts["control_retained"]}/24 for the preserved control: {counts["gained"]} gained and {counts["lost"]} lost.** Candidate detection remains 18/24. These are conditional measurement results, not accepted physical events.', '',
        'The only runtime change anchors the onset to connected candidate edge cells containing the original seed. Later disconnected members keep their existing exclusion masks. Candidate finding, original groups, background fits, frozen scales, thresholds, context and the one-reassessment limit remain fixed. This changes RTC Learn evidence consumed by the existing Consider chain; it produces no Apply plan.', '',
        '| Background / injection | Coordinate | Detected | Control retained | Trial retained | Trial first terminal stage |', '|---|---|---|---|---|---|']
    for r in rows:
        if r['jump_injected']:
            stage=r['trial']['first_decisive_stage']
            if not r['candidate_detected']:stage='candidate miss (supplied-location probe: '+stage+')'
            lines.append(f'| {r["background_identity"].split(":sha256:")[0]} / {r["trial_name"]} | {r["coordinate"]} | {r["candidate_detected"]} | {r["control"]["end_to_end_retained"]} | {r["trial"]["end_to_end_retained"]} | {stage} |')
    lines += ['',
        'The six gains are x and r for the sharp step, three-cell transition and sharp step plus neighboring spike on 152385/network 4/channel 61. The sharp bound now covers three native cells (about 24.576 ms), compared with the earlier 0.852-second bound. Its retained short-fit post support is 104 samples (89 with the neighboring spike), rather than 20; these cases need no additional refit. The sharp bound covers the known instantaneous onset with signed endpoint errors of −1.5 and +1.5 native cells.', '',
        'Three detected x-coordinate cases on 152430/network 8/channel 253 remain unresolved: the sharp and three-cell transitions fail the unchanged initial consistency check; the sharp-plus-spike case now encounters a competing neighbor exclusion while measuring its bound. Its r coordinate remains retained with a shorter bound. This trial does not tune away those failures or claim that every real-corpus loss is recoverable.', '',
        '## Fitting support and truth error', '',
        '| Case / coordinate | Short pre/post: control → trial | Offset: control → trial | Added offset | Bound error: control → trial (native cells) |', '|---|---|---|---|---|']
    for r in rows:
        if r['jump_injected'] and r['trial_name'] in ['sharp_step','sharp_plus_neighbor_spike'] and r['background'] in [1,2]:
            b,t=r['control'],r['trial'];be=b['actual_transition'].get('boundary_error_samples');te=t['actual_transition'].get('boundary_error_samples')
            lines.append(f'| {r["background"]}/{r["trial_name"]}/{r["coordinate"]} | {b["short_counts"]} → {t["short_counts"]} | {fmt(b["fitted_offset"])} → {fmt(t["fitted_offset"])} | {fmt(r["injected_offset"])} | {be} → {te} |')
    lines += ['', 'Background 1 is 152385/network 4/channel 61; background 2 is 152430/network 8/channel 253. Offset values retain each coordinate’s original units. The complete [comparison JSON](comparison.json) and [TSV](comparison.tsv) include every control, unavailable fit, signed boundary error in seconds/cells and the incremental error against a same-support uninjected fit. No accuracy denominator excludes a miss or unresolved injection.', '',
        'Four figures compare the preserved reassessment, the onset trial and a separately labelled known-support fit. Their masks, actual fitting samples, fitted states, offsets, dispositions and residuals are explicit:', '',
        '- [Previously lost sharp step](sharp.png)', '- [Sharp step with neighboring spike](sharp-neighbor.png)',
        '- [Complex real background](complex-background.png)', '- [Matched synthetic success](synthetic.png)', '',
        f'The synthetic no-jump, spike and finite-pulse controls retain {counts["synthetic_control_retained"]}/{counts["synthetic_control_coordinates"]} persistent-jump measurements. The six twelve-cell-ramp coordinate misses remain misses (about 98.304 ms transitions at the original 20-sigma added amplitudes). Their supplied-location diagnostics remain separate in the linked audited trial records. Real unmodified comparisons are not assumed event-free.', '',
        'The comparison independently verifies identical original samples, candidate membership, initial and known-support fits, scale roles and recovery evidence, plus the new onset components and all neighbor masks. The shared truth audit checks fitting populations, injected values, model losses and explicit-anchor/production parity. An available offset is not a calibrated parameter uncertainty; sigma_delta remains the approved fixed empirical tolerance.', '',
        f'Source: `{a.source}`. Control source: `8ef77477823a5393f9cc7860d8f963509aee0fc8`, closure `019a0ee3fe710c1a62da3107286cf3c92794a5ca`. Full trial audits: [{a.trial_truth.resolve()/"truth-analysis.json"}]({a.trial_truth.resolve()/"truth-analysis.json"}).', '',
        'Local compilation, focused/broad tests and independent exact-SHA reviews are recorded in the external completion receipt. This report does not itself establish acceptance, Unity/Spack qualification, whole-corpus performance, event identity, timing uncertainty, real scan binding or production readiness. No full 143-file corpus rerun was performed.', '']
    report='\n'.join(lines)
    for name in ['comparison.json','comparison.tsv',*(x[2] for x in selected)]:
        report=report.replace(']('+name+')',']('+str((a.output/name).resolve())+')')
    (a.output/'README.md').write_text(report)
    print(json.dumps(dict(counts=dict(counts),audits=dict(audited),report=str(a.output/'README.md')),indent=2))


if __name__=='__main__':
    main()
