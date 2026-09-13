"""Compact decision report and full saved-case tables; standard library only."""
from pathlib import Path
import json,math
HERE=Path(__file__).resolve().parent
read=lambda name:json.loads((HERE/name).read_text())

def yn(v):return 'yes' if v else 'no'
def main():
    rows=read('REPEATABILITY_AND_RESPONSE.json');peaks=read('PEAKS.json');costs=read('TRAJECTORY_COSTS.json');gains=read('APPARENT_BENEFITS.json')
    def row(array,arm,num,den):return next(r for r in rows if r['array']==array and r['arm']==arm and r['numerator']==num and r['denominator']==den)
    identity_checks=0
    for array in ['a1100','a1400','a2000']:
        for arm in ['P','C']:
            rh=row(array,arm,'H_20260912','H_20260911')['ratio'];rd=row(array,arm,'D_20260912','D_20260911')['ratio']
            m1=row(array,arm,'D_20260911','H_20260911')['ratio'];m2=row(array,arm,'D_20260912','H_20260912')['ratio']
            x12=row(array,arm,'D_20260911','H_20260912')['ratio'];x21=row(array,arm,'D_20260912','H_20260911')['ratio']
            assert math.isclose(m2,m1*rd/rh,rel_tol=2e-15)
            assert math.isclose(x12,m1/rh,rel_tol=2e-15)
            assert math.isclose(x21,m1*rd,rel_tol=2e-15)
            identity_checks+=3
    lines=['# All repeatability and response pairs','',
        'P = pixelwise reference; C = starlet. Seeds 1/2 are 20260911/20260912. H/H and D/D use seed2/seed1 and expected value one. D/H expects 0.9; raw response success also requires D/H <0.95. Error is 100*(ratio/expected − 1). Every pair is retained, including withheld and numerically incomplete fits.','',
        'The cost columns always describe C/P full-trajectory wall cost for the numerator and denominator observations. They apply to starlet advancement, not to baseline accuracy. No pair-average cost or per-array apportionment is used. Numerical flags show numerator/denominator at 60″, followed by the corresponding 52″ flags; the primary measurement remains 60″.','',
        '| Array | Arm | Family | Pair | Ratio | Signed error % | Raw pass | Peak usable | Usable pass | Numerical 60″; 52″ | C/P costs numerator; denominator | Both costs ≤2 |',
        '| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        n,d=r['members'];num=n['case'].replace('_20260911','1').replace('_20260912','2');den=d['case'].replace('_20260911','1').replace('_20260912','2')
        nf=f"{yn(n['numerical_complete'])}/{yn(d['numerical_complete'])}; {yn(n['inner_numerical_complete'])}/{yn(d['inner_numerical_complete'])}"
        cr=f"{n['matched_trajectory_cost']['ratio']:.4f}; {d['matched_trajectory_cost']['ratio']:.4f}"
        lines.append(f"| {r['array']} | {r['arm']} | {r['family']} {r['pairing']} | {num}/{den} | {r['ratio']:.7f} | {r['signed_error_pct']:+.4f} | {yn(r['raw_requirement_pass'])} | {yn(r['usable'])} | {yn(r['usable_requirement_pass'])} | {nf} | {cr} | {yn(r['both_trajectory_costs_pass'])} |")
    lines += ['','## Every contributing absolute peak and judgment','',
        'All original repaired judgments and inner-probe fit records are retained verbatim in PEAKS.json. Here source/association/centroid/peak are separate saved usability judgments; shape and support warnings are retained without adding a new veto.','',
        '| Array | State | Arm | Peak | Absolute error % | Source/association/centroid/peak | Numerical 60″/52″ | Shape/support warning | Remaining limitation or peak reason |',
        '| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |']
    for p in peaks:
        j=p['judgments'];values='/'.join(yn(j[k]) for k in ['source_evidence_present','associated_source','centroid_usable','peak_response_usable'])
        reasons=j['limitations'][:]
        if not j['peak_score_pass']:reasons.append('low peak score')
        if not j['peak_stability_pass']:reasons.append('peak domain sensitivity')
        lines.append(f"| {p['array']} | {p['case']} | {p['arm']} | {p['peak']:.6f} | {p['absolute_error_pct']:+.4f} | {values} | {yn(p['numerical_complete'])}/{yn(p['inner_numerical_complete'])} | {yn(j['shape_warning'])}/{yn(j['support_warning'])} | {', '.join(reasons) or 'none'} |")
    (HERE/'ALL_CASES.md').write_text('\n'.join(lines)+'\n')
    text='''# Decision audit: close this starlet experiment for POINT

2026-09-13 · SCI-FRUIT v0.1 development · r0.1

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md),
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md),
[owner-directed audit scope](SCOPE.md), [existing operational and cost rules](../fruit_point_nominal_utility_2026-09-13/PROTOCOL.md)
and [repaired common readout](../fruit_point_profiled_readout_2026-09-13/SCIENTIFIC_REPORT.md).
This is a saved-products decision, with no new fitting or reduction.

**Recommendation: no operationally viable advantage has been demonstrated;
close this starlet experiment for POINT. It has not earned an independent
replication on the present evidence.** Preserve the repaired common readout
and every previous result. This disposition concerns this tested candidate,
not all possible wavelet methods or future scientific uses.

## Compact decision table

All measurements below use the repaired 60″ pass-6 readout. Error is signed and
relative to the declared ratio: one for unchanged H/H and D/D, 0.9 for D/H.
Unchanged ratios use seed2/seed1. Seeds 1/2 are 20260911/20260912. The existing
5% band is unchanged; D/H also must be below 0.95. D/D is the requested saved
repeatability diagnostic, not a retroactive relabeling of an earlier gate.

| Array | Unchanged H/H; D/D error: reference → starlet | Matched D/H usable successes | Crossed D/H usable successes | Operational assessment |
| --- | --- | --- | --- | --- |
| a1100 | −2.285%; −2.245% → −2.816%; −2.277%; all usable | 2/2 → 2/2 | 2/2 → 2/2 | Reference already meets all these requirements. Starlet is slower; no new operational success. |
| a1400 | −6.113%; −2.047% → −5.004%; −3.948%; all pairs withheld | 0/2 → 0/2 | 0/2 → 0/2 | Improved raw matched ratios provide no usable gain measurement. |
| a2000 | −11.665%; −10.945% → −6.585%; −7.822%; C repeats usable but fail | 1/2 → 2/2* | 0/2 → 0/2 | Better matched recovery does not provide reliable change discrimination; the incremental usable pair depends on an incomplete fit. |

The denominators retain every pair. P's a2000 H/H is usable but fails; its D/D
is withheld and its raw repeatability error exceeds 5%. C's a2000 H/H involves the incomplete fit,
but **D/D is numerically complete and still shows a false 7.822% loss**. Thus
incomplete fitting alone cannot explain away the failed repeatability.

C's matched D/H errors, in seed1/seed2 order, are a1100 **+0.215%, +0.771%**;
a1400 **−1.452%, −0.357%**; a2000 **+0.538%, −0.793%**. All six raw values
satisfy the response requirement, but only four pairs are usable. C's crossed
D/H errors, in D1/H2 then D2/H1 order, are **+3.119%, −2.066%**;
**+3.739%, −5.343%**; **+7.625%, −7.325%**, respectively. Only the two a1100
crossed pairs are usable and within the unchanged requirement. Crossed
recombinations are dependent diagnostics, not more independent realizations.

At a2000, D1/H2 = 0.96863 also fails the existing below-0.95 degradation test:
it reports only a 3.14% loss for the imposed 10% loss. D2/H1 = 0.83407 reports
a 16.59% loss. Matched ratios partly cancel the shared repeat differences:
D2/H2 = (D1/H1) × (D2/D1)/(H2/H1). That identity explains why good matched
recovery can coexist with poor repeatability; it does not identify the physical
origin of the noise-dependent map response.

## The cost of each required observation

These are the saved full seven-pass, all-array wall times, including the
original evaluation and output work. They are not retimed or replaced with
isolated repaired-fitter timings. The existing C/P ≤2 requirement applies to
each observation needed for a claimed benefit. A passing partner cannot offset
a failing trajectory, and choosing one array cannot divide this measured cost.

| State | Reference seconds | Starlet seconds | C/P | Existing cost requirement |
| --- | ---: | ---: | ---: | --- |
'''
    for c in costs:text+=f"| {c['case']} | {c['P_seconds']:.4f} | {c['C_seconds']:.4f} | {c['ratio']:.4f} | {'pass' if c['existing_cost_pass'] else 'fail'} |\n"
    text+='''
Every pair requiring H1 retains its **2.114× cost failure**. No natural array
subset is faster: the source-case ratios range from 1.588× to 2.114×. Restricting
attention to seed2 would be selection by a favorable noise realization.

## Every starlet-only usable success, with its limitations

The saved H/D inventory contains exactly three instances where C has an
accurate usable result and P does not, under the existing requirements:

| Apparent benefit | Reference → starlet error | Saved matched cost | Depends on incomplete H2/a2000? | Why it does not define a replication subset |
| --- | --- | --- | --- | --- |
| a2000 D2/H2 response | −2.562% but withheld → −0.793%, usable | D2 1.613×; H2 1.588×, both pass | Yes | The incremental matched pair lacks independent numerical completion; unchanged H/H and D/D and both crossed pairs fail. |
| a2000 H2 absolute peak | −13.308% → −4.474%, both usable | H2 1.588×, passes | Yes | The apparent absolute recovery gain depends on the same incomplete fit and does not establish repeatable response. |
| a2000 D1 absolute peak | −5.147% → +2.810%, both usable | D1 1.938×, passes | No | D2 remains −5.231%, outside 5%, and the complete D/D pair fails repeatability. Choosing D1 alone would select a favorable realization. |

The other a2000 matched pair improves raw error from −3.349% to +0.538%,
with complete usable fits in both arms, but P already passes and H1 exceeds
cost. Raw a1400 matched recovery also improves, without dependence on the
incomplete fit, but every a1400 pair is withheld. These are retained scientific
observations, not qualifying operational benefits.

The single incomplete primary fit is C/H2/a2000 at 60″. Its flag propagates to
H/H, matched D2/H2 and crossed D1/H2. The complete D/D and D2/H1 failures
remain independent of that fit. No case is dropped and no new numerical veto
is substituted for its saved usability judgment.

## Preserved outcomes and stopping point

All 12 H/D centroids per arm remain usable within 1″. The preserved shifted,
null and boundary safeguards and earlier T/H health results are unchanged;
no new repaired T/H claim is manufactured. C's reduced false null feedback is
retained, while both arms already reported zero null sources. It does not
resolve this failed source-response comparison. No standalone image-quality,
exterior-RMS, model-image agreement or shape-only veto decides this disposition.

The audit contains all 24 contributing peaks, 36 repeat/response pairs, both
members' full saved judgments and 60″/52″ numerical flags. [All cases](ALL_CASES.md),
[apparent-benefit records](APPARENT_BENEFITS.json), [trajectory costs](TRAJECTORY_COSTS.json)
and [verification](VERIFICATION.json) retain the evidence. The prior reports,
repaired fitter, reduction products and opaque review archives remain unchanged.

**Close the current POINT starlet experiment; retain the common readout repair.**
No subset selected by noise, fitted amplitude correction, new tolerance, fitter
change, FRUIT run, 129081 access, OOF work or independent replication follows.
'''
    (HERE/'DECISION_AUDIT.md').write_text(text)
    disposition=dict(identity='SCI-FRUIT-POINT-SAVED-DECISION-AUDIT@r0.1',
        recommendation='close_this_starlet_experiment_for_POINT',independent_replication_earned=False,
        defensible_operational_subset=None,scope='tested nominal central starlet candidate, not all wavelet methods',
        repaired_common_readout='preserved',previous_registered_results='preserved',
        starlet_only_usable_successes=3,successes_depending_on_incomplete_fit=2,
        other_complete_success_limited_to_one_noise_realization=True,
        standalone_image_quality_veto=False,changed_tolerances=False,amplitude_correction=False,
        new_experiment_started=False,reserved_observation_opened=False)
    (HERE/'DISPOSITION.json').write_text(json.dumps(disposition,indent=2)+'\n')
    verification=read('VERIFICATION.json');verification['repeat_and_response_cancellation_identities']=identity_checks
    (HERE/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(disposition)
if __name__=='__main__':main()
