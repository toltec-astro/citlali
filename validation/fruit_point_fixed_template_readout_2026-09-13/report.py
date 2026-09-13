"""Report all registered readouts; no further fitting."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import HERE,ARRAYS,SEEDS,read,write


def main():
    rows=read(HERE/'READOUTS.json');e=read(HERE/'COMPARISON.json');w=read(HERE/'FIT_OBJECTIVE_WITNESSES.json')
    lines=['# Complete fixed-template readout evidence','', 'Fixed = truth-assisted signed amplitude plus free plane. Free = retained nonlinear Gaussian plus plane. All errors are numerical diagnostics; original 60-domain usability is carried unchanged and is not a fixed-template admission policy. All fits use the fixed terminal pass 6. Backgrounds below are evaluated at the known injected center for a common comparison coordinate.','', '| Arm | Case | Array | Radius | Free error % | Fixed error % | Original peak usable | Free background | Fixed background | Fixed minus free plane (b0,bx,by) | Fixed SSE / free SSE |','| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |']
    for r in rows:
        delta=np.array(r['background'])-r['free_fit']['background']
        lines.append(f"| {r['arm']} | {r['case']} | {r['array']} | {r['radius_arcsec']} | {100*r['free_relative_peak_error']:+.4f} | {100*r['relative_peak_error']:+.4f} | {'yes' if r['original_peak_usable'] else 'no'} | {r['free_background_at_truth_center']:.4f} | {r['fitted_background_at_truth_center']:.4f} | {', '.join(f'{x:+.4f}' for x in delta)} | {r['sse']/r['free_fit']['sse']:.7f} |")
    lines+=['','## All H/D ratios','', 'Four ordered seed pairs per array, domain and readout. Seed 1 = 20260911; seed 2 = 20260912. Crossed pairs are not independent new cases.','', '| Arm | Radius | Array | Readout | H1/D1 error % | H2/D2 error % | H1/D2 error % | H2/D1 error % |','| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |']
    for arm in ['P','C']:
        for radius in [60,52]:
            for array in ARRAYS:
                for kind in ['free','fixed']:
                    rr=[r for r in e['ratios'] if r['arm']==arm and r['radius_arcsec']==radius and r['array']==array and r['readout']==kind]
                    cells=[next(r['relative_error']*100 for r in rr if r['numerator_seed']==i and r['denominator_seed']==j) for i,j in [(SEEDS[0],SEEDS[0]),(SEEDS[1],SEEDS[1]),(SEEDS[0],SEEDS[1]),(SEEDS[1],SEEDS[0])]]
                    lines.append(f"| {arm} | {radius} | {array} | {kind} | "+' | '.join(f'{x:+.4f}' for x in cells)+' |')
    lines+=['','## H and D seed changes','', '| Arm | Radius | Array | State | Free change % | Fixed change % |','| --- | ---: | --- | --- | ---: | ---: |']
    for arm in ['P','C']:
        for radius in [60,52]:
            for array in ARRAYS:
                for state in ['H','D']:
                    rr=[r for r in e['seed_changes'] if r['arm']==arm and r['radius_arcsec']==radius and r['array']==array and r['state']==state]
                    vals={r['readout']:r['relative_change']*100 for r in rr}
                    lines.append(f"| {arm} | {radius} | {array} | {state} | {vals['free']:+.4f} | {vals['fixed']:+.4f} |")
    lines+=['','## 52-domain versus 60-domain change','', '| Arm | State | Seed | Array | Free peak change % | Fixed peak change % | Free background change | Fixed background change |','| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in e['domain_changes']:
        lines.append(f"| {r['arm']} | {r['state']} | {r['seed']} | {r['array']} | {100*r['free_peak_relative_change']:+.4f} | {100*r['fixed_peak_relative_change']:+.4f} | {r['free_background_at_truth_change']:+.4f} | {r['fixed_background_at_truth_change']:+.4f} |")
    (HERE/'READOUT_TABLES.md').write_text('\n'.join(lines)+'\n')
    colors={'P':'#246eaa','C':'#d95b24'}
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for a,array in enumerate(ARRAYS):
        for radius,marker in [(60,'o'),(52,'^')]:
            for arm in ['P','C']:
                for r in rows:
                    if r['array']==array and r['arm']==arm and r['radius_arcsec']==radius:
                        axs[0,a].scatter(100*r['free_relative_peak_error'],100*r['relative_peak_error'],color=colors[arm],marker=marker,s=40,alpha=.8)
                pairs=[r for r in e['ratios'] if r['array']==array and r['arm']==arm and r['radius_arcsec']==radius and r['pairing']=='crossed']
                for i,j in [(SEEDS[0],SEEDS[1]),(SEEDS[1],SEEDS[0])]:
                    pair={r['readout']:r for r in pairs if r['numerator_seed']==i and r['denominator_seed']==j}
                    axs[1,a].scatter(100*pair['free']['relative_error'],100*pair['fixed']['relative_error'],color=colors[arm],marker=marker,s=40,alpha=.8)
        for row,lims in [(0,(-25,20)),(1,(-18,18))]:
            ax=axs[row,a];ax.plot(lims,lims,color='gray',lw=.6,ls='--')
            ax.axhspan(-5,5,color='green',alpha=.08);ax.axvspan(-5,5,color='green',alpha=.05)
            ax.set_xlim(lims);ax.set_ylim(lims);ax.grid(alpha=.2)
            ax.set_xlabel('Retained free-Gaussian error (%)')
        axs[0,a].set_title(array)
    axs[0,0].set_ylabel('Fixed-template absolute peak error (%)')
    axs[1,0].set_ylabel('Fixed-template crossed H/D error (%)')
    fig.suptitle('Same terminal maps, two readouts\nFixed template uses injected shape; all numerical results shown, including originally withheld cases')
    handles=[Line2D([],[],marker='o',ls='',color=colors[a],label=a) for a in ['P','C']]+[Line2D([],[],marker=m,ls='',color='gray',label=f'{radius} arcsec domain') for radius,m in [(60,'o'),(52,'^')]]
    fig.legend(handles=handles,loc='outside lower center',ncol=4)
    fig.savefig(HERE/'READOUT_COMPARISON.png',dpi=160);plt.close(fig)
    report='''# Fixed-template readout result: a mixed map and measurement problem

2026-09-13 · SCI-FRUIT v0.1 development · r0.1

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md),
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md)
and the [authorized protocol](PROTOCOL.md). The owner approved the exact
48-fit saved-map comparison proposed by the earlier peak diagnosis. Starlet
remains **parked for POINT**; all registered outcomes, availability labels and
the failed image-stability qualification remain unchanged.

**The candidate maps contain substantially better recovered amplitude along
the known source shape in these cases, while the common free-Gaussian readout
introduces substantial errors. Some crossed-noise error nevertheless remains
in the maps.** This is a mixed result, with direct evidence of inadequate
numerical solution selection in six of the saved source fits. It does not
qualify a deployable fixed-template method or reverse the parking decision.

## What changed when only the measurement changed

The same 24 terminal H/D array maps were measured with a signed amplitude for
the true injected shape plus a free plane, on both existing domains. The
injected peak was not imposed: amplitudes were solved freely. No PTC, feedback
or free-Gaussian fit was rerun. The old fit is compared on identical pixels.
The following counts describe raw errors, including originally withheld peaks;
they are not new usable-result or qualification counts.

| Radius | Readout | P absolute peaks within 5% | C absolute peaks within 5% | P crossed H/D within 5% | C crossed H/D within 5% |
| --- | --- | ---: | ---: | ---: | ---: |
'''
    for radius in [60,52]:
        for kind in ['free','fixed']:
            def count(arm,quantity):
                r=next(r for r in e['summary'] if r['arm']==arm and r['radius_arcsec']==radius and r['readout']==kind and r['quantity']==quantity)
                return f"{r['within_5pct']}/{r['required']}"
            report+=f"| {radius} arcsec | {'Saved free Gaussian' if kind=='free' else 'Fixed true shape'} | {count('P','absolute_peak')} | {count('C','absolute_peak')} | {count('P','crossed_H_over_D')} | {count('C','crossed_H_over_D')} |\n"
    report+='''
At radius 60, the fixed-template C peak errors span **-3.87% to +5.49%**;
only first-seed H a1400 misses 5%. The saved free-Gaussian C errors span
**-5.23% to +16.69%**. The strong absolute-amplitude improvement survives the
52-arcsec comparison (11/12 within 5%, maximum error 6.27%). It therefore does
not depend on choosing the more favorable of the two domains.

For P, all twelve fixed-template amplitudes fall below truth: **-5.08% to
-22.67%** at radius 60. All its saved free-Gaussian H/D widths are narrower
than the injected widths. Allowing a narrower profile raises the fitted peak
and can make peak agreement look better despite less recovered amplitude
along the known source shape. This supports a profile/response discrepancy in
the reference maps, with nuisance still present; it is not an independent
measurement of the PTC transfer function.

Matched-noise gain remains favorable for C: all six fixed-template H/D ratios
are within 5%, maximum error **2.56%**, on both domains. That is weaker numerical
agreement than the original 0.80% matched result, but still inside the same
5% descriptive comparison. P remains 5/6 matched, maximum 5.45% at radius 60.
All ratios and directions are retained in [the tables](READOUT_TABLES.md).

## What remains in the maps

The fixed template does not eliminate between-noise variation. At radius 60,
C's H1/D2 crossed errors remain **+7.27% in a1400** and **+6.72% in a2000**;
the reverse directions are -3.78% and -4.82%. At radius 52 the corresponding
positive errors are **+8.31%** and **+6.17%**. Both domains therefore retain two
crossed errors above 5%. These depend on source-aligned content in the saved
maps, not solely on the nonlinear free-shape fit.

For C, fixed-template H seed changes are +0.056%, -6.180%, -4.884% across
1100/1400/2000; D changes are +0.125%, -4.393%, -6.229%. P retains still larger
changes in the two longer-wavelength arrays. Matched ratios partly cancel
these seed effects. Two reused realizations cannot establish an uncertainty,
false-alarm rate or population-level bias. This experiment does not separate
residual nuisance from cleaning/feedback transfer effects.

## A concrete numerical problem in the common readout

The fixed-template solution, including its fitted plane, is a feasible member
of the free Gaussian-plus-plane model family: its center, widths, angle and
amplitude satisfy that family's bounds. Yet **six saved free fits have a
higher residual sum of squares than this known feasible solution**. Their
saved costs were independently reproduced on exactly the same pixels.

| Saved fit with a known better feasible solution | Domain | Reduction in saved fit SSE |
| --- | ---: | ---: |
'''
    for q in w['witnesses']:
        report+=f"| {q['arm']} {q['case']} {q['array']} | {q['radius_arcsec']} arcsec | {100*q['relative_improvement']:.4f}% |\n"
    report+='''
The four inner-domain cases are exactly the four distinct H/D states rejected
by the registered domain-sensitivity check. This establishes numerical
solution-selection failures in the common measurement. It is stronger evidence
than simply observing that two different fit domains give different peaks.
It does **not** prove a repaired fitter would make these measurements usable:
that would require actual authorized refitting and qualification, and the
low-score rejections remain a separate issue.

Across all 24 fixed-template domain pairs, the largest amplitude change between
52 and 60 arcsec is **0.954%**, versus **3.240%** for the saved free fits. That
smaller sensitivity uses the known source shape and therefore cannot be used
to relabel the original measurements. Central fitted-plane coefficients and
values at the known source center are recorded for both readouts; the separately
saved outer-annulus plane remains distinct. The remaining cases where the free
fit has lower map residual cost but worse peak truth agreement also show why
least-squares completion alone is not an astronomical accuracy certificate.

## Consequence and limits

The earlier diagnosis is now narrower: the reference loses substantial
known-shape amplitude in these saved cases; the candidate restores it much
better; free-shape readout errors obscure that difference; and some noise
dependence remains even with an exact source template. There is concrete
reason to address the common source fitter before using its domain-sensitivity
failures to judge another feedback change. No fitter repair or new experiment
has been started here.

The fixed template deliberately knows the injected centroid, shape and
orientation. It is an oracle diagnostic ruler, not a POINT recipe or proof of
its usability. Original peak labels, pointing outcomes, null safeguards,
noise-related failures and runtime limits all stand. The original candidate
cost misses are untouched, and no independent pointing or reserved observation
was evaluated. Starlet stays parked; no OOF or production claim follows.

## Verification and provenance

All 48 linear fits are finite and rank four, with design condition numbers
12.14–14.61. Maximum normalized normal-equation residual is 1.87e-16. The
48 saved free-fit costs were reproduced, and each fixed solution was independently
checked as a feasible Gaussian-family model without another fit. Execution took
1.09 seconds overall, including read/preservation work; the 48 linear solves
and their residual checks took 6.86 milliseconds and process peak RSS was
57.75 MiB. This diagnostic timing is not a new FRUIT latency result.

The [source/input freeze](FREEZE.json) was committed at
`e8ea8b74cfe1dbead2d38604f7f2225fe4b186ca` before fitting, from clean parent
`5d0ab7e7d95786fcc02822938452c79910b7e725`. Exactly 48 fits were executed, with
zero verification refits, nonlinear fits, feedback fits, PTC calls, pre-PTC
reads or reserved observations. All 256 prior packet payloads, 1804 prior
external products, 57 frozen authority payloads and opaque archive statuses
remain unchanged. [Verification](VERIFICATION.json),
[objective witnesses](FIT_OBJECTIVE_WITNESSES.json),
[complete numerical comparison](COMPARISON.json),
[figure](READOUT_COMPARISON.png) and [result manifest](RESULT_MANIFEST.json)
preserve every case and original label.
'''
    (HERE/'SCIENTIFIC_REPORT.md').write_text(report)
    write(HERE/'DISPOSITION.json',dict(candidate='parked_for_POINT',registered_outcomes='unchanged',diagnostic='mixed_map_and_readout_effects',
        full_domain_absolute_within_5pct=dict(P_free=6,P_fixed=0,C_free=4,C_fixed=11,required=12),
        full_domain_crossed_within_5pct=dict(P_free=3,P_fixed=4,C_free=2,C_fixed=4,required=6),
        known_feasible_lower_cost_source_fits=6,inner_fit_witnesses_for_H_D_domain_rejections=4,
        interpretation='Candidate has better recovered amplitude along the known source shape; common source-fit numerical failures obscure some outcomes, while crossed-noise map variation persists.',
        operational_policy_changed=False,next_experiment_started=False,production_qualified=False))
if __name__=='__main__':main()
