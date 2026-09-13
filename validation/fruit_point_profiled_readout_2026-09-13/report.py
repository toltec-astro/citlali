"""Render the bounded readout result and complete numerical tables from saved JSON."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def fmt(x,n=5):return '—' if x is None else f'{x:.{n}g}'
def vec(v,n=5):return '—' if v is None else ', '.join(fmt(float(x),n) for x in v)
def percent(x):return '—' if x is None else f'{100*x:+.4f}'

def main():
    c=read(HERE/'FIT_COMPARISON.json');m=read(HERE/'MEASUREMENTS.json');s=read(HERE/'SUMMARY.json')
    w=read(HERE/'WITNESS_RESULTS.json');r=read(HERE/'RESULTS.json');extra=read(HERE/'SUPPLEMENTAL_DIAGNOSTICS.json')
    done=read(HERE/'COMPLETE.json');verification=read(HERE/'VERIFICATION.json')
    lookup={(q['case'],q['arm'],q['array']):q for q in m}
    lines=['# Complete readout sequence','',
           'P = pixelwise reference; C = starlet. Every row uses saved terminal pass 6. Peak is the continuous free-profile Gaussian peak. Linear means coefficients re-solved at the original free geometry; it is diagnostic only. The truth-template coefficient is a different quantity and is reported separately. No rows are dropped for unavailability.','',
           '| Case / arm / array | Radius | Original peak | Linear peak | Repaired peak | Original error % | Linear error % | Repaired error % | Template coefficient | Template error % |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for q in c:
        o,l,n=q['original'],q['linear'],q['repaired'];t=q.get('truth_template')
        lines.append(f"| {q['case']} / {q['arm']} / {q['array']} | {q['radius']} | {fmt(o['peak'])} | {fmt(None if l is None else l['peak'])} | {fmt(n.get('peak'))} | {percent(o['peak_error'])} | {percent(None if l is None else l['peak_error'])} | {percent(n['peak_error'])} | {fmt(None if t is None else t['amplitude'])} | {percent(None if t is None else t['peak_error'])} |")
    lines += ['', '## Geometry, background and objective','',
        'Centroid and widths are arcsec; widths are sorted FWHM major/minor; angle is the corresponding orientation modulo pi in radians. Plane coefficients multiply (1,x/90,y/90). Re-solving the linear coefficients leaves original geometry exactly fixed. N has no source truth.','',
        '| Case / arm / array | Radius | Readout | Centroid | Widths | Angle | Plane b0,bx,by | Background at fitted center | Centroid error | SSE |',
        '| --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |']
    for q in c:
        for kind in ['original','linear','repaired']:
            v=q[kind]
            if v is None:continue
            lines.append(f"| {q['case']} / {q['arm']} / {q['array']} | {q['radius']} | {kind} | {vec(v['centroid'])} | {vec(v['widths'])} | {fmt(v['angle_rad'])} | {vec(v['background'])} | {fmt(v['background_at_fit_center'])} | {fmt(v['centroid_error'])} | {fmt(v['sse'],9)} |")
    (HERE/'READOUT_TABLES.md').write_text('\n'.join(lines)+'\n')
    lines=['# Historical and new judgments','',
        'These are separate saved-map readout results. Numerical completion is independent of the unchanged data-only rules. S = source evidence; A = associated evidence; C = centroid usable; P = peak usable; shape/support are warnings. The original labels are not overwritten.','',
        '| Case / arm / array | Readout | S/A/C/P | Shape/support | Peak score | Peak domain change % | Centroid domain change arcsec | Limitations |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for q in m:
        for kind in ['original','repaired']:
            v=q[kind];j=v['judgments'];p=v['fixed_inner_domain_probe']
            vals='/'.join('yes' if j[k] else 'no' for k in ['source_evidence_present','associated_source','centroid_usable','peak_response_usable'])
            lines.append(f"| {q['case']} / {q['arm']} / {q['array']} | {kind} | {vals} | {j['shape_warning']}/{j['support_warning']} | {fmt(v['original'].get('empirical_peak_score'))} | {percent(p.get('peak_relative_difference'))} | {fmt(p.get('centroid_difference_arcsec'))} | {', '.join(j['limitations']) or 'none'} |")
    (HERE/'JUDGMENT_TABLES.md').write_text('\n'.join(lines)+'\n')
    lines=['# H/D gain ratios: all readouts and pairings','',
        'Expected H/D = 100/90. Matched and crossed ratios reuse two noise realizations and are not independent additional trials. Usability belongs to the 60-arcsec readout with its inner probe; 52-arcsec ratios and both coefficient-only readouts have no separate operational availability.','',
        '| Arm / array | Radius | H seed / D seed | Pairing | Readout | Ratio | Error % | Raw within 5% | Pair usable |',
        '| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |']
    for q in read(HERE/'RATIOS.json'):
        usable=str(q['full_domain_pair_usable']) if q['usability_applies_to_this_radius'] else 'diagnostic'
        lines.append(f"| {q['arm']} / {q['array']} | {q['radius']} | {q['numerator_seed']} / {q['denominator_seed']} | {q['pairing']} | {q['readout']} | {fmt(q['ratio'],7)} | {percent(q['error'])} | {q['raw_within_5pct']} | {usable} |")
    (HERE/'RATIO_TABLES.md').write_text('\n'.join(lines)+'\n')
    lines=['# Numerical completion and attempted-solution spread','',
        'Every attempt and termination message is preserved in RESULTS.json. Similar cost uses the prospectively declared tolerance, not truth. Signed peaks may be uninterpretable for null/support-limited maps.','',
        '| Problem | Selected start | Finite/rank | First-order complete | Relative projected gradient | Minimum-to-maximum cost | All-start peak span % | All-start centroid diameter | Similar-cost peak span % | Similar-cost centroid diameter | Seconds |',
        '| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    byid={q['id']:q for q in c}
    for q in r:
        f=q['repaired'];a=f['attempts'][f['selected_start']];d=a['diagnostics'];cc=byid[q['id']];all_=cc['all_start_spread'];close=cc['similar_cost_spread']
        lines.append(f"| {q['id']} | {f['selected_start']} | {d['finite']}/{d['rank']} | {d['numerical_complete']} | {fmt(d['relative_scaled_projected_gradient'])} | {fmt(all_['sse_min'],9)}–{fmt(all_['sse_max'],9)} | {percent(all_['relative_peak_span'])} | {fmt(all_['centroid_diameter'])} | {percent(close['relative_peak_span'])} | {fmt(close['centroid_diameter'])} | {fmt(f['seconds'])} |")
    (HERE/'NUMERICAL_TABLES.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.5,8),layout='constrained')
    for ax,arm in zip(axes[0],['P','C']):
        qq=[q for q in c if q['primary'] and q['radius']==60 and q['arm']==arm]
        xx=np.arange(len(qq))
        ax.axhspan(-5,5,color='#e8f0e5');ax.axhline(0,color='0.7',lw=.7)
        for j,q in enumerate(qq):ax.plot([j-.12,j+.12],[q['original']['peak_error']*100,q['repaired']['peak_error']*100],color='0.6',lw=1)
        ax.scatter(xx-.12,[q['original']['peak_error']*100 for q in qq],color='#777777',s=28,label='Original free fit')
        ax.scatter(xx+.12,[q['repaired']['peak_error']*100 for q in qq],color='#176b8f',s=28,label='Profiled free fit')
        ax.set_xticks(xx,[q['case'].split('_')[0]+('1' if q['case'].endswith('11') else '2')+'\n'+q['array'][1:] for q in qq],fontsize=8)
        ax.set(title=('Pixelwise reference' if arm=='P' else 'Starlet candidate')+' — free-profile peak',ylabel='Error against injected peak (%)',ylim=(-20,20))
        ax.legend(loc='lower left',fontsize=8)
    ax=axes[1,0];qq=[q for q in m if q['primary']]
    ax.axhline(1,color='#aa4d32',ls='--',label='Unchanged 1% domain rule')
    ax.scatter(np.arange(24)-.12,[100*q['original']['fixed_inner_domain_probe']['peak_relative_difference'] for q in qq],color='#777777',label='Original')
    ax.scatter(np.arange(24)+.12,[100*q['repaired']['fixed_inner_domain_probe']['peak_relative_difference'] for q in qq],color='#176b8f',label='Profiled')
    ax.set(title='H/D peak sensitivity to 52″ versus 60″ domain',xlabel='All 24 H/D array maps, registered order',ylabel='Absolute peak change (%)');ax.legend(fontsize=8)
    ax=axes[1,1]
    for primary,col,label in [(True,'#176b8f','48 H/D problems'),(False,'#b67538','36 safeguards')]:
        ii=[i for i,q in enumerate(r) if q['primary']==primary]
        yy=[r[i]['repaired']['attempts'][r[i]['repaired']['selected_start']]['diagnostics']['relative_scaled_projected_gradient'] for i in ii]
        ax.scatter(ii,yy,color=col,s=18,label=label)
    ax.axhline(1e-6,color='#aa4d32',ls='--');ax.set_yscale('log')
    ax.set(title='Independent first-order completion: 65/84',xlabel='All 84 map/domain problems',ylabel='Relative projected gradient');ax.legend(fontsize=8)
    fig.suptitle('Saved-map source readout repair — no feedback or cleaning rerun',fontsize=15)
    fig.savefig(HERE/'READOUT_REPAIR.png',dpi=170);plt.close(fig)
    report='''# Source fitter repair: better availability, unchanged absolute peak counts

2026-09-13 · SCI-FRUIT v0.1 development · r0.1

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md),
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md)
and [owner-authorized frozen protocol](PROTOCOL.md). This is a new terminal
saved-map readout result. Every previous packet, map, parameter, label and
registered outcome is preserved. **Starlet remains parked for POINT.**

## Fitter verdict

**The repair clears all six known objective defects without truth-assisted
initialization, but does not fully meet its independent numerical-completion
criterion.** All 84 selected solutions are finite, in bounds, rank four and no
worse in objective than their saved counterparts. The three existing map-derived
starts, model, coefficient constraints, domains and scientific rules were retained.
The new route profiles amplitude and plane coefficients while optimizing free
geometry, with fixed numerical scaling and tolerances declared before execution.

The intermediate 48 linear re-solves make essentially no change: largest
relative peak change is 2.97e-8, plane-coefficient change 6.47e-8, and relative
objective improvement 1.94e-16. None clears a witness. The old linear part was
already solved; the consequential problem lies in geometry selection or its
coupling to the linear parameters.

All six defective saved fits sit at **theta = +pi**, with an outward angle
gradient. Their old solver terminated on objective change. This is specific
evidence consistent with trapping at the angle coordinate boundary, where the
physical Gaussian orientation is periodic. Remaining bound-aware gradients are
also larger than the new diagnostic criterion. The original analytic derivative
passes an independent finite-difference fixture. These observations do not isolate
which of profiling, numerical scaling, termination or path selection was decisive;
this was one repair, not a factorial solver study. No angle bound was changed.

| Known witness | Domain | Original SSE | Linear re-solve SSE | Repaired SSE | Feasible witness SSE |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    for q in w:
        report+=f"| {q['case']} / {q['arm']} / {q['array']} | {q['radius']}″ | {q['original_sse']:.2f} | {q['linear_sse']:.2f} | {q['repaired_sse']:.2f} | {q['witness_sse']:.2f} |\n"
    report+='''
All 252 starts terminated on small objective change, rather than the solver's
gradient condition or evaluation limit. The independent bound-aware criterion
passes **65/84 selected fits: 47/48 H/D and 18/36 safeguards**. The sole H/D miss
is C/H second-seed a2000 at 60″ (1.234e-6 versus the registered 1e-6 criterion).
The other misses are one shifted-source, nine null and eight boundary fits;
none is hidden or rerun. The largest residual criterion is 4.92e-4 in a null
inner fit. Optimizer success is not being promoted to a completion certificate.

Different starts sometimes find substantially different higher-cost solutions:
17 problems have all-start peak spreads above 5% and centroid diameters above 1″.
Among the prospectively defined similar-cost solutions, however, the largest
peak spread is **0.0397%** and centroid diameter **0.00853″**. For H/D alone these
are 0.000396% and 0.0000294″. Those are small compared with the operational
budgets, but are not a global-optimum or unique-parameter certificate. Nearly
circular orientation is never an extra rejection. [All numerical records](NUMERICAL_TABLES.md)
retain the incomplete judgments and every start in the linked JSON.

## POINT interpretation

**Repairing the common readout restores four withheld H/D peaks. It does not
improve the absolute peak-accuracy counts or establish a sufficient candidate
advantage.** Numerical and operational labels below are new results on the
same saved maps; they do not replace the registered historical labels.

At 60″, out of all 12 H/D peaks or six H/D ratios per arm:

| Quantity | Pixelwise original → repaired | Starlet original → repaired |
| --- | ---: | ---: |
| Raw absolute peak error within 5% | 6/12 → 6/12 | 4/12 → 4/12 |
| Usable peaks | 6/12 → 7/12 | 6/12 → 9/12 |
| Usable absolute peaks within 5% | 2/12 → 3/12 | 3/12 → 4/12 |
| Raw matched-noise gain within 5% | 5/6 → 5/6 | 6/6 → 6/6 |
| Usable matched gain within 5% | 2/6 → 3/6 | 1/6 → 4/6 |
| Raw crossed-noise gain within 5% | 3/6 → 3/6 | 2/6 → 3/6 |
| Usable crossed gain within 5% | 1/6 → 2/6 | 1/6 → 2/6 |
| Usable centroid within 1″ | 12/12 → 12/12 | 12/12 → 12/12 |

These usability counts apply the unchanged data-only rules; they are not full
numerical qualification counts. The nine usable C peaks include the one H/D
fit that misses the independent first-order criterion described above.

Linear re-solving at the old geometry leaves every corresponding accuracy count
unchanged. At 52″, repaired absolute counts are P 6/12 and C **4/12**, compared
with original P 6/12 and C 5/12. Thus lower objective does not necessarily improve
truth agreement. Both repaired domains have P 5/6 and C 6/6 raw matched ratios,
and 3/6 crossed ratios in each arm. At 60″ the largest repaired raw absolute
errors remain **15.53% (P)** and **16.69% (C)**; largest matched-ratio errors are
5.96% and 1.47%, and crossed errors 16.18% and 7.90%.

The old truth-template coefficient still gives P 0/12 and C 11/12 within 5%
on both domains. It measures amplitude against the injected profile, not the
free-profile peak of a distorted map, and is neither the target of this repair
nor evidence for whole-source fidelity. The earlier interpretation is narrowed
accordingly. Remaining crossed-noise error could mix map noise, background
coupling and noise-dependent cleaning response; two reused realizations cannot
separate them.

## The four domain-rejection cases

All four inner fits now beat their known feasible witness. Their original peak
domain-sensitivity failures disappear under the **unchanged 1% rule**, and all
four newly pass the existing peak-usability rule:

| Case | Original domain change | Repaired domain change | Old → new peak usable |
| --- | ---: | ---: | --- |
'''
    for q in m:
        if q['primary'] and not q['original']['judgments']['peak_stability_pass']:
            old=q['original']['fixed_inner_domain_probe']['peak_relative_difference']*100
            new=q['repaired']['fixed_inner_domain_probe']['peak_relative_difference']*100
            report+=f"| {q['case']} / {q['arm']} / {q['array']} | {old:.4f}% | {new:.4f}% | no → yes |\n"
    report+='''
Across all H/D states, maximum domain changes fall from 3.240% to 0.0366%
for P and 2.552% to 0.0589% for C. Low peak-score withholding remains separate:
five P peaks and three C peaks remain withheld. Shape warning changes only for
C/H second-seed a1400, from warning to no warning; the new fit changed, not the
warning threshold. H/D maximum centroid error stays near 0.589″ for P and improves
from 0.716″ to 0.581″ for C. The largest C centroid change is 0.1405″; all source
association and support judgments remain data-only.

## Safeguards and cost

The single saved shifted-source state retains all three usable centroids and
peaks in each arm, with maximum centroid error about 0.362″. The null state
reports **zero source evidence, usable centroids or usable peaks** in both arms.
All three near-boundary maps in each arm still carry support warnings and
withhold precision centroids and peaks, even where the raw centroid improves.
No favorable boundary fit removes the support limitation. The retained shape
warnings also remain. These are finite safeguards, not a false-positive rate.

The full normal fitter took **2.085 s for 84 fits**, median 19.1 ms and range
13.6–97.3 ms per map/domain. This includes initialization, all three starts,
analytic derivatives and 4,776 linear solves, of which 84 initialize the plane.
There were 4,692 profiled residual calculations, maximum 189 per start,
zero numerical-difference evaluations in the benchmark, and no work-cap increase.
The separate 48 linear diagnostics took 24.1 ms; applying the saved-evidence
judgments took 66.7 ms. The run including I/O and preservation checks took
5.01 s at 106.4 MiB peak RSS. Verification/report costs are separate.

Archived terminal evaluation for the same maps took 5.66 s, but includes both
fit domains and other evaluator work, with no isolated old-fitter timings.
This is context, not a controlled fitter speedup measurement. The new readout
does not erase the starlet trajectories' recorded runtime failures.

## Verification and stopping point

Independent saved-parameter checks reproduce all 252 attempted models, costs,
gradients and linear ranks, all 84 original costs, and the 48 fixed-geometry
results; no verification refit was used. The largest normalized linear normal
residual is 4.08e-16. Analytic-fixture derivative discrepancies are 4.16e-10 for
the profiled residual and 9.85e-11 for the old full model. The unchanged evaluator
reproduced every one of the 42 historical terminal judgments before the run.

The source/input freeze was committed before execution; the repaired outputs
were saved and hash-frozen before witness and truth scoring. All 281 prior
packet payloads, 1,804 external products, 57 frozen authority payloads and opaque
archive statuses remain unchanged. No PTC, feedback fits, new reductions, new
noise, observations, reserved pointings, thresholds, OOF or production changes.

**Stop here.** The common readout is more trustworthy for this bounded comparison,
with identified numerical-completion limitations. Starlet stays parked. No wider
repair, trajectory rerun, noise campaign or method change is initiated.

[Complete readout sequence, geometry and backgrounds](READOUT_TABLES.md) ·
[Old/new judgments](JUDGMENT_TABLES.md) · [Every gain ratio](RATIO_TABLES.md) ·
[Numerical completion and ambiguity](NUMERICAL_TABLES.md) ·
[Verification](VERIFICATION.json) · [Manifest](RESULT_MANIFEST.json).

![Saved-map readout results](READOUT_REPAIR.png)
'''
    (HERE/'SCIENTIFIC_REPORT.md').write_text(report)
    write(HERE/'DISPOSITION.json',dict(identity='SCI-FRUIT-POINT-PROFILED-READOUT-RESULT@r0.1',
        fitter_verdict='six_known_witnesses_cleared; incomplete independent first-order qualification (65/84)',
        POINT_verdict='four_H_D_peak_availability_rejections_removed; absolute_peak_accuracy_counts_unchanged_at_60arcsec',
        strongest_supported_mechanism='geometry_search_or_coupling; all six old witnesses at angular upper bound; linear coefficients already solved',
        candidate='parked_for_POINT',registered_outcomes='preserved',truth_independent_initialization=True,
        no_production_readiness=True,follow_on_experiment_started=False))
    print('report, complete tables and figure written')
if __name__=='__main__':main()
