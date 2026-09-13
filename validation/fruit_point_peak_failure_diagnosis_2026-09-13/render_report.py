"""Readable summaries and saved-pass plots; no refitting."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from diagnose import HERE,read,ARRAYS,SEEDS,write


def main():
    traces=read(HERE/'TRACES.json');ratios=read(HERE/'RATIOS.json');changes=read(HERE/'SEED_RESPONSE.json');scales=read(HERE/'SCALING_CHECKS.json')
    lookup={(q['case'],q['arm'],q['array'],q['pass_index']):q for q in traces}
    summary=read(HERE/'SUMMARY.json')
    lines=['# Exact withheld-peak reasons and retained trajectories','','The rule requires a valid associated compact source, an available inner-domain probe with <=1% peak change, and peak / outer MAD >5. Shape warnings alone do not reject these terminal compact peaks. The score is an empirical map-contrast statistic, not a calibrated detection probability.','', 'Seed 1 = 20260911; seed 2 = 20260912. Accuracy columns are independent of availability.','', '| Arm | State | Seed | Array | Exact failed predicate(s) | Score (must exceed 5) | Domain peak change % (must be <=1) | Raw peak error % |','| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    def reason(q):
        return ', '.join(q['withholding_reasons']) or 'usable'
    for q in traces:
        name=q['case'].rsplit('_',1)[0]
        if q['pass_index']==6 and name in ['H','H-shift','D','T'] and q['withholding_reasons']:
            s=1 if q['case'].endswith('20260911') else 2
            lines.append(f"| {q['arm']} | {name} | {s} | {q['array']} | {reason(q)} | {q['peak_score']:.6f} | {100*q['peak_domain_change']:.6f} | {100*q['source_truth_score']['peak_relative_error']:+.4f} |")
    lines+=['','All primary H/D/T terminal rejections have one failed predicate; the shifted-H C a1400 second-seed case fails both score and domain checks. Null, gross-coma, boundary and discovery limitations are retained in TRACES.json for all 714 array-pass records.','', '## All primary H/D/T passes','', 'The fitted central plane and the outer-annulus plane are different saved fits. Below, each background is evaluated at the saved source-fit center. No new fit is performed. Raw centroid jumps and unavailable peaks remain; rows are ordered by case and pass, not by truth error.','', '| Case | Arm | Array | Pass | Wall s | Peak | Major / minor arcsec | Central-fit background | Outer background | Score | Domain change % | Peak status | Shape/support warning |','| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |']
    for q in sorted(traces,key=lambda q:(q['case'],q['arm'],q['array'],q['pass_index'])):
        if q['case'].rsplit('_',1)[0] not in ['H','D','T']:continue
        f=q['fit'];w=f['widths'];j=q['judgments']
        lines.append(f"| {q['case']} | {q['arm']} | {q['array']} | {q['pass_index']} | {q['wall_seconds']:.3f} | {f['peak']:.3f} | {w[0]:.3f} / {w[1]:.3f} | {f['background_at_fitted_center']:.4f} | {q['outer_background_at_fitted_center']:.4f} | {q['peak_score']:.4f} | {100*q['peak_domain_change']:.4f} | {reason(q)} | {j['shape_warning']}/{j['support_warning']} |")
    (HERE/'WITHHOLDING_AND_PASS_TRACES.md').write_text('\n'.join(lines)+'\n')
    lines=['# Matched and crossed saved peak ratios','','Raw relative errors in percent. U=both peaks usable; W=one or both withheld. Pass 6 is fixed. Crossed pairs are dependent recombinations of two saved realizations, not additional independent trials.','', '| Arm | Array | Ratio | Seed 1 / 1 | Seed 2 / 2 | Cross 1 / 2 | Cross 2 / 1 |','| --- | --- | --- | --- | --- | --- | --- |']
    for arm in ['P','C']:
        for a in ARRAYS:
            for family in ['H/D','D/H','T/H']:
                cell=[]
                for i,j in [(SEEDS[0],SEEDS[0]),(SEEDS[1],SEEDS[1]),(SEEDS[0],SEEDS[1]),(SEEDS[1],SEEDS[0])]:
                    r=next(r for r in ratios if r['arm']==arm and r['array']==a and r['family']==family and r['pass_index']==6 and r['numerator_seed']==i and r['denominator_seed']==j)
                    cell.append(f"{100*r['relative_error']:+.4f} ({'U' if r['available'] else 'W'})")
                lines.append(f"| {arm} | {a} | {family} | "+' | '.join(cell)+' |')
    lines+=['','## How common seed response cancels in matched H/D','', 'Let h=H2/H1 and d=D2/D1. The ratio of matched gains is h/d; H1/D2 divided by H1/D1 is 1/d. These are exact identities of the saved peak readouts, not a fitted noise model.','', '| Arm | Array | H seed change % | D seed change % | Matched gain change % | H width-area change % | D width-area change % | H fitted background change | D fitted background change |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in changes:
        if s['pass_index']!=6:continue
        h,d=s['H'],s['D']
        lines.append(f"| {s['arm']} | {s['array']} | {100*(h['peak_ratio']-1):+.4f} | {100*(d['peak_ratio']-1):+.4f} | {100*(s['matched_gain_ratio_seed2_over_seed1']-1):+.4f} | {100*(h['width_product_ratio']-1):+.4f} | {100*(d['width_product_ratio']-1):+.4f} | {h['center_background_difference']:+.4f} | {d['center_background_difference']:+.4f} |")
    (HERE/'MATCHED_AND_CROSSED_RATIOS.md').write_text('\n'.join(lines)+'\n')
    # All H/D passes, every array/arm/seed. No choice of favorable iteration.
    fig,axs=plt.subplots(6,3,figsize=(14,18),layout='constrained')
    colors={'P':'#176b9e','C':'#d65a23'}
    for a,name in enumerate(ARRAYS):
        for arm in ['P','C']:
            for seed,marker in zip(SEEDS,['o','^']):
                for state,ls,truth in [('H','-',100.),('D','--',90.)]:
                    qq=[lookup[f'{state}_{seed}',arm,name,k] for k in range(7)]
                    values=[[100*(q['fit']['peak']/truth-1) for q in qq],
                        [q['fit']['widths'][0] for q in qq],[q['fit']['widths'][1] for q in qq],
                        [q['fit']['background_at_fitted_center'] for q in qq],
                        [q['peak_score'] for q in qq],[100*q['peak_domain_change'] for q in qq]]
                    for row,v in enumerate(values):
                        ax=axs[row,a];ax.plot(range(7),v,color=colors[arm],ls=ls,alpha=.65,lw=.9)
                        for k,q in enumerate(qq):ax.scatter(k,v[k],s=18,marker=marker,edgecolors=colors[arm],facecolors='white' if q['withholding_reasons'] else colors[arm],linewidths=.8)
        for row in range(6):axs[row,a].grid(alpha=.15);axs[row,a].set_xticks(range(7))
        axs[0,a].set_title(name);axs[-1,a].set_xlabel('Retained pass (terminal = 6)')
        axs[0,a].axhspan(-5,5,color='green',alpha=.08)
        axs[4,a].axhline(5,color='black',lw=.7)
        axs[5,a].axhline(1,color='black',lw=.7)
        axs[5,a].set_yscale('symlog',linthresh=1)
    for row,label in enumerate(['Raw peak error (%)','Major FWHM (arcsec)','Minor FWHM (arcsec)','Central-fit background\nat fitted centroid','Peak / outer MAD','Domain peak change (%)\nlog scale above 1%']):axs[row,0].set_ylabel(label)
    handles=[Line2D([],[],color=colors[a],label=a) for a in ['P','C']]+[Line2D([],[],color='gray',ls=ls,label=label) for ls,label in [('-','H'),('--','D')]]+[Line2D([],[],marker=m,color='gray',ls='',label=f'seed {i}') for m,i in [('o',1),('^',2)]]+[Line2D([],[],marker='o',color='gray',ls='',markerfacecolor='white',label='withheld peak')]
    fig.legend(handles=handles,loc='outside lower center',ncol=7)
    fig.suptitle('Saved H/D trajectories — numerical measurements and availability kept separate\nAll arrays, both arms, both seeds, all seven passes')
    fig.savefig(HERE/'SAVED_PEAK_TRACES.png',dpi=130);plt.close(fig)
    write(HERE/'DIAGNOSIS_DISPOSITION.json',dict(candidate='parked_for_POINT',registered_outcomes='unchanged',
        strongest_supported_effect='Same-seed peak shifts partly cancel in matched ratios; the accuracy does not carry over to crossed-noise pairs.',
        exact_withholding='Primary P: six score and two domain failures. Primary C: four score and three domain failures. No terminal primary shape-only veto or unfinished solve.',
        demonstrated_readout_effect='P T/H map scaling agrees near 1e-13 while saved Gaussian-plus-plane fits can differ in peak, width and attainable fit objective; first-seed terminal score crosses five.',
        unresolved='The dominant contribution to between-noise peak errors: source/nuisance structure in reconstructed total maps versus amplification by the free-shape Gaussian-plus-plane readout.',
        next_experiment='One saved-map controlled readout comparison: fixed truth-shaped unit-peak source template plus free plane versus the retained free Gaussian fit, in both existing domains, on the 24 terminal H/D array maps. Diagnostic only; 48 linear solves, no feedback/PTC, no availability-policy change.',
        next_experiment_executed=False))

if __name__=='__main__':main()
