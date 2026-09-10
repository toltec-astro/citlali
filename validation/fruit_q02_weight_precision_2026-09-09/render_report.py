#!/usr/bin/env python3
"""Render the fixed approved metrics from saved results; never opens raw inputs."""
import os
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']='/private/tmp/fruit-q02-precision-mpl-20260909'
os.environ['XDG_CACHE_HOME']='/private/tmp/fruit-q02-precision-cache-20260909'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RESULT=HERE/'attempt_01'
OUT=HERE/'report'
OUT.mkdir(exist_ok=True)
record=json.loads((RESULT/'SUMMARY.json').read_text())
rows=record['windows']
arrays=['a1100','a1400','a2000']
fmt=lambda x: 'unavailable' if x is None else f'{100*x:.1f}%'
text=['# Full predeclared result tables', '',
      'Derived only from attempt_01 saved products. Every original K/tau setting is retained.',
      'Fractions and quantiles are descriptive; there is no certified precision pass.',
      'Percentiles below use evaluation-occurrence counts unless explicitly labeled otherwise.', '',
      '## All three windows for each observation and array', '']
for obs in [123424,152389]:
    for array in arrays:
        rr=sorted((s for s in rows if s['obsnum']==obs and s['array']==array),key=lambda s:s['K'])
        text += [f'### {obs} {array}', '',
                 '| Chunks | Required groups | Missing weights (groups) | Missing weights (evaluation share) | Central evaluation share missing weights | Normalized precision |',
                 '| --- | ---: | ---: | ---: | ---: | --- |']
        for s in rr:
            miss=s['weight_unavailable']
            text.append(f"| {s['K']} | {s['required_groups']:,} | {miss['groups']:,} | {100*miss['evaluation_share']:.5f}% | {100*miss['central_evaluation_share']:.5f}% | {s['normalized_uncertainty_state']} |")
        text += ['', '| Chunks | Tau (s) | Normalized p10 / p50 / p90 | Groups with p / all groups | At/below 20% among available / all evaluation occurrences | Unknown evaluation share | Raw inverse-scatter p50 |',
                 '| --- | ---: | --- | --- | --- | --- | --- |']
        for s in rr:
            for p in s['precision']:
                n=p['normalized']; raw=p['unnormalized']
                q=' / '.join(fmt(x) for x in n['occurrence_weighted_quantiles_10_50_90'])
                fractions=fmt(n['below_goal_fraction_of_available_occurrences'])+' / '+fmt(n['below_goal_fraction_of_all_occurrences'])
                text.append(f"| {s['K']} | {p['tau_seconds']} | {q} | {n['available_groups']:,} / {n['required_groups']:,} | {fractions} | {fmt(n['unknown_fraction_of_all_occurrences'])} | {fmt(raw['occurrence_weighted_quantiles_10_50_90'][1])} |")
        text += ['', 'A zero fraction among all occurrences when everything is unknown means zero known-below-target entries; it is not a measured failure fraction.', '']
text += ['## Four-chunk stability and coefficient distribution', '',
         'The child-scatter ratio is maximum/minimum over the four original chunks. Its percentiles use only groups with all four child scatters available; their support is shown explicitly.', '',
         '| Observation / array | Groups with child ratio / required | Evaluation share with child ratio | Child ratio p50 / p90 | Mean-change contribution p50 / p90 | Gamma p10 / p50 / p90 |',
         '| --- | --- | --- | --- | --- | --- |']
from run_weight_precision import quantiles,clean
extra=[]
for obs in [123424,152389]:
    with np.load(RESULT/f'{obs}_K4_groups.npz',allow_pickle=False) as d:
        for a,array in enumerate(arrays):
            s=next(s for s in rows if s['obsnum']==obs and s['array']==array and s['K']==4)
            required=(d['e']>0)&(d['array'][None,:]==a)
            available=required&np.isfinite(d['child_scatter_ratio'])
            support=float(d['e'][available].sum()/d['e'][required].sum())
            gamma_q=quantiles(d['gamma'][required],d['e'][required])
            ratio=s['child_scatter_ratio_quantiles_10_50_90']; between=s['mean_change_fraction_quantiles_10_50_90']
            gtext=' / '.join('unavailable' if x is None else f'{x:.3f}' for x in gamma_q)
            text.append(f"| {obs} {array} | {available.sum():,} / {required.sum():,} | {100*support:.2f}% | {ratio[1]:.2f} / {ratio[2]:.2f} | {fmt(between[1])} / {fmt(between[2])} | {gtext} |")
            extra.append(dict(obsnum=obs,array=array,child_ratio_available_groups=int(available.sum()),
                         child_ratio_available_evaluation_share=support,gamma_quantiles=gamma_q,
                         zero_normalized_precision_entries=int(np.sum(required[:,:,None]&np.isfinite(d['p_normalized'])&(d['p_normalized']==0)))))
text += ['', 'Gamma is the candidate normalized gridding coefficient, not inverse noise covariance. Relative scatter differences do not by themselves establish map benefit.', '',
         '## Access and cost', '',
         'Both original discovery hashes were rechecked unchanged after the run. No additional observation was opened. Evaluation signal access was limited to finite/mask predicates; no evaluation moment or map was computed.', '',
         'All flag-zero, finite-coordinate candidate occurrences had finite signal in these exports. The nonfinite 152389 values were already excluded by flags; signal finiteness added no further exclusion. Raw nonfinite counters include inactive/flagged slots and must not be treated as extra losses.', '',
         'See attempt_01/COMPLETION.json, RUN_START.json and PRODUCT_MANIFEST.json for exact environment, resource use, settings and identities.']
(OUT/'FULL_RESULT_TABLES.md').write_text('\n'.join(text)+'\n')
(OUT/'DERIVED_TABLE_COUNTS.json').write_text(json.dumps(clean(extra),indent=2)+'\n')

fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
for row,obs in enumerate([123424,152389]):
    for col,array in enumerate(arrays):
        ax=axes[row,col]
        s=next(s for s in rows if s['obsnum']==obs and s['array']==array and s['K']==4)
        x=[p['tau_seconds'] for p in s['precision']]
        q=np.array([p['normalized']['occurrence_weighted_quantiles_10_50_90'] for p in s['precision']],dtype=float)*100
        if np.isfinite(q).any():
            ax.fill_between(x,q[:,0],q[:,2],color='#88b7cb',alpha=.35,label='10th–90th percentiles')
            ax.plot(x,q[:,1],'o-',color='#1b657e',label='Median')
            ax.plot(x,q[:,2],':',color='#1b657e',lw=1)
        else:
            ax.text(.5,.52,'Unavailable\nOne required group has no training',ha='center',va='center',transform=ax.transAxes,fontsize=11)
        ax.axhline(20,color='#b45a28',ls='--',lw=1,label='Provisional 20% goal')
        ax.set(title=f'{obs} · {array}',xscale='log',xticks=[.125,.25,.5,1,2,4],ylim=(0,60))
        ax.set_xticklabels(['.125','.25','.5','1','2','4'])
        ax.grid(alpha=.15)
        if row==1:ax.set_xlabel('Correlation span (s)')
        if col==0:ax.set_ylabel('Estimated fractional SD (%)')
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle('Four-chunk windows: normalized relative-weight precision\nOccurrence-weighted distributions; sensitivity across assumptions, not confidence intervals',fontsize=12)
fig.tight_layout()
fig.savefig(OUT/'normalized_precision_K4.png',dpi=180)
plt.close(fig)
print('Report tables and normalized-precision figure rendered from saved products only.')
