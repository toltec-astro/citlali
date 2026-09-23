"""Figures and owner-facing tables for the bounded post-FRUIT companion."""
import numpy as np
from point_empirical import coordinates


def sensitivity_plot_values(variants,index):
    available=[v for v in variants if v['status']=='calculated' and v['comparison'].get('scatter_ratio') is not None]
    return [v['comparison']['scatter_ratio'][index] for v in available],[v['name'] for v in available]


def figures(out,array,cycles,models,rows,summary,variants,checks,context):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['M0','T1','T3','T5','T10']; colors=plt.cm.viridis(np.linspace(.05,.95,5))
    h=cycles[0]['header'];x,y=coordinates(cycles[0]['data']['PRIMARY'].shape,h)
    dx=h['CDELT1']*3600;dy=h['CDELT2']*3600
    extent=(x[0,0]-dx/2,x[0,-1]+dx/2,y[0,0]-dy/2,y[-1,0]+dy/2)
    lim=np.quantile(abs(cycles[0]['data']['PRIMARY'][models[0]['background_mask']]),.98)
    fig,axs=plt.subplots(2,5,figsize=(18,7),layout='constrained')
    for j,(rec,m,label) in enumerate(zip(cycles,models,labels)):
        im=axs[0,j].imshow(np.where(m['background_mask'],rec['data']['PRIMARY'],np.nan),origin='lower',extent=extent,cmap='RdBu_r',vmin=-lim,vmax=lim)
        axs[0,j].contour(x,y,m['source_mask'],levels=[.5],colors='k',linewidths=.6)
        axs[0,j].set(title=label,xlabel='East (arcsec)',ylabel='North (arcsec)')
        axs[1,j].hist(m['z'][m['background_mask']],bins=np.linspace(-5,5,51),density=True,histtype='step',color=colors[j])
        z=np.linspace(-5,5,200);axs[1,j].plot(z,np.exp(-z*z/2)/np.sqrt(2*np.pi),'k:',lw=1)
        axs[1,j].set(yscale='log',ylim=(.0005,1),xlabel='Normalized background',title='Histogram; no clipping')
    fig.colorbar(im,ax=axs[0,:],shrink=.6,label='mJy / nominal beam; common stretch')
    fig.suptitle(array+' · identical comparison support and four-FWHM union mask (white excluded)')
    fig.savefig(out/'background-cycles.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,m,label,color in zip(rows,models,labels,colors):
        bins=m['summary']['bins'];mid=[(b['coverage_min']+b['coverage_max_exclusive'])/2 for b in bins]
        axs[0,0].plot(mid,[np.sqrt(b['second_moment_about_global_mean']) for b in bins],'.-',label=label,color=color)
        L=m['empirical'].shape[0]//2;lag=np.arange(-L,L+1)*2
        axs[0,1].plot(lag,m['empirical'][L],color=color,label=label)
        axs[0,2].plot(lag,m['empirical'][:,L],color=color,label=label)
    axs[0,0].set(xlabel='Coverage / accepted samples',ylabel='Second-moment scale (mJy / nominal beam)',title='About one training mean')
    axs[0,0].legend();axs[0,1].set(xlabel='East lag (arcsec)',ylabel='Signed correlation',title='Pair-count corrected')
    axs[0,2].set(xlabel='North lag (arcsec)',ylabel='Signed correlation',title='Pair-count corrected')
    for ax in axs[0,1:]:ax.axhline(0,color='grey',lw=.7);ax.set_ylim(-.25,1.05)
    for j,name in enumerate(['Template','Aperture 1 FWHM','Aperture 1.5 FWHM','Aperture 2 FWHM']):
        val=np.array([r['fixed_background_scatter'][j] for r in rows]);diag=np.array([r['diagonal_scatter'][j] for r in rows]);large=np.array([r['larger_mask_fixed_background_scatter'][j] for r in rows],dtype=float)
        axs[1,0].plot(range(5),val/val[0],'.-',label=name)
        axs[1,1].plot(range(5),val/diag,'.-',label=name)
        axs[1,2].plot(range(5),large/val,'.-',label=name)
    for ax in axs[1]:ax.set_xticks(range(5),labels);ax.axhline(1,color='grey',lw=.7)
    axs[1,0].set(ylabel='Scatter / M0 scatter',title='Fixed-background measurement scale');axs[1,0].legend(fontsize=8)
    axs[1,1].set(ylabel='Correlated / diagonal scatter',title='Effect of signed covariance')
    axs[1,2].set(ylabel='Five / four FWHM mask scatter',title='Larger-mask sensitivity')
    fig.suptitle(array+' · matched linear operators; actual refitted background propagated separately at T10')
    fig.savefig(out/'noise-evolution.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    base=models[-1];im=axs[0,0].imshow(np.where(base['valid'],base['scale'],np.nan),origin='lower',extent=extent)
    fig.colorbar(im,ax=axs[0,0],shrink=.7,label='mJy / nominal beam');axs[0,0].set(title='T10 pixel scale and validity',xlabel='East (arcsec)',ylabel='North (arcsec)')
    axs[0,0].contour(x,y,base['source_mask'],levels=[.5],colors='white',linewidths=.7)
    bins=rows[-1]['applicability']['overlapping_coverage_bins']
    for label,style in [('north','o-'),('south','s-')]:
        rr=[r for r in bins if r['region']==label]
        axs[0,1].plot([np.mean(r['coverage_range']) for r in rr],[r['normalized_second_moment'] for r in rr],style,label=label)
        for ri,r in enumerate(rr):
            cv=np.array(r['correlation']);axs[0,2].plot(np.arange(-8,9)*2,cv[8],alpha=.6,ls='-' if label=='north' else '--',label=label if ri==0 else None)
    axs[0,1].set(xlabel='Coverage; same bin edges',ylabel='Normalized second moment',title='North / south at overlapping depth');axs[0,1].legend()
    axs[0,2].set(xlabel='East lag (arcsec)',ylabel='Normalized correlation',title='Each overlapping coverage bin');axs[0,2].legend()
    ap=rows[-1]['applicability'];nets=sorted(ap['regions']['source_fit']['network_fractions'],key=int)
    for j,label in enumerate(['source_fit','background','north','south']):
        axs[1,0].plot(nets,[ap['regions'][label]['network_fractions'].get(n,0) for n in nets],'.-',label=label)
    axs[1,0].set(xlabel='Network',ylabel='Fraction of accepted contributions',title='Contribution mixture, not noise attribution');axs[1,0].legend(fontsize=8)
    for j,name in enumerate(['peak','template','aperture2']):
        idx=summary['order'].index(name)
        vals,names=sensitivity_plot_values(variants,idx)
        axs[1,1].plot(names,vals,'o-',label=name)
    axs[1,1].axhline(1,color='grey',lw=.7);axs[1,1].set(ylabel='Variant / matched base64 scatter',title='Model sensitivity, not confidence interval');axs[1,1].legend(fontsize=8)
    im=axs[1,2].imshow(summary['correlation'] if summary['correlation'] is not None else np.full((5,5),np.nan),vmin=-1,vmax=1,cmap='RdBu_r');fig.colorbar(im,ax=axs[1,2],shrink=.7)
    axs[1,2].set_xticks(range(5),summary['order'],rotation=40,ha='right');axs[1,2].set_yticks(range(5),summary['order']);axs[1,2].set_title('T10 joint measurement correlation')
    fig.suptitle(array+' · T10 representativeness and coupled source measurements')
    fig.savefig(out/'T10-applicability.png',dpi=150);plt.close(fig)


def report(out,results,runtime):
    lines=['# POINT 152389: post-FRUIT A+B companion','',
        'T10 remains the total map made using m9. Saved m10 remains unapplied. All measurements reproduce their bound references within rtol=atol=1e-10. The inherited a1100 network 3 / scan 5 nonconvergence is unchanged.','',
        'These calculations assume source-masked background covariance transfers to the source despite adaptive feedback. They are conditional additive-noise estimates, not an exact conditional law propagated through a frozen cleaner, full-FRUIT errors, response or calibration estimates. No published measurements, weights or feedback states changed.','',
        'The fit is the unchanged unweighted seven-parameter elliptical Gaussian plus constant in the 40-arcsecond-radius AST-centered disk. Original M0 template and apertures stay fixed. Each trial uses its own fitted constant for the template/apertures; their covariance is retained. Aperture values are map brightness times angular area, with no integrated-flux conversion.','',
        '| Array | Measurement | Observed | Noiseless reference | Additive SD | Relative SD | MC mean shift | Units |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    unavailable=[r for r in results if r.get('status')=='unavailable']
    results=[r for r in results if r.get('status')!='unavailable']
    for r in results:
        for m in r['source']['measurements']:
            fmt=lambda x:'unavailable' if x is None else f'{x:.6g}'
            lines.append('| '+' | '.join([r['array'],m['name'],fmt(m['observed']),fmt(m['noiseless_reference']),fmt(m['scatter']),fmt(m['relative_scatter']),fmt(m['mean_shift']),m['units']])+' |')
    lines+=['','SD uses B−1, not division by sqrt(B). Base256 gives approximately4.43% finite-Monte-Carlo relative precision on SD under Gaussian sampling. Mean shifts are not applied as corrections. Model representativeness and omitted upstream/adaptive/calibration/source-response effects are separate limitations.','',
        '## Matched map comparison','',
        'A fixed common support and four-major-FWHM union source exclusion is used at every selected cycle; five-FWHM union is the larger-mask check. One ordinary background mean is removed for learning, with no clipping, gradient subtraction or Gaussian-source subtraction. Off-source structure is background residual structure; its stochastic interpretation is conditional.','',
        '| Array | Cycle | Mean | Median | Centered SD | Robust scale | Linear template SD | Linear aperture2 SD |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        for row in r['cycle_comparison']:
            if row.get('status')=='unavailable':
                lines.append(f"| {r['array']} | T{row['cycle']} | unavailable: {row['cause']} | — | — | — | — | — |");continue
            b=row['background'];vals=[b[k] for k in ['mean','median','std','robust_scale']]+[row['fixed_background_scatter'][0],row['fixed_background_scatter'][-1]]
            lines.append(f"| {r['array']} | {'M0' if row['cycle']==0 else 'T'+str(row['cycle'])} | "+' | '.join(f'{v:.5g}' for v in vals)+' |')
    lines+=['','Linear template units: mJy/nominal beam; linear aperture2: mJy/nominal beam arcsec². These analytic projections hold the background fixed and are narrower than the true T10 estimators above. Diagonal comparisons, second moments about zero/training mean, coverage bins, pair counts, mask sensitivity and regional correlations are in the machine-readable tables. Same-data cycle differences are not independent realizations or evidence of improved source response.','',
        '## Model checks and applicability','',
        'The model retains signed anisotropic correlation, uses mask-aware pair counts, cosine-product lag taper and nonnegative spectral repair. Unsupported long lags are unknown; computational truncation is not physical absence. Pair counts do not remove finite-window bias from estimating the mean. The common100×100 generator grid covers the entire41×41 measurement stamp. A twice-sized grid embeds identical frequencies, preserving the scientific covariance exactly; it does not learn new long modes.','']
    for r in results:
        a=r['array'];s=r['source'];checks=r['propagation_checks'];root=out/a
        lines += [f'### {a}','',f"256 trials: {s['failures']} failures, {s['bound_limited']} finite bound-limited fits. Pixel-amplitude modulation check: {checks['modulation_second_moment']:.5f} against1 (MC SE {checks['second_moment_MC_SE']:.5f}).",
            '',f"[Comparable backgrounds]({root/'background-cycles.png'}) · [Noise evolution]({root/'noise-evolution.png'}) · [T10 applicability]({root/'T10-applicability.png'})" if (root/'background-cycles.png').exists() else "Cycle comparison figures unavailable; see retained per-cycle diagnostics.",
            '',f"[All source statistics]({root/'source-uncertainty.json'}) · [Trials]({root/'trials.json'}) · [Background cross-covariance]({root/'background-coupling.json'}) · [Analytic/dense/generation checks]({root/'propagation-checks.json'}) · [Sensitivity]({root/'sensitivity.json'})",'',
            '| Variant | Peak SD / matched base64 | Source depth supported | Status |','|---|---:|---|---|']
        for v in r['sensitivity']:
            if v['status']=='calculated':
                ratio=v['comparison'].get('scatter_ratio');text='unavailable' if ratio is None else f'{ratio[0]:.4f}'
                lines.append(f"| {v['name']} | {text} | {v['source_depth_supported']} | calculated; regional transfer remains assumed |")
            else:lines.append(f"| {v['name']} | — | — | {v['cause']} |")
        lines+=['','The base64 prefix is the comparator for each64-trial variant; paired bootstrap intervals in the table describe Monte Carlo precision only. Four variants were prescribed before estimates. Unsupported source depth and regional/mixture transfer are reported separately; no sensitivity range is a confidence interval.','']
    lines += ['## Interpretation boundaries','',
        'Per-array scientific dispositions and the five owner questions are completed in ASSESSMENT.md after inspecting these diagnostics. No automated fit-quality threshold declares observational representativeness. The a2000 coverage/network11/12 comparison records overlap and confounding without claiming causation or independent per-network variances. No mixture-aware alternative was fitted.','',
        'Blank positions are remeasured on each actual saved cycle only when fully supported by the new exclusion. Their raw linear estimator matches the earlier empirical diagnostic; it does not include a bright-source fitted background. Placement overlaps and local depth/predictions are retained.','',
        f"Companion runtime: {runtime['wall_seconds']:.3f}s; four workers by default, numerical-library threads recorded in RUN.json. Inputs verified before and after. No native RTC/PTC/MAP/FRUIT execution. Independent synthetic validation is reported separately. Model arrays, masks, trials, representative FITS realizations, source/parent identities and checksums permit regeneration."]
    for r in unavailable:
        lines += ['',f"{r['array']}: unavailable — {r['cause']}. Other arrays remain independent.", f"[Preserved cycle diagnostics]({out/r['array']/'cycle-comparison.json'})"]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
