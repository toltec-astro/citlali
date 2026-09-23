"""Figures and compact numerical tables for the bounded admission experiment."""
import numpy as np
from point_empirical import coordinates
from point_admission import CASES


def figures(path, array, signal, parameters, header, outputs):
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.lines import Line2D
    x,y=coordinates(signal.shape,header)
    dx=header['CDELT1']*3600;dy=header['CDELT2']*3600
    extent=[x[0,0]-dx/2,x[0,-1]+dx/2,y[0,0]-dy/2,y[-1,0]+dy/2]
    ref=outputs['reference'];brightness=ref['brightness']
    colors=['black','#2ca02c','#d62728','#9467bd']
    fig,axs=plt.subplots(2,3,figsize=(14,9),layout='constrained')
    peak=float(np.nanmax(brightness));wing=max(1.,float(np.nanmedian(np.abs(ref['excluded'])))*3)
    panels=[(brightness,'Source-relative observed map',peak),
            (ref['candidate'],'Reference candidate: threshold 3',peak),
            (ref['excluded'],'Excluded signed brightness: reference',max(30.,peak*.08))]
    for name,title in [('permissive','Added by threshold 2'),('restrictive','Removed by threshold 5'),('scale_stress','Removed by doubled noise scale')]:
        selected=outputs[name]['selected']
        changed=selected & ~ref['selected'] if name=='permissive' else ref['selected'] & ~selected
        panels.append((np.where(ref['valid'],np.where(changed,brightness,0.),np.nan),title,max(30.,peak*.08)))
    for ax,(data,title,limit) in zip(axs.flat,panels):
        im=ax.imshow(data,origin='lower',extent=extent,cmap='RdBu_r',norm=SymLogNorm(linthresh=wing,vmin=-limit,vmax=limit))
        ax.plot(parameters[1],parameters[2],'x',color='#ff9900',ms=7)
        ax.set(title=title,xlim=(parameters[1]+20,parameters[1]-20),ylim=(parameters[2]-20,parameters[2]+20),xlabel='East offset (arcsec)',ylabel='North offset (arcsec)')
        fig.colorbar(im,ax=ax,label='mJy / nominal beam',shrink=.8)
    for color,(name,_,_) in zip(colors,CASES):
        if outputs[name]['selected'].any():axs[0,0].contour(x,y,outputs[name]['selected'].astype(float),levels=[.5],colors=[color],linewidths=1)
    axs[0,0].legend([Line2D([0],[0],color=c) for c in colors],['t=3','t=2','t=5','t=3, scale×2'],fontsize=8)
    fig.suptitle(f'{array} · provisional bright-POINT candidate · measured-centroid zoom\nSigned colors emphasize wings; analysis remains the original 40″ radius disk. No smoothing.',fontsize=12)
    fig.savefig(path,dpi=160);plt.close(fig)


def fmt(value,places=3):
    return 'unavailable' if value is None else f'{value:.{places}f}'


def report(out,summaries,runtime):
    lines=['# POINT 152389: provisional connected-support candidate','',
        'This is a read-only candidate-model experiment. Candidate fit measurements are diagnostics, not new observational fluxes or FRUIT results. The recommendation and scientific assessment are provided separately in ASSESSMENT.md.','',
        'All peak/template quantities are mJy/nominal beam; centroids and FWHM are arcseconds. Background reference D=M−b0 uses each observation/trial’s own descriptive fitted constant. Domain: unchanged 40″ radius about AST center; association: measured centroid, nearest pixel, exact half ties toward increasing index. Eight-connectivity, positive emission only. Invalid/outside-domain pixels remain NaN; valid exclusions are zero.','',
        'Four controls only: threshold 3 reference, threshold 2, threshold 5, threshold 3 with doubled pixel-noise scale. The last is exactly threshold 6 under the original scale, not an independent spatial-model test. Ratios are not calibrated detection significances.','']
    for array,s in summaries.items():
        ref=s['reference']['fit'];p=ref['parameters'];w=ref['description']['measured_fwhm_major_minor_arcsec']
        lines += [f'## {array}','',f'Unmasked source-relative reference: peak {p[0]:.3f}; centroid ({p[1]:.3f}, {p[2]:.3f}); FWHM {w[0]:.3f} × {w[1]:.3f}. Saved background removed: {s["saved_parameters"][6]:.5f}. Fixed-template projection {s["reference"]["fixed_projection"]:.3f}.','',
            '| Case | Area arcsec² (pixels) | Peak | Δ peak % | Centroid east, north | FWHM major × minor | Template excluded % | Added/removed pixels vs reference | Boundary / unsupported contact | Admission / fit / constrained |',
            '|---|---:|---:|---:|---|---|---:|---|---|---|']
        for name,_,_ in CASES:
            r=s['cases'][name];f=r['fit'];success=f['status']=='success';p=f.get('parameters');w=f.get('description',{}).get('measured_fwhm_major_minor_arcsec')
            comparison=r['comparison']
            lines.append(f"| {name} | {r['area_arcsec2']:.0f} ({r['pixels']}) | {fmt(p[0] if success else None)} | {fmt(100*comparison['amplitude_fraction'] if comparison else None,2)} | {f'{p[1]:.3f}, {p[2]:.3f}' if success else 'unavailable'} | {f'{w[0]:.3f} × {w[1]:.3f}' if success else 'unavailable'} | {fmt(100*r['fixed_fraction_excluded'] if r['fixed_fraction_excluded'] is not None else None,2)} | {r['added_vs_reference']} / {r['removed_vs_reference']} | {r['boundary_contact']} / {r['unsupported_contact']} | {r['status']} / {f['status']} / {f.get('constrained','—')} |")
        lines += ['',f"Jointly unsupported analysis pixels: {s['cases']['reference']['unsupported_domain_pixels']}. Extents, anchor ratios, full fits, constraints and failure causes are in [measurements]({out/array/'measurements.json'}).",'',
                  f"![{array} source structure and threshold differences]({out/array/'candidates.png'})",'',
                  '| Case | Retained / 32 | Fit failures or unavailable / constrained | Area median [min,max] arcsec² | Candidate amplitude scatter | Paired Δ amplitude % median [min,max] | Max centroid shift arcsec | Max absolute FWHM change % (major/minor) |',
                  '|---|---:|---|---|---:|---|---:|---|']
        for name,_,_ in CASES:
            r=s['simulation_summary'][name];area=r['area_arcsec2'];d=r['amplitude_fraction_change'];c=r['centroid_distance_change_arcsec']
            width=[]
            for key in ['major_fwhm_fraction_change','minor_fwhm_fraction_change']:
                v=r[key];width.append(fmt(100*max(abs(v['minimum']),abs(v['maximum'])) if v['n'] else None,2))
            interval=lambda v,factor: f"{factor*v['median']:.2f} [{factor*v['minimum']:.2f},{factor*v['maximum']:.2f}]" if v['n'] else 'unavailable'
            lines.append(f"| {name} | {r['source_retained']} | {r['candidate_fit_failures_or_unavailable']} / {r['constrained']} | {interval(area,1)} | {fmt(r['amplitude'].get('scatter'))} | {interval(d,100)} | {fmt(c.get('maximum'))} | {' / '.join(width)} |")
        lines += ['', 'All finite constrained outcomes remain included. Any failure makes the displayed successful-fit distribution partial, with failures counted separately. Trials refit their own raw source centroid/background before admission. The 32 existing deterministic noise fields are shared across all cases; the baseline is the saved clean Gaussian+constant, not the observed noisy map.','']
    lines += [f'Companion wall time {runtime["wall_seconds"]:.3f} s. Exact inputs/code/seeds, timing and hashes accompany the FITS candidates, masks and trial cubes. No completed uncertainty campaign or upstream operation rerun.','',
        'A2000 retains the additive model’s spatial-representativeness limitation. Conditional simulations cannot validate its spatial noise model or the real source’s non-Gaussian structure. Original maps and seven-parameter measurements are unchanged.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
