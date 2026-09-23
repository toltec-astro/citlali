#!/usr/bin/env python3
"""Report saved POINT trajectory only; never executes feedback or changes candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
from point_fruit_trajectory import verify_manifest, manifest, load, selection_difference, bound, difference
from point_fruit_diagnostics import patch_map
from point_additive_diagnostics import save_json


def report(run, output):
    import matplotlib.pyplot as plt
    from point_empirical import coordinates
    verify_manifest(run)
    output.mkdir(parents=True, exist_ok=False)
    result = json.loads((run/'RUN.json').read_text())
    rows = {}
    for array, outcome in result['arrays'].items():
        dest = output/array
        dest.mkdir()
        histories = {int(k):v for k,v in outcome['histories'].items()}
        indices = sorted(histories)
        measures = [histories[k]['measurement'] for k in indices]
        pars = np.array([m['fit'].get('parameters', [np.nan]*7) for m in measures])
        widths = np.sort(pars[:,3:5], axis=1)[:,::-1]*np.sqrt(8*np.log(2))
        candidates = [load(Path(histories[k]['candidate']['path']))[0] if histories[k].get('candidate') else None for k in indices]
        pixels = [int(c['ADMITTED'].sum()) if c is not None else np.nan for c in candidates]
        fixed = [m['frozen_template_projection'] for m in measures]
        apertures = np.array([[x['value_mJy_per_beam_arcsec2'] for x in m['apertures']] for m in measures])
        # Include T1-M0, which predates the continuation's per-cycle diagnostics.
        if 1 in histories and 'differences' not in histories[1]:
            current, _ = load(Path(histories[1]['total']['path']))
            original, _ = load(Path(histories[0]['total']['path']))
            newpatch, _, _, h = patch_map(Path(histories[1]['total']['path']))
            oldpatch, _, _, _ = patch_map(Path(histories[0]['total']['path']))
            x,y=coordinates(newpatch.shape,h);disk=np.hypot(x,y)<=40
            a0,a1=pars[0,0],pars[1,0]
            histories[1]['differences']={'1':dict(from_cycle=0,fitted_amplitude=a1-a0,
                fitted_amplitude_fraction=(a1-a0)/a0,
                full_map=difference(current['PRIMARY'],original['PRIMARY'],current['SCI_SUPPORT'].astype(bool),a0),
                fixed_disk=difference(newpatch,oldpatch,disk,a0),
                fixed_disk_background_subtracted=difference(newpatch-pars[1,6],oldpatch-pars[0,6],disk,a0),
                selection=selection_difference(candidates[1],candidates[0]))}
        rows[array] = []
        for j,k in enumerate(indices):
            d = histories[k].get('differences', {})
            rows[array].append(dict(cycle=k, amplitude=pars[j,0], frozen_template=fixed[j],
                widths_arcsec=widths[j].tolist(), east_north_arcsec=pars[j,1:3].tolist(), constant=pars[j,6],
                selected_pixels=pixels[j], apertures=apertures[j].tolist(), background=measures[j]['background'],
                differences=d, fit_status=measures[j]['fit']['status'], valid=histories[k].get('ready',True)))
        fig,axes=plt.subplots(3,3,figsize=(15,12),constrained_layout=True)
        ax=axes[0,0];ax.plot(indices,pars[:,0],'o-',label='Gaussian amplitude');ax.plot(indices,fixed,'s-',label='Frozen template');ax.set_ylabel('mJy/nominal beam');ax.legend()
        ax=axes[0,1]
        for n in range(3):ax.plot(indices,apertures[:,n],'o-',label=f'{[1,1.5,2][n]} × original major FWHM')
        ax.set_ylabel('mJy/nominal-beam arcsec²');ax.legend()
        ax=axes[0,2];ax.plot(indices,widths[:,0],'o-',label='Major');ax.plot(indices,widths[:,1],'o-',label='Minor');ax.set_ylabel('FWHM [arcsec]');ax.legend()
        ax=axes[1,0];ax.plot(indices,pars[:,1],'o-',label='East');ax.plot(indices,pars[:,2],'o-',label='North');ax.set_ylabel('Centroid offset [arcsec]');ax.legend()
        ax=axes[1,1];ax.plot(indices,pixels,'o-');ax.set_ylabel('Admitted pixels (4 arcsec² each)')
        ax=axes[1,2];ax.plot(indices,pars[:,6],'o-',label='Fitted constant');ax.plot(indices,[m['background']['median'] for m in measures],'s-',label='Fixed background median');ax.set_ylabel('mJy/nominal beam');ax.legend()
        ax=axes[2,0];ax.plot(indices,[m['background']['std'] for m in measures],'o-',label='Ordinary SD');ax.plot(indices,[m['background']['robust_scale'] for m in measures],'s-',label='Robust scale');ax.set_ylabel('Fixed background scale [mJy/beam]');ax.legend()
        ax=axes[2,1]
        for lag in (1,2):ax.plot(indices[lag:],pars[lag:,0]-pars[:-lag,0],'o-',label=f'A(k) − A(k−{lag})')
        ax.axhline(0,color='k',lw=.5);ax.set_ylabel('Amplitude difference [mJy/beam]');ax.legend()
        ax=axes[2,2]
        for lag in (1,2):
            ks=[k for k in indices if str(lag) in histories[k].get('differences',{})]
            ax.plot(ks,[histories[k]['differences'][str(lag)]['fixed_disk']['rms_over_M0_fitted_amplitude'] for k in ks],'o-',label=f'Map lag {lag}')
        ax.set_ylabel('40″-disk map RMS / M0 amplitude');ax.legend()
        for ax in axes.flat:ax.set_xlabel('Feedback cycle (0 = M0)');ax.set_xticks(indices);ax.grid(alpha=.2)
        fig.suptitle(array+' — fixed-definition trajectory; no post-FRUIT error bars')
        fig.savefig(dest/'trajectory.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for lag in (1,2):
            ks=[k for k in indices if str(lag) in histories[k].get('differences',{})]
            entries=[histories[k]['differences'][str(lag)]['selection'] for k in ks]
            axes[0].plot(ks,[e['added'] for e in entries],'o-',label=f'Added vs k−{lag}')
            axes[1].plot(ks,[e['removed'] for e in entries],'o-',label=f'Removed vs k−{lag}')
            axes[2].plot(ks,[e['retained_brightness_rms_change'] for e in entries],'o-',label=f'Retained vs k−{lag}')
        for ax in axes:ax.set_xlabel('Cycle');ax.legend();ax.grid(alpha=.2)
        axes[0].set_ylabel('Pixels added');axes[1].set_ylabel('Pixels removed');axes[2].set_ylabel('Retained brightness RMS [mJy/nominal beam]')
        fig.suptitle(array+' — replacement support and retained brightness changes');fig.savefig(dest/'selection-changes.png',dpi=150);plt.close(fig)
        shown=[k for k in (0,1,2,3,5,10) if k in histories]
        images={k:patch_map(Path(histories[k]['total']['path']))[0] for k in indices}
        _,_,_,h=patch_map(Path(histories[0]['total']['path']));x,y=coordinates(images[0].shape,h)
        extent=[x[0,0]+1,x[0,-1]-1,y[0,0]-1,y[-1,0]+1]
        vmax=max(float(np.nanmax(v)) for v in images.values())
        fig,axes=plt.subplots(2,len(shown),figsize=(3*len(shown),6),squeeze=False,constrained_layout=True)
        for j,k in enumerate(shown):
            for r,(low,high,cmap) in enumerate([(-30,vmax,'magma'),(-40,40,'RdBu_r')]):
                im=axes[r,j].imshow(images[k],origin='lower',extent=extent,vmin=low,vmax=high,cmap=cmap)
                axes[r,j].set_title('M0' if k==0 else f'T{k}');axes[r,j].set_xlabel('East [arcsec]')
                if j==0:axes[r,j].set_ylabel('North [arcsec]')
            # Each row has one shared, fixed brightness scale.
        for r in range(2):fig.colorbar(axes[r,-1].images[0],ax=list(axes[r,:]),shrink=.8,label='mJy/nominal beam')
        fig.suptitle(array+' — same display range across cycles; lower row shows signed outskirts')
        fig.savefig(dest/'selected-maps.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        for k,m in zip(indices,measures):
            for ax in axes:ax.plot(np.arange(1,40,2),m['profile'],label='M0' if k==0 else f'T{k}',color=plt.cm.viridis(k/10))
        axes[1].set_ylim(-40,40);axes[0].legend(ncol=2)
        for ax in axes:ax.set_xlabel('Radius from original centroid [arcsec]');ax.set_ylabel('Mean brightness [mJy/nominal beam]');ax.axhline(0,color='k',lw=.5)
        fig.suptitle(array+' — fixed-centroid radial profiles and signed bowls');fig.savefig(dest/'profiles.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(2,5,figsize=(15,6),constrained_layout=True)
        for ax,k in zip(axes.flat,range(1,11)):
            if k not in histories:ax.set_visible(False);continue
            c=candidates[indices.index(k)]
            if c is None:
                ax.set_title(f'm{k}: unavailable');ax.axis('off');continue
            ax.imshow(c['ADMITTED'],origin='lower',extent=extent,vmin=0,vmax=1,cmap='Greys');ax.set_title(f'm{k}: {int(c["ADMITTED"].sum())} pixels');ax.set_xlim(20,-20);ax.set_ylim(-25,15)
        fig.suptitle(array+' — replacement support; m10 unapplied');fig.savefig(dest/'candidate-support.png',dpi=150);plt.close(fig)
    save_json(output/'trajectory-table.json',rows)
    save_json(output/'provenance.json',dict(run_manifest=bound(run/'FILES_SHA256SUMS'),scope='saved-product reporting only; no scientific execution'))
    lines=['# POINT 152389 — trajectory through ten total feedback cycles','','New cycles restart from original y, use unity-gain threshold-3 replacement candidates, and relearn the unchanged PTC recipe. M0 and the existing T1 are referenced unchanged. The frozen first-pass selection scale is provisional, not post-FRUIT uncertainty or significance; the a2000 spatial-noise qualification remains.','',
    '## Definitions','','Amplitudes and fitted constants: mJy/nominal beam. Widths and tangent east/north offsets: arcsec, FK5/J2000. Nominal beams remain 5/7.5/10 arcsec. The unchanged seven-parameter Gaussian+constant fit uses the original 40″ AST-centered disk. Fixed template, aperture center/radii and background exclusion remain bound to M0; free fit parameters evolve. Aperture sums retain mJy/nominal-beam arcsec² units, with no integrated-flux conversion. Profiles use 2″ annuli about the original centroid. Background is within the original 40″ disk, outside three original major FWHM.','',
    'Lag differences are current minus k−1 or k−2. Map RMS, signed mean and maximum absolute differences are computed over either the full common science support or the fixed 40″ disk, explicitly named in JSON. RMS normalization divides by the original M0 fitted amplitude. A separate disk metric subtracts each map’s own fitted constant. Candidate differences count added/removed/retained pixels; retained-pixel brightness differences use only the intersection of selected supports. No uncertainty normalization is inferred.','']
    for a,table in rows.items():
        outcome=result['arrays'][a]
        lines += [f'## {a}','',f'Stop: {outcome["stop"]}.','',
                  '| Map | Fit amplitude | Frozen template | Major / minor FWHM | East / north | Constant | Selected pixels |',
                  '|---|---:|---:|---|---|---:|---:|']
        for r in table:
            pair=lambda v:' / '.join(f'{x:.3f}' for x in v)
            lines.append(f'| {"M0" if r["cycle"]==0 else "T"+str(r["cycle"])} | {r["amplitude"]:.3f} | {r['frozen_template'] if r['frozen_template'] is not None else 'unavailable'} | {pair(r["widths_arcsec"])} | {pair(r["east_north_arcsec"])} | {r["constant"]:.3f} | {r["selected_pixels"]} |')
        lines+=['','Figures: '+', '.join(f'[{label}]({output/a/name})' for label,name in [('trajectory','trajectory.png'),('M0/T1/T2/T3/T5/T10','selected-maps.png'),('profiles','profiles.png'),('candidate support','candidate-support.png'),('selection changes','selection-changes.png')])+'.','',f'All cycle products and receipts: [{a}]({run/a}).','']
    lines+=['## Execution and evidence','',f'Whole execution wall time: {result["wall_seconds"]:.3f} s; {result["array_workers"]} array workers, one numerical-library thread each. Per-cycle native/diagnostic wall, native CPU, stage timing and peak memory are preserved in the receipts and RUN files. Shared original inputs are referenced. No RTC/CAL rerun, threshold/rank tuning, uncertainty campaign, cycle 11 or best-map selection.','',f'Numerical trajectory and all fixed-aperture/background/difference values: [trajectory-table.json]({output/"trajectory-table.json"}). Full lineage and guards: [RUN.json]({run/"RUN.json"}).','',
    'T contains the restored feedback model; increased amplitude, source presence or a shallower bowl is not independent evidence of calibrated recovery. The authorized stop is ten cycles completed with m10 unapplied, or the recorded array-specific validity failure.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    manifest(output)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.run,a.output)
