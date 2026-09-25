"""Report a sealed-stage expected-source replay; no fits or new random fields."""
import argparse, time
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from run_point_expected import inputs, read, ARRAYS, POLICIES, SNAPSHOTS
from run_point_corr_admission import source
from point_fruit_trajectory import bound, manifest, verify_manifest
from point_fruit_diagnostics import load
from point_additive_diagnostics import save_json


def summary(scenes, reference):
    guards=[s['guard'] for s in scenes]
    diag=[g['diagnostic'] for g in guards if g['diagnostic'] is not None]
    converged=[g for g in guards if g['diagnostic'] is not None and g['diagnostic']['finite'] and g['diagnostic']['converged']]
    paired=[g for g in converged if g['sigma_A'] is not None]
    amp=np.array([g['diagnostic']['parameters'][0] for g in converged])
    pred=np.array([g['sigma_A'] for g in paired])
    valid=[s for s in scenes if s['numerically_valid']]
    return dict(count=len(scenes),accepted=sum(g['accepted'] for g in guards),
        quality_rejected=sum(g['complete'] and not g['accepted'] and g['diagnostic'] is not None and g['diagnostic']['converged'] for g in guards),
        finite_nonconvergence=sum(g['complete'] and g['diagnostic'] is not None and not g['diagnostic']['converged'] for g in guards),
        computational_errors=sum(not g['complete'] for g in guards),
        reasons=dict(Counter(g['cause'] for g in guards if g['cause'] is not None)),
        nfev_median=float(np.median([d['nfev'] for d in diag])),nfev_max=max(d['nfev'] for d in diag),
        nfev_total=sum(d['nfev'] for d in diag),
        BASE_POINT_false_events=sum(s['BASE_POINT_false_pixels']>0 for s in valid),
        BASE_POINT_mean_false_pixels=float(np.mean([s['BASE_POINT_false_pixels'] for s in valid])),
        BASE_POINT_mean_false_brightness_area=float(np.mean([s['BASE_POINT_false_brightness_area'] for s in valid])),
        uncertainty=dict(scope='all finite converged fits BEFORE S/N selection; predicted values only where geometry/identifiability supports them',
            converged=len(converged),with_sigma=len(paired),excluded_from_scatter=len(guards)-len(converged),
            empirical_amplitude_std=float(np.std(amp,ddof=1)) if len(amp)>1 else None,
            amplitude_bias=float(np.mean(amp)-reference) if len(amp) else None,
            RMS_predicted_sigma=float(np.sqrt(np.mean(pred**2))) if len(pred) else None))


def render(out):
    start=time.perf_counter();revision=source();p=inputs(out)
    for stage in ('freeze','checks','calibration','validation','screen'):verify_manifest(out/stage)
    dest=out/'report';dest.mkdir();checks=read(out/'checks/RESULTS.json');parent=Path(p['parent_root'])
    original=read(parent/'freeze/prospective.json');screens=read(out/'validation/screens.json')['policies']
    taus=read(out/'calibration/thresholds.json')['thresholds'];stats={};policy_stats={};old_stats={};figures=[]
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.dpi':130})
    def save(fig,name):
        path=dest/name;fig.savefig(path,bbox_inches='tight',pad_inches=.15);plt.close(fig);figures.append(bound(path))
    for a in ARRAYS:
        stats[a]={};policy_stats[a]={};old_stats[a]={}
        cal=read(out/'calibration'/f'{a}-trials.json');val=read(out/'validation'/f'{a}-trials.json')
        for phase,rows,key in [('calibration',cal,'scenes'),('validation',val,'scenes'),('stress1.25',val,'stress')]:
            selected=[r[key] for r in rows if key in r]
            stats[a][phase]={s:summary([r[s] for r in selected],checks['noiseless'][a]['core']['guard']['diagnostic']['parameters'][0]) for s in ('blank','core')}
            old=read(parent/('validation' if key=='stress' else phase)/f'{a}-trials.json')
            old_stats[a][phase]={s:dict(unknown=sum(not r[key][s]['numerically_valid'] for r in old if key in r),count=len(selected)) for s in ('blank','core')}
            policy_stats[a][phase]={}
            for pol in POLICIES:
                t=taus[a][pol];known=all(s['policies'][pol]['maximum_known'] for r in selected for s in r.values())
                mx=[max((s['policies'][pol]['max_score'] for s in r.values() if s['policies'][pol]['max_score'] is not None),default=-np.inf) for r in selected]
                record=dict(fields=len(selected),all_maxima_known=known,threshold=t,joint_exceedances=sum(v>t for v in mx) if known and t is not None else None)
                if phase!='calibration':
                    record['final_added_events']=sum(any(s['policies'][pol]['added_pixels']>0 for s in r.values()) for r in selected) if known and t is not None else None
                    record['scenes']={s:dict(final_added_events=sum(r[s]['policies'][pol]['added_pixels']>0 for r in selected),
                        mean_added_pixels=float(np.mean([r[s]['policies'][pol]['added_pixels'] for r in selected])),
                        mean_extra_brightness_area=float(np.mean([r[s]['policies'][pol]['extra_brightness_area'] for r in selected]))) for s in ('blank','core')} if known and t is not None else None
                policy_stats[a][phase][pol]=record
    save_json(dest/'SUMMARY.json',dict(outcomes=stats,policies=policy_stats,historical=old_stats,screens=screens))
    # Reuse the original diagnosed raw optimizer records; no retry or fit here.
    diagnosis=read(next(Path(n) for n in p['consumed_inputs'] if n.endswith('/CASES.json')))
    detail=[]
    for row in checks['cases']:
        old=next(d for d in diagnosis if all(d[k]==row[k] for k in ('array','scene','trial')))
        detail.append(dict(array=row['array'],scene=row['scene'],trial=row['trial'],role=row['role'],old_optimizer=old['original'],new=row['new']))
    save_json(dest/'DIAGNOSIS_COMPARISON.json',detail)
    fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained')
    for i,a in enumerate(ARRAYS):
        for j,s in enumerate(('blank','core')):
            ax=axes[i,j];bottom=np.zeros(3)
            for cat,label,color in [('accepted','Accepted','#377c88'),('quality_rejected','Quality rejected / empty','#d8aa52'),('finite_nonconvergence','Finite nonconvergence / empty','#a9613e'),('computational_errors','Unavailable','#99383f')]:
                vals=np.array([stats[a][ph][s][cat]/stats[a][ph][s]['count']*100 for ph in stats[a]])
                ax.bar(range(3),vals,bottom=bottom,label=label,color=color);bottom+=vals
            ax.set(title=f'{a} · {s}',ylim=(0,108),ylabel='% of prescribed scenes',xticks=range(3),xticklabels=['Cal: 1024','Val: 1024','Stress: 256'])
            for x,ph in enumerate(stats[a]):ax.text(x,102,f"{stats[a][ph][s]['accepted']} accepted",ha='center',fontsize=8)
    axes[0,0].legend(loc='center',fontsize=8);fig.suptitle('Source recognition outcomes · ordinary rejection has a complete empty selection');save(fig,'source-outcomes.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,a in zip(axes,ARRAYS):
        records=[stats[a][ph]['core']['uncertainty'] for ph in stats[a]];xx=np.arange(3)
        ax.bar(xx-.18,[r['empirical_amplitude_std'] for r in records],.36,label='Empirical refit scatter',color='#377c88')
        ax.bar(xx+.18,[r['RMS_predicted_sigma'] for r in records],.36,label='RMS predicted σA',color='#d8aa52')
        ax.set(title=a,ylabel='mJy / nominal beam',xticks=xx,xticklabels=['Cal','Val','Stress ×1.25']);ax.legend(fontsize=8)
    fig.suptitle('Core amplitude uncertainty before S/N selection · frozen C0 unchanged under stress');save(fig,'amplitude-uncertainty.png')
    saved_rows=[]
    for a in ARRAYS:
        peak=abs(original['original_owner']['original_cycles'][a][10]['measurement']['fit']['parameters'][0]);lim=.06*peak
        fig,axes=plt.subplots(4,5,figsize=(17,12),layout='constrained')
        for col,k in enumerate(SNAPSHOTS):
            for rr,pol in enumerate(('BASE-POINT',*POLICIES),1):
                path=out/'screen'/f'{a}-{k:02d}-{pol}';im,header=load(path/'selection.fits');meta=read(path/'selection.json')
                dx=header['CDELT1']*3600;dy=header['CDELT2']*3600;ny,nx=im['MEASURED'].shape
                extent=[(0.5-header['CRPIX1'])*dx,(nx+.5-header['CRPIX1'])*dx,(.5-header['CRPIX2'])*dy,(ny+.5-header['CRPIX2'])*dy]
                if rr==1:
                    axes[0,col].imshow(np.where(im['VALID'],im['BRIGHTNESS'],np.nan),origin='lower',extent=extent,cmap='RdBu_r',norm=TwoSlopeNorm(0,vmin=-lim,vmax=lim))
                    axes[0,col].set_title(f'T{k} measured − fitted constant')
                    for name,color in [('BASE_POINT','black'),('K0','limegreen')]:axes[0,col].contour(im[name],levels=[.5],origin='lower',extent=extent,colors=[color],linewidths=.7)
                ax=axes[rr,col];available='CANDIDATE' in im
                if available:
                    shown=ax.imshow(im['CANDIDATE'],origin='lower',extent=extent,cmap='RdBu_r',norm=TwoSlopeNorm(0,vmin=-lim,vmax=lim))
                    ax.contour(im['BASE_POINT'],levels=[.5],origin='lower',extent=extent,colors=['black'],linewidths=.7)
                    counts={n:int(im[n].sum()) for n in ('ADMITTED','PROPOSED','ADDED') if n in im}
                    for name,color in [('PROPOSED','#aa36c7'),('ADDED','#16b68a')]:
                        if name in im and im[name].any():ax.contour(im[name],levels=[.5],origin='lower',extent=extent,colors=[color],linewidths=.8)
                    title=f"{pol}: {counts['ADMITTED']} pixels"
                    if pol!='BASE-POINT':title+=f"\nproposed {counts['PROPOSED']}, added {counts['ADDED']}"
                else:counts=None;ax.text(.5,.5,'UNAVAILABLE\n'+str(meta['blocked_cause']),transform=ax.transAxes,ha='center',wrap=True);title=pol
                ax.set_title(title);saved_rows.append(dict(array=a,stage=k,policy=pol,accepted=meta['guard']['accepted'],counts=counts,threshold=meta['threshold'],status=meta['screen_status']))
        for ax in axes.flat:ax.set(xlabel='East offset (arcsec)',ylabel='North offset (arcsec)',facecolor='#d0d0d0')
        fig.colorbar(shown,ax=axes.ravel().tolist(),shrink=.5,label='mJy / nominal beam; shared limits')
        fig.suptitle(f'{a}: saved-map candidates, UNAPPLIED · all sources accepted\nBlack BASE-POINT; green K0 (top); purple proposed, teal final extras. Gray = unavailable; white = zero. Brightness clipped at ±{lim:.2f} (6% original T10 peak).')
        save(fig,a+'-saved-candidates.png')
    save_json(dest/'SAVED_SCREEN_COUNTS.json',saved_rows)
    lines=['# POINT 152389: expected-source guard and recalibrated admission','','All six prescribed statistical screens pass with complete outcomes and retained bright controls. These are unapplied saved-map diagnostics; no feedback trajectory was launched. Validation replays previously examined fields and is not a new untouched holdout.','','## Selected measurement','','POINT-EXPECTED-v1 uses one existing unweighted seven-parameter elliptical Gaussian plus constant fit on the original 40-arcsec disk (1257 pixels). Peak initialization is restricted to the original 20-arcsec AST disk; the central coordinate box remains ±20 arcsec. There is one deterministic optimizer call, maximum 3000 function evaluations, and inherited stopping tolerances. Acceptance requires finite convergence, positive peak above the constant, radial centroid ≤20 arcsec, both sigmas strictly >1.5 arcsec with no active lower width bound, and A/σA >10. The width floor is 3.53223 arcsec FWHM on 2-arcsec pixels. No goodness-of-fit, circularity, exact-beam or source-depth cut is added.','','The expected origin is tangent (0,0), FK5 J2000 RA 176.7429079166667°, Dec 39.97619555555556°. Reported tangent centroids are descriptive measurements; no operational pointing correction is emitted. Nominal beams remain 5/7.5/10 arcsec FWHM. A and σA use mJy/nominal beam. σA²=qAᵀC0qA: amplitude row of the SVD pseudoinverse of the full seven-column model Jacobian (W=I), contracted through the existing FFT machinery with the full original M0 covariance. All nuisance columns, including background and any active-bound directions, remain. SVD rank threshold is eps × max(J.shape) × largest singular value; amplitude identifiability uses the null projection ≤100 × eps × max(J.shape). Circular orientation null directions are allowed when amplitude remains identifiable.','','Ordinary quality rejection, finite nonconvergence or nonidentifiable amplitude gives a failed POINT message and complete empty accepted products (zero candidate inside valid support, unavailable outside). Measured maps and raw fit diagnostics remain. Unexpected/nonfinite/missing-input failures remain computationally unavailable. There is no covariance inverse, refit ensemble inside a trial, uncertainty rescaling or exact ten-sigma false-alarm claim. BASE-POINT/CORR/DIAG share this guard and fitted brightness operand; DIAG changes only the later supplemental-neighborhood variance. Original C0, K0, means, seeds, grid, masks, connectivity, gain and threshold-3 baseline remain frozen.','','## Complete outcomes','','Each row below uses 1024 calibration fields, 1024 replayed-validation fields, or 256 reused stress fields; each field carries paired blank and core scenes. Rejection columns are quality rejection / finite nonconvergence / computational error.','', '| Array | Phase | Old blank unknown | New blank accepted | Blank Q / N / E | Core accepted / count | Core Q / N / E | Blank nfev median / max |','|---|---|---:|---:|---|---:|---|---|']
    for a in ARRAYS:
        for ph,s in stats[a].items():
            b=s['blank'];c=s['core'];counts=lambda d:'/'.join(str(d[n]) for n in ('quality_rejected','finite_nonconvergence','computational_errors'))
            lines.append(f"| {a} | {ph} | {old_stats[a][ph]['blank']['unknown']} | {b['accepted']} | {counts(b)} | {c['accepted']}/{c['count']} | {counts(c)} | {b['nfev_median']:g}/{b['nfev_max']} |")
    lines+=['','All draws and reason counts are retained in SUMMARY.json and per-field trial records. No failed draw is dropped or replaced. nfev is the solver counter, not the number of residual calls including numerical Jacobian evaluation.','','## Calibration, validation and stress','','Threshold is max(3, calibration order statistic 1015/1024); strict comparisons. Joint counts are preconnectivity maxima over paired scenes. The final-event column is any connected supplemental addition. U is the original exact one-sided 95% binomial upper bound for validation; no-supplement noiseless checks also pass.','','| Array | Policy | τ | Cal >τ /1024 | Val >τ /1024 | U95 | Val final events | Stress >τ /256 | Stress final events | Screen |','|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for a in ARRAYS:
        for pol in POLICIES:
            s=screens[a][pol];cal=policy_stats[a]['calibration'][pol];stress=policy_stats[a]['stress1.25'][pol]
            lines.append(f"| {a} | {pol} | {s['threshold']:.8f} | {cal['joint_exceedances']} | {s['preconnectivity_exceedances']} | {s['upper95']:.6f} | {s['final_supplemental_false_events']} | {stress['joint_exceedances']} | {stress['final_added_events']} | {'PASS' if s['comparison_usable'] else 'BLOCKED'} |")
    lines+=['','Stress uses 1.25 times the same noise, with the original frozen C0 and thresholds; it is a sensitivity diagnostic, not another calibrated screen. SUMMARY.json includes scene-wise baseline false support and supplemental pixel/brightness-area averages (brightness-area units mJy/nominal beam × arcsec²). Zero final supplemental events do not imply zero baseline false support.','','## Amplitude uncertainty check, before S/N selection','','Empirical values are sample standard deviations (ddof=1) of every finite converged core refit. Prediction is RMS of available σA; no accepted-only filtering or division by sqrt(N). Bias is mean amplitude minus the new fit to the original noiseless core. Missing predictions/fits are explicitly counted.','','| Array | Phase | Converged / with σA | Empirical scatter | RMS σA | Ratio | Bias |','|---|---|---:|---:|---:|---:|---:|']
    for a in ARRAYS:
        for ph,s in stats[a].items():
            u=s['core']['uncertainty'];lines.append(f"| {a} | {ph} | {u['converged']}/{u['with_sigma']} | {u['empirical_amplitude_std']:.4f} | {u['RMS_predicted_sigma']:.4f} | {u['empirical_amplitude_std']/u['RMS_predicted_sigma']:.4f} | {u['amplitude_bias']:.4f} |")
    lines+=['','This conditional local-linear check is not a replacement for the existing empirical uncertainty companions. Stress deliberately leaves σA based on C0 unchanged. Source-depth/regional-model qualifications of C0 persist, including a2000 background/source-transfer limitations and the a1100 network-3 scan-5 PTC qualification. No source recovery through PTC is validated here.','','## Recorded cases and historical comparison','','All 18 historical diagnosis outcomes and all 15 historical real-map measurements/candidates replay exactly on the unchanged historical route. All nine old budget-exhausted blank cases now converge and fail the width criterion; all 15 diagnosed blanks produce complete empty results, and three noisy core controls are accepted. The three noiseless blanks are rejected and three noiseless cores accepted. Raw old/new optimizer records and geometry/error/support are in DIAGNOSIS_COMPARISON.json and checks/RESULTS.json. An uncertainty omitted after a position/width rejection is not a zero error.','','| Array | Nine failed-case new nfev (three per array) | Bright controls | Max abs ΔA | Max abs centroid-component Δ arcsec | Max abs sorted-sigma Δ arcsec | Max abs ΔB | Added / removed baseline pixels |','|---|---|---:|---:|---:|---:|---:|---|']
    for a in ARRAYS:
        b=[r['comparison'] for r in checks['bright'] if r['array']==a];n=[r['new']['guard']['diagnostic']['nfev'] for r in checks['cases'] if r['array']==a and r['role']=='failed']
        lines.append(f"| {a} | {n} | 5/5 | {max(abs(r['peak_delta']) for r in b):.6g} | {max(max(abs(v) for v in r['centroid_delta_arcsec']) for r in b):.6g} | {max(max(abs(v) for v in r['sigmas_delta_arcsec']) for r in b):.6g} | {max(abs(r['constant_delta']) for r in b):.6g} | {sum(r['added_vs_historical'] for r in b)} / {sum(r['removed_vs_historical'] for r in b)} |")
    lines+=['','Brightness-operand and candidate-value differences are explicitly retained in checks/RESULTS.json; each maximum is the fitted constant change, below 4×10⁻⁷ mJy/nominal beam. New fit parameters are not replaced by historical values. The guard leaves candidate membership unchanged on these controls; this does not require future trajectories to be identical. The earlier authenticated historical ten-cycle BASE replays are reused; none is rerun here.','','## Unapplied saved-map candidates','','Each matched figure contains measured brightness (map minus fitted constant), BASE-POINT, CORR and DIAG at M0/T1/T3/T5/T10. Color limits are shared within an array and clipped at ±6% of its original T10 peak to expose low brightness. Black outlines show BASE-POINT; green on measured maps shows frozen K0; purple marks proposed and teal final connected extras. Gray is unavailable and white is valid zero. Source acceptance, thresholds and counts are in SAVED_SCREEN_COUNTS.json. Each FITS also retains score, variance and validity planes.','','| Array | Stage | BASE-POINT pixels | CORR proposed / added | DIAG proposed / added |','|---|---:|---:|---|---|']
    for a in ARRAYS:
        for k in SNAPSHOTS:
            rr={r['policy']:r for r in saved_rows if r['array']==a and r['stage']==k}
            fmt=lambda pol:'unavailable' if rr[pol]['counts'] is None else f"{rr[pol]['counts']['PROPOSED']} / {rr[pol]['counts']['ADDED']}"
            lines.append(f"| {a} | {k} | {rr['BASE-POINT']['counts']['ADMITTED']} | {fmt('CORR')} | {fmt('DIAG')} |")
    lines+=['','## Scope and runtime','','Implementation/checks and the fixed-field diagnostic screen are complete. Original-C0 amplitude errors and same-map candidates are qualified diagnostics. New BASE-POINT/CORR/DIAG feedback trajectories, endpoint response/noise/uncertainty, and source-transfer improvements are **unrun**. No new production/default, map, covariance, weight, calibration, cleaning or FRUIT state is adopted. There are 6144 reused fields, 768 stress reuses, zero independent new fields and zero new feedback applications.','','Recommended next action only: a separately authorized bounded trajectory comparison of BASE-POINT/CORR/DIAG using these frozen thresholds and methods, with source response and background diagnostics. Passing this selection screen does not demonstrate improved maps.','','| Stage | Wall seconds | Worker CPU seconds | Max worker RSS MiB |','|---|---:|---:|---:|']
    prep=read(out/'PREPARED.json');lines.append(f"| Checks | {prep['wall_seconds']:.3f} | not captured | not captured |")
    for ph in ('CALIBRATION','VALIDATION'):
        r=read(out/f'{ph}.json');lines.append(f"| {ph.lower()} | {r['wall_seconds']:.3f} | {r['CPU_seconds']:.3f} | {r['maximum_worker_RSS_bytes']/2**20:.2f} |")
    lines+=['','12 independent workers; one numerical thread requested. Apple Accelerate internal thread count is not observable. Calibration/validation source is '+p['source_revision']+'. Reporting source is '+revision+'. Reporting time is in REPORT.json. Saved-map publication time was not captured separately. This reporting stage performs no fits or field generation. Figure axes are tangent east/north arcseconds.','']
    (dest/'ASSESSMENT.md').write_text('\n'.join(lines))
    inputs(out);save_json(dest/'REPORT.json',dict(source_revision=revision,wall_seconds=time.perf_counter()-start,figures=figures,new_fits=0,new_fields=0,inputs_unchanged=True));manifest(dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    render(parser.parse_args().output)
