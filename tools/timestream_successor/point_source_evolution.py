"""Compact descriptions of executed fits; no additional optimization or acceptance."""
import json
import numpy as np
from point_fruit_trajectory import bound
from point_additive_diagnostics import save_json


def describe(measurement):
    g=measurement['guard'];p=g.get('accepted_parameters')
    if p is None:return dict(available=False,guard_status=g['status'],reason=g['cause'])
    major,minor=sorted(np.asarray(p[3:5])*np.sqrt(8*np.log(2)),reverse=True)
    return dict(available=True,peak_above_constant_mJy_beam=p[0],constant_mJy_beam=p[6],
                centroid_tangent_arcsec=p[1:3],frame=g['expected_origin_sky'],
                major_FWHM_arcsec=float(major),minor_FWHM_arcsec=float(minor),
                axis_ratio=float(major/minor),geometric_mean_FWHM_arcsec=float(np.sqrt(major*minor)),
                conditional_frozen_M0_sigma_A=g['sigma_A'],uncertainty_scope='frozen M0 guard covariance, not cycle-specific or paired-difference uncertainty',
                auxiliary_support=measurement.get('auxiliary_support'))


def export(out,k,map_path,measurement):
    path=out/'SOURCE-EVOLUTION.json';rows=json.loads(path.read_text()) if path.exists() else []
    rows=[r for r in rows if r['cycle']!=k]
    rows.append(dict(cycle=k,label='M0' if k==0 else 'T'+str(k),map=bound(map_path),**describe(measurement)))
    save_json(path,sorted(rows,key=lambda r:r['cycle']))


def compare_saved(root,repo):
    """Six executed B fits plus unchanged native-frame historical benchmarks."""
    from pathlib import Path
    from point_fruit_trajectory import verified
    import matplotlib.pyplot as plt
    out=root/'report';out.mkdir(exist_ok=True)
    c=repo/'validation/point152420_map_comparison_2026-09-29'
    earlier=json.loads((c/'results/metrics.json').read_text())
    historical=json.loads((c/'redu09/results/metrics.json').read_text())
    for b in earlier['inputs']+historical['inputs']:verified(b)
    rows=json.loads((root/'B-REF/SOURCE-EVOLUTION.json').read_text())
    assert [r['cycle'] for r in rows]==list(range(6))
    benchmarks=[]
    for label,record in [('Earlier V2 (no FRUIT, 1 arcsec)',earlier['fits']['old_native']),('redu09 (FRUIT index 4, 1 arcsec)',historical['fits']['redu09_native'])]:
        p=record['parameters'];major,minor=record['fwhm_major_minor_arcsec']
        benchmarks.append(dict(label=label,peak_above_constant_mJy_beam=p[0],constant_mJy_beam=p[6],
            centroid_tangent_arcsec=p[1:3],frame='native AltAz offsets; not registered to successor FK5',
            major_FWHM_arcsec=major,minor_FWHM_arcsec=minor,axis_ratio=major/minor,
            geometric_mean_FWHM_arcsec=float(np.sqrt(major*minor)),uncertainty=None))
    runtime=json.loads((root/'B-REF/LAUNCH.json').read_text())
    timing=[json.loads(s) for s in (root/'B-REF/timing.jsonl').read_text().splitlines()]
    stages=[{k:e[k] for k in ('name','seconds','scope')} for e in timing if e['event']=='span']
    record=dict(successor=rows,historical=benchmarks,inputs=[bound(c/'results/metrics.json'),bound(c/'redu09/results/metrics.json')],
                raw_to_final_seconds=runtime['exit_seconds'],stage_timings=stages,
                qualifications=['Same free seven-parameter descriptive fit inside 40 arcsec; historical 1 arcsec vs successor 2 arcsec pixels.',
                'Native AltAz and FK5 tangent frames not registered; no angle or pixelwise differences.',
                'Historical FRUIT index 4 is not evidence of exactly five successor feedback applications.',
                'Guard sigma_A uses frozen M0 covariance. Only T5 receives endpoint covariance; no ensembles added.',
                'Brightness and width changes alone establish neither correct astronomy nor contamination removal.',
                'Stage times include nested spans and must not be added.'])
    save_json(out/'source-evolution.json',record)
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,key,title in zip(axs.ravel(),('peak_above_constant_mJy_beam','constant_mJy_beam','major_FWHM_arcsec','axis_ratio'),('Peak above constant (mJy/beam)','Fitted constant (mJy/beam)','Major FWHM (arcsec)','Major / minor FWHM')):
        ax.plot(range(6),[r.get(key,np.nan) for r in rows],marker='o',label='Successor B (2 arcsec)')
        for b,color in zip(benchmarks,('gray','tab:orange')):ax.axhline(b[key],color=color,ls='--',label=b['label'])
        ax.set(xticks=range(6),xticklabels=['M0','T1','T2','T3','T4','T5'],ylabel=title)
    axs[0,0].legend(fontsize=7);fig.suptitle('POINT 152420 a1100: source evolution and historical benchmarks')
    fig.savefig(out/'source-evolution.png',dpi=150);plt.close(fig)
    save_json(out/'SOURCE-MANIFEST.json',[bound(out/n) for n in ('source-evolution.json','source-evolution.png')])
