"""Saved-product assessment for the bounded high-coverage POINT replacement.
No fits, covariance learns, policy reevaluations, or native applications.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from point_guarded_feedback import POLICIES,BRANCHES
from point_response_cd import vector_measurement,MEASUREMENT_NAMES
from point_fruit_diagnostics import load as native_load,patch_map as native_patch_map
from point_additive_diagnostics import save_json

ROOT=None
ORIGINAL_ROOT_NAME='successor-point-152430-a2000-no-upper-coverage-veto-20260926'
def product_path(path):
    p=Path(path)
    if ORIGINAL_ROOT_NAME in p.parts:
        p=ROOT.joinpath(*p.parts[p.parts.index(ORIGINAL_ROOT_NAME)+1:])
    if not p.is_file() and p.suffix=='.fits' and p.with_suffix('.fits.gz').is_file():p=p.with_suffix('.fits.gz')
    if not p.is_file():raise FileNotFoundError('Required retained product absent: '+str(p))
    return p
def load(path):return native_load(product_path(path))
def patch_map(path):return native_patch_map(product_path(path))

SIGMA_NAMES={'peak':'peak','east':'cx','north':'cy','background':'constant','fixed_template':'template','aperture_1':'aperture1','aperture_1p5':'aperture1.5','aperture_2':'aperture2'}
def read(p):return json.loads(p.read_text())
def unit(n):return 'mJy/nominal beam * arcsec^2' if n.startswith('aperture') else 'arcsec^2' if n.startswith('ellipse') else 'arcsec' if n in ('east','north','FWHM_major','FWHM_minor') else 'mJy/nominal beam'
def num(x):return 'unresolved' if x is None else f'{x:.7g}'
def report(out):
    global ROOT
    ROOT=out.resolve()
    start=time.perf_counter();d=out/'report';d.mkdir(exist_ok=True);matrix=[];summaries={};hist=[];effects=[];responses=[];drifts=[]
    for pol in POLICIES:
        for b in BRANCHES:
            p=out/'run'/pol/b/'a2000/SUMMARY.json';r=read(p) if p.exists() else None
            status='complete' if r and r['ready'] and r['completed_cycle']==10 else 'stopped' if r else 'not executed'
            matrix.append(dict(policy=pol,branch=b,status=status,completed_cycle=r.get('completed_cycle') if r else None,reason=r.get('stop_reason') if r else 'dependent work unavailable',feedback_attempts=r.get('actual_feedback_applications',0) if r else 0))
            if not r:continue
            summaries[(pol,b)]=r
            for h in r['histories']:
                if 'measurement' not in h:continue
                m=h['measurement'];v=vector_measurement(m);ad=h.get('admission',{})
                hist.append(dict(policy=pol,branch=b,cycle=h['cycle_index'],values=dict(zip(MEASUREMENT_NAMES,v.tolist())) if v is not None else None,accepted=m['guard']['accepted'],selected_pixels=ad.get('selected_pixels'),added_pixels=ad.get('added_pixels'),raw_apertures=m.get('raw_apertures'),background_terms=m.get('aperture_background_terms')))
                if 'response' in h:responses.append(dict(policy=pol,branch=b,cycle=h['cycle_index'],**h['response']))
    for pol in POLICIES:
        ref=summaries.get((pol,'REF'))
        if not ref:continue
        sig={r['name']:r['conditional_sigma'] for r in ref.get('endpoint_uncertainty',[])}
        for b in BRANCHES:
            r=summaries.get((pol,b))
            if not r:continue
            for h in r['histories']:
                k=h['cycle_index'];rh=next((q for q in ref['histories'] if q['cycle_index']==k),None)
                if not rh or 'measurement' not in h or 'measurement' not in rh:continue
                a=vector_measurement(h['measurement']);v=vector_measurement(rh['measurement'])
                if a is None or v is None:continue
                comparisons=[('branch_minus_own_REF',a-v)] if b!='REF' else []
                ideal=h.get('ideal',{}).get('comparisons')
                if ideal:comparisons += [('actual_minus_ideal',np.array([q['absolute_discrepancy'] for q in ideal])),('ideal_increment',np.array([q['ideal_increment'] for q in ideal]))]
                for label,z in comparisons:
                    for name,value,full in zip(MEASUREMENT_NAMES,z,v):
                        sigma=sig.get(SIGMA_NAMES.get(name)) if k==10 else None
                        effects.append(dict(policy=pol,branch=b,cycle=k,comparison=label,measurement=name,effect=float(value),units=unit(name),own_REF=float(full),fraction_of_source=float(value/full) if abs(full)>1e-12 else None,conditional_endpoint_sigma=sigma,signed_effect_over_sigma=float(value/sigma) if sigma is not None and sigma>0 else None,scale_scope='own-policy REF T10 model-based conditional estimator sigma; practical comparison, not paired-difference significance' if k==10 else 'no endpoint scale applied to earlier cycles'))
            if r['completed_cycle']==10 and len(r['histories'])==11:
                a=vector_measurement(r['histories'][-1]['measurement']);z=vector_measurement(r['histories'][-2]['measurement'])
                if a is not None and z is not None:drifts.append(dict(policy=pol,branch=b,T9_to_T10=dict(zip(MEASUREMENT_NAMES,(a-z).tolist()))))
    cross=[];absorption=[];asymmetry=[]
    for pol in ('CORR','DIAG'):
        for b in BRANCHES:
            a=summaries.get((pol,b));base=summaries.get(('BASE-POINT',b))
            if a and base and a['completed_cycle']==base['completed_cycle']==10:
                av=vector_measurement(a['histories'][-1]['measurement']);bv=vector_measurement(base['histories'][-1]['measurement'])
                if av is not None and bv is not None:cross.append(dict(policy=pol,branch=b,comparison='policy_minus_BASE_POINT',measurement_difference=dict(zip(MEASUREMENT_NAMES,(av-bv).tolist()))))
    for (pol,b),r in summaries.items():
        if r['completed_cycle']!=10:continue
        ideal=r['histories'][-1].get('ideal')
        if ideal:absorption.append(dict(policy=pol,branch=b,rows=ideal['raw_aperture_and_background_increments']))
    for pol in POLICIES:
        plus=summaries.get((pol,'C-core-plus'));minus=summaries.get((pol,'C-core-minus'));ref=summaries.get((pol,'REF'))
        if plus and minus and ref and all(r['completed_cycle']==10 for r in (plus,minus,ref)):
            a,b,v=[vector_measurement(r['histories'][-1]['measurement']) for r in (plus,minus,ref)]
            if all(z is not None for z in (a,b,v)):asymmetry.append(dict(policy=pol,definition='plus increment + minus increment; zero for an odd signed measurement response',signed_sum=dict(zip(MEASUREMENT_NAMES,(a+b-2*v).tolist()))))
    save_json(d/'policy-contrasts.json',cross);save_json(d/'background-absorption.json',absorption);save_json(d/'compact-sign-asymmetry.json',asymmetry)
    attempts=[json.loads(l) for l in (out/'execution-attempts.jsonl').read_text().splitlines()] if (out/'execution-attempts.jsonl').exists() else []
    phases={p.stem:read(p) for p in out.glob('PHASE-*.json')};runs=[read(p) for p in out.rglob('RUN.json')]
    storage=[json.loads(l) for l in (out/'storage.jsonl').read_text().splitlines()] if (out/'storage.jsonl').exists() else []
    counts=dict(trajectories_started=len(summaries),trajectories_complete=sum(r['status']=='complete' for r in matrix),feedback_attempts=sum(r['kind']=='feedback' for r in attempts),initial_physical_inputs=1+sum(r['kind']=='injected_initial' for r in attempts),ideal_controls=sum('ideal' in h for r in summaries.values() for h in r['histories']),endpoint_attempts=len(list(out.rglob('ENDPOINT-ATTEMPT.json'))),native_invocations=len(runs),input_transfers=0,input_transfer_bytes=0)
    assert counts['feedback_attempts']<=150 and counts['trajectories_started']<=15 and counts['initial_physical_inputs']<=4 and counts['ideal_controls']<=9 and counts['endpoint_attempts']<=3
    resources=dict(native_wall_seconds_sum=sum(r['wall_seconds'] for r in runs),upstream_seconds=read(out/'UPSTREAM.json').get('seconds'),phase_wall_seconds={k:v.get('seconds') for k,v in phases.items()},phase_peak_family_rss_bytes={k:v.get('peak_family_rss_bytes') for k,v in phases.items()},native_max_sampled_rss_bytes=max((r.get('polled_peak_rss_bytes',0) for r in runs),default=0),largest_observed_stage_bytes=max((r['bytes'] for r in storage),default=0),ceiling_bytes=read(out/'freeze/budget.json')['peak_ceiling_bytes'],disk_measurement='stage snapshots plus enforced in-flight allocation reservations; not a continuous filesystem high-water measurement',numeric_threads=1,phase_workers={k:v.get('workers') for k,v in phases.items()})
    for name,value in [('matrix',matrix),('histories',hist),('effects',effects),('responses',responses),('terminal-drift',drifts),('execution-counts',counts),('resources',resources)]:save_json(d/(name+'.json'),value)
    lines=['# POINT 152430 a2000: replacement comparison','','New-run measurements only. High coverage alone is informational. All uncertainty scales are model-based conditional propagation, not an empirically calibrated false-admission probability or paired-difference significance. Apertures are brightness × angular area, not integrated flux.','','## Completion','','| Policy | Branch | Status | Last cycle | Feedback attempts | Reason |','|---|---|---|---:|---:|---|']
    lines += [f"| {r['policy']} | {r['branch']} | {r['status']} | {r['completed_cycle']} | {r['feedback_attempts']} | {r['reason'] or ''} |" for r in matrix]
    lines += ['','## Leading T10 measurement differences','','Source responses are compared to their own policy REF. Actual-minus-ideal isolates response mismatch against the single saved ideal map-addition control; seed-minus-REF is initialization sensitivity.','','| Policy | Branch | Comparison | Measurement | Signed effect | % whole REF | Conditional sigma | Effect / sigma |','|---|---|---|---|---:|---:|---:|---:|']
    for e in effects:
        if e['cycle']==10 and e['measurement'] in ('peak','fixed_template','aperture_2') and (e['comparison']=='actual_minus_ideal' or (e['branch']=='D-seed-shoulder' and e['comparison']=='branch_minus_own_REF')):
            lines.append(f"| {e['policy']} | {e['branch']} | {e['comparison']} | {e['measurement']} | {num(e['effect'])} | {num(None if e['fraction_of_source'] is None else 100*e['fraction_of_source'])} | {num(e['conditional_endpoint_sigma'])} | {num(e['signed_effect_over_sigma'])} |")
    lines += ['','## Measurement definitions','','POINT-EXPECTED-v1: seven-parameter Gaussian plus constant; W=I; one bounded fit per full map. Peak is amplitude above the fitted constant in mJy/nominal beam. Centroid is east/north arcsec. Major/minor FWHM are sorted principal widths; ellipse covariance components avoid angle gauge ambiguity. Fixed template uses the original M0 template and subtracts each map’s fitted constant. Apertures use fixed M0 centroid and radii 1, 1.5, 2 × M0 major FWHM with 4 arcsec² pixels. Raw aperture and fitted-background contributions are separately retained in histories. Endpoint sensitivity operators include refitted constant nuisance and spatial covariance. Angle/width errors are not substituted for sorted or tensor-shape uncertainties. Unsupported uncertainty scales remain unresolved.','','M0–T10 compact history is in histories.json; the stage summary table below selects M0/T1/T3/T5/T10. g and eta use the fixed own-REF ordinary-MAP injection controls and unweighted common support. Controls bypass PTC; this experiment tests post-RTC/CAL response and does not establish astronomical flux calibration.','','## Selected histories','','| Policy | Branch | Cycle | Peak | Major/minor FWHM | Template | Aperture 2 | Selected / added |','|---|---|---:|---:|---|---:|---:|---|']
    for r in hist:
        if r['cycle'] not in (0,1,3,5,10) or r['values'] is None:continue
        v=r['values'];lines.append(f"| {r['policy']} | {r['branch']} | {r['cycle']} | {num(v['peak'])} | {num(v['FWHM_major'])}/{num(v['FWHM_minor'])} | {num(v['fixed_template'])} | {num(v['aperture_2'])} | {r['selected_pixels']}/{r['added_pixels']} |")
    lines += ['','## Execution and resources','',f'Counts: `{json.dumps(counts)}`.', '',f'Resources: `{json.dumps(resources)}`.','', 'The native processing guards, subtract/restore checks, control audits, shape definitions, exact inputs and configuration remain in compact products. Numeric libraries use one thread; outer workers are in resources.json. No threshold/covariance retuning, extra cycles, other observations/arrays, or JINC.']
    (d/'MEASUREMENTS.md').write_text('\n'.join(lines)+'\n')
    # Matched scales: each column has one scale across policies.
    fig,ax=plt.subplots(3,3,figsize=(11,10),constrained_layout=True)
    data={}
    for pi,pol in enumerate(POLICIES):
        for bi,b in enumerate(('REF','C-shoulder','D-seed-shoulder')):
            r=summaries.get((pol,b))
            if r and r['completed_cycle']==10 and 'total' in r['histories'][-1]:
                z,v,_,_=patch_map(r['histories'][-1]['total']['path']);ref=summaries.get((pol,'REF'));rz,rv,_,_=patch_map(ref['histories'][-1]['total']['path']);data[(pi,bi)]=(np.where(v,z if bi==0 else z-rz,np.nan))
    for bi in range(3):
        arrays=[v for (p,b),v in data.items() if b==bi];lim=max((np.nanmax(abs(z)) for z in arrays),default=1.)
        for pi,pol in enumerate(POLICIES):
            a=ax[pi,bi];z=data.get((pi,bi));a.set_title(pol+' '+('REF T10','shoulder − REF','seed − REF')[bi],fontsize=10)
            if z is None:a.text(.5,.5,'Unavailable',ha='center',transform=a.transAxes);a.set_axis_off();continue
            im=a.imshow(z,origin='lower',extent=[41,-41,-41,41],cmap='viridis' if bi==0 else 'RdBu_r',vmin=0 if bi==0 else -lim,vmax=lim);fig.colorbar(im,ax=a,shrink=.75,label='mJy/nominal beam');a.set_xlabel('east (arcsec)');a.set_ylabel('north (arcsec)')
    fig.savefig(d/'matched-endpoints.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for pol in POLICIES:
        r=[h for h in hist if h['policy']==pol and h['branch']=='REF' and h['values']]
        for a,n in zip(ax,('peak','aperture_2')):a.plot([q['cycle'] for q in r],[q['values'][n] for q in r],'.-',label=pol);a.set_xlabel('Cycle');a.set_ylabel(n+' ('+unit(n)+')');a.legend()
    fig.savefig(d/'reference-histories.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(11,3.8),constrained_layout=True)
    for a,pol in zip(ax,POLICIES):
        r=summaries.get((pol,'REF'));a.set_title(pol+' T10 admission')
        if not r or r['completed_cycle']!=10:a.set_axis_off();continue
        path=Path(r['histories'][-1]['candidate']['path']).parent/'admission.fits';z,h=load(path)
        mask=z['BASE_POINT']+2*z['ADDED'];im=a.imshow(mask,origin='lower',extent=[41,-41,-41,41],vmin=0,vmax=2,cmap='viridis')
        ex=np.argwhere(z.get('NOISE_EXTRAP',np.zeros_like(mask)))
        if len(ex):a.plot(40-2*ex[:,1],-40+2*ex[:,0],'rx',label='high coverage');a.legend(fontsize=7)
        a.set_xlabel('east (arcsec)');a.set_ylabel('north (arcsec)')
    
    if any(r['completed_cycle']==10 for (pol,b),r in summaries.items() if b=='REF'):fig.colorbar(im,ax=ax,ticks=[0,1,2],label='0 excluded, 1 BASE, 2 supplement')
    fig.savefig(d/'admission-masks.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Common residual color scale makes policy differences visually comparable.
    fig,ax=plt.subplots(3,3,figsize=(10,10),constrained_layout=True);residuals={}
    for pi,pol in enumerate(POLICIES):
        for bi,b in enumerate(BRANCHES[1:4]):
            r=summaries.get((pol,b))
            if r and r['completed_cycle']==10:
                p=Path(r['histories'][-1]['total']['path']).parents[2]/'response.fits'
                if product_path(p).exists():
                    q,h=load(p);z,v,_,_,_=__import__('point_empirical').source_patch(q['RESIDUAL'],q['VALID'].astype(bool),np.ones(q['RESIDUAL'].shape),h);residuals[(pi,bi)]=np.where(v,z,np.nan)
    lim=max((np.nanmax(abs(z)) for z in residuals.values()),default=1.)
    for pi,pol in enumerate(POLICIES):
        for bi,b in enumerate(BRANCHES[1:4]):
            a=ax[pi,bi];a.set_title(pol+' '+b,fontsize=9);z=residuals.get((pi,bi))
            if z is None:a.set_axis_off();continue
            im=a.imshow(z,origin='lower',extent=[41,-41,-41,41],cmap='RdBu_r',vmin=-lim,vmax=lim);a.set_xlabel('east (arcsec)');a.set_ylabel('north (arcsec)')
    if residuals:fig.colorbar(im,ax=ax,label='(branch − own REF) − control [mJy/nominal beam]',shrink=.65)
    fig.savefig(d/'response-residuals.png',dpi=160,bbox_inches='tight');plt.close(fig)
    save_json(d/'REPORT.json',dict(saved_product_only=True,seconds=time.perf_counter()-start,counts=counts))
    print(json.dumps(counts))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);report(p.parse_args().output)
