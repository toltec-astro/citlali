"""Read compact PTC experiment products. Never learn, fit, or launch science."""
import json
from pathlib import Path
import numpy as np
from point_fruit_diagnostics import load, patch_map
from point_response_cd import vector_measurement, MEASUREMENT_NAMES
from point_additive_diagnostics import save_json
from compare_point_pipeline import available
from inspect_forward_map import require

SIGMA={'peak':'peak','east':'cx','north':'cy','background':'constant','fixed_template':'template','aperture_1':'aperture1','aperture_1p5':'aperture1.5','aperture_2':'aperture2'}
BRANCHES=('REF','C-core-plus','C-core-minus','C-shoulder')
def read(p):return json.loads(p.read_text())
def num(x):return 'unresolved' if x is None else f'{x:.6g}'
def units(n):return 'mJy/nominal beam arcsec^2' if n.startswith('aperture') else 'arcsec^2' if n.startswith('ellipse') else 'arcsec' if n in ('east','north','FWHM_major','FWHM_minor') else 'mJy/nominal beam'

def equivalent(sigma,gain,distortion,branch):
    if branch=='C-shoulder':return None,'broader spatial response; no scalar noise correction'
    if sigma is None or gain is None or not np.isfinite(sigma+gain) or sigma<=0 or abs(gain)<1e-8:return None,'unsupported or degenerate response'
    if gain<=0:return None,'sign reversal'
    if distortion is None or not np.isfinite(distortion) or distortion>=1:return None,'severe or unavailable spatial distortion (eta >= 1)'
    return sigma/abs(gain),'input-equivalent conditional measurement uncertainty; not calibrated sensitivity'


def report(root):
    require((root/'EXECUTION.json').is_file(),'missing execution; report never reruns science')
    d=root/'report';d.mkdir(exist_ok=True);execution=read(root/'EXECUTION.json');summaries={};histories=[];effects=[];ref_rows=[];support=[];network=[];asymmetry=[]
    for rank in (5,10,15):
        for branch in BRANCHES:
            p=root/f'rank-{rank}/run/BASE-POINT/{branch}/a2000/SUMMARY.json'
            if not p.exists():continue
            result=read(p);summaries[rank,branch]=result
            for h in result['histories']:
                if 'measurement' not in h:continue
                v=vector_measurement(h['measurement']);k=h['cycle_index']
                histories.append(dict(rank=rank,branch=branch,cycle=k,values=dict(zip(MEASUREMENT_NAMES,v.tolist())) if v is not None else None,selected_pixels=h.get('admission',{}).get('selected_pixels'),measurement=h['measurement'],response=h.get('response'),ideal=h.get('ideal')))
        ref=summaries.get((rank,'REF'))
        if not ref:continue
        sig={r['name']:r['conditional_sigma'] for r in ref.get('endpoint_uncertainty',[])}
        for h in ref['histories']:
            if 'measurement' not in h:continue
            k=h['cycle_index'];m=h['measurement'];v=vector_measurement(m)
            if v is None:continue
            total=root/f'rank-{rank}/M0/products/a2000/map.fits' if k==0 else root/f'rank-{rank}/run/BASE-POINT/REF/a2000/cycle-{k:02d}/native/products/total.fits'
            im,header=load(available(total));use=im['SCI_SUPPORT'].astype(bool)
            ref_rows.append(dict(rank=rank,cycle=k,values=dict(zip(MEASUREMENT_NAMES,v.tolist())),conditional_sigmas=sig if k==10 else {},M0_C0_guard_sigma=m['guard'].get('sigma_A'),support_pixels=int(use.sum()),contributing_samples=int(im['HITS'].sum()),coverage=dict(min=float(im['NORMALIZATION'][use].min()),median=float(np.median(im['NORMALIZATION'][use])),max=float(im['NORMALIZATION'][use].max())),background=m['background']))
            baseline=root/f'rank-10/M0/products/a2000/map.fits' if k==0 else root/f'rank-10/run/BASE-POINT/REF/a2000/cycle-{k:02d}/native/products/total.fits'
            if available(baseline).is_file():
                b,bh=load(available(baseline));same=im['PRIMARY'].shape==b['PRIMARY'].shape and all(header[key]==bh[key] for key in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2'))
                row=dict(rank=rank,cycle=k,same_grid=same)
                if same:
                    bv=b['SCI_SUPPORT'].astype(bool);lost=bv&~use;gained=use&~bv
                    row.update(planes_equal={n:bool(np.array_equal(im[n],b[n],equal_nan=True)) for n in ('HITS','NORMALIZATION','SCI_SUPPORT','NORM_SUPPORT')},lost_pixels=int(lost.sum()),gained_pixels=int(gained.sum()),lost_yx=np.argwhere(lost).tolist(),gained_yx=np.argwhere(gained).tolist())
                support.append(row)
        for branch in BRANCHES[1:]:
            r=summaries.get((rank,branch))
            if not r or r['completed_cycle']!=10:continue
            h=r['histories'][-1];ideal=h.get('ideal');rv=vector_measurement(ref['histories'][-1]['measurement'])
            if not ideal or not ideal['comparisons'] or rv is None:continue
            for q,full in zip(ideal['comparisons'],rv):
                n=q['name'];sigma=sig.get(SIGMA.get(n));g=q['ratio'];eta=h['response']['source_disk']['eta']
                eq,why=equivalent(sigma,g,eta,branch)
                effects.append(dict(rank=rank,branch=branch,measurement=n,units=units(n),own_REF=float(full),**q,fraction_of_source=q['absolute_discrepancy']/full if abs(full)>1e-8 else None,conditional_sigma=sigma,effect_over_sigma=q['absolute_discrepancy']/sigma if sigma else None,sigma_equiv=eq,sigma_equiv_scope=why,spatial_response=h['response']))
        plus=summaries.get((rank,'C-core-plus'));minus=summaries.get((rank,'C-core-minus'))
        if plus and minus and all(s.get('ready') and s['completed_cycle']==10 and 'measurement' in s['histories'][-1] for s in (plus,minus,ref)):
            a,b,r=[vector_measurement(s['histories'][-1]['measurement']) for s in (plus,minus,ref)]
            if all(x is not None for x in (a,b,r)):asymmetry.append(dict(rank=rank,definition='plus increment plus minus increment; zero for odd signed response',values=dict(zip(MEASUREMENT_NAMES,(a+b-2*r).tolist()))))
        for stage,file in [('M0','M0-PTC.json'),('last','LAST-PTC.json')]:
            p=root/f'rank-{rank}'/file
            if not p.exists():continue
            for nw in (11,12):
                records=[r for r in read(p) if r['network']==nw];ss=[r['segment'] for r in records];old=[s.get('source_segment',s) for s in ss]
                network.append(dict(rank=rank,network=nw,stage=stage,segments=len(ss),requested_ranks=sorted(set(r['requested_rank'] for r in records)),realized_ranks=sorted(set(r['realized_rank'] for r in records)),detectors=sorted(set(x for s in old for x in s['detector_indices'])),eligible_entries=sum(s['eligible'] for s in old),scheduled_times=sum(s['scheduled_times'] for s in old),retained_entries=sum(s['retained'] for s in ss),all_converged=all(s['converged'] for s in ss),terminations=[s['stopping_reason'] for s in ss],iterations=[s['iterations'] for s in ss],objectives=[s['objective'] for s in ss],power_diagnostic='existing final observed-entry residual objective per segment; before/after spectra/correlation not retained by this route'))
    attempts=[json.loads(s) for s in (root/'attempts.jsonl').read_text().splitlines()] if (root/'attempts.jsonl').exists() else []
    counts={kind:sum(r['kind']==kind for r in attempts) for kind in ('parent','trajectory','feedback','noise','endpoint','ideal')}
    runs=[read(p) for p in root.rglob('RUN.json')];storage=[json.loads(s) for p in root.glob('rank-*/storage.jsonl') for s in p.read_text().splitlines()]
    resources=dict(wall_seconds=execution.get('seconds'),maximum_native_process_sampled_RSS=max((r.get('polled_peak_rss_bytes',0) for r in runs),default=0),maximum_stage_storage=max((r['bytes'] for r in storage),default=0),storage_ceiling=read(root/'PREFLIGHT.json')['resource_budget']['peak_ceiling_bytes'],branch_workers_maximum=3,numeric_library_threads=1,scope='per-native-process sampled RSS; stage storage snapshots plus allocation reservations, not continuous system or filesystem high water')
    drift=[]
    for (rank,branch),r in summaries.items():
        if r.get('ready') and r['completed_cycle']==10 and len(r['histories'])>=2 and all('measurement' in h for h in r['histories'][-2:]):
            a,b=[vector_measurement(h['measurement']) for h in r['histories'][-2:]]
            if a is not None and b is not None:drift.append(dict(rank=rank,branch=branch,T9_to_T10=dict(zip(MEASUREMENT_NAMES,(b-a).tolist()))))
    for name,value in [('matrix',execution['matrix']),('histories',histories),('REF',ref_rows),('effects',effects),('support',support),('network-PTC',network),('compact-asymmetry',asymmetry),('drift',drift),('counts',counts),('resources',resources)]:save_json(d/(name+'.json'),value)
    lines=['# PTC ranks 5, 10 and 15: POINT 152430/a2000','',f"Execution: {execution['status']}. Original-input RTC/CAL parent generations: {counts['parent']}; trajectories: {counts['trajectory']}; feedback attempts: {counts['feedback']}; noise packages: {counts['noise']}; endpoint analyses: {counts['endpoint']}; ideal controls: {counts['ideal']}.",'','Rank10 remains the reference. Measurement differences are method sensitivity, not absolute flux truth. Conditional scales are not paired p-values, blank-field sensitivity, or a full adaptive uncertainty law. RTC/CAL transfer was not tested.','', '| Rank | Branch | Status | Last cycle | Reason |','|---|---|---|---:|---|']
    for row in execution['matrix']:lines.append(f"| {row['rank']} | {row['branch']} | {row['status']} | {row.get('completed_cycle','')} | {row.get('reason') or ''} |")
    lines += ['','## Unmodified reference measurements','','Peak/template: mJy/nominal beam. Aperture: mJy/nominal beam arcsec². Widths and positions: arcsec. Fixed apertures/template are common across ranks; fitted source parameters remain free.','','| Rank | Stage | Peak | Major / minor FWHM | Template | Aperture 2 |','|---|---|---:|---|---:|---:|']
    for row in ref_rows:
        if row['cycle'] not in (0,1,3,5,10):continue
        v=row['values'];lines.append(f"| {row['rank']} | {'M0' if row['cycle']==0 else 'T'+str(row['cycle'])} | {num(v['peak'])} | {num(v['FWHM_major'])} / {num(v['FWHM_minor'])} | {num(v['fixed_template'])} | {num(v['aperture_2'])} |")
    lines += ['','## Endpoint response and conditional uncertainty','','Actual-minus-ideal is the departure from own REF plus the ordinary-MAP injection control. Source percentages use the full own REF measurement, not the artificial increment. Compact plus/minus gains are separate; distorted shoulder response has no scalar uncertainty correction.','','| Rank | Signal | Q | Actual − ideal | % full source | / conditional σ | Gain | σ | σ equivalent |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for e in effects:
        if e['measurement'] in ('peak','fixed_template','aperture_2'):
            lines.append(f"| {e['rank']} | {e['branch']} | {e['measurement']} | {num(e['absolute_discrepancy'])} | {num(None if e['fraction_of_source'] is None else 100*e['fraction_of_source'])} | {num(e['effect_over_sigma'])} | {num(e['ratio'])} | {num(e['conditional_sigma'])} | {num(e['sigma_equiv'])} |")
    lines+=['','## Network populations','','| Rank | Network | Stage | Segments | Retained entries | Realized ranks | Converged |','|---|---|---|---:|---:|---|---|']
    for n in network:lines.append(f"| {n['rank']} | {n['network']} | {n['stage']} | {n['segments']} | {n['retained_entries']} | {n['realized_ranks']} | {n['all_converged']} |")
    lines+=['','Native per-segment objectives, terminations, detector identities, eligibility and iterations are retained in network-PTC.json and M0/LAST receipts. The route does not publish before/after residual spectra or correlation summaries; no new spectral method was introduced. Lower variance alone is not success.','', 'All-cycle raw apertures, fitted-background contributions, response morphology, admitted pixels and T9–T10 changes remain in compact JSON. Full-support comparisons and any lost/gained pixel locations are in support.json. See the owner assessment for the interpretation and single next recommendation.','',f"Runtime/storage: {json.dumps(resources)}",'','Local AppleClang/Homebrew evidence only; no Unity/Spack qualification.']
    (d/'REPORT.md').write_text('\n'.join(lines)+'\n')
    figures(root,d,summaries,ref_rows,effects)
    return dict(counts=counts,resources=resources)


def figures(root,d,summaries,refs,effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for rank in (5,10,15):
        rows=[r for r in refs if r['rank']==rank]
        for ax,key in zip(axes,('peak','fixed_template','aperture_2')):
            ax.plot([r['cycle'] for r in rows],[r['values'][key] for r in rows],'.-',label=f'rank {rank}');ax.set(title=key,xlabel='Application (0 = M0)',ylabel=units(key));ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(d/'REF-histories.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(10,9),squeeze=False);images=[]
    for i,rank in enumerate((5,10,15)):
        for j,branch in enumerate(BRANCHES[1:]):
            p=root/f'rank-{rank}/run/BASE-POINT/{branch}/a2000/cycle-10/response.fits'
            if not p.exists() and not Path(str(p)+'.gz').exists():continue
            im,h=load(available(p));cy,cx=int(round(h['CRPIX2']-1)),int(round(h['CRPIX1']-1));z=im['RESIDUAL'][cy-20:cy+21,cx-20:cx+21];images.append((i,j,z))
    limit=max((float(np.nanmax(abs(z))) for _,_,z in images),default=1)
    for i,j,z in images:
        plot=axes[i,j].imshow(z,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit,extent=(41,-41,-41,41));axes[i,j].set(title=f'rank {(5,10,15)[i]} {BRANCHES[j+1]}',xlabel='East (arcsec)',ylabel='North (arcsec)')
    if images:fig.colorbar(plot,ax=axes.ravel().tolist(),label='actual − own REF − ideal injection (mJy/nominal beam)',shrink=.6)
    fig.savefig(d/'response-residuals.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for ax,key in zip(axes,('peak','fixed_template','aperture_2')):
        for branch in BRANCHES[1:3]:
            rows=[e for e in effects if e['branch']==branch and e['measurement']==key and e['sigma_equiv'] is not None]
            ax.plot([e['rank'] for e in rows],[e['sigma_equiv'] for e in rows],'.-',label=branch)
        rows=[r for r in refs if r['cycle']==10];ax.plot([r['rank'] for r in rows],[r['conditional_sigmas'].get(SIGMA[key],np.nan) for r in rows],'k--',label='conditional σ')
        ax.set(title=key,xlabel='PTC rank',ylabel=units(key),xticks=[5,10,15]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(d/'response-uncertainty.png',dpi=160);plt.close(fig)
