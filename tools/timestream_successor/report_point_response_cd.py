#!/usr/bin/env python3
"""Assess the fixed C+D campaign; no native execution or feedback changes."""
import argparse
from pathlib import Path
import time
import numpy as np
from point_response_cd import (BRANCHES,injection_reference,response_metrics,measurement_response,
                               vector_measurement,MEASUREMENT_NAMES,shape_projection)
from run_point_response_cd import json_read,recover,ARRAYS,source
from point_fruit_trajectory import bound,verified,verify_manifest,manifest,load,selection_difference
from point_fruit_diagnostics import patch_map
from point_additive_diagnostics import save_json
from point_empirical_io import write_fits
from point_empirical import coordinates,source_patch
from inspect_forward_map import require,read_yaml,digest


def aligned(a,ha,b,hb):
    """Align AST-centered equal-scale grids on their union without filling support."""
    for key in ('CDELT1','CDELT2','CRVAL1','CRVAL2','RADESYS','EQUINOX'):
        require(ha[key]==hb[key],'foreign paired WCS '+key)
    cx=max(ha['CRPIX1'],hb['CRPIX1']);cy=max(ha['CRPIX2'],hb['CRPIX2'])
    nr=int(cy+max(a.shape[0]-ha['CRPIX2'],b.shape[0]-hb['CRPIX2']))
    nc=int(cx+max(a.shape[1]-ha['CRPIX1'],b.shape[1]-hb['CRPIX1']))
    h=ha.copy();h['CRPIX1']=cx;h['CRPIX2']=cy
    def place(v,vh):
        dy=cy-vh['CRPIX2'];dx=cx-vh['CRPIX1'];require(dx==int(dx) and dy==int(dy),'nonaligned paired pixel grid')
        out=np.full((nr,nc),np.nan);out[int(dy):int(dy)+v.shape[0],int(dx):int(dx)+v.shape[1]]=v
        return out
    return place(a,ha),place(b,hb),h,place


def candidate_comparison(current, reference, shoulder):
    """Failed/constrained fits can publish total maps without replacement models."""
    if not current.get('candidate'):
        return dict(candidate_status='unavailable',candidate_unavailable_reason=current.get('stop_reason'),
                    candidate_vs_REF=None,candidate_area_arcsec2=None,replacement_shoulder_projection=None)
    cs,_=load(verified(current['candidate']));rs,_=load(verified(reference['candidate']))
    cv=cs['VALID'].astype(bool)&rs['VALID'].astype(bool)
    cdiff=np.where(cv,cs['CANDIDATE']-rs['CANDIDATE'],np.nan)
    p=shape_projection(cdiff,cv,shoulder)
    return dict(candidate_status='available',candidate_vs_REF=selection_difference(cs,rs),
        candidate_area_arcsec2=int(cs['ADMITTED'].sum())*4,replacement_shoulder_projection=p['value'])


MEASUREMENT_UNITS=['mJy/nominal beam','arcsec','arcsec','arcsec','arcsec',
    'arcsec² (shape tensor)','arcsec² (shape tensor)','arcsec² (shape tensor)',
    'mJy/nominal beam','mJy/nominal beam',
    'mJy/nominal beam × arcsec²','mJy/nominal beam × arcsec²','mJy/nominal beam × arcsec²']


def fit_control_notes(results):
    """Expose retained failed/constrained measurement controls without refitting."""
    lines=['### Measurement-control fit flags','',
           'Successful constrained fits remain reported, with their original values and bounds. '
           'Their measurement comparisons are qualified; map-response diagnostics are independent.','',
           '| Branch | Cycle | Control | Status | Constrained |',
           '|---|---:|---|---|---|']
    found=False
    for branch,rows in results.items():
        for row in rows:
            for name,measurement in row.get('measurement_response',{}).items():
                if name not in ('reference','actual','ideal_map_addition'):continue
                fit=measurement['fit']
                if fit['status']!='success' or fit.get('constrained',False):
                    found=True
                    lines.append(f"| {branch} | {row['cycle']} | {name} | {fit['status']} | {fit.get('constrained','unavailable')} |")
    if not found:lines.append('| All available controls | — | — | success | False |')
    return lines


def measurement_tables(out, array, results):
    lines=[f'# {array}: measurement response and remaining drift','',
        'All values retain the original fit, template, aperture centers/radii and background definitions. '
        'Shape tensor is the Gaussian squared-width tensor, not an uncertainty covariance. '
        'Peak/template/background are mJy/nominal beam; aperture integrals are brightness times angular area.',
        'Ratios are secondary deterministic diagnostics; unavailable if the ideal increment is too small. '
        'No signed-difference positive-source fit, uncertainty conversion or flux correction is used.','']
    units=dict(zip(MEASUREMENT_NAMES,MEASUREMENT_UNITS))
    for branch,rows in results.items():
        final=next((r for r in rows if r['cycle']==10),None)
        if final is None:continue
        lines.extend([f'## {branch} at T10',''])
        if branch.startswith('C-'):
            comparisons=final['measurement_response']['comparisons']
            for control in ('reference','actual','ideal_map_addition'):
                fit=final['measurement_response'][control]['fit']
                lines.append(f"{control}: status={fit['status']}; constrained={fit.get('constrained','unavailable')}.")
            lines.append('')
            lines.extend(['| Measurement | Unit | Actual increment | Ideal increment | Actual − ideal | Ratio |',
                          '|---|---|---:|---:|---:|---:|'])
            for row in comparisons or []:
                lines.append(f"| {row['name']} | {units[row['name']]} | {fmt(row['actual_increment'])} | {fmt(row['ideal_increment'])} | {fmt(row['absolute_discrepancy'])} | {fmt(row['ratio'])} |")
            lines.extend(['','| Fixed aperture | Actual raw increment | Ideal raw increment | Actual background term | Ideal background term |',
                          '|---|---:|---:|---:|---:|'])
            for row in final['measurement_response'].get('raw_aperture_and_background_increments',[]):
                if row['available']:lines.append(f"| {row['factor']} × M0 major FWHM | {fmt(row['actual_raw_increment'])} | {fmt(row['ideal_raw_increment'])} | {fmt(row['actual_background_increment'])} | {fmt(row['ideal_background_increment'])} |")
            lines.append('Raw minus fitted-background term is the reported source-relative aperture increment; no missing-area/aperture correction.')
        else:
            lines.extend(['| Measurement | Unit | T10 − REF T10 | T10 − T9 within branch |',
                          '|---|---|---:|---:|'])
            for name in MEASUREMENT_NAMES:
                lines.append(f"| {name} | {units[name]} | {fmt(final.get('measurement_increments',{}).get(name))} | {fmt(final.get('previous_cycle_measurement_drift',{}).get(name))} |")
        lines.append('')
    lines.extend(fit_control_notes(results))
    (out/'MEASUREMENTS.md').write_text('\n'.join(lines)+'\n')


def assess(out):
    began=time.perf_counter();revision=source();freeze=json_read(out/'FREEZE.json')
    verify_manifest(out/'freeze',freeze['manifest']['sha256']);owner=json_read(verified(freeze['prospective']))
    roots,fixeds=recover();report=out/'report';report.mkdir(exist_ok=False)
    summaries={};runtime=[];status=[]
    for branch in BRANCHES:
        for array in ARRAYS:
            p=out/f'run/{branch}/{array}/SUMMARY.json'
            if not p.exists():status.append(dict(array=array,branch=branch,status='not-run'));continue
            verify_manifest(p.parent);s=json_read(p);summaries[branch,array]=s
            status.append(dict(array=array,branch=branch,status='T10 complete' if s['ready'] and s['completed_cycle']==10 else 'stopped',
                completed=s['completed_cycle'],reason=s['stop_reason']))
            for c in s['histories']:
                if c.get('reused_M0'):continue
                prod=Path(c['total']['path']).parent if c.get('total') else None
                if not prod:continue
                r=read_yaml(prod/'receipt.yaml');run=json_read(prod.parent/'RUN.json')
                runtime.append(dict(array=array,branch=branch,cycle=c['cycle_index'],wall=r['wall_seconds'],cpu=r['cpu_seconds'],
                    rss_bytes=r['peak_rss_native_units']*(1 if r['peak_rss_unit']=='bytes' else 1024),
                    learn=sum(s['learning_seconds'] for s in r['segments']),apply=sum(s['application_seconds'] for s in r['segments']),
                    prepare=sum(s['preparation_seconds'] for s in r['segments']),MAP=r['MAP_seconds'],
                    serialization=r.get('serialization_seconds'),verification=r.get('verification_seconds'),
                    loading_projection_serialization=sum(s['loading_projection_serialization_seconds'] for s in r['segments']),
                    diagnostics=c.get('diagnostic_seconds'),output_bytes=run['output_bytes']))
    allresults={};texts=['# POINT 152389 C+D finite response and robustness','',
        'Brightness: mJy/nominal beam. Offsets: tangent east/north arcsec. Apertures: mJy/nominal beam × arcsec²; no aperture/flux correction.',
        'Same seven-parameter unweighted Gaussian plus constant, original fixed template/apertures and frozen admission scale. No signed-difference source fits.',
        '', '| Array | Branch | Status | Last cycle | Reason |','|---|---|---|---:|---|']
    for s in status:texts.append(f"| {s['array']} | {s['branch']} | {s['status']} | {s.get('completed','—')} | {s.get('reason') or ''} |")
    for array in ARRAYS:
        fixed=fixeds[array];d=report/array;d.mkdir();ref=summaries.get(('REF',array))
        if not ref or not ref['ready'] or ref['completed_cycle']!=10:continue
        refs={c['cycle_index']:c for c in ref['histories']};results={};map_images={}
        for branch in BRANCHES[1:]:
            summary=summaries.get((branch,array))
            if not summary:continue
            dest=d/branch;dest.mkdir();rows=[];injection_hashes=None
            for c in summary['histories']:
                k=c['cycle_index'];rc=refs[k]
                if not c.get('total'):
                    save_json(dest/f'cycle-{k:02d}-unavailable.json',dict(cycle=k,branch=branch,
                        status='unavailable native product',stop_reason=c.get('stop_reason')))
                    continue
                bp=verified(c['total']);rp=verified(rc['total'])
                b,bh=load(bp);r,rh=load(rp)
                br,rr,h,place=aligned(b['PRIMARY'],bh,r['PRIMARY'],rh)
                bv=place(b['SCI_SUPPORT'],bh)==1;rv=place(r['SCI_SUPPORT'],rh)==1;common=bv&rv
                delta=np.where(common,br-rr,np.nan)
                lost=rv&~bv;gained=bv&~rv
                norm_changed=common & (place(b['NORMALIZATION'],bh)!=place(r['NORMALIZATION'],rh))
                row=dict(cycle=k,branch=branch,measurement=c['measurement'],reference_measurement=rc['measurement'],
                    support=dict(common=int(common.sum()),lost=int(lost.sum()),gained=int(gained.sum()),
                        changed_normalization=int(norm_changed.sum())),
                    map_difference_rms=float(np.sqrt(np.mean(delta[common]**2))),
                    map_difference_max_abs=float(np.max(abs(delta[common]))),candidate_turnover=c.get('candidate_turnover'),native_receipt=c.get('receipt'),
                    PTC_states=c.get('PTC_segments',rc.get('PTC_segments',[])))
                vbase=vector_measurement(rc['measurement']);vbranch=vector_measurement(c['measurement'])
                if vbase is not None and vbranch is not None:
                    row['measurement_increments']=dict(zip(MEASUREMENT_NAMES,vbranch-vbase))
                shoulder,ih=load(verified(owner['injections'][array]['C-shoulder']));sh=shoulder['INJECTION']
                row.update(candidate_comparison(c,rc,sh))
                ds,dv,_,dh,_=source_patch(delta,common,np.ones(delta.shape),h)
                row['total_shoulder_projection_domain']=shape_projection(ds,dv,sh)
                row['total_shoulder_projection']=row['total_shoulder_projection_domain']['value']
                row['profile_radius_arcsec']=((fixed.edges[:-1]+fixed.edges[1:])/2).tolist()
                if branch.startswith('D-') and k==10:map_images[branch,k]=(ds,None,None)
                row['difference_profile']=[float(np.mean(ds[m])) if np.any(m) else None for m in
                    [dv&(fixed.radius>=lo)&(fixed.radius<hi) for lo,hi in zip(fixed.edges[:-1],fixed.edges[1:])]]
                planes=[('DELTA',delta,'mJy/nominal beam'),('COMMON',common.astype('u1'),'1'),
                        ('LOST',lost.astype('u1'),'1'),('GAINED',gained.astype('u1'),'1')]
                if branch.startswith('C-'):
                    raster,ih=load(verified(owner['injections'][array][branch]))
                    known,kv,kh,audit=injection_reference(rp.parent,bp.parent,raster['INJECTION'],ih)
                    hashes=[s['files']['injection.f64'] for s in read_yaml(bp.parent/'receipt.yaml')['segments']]
                    if injection_hashes is None:injection_hashes=hashes
                    require(hashes==injection_hashes,'injection changed/accumulated across cycles')
                    row['projection_audit']=audit;ki=place(known,kh)
                    row['map_response']=response_metrics(delta,ki,common)
                    planes.extend([('KNOWN',ki,'mJy/nominal beam'),('RESPONSE_RESIDUAL',np.where(common,delta-ki,np.nan),'mJy/nominal beam')])
                    bs,bsv,_,_=patch_map(bp);rsig,rsv,_,_=patch_map(rp)
                    isg,iv,_,_,_=source_patch(known,kv,np.ones(known.shape),kh)
                    row['measurement_response']=measurement_response(fixed,rsig,bs,bsv&rsv&iv,isg)
                    row['state_transition']=bool(lost.any() or gained.any() or norm_changed.any() or
                        any(p.get('output_cause_changed',0) or p['converged']!=p['parent_converged'] for p in row['PTC_states']))
                    if row['state_transition']:
                        row['response_qualification']='common-support diagnostic with downstream state change; not brightness-only response'
                    if k in (0,10):map_images[branch,k]=(ds,isg,ds-isg)
                if len(rows) and vbranch is not None:
                    old=vector_measurement(rows[-1]['measurement'])
                    if old is not None:row['previous_cycle_measurement_drift']=dict(zip(MEASUREMENT_NAMES,vbranch-old))
                write_fits(dest/f'cycle-{k:02d}.fits',h,planes,companion='POINT-CD-DIAGNOSTIC')
                save_json(dest/f'cycle-{k:02d}.json',row);rows.append(row)
            results[branch]=rows
        # Even finite difference on common supported pixels, never a source fit.
        plus=summaries.get(('C-core-plus',array),{}).get('histories',[])
        minus={r['cycle_index']:r for r in summaries.get(('C-core-minus',array),{}).get('histories',[])}
        even=[]
        for p in plus:
            k=p['cycle_index'];m=minus.get(k)
            if not m or not p.get('total') or not m.get('total'):continue
            ps,pv,_,h=patch_map(verified(p['total']));ms,mv,_,_=patch_map(verified(m['total']));rs,rv,_,_=patch_map(verified(refs[k]['total']))
            valid=pv&mv&rv;v=np.where(valid,ps+ms-2*rs,np.nan)
            even.append(dict(cycle=k,rms=float(np.sqrt(np.mean(v[valid]**2))),max_abs=float(np.max(abs(v[valid])))))
            write_fits(d/f'even-{k:02d}.fits',h,[('EVEN',v,'mJy/nominal beam'),('VALID',valid.astype('u1'),'1')],companion='POINT-CD-EVEN')
        allresults[array]=dict(branches=results,even=even,measurement_units=dict(zip(MEASUREMENT_NAMES,MEASUREMENT_UNITS)),even_domain='common supported pixels in the 41x41 source stamp')
        measurement_tables(d,array,results)
        save_json(d/'results.json',allresults[array]);draw(d,array,results,map_images,even,fixed)
        texts.extend(['',f'## {array}', '',f'[Map response and profiles]({d}/response.png) · [Signed source-region maps]({d}/maps.png) · [Seed and threshold comparisons]({d}/robustness.png) · [Unthresholded D difference maps]({d}/D-difference-maps.png) · [Measurement comparison tables]({d}/MEASUREMENTS.md) · [All numerical records]({d}/results.json)',
            '', '| Intervention | Cycle | g | η | Actual peak increment | Ideal peak increment | Peak discrepancy |','|---|---:|---:|---:|---:|---:|---:|'])
        for b,rows in results.items():
            if not b.startswith('C-'):continue
            for row in rows:
                if row['cycle'] not in (0,1,3,5,10):continue
                q=row['map_response'];comp=row['measurement_response']['comparisons'];p=comp[0] if comp else {}
                texts.append(f"| {b} | {row['cycle']} | {q['g']:.6f} | {q['eta']:.6f} | {fmt(p.get('actual_increment'))} | {fmt(p.get('ideal_increment'))} | {fmt(p.get('absolute_discrepancy'))} |")
        texts.extend(['','| Initialization / method | Cycle | Peak − REF | Background − REF | Total shoulder projection | Replacement shoulder projection | Candidate area arcsec² |','|---|---:|---:|---:|---:|---:|---:|'])
        for b,rows in results.items():
            if not b.startswith('D-'):continue
            for row in rows:
                if row['cycle'] not in (1,3,5,10):continue
                q=row.get('measurement_increments',{})
                texts.append(f"| {b} | {row['cycle']} | {fmt(q.get('peak'))} | {fmt(q.get('background'))} | {row['total_shoulder_projection']:.6f} | {fmt(row['replacement_shoulder_projection'])} | {row['candidate_area_arcsec2']} |")
        texts.extend(['']+fit_control_notes(results))
    save_json(report/'summary.json',dict(source_revision=revision,prospective=bound(out/'freeze/prospective.json'),
        status=status,arrays=allresults,runtime=runtime,report_seconds=time.perf_counter()-began))
    texts.extend(['','This is finite full-procedure response at the saved post-RTC/CAL boundary. Paired maps share observed noise; changes include altered cleaning of original source/background. g and η are unweighted diagnostics, not uncertainty or significance.',
        'The ideal final-map addition is a measurement control only. Ratios with |ideal increment| ≤1e−8 in their native units are unavailable. Ellipse tensor and eigenwidths avoid position-angle gauge ambiguity. Raw aperture and fitted-background increments are separate in each cycle JSON.',
        'Seed endpoint spread and remaining drift are not an uncertainty distribution or evidence of asymptotic fixed points. Threshold5 remains a separate method comparison. Original and experimental m10 remain unapplied.'])
    (report/'REPORT.md').write_text('\n'.join(texts)+'\n');manifest(report)


def fmt(v):return 'unavailable' if v is None or not np.isfinite(v) else f'{v:.6g}'


def draw(out,array,results,images,even,fixed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for branch,rows in results.items():
        if not branch.startswith('C-'):continue
        k=[r['cycle'] for r in rows];axs[0,0].plot(k,[r['map_response']['g'] for r in rows],'.-',label=branch)
        axs[0,1].plot(k,[r['map_response']['eta'] for r in rows],'.-',label=branch)
        for row in rows:
            if row['cycle'] in (0,10):axs[1,0].plot(row['profile_radius_arcsec'],row['difference_profile'],label=f'{branch} T{row["cycle"]}')
    axs[0,0].set_xlabel('cycle (0 = M0)');axs[0,1].set_xlabel('cycle (0 = M0)')
    axs[0,0].axhline(1,color='k',lw=.7);axs[0,0].set_ylabel('shape projection g');axs[0,1].set_ylabel('relative residual norm η')
    axs[1,0].set(xlabel='radius from original centroid [arcsec]',ylabel='signed Δ brightness [mJy/nominal beam]')
    axs[1,1].plot([r['cycle'] for r in even],[r['rms'] for r in even],'.-');axs[1,1].set(xlabel='cycle (0 = M0)',ylabel='even response RMS, 41×41 stamp [mJy/nominal beam]')
    for ax in axs.flat:ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=7);fig.suptitle(array+' — finite paired map response')
    fig.savefig(out/'response.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(3,4,figsize=(13,10),constrained_layout=True)
    extent=[41,-41,-41,41]
    for i,b in enumerate(BRANCHES[1:4]):
        zero=images.get((b,0));ten=images.get((b,10))
        planes=[zero[1] if zero else None,zero[0] if zero else None,ten[0] if ten else None,ten[2] if ten else None]
        for j,(v,title) in enumerate(zip(planes,['known I_map','Δ at M0','Δ at T10','Δ−I_map at T10'])):
            ax=axs[i,j];ax.set_title(b+'\n'+title,fontsize=9)
            if v is None:ax.text(.5,.5,'unavailable',ha='center',transform=ax.transAxes);continue
            vmax=max(float(np.nanmax(abs(v))),1e-8)
            im=ax.imshow(v,origin='lower',extent=extent,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
            fig.colorbar(im,ax=ax,shrink=.7);ax.set_xlabel('east [arcsec]');ax.set_ylabel('north [arcsec]')
    fig.suptitle(array+' — signed brightness; each panel has its own color scale [mJy/nominal beam]')
    fig.savefig(out/'maps.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for b,rows in results.items():
        if not b.startswith('D-'):continue
        k=[r['cycle'] for r in rows];p=[r.get('measurement_increments',{}).get('peak',np.nan) for r in rows]
        target=axs[1,1] if b=='D-threshold-5' else axs[0,0];target.plot(k,p,'.-',label=b)
        if b!='D-threshold-5':axs[1,0].plot(k,[r['candidate_area_arcsec2'] for r in rows],'.-',label=b)
        if b=='D-seed-shoulder':
            axs[0,1].plot(k,[r['total_shoulder_projection'] for r in rows],'.-',label='seed-only: total')
            axs[0,1].plot(k,[r['replacement_shoulder_projection'] for r in rows],'.-',label='seed-only: replacement')
    c=results.get('C-shoulder',[])
    axs[0,1].plot([r['cycle'] for r in c],[r['total_shoulder_projection'] for r in c],'.-',label='data shoulder: total')
    labels=['seed peak − REF [mJy/nominal beam]','projection onto shoulder raster','candidate area [arcsec²]','threshold5 peak − REF [mJy/nominal beam]']
    for ax,label in zip(axs.flat,labels):
        ax.set(xlabel='cycle (0 = M0)',ylabel=label);ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    fig.suptitle(array+' — initialization memory separate from threshold sensitivity')
    fig.savefig(out/'robustness.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(14,4.5),constrained_layout=True)
    for ax,b in zip(axs,BRANCHES[4:]):
        v=images.get((b,10),(None,))[0]
        ax.set_title(b+'\nT10 − REF T10',fontsize=10)
        if v is None:ax.text(.5,.5,'unavailable',ha='center',transform=ax.transAxes);continue
        vmax=max(float(np.nanmax(abs(v))),1e-12)
        im=ax.imshow(v,origin='lower',extent=extent,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        fig.colorbar(im,ax=ax,shrink=.8);ax.set_xlabel('east [arcsec]');ax.set_ylabel('north [arcsec]')
    fig.suptitle(array+' — unthresholded map differences; independent color scales [mJy/nominal beam]')
    fig.savefig(out/'D-difference-maps.png',dpi=150);plt.close(fig)



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    assess(p.parse_args().output.resolve())
