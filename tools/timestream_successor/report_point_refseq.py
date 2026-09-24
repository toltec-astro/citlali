#!/usr/bin/env python3
"""Saved paired a2000 results only: original measurements, no new source fits."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from run_point_refseq import *
from point_response_cd import vector_measurement,MEASUREMENT_NAMES
from report_point_frozen_shoulder import table,fmt


def candidate_evidence(candidate,stage,ctx):
    c,h=load(candidate['path']);signal,support,_,_=patch_map(stage['total']['path']);p=np.asarray(stage['measurement']['fit']['parameters'])
    v=c['VALID'].astype(bool);m=c['ADMITTED'].astype(bool);disk=np.hypot(ctx['x'],ctx['y'])<=40
    require(np.array_equal(v,disk&support&ctx['noise_valid']&np.isfinite(signal)&np.isfinite(ctx['scale'])&(ctx['scale']>0)),'candidate validity changed')
    ratio=np.full(signal.shape,np.nan);ratio[v]=(signal[v]-p[6])/ctx['scale'][v];check=close(ratio[v],c['RATIO'][v],'candidate ratio')
    labels,_=label(v&(ratio>=3),structure=np.ones((3,3)))
    col=int(np.floor(p[1]/(h['CDELT1']*3600)+h['CRPIX1']-.5));row=int(np.floor(p[2]/(h['CDELT2']*3600)+h['CRPIX2']-.5));anchor=int(labels[row,col]);require(anchor>0 and np.array_equal(m,labels==anchor),'connected candidate mismatch')
    return c,dict(candidate=candidate,formation_stage=stage['cycle_index'],application_stage=stage['cycle_index']+1 if stage['cycle_index']<10 else None,
        selected_pixels=int(m.sum()),area_arcsec2=int(m.sum())*4,fitted_constant=float(p[6]),source_anchor=[row,col],source_component=anchor,ratio_check=check,membership_exact=True)


def new_candidate_evidence(stage,ctx):
    if not stage.get('ready') or not stage.get('candidate_available'):
        return None,dict(status='unavailable',reason=stage.get('stop_reason'),
                         preserved_candidate=stage.get('candidate'),selected_pixels=None,area_arcsec2=None)
    c,record=candidate_evidence(stage['candidate'],stage,ctx)
    record['continuation_ready']=True
    return c,record


def pair_candidates(new,old,ctx):
    require(np.array_equal(new['VALID'],old['VALID']),'candidate comparison support transition')
    result=selection_difference(new,old);a=new['ADMITTED'].astype(bool);b=old['ADMITTED'].astype(bool);pixels=[]
    for y,x in np.argwhere(a!=b):
        pixels.append(dict(row=int(y),column=int(x),east=float(ctx['x'][y,x]),north=float(ctx['y'][y,x]),new_selected=bool(a[y,x]),old_selected=bool(b[y,x]),new_ratio=float(new['RATIO'][y,x]),old_ratio=float(old['RATIO'][y,x]),new_margin=float(new['RATIO'][y,x]-3),old_margin=float(old['RATIO'][y,x]-3)))
    return dict(**result,changed_membership=pixels)


def report(out):
    started=time.perf_counter();revision=source();fr=json.loads((out/'FREEZE.json').read_text());w=json.loads(verified(fr['prospective']).read_text());auth=auth_work(w)
    owner=auth.json(CD/'freeze/prospective.json');ctx,fixed=contexts(auth,owner)
    branches={b:json.loads((out/'run'/b/'SUMMARY.json').read_text()) for b in BRANCHES};require(all(b['complete'] for b in branches.values()),'complete pair required for this report')
    dest=out/'report';dest.mkdir();rows=[];candall={}
    for k in range(11):
        stages={b:branches[b]['stages'][k] for b in BRANCHES};r,h=[load(stages[b]['total']['path'])[0] for b in BRANCHES];header=load(stages[BRANCHES[0]]['total']['path'])[1]
        old,oh=auth.fits(CD/f'report/a2000/C-shoulder/cycle-{k:02d}.fits');oldrec=auth.json(CD/f'report/a2000/C-shoulder/cycle-{k:02d}.json')
        domain=old['COMMON'].astype(bool)&np.isfinite(old['KNOWN']);known=old['KNOWN'];delta=h['PRIMARY']-r['PRIMARY'];adaptive=old['DELTA'];e=delta-adaptive
        require(np.all(r['SCI_SUPPORT'][domain]) and np.all(h['SCI_SUPPORT'][domain]),'original headline domain unavailable')
        measures={name:map_measures(v,known,domain,ctx,header) for name,v in [('SEQ',delta),('ADAPTIVE',adaptive),('E',e)]}
        for m in measures.values():m['norm_over_known']=float(np.sqrt(m['g']**2+m['orthogonal_distortion']**2))
        row=dict(stage=k,measurements=measures,change={n:measures['SEQ'][n]-measures['ADAPTIVE'][n] for n in ('g','eta','orthogonal_distortion')},
            branch_measurements={b:stages[b]['measurement'] for b in BRANCHES},original_ideal=oldrec['measurement_response']['ideal_map_addition'],
            candidates={},candidate_changes={},estimator_response=[],raw_background={})
        bv=vector_measurement(stages[BRANCHES[0]]['measurement']);hv=vector_measurement(stages[BRANCHES[1]]['measurement']);iv=vector_measurement(row['original_ideal'])
        ov=vector_measurement(oldrec['measurement_response']['reference']);av=vector_measurement(oldrec['measurement_response']['actual'])
        close(bv,ov,'saved ideal reference applicability',0,0)
        if iv is not None and hv is not None:
            for name,actual,previous,ideal in zip(MEASUREMENT_NAMES,hv-bv,av-ov,iv-ov):
                row['estimator_response'].append(dict(name=name,sequence_increment=float(actual),adaptive_increment=float(previous),ideal_increment=float(ideal),sequence_minus_ideal=float(actual-ideal),sequence_minus_adaptive=float(actual-previous),ratio=float(actual/ideal) if np.isfinite(actual+ideal) and abs(ideal)>1e-8 else None))
        for branch in BRANCHES:
            new,nr=new_candidate_evidence(stages[branch],ctx)
            prev=w['old_stages'][branch]['histories'][k];oldc,orr=candidate_evidence(prev['candidate'],prev,ctx)
            row['candidates'][branch]=dict(new=nr,old=orr);row['candidate_changes'][branch]=pair_candidates(new,oldc,ctx) if new is not None else dict(retained=None,added=None,removed=None,retained_brightness_rms_change=None,status='unavailable')
            candall[(branch,k)]=(new,oldc)
        row['paired_new_candidates']=pair_candidates(candall[(BRANCHES[1],k)][0],candall[(BRANCHES[0],k)][0],ctx) if all(candall[(b,k)][0] is not None for b in BRANCHES) else dict(status='unavailable')
        cb=hv[8]-bv[8] if hv is not None else None
        for name,raw in measures['SEQ']['linear'].items():
            if name=='template':weight=float(ctx['q'].sum());actual=hv[9]-bv[9] if hv is not None else None
            else:
                i=['aperture_1','aperture_1p5','aperture_2'].index(name);weight=float(np.count_nonzero(ctx['radius']<=[1,1.5,2][i]*ctx['major'])*4);actual=hv[10+i]-bv[10+i] if hv is not None else None
            row['raw_background'][name]=dict(raw=raw,background=-weight*cb if cb is not None else None,source_relative=float(actual) if actual is not None else None,check=close(raw-weight*cb,actual,'new raw/background '+name) if actual is not None else 'unavailable fitted background')
        if k:
            prior=rows[-1];row['previous_stage_drift']={n:measures['SEQ'][n]-prior['measurements']['SEQ'][n] for n in ('g','eta','orthogonal_distortion')}
            row['branch_estimator_drift']={b:{n:float(x) for n,x in zip(MEASUREMENT_NAMES,vector_measurement(stages[b]['measurement'])-vector_measurement(branches[b]['stages'][k-1]['measurement']))} for b in BRANCHES if vector_measurement(stages[b]['measurement']) is not None and vector_measurement(branches[b]['stages'][k-1]['measurement']) is not None}
        planes=dict(KNOWN=known,ADAPTIVE=adaptive,SEQUENCE=delta,E=e,ADAPTIVE_RESIDUAL=adaptive-known,SEQUENCE_RESIDUAL=delta-known,
            ADAPTIVE_ORTHOGONAL=adaptive-measures['ADAPTIVE']['g']*known,SEQUENCE_ORTHOGONAL=delta-measures['SEQ']['g']*known)
        write_fits(dest/f'stage-{k:02d}.fits',header,[(n,v,'mJy/nominal beam') for n,v in planes.items()]+[('COMMON',domain.astype('u1'),'1')],companion='POINT-REFSEQ-RESPONSE')
        save_json(dest/f'stage-{k:02d}.json',row);rows.append(row)
    endpoints=json.loads((out/'endpoints/SUMMARY.json').read_text())
    figures(dest,rows,candall,ctx,out)
    lines=['# POINT 152389 a2000 reference-basis-sequence counterfactual','',
        'Two completed trajectories, stage-specific original REF bases with fresh branch means/coefficients and ordinary threshold-3 candidates. Exact regenerated REF replay gates the shoulder. No new observation learning, m10 application or cycle11. Finite dataset-specific intervention; no default adoption, uncertainty interpretation or absolute flux correction.','',
        '[Response trajectories](response.png) · [T10 matched maps](T10-maps.png) · [Candidate histories and masks](candidates.png) · [Endpoint/history accounting](T10-accounting.png)','',
        'g is unweighted known-shape projection; eta is total normalized residual norm; rho is shape-orthogonal distortion. The full original per-stage common supported domain includes known-shape zeros. Map panels are41×41-pixel source stamps, east left/north up; metrics use full support.','',
        table(['Stage','g seq','g adaptive','Δg','eta seq','eta adaptive','rho seq','rho adaptive','||E||/||known||'],[[r['stage'],fmt(r['measurements']['SEQ']['g']),fmt(r['measurements']['ADAPTIVE']['g']),fmt(r['change']['g'])]+[fmt(r['measurements'][b][n]) for n in ('eta','orthogonal_distortion') for b in ('SEQ','ADAPTIVE')]+[fmt(r['measurements']['E']['norm_over_known'])] for r in rows]),'',
        '## Candidate formation histories','',
        'Rows refer to newly formed mk afterTk (M0 for0), applied atT(k+1). m10 is saved and unapplied. Membership changes below compare new shoulder with old adaptive shoulder; every reference comparison is preserved in JSON, including any nonzero residual. Actual ratios/margins and connected source membership are checked and archived for all22 new candidates.','',
        table(['Candidate','New H pixels','Old H pixels','Retained','Added','Removed','Retained brightness RMS change'],[[r['stage'],r['candidates'][BRANCHES[1]]['new']['selected_pixels'],r['candidates'][BRANCHES[1]]['old']['selected_pixels']]+[r['candidate_changes'][BRANCHES[1]][n] for n in ('retained','added','removed')]+[fmt(r['candidate_changes'][BRANCHES[1]]['retained_brightness_rms_change'])] for r in rows]),'',
        '## M0/T10 independent accounting','',
        'DC reuses the authenticated original centered direct map after checking operator/domain identity. FC_SEQ and HISTORY are native applications to actual operand differences, never closure remainders. E=−CQ_old+HISTORY. T10 uses appliedm9; reference-branch history pieces are retained.','']
    for ep in endpoints['results']:
        lines += [f"Stage{ep['stage']}",'',table(['Term','Projection','Raw template','Raw aperture2'],[[n,fmt(ep['measurements'][n]['g']),fmt(ep['measurements'][n]['linear']['template']),fmt(ep['measurements'][n]['linear']['aperture_2'])] for n in ('DC','FC','AVOIDED','HISTORY','E','HISTORY_H_RETAIN','HISTORY_H_ADD','HISTORY_H_REMOVE','HISTORY_R_RETAIN','HISTORY_R_ADD','HISTORY_R_REMOVE')]),'',f"Closure checks: {ep['checks']}",'']
    lines += ['## Original estimator response at every stage','', 'Fits are only on full new branch maps. Ideal comparators are the saved C+D map-addition controls. Failed/constrained flags remain in stage JSON. Width eigenvalues and ellipse tensors are gauge safe; no signed component is fitted. Peak/template/background: mJy/nominal beam; centroid/FWHM: arcsec; ellipse tensor: arcsec²; apertures: mJy/nominal-beam arcsec². Ratios are unavailable when the ideal denominator is ≤1e−8 in its native units.','']
    for row in rows:
        lines += [f"### Stage{row['stage']}",'',f"Fit status/constraints: { {b:(m['fit']['status'],m['fit'].get('constrained',False)) for b,m in row['branch_measurements'].items()} }; ideal: {row['original_ideal']['fit']['status']}, constrained={row['original_ideal']['fit'].get('constrained',False)}",'',table(['Measurement','New paired increment','Old adaptive increment','Saved ideal increment','New − ideal','New − old','Ratio'],[[q['name']]+[fmt(q[n]) for n in ('sequence_increment','adaptive_increment','ideal_increment','sequence_minus_ideal','sequence_minus_adaptive','ratio')] for q in row['estimator_response']]),'',table(['Linear measurement','Raw increment','Fitted-background term','Source-relative increment'],[[n]+[fmt(q[j]) for j in ('raw','background','source_relative')] for n,q in row['raw_background'].items()]),'']
    lines += ['## Scope and runtime','', 'Original a2000 background-model qualification persists. Paired deterministic differences are not independent noise realizations. No A+B errors, significance, quadrature combination or total uncertainty budget is attached. eta/rho are not PSF broadening or validation of the final5% photometry/PSF goals.','']
    for b,s in branches.items():lines.append(f"{b}: {s['maps_completed']} maps, {s['candidates_available']} candidates, wall{s['wall_seconds']:.3f}s, parentCPU{s['parent_cpu_seconds']:.3f}s. Three segment Apply workers; publication/MAP accumulation ordered, one library thread. Aggregate memory not measured.")
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save_json(dest/'SUMMARY.json',dict(source_revision=revision,rows=rows,endpoint_summary=endpoints,report_seconds=time.perf_counter()-started,original_verification_seconds=auth.verify_all()))
    manifest(dest)


def figures(dest,rows,candidates,ctx,out):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,n,title in zip(axs,('g','eta','orthogonal_distortion'),('Shoulder projection g','Total distortion eta','Orthogonal distortion rho')):
        for b in ('ADAPTIVE','SEQ'):ax.plot(range(11),[r['measurements'][b][n] for r in rows],'.-',label=b)
        ax.set(xlabel='map-producing stage (0=M0)',title=title);ax.grid(alpha=.2);ax.legend()
    fig.savefig(dest/'response.png',dpi=150,bbox_inches='tight');plt.close(fig)
    maps,header=load(dest/'stage-10.fits');domain=maps['COMMON'].astype(bool);extent=[41,-41,-41,41]
    names=['KNOWN','ADAPTIVE','SEQUENCE','E','ADAPTIVE_RESIDUAL','SEQUENCE_RESIDUAL','ADAPTIVE_ORTHOGONAL','SEQUENCE_ORTHOGONAL']
    patches={n:source_patch(maps[n],domain,np.ones(domain.shape),header)[0] for n in names}
    signal_scale=max(float(np.nanmax(abs(patches[n]))) for n in names[:3]);error_scale=max(float(np.nanmax(abs(patches[n]))) for n in names[3:])
    fig,axs=plt.subplots(2,4,figsize=(14,8),constrained_layout=True)
    for ax,n in zip(axs.flat,names):
        scale=signal_scale if n in names[:3] else error_scale;im=ax.imshow(patches[n],origin='lower',extent=extent,cmap='RdBu_r',vmin=-scale,vmax=scale);fig.colorbar(im,ax=ax,shrink=.7);ax.set(title=n,xlabel='east [arcsec]',ylabel='north [arcsec]')
    fig.suptitle('a2000 T10 — shared signal scale; shared difference/distortion scale [mJy/nominal beam]');fig.savefig(dest/'T10-maps.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for i,(name,c) in enumerate(zip(('old H applied m9','new H applied m9'),candidates[(BRANCHES[1],9)][::-1])):
        axs[0,i].imshow(c['ADMITTED'],origin='lower',extent=extent,cmap='Greys',vmin=0,vmax=1);axs[0,i].contour(ctx['x'],ctx['y'],ctx['h'],levels=np.nanmax(ctx['h'])*np.array([.1,.3,.6,.9]),colors='tab:red');axs[0,i].set(title=name,xlabel='east [arcsec]',ylabel='north [arcsec]')
    for which in ('new','old'):axs[1,0].plot(range(11),[r['candidates'][BRANCHES[1]][which]['area_arcsec2'] for r in rows],'.-',label=which+' shoulder')
    axs[1,0].set(title='Formed candidate area',xlabel='candidate index (m10 unapplied)',ylabel='arcsec²');axs[1,0].legend()
    for n in ('added','removed'):axs[1,1].plot(range(11),[r['candidate_changes'][BRANCHES[1]][n] for r in rows],'.-',label=n)
    axs[1,1].set(title='New vs old shoulder support',xlabel='formed candidate index',ylabel='pixels');axs[1,1].legend();fig.savefig(dest/'candidates.png',dpi=150,bbox_inches='tight');plt.close(fig)
    a,h=load(out/'endpoints/10/accounting.fits');ep=json.loads((out/'endpoints/10/result.json').read_text());dm=a['COMMON'].astype(bool)
    fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,n in zip(axs.flat[:4],('AVOIDED','HISTORY','E','CLOSURE')):
        v=source_patch(a[n],dm,np.ones(dm.shape),h)[0];scale=max(float(np.nanmax(abs(v))),1e-15);im=ax.imshow(v,origin='lower',extent=extent,cmap='RdBu_r',vmin=-scale,vmax=scale);fig.colorbar(im,ax=ax,shrink=.7);ax.set(title=n,xlabel='east [arcsec]',ylabel='north [arcsec]')
    for n in ('AVOIDED','HISTORY','E'):
        axs[1,1].plot(np.arange(1,41,2),ep['measurements'][n]['profile'],label=n)
        axs[1,2].plot([1,1.5,2],[ep['measurements'][n]['linear'][j] for j in ('aperture_1','aperture_1p5','aperture_2')],'.-',label=n)
    axs[1,1].set(title='Signed profiles',xlabel='radius from original center [arcsec]',ylabel='mJy/nominal beam');axs[1,2].set(title='Raw fixed apertures',xlabel='radius / original M0 major FWHM',ylabel='mJy/nominal beam × arcsec²')
    for ax in axs[1,1:]:ax.axhline(0,color='gray',lw=.5);ax.legend()
    fig.suptitle('a2000 T10: E = avoided old subspace term + changed history; independent signed scales');fig.savefig(dest/'T10-accounting.png',dpi=150,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.output)
