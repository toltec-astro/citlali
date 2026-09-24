#!/usr/bin/env python3
"""Saved-state component tables and figures; no source fitting or native calls."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from point_frozen_shoulder import *
from point_fruit_trajectory import manifest,bound
from run_point_response_cd import source


def fmt(v):return 'unavailable' if v is None else f'{v:.8g}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def figures(out,array,rows,ctx):
    import matplotlib.pyplot as plt
    good=[r for r in rows if r['status']=='pass'];ks=[r['cycle'] for r in good];extent=[41,-41,-41,41]
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for names,ax,title in [(PRIMARY,axs[0,0],'All-state-frozen reference anchor'),(CENTERED,axs[0,1],'Original mean rule recomputed; basis fixed'),(('F_RETAIN','F_ADD','F_REMOVE'),axs[1,0],'Primary feedback-effect partition'),(('FC_RETAIN','FC_ADD','FC_REMOVE'),axs[1,1],'Mean-aware feedback-effect partition')]:
        for n in names:ax.plot(ks,[r['measurements'][n]['g'] for r in good],'.-',label=n)
        ax.plot(ks,[r['measurements']['DELTA']['g'] for r in good],'k--',label='adaptive total');ax.axhline(0,color='grey',lw=.5);ax.set(title=title,xlabel='saved output cycle (0 = M0)',ylabel='signed projection onto known shoulder');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle(array+' — conditional signed accounting; terms can cancel');fig.savefig(out/'projections.png',dpi=150,bbox_inches="tight",pad_inches=.15);plt.close(fig)
    final=next((r for r in good if r['cycle']==10),None)
    if not final:return
    maps,h=load(out.parents[1]/'pairs'/array/'10/components.fits');domain=maps['COMMON'].astype(bool)
    selected=['KNOWN','DELTA','D','F','CMU','CQ','DC','FC','CENTER','PROJECTION','ORTHOGONAL','CLOSURE']
    fig,axs=plt.subplots(3,4,figsize=(15,11),constrained_layout=True)
    for ax,n in zip(axs.flat,selected):
        v,_,_,_,_=source_patch(maps[n],domain,np.ones(domain.shape),h);scale=max(float(np.nanmax(abs(v))),1e-15)
        im=ax.imshow(v,origin='lower',extent=extent,cmap='RdBu_r',vmin=-scale,vmax=scale);fig.colorbar(im,ax=ax,shrink=.72);ax.set(title=n,xlabel='east [arcsec]',ylabel='north [arcsec]')
    fig.suptitle(array+' T10 — independent signed scales [mJy/nominal beam]; 41×41-pixel stamp, full-support metrics in tables')
    fig.savefig(out/'T10-components.png',dpi=150,bbox_inches="tight",pad_inches=.15);plt.close(fig)
    applied,_=load(out.parents[1]/'pairs'/array/'10/applied-masks.fits');fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,b in zip(axs[0,:2],['R','H']):
        ax.imshow(applied['MASK_'+b],origin='lower',extent=extent,cmap='Greys',vmin=0,vmax=1);ax.contour(ctx['x'],ctx['y'],ctx['h'],levels=np.nanmax(ctx['h'])*np.array([.1,.3,.6,.9]),colors='tab:red',linewidths=.8);ax.set(title='applied '+b+' m9 mask; shoulder contours',xlabel='east [arcsec]',ylabel='north [arcsec]')
    for n in ('DELTA','D','F','CMU','CQ'):axs[0,2].plot(np.arange(1,41,2),final['measurements'][n]['profile'],label=n)
    axs[0,2].set(title='Signed profiles from original centroid',xlabel='radius [arcsec]',ylabel='mJy/nominal beam');axs[0,2].legend(fontsize=8);axs[0,2].axhline(0,color='grey',lw=.5)
    for n in ('F_RETAIN','F_ADD','F_REMOVE'):axs[1,0].plot(np.arange(1,41,2),final['measurements'][n]['profile'],label=n)
    axs[1,0].set(title='Propagated candidate pieces',xlabel='radius [arcsec]',ylabel='mJy/nominal beam');axs[1,0].legend(fontsize=8);axs[1,0].axhline(0,color='grey',lw=.5)
    for n in PRIMARY+('DELTA',):axs[1,1].plot([1,1.5,2],[final['measurements'][n]['linear'][k] for k in ['aperture_1','aperture_1p5','aperture_2']],'.-',label=n)
    axs[1,1].set(title='Raw fixed-aperture signed accounting',xlabel='radius / original M0 major FWHM',ylabel='mJy/nominal beam × arcsec²');axs[1,1].legend(fontsize=8)
    x=np.arange(4);axs[1,2].bar(x-.18,[final['measurements'][n]['g'] for n in PRIMARY],width=.36,label='REF anchor');axs[1,2].bar(x+.18,[final['measurements'][n]['g'] for n in REVERSE],width=.36,label='shoulder anchor');axs[1,2].set(xticks=x,xticklabels=PRIMARY,title='T10 anchor dependence',ylabel='signed projection');axs[1,2].legend(fontsize=8);axs[1,2].axhline(0,color='grey',lw=.5)
    fig.suptitle(array+' — historical applied masks, geometric overlap and propagated effects are distinct');fig.savefig(out/'T10-masks-profiles.png',dpi=150,bbox_inches="tight",pad_inches=.15);plt.close(fig)


def report(root):
    started=time.perf_counter();revision=source();dest=root/'report';require(not dest.exists(),'fresh report required');dest.mkdir()
    auth,owner=ancestors();run=json.loads((root/'RUN.json').read_text());records={a:sorted([json.loads(p.read_text()) for p in (root/'pairs'/a).glob('*/result.json')],key=lambda r:r['cycle']) for a in ARRAYS}
    text=['# POINT 152389 — saved frozen-PTC shoulder accounting','',
        'Conditional decomposition of completed REF/C-shoulder histories, not a new reduction, unique causal budget or uncertainty estimate. Each T10 uses its saved m9. All original and diagnostic m10 remain unapplied.','',
        'D: direct increment under all saved reference state; F: changed historical feedback operand including subtraction and restoration; CMU: saved mean change at fixed reference projection; CQ: learned-subspace change on the full shoulder residual. DC/FC regroup the original mean rule with the fixed basis. CENTER is the centering-only direct change; PROJECTION is the subsequent direct projection change.','',
        'Brightness and template units: mJy/nominal beam; raw apertures: mJy/nominal beam × arcsec². Original fixed weights/centers/radii and saved constants; no refits or flux correction. Projections use the exact C+D full common supported domain with known injection-map zeros retained. Plots show source stamps.','',table(['Array','Snapshot','Outcome','Reason'],[[a,r['cycle'],r['status'],r.get('reason','')] for a in ARRAYS for r in records[a]])]
    for a in ARRAYS:
        d=dest/a;d.mkdir();rows=records[a];good=[r for r in rows if r['status']=='pass'];ctx=saved_context(auth,owner,a);figures(d,a,rows,ctx)
        text+=['',f'## {a}','',f'[Projection trajectories]({a}/projections.png) · [T10 signed components]({a}/T10-components.png) · [Applied masks, profiles and apertures]({a}/T10-masks-profiles.png) · [Complete linear measurements]({a}/LINEAR.md)','',table(['Cycle','g adaptive','eta','Orthogonal distortion','D','F','CMU','CQ','DC','FC','CENTER','PROJECTION','Closure max abs'],[[r['cycle']]+[fmt(r['measurements']['DELTA'][n]) for n in ['g','eta','orthogonal_distortion']]+[fmt(r['measurements'][n]['g']) for n in PRIMARY+('DC','FC','CENTER','PROJECTION')]+[fmt(r['checks']['CLOSURE']['max_abs'])] for r in good])]
        linear=['# '+a+' fixed linear measurements','', 'Each component is measured with the original fixed linear weights or raw aperture sums. Nonlinear refitted-background differences are reported separately and never distributed among components.','']
        for r in good:
            k=r['cycle'];linear+=['',f'## Snapshot {k}','',table(['Component','g','Template','Aperture 1','Aperture 1.5','Aperture 2'],[[n,fmt(m['g'])]+[fmt(m['linear'][j]) for j in ['template','aperture_1','aperture_1p5','aperture_2']] for n,m in r['measurements'].items()]),'',table(['Saved estimator','Raw paired change','Fitted-background term','Saved source-relative change'],[[n]+[fmt(q[j]) for j in ['raw_difference','background_term','saved_estimator_difference']] for n,q in r['background']['terms'].items()])]
            if k:
                ad=r['admission'];linear+=['',f"Applied candidate pixel counts: {ad['counts']}. Geometric overlap only:",'',table(['Anchor','Brightness-area inside','Squared-brightness inside'],[[b]+[fmt(ad['branches'][b]['geometric_overlap'][j]['inside']) for j in ['brightness_area','squared_brightness']] for b in ['R','H']]),'',f'[Saved ratios and connected-component membership](../../pairs/{a}/{k:02d}/result.json)']
        (d/'LINEAR.md').write_text('\n'.join(linear)+'\n')
        final=next((r for r in good if r['cycle']==10),None)
        if final:
            text+=['','T10 reverse-anchor check:', '',table(['Term','REF anchor g','Shoulder anchor g','Shift'],[[n,fmt(final['measurements'][n]['g']),fmt(final['measurements']['REV_'+n]['g']),fmt(final['measurements']['REV_'+n]['g']-final['measurements'][n]['g'])] for n in PRIMARY]),'', 'Candidate effect partition (signed; no causal threshold attribution):','',table(['Piece','F projection','FC projection'],[[n,fmt(final['measurements']['F_'+n]['g']),fmt(final['measurements']['FC_'+n]['g'])] for n in ['RETAIN','ADD','REMOVE']])]
    native=[n for rows in records.values() for r in rows for n in r.get('native',[])];text+=['','## Runtime and validation scope','',f"Processed {sum(r['status']=='pass' for rows in records.values() for r in rows)}/15 pairs. Execution wall {run['wall_seconds']:.3f}s with {run['workers']} array workers and one numerical thread each, on {run['available_logical_cpus']} logical CPUs. Original-input verification before/after {run['before_verification_seconds']:.3f}/{run['after_verification_seconds']:.3f}s.",'',table(['Native aggregate','Value'],[[key,fmt(sum(n[key] for n in native))] for key in ['loading_verification_seconds','Apply_seconds','mean_seconds','MAP_seconds','serialization_seconds','wall_seconds','cpu_seconds']]+[['Maximum individual native-process RSS GiB',fmt(max((n['peak_rss_native_units']*(1 if n['peak_rss_unit']=='bytes' else 1024) for n in native),default=0)/2**30)]]),'',
        'The mean_seconds timer includes native preparation/grouping and centering. Summed native time is work, not campaign elapsed time. RSS is an individual-process high-water mark, not aggregate campaign memory. Metadata and parent loading, Python diagnostics and complete phase times are retained in RUN.json and pair receipts. Local cached AppleClang validation is supplemental, not Spack V2. Zero Learn/admission/source-fit calls are established by the restricted entry-point call graph and saved-state source equivalence; no runtime counter is claimed to intercept arbitrary external calls.','',
        'Inherited a1100 network3/scan5 remains failed/excluded under its original causes; a2000 background-model qualification remains. Five a1100 shoulder ideal-comparator angle constraints from C+D remain qualified whenever those old peak results are referenced; no such fit is used to make this map accounting close.']
    (dest/'REPORT.md').write_text('\n'.join(text)+'\n');save_json(dest/'summary.json',dict(source_revision=revision,arrays=records,run=run,report_seconds=time.perf_counter()-started));manifest(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);report(p.parse_args().output)
