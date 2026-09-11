#!/usr/bin/env python3
"""Evaluation only: fixed truth apertures, matched responses and charged timing."""
from pathlib import Path
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
import rbf,truths
b=rbf.base

def read(p):return json.loads(Path(p).read_text())
def normerr(z,t,mask):return float(np.linalg.norm((z-t)[mask])/np.linalg.norm(t[mask]))
def centroid(z,x,y,mask):
    flux=float(np.sum(z[mask]));return None if flux<=0 else [float(np.dot(z[mask],v[mask])/flux) for v in [x,y]]
def measure(root,dest):
    geo=np.load(root/'geometry.npz');x=geo['x'];y=geo['y'];D=geo['D'];O=geo['O'];B=np.column_stack([np.ones(len(x)),x/90,y/90]);radius=np.hypot(x-17,y+11)
    rows=[];raw=[];timing=[];nulls=[]
    for case in read(root/'CASES.json'):
        name=case['case'];truth=np.load(root/f'{name}_truth.npz')['truth'] if case['truth'] else None
        for arm in ['P','G','R']:
            for path in sorted((root/name/arm).glob('pass[0-9][0-9].json')):
                receipt=read(path);k=receipt['pass_index'];maps=np.load(path.with_name(f'pass{k:02d}_maps.npz'))
                timing.append({key:receipt[key] for key in receipt if key not in ['fit','decision']})
                delta=None
                if case['null']:
                    npth=root/case['null']/arm/f'pass{k:02d}_maps.npz'
                    if npth.exists():delta=maps['total']-np.load(npth)['total']
                for a,array in enumerate(b.ARRAYS):
                    common=dict(case=name,arm=arm,array=array,pass_index=k,passes=k+1,method_map_seconds=receipt['cumulative_method_map_seconds'],cold_map_seconds=receipt['first_observation_map_seconds'],wall_seconds=receipt['cumulative_wall_seconds'])
                    dec=receipt['decision'][a];model=maps['next_model'][a];applied=maps['applied_model'][a]
                    item=dict(**common,fit=receipt['fit'][a],decision=dec,total_D_peak=float(np.max(maps['total'][a,D[a]])),total_O_rms=float(np.sqrt(np.mean(maps['total'][a,O[a]]**2))),
                        next_integrated_brightness=float(4*np.sum(model[D[a]])),next_rms=float(np.sqrt(np.mean(model[D[a]]**2))),applied_rms=float(np.sqrt(np.mean(applied[D[a]]**2))),
                        next_support_pixels=int(np.count_nonzero(model)),D_pixels=int(D[a].sum()),O_pixels=int(O[a].sum()))
                    raw.append(item)
                    if name.startswith(('null','background')):nulls.append(item)
                    if delta is None:continue
                    response=delta[a];t=truth[a];err=response-t
                    r=dict(**common,direct_D_error_rms=float(np.sqrt(np.mean(err[D[a]]**2))),exterior_error_rms=float(np.sqrt(np.mean(err[O[a]]**2))),next_model=dec,
                        raw_signed_integrated_D=float(4*np.sum(response[D[a]])),raw_signed_centroid_D=centroid(response,x,y,D[a]))
                    if name.startswith('background'):
                        r['paired_response_D_rms']=r['direct_D_error_rms'];rows.append(r);continue
                    mask=D[a]&(radius<=60);beta=np.linalg.lstsq(B[O[a]],response[O[a]],rcond=None)[0];adjusted=response-b.mm(B,beta)
                    truec=centroid(t,x,y,mask);c=centroid(adjusted,x,y,mask);cerror=None if c is None else float(np.linalg.norm(np.asarray(c)-truec))
                    regions={label:D[a]&(radius>low)&(radius<=high) for label,low,high in [('core',-1,15),('shoulder',15,35),('tail',35,60),('wings',15,60)]}
                    regional={label:dict(relative_L2=normerr(adjusted,t,m),raw_relative_L2=normerr(response,t,m),integrated_brightness_error=float(np.sum(adjusted[m])/np.sum(t[m])-1),truth_brightness=float(4*np.sum(t[m]))) for label,m in regions.items()}
                    r.update(relative_L2=normerr(adjusted,t,mask),raw_relative_L2=normerr(response,t,mask),centroid=c,truth_centroid=truec,centroid_error_arcsec=cerror,
                        integrated_brightness_error=float(np.sum(adjusted[mask])/np.sum(t[mask])-1),raw_integrated_brightness_error=float(np.sum(response[mask])/np.sum(t[mask])-1),
                        nuisance_plane=beta,regions=regional,truth_concentration=truths.concentration(t,D[a]),next_model_relative_L2=normerr(model,t,mask),applied_model_relative_L2=normerr(applied,t,mask))
                    if arm=='R':
                        diag=dec['concentration'];tc=r['truth_concentration'];rr=diag['raw_fit_diagnostic']
                        r['concentration_available']=diag['available'];r['raw_fitted_Aeff_bias']=None if rr is None else rr['effective_area_arcsec2']/tc['effective_area_arcsec2']-1
                        r['scientific_Aeff_bias']=r['raw_fitted_Aeff_bias'] if diag['available'] else None
                    if name.startswith(('gauss','mismatch')):
                        tr=b.fit_source(t,x,y,D[a]);fit=b.fit_source(response,x,y,D[a]);p=fit['parameters'];plane=p[6]+p[7]*x/90+p[8]*y/90;source_domain=D[a]&(radius<=40)
                        amp=fit['peak']/tr['peak']-1;width=np.asarray(fit['widths'])/tr['widths']-1;distance=float(np.linalg.norm(np.asarray(fit['centroid'])-tr['centroid']));scaled=distance/min(tr['widths'])
                        r.update(amplitude_bias=amp,width_bias=width,gaussian_centroid_error_arcsec=distance,centroid_error_minor_fwhm=scaled,fit=fit,
                            source_error_rms=float(np.sqrt(np.mean(err[source_domain]**2))),aperture_fraction=float(np.sum(response[source_domain]-plane[source_domain])/np.sum(t[source_domain])),
                            joint_target=bool(abs(amp)<=.05 and np.all(abs(width)<=.05) and scaled<=.05 and not fit['boundary_rejection']))
                    else:
                        bright=name=='coma_bright';r['joint_target']=bool(abs(r['integrated_brightness_error'])<=(.1 if bright else .15) and cerror is not None and cerror<=(.4 if bright else .8) and r['relative_L2']<=(.15 if bright else .25) and abs(regional['wings']['integrated_brightness_error'])<=(.2 if bright else .3))
                    rows.append(r)
    final=[r for r in rows if r['pass_index']==6];first=[]
    for case in ['gauss20260911','gauss20260912','coma_bright','coma_half']:
        for array in b.ARRAYS:
            for arm in ['P','G','R']:
                good=[r for r in rows if r['case']==case and r['array']==array and r['arm']==arm and r.get('joint_target')]
                first.append(dict(case=case,array=array,arm=arm,first=good[0] if good else None))
    failure=[dict(path=str(p.relative_to(root)),record=read(p)) for p in root.rglob('*FAILURE.json')]
    result=dict(root=str(root),response=rows,final=final,raw=raw,null_and_background=nulls,timing=timing,first_joint_targets=first,failures=failure,
        setup=read(root/'SETUP.json'),campaign=read(root/'CAMPAIGN_COMPLETE.json'),OG=read(rbf.PREV/'NUMERICAL_EVIDENCE_R0.2.json')['OG'])
    b.write(dest,result)
    for r in final:
        if r['case'].startswith('background'):continue
        print(r['case'],r['array'],r['arm'],'L2',round(r['relative_L2'],4),'flux',round(r['integrated_brightness_error'],4),'centroid',round(r['centroid_error_arcsec'],3) if r['centroid_error_arcsec'] is not None else None,'wing',round(r['regions']['wings']['integrated_brightness_error'],3),'joint',r['joint_target'],'Aeff',r.get('scientific_Aeff_bias'))
    print('Failures',failure)
    print('RBF null/background admissions',[(r['case'],r['array'],r['pass_index'],r['next_rms']) for r in nulls if r['arm']=='R' and r['decision']['admitted']])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('dest',type=Path);a=p.parse_args()
    with threadpool_limits(limits=4):measure(a.root,a.dest)
