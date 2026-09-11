#!/usr/bin/env python3
"""Evaluate the predeclared paired response and timing; no feedback choices."""
from pathlib import Path
import argparse,json,re
import numpy as np
from astropy.io import fits
from threadpoolctl import threadpool_limits
import experiment as e

def load(p):return json.loads(Path(p).read_text())

def analyze(root,dest):
    geom=np.load(root/'fixed_geometry.npz');x=geom['x'];y=geom['y'];D=geom['D'];O=geom['O'];rows=[];truth_fits={}
    for source in ['gauss20260911','gauss20260912','mismatch20260911']:
        truth=np.load(root/f'{source}_truth.npz')['truth'];null='null'+source[-8:]
        truth_fits[source]=[e.fit_source(truth[a],x,y,D[a]) for a in range(3)]
        for arm in ['P','G']:
            for k in range(7):
                sp=root/source/arm/f'pass{k:02d}_maps.npz';npth=root/null/arm/f'pass{k:02d}_maps.npz'
                if not sp.exists() or not npth.exists():continue
                delta=np.load(sp)['total']-np.load(npth)['total'];receipt=load(root/source/arm/f'pass{k:02d}.json')
                for a in range(3):
                    tr=truth_fits[source][a];fit=e.fit_source(delta[a],x,y,D[a]);p=fit['parameters'];plane=p[6]+p[7]*x/90+p[8]*y/90
                    source_domain=D[a]&(np.hypot(x-17,y+11)<=40);err=delta[a]-truth[a]
                    amplitude_bias=(fit['peak']/tr['peak']-1);width_bias=np.asarray(fit['widths'])/tr['widths']-1;centroid=np.linalg.norm(np.asarray(fit['centroid'])-tr['centroid']);centroid_scaled=centroid/min(tr['widths'])
                    joint=abs(amplitude_bias)<=.05 and np.all(abs(width_bias)<=.05) and centroid_scaled<=.05 and not fit['boundary_rejection']
                    rows.append(dict(case=source,arm=arm,array=e.ARRAYS[a],pass_index=k,passes=k+1,
                        amplitude_bias=amplitude_bias,width_bias=width_bias,centroid_error_arcsec=centroid,centroid_error_minor_fwhm=centroid_scaled,
                        direct_D_error_rms=np.sqrt(np.mean(err[D[a]]**2)),source_error_rms=np.sqrt(np.mean(err[source_domain]**2)),
                        exterior_error_rms=np.sqrt(np.mean(err[O[a]]**2)),aperture_fraction=np.sum(delta[a,source_domain]-plane[source_domain])/np.sum(truth[a,source_domain]),
                        peak=fit['peak'],widths=fit['widths'],centroid=fit['centroid'],fit_boundary=fit['boundary_rejection'],joint_target=bool(joint),
                        method_map_seconds=receipt['cumulative_method_map_seconds'],development_wall_seconds=receipt['cumulative_wall_seconds']))
    real=[];null=[];fail=[]
    for case in ['real123424','null20260911','null20260912']:
        for arm in ['P','G']:
            for path in sorted((root/case/arm).glob('pass[0-9][0-9].json')):
                rec=load(path)
                for a in range(3):
                    item=dict(case=case,arm=arm,array=e.ARRAYS[a],pass_index=rec['pass_index'],passes=rec['cleaning_passes'],fit=rec['fit'][a],decision=rec['decision'][a],method_map_seconds=rec['cumulative_method_map_seconds'],development_wall_seconds=rec['cumulative_wall_seconds'])
                    (real if case=='real123424' else null).append(item)
    for p in root.rglob('*FAILURE.json'):fail.append({'path':str(p.relative_to(root)),'record':load(p)})
    og=load(e.HERE/'OG_BENCHMARK.json');log=Path(og['log']).read_text();times={int(i):float(t) for i,t in re.findall(r'profile stage=reduction.iteration context=fruit_iter=(\d+) elapsed_s=([\d.]+)',log)}
    oswall=re.search(r'^real\s+([\d.]+)',log,re.M);peak=re.search(r'(\d+)\s+maximum resident set size',log)
    benchmark=[]
    for k in range(7):
        for a,array in enumerate(e.ARRAYS):
            path=Path(og['root'])/f'redu{k:02d}/123424/raw'/f'toltec_commissioning_{array}_pointing_123424_citlali.fits'
            with fits.open(path) as f:
                z=np.asarray(f['signal_I'].data).squeeze();w=np.asarray(f['weight_I'].data).squeeze();h=f['signal_I'].header
                assert h['CUNIT1']==h['CUNIT2']=='arcsec'
                xx=(np.arange(z.shape[1])+1-h['CRPIX1'])*h['CDELT1']+h['CRVAL1'];yy=(np.arange(z.shape[0])+1-h['CRPIX2'])*h['CDELT2']+h['CRVAL2'];xx,yy=np.meshgrid(xx,yy);d=(xx*xx+yy*yy<=90**2)&(w>0)&np.isfinite(z)
                fit=e.fit_source(z.ravel(),xx.ravel(),yy.ravel(),d.ravel())
            benchmark.append(dict(array=array,pass_index=k,fit=fit,cumulative_iteration_seconds=sum(times.get(i,0) for i in range(k+1)),path=str(path)))
    first=[]
    for case in ['gauss20260911','gauss20260912']:
        for array in e.ARRAYS:
            for arm in ['P','G']:
                good=[r for r in rows if r['case']==case and r['array']==array and r['arm']==arm and r['joint_target']]
                first.append(dict(case=case,array=array,arm=arm,first=good[0] if good else None))
    record=dict(root=str(root),response=rows,real=real,null=null,failures=fail,first_joint_targets=first,truth_fits=truth_fits,
        OG=dict(identity=og['identity'],full_wall_seconds=float(oswall[1]) if oswall else None,peak_rss_bytes=int(peak[1]) if peak else None,iteration_seconds=times,sequence=benchmark,
        limit='Historical-recurrence control has different input/upstream/JINC/weights/runtime scope and diagnostics. Core-vs-OG speed ratio is not end-to-end performance evidence.'),campaign=load(root/'CAMPAIGN_COMPLETE.json') if (root/'CAMPAIGN_COMPLETE.json').exists() else None)
    e.write(dest,record)
    print('Failures',len(fail),'OG wall',record['OG']['full_wall_seconds'])
    for case in ['gauss20260911','gauss20260912','mismatch20260911']:
        print(case)
        for array in e.ARRAYS:
            last=[r for r in rows if r['case']==case and r['array']==array and r['pass_index']==6]
            print(array,[(r['arm'],round(100*r['amplitude_bias'],2),np.round(100*np.array(r['width_bias']),2).tolist(),round(r['centroid_error_arcsec'],3),round(r['direct_D_error_rms'],3),r['joint_target']) for r in last])
    for item in first:print('FIRST',item['case'],item['array'],item['arm'],None if item['first'] is None else (item['first']['passes'],round(item['first']['method_map_seconds'],3)))
    print('Null final',[(r['case'],r['array'],r['decision']['model_peak'],r['decision']['reason']) for r in null if r['arm']=='G' and r['pass_index']==6])
    print('Real final',[(r['arm'],r['array'],round(r['fit']['peak'],2),np.round(r['fit']['centroid'],2).tolist(),np.round(r['fit']['widths'],2).tolist()) for r in real if r['pass_index']==6])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=4):analyze(args.root,args.output)
