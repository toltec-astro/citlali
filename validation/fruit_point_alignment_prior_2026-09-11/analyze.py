#!/usr/bin/env python3
"""Post-run evaluation only; true positions never enter feedback inference."""
from pathlib import Path
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
import experiment as e

def read(p):return json.loads(Path(p).read_text())

def metrics(delta,truth,x,y,D,O,local=False):
    error=delta-truth
    out=dict(direct_D_error_rms=np.sqrt(np.mean(error[D]**2)),exterior_error_rms=np.sqrt(np.mean(error[O]**2)))
    if not np.any(truth):
        out.update(absent_truth=True,joint_target=None,fit=None,truth_fit=None)
        return out
    tf=e.b.fit_source(truth,x,y,D);center=tf['centroid'];r=np.hypot(x-center[0],y-center[1]);ap=D&(r<=40)
    fit_domain=D&(r<=20) if local else D
    fit=e.b.fit_source(delta,x,y,fit_domain);p=fit['parameters'];plane=p[6]+p[7]*x/90+p[8]*y/90
    bias=fit['peak']/tf['peak']-1;wb=np.array(fit['widths'])/tf['widths']-1
    ce=np.linalg.norm(np.array(fit['centroid'])-tf['centroid']);cw=ce/min(tf['widths'])
    out.update(absent_truth=False,fit=fit,truth_fit=tf,amplitude_bias=bias,width_bias=wb,centroid_error_arcsec=ce,centroid_error_minor_fwhm=cw,
        aperture_fraction=np.sum(delta[ap]-plane[ap])/np.sum(truth[ap]),signed_raw_aperture_fraction=np.sum(delta[ap])/np.sum(truth[ap]),
        source_error_rms=np.sqrt(np.mean(error[ap]**2)),joint_target=bool(abs(bias)<=.05 and np.all(abs(wb)<=.05) and cw<=.05 and not fit['boundary_rejection']),
        evaluator='unconstrained fit on fixed truth-centered 20 arcsec radius' if local else 'unconstrained full-D fit')
    if local:out['full_domain_free_fit']=e.b.fit_source(delta,x,y,D)
    return out

def analyze(root,dest):
    geom=np.load(root/'fixed_geometry.npz');x=geom['x'];y=geom['y'];D=geom['D'];O=geom['O'];cases=read(root/'CASES.json')
    receipts=[];response=[]
    for case in cases:
        name=case['case'];truth=np.load(root/f'{name}_truth.npz')['truth'] if case['truth'] else None
        for arm in ['G','J']:
            for k in range(7):
                p=root/name/arm/f'pass{k:02d}.json'
                if not p.exists():continue
                rec=read(p);receipts.append(rec)
                if truth is None:continue
                n=root/case['null']/arm/f'pass{k:02d}_maps.npz'
                if not n.exists():continue
                delta=np.load(root/name/arm/f'pass{k:02d}_maps.npz')['total']-np.load(n)['total']
                for a,array in enumerate(e.b.ARRAYS):
                    m=metrics(delta[a],truth[a],x,y,D[a],O[a],name.startswith('source_contaminant'))
                    response.append(dict(case=name,arm=arm,array=array,passes=k+1,pass_index=k,method_map_seconds=rec['cumulative_method_map_seconds'],development_wall_seconds=rec['cumulative_wall_seconds'],**m))
    first=[]
    for name in [c['case'] for c in cases if c['truth']]:
        for array in e.b.ARRAYS:
            for arm in ['G','J']:
                good=[r for r in response if r['case']==name and r['array']==array and r['arm']==arm and r['joint_target']]
                first.append(dict(case=name,array=array,arm=arm,passes=good[0]['passes'] if good else None,method_map_seconds=good[0]['method_map_seconds'] if good else None))
    failures=[dict(path=str(p.relative_to(root)),record=read(p)) for p in root.rglob('*FAILURE.json')]
    record=dict(root=str(root),cases=cases,response=response,receipts=receipts,first_joint_targets=first,failures=failures,campaign=read(root/'CAMPAIGN_COMPLETE.json'))
    e.b.write(dest,record)
    print('CAMPAIGN',record['campaign'],'FAILURES',len(failures))
    for name in [c['case'] for c in cases if c['truth']]:
        print(name)
        for a in e.b.ARRAYS:
            final=[r for r in response if r['case']==name and r['array']==a and r['passes']==7]
            print(a,[(r['arm'],round(100*r['amplitude_bias'],2) if 'amplitude_bias' in r else None,np.round(100*np.array(r['width_bias']),2).tolist() if 'width_bias' in r else None,round(r.get('centroid_error_arcsec',0),3),r['joint_target'],round(r['direct_D_error_rms'],3)) for r in final])
    for name in ['real123424','null20260911','null20260912','absent20260911','contaminant20260911','outside20260911']:
        rows=[r for r in receipts if r['case']==name and r['arm']=='J'];print('J DECISIONS',name,[(r['pass_index'],[x['reason'] for x in r['decision']],r['prior']['a2000_offset_radius']) for r in rows])
    for r in receipts:
        if r['case']=='real123424' and r['pass_index']==6:print('REAL',r['arm'],r['cumulative_method_map_seconds'],[(f['peak'],f['centroid'],f['widths']) for f in r['fit']],r['prior'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('dest',type=Path);a=p.parse_args()
    with threadpool_limits(limits=4):analyze(a.root,a.dest)
