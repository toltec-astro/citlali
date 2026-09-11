#!/usr/bin/env python3
"""Noiseless representation and grid-phase check before empirical trajectories."""
from pathlib import Path
import argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
import rbf_r02 as rbf
import truths

def main(out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    data=rbf.base.Data(json.loads((rbf.PREV/'INPUT_PREFLIGHT.json').read_text()))
    images={'diffraction':truths.optical_image(False),'coma':truths.optical_image(True)}
    rows=[];identities=[]
    np.savez_compressed(out/'geometry.npz',x=data.x,y=data.ygrid,D=data.D,O=data.O,shape=data.shape)
    for a,array in enumerate(rbf.base.ARRAYS):
        systems={lam:rbf.Basis(data,a,lam) for lam in [0.,rbf.CONFIG['lambda_value']]};identities.extend([dict(array=array,**s.identity) for s in systems.values()])
        for kind in ['compact','diffraction','coma']:
            for phase in [0,1]:
                center=np.array([17.,-11.])+phase*np.array([2.,1.])
                if kind=='compact':
                    p=np.array([100.,*center,np.log(12.),np.log(8.),np.deg2rad(25),0,0,0]);z=np.where(data.D[a],rbf.base.gaussian(p,data.x,data.ygrid),0.)
                else:z=truths.map_truth(data,images[kind],translation=center)[a]
                tc=truths.concentration(z,data.D[a])
                for lam,system in systems.items():
                    coeff,beta,fit,model=system.fit_only(z);diag=system.concentration(coeff)['raw_fit_diagnostic'];err=model-z
                    result=dict(array=array,kind=kind,phase=phase,translation=center,lambda_value=lam,
                        relative_L2=float(np.linalg.norm(err[data.D[a]])/np.linalg.norm(z[data.D[a]])),
                        integrated_brightness_error=diag['integrated_brightness']/tc['integrated_brightness']-1,
                        effective_area_error=diag['effective_area_arcsec2']/tc['effective_area_arcsec2']-1,
                        concentration=diag,truth_concentration=tc,background=beta,solver=fit)
                    rows.append(result);name=f'{array}_{kind}_phase{phase}_lambda{lam:g}.npz'
                    np.savez_compressed(out/name,truth=z,model=model,coefficients=coeff)
                    print(array,kind,phase,lam,'L2',result['relative_L2'],'area error',result['effective_area_error'],flush=True)
        system=systems[rbf.CONFIG['lambda_value']];plane=13+7*data.x/90-4*data.ygrid/90
        coeff,beta,fit,model=system.fit_only(plane)
        rows.append(dict(array=array,kind='plane_only',max_source=float(model.max()),background=beta,solver=fit))
    rbf.base.write(out/'REPRESENTATION.json',dict(config=rbf.CONFIG,truth_config=truths.PSF_CONFIG,rows=rows,bases=identities,wall_seconds=time.monotonic()-start,
        raw_representation_pass=all(r['relative_L2']<=.05 for r in rows if r.get('lambda_value')==0.)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    with threadpool_limits(limits=4):main(a.output)
