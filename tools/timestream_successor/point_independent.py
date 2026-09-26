"""Frozen-method bindings for the single independent a2000 POINT campaign.

No threshold calibration, ensemble, science defaults or native numerical rules.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from point_additive import learn,NoiseRequest,linear_weights,variance
from point_empirical import coordinates,patch,source_patch
from point_expected import measure_expected,candidate,jacobian,amplitude_sensitivity
from point_corr_admission import Neighborhood
from point_guarded_feedback import measurement
from point_additive_diagnostics import save_json
from point_fruit_diagnostics import load,patch_map
from inspect_forward_map import require


def initialize(path,dest):
    full,h=load(path);signal=full['PRIMARY'];support=full['SCI_SUPPORT'].astype(bool);coverage=full['NORMALIZATION']
    s,v,_,ph=patch_map(path);f=SimpleNamespace(header=ph,beam=10.)
    models={}
    def covariance(p):
        models['C0']=learn(signal,coverage,support,h,p,NoiseRequest(mask_fwhm=4.,high_coverage_usable=True),stamp_size=49)
        models['BASE']=learn(signal,coverage,support,h,p,NoiseRequest(mask_fwhm=3.,high_coverage_usable=True),stamp_size=49,amplitude_only=True)
        cy,cx=int(round(h['CRPIX2']-1)),int(round(h['CRPIX1']-1))
        models['scale']=patch(models['C0']['scale'],cy,cx,20)
        models['base_scale']=patch(models['BASE']['scale'],cy,cx,20)
        models['noise_valid']=patch(models['BASE']['valid'],cy,cx,20).astype(bool)
        return models['scale'],models['C0']['psd']
    guard=measure_expected(s,v,f,None,None,initialize_covariance=covariance)
    dest.mkdir(parents=True);save_json(dest/'guard.json',guard)
    if not guard['accepted']:return None,guard
    p=np.asarray(guard['accepted_parameters']);x,y=coordinates(s.shape,ph);major=max(p[3:5])*np.sqrt(8*np.log(2));q,_=linear_weights(ph,s.shape,p)
    radius=np.hypot(x-p[1],y-p[2]);f=SimpleNamespace(header=ph,beam=10.,parameters=p,x=x,y=y,major=major,radius=radius,weights=q,disk=np.hypot(x,y)<=40,background=(np.hypot(x,y)<=40)&(radius>3*major),edges=np.arange(0,42,2),scale=models['base_scale'],noise_valid=models['noise_valid'])
    base=candidate(s,v,f,guard);k0=base['selected'].copy()
    np.savez_compressed(dest/'context.npz',original=s,support=v,full_support=support,scale=models['scale'],psd=models['C0']['psd'],empirical=models['C0']['empirical'],pairs=models['C0']['pairs'],base_scale=models['base_scale'],noise_valid=models['noise_valid'],parameters=p,q=q,k0=k0)
    save_json(dest/'geometry.json',dict(full_WCS={k:h[k] for k in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','RADESYS','EQUINOX')},header={k:val for k,val in ph.items() if k not in ('COMMENT','HISTORY','')},major=major,beam=10.,guard=guard,mask_rule='M0-only four-FWHM circle; fixed before feedback',M0_covariance_learns=1,BASE_amplitude_only_initializations=1,optimizer_calls=1))
    for name in ('C0','BASE'):
        m=models[name];np.savez_compressed(dest/(name+'.npz'),**{k:m[k] for k in ('scale','valid','coverage_extrapolated','source_mask','background_mask')},**({k:m[k] for k in ('psd','empirical','pairs')} if name=='C0' else {}));save_json(dest/(name+'.json'),m['summary'])
    sanity={}
    for name in ('C0','BASE'):
        m=models[name];high=m['coverage_extrapolated'];use=m['valid'];require(np.isfinite(m['scale'][use]).all() and (m['scale'][use]>0).all(),'unusable amplitude continuation')
        tail=m['summary']['bins'][-1];expected=np.sqrt(tail['second_moment_about_global_mean']*(m['summary'].get('zero_lag_before_normalization',m['summary'].get('spectrum_fit',{}).get('zero_lag_before_normalization'))))
        require(np.allclose(m['scale'][high],expected,rtol=2e-15,atol=0),'unexpected high-coverage continuation')
        sanity[name]=dict(**m['summary']['coverage_transfer'],maximum_extrapolated_scale=float(m['scale'][high].max()) if high.any() else None,minimum_extrapolated_scale=float(m['scale'][high].min()) if high.any() else None,training_endpoint_scale=float(expected),upper_endpoint_jump=0.,reason='same terminal bin applies on both sides of the training maximum; finite positive constant tail, no singularity or collapse; no additional clamp or noise law introduced')
    save_json(dest/'continuation-sanity.json',sanity)
    c=context(dest);return c,guard


def context(dest):
    import json
    g=json.loads((dest/'geometry.json').read_text());c=dict(np.load(dest/'context.npz'));p=c['parameters'];h=g['header'];x,y=coordinates(c['original'].shape,h);r=np.hypot(x-p[1],y-p[2])
    f=SimpleNamespace(header=h,beam=10.,parameters=p,x=x,y=y,radius=r,major=g['major'],weights=c['q'],scale=c['base_scale'],noise_valid=c['noise_valid'],disk=np.hypot(x,y)<=40,background=(np.hypot(x,y)<=40)&(r>3*g['major']),edges=np.arange(0,42,2))
    require(json.loads((dest/'BASE.json').read_text())['request']['high_coverage_usable'] and json.loads((dest/'C0.json').read_text())['request']['high_coverage_usable'],'wrong saved high-coverage semantics')
    fullh=g['full_WCS'];cy,cx=int(round(fullh['CRPIX2']-1)),int(round(fullh['CRPIX1']-1))
    c['noise_high_extrapolated']=patch(np.load(dest/'BASE.npz')['coverage_extrapolated'],cy,cx,20)
    c['covariance_high_extrapolated']=patch(np.load(dest/'C0.npz')['coverage_extrapolated'],cy,cx,20)
    c.update(header=h,fixed=f,major=g['major'],beam=10.,guard=g['guard'],full_WCS=g['full_WCS'],full_source_mask=np.load(dest/'C0.npz')['source_mask']);c['neighborhood']=Neighborhood(h,c['original'].shape,10.,c['scale'],c['psd'],c['k0'],empirical=c['empirical'])
    require(c['neighborhood'].lags_supported,'required covariance lags unavailable');return c


def endpoint(path,c,record,dest):
    """One endpoint covariance; conditional local estimator sensitivities, no refit.

    All nuisance directions including the constant enter propagation. Circular
    angle nonidentifiability is exposed per-parameter, not forced invertible.
    """
    full,h=load(path);s,v,_,ph=patch_map(path);p=np.asarray(record['guard']['accepted_parameters']);f=c['fixed']
    require(all(h[k]==val for k,val in c['full_WCS'].items()),'endpoint frozen WCS differs')
    require(full['PRIMARY'].shape==c['full_source_mask'].shape,'endpoint frozen source-mask geometry differs')
    model=learn(full['PRIMARY'],full['NORMALIZATION'],full['SCI_SUPPORT'].astype(bool)&c['full_support'],h,c['parameters'],NoiseRequest(mask_fwhm=4.,high_coverage_usable=True),stamp_size=49,source_mask=c['full_source_mask'])
    cy,cx=int(round(h['CRPIX2']-1)),int(round(h['CRPIX1']-1));scale=patch(model['scale'],cy,cx,20);valid=patch(model['valid'],cy,cx,20);extrapolated=patch(model['coverage_extrapolated'],cy,cx,20)
    roi=f.disk;J=jacobian(p,f.x[roi],f.y[roi]);U,sigma,Vt=np.linalg.svd(J,full_matrices=False);rank=int(np.sum(sigma>np.finfo(float).eps*max(J.shape)*sigma[0]));pinv=(Vt[:rank].T/sigma[:rank])@U[:,:rank].T
    projector=Vt[:rank].T@Vt[:rank];tol=100*np.finfo(float).eps*max(J.shape)
    def response(row):
        z=np.zeros(roi.shape);z[roi]=row;return z
    W=[response(pinv[i]) for i in range(7)];identifiable=[bool(np.linalg.norm(np.eye(7)[i]-projector[i])<=tol) for i in range(7)]
    linear=[f.weights,*[4.*(f.radius<=a*f.major) for a in (1.,1.5,2.)]]
    ops=W+[q-np.sum(q)*W[6] for q in linear]
    names=['peak','cx','cy','sx','sy','angle','constant','template','aperture1','aperture1.5','aperture2'];rows=[]
    for i,(name,q) in enumerate(zip(names,ops)):
        usable=identifiable[i] if i<7 else identifiable[6]
        domain=q!=0;transfer=bool(np.all(valid[domain]));vv=variance(model['psd'],scale,q) if usable else None
        rows.append(dict(name=name,conditional_sigma=float(np.sqrt(vv)) if transfer and vv is not None and vv>=0 else None,source_transfer_supported=transfer,source_transfer_empirically_calibrated=False,high_coverage_operator_pixels=int(np.sum(extrapolated&domain)),identifiable=usable,scope='conditional background covariance transfer including refitted constant nuisance; not paired-difference significance or model-estimation error',qualification='conditional model-based high-coverage continuation' if transfer and np.any(extrapolated&domain) else None if transfer else 'low coverage or independent unusable input; measurement uncertainty unresolved'))
    dest.mkdir();save_json(dest/'uncertainties.json',rows);save_json(dest/'model.json',model['summary']);np.savez_compressed(dest/'model.npz',psd=model['psd'],source_scale=scale,source_valid=valid,coverage_extrapolated=extrapolated,source_mask=model['source_mask'],background_mask=model['background_mask'],empirical=model['empirical'],pairs=model['pairs'],operators=np.asarray(ops));return rows
