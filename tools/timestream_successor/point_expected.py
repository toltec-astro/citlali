"""POINT-EXPECTED-v1: owner-selected diagnostic source guard, never a default.

Parameter order A,cx,cy,sx,sy,angle,constant; coordinates/sigmas arcsec;
brightness and conditional amplitude uncertainty mJy/nominal beam. Fit weights
remain identity. The PSD/scale are the frozen full original-M0 covariance.
"""
import numpy as np
from inspect_point_source import solve_gaussian
from point_empirical import coordinates
from point_admission import admit
from point_additive import variance

METHOD = 'POINT-EXPECTED-v1'
SIGMA_FLOOR = 1.5
FAILURE = 'POINT fit failed: no source satisfying the expected POINT criteria was found.'


def jacobian(p, x, y):
    """Analytic model derivatives of all seven free parameters, including B."""
    A,cx,cy,sx,sy,t,B = p
    c,s = np.cos(t),np.sin(t)
    u = c*(x-cx)+s*(y-cy); v = -s*(x-cx)+c*(y-cy)
    g = np.exp(-.5*((u/sx)**2+(v/sy)**2)); ag=A*g
    return np.column_stack((g,ag*(c*u/sx**2-s*v/sy**2),
        ag*(s*u/sx**2+c*v/sy**2),ag*u*u/sx**3,ag*v*v/sy**3,
        ag*u*v*(1/sy**2-1/sx**2),np.ones_like(x)))


def amplitude_sensitivity(J):
    """Amplitude row of J+ for W=I; keep all nuisance directions at bounds.

    Rank threshold is eps*max(shape)*largest singular value. Identifiability
    checks the amplitude basis vector's null-space projection against
    100*eps*max(shape), a numerical projector tolerance, not a quality cut.
    Circular orientation may be null while amplitude remains identifiable.
    """
    if J.ndim!=2 or J.shape[1]!=7 or not np.isfinite(J).all():
        raise ValueError('nonfinite or malformed model Jacobian')
    U,s,Vt=np.linalg.svd(J,full_matrices=False)
    tol=np.finfo(float).eps*max(J.shape)*s[0]
    rank=int(np.sum(s>tol)); ident_tol=100*np.finfo(float).eps*max(J.shape)
    unit=np.eye(7)[0]
    null_projection=float(np.linalg.norm(unit-np.einsum('ij,i->j',Vt[:rank],Vt[:rank,0],optimize=False)))
    info=dict(rank=rank,rank_tolerance=float(tol),singular_values=s.tolist(),
              amplitude_null_projection=null_projection,identifiability_tolerance=ident_tol)
    if null_projection>ident_tol:return None,info
    q=np.einsum('ij,j->i',U[:,:rank],Vt[:rank,0]/s[:rank],optimize=False)
    if not np.isfinite(q).all():raise ValueError('nonfinite amplitude sensitivity')
    return q,info


def geometry_rejection(p,active):
    if p[0]<=0:return 'nonpositive peak amplitude above fitted constant'
    if np.hypot(p[1],p[2])>20:return 'centroid outside expected-source 20 arcsec disk'
    if min(p[3:5])<=SIGMA_FLOOR or any(active[i]==-1 for i in (3,4)):
        return 'sigma does not strictly exceed 1.5 arcsec or lower width bound is active'
    return None


def snr_rejection(amplitude,sigma):
    return None if amplitude/sigma>10 else 'fitted amplitude S/N does not strictly exceed 10'


def measure_expected(signal,support,fixed,scale,psd,*,initialize_covariance=None):
    """One attempt, fresh accepted state; ordinary rejection != unknown input.

    Invalid covariance/input and nonfinite numerical results are computationally
    unavailable, never completed nulls. Unexpected solver exceptions propagate.
    """
    x,y=coordinates(signal.shape,fixed.header);roi=np.hypot(x,y)<=40
    central=np.hypot(x,y)<=20
    result=dict(method=METHOD,status='failed',accepted=False,complete=True,cause=None,
                accepted_parameters=None,pointing_correction=None,
                accepted_centroid_tangent_arcsec=None,
                expected_origin_tangent_arcsec=[0.,0.],fit_pixels=int(roi.sum()),
                expected_origin_sky=dict(ra_deg=fixed.header.get('CRVAL1'),dec_deg=fixed.header.get('CRVAL2'),
                    frame=fixed.header.get('RADESYS'),equinox=fixed.header.get('EQUINOX')),
                sigma_A=None,SNR_A=None,diagnostic=None,optimizer_calls=0)

    def reject(cause,computational=False):
        result.update(status='unavailable' if computational else 'failed',complete=not computational,cause=cause,
                      message=('POINT computation unavailable.' if computational else FAILURE)+' '+cause)
        return result

    if support.shape!=signal.shape or (initialize_covariance is None and scale.shape!=signal.shape):
        return reject('input/covariance shape mismatch',True)
    if np.any(roi&~(support&np.isfinite(signal))) or roi.sum()<30 or central.sum()<10:
        return reject('incomplete or nonfinite original fit support',True)
    if initialize_covariance is None:
        if np.any(~np.isfinite(scale[roi])) or np.any(scale[roi]<=0):
            return reject('missing or invalid frozen covariance amplitude',True)
        if psd.ndim!=2 or not np.isfinite(psd).all() or np.any(psd<0) or not psd.mean()>0 or max(signal.shape)>=min(psd.shape)/2+1:
            return reject('invalid or insufficiently padded frozen covariance spectrum',True)
    if not np.isfinite(fixed.beam) or fixed.beam<=0:return reject('invalid nominal initialization scale',True)
    ix=np.argmax(np.where(central,signal,-np.inf));background=float(np.median(signal[roi]))
    initial=[max(0.,float(signal.flat[ix])-background),float(x.flat[ix]),float(y.flat[ix]),
             max(SIGMA_FLOOR,fixed.beam/np.sqrt(8*np.log(2))),max(SIGMA_FLOOR,fixed.beam/np.sqrt(8*np.log(2))),0.,background]
    fit=solve_gaussian(signal,roi,x,y,initial,sigma_floor=SIGMA_FLOOR)
    result['optimizer_calls']=1
    p=fit.x
    finite=bool(np.isfinite(p).all() and np.isfinite(fit.fun).all() and np.isfinite(fit.jac).all()
                and np.isfinite(fit.cost) and np.isfinite(fit.optimality))
    result['diagnostic']=dict(initial=initial,parameters=p.tolist(),status=int(fit.status),message=fit.message,
        converged=bool(fit.success),finite=finite,nfev=fit.nfev,njev=fit.njev,cost=float(fit.cost),
        optimality=float(fit.optimality),active_mask=fit.active_mask.tolist(),sigma_floor_arcsec=SIGMA_FLOOR,
        sigma_floor_FWHM_arcsec=SIGMA_FLOOR*np.sqrt(8*np.log(2)),centroid_radius_arcsec=float(np.hypot(p[1],p[2])))
    if not finite:return reject('nonfinite optimizer calculation',True)
    if not fit.success:return reject('finite optimizer nonconvergence: '+fit.message)
    cause=geometry_rejection(p,fit.active_mask)
    if cause:return reject(cause)
    if initialize_covariance is not None:
        # One M0 numerical attempt supplies mask geometry before covariance
        # exists. Complete this same fit's guard; never retry or relax it.
        if scale is not None or psd is not None:
            return reject('competing initialized and supplied covariance',True)
        try:
            scale,psd=initialize_covariance(p.copy())
        except ValueError as exc:
            return reject('initial covariance unavailable: '+str(exc),True)
        if scale.shape!=signal.shape:
            return reject('initialized covariance shape mismatch',True)
    if np.any(~np.isfinite(scale[roi])) or np.any(scale[roi]<=0):
        return reject('missing or invalid frozen covariance amplitude',True)
    if psd.ndim!=2 or not np.isfinite(psd).all() or np.any(psd<0) or not psd.mean()>0 or max(signal.shape)>=min(psd.shape)/2+1:
        return reject('invalid or insufficiently padded frozen covariance spectrum',True)
    try:
        q,info=amplitude_sensitivity(jacobian(p,x[roi],y[roi]))
        result['amplitude_sensitivity']=info
        if q is None:return reject('amplitude not locally identifiable with free nuisance parameters')
        weights=np.zeros(signal.shape);weights[roi]=q
        v=variance(psd,scale,weights)
    except (ValueError,np.linalg.LinAlgError) as exc:
        return reject('amplitude covariance calculation: '+str(exc),True)
    if not np.isfinite(v) or v<=0:return reject('nonpositive or nonfinite amplitude variance',True)
    result.update(sigma_A=float(np.sqrt(v)),SNR_A=float(p[0]/np.sqrt(v)))
    if not np.isfinite(result['SNR_A']):return reject('nonfinite amplitude S/N calculation',True)
    cause=snr_rejection(p[0],result['sigma_A'])
    if cause:return reject(cause)
    result.update(status='accepted',accepted=True,accepted_parameters=p.tolist(),
                  accepted_centroid_tangent_arcsec=p[1:3].tolist(),cause=None,message='Expected POINT source accepted.')
    return result


def candidate(signal,support,fixed,guard):
    """Outward empty accepted products on failure; diagnostics retain status."""
    if guard['accepted']:
        return admit(signal,support,fixed.scale,fixed.noise_valid,fixed.header,
                     np.array(guard['accepted_parameters']),3.,1.)
    x,y=coordinates(signal.shape,fixed.header);domain=np.hypot(x,y)<=40
    valid=domain&support&np.isfinite(signal)&fixed.noise_valid&np.isfinite(fixed.scale)&(fixed.scale>0)
    empty=np.where(valid,0.,np.nan); selected=np.zeros(signal.shape,bool)
    return dict(candidate=empty,selected=selected,valid=valid,domain=domain,
                excluded=np.full(signal.shape,np.nan),brightness=np.full(signal.shape,np.nan),
                z=np.full(signal.shape,np.nan),summary=dict(status='expected_source_'+guard['status'],
                cause=guard['cause'],pixels=0,area_arcsec2=0.,threshold=3.,anchor_row_column=None))


def infer_expected(signal,support,fixed,neighborhood,scale,psd):
    guard=measure_expected(signal,support,fixed,scale,psd)
    base=candidate(signal,support,fixed,guard)
    fields=neighborhood.scores(base) if guard['accepted'] and base['selected'].any() else None
    return guard,base,fields
