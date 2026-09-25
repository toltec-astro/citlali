"""Arithmetic only for cached POINT products; no inference or selection imports.

Rows/columns are saved north/east indices; brightness mJy/nominal beam, angular
coordinates arcsec. Unavailable values remain NaN; admitted model zeros are real
zeros. Shape-domain norms and full-map norms are deliberately distinct.
"""
import numpy as np


def require(ok,message):
    if not ok:raise ValueError(message)


def timing(k):
    require(isinstance(k,int) and 0<=k<=10,'stage outside saved history')
    return dict(map_stage=k,applied_candidate=None if k==0 else k-1,
        selected_candidate=k,next_map=None if k==10 else k+1,
        candidate_status='saved unapplied' if k==10 else 'imposed initial seed only when seed branch' if k==0 else 'applied only to next map')


def paired(branch,reference):
    for key in ('array','policy','cycle_index'):require(branch[key]==reference[key],'mismatched '+key)
    require(reference['branch']=='REF','comparison requires own REF')


def difference(a,av,b,bv):
    require(a.shape==b.shape==av.shape==bv.shape,'unaligned arrays')
    valid=av&bv;require(np.isfinite(a[valid]).all() and np.isfinite(b[valid]).all(),'nonfinite available pixel')
    return np.where(valid,a-b,np.nan),valid,dict(common=int(valid.sum()),lost=int((bv&~av).sum()),gained=int((av&~bv).sum()))


def magnitude(v,valid):
    require(np.isfinite(v[valid]).all(),'nonfinite declared domain')
    x=v[valid]
    return dict(pixels=int(valid.sum()),norm=float(np.linalg.norm(x)),rms=float(np.sqrt(np.mean(x*x))) if x.size else None,
        maximum_absolute=float(np.max(abs(x))) if x.size else None,signed_sum=float(x.sum()))


def shape_components(v,valid,shape):
    domain=valid&np.isfinite(shape);require(np.isfinite(v[domain]).all(),'unavailable shape projection')
    q=shape[domain];norm2=float(np.sum(q*q));require(norm2>0,'empty shape normalization')
    alpha=float(np.sum(v[domain]*q)/norm2);parallel=np.where(domain,alpha*shape,np.nan);residual=np.where(domain,v-parallel,np.nan)
    full=magnitude(v,domain);par=magnitude(parallel,domain);orth=magnitude(residual,domain)
    require(np.isclose(full['norm']**2,par['norm']**2+orth['norm']**2,rtol=1e-11,atol=1e-8),'orthogonal bookkeeping identity')
    return dict(coefficient=alpha,shape_norm2=norm2,domain_pixels=int(domain.sum()),total=full,parallel=par,orthogonal=orth),parallel,residual,domain


def aperture(v,valid,masks,area=4.):
    return [dict(available=bool(np.all(valid[m]) and m.any()),pixels=int(m.sum()),
                 value=float(np.sum(v[m])*area) if np.all(valid[m]) and m.any() else None) for m in masks]


def split_candidate(seed,ref,seed_ad,ref_ad,bs,br,k):
    """Disjoint membership terms; k0 is explicitly not an ordinary candidate."""
    require(k>=1,'imposed m0 cannot use ordinary brightness partition')
    valid=seed['VALID'].astype(bool)&ref['VALID'].astype(bool)
    ms=seed['ADMITTED'].astype(bool);mr=ref['ADMITTED'].astype(bool)
    for c,a in ((seed,seed_ad),(ref,ref_ad)):
        m=c['ADMITTED'].astype(bool);v=c['VALID'].astype(bool)
        require(not (m&~v).any(),'selected unavailable pixel')
        require(np.array_equal(c['CANDIDATE'][m],a['BRIGHTNESS'][m]),'changed saved selected brightness')
        require(np.all(c['CANDIDATE'][v&~m]==0) and np.isnan(c['CANDIDATE'][~v]).all(),'zero versus unavailable semantics')
    masks=dict(shared=valid&ms&mr,seed_only=valid&ms&~mr,REF_only=valid&mr&~ms)
    values=dict(shared=seed_ad['BRIGHTNESS']-ref_ad['BRIGHTNESS'],seed_only=seed_ad['BRIGHTNESS'],REF_only=-ref_ad['BRIGHTNESS'])
    constants=dict(shared=-(bs-br),seed_only=-bs,REF_only=br)
    terms={n:np.where(valid,np.where(m,values[n],0.),np.nan) for n,m in masks.items()}
    bg={n:np.where(valid,np.where(m,constants[n],0.),np.nan) for n,m in masks.items()}
    delta,_,_=difference(seed['CANDIDATE'],valid,ref['CANDIDATE'],valid)
    err=float(np.max(abs(sum(terms.values())[valid]-delta[valid])));require(err<=1e-8,'mask partition failed')
    measured={n:np.where(valid,terms[n]-bg[n],np.nan) for n in terms}
    return dict(masks=masks,terms=terms,background=bg,measured=measured,valid=valid,delta=delta,error=err)


def memberships(ad,previous=None):
    b=ad['BASE_POINT'].astype(bool);p=ad['PROPOSED'].astype(bool);e=ad['ADDED'].astype(bool);m=ad['ADMITTED'].astype(bool)
    masks=dict(baseline=b,supplement=e,proposed=p,rejected_connectivity=p&~e,final=m)
    if previous is not None:
        pm=previous['ADMITTED'].astype(bool);pb=previous['BASE_POINT'].astype(bool);pe=previous['ADDED'].astype(bool)
        masks.update(newly_selected=m&~pm,dropped=pm&~m,shared_selected=m&pm,supplement_to_baseline=pe&b,baseline_to_supplement=pb&e)
    return masks


def score_audit(ad,policy):
    valid=ad[policy+'_VALID'].astype(bool);e=ad['EVIDENCE'];v=ad[policy+'_VAR'];z=ad[policy+'_SCORE']
    require(np.isfinite(e[valid]).all() and np.all(v[valid]>0),'invalid saved available evidence')
    error=float(np.max(abs(e[valid]/np.sqrt(v[valid])-z[valid]))) if valid.any() else 0.
    require(error<=1e-8,'saved score arithmetic mismatch')
    return dict(available_pixels=int(valid.sum()),error=error,negative_signed_evidence=int((valid&(e<0)).sum()),
        numerator_variance_reconstruction='unavailable: individual archived weights absent; no reconstruction attempted')


def signed_response(delta,known,domain):
    require(np.isfinite(delta[domain]).all() and np.isfinite(known[domain]).all(),'unavailable signed response')
    n=float(np.sum(known[domain]**2));require(n>0,'zero signed reference')
    g=float(np.sum(delta[domain]*known[domain])/n)
    eta=float(np.linalg.norm((delta-known)[domain])/np.sqrt(n));rho=float(np.linalg.norm((delta-g*known)[domain])/np.sqrt(n))
    require(abs(eta*eta-(g-1)**2-rho*rho)<1e-10,'signed response identity')
    return dict(g=g,eta=eta,rho=rho,pixels=int(domain.sum()))


def signed_even(plus,minus,kplus,kminus,valid):
    even=np.where(valid,plus+minus,np.nan);ideal=np.where(valid,kplus+kminus,np.nan)
    return even,ideal,np.where(valid,even-ideal,np.nan)


def percent(value,denominator):
    return 100.*value/denominator if np.isfinite(denominator) and abs(denominator)>1e-8 else None
