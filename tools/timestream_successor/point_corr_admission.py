"""Diagnostic FRUIT selection only: frozen-covariance leave-center/core-out evidence.

Grid arrays are north-row/east-column; physical offsets arcsec; b is unchanged
MAP brightness minus the current descriptive fitted constant (mJy/nominal beam).
C0 is the authenticated M0 companion model, never new MAP/source-fit weights.
"""
import hashlib
import numpy as np
from scipy.ndimage import label
from scipy.stats import beta, binom
from point_admission import admit, fit_diagnostic, CONNECTIVITY
from point_empirical import coordinates
from inspect_forward_map import require

METHOD='POINT152389-CORR-NEIGHBOR-v1'
POLICIES=('CORR','DIAG')


def trial_seed(array,phase,index):
    require(phase in ('calibration','validation') and 0<=index<1024,'fixed stream identity')
    token=f'{METHOD}|{array}|{phase}|{index:04d}'
    return int.from_bytes(hashlib.sha256(token.encode('utf-8')).digest()[:16],'big')


def threshold(maxima):
    require(len(maxima)==1024,'exact calibration count required')
    a=np.array([-np.inf if v is None else v for v in maxima],float)
    require(not np.isnan(a).any() and not np.isposinf(a).any(),'invalid calibration score')
    return float(max(3.,np.sort(a)[1014]))


def binomial_upper(k,n=1024):
    require(0<=k<=n and n>0,'invalid binomial counts')
    u=1. if k==n else float(beta.ppf(.95,k+1,n-k))
    if k==0:require(abs(u-(-np.expm1(np.log(.05)/n)))<1e-14,'zero event bound')
    if k<n:require(abs(binom.cdf(k,n,u)-.05)<1e-10,'binomial inversion check')
    return u


class Neighborhood:
    """Compact per-anchor neighborhoods; one stationary local correlation matrix.

    Full circular footprint availability is tested BEFORE deliberate exclusions.
    No noise-valid interpolation flag is converted to zero covariance: finite
    endpoint continuation is computational and reported as extrapolated.
    """
    def __init__(self,header,shape,beam,scale,psd,k0,*,empirical=None):
        self.shape=tuple(shape);self.header=header;self.beam=beam
        self.scale=np.asarray(scale);self.k0=np.asarray(k0,bool)
        require(self.scale.shape==self.shape and self.k0.shape==self.shape,'neighborhood shape')
        require(np.isfinite(psd).all() and (psd>=0).all() and psd.mean()>0,'invalid covariance spectrum')
        self.x,self.y=coordinates(shape,header);self.disk=np.hypot(self.x,self.y)<=40
        dy=abs(header['CDELT2']*3600);dx=abs(header['CDELT1']*3600)
        hy=int(np.ceil(beam/dy));hx=int(np.ceil(beam/dx))
        off=np.array([(r,c) for r in range(-hy,hy+1) for c in range(-hx,hx+1) if (r*dy)**2+(c*dx)**2<=beam**2])
        self.offsets=off;self.center=np.all(off==0,axis=1)
        self.weights=np.exp(-4*np.log(2)*((off[:,0]*dy)**2+(off[:,1]*dx)**2)/beam**2)
        self.anchors=np.argwhere(self.disk);loc=self.anchors[:,None,:]+off[None,:,:]
        inside=np.all((loc>=0)&(loc<np.array(shape)),axis=(1,2));self.inside=inside
        safe=np.clip(loc,0,np.array(shape)-1);self.indices=safe[:,:,0]*shape[1]+safe[:,:,1]
        self.anchor_indices=self.anchors[:,0]*shape[1]+self.anchors[:,1]
        rho=np.fft.ifft2(psd).real;d=off[:,None,:]-off[None,:,:]
        require(max(shape)<min(psd.shape)/2+1,'covariance padding insufficient for joint field')
        self.rho=rho[d[:,:,0]%rho.shape[0],d[:,:,1]%rho.shape[1]]
        require(np.allclose(self.rho,self.rho.T,rtol=0,atol=1e-13),'covariance symmetry')
        self.zero=float(rho[0,0]);self.lags_supported=True
        if empirical is not None:
            middle=np.array(empirical.shape)//2;idx=d+middle
            self.lags_supported=bool(np.all(idx>=0) and np.all(idx<np.array(empirical.shape)) and np.isfinite(empirical[idx[:,:,0],idx[:,:,1]]).all())

    def scores(self,base):
        B=base['selected'];valid=base['valid'];b=base['brightness'];excluded=self.k0|B
        ii=self.indices;ai=self.anchor_indices
        full=self.inside & valid.ravel()[ii].all(axis=1) & np.isfinite(self.scale.ravel()[ii]).all(axis=1) & (self.scale.ravel()[ii]>0).all(axis=1)
        full &= self.lags_supported
        anchor=valid.ravel()[ai]&~excluded.ravel()[ai]
        w=self.weights[None,:]*(~excluded.ravel()[ii]);w[:,self.center]=0
        norm=w.sum(axis=1);q=np.divide(w,norm[:,None],out=np.zeros_like(w),where=norm[:,None]>0)
        conc=np.sum(q*q,axis=1);neff=np.divide(1.,conc,out=np.zeros_like(conc),where=conc>0)
        available=full&anchor&(norm>0)&(neff>=3)
        # nan values in unusable rows never make a test available.
        e=np.sum(q*np.nan_to_num(b.ravel()[ii]),axis=1)
        u=q*np.where(np.isfinite(self.scale.ravel()[ii]),self.scale.ravel()[ii],0.);vc=np.einsum('pi,ij,pj->p',u,self.rho,u,optimize=False);vd=np.sum(u*u,axis=1)*self.zero
        fields={}
        for name,values in [('evidence',e),('neff_weights',neff),('full_footprint',full.astype(float))]:
            a=np.full(self.shape,np.nan);a.ravel()[ai]=values;fields[name]=a
        fields['policies']={}
        for name,v in zip(POLICIES,(vc,vd)):
            ok=available&np.isfinite(e)&np.isfinite(v)&(v>0)
            z=np.divide(e,np.sqrt(np.maximum(v,0)),out=np.full_like(e,np.nan),where=ok)
            zs=np.full(self.shape,np.nan);vs=zs.copy();av=np.zeros(self.shape,bool)
            zs.ravel()[ai]=z;vs.ravel()[ai]=v;av.ravel()[ai]=ok
            fields['policies'][name]=dict(score=zs,variance=vs,available=av,numerical_failures=int(np.sum(available&~(np.isfinite(e)&np.isfinite(v)&(v>0)))))
        fields['eligible_anchors']=int(anchor.sum());return fields


def base_from_fit(signal,support,fixed,fit):
    # Retain finite constrained measurements in accounting, but preserve the
    # inherited no-candidate selection outcome. Do not turn it into a failure.
    if fit['status']!='success' or fit.get('constrained',False):return None
    return admit(signal,support,fixed.scale,fixed.noise_valid,fixed.header,np.asarray(fit['parameters']),3.,1.)


def infer(signal,support,fixed,neighborhood,fit=None):
    fit=fit_diagnostic(signal,support,fixed.header,fixed.beam) if fit is None else fit
    base=base_from_fit(signal,support,fixed,fit)
    fields=neighborhood.scores(base) if base is not None and base['selected'].any() else None
    return fit,base,fields


def select(base,fields,k0,policy,tau):
    if base is None:return None
    c={k:(v.copy() if isinstance(v,np.ndarray) else v.copy() if isinstance(v,dict) else v) for k,v in base.items()}
    B=base['selected'];extra=np.zeros(B.shape,bool)
    if B.any() and fields is not None:
        f=fields['policies'][policy]
        extra=base['valid']&~(k0|B)&f['available']&(base['brightness']>0)&(f['score']>tau)
    labels,_=label(B|extra,structure=CONNECTIVITY);connected=np.unique(labels[B]);connected=connected[connected!=0]
    selected=np.isin(labels,connected);added=selected&~B
    require(not np.any(B&~selected),'supplement removed BASE')
    c['selected']=selected;c['candidate']=np.where(base['valid'],np.where(selected,base['brightness'],0.),np.nan)
    c['excluded']=np.where(base['valid'],base['brightness']-c['candidate'],np.nan)
    c['proposed_extra']=extra;c['added']=added
    c['summary'].update(policy=policy,threshold_supplement=tau,base_pixels=int(B.sum()),proposed_extra_pixels=int(extra.sum()),added_pixels=int(added.sum()),pixels=int(selected.sum()),extra_brightness_area=float(np.sum(base['brightness'][added])*4),total_brightness_area=float(np.sum(c['candidate'][selected])*4),unavailable_tests=fields['eligible_anchors']-int(fields['policies'][policy]['available'].sum()) if fields else None)
    return c


def maximum(base,fields,policy):
    if base is None or fields is None:return None
    f=fields['policies'][policy];use=f['available']&(base['brightness']>0)
    return float(np.max(f['score'][use])) if use.any() else None
