"""Arithmetic for the explicitly approved, isolated Q02 r0.6 screen."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import importlib.util
from pathlib import Path
import numpy as np
from scipy import stats

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
OLD=HERE.parent/'fruit_q02_weight_precision_2026-09-09/run_weight_precision.py'
spec=importlib.util.spec_from_file_location('q02_precision_reader',OLD)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
require,write_json,digest=legacy.require,legacy.write_json,legacy.digest
ARRAYS=legacy.ARRAYS
POLICIES=['U','N4U']
TAIL=.05/256

def weight_generation(n,mu,v,e):
    """Input is training sufficient statistics plus evaluation counts, never values."""
    n,mu,v,e=map(np.asarray,(n,mu,v,e))
    raw=np.full(v.shape,np.nan)
    required=e>0
    reason=np.zeros(v.shape,np.uint8)  # 0 valid, 1 unused, 2 n<2, 3 scatter, 4 reciprocal
    reason[~required]=1
    reason[required & (n<2)]=2
    reason[required & (n>=2) & (~np.isfinite(v) | (v<=0))]=3
    eligible=required & (n>=2) & np.isfinite(v) & (v>0)
    with np.errstate(all='ignore'):np.divide(1.,v,out=raw,where=eligible)
    reason[eligible & (~np.isfinite(raw) | (raw<=0))]=4
    H=required & (reason==0); L=required & ~H
    gamma=np.full(v.shape,np.nan)
    if not H.any():return dict(n=n,mu=mu,v=v,e=e,raw=raw,H=H,L=L,reason=reason,gamma=gamma,mean=np.nan,available=False)
    with np.errstate(all='ignore'):mean=np.sum(e[H]*raw[H])/e[H].sum()
    require(np.isfinite(mean) and mean>0,'Normalization unavailable')
    gamma[H]=raw[H]/mean;gamma[L]=1.
    require(np.all(np.isfinite(gamma[required]) & (gamma[required]>0)),'Normalized weight unavailable')
    require(np.isclose(np.sum(e[required]*gamma[required])/e[required].sum(),1,rtol=1e-12,atol=1e-14),'Weight mean differs from one')
    return dict(n=n,mu=mu,v=v,e=e,raw=raw,H=H,L=L,reason=reason,gamma=gamma,mean=mean,available=True)

def from_training(b,valid,e):
    return weight_generation(*legacy.moments(b,valid),e)

def support(q,numerator,c=.1):
    positive=np.isfinite(q)&(q>0)
    ordered=np.sort(q[positive]);n=len(ordered)
    index=(int(np.floor(.75*n))+n)//2 if n else -1
    star=float(ordered[index]) if n else 0.
    m=np.full(q.shape,np.nan)
    with np.errstate(all='ignore'):np.divide(numerator,q,out=m,where=positive)
    finite=np.isfinite(numerator)&np.isfinite(m)
    norm=positive&(q>=star*c/10)&finite;sci=positive&(q>=star*c)&finite
    cause=np.zeros(q.shape,np.uint8) # 1 Q absent, 2 arithmetic, 3 below norm, 4 norm only
    cause[~positive]=1;cause[positive&~finite]=2
    cause[positive&finite&~norm]=3;cause[norm&~sci]=4
    return dict(map=np.where(sci,m,np.nan),norm_map=np.where(norm,m,np.nan),S_norm=norm,S_sci=sci,cause=cause,Qstar=star,index=index)

def accumulate(pixel,b,gamma,npix):
    require(pixel.shape==b.shape==gamma.shape,'Occurrence shape mismatch')
    require(np.all((pixel>=0)&(pixel<npix)),'Pixel outside fixed grid')
    require(np.isfinite(b).all() and np.all(np.isfinite(gamma)&(gamma>0)),'Invalid admitted occurrence')
    q=np.bincount(pixel,weights=gamma,minlength=npix)
    numerator=np.bincount(pixel,weights=gamma*b,minlength=npix)
    require(np.isfinite(numerator).all() and np.isfinite(q).all(),'Accumulator overflow')
    count=np.bincount(pixel,minlength=npix)
    q2=np.bincount(pixel,weights=gamma*gamma,minlength=npix)
    return dict(numerator=numerator,Q=q,count=count,Q2=q2,**support(q,numerator))

def pixel_indices(x,y,h):
    require(np.isfinite(x).all() and np.isfinite(y).all(),'Invalid admitted coordinates')
    return np.floor(x/h+.5).astype(np.int64),np.floor(y/h+.5).astype(np.int64)

def moments_map(m,x,y):
    if not np.isfinite(m).all() or not len(m):return np.full(4,np.nan)
    total=m.sum()
    if total<=0:return np.full(4,np.nan)
    cx,cy=np.sum(m*x)/total,np.sum(m*y)/total
    vx,vy=np.sum(m*(x-cx)**2)/total,np.sum(m*(y-cy)**2)/total
    return np.array([cx,cy,np.sqrt(vx) if vx>0 else np.nan,np.sqrt(vy) if vy>0 else np.nan])

def source_metrics(m,C,O,T,x,y):
    if not C.any() or not O.any() or not np.isfinite(m[C|O]).all():return dict(amplitude=np.nan,centroid_x=np.nan,centroid_y=np.nan,width_x=np.nan,width_y=np.nan)
    s=m[C]-m[O].mean();den=np.dot(T[C],T[C])
    amp=np.dot(T[C],s)/den if den>0 else np.nan
    values=moments_map(s,x[C],y[C])
    return dict(zip(['amplitude','centroid_x','centroid_y','width_x','width_y'],np.r_[amp,values]))

def ring_metrics(m,mask,shape):
    n=int(mask.sum());v=m[mask]
    if not n or not np.isfinite(v).all():return dict(pixels=n,available=False)
    ans=dict(pixels=n,available=True,mean=float(v.mean()),power=float(np.mean(v*v)),centered_power=float(np.var(v)))
    a=m.reshape(shape);r=mask.reshape(shape)
    for name,sl1,sl2 in [('x',(slice(None),slice(None,-1)),(slice(None),slice(1,None))),('y',(slice(None,-1),slice(None)),(slice(1,None),slice(None)))]:
        use=r[sl1]&r[sl2];p,q=a[sl1][use],a[sl2][use]
        cov=float(np.mean(p*q)-p.mean()*q.mean()) if len(p) else np.nan
        den=np.std(p)*np.std(q) if len(p) else np.nan
        ans[name]=dict(pairs=len(p),covariance=cov,correlation=cov/den if den>0 else np.nan)
    return ans

def continuous_bounds(values,margin=0.,absolute=False):
    v=np.asarray(values)
    if not np.isfinite(v).all() or len(v)<2:
        return dict(n=len(v),available=False,disposition='inconclusive',margin=margin)
    mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(len(v)));half=float(stats.t.ppf(1-TAIL,len(v)-1)*se)
    lo,hi=mean-half,mean+half
    passed=(lo>=-margin and hi<=margin) if absolute else hi<=margin
    failed=(hi< -margin or lo>margin) if absolute else lo>margin
    return dict(n=len(v),available=True,mean=mean,se=se,lower=lo,upper=hi,margin=margin,absolute=absolute,disposition='pass' if passed else 'failure' if failed else 'inconclusive')

def binomial_bounds(success,n):
    lo=float(stats.beta.ppf(TAIL,success,n-success+1)) if success else 0.
    hi=float(stats.beta.ppf(1-TAIL,success+1,n-success)) if success<n else 1.
    return dict(successes=int(success),n=int(n),lower=lo,upper=hi)

def tail_registry():
    tails=[]
    for row in range(9):
        tails.extend(f'{row}/noise/{side}' for side in ['lower','upper'])
        for metric,axes in [('amplitude',['scalar']),('centroid',['x','y']),('width',['x','y'])]:
            tails.extend(f'{row}/{metric}/{axis}/{arm}/{side}' for axis in axes for arm in POLICIES for side in ['lower','upper'])
        tails.extend(f'{row}/excursion/{arm}/{side}' for arm in POLICIES for side in ['lower','upper'])
    tails.extend(f'{row}/analytic_U/{side}' for row in range(4) for side in ['lower','upper'])
    require(len(tails)==len(set(tails))==242,'Tail enumeration mismatch')
    return tails
