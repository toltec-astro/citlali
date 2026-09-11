#!/usr/bin/env python3
"""Isolated, preregistered POINT feedback experiment; no production imports."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, platform, resource, time, traceback
from pathlib import Path
import numpy as np
import scipy
from scipy import linalg, optimize, signal
import netCDF4
from threadpoolctl import threadpool_limits, threadpool_info

HERE=Path(__file__).resolve().parent
ARRAYS=['a1100','a1400','a2000']
FWHM=2*np.sqrt(2*np.log(2))
RAD=648000/np.pi

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def native(v):
    if isinstance(v,dict): return {str(k):native(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [native(x) for x in v]
    if isinstance(v,np.ndarray): return native(v.tolist())
    if isinstance(v,np.generic): return native(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v

def write(p,v):
    Path(p).write_text(json.dumps(native(v),indent=2,allow_nan=False)+'\n')

def mm(a,b):
    # macOS Accelerate may leave floating-point status flags set despite a finite
    # product. Local suppression is paired with an explicit finite-result gate.
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        result=np.matmul(a,b)
    if not np.isfinite(result).all(): raise FloatingPointError('nonfinite matrix product')
    return result

def gaussian(p,x,y,with_plane=False):
    A,cx,cy,lx,ly,ang,b0,bx,by=p
    c,s=np.cos(ang),np.sin(ang); dx=x-cx;dy=y-cy
    u=c*dx+s*dy; v=-s*dx+c*dy
    G=A*np.exp(-.5*((u/(np.exp(lx)/FWHM))**2+(v/(np.exp(ly)/FWHM))**2))
    return G+b0+bx*x/90+by*y/90 if with_plane else G

def model_jac(p,x,y):
    A,cx,cy,lx,ly,ang,*_=p;c,s=np.cos(ang),np.sin(ang)
    u=c*(x-cx)+s*(y-cy);v=-s*(x-cx)+c*(y-cy)
    ix=(FWHM/np.exp(lx))**2;iy=(FWHM/np.exp(ly))**2
    E=np.exp(-.5*(u*u*ix+v*v*iy));G=A*E
    return np.column_stack([E,G*(u*c*ix-v*s*iy),G*(u*s*ix+v*c*iy),G*u*u*ix,G*v*v*iy,G*u*v*(iy-ix),np.ones_like(x),x/90,y/90])

def fit_source(z,x,y,domain):
    good=domain&np.isfinite(z); xx=x[good];yy=y[good];zz=z[good]
    if len(zz)<100: raise ValueError('insufficient source-fit support')
    H=np.column_stack([np.ones(len(zz)),xx/90,yy/90]);b=np.linalg.lstsq(H,zz,rcond=None)[0]
    r=zz-mm(H,b);imax=np.argmax(r);cx=np.clip(xx[imax],-79,79);cy=np.clip(yy[imax],-79,79)
    scale=max(float(np.std(r)),1e-12); lower=[-np.inf,-80,-80,np.log(4),np.log(4),-np.pi,-np.inf,-np.inf,-np.inf]
    upper=[np.inf,80,80,np.log(60),np.log(60),np.pi,np.inf,np.inf,np.inf]
    results=[]
    for w in [6.,18.,42.]:
        p=np.array([r[imax],cx,cy,np.log(w),np.log(w),.2,*b])
        res=optimize.least_squares(lambda p:(gaussian(p,xx,yy,True)-zz)/scale,p,
            jac=lambda p:model_jac(p,xx,yy)/scale,bounds=(lower,upper),max_nfev=250,
            x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-8)
        if res.success and np.isfinite(res.x).all(): results.append(res)
    if not results: raise ValueError('all three numerical source-fit starts failed')
    res=min(results,key=lambda z:z.cost);p=res.x; widths=np.exp(p[3:5]);order=np.argsort(widths)[::-1]
    boundary=bool(np.any(np.abs(p[1:3])>=79.999) or np.any(widths<=4.001) or np.any(widths>=59.999))
    rem=zz-gaussian(p,xx,yy,True)
    return dict(parameters=p,peak=p[0],centroid=p[1:3],widths=widths[order],angle_rad=(p[5]+(np.pi/2 if order[0] else 0))%np.pi,
        gaussian_integral=p[0]*2*np.pi*np.prod(widths/FWHM),background=p[6:9],residual_rms=np.sqrt(np.mean(rem**2)),
        sse=np.sum(rem**2),boundary_rejection=boundary,solver_status=int(res.status),nfev=int(res.nfev),successful_starts=len(results))

def support(q,c=.1):
    v=np.sort(q[q>0]);N=len(v);ref=v[(int(np.floor(.75*N))+N)//2] if N else 0.
    return (q>0)&(q>=ref*c/10),(q>0)&(q>=ref*c)

class Data:
    def __init__(self,record):
        path=Path(record['input']['path']);assert digest(path)==record['input']['sha256']
        with netCDF4.Dataset(path) as f:
            f.set_auto_mask(False);self.y=np.asarray(f['signal'][:],float)
            flags=f['flags'][:];apt=f['apt_flag'][:];self.ar=f['apt_array'][:].astype(int);self.nw=f['apt_nw'][:].astype(int)
            self.uid=f['apt_uid'][:];self.valid=(flags==0)&(apt[None,:]==0)
            if not np.isfinite(self.y[self.valid]).all(): raise ValueError('nonfinite required input')
            self.edges=np.asarray(record['recovered_edges']);el=f['TelElAct'][:][:,None]
            x=f['az_phys'][:][:,None]*RAD+np.cos(el)*f['apt_x_t'][:]-np.sin(el)*f['apt_y_t'][:]+f['pointing_offset_az'][:][:,None]
            y=f['alt_phys'][:][:,None]*RAD+np.cos(el)*f['apt_y_t'][:]+np.sin(el)*f['apt_x_t'][:]+f['pointing_offset_alt'][:][:,None]
        if not np.isfinite(x[self.valid]).all() or not np.isfinite(y[self.valid]).all(): raise ValueError('nonfinite coordinate')
        j=np.floor(x/2+.5).astype(np.int32);k=np.floor(y/2+.5).astype(np.int32)
        j0,j1=int(j[self.valid].min()),int(j[self.valid].max());k0,k1=int(k[self.valid].min()),int(k[self.valid].max())
        self.shape=(k1-k0+1,j1-j0+1);self.npix=int(np.prod(self.shape))
        if self.npix>1_000_000: raise ValueError('grid exceeds fixed resource bound')
        self.pixel=np.clip((k-k0)*self.shape[1]+j-j0,0,self.npix-1).astype(np.int32)
        # Pixel entries outside valid input are inaccessible to every use.
        ky,jx=np.mgrid[k0:k1+1,j0:j1+1];self.x=2*jx.ravel();self.ygrid=2*ky.ravel();r=np.hypot(self.x,self.ygrid)
        self.cols=[np.flatnonzero(self.ar==a) for a in range(3)]
        self.Q=[];self.D=[];self.O=[];self.S=[]
        for a,cols in enumerate(self.cols):
            good=self.valid[:,cols];q=np.bincount(self.pixel[:,cols][good],minlength=self.npix).astype(float)
            sn,ss=support(q);self.Q.append(q);self.S.append(sn);self.D.append(ss&(r<=90));self.O.append(ss&(r>90)&(r<=150))
            if self.D[-1].sum()<100 or self.O[-1].sum()<100: raise ValueError('insufficient fixed region')
        self.Q=np.array(self.Q);self.S=np.array(self.S);self.D=np.array(self.D);self.O=np.array(self.O)
        self.groups=[]
        for c,(lo,hi) in enumerate(zip(self.edges[:-1],self.edges[1:])):
            for g in np.unique(self.nw):
                cols=np.flatnonzero((self.nw==g)&(self.valid[lo:hi].sum(axis=0)>1))
                if not len(cols): continue
                if len(cols)<=5 or len(np.unique(self.ar[cols]))!=1: raise ValueError('group/rank infeasible')
                mask=self.valid[lo:hi,cols].astype(float);counts=mask.sum(axis=0)
                self.groups.append((int(c),int(g),int(lo),int(hi),cols,mask,counts,mm(mask.T,mask)-1))
    def project(self,model):
        return np.where(self.valid,model[self.ar[None,:],self.pixel],0.)
    def grid(self,tod):
        maps=np.full((3,self.npix),np.nan)
        for a,cols in enumerate(self.cols):
            good=self.valid[:,cols];values=tod[:,cols][good]
            if not np.isfinite(values).all(): raise ValueError('required clean output unavailable')
            numer=np.bincount(self.pixel[:,cols][good],weights=values,minlength=self.npix)
            maps[a,self.S[a]]=numer[self.S[a]]/self.Q[a,self.S[a]]
        return maps
    def clean(self,parent,model):
        rho=parent-self.project(model);cleaned=np.full(parent.shape,np.nan);saved={};guards=[]
        for c,g,lo,hi,cols,mask,counts,den in self.groups:
            raw=rho[lo:hi,cols];lam=np.sum(np.where(mask!=0,raw,0),axis=0)/counts
            centered=np.where(mask!=0,raw-lam,0);num=mm(centered.T,centered)
            cov=np.divide(num,den,out=np.zeros_like(num),where=den>0)
            ev,A=linalg.eigh(cov,subset_by_index=[len(cols)-5,len(cols)-1],driver='evr',check_finite=False)
            if ev[0]<=0 or not np.isfinite(ev).all(): raise ValueError(f'nonpositive requested mode {c}/{g}')
            N=np.einsum('td,di,dj->tij',mask,A,A,optimize=True);rhs=mm(centered,A)
            active=mask.sum(axis=1)>0;vals=np.linalg.eigvalsh(N[active]);ratios=vals[:,0]/vals[:,-1]
            if np.any(ratios<=1e-10): raise ValueError(f'group-time rank guard {c}/{g}')
            coeff=np.zeros_like(rhs);coeff[active]=np.linalg.solve(N[active],rhs[active,:,None])[...,0]
            value=(raw-lam)-mm(coeff,A.T)
            # Invalid entries have no retention or map influence.
            cleaned[lo:hi,cols]=np.where(mask!=0,value,np.nan)
            prefix=f'c{c:02d}_nw{g:02d}';saved[prefix+'_columns']=cols;saved[prefix+'_location']=lam;saved[prefix+'_basis']=A;saved[prefix+'_eigenvalues']=ev
            guards.append(float(ratios.min()))
        return cleaned+self.project(model),saved,min(guards)

def detector_scales(data):
    pieces=[[] for _ in range(data.y.shape[1])]
    for lo,hi in zip(data.edges[:-1],data.edges[1:]):
        d=np.diff(data.y[lo:hi],axis=0);good=data.valid[lo:hi-1]&data.valid[lo+1:hi]
        for j in np.flatnonzero(good.any(axis=0)): pieces[j].append(d[good[:,j],j])
    result=np.zeros(data.y.shape[1])
    for j,parts in enumerate(pieces):
        if parts:
            v=np.concatenate(parts);result[j]=1.4826*np.median(np.abs(v-np.median(v)))/np.sqrt(2)
            if not np.isfinite(result[j]) or result[j]<=0: raise ValueError('invalid synthetic scale')
    return result

def nuisance(data,scales,seed):
    rng=np.random.Generator(np.random.PCG64(seed));N=data.y.shape[0]
    e=rng.standard_normal(data.y.shape);e=signal.lfilter([np.sqrt(1-.3**2)],[1,-.3],e,axis=0)*scales
    for g in np.unique(data.nw):
        cols=np.flatnonzero(data.nw==g);common=[]
        for phi in [.98,.90,.70]:
            z=signal.lfilter([np.sqrt(1-phi*phi)],[1,-phi],rng.standard_normal(N));common.append(z)
        load=rng.standard_normal((3,len(cols)))*scales[cols]*5
        e[:,cols]+=mm(np.column_stack(common),load)
    return e

def truth_map(data,mismatch=False):
    p=np.array([80. if mismatch else 100.,17.,-11.,np.log(12.),np.log(8.),np.deg2rad(25),0,0,0])
    G=gaussian(p,data.x,data.ygrid)
    if mismatch:
        p[0]=20;p[1]+=10;p[2]+=6;G+=gaussian(p,data.x,data.ygrid)
    return np.where(data.D,G[None,:],0)

def describe(data,maps):
    result=[]
    for a in range(3):
        fit=fit_source(maps[a],data.x,data.ygrid,data.D[a]);p=fit['parameters'];r=np.hypot(data.x-p[1],data.ygrid-p[2])
        plane=p[6]+p[7]*data.x/90+p[8]*data.ygrid/90
        ap=data.D[a]&(r<=40);values=maps[a,ap]-plane[ap]
        fit.update(aperture_sum_arcsec2=float(np.sum(values)*4),exterior_rms=float(np.sqrt(np.mean(maps[a,data.O[a]]**2))),support_pixels=int(data.D[a].sum()))
        result.append(fit)
    return result

def inference(data,maps,arm,fit):
    model=np.zeros_like(maps);records=[]
    for a in range(3):
        v=maps[a,data.O[a]];R=1.4826*np.median(np.abs(v-np.median(v)))
        if not np.isfinite(R) or R<=0: raise ValueError('invalid empirical scatter score')
        if arm=='P':
            admitted=data.D[a]&(maps[a]>3*R);model[a,admitted]=maps[a,admitted];reason='pixelwise_score'
        else:
            accepted=fit[a]['peak']>3*R and not fit[a]['boundary_rejection']
            if accepted: model[a,data.D[a]]=gaussian(fit[a]['parameters'],data.x[data.D[a]],data.ygrid[data.D[a]])
            reason='coherent_source' if accepted else ('compact_model_boundary_rejection' if fit[a]['boundary_rejection'] else 'below_positive_peak_score')
        records.append(dict(scale=R,threshold=3.,admitted_pixels=int(np.count_nonzero(model[a])),reason=reason,model_peak=float(model[a].max())))
    return model,records

class Run:
    def __init__(self,out):self.out=out;self.start=time.monotonic()
    def guard(self):
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system()!='Darwin':rss*=1024
        if time.monotonic()-self.start>7200 or rss>8*2**30: raise RuntimeError('campaign time/memory resource limit')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>4*2**30: raise RuntimeError('retained-output resource limit')
        return int(rss)
    def trajectory(self,data,parent,case,arm):
        out=self.out/case/arm;out.mkdir(parents=True,exist_ok=False);model=np.zeros((3,data.npix));start=time.monotonic();rows=[];evaluation_seconds=0.
        try:
            for k in range(7):
                self.guard();t=time.monotonic();tod,state,guard=data.clean(parent,model);clean_seconds=time.monotonic()-t
                t=time.monotonic();maps=data.grid(tod);map_seconds=time.monotonic()-t;del tod
                map_ready_method_seconds=time.monotonic()-start-evaluation_seconds
                t=time.monotonic();fits=describe(data,maps);fit_seconds=time.monotonic()-t
                t=time.monotonic();next_model,decision=inference(data,maps,arm,fits);policy_seconds=time.monotonic()-t
                if arm=='P': evaluation_seconds+=fit_seconds
                # Terminal next-model inference is retained, never applied; map-ready timing precedes it.
                delta=float(np.sqrt(np.mean((next_model-model)**2)))
                np.savez_compressed(out/f'pass{k:02d}_maps.npz',total=maps,applied_model=model,next_model=next_model)
                np.savez_compressed(out/f'pass{k:02d}_state.npz',**state)
                row=dict(case=case,arm=arm,pass_index=k,cleaning_passes=k+1,clean_seconds=clean_seconds,map_seconds=map_seconds,
                    source_fit_seconds=fit_seconds,policy_seconds=policy_seconds+(fit_seconds if arm=='G' else 0),
                    cumulative_wall_seconds=time.monotonic()-start,cumulative_evaluation_only_seconds=evaluation_seconds,
                    cumulative_method_map_seconds=map_ready_method_seconds,fit=fits,decision=decision,
                    model_change_rms=delta,min_application_rank_ratio=guard,peak_rss_bytes=self.guard())
                write(out/f'pass{k:02d}.json',row);rows.append(row);model=next_model
                print(json.dumps(native({key:row[key] for key in ['case','arm','pass_index','cumulative_wall_seconds']})),flush=True)
            write(out/'COMPLETE.json',dict(status='complete',passes=7,wall_seconds=time.monotonic()-start,peak_rss_bytes=self.guard()))
        except Exception as e:
            write(out/'FAILURE.json',dict(status='failed',error=str(e),traceback=traceback.format_exc(),completed_passes=len(rows),wall_seconds=time.monotonic()-start));print('FAILED',case,arm,str(e),flush=True)
        return rows

def verify_freeze():
    f=json.loads((HERE/'FREEZE.json').read_text())
    for row in f['files']:
        if digest(HERE/row['path'])!=row['sha256']:raise ValueError('changed registered source '+row['path'])
    return f

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    freeze=verify_freeze();run=Run(out)
    write(out/'RUN_START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=freeze,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,netcdf=netCDF4.__version__,platform=platform.platform(),pid=os.getpid()))
    try:
        with threadpool_limits(limits=4):
            write(out/'THREAD_POOLS.json',threadpool_info());data=Data(json.loads((HERE/'INPUT_PREFLIGHT.json').read_text()))
            scales=detector_scales(data)
            np.savez_compressed(out/'fixed_geometry.npz',x=data.x,y=data.ygrid,shape=data.shape,Q=data.Q,D=data.D,O=data.O,normalization_support=data.S,detector_scales=scales,uid=data.uid,network=data.nw,array=data.ar,edges=data.edges,valid_mask=data.valid)
            write(out/'INPUT_READY.json',dict(wall_seconds=time.monotonic()-run.start,input_shape=data.y.shape,map_shape=data.shape,valid_occurrences=int(data.valid.sum()),groups=len(data.groups),scale_quantiles=np.quantile(scales[scales>0],[0,.5,1])))
            # Fixed balanced order avoids confounding arm order with every case.
            for arm in ['P','G']:run.trajectory(data,data.y,'real123424',arm)
            for seed in [20260911,20260912]:
                noise=nuisance(data,scales,seed)
                for case,truth in [(f'null{seed}',None),(f'gauss{seed}',truth_map(data))]+([(f'mismatch{seed}',truth_map(data,True))] if seed==20260911 else []):
                    parent=noise if truth is None else noise+data.project(truth)
                    if truth is not None: np.savez_compressed(out/f'{case}_truth.npz',truth=truth)
                    for arm in (['G','P'] if case.startswith('null') or case.startswith('mismatch') else ['P','G']):run.trajectory(data,parent,case,arm)
                    del parent
                del noise
            verify_freeze();write(out/'CAMPAIGN_COMPLETE.json',dict(status='campaign_finished_see_each_trajectory',wall_seconds=time.monotonic()-run.start,peak_rss_bytes=run.guard(),reserved_129081='not opened'))
    except BaseException as e:
        write(out/'CAMPAIGN_FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-run.start));raise
    finally:
        files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='PRODUCT_MANIFEST.json']
        write(out/'PRODUCT_MANIFEST.json',dict(files=files,bytes=sum(x['bytes'] for x in files)))
if __name__=='__main__':main()
