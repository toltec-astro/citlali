#!/usr/bin/env python3
"""Frozen POINT positional-prior screen; predecessor cleaning is imported unchanged."""
from pathlib import Path
import argparse, datetime, json, os, platform, sys, time, traceback
import numpy as np
from scipy import optimize
from threadpoolctl import threadpool_limits, threadpool_info

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'fruit_point_coherent_feedback_2026-09-11'
sys.path.insert(0,str(PREV))
import experiment_r02 as b

FREE=[0,3,4,5,6,7,8]

def unpack_pair(q,a):
    p=np.zeros(9);p[1:3]=q[:2];p[FREE]=q[2+7*a:9+7*a];return p

def pair_jac(q,a,x,y):
    J=b.model_jac(unpack_pair(q,a),x,y);out=np.zeros((len(x),16))
    out[:,:2]=J[:,1:3];out[:,2+7*a:9+7*a]=J[:,FREE];return out

def unpack_offset(q,center):
    p=np.zeros(9);p[1:3]=center+q[0]*np.array([np.cos(q[1]),np.sin(q[1])]);p[FREE]=q[2:];return p

def offset_jac(q,center,x,y):
    J=b.model_jac(unpack_offset(q,center),x,y);out=np.empty((len(x),9))
    c,s=np.cos(q[1]),np.sin(q[1]);out[:,0]=J[:,1]*c+J[:,2]*s
    out[:,1]=q[0]*(-J[:,1]*s+J[:,2]*c);out[:,2:]=J[:,FREE];return out

def record_fit(p,z,x,y,result,nstarts):
    widths=np.exp(p[3:5]);order=np.argsort(widths)[::-1];rem=z-b.gaussian(p,x,y,True)
    boundary=bool(np.any(abs(p[1:3])>=79.999) or np.any(widths<=4.001) or np.any(widths>=59.999))
    return dict(parameters=p,peak=p[0],centroid=p[1:3],widths=widths[order],
        angle_rad=(p[5]+(np.pi/2 if order[0] else 0))%np.pi,gaussian_integral=p[0]*2*np.pi*np.prod(widths/b.FWHM),
        background=p[6:],residual_rms=np.sqrt(np.mean(rem**2)),sse=np.sum(rem**2),boundary_rejection=boundary,
        solver_status=int(result.status),nfev=int(result.nfev),successful_starts=nstarts)

def solve(fun,jac,q,lo,hi):
    return optimize.least_squares(fun,q,jac=jac,bounds=(lo,hi),max_nfev=250,
        x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-8)

def prior_fits(data,maps,free_fits):
    xy=[(data.x[data.D[a]],data.ygrid[data.D[a]]) for a in range(3)]
    zz=[maps[a,data.D[a]] for a in range(3)]
    if any(len(z)<100 or not np.isfinite(z).all() for z in zz):raise ValueError('invalid prior-fit domain')
    flo=[-np.inf,np.log(4),np.log(4),-np.pi,-np.inf,-np.inf,-np.inf]
    fhi=[np.inf,np.log(60),np.log(60),np.pi,np.inf,np.inf,np.inf]
    lo=np.r_[[-80.,-80.],flo,flo];hi=np.r_[[80.,80.],fhi,fhi]
    scale=max(float(np.std(np.concatenate(zz[:2]))),1e-12)
    def fun(q):return np.concatenate([(b.gaussian(unpack_pair(q,a),*xy[a],True)-zz[a])/scale for a in range(2)])
    def jac(q):return np.vstack([pair_jac(q,a,*xy[a])/scale for a in range(2)])
    starts=[np.asarray(free_fits[a]['centroid']) for a in range(2)]
    starts.append((starts[0]+starts[1])/2);results=[]
    for center in starts:
        q=np.r_[np.clip(center,-79.99,79.99),np.asarray(free_fits[0]['parameters'])[FREE],np.asarray(free_fits[1]['parameters'])[FREE]]
        q=np.maximum(np.minimum(q,hi-1e-8),lo+1e-8)
        res=solve(fun,jac,q,lo,hi)
        if res.success and np.isfinite(res.x).all():results.append(res)
    if not results:raise ValueError('all reference-pair fit starts failed')
    best=min(results,key=lambda r:r.cost);center=best.x[:2]
    fits=[record_fit(unpack_pair(best.x,a),zz[a],*xy[a],best,len(results)) for a in range(2)]
    delta=np.asarray(free_fits[2]['centroid'])-center;phi=np.arctan2(delta[1],delta[0])
    scale2=max(float(np.std(zz[2])),1e-12);lo2=np.r_[[0.,-np.inf],flo];hi2=np.r_[[2.,np.inf],fhi]
    def f2(q):return (b.gaussian(unpack_offset(q,center),*xy[2],True)-zz[2])/scale2
    def j2(q):return offset_jac(q,center,*xy[2])/scale2
    results2=[]
    for w in [6.,18.,42.]:
        p=np.asarray(free_fits[2]['parameters']).copy();p[3:5]=np.log(w)
        q=np.r_[[1.,phi],p[FREE]];q=np.maximum(np.minimum(q,hi2-1e-8),lo2+1e-8)
        res=solve(f2,j2,q,lo2,hi2)
        if res.success and np.isfinite(res.x).all():results2.append(res)
    if not results2:raise ValueError('all conditional a2000 fit starts failed')
    best2=min(results2,key=lambda r:r.cost);p=unpack_offset(best2.x,center)
    fits.append(record_fit(p,zz[2],*xy[2],best2,len(results2)))
    info=dict(common_centroid=center,a2000_offset=p[1:3]-center,a2000_offset_radius=best2.x[0],
        a2000_prior_tension=bool(best2.x[0]>=1.999),reference_pair_sse=2*best.cost*scale**2)
    return fits,info

def prior_inference(data,maps,fits,info):
    model,records=b.inference(data,maps,'G',fits)
    anchor=any(r['reason']=='coherent_source' for r in records[:2])
    cause='relative_position_prior' if anchor else 'no_reference_pair_anchor'
    if info['a2000_prior_tension']:cause='a2000_offset_boundary_tension'
    if not anchor or info['a2000_prior_tension']:
        model[2]=0;records[2].update(reason=cause,admitted_pixels=0,model_peak=0.)
    info=dict(info,reference_pair_anchor=anchor)
    return model,records,info

def shifted_truth(data,kind):
    truth=b.truth_map(data)
    if kind in ['near','outside']:
        offsets=[(0,0),(.2,-.1),(1.2,-1.2)] if kind=='near' else [(0,0),(0,0),(4,0)]
        for a,(dx,dy) in enumerate(offsets):
            p=np.array([100.,17+dx,-11+dy,np.log(12.),np.log(8.),np.deg2rad(25),0,0,0])
            truth[a]=np.where(data.D[a],b.gaussian(p,data.x,data.ygrid),0)
    elif kind=='absent':truth[2]=0
    else:raise ValueError(kind)
    return truth

def contaminant(data):
    p=np.array([180.,-24.,24.,np.log(18.),np.log(10.),.3,0,0,0]);z=np.zeros((3,data.npix))
    z[2]=np.where(data.D[2],b.gaussian(p,data.x,data.ygrid),0);return z

class Run(b.Run):
    def trajectory(self,data,parent,case,arm):
        out=self.out/case/arm;out.mkdir(parents=True,exist_ok=False);model=np.zeros((3,data.npix));start=time.monotonic();rows=[]
        try:
            for k in range(7):
                self.guard();t=time.monotonic();tod,state,guard=data.clean(parent,model);clean_seconds=time.monotonic()-t
                t=time.monotonic();maps=data.grid(tod);map_seconds=time.monotonic()-t;del tod
                ready=time.monotonic()-start
                t=time.monotonic();fits=b.describe(data,maps);fit_seconds=time.monotonic()-t
                t=time.monotonic()
                if arm=='J':
                    policy_fits,info=prior_fits(data,maps,fits)
                    next_model,decision,info=prior_inference(data,maps,policy_fits,info)
                else:
                    policy_fits=fits;info=None;next_model,decision=b.inference(data,maps,'G',fits)
                policy_seconds=fit_seconds+time.monotonic()-t
                delta=float(np.sqrt(np.mean((next_model-model)**2)))
                np.savez_compressed(out/f'pass{k:02d}_maps.npz',total=maps,applied_model=model,next_model=next_model)
                np.savez_compressed(out/f'pass{k:02d}_state.npz',**state)
                row=dict(case=case,arm=arm,pass_index=k,cleaning_passes=k+1,clean_seconds=clean_seconds,map_seconds=map_seconds,
                    source_fit_seconds=fit_seconds,policy_seconds=policy_seconds,cumulative_method_map_seconds=ready,
                    cumulative_wall_seconds=time.monotonic()-start,cumulative_evaluation_only_seconds=0.,fit=fits,policy_fit=policy_fits,
                    prior=info,decision=decision,model_change_rms=delta,min_application_rank_ratio=guard,peak_rss_bytes=self.guard())
                b.write(out/f'pass{k:02d}.json',row);rows.append(row);model=next_model
                print(json.dumps({key:row[key] for key in ['case','arm','pass_index','cumulative_wall_seconds']}),flush=True)
            b.write(out/'COMPLETE.json',dict(status='complete',passes=7,wall_seconds=time.monotonic()-start,peak_rss_bytes=self.guard()))
        except Exception as error:
            b.write(out/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc(),completed_passes=len(rows),wall_seconds=time.monotonic()-start))
            print('FAILED',case,arm,str(error),flush=True)
        return rows

def verify_freeze():
    freeze=json.loads((HERE/'FREEZE.json').read_text())
    for r in freeze['files']:
        if b.digest(HERE/r['path'])!=r['sha256']:raise ValueError('changed frozen file '+r['path'])
    return freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    freeze=verify_freeze();out=args.output;out.mkdir(parents=True,exist_ok=False);run=Run(out)
    b.write(out/'RUN_START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=freeze,python=platform.python_version(),
        numpy=np.__version__,scipy=b.scipy.__version__,netcdf=b.netCDF4.__version__,platform=platform.platform(),pid=os.getpid()))
    try:
        with threadpool_limits(limits=4):
            b.write(out/'THREAD_POOLS.json',threadpool_info());data=b.Data(json.loads((PREV/'INPUT_PREFLIGHT.json').read_text()));scales=b.detector_scales(data)
            np.savez_compressed(out/'fixed_geometry.npz',x=data.x,y=data.ygrid,shape=data.shape,Q=data.Q,D=data.D,O=data.O,normalization_support=data.S,
                detector_scales=scales,uid=data.uid,network=data.nw,array=data.ar,edges=data.edges,valid_mask=data.valid)
            b.write(out/'INPUT_READY.json',dict(wall_seconds=time.monotonic()-run.start,input_shape=data.y.shape,map_shape=data.shape,groups=len(data.groups)))
            cases=[]
            def execute(name,parent,truth=None,null=None):
                if truth is not None:np.savez_compressed(out/f'{name}_truth.npz',truth=truth)
                cases.append(dict(case=name,null=null,truth=truth is not None))
                for arm in (['G','J'] if len(cases)%2 else ['J','G']):run.trajectory(data,parent,name,arm)
            execute('real123424',data.y)
            for seed in [20260911,20260912]:
                noise=b.nuisance(data,scales,seed);null=f'null{seed}'
                execute(null,noise)
                truth=b.truth_map(data);execute(f'gauss{seed}',noise+data.project(truth),truth,null)
                if seed==20260911:
                    truth=b.truth_map(data,True);execute('mismatch20260911',noise+data.project(truth),truth,null)
                    for kind in ['near','outside','absent']:
                        truth=shifted_truth(data,kind);execute(kind+'20260911',noise+data.project(truth),truth,null)
                    feature=contaminant(data);np.savez_compressed(out/'contaminant_feature.npz',feature=feature)
                    background=noise+data.project(feature);execute('contaminant20260911',background)
                    truth=b.truth_map(data);execute('source_contaminant20260911',background+data.project(truth),truth,'contaminant20260911')
                    del background,feature
                del noise
            b.write(out/'CASES.json',cases);verify_freeze()
            b.write(out/'CAMPAIGN_COMPLETE.json',dict(wall_seconds=time.monotonic()-run.start,peak_rss_bytes=run.guard(),cases=len(cases),reserved_129081='not opened'))
    except BaseException as error:
        b.write(out/'CAMPAIGN_FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()));raise
    finally:
        files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=b.digest(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='PRODUCT_MANIFEST.json']
        b.write(out/'PRODUCT_MANIFEST.json',dict(files=files,bytes=sum(r['bytes'] for r in files)))

if __name__=='__main__':main()
