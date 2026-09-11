#!/usr/bin/env python3
"""Matched P/G/R feedback trajectories on immutable registered parents."""
from pathlib import Path
import argparse,datetime,json,os,platform,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
import rbf_r02 as rbf
import truths

b=rbf.base;H=Path(__file__).resolve().parent

def free_evaluation(data,maps):
    result=[]
    for a in range(3):
        try:
            f=b.fit_source(maps[a],data.x,data.ygrid,data.D[a]);p=f['parameters'];plane=p[6]+p[7]*data.x/90+p[8]*data.ygrid/90
            ap=data.D[a]&(np.hypot(data.x-p[1],data.ygrid-p[2])<=40)
            f.update(available=True,aperture_sum_arcsec2=float(np.sum(maps[a,ap]-plane[ap])*4),exterior_rms=float(np.sqrt(np.mean(maps[a,data.O[a]]**2))),support_pixels=int(data.D[a].sum()))
        except Exception as error:f=dict(available=False,error=str(error))
        result.append(f)
    return result

class Run(b.Run):
    def __init__(self,out,bases):super().__init__(out);self.bases=bases;self.setup=sum(x.setup_seconds for x in bases)
    def trajectory(self,data,parent,case,arm):
        out=self.out/case/arm;out.mkdir(parents=True,exist_ok=False);model=np.zeros((3,data.npix));rows=[];evaluation_seconds=0.;start=time.monotonic()
        for basis in self.bases:basis.reset()
        try:
            for k in range(7):
                self.guard();t=time.monotonic();tod,state,guard=data.clean(parent,model);clean_seconds=time.monotonic()-t
                t=time.monotonic();maps=data.grid(tod);map_seconds=time.monotonic()-t;del tod
                ready=time.monotonic()-start-evaluation_seconds
                t=time.monotonic()
                if arm=='R':
                    next_model=np.zeros_like(model);coefficients={};decisions=[]
                    for a,basis in enumerate(self.bases):
                        v=maps[a,data.O[a]];scale=1.4826*np.median(abs(v-np.median(v)))
                        if not np.isfinite(scale) or scale<=0:raise ValueError('invalid scatter scale')
                        z,coeff,record=basis.infer(maps[a],scale);next_model[a]=z;coefficients[b.ARRAYS[a]]=coeff
                        record.update(admitted_pixels=int(np.count_nonzero(z)),model_peak=float(z.max()));decisions.append(record)
                    diagnostic_seconds=sum(d['diagnostic_seconds'] for d in decisions)
                    policy_seconds=time.monotonic()-t-diagnostic_seconds;t=time.monotonic();fits=free_evaluation(data,maps);eval_seconds=time.monotonic()-t;evaluation_seconds+=eval_seconds
                else:
                    diagnostic_seconds=0.
                    fits=b.describe(data,maps);fit_seconds=time.monotonic()-t
                    t=time.monotonic();next_model,decisions=b.inference(data,maps,arm,fits);policy_seconds=time.monotonic()-t
                    eval_seconds=fit_seconds if arm=='P' else 0.
                    if arm=='P':evaluation_seconds+=fit_seconds
                    else:policy_seconds+=fit_seconds
                t=time.monotonic()
                if arm=='R':np.savez_compressed(out/f'pass{k:02d}_coefficients.npz',**coefficients)
                np.savez_compressed(out/f'pass{k:02d}_maps.npz',total=maps,applied_model=model,next_model=next_model)
                np.savez_compressed(out/f'pass{k:02d}_state.npz',**state);output_seconds=time.monotonic()-t
                row=dict(case=case,arm=arm,pass_index=k,cleaning_passes=k+1,clean_seconds=clean_seconds,map_seconds=map_seconds,
                    policy_seconds=policy_seconds,diagnostic_seconds=diagnostic_seconds,evaluation_seconds=eval_seconds,output_seconds=output_seconds,cumulative_evaluation_only_seconds=evaluation_seconds,
                    cumulative_method_map_seconds=ready,first_observation_map_seconds=ready+(self.setup if arm=='R' else 0.),
                    cumulative_wall_seconds=time.monotonic()-start,fit=fits,decision=decisions,model_change_rms=float(np.sqrt(np.mean((next_model-model)**2))),
                    min_application_rank_ratio=guard,peak_rss_bytes=self.guard())
                b.write(out/f'pass{k:02d}.json',row);rows.append(row);model=next_model
                print(json.dumps({x:row[x] for x in ['case','arm','pass_index','cumulative_wall_seconds']}),flush=True)
            b.write(out/'COMPLETE.json',dict(passes=7,wall_seconds=time.monotonic()-start,peak_rss_bytes=self.guard()))
        except Exception as error:
            b.write(out/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc(),completed_passes=len(rows),wall_seconds=time.monotonic()-start));print('FAILED',case,arm,str(error),flush=True)

def verify_freeze():
    freeze=json.loads((H/'FREEZE_R0.2.json').read_text())
    for r in freeze['files']:
        if b.digest(H/r['path'])!=r['sha256']:raise ValueError('changed frozen file '+r['path'])
    return freeze

def main(out):
    freeze=verify_freeze();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    b.write(out/'RUN_START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=freeze,python=platform.python_version(),numpy=np.__version__,
        scipy=b.scipy.__version__,netcdf=b.netCDF4.__version__,platform=platform.platform(),pid=os.getpid()))
    try:
        with threadpool_limits(limits=4):
            b.write(out/'THREAD_POOLS.json',threadpool_info());data=b.Data(json.loads((rbf.PREV/'INPUT_PREFLIGHT.json').read_text()));scales=b.detector_scales(data)
            common_setup=time.monotonic()-start;bases=[rbf.Basis(data,a) for a in range(3)];run=Run(out,bases);run.start=start
            b.write(out/'SETUP.json',dict(common_input_seconds=common_setup,basis_seconds=sum(x.setup_seconds for x in bases),bases=[x.identity for x in bases],config=rbf.CONFIG,truth_config=truths.PSF_CONFIG))
            np.savez_compressed(out/'geometry.npz',x=data.x,y=data.ygrid,D=data.D,O=data.O,S=data.S,Q=data.Q,shape=data.shape,uid=data.uid,array=data.ar,network=data.nw,edges=data.edges,valid=data.valid,scales=scales)
            for a,basis in enumerate(bases):np.savez_compressed(out/f'basis_{b.ARRAYS[a]}.npz',centers=basis.centers,b=basis.b,H_data=basis.H.data,H_indices=basis.H.indices,H_indptr=basis.H.indptr,H_shape=basis.H.shape,
                phi_data=basis.phi.data,phi_indices=basis.phi.indices,phi_indptr=basis.phi.indptr,phi_shape=basis.phi.shape)
            cases=[]
            def execute(name,parent,truth=None,null=None):
                if truth is not None:np.savez_compressed(out/f'{name}_truth.npz',truth=truth)
                cases.append(dict(case=name,truth=truth is not None,null=null))
                order=[['P','G','R'],['R','P','G'],['G','R','P']][(len(cases)-1)%3]
                for arm in order:run.trajectory(data,parent,name,arm)
            execute('real123424',data.y)
            for seed in [20260911,20260912]:
                noise=b.nuisance(data,scales,seed);null=f'null{seed}';execute(null,noise)
                truth=b.truth_map(data);execute(f'gauss{seed}',noise+data.project(truth),truth,null)
                if seed==20260911:
                    truth=b.truth_map(data,True);execute('mismatch20260911',noise+data.project(truth),truth,null)
                    image=truths.optical_image(True)
                    for brightness,label in [(1.,'bright'),(.5,'half')]:
                        truth=truths.map_truth(data,image,brightness);execute('coma_'+label,noise+data.project(truth),truth,null)
                    plane=np.tile(13+7*data.x/90-4*data.ygrid/90,(3,1))
                    execute('background20260911',noise+data.project(plane),np.zeros_like(plane),null)
                    del image,plane
                del noise
            b.write(out/'CASES.json',cases);verify_freeze();b.write(out/'CAMPAIGN_COMPLETE.json',dict(cases=len(cases),wall_seconds=time.monotonic()-start,peak_rss_bytes=run.guard(),
                reserved_129081='historically characterized; no new RBF comparison performed'))
    except BaseException as error:b.write(out/'CAMPAIGN_FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()));raise
    finally:
        files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=b.digest(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='PRODUCT_MANIFEST.json']
        b.write(out/'PRODUCT_MANIFEST.json',dict(files=files,bytes=sum(x['bytes'] for x in files)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
