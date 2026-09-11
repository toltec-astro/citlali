#!/usr/bin/env python3
"""Approved one-bootstrap screen; map additions do not rerun learning."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,datetime,resource,platform,signal,os,hashlib,traceback
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
import starlet
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'fruit_point_coherent_feedback_2026-09-11'))
import experiment_r02 as b
sys.path.insert(0,str(H.parent/'fruit_point_rbf_feedback_2026-09-11'))
import truths
OLD=Path('/private/tmp/sci-fruit-point-rbf-feedback-20260911-r0.1')
OUT=Path('/private/tmp/sci-fruit-point-starlet-preliminary-20260911-r0.1')
read=lambda p:json.loads(Path(p).read_text())

def oracle(g,observed,paired,reference,out):
    # Same three-component realized-template definition as the frozen audit.
    x,y,Q,S,D,O=g.x,g.ygrid,g.Q,g.S,g.D,g.O
    B=np.column_stack([np.ones(len(x)),x/90,y/90]);rad=np.hypot(x-17,y+11)
    lookup={(int(xx),int(yy)):i for i,(xx,yy) in enumerate(zip(x,y))};rows=[];saved={}
    for a,array in enumerate(b.ARRAYS):
        q=np.sqrt(Q[a]/np.median(Q[a,S[a]]));q[~S[a]]=0
        def residual(z):
            beta=np.linalg.lstsq(B[O[a]],z[O[a]],rcond=None)[0]
            return np.where(S[a],q*(z-b.mm(B,beta)),0)
        Y=residual(observed[a]);N=residual(paired[a]);R=residual(reference[a]);T=Y-N
        masks=[D[a]&(rad>lo)&(rad<=hi) for lo,hi in [(-1,15),(15,35),(35,60)]]
        footprint=np.flatnonzero(np.logical_or.reduce(masks));norms=np.array([np.linalg.norm(T[m]) for m in masks])
        if not np.all(norms>0):raise ValueError('empty processed template component')
        filters=np.array([np.where(m,T/n,0) for m,n in zip(masks,norms)])
        f=b.mm(filters,Y);fn=b.mm(filters,N);targetQ=np.median(Q[a,footprint]);vectors=[];positions=[];ratios=[]
        for dy in range(-80,81,20):
            for dx in range(-80,81,20):
                shifted=np.array([lookup.get((int(x[i]+dx),int(y[i]+dy)),-1) for i in footprint])
                if np.any(shifted<0) or not np.all(S[a,shifted]):continue
                ratio=np.median(Q[a,shifted])/targetQ
                if not .5<=ratio<=2:continue
                vectors.append(b.mm(filters[:,footprint],R[shifted]));positions.append((dx,dy));ratios.append(ratio)
        if len(vectors)<16:raise ValueError('too few complete coverage-compatible placements')
        features=np.array(vectors);C=np.cov(features,rowvar=False,ddof=1);Cs=.9*C+.1*np.diag(np.diag(C));cond=float(np.linalg.cond(Cs))
        if not np.isfinite(Cs).all() or not np.all(np.linalg.eigvalsh(Cs)>0) or cond>1e8:raise ValueError('unusable projection covariance')
        v=np.linalg.solve(Cs,norms);w=v/np.dot(norms,v);sigma=float(np.sqrt(np.dot(w,b.mm(Cs,w))));mean=features.mean(axis=0)
        amp=float(np.dot(w,f-mean));nullamp=float(np.dot(w,fn-mean));paired_amp=amp-nullamp
        assert np.isfinite(sigma) and sigma>0 and np.isclose(paired_amp,1,rtol=1e-10,atol=1e-10)
        scores=b.mm(features-mean,w)/sigma
        rows.append(dict(array=array,source_score=amp/sigma,paired_null_score=nullamp/sigma,processed_response_score=paired_amp/sigma,
            amplitude=amp,paired_null_amplitude=nullamp,empirical_scale=sigma,expected=norms,covariance=C,shrunk_covariance=Cs,
            weights=w,condition_number=cond,placements=len(vectors),reference_score_range=[scores.min(),scores.max()],five_scale_gate=bool(amp/sigma>5)))
        saved[array+'_filters']=filters;saved[array+'_features']=features;saved[array+'_positions']=positions;saved[array+'_coverage_ratios']=ratios
    np.savez_compressed(out/'ORACLE_FEATURES.npz',**saved)
    b.write(out/'ORACLE.json',dict(rows=rows,all_arrays_above_five=all(r['five_scale_gate'] for r in rows),
        meaning='descriptive realized processed-template diagnostic; not calibrated significance; never feedback evidence'))
    return rows

def metrics(g,u,t,a,phase,compact):
    D=g.D[a];r=np.hypot(g.x-phase[0],g.ygrid-phase[1]);ap=D&(r<=60);wing=D&(r>15)&(r<=60)
    def center(z):
        total=z[ap].sum()
        return np.array([np.dot(z[ap],g.x[ap]),np.dot(z[ap],g.ygrid[ap])])/total if total>0 else None
    c,tc=center(u),center(t)
    result=dict(integrated_brightness=4*u[D].sum(),truth_integrated_brightness=4*t[D].sum(),
        brightness_error=u[D].sum()/t[D].sum()-1,relative_L2=np.linalg.norm((u-t)[ap])/np.linalg.norm(t[ap]),
        centroid=c,truth_centroid=tc,centroid_error=None if c is None else np.linalg.norm(c-tc),
        wing_error=u[wing].sum()/t[wing].sum()-1,exterior_D_brightness=4*u[D&~ap].sum(),
        truth_exterior_D_brightness=4*t[D&~ap].sum(),negative_pixels=int(np.sum(u<0)),outside_D_nonzero=int(np.count_nonzero(u[~D])))
    if compact:
        try:
            f=b.fit_source(u,g.x,g.ygrid,D);tf=b.fit_source(t,g.x,g.ygrid,D)
            result.update(amplitude_bias=f['peak']/tf['peak']-1,width_bias=f['widths']/tf['widths']-1,
                gaussian_centroid_error=np.linalg.norm(f['centroid']-tf['centroid']),fit_boundary=f['boundary_rejection'],fit=f)
        except (ValueError,FloatingPointError) as e:result['fit_unavailable']=str(e)
    return result

def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    freeze=read(H/'FREEZE.json')
    def checkfreeze():
        for row in freeze['files']:assert b.digest(row['path'])==row['sha256'],row['path']
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
        if time.monotonic()-start>1800 or rss>8*2**30:raise RuntimeError('execution time/RSS bound')
        size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        if size>2*2**30:raise RuntimeError('output size bound')
        return dict(seconds=time.monotonic()-start,peak_rss_bytes=rss,output_bytes=size)
    def timeout(*_):raise RuntimeError('30 minute execution limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    checkfreeze();b.write(OUT/'START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_sha256=b.digest(H/'FREEZE.json'),pid=os.getpid(),python=platform.python_version(),numpy=np.__version__))
    try:
        with threadpool_limits(limits=4):
            b.write(OUT/'THREAD_POOLS.json',threadpool_info())
            archived=np.load(OLD/'geometry.npz')
            g=SimpleNamespace(x=archived['x'],ygrid=archived['y'],shape=tuple(archived['shape']),D=archived['D'],O=archived['O'],S=archived['S'],Q=archived['Q'])
            g.npix=len(g.x);estimators=[starlet.Estimator(g.x,g.ygrid,g.shape,g.S[a],g.D[a],g.O[a],g.Q[a]) for a in range(3)]
            first={c['case']:np.load(OLD/c['case']/'R/pass00_maps.npz')['total'] for c in read(OLD/'CASES.json')}
            t0=time.monotonic();cal=[e.calibrate(first['null20260911'][a]) for a,e in enumerate(estimators)]
            b.write(OUT/'CALIBRATION.json',dict(rows=cal,seconds=time.monotonic()-t0,source='null20260911 map 1 only'))
            print('calibration',[(b.ARRAYS[a],r['available'],r['failures']) for a,r in enumerate(cal)],flush=True)
            # Freeze analytic identity above; retain its actual sampled values
            # and digest BEFORE the single injected-parent learning operation.
            image=truths.optical_image();t4=truths.map_truth(g,image,brightness=4.)
            np.savez_compressed(OUT/'coma4_truth.npz',truth=t4)
            b.write(OUT/'TRUTH_BINDING.json',dict(config=truths.PSF_CONFIG,brightness=4.,seed=20260911,truth_sha256=b.digest(OUT/'coma4_truth.npz')))
            t0=time.monotonic();data=b.Data(read(H.parent/'fruit_point_coherent_feedback_2026-09-11/INPUT_PREFLIGHT.json'))
            for key,actual in [('x',data.x),('y',data.ygrid),('D',data.D),('O',data.O),('S',data.S),('Q',data.Q),('valid',data.valid)]:np.testing.assert_array_equal(archived[key],actual)
            scales=archived['scales'];parent=b.nuisance(data,scales,20260911)+data.project(t4)
            parent_digest=hashlib.sha256(np.ascontiguousarray(parent).tobytes()).hexdigest()
            b.write(OUT/'BOOTSTRAP_STARTED.json',dict(parent_sha256_float64_C=parent_digest,parent_shape=parent.shape,seed=20260911,truth_sha256=b.digest(OUT/'coma4_truth.npz'),setup_seconds=time.monotonic()-t0,planned_cleaning_passes=1))
            t0=time.monotonic();tod,state,rank_guard=data.clean(parent,np.zeros((3,g.npix)));clean_seconds=time.monotonic()-t0
            t0=time.monotonic();newmap=data.grid(tod);map_seconds=time.monotonic()-t0
            np.savez_compressed(OUT/'coma4_bootstrap_maps.npz',total=newmap,applied_model=np.zeros_like(newmap))
            np.savez_compressed(OUT/'coma4_bootstrap_state.npz',**state)
            b.write(OUT/'BOOTSTRAP_COMPLETE.json',dict(cleaning_passes=1,clean_seconds=clean_seconds,map_seconds=map_seconds,rank_guard=rank_guard,groups=len(data.groups),**guard()))
            del parent,tod,state,data
            diag=oracle(g,newmap,first['null20260911'],first['null20260912'],OUT)
            print('coma4 oracle',[r['source_score'] for r in diag],flush=True)
            if not all(r['five_scale_gate'] for r in diag):
                b.write(OUT/'STOP.json',dict(reason='brighter_coma_regime_unavailable',**guard()));return
            rows=[]
            def case(name,maps,truth=None,phase=(17,-11),compact=False,kind='saved_processed'):
                guard();models=[];rec=[];omegas=[];trials=[]
                for a,e in enumerate(estimators):
                    u,d,omega=e.infer(maps[a]);models.append(u);omegas.append(np.zeros((5,g.npix),bool) if omega is None else omega)
                    trials.append(e.last_trial)
                    if truth is not None:d['model_metrics']=metrics(g,u,truth[a],a,phase,compact)
                    d.update(array=b.ARRAYS[a],case=name,kind=kind,phase=list(phase));rec.append(d)
                np.savez_compressed(OUT/(name+'.npz'),input_total=maps,model=np.array(models),solver_trial=np.array(trials),selected=np.array(omegas))
                b.write(OUT/(name+'.json'),dict(rows=rec,array_time_median=np.median([r['seconds'] for r in rec]),array_time_max=max(r['seconds'] for r in rec),serial_inference_seconds=sum(r['seconds'] for r in rec)))
                rows.extend(rec);print(name,[(r['reason'],round(r['seconds'],3)) for r in rec],flush=True)
            for c in read(OLD/'CASES.json'):
                name=c['case'];tp=OLD/(name+'_truth.npz');truth=np.load(tp)['truth'] if tp.exists() and not name.startswith('background') else None
                case('saved_'+name,first[name],truth=truth,compact=name.startswith('gauss'))
            case('saved_coma4',newmap,t4)
            for phase in [(17,-11),(19,-10)]:
                p=np.array([100.,*phase,np.log(12.),np.log(8.),np.deg2rad(25),0,0,0]);compact_truth=np.where(g.D,b.gaussian(p,g.x,g.ygrid)[None,:],0)
                prefix='noiseless_phase'+('0' if phase==(17,-11) else '1')
                case(prefix+'_compact',compact_truth,compact_truth,phase,True,'noiseless')
                for brightness in [1.,.5,4.]:
                    truth=truths.map_truth(g,image,brightness=brightness,translation=phase)
                    case(prefix+'_coma'+str(brightness).replace('.','p'),truth,truth,phase,False,'noiseless')
            for name,truth,seed,compact in [('compact_seed1',b.truth_map(g),20260911,True),('compact_seed2',b.truth_map(g),20260912,True),
                ('mismatch',b.truth_map(g,True),20260911,False)]+[(f'coma{v}',truths.map_truth(g,image,brightness=v),20260911,False) for v in [1,.5,4]]:
                case('map_add_'+name.replace('.','p'),truth+first[f'null{seed}'],truth,compact=compact,kind='estimator_only_map_addition')
            plane=7+2*g.x/90-3*g.ygrid/90
            case('pure_plane',np.where(g.S,plane[None,:],np.nan),kind='pure_plane')
            b.write(OUT/'RESULTS.json',dict(rows=rows,scope='estimator tests only; no full FRUIT trajectories',calibration=cal,oracle=diag))
            checkfreeze();b.write(OUT/'COMPLETE.json',dict(cleaning_passes=1,estimator_array_cases=len(rows),**guard()))
    except BaseException as e:
        b.write(OUT/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),elapsed=time.monotonic()-start));raise
    finally:
        signal.alarm(0)
        b.write(OUT/'PRODUCT_MANIFEST.json',dict(files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=b.digest(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='PRODUCT_MANIFEST.json']))
if __name__=='__main__':main()
