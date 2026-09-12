"""One frozen spatial audit and matched saved-map test. No PTC/input reads."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,datetime,resource,platform,signal,traceback,os
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
import candidate
H=Path(__file__).resolve().parent;PREV=H.parent/'fruit_point_starlet_preliminary_2026-09-11'
sys.path.insert(0,str(PREV));import run_screen as previous
b=previous.b;truths=previous.truths
OLD=previous.OUT;RBF=previous.OLD;OUT=Path('/private/tmp/sci-fruit-starlet-central-domain-20260912-r0.1')
read=lambda p:json.loads(Path(p).read_text())
RADIUS=60.

def load_geometry():
    a=np.load(RBF/'geometry.npz')
    g=SimpleNamespace(x=a['x'],ygrid=a['y'],shape=tuple(a['shape']),D=a['D'],O=a['O'],S=a['S'],Q=a['Q'])
    g.npix=len(g.x);return g

def make_estimator(g,a,arm,null):
    e=candidate.Estimator(g.x,g.ygrid,g.shape,g.S[a],g.D[a],g.O[a],g.Q[a]);cal=e.calibrate(null[a])
    if not cal['available']:raise ValueError('unchanged calibration unavailable')
    if arm=='C':e.D=e.D&(np.hypot(g.x,g.ygrid)<=RADIUS)
    return e,cal

def spatial_audit(g,estimators):
    radius=np.hypot(g.x,g.ygrid);rows=[];locations=[];models=[]
    for name in ['saved_null20260911','saved_null20260912','saved_background20260911']:
        p=np.load(OLD/(name+'.npz'))
        for a,e in enumerate(estimators):
            Y,_=e.residual(p['input_total'][a]);W=candidate.analysis(Y.reshape(g.shape)).reshape(5,-1);sigma=e.sigmas[:,e.stratum]
            selected=p['selected'][a];eligible=e.valid&g.D[a]
            np.testing.assert_array_equal(selected,eligible&(abs(W)>=5*sigma))
            for j in range(5):
                for lo,hi in [(-1,30),(30,60),(60,90)]:
                    for q in range(2):
                        mask=eligible[j]&(radius>lo)&(radius<=hi)&(e.stratum==q);n=int(mask.sum());k=int(np.sum(selected[j]&mask));z=abs(W[j,mask]/sigma[j,mask])
                        rows.append(dict(case=name,array=b.ARRAYS[a],band=j+1,inner_arcsec=max(lo,0),outer_arcsec=hi,stratum=q,eligible=n,selected=k,
                            rate=k/n if n else None,abs_standardized_quantiles=np.quantile(z,[.5,.9,.95,.99,.999,1]) if n else None,
                            standardized_spatial_MAD=candidate.mad(W[j,mask]/sigma[j,mask]) if n else None,assigned_sigma=e.sigmas[j,q],
                            Q_quantiles=np.quantile(g.Q[a,mask],[0,.5,1]) if n else None))
                for i in np.flatnonzero(selected[j]):locations.append(dict(case=name,array=b.ARRAYS[a],band=j+1,x=g.x[i],y=g.ygrid[i],radius=radius[i],Q=g.Q[a,i],Q_over_median_D=g.Q[a,i]/e.q_split,stratum=int(e.stratum[i]),
                    coefficient=W[j,i],assigned_sigma=sigma[j,i],standardized=W[j,i]/sigma[j,i],inside_central=bool(radius[i]<=RADIUS)))
            u=p['model'][a];central=g.D[a]&(radius<=RADIUS);outer=g.D[a]&~central;total=float(4*u.sum())
            def peak(mask):
                ix=np.flatnonzero(mask);i=ix[np.argmax(u[ix])]
                return dict(value=u[i],x=g.x[i],y=g.ygrid[i],radius=radius[i],Q=g.Q[a,i])
            models.append(dict(case=name,array=b.ARRAYS[a],total_brightness=total,central_brightness=float(4*u[central].sum()),outer_brightness=float(4*u[outer].sum()),
                central_fraction=float(4*u[central].sum()/total) if total>0 else None,central_peak=peak(central),outer_peak=peak(outer)))
    result=dict(rings=rows,selected_locations=locations,false_models=models,central_radius=RADIUS,noise_statement='Assigned band/Q-stratum scales; spatial descriptive diagnostics, not calibrated local coefficient uncertainties')
    b.write(OUT/'SPATIAL_AUDIT.json',result)
    print('spatial audit selected inside/outside',sum(r['inside_central'] for r in locations),sum(not r['inside_central'] for r in locations),flush=True)
    return result

def clipping(g,truth,a,phase):
    radius=np.hypot(g.x,g.ygrid);domain=g.D[a];central=domain&(radius<=RADIUS);r=np.hypot(g.x-phase[0],g.ygrid-phase[1]);ap=domain&(r<=60);w=domain&(r>15)&(r<=60)
    outside=domain&~central;t=truth[a]
    return dict(brightness_fraction_outside_C=float(t[outside].sum()/t[domain].sum()),wing_fraction_outside_C=float(t[w&outside].sum()/t[w].sum()),
        minimum_relative_L2_from_excluded_truth=float(np.linalg.norm(t[ap&outside])/np.linalg.norm(t[ap])),truth_center=phase)

def cases(g):
    oldrows=read(PREV/'NUMERICAL_RESULTS.json')['rows'];names=sorted({r['case'] for r in oldrows})
    for name in names:
        rec=next(r for r in oldrows if r['case']==name);p=np.load(OLD/(name+'.npz'));z=p['input_total'];truth=None
        if name.startswith('noiseless'):truth=z
        elif name in ['saved_coma4','map_add_coma4']:truth=np.load(OLD/'coma4_truth.npz')['truth']
        elif name.startswith('saved_') and name!='saved_background20260911':
            tp=RBF/(name.removeprefix('saved_')+'_truth.npz')
            if tp.exists():truth=np.load(tp)['truth']
        elif name.startswith('map_add_'):
            truthname={'map_add_compact_seed1':'gauss20260911','map_add_compact_seed2':'gauss20260912','map_add_mismatch':'mismatch20260911','map_add_coma1':'coma_bright','map_add_coma0p5':'coma_half'}[name]
            truth=np.load(RBF/(truthname+'_truth.npz'))['truth']
        compact=('compact' in name or 'gauss' in name);coma4=('coma4' in name)
        yield dict(name=name,maps=z,truth=truth,phase=tuple(rec['phase']),kind=rec['kind'],compact=compact,coma4=coma4,original=True)
    phase=(35.,-11.);p=np.array([100.,*phase,np.log(12.),np.log(8.),np.deg2rad(25),0,0,0]);compact=np.where(g.D,b.gaussian(p,g.x,g.ygrid)[None,:],0.)
    coma=truths.map_truth(g,truths.optical_image(),4.,phase);null=np.load(OLD/'saved_null20260911.npz')['input_total']
    for label,t in [('compact',compact),('coma4',coma)]:
        for kind,z in [('noiseless',t),('map_add',t+null)]:
            yield dict(name='offset_'+kind+'_'+label,maps=z,truth=t,phase=phase,kind='noiseless' if kind=='noiseless' else 'estimator_only_map_addition',compact=label=='compact',coma4=label=='coma4',original=False)

def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic();freeze=read(H/'FREEZE.json')
    def verify_freeze():
        for r in freeze['files']:assert b.digest(r['path'])==r['sha256'],r['path']
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024);seconds=time.monotonic()-start
        size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        if seconds>600 or rss>4*2**30 or size>2**30:raise RuntimeError('screen resource bound')
        return dict(seconds=seconds,peak_rss_bytes=rss,output_bytes=size)
    def timeout(*_):raise RuntimeError('ten minute bound')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    verify_freeze();b.write(OUT/'START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_sha256=b.digest(H/'FREEZE.json'),pid=os.getpid(),new_cleaning_passes=0))
    try:
        with threadpool_limits(limits=4):
            b.write(OUT/'THREAD_POOLS.json',dict(pools=threadpool_info(),requested_environment={k:os.getenv(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}))
            g=load_geometry();null=np.load(OLD/'saved_null20260911.npz')['input_total'];estimators={};cal=[]
            for arm in ['F','C']:
                estimators[arm]=[]
                for a in range(3):
                    e,record=make_estimator(g,a,arm,null);estimators[arm].append(e)
                    if arm=='F':cal.append(record)
            assert b.native(cal)==read(PREV/'NUMERICAL_RESULTS.json')['calibration']
            for a in range(3):
                for field in ['S','O','valid','stratum','sigmas']:
                    np.testing.assert_array_equal(getattr(estimators['F'][a],field),getattr(estimators['C'][a],field))
            b.write(OUT/'SETUP.json',dict(calibration=cal,setup_seconds=time.monotonic()-start,central_radius=RADIUS,full_D_pixels=g.D.sum(axis=1),central_D_pixels=[e.D.sum() for e in estimators['C']],solver_normalization_fallback=[e.normalization_fallback for e in estimators['F']]))
            audit=spatial_audit(g,estimators['F']);rows=[];clipped=[];matched=[]
            # Case list/truths are generated once and stored before inference.
            allcases=list(cases(g));assert len(allcases)==29
            for c in allcases:
                if c['truth'] is not None:
                    np.savez_compressed(OUT/(c['name']+'_truth.npz'),truth=c['truth'])
                    clipped.extend([dict(case=c['name'],array=b.ARRAYS[a],**clipping(g,c['truth'],a,c['phase'])) for a in range(3)])
            b.write(OUT/'TRUTH_CLIPPING.json',clipped)
            for c in allcases:
                guard();name=c['name'];z=c['maps']
                for arm in ['F','C']:
                    models=[];trials=[];selected=[];records=[]
                    for a,e in enumerate(estimators[arm]):
                        u,rec,omega=e.infer(z[a]);models.append(u);trials.append(e.last_trial);selected.append(omega)
                        rec.update(case=name,array=b.ARRAYS[a],arm=arm,kind=c['kind'],phase=c['phase'],compact=c['compact'],coma4=c['coma4'],original=c['original'])
                        if c['truth'] is not None:rec['model_metrics']=previous.metrics(g,u,c['truth'][a],a,c['phase'],c['compact'])
                        central=g.D[a]&(np.hypot(g.x,g.ygrid)<=RADIUS)
                        rec.update(central_model_brightness=4*u[central].sum(),outer_model_brightness=4*u[g.D[a]&~central].sum(),model_peak=float(u.max()))
                        if arm=='F' and c['original'] and rec['outer_MAD']>0:
                            saved=np.load(OLD/(name+'.npz'))
                            exact=all(np.array_equal(now,saved[key][a]) for now,key in [(u,'model'),(e.last_trial,'solver_trial'),(omega,'selected')])
                            matched.append(dict(case=name,array=b.ARRAYS[a],bitwise_identical=exact))
                            if not exact:raise AssertionError('nonzero-MAD full-domain control changed')
                        records.append(rec);rows.append(rec)
                    np.savez_compressed(OUT/f'{name}_{arm}.npz',input_total=z,model=np.array(models),solver_trial=np.array(trials),selected=np.array(selected))
                    b.write(OUT/f'{name}_{arm}.json',dict(rows=records,median_array_seconds=np.median([r['seconds'] for r in records]),max_array_seconds=max(r['seconds'] for r in records),serial_seconds=sum(r['seconds'] for r in records)))
                    print(name,arm,[(r['reason'],round(r['seconds'],3)) for r in records],flush=True)
            assert len(rows)==174;verify_freeze()
            b.write(OUT/'RESULTS.json',dict(rows=rows,matched_controls=matched,calibration=cal))
            b.write(OUT/'COMPLETE.json',dict(cases=29,estimator_calls=len(rows),new_cleaning_passes=0,full_trajectories=0,**guard()))
    except BaseException as e:
        b.write(OUT/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),**guard()));raise
    finally:
        signal.alarm(0)
        b.write(OUT/'PRODUCT_MANIFEST.json',dict(files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=b.digest(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='PRODUCT_MANIFEST.json']))
if __name__=='__main__':main()
