"""Reproducible frozen preparation and paired null calibration/validation stages."""
import argparse,json,os,platform,resource,shutil,subprocess,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import numpy as np
import psutil
from point_frozen_shoulder import ancestors,CD,ARRAYS,saved_context
from point_fruit_trajectory import bound,verified,manifest,verify_manifest,FixedMeasurements
from point_fruit_diagnostics import patch_map,load
from point_additive import draw_noise
from point_additive_diagnostics import save_json
from point_empirical import patch,coordinates
from point_admission import admit
from inspect_point_source import gaussian
from inspect_forward_map import require,digest
from run_point_response_cd import source,AB
from point_corr_admission import Neighborhood,infer,select,maximum,trial_seed,threshold,binomial_upper,POLICIES
REPO=Path(__file__).resolve().parents[2]
V=REPO/'validation/point_152389_correlation_admission_2026-09-24'
SNAPSHOTS=(0,1,3,5,10)


def read(p):return json.loads(Path(p).read_text())

def fixed_from(ctx):
 f=FixedMeasurements.__new__(FixedMeasurements)
 for name,key in [('header','header'),('parameters','p'),('x','x'),('y','y'),('radius','radius'),('major','major'),('weights','q'),('scale','scale'),('noise_valid','noise_valid')]:setattr(f,name,ctx[key])
 f.beam=ctx['beam'];f.disk=np.hypot(f.x,f.y)<=40;f.background=f.disk&(f.radius>3*f.major);f.edges=np.arange(0,42,2)
 return f


def prepare(out,binary):
 start=time.perf_counter();revision=source();require(not out.exists(),'fresh destination required')
 version=subprocess.check_output([str(binary),'--version'],text=True);require(revision[:9] in version,'build/source mismatch')
 auth,owner=ancestors();jc=read(REPO/'validation/point_152389_map_only_jinc_full_2026-09-24/COMPLETION.json');jp=Path(jc['product_root']);auth.manifest(jp,jc['root_manifest']['sha256'])
 for n in ('ASSESSMENT.md','FINAL_REVIEW.md','COMPLETION.json'):auth.check(jp/n)
 out.mkdir();dest=out/'freeze';dest.mkdir();shutil.copy2(V/'OWNER_DIRECTIVE.md',out/'OWNER_DIRECTIVE.md')
 methods=dict(inherited='BASE threshold ratio>=3, own fitted constant and nearest fitted-centroid anchor, 8-connected; constrained fit retained numerically but no candidate; source association absence has no supplement',new='CORR-NEIGHBOR-v1/DIAG, center/K0/current B exclusion, positive b, strict score threshold, full circular footprint, neff_weights>=3, noniterative connection then exact original b',C0='reused original A+B REF M0 model, original four-FWHM union source/bowl exclusion and settings; distinct from additive-r2 BASE admission scale',covariance='ifft2(saved PSD), original padded generator and amplitude continuation flags, no new fitting or PSD repair',limits='no source-flux ensemble; thresholds frozen before validation/real scores; six6144 fields total; first256validation reused1.25; gate fail cannot retune')
 record=dict(identity='POINT-152389-CORR-NEIGHBOR-PROSPECTIVE-2026-09-24',source_revision=revision,binary=bound(binary),version=version,directive=bound(out/'OWNER_DIRECTIVE.md'),methods=methods,original_owner=owner,arrays={},seeds={a:{ph:[str(trial_seed(a,ph,i)) for i in range(1024)] for ph in ('calibration','validation')} for a in ARRAYS},generator='point_additive.draw_noise; numpy.default_rng(seed integer)/PCG64, real normal field, FFT multiply sqrt(saved PSD), centered crop, amplitude multiplication',calibration_fields=1024,validation_fields=1024,stress_fields=256,stress_multiplier=1.25,order_statistic_one_based=1015,threshold_floor=3.,launch_upper95_limit=.02,workers=min(12,max(1,os.cpu_count()-2)),threads_per_worker=1,host=dict(platform=platform.platform(),cpus=os.cpu_count(),memory_bytes=psutil.virtual_memory().total),trajectory_matrix=dict(policies=['CORR','DIAG'],branches=['REF','C-core-plus','C-core-minus','C-shoulder','D-seed-shoulder'],cycles=10,BASE_REF_replays=3,maximum_feedback_applications=330),tolerances=owner['reference_tolerances'])
 for a in ARRAYS:
  d=dest/a;d.mkdir();ctx=saved_context(auth,owner,a);ctx['beam']={'a1100':5.,'a1400':7.5,'a2000':10.}[a];f=fixed_from(ctx)
  c0=AB/'companion'/a/'M0';model_path=auth.check(c0/'model.npz');meta=auth.json(c0/'model.json')
  with np.load(model_path) as z:model={k:z[k].copy() for k in z.files}
  for n in ('frozen-domains.npz','frozen-domains.json'):auth.check(AB/'companion'/a/n)
  base,ch=auth.fits(owner['original_cycles'][a][0]['candidate']['path']);k0=base['ADMITTED'].astype(bool)
  original,sv,_,h=patch_map(auth.check(owner['original_cycles'][a][0]['total']['path']))
  require(original.shape==model['source_scale'].shape==k0.shape,'C0 source shape')
  require(np.array_equal(model['source_support'],sv),'C0 original M0 support mismatch')
  p0=np.asarray(owner['original_cycles'][a][0]['measurement']['fit']['parameters']);blank=np.full(original.shape,p0[6]);unit=p0.copy();unit[6]=0;core=blank+np.where(k0,gaussian(unit,ctx['x'],ctx['y']),0.)
  np.savez(d/'inputs.npz',original=original,support=sv,psd=model['psd'],scale=model['source_scale'],empirical=model['empirical'],pairs=model['pairs'],k0=k0,blank=blank,core=core,parameters=ctx['p'],x=ctx['x'],y=ctx['y'],radius=ctx['radius'],q=ctx['q'],base_scale=ctx['scale'],noise_valid=ctx['noise_valid'])
  headers={k:v for k,v in h.items() if k not in ('COMMENT','HISTORY','')};save_json(d/'geometry.json',dict(header=headers,beam=f.beam,major=f.major,model=meta))
  replay=[]
  # Original candidates are replayed using their exact archived required fit;
  # no new source fit and no supplemental real scores are opened here.
  for branch in ('REF','C-core-plus','C-core-minus','C-shoulder','D-seed-shoulder'):
   summary=auth.json(CD/'run'/branch/a/'SUMMARY.json')
   for state in summary['histories']:
    total=auth.check(state['total']['path']);signal,support,_,_=patch_map(total)
    candidate=auth.check(state['candidate']['path']);old,_=load(candidate);fit=state['measurement']['fit']
    if branch=='D-seed-shoulder' and state['cycle_index']==0:
     replay.append(dict(branch=branch,stage=0,status='authenticated first-candidate-only bypass'));continue
    new=admit(signal,support,f.scale,f.noise_valid,f.header,np.asarray(fit['parameters']),3.,1.)
    for field,key in [('CANDIDATE','candidate'),('ADMITTED','selected'),('VALID','valid'),('RATIO','z')]:require(np.array_equal(old[field],new[key],equal_nan=True),'original candidate replay '+branch+field)
    replay.append(dict(branch=branch,stage=state['cycle_index'],status='exact candidate replay',total=auth.ref(total),candidate=auth.ref(candidate)))
   # Authenticate full used historical trajectory (all manifested files).
   prefix=str((CD/'run'/branch/a).resolve())+os.sep
   for name in list(auth.expected):
    if name.startswith(prefix):auth.check(name)
  record['arrays'][a]=dict(inputs=bound(d/'inputs.npz'),geometry=bound(d/'geometry.json'),C0=auth.ref(model_path),C0_metadata=auth.ref(c0/'model.json'),K0=auth.ref(owner['original_cycles'][a][0]['candidate']['path']),K0_pixels=int(k0.sum()),candidate_replay=replay,covariance_reconstructed=False)
 record['consumed_inputs']=auth.consumed;record['authentication_seconds']=auth.seconds;record['final_authentication_seconds']=auth.verify_all();record['wall_seconds']=time.perf_counter()-start;record['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 save_json(dest/'prospective.json',record);manifest(dest);save_json(out/'PREPARED.json',dict(prospective=bound(dest/'prospective.json'),manifest=bound(dest/'FILES_SHA256SUMS')))


def context(out,a):
 p=read(out/'freeze/prospective.json');r=p['arrays'][a];g=read(verified(r['geometry']))
 with np.load(verified(r['inputs'])) as z:c={k:z[k].copy() for k in z.files}
 c.update(g);f=fixed_from(dict(header=g['header'],p=c['parameters'],x=c['x'],y=c['y'],radius=c['radius'],major=g['major'],q=c['q'],scale=c['base_scale'],noise_valid=c['noise_valid'],beam=g['beam']))
 c['fixed']=f;c['neighborhood']=Neighborhood(f.header,c['original'].shape,f.beam,c['scale'],c['psd'],c['k0'],empirical=c['empirical']);return c

_GLOBAL=None

def init_worker(out,a):
 global _GLOBAL
 _GLOBAL=context(Path(out),a)


def scene(c,data,tau,source_mask=None):
 fit,base,fields=infer(data,c['support'],c['fixed'],c['neighborhood'])
 valid=fit['status']=='success' and np.isfinite(fit['parameters']).all()
 rec=dict(fit=fit,numerically_valid=bool(valid),candidate_status='fit_unavailable' if not valid else 'finite_constrained_no_candidate' if fit.get('constrained') else base['summary']['status'],BASE_pixels=int(base['selected'].sum()) if base is not None else 0,BASE_brightness_area=float(np.sum(base['candidate'][base['selected']])*4) if base is not None else 0,policies={})
 false=(base['selected']&~source_mask) if base is not None and source_mask is not None else (base['selected'] if base is not None else np.zeros(c['support'].shape,bool))
 rec['BASE_false_pixels']=int(false.sum());rec['BASE_false_brightness_area']=float(np.sum(base['brightness'][false])*4) if base is not None else 0
 for pol in POLICIES:
  r=dict(max_score=maximum(base,fields,pol))
  if tau is not None:
   selected=select(base,fields,c['k0'],pol,tau[pol]);r['summary']=selected['summary'] if selected is not None else None
   r['added_pixels']=int(selected['added'].sum()) if selected is not None else 0
   r['extra_brightness_area']=float(np.sum(selected['candidate'][selected['added']])*4) if selected is not None else 0
   r['total_pixels']=int(selected['selected'].sum()) if selected is not None else 0
  rec['policies'][pol]=r
 return rec


def trial(payload):
 a,phase,index,tau=payload;c=_GLOBAL;start=time.perf_counter();cpu=time.process_time();seed=trial_seed(a,phase,index);noise=draw_noise(c['psd'],c['scale'],seed)
 rec=dict(array=a,phase=phase,trial=index,seed=str(seed),scenes={})
 for name in ('blank','core'):rec['scenes'][name]=scene(c,np.where(c['support'],c[name]+noise,np.nan),tau,c['k0'] if name=='core' else None)
 rec['maxima']={pol:max([r['policies'][pol]['max_score'] for r in rec['scenes'].values() if r['policies'][pol]['max_score'] is not None],default=None) for pol in POLICIES}
 if phase=='validation' and index<256:rec['stress']={name:scene(c,np.where(c['support'],c[name]+1.25*noise,np.nan),tau,c['k0'] if name=='core' else None) for name in ('blank','core')}
 rec.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);return rec


def simulate(out,phase,review):
 revision=source();p=read(out/'freeze/prospective.json');verify_manifest(out/'freeze');require(review.is_file(),'independent exact-source preflight review required')
 dest=out/phase;require(not dest.exists(),'no replacement trials');dest.mkdir();start=time.perf_counter();allrows={};taus=None
 if phase=='validation':taus=read(verified(read(out/'CALIBRATION.json')['thresholds']))['thresholds']
 for a in ARRAYS:
  tau=None if taus is None else taus[a];rows=[]
  with ProcessPoolExecutor(max_workers=p['workers'],initializer=init_worker,initargs=(str(out),a)) as pool:
   for row in pool.map(trial,[(a,phase,i,tau) for i in range(1024)],chunksize=8):rows.append(row)
  save_json(dest/(a+'-trials.json'),rows);allrows[a]=rows
  print(json.dumps(dict(phase=phase,array=a,trials=len(rows),failed=sum(not s['numerically_valid'] for r in rows for s in r['scenes'].values()))),flush=True)
 result=dict(source_revision=revision,prospective=bound(out/'freeze/prospective.json'),review=bound(review),wall_seconds=time.perf_counter()-start,independent_fields=3072,workers=p['workers'],numeric_threads=1,CPU_seconds=sum(r['cpu_seconds'] for rows in allrows.values() for r in rows),maximum_worker_highwater_bytes=max(r['peak_rss_bytes'] for rows in allrows.values() for r in rows))
 if phase=='calibration':
  ts={a:{pol:threshold([r['maxima'][pol] for r in rows]) for pol in POLICIES} for a,rows in allrows.items()}
  save_json(dest/'thresholds.json',dict(thresholds=ts,order_statistic=1015,strict_exceedance=True,frozen_before_validation=True));result['thresholds']=bound(dest/'thresholds.json')
 else:
  gates={}
  for a,rows in allrows.items():
   c=context(out,a);controls={name:scene(c,np.where(c['support'],c[name],np.nan),taus[a],c['k0'] if name=='core' else None) for name in ('blank','core')};gates[a]={}
   calibration=read(out/'calibration'/(a+'-trials.json'))
   complete=all(s['numerically_valid'] for rr in (rows,calibration) for r in rr for s in r['scenes'].values())
   for pol in POLICIES:
    k=sum(r['maxima'][pol] is not None and r['maxima'][pol]>taus[a][pol] for r in rows);u=binomial_upper(k);false=sum(any(s['policies'][pol]['added_pixels']>0 for s in r['scenes'].values()) for r in rows)
    clean=all(s['numerically_valid'] and s['policies'][pol]['added_pixels']==0 for s in controls.values())
    gates[a][pol]=dict(pass_gate=bool(complete and u<=.02 and clean),**{'pass':bool(complete and u<=.02 and clean)},threshold=taus[a][pol],preconnectivity_exceedances=k,upper95=u,final_false_admission_events=false,complete_numerical_outcomes=complete,noiseless_controls=controls,qualifications='conditional frozen M0 covariance/scene model; retained observational limitations; not a ten-cycle false-feedback guarantee')
  save_json(dest/'gates.json',dict(policies=gates));result['gates']=bound(dest/'gates.json')
 manifest(dest);save_json(out/(phase.upper()+'.json'),result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','calibration','validation']);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path);p.add_argument('--review',type=Path);a=p.parse_args()
 if a.stage=='prepare':prepare(a.output,a.binary)
 else:simulate(a.output,a.stage,a.review)
