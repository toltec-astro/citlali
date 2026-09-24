"""Finite authorized JINC comparison matrix; certificate-gated publication.

No upstream science call: native entry is only jinc_diagnostic. Frozen saved
rejoined-descendant operands and geometry are the entire observation input.
"""
import argparse,time,subprocess,os,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import psutil
from astropy.io import fits
from point_jinc_full import (read,source,REPO,PROFILE,ARRAYS,MATRIX,key,header_record)
from point_fruit_trajectory import bound,verified,manifest,verify_manifest,FixedMeasurements
from point_fruit_diagnostics import load
from point_additive_diagnostics import save_json
from point_empirical_io import write_fits
from point_empirical import source_patch,coordinates
from point_response_cd import vector_measurement,RATIO_FLOOR
from inspect_forward_map import require,read_yaml,digest

WORKERS=12
SEEDS={'MAP':241523890,'JINC':241523891}

def fixed_for(out,a):
 d=out/'prepared'/a;info=read(d/'fixed.json');f=FixedMeasurements.__new__(FixedMeasurements)
 f.header=fits.Header(info['header']);f.major=info['major'];f.beam=info['beam']
 with np.load(d/'fixed.npz') as z:
  for name in z.files:setattr(f,name,z[name].copy())
 f.disk=np.hypot(f.x,f.y)<=40;f.background=f.disk&(f.radius>3*f.major);f.edges=np.arange(0,42,2)
 return f

def measure_full(data,header,fixed):
 s,v,_,_,_=source_patch(data['PRIMARY'],data['SCI_SUPPORT'].astype(bool),data['HITS'],header)
 m=fixed.measure(s,v);raw=[]
 for factor in (1.,1.5,2.):
  roi=fixed.radius<=factor*fixed.major;good=bool(v[roi].all() and np.isfinite(s[roi]).all())
  raw.append(dict(factor=factor,pixels=int(roi.sum()),raw_brightness_area=float(s[roi].sum()*4) if good else None,units='mJy/nominal beam arcsec^2',available=good))
 m['raw_apertures']=raw
 if m['fit']['status']=='success':
  from point_response_cd import ellipse
  par=m['fit']['parameters'];m['shape']=dict(ellipse_arcsec2=ellipse(par),minor_major_sigma_ratio=min(par[3:5])/max(par[3:5]),angle_interpretation='ellipse tensor is gauge-safe; fitted angle becomes undefined at circularity and weakly determined as axis ratio approaches1; no added threshold or rejection')
 if m['fit']['status']!='success':
  m['frozen_template_projection']=None
  for aperture in m['apertures']:
   aperture['value_mJy_per_beam_arcsec2']=None
   aperture['background_status']='unavailable fitted constant'
   aperture['available']=False
 return m

def ordinary(g,t,h):
 shape=(h['NAXIS2'],h['NAXIS1']);pixel=np.floor(g[:,1]-.5).astype('i8')*shape[1]+np.floor(g[:,0]-.5).astype('i8');n=np.zeros(np.prod(shape));c=np.zeros(n.size);ab=np.zeros(n.size)
 np.add.at(n,pixel,t);np.add.at(c,pixel,1.);np.add.at(ab,pixel,abs(t))
 n=n.reshape(shape);c=c.reshape(shape);ab=ab.reshape(shape);u=2.**-53;gamma=np.nextafter((c+3)*u/(1-(c+3)*u),np.inf)
 e=np.nextafter(gamma*ab/(1-gamma),np.inf);s=c>0;m=np.divide(n,c,out=np.full(shape,np.nan),where=s)
 error=np.divide(e+np.nextafter(u/(1-u)*abs(n),np.inf),c,out=np.full(shape,np.inf),where=s)
 return dict(PRIMARY=m,NUMERATOR=n,NORMALIZATION=c,HITS=c,NORM_SUPPORT=s.astype('u1'),SCI_SUPPORT=s.astype('u1'),ERROR=error)

def map_job(job):
 """Each native process gets one accumulation worker; independent maps parallel."""
 out,prepared,s,role,value,helper=job;a=s['array'];ar=prepared['arrays'][a];d=out/'grids'/s['key']/role;d.mkdir(parents=True)
 q=dict(schema='point-152389-jinc-diagnostic-v1',mode='accumulate',spec=ar['cache']['spec'],profile=PROFILE,profile_binding=bound(out/'freeze/profile.json'),cache_sha256=ar['cache']['cache']['sha256'],coefficient_errors=ar['cache']['coefficient_errors'],samples=s['samples'],workers=1,frozen_peak=ar['peak']['P_mJy_per_nominal_beam'],geometry_payload=s['geometry'],value_payload=value,prospective=bound(out/'freeze/prospective.json'),parent_state=bound(out/'prepared'/a/s['key']/'STATE.json'),lifecycle='provisional sums; independently checked before scientific publication')
 save_json(d/'request.json',q);start=time.perf_counter()
 with (d/'native.log').open('w') as log:subprocess.run([str(helper),str(d/'request.json'),str(d/'native')],check=True,stdout=log,stderr=subprocess.STDOUT)
 r=read_yaml(d/'native/receipt.yaml');require(r['request_sha256']==digest(d/'request.json'),'native request binding')
 for p,sha in r['files'].items():require(digest(d/'native'/p)==sha,'native output changed')
 shape=(ar['header']['NAXIS2'],ar['header']['NAXIS1']);v={n:np.fromfile(d/'native'/(n+'.f64'),dtype='<f8').reshape(shape) for n in ('N','C','Q','T','m','absolute-C','tolerance-lower',*[f'error-{i}' for i in range(5)])};causes=np.fromfile(d/'native/causes.u8',dtype='u1').reshape(shape);counts=np.fromfile(d/'native/members.u64',dtype='<u8').reshape(shape);good=causes==0
 require(np.all(np.isfinite(v['m'][good])) and np.all(v['error-4'][good]<=v['tolerance-lower'][good]),'signal certificate mismatch')
 for n,i in [('Q',2),('T',3)]:
  lower=np.nextafter(v[n]-v[f'error-{i}'],-np.inf);tol=np.nextafter(np.nextafter(1e-6,-np.inf)*lower,-np.inf)
  require(np.all(lower[good]>0) and np.all(v[f'error-{i}'][good]<=tol[good]),'ancillary certificate mismatch')
 require(np.all(abs(v['C'][good])>v['error-1'][good]),'denominator separation mismatch')
 header=fits.Header(ar['header']);write_fits(d/'map.fits',header,[('PRIMARY',np.where(good,v['m'],np.nan),'mJy/nominal beam'),('NUMERATOR',v['N'],'mJy/nominal beam'),('NORMALIZATION',v['C'],'1'),('QUADRATIC',v['Q'],'1'),('COEFFICIENT_SQUARED_TIME',v['T'],'s')],companion='POINT-JINC-DIAGNOSTIC')
 write_fits(d/'certificate.fits',header,[('SCI_SUPPORT',good.astype('u1'),'1'),('CAUSES',causes,'bitmask'),('MEMBERS',counts.astype('f8'),'1'),('ABS_C',v['absolute-C'],'1'),('ERROR_N',v['error-0'],'mJy/nominal beam'),('ERROR_C',v['error-1'],'1'),('ERROR_Q',v['error-2'],'1'),('ERROR_T',v['error-3'],'s'),('ERROR_M',v['error-4'],'mJy/nominal beam'),('TOLERANCE_LOWER',v['tolerance-lower'],'mJy/nominal beam')],companion='POINT-JINC-NUMERICAL-CERTIFICATE')
 record=dict(state=s['key'],role=role,array=a,map=bound(d/'map.fits'),diagnostics=bound(d/'certificate.fits'),request=bound(d/'request.json'),native_receipt=bound(d/'native/receipt.yaml'),profile=PROFILE,accepted_pixels=int(good.sum()),negative_C_accepted=int((good&(v['C']<0)).sum()),no_members=int((counts==0).sum()),causes={str(i):int((causes==i).sum()) for i in np.unique(causes)},max_error_m=float(v['error-4'][good].max(initial=0)),max_error_fraction=float((v['error-4'][good]/v['tolerance-lower'][good]).max(initial=0)),wall_seconds=time.perf_counter()-start,native=r,lifecycle='certificate checked; only cause0 signal released')
 save_json(d/'CERTIFICATE.json',record);manifest(d);return record

def read_map(out,state,role='total',operator='JINC'):
 if operator=='MAP':return load(out/'ordinary'/state/(role+'.fits'))
 d=out/'grids'/state/role;v,h=load(d/'map.fits');c,_=load(d/'certificate.fits');v.update(SCI_SUPPORT=c['SCI_SUPPORT'],HITS=c['MEMBERS'],ERROR=c['ERROR_M'],ERROR_C=c['ERROR_C'],ABS_C=c['ABS_C']);return v,h

def freeze(out,review):
 prepared=read(out/'PREPARED.json');require(not (out/'freeze/prospective.json').exists(),'prospective exists');rev=source()
 for n in ('prepared','ast','cache'):verify_manifest(out/n)
 require(Path(review).is_file(),'independent preflight review required');sh=bound(review)
 from point_frozen_shoulder import ancestors,CD
 auth,owner=ancestors();benchmark={}
 for a in ARRAYS:
  domains={};refs={}
  for b in ('C-core-plus','C-core-minus','C-shoulder'):
   path=CD/f'report/{a}/{b}/cycle-10.fits';v,h=auth.fits(path);domains[b]=v['COMMON'].astype(bool);refs[b]=auth.ref(path)
   require(v['COMMON'].shape==(prepared['arrays'][a]['header']['NAXIS2'],prepared['arrays'][a]['header']['NAXIS1']),'benchmark target shape')
   for k in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2'):require(h[k]==prepared['arrays'][a]['header'][k],'benchmark WCS differs')
  np.savez(out/'freeze'/(a+'-benchmark-domains.npz'),**domains);benchmark[a]=dict(original_report_maps=refs,realized_masks=bound(out/'freeze'/(a+'-benchmark-domains.npz')))
 record=dict(schema='point-152389-jinc-prospective-v1',benchmark_domains=benchmark,prepared=bound(out/'PREPARED.json'),profile=bound(out/'freeze/profile.json'),source_revision=rev,native_source_revision='a5c45cd7e38c9e48b0b37a73ba260859a3f52700',native_source_compatibility='subsequent commits before preparation change only Python orchestration/documentation; native helper/AST identical to reviewed a5',native_binary=prepared['helper'],cli=prepared['cli'],independent_review=sh,ordinary_replay='all15 states and exact same-sample membership passed during preparation',measurement='saved unweighted seven-parameter Gaussian plus constant; 40arcsec original disk; original M0 template and apertures; finite constrained fits retained',ratio_floor=RATIO_FLOOR,seeds=SEEDS,generator='point_post_fruit.ensemble/point_additive.draw_noise; numpy default_rng PCG64; [master_by_operator,integer_wavelength,10,0,trial]',trial_count=256,matrix=[dict(branch=b,stage=k) for b,k in MATRIX],injection_references=['C-core-plus','C-core-minus','C-shoulder'],phase_offsets=prepared['arrays']['a1100']['phases'],noise_mask='bound prepared exact A+B exclusions square dilated h+1 once, unchanged both maps/M0T10',covariate='ordinary HITS or JINC positive footprint membership counts; learned empirical amplitude relation, no precision interpretation',domains='original benchmark complete common SCI_SUPPORT over original full destination, recovered exact masks from C+D report; source fits retain40arcsec disk; operator-native and paired common response domains separately',native_workers_per_job=1,max_simultaneous_jobs=WORKERS,library_threads=1,upstream_scope='zero upstream calls structurally: campaign invokes only standalone gridding helper; consumes historical z and applied m9',uniform_grant='original MAP payload grant is lineage; separate owner SCI-PTC/JINC r0.4 UFC-OD-009 grants same gamma1 into diagnostic rejoined-descendant; no new production grant',rejoin_arithmetic='triangle of full map bounds plus fixed-input fl(z+v) bound gamma1*maxabs(total)*sumabs(k)/separated_C and rounded subtraction; M0 exact-zero operand reused',resource_policy='12 independent map processes, one native worker and one BLAS thread each; 14logical CPUs/48GiB host')
 save_json(out/'freeze/prospective.json',record);manifest(out/'freeze');return record

def grid(out):
 start=time.perf_counter();p=read(out/'PREPARED.json');f=read(out/'freeze/prospective.json');require(bound(out/'PREPARED.json')==f['prepared'],'prepared changed');helper=verified(p['helper']);verified(f['independent_review']);states=p['states'];results=[];(out/'ordinary').mkdir();(out/'grids').mkdir()
 # Same inputs, same occurrence order; primary ordinary signal replay and saved support.
 for s in states:
  a=s['array'];h=fits.Header(p['arrays'][a]['header']);g=np.fromfile(verified(s['geometry']),dtype='<f8').reshape(-1,4);d=out/'ordinary'/s['key'];d.mkdir();old,_=load(verified(s['archived_total']));support=old['SCI_SUPPORT'].astype(bool)
  controls={}
  for role in ('total','clean','operand'):
   t=np.fromfile(verified(s[role]),dtype='<f8');v=ordinary(g,t,h);v['SCI_SUPPORT']=support.astype('u1');v['PRIMARY'][~support]=np.nan;controls[role]=v
   if role=='total':
    require(np.allclose(v['PRIMARY'][support],old['PRIMARY'][support],rtol=1e-11,atol=1e-9),'ordinary signal replay')
    require(np.array_equal(v['HITS'],old['HITS']),'ordinary count replay')
   write_fits(d/(role+'.fits'),h,[(n,v[n], 'mJy/nominal beam' if n in ('PRIMARY','NUMERATOR','ERROR') else '1') for n in ('PRIMARY','NUMERATOR','NORMALIZATION','HITS','NORM_SUPPORT','SCI_SUPPORT','ERROR')],companion='POINT-ORDINARY-REPLAY')
  difference=abs(controls['total']['PRIMARY']-controls['clean']['PRIMARY']-controls['operand']['PRIMARY']);bound0=sum(v['ERROR'] for v in controls.values())+32*np.finfo(float).eps*(abs(controls['total']['PRIMARY'])+abs(controls['clean']['PRIMARY'])+abs(controls['operand']['PRIMARY']))
  require(np.all(difference[support]<=bound0[support]),'ordinary rejoin bound');save_json(d/'rejoin.json',dict(max_abs=float(difference[support].max()),propagated_bound_max=float(bound0[support].max()),domain_pixels=int(support.sum())))
  if s['branch']=='REF' and s['stage']==10:
   refs=[(b,r['values']) for b,r in p['arrays'][a]['injection_references'].items()]+[(f"phase-{r['phase']}",r['values']) for r in p['arrays'][a]['phases']]
   for role,value in refs:
    v=ordinary(g,np.fromfile(verified(value),dtype='<f8'),h);v['SCI_SUPPORT']=support.astype('u1');v['PRIMARY'][~support]=np.nan
    write_fits(d/(role+'.fits'),h,[(n,v[n],'mJy/nominal beam' if n in ('PRIMARY','NUMERATOR','ERROR') else '1') for n in ('PRIMARY','NUMERATOR','NORMALIZATION','HITS','NORM_SUPPORT','SCI_SUPPORT','ERROR')],companion='POINT-ORDINARY-REFERENCE')
 # REF/control wave completes before dependent perturbation waves.
 for phase in ('REF','C-core-plus','C-core-minus','C-shoulder','fixtures'):
  jobs=[]
  for s in states:
   if phase=='fixtures' and s['branch']=='REF' and s['stage']==10:
    ar=p['arrays'][s['array']];jobs.extend((out,p,s,b,r['values'],helper) for b,r in ar['injection_references'].items());jobs.extend((out,p,s,f"phase-{r['phase']}",r['values'],helper) for r in ar['phases'])
   elif phase==s['branch']:
    jobs.extend((out,p,s,role,s[role],helper) for role in (('total',) if s['stage']==0 else ('total','clean','operand')))
  with ThreadPoolExecutor(max_workers=WORKERS) as pool:
   for r in pool.map(map_job,jobs):results.append(r);print(r['state'],r['role'],r['accepted_pixels'],flush=True)
 for s in states:
  d=out/'grids'/s['key'];a=p['arrays'][s['array']]
  if s['stage']==0:
   save_json(d/'REJOIN.json',dict(status='exact reuse',identity='M0 operand is identically zero; cleaned payload identical total',zero_operand=bool(np.all(np.fromfile(verified(s['operand']),dtype='<f8')==0)),clean_total_identical=s['clean']['sha256']==s['total']['sha256']));continue
  maps=[read_map(out,s['key'],role)[0] for role in ('total','clean','operand')];m,z,v=maps;use=np.logical_and.reduce([x['SCI_SUPPORT'].astype(bool) for x in maps]);den=np.nextafter(abs(m['NORMALIZATION'])-m['ERROR_C'],-np.inf)
  maxt=float(abs(np.fromfile(verified(s['total']),dtype='<f8')).max());source_add=np.nextafter((2.**-53/(1-2.**-53))*maxt*np.nextafter(m['ABS_C']+m['ERROR_C'],np.inf)/den,np.inf)
  # ABS_C error is bounded by relative accumulation gamma plus coefficient EC;
  # the additional directed factor encloses that positive sum's own rounding.
  gamma=(m['HITS']+7)*2.**-53/(1-(m['HITS']+7)*2.**-53);source_add=np.nextafter(source_add/(1-gamma),np.inf)
  b=np.nextafter(sum(x['ERROR'] for x in maps)+source_add+32*np.finfo(float).eps*sum(abs(x['PRIMARY']) for x in maps),np.inf);delta=abs(m['PRIMARY']-z['PRIMARY']-v['PRIMARY']);require(np.all(delta[use]<=b[use]),'JINC rejoin beyond propagated bound')
  save_json(d/'REJOIN.json',dict(status='passed',pixels=int(use.sum()),unavailable_pixels=int((~use).sum()),max_abs=float(delta[use].max()),max_bound=float(b[use].max()),max_fraction=float((delta[use]/b[use]).max()),arithmetic='triangle of three certificates + sample rejoin rounding + subtraction rounding'))
 record=dict(status='matrix accumulated and certificate checked',primary_maps=30,logical_injection_references=18,phase_maps=54,actual_native_accumulations=len(results),JINC=results,wall_seconds=time.perf_counter()-start,worker_jobs=WORKERS,native_workers_per_job=1,library_threads=1)
 save_json(out/'GRIDS.json',record);manifest(out/'ordinary');manifest(out/'grids');return record

def measured_grid(out):
 stop=threading.Event();samples=[];process=psutil.Process();start=time.perf_counter()
 def monitor():
  while not stop.is_set():
   members=[]
   for p in [process,*process.children(recursive=True)]:
    try:
     cpu=p.cpu_times();members.append(dict(pid=p.pid,rss=p.memory_info().rss,cpu=cpu.user+cpu.system,threads=p.num_threads()))
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   samples.append(dict(seconds=time.perf_counter()-start,processes=members,aggregate_rss=sum(p['rss'] for p in members)))
   stop.wait(1.)
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:return grid(out)
 finally:
  stop.set();thread.join();save_json(out/'GRID_RESOURCES.json',dict(samples=samples,wall_seconds=time.perf_counter()-start,sampled_aggregate_rss_max=max((s['aggregate_rss'] for s in samples),default=0),meaning='one-second sampled process RSS sum; distinct from per-process high-water marks; CPU totals from native receipts'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','grid']);p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path);a=p.parse_args();freeze(a.output,a.review) if a.stage=='freeze' else measured_grid(a.output)
