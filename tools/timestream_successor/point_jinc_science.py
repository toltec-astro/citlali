"""Unchanged POINT estimators and A+B covariance applied after gridding gates."""
import argparse,time
from pathlib import Path
import numpy as np
from scipy.signal import convolve2d
from astropy.io import fits
from point_jinc_campaign import read_map,fixed_for,measure_full,SEEDS
from point_jinc_full import read,ARRAYS,key,HALF
from point_response_cd import vector_measurement,MEASUREMENT_NAMES,RATIO_FLOOR,ellipse
from point_post_fruit import (measurement_values,context_for,ensemble,joint_summary,propagation_checks,applicability,trial)
from point_additive import learn,NoiseRequest,variance
from run_point_post_fruit import larger_comparison,save_model
from point_additive_diagnostics import save_json,regional_summary
from point_fruit_trajectory import verified,bound,manifest,verify_manifest
from point_empirical import coordinates,source_patch,stats
from point_empirical_io import write_fits
from inspect_point_source import gaussian
from inspect_forward_map import require

OPERATORS=('MAP','JINC');PERTURBATIONS=('C-core-plus','C-core-minus','C-shoulder')

def response(delta,known,domain,available):
 if np.any(domain&~available):return dict(status='unavailable',required_pixels=int(domain.sum()),missing_pixels=int((domain&~available).sum()))
 d=delta[domain];h=known[domain];norm=np.linalg.norm(h)
 if not len(h) or norm<=RATIO_FLOOR:return dict(status='unavailable',cause='negligible reference norm',pixels=len(h))
 g=float(np.dot(h,d)/norm**2);eta=float(np.linalg.norm(d-h)/norm);rho=float(np.linalg.norm(d-g*h)/norm)
 require(abs(eta**2-(g-1)**2-rho**2)<1e-10,'response identity')
 return dict(status='available',pixels=len(h),g=g,eta=eta,rho=rho,identity_error=eta**2-(g-1)**2-rho**2,reference_norm=float(norm),residual_rms=float(np.sqrt(np.mean((d-h)**2))))

def science(out):
 start=time.perf_counter();p=read(out/'PREPARED.json');freeze=read(out/'freeze/prospective.json');verify_manifest(out/'ordinary');verify_manifest(out/'grids');dest=out/'science';dest.mkdir();measurements={};replay=[]
 # Exactly one accepted real-map fit per primary state/operator.
 for s in p['states']:
  a=s['array'];fixed=fixed_for(out,a)
  for op in OPERATORS:
   d,h=read_map(out,s['key'],operator=op);m=measure_full(d,h,fixed);measurements[s['key'],op]=m
   if op=='MAP':
    av,bv=vector_measurement(m),vector_measurement(s['measurement']);require(av is not None and bv is not None,'ordinary measurement unavailable')
    delta=abs(av-bv);ok=np.allclose(av,bv,rtol=1e-10,atol=1e-10)
    row=dict(state=s['key'],pass_replay=bool(ok),max_absolute=float(delta.max()),new=m,archived=s['measurement'],rtol=1e-10,atol=1e-10);replay.append(row)
    save_json(dest/'ordinary-measurement-replay.json',replay);require(ok,'ordinary measurement replay '+s['key'])
   save_json(dest/(s['key']+'-'+op+'-measurement.json'),m)
 results={}
 for a in ARRAYS:
  fixed=fixed_for(out,a);ad=dest/a;ad.mkdir();refkey=key(a,'REF',10);ref={op:read_map(out,refkey,operator=op)[0] for op in OPERATORS};h=read_map(out,refkey)[1];x,y=coordinates(ref['MAP']['PRIMARY'].shape,h);orig=np.load(verified(freeze['benchmark_domains'][a]['realized_masks']));rows=[]
  for b in PERTURBATIONS:
   branch={op:read_map(out,key(a,b,10),operator=op)[0] for op in OPERATORS};known={op:read_map(out,refkey,b,op)[0] for op in OPERATORS};native={op:branch[op]['SCI_SUPPORT'].astype(bool)&ref[op]['SCI_SUPPORT'].astype(bool)&known[op]['SCI_SUPPORT'].astype(bool) for op in OPERATORS};common=native['MAP']&native['JINC'];original=orig[b]
   for op in OPERATORS:
    dr=ad/(b+'-'+op);dr.mkdir();r,t,k=ref[op],branch[op],known[op];use=native[op];delta=t['PRIMARY']-r['PRIMARY'];resid=delta-k['PRIMARY'];bm=np.nextafter(t['ERROR']+r['ERROR']+k['ERROR']+32*np.finfo(float).eps*(abs(t['PRIMARY'])+abs(r['PRIMARY'])+abs(k['PRIMARY'])),np.inf)
    base=measurements[refkey,op];actual=measurements[key(a,b,10),op];ideal_data=dict(r);ideal_data['PRIMARY']=r['PRIMARY']+k['PRIMARY'];ideal_data['SCI_SUPPORT']=(r['SCI_SUPPORT'].astype(bool)&k['SCI_SUPPORT'].astype(bool)).astype('u1');ideal=measure_full(ideal_data,h,fixed)
    va,vb,vi=map(vector_measurement,(actual,base,ideal));comparison=None
    if all(v is not None for v in (va,vb,vi)):
     comparison=[dict(name=n,actual_increment=float(da),ideal_increment=float(di),discrepancy=float(da-di),ratio=float(da/di) if abs(di)>RATIO_FLOOR else None) for n,da,di in zip(MEASUREMENT_NAMES,va-vb,vi-vb)]
    profiles=[]
    for lo,hi in zip(np.arange(0,100,4),np.arange(4,104,4)):
     roi=(np.hypot(x-fixed.parameters[1],y-fixed.parameters[2])>=lo)&(np.hypot(x-fixed.parameters[1],y-fixed.parameters[2])<hi)&use
     profiles.append(dict(radius_arcsec=(lo+hi)/2,pixels=int(roi.sum()),H=float(k['PRIMARY'][roi].mean()) if roi.any() else None,delta=float(delta[roi].mean()) if roi.any() else None,residual=float(resid[roi].mean()) if roi.any() else None))
    record=dict(array=a,operator=op,branch=b,original_benchmark=response(delta,k['PRIMARY'],original,use),native=response(delta,k['PRIMARY'],use,use),paired_common=response(delta,k['PRIMARY'],common,use),support=dict(original_pixels=int(original.sum()),native_pixels=int(use.sum()),common_pixels=int(common.sum()),lost_original=int((original&~use).sum()),gained_original=int((use&~original).sum()),branch_lost_REF=int((r['SCI_SUPPORT'].astype(bool)&~t['SCI_SUPPORT'].astype(bool)).sum()),branch_gained_REF=int((~r['SCI_SUPPORT'].astype(bool)&t['SCI_SUPPORT'].astype(bool)).sum())),measurement_comparison=comparison,reference=base,actual=actual,ideal_map_addition=ideal,profiles=profiles,numerical_residual_bound_max=float(bm[use].max()) if use.any() else None,residual_within_numerical_bound_pixels=int((abs(resid[use])<=bm[use]).sum()),scope='regridded differences between completed adaptive ordinary histories; no response correction or upstream recovery')
    save_json(dr/'RESPONSE.json',record);rows.append(record)
    write_fits(dr/'response.fits',h,[(n,np.where(use,v,np.nan),'mJy/nominal beam') for n,v in [('H',k['PRIMARY']),('DELTA',delta),('RESIDUAL',resid),('BOUND',bm)]]+[(n,v.astype('u1'),'1') for n,v in [('NATIVE',use),('PAIRED_COMMON',common),('ORIGINAL_DOMAIN',original)]],companion='POINT-JINC-RESPONSE')
  even=[]
  for op in OPERATORS:
   plus,_=read_map(out,key(a,'C-core-plus',10),operator=op);minus,_=read_map(out,key(a,'C-core-minus',10),operator=op);r=ref[op];use=plus['SCI_SUPPORT'].astype(bool)&minus['SCI_SUPPORT'].astype(bool)&r['SCI_SUPPORT'].astype(bool);v=plus['PRIMARY']+minus['PRIMARY']-2*r['PRIMARY'];b=plus['ERROR']+minus['ERROR']+2*r['ERROR']+32*np.finfo(float).eps*(abs(plus['PRIMARY'])+abs(minus['PRIMARY'])+2*abs(r['PRIMARY']))
   write_fits(ad/('even-'+op+'.fits'),h,[('EVEN',np.where(use,v,np.nan),'mJy/nominal beam'),('SUPPORT',use.astype('u1'),'1'),('BOUND',b,'mJy/nominal beam')],companion='POINT-JINC-EVEN');even.append(dict(operator=op,full=stats(v[use]),source_disk=stats(v[use&(np.hypot(x,y)<=40)]),bound_max=float(b[use].max()) if use.any() else None))
  phases=[];p0=np.array(p['arrays'][a]['compact_profile']['parameters'])
  for ph in p['arrays'][a]['phases']:
   for op in OPERATORS:
    v,vh=read_map(out,refkey,f"phase-{ph['phase']}",op);m=measure_full(v,vh,fixed);fit=m['fit'];expected=p0[1:3]+ph['offset_tangent_arcsec'];use=v['SCI_SUPPORT'].astype(bool);signed=v['PRIMARY'][use];record=dict(array=a,operator=op,phase=ph['phase'],offset_pixels=ph['offset_pixels'],true_centroid=expected,measurement=m,centroid_error_arcsec=np.array(fit['parameters'][1:3])-expected if fit['status']=='success' else None,ellipse=ellipse(fit['parameters']) if fit['status']=='success' else None,minimum=float(signed.min()) if signed.size else None,negative_brightness_area=float(signed[signed<0].sum()*4),negative_pixels=int((signed<0).sum()),support_pixels=int(use.sum()),normalization=p0[0],taper='same saved30..36arcsec taper shifted with continuous profile; full destination; no renormalization')
    phases.append(record)
  save_json(ad/'PHASES.json',phases);save_json(ad/'EVEN.json',even);results[a]=dict(responses=rows,phases=phases,even=even)
 record=dict(arrays=results,ordinary_replay=replay,source_fits=30,ideal_control_fits=18,phase_fits=54,wall_seconds=time.perf_counter()-start);save_json(out/'SCIENCE.json',record);manifest(dest);return record

def contributions(out,p,a,operator):
 s=next(s for s in p['states'] if s['key']==key(a,'REF',10));g=np.fromfile(verified(s['geometry']),dtype='<f8').reshape(-1,4);net=np.fromfile(out/'prepared'/a/s['key']/'network.i16',dtype='<i2');shape=(p['arrays'][a]['header']['NAXIS2'],p['arrays'][a]['header']['NAXIS1']);pixel=np.floor(g[:,1]-.5).astype('i8')*shape[1]+np.floor(g[:,0]-.5).astype('i8');result={}
 for n in np.unique(net):
  v=np.bincount(pixel[net==n],minlength=np.prod(shape)).reshape(shape)
  if operator=='JINC':v=np.rint(convolve2d(v,np.ones((2*HALF[a]+1,)*2),mode='same',boundary='fill')).astype('i8')
  result[str(n)]=v
 actual,_=read_map(out,s['key'],operator=operator);require(np.array_equal(sum(result.values()),actual['HITS']),'network footprint membership mismatch');return result

def noise(out):
 start=time.perf_counter();p=read(out/'PREPARED.json');frozen=read(out/'freeze/prospective.json');science=read(out/'SCIENCE.json');dest=out/'noise';dest.mkdir();allresults=[]
 for a in ARRAYS:
  fixed=fixed_for(out,a);maps={(op,k):read_map(out,key(a,'REF',k),operator=op)[0] for op in OPERATORS for k in (0,10)};h=read_map(out,key(a,'REF',10))[1];common=np.logical_and.reduce([v['SCI_SUPPORT'].astype(bool) for v in maps.values()]);masks=np.load(out/'prepared'/a/'noise-masks.npz');ad=dest/a;ad.mkdir();np.savez(ad/'domains.npz',common_support=common,base=masks['base'],larger=masks['larger']);summary=[]
  for op in OPERATORS:
   od=ad/op;od.mkdir();counts=contributions(out,p,a,op);np.savez(od/'network-memberships.npz',**counts);base=None;c=None;request=NoiseRequest(mask_fwhm=4.,seed=SEEDS[op])
   for k in (0,10):
    d=od/('M0' if k==0 else 'T10');d.mkdir();v=maps[op,k];meas=read(out/'science'/(key(a,'REF',k)+'-'+op+'-measurement.json'));t=time.perf_counter();native=v['SCI_SUPPORT'].astype(bool);record=dict(array=a,operator=op,stage=k,native_background=stats(v['PRIMARY'][native&~masks['base']]),common_background=stats(v['PRIMARY'][common&~masks['base']]),native_background_pixels=int((native&~masks['base']).sum()),fixed_common_background_pixels=int((common&~masks['base']).sum()),covariate='positive same-operator sample footprint membership count; no precision meaning',training_support='fixed intersection of MAP/JINC M0/T10 native support; exact frozen h+1 dilated A+B mask',qualification='a2000 spatial background model remains qualified; a1100 inherited nw3/scan5 omission retained; separable empirical transfer is a model assumption')
    try:
     m=learn(v['PRIMARY'],v['HITS'],common,h,fixed.parameters,request,stamp_size=49,source_mask=masks['base']);require(meas['fit']['status']=='success','source fit unavailable for noise baseline');ctx=context_for(v['PRIMARY'],v['HITS'],native,h,fixed,meas['fit']['parameters'],m)
     b=v['PRIMARY'][m['background_mask']];large=larger_comparison(v['PRIMARY'],v['HITS'],common,native,h,fixed,meas['fit']['parameters'],request,masks['larger'],ctx['weights']);ap=applicability(v['PRIMARY'],v['HITS'],h,fixed.parameters,m,counts,ctx['weights'])
     record.update(status='available',background=stats(b),second_moment_about_zero=float(np.mean(b*b)),second_moment_about_training_mean=float(np.mean((b-b.mean())**2)),analytic_scatter=[np.sqrt(variance(m['psd'],ctx['scale'],w)) for w in ctx['weights']],diagonal_scatter=[np.sqrt(np.sum((w*ctx['scale'])**2)) for w in ctx['weights']],larger_mask=large,applicability=ap,source_depth_in_background_range=ctx['complete_amplitude_support'],outside_source_range_pixels=ctx['outside_amplitude_range_fit_pixels'],model=m['summary'])
     save_model(d,m,ctx,h);save_json(d/'regions.json',regional_summary(m,ctx))
     if k==10:base=m;c=ctx
    except ValueError as exc:record.update(status='unavailable',cause=str(exc))
    record['inference_diagnostics_wall_seconds']=time.perf_counter()-t;save_json(d/'BACKGROUND.json',record);summary.append(record)
   if c is None:
    allresults.append(dict(array=a,operator=op,status='unavailable',cause='T10 model/domain unavailable',trials=0,background=summary[-2:]));continue
   t=time.perf_counter();records=ensemble(c,256,SEEDS[op],a,'JINC-comparison-'+op,12);observed=measurement_values(read(out/'science'/(key(a,'REF',10)+'-'+op+'-measurement.json')));joint=joint_summary(records,c['noiseless'],observed);save_json(od/'trials.json',records);save_json(od/'UNCERTAINTY.json',joint);save_json(od/'noiseless.json',c['noiseless_record']);ensemble_seconds=time.perf_counter()-t
   checks=propagation_checks(c,records,SEEDS[op],a);save_json(od/'propagation.json',checks)
   # Repeat identical trial0 to check deterministic regeneration, not another draw.
   regenerated=trial((c,tuple(records[0]['seed']),records[0]['identity'],0));require(regenerated['status']==records[0]['status'] and regenerated['values']==records[0]['values'] and regenerated['linear_noise']==records[0]['linear_noise'],'trial regeneration differs')
   plus=measurement_values(read(out/'science'/(key(a,'C-core-plus',10)+'-'+op+'-measurement.json')));minus=measurement_values(read(out/'science'/(key(a,'C-core-minus',10)+'-'+op+'-measurement.json')));delta=.1*p['arrays'][a]['peak']['P_mJy_per_nominal_beam'];ratios=[]
   for i,j in enumerate(joint['measurements']):
    valid=all(z is not None for z in (plus[i],minus[i],observed[i]));slope=(plus[i]-minus[i])/(2*delta) if valid else None;scatter=j['scatter'];ratios.append(dict(measurement=j['name'],units=j['units'],positive_response=(plus[i]-observed[i])/delta if valid else None,negative_response=(minus[i]-observed[i])/(-delta) if valid else None,even=plus[i]+minus[i]-2*observed[i] if valid else None,slope=slope,output_scatter=scatter,input_referenced_scatter=scatter/abs(slope) if scatter is not None and slope is not None and abs(slope)>RATIO_FLOOR else None,ratio_floor=RATIO_FLOOR))
   save_json(od/'RESPONSE_REFERENCED.json',ratios)
   linear=np.asarray([r['linear_noise'] for r in records]);save_json(od/'LINEAR_UNCERTAINTY.json',dict(trials=256,names=['template_raw','aperture1_raw','aperture1.5_raw','aperture2_raw'],scatter=linear.std(axis=0,ddof=1),covariance=np.cov(linear,rowvar=False,ddof=1),independently_valid=True,scope='fixed-background linear operators; no fitted-constant coupling'))
   good=[r for r in records if r['status']=='success'];coupling=[]
   if len(good)>1:
    bg=np.array([r['fit']['parameters'][6] for r in good]);ln=np.array([r['linear_noise'] for r in good])
    for i,w in enumerate(c['weights']):
     area=w.sum();cross=np.cov(ln[:,i],bg,ddof=1)[0,1];pred=ln[:,i].var(ddof=1)+area**2*bg.var(ddof=1)-2*area*cross;actual=np.array([r['values'][i+1] for r in good]).var(ddof=1);require(np.isclose(pred,actual,rtol=1e-10,atol=1e-10),'same-trial background coupling')
     coupling.append(dict(name=joint['measurements'][i+1]['name'],raw_variance=ln[:,i].var(ddof=1),background_variance=bg.var(ddof=1),cross_term=-2*area*cross,total_variance=pred,actual_variance=actual,primary_complete=joint['primary_available']))
   save_json(od/'background-coupling.json',coupling)
   allresults.append(dict(array=a,operator=op,status='complete' if joint['primary_available'] else 'complete trials; primary nonlinear aggregates unavailable',trials=256,uncertainty=joint,response_referenced=ratios,background=summary[-2:],propagation=checks,ensemble_wall_seconds=ensemble_seconds,trial_cpu_seconds=sum(r['cpu_seconds'] for r in records),regeneration_exact=True,source_baseline='fitted seven-parameter Gaussian plus constant; non-Gaussian signed JINC structure not represented; conditional additive model, no full-FRUIT variance'))
  save_json(ad/'BACKGROUND.json',summary)
 result=dict(results=allresults,trial_count=sum(r['trials'] for r in allresults),wall_seconds=time.perf_counter()-start,interpretation='conditional additive scatter under transferred empirical covariance; no physical cross-map covariance, absolute calibration, correction or feedback admission claim');save_json(out/'NOISE.json',result);manifest(dest);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['science','noise']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();science(a.output) if a.stage=='science' else noise(a.output)
