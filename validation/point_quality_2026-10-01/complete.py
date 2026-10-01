"""Close the bounded batch using saved products only; no native invocation or retry."""
import os
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-quality-close-mpl',XDG_CACHE_HOME='/private/tmp/point-quality-close-cache')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,importlib.util,re,gzip,shutil
from pathlib import Path
import numpy as np
from astropy.io import fits
D=Path(__file__).resolve().parent;W=D.parents[1];sys.path.insert(0,str(W/'tools/timestream_successor'));sys.path.insert(0,str(D))
from point_quality_experiment import ROOT,ROLES,LIMITS,PRIOR,CAP
from point_fruit_trajectory import bound,verified
from point_additive_diagnostics import save_json
from point_empirical import coordinates
from point_storage import tree_bytes
from inspect_forward_map import require
import saved_scorecard as score
OUT=D/'results/completed'
def read(p):return json.loads(Path(p).read_text())
def transfer(actual,ideal,amplitude,sigma):
 require(np.isfinite([actual,ideal,amplitude,sigma]).all() and abs(ideal)>1e-8 and abs(amplitude)>1e-8 and sigma>0,'invalid amplitude response')
 gain=actual/amplitude;require(abs(gain)>1e-8,'unresolved input-equivalent uncertainty')
 return dict(actual_peak_increment=actual,ideal_estimator_increment=ideal,analytic_input_amplitude=amplitude,input_amplitude_gain=gain,ideal_estimator_gain=actual/ideal,sigma_A=sigma,input_equivalent_sigma_A=sigma/abs(gain),ideal_estimator_equivalent_sigma_A=sigma/abs(actual/ideal))
def exposure():
 spec=importlib.util.spec_from_file_location('population_saved',W/'validation/rtc_two_stage_2026-09-28/two_stage_population.py');pop=importlib.util.module_from_spec(spec);spec.loader.exec_module(pop);pop.B=ROOT/'B-REF';rows=[]
 for nw in range(1,5):
  base,r,pr,stages,pix,off,distance,lon,lat=pop.network(nw);shape=pix.shape;probe=base/'source-probe';mapping=pop.arr(probe/'mapping-available.u8','u1',shape).astype(bool);pair=pop.arr(probe/'input-valid.u8','u1',shape).astype(bool)&pop.arr(probe/'input-r-valid.u8','u1',shape).astype(bool)
  inside=mapping&np.isfinite(lon)&np.isfinite(lat)&(np.hypot(lon,lat)<=40);retained=np.isfinite(stages['T5']);mapped=pix>=0;dt=r['native_integration_seconds'];episodes=[]
  for det in range(pr['detectors']):
   for rid,(start,end) in enumerate(r['native_runs']):
    for lo,hi in pop.runs(inside[start:end,det]):
     lo+=start;hi+=start
     episodes.append(dict(network=nw,channel=pr['channels'][det],detector=det,physical_run=rid,first=int(lo),past_last=int(hi),geometric_samples=int(hi-lo),original_paired_samples=int(pair[lo:hi,det].sum()),retained_samples=int(retained[lo:hi,det].sum()),map_samples=int(mapped[lo:hi,det].sum())))
  save_json(OUT/f'source-crossings-network{nw}.json',episodes)
  rows.append(dict(network=nw,detectors=pr['detectors'],physical_runs=r['native_runs'],integration_seconds=dt,native_cell_count=int(pair.size),original_paired_cells=int(pair.sum()),retained_cells=int(retained.sum()),MAP_contributions=int(mapped.sum()),retained_detector_count=int(np.any(retained,axis=0).sum()),source_geometry_cells=int(inside.sum()),source_original_paired_cells=int((inside&pair).sum()),source_retained_cells=int((inside&retained).sum()),source_retained_original_paired_cells=int((inside&retained&pair).sum()),source_retained_without_original_pair=int((inside&retained&~pair).sum()),source_MAP_contributions=int((inside&mapped).sum()),source_original_detector_seconds=float((inside&pair).sum()*dt),source_retained_detector_seconds=float((inside&retained).sum()*dt),source_crossing_episodes=len(episodes),episodes_with_original_pairs=sum(x['original_paired_samples']>0 for x in episodes),episodes_with_any_retained=sum(x['retained_samples']>0 for x in episodes),original_paired_episodes_with_any_retained=sum(x['original_paired_samples']>0 and x['retained_samples']>0 for x in episodes),original_paired_episodes_with_no_retained=sum(x['original_paired_samples']>0 and x['retained_samples']==0 for x in episodes),episodes_with_no_retained=sum(x['retained_samples']==0 for x in episodes),receipt=bound(base/'receipt.yaml'),source_probe=bound(probe/'receipt.yaml')))
  print('support network',nw,flush=True)
 save_json(OUT/'support.json',dict(networks=rows,qualification='Saved native RTC/PTC/MAP support; no new processing. Crossing=contiguous geometry-available episode inside commanded40arcsec disk, separately within each physical run/detector; not independent sky measurement. Exposure is detector-seconds, not elapsed time. Original invalid/mapping-unavailable data remain unavailable. Shared helper verifies REF M0/T5 operator/support equality.'))
 return rows

def figures(maps):
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
 for col,(name,d) in enumerate(maps.items()):
  x,y,s,v=[d[k] for k in ('x','y','signal','support')];dx=x[0,1]-x[0,0];dy=y[1,0]-y[0,0];extent=[x[0,0]-dx/2,x[0,-1]+dx/2,y[0,0]-dy/2,y[-1,0]+dy/2]
  for row,lim in enumerate((60,100)):
   im=axs[row,col].imshow(np.where(v,s,np.nan),origin='lower',extent=extent,cmap='RdBu_r',vmin=-80,vmax=650 if row==0 else 80);axs[row,col].set(xlim=(lim,-lim),ylim=(-lim,lim),title=name+' / '+d['header']['CTYPE1'],xlabel='native x (arcsec)',ylabel='native y (arcsec)');axs[row,col].plot(0,0,'k+',ms=5)
   for radius in ([40] if row==0 else [60,100]):axs[row,col].add_patch(plt.Circle((0,0),radius,fill=False,color='black',ls=':',lw=.6))
   fig.colorbar(im,ax=axs[row,col],label='mJy / nominal beam',shrink=.7)
 fig.suptitle('POINT152420 completed: R operational / current S M0 / current S T5\nNative frames not registered; R includes network0; S networks1–4; no rescaling')
 fig.savefig(OUT/'point-152420-completed.png',dpi=130);plt.close(fig)
 a=np.load(ROOT/'report/response-B/response-maps.npz');fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
 for row,branch in enumerate(('C-core-plus','C-core-minus')):
  key=branch+'_T5_';n=a[key+'REF'].shape[0];xy=np.arange(n)*2-(n-1);x,y=np.meshgrid(-xy,xy);valid=a[key+'support'].astype(bool)&(np.hypot(x,y)<=40)
  for col,(label,z,limit) in enumerate([('ideal compact',a[key+'ideal'],25),('injected minus own REF',a[key+'injected']-a[key+'REF'],25),('response minus ideal',a[key+'residual'],2.5)]):
   im=axes[row,col].imshow(np.where(valid,z,np.nan),origin='lower',extent=[n,-n,-n,n],cmap='RdBu_r',vmin=-limit,vmax=limit);axes[row,col].set(xlim=(40,-40),ylim=(-40,40),title=branch+' / '+label,xlabel='tangent east (arcsec)',ylabel='tangent north (arcsec)');fig.colorbar(im,ax=axes[row,col],label='mJy / nominal beam',shrink=.7)
 fig.suptitle('POINT152420 T5 compact response: common40arcsec support\nOwn references, independently relearned RTC/PTC/feedback; residual display has separate scale')
 fig.savefig(OUT/'compact-response.png',dpi=130);plt.close(fig)

def main():
 started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);status=read(ROOT/'STATUS.json');account=read(ROOT/'ACCOUNTING.json');freeze=read(ROOT/'FREEZE.json');require(status['state']=='COMPLETE_PENDING_OWNER_REVIEW' and status['exit_code']==0,'batch not complete')
 require(account['charged_attempts']==LIMITS and len(account['receipts'])==12 and all(r['Apply_performed'] for r in account['receipts']),'unexpected accounting')
 for row in account['receipts']:verified(row['receipt'])
 roles=[];bindings=[bound(ROOT/name) for name in ('STATUS.json','ACCOUNTING.json','FREEZE.json','PREFLIGHT_REVIEW.json')]
 for role in ROLES:
  out=ROOT/role;s=read(out/'run/BASE-POINT/REF/a1100/SUMMARY.json');launch=read(out/'LAUNCH.json');execution=read(out/'EXECUTION.json');q=read(out/'run/BASE-POINT/REF/a1100/cycle-05/request.json')
  require(execution['status']=='completed' and execution['completed_cycle']==5 and execution['identity']['source_revision']==freeze['source_revision'] and launch['exit_code']==0,'execution failure')
  require(s['ready'] and s['completed_cycle']==5 and s['actual_feedback_applications']==5 and q['candidate']==s['histories'][4]['candidate'] and q['candidate']!=s['histories'][5]['candidate'],'T5/m4 binding failure');verified(q['candidate'])
  require(not (out/'run/BASE-POINT/REF/a1100/cycle-06').exists(),'extra feedback')
  errors=[str(p) for p in out.rglob('run.log') if re.search(r'\[(?:error|critical)\]',p.read_text(),re.I)];require(not errors,'unexpected native errors')
  timings=[json.loads(line) for line in (out/'timing.jsonl').read_text().splitlines()];spans={}
  for row in timings:
   if row['event']=='span':spans[row['name']]=spans.get(row['name'],0)+row['seconds']
  roles.append(dict(role=role,source_revision=freeze['source_revision'],M0=s['histories'][0]['measurement'],T5=s['histories'][5]['measurement'],endpoint_uncertainty=s['endpoint_uncertainty'],T5_consumes_m4=True,m5_applied=False,unexpected_error_logs=errors,wall_seconds=launch['exit_seconds'],sampled_tree_RSS_bytes=launch['aggregate_tree_sampled_RSS'],stage_spans_seconds=spans,stage_scope='inclusive nested wall spans are not additive',feedback_trajectory_wall_seconds=s['wall_seconds'],history_source=bound(out/'run/BASE-POINT/REF/a1100/SUMMARY.json')))
  bindings += [bound(out/name) for name in ('LAUNCH.json','EXECUTION.json','timing.jsonl','COMPLETED-SCRATCH-REMOVALS.json','freeze/prospective.json','run/BASE-POINT/REF/a1100/SUMMARY.json','run/BASE-POINT/REF/a1100/cycle-05/request.json')]
 rawresponse=read(ROOT/'report/response-B/source-responses.json');shape=read(ROOT/'B-REF/SHAPES.json');sigma=next(r['conditional_sigma'] for r in roles[0]['endpoint_uncertainty'] if r['name']=='peak');response=[]
 for r in rawresponse:
  p=next(x for x in r['measurement']['comparisons'] if x['name']=='peak');amp=shape['rasters'][r['branch']]['analytic_peak'];sigma_here=sigma if r['cycle']==5 else r['M0_guard_sigma_A'];t=transfer(p['actual_increment'],p['ideal_increment'],amp,sigma_here)
  response.append(dict(branch=r['branch'],cycle=r['cycle'],**t,spatial=r['spatial'],support=r['support'],uncertainty_scope='T5 endpoint conditional covariance divided by finite perturbation gain; not differential-gain uncertainty' if r['cycle']==5 else 'M0 guard reference covariance divided by finite perturbation gain',centroid_discrepancy_arcsec={x['name']:x['absolute_discrepancy'] for x in r['measurement']['comparisons'] if x['name'] in ('east','north')}))
 # Independent scalar/raster normalization check from the recorded physical parameters.
 normalization=[]
 for branch in ('C-core-plus','C-core-minus'):
  sh=shape['rasters'][branch];path=verified(sh)
  with fits.open(path) as h:
   z=h[0].data;valid=h['VALID'].data.astype(bool);x,y=coordinates(z.shape,h[0].header);p=np.array(sh['parameters']);xx=x-p[1];yy=y-p[2];u=np.cos(p[5])*xx+np.sin(p[5])*yy;v=-np.sin(p[5])*xx+np.cos(p[5])*yy;rad=np.hypot(x,y);taper=np.ones_like(rad);taper[rad>=36]=0;mid=(rad>30)&(rad<36);taper[mid]=.5+.5*np.cos(np.pi*(rad[mid]-30)/6);expected=p[0]*np.exp(-.5*((u/p[3])**2+(v/p[4])**2))*taper
   err=float(np.max(abs(z[valid]-expected[valid])));require(err<1e-12,'independent physical raster normalization failed');normalization.append(dict(branch=branch,analytic_amplitude=p[0],discrete_peak=sh['discrete_signed_peak'],maximum_absolute_error=err,raster=bound(path)))
 paths=dict(R=score.L/'citlali-validation/v2/science/pointings/reduced/redu00/152420/raw/toltec_commissioning_a1100_pointing_152420_citlali.fits',M0=ROOT/'B-REF/M0/products/a1100/map.fits',T5=ROOT/'B-REF/run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits')
 maps={name:score.m.load(path,name=='R') for name,path in paths.items()};metrics={name:dict(input=bound(paths[name]),**score.metrics(d)) for name,d in maps.items()};common=score.common_pair(maps['M0'],maps['T5'])
 old=score.m.load(ROOT.with_name('successor-rtc-point-transfer-completion-20260930')/'B-REF/run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits',False);require(np.array_equal(old['x'],maps['T5']['x']) and np.array_equal(old['y'],maps['T5']['y']),'old/new grids differ');v=old['support']&maps['T5']['support'];delta=maps['T5']['signal']-old['signal'];old_metrics=score.metrics(old)
 repeat=dict(old_map=bound(ROOT.with_name('successor-rtc-point-transfer-completion-20260930')/'B-REF/run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits'),peak_change=metrics['T5']['fit']['amplitude']-old_metrics['fit']['amplitude'],map_change_common=score.finite_stats(delta[v]),maximum_absolute_map_change=float(abs(delta[v]).max()),old_pixels=int(old['support'].sum()),new_pixels=int(maps['T5']['support'].sum()),lost=int((old['support']&~maps['T5']['support']).sum()),gained=int((maps['T5']['support']&~old['support']).sum()),qualification='Corrected B-to-B version comparison, not evidence of R/S improvement')
 support=exposure();figures(maps)
 storage=dict(prior=[dict(path=str(p),bytes=tree_bytes(p)) for p in PRIOR],current_bytes=tree_bytes(ROOT),ceiling=CAP);storage['shared_bytes']=sum(p['bytes'] for p in storage['prior'])+storage['current_bytes'];require(storage['shared_bytes']<=CAP,'shared storage exceeded')
 for p in (ROOT/'report/response-B').iterdir():
  if p.is_file():shutil.copyfile(p,OUT/p.name);bindings.append(bound(p))
 bindings += [bound(ROOT/'B-REF/SHAPES.json')]
 save_json(OUT/'metrics.json',dict(schema='point-quality-completion-v1',executed_source=freeze['source_revision'],bindings=bindings,accounting=account,roles=roles,response=response,normalization=normalization,scorecard=metrics,common_support=common,corrected_version_comparison=repeat,support=support,storage=storage,reduction_wall_seconds=sum(r['wall_seconds'] for r in roles),response_extraction_seconds=read(ROOT/'report/response-B/response-extraction.json')['seconds'],completion_evaluation_seconds=time.monotonic()-started,new_observation_starts_in_this_analysis=0,qualification='Input-amplitude gains use the signed analytic amplitude of the fixed raster family. Ideal-estimator gains are separately reported. These are finite10percent perturbations with shared data/calibration/context; sigma/g is conditional local finite-response uncertainty, not absolute calibration, unbiasedness, paired significance or a measured sensitivity advantage over R.'))
 print(json.dumps(dict(peaks=[r['T5']['guard']['accepted_parameters'][0] for r in roles],sigma_A=sigma,response=response,version=repeat,storage=storage),indent=2))
if __name__=='__main__':main()
