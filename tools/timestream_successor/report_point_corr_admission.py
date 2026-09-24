"""Report the fixed admission experiment, including unavailable failed-gate cells."""
import argparse,time,json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from run_point_corr_admission import read,context,ARRAYS,source
from point_corr_admission import infer,maximum,POLICIES,trial_seed
from point_additive import draw_noise
from point_fruit_trajectory import bound,verified,manifest,verify_manifest
from point_fruit_diagnostics import load
from point_additive_diagnostics import save_json
from inspect_forward_map import require


def fixed_checks(out):
 results={}
 for a in ARRAYS:
  c=context(out,a);n=c['neighborhood'];base=dict(selected=c['k0'],valid=c['fixed'].disk&c['support'],brightness=np.zeros(c['support'].shape));fields=n.scores(base)
  loc=np.argwhere(fields['policies']['CORR']['available']);r,col=loc[np.argmin(np.hypot(n.x[loc[:,0],loc[:,1]],n.y[loc[:,0],loc[:,1]]))]
  ix=np.flatnonzero(n.anchor_indices==r*c['support'].shape[1]+col)[0];indices=n.indices[ix]
  w=n.weights*(~c['k0'].ravel()[indices]);w[n.center]=0;q=w/w.sum();u=q*c['scale'].ravel()[indices]
  direct=float(sum(u[i]*n.rho[i,j]*u[j] for i in range(len(u)) for j in range(len(u))))
  analytic=fields['policies']['CORR']['variance'][r,col];require(abs(direct-analytic)<=1e-11*analytic,'actual dense qCq mismatch')
  rows=read(out/'validation'/(a+'-trials.json'));values=[];checks=[]
  for i,row in enumerate(rows):
   noise=draw_noise(c['psd'],c['scale'],trial_seed(a,'validation',i));values.append(float(np.sum(q*noise.ravel()[indices])))
   if i in (0,1,255):
    again=draw_noise(c['psd'],c['scale'],int(row['seed']));require(np.array_equal(again,noise),'noise regeneration differs')
    for mult,key in ((1.,'scenes'),(1.25,'stress')):
     for name,record in row[key].items():
      data=np.where(c['support'],c[name]+mult*noise,np.nan);fit,b,f=infer(data,c['support'],c['fixed'],n,fit=record['fit'])
      for pol in POLICIES:
       got=maximum(b,f,pol);expected=record['policies'][pol]['max_score']
       require((got is None and expected is None) or (got is not None and expected is not None and abs(got-expected)<1e-12),'stored complete-procedure score regeneration differs')
      checks.append(dict(trial=i,scene=name,multiplier=mult,score_reproduced=True,fit_reused=True))
  ratio=float(np.var(values,ddof=1)/analytic);limits=chi2.ppf([.0005,.9995],1023)/1023
  require(limits[0]<=ratio<=limits[1],'actual fixed-row generation variance check failed')
  results[a]=dict(anchor_row_column=[int(r),int(col)],analytic_variance=float(analytic),direct_pair_variance=direct,regenerated_validation_sample_variance=float(np.var(values,ddof=1)),variance_ratio=ratio,MC999_interval=limits,score_checks=checks,independent_new_draws=0,scope='existing validation seeds regenerated; fixed test row, no new fit or policy/covariance change')
 return results


def scene_summary(rows,name,key='scenes'):
 allscenes=[r[key][name] for r in rows if key in r];good=[s for s in allscenes if s['numerically_valid']];failure=len(allscenes)-len(good)
 return dict(count=len(allscenes),failures=failure,constrained=sum(s['fit'].get('constrained',False) for s in good),candidate_absent=sum(s['candidate_status']!='retained' for s in good),known_outcomes=len(good),BASE_false_events=sum(s['BASE_false_pixels']>0 for s in good),BASE_false_pixels_mean=float(np.mean([s['BASE_false_pixels'] for s in good])) if good else None,BASE_false_brightness_area_mean=float(np.mean([s['BASE_false_brightness_area'] for s in good])) if good else None,scope='all outcomes' if not failure else 'partial valid outcomes only; failed raw zero placeholders excluded')


def render(out):
 start=time.perf_counter();revision=source();dest=out/'report';dest.mkdir();p=read(out/'freeze/prospective.json');gate=read(out/'validation/gates.json');checks=fixed_checks(out);save_json(dest/'CHECKS.json',checks);stats={};figs=[]
 plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.dpi':140})
 def save(fig,name):
  path=dest/name;fig.savefig(path,bbox_inches='tight',pad_inches=.15);plt.close(fig);figs.append(bound(path))
 fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
 for row,a in enumerate(ARRAYS):
  cal=read(out/'calibration'/(a+'-trials.json'));val=read(out/'validation'/(a+'-trials.json'));stats[a]={}
  for phase,rows,key in [('calibration',cal,'scenes'),('validation',val,'scenes'),('stress1.25',val,'stress')]:
   stats[a][phase]={scene:scene_summary(rows,scene,key) for scene in ('blank','core')}
  for column,name in enumerate(('blank','core')):
   ax=axes[row,column];xx=np.arange(3);records=[stats[a][phase][name] for phase in stats[a]]
   fail=np.array([r['failures']/r['count']*100 for r in records]);con=np.array([r['constrained']/r['count']*100 for r in records])
   ax.bar(xx,fail,color='#b44132',label='Solver failure: unknown');ax.bar(xx,con,bottom=fail,color='#d9ab50',label='Finite constrained: no candidate');ax.bar(xx,100-fail-con,bottom=fail+con,color='#447a92',label='Other valid outcome')
   ax.set(xticks=xx,xticklabels=['Calibration\n1024','Validation\n1024','Stress\n256 reused'],ylim=(0,105),ylabel='Percent of prescribed outcomes',title=f'{a} · {name}')
   for x,r in zip(xx,records):ax.text(x,101,f"{r['failures']} failed",ha='center',fontsize=8)
 axes[0,0].legend(fontsize=8,loc='center left');fig.suptitle('Complete null outcomes · failures are neither absence nor empty searches');save(fig,'null-outcomes.png')
 # No invalid threshold line is displayed. These are conditional scene-score distributions.
 fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
 for row,a in enumerate(ARRAYS):
  rows=read(out/'validation'/(a+'-trials.json'))
  for col,name in enumerate(('blank','core')):
   ax=axes[row,col]
   for pol,color in [('CORR','#315b89'),('DIAG','#b74b2c')]:
    vals=[s['policies'][pol]['max_score'] for r in rows for s in [r['scenes'][name]] if s['numerically_valid'] and s['policies'][pol]['max_score'] is not None]
    ax.hist(vals,bins=25,histtype='step',color=color,label=f'{pol}: {len(vals)} finite maxima')
   ax.set(title=f'{a} · {name}',xlabel='Nominal scene search maximum',ylabel='Count');ax.legend(fontsize=8)
 fig.suptitle('Validation scene scores · finite searches only; no calibrated threshold');save(fig,'nominal-score-distributions.png')
 # Same-map real diagnostics: all policy masks unavailable, BASE is explicit.
 for a in ARRAYS:
  c=context(out,a);fig,axes=plt.subplots(3,5,figsize=(16,9),layout='constrained');extent=[41,-41,-41,41]
  for col,k in enumerate((0,1,3,5,10)):
   images,_=load(out/'screen'/f'{a}-{k:02d}-CORR'/'admission.fits');b=images['BRIGHTNESS'];peak=abs(p['original_owner']['original_cycles'][a][10]['measurement']['fit']['parameters'][0]);lim=.06*peak
   ax=axes[0,col];im=ax.imshow(b,origin='lower',extent=extent,cmap='RdBu_r',norm=TwoSlopeNorm(0,vmin=-lim,vmax=lim));ax.contour(images['BASE'],levels=[.5],colors=['black'],linewidths=1,origin='lower',extent=extent);ax.contour(images['K0'],levels=[.5],colors=['limegreen'],linewidths=.8,origin='lower',extent=extent);ax.set_title(f'T{k}: measured brightness');fig.colorbar(im,ax=ax,shrink=.65,label='mJy/nominal beam')
   for rr,pol in ((1,'CORR'),(2,'DIAG')):
    im=axes[rr,col].imshow(images[pol+'_SCORE'],origin='lower',extent=extent,cmap='RdBu_r',vmin=-5,vmax=5);fig.colorbar(im,ax=axes[rr,col],shrink=.65,label='Nominal Z');axes[rr,col].set_title(f'{pol}: threshold unavailable')
  for ax in axes.flat:ax.set(xlabel='East offset (arcsec)',ylabel='North offset (arcsec)')
  fig.suptitle(f'{a}: same-map evidence · black BASE, green fixed K0 · gray/blank = no test\nBrightness clipped at ±6% original T10 peak; score display clipped at ±5, no admission implied');save(fig,a+'-saved-evidence.png')
 # Numerical geometry/variance and original M0 noise model.
 fig,axes=plt.subplots(3,3,figsize=(11,10),layout='constrained')
 for row,a in enumerate(ARRAYS):
  images,_=load(out/'screen'/f'{a}-00-CORR'/'admission.fits');ratio=np.divide(images['CORR_VAR'],images['DIAG_VAR'],out=np.full((41,41),np.nan),where=images['CORR_VALID'].astype(bool)&images['DIAG_VALID'].astype(bool));c=context(out,a)
  for col,(data,title,cm) in enumerate([(images['CORR_VALID'],a+' available supplemental tests','gray'),(ratio,'CORR / DIAG variance','viridis'),(np.where(c['support']&(c['scale']>0),c['scale'],np.nan),'Frozen M0 amplitude (mJy/beam)','viridis')]):
   im=axes[row,col].imshow(data,origin='lower',extent=[41,-41,-41,41],cmap=cm);axes[row,col].set(title=title,xlabel='East (arcsec)',ylabel='North (arcsec)');fig.colorbar(im,ax=axes[row,col],shrink=.75)
 fig.suptitle('Numerically supported original-M0 covariance · full footprints precede core/center exclusions');save(fig,'covariance-and-support.png')
 save_json(dest/'NULL_SUMMARY.json',stats)
 # Realized exact BASE replays and explicit failed-gate matrix.
 matrix=[];replays={};fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained')
 for row,a in enumerate(ARRAYS):
  rp=out/'run/BASE/REF'/a/'SUMMARY.json';s=read(rp);replays[a]=s;h=s['histories'];original=p['original_owner']['original_cycles'][a]
  matrix.append(dict(array=a,policy='BASE',branch='REF',status='completed' if s['ready'] and s['completed_cycle']==10 else 'unavailable',last_stage=s['completed_cycle']))
  for pol in POLICIES:
   for branch in ('REF','C-core-plus','C-core-minus','C-shoulder','D-seed-shoulder'):matrix.append(dict(array=a,policy=pol,branch=branch,status='gate-blocked',cause='calibration has unknown joint maxima; no calibrated threshold'))
  k=[r['cycle_index'] for r in h];new=[r['measurement']['fit']['parameters'][0] for r in h];old=[original[j]['measurement']['fit']['parameters'][0] for j in k]
  axes[row,0].plot(k,old,label='Original BASE',color='black');axes[row,0].plot(k,new,'o',fillstyle='none',label='Current replay',color='#39729b');axes[row,0].set(title=a,xlabel='Feedback stage',ylabel='Fitted peak (mJy/nominal beam)');axes[row,0].legend()
  delta=np.array(new)-old;axes[row,1].plot(k,delta,'o-');axes[row,1].set(title='Replay − original',xlabel='Feedback stage',ylabel='Peak difference (mJy/nominal beam)',ylim=(-1e-8,1e-8));axes[row,1].axhline(0,color='gray',lw=.7)
 fig.suptitle('BASE controls only · no CORR/DIAG trajectory, endpoint, response or seed-persistence comparison');save(fig,'BASE-replay.png')
 save_json(dest/'MATRIX.json',dict(trajectories=matrix,calibration_fields=3072,validation_fields=3072,stress_reuses=768,new_policy_feedback_applications=0,BASE_feedback_applications=sum(len(s['histories'])-1 for s in replays.values()),endpoint_covariance='not run: no completed new-policy REF',compact_shoulder_seed_response='unavailable for new policies: failed launch gates',saved_map_screen='15 exact BASE screens;30 nominal policy evidence screens, thresholded masks unavailable'))
 lines=['# POINT 152389 correlation-aware admission: failed-gate result','','The numerical covariance and selection implementation pass their checks, but the prescribed complete null procedure cannot calibrate a threshold. Blank-scene descriptive fits fail; their search maxima are unknown. No CORR or DIAG feedback trajectory was launched.','','| Array | Calibration blank failures /1024 | Validation blank failures /1024 | Stress blank failures /256 |','|---|---:|---:|---:|']
 for a in ARRAYS:lines.append(f"| {a} | {stats[a]['calibration']['blank']['failures']} | {stats[a]['validation']['blank']['failures']} | {stats[a]['stress1.25']['blank']['failures']} |")
 lines.extend(['','All core-scene outcomes are numerically valid, including finite constrained fits whose inherited candidate is absent. Legitimate no-positive-central-peak outcomes remain separate from solver nonconvergence. The prescribed field counts are complete; stress reuses existing fields.','','All six calibrated thresholds, binomial upper bounds, supplemental false-admission rates, extra-brightness biases and real supplemental masks are **unavailable**, not zero. The original intermediate thresholds remain preserved as invalid diagnostics. No failed draw was replaced and no solver or science selection was retuned.','','## What is and is not learned','','The fixed K0/current BASE/anchor exclusions, original measured brightness publication, full footprints, covariance variances and deterministic generation pass numerical tests. Same-map figures expose the nominal evidence without implying admission. Since the common source-measurement procedure blocks calibration, this experiment cannot compare CORR with DIAG or measure new-policy compact/shoulder response, seed persistence, endpoint background, terminal drift or measurement precision. A larger nominal score does not establish a useful admitted region.','','The three BASE M0–T10 controls replay the original maps, measurements, candidate values/masks, applied operands and same-stage PTC state under the unchanged recipe. Their response and history remain the original reference; they are not new-policy results.','','## Partial BASE null diagnostics','','Only numerically valid outcomes enter these averages. Blank rows are partial populations; failed placeholders are excluded. Units of brightness-area are mJy/nominal beam × arcsec². These describe inherited BASE false support, not calibrated supplemental false admission.','','| Array | Phase | Scene | Valid / prescribed | BASE false events | Mean false pixels | Mean false brightness-area |','|---|---|---|---:|---:|---:|---:|'])
 for a in ARRAYS:
  for phase,scenes in stats[a].items():
   for name,r in scenes.items():lines.append(f"| {a} | {phase} | {name} | {r['known_outcomes']}/{r['count']} | {r['BASE_false_events']} | {r['BASE_false_pixels_mean']:.4g} | {r['BASE_false_brightness_area_mean']:.5g} |")
 lines.extend(['','## Cost and execution scope','',f"Preparation: {p['wall_seconds']:.2f} s wall, including {p['authentication_seconds']:.2f} s initial authentication and {p['final_authentication_seconds']:.2f} s repeated original verification.",'', '| Phase | Wall seconds | Summed trial CPU seconds | Maximum worker high-water MiB |','|---|---:|---:|---:|'])
 for phase in ('CALIBRATION','VALIDATION'):
  rr=read(out/(phase+'.json'));lines.append(f"| {phase} | {rr['wall_seconds']:.2f} | {rr['CPU_seconds']:.2f} | {rr['maximum_worker_highwater_bytes']/2**20:.2f} |")
 fr=read(out/'FEEDBACK-BASE.json');lines.extend(['',f"BASE replay: {fr['wall_seconds']:.2f} s wall, {fr['workers']} concurrent native jobs, sampled aggregate RSS {fr['sampled_aggregate_RSS_bytes']/2**30:.3f} GiB.",'Twelve independent simulation workers; one library thread requested. Apple Accelerate internal thread counts are not observable. High-water and aggregate RSS are distinct measurements; CPU totals are not elapsed time. All execution was local in the existing environment, not Unity/Spack replication.'])
 lines.extend(['','## One next action, not executed','','Recommend a bounded investigation of the inherited POINT solver’s blank-scene failures and a separately reviewed source-absence/candidate-outcome rule before repeating admission calibration. This is not permission to relabel current nonconvergence as absence, change the solver here, or choose a threshold from survivors.','','The original a1100 network-3/scan-5 and a2000 background/source-transfer qualifications remain. C0 is the original A+B M0 model and remains frozen; conditional simulations do not validate observed source recovery, astronomical significance, or a ten-cycle false-feedback rate. MAP coefficients, source-fit weights, calibration, feedback gain and production defaults were not changed.','','## Diagnostic figures',''])
 for f in figs:lines.append(f"![{Path(f['path']).stem}]({f['path']})\n")
 lines.extend([f"[Every trajectory and unavailable product]({dest/'MATRIX.json'})",f"[Detailed null outcomes]({dest/'NULL_SUMMARY.json'})",f"[Actual covariance and seed-regeneration checks]({dest/'CHECKS.json'})"])
 (dest/'ASSESSMENT.md').write_text('\n'.join(lines)+'\n');save_json(dest/'REPORT.json',dict(source_revision=revision,figures=figs,wall_seconds=time.perf_counter()-start,checks=bound(dest/'CHECKS.json'),matrix=bound(dest/'MATRIX.json')));manifest(dest)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.output)
