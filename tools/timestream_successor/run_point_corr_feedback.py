"""Gated native recurrence and saved-map selection using frozen CORR/DIAG policy."""
import argparse,json,shutil,time,threading,resource
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import psutil
from run_point_corr_admission import context,read,REPO,ARRAYS,SNAPSHOTS
from run_point_response_cd import source,replay_check
from point_fruit_trajectory import bound,verified,manifest,verify_manifest,candidate_guard,selection_difference
from point_fruit_diagnostics import load,patch_map,learned_state
from point_response_cd import processing_guard,seed_candidate,save_candidate
from point_corr_admission import infer,select,POLICIES
from point_additive_diagnostics import save_json
from point_empirical_io import write_fits
from point_jinc_campaign import exact_header
from run_point_fruit_ten import ready_state
from run_successor_observation import run as invoke
from inspect_forward_map import require,read_yaml,digest

BRANCHES=('REF','C-core-plus','C-core-minus','C-shoulder','D-seed-shoulder')


def publish_policy(dest,c,signal,support,fit,policy,tau,prepared=None):
 rawfit,base,fields=infer(signal,support,c['fixed'],c['neighborhood'],fit=fit) if prepared is None else prepared
 result=base if policy=='BASE' else None if tau is None else select(base,fields,c['k0'],policy,tau)
 summary=dict(fit=rawfit,BASE=None if base is None else base['summary'],candidate=None if result is None else result['summary'],unavailable_cause='calibrated threshold unavailable' if policy!='BASE' and tau is None else None)
 save_json(dest/'admission.json',summary)
 if base is None:return None,summary
 header=exact_header(c['header']);path=None
 if result is not None:
  path=dest/'candidate.fits';save_candidate(path,header,result)
 planes=[('BASE',base['selected'].astype('u1'),'1'),('K0',c['k0'].astype('u1'),'1'),('BRIGHTNESS',base['brightness'],'mJy/nominal beam'),('VALID',base['valid'].astype('u1'),'1')]
 if fields is not None:
  planes.extend([('NEFF_W',fields['neff_weights'],'1'),('EVIDENCE',fields['evidence'],'mJy/nominal beam')])
  for pol in POLICIES:
   f=fields['policies'][pol];planes.extend([(pol+'_SCORE',f['score'],'1'),(pol+'_VAR',f['variance'],'(mJy/nominal beam)^2'),(pol+'_VALID',f['available'].astype('u1'),'1')])
 if result is not None:planes.append(('ADMITTED',result['selected'].astype('u1'),'1'))
 if result is not None and policy!='BASE':planes.extend([('PROPOSED',result['proposed_extra'].astype('u1'),'1'),('ADDED',result['added'].astype('u1'),'1')])
 write_fits(dest/'admission.fits',header,planes,companion='POINT-CORR-DIAGNOSTIC')
 return path,summary


def screen(out):
 rev=source();p=read(out/'freeze/prospective.json');ts=read(out/'calibration-disposition/thresholds.json')['thresholds'];dest=out/'screen';dest.mkdir();rows=[]
 for a in ARRAYS:
  c=context(out,a)
  for k in SNAPSHOTS:
   state=p['original_owner']['original_cycles'][a][k];signal,support,_,_=patch_map(verified(state['total']));fit=state['measurement']['fit']
   prepared=infer(signal,support,c['fixed'],c['neighborhood'],fit=fit)
   for pol in ('BASE',*POLICIES):
    d=dest/f'{a}-{k:02d}-{pol}';d.mkdir();path,summary=publish_policy(d,c,signal,support,fit,pol,ts[a].get(pol),prepared)
    if pol=='BASE':
     old,_=load(verified(state['candidate']));new,_=load(path)
     for n in ('CANDIDATE','ADMITTED','VALID','RATIO'):require(np.array_equal(old[n],new[n],equal_nan=True),'saved screen BASE replay mismatch')
    rows.append(dict(array=a,stage=k,policy=pol,total=state['total'],candidate=bound(path) if path else None,summary=summary))
 save_json(dest/'RESULTS.json',dict(source_revision=rev,rows=rows,applied=False));manifest(dest)


def history_source(p,a,branch):
 from point_frozen_shoulder import CD
 path=CD/'run'/branch/a/'SUMMARY.json'
 require(str(path) in p['consumed_inputs'],'historical summary not frozen')
 from inspect_forward_map import digest
 require(digest(path)==p['consumed_inputs'][str(path)],'historical summary changed')
 return read(path)


def upstream_replay(original,products,receipt):
 expected=read_yaml(verified(original['receipt']));require(len(expected['segments'])==len(receipt['segments']),'BASE segment count changed')
 records=[]
 for old,new in zip(expected['segments'],receipt['segments']):
  for name in ('source_segment','source_receipt_sha256','projection_support_counts','rank','converged','stopping_reason','iterations','objective','retained','application_failed_times','input_eligibility_changed','output_cause_changed'):
   require(old[name]==new[name],'BASE same-stage PTC outcome differs: '+name)
  source=new['source_segment'];stem=next(n[:-len('-input.f64')] for n in source['files'] if n.endswith('-input.f64'));directory=products/f"network-{source['network']}-{stem}"
  for name in ('operand.f64','model-support.u8','mean.f64','basis.f64','cleaned.f64','causes.u8'):
   require(digest(directory/name)==new['files'][name]==old['files'][name],'BASE same-stage numerical state differs: '+name)
  records.append(dict(network=source['network'],scan=source['scan'],exact_files=6,exact_outcomes=True))
 return dict(archived_same_stage_receipt=original['receipt'],segments=records,all_numerical_files_exact=True)


def execute_branch(out,binary,p,a,pol,branch):
 start=time.perf_counter();c=context(out,a);dest=out/'run'/pol/branch/a;dest.mkdir(parents=True);tau=read(out/'calibration-disposition/thresholds.json')['thresholds'][a].get(pol)
 original=p['original_owner'];initial=history_source(p,a,branch if branch.startswith('C-') else 'REF')['histories'][0]
 init=dest/'initial';init.mkdir();sig,sup,_,_=patch_map(verified(initial['total']));measurement=initial['measurement']
 path,summary=publish_policy(init,c,sig,sup,measurement['fit'],pol,tau)
 require(path is not None,'initial candidate unavailable')
 if branch=='D-seed-shoulder':
  ownref=read(out/'run'/pol/'REF'/a/'SUMMARY.json')['histories'][0];own,_=load(verified(ownref['candidate']));shoulder,_=load(verified(original['injections'][a]['C-shoulder']))
  seeded=seed_candidate(own,'D-seed-shoulder',shoulder['INJECTION']);path=init/'seed.fits'
  write_fits(path,exact_header(c['header']),[(n,seeded[n],'mJy/nominal beam' if n=='CANDIDATE' else '1') for n in ('CANDIDATE','ADMITTED','VALID','RATIO')],companion='POINT-CORR-SEED')
 candidate_guard(load(path)[0],c['header'])
 owner=dict(identity='POINT-152389-CORR-NEIGHBOR-OWNER-2026-09-24',outside_disk_no_feedback=True,direct_identical_operand_rejoin=True,gain=1,nonzero_cycles=10,policy=pol,prospective=bound(out/'freeze/prospective.json'),parents=original['parents'],injections=original['injections'],seeds={a:{branch:bound(path)}})
 if pol!='BASE':owner['launch_gate']=bound(out/'validation/gates.json')
 op=dest/'owner.json';save_json(op,owner)
 prior=ready_state(a,0,path,verified(initial['total']),verified(initial['receipt']),original['parents'][a],True)
 prior.update(branch=branch,owner_sha256=bound(op)['sha256'],measurement=measurement,reused_M0=True,candidate_summary=summary)
 sp=init/'state.json';save_json(sp,prior);prior['state_path']=str(sp);history=[prior]
 for k in range(1,11):
  if not prior['ready']:break
  d=dest/f'cycle-{k:02d}';d.mkdir();tic=time.perf_counter();rec=dict(array=a,policy=pol,branch=branch,cycle_index=k,ready=False)
  request=dict(schema='citlali-point-fruit-corr-neighbor-v1',method='point-152389-corr-neighbor-v1',mode='reference',array=a,branch=branch,cycle_index=k,candidate=prior['candidate'],owner_binding=bound(op),previous_state=bound(Path(prior['state_path'])),**original['parents'][a])
  if branch.startswith('C-'):request['injection']=original['injections'][a][branch]
  save_json(d/'request.json',request)
  try:
   require(invoke(binary,d/'request.json',d/'native',1,'point-fruit')==0,'native cycle failed')
   products=d/'native/products';receipt=read_yaml(products/'receipt.yaml');signal,support,_,_=patch_map(products/'total.fits');fitstart=time.perf_counter();measurement=c['fixed'].measure(signal,support)
   path,summary=publish_policy(d,c,signal,support,measurement['fit'],pol,tau)
   rec.update(total=bound(products/'total.fits'),receipt=bound(products/'receipt.yaml'),measurement=measurement,admission=summary,guards=processing_guard(products,receipt),PTC_segments=learned_state(products,receipt))
   require(path is not None and summary['candidate']['status']=='retained','unavailable source candidate')
   candidate_guard(load(path)[0],c['header']);rec['candidate']=bound(path)
   if pol=='BASE':
    rec['replay']=replay_check(original['original_cycles'][a][k],products,measurement,path,c['beam'],reuse_fits=True)
    archived=history_source(p,a,'REF')['histories'][k];rec['upstream_replay']=upstream_replay(archived,products,receipt)
   old,_=load(verified(prior['candidate']));new,_=load(path);rec['candidate_turnover']=selection_difference(new,old)
   restore,rv,_,_=patch_map(products/'restored.fits');use=rv&old['VALID'].astype(bool);err=float(np.max(abs((restore-old['CANDIDATE'])[use])));require(err<1e-8,'applied operand remap mismatch');rec['guards']['operand_remap_max_abs']=err
   state=ready_state(a,k,path,products/'total.fits',products/'receipt.yaml',original['parents'][a],True);state.update(branch=branch,owner_sha256=bound(op)['sha256']);sp=d/'state.json';save_json(sp,state)
   rec.update(ready=True,state_path=str(sp),diagnostic_seconds=time.perf_counter()-fitstart)
  except (ValueError,RuntimeError,KeyError,OSError) as exc:rec['stop_reason']=str(exc)
  rec['wall_seconds']=time.perf_counter()-tic;save_json(d/'cycle.json',rec);history.append(rec);prior=rec
  print(json.dumps(dict(array=a,policy=pol,branch=branch,cycle=k,ready=rec['ready'],reason=rec.get('stop_reason'))),flush=True)
 result=dict(array=a,policy=pol,branch=branch,ready=history[-1]['ready'],completed_cycle=history[-1]['cycle_index'],histories=history,wall_seconds=time.perf_counter()-start,m10_applied=False)
 save_json(dest/'SUMMARY.json',result);manifest(dest);return result


def execute(out,phase,review):
 revision=source();p=read(out/'freeze/prospective.json');rv=read(review);require(rv['source_revision']==revision and rv['disposition']=='pass','reviewed execution source required');binary=verified(p['binary'])
 # The exact frozen binary is retained; later Python-only reviewed orchestration
 # is explicitly bound separately, never mislabeled as the native build source.
 gates=read(out/'validation/gates.json')['policies'];tasks=[];skipped=[]
 for a in ARRAYS:
  for pol in (('BASE',) if phase=='BASE' else POLICIES):
   for branch in (('REF',) if phase in ('BASE','REF') else BRANCHES[1:]):
    cause=None
    if pol!='BASE' and not gates[a][pol]['pass']:cause='policy null/numerical launch gate failed'
    if pol!='BASE':
     base=out/'run/BASE/REF'/a/'SUMMARY.json'
     if not base.exists() or not read(base)['ready'] or read(base)['completed_cycle']!=10:cause='BASE REF replay unavailable'
    if phase=='branches' and cause is None:
     ref=out/'run'/pol/'REF'/a/'SUMMARY.json'
     if not ref.exists() or not read(ref)['ready'] or read(ref)['completed_cycle']!=10:cause='own policy REF unavailable'
    if cause:skipped.append(dict(array=a,policy=pol,branch=branch,cause=cause))
    else:tasks.append((a,pol,branch))
 start=time.perf_counter();results=[];workers=min(p['workers'],max(1,len(tasks)));stop=threading.Event();rss=[]
 def monitor():
  parent=psutil.Process()
  while not stop.is_set():
   total=0
   for child in [parent,*parent.children(recursive=True)]:
    try:total+=child.memory_info().rss
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   rss.append(total);stop.wait(.25)
 watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(execute_branch,out,binary,p,*task) for task in tasks]
  for future in as_completed(futures):results.append(future.result())
 stop.set();watcher.join()
 save_json(out/('FEEDBACK-'+phase+'.json'),dict(source_revision=revision,native_binary=p['binary'],native_source_revision=p['source_revision'],review=bound(review),results=[{k:r[k] for k in ('array','policy','branch','ready','completed_cycle','wall_seconds')} for r in results],not_run=skipped,workers=workers,threads_per_native=1,requested_library_threads=1,actual_library_threads='Apple Accelerate internal count not observable',sampled_aggregate_RSS_bytes=max(rss,default=0),rss_sampling_seconds=.25,wall_seconds=time.perf_counter()-start))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen','BASE','REF','branches']);p.add_argument('--output',required=True,type=Path);p.add_argument('--review',type=Path);a=p.parse_args()
 if a.stage=='screen':screen(a.output)
 else:execute(a.output,a.stage,a.review)
