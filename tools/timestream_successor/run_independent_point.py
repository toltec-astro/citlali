"""Bounded ordinary-MAP POINT152430 a2000 frozen-method validation command."""
import argparse,json,os,shutil,subprocess,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import psutil
from point_independent import initialize,context,endpoint
from point_guarded_feedback import POLICIES,BRANCHES,infer,publish,measurement,continuation_reason,ideal_comparison
from point_expected import candidate
from point_response_cd import shapes,seed_candidate,processing_guard,injection_reference,response_metrics,direct_raster
from point_fruit_trajectory import bound,verified,selection_difference
from point_fruit_diagnostics import load,patch_map,learned_state
from point_empirical_io import write_fits
from point_empirical_io import exact_header
from point_empirical import source_patch
from point_additive_diagnostics import save_json
from point_fruit_trajectory import ready_state
from run_successor_observation import run as invoke
from inspect_forward_map import require,read_yaml,digest

from point_processing import call, cleanup_cycle, execute_branch, owner, parents, read, record_attempt, request, size, upstream

REPO=Path(__file__).resolve().parents[2]
_LOCK=threading.Lock();_RESERVED=0


def check(out,binary,review):
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'clean committed source required')
    r=read(review);require(r['disposition']=='pass' and r['source_revision']==rev and r['prospective_sha256']==digest(out/'freeze/prospective.json'),'exact source/prospective preflight review required')
    require(digest(binary)==r['binary_sha256'],'reviewed binary differs')
    require(rev[:9] in subprocess.check_output([str(binary),'--version'],text=True),'native/source revision mismatch')
    require(all(os.environ.get(n)=='1' for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']),'explicit one-thread numeric libraries required')
    prospective=read(out/'freeze/prospective.json')
    verified(prospective['dependencies']);verified(prospective['budget']);verified(prospective['generated_inputs'])
    for ref in [*read(out/'freeze/dependencies.json')['inputs'],*read(verified(prospective['generated_inputs']))]:verified(ref)
    for name in ('realized.json','shapes-manifest.json'):
        manifest=out/'freeze'/name
        if manifest.exists():
            for ref in read(manifest):verified(ref)















def initial_injections(out,binary):
    c=context(out/'freeze/a2000');base=read(out/'run/BASE-POINT/REF/a2000/SUMMARY.json')
    require(base['ready'] and base['completed_cycle']==10,'BASE reference failure blocks shapes')
    d=out/'freeze/shapes';d.mkdir();ref0=load(verified(base['histories'][0]['candidate']))[0]
    rasters,info=shapes(base['histories'][-1]['measurement']['fit']['parameters'],c['major'],c['header'],ref0['VALID'].astype(bool))
    for b,z in rasters.items():write_fits(d/(b+'.fits'),exact_header(c['header']),[('INJECTION',z,'mJy/nominal beam'),('VALID',ref0['VALID'],'1')],companion='POINT-INDEPENDENT-INJECTION')
    save_json(out/'SHAPES.json',dict(source=base['histories'][-1]['total'],M0_major=c['major'],shapes=info,rasters={b:bound(d/(b+'.fits')) for b in rasters},frozen_before_other_policy_endpoints=True))
    save_json(out/'freeze/shapes-manifest.json',[bound(out/'SHAPES.json'),*[bound(p) for p in sorted(d.glob('*.fits'))]])
    for branch in BRANCHES[1:4]:
        try:initial_injection(out,binary,branch,base,c)
        except (ValueError,RuntimeError,KeyError,OSError) as exc:
            save_json(out/'initial'/branch/'INITIAL.json',dict(available=False,reason=str(exc)))

def initial_control(reference,injected,raster,c,dest):
    """M0 control for the aligned containing-pixel, uniform-gamma MAP.

    No fit or PTC: identical grid makes normalized gridding of P(I) equal I
    at each supported output pixel. T1–T10 additionally audit native streams.
    """
    ref,h=load(reference);actual,ah=load(injected);z,zh=load(raster)
    require(all(h[k]==ah[k] for k in ('CRPIX1','CRPIX2','CDELT1','CDELT2','CRVAL1','CRVAL2')),'M0 control grid mismatch')
    require(np.array_equal(ref['HITS'],ref['NORMALIZATION']),'M0 nonuniform coefficient')
    known=direct_raster(z['INJECTION'],zh,ref['PRIMARY'].shape,h)
    common=ref['SCI_SUPPORT'].astype(bool)&actual['SCI_SUPPORT'].astype(bool)
    delta=actual['PRIMARY']-ref['PRIMARY'];known[~ref['SCI_SUPPORT'].astype(bool)]=np.nan
    ks,vs,_,_,_=source_patch(known,ref['SCI_SUPPORT'].astype(bool),ref['NORMALIZATION'],h)
    ds,dv,_,_,_=source_patch(delta,common,ref['NORMALIZATION'],h)
    write_fits(dest/'response.fits',h,[('DELTA',delta,'mJy/nominal beam'),('CONTROL',known,'mJy/nominal beam'),('RESIDUAL',delta-known,'mJy/nominal beam'),('VALID',common.astype('u1'),'1')],companion='POINT-INDEPENDENT-M0-CONTROL')
    return dict(audit=dict(definition='aligned containing-pixel ordinary MAP of injection on original M0 support; uniform gamma=1, no PTC/centering',REF_counts_normalization_exact=True,native_stream_audit='performed separately for every T1–T10 control'),full=response_metrics(delta,known,common),source_disk=response_metrics(ds,ks,c['fixed'].disk&dv&vs))


def initial_injection(out,binary,branch,base,c):
        dest=out/'initial'/branch;dest.mkdir(parents=True);op=dest/'owner.json';save_json(op,owner(out,'BASE-POINT',branch));q=dest/'request.json';save_json(q,request(out,op,'BASE-POINT',branch,0,verified(base['histories'][0]['candidate'])))
        require(call(out,binary,q,dest/'native')==0,'injected M0 native failure')
        products=dest/'native/products';receipt=read_yaml(products/'receipt.yaml');guards=processing_guard(products,receipt,allow_historical_failure=False)
        s,v,_,_=patch_map(products/'total.fits');m,_=infer(c,s,v);response=initial_control(out/'M0/products/a2000/map.fits',products/'total.fits',out/f'freeze/shapes/{branch}.fits',c,dest);save_json(dest/'INITIAL.json',dict(available=True,total=bound(products/'total.fits'),receipt=bound(products/'receipt.yaml'),measurement=m,guards=guards,response=response))
        cleanup_cycle(out,products,keep_projection=False,keep_injected=True)




def execute(out,binary,phase):
    require(read(out/'UPSTREAM.json')['status']=='accepted','initial source guard unavailable')
    record=out/('PHASE-'+phase+'.json');require(not record.exists(),'no implicit phase retry');start=time.perf_counter();tasks=[]
    if phase=='base':tasks=[('BASE-POINT','REF')];workers=1
    else:
        profile=read(out/'PHASE-base.json');require(profile['results'][0]['ready'] and profile['results'][0]['completed_cycle']==10,'failed baseline reference')
        if phase=='shapes':
            save_json(record,dict(status='running'))
            try:
                initial_injections(out,binary);save_json(record,dict(status='completed',seconds=time.perf_counter()-start))
            except (ValueError,RuntimeError,KeyError,OSError) as exc:save_json(record,dict(status='failed',reason=str(exc)))
            return
        require((out/'SHAPES.json').exists(),'shapes must be frozen before other policy endpoints or branches')
        if phase=='references':tasks=[('CORR','REF'),('DIAG','REF')]
        elif phase=='responses':
            for pol in POLICIES:
                ref=out/'run'/pol/'REF/a2000/SUMMARY.json'
                if ref.exists() and read(ref)['ready'] and read(ref)['completed_cycle']==10:tasks.extend((pol,b) for b in BRANCHES[1:])
        budget=read(out/'freeze/budget.json');rss=profile['peak_family_rss_bytes'];workers=min(len(tasks),12,max(1,os.cpu_count()-2),max(1,int(psutil.virtual_memory().available//max(2*rss,1))))
    save_json(record,dict(status='running',tasks=tasks,workers=workers))
    stop=threading.Event();samples=[]
    def monitor():
        p=psutil.Process()
        while not stop.is_set():
            total=0
            for child in [p,*p.children(recursive=True)]:
                try:total+=child.memory_info().rss
                except psutil.Error:pass
            samples.append(total);stop.wait(.5)
    t=threading.Thread(target=monitor,daemon=True);t.start();results=[]
    try:
        with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
            futures={pool.submit(execute_branch,out,binary,*q):q for q in tasks}
            for f in as_completed(futures):
                try:r=f.result();results.append({k:r[k] for k in ('policy','branch','ready','completed_cycle','stop_reason','actual_feedback_applications','wall_seconds')})
                except Exception as e:results.append(dict(policy=futures[f][0],branch=futures[f][1],ready=False,stop_reason=str(e)))
    finally:stop.set();t.join()
    save_json(record,dict(status='completed',results=results,workers=workers,numeric_threads=1,seconds=time.perf_counter()-start,peak_family_rss_bytes=max(samples,default=0),final_bytes=size(out)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['upstream','base','shapes','references','responses']);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();check(a.output,a.binary,a.review)
    if a.stage=='upstream':
        try:upstream(a.output,a.binary)
        except (ValueError,RuntimeError,KeyError,OSError) as exc:
            if not (a.output/'UPSTREAM.json').exists():save_json(a.output/'UPSTREAM.json',dict(status='failed',reason=str(exc)))
            raise
    else:execute(a.output,a.binary,a.stage)
