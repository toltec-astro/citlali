"""Bounded saved-M0 frozen-basis controls; never an independently learned reduction.

The original RTC/AST and rank-10 basis stay fixed. Means and masked coefficients
are evaluated on each role's samples. Historical conversion precedes centering
and projection. All full-profile controls count against the new trajectory budget.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-amplitude-mpl',XDG_CACHE_HOME='/private/tmp/point-amplitude-cache')
from pathlib import Path
import argparse,fcntl,json,re,shutil,subprocess,sys,time,resource
import numpy as np
from astropy.io import fits
import scipy,astropy
from calibration_audit import D,R,H,S,HST,yread,array,bound,digest,save,tables
W=D.parents[1];sys.path.insert(0,str(W/'tools/timestream_successor'))
from inspect_point_source import solve_gaussian
from point_storage import tree_bytes
from point_transfer_experiment import PRIOR,ROOT as OLD_ROOT
ROOT=R.parents[1]/'successor-point152420-amplitude-20260930'
ROLES=('M0-replay','common-current','common-historical-conversion')
RESERVATION=100_000_000

def require(ok,message):
 if not ok:raise RuntimeError(message)

def apply_basis(values,eligible,basis):
 """Same observed-entry centering and rank cutoff as native ptc_apply.
 Excluded cells never enter arithmetic. Rank-deficient time patterns unavailable.
 """
 require(values.shape==eligible.shape and basis.shape[0]==values.shape[1],'PTC shapes')
 require(np.isfinite(values[eligible]).all() and np.isfinite(basis).all(),'PTC finite payload')
 counts=eligible.sum(axis=0);mean=np.divide(np.sum(np.where(eligible,values,0),axis=0,dtype=np.longdouble),counts,out=np.zeros(len(counts),np.longdouble),where=counts>0).astype(float)
 clean=np.zeros_like(values);valid=np.zeros_like(eligible);patterns,inverse=np.unique(eligible,axis=0,return_inverse=True);fail=0
 for i,mask in enumerate(patterns):
  times=np.flatnonzero(inverse==i);cols=np.flatnonzero(mask)
  if len(cols)==0:continue
  b=basis[cols];e,u=np.linalg.eigh(np.einsum('ji,jk->ik',b,b));cut=max(0.,e[-1])*1e-10
  if np.count_nonzero(e>cut)!=basis.shape[1]:fail+=len(times);continue
  projector=np.einsum('ij,jk->ik',np.einsum('ij,kj->ik',u*(1/e),u),b.T)
  z=values[np.ix_(times,cols)]-mean[cols]
  coeff=np.einsum('ij,kj->ik',z,projector)
  clean[np.ix_(times,cols)]=z-np.einsum('ij,kj->ik',coeff,b);valid[np.ix_(times,cols)]=True
 require(np.isfinite(clean[valid]).all(),'nonfinite PTC output')
 return clean,valid,dict(rank_failed_times=fail,eligible=int(eligible.sum()),retained=int(valid.sum()))

def normalization(count,cut=.5):
 positive=np.sort(count[count>0]);require(len(positive)>0,'empty map')
 ref=positive[(3*len(positive)//4+len(positive))//2]
 return (count>0)&(count>=cut*ref),float(ref)

def segments():
 plan=yread(R/'M0/products/a1100/plan.yaml');receipts={}
 require(plan['observation']=='obs:152420_0_2' and plan['array']=='a1100' and plan['coverage_cut']==.5,'parent MAP scope')
 for col in plan['ordered_column_domain']:
  nw,sid=map(int,re.search(r'network-(\d+)-segment-(\d+)',col['publication_file']).groups());p=R/f'upstream/products/network{nw}/donor-continuity/ptc'
  if nw not in receipts:receipts[nw]=yread(p/'receipt.yaml')
  rr=receipts[nw];require(rr['rank']==10 and float(rr['relative_rank_tolerance'])==1e-10,'parent PTC rank policy')
  s=next(s for s in rr['segments'] if s['scan']==sid)
  yield nw,sid,p,s,col

def checkpoint(sid,p,s):
 nt,nd,no=s['scheduled_times'],s['detectors'],s['output_detectors'];stem=p/f'segment-{sid}'
 a=lambda name,dtype,shape:array(Path(str(stem)+'-'+name),dtype,shape)
 return dict(y=a('input.f64','<f8',(nt,nd)),eligible=a('eligible.u8','u1',(nt,nd)).astype(bool),basis=a('basis.f64','<f8',(nd,10)),native=a('cleaned.f64','<f8',(nt,no)),causes=a('causes.u8','u1',(nt,no)),slots=a('slots.i64','<i8',(nt,)),cols=np.array(s['fit_columns_in_output']),ids=np.array(s['detector_indices']),shape=(nt,no))

def capacity():
 paths=PRIOR+[OLD_ROOT];holdings=[dict(path=str(p),bytes=tree_bytes(p)) for p in paths]
 used=sum(x['bytes'] for x in holdings)+tree_bytes(ROOT);free=shutil.disk_usage(ROOT.parent).free
 require(used+RESERVATION<=30_000_000_000 and free>=RESERVATION+2_000_000_000,'shared storage reservation')
 return dict(holdings=holdings,used_bytes=used,reservation_bytes=RESERVATION,ceiling_bytes=30_000_000_000,free_bytes=free,free_reserve_bytes=2_000_000_000)

def inputs():
 files={H,S,HST,D/'OWNER_DIRECTIVE.md',D/'BATCH1.md',D/'results/calibration-audit.json',R/'freeze/prospective.json',R/'M0/products/a1100/map.fits',R/'M0/products/a1100/plan.yaml',R/'M0/products/a1100/operator-pixel.i32'}
 for nw,sid,p,s,col in segments():
  files.add(p/'receipt.yaml')
  for suffix in ('input.f64','eligible.u8','basis.f64','cleaned.f64','causes.u8','slots.i64'):
   f=p/f'segment-{sid}-{suffix}';require(digest(f)==s['files'][f.name],'checkpoint receipt hash '+str(f));files.add(f)
  up=p.parent.parent;files.update([up/'receipt.yaml',up/'source-probe/receipt.yaml',up/'source-probe/cal-multiplier.f64',p.parent/'cal/receipt.yaml'])
 return sorted(files)

def environment():return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__)
def revision():return subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()
def source_files():return [Path(__file__),D/'calibration_audit.py',W/'tools/timestream_successor/inspect_point_source.py',W/'tools/timestream_successor/point_storage.py',W/'tools/timestream_successor/point_transfer_experiment.py']
def prepare():
 require(not ROOT.exists(),'campaign destination exists; inspect instead of overwriting')
 require(not subprocess.check_output(['git','-C',str(W),'status','--porcelain'],text=True).strip(),'freeze requires clean committed source')
 files=inputs();cap=capacity();ROOT.mkdir()
 save(ROOT/'FREEZE.json',dict(schema='point152420-frozen-calibration-batch1-v1',source_revision=revision(),environment=environment(),roles=list(ROLES),inputs=[bound(p) for p in files],source=[bound(p) for p in source_files()],capacity=cap,budget=dict(planned_total=6,replacements_total=2,absolute_total=8,batch_planned=3,starts=0),mode='M0 frozen basis, recomputed observed means and coefficients; no feedback, no RTC/PTC learning; original and diagnostic CAL identities distinct'))
 save(ROOT/'STATUS.json',dict(status='READY',starts=0,roles=list(ROLES)));print(ROOT)

def run_role(role):
 numerator=None;count=None;unclean_num=None;details=[];h,s,hd,sd=tables();mults={}
 with fits.open(R/'M0/products/a1100/map.fits') as f:
  original=f[0].data.copy();header=f[0].header.copy();support=f['SCI_SUPPORT'].data.astype(bool);hits=f['HITS'].data.copy()
 size=original.size;numerator=np.zeros(size);count=np.zeros(size,dtype=np.int64);unclean_num=np.zeros(size)
 pixels=array(R/'M0/products/a1100/operator-pixel.i32','<i4')
 for nw,sid,p,seg,col in segments():
  a=checkpoint(sid,p,seg);y=a['y'];eligible=a['eligible'].copy()
  if role!='M0-replay':
   common=[];hf=[]
   for channel in a['ids']:
    hi,si=hd[(nw,int(channel))],sd[(nw,int(channel))];v=float(h['flxscale'][hi]);common.append(h['flag'][hi]==0 and s['flag'][si]==0 and np.isfinite(v) and v>0);hf.append(v)
   eligible&=np.array(common)[None,:]
   if role=='common-historical-conversion':
    if nw not in mults:
     pr=yread(p.parent.parent/'source-probe/receipt.yaml');require(pr['channels']==list(range(pr['detectors'])),'native channel ordering')
     rr=yread(p.parent.parent/'receipt.yaml');require({(r['factor'],r['phase_native_rows']) for r in rr['donor_continuity']['realized_detectors']}=={(1,0)},'native sample grid')
     mults[nw]=np.memmap(p.parent.parent/'source-probe/cal-multiplier.f64',dtype='<f8',mode='r',shape=(pr['native_rows'],pr['detectors']))
    scale=np.asarray(mults[nw][a['slots'][:,None],a['ids']]);require(np.all(np.isfinite(scale[eligible])&(scale[eligible]>0)),'conversion available')
    ratio=np.divide(np.asarray(hf)[None,:],scale,out=np.zeros_like(scale),where=eligible)
    y=np.where(eligible,y,0)*ratio
  clean,valid,diag=apply_basis(y,eligible,a['basis']);full=np.zeros(a['shape']);keep=np.zeros(a['shape'],bool);full[:,a['cols']]=clean;keep[:,a['cols']]=valid
  old=a['causes']==0;require(not np.any(keep&~old),'new support outside bound MAP domain')
  if role=='M0-replay':
   require(np.array_equal(keep,old),'native application support mismatch')
   err=float(np.max(abs(full[old]-a['native'][old])));require(err<=1e-5,'native cleaned replay differs');diag['native_cleaned_maxabs']=err
  first=col['first_input_row'];n=int(old.sum());require(n==col['input_rows'],'MAP row count')
  pix=pixels[first:first+n];sel=keep[old]&(pix>=0);v=full[old][sel];idx=pix[sel]
  numerator+=np.bincount(idx,weights=v,minlength=size);count+=np.bincount(idx,minlength=size)
  raw=np.zeros(a['shape']);raw[:,a['cols']]=np.where(eligible,y,0);unclean_num+=np.bincount(idx,weights=raw[old][sel],minlength=size)
  diag.update(network=nw,segment=sid,map_contributions=int(sel.sum()));details.append(diag)
  print(role,nw,sid,diag,flush=True)
 science,ref=normalization(count);image=np.divide(numerator,count,out=np.full(size,np.nan),where=count>0).reshape(original.shape);science=science.reshape(original.shape)
 if role=='M0-replay':
  require(np.array_equal(count.reshape(hits.shape),hits) and np.array_equal(science,support),'native MAP count/support mismatch')
  err=float(np.max(abs(image[science]-original[science])));require(err<1e-6,'native MAP replay mismatch')
 else:err=None
 yy,xx=np.indices(image.shape);x=(xx+1-header['CRPIX1'])*header['CDELT1']*3600;y=(yy+1-header['CRPIX2'])*header['CDELT2']*3600;roi=np.hypot(x,y)<=40
 require(np.all(science[roi]),'incomplete fixed 40 arcsec fit domain; do not silently restrict')
 bg=float(np.median(image[roi]));i=np.argmax(np.where(np.hypot(x,y)<=20,image,-np.inf));fit=solve_gaussian(image,roi,x,y,[max(0.,image.flat[i]-bg),x.flat[i],y.flat[i],5/np.sqrt(8*np.log(2)),5/np.sqrt(8*np.log(2)),0,bg],sigma_floor=1.5);require(fit.success,'fit failed')
 out=ROOT/role;out.mkdir();header['HISTORY']='Diagnostic frozen M0 basis. No independent RTC/PTC learning or FRUIT.';header['HISTORY']='Historical-conversion role replaces post-RTC CAL before PTC centering.'
 hdus=[fits.PrimaryHDU(image,header),fits.ImageHDU(science.astype('u1'),name='SCI_SUPPORT'),fits.ImageHDU(count.reshape(image.shape),name='HITS'),fits.ImageHDU(numerator.reshape(image.shape),name='NUMERATOR'),fits.ImageHDU(unclean_num.reshape(image.shape),name='UNCLEAN_NUMERATOR')];fits.HDUList(hdus).writeto(out/'map.fits',checksum=True)
 rr=np.hypot(x-fit.x[1],y-fit.x[2]);ann=(rr>=45)&(rr<60);bg2=float(np.median(image[ann&science]));profile=[]
 for radius in (4,8,12,20,30,40,50,60):
  inside=rr<radius;complete=bool(np.all(science[inside]));profile.append(dict(radius=radius,pixels=int(inside.sum()),supported=int(np.sum(inside&science)),integral_minus_annulus=float(np.sum(image[inside]-bg2)*4) if complete else None))
 return dict(role=role,map=bound(out/'map.fits'),parameters=fit.x.tolist(),peak=float(fit.x[0]),fwhm=np.sort(fit.x[3:5]*np.sqrt(8*np.log(2)))[::-1].tolist(),constant=float(fit.x[6]),angular_integral=float(2*np.pi*fit.x[0]*fit.x[3]*fit.x[4]),cost=float(fit.cost),active=fit.active_mask.tolist(),native_map_maxabs=err,normalization_reference=ref,science_pixels=int(science.sum()),native_science_pixels_lost=int(np.sum(support&~science)),native_science_pixels_gained=int(np.sum(science&~support)),contributions=int(count.sum()),background_annulus_median=bg2,background_annulus_supported=int(np.sum(ann&science)),profiles=profile,segments=details)

def run():
 require(not subprocess.check_output(['git','-C',str(W),'status','--porcelain'],text=True).strip(),'launch requires clean committed source')
 freeze=json.loads((ROOT/'FREEZE.json').read_text());require(revision()==freeze['source_revision'],'source HEAD changed');require(environment()==freeze['environment'],'environment changed')
 for r in freeze['inputs']+freeze['source']:require(digest(r['path'])==r['sha256'],'frozen input/source changed: '+r['path'])
 require(not (ROOT/'attempts.jsonl').exists(),'batch already started; resume assessment, never blind retry')
 started=time.time();results=[]
 for role in ROLES:
  cap=capacity()
  with (ROOT/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(role=role,kind='planned',epoch=time.time()))+'\n');f.flush();os.fsync(f.fileno())
  save(ROOT/'STATUS.json',dict(status='RUNNING',role=role,starts=len(results)+1,pid=os.getpid()))
  result=run_role(role)
  if role=='common-historical-conversion':
   with fits.open(ROOT/'common-current/map.fits') as a, fits.open(ROOT/role/'map.fits') as b:
    for name in ('HITS','SCI_SUPPORT'):require(np.array_equal(a[name].data,b[name].data),'matched conversion support changed')
  results.append(result);save(ROOT/role/'RESULT.json',result);save(ROOT/'RESULTS.json',results)
  require(tree_bytes(ROOT)<=RESERVATION,'batch exceeded reserved compact storage')
 save(ROOT/'STATUS.json',dict(status='COMPLETE_PENDING_ASSESSMENT',starts=len(results),seconds=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,capacity=capacity(),interpretation='Frozen operator effects only; photometric acceptance open'))
 print(json.dumps([dict(role=r['role'],peak=r['peak'],fwhm=r['fwhm']) for r in results],indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
 if args.action=='prepare':prepare()
 else:
  with (ROOT/'LOCK').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   try:run()
   except Exception as e:
    save(ROOT/'FAILURE.json',dict(error=repr(e),epoch=time.time()))
    state=json.loads((ROOT/'STATUS.json').read_text())
    if state.get('status')=='RUNNING' and state.get('pid')==os.getpid():
     state.update(status='FAILED',error=repr(e));save(ROOT/'STATUS.json',state)
    raise
