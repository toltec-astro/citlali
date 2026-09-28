"""Read completed native products once; summarize full retained physical-run population.
No fitting, masking, scientific processing, or new uncertainty ensemble.
"""
from pathlib import Path
import json,sys,time
import numpy as np,yaml
from astropy.io import fits
from scipy.ndimage import uniform_filter1d
W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_additive_diagnostics import save_json
from point_rtc_response import native_rows
from inspect_forward_map import require,digest
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-retained-residuals-20260928');B=R/'B-REF-rep1';D=R/'report'
def read(p):return json.loads(p.read_text())
def y(p):return yaml.load(p.read_text(),Loader=yaml.CSafeLoader)
def arr(p,t,shape=None):
 a=np.fromfile(p,t);return a.reshape(shape) if shape else a

def mad(a):
 return float(1.4826*np.median(abs(a-np.median(a)))) if len(a) else None

def runs(mask):return np.flatnonzero(np.diff(np.r_[False,mask,False].astype('i1'))).reshape(-1,2)

def summarize(z,off,lo,hi):
 good=np.isfinite(z);quiet=good&off;v=z[quiet];allv=z[good]
 q=dict(retained=int(good.sum()),background=int(quiet.sum()))
 if not len(v):return q
 median=float(np.median(v));diff=np.diff(z)[quiet[:-1]&quiet[1:]];scale=mad(diff);rms=float(np.std(v));peak=int(np.argmax(np.where(quiet,abs(z-median),-1)))
 q.update(median=median,background_std=rms,adjacent_difference_MAD=scale,q01_q05_q50_q95_q99=np.percentile(v,[1,5,50,95,99]).tolist(),minimum=float(v.min()),maximum=float(v.max()),maximum_abs_row=lo+peak,maximum_abs_value=float(z[peak]),all_min=float(allv.min()),all_max=float(allv.max()))
 for threshold in (6,12):
  tail=quiet&(abs(z-median)>threshold*scale) if scale else np.zeros(len(z),bool);rr=runs(tail)
  q[f'tail_{threshold}MAD_count']=int(tail.sum());q[f'tail_{threshold}MAD_longest_rows']=int(max((b-a for a,b in rr),default=0))
 # 61 samples ~0.5 s. Descriptive local mean offset, only fully supported background.
 width=61
 valid=uniform_filter1d(quiet.astype(float),width,mode='constant')>.999999
 smooth=uniform_filter1d(np.where(quiet,z-median,0),width,mode='constant')
 if valid.any():
  row=int(np.argmax(np.where(valid,abs(smooth),-1)));q.update(max_half_second_mean=float(smooth[row]),max_half_second_mean_row=lo+row)
 # Adjacent 0.5s means with a central 0.08s gap, strictly within retained runs.
 gap=10;offset=(width+gap)//2
 contrast=np.roll(smooth,-offset)-np.roll(smooth,offset);ok=np.roll(valid,-offset)&np.roll(valid,offset)
 ok[:width+gap]=False;ok[-width-gap:]=False
 if ok.any():
  row=int(np.argmax(np.where(ok,abs(contrast),-1)));q.update(max_half_second_contrast=float(contrast[row]),max_half_second_contrast_row=lo+row)
 return q

def network(nw):
 base=B/f'upstream/products/network{nw}';receipt=y(base/'receipt.yaml');pr=y(base/'source-probe/receipt.yaml');n,nd=pr['native_rows'],pr['detectors'];shape=n,nd
 g=read(B/'freeze/measurement-geometry.json')['parameters'];probe=base/'source-probe'
 lon=arr(probe/'longitude-deg.f64','<f8',shape)*3600;lat=arr(probe/'latitude-deg.f64','<f8',shape)*3600;mapping=arr(probe/'mapping-available.u8','u1',shape).astype(bool)
 radius=4*max(g[3:5])*np.sqrt(8*np.log(2));distance=np.hypot(lon-g[1],lat-g[2]);off=(distance>radius)&mapping
 stages={k:np.full(shape,np.nan) for k in ('CAL','M0')};pixmap=np.full(shape,-2,dtype='<i4');ptc=base/'donor-continuity/ptc';ptcr=y(ptc/'receipt.yaml')
 product=B/'M0/products/a1100';mr=y(product/'plan.yaml');pixel=arr(product/'operator-pixel.i32','<i4')
 for s in ptcr['segments']:
  sid=s['scan'];nt=s['scheduled_times'];nf=s['detectors'];no=s['output_detectors'];ids=np.asarray(s['detector_indices']);outids=np.asarray(s['output_detector_indices']);rows=native_rows(receipt,arr(ptc/f'segment-{sid}-slots.i64','<i8'))
  z=arr(ptc/f'segment-{sid}-input.f64','<f8',(nt,nf));good=arr(ptc/f'segment-{sid}-eligible.u8','u1',(nt,nf))>0;stages['CAL'][rows[:,None],ids]=np.where(good,z,np.nan)
  for name,path,stem in [('M0',ptc,f'segment-{sid}-')]:
   z=arr(path/f'{stem}cleaned.f64','<f8',(nt,no));good=arr(path/f'{stem}causes.u8','u1',(nt,no))==0
   stages[name][rows[:,None],outids]=np.where(good,z,np.nan)
   if name=='M0':
    group=next(q for q in mr['ordered_column_domain'] if f'network-{nw}-segment-{sid}-' in q['publication_file']);first=group['first_input_row'];nr=int(good.sum());require(nr==s['retained'],'source support changed');pp=np.full((nt,no),-2,dtype='<i4');pp[good]=pixel[first:first+nr];require((pixmap[rows[:,None],outids]==-2).all(),'duplicate scheduled occurrence');pixmap[rows[:,None],outids]=pp
 return base,receipt,pr,stages,pixmap,off,distance,lon,lat

def main():
 D.mkdir(exist_ok=True)
 start=time.monotonic();population=[];network_records=[];packed={};mapshape=None;projected=None;counted=None
 for nw in (1,2,3,4):
  base,r,pr,stages,pix,off,distance,lon,lat=network(nw)
  with fits.open(B/'M0/products/a1100/map.fits') as h:
   mapshape=h[0].data.shape;hits=h['HITS'].data.copy();norm=h['NORMALIZATION'].data.copy();support=h['SCI_SUPPORT'].data.astype(bool);header={k:v for k,v in h[0].header.items() if k not in ('COMMENT','HISTORY','')}
  require(np.array_equal(hits,norm),'diagnostic projection requires verified existing uniform coefficients')
  if projected is None:projected=np.zeros(mapshape);counted=np.zeros(mapshape,dtype='i8')
  good=pix>=0;counted+=np.bincount(pix[good],minlength=hits.size).reshape(mapshape);projected+=np.bincount(pix[good],weights=stages['M0'][good],minlength=hits.size).reshape(mapshape)
  nr=[]
  for d in range(pr['detectors']):
   for ri,(lo,hi) in enumerate(r['native_runs']):
    rr=dict(network=nw,detector=d,channel=pr['channels'][d],occurrence=r['detectors'][d]['occurrence'],physical_run=ri,native_interval=[lo,hi],MAP_contributions=int((pix[lo:hi,d]>=0).sum()),source_contributions=int(((pix[lo:hi,d]>=0)&~off[lo:hi,d]).sum()),T5='unavailable: failed first FRUIT call; not zero',stages={k:summarize(z[lo:hi,d],off[lo:hi,d],lo,hi) for k,z in stages.items()});population.append(rr);nr.append(rr)
  network_records.append(dict(network=nw,detectors=pr['detectors'],physical_runs=r['native_runs'],fully_unavailable=sum(q['stages']['M0']['retained']==0 for q in nr),retained_detectors=sum(q['stages']['M0']['retained']>0 for q in nr),retained_cells=int(np.isfinite(stages['M0']).sum()),MAP_contributions=int(good.sum()),source_contributions=int((good&~off).sum()),taps=67,factor=1,native_dt=r['native_integration_seconds']))
  print(nw,network_records[-1],flush=True)
 require(np.array_equal(counted,hits),'native operator reconstruction differs from M0 HITS')
 projected=np.divide(projected,hits,out=np.full(mapshape,np.nan),where=hits>0)
 np.savez_compressed(D/'residual-projection.npz',M0_residual=projected,HITS=hits,NORMALIZATION=norm,SCI_SUPPORT=support)
 save_json(D/'population.json',population);save_json(D/'population-summary.json',dict(networks=network_records,units='mJy/nominal beam for CAL/M0/T5; original x/r dimensionless retained in excerpts',background='outside 4 major FWHM of fixed previous B M0 fit; descriptive only, no scientific support change',rules='same rule on every detector/physical run, unavailable is absent not zero; adjacent difference MAD descriptive only; 61-row local means/contrasts require complete accepted background support',diagnostic_projection='actual M0 cleaned PTC residual, no restored operand; T5 population unavailable due failed first FRUIT call; native ordered operator-pixel and uniform coefficients/HITS verified, contribution diagnostic not independent astronomical map',seconds=time.monotonic()-start))
 save_json(D/'map-header.json',header)
 print('done',len(population),'physical detector/runs',flush=True)
if __name__=='__main__':main()
