"""Saved-only compact evidence reporting: no original inputs, fits, or reductions."""
from pathlib import Path
import argparse,json,collections,gzip
import numpy as np,yaml
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def stats(v,rows,center):
 good=np.isfinite(v);pre=good&(rows<center-12);post=good&(rows>center+12)
 idx=np.flatnonzero(good); pairs=np.diff(rows[idx])==1;dv=np.diff(v[idx])[pairs];mad=float(1.4826*np.median(abs(dv-np.median(dv)))) if len(dv) else None
 a=float(np.median(v[pre])) if pre.any() else None;b=float(np.median(v[post])) if post.any() else None
 return dict(n=int(good.sum()),pre_n=int(pre.sum()),post_n=int(post.sum()),pre_median=a,post_median=b,post_minus_pre=b-a if a is not None and b is not None else None,adjacent_difference_MAD=mad)
def main(root,out):
 out.mkdir(parents=True,exist_ok=True);report=root/'report';facts={};table=['# Saved POINT step metrics','', '| Observation | nw | events / admitted | affected D / all D | valid detector-s | RTC retained detector-s | CAL/MAP contributions |','|---|---:|---:|---:|---:|---:|---:|'];wstats=[]
 for name in ('N14-REF-rep1',):
  p=report/name;pop=read(p/'population.json');win=read(p/'windows.json');z=np.load(p/'windows.npz');mask=np.load(p/'population-masks.npz');ep=p/'event-chain.yaml';ev=yaml.load(ep.read_bytes() if ep.exists() else gzip.decompress(Path(str(ep)+'.gz').read_bytes()),Loader=yaml.CSafeLoader)
  facts[name]=dict(population=pop,map_support=read(p/'map-support.json'),review={kind:dict(collections.Counter((q['adjudication']['label'] if isinstance(q['adjudication'],dict) else q['adjudication'].split(':')[0]) for q in win if q['kind']==kind)) for kind in ('event','unflagged')})
  fig,ax=plt.subplots(4,3,figsize=(16,14),layout='constrained')
  for i,n in enumerate(pop):
   nw=n['network'];nt,nd=n['shape'];dt=n['cadence_seconds'];extent=[0,nt*dt,-.5,nd-.5];events=[e for e in ev if e['network']==nw];cx=[];cy=[]
   for e in events:
    for c in e['chain']['candidates']:cx.append(c['earlier_row']*dt);cy.append(e['detector'])
   ax[i,0].scatter(cx,cy,s=3);ax[i,0].set_xlim(0,nt*dt);ax[i,0].set_ylim(-.5,nd-.5)
   for j,key in enumerate(('transition','RTC_x'),1):
    a=np.unpackbits(mask[f'n{nw}_{key}'],axis=1)[:,:nt].astype(bool);a=~a if key=='RTC_x' else a
    if key=='transition':
     ds,ts=np.nonzero(a);ax[i,j].scatter(ts*dt,ds,s=4,color='purple');ax[i,j].set_xlim(0,nt*dt);ax[i,j].set_ylim(-.5,nd-.5)
    else:ax[i,j].imshow(a,aspect='auto',origin='lower',extent=extent,interpolation='nearest',cmap='Greys',vmin=0,vmax=1)
   for j,title in enumerate(('Original candidates (coordinate instances)','Accepted physical transitions','Realized RTC x exclusions (all causes)')):ax[i,j].set_title(f'nw{nw}: {title}',fontsize=10);ax[i,j].set_xlabel('seconds from first native sample');ax[i,j].set_ylabel('detector identity')
   table.append(f"| {n['observation']} | {nw} | {n['event_groups']} / {n['admitted_jump_groups']} | {n['admitted_detector_count']} / {nd} | {n['input_valid_detector_seconds']:.3f} | {n['RTC_retained_x_cells']*n['integration_seconds']:.3f} | {n['CAL_eligible_scheduled_cells']} |")
  fig.savefig(out/f'{name}-population.png',dpi=130);plt.close(fig)
  for q in win:
   get=lambda k:z[q['key']+'_'+k];values={k:get(k).copy() for k in ('raw_x','raw_r','RTC_x','RTC_r','CAL','M0','T5_residual','T5_total')}
   values['raw_x'][~get('valid_x')]=np.nan;values['raw_r'][~get('valid_r')]=np.nan;values['RTC_x'][(get('state')&1)==0]=np.nan;values['RTC_r'][(get('state')&2)==0]=np.nan
   wstats.append(dict(run=name,key=q['key'],kind=q['kind'],adjudication=q['adjudication'],stage={k:stats(v,get('native_row'),q['center']) for k,v in values.items()}))
  # Complete event/disposition population, stable keys map directly into event-chain.yaml.
  lines=['| obs/nw/D/event | seed unix s | x/r jump admission | disposition | RTC excluded at seed |','|---|---:|---|---|---|']
  for e in ev:lines.append(f"| {e['observation']}/{e['network']}/{e['detector']}/{e['event']} | {e['seed_unix']:.6f} | {'/'.join(str(c['admitted']) for c in e['chain']['coordinates'])} | {e['consider']['class']} | {e['applied']['excluded_at_seed']} |")
  (out/f'{name}-events.md').write_text('\n'.join(lines)+'\n')
 response=read(report/'response/source-responses.json');facts['active_selection']=read(report/'response/active-selection-summary.json');facts['timing']=read(report/'timing.json');facts['resources']=read(report/'resources.json');effects=[]
 table+=['','Ratios compare signed increments to own ideal on full REF support; position/width discrepancies are absolute arcsec. T5 sigmas are conditional scales, not paired-difference significance.','', '| branch | stage | peak | template | ap1 | ap1.5 | ap2 | spatial g | distortion eta |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 fig,axes=plt.subplots(3,3,figsize=(12,11),layout='constrained');maps=np.load(report/'response/response-maps.npz')
 for i,branch in enumerate(('C-core-plus','C-core-minus','C-shoulder')):
  for j,(key,title) in enumerate((('ideal','ideal addition'),('injected','actual minus REF'),('residual','response minus ideal'))):
   v=maps[f'{branch}_T5_{key}'].copy()
   if key=='injected':v-=maps[f'{branch}_T5_REF']
   support=maps[f'{branch}_T5_support'];v[~support]=np.nan
   artist=axes[i,j].imshow(v,origin='lower',cmap='RdBu_r',vmin=-np.nanmax(abs(v)),vmax=np.nanmax(abs(v)));axes[i,j].set_title(branch+' T5 '+title,fontsize=9);fig.colorbar(artist,ax=axes[i,j],label='mJy/beam')
 for r in response:
  comp={q['name']:q for q in r['measurement']['comparisons']};names=('peak','fixed_template','aperture_1','aperture_1p5','aperture_2');ratios=[comp[k]['ratio'] for k in names]
  table.append('| '+r['branch']+f" | T{r['cycle']} | "+' | '.join(f'{v:.5f}' if v is not None else 'unavailable' for v in ratios)+f" | {r['spatial']['g']:.5f} | {r['spatial']['eta']:.5f} |")
  effects.append(dict(branch=r['branch'],cycle=r['cycle'],measurements=comp,spatial=r['spatial'],RTC_spatial=r['RTC_spatial'],unsupported=r['unsupported_regions'],scales=r['endpoint_conditional_sigmas'],raw_aperture_background=r['measurement']['raw_aperture_and_background_increments']))
 fig.savefig(out/'response.png',dpi=130);plt.close(fig)
 table+=['','Window metrics use all finite samples except ±12 native rows around the declared center, independent pre/post medians and adjacent finite-difference MAD. Descriptive contrast only: slopes, source structure and colored noise are not removed. No per-step significance threshold or new fit. Raw x/r and RTC use Δf/f; CAL/PTC use mJy/beam. Excluded stage values are unavailable, never zero.']
 (out/'METRICS.md').write_text('\n'.join(table)+'\n');save(out/'report-facts.json',facts);save(out/'window-statistics.json',wstats);save(out/'effects.json',effects)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
