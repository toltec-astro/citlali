"""Static scientific figures from machine-readable saved summaries only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS=dict(current10='#1b6ca8',current15='#9b59b6',warm10='#db7c26',covariance10='#218c74',warm15='#b14c13',covariance15='#146b52')
LABELS=dict(current10='Current rank10',current15='Current rank15',warm10='Warm ALS rank10',covariance10='Covariance rank10',warm15='Warm ALS rank15',covariance15='Covariance rank15')
def read(p):return json.loads(p.read_text())
def plot(root,figure_dir=None):
 d=root/'report';figdir=figure_dir or d/'figures';figdir.mkdir(parents=True,exist_ok=True);timing=read(d/'method-timing.json');measure=read(d/'science-measurements.json');effect=read(d/'science-responses.json');a=read(d/'PHASE-A-timing.json')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':160})
 fig,axes=plt.subplots(1,2,figsize=(12,4.8))
 for t in timing:
  f=t['family'];r=t['bycycle'];ks=sorted(map(int,r));x=np.array(ks);y=np.array([r[str(k)]['median'] for k in ks]);lo=[r[str(k)]['minimum'] for k in ks];hi=[r[str(k)]['maximum'] for k in ks]
  axes[0].plot(x,y,label=LABELS[f],color=COLORS[f]);axes[0].fill_between(x,lo,hi,color=COLORS[f],alpha=.14)
  if len(y)>1:axes[1].plot(x[1:],np.diff(y),'.-',label=LABELS[f],color=COLORS[f])
 for rank in (10,15):
  rows=[r for r in a if r['rank']==rank and r['N']<10 and 'terminating_exit' in r];axes[0].scatter([r['N'] for r in rows],[r['terminating_exit'] for r in rows],marker='x',s=40,color=COLORS['current'+str(rank)],label=f'Actual exit rank{rank} (single)')
 axes[0].set(xlabel='Feedback count N',ylabel='Original inputs → usable measurement, seconds',title='Prefix latency: median and observed range');axes[0].legend(fontsize=8,loc='upper left');axes[1].set(xlabel='Feedback count N',ylabel='Difference of prefix medians, seconds',title='Marginal cost (descriptive; not a paired interval)');axes[1].legend(fontsize=8)
 fig.suptitle('Recorded sequence: growing storage-accounting overhead confounds method comparisons\nPOINT152430 / a2000 / networks11–12 · Apple M4 Pro · warm OS cache allowed',fontsize=13);fig.tight_layout();fig.savefig(figdir/'runtime.png');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
 fields=[('east','Pointing east / arcsec'),('north','Pointing north / arcsec'),('peak','Peak / mJy per nominal beam'),('FWHM_major','Major FWHM / arcsec'),('fixed_template','Fixed template / mJy per nominal beam'),('aperture_2','Aperture2 / mJy per nominal beam arcsec²')]
 for ax,(key,label) in zip(axes.flat,fields):
  for t in timing:
   f=t['family'];rows=[r for r in measure if r['family']==f];ax.plot([r['cycle'] for r in rows],[r['values'][key] for r in rows],'.-',label=LABELS[f],color=COLORS[f])
  ax.set(ylabel=label);ax.grid(alpha=.15)
  constrained=[r for r in measure if r.get('fit_constrained')]
  if key=='FWHM_major' and constrained:
   ax.scatter([r['cycle'] for r in constrained],[r['values'][key] for r in constrained],facecolors='none',edgecolors='black',s=90,zorder=5)
   ax.set_title('Open circle: accepted fit at angle bound',fontsize=9)
 for ax in axes[-1]:ax.set_xlabel('Feedback count N')
 axes[0,0].legend(fontsize=8);fig.suptitle('Measured source evolution · common geometry · no truth or automatic-stop claim');fig.tight_layout();fig.savefig(figdir/'measurements.png');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for t in timing:
  f=t['family'];r=t['bycycle']
  for branch,style in [('C-core-plus','-'),('C-core-minus','--')]:
   for ax,key in [(axes[0,0],'peak'),(axes[0,1],'aperture_2')]:
    rows=[e for e in effect if e['family']==f and e['branch']==branch and e['measurement']==key and e['sigma_equiv'] is not None];ax.plot([r[str(e['cycle'])]['median'] for e in rows],[e['sigma_equiv'] for e in rows],style+'o',ms=4,color=COLORS[f],label=LABELS[f]+(' +' if branch.endswith('plus') else ' −'))
  rows=[e for e in effect if e['family']==f and e['branch']=='C-shoulder' and e['measurement']=='aperture_2'];x=[r[str(e['cycle'])]['median'] for e in rows];axes[1,0].plot(x,[100*e['ratio'] for e in rows],'o-',color=COLORS[f],label=LABELS[f]);axes[1,1].plot(x,[e['spatial_response']['eta'] for e in rows],'o-',color=COLORS[f],label=LABELS[f])
 axes[0,0].set(title='Compact peak: conditional σ / |gain|',ylabel='mJy per nominal beam');axes[0,1].set(title='Compact aperture2: conditional σ / |gain|',ylabel='mJy per nominal beam arcsec²');axes[1,0].set(title='Shoulder: systematic aperture recovery',ylabel='Actual / ideal increment, %');axes[1,0].axhline(100,color='gray',lw=.8);axes[1,1].set(title='Shoulder: spatial distortion η',ylabel='Residual norm / ideal-control norm')
 for ax in axes.flat:ax.set_xlabel('Measured prefix readiness, seconds');ax.grid(alpha=.15)
 axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=8);fig.suptitle('Conditional statistical scales (top) and systematic broad response (bottom)\nOwn REF noise · signed controls separate · no scalar shoulder correction · time has run-order overhead');fig.tight_layout();fig.savefig(figdir/'response-versus-time.png');plt.close(fig)
 # Source-region maps are display diagnostics from saved products, no fit.
 from compare_point_pipeline import available
 from point_fruit_diagnostics import load,patch_map
 from point_empirical import source_patch
 retained=Path(__file__).resolve().parents[2]/'validation/ptc_rank_response_2026-09-26/compact'
 families=[('current10',root/'A-rank10-N10-rep1',retained/'rank-10'),('warm10',root/'B-warm-rank10-N10-rep1',root/'B-warm-rank10-N10-rep1'),('covariance10',root/'B-covariance-rank10-N10-rep1',root/'B-covariance-rank10-N10-rep1')]
 base=available(families[0][1]/'run/BASE-POINT/REF/a2000/cycle-10/native/products/total.fits');bs,bv,_,_=patch_map(base)
 fig,axes=plt.subplots(3,3,figsize=(12,10))
 for i,(family,out,controls) in enumerate(families):
  path=available(out/'run/BASE-POINT/REF/a2000/cycle-10/native/products/total.fits');full,h=load(path);s,v,_,_=patch_map(path)
  response,rh=load(available(controls/'run/BASE-POINT/C-shoulder/a2000/cycle-10/response.fits'));res,rv,_,_,_=source_patch(response['RESIDUAL'],response['VALID'].astype(bool),full['NORMALIZATION'],rh)
  arrays=[np.where(v,s,np.nan),np.where(v&bv,s-bs,np.nan),np.where(rv,res,np.nan)];limits=[(-5,50),(-8,8),(-3,3)]
  for j,(im,(lo,hi)) in enumerate(zip(arrays,limits)):
   artist=axes[i,j].imshow(im,origin='lower',extent=(41,-41,-41,41),vmin=lo,vmax=hi,cmap='RdBu_r');axes[i,j].set(xlabel='East offset / arcsec',ylabel='North offset / arcsec',title=LABELS[family]+[' · T10 signal',' · T10 minus current',' · shoulder actual−ideal'][j]);fig.colorbar(artist,ax=axes[i,j],shrink=.75,label='mJy per nominal beam')
 fig.suptitle('T10 source structure · signal display clipped at50 to expose the wings\nMatched full support · shoulder residual is a systematic response diagnostic',fontsize=13);fig.tight_layout();fig.savefig(figdir/'source-structure.png');plt.close(fig)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--figure-dir',type=Path);a=p.parse_args();plot(a.root,a.figure_dir)
