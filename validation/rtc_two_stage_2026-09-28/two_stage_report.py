"""Saved diagnostic-only rendering: no original inputs or pipeline execution."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def render(compact,dest):
 dest.mkdir(parents=True,exist_ok=True);r=compact/'report';pop=read(r/'population.json');windows=read(r/'new-windows.json');packed=np.load(r/'new-windows.npz');maps=np.load(r/'direct-map.npz')
 plt.rcParams.update({'font.size':8,'figure.dpi':100,'savefig.dpi':130})
 fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
 for nw,ax in zip(range(1,5),axes.ravel()):
  pp=[q for q in pop if q['network']==nw and q['stages']['T5'].get('adjacent_difference_MAD')]
  x=np.array([q['stages']['M0']['adjacent_difference_MAD'] for q in pp]);y=np.array([q['stages']['T5']['adjacent_difference_MAD'] for q in pp]);c=[q['T5_map_contribution']['fit_disk_max_abs'] for q in pp]
  im=ax.scatter(x,y,c=c,s=9,cmap='viridis');ax.set_xscale('log');ax.set_yscale('log');lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],'k:',lw=.7);ax.set(title=f'Network {nw}: {len(pp)} retained detector/runs',xlabel='M0 difference MAD (mJy/beam)',ylabel='T5 difference MAD (mJy/beam)');fig.colorbar(im,ax=ax,label='Max whole-run contribution in fit disk (mJy/beam)')
 fig.suptitle('Full retained population • descriptive scales, not event uncertainty');fig.savefig(dest/'population.png');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
 for ax,key,title,limits in zip(axes.ravel(),('source_patch','source_fit_residual','source_normalized_residual','source_scale'),('B T5 signal (mJy/beam)','Accepted fit residual (mJy/beam)','Residual / saved local noise','Saved local noise (mJy/beam)'),((-60,440),(-70,70),(-3.5,3.5),(20,24))):
  im=ax.imshow(maps[key],origin='lower',extent=(41,-41,-41,41),cmap='coolwarm' if key!='source_scale' else 'viridis',vmin=limits[0],vmax=limits[1]);ax.set(title=title,xlabel='Tangent x (arcsec)',ylabel='Tangent y (arcsec)');fig.colorbar(im,ax=ax)
 fig.suptitle('New unchanged B through T5 • existing fit and uncertainty package');fig.savefig(dest/'final-map.png');plt.close(fig)
 for nw in range(1,5):
  qq=[q for q in windows if q['network']==nw];fig,axes=plt.subplots(5,len(qq),figsize=(16,11),constrained_layout=True)
  for col,q in enumerate(qq):
   k=q['key'];t=packed[k+'_time'];axes[0,col].set_title(f"{k} D{q['detector']}"+('\nT5 mean target' if col==3 else '\nprior M0 target/control'))
   for row,field in enumerate(('raw_x','raw_r','CAL','M0','T5')):
    z=packed[k+'_'+field];ax=axes[row,col];ax.plot(t,z,lw=.8)
    if row<2:
     peers=packed[k+('_peer_x' if row==0 else '_peer_r')]
     for j in range(peers.shape[1]):
      if np.isfinite(peers[:,j]).any() and np.isfinite(z).any():ax.plot(t,peers[:,j]-np.nanmedian(peers[:,j])+np.nanmedian(z),lw=.6,alpha=.45)
    ax.axvline(t[np.argmin(abs(packed[k+'_row']-q['center']))],color='k',ls=':',lw=.7);ax.set_ylabel(field+(' (mJy/beam)' if row>1 else ' (dimensionless)'));ax.grid(alpha=.15)
    if row==4:ax.set_xlabel('Physical-run time (s)')
  fig.suptitle(f'Network {nw} • original x/r, CAL, M0 and actual T5 residual\nFaded: adjacent detector indices recentered for context; not necessarily physical neighbors');fig.savefig(dest/f'windows-n{nw}.png');plt.close(fig)
 rows=['| Network | Total runs | Retained at T5 | Unavailable | Median CAL / M0 / T5 difference MAD (mJy/beam) |','|---|---:|---:|---:|---|']
 for nw in range(1,5):
  allq=[q for q in pop if q['network']==nw];qq=[q for q in allq if q['stages']['T5'].get('adjacent_difference_MAD')];vals=[np.median([q['stages'][stage]['adjacent_difference_MAD'] for q in qq]) for stage in ('CAL','M0','T5')];rows.append(f'| {nw} | {len(allq)} | {len(qq)} | {len(allq)-len(qq)} | '+ ' / '.join(f'{v:.2f}' for v in vals)+' |')
 rows+=['','Unavailable is absent, never zero. Physical runs are not numerical scans. Contribution maps are not isolated contaminant biases or counterfactual quarantine effects.','']
 (dest/'DIAGNOSTICS.md').write_text('\n'.join(rows))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--compact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.compact,a.output)
