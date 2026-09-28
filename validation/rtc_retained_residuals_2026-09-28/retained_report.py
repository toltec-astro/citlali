"""Render only packed diagnostics. Never imports science processing or fits a model."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def render(compact,dest):
 dest.mkdir(parents=True,exist_ok=True);r=compact/'report';pop=read(r/'population.json');summary=read(r/'population-summary.json');windows=read(r/'new-windows.json');packed=np.load(r/'new-windows.npz');direct=read(r/'direct-map.json');maps=np.load(r/'direct-map.npz');old=read(r/'old-window-projections.json')['records']
 plt.rcParams.update({'font.size':9,'figure.dpi':110,'savefig.dpi':130})
 fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
 for nw,ax in zip((1,2,3,4),axes.ravel()):
  q=[p for p in pop if p['network']==nw and p['stages']['M0'].get('adjacent_difference_MAD') and p['stages']['CAL'].get('adjacent_difference_MAD')];x=[p['stages']['CAL']['adjacent_difference_MAD'] for p in q];z=[p['stages']['M0']['adjacent_difference_MAD'] for p in q];ax.loglog(x,z,'.',ms=3);bounds=[min(x+z),max(x+z)];ax.plot(bounds,bounds,'k:',lw=1);ax.set(title=f'Network {nw}: {len(q)} retained runs',xlabel='CAL adjacent-difference MAD (mJy/beam)',ylabel='M0 adjacent-difference MAD (mJy/beam)');ax.grid(alpha=.2)
 fig.suptitle('Complete CAL/M0 population • descriptive scales, not event uncertainty\nT5 population unavailable after rejected first FRUIT request');fig.savefig(dest/'population.png');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
 for ax,key,title,limits in zip(axes.ravel(),('source_patch','source_fit_residual','source_normalized_residual','source_scale'),('Saved B T5 signal (mJy/beam)','Saved fit residual (mJy/beam)','Residual / saved local noise scale','Saved local noise scale (mJy/beam)'),((-60,440),(-70,70),(-3.5,3.5),(20,24))):
  im=ax.imshow(maps[key],origin='lower',extent=(41,-41,-41,41),cmap='coolwarm' if key!='source_scale' else 'viridis',vmin=limits[0],vmax=limits[1]);ax.set(title=title,xlabel='AST tangent x (arcsec)',ylabel='AST tangent y (arcsec)');fig.colorbar(im,ax=ax)
 fig.suptitle('Unmodified prior T5 • accepted fit and existing noise • no new fit');fig.savefig(dest/'final-map.png');plt.close(fig)
 for nw in (1,2,3,4):
  qq=[q for q in windows if q['network']==nw];fig,axes=plt.subplots(4,3,figsize=(14,10),constrained_layout=True)
  for col,q in enumerate(qq):
   key=q['key'];t=packed[key+'_time'];axes[0,col].set_title(f"{key} D{q['detector']}\n"+('peak target' if col==0 else '0.5-second target' if col==1 else 'typical-scale control'))
   for row,field in enumerate(('raw_x','raw_r','CAL','M0')):
    z=packed[key+'_'+field];ax=axes[row,col];ax.plot(t,z,lw=.8)
    if row<2:
     peers=packed[key+('_peer_x' if row==0 else '_peer_r')];
     for j in range(peers.shape[1]):
      if np.isfinite(peers[:,j]).any() and np.isfinite(z).any():ax.plot(t,peers[:,j]-np.nanmedian(peers[:,j])+np.nanmedian(z),lw=.6,alpha=.45)
    ax.axvline(t[np.argmin(abs(packed[key+'_row']-q['center']))],color='k',ls=':',lw=.7);ax.set_ylabel(field+(' (mJy/beam)' if row>1 else ' (dimensionless)'));ax.grid(alpha=.15)
    if row==3:ax.set_xlabel('Physical-run elapsed time (s)')
  fig.suptitle(f'Network {nw}: retained M0 audit • faded x/r = recentered adjacent detector indices\nTargets selected from M0; not proof of non-astronomical origin or T5 persistence');fig.savefig(dest/f'windows-n{nw}.png');plt.close(fig)
 # Machine-readable compact table retains every detector/run, including unavailable.
 rows=['| Network | Total runs | M0 retained | M0 unavailable | M0 MAP contributions | Median CAL → M0 difference MAD (mJy/beam) |','|---|---:|---:|---:|---:|---|']
 for nw in (1,2,3,4):
  n=next(n for n in summary['networks'] if n['network']==nw);q=[q for q in pop if q['network']==nw and 'adjacent_difference_MAD' in q['stages']['M0']];vals=[np.median([a['stages'][stage]['adjacent_difference_MAD'] for a in q]) for stage in ('CAL','M0')];rows.append(f"| {nw} | {n['detectors']} | {n['retained_detectors']} | {n['fully_unavailable']} | {n['MAP_contributions']:,} | {vals[0]:.2f} → {vals[1]:.2f} |")
 rows+=['','All stages use the same physical-run identity; no numerical scan boundary becomes a physical gap. T5 full-population values are unavailable, never zero.','','| Prior window | D | T5 sample range (mJy/beam) | MAP / science-supported contributions | Max science projection (mJy/beam) | Max within 40 arcsec |','|---|---:|---|---:|---:|---:|']
 for q in old:
  if q['key'] not in ('n1_w4','n1_w5','n2_w4','n2_w6','n3_w6'):continue
  rows.append(f"| {q['key']} | {q['detector']} | {q['T5_residual_min']:.1f} to {q['T5_residual_max']:.1f} | {q['actual_MAP_contributions']} / {q['science_supported_MAP_contributions']} | {q['max_abs_projection_science_mJy_beam']:.3f} | {q['max_abs_projection_40arcsec_mJy_beam']:.3f} |")
 rows+=['','These are selected-excerpt contribution diagnostics using authenticated actual T5 support, unit native coefficients and full normalization. They are not independent sky maps or predicted effects of quarantine. PTC/FRUIT would recompute after any veto.','',f"Saved B T5 background: N={direct['background_raw']['n']}, scatter={direct['background_raw']['std']:.3f} mJy/beam. Within 40 arcsec, fit-residual scatter={direct['source_disk_fit_residual']['std']:.3f} mJy/beam; median saved local noise={direct['source_noise_scale']['median']:.3f} mJy/beam. These descriptive single-map comparisons neither establish nor exclude systematic contamination."]
 (dest/'DIAGNOSTICS.md').write_text('\n'.join(rows)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--compact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.compact,a.output)
