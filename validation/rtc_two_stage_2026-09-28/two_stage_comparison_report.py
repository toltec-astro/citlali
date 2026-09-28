"""Render packed comparison only; no fitting or native input access."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(compact,dest):
 r=compact/'report';dest.mkdir(parents=True,exist_ok=True);c=json.loads((r/'B-Q-comparison.json').read_text());m=np.load(r/'B-Q-maps.npz');qr=json.loads((r/'response-Q/source-responses.json').read_text());br=json.loads((r/'sealed-B-source-responses.json').read_text())
 plt.rcParams.update({'font.size':9,'figure.dpi':110,'savefig.dpi':130})
 fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
 for ax,key,title,vlim in zip(axes.ravel(),('B_residual','Q_residual','difference','normalization_difference'),('B T5 residual to accepted fit','Q T5 residual to accepted fit','Actual end-to-end Q minus B','Normalization Q minus B'),(100,100,100,None)):
  z=m[key].copy();z[~m['B_support']]=np.nan
  if key=='Q_residual':z[~m['Q_support']]=np.nan
  im=ax.pcolormesh(m['x_arcsec'],m['y_arcsec'],z,shading='nearest',cmap='coolwarm',vmin=-vlim if vlim else None,vmax=vlim);ax.contour(m['x_arcsec'],m['y_arcsec'],m['B_event_footprint'],levels=[.5],colors=['black'],linewidths=.8);ax.set(title=title,xlabel='Tangent x (arcsec)',ylabel='Tangent y (arcsec)',aspect='equal');fig.colorbar(im,ax=ax,label='MAP contributions' if key=='normalization_difference' else 'mJy/beam')
 fig.suptitle('B-supported domain retained explicitly; missing Q is blank\nBlack outline: baseline-selected D88 excerpt footprint; signal colors clipped at ±100');fig.savefig(dest/'B-Q-maps.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
 for branch,color in zip(('C-core-plus','C-core-minus','C-shoulder'),('tab:blue','tab:orange','tab:green')):
  for family,rr,style in [('B',br,':'),('Q',qr,'-')]:
   ss=sorted([x for x in rr if x['branch']==branch],key=lambda x:x['cycle']);axes[0].plot([x['cycle'] for x in ss],[x['spatial']['g'] for x in ss],style+'o',color=color,label=family+' '+branch)
 axes[0].axhline(1,color='k',lw=.7);axes[0].set(xlabel='Feedback cycle',ylabel='Finite-response spatial gain',title='Exact historical rasters; each own REF');axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
 final=c['cycles'][-1];selected=[q for q in final['measurements'] if q['name'] in ('peak','cx','cy','FWHM_major','FWHM_minor','template','aperture2')];axes[1].bar([q['name'] for q in selected],[q['delta_over_B_sigma'] for q in selected]);axes[1].axhline(0,color='k',lw=.7);axes[1].set(ylabel='Q−B / B conditional scale',title='Scale comparison, not paired significance');axes[1].tick_params(axis='x',labelrotation=35);axes[1].grid(axis='y',alpha=.2);fig.savefig(dest/'B-Q-response-and-measurements.png');plt.close(fig)
 rows=['| Quantity | B | Q | Q−B | B conditional scale |','|---|---:|---:|---:|---:|']
 for q in final['measurements']:rows.append(f"| {q['name']} | {q['B']:.6g} | {q['Q']:.6g} | {q['delta']:.6g} | {q['B_conditional_sigma']:.6g} |")
 rows+=['','Peak/template/constant in mJy/beam; centroids/FWHM in arcsec; major-axis angle in radians; apertures in mJy/beam arcsec². Conditional scales omit model-estimation and policy-selection uncertainty. Angle constrained in B requires caution.','','| Cycle | B science pixels | Q science pixels | Lost B pixels | B contributions | Q contributions |','|---|---:|---:|---:|---:|---:|']
 for q in c['cycles']:rows.append(f"| {q['cycle']} | {q['B_science_pixels']} | {q['Q_science_pixels']} | {q['lost_B_science_pixels']} | {q['B_contributions']} | {q['Q_contributions']} |")
 rows+=['','Q absence is not a zero residual; comparisons on shared pixels are explicitly conditional. Whole-run removal changes normalization and PTC/FRUIT relearning. No direct selected-excerpt contribution is called an isolated contaminant bias.','']
 (dest/'COMPARISON.md').write_text('\n'.join(rows))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--compact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.compact,a.output)
