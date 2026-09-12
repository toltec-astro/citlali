"""Static retained-product figures; no inference or cleaning."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import run_comparison as run
H=run.H;g=run.load_geometry();shape=g.shape;x,y=g.x,g.ygrid
extent=[x.min()-1,x.max()+1,y.min()-1,y.max()+1]
audit=json.loads((run.OUT/'SPATIAL_AUDIT.json').read_text());old=np.load(run.OLD/'saved_null20260911.npz')
fig,axs=plt.subplots(3,2,figsize=(10.2,11),layout='constrained')
for a,array in enumerate(run.b.ARRAYS):
    for j in range(2):
        ax=axs[a,j]
        if j==0:
            q=np.full(g.npix,np.nan);q[g.D[a]]=np.log10(g.Q[a,g.D[a]]/np.median(g.Q[a,g.D[a]]))
            im=ax.imshow(q.reshape(shape),origin='lower',extent=extent,cmap='viridis',vmin=-.75,vmax=.25,interpolation='nearest')
            for case,color,mark in [('saved_null20260911','red','x'),('saved_null20260912','white','+')]:
                loc=[r for r in audit['selected_locations'] if r['case']==case and r['array']==array]
                ax.scatter([r['x'] for r in loc],[r['y'] for r in loc],c=color,marker=mark,s=45,linewidth=1.2,label=case.removeprefix('saved_null'))
            if a==0:ax.legend(loc='upper right',fontsize=8)
            fig.colorbar(im,ax=ax,shrink=.8,label='log10[coverage / median(D)]')
        else:
            z=np.where(g.D[a],old['model'][a],np.nan)
            im=ax.imshow(z.reshape(shape),origin='lower',extent=extent,cmap='magma',vmin=0,vmax=np.nanmax(z),interpolation='nearest')
            rec=next(r for r in audit['false_models'] if r['case']=='saved_null20260911' and r['array']==array)
            ax.text(.02,.97,f"Brightness inside 60″: {100*rec['central_fraction']:.3g}%",transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor='white',alpha=.9,edgecolor='none'),fontsize=9)
            fig.colorbar(im,ax=ax,shrink=.8,label='False model, legacy brightness')
        ax.add_patch(Circle((0,0),60,fill=False,color='cyan',linestyle='--',linewidth=1.4));ax.add_patch(Circle((0,0),90,fill=False,color='gray',linewidth=.6))
        ax.set(xlim=(-94,94),ylim=(-94,94),xlabel='Map x (arcsec)',ylabel=array+'\nMap y (arcsec)' if j==0 else '')
        if a==0:ax.set_title(['Coverage and selected null coefficients','Original full-domain null model, seed 20260911'][j],fontsize=10)
fig.suptitle('The retained null failures originate outside the fixed 60″ central region\nDashed circle: tested domain; some reconstructed false brightness spreads inward',fontsize=12)
fig.savefig(H/'SPATIAL_EVIDENCE.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.6,5.2),layout='constrained')
names=['noiseless_phase0_coma4p0','offset_noiseless_coma4'];tt=[np.load(run.OUT/(n+'_truth.npz'))['truth'][0] for n in names]
clip=json.loads((run.OUT/'TRUTH_CLIPPING.json').read_text());vmax=max(t.max() for t in tt)
for ax,name,t in zip(axs,names,tt):
    z=np.where(g.D[0],t,np.nan);im=ax.imshow(z.reshape(shape),origin='lower',extent=extent,cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
    rec=next(r for r in clip if r['case']==name and r['array']=='a1100')
    ax.add_patch(Circle((0,0),60,fill=False,color='cyan',linestyle='--',linewidth=1.4));ax.scatter(*rec['truth_center'],marker='+',s=70,c='white')
    ax.set(xlim=(-94,94),ylim=(-94,94),xlabel='Map x (arcsec)',ylabel='Map y (arcsec)',title=f"Truth center {tuple(rec['truth_center'])}\nBrightness excluded: {100*rec['brightness_fraction_outside_C']:.2f}%")
fig.colorbar(im,ax=axs,shrink=.8,label='True coma, legacy map brightness')
fig.suptitle('A hard 60″ reconstruction domain also removes real comatic structure\nSame PSF and brightness; a1100 mask shown; crosses identify truth for evaluation only',fontsize=12)
fig.savefig(H/'DOMAIN_CLIPPING.png',dpi=160);plt.close(fig)
