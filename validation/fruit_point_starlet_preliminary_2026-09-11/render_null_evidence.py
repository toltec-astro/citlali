"""Static scientific evidence from retained outputs; no inference or learning."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
H=Path(__file__).resolve().parent;out=Path('/private/tmp/sci-fruit-point-starlet-preliminary-20260911-r0.1')
g=np.load('/private/tmp/sci-fruit-point-rbf-feedback-20260911-r0.1/geometry.npz');shape=tuple(g['shape']);x,y=g['x'],g['y'];D=g['D']
a=np.load(out/'saved_null20260911.npz');c=np.load(out/'saved_null20260912.npz')
extent=[x.min()-1,x.max()+1,y.min()-1,y.max()+1]
fig,axs=plt.subplots(3,3,figsize=(12.6,10),layout='constrained')
for j,array in enumerate(['a1100','a1400','a2000']):
    noise=a['input_total'][j];vmax=max(a['model'][j].max(),c['model'][j].max());limit=np.nanpercentile(abs(noise),98)
    for k,field in enumerate([noise,a['model'][j],c['model'][j]]):
        ax=axs[j,k];z=np.where(D[j],field,np.nan).reshape(shape)
        im=ax.imshow(z,origin='lower',extent=extent,cmap='RdBu_r' if k==0 else 'magma',vmin=-limit if k==0 else 0,vmax=limit if k==0 else vmax,interpolation='nearest')
        ax.add_patch(Circle((0,0),90,fill=False,lw=.6,color='gray'));ax.set(xlim=(-94,94),ylim=(-94,94),xlabel='Map x (arcsec)',ylabel=(array+'\nMap y (arcsec)') if k==0 else '')
        ax.set_title(['Noise-only input, seed 20260911','Admitted model, calibration null','Returned model, other null'][k] if j==0 else '')
        if k>0:ax.text(.025,.975,f"Peak {field.max():.1f}",transform=ax.transAxes,ha='left',va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        if k in [0,2]:fig.colorbar(im,ax=ax,shrink=.8,label='Legacy map brightness')
fig.suptitle('Rejected preliminary starlet candidate: noise admitted as source feedback\nTrue astronomical signal is zero in every panel; models were never applied to PTC',fontsize=13)
fig.savefig(H/'NULL_ADMISSION.png',dpi=160);plt.close(fig)
