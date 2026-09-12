"""Standalone scientific figures of saved-map numerical stability."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import b, read, setup, HERE, OUT

def main():
    evidence = read(HERE/'STOPPING_EVIDENCE.json')
    rows = [r for r in evidence['problems'] if r['comparison'] is not None]
    colors = dict(real123424='#2563eb', H='#15803d', **{'H-shift':'#15803d'},
                  D='#15803d', T='#15803d', C='#b45309', E='#be123c')
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax, xkey, ykey, title in [
        (axes[0],'sampled_peak_relative_difference','brightness_centroid_difference_arcsec','Direct model readouts'),
        (axes[1],'fitted_peak_relative_difference','fitted_centroid_difference_arcsec','Free Gaussian readouts')]:
        seen = set()
        for r in rows:
            c = r['comparison']
            if xkey not in c:
                continue
            state = r['case'].split('_')[0]
            group = 'real123424' if state=='real123424' else 'compact / mild / response loss' if state in ['H','H-shift','D','T'] else 'gross coma' if state=='C' else 'near boundary'
            ax.scatter(max(100*c[xkey],1e-7),max(c[ykey],1e-7),color=colors[state],s=30,alpha=.7,
                       edgecolors='white',linewidths=.3,label=group if group not in seen else None)
            seen.add(group)
        ax.axvline(.5,color='black',lw=1,label='Numerical allocation')
        ax.axhline(.1,color='black',lw=1)
        ax.axvline(5,color='gray',ls='--',lw=1,label='Operational budget (for scale)')
        ax.axhline(1,color='gray',ls='--',lw=1)
        ax.set(xscale='log',yscale='log',xlabel='Peak difference (%)',ylabel='Centroid difference (arcsec)',title=title)
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc='lower center', bbox_to_anchor=(.5,-.1), ncol=3)
    fig.suptitle('Declared 1e-4 stop versus tighter 1e-6 stop\nSame saved problem and uninterrupted optimizer path',fontsize=12)
    fig.savefig(HERE/'STOPPING_STABILITY.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

    data, estimators = setup()
    fig, axes = plt.subplots(3,3,figsize=(10,10),constrained_layout=True)
    extent=[data.x.min()-1,data.x.max()+1,data.ygrid.min()-1,data.ygrid.max()+1]
    for a,e in enumerate(estimators):
        name=f'H_20260911_pass00_{b.ARRAYS[a]}'
        with np.load(OUT/'stopping'/(name+'.npz')) as z:
            loose,tight=z['declared'],z['tight']
        maximum=max(loose.max(),tight.max())
        delta=loose-tight
        for j,(u,title) in enumerate([(loose,'Declared stop'),(tight,'Tighter stop'),(delta,'Declared minus tighter')]):
            canvas=np.where(e.D,u,np.nan).reshape(e.shape)
            kw=dict(vmin=0,vmax=maximum,cmap='viridis') if j<2 else dict(vmin=-max(abs(delta)),vmax=max(abs(delta)),cmap='RdBu_r')
            im=axes[a,j].imshow(canvas,origin='lower',extent=extent,**kw)
            axes[a,j].set(xlim=(-65,65),ylim=(-65,65),xlabel='x (arcsec)',ylabel=f'{b.ARRAYS[a]} · y (arcsec)')
            if a==0:
                axes[a,j].set_title(title)
            fig.colorbar(im,ax=axes[a,j],shrink=.72,label='Inherited map units')
    fig.suptitle('H, first nuisance realization, bootstrap feedback models\nNumerical comparison only; no new FRUIT trajectory',fontsize=13)
    fig.savefig(HERE/'FEEDBACK_STOPPING_MAPS.png',dpi=150)
    plt.close(fig)

if __name__=='__main__':
    main()
