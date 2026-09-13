"""Scientific figures of paired effects; no smoothing of either feedback path."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import b,read,HERE,OUT,SOLUTIONS,SAVED,old

def main():
    evidence=read(HERE/'DECISION_EVIDENCE.json')
    rows=evidence['measurements']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True)
    names=[('H' if r['case'].startswith('H_') else 'T' if r['case'].startswith('T_') else 'Real')+'\n'+r['array'] for r in rows]
    for ax,key,availability,mult,small,full,title,ylabel in [
        (axes[0],'peak_relative_change','both_peak_usable',100,.5,5,'Next-map fitted peak','Absolute change (%)'),
        (axes[1],'centroid_change_arcsec','both_centroid_usable',1,.1,1,'Next-map fitted centroid','Movement (arcsec)')]:
        for i,r in enumerate(rows):
            ax.scatter(i,max(abs(r[key])*mult,1e-7),s=50,facecolors='#2563eb' if r[availability] else 'none',edgecolors='#2563eb',linewidths=1.5)
        ax.axhline(small,color='black',lw=1,label='Numerical allocation')
        ax.axhline(full,color='gray',ls='--',lw=1,label='Full operational scale')
        ax.set(yscale='log',title=title,ylabel=ylabel,xticks=range(9),xticklabels=names)
        ax.tick_params(axis='x',labelsize=7)
        ax.grid(axis='y',alpha=.2)
    for i,r in enumerate(rows):
        width=100*np.array(r['width_relative_change'])
        axes[2].scatter(i-.1,width[0],marker='^',color='#b45309',label='Major FWHM' if i==0 else None)
        axes[2].scatter(i+.1,width[1],marker='v',color='#15803d',label='Minor FWHM' if i==0 else None)
    axes[2].axhline(0,color='gray',lw=1)
    axes[2].set(title='Next-map shape diagnostic',ylabel='FWHM change (%)',xticks=range(9),xticklabels=names)
    axes[2].tick_params(axis='x',labelsize=7);axes[2].legend(fontsize=8);axes[0].legend(fontsize=8)
    fig.suptitle('Tighter versus nominal feedback: one PTC-relearning step\nOpen markers: at least one measurement withheld; no trajectory or policy qualification',fontsize=12)
    fig.savefig(HERE/'NEXT_MAP_SENSITIVITY.png',dpi=170)
    plt.close(fig)

    data,estimators=old.setup()
    pairs={(r['case'],r['pass_index'],r['array']):r for r in read(SOLUTIONS/'stopping/RECEIPTS.json')}
    chosen=[('H_20260911',0,2),('T_20260911',1,1),('real123424',0,1)]
    fig,axes=plt.subplots(3,4,figsize=(13.5,10),constrained_layout=True)
    extent=[data.x.min()-1,data.x.max()+1,data.ygrid.min()-1,data.ygrid.max()+1]
    for row,(case,k,a) in enumerate(chosen):
        e=estimators[a];pair=pairs[case,k,b.ARRAYS[a]]
        with np.load(SOLUTIONS/'stopping'/(pair['name']+'.npz')) as z:n,t=z['declared'],z['tight']
        with np.load(OUT/'screen'/f'{case}_pass{k:02d}_difference.npz') as z:core=z['source_core'][a]
        with np.load(OUT/'replay'/f'{case}_after_pass{k:02d}'/'PAIRED_DIFFERENCES.npz') as z:next_delta=z['next_total'][a]
        maximum=max(n.max(),t.max())
        for col,(u,title,mask) in enumerate([(n,'Nominal feedback',e.D),(t,'Tighter feedback',e.D),
                                           (t-n,'Feedback difference',e.D),(next_delta,'Next total-map\ndifference',data.D[a])]):
            kw=dict(cmap='viridis',vmin=0,vmax=maximum) if col<2 else dict(cmap='RdBu_r',vmin=-max(abs(u[mask])),vmax=max(abs(u[mask])))
            im=axes[row,col].imshow(np.where(mask,u,np.nan).reshape(data.shape),extent=extent,origin='lower',interpolation='nearest',**kw)
            axes[row,col].contour(data.x.reshape(data.shape),data.ygrid.reshape(data.shape),core.reshape(data.shape),levels=[.5],colors='white',linewidths=.65)
            if col<2:
                j=int(np.argmax(u));axes[row,col].plot(data.x[j],data.ygrid[j],marker='+',color='red',ms=9,mew=1.2)
            axes[row,col].set(xlim=(-65,65),ylim=(-65,65),xlabel='x (arcsec)',ylabel=('H' if case.startswith('H_') else 'T' if case.startswith('T_') else 'Real')+' '+b.ARRAYS[a]+'\ny (arcsec)')
            if row==0:axes[row,col].set_title(title)
            fig.colorbar(im,ax=axes[row,col],shrink=.72,label='Inherited map units')
    fig.suptitle('Large feedback differences can have smaller central-source effects\nRed crosses: model maxima. White outline: original data-only source core. Color scales differ by panel.',fontsize=12)
    fig.savefig(HERE/'FEEDBACK_AND_NEXT_MAP.png',dpi=150)
    plt.close(fig)

if __name__=='__main__':main()
