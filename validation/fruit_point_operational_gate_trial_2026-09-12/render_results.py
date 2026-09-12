"""Static scientific figures from retained trial products; no inference or replay."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import b, read, HERE, OUT

def main():
    rows=read(OUT/'RECEIPTS.json');lookup={(r['case'],r['arm'],r['pass_index']):r for r in rows}
    colors={'P':'#2579ac','C':'#bc4e29'}
    fig,ax=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    counts=np.zeros((3,2),int)
    for a in range(3):
        for arm in ['P','C']:
            for seed,style in [(20260911,'-'),(20260912,'--')]:
                rr=[r for r in rows if r['case']==f'H_{seed}' and r['arm']==arm]
                available=[r for r in rr if r['next_model_available'] and r['measurement'][a]['measurement_available']]
                counts[a,0]+=len(available)
                ax[a,0].plot([r['cumulative_wall_seconds'] for r in available],
                             [r['truth_score'][a]['centroid_error_arcsec'] for r in available],style+'o',
                             color=colors[arm],ms=3)
                points=[]
                for k in range(7):
                    h,d=lookup.get((f'H_{seed}',arm,k)),lookup.get((f'D_{seed}',arm,k))
                    if h and d and h['next_model_available'] and d['next_model_available'] and h['measurement'][a]['measurement_available'] and d['measurement'][a]['measurement_available']:
                        ratio=h['measurement'][a]['fit']['peak']/d['measurement'][a]['fit']['peak']
                        points.append((h['cumulative_wall_seconds']+d['cumulative_wall_seconds'],100*(ratio/(1/.9)-1)))
                if points:
                    counts[a,1]+=len(points)
                    x,y=np.array(points).T;ax[a,1].plot(x,y,style+'o',color=colors[arm],ms=3)
        ax[a,0].axhline(1,color='black',ls=':',lw=1)
        ax[a,1].axhspan(-5,5,color='gray',alpha=.13)
        ax[a,1].axhline(0,color='black',lw=.6)
        ax[a,0].set_ylabel(b.ARRAYS[a]+'\ncentroid error (arcsec)')
        ax[a,1].set_ylabel('Peak-gain error (%)')
        ax[a,0].set_ylim(0,1.12);ax[a,0].set_xlim(0,10)
        ax[a,1].set_ylim(-7,7);ax[a,1].set_xlim(0,20)
        for j in range(2):
            if counts[a,j]==0:
                ax[a,j].text(.5,.25,'No eligible measurements\nunder the frozen common gate',
                             transform=ax[a,j].transAxes,ha='center',fontsize=10,color='#555555')
        for panel in ax[a]:panel.grid(alpha=.2)
    ax[0,0].set_title('Healthy source: usable measured offsets')
    ax[0,1].set_title('Before/after pair: usable peak gain')
    ax[-1,0].set_xlabel('Cumulative full development wall time (s)')
    ax[-1,1].set_xlabel('Sum of before/after development wall times (s)')
    fig.legend(handles=[Line2D([0],[0],color=colors['P'],label='Pixelwise reference'),
                        Line2D([0],[0],color=colors['C'],label='Central starlet'),
                        Line2D([0],[0],color='gray',ls='-',label='Seed 20260911'),
                        Line2D([0],[0],color='gray',ls='--',label='Seed 20260912')],loc='outside lower center',ncol=4)
    fig.suptitle('POINT operational trial — retained iteration sequence\nCentral starlet has no eligible accuracy curves: source trajectories stopped',fontsize=12)
    fig.savefig(HERE/'ACCURACY_VS_TIME.png',dpi=160)
    plt.close(fig)
    g=np.load(OUT/'geometry.npz');x,y=g['x'],g['y'];shape=tuple(g['shape'])
    extent=[x.min()-1,x.max()+1,y.min()-1,y.max()+1]
    fig,axes=plt.subplots(4,3,figsize=(11,12),constrained_layout=True)
    for i,(case,arm) in enumerate([('H_20260911','P'),('H_20260911','C'),('C_20260911','P'),('C_20260911','C')]):
        rr=[r for r in rows if r['case']==case and r['arm']==arm]
        if not rr:continue
        r=rr[-1];k=r['pass_index'];p=np.load(OUT/case/arm/f'pass{k:02d}_maps.npz')
        terminal=(OUT/case/arm/'COMPLETE.json').exists()
        for a in range(3):
            values=np.where(g['S'][a],p['total'][a],np.nan).reshape(shape)
            mask=g['D'][a];v=p['total'][a,mask];lo,hi=np.quantile(v,[.02,.99])
            im=axes[i,a].imshow(values,origin='lower',extent=extent,cmap='RdBu_r',vmin=min(lo,-1),vmax=max(hi,1))
            axes[i,a].add_patch(plt.Circle((0,0),60,fill=False,color='black',ls='--',lw=.8))
            axes[i,a].set_xlim(-90,90);axes[i,a].set_ylim(-90,90)
            axes[i,a].set_title(f'{b.ARRAYS[a]} — {r["measurement"][a]["status"]}',fontsize=9)
            fig.colorbar(im,ax=axes[i,a],shrink=.75,label='Inherited map brightness')
        axes[i,0].set_ylabel(f'{case.split("_")[0]} / {arm}, map {k+1}\n'+('terminal' if terminal else 'failed trajectory; retained map')+'\ny (arcsec)')
    for a in axes[-1]:a.set_xlabel('x (arcsec)')
    fig.suptitle('Reconstructed totals preserve source and deformation evidence\nDashed circle: fixed 60″ feedback/evaluation boundary; panels use separate display scales',fontsize=12)
    fig.savefig(HERE/'TOTAL_MAPS.png',dpi=150)
    plt.close(fig)

if __name__=='__main__':main()
