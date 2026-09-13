"""Static POINT output/timing figures; no new inference or reductions."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import b,read,HERE,OUT

COLORS={'P':'#186cab','C':'#d95c20'}

def main():
    rows=read(OUT/'RECEIPTS.json');lookup={(r['case'],r['arm'],r['pass_index']):r for r in rows}
    evidence=read(HERE/'DECISION_EVIDENCE.json')
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for a,name in enumerate(b.ARRAYS):
        for arm in ['P','C']:
            color=COLORS[arm]
            for seed in [20260911,20260912]:
                for case in ['H','H-shift','D']:
                    rr=[lookup.get((f'{case}_{seed}',arm,k)) for k in range(7)]
                    rr=[r for r in rr if r and r['truth_score'][a].get('centroid_error_arcsec') is not None]
                    x=[r['cumulative_wall_seconds'] for r in rr];y=[r['truth_score'][a]['centroid_error_arcsec'] for r in rr]
                    axes[0,a].plot(x,y,color=color,alpha=.45,lw=.9)
                    for r,xx,yy in zip(rr,x,y):
                        ok=r['measurement'][a]['judgments']['centroid_usable']
                        axes[0,a].scatter(xx,yy,s=15,edgecolor=color,facecolor=color if ok else 'white',zorder=3)
                rr=[(lookup.get((f'H_{seed}',arm,k)),lookup.get((f'D_{seed}',arm,k))) for k in range(7)]
                xx=[];yy=[];available=[]
                for h,d in rr:
                    if not h or not d:continue
                    hm,dm=h['measurement'][a],d['measurement'][a]
                    if 'fit' not in hm['original'] or 'fit' not in dm['original']:continue
                    xx.append(h['cumulative_wall_seconds']+d['cumulative_wall_seconds'])
                    yy.append(100*(hm['original']['fit']['peak']/dm['original']['fit']['peak']/(1/.9)-1))
                    available.append(hm['judgments']['peak_response_usable'] and dm['judgments']['peak_response_usable'])
                axes[1,a].plot(xx,yy,color=color,alpha=.7,lw=1)
                for x,y,ok in zip(xx,yy,available):axes[1,a].scatter(x,y,s=22,edgecolor=color,facecolor=color if ok else 'white',zorder=3)
        axes[0,a].set_title(name)
        if a==1:
            axes[0,a].set_yscale('symlog',linthresh=1)
            axes[0,a].set_yticks([0,.5,1,10,60],['0','0.5','1','10','60'])
            axes[0,a].set_title(name+' (log scale above 1 arcsec)')
        axes[0,a].axhline(1,color='black',ls='--',lw=.8)
        axes[1,a].axhspan(-5,5,color='#39904a',alpha=.1)
        axes[1,a].axhline(0,color='gray',lw=.5)
        axes[0,a].set_xlabel('Cumulative wall time per observation (s)')
        axes[1,a].set_xlabel('Sum of paired cumulative wall times (s)')
        for ax in axes[:,a]:ax.grid(alpha=.2)
    axes[0,0].set_ylabel('H / shifted H / D centroid error (arcsec)')
    axes[1,0].set_ylabel('H/D peak-ratio error (%)')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=COLORS[a],label=a) for a in ['P','C']]+[Line2D([],[],marker='o',color='gray',ls='',label='usable',markerfacecolor='gray'),Line2D([],[],marker='o',color='gray',ls='',label='withheld; raw error',markerfacecolor='white')]
    fig.legend(handles=handles,loc='outside lower center',ncol=4)
    fig.suptitle('All seven retained passes — no truth-selected terminal\nP: pixelwise reference; C: fixed nominal starlet')
    fig.savefig(HERE/'POINT_RECOVERY_VS_TIME.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,5.5),layout='constrained')
    keys=[('U2_gain','H_20260911/D_20260911','H/D · seed 1'),('U2_gain','H_20260912/D_20260912','H/D · seed 2'),('U2_U4_unchanged','H_seed2/seed1','Unchanged H'),('U5_health','T_20260911/H_20260911','T/H · seed 1'),('U5_health','T_20260912/H_20260912','T/H · seed 2')]
    for a,name in enumerate(b.ARRAYS):
        ax=axes[a]
        for i,(use,case,label) in enumerate(keys):
            for arm,offset in [('P',-.12),('C',.12)]:
                s=next(s for s in evidence['use_scores'] if s['use']==use and s['case']==case and s['arm']==arm and s['array']==name)
                if s['raw_relative_error'] is not None:
                    ax.scatter(100*s['raw_relative_error'],i+offset,s=65,edgecolor=COLORS[arm],facecolor=COLORS[arm] if s['available'] else 'white',lw=1.5)
        ax.axvspan(-5,5,color='#39904a',alpha=.12);ax.axvline(0,color='gray',lw=.6)
        ax.set_yticks(range(len(keys)),[k[2] for k in keys]);ax.invert_yaxis();ax.grid(axis='x',alpha=.25)
        ax.set_xlabel('Error relative to imposed peak ratio (%)');ax.set_title(name)
    fig.legend(handles=handles,loc='outside lower center',ncol=4)
    fig.suptitle('Fixed terminal pass 6 — missing usable ratios remain failures')
    fig.savefig(HERE/'TERMINAL_PEAK_RATIOS.png',dpi=160);plt.close(fig)
    with np.load(OUT/'geometry.npz') as z:x,y,shape,D=z['x'],z['y'],tuple(z['shape']),z['D']
    p=np.load(OUT/'real123424/P/pass06_maps.npz')['total'];c=np.load(OUT/'real123424/C/pass06_maps.npz')['total']
    fig,axes=plt.subplots(3,3,figsize=(11,10),layout='constrained')
    for a,name in enumerate(b.ARRAYS):
        vmax=np.nanpercentile(np.r_[p[a,D[a]],c[a,D[a]]],99.5)
        extent=[x.min()-1,x.max()+1,y.min()-1,y.max()+1]
        for j,(z,label) in enumerate([(p,'P'),(c,'C'),(c-p,'C minus P')]):
            im=axes[j,a].imshow(z[a].reshape(shape),origin='lower',extent=extent,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
            axes[j,a].set(xlim=(-90,90),ylim=(-90,90),title=f'{name} · {label}',xlabel='x (arcsec)',ylabel='y (arcsec)')
            axes[j,a].add_patch(plt.Circle((0,0),60,fill=False,color='black',lw=.6))
        fig.colorbar(im,ax=axes[:,a],shrink=.6,label='Inherited map units (mJy/beam)')
    fig.suptitle('Discovery 123424: terminal total maps and signed differences\nDiagnostic display; no real-data truth or qualified morphology claim')
    fig.savefig(HERE/'DISCOVERY_TOTAL_MAPS.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
