#!/usr/bin/env python3
from pathlib import Path
import json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def render(record,root,dest):
    data=json.loads(record.read_text());dest.mkdir(exist_ok=True)
    colors={'P':'#687886','G':'#126c9a'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    fig,axes=plt.subplots(3,2,figsize=(11,10),sharex=True)
    for i,a in enumerate(['a1100','a1400','a2000']):
        for case,ls in [('gauss20260911','-'),('gauss20260912','--')]:
            for arm in ['P','G']:
                r=[x for x in data['response'] if x['case']==case and x['array']==a and x['arm']==arm]
                t=[x['method_map_seconds'] for x in r];b=[100*x['amplitude_bias'] for x in r];w=[100*max(abs(np.array(x['width_bias']))) for x in r]
                label=('Pixelwise' if arm=='P' else 'Coherent source')+(' · seed 1' if ls=='-' else ' · seed 2')
                axes[i,0].plot(t,b,ls,marker='o',ms=3,color=colors[arm],label=label)
                axes[i,1].plot(t,w,ls,marker='o',ms=3,color=colors[arm])
        axes[i,0].axhspan(-5,5,color='#5aaf73',alpha=.13);axes[i,1].axhspan(0,5,color='#5aaf73',alpha=.13)
        axes[i,0].axhline(0,lw=.7,color='black',alpha=.4)
        axes[i,0].set_ylabel(a+'\nAmplitude error (%)');axes[i,1].set_ylabel('Maximum width error (%)')
        for ax in axes[i]:ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);axes[2,0].set_xlabel('Cumulative method map-ready wall time (s)');axes[2,1].set_xlabel('Cumulative method map-ready wall time (s)')
    fig.suptitle('Known injected Gaussian: recovery including PTC relearning',fontsize=15)
    fig.text(.5,.016,'Two declared nuisance realizations; paired null subtraction. Shading marks provisional 5% targets.\nTiming includes preceding candidate fits and excludes reference evaluation-only fits; not full OG latency.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.96]);fig.savefig(dest/'known_signal_recovery.png');plt.close(fig)
    geom=np.load(root/'fixed_geometry.npz');shape=tuple(geom['shape']);x=geom['x'].reshape(shape);y=geom['y'].reshape(shape)
    P=np.load(root/'real123424/P/pass06_maps.npz')['total'];G=np.load(root/'real123424/G/pass06_maps.npz')['total'];M=np.load(root/'real123424/G/pass06_maps.npz')['applied_model']
    extent=[x.min()-1,x.max()+1,y.min()-1,y.max()+1];view=(abs(x.ravel())<=45)&(abs(y.ravel())<=45)
    fig,axes=plt.subplots(3,4,figsize=(13,10))
    for i,a in enumerate(['a1100','a1400','a2000']):
        vmax=max(np.nanpercentile(P[i,view],99),np.nanpercentile(G[i,view],99),1);vmin=min(np.nanpercentile(P[i,view],1),np.nanpercentile(G[i,view],1),0)
        for j,z in enumerate([P[i],G[i],M[i],G[i]-P[i]]):
            if j==3:
                v=max(np.nanpercentile(abs(z[view]),99),1);im=axes[i,j].imshow(z.reshape(shape),origin='lower',extent=extent,cmap='RdBu_r',vmin=-v,vmax=v)
            else:im=axes[i,j].imshow(z.reshape(shape),origin='lower',extent=extent,cmap='viridis',vmin=vmin,vmax=vmax)
            axes[i,j].set_xlim(-45,45);axes[i,j].set_ylim(-45,45);axes[i,j].set_aspect('equal');fig.colorbar(im,ax=axes[i,j],shrink=.70,pad=.025)
            if j==0:axes[i,j].set_ylabel(a+'\nElevation offset (arcsec)')
            if i==2:axes[i,j].set_xlabel('Azimuth offset (arcsec)')
    for ax,title in zip(axes[0],['Pixelwise total','Coherent total','Applied source feedback','Coherent − pixelwise']):ax.set_title(title)
    fig.suptitle('123424: retained seventh-pass products',fontsize=15)
    fig.text(.5,.012,'Central display only; metrics use fixed support. Colors: legacy mJy/beam, clipped at row 1st/99th percentiles; difference ±99th.\nThe fitted background never enters source feedback. A larger peak alone is not evidence of better recovery.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.96]);fig.savefig(dest/'real_point_maps.png');plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('record',type=Path);p.add_argument('root',type=Path);p.add_argument('dest',type=Path);a=p.parse_args();render(a.record,a.root,a.dest)
