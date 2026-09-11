#!/usr/bin/env python3
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(record,root,dest):
    d=json.loads(record.read_text());dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    colors={'G':'#697783','J':'#176f9c'};labels={'G':'Independent source fits','J':'Relative-position prior'}
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(case,title) in zip(axes.ravel(),[('gauss20260911','Aligned source · seed 1'),('gauss20260912','Aligned source · seed 2'),('near20260911','Small genuine array offsets'),('source_contaminant20260911','Source plus remote contaminant')]):
        for arm in ['G','J']:
            r=[r for r in d['response'] if r['case']==case and r['array']=='a2000' and r['arm']==arm]
            ax.plot([r['method_map_seconds'] for r in r],[100*r['amplitude_bias'] for r in r],marker='o',ms=4,color=colors[arm],label=labels[arm])
        ax.axhspan(-5,5,color='#57a16b',alpha=.15);ax.axhline(0,lw=.6,color='black',alpha=.4);ax.grid(alpha=.15)
        ax.set_title(title);ax.set_ylabel('a2000 recovered peak error (%)');ax.set_xlabel('Cumulative method map-ready wall time (s)')
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Known-signal response with PTC relearning',fontsize=15)
    fig.text(.5,.014,'Paired background subtraction; free evaluation fits. Green band: provisional ±5% amplitude target.\nContaminant case uses the predeclared 20″ evaluation aperture; the other cases use the full fixed domain.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.96]);fig.savefig(dest/'a2000_recovery.png');plt.close(fig)
    g=np.load(root/'fixed_geometry.npz');shape=tuple(g['shape']);x=g['x'];y=g['y'];extent=[x.min()-1,x.max()+1,y.min()-1,y.max()+1];view=(abs(x)<=45)&(abs(y)<=45)
    fig,axes=plt.subplots(2,3,figsize=(12,8.5))
    for i,(case,label) in enumerate([('real123424','Real 123424'),('source_contaminant20260911','Known source + contaminant')]):
        G=np.load(root/case/'G/pass06_maps.npz');J=np.load(root/case/'J/pass06_maps.npz')
        datasets=[G['total'][2],J['total'][2],J['applied_model'][2]]
        vmin=min(np.nanpercentile(z[view],1) for z in datasets[:2]);vmax=max(np.nanpercentile(z[view],99) for z in datasets[:2])
        for j,z in enumerate(datasets):
            ax=axes[i,j];im=ax.imshow(z.reshape(shape),origin='lower',extent=extent,cmap='viridis',vmin=vmin,vmax=vmax)
            ax.set_xlim(-45,45);ax.set_ylim(-45,45);ax.set_aspect('equal');fig.colorbar(im,ax=ax,shrink=.75,pad=.025)
            if i:ax.scatter([17],[-11],marker='+',color='white',s=90,linewidth=1.3)
            if j==0:ax.set_ylabel(label+'\nElevation offset (arcsec)')
            if i:ax.set_xlabel('Azimuth offset (arcsec)')
    for ax,title in zip(axes[0],['Independent-fit total','Prior-candidate total','Prior-candidate applied feedback']):ax.set_title(title)
    fig.suptitle('a2000: seventh-pass maps and astronomical feedback',fontsize=15)
    fig.text(.5,.014,'Colors: legacy mJy/beam; shared row limits clipped at 1st/99th percentiles. White cross: synthetic source truth only.\nReal a2000 feedback is zero after offset-boundary rejection; the prior does not remove the unrelated map feature.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.96]);fig.savefig(dest/'a2000_maps.png');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('record',type=Path);p.add_argument('root',type=Path);p.add_argument('dest',type=Path);a=p.parse_args();render(a.record,a.root,a.dest)
