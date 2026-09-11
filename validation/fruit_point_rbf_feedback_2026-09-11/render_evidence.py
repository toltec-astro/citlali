#!/usr/bin/env python3
"""Static scientific comparisons, no feedback or gate decisions."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import sparse
import rbf
b=rbf.base;H=Path(__file__).resolve().parent
COLORS={'P':'#777777','G':'#007c91','R1':'#e4a044','R2':'#ab2850'}
def read(p):return json.loads(Path(p).read_text())
def render():
    out=H/'figures';out.mkdir(exist_ok=True);r1=read(H/'NUMERICAL_EVIDENCE_R0.1.json');r2=read(H/'NUMERICAL_EVIDENCE_R0.2.json');root=Path(r2['root']);geo=np.load(root/'geometry.npz');x=geo['x'];y=geo['y'];D=geo['D'];shape=tuple(geo['shape']);extent=[x.min()-1,x.max()+1,y.min()-1,y.max()+1]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
    def save(fig,name):fig.savefig(out/(name+'.png'),bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,case in enumerate(['coma_bright','coma_half']):
        for a,array in enumerate(b.ARRAYS):
            aa=ax[row,a]
            for label,record,arm in [('P',r2,'P'),('G',r2,'G'),('R1',r1,'R'),('R2',r2,'R')]:
                rr=[v for v in record['response'] if v['case']==case and v['array']==array and v['arm']==arm]
                aa.plot([v['cold_map_seconds'] for v in rr],[100*v['relative_L2'] for v in rr],'.-',color=COLORS[label],label=label,alpha=.85)
            aa.axhline(15 if row==0 else 25,color='black',lw=.7,ls='--',label='fidelity ceiling');aa.set_title(f'{case.replace("_"," ")} · {array}');aa.set_xlabel('Time to map, setup included (s)');aa.set_ylabel('Aperture image error (%)');aa.grid(alpha=.15)
    ax[0,0].legend(ncol=3,fontsize=8);fig.suptitle('Comatic output recovery across retained passes\nPaired response; fixed aperture and outer-plane correction; lower error is better');save(fig,'coma_error_vs_time')
    for case in ['coma_bright','coma_half']:
        truth=np.load(root/f'{case}_truth.npz')['truth'];fig,ax=plt.subplots(3,5,figsize=(14,8),layout='constrained')
        for a,array in enumerate(b.ARRAYS):
            values=[truth[a]]
            for arm in ['P','G','R']:
                values.append(np.load(root/case/arm/'pass06_maps.npz')['total'][a]-np.load(root/'null20260911'/arm/'pass06_maps.npz')['total'][a])
            values.append(np.load(root/case/'R/pass06_maps.npz')['next_model'][a]);vmax=truth[a].max();norm=TwoSlopeNorm(0,-.25*vmax,vmax)
            for j,(label,z) in enumerate(zip(['Truth','P output − paired null','G output − paired null','R2 output − paired null','R2 next feedback'],values)):
                aa=ax[a,j];im=aa.imshow(z.reshape(shape),origin='lower',extent=extent,norm=norm,cmap='RdBu_r');aa.set_xlim(-53,87);aa.set_ylim(-81,59);aa.set_aspect('equal');aa.set_title(f'{array} · {label}');aa.set_xlabel('x (arcsec)');aa.set_ylabel('y (arcsec)')
                for radius in [15,35,60]:aa.add_patch(plt.Circle((17,-11),radius,fill=False,color='#666666',lw=.5,alpha=.6))
            fig.colorbar(im,ax=ax[a,:],shrink=.8,label='Legacy mJy/beam')
        fig.suptitle(f'{case.replace("_"," ").title()} — seventh output, fixed core/shoulder/tail regions\nPaired difference is a response diagnostic; feedback is an absolute, distinct product');save(fig,case+'_maps')
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for col,array in enumerate(b.ARRAYS):
        for row,case in enumerate(['gauss20260911','gauss20260912']):
            aa=ax[row,col]
            for label,record,arm in [('P',r2,'P'),('G',r2,'G'),('R1',r1,'R'),('R2',r2,'R')]:
                rr=[v for v in record['response'] if v['case']==case and v['array']==array and v['arm']==arm]
                aa.plot([v['cold_map_seconds'] for v in rr],[100*v['amplitude_bias'] for v in rr],'.-',color=COLORS[label],label=label)
            aa.axhspan(-5,5,color='#229966',alpha=.1);aa.set_title(f'Compact seed {case[-8:]} · {array}');aa.set_xlabel('Time to map, setup included (s)');aa.set_ylabel('Fitted amplitude bias (%)');aa.grid(alpha=.15)
    ax[0,0].legend(ncol=4,fontsize=8);fig.suptitle('Compact recovery — fixed ±5% amplitude band');save(fig,'compact_amplitude_vs_time')
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for a,array in enumerate(b.ARRAYS):
        for label,record,style in [('R1',r1,'--'),('R2',r2,'-')]:
            for case,marker in [('coma_bright','o'),('coma_half','s')]:
                rr=[v for v in record['response'] if v['case']==case and v['array']==array and v['arm']=='R'];bias=[100*v['raw_fitted_Aeff_bias'] if v['raw_fitted_Aeff_bias'] is not None else np.nan for v in rr]
                ax[a].plot([v['passes'] for v in rr],bias,marker=marker,ls=style,color=COLORS[label],label=f'{label} {case[5:]}',mfc='none')
        ax[a].set_title(array);ax[a].set_xlabel('Cleaning passes');ax[a].set_ylabel('Raw fitted effective-area bias (%)');ax[a].grid(alpha=.15)
    ax[0].legend(fontsize=8);fig.suptitle('Diagnostic audit of raw fits, including rejected sources\nOpen markers do not indicate a valid focus measurement; admission state is retained in the metrics');save(fig,'concentration_bias')
    fig,ax=plt.subplots(3,4,figsize=(12,9),layout='constrained')
    for a,array in enumerate(b.ARRAYS):
        vals=[np.load(root/'real123424'/arm/'pass06_maps.npz')['total'][a] for arm in ['P','G','R']]+[np.load(root/'real123424/R/pass06_maps.npz')['next_model'][a]]
        v=max(np.nanmax(z[D[a]]) for z in vals);norm=TwoSlopeNorm(0,-.1*v,v)
        for j,(label,z) in enumerate(zip(['P output','G output','R2 output','R2 next feedback'],vals)):
            im=ax[a,j].imshow(z.reshape(shape),origin='lower',extent=extent,norm=norm,cmap='RdBu_r');ax[a,j].set_xlim(-90,90);ax[a,j].set_ylim(-90,90);ax[a,j].set_title(f'{array} · {label}');ax[a,j].set_xlabel('x (arcsec)');ax[a,j].set_ylabel('y (arcsec)')
        fig.colorbar(im,ax=ax[a,:],shrink=.7,label='Legacy mJy/beam')
    fig.suptitle('Real 123424 — seventh output and separate feedback\nNo truth claim; each row shares a display scale');save(fig,'real_maps')
if __name__=='__main__':render()
