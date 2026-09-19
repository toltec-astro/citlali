"""Compact scientific figures. Numerical claims live in the result receipt."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plots(r,out):
    profiles=[p for p in r['profiles'] if p['state']=='available']
    if not profiles:return
    fig,axes=plt.subplots(len(profiles),3,figsize=(14,4*len(profiles)),squeeze=False)
    for ax,p in zip(axes,profiles):
        f=np.asarray(p['frequency_hz']);e=p['pooled'];d=len(r['cohort_columns'])
        ax[0].semilogy(f[1:],np.asarray(e['trace_psd'])[1:]/d,label='total / detector',color='0.3')
        ev=np.asarray(e['leading_eigenvalues'])
        ax[0].semilogy(f[1:],ev[1:,0]/d,label='leading mode / detector')
        ax[0].set_ylabel('(mJy/beam)²/Hz');ax[0].legend(fontsize=8)
        for k,c in zip((1,3,10),np.asarray(e['leading_fractions_1_3_10']).T):ax[1].plot(f[1:],c[1:],label=f'first {k}')
        ax[1].set_ylim(0,1.02);ax[1].set_ylabel('Fraction of spectral power');ax[1].legend(fontsize=8)
        locals=[x for x in p['local'] if 'trace_psd' in x]
        for hz in (.5,11,20):
            if hz>f[-1]:continue
            fi=int(np.argmin(abs(f-hz)));xx=[(x['pool']+.5)*p['pool_seconds'] for x in locals]
            ax[2].plot(xx,[x['leading_fractions_1_3_10'][fi][0] for x in locals],'.-',label=f'{f[fi]:.2f} Hz')
        ax[2].set_ylim(0,1.02);ax[2].set_xlabel('Seconds from observation start');ax[2].set_ylabel('Leading fraction, local pools');ax[2].legend(fontsize=8)
        for a in ax[:2]:a.set_xlabel('Frequency (Hz)');a.axvline(11,color='0.5',lw=.7,ls=':')
        for a in ax:a.grid(alpha=.18)
        ax[0].set_title(f'{p["actual_fft_seconds"]:.3f}s Hann · Δf={p["bin_spacing_hz"]:.3f} Hz')
        ax[1].set_title(f'{e["realizations"]} realizations · rank limit {e["estimator_rank_limit"]}')
        ax[2].set_title(f'{p["pool_seconds"]:g}s pools; finite-support selection')
    fig.suptitle(f'152390 · network {r["network"]} · {d} detectors · CAL input SPOD (diagnostic)',y=1)
    fig.tight_layout();fig.savefig(out/'spectral-overview.png',dpi=150);plt.close(fig)
    p=profiles[0];f=np.asarray(p['frequency_hz']);locals=[x for x in p['local'] if 'trace_psd' in x]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,hz in zip(axes[0],(.5,11)):
        fi=int(np.argmin(abs(f-hz)));m=len(p['local']);matrix=np.full((m,m),np.nan);np.fill_diagonal(matrix,1)
        for row in p['pairwise_pool_comparisons']:
            i,j=row['pools'];v=row['by_frequency'][str(fi)]['1']
            if v:matrix[i,j]=matrix[j,i]=v['mean_cos2']
        im=ax.imshow(matrix,vmin=0,vmax=1,origin='lower',cmap='viridis');fig.colorbar(im,ax=ax,label='Leading-subspace overlap cos²')
        ax.set_title(f'{f[fi]:.2f} Hz; {p["pool_seconds"]:g}s pool index');ax.set_xlabel('Pool');ax.set_ylabel('Pool')
    for ax,hz in zip(axes[1],(.5,11)):
        fi=int(np.argmin(abs(f-hz)));xx=[(x['pool']+.5)*p['pool_seconds'] for x in locals]
        ax.plot(xx,[x['to_pooled'][str(fi)]['1']['mean_cos2'] for x in locals],'.-',label='local vs pooled (not independent)')
        ys=[x['split_half'].get(str(fi),{}).get('1') for x in locals]
        ax.plot(xx,[v['mean_cos2'] if v else np.nan for v in ys],'s--',label='separated split-half control')
        ax.set_ylim(0,1.02);ax.set_title(f'{f[fi]:.2f} Hz');ax.set_xlabel('Seconds');ax.set_ylabel('Overlap cos²');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'pattern-stability.png',dpi=150);plt.close(fig)
    # Patterns display complex shape without selecting high-amplitude detectors.
    arrays=np.load(out/f'fft{p["requested_fft_seconds"]:g}-modes.npz')
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    for ax,hz in zip(axes,(.5,11)):
        fi=int(np.argmin(abs(f-hz)));u=arrays[f'pooled_mode_{fi}'][:,0]
        u=u*np.exp(-1j*np.angle(u[np.argmax(abs(u))]))
        ax.plot(u.real,label='real');ax.plot(u.imag,label='imaginary');ax.set_ylabel('Unit-norm loading');ax.set_title(f'Pooled {f[fi]:.2f} Hz; global phase fixed for display only');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Fixed cohort detector position (identities in receipt)');fig.tight_layout();fig.savefig(out/'detector-patterns.png',dpi=150);plt.close(fig)
