"""Headless scientific figures for the empirical POINT companion."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_background(path,array,data,bg,beam):
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    s=data['PRIMARY'];good=bg['background_mask'];limit=np.percentile(abs(s[good]),99)
    im=ax[0,0].imshow(s,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax[0,0].contour(bg['source_mask'],[.5],colors='lime');ax[0,0].set_title('Signed signal; green = excluded source region')
    fig.colorbar(im,ax=ax[0,0],label='mJy / nominal beam')
    im=ax[0,1].imshow(bg['scale'],origin='lower',cmap='viridis');ax[0,1].set_title('Background scatter pooled by coverage')
    fig.colorbar(im,ax=ax[0,1],label='Sample standard deviation (mJy / nominal beam)')
    for b in bg['records']:
        use=good&(bg['bin_index']==b['bin']);values=s[use]
        ax[1,0].hist(values,bins=50,histtype='step',density=True,label=f"bin {b['bin']}: n={b['n']}")
    ax[1,0].set(xlabel='Background signal (mJy / nominal beam)',ylabel='Density',title='Source-masked background distributions')
    ax[1,0].legend(fontsize=8)
    centers=[(b['coverage_min']+b['coverage_max_exclusive'])/2 for b in bg['records']]
    ax[1,1].plot(centers,[b['std'] for b in bg['records']],'o-',label='Ordinary scatter')
    ax[1,1].plot(centers,[b['robust_scale'] for b in bg['records']],'s-',label='1.4826 × MAD')
    ax[1,1].plot(centers,[b['median'] for b in bg['records']],'x-',label='Median')
    ax[1,1].set(xlabel='Retained sample coverage',ylabel='mJy / nominal beam',title='No inverse-square-root coverage law imposed')
    ax[1,1].legend();fig.suptitle(f'{array} · empirical background · nominal beam {beam:g} arcsec')
    fig.savefig(path,dpi=150);plt.close(fig)


def draw_measurements(path,array,experiment,bg):
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained');trials=experiment['trials']
    fixed=[t['fixed']['amplitude'] for t in trials if t['fixed']]
    errors=[t['amplitude_error'] for t in experiment['refit_trials'] if t['status']=='success']
    ax[0,0].hist(fixed,bins=20);ax[0,0].set(title='Blank fixed-template amplitudes',xlabel='mJy / nominal beam',ylabel='Locations')
    ax[0,1].hist(errors,bins=20);
    if not errors:ax[0,1].text(.5,.5,'No complete matched 40″ blank patches',ha='center',va='center',transform=ax[0,1].transAxes,fontsize=9)
    ax[0,1].axvline(0,color='k',lw=.8)
    ax[0,1].set(title='Injected Gaussian: refit minus input',xlabel='Amplitude error (mJy / nominal beam)',ylabel='Trials')
    ax[0,2].scatter([t['coverage_ratio'] for t in trials],fixed,s=12)
    ax[0,2].set(title='Coverage matching',xlabel='Template-weighted coverage / source coverage',ylabel='Blank amplitude (mJy / nominal beam)')
    for i,a in enumerate(experiment['apertures']):
        vals=[t['apertures'][i]['value_mJy_per_beam_arcsec2'] for t in trials if t['apertures'][i]['available']]
        ax[1,0].hist(vals,bins=20,histtype='step',label=f"radius {a['source']['radius_measured_fwhm']:g} FWHM")
    ax[1,0].set(title='Blank aperture measurements',xlabel='mJy / nominal beam × arcsec²',ylabel='Locations');ax[1,0].legend(fontsize=8)
    radial=experiment['residual_radial'];ax[1,1].plot([np.mean(r['radius_measured_fwhm']) for r in radial],[r['residual']['mean'] for r in radial],'o-');ax[1,1].axhline(0,color='k',lw=.8)
    ax[1,1].set(title='Observed source minus fitted model',xlabel='Radius / measured major FWHM',ylabel='Mean residual (mJy / nominal beam)')
    sens=bg['sensitivity'];ax[1,2].plot([s['mask_radius_arcsec'] for s in sens],[s['overall']['std'] for s in sens],'o-',label='Ordinary scatter')
    ax[1,2].plot([s['mask_radius_arcsec'] for s in sens],[s['overall']['robust_scale'] for s in sens],'s-',label='Robust scale')
    ax[1,2].set(title='Source-mask sensitivity',xlabel='Excluded radius (arcsec)',ylabel='mJy / nominal beam');ax[1,2].legend(fontsize=8)
    fig.suptitle(f"{array} · conditional map-domain measurements · {len(trials)} overlapping blank locations")
    fig.savefig(path,dpi=150);plt.close(fig)


def draw_splits(path,array,splits,header):
    fig,ax=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    a,b=splits['signals'];nx=a.shape[1];ny=a.shape[0]
    dx=header['CDELT1']*3600;dy=header['CDELT2']*3600
    extent=[(.5-header['CRPIX1'])*dx,(nx+.5-header['CRPIX1'])*dx,(.5-header['CRPIX2'])*dy,(ny+.5-header['CRPIX2'])*dy]
    limit=max(np.nanmax(abs(a)),np.nanmax(abs(b)))
    for i,(s,title) in enumerate([(a,'A: scans 0,2,4'),(b,'B: scans 1,3,5')]):
        im=ax[i].imshow(s,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit);ax[i].set_title(title)
        fig.colorbar(im,ax=ax[i],label='mJy / nominal beam',shrink=.8)
    d=splits['difference'];limit=np.nanpercentile(abs(d),99.5)
    im=ax[2].imshow(d,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit);ax[2].set_title('A − B on common science support')
    fig.colorbar(im,ax=ax[2],label='mJy / nominal beam',shrink=.8)
    im=ax[3].imshow(splits['balance'],origin='lower',extent=extent,cmap='coolwarm',vmin=0,vmax=1);ax[3].set_title('Coverage fraction from A')
    fig.colorbar(im,ax=ax[3],label='A / (A + B)',shrink=.8)
    for a in ax[:3]:a.set(xlim=(30,-30),ylim=(-30,30),xlabel='East offset (arcsec)',ylabel='North offset (arcsec)')
    ax[3].set(xlabel='East offset (arcsec)',ylabel='North offset (arcsec)')
    fig.suptitle(f'{array} · source zoom / full coverage · saved PTC split; no upstream relearning');fig.savefig(path,dpi=150);plt.close(fig)
