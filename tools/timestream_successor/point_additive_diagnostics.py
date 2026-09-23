"""Read-only applicability evidence and bounded visualization for additive POINT."""
import json
from pathlib import Path
import numpy as np
from dataclasses import replace
from point_additive import learn,NoiseRequest,make_context,variance,experiment,draw_noise,influence,projected_ranges,broad_mode_response
from point_empirical import coordinates,patch,stats
from inspect_forward_map import digest,read_yaml,require
from point_empirical_io import dump


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def save_json(path,value):dump(path,clean(value))


def verify_companion(root,map_hashes):
    root=Path(root).resolve();manifest=root/'FILES_SHA256SUMS'
    for line in manifest.read_text().splitlines():
        h,name=line.split('  ',1);path=(root/name).resolve()
        require(path.is_relative_to(root) and digest(path)==h,'earlier companion manifest mismatch')
    run=json.loads((root/'RUN.json').read_text())
    require(run['parent_maps_unchanged']==map_hashes,'earlier companion has foreign MAP parents')
    return dict(path=str(root),manifest_sha256=digest(manifest),runtime=run)


def regional_summary(model,context):
    results=[]
    for region in model['regions']:
        cv=region['covariance'];L=(cv.shape[0]-1)//2
        # Region zero-lag second moment records amplitude differences separately
        # from its normalized correlation shape; no radial averaging.
        corr=region['correlation'];base=model['empirical'];b=(base.shape[0]-1)//2
        difference=corr-base[b-L:b+L+1,b-L:b+L+1]
        results.append(dict(**region,correlation_RMS_difference=float(np.nanmean(difference**2)**.5)))
    return results


def populations(root,plan,model,context,header,parameters):
    """Use existing MAP operator locations; no new per-sample object or PTC run."""
    root=Path(root);x,y=coordinates(model['scale'].shape,header);p=parameters
    masks={'source_fit':np.hypot(x,y)<=40,'background':model['background_mask']}
    for name,mask in [('north',y>=p[2]),('south',y<p[2]),('east',x>=p[1]),('west',x<p[1])]:masks['background_'+name]=model['background_mask']&mask
    op=np.memmap(root/'operator-pixel.i32',dtype='<i4',mode='r');counts={k:{} for k in masks};scans={k:{} for k in masks}
    for col in plan['ordered_column_domain']:
        pub=read_yaml(root/col['publication_file']);pixels=op[col['first_input_row']:col['first_input_row']+col['input_rows']];pixels=pixels[pixels>=0]
        network=str(pub['network']);scan=str(pub['source_segment']['scan'])
        for name,mask in masks.items():
            n=int(np.sum(mask.ravel()[pixels]));counts[name][network]=counts[name].get(network,0)+n;scans[name][scan]=scans[name].get(scan,0)+n
    def fractions(d):
        total=sum(d.values());return {k:v/total for k,v in d.items()} if total else {}
    return dict(network_counts=counts,network_fractions={k:fractions(v) for k,v in counts.items()},chunk_counts=scans,
        chunk_fractions={k:fractions(v) for k,v in scans.items()},
        scope='exact accepted sample contributions by network/chunk; inherited cleaning response not estimated; no scan-direction vector is inferred from counts')


def blank_checks(model,context,earlier,header,seed,array_id,workers,inner=32):
    half=context['q'].shape[0]//2;fixedroi=context['q']!=0;rows=[]
    for r in earlier['trials']:
        row,col=r['row'],r['column'];s=patch(model['scale'],row,col,half);valid=patch(model['valid'],row,col,half)
        available=bool(s.shape==context['q'].shape and np.all(valid[fixedroi]))
        pred=variance(model['psd'],s,context['q']) if np.isfinite(s[fixedroi]).all() else None
        rows.append(dict(row=row,column=col,measured=r['fixed']['amplitude'],predicted_variance=pred,
            amplitude_supported=available,coverage_ratio=r['coverage_ratio']))
    refits=[]
    # Reproduce the prior source-plus-background experiment's location and its
    # exact source shape. This is not a free fit to pure noise. Cap only the
    # diagnostic budget, choose the first deterministic positions before values.
    for r in earlier['refit_trials'][:4]:
        row,col=r['row'],r['column'];s=patch(model['scale'],row,col,half);valid=patch(model['valid'],row,col,half)
        if not np.isfinite(s[context['roi']]).all():
            refits.append(dict(row=row,column=col,status='unavailable amplitude'));continue
        c=context.copy();c['scale']=np.nan_to_num(s)
        # Free background makes the baseline constant arbitrary for amplitude;
        # use the learned ordinary background mean, matching source+background.
        c['baseline']=context['baseline']-context['reference_parameters'][6]+model['summary']['mean_removed']
        result,trials=experiment(c,inner,seed,array_id+1000+row*1000+col,workers)
        refits.append(dict(row=row,column=col,amplitude_supported=bool(np.all(valid[context['roi']])),model=result,trials=trials,
            existing_refit=r))
    selected=[r for r in rows if r['predicted_variance'] is not None and r['amplitude_supported']]
    return dict(fixed_locations=rows,existing_fixed=earlier['fixed_template'],
        predicted_RMS_fixed_scatter=float(np.sqrt(np.mean([r['predicted_variance'] for r in selected]))) if selected else None,
        matched_existing_scatter=stats([r['measured'] for r in selected]),matched_locations=len(selected),
        refit_locations=refits,existing_refit=earlier['refit'],
        caveat='placements overlap; RMS individual predicted variance and observed centered spatial scatter differ when placement errors correlate or means vary; comparison is descriptive, without independent-placement significance',
        refit_scope='same saved source-plus-background estimator at at most four prespecified existing locations; no pure-noise nonlinear fits')


def draw(out,array,signal,header,model,context,trials,sensitivities,regions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,data,title in [(axs[0,0],np.where(model['background_mask'],signal,np.nan),'Training background; source excluded'),
                           (axs[0,1],np.where(model['valid'],model['scale'],np.nan),'Pixel-noise amplitude (mJy/nominal beam)'),
                           (axs[0,2],model['empirical'],'Mask-corrected directional correlation')]:
        im=ax.imshow(data,origin='lower',cmap='RdBu_r' if ax!=axs[0,1] else 'viridis');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.75)
    axs[0,0].contour(model['source_mask'],levels=[.5],colors='black',linewidths=.7)
    im=axs[1,0].imshow(np.fft.fftshift(model['psd']),origin='lower',cmap='magma');axs[1,0].set_title('Nonnegative normalized 2D spectrum');fig.colorbar(im,ax=axs[1,0],shrink=.75)
    L=(model['empirical'].shape[0]-1)//2;rho=np.fft.ifft2(model['psd']).real;lags=np.arange(-L,L+1)
    axs[1,1].plot(lags*2,model['empirical'][L,:],label='empirical east');axs[1,1].plot(lags*2,model['empirical'][:,L],label='empirical north')
    axs[1,1].plot(lags*2,rho[0,lags%len(rho)],'--',label='model east');axs[1,1].plot(lags*2,rho[lags%len(rho),0],'--',label='model north');axs[1,1].legend(fontsize=8);axs[1,1].set(xlabel='Lag (arcsec)',ylabel='Correlation',title='Both directions; negative values retained')
    noise=draw_noise(model['psd'],context['scale'],[23152389,int(array[1:]),0]);noise=np.where(context['support'],noise,np.nan)
    im=axs[1,2].imshow(noise,origin='lower',cmap='RdBu_r');fig.colorbar(im,ax=axs[1,2],shrink=.75);axs[1,2].set_title('One noise patch at actual source depth')
    fig.suptitle(array+' · conditional additive-noise model · amplitude image is not fitted-amplitude uncertainty')
    fig.savefig(out/'noise-model.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    amps=[r['amplitude']-context['A0'] for r in trials if r['status']=='success'];axs[0,0].hist(amps,bins=24);axs[0,0].set(title='All successful refits, including bound-limited',xlabel='Fitted amplitude − A0 (mJy/nominal beam)')
    z=model['z'][model['background_mask']];axs[0,1].hist(z,bins=60,density=True,histtype='step');v=np.linspace(min(z),max(z),200);axs[0,1].plot(v,np.exp(-v*v/2)/np.sqrt(2*np.pi),label='unit Gaussian');axs[0,1].set(yscale='log',title='Background tails: no clipping',xlabel='Normalized residual');axs[0,1].legend()
    labels=[r['name'] for r in sensitivities if 'experiment' in r];vals=[r['experiment']['candidate_refit_scatter'] for r in sensitivities if 'experiment' in r]
    axs[1,0].plot(range(len(vals)),vals,'o');axs[1,0].set_xticks(range(len(vals)),labels,rotation=45,ha='right');axs[1,0].set(title='Prespecified noise-model sensitivity',ylabel='Conditional refit scatter')
    for r in regions:
        c=np.asarray(r['correlation']);ll=c.shape[0]//2;axs[1,1].plot(np.arange(-ll,ll+1)*2,c[ll,:],label=r['name'])
    axs[1,1].legend(fontsize=8);axs[1,1].set(title='Disjoint region / coverage comparisons',xlabel='East lag (arcsec)',ylabel='Normalized correlation')
    fig.suptitle(array+' · estimator sampling and background representativeness')
    fig.savefig(out/'measurement-and-sensitivity.png',dpi=140);plt.close(fig)
