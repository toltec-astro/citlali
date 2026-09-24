"""Unchanged POINT measurements and first-cycle comparisons; no feedback apply."""
from pathlib import Path
import numpy as np
from astropy.io import fits
from point_empirical import coordinates,source_patch,aperture_measurements,stats
from point_admission import admit,fit_diagnostic,projection
from point_empirical_io import write_fits
from point_additive_diagnostics import save_json
from inspect_forward_map import require,read_yaml,digest


def basis_distance(old_path,new_path,rows,columns):
    """Saved native matrices are row-major, including column-major Eigen bases."""
    if not rows or not columns:return None
    old=np.fromfile(old_path,dtype='<f8').reshape(rows,columns)
    new=np.fromfile(new_path,dtype='<f8').reshape(rows,columns)
    for b in [old,new]:
        require(np.isfinite(b).all() and np.allclose(np.einsum('ij,ik->jk',b,b),np.eye(columns),rtol=0,atol=1e-10),'saved basis is not the declared orthonormal column matrix')
    return float(np.linalg.norm(np.einsum('ik,jk->ij',old,old)-np.einsum('ik,jk->ij',new,new)))


def learned_state(cycle,receipt):
    rows=[]
    for s in receipt['segments']:
        source=s['source_segment'];name=next(n for n in source['files'] if n.endswith('-basis.f64'))
        index=name.split('-')[1];old=Path(s['source_root'])/name;new=cycle/f"network-{source['network']}-segment-{index}"/'basis.f64'
        require(digest(old)==source['files'][name] and digest(new)==s['files']['basis.f64'],'saved basis identity changed')
        row={k:s.get(k) for k in ['converged','stopping_reason','rank','iterations','retained','output_cause_changed','input_eligibility_changed','projection_support_counts']}
        row.update(network=source['network'],scan=source['scan'],parent_converged=source['converged'],
            basis_projector_Frobenius_difference=basis_distance(old,new,source['basis_rows'],source['basis_columns']),
            comparison_definition='Frobenius norm of BB^T difference; verified row-major saved orthonormal bases; not raw eigenvector difference',
            native_diagnostic_value=s.get('basis_projector_Frobenius_difference'))
        rows.append(row)
    return rows


def load(path):
    with fits.open(path,checksum=True,memmap=False) as f:
        return {h.name:np.array(h.data) for h in f},f[0].header.copy()


def patch_map(path):
    data,h=load(path)
    s,v,n,ch,_=source_patch(data['PRIMARY'],data['SCI_SUPPORT'].astype(bool),data['NORMALIZATION'],h)
    return s,v,n,ch


def compare_zero(original,replay,beam,*,precomputed_fits=None):
    a,ha=load(original);b,hb=load(replay)
    require(a['PRIMARY'].shape==b['PRIMARY'].shape,'zero MAP shape changed')
    for key in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','RADESYS','EQUINOX']:
        require(ha[key]==hb[key],f'zero WCS changed: {key}')
    for name in ['SCI_SUPPORT','NORM_SUPPORT','NORMALIZATION','HITS']:
        require(np.array_equal(a[name],b[name]),f'zero support/normalization changed: {name}')
    mask=a['SCI_SUPPORT'].astype(bool);error=float(np.max(np.abs(a['PRIMARY'][mask]-b['PRIMARY'][mask])))
    require(np.allclose(a['PRIMARY'],b['PRIMARY'],rtol=1e-11,atol=1e-9,equal_nan=True),'zero signal mismatch')
    require(np.allclose(a['NUMERATOR'],b['NUMERATOR'],rtol=1e-11,atol=1e-7),'zero numerator mismatch')
    if precomputed_fits is None:
        fa=fit_diagnostic(a['PRIMARY'],mask,ha,beam);fb=fit_diagnostic(b['PRIMARY'],mask,hb,beam)
    else:
        fa,fb=precomputed_fits
    require(fa['status']==fb['status']=='success','zero source fit unavailable')
    require(np.allclose(fa['parameters'],fb['parameters'],rtol=1e-8,atol=1e-8),'zero source fit changed')
    return dict(pass_control=True,maximum_signal_difference=error,support_hits_normalization_exact=True,
                tolerances=dict(signal_absolute=1e-9,signal_relative=1e-11,numerator_absolute=1e-7,fit_absolute_relative=1e-8),
                tolerance_basis='floating arithmetic/serialization allowances; materially smaller than empirical pixel scales; support and coefficients exact',source_fit=fb)


def replacement(total,support,scale,noise_valid,header,beam):
    fit=fit_diagnostic(total,support,header,beam)
    if fit['status']!='success' or fit.get('constrained',False):
        return fit,None
    return fit,admit(total,support,scale,noise_valid,header,np.asarray(fit['parameters']),3.,1.)


def diagnose(array,maps,admission,additive,zero,cycle,out):
    out.mkdir(exist_ok=False)
    old,vo,coverage,h=patch_map(maps/'map.fits');z,vz,_,_=patch_map(zero/'total.fits');total,vt,ct,ht=patch_map(cycle/'total.fits')
    require(all(h[k]==ht[k] for k in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2']),'source stamp WCS changed')
    m0,_=load(admission/'candidate-reference.fits');relative,_=load(admission/'source-relative.fits')
    noise=np.load(additive/'model.npz');q=noise['weights'];scale=relative['PIXEL_SCALE'];nv=relative['NOISE_VALID'].astype(bool)
    beam=read_yaml(maps/'bundle.yaml')['nominal_beam_fwhm_arcsec']
    fit,m1=replacement(total,vt,scale,nv,h,beam)
    measurements={};planes={'M0':(old,vo),'zero':(z,vz),'T1':(total,vt),'m0':(m0['CANDIDATE'],m0['VALID'].astype(bool))}
    if m1 is not None:
        planes['m1']=(m1['candidate'],m1['valid'])
        write_fits(out/'m1.fits',h,[('CANDIDATE',m1['candidate'],'mJy/nominal beam'),('ADMITTED',m1['selected'].astype('u1'),'1'),('VALID',m1['valid'].astype('u1'),'1'),('RATIO',m1['z'],'1')],companion='FRUIT-UNAPPLIED')
    p=fit_diagnostic(old,vo,h,beam)['parameters'];x,y=coordinates(old.shape,h);major=max(p[3:5])*np.sqrt(8*np.log(2))
    for name,(signal,support) in planes.items():
        f=fit_diagnostic(signal,support,h,beam);bg=f['parameters'][6] if f['status']=='success' and name in ['M0','zero','T1'] else 0.
        relative_signal=signal-bg
        measurements[name]=dict(fit=f,frozen_template_projection=projection(relative_signal,support,q),
            fixed_template_definition='original unit-peak template q=t/sum(t²), original 3-major-FWHM footprint; subtract own fitted constant for science maps; candidates already source-relative',
            template_unit='mJy/nominal-beam',apertures=aperture_measurements(relative_signal,support,x,y,p[1:3],major,beam,4.),
            aperture_definition='fixed original centroid/radii, sum source-relative brightness times 4 arcsec²; no missing-area or aperture correction; not integrated flux')
    r,vr,_,_=patch_map(cycle/'residual.fits');a,va,_,_=patch_map(cycle/'restored.fits')
    common=vt&vo;delta=np.where(common,total-old,np.nan)
    # The owner-approved containing-pixel projection/remapping identity is
    # measured on actual new output support, rather than assumed.
    model_supported=va&m0['VALID'].astype(bool)
    remap_error=float(np.max(np.abs(a[model_supported]-m0['CANDIDATE'][model_supported])))
    receipt=read_yaml(cycle/'receipt.yaml')
    changes=dict(added=int((m1['selected']&~m0['ADMITTED'].astype(bool)).sum()),removed=int((m0['ADMITTED'].astype(bool)&~m1['selected']).sum())) if m1 else None
    bgdomain=(np.hypot(x-p[1],y-p[2])>3*major)&(np.hypot(x,y)<=40)
    report=dict(array=array,measurements=measurements,m1_status=m1['summary'] if m1 else dict(status='withheld',cause='failed/unavailable/constrained total fit'),replacement_changes=changes,
        pixel_scale='frozen first-pass additive companion-r2; selection control only, not post-FRUIT uncertainty or calibrated significance',
        source_support=dict(lost=int((vo&~vt).sum()),added=int((vt&~vo).sum())),
        projected_remapped_model_maximum_difference=remap_error,decomposition_error=receipt['decomposition_max_abs_mJy_per_nominal_beam'],
        source_masked_background_in_40arcsec_stamp={n:stats(s[v&bgdomain]) for n,(s,v) in planes.items() if n in ['M0','zero','T1']},
        PTC_segments=learned_state(cycle,receipt))
    save_json(out/'measurements.json',report)
    write_fits(out/'comparison.fits',h,[('T1_MINUS_M0',delta,'mJy/nominal beam'),('SOURCE_COMMON',common.astype('u1'),'1'),('M0_SUPPORT',vo.astype('u1'),'1'),('T1_SUPPORT',vt.astype('u1'),'1'),('COVERAGE_DIFFERENCE',ct-coverage,'count')],companion='FRUIT-COMPARISON')
    figures(out,array,planes,r,a,delta,coverage,ct,h,m0,m1,p)
    report['full_map_support']=full_support(out,array,maps/'map.fits',cycle/'total.fits')
    save_json(out/'measurements.json',report)
    return report


def full_support(out,array,old_path,new_path):
    import matplotlib.pyplot as plt
    old,ho=load(old_path);new,hn=load(new_path)
    for k in ['CDELT1','CDELT2','CRVAL1','CRVAL2']:require(ho[k]==hn[k],'unaligned full MAP grid')
    shape=tuple(max(a,b) for a,b in zip(old['PRIMARY'].shape,new['PRIMARY'].shape))
    def pad(v,fill):
        result=np.full(shape,fill,dtype=v.dtype);r=(shape[0]-v.shape[0])//2;c=(shape[1]-v.shape[1])//2
        result[r:r+v.shape[0],c:c+v.shape[1]]=v;return result
    a,b=pad(old['PRIMARY'],np.nan),pad(new['PRIMARY'],np.nan)
    vo,vn=pad(old['SCI_SUPPORT'],0).astype(bool),pad(new['SCI_SUPPORT'],0).astype(bool)
    common=vo&vn;difference=np.where(common,b-a,np.nan);coverage=pad(new['HITS'],0)-pad(old['HITS'],0)
    changes=vo.astype(int)+2*vn.astype(int);maximum=max(float(np.nanmax(a)),float(np.nanmax(b)))
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,(title,v,lo,hi) in zip(axes.flat,[('M0',a,-30,maximum),('T1',b,-30,maximum),('T1 − M0',difference,-40,40),('T1 outskirts',b,-40,40),('coverage change',coverage,-max(1,np.max(abs(coverage))),max(1,np.max(abs(coverage)))),('support: 1 lost, 2 added, 3 both',changes,0,3)]):
        im=ax.imshow(v,origin='lower',vmin=lo,vmax=hi,cmap='RdBu_r' if lo==-hi else 'magma');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle(array+' full footprint — aligned pixels; brightness mJy/nominal beam');fig.savefig(out/'full-footprint.png',dpi=150);plt.close(fig)
    return dict(original_pixels=int(vo.sum()),new_pixels=int(vn.sum()),lost=int((vo&~vn).sum()),added=int((vn&~vo).sum()),coverage_changed_pixels=int(np.count_nonzero(coverage)),maximum_coverage_difference=int(np.max(abs(coverage))))


def figures(out,array,planes,r,a,delta,c0,c1,h,m0,m1,p):
    import matplotlib.pyplot as plt
    x,y=coordinates(delta.shape,h);extent=[x[0,0]+1,x[0,-1]-1,y[0,0]-1,y[-1,0]+1]
    vmax=max(np.nanmax(planes[n][0]) for n in ['M0','zero','T1']);fig,axes=plt.subplots(3,3,figsize=(13,11),constrained_layout=True)
    entries=[('M0',planes['M0'][0],-30,vmax),('zero',planes['zero'][0],-30,vmax),('T1',planes['T1'][0],-30,vmax),
        ('R1: cleaned residual',r,-40,40),('A1: restored model',a,-30,vmax),('T1 − M0',delta,-max(1,np.nanmax(abs(delta))),max(1,np.nanmax(abs(delta)))),
        ('m0',planes['m0'][0],-30,vmax),('m1 (unapplied)',planes.get('m1',(np.full(delta.shape,np.nan),None))[0],-30,vmax),('new − old coverage',c1-c0,-max(1,np.max(abs(c1-c0))),max(1,np.max(abs(c1-c0))))]
    for ax,(name,v,lo,hi) in zip(axes.flat,entries):
        im=ax.imshow(v,origin='lower',extent=extent,vmin=lo,vmax=hi,cmap='RdBu_r' if lo==-hi else 'magma');ax.set_title(name);ax.set_xlabel('east offset [arcsec]');ax.set_ylabel('north offset [arcsec]');fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle(array+' — first feedback cycle; brightness mJy/nominal beam, coverage counts');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True);rad=np.hypot(x-p[1],y-p[2]);edges=np.arange(0,42,2);profiles={}
    for name,(v,support) in planes.items():
        profile=[float(np.mean(v[support&(rad>=lo)&(rad<hi)])) if np.any(support&(rad>=lo)&(rad<hi)) else np.nan for lo,hi in zip(edges[:-1],edges[1:])];profiles[name]=profile
        axes[0].plot((edges[:-1]+edges[1:])/2,profile,label=name);axes[1].plot((edges[:-1]+edges[1:])/2,profile,label=name)
    axes[0].legend();axes[1].set_ylim(-40,40)
    for ax in axes[:2]:ax.set_xlabel('radius from original centroid [arcsec]');ax.set_ylabel('mean brightness [mJy/nominal beam]');ax.axhline(0,color='k',lw=.5)
    axes[0].set_title('Fixed-centroid radial profiles');axes[1].set_title('Outskirts and signed bowls')
    changed=np.zeros(delta.shape);changed[m0['ADMITTED'].astype(bool)]=1
    if m1 is not None:changed[m1['selected']]+=2
    axes[2].imshow(changed,origin='lower',extent=extent,vmin=0,vmax=3,cmap='viridis');axes[2].set_title('Support: 0 neither, 1 removed, 2 added, 3 both');axes[2].set_xlabel('east [arcsec]');axes[2].set_ylabel('north [arcsec]')
    fig.savefig(out/'profiles-support.png',dpi=150);plt.close(fig);save_json(out/'profiles.json',dict(radius_arcsec=(edges[:-1]+edges[1:])/2,profiles=profiles,center=p[1:3]))
