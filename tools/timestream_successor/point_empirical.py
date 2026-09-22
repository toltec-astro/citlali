"""Map-domain empirical diagnostics; no MAP mutation or statistical-significance claim.

This module reuses the explicitly selected saved POINT descriptive method. It
is not the unavailable SCI-POINT compatibility/formal-error numerical method.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.signal import fftconvolve
from inspect_point_source import measure, gaussian

METHOD = 'point-saved-descriptive-empirical-v1'

@dataclass(frozen=True)
class Request:
    source_mask_fwhm: float = 3.0
    mask_sweep_fwhm: tuple = (2., 3., 4.)
    coverage_bins: int = 5
    minimum_background_pixels: int = 100
    coverage_tolerance: float = .30
    minimum_stamp_fraction: float = 1.0
    maximum_blanks: int = 100
    minimum_blanks: int = 5
    minimum_spacing_fwhm: float = 2.0
    seed: int = 152389

    def __post_init__(self):
        if not (self.source_mask_fwhm>0 and self.coverage_bins>=2 and
                self.minimum_background_pixels>=2 and 0<self.coverage_tolerance<1 and
                self.minimum_stamp_fraction==1 and self.minimum_blanks>=3 and
                self.maximum_blanks>=self.minimum_blanks and self.minimum_spacing_fwhm>0):
            raise ValueError('invalid empirical request')
        if not self.mask_sweep_fwhm or min(self.mask_sweep_fwhm)<=0:
            raise ValueError('invalid source-mask sensitivity radii')


def stats(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return {'n':0,'median':None,'mean':None,'std':None,'robust_scale':None}
    med=float(np.median(a))
    return dict(n=len(a),median=med,mean=float(a.mean()),
                std=float(a.std(ddof=1)) if len(a)>1 else None,
                robust_scale=float(1.482602218505602*np.median(np.abs(a-med))))


def coordinates(shape,header):
    y,x=np.indices(shape)
    return ((x+1-header['CRPIX1'])*header['CDELT1']*3600,
            (y+1-header['CRPIX2'])*header['CDELT2']*3600)


def background(signal,coverage,support,header,center,mask_scale_fwhm_arcsec,request):
    x,y=coordinates(signal.shape,header)
    good=support & np.isfinite(signal) & np.isfinite(coverage) & (coverage>0)
    r=np.hypot(x-center[0],y-center[1]);source=r<=request.source_mask_fwhm*mask_scale_fwhm_arcsec
    bg=good & ~source
    if bg.sum()<request.minimum_background_pixels:
        raise ValueError('insufficient source-masked background support')
    edges=np.unique(np.quantile(coverage[bg],np.linspace(0,1,request.coverage_bins+1)))
    if len(edges)<2:edges=np.array([coverage[bg][0]-.5,coverage[bg][0]+.5])
    edges[-1]=np.nextafter(edges[-1],np.inf)
    bins=np.searchsorted(edges,coverage,side='right')-1
    scale=np.full(signal.shape,np.nan);robust=scale.copy();median=scale.copy()
    records=[]
    for b in range(len(edges)-1):
        s=stats(signal[bg & (bins==b)])
        s.update(bin=b,coverage_min=float(edges[b]),coverage_max_exclusive=float(edges[b+1]))
        s['available']=s['n']>=request.minimum_background_pixels
        if s['available']:
            use=good & (bins==b);scale[use]=s['std'];robust[use]=s['robust_scale'];median[use]=s['median']
        records.append(s)
    sensitivity=[]
    for factor in request.mask_sweep_fwhm:
        m=good & (r>factor*mask_scale_fwhm_arcsec)
        sensitivity.append(dict(mask_radius_arcsec=factor*mask_scale_fwhm_arcsec,overall=stats(signal[m]),
            bins=[dict(bin=b,**stats(signal[m & (bins==b)])) for b in range(len(edges)-1)]))
    # These spatial comparisons expose failures of coverage-only exchangeability.
    spatial=[]
    for sx,sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        m=bg & ((x-center[0])*sx>=0) & ((y-center[1])*sy>=0)
        spatial.append(dict(quadrant=[sx,sy],bins=[dict(bin=b,**stats(signal[m & (bins==b)])) for b in range(len(edges)-1)]))
    return dict(records=records,sensitivity=sensitivity,spatial=spatial,edges=edges,
                source_mask=source,background_mask=bg,bin_index=bins,scale=scale,
                robust_scale=robust,median=median,valid=np.isfinite(scale))


def stamp_header(header,half):
    h=header.copy();h['CRPIX1']=half+1;h['CRPIX2']=half+1
    return h


def patch(a,row,col,half):
    return a[row-half:row+half+1,col-half:col+half+1]


def source_patch(signal,support,coverage,header):
    step=abs(header['CDELT1'])*3600
    if not np.isclose(step,abs(header['CDELT2'])*3600,rtol=0,atol=1e-10):
        raise ValueError('companion requires square tangent-grid pixels')
    half=int(np.ceil(40/step));row=int(round(header['CRPIX2']-1));col=int(round(header['CRPIX1']-1))
    if abs(header['CRPIX1']-1-col)>1e-8 or abs(header['CRPIX2']-1-row)>1e-8:
        raise ValueError('saved descriptive companion requires integer AST-center pixel')
    if min(row,col)<half or row+half>=signal.shape[0] or col+half>=signal.shape[1]:
        raise ValueError('source fit region is truncated by image boundary')
    h=stamp_header(header,half)
    return patch(signal,row,col,half),patch(support,row,col,half),patch(coverage,row,col,half),h,half


def fixed_amplitude(data,support,template,roi):
    use=support & roi & np.isfinite(data) & np.isfinite(template)
    if not np.array_equal(use,roi) or use.sum()<3:return None
    denominator=float(np.sum(template[use]**2))
    if denominator<=0:return None
    return dict(amplitude=float(np.sum(template[use]*data[use])/denominator),
                pixels=int(use.sum()),template_square_sum=denominator)


def aperture_measurements(data,support,x,y,center,major,beam,pixel_area):
    r=np.hypot(x-center[0],y-center[1]);good=support & np.isfinite(data);out=[]
    for factor in [1.,1.5,2.]:
        ap=r<=factor*major;use=ap & good
        available=bool(use.sum()==ap.sum() and ap.sum()>0)
        # Retain calibrated map units times angular area. No background
        # subtraction, missing-area correction, or aperture correction.
        value=float(np.sum(data[use])*pixel_area) if available else None
        out.append(dict(radius_measured_fwhm=factor,radius_arcsec=factor*major,value_mJy_per_beam_arcsec2=value,
            available=available,aperture_pixels=int(use.sum()),expected_pixels=int(ap.sum())))
    return out


def select_blanks(coverage,support,bgmask,template,roi,half,source_q,beam,step,request):
    # Correlation, not convolution: the footprint can be offset from the
    # AST-center stamp anchor by the fitted source centroid.
    kernel=roi[::-1,::-1].astype(float)
    fractions=fftconvolve(support.astype(float),kernel,mode='same')/kernel.sum()
    # Source-mask intersection is forbidden even if the allowed missing fraction
    # would otherwise conceal it. Outside map coverage is a different fact.
    source_forbidden=support & ~bgmask
    overlaps=fftconvolve(source_forbidden.astype(float),kernel,mode='same')>.5
    weighted=template**2*roi;weighted/=weighted.sum()
    qmetric=fftconvolve(coverage*support,weighted[::-1,::-1],mode='same')
    qsource=float(np.sum(source_q*weighted))
    eligible=(fractions>=request.minimum_stamp_fraction-1e-12)&~overlaps
    eligible &= (qmetric>=qsource*(1-request.coverage_tolerance))&(qmetric<=qsource*(1+request.coverage_tolerance))
    eligible[:half,:]=False;eligible[-half:,:]=False;eligible[:,:half]=False;eligible[:,-half:]=False
    candidates=np.argwhere(eligible);rng=np.random.default_rng(request.seed);rng.shuffle(candidates)
    # First choose a separated subset, then fill additional recorded placements.
    # The latter improve spatial sampling but are never counted as independent.
    extent=np.argwhere(roi);diameter=float(2*np.max(np.linalg.norm(extent-extent.mean(axis=0),axis=1)))
    chosen=[];spacing=request.minimum_spacing_fwhm*beam/step
    for row,col in candidates:
        if all(np.hypot(row-a,col-b)>diameter for a,b in chosen):chosen.append((int(row),int(col)))
        if len(chosen)>=request.maximum_blanks:break
    nonoverlap=chosen.copy()
    for row,col in candidates:
        if all(np.hypot(row-a,col-b)>=spacing for a,b in chosen):chosen.append((int(row),int(col)))
        if len(chosen)>=request.maximum_blanks:break
    for row,col in chosen:
        if not np.all(patch(bgmask,row,col,half)[roi]):
            raise ValueError('selected blank footprint intersects missing/source support')
    return chosen,dict(candidate_count=len(candidates),selected_count=len(chosen),
        fit_footprint_disjoint_subset_count=len(nonoverlap),nonoverlapping_subset=nonoverlap,
        source_template_squared_coverage=qsource,coverage_metric='sum(template**2 * coverage)/sum(template**2); absent support contributes zero',
        minimum_spacing_arcsec=spacing*step,fit_support_fraction_min=request.minimum_stamp_fraction,
        coverage_ratio_range=[1-request.coverage_tolerance,1+request.coverage_tolerance],
        selection='geometry/support/coverage only; deterministic seeded order; no signal clipping',
        dependence='fit footprints can overlap; selected count is not an independent sample count'),qmetric,fractions


def source_experiment(signal,coverage,support,header,beam,parameters,bgmask,request):
    data,valid,q,h,half=source_patch(signal,support,coverage,header)
    x,y=coordinates(data.shape,h);roi=np.hypot(x,y)<=40;step=abs(h['CDELT1'])*3600
    p=np.asarray(parameters);major=float(max(p[3:5])*np.sqrt(8*np.log(2)))
    unit=p.copy();unit[0]=1;unit[6]=0;template=gaussian(unit,x,y)
    fixedroi=np.hypot(x-p[1],y-p[2])<=3*major
    if np.any(fixedroi & ~roi):raise ValueError('fixed-template footprint exceeds saved fit stamp')
    source_fixed=fixed_amplitude(data,valid,template,fixedroi)
    source_ap=aperture_measurements(data,valid,x,y,p[1:3],major,beam,step**2)
    chosen,selection,qmetric,fractions=select_blanks(coverage,support,bgmask,template,fixedroi,half,q,major,step,request)
    trials=[]
    for row,col in chosen:
        d=patch(signal,row,col,half);v=patch(support,row,col,half)
        fixed=fixed_amplitude(d,v,template,fixedroi)
        apertures=aperture_measurements(d,v,x,y,p[1:3],major,beam,step**2)
        trials.append(dict(row=row,column=col,coverage_ratio=float(qmetric[row,col]/selection['source_template_squared_coverage']),
            fit_support_fraction=float(fractions[row,col]),fixed=fixed,apertures=apertures))
    refit_locations,refit_selection,qmetric,fractions=select_blanks(coverage,support,bgmask,template,roi,half,q,major,step,request)
    refits=[]
    for row,col in refit_locations:
        d=patch(signal,row,col,half);v=patch(support,row,col,half)&roi
        record=dict(row=row,column=col,coverage_ratio=float(qmetric[row,col]/refit_selection['source_template_squared_coverage']),support_fraction=float(fractions[row,col]))
        try:
            desc,fit,_,_=measure(d+p[0]*template,v,h,beam)
            record.update(status='success',parameters=fit.tolist(),amplitude_error=float(fit[0]-p[0]),
                centroid_error_arcsec=(fit[1:3]-p[1:3]).tolist(),constraint_active=desc['fit_constraint_active'])
        except (ValueError,RuntimeError) as e:record.update(status='failed',cause=str(e))
        refits.append(record)
    fixedstats=stats([t['fixed']['amplitude'] for t in trials if t['fixed'] is not None])
    errors=stats([t['amplitude_error'] for t in refits if t['status']=='success'])
    available=fixedstats['n']>=request.minimum_blanks
    apresults=[]
    for i,a in enumerate(source_ap):
        s=stats([t['apertures'][i]['value_mJy_per_beam_arcsec2'] for t in trials if t['apertures'][i]['available']])
        apresults.append(dict(source=a,blank=s,scatter_available=s['n']>=request.minimum_blanks))
    model=gaussian(p,x,y);residual=data-model
    radial=[];r=np.hypot(x-p[1],y-p[2])
    for lo,hi in zip([0,1,2,3],[1,2,3,4]):
        use=valid&roi&(r>=lo*major)&(r<hi*major)
        radial.append(dict(radius_measured_fwhm=[lo,hi],residual=stats(residual[use])))
    return dict(method=METHOD,request=asdict(request),selection=selection,source_parameters=p.tolist(),
        measured_major_fwhm_arcsec=major,fixed_template_radius_arcsec=3*major,
        fixed_template=dict(source=source_fixed,blank=fixedstats,scatter_available=available,
            disjoint_subset_blank=stats([t['fixed']['amplitude'] for t in trials if (t['row'],t['column']) in selection['nonoverlapping_subset'] and t['fixed'] is not None])),
        refit=dict(injected_amplitude=float(p[0]),error=errors,selection=refit_selection,
            scatter_available=errors['n']>=request.minimum_blanks and len(refits)==errors['n'] ,
            failed=sum(t['status']!='success' for t in refits),constrained=sum(t.get('constraint_active',False) for t in refits),
            interpretation='complete 40 arcsec map-domain fit footprints; scatter/bias relative to injected fitted Gaussian; unavailable if too few patches, any failed refit; finite bound-limited outcomes stay included and counted; not observed-source truth or upstream recovery'),
        apertures=apresults,residual_radial=radial,source_residual=stats(residual[valid&roi]),trials=trials,refit_trials=refits)
