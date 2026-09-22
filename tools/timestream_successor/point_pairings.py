"""Complete-footprint, conditional first-pass reconstruction comparisons.

No response equivalence, noise-only null, or uncertainty claim is made here.
"""
from itertools import combinations
import numpy as np
from point_empirical import coordinates, fixed_amplitude, aperture_measurements, stats
from point_empirical_io import threshold
from inspect_point_source import gaussian, measure
from inspect_forward_map import require


def balanced_partitions(scans):
    scans=tuple(scans)
    require(len(scans)>=2 and len(scans)%2==0 and len(set(scans))==len(scans),
            'balanced comparisons require unique, even-count chronological scans')
    return [(a,tuple(s for s in scans if s not in a))
            for a in combinations(scans,len(scans)//2) if scans[0] in a]


def footprints(shape,header,parameters):
    x,y=coordinates(shape,header);p=np.asarray(parameters)
    require(x.min()<=-40 and x.max()>=40 and y.min()<=-40 and y.max()>=40,'image truncates declared 40 arcsec disk')
    major=float(max(p[3:5])*np.sqrt(8*np.log(2)));r=np.hypot(x-p[1],y-p[2])
    regions={'core':r<=major,'template':r<=3*major,'fit':np.hypot(x,y)<=40}
    regions['fit_periphery']=regions['fit'] & ~regions['template']
    for factor in (1.,1.5,2.):regions[f'aperture_{factor:g}']=r<=factor*major
    require(all(v.any() for v in regions.values()),'empty measurement footprint')
    require(not np.any(regions['template'] & ~regions['fit']),'template outside fixed fit disk')
    unit=p.copy();unit[0]=1;unit[6]=0
    return regions,gaussian(unit,x,y),major,x,y


def normalized_sum(numerators,normalizations,indices,cut):
    n=np.sum(numerators[list(indices)],axis=0);q=np.sum(normalizations[list(indices)],axis=0)
    support,reference=threshold(q,cut);signal=np.full(n.shape,np.nan)
    np.divide(n,q,out=signal,where=support)
    return dict(numerator=n,normalization=q,support=support,signal=signal,coverage_reference=reference)


def quantiles(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return np.quantile(values,[0,.1,.5,.9,1]).tolist() if len(values) else None


def local_coverage(product,regions):
    q=product['normalization'];valid=product['support'] & np.isfinite(product['signal']);out={}
    for name,roi in regions.items():
        n=int(roi.sum());used=int((roi&valid).sum())
        out[name]=dict(expected_pixels=n,supported_pixels=used,complete=used==n,
            supported_fraction=used/n,normalization_quantiles_including_zero=quantiles(q[roi]),
            normalization_total=float(q[roi].sum()),zero_hit_pixels=int(np.sum(q[roi]==0)))
    return out


def measured(product,regions,template,header,beam,parameters,major,x,y):
    valid=product['support'] & np.isfinite(product['signal']);d=product['signal']
    coverage=local_coverage(product,regions)
    if coverage['fit']['complete']:
        try:
            description,_,_,_=measure(d,valid,header,beam)
            require(description['fit_pixels']==coverage['fit']['expected_pixels'],'fitter changed declared footprint')
            fit=dict(status='success',**description)
        except (ValueError,RuntimeError) as e:fit=dict(status='unavailable',cause=str(e))
    else:fit=dict(status='unavailable',cause='incomplete declared 40 arcsec source-fit footprint; no intersection or reduced-domain fit')
    return dict(coverage_reference=product['coverage_reference'],coverage=coverage,fit=fit,
        fixed_template=fixed_amplitude(d,valid,template,regions['template']),
        apertures=aperture_measurements(d,valid,x,y,parameters[1:3],major,beam,abs(header['CDELT1']*3600)**2),
        fit_periphery_signal=stats(d[valid&regions['fit_periphery']]),
        fit_background_definition='constant fitted jointly on the entire complete 40 arcsec disk; peripheral statistics are descriptive only; no annulus subtraction')


def compared(a,b,ma,mb,regions,template):
    common=a['support'] & b['support'];difference=np.where(common,a['signal']-b['signal'],np.nan)
    qa,qb=a['normalization'],b['normalization'];local={}
    for name,roi in regions.items():
        both=roi&common;positive=roi&(qa>0)&(qb>0)
        local[name]=dict(both_complete=ma['coverage'][name]['complete'] and mb['coverage'][name]['complete'],
            expected_pixels=int(roi.sum()),common_supported_pixels=int(both.sum()),
            ratio_QA_QB_positive_pixels=quantiles(qa[positive]/qb[positive]),ratio_pixels=int(positive.sum()),
            fraction_A_quantiles=quantiles(qa[roi]/np.where((qa+qb)[roi]>0,(qa+qb)[roi],np.nan)))
    fit_ok=ma['fit']['status']==mb['fit']['status']=='success'
    projected=fixed_amplitude(difference,common,template,regions['template'])
    report=dict(local_coverage=local,fit_comparison_available=fit_ok,
        fitted_amplitude_A_minus_B=(ma['fit']['fitted_amplitude_above_constant_mJy_per_nominal_beam']-mb['fit']['fitted_amplitude_above_constant_mJy_per_nominal_beam']) if fit_ok else None,
        fixed_template_A_minus_B=projected,
        fixed_template_unavailable_cause=None if projected else 'incomplete identical template footprint in one or both subsets',
        interpretation='first-pass reconstruction difference; relative cleaning response unresolved; not a noise-only null, final flux repeatability, significance or uncertainty estimate')
    return difference,common,report


def population_summary(member_records,indices,regions):
    """Bounded per-detector counts; original occurrence keys live in inventory."""
    selected=[r for r in member_records if r['scan'] in indices];out={}
    for name in regions:
        bynetwork={}
        for r in selected:
            n=r['network'];v=np.asarray(r['region_counts'][name],dtype=np.int64)
            if n not in bynetwork:bynetwork[n]=v.copy()
            else:
                require(len(bynetwork[n])==len(v),'detector inventory changed between chunks')
                bynetwork[n]+=v
        out[name]={str(n):dict(contributions=int(v.sum()),contributing_detectors=int(np.sum(v>0)),
                              contributions_per_output_detector=v.tolist()) for n,v in sorted(bynetwork.items())}
    return out


def population_comparison(a,b):
    out={}
    for region in a:
        networks={};totala=sum(v['contributions'] for v in a[region].values());totalb=sum(v['contributions'] for v in b[region].values())
        for n in sorted(set(a[region])|set(b[region])):
            va=np.asarray(a[region].get(n,b[region].get(n))['contributions_per_output_detector'])
            vb=np.asarray(b[region].get(n,a[region].get(n))['contributions_per_output_detector'])
            if n not in a[region]:va=np.zeros_like(vb)
            if n not in b[region]:vb=np.zeros_like(va)
            require(len(va)==len(vb),'pair detector occurrence domains differ')
            networks[n]=dict(contributions_A=int(va.sum()),contributions_B=int(vb.sum()),
                detectors_A=int(np.sum(va>0)),detectors_B=int(np.sum(vb>0)),detectors_both=int(np.sum((va>0)&(vb>0))),
                detectors_A_only=int(np.sum((va>0)&(vb==0))),detectors_B_only=int(np.sum((vb>0)&(va==0))),
                contribution_fraction_A=float(va.sum()/totala) if totala else None,
                contribution_fraction_B=float(vb.sum()/totalb) if totalb else None)
        out[region]=dict(networks=networks,detectors_A=sum(v['detectors_A'] for v in networks.values()),
            detectors_B=sum(v['detectors_B'] for v in networks.values()),detectors_both=sum(v['detectors_both'] for v in networks.values()))
    return out
