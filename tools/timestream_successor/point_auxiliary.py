"""Independent POINT fixed-template window on the existing tangent pixel grid.

B/M0 owns this geometry. Brightness is mJy/nominal beam; offsets/sigmas
are arcseconds. Missing measurement pixels make the projection unavailable;
weights are never renormalized to surviving support. Fit/feedback stay at 40".
"""
import numpy as np
from point_empirical import coordinates
from inspect_point_source import gaussian
from inspect_forward_map import require

SCHEMA = 'point-auxiliary-template-v1'


def geometry(header, shape, parameters):
    p = np.asarray(parameters, dtype=float)
    require(p.shape == (7,) and np.isfinite(p).all() and min(p[3:5]) > 0,
            'invalid auxiliary template parameters')
    step = np.array([header['CDELT1'], header['CDELT2']])*3600
    require(np.isfinite(step).all() and np.all(step != 0), 'invalid auxiliary pixel grid')
    radius = 3*max(p[3:5])*np.sqrt(8*np.log(2))
    # A symmetric origin-centered rectangle contains both the complete intended
    # circle and the old fit/constant operator; it does not move either center.
    half = np.maximum(np.ceil((np.abs(p[1:3])+radius)/np.abs(step)).astype(int),
                      np.array(shape[::-1])//2)
    h = {k:v for k,v in header.items() if k not in ('COMMENT','HISTORY','')}; h.update(CRPIX1=int(half[0]+1), CRPIX2=int(half[1]+1))
    h.update(NAXIS1=int(2*half[0]+1),NAXIS2=int(2*half[1]+1))
    result = dict(schema=SCHEMA, header=h, shape=[int(2*half[1]+1), int(2*half[0]+1)],
                  parameters=p.tolist(), radius_arcsec=float(radius),
                  fit_feedback_radius_arcsec=40., normalization='unit Gaussian divided by sum of squared unit template on complete three-major-FWHM circle',
                  support_policy='all nonzero template pixels required; no clipping or support renormalization')
    return result


def weights(g):
    require(g['schema'] == SCHEMA, 'unknown auxiliary geometry')
    x,y = coordinates(tuple(g['shape']),g['header']); p=np.array(g['parameters'])
    radius=3*max(p[3:5])*np.sqrt(8*np.log(2))
    require(np.isclose(radius,g['radius_arcsec'],rtol=0,atol=1e-12), 'auxiliary footprint changed')
    require(x.min() <= p[1]-radius and x.max() >= p[1]+radius and
            y.min() <= p[2]-radius and y.max() >= p[2]+radius,
            'auxiliary geometry clips intended footprint')
    unit=p.copy();unit[0]=1.;unit[6]=0.
    roi=np.hypot(x-p[1],y-p[2])<=radius
    q=np.where(roi,gaussian(unit,x,y),0.);norm=float(np.sum(q*q))
    require(norm>0 and np.isfinite(norm),'invalid auxiliary normalization')
    return q/norm,roi


def slices(shape, header, g):
    """Exact physical pixel correspondence, without interpolation or data padding."""
    for k in ('CDELT1','CDELT2','CRVAL1','CRVAL2','RADESYS','EQUINOX'):
        require(header.get(k)==g['header'].get(k),'auxiliary full-map WCS differs: '+k)
    offset=np.array([header['CRPIX2']-g['header']['CRPIX2'],
                     header['CRPIX1']-g['header']['CRPIX1']])
    start=np.rint(offset).astype(int)
    require(np.allclose(offset,start,rtol=0,atol=1e-8),'auxiliary grids do not align')
    end=start+g['shape']
    require(np.all(start>=0) and np.all(end<=shape),'auxiliary window outside full map')
    return tuple(slice(int(a),int(b)) for a,b in zip(start,end))


def sample(signal, support, header, g):
    try:
        sl=slices(signal.shape,header,g)
    except ValueError as exc:
        return dict(available=False,reason=str(exc),missing_pixels=None,nonfinite_pixels=None)
    s=np.asarray(signal[sl]);v=np.asarray(support[sl],bool);q,roi=weights(g)
    missing=int(np.sum(roi&~v));nonfinite=int(np.sum(roi&~np.isfinite(s)))
    return dict(available=not (missing or nonfinite), reason=None if not (missing or nonfinite) else 'incomplete auxiliary template support',
                missing_pixels=missing,nonfinite_pixels=nonfinite,signal=s,support=v,
                template_pixels=int(roi.sum()),supported_template_pixels=int(np.sum(roi&v&np.isfinite(s))))


def from_map(path,c):
    if 'auxiliary' not in c:return None
    from point_fruit_diagnostics import load
    data,h=load(path)
    return sample(data['PRIMARY'],data['SCI_SUPPORT'].astype(bool),h,c['auxiliary'])


def projection(c, payload, constant):
    if payload is None:return None,dict(available=False,reason='auxiliary full-map sample not supplied')
    info={k:v for k,v in payload.items() if k not in ('signal','support')}
    if not payload['available']:return None,info
    q,_=weights(c['auxiliary']);use=q!=0
    return float(np.sum(q[use]*(payload['signal'][use]-constant))),info


def embed(operator, header, g):
    """Embed a linear operator, not observed data; absent coefficients are zero."""
    result=np.zeros(g['shape']);sl=slices(tuple(g['shape']),g['header'],dict(header=header,shape=list(operator.shape)))
    result[sl]=operator
    return result


def covariance_spectrum(model,g):
    size=max(g['shape'])
    if size<=49:return model['psd']
    from point_additive import spectrum_from_pairs
    request=model['summary']['request']
    return spectrum_from_pairs(model['empirical'],request['lag_pixels'],size,
                               request['taper_power'],request['domain_factor'])[0]
