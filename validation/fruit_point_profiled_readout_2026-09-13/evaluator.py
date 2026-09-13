"""Unchanged data-only judgments applied to supplied terminal fits, never refitting."""
import copy
import numpy as np
from common import gaussian,analysis,eligible,mad,judgments

def evidence(z,x,y,S,central,O,Q,shape,cal,old):
    beta=np.array(old['original']['background'])
    plane=beta[0]+beta[1]*x/90+beta[2]*y/90
    Y=np.where(S,z-plane,0.)
    bands=analysis(Y.reshape(shape)).reshape(5,-1)
    valid=eligible(S.reshape(shape)).reshape(5,-1)
    stratum=(Q>cal['q_median']).astype(int)
    sigmas=np.zeros((5,2))
    for row in cal['bands']:sigmas[row['band']-1,row['stratum']]=row['scatter']
    positive=valid&central&(bands>=5*sigmas[:,stratum])
    np.testing.assert_array_equal(positive.sum(axis=1),old['original']['positive_counts'])
    assert bool(positive[1:].any())==old['original']['source_present']
    return dict(positive=positive,R=mad(z[O]))

def evaluate(z,x,y,D,central,O,old,evidence,full,inner):
    original={k:copy.deepcopy(old['original'][k]) for k in
              ['source_present','positive_counts','source_evidence','background','support_pixels','exterior_rms']}
    original['measurement_available']=False
    associated=False;probe=dict(available=False)
    if full is None:
        original.update(status='unassessable_measurement' if original['source_present'] else 'no_source_established',
                        fit_error='experimental readout unavailable',distortion_warning=False,boundary_warning=False)
    else:
        fit=copy.deepcopy(full);p=np.array(fit['parameters'])
        G=gaussian(p,x,y);plane=p[6]+p[7]*x/90+p[8]*y/90
        core=central&(G>=.1*max(p[0],0.));denom=np.linalg.norm(G[core])
        shape_error=float(np.linalg.norm((z-plane-G)[core])/denom) if denom>0 else None
        radius=float(np.hypot(*p[1:3]));R=evidence['R']
        interior=np.hypot(x-p[1],y-p[2])<=1.5*max(fit['widths'])
        fraction=float(np.sum(interior&central)/max(int(interior.sum()),1))
        boundary=radius>=52 or fraction<.95
        distorted=bool(max(fit['widths'])>24 or shape_error is None or shape_error>.25)
        adequate=bool(original['source_present'] and p[0]>5*R and not fit['boundary_rejection'] and
                      not boundary and not distorted and int(core.sum())>=10)
        aperture=D&(np.hypot(x-p[1],y-p[2])<=40)
        original.update(fit=fit,fit_radius=radius,empirical_peak_score=float(p[0]/R) if R>0 else None,
            shape_error=shape_error,shape_core_pixels=int(core.sum()),aperture_brightness=float(4*np.sum((z-plane)[aperture])),
            source_support_fraction=fraction,boundary_warning=bool(boundary),distortion_warning=distorted,
            measurement_available=adequate,status=('no_source_established' if not original['source_present'] else
                'unassessable_support' if boundary else 'degraded_shape' if distorted else
                'adequate_for_finite_screen' if adequate else 'unassessable_measurement'))
        associated=bool(np.any(evidence['positive'][1:]&core))
        if inner is not None:
            ok=not inner['boundary_rejection'] and inner['peak']>0 and fit['peak']>0 and np.isfinite(inner['parameters']).all()
            probe=dict(available=bool(ok),fit=copy.deepcopy(inner),
                centroid_difference_arcsec=float(np.linalg.norm(np.array(inner['centroid'])-p[1:3])),
                peak_relative_difference=float(abs(inner['peak']/p[0]-1)) if p[0]!=0 else None)
    return dict(original=original,judgments=judgments(original,associated,probe),fixed_inner_domain_probe=probe)
