"""Common POINT measurements on total maps; truth never enters this evaluator."""
import numpy as np
import candidate
from common import b

def measure(data, maps, estimators):
    records = []
    for a, e in enumerate(estimators):
        Y, background = e.residual(maps[a])
        bands = candidate.analysis(Y.reshape(e.shape)).reshape(5, -1)
        sigma = e.sigmas[:, e.stratum]
        positive = e.valid & e.D & (bands >= 5*sigma)
        # Common, data-only presence screen: positive detail bands 2--4/coarse.
        # Its empirical scales are not calibrated detection significances.
        source = bool(positive[1:].any())
        rec = dict(source_present=source, positive_counts=positive.sum(axis=1),
                   source_evidence='positive_band_2_to_5_coefficient_above_5_empirical_scales',
                   background=background, measurement_available=False,
                   support_pixels=int(data.D[a].sum()),
                   exterior_rms=float(np.sqrt(np.mean(maps[a, data.O[a]]**2))))
        try:
            fit = b.fit_source(maps[a], data.x, data.ygrid, e.D)
            p = fit['parameters']
            radius = float(np.hypot(*p[1:3]))
            G = b.gaussian(p, data.x, data.ygrid)
            plane = p[6]+p[7]*data.x/90+p[8]*data.ygrid/90
            core = e.D & (G >= .1*max(p[0], 0.))
            denom = np.linalg.norm(G[core])
            shape_error = float(np.linalg.norm((maps[a]-plane-G)[core])/denom) if denom > 0 else None
            R = candidate.mad(maps[a, data.O[a]])
            interior = np.hypot(data.x-p[1], data.ygrid-p[2]) <= 1.5*max(fit['widths'])
            fraction = float(np.sum(interior & e.D)/max(int(interior.sum()), 1))
            boundary = radius >= 52 or fraction < .95
            distorted = bool(max(fit['widths']) > 24 or shape_error is None or shape_error > .25)
            adequate = bool(source and p[0] > 5*R and not fit['boundary_rejection']
                            and not boundary and not distorted and int(core.sum()) >= 10)
            aperture = data.D[a] & (np.hypot(data.x-p[1], data.ygrid-p[2]) <= 40)
            rec.update(fit=fit, fit_radius=radius, empirical_peak_score=float(p[0]/R) if R > 0 else None,
                       shape_error=shape_error, shape_core_pixels=int(core.sum()),
                       aperture_brightness=float(4*np.sum((maps[a]-plane)[aperture])),
                       source_support_fraction=fraction, boundary_warning=bool(boundary),
                       distortion_warning=distorted, measurement_available=adequate,
                       status=('no_source_established' if not source else
                               'unassessable_support' if boundary else
                               'degraded_shape' if distorted else
                               'adequate_for_finite_screen' if adequate else 'unassessable_measurement'))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
            rec.update(status='unassessable_measurement' if source else 'no_source_established',
                       fit_error=str(err), distortion_warning=False, boundary_warning=False)
        records.append(rec)
    return records

def score_truth(data, maps, measurements, truth, definition):
    """External finite-case ruler. It cannot affect measurement or feedback."""
    result = []
    for a, rec in enumerate(measurements):
        D = data.D[a]
        norm = np.linalg.norm(truth[a,D])
        r = dict(total_image_relative_L2=float(np.linalg.norm((maps[a]-truth[a])[D])/norm) if norm > 0 else None,
                 signed_total_brightness=float(4*np.sum(maps[a,D])),
                 truth_brightness=float(4*np.sum(truth[a,D])))
        if definition.get('centroid') and 'fit' in rec:
            r['centroid_error_arcsec'] = float(np.linalg.norm(np.array(rec['fit']['centroid'])-definition['centroid']))
            r['centroid_error_vector_arcsec'] = np.array(rec['fit']['centroid'])-definition['centroid']
        if definition.get('peak') and 'fit' in rec:
            r['peak_relative_error'] = float(rec['fit']['peak']/definition['peak'][a]-1)
        if definition.get('widths') and 'fit' in rec:
            r['width_relative_error'] = np.array(rec['fit']['widths'])/definition['widths']-1
        result.append(r)
    return result
