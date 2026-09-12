"""Separate data-only POINT judgments. Truth and case names are not inputs."""
import copy
import numpy as np
from common import b, candidate, module, PRIOR

old_evaluation = module('repair_original_evaluation', PRIOR/'evaluation.py')

def judgments(original, associated, probe):
    """Pure admission rule; no truth errors, arm identity, or nominal source."""
    fit = original.get('fit')
    limitations = []
    if fit is None:
        limitations.append('fit_unavailable')
    else:
        if not np.isfinite(fit['parameters']).all() or fit['peak'] <= 0:
            limitations.append('nonfinite_or_nonpositive_fit')
        if fit['boundary_rejection']:
            limitations.append('fit_parameter_limit')
        if original.get('shape_core_pixels', 0) < 10:
            limitations.append('insufficient_source_core')
        if max(fit['widths']) > 24:
            limitations.append('outside_compact_measurement_domain')
    if original.get('boundary_warning', False):
        limitations.append('source_support_limit')
    if not original['source_present']:
        limitations.append('source_not_established')
    elif not associated:
        limitations.append('fit_not_associated_with_source_evidence')
    base_ok = not limitations
    stable_center = probe.get('available', False) and probe['centroid_difference_arcsec'] <= .25
    stable_peak = probe.get('available', False) and probe['peak_relative_difference'] <= .01
    score = original.get('empirical_peak_score')
    return dict(source_evidence_present=original['source_present'], associated_source=bool(associated),
        centroid_usable=bool(base_ok and stable_center),
        peak_response_usable=bool(base_ok and stable_peak and score is not None and score > 5),
        shape_warning=bool(original.get('distortion_warning', False)),
        support_warning=bool(original.get('boundary_warning', False)),
        limitations=limitations, centroid_stability_pass=bool(stable_center),
        peak_stability_pass=bool(stable_peak), peak_score_pass=bool(score is not None and score > 5))

def measure(data, maps, estimators, originals=None):
    originals = old_evaluation.measure(data, maps, estimators) if originals is None else originals
    result = []
    for a, (e, original) in enumerate(zip(estimators, originals)):
        original = copy.deepcopy(original)
        associated = False
        probe = dict(available=False)
        if 'fit' in original:
            p = np.array(original['fit']['parameters'])
            G = b.gaussian(p, data.x, data.ygrid)
            core = e.D & (G >= .1*max(p[0], 0))
            Y, _ = e.residual(maps[a])
            bands = candidate.analysis(Y.reshape(e.shape)).reshape(5, -1)
            positive = e.valid & e.D & (bands >= 5*e.sigmas[:, e.stratum])
            associated = bool(np.any(positive[1:] & core))
            try:
                inner = b.fit_source(maps[a], data.x, data.ygrid,
                                     e.D & (np.hypot(data.x, data.ygrid) <= 52))
                ok = (not inner['boundary_rejection'] and inner['peak'] > 0
                      and original['fit']['peak'] > 0 and np.isfinite(inner['parameters']).all())
                probe = dict(available=bool(ok), fit=inner,
                    centroid_difference_arcsec=float(np.linalg.norm(np.array(inner['centroid'])-p[1:3])),
                    peak_relative_difference=float(abs(inner['peak']/p[0]-1)) if p[0] != 0 else None)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
                probe['error'] = str(err)
        result.append(dict(original=original, judgments=judgments(original, associated, probe),
                           fixed_inner_domain_probe=probe))
    return result
