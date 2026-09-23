"""Provisional bright-POINT connected-support candidate; no FRUIT application."""
import numpy as np
from scipy.ndimage import label, binary_dilation, binary_erosion
from inspect_point_source import measure
from point_empirical import coordinates

METHOD = 'bright-point-connected-support-prototype-v1'
CASES = (('reference', 3., 1.), ('permissive', 2., 1.),
         ('restrictive', 5., 1.), ('scale_stress', 3., 2.))
CONNECTIVITY = np.ones((3, 3), dtype=bool)


def anchor_pixel(center, header):
    """Zero-based row,column; nearest pixel with exact half ties upward in index."""
    column = center[0] / (header['CDELT1'] * 3600) + header['CRPIX1'] - 1
    row = center[1] / (header['CDELT2'] * 3600) + header['CRPIX2'] - 1
    return int(np.floor(row + .5)), int(np.floor(column + .5))


def admit(signal, map_support, scale, noise_valid, header, parameters, threshold, multiplier):
    if threshold <= 0 or multiplier <= 0 or not np.isfinite([threshold, multiplier]).all():
        raise ValueError('positive finite experimental controls required')
    x, y = coordinates(signal.shape, header)
    domain = np.hypot(x, y) <= 40
    valid = domain & map_support & np.isfinite(signal) & noise_valid & np.isfinite(scale) & (scale > 0)
    brightness = signal - parameters[6]
    z = np.full(signal.shape, np.nan)
    np.divide(brightness, scale * multiplier, out=z, where=valid)
    eligible = valid & (z >= threshold)
    row, column = anchor_pixel(parameters[1:3], header)
    selected = np.zeros(signal.shape, bool)
    if not (0 <= row < signal.shape[0] and 0 <= column < signal.shape[1]) or not domain[row, column]:
        status = 'anchor_outside_domain'
    elif not valid[row, column]:
        status = 'anchor_invalid'
    elif not eligible[row, column]:
        status = 'anchor_below_threshold'
    else:
        components, _ = label(eligible, structure=CONNECTIVITY)
        selected = components == components[row, column]
        status = 'retained'
    candidate = np.where(valid, np.where(selected, brightness, 0.), np.nan)
    excluded = np.where(valid, brightness - candidate, np.nan)
    boundary = domain & ~binary_erosion(domain, structure=CONNECTIVITY, border_value=0)
    adjacent = binary_dilation(selected, structure=CONNECTIVITY)
    unsupported = domain & ~valid
    area = abs(header['CDELT1'] * header['CDELT2']) * 3600**2
    summary = dict(status=status, anchor_row_column=[row, column],
        anchor_ratio=float(z[row, column]) if 0 <= row < signal.shape[0] and 0 <= column < signal.shape[1] and valid[row, column] else None,
        pixels=int(selected.sum()), area_arcsec2=float(selected.sum() * area),
        extent_east_north_arcsec=[[float(x[selected].min()), float(x[selected].max())],
                                  [float(y[selected].min()), float(y[selected].max())]] if selected.any() else None,
        boundary_contact=bool(np.any(selected & boundary)),
        unsupported_contact=bool(np.any(adjacent & unsupported)),
        unsupported_domain_pixels=int(unsupported.sum()),
        unassociated_above_threshold_pixels=int((eligible & ~selected).sum()),
        threshold=threshold, noise_scale_multiplier=multiplier, equivalent_original_scale_threshold=threshold * multiplier)
    return dict(candidate=candidate, selected=selected, valid=valid, domain=domain,
                excluded=excluded, brightness=np.where(domain & map_support, brightness, np.nan), z=z, summary=summary)


def fit_diagnostic(signal, support, header, beam):
    x, y = coordinates(signal.shape, header)
    domain = np.hypot(x, y) <= 40
    if np.any(domain & ~(support & np.isfinite(signal))):
        return dict(status='unavailable', cause='incomplete original 40 arcsec radius fit footprint')
    try:
        desc, p, _, _ = measure(signal, support, header, beam)
        return dict(status='success', parameters=p.tolist(), description=desc,
                    constrained=desc['fit_constraint_active'])
    except (ValueError, RuntimeError) as exc:
        return dict(status='failed', cause=str(exc))


def projection(signal, valid, q):
    roi = q != 0
    if np.any(roi & ~(valid & np.isfinite(signal))):
        return None
    return float(np.sum(q[roi] * signal[roi]))


def compare_fit(candidate, reference):
    if candidate['status'] != 'success' or reference['status'] != 'success':
        return None
    p, r = np.asarray(candidate['parameters']), np.asarray(reference['parameters'])
    widths = np.asarray(candidate['description']['measured_fwhm_major_minor_arcsec'])
    original = np.asarray(reference['description']['measured_fwhm_major_minor_arcsec'])
    return dict(amplitude_delta=float(p[0]-r[0]), amplitude_fraction=float(p[0]/r[0]-1),
                centroid_delta_arcsec=(p[1:3]-r[1:3]).tolist(),
                centroid_distance_arcsec=float(np.linalg.norm(p[1:3]-r[1:3])),
                fwhm_delta_arcsec=(widths-original).tolist(), fwhm_fraction=(widths/original-1).tolist())


def evaluate(signal, support, scale, noise_valid, header, parameters, beam, q):
    """All four controls share one raw measurement/background/anchor per realization."""
    source_relative = signal - parameters[6]
    reference = fit_diagnostic(source_relative, support, header, beam)
    reference_projection = projection(source_relative, support, q)
    outputs = {}
    for name, threshold, multiplier in CASES:
        result = admit(signal, support, scale, noise_valid, header, parameters, threshold, multiplier)
        fit = fit_diagnostic(result['candidate'], result['valid'], header, beam)
        fixed = projection(result['candidate'], result['valid'], q)
        excluded_fixed = projection(result['excluded'], result['valid'], q)
        result['summary'].update(fit=fit, comparison=compare_fit(fit, reference), fixed_projection=fixed,
            excluded_fixed_projection=excluded_fixed,
            fixed_fraction_excluded=excluded_fixed/reference_projection if excluded_fixed is not None and reference_projection else None)
        outputs[name] = result
    refmask = outputs['reference']['selected']
    for result in outputs.values():
        added = result['selected'] & ~refmask; removed = refmask & ~result['selected']
        result['summary'].update(added_vs_reference=int(added.sum()), removed_vs_reference=int(removed.sum()))
    return dict(fit=reference, fixed_projection=reference_projection), outputs


def distribution(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                scatter=float(values.std(ddof=1)) if len(values)>1 else None) if len(values) else dict(n=0)


def summarize_trials(records):
    result = {}
    for name, _, _ in CASES:
        rows = [r['cases'][name] for r in records if r['raw_fit']['status']=='success']
        fitted = [r for r in rows if r['fit']['status']=='success']
        failures = len(records)-len(fitted)
        result[name] = dict(trials=len(records), raw_fit_failures=len(records)-len(rows),
            raw_constrained=sum(r['raw_fit'].get('constrained',False) for r in records),
            reference_fit_failures=sum(r.get('reference',{}).get('fit',{}).get('status','success')!='success' for r in records if r['raw_fit']['status']=='success'),
            source_retained=sum(r['status']=='retained' for r in rows),
            candidate_fit_failures_or_unavailable=failures,
            constrained=sum(r['fit']['constrained'] for r in fitted),
            boundary_contacts=sum(r['boundary_contact'] for r in rows),
            unsupported_contacts=sum(r['unsupported_contact'] for r in rows),
            partial_distribution_only=bool(failures),
            area_arcsec2=distribution([r['area_arcsec2'] for r in rows]),
            amplitude=distribution([r['fit']['parameters'][0] for r in fitted]),
            amplitude_fraction_change=distribution([r['comparison']['amplitude_fraction'] for r in fitted if r['comparison'] is not None]),
            centroid_distance_change_arcsec=distribution([r['comparison']['centroid_distance_arcsec'] for r in fitted if r['comparison'] is not None]),
            major_fwhm_fraction_change=distribution([r['comparison']['fwhm_fraction'][0] for r in fitted if r['comparison'] is not None]),
            minor_fwhm_fraction_change=distribution([r['comparison']['fwhm_fraction'][1] for r in fitted if r['comparison'] is not None]))
    return result
