"""Bounded POINT recovery experiment products, not production source selection."""

import numpy as np

ARRAYS = ("a1100", "a1400", "a2000")
ROLES = ("REF", "CORE+", "CORE-", "SHOULDER", "SEED")


def signals(parameters, major0):
    p = np.array(parameters, dtype=float)
    if (
        not np.isfinite(major0)
        or major0 <= 0
        or p.shape != (7,)
        or not np.isfinite(p).all()
        or p[0] <= 0
        or min(p[3:5]) <= 0
    ):
        raise ValueError("unavailable frozen source shape")
    y, x = np.indices((41, 41))
    x = (20 - x) * 2.0
    y = (y - 20) * 2.0
    radius = np.hypot(x, y)
    taper = np.where(
        radius <= 30,
        1.0,
        np.where(radius < 36, (1 + np.cos(np.pi * (radius - 30) / 6)) / 2, 0.0),
    )

    def evaluate(q):
        dx, dy = x - q[1], y - q[2]
        u = np.cos(q[5]) * dx + np.sin(q[5]) * dy
        v = -np.sin(q[5]) * dx + np.cos(q[5]) * dy
        return q[0] * np.exp(-0.5 * ((u / q[3]) ** 2 + (v / q[4]) ** 2)) * taper

    compact = p.copy()
    compact[0] *= 0.1
    shoulder = p.copy()
    shoulder[0] *= 0.05
    shoulder[1] += 0.5 * major0
    shoulder[3:5] *= 2
    return {
        "REF": np.zeros((41, 41)),
        "CORE+": evaluate(compact),
        "CORE-": -evaluate(compact),
        "SHOULDER": evaluate(shoulder),
    }


def response(delta, truth, support):
    if (
        not np.all(support)
        or not np.isfinite(delta).all()
        or not np.isfinite(truth).all()
    ):
        return {"available": False, "reason": "fixed benchmark support unavailable"}
    power = float(np.sum(truth * truth))
    if power <= 0:
        raise ValueError("empty truth norm")
    gain = float(np.sum(delta * truth) / power)
    eta = float(np.sqrt(np.sum((delta - truth) ** 2) / power))
    rho = float(np.sqrt(np.sum((delta - gain * truth) ** 2) / power))
    if not np.isclose(eta * eta, (gain - 1) ** 2 + rho * rho, rtol=1e-10, atol=1e-12):
        raise ValueError("response decomposition failed")
    return dict(available=True, gain=gain, distortion=eta, orthogonal_distortion=rho)


def geometry(header, shape, parameters):
    """Freeze exact pixel-center footprints; no fits or support clipping.

    This is the current point_empirical aperture convention and point_auxiliary
    three-major-FWHM unit-template convention, expressed directly in AltAz.
    """
    from inspect_point_source import gaussian

    p = np.asarray(parameters, float)
    if p.shape != (7,) or not np.isfinite(p).all() or min(p[3:5]) <= 0:
        raise ValueError("invalid frozen M0 geometry")
    if header["CDELT1"] != -2 or header["CDELT2"] != 2:
        raise ValueError("recovery requires the fixed 2 arcsecond grid")
    yy, xx = np.indices(shape)
    x = (xx + 1 - header["CRPIX1"]) * -2.0
    y = (yy + 1 - header["CRPIX2"]) * 2.0
    major = max(p[3:5]) * np.sqrt(8 * np.log(2))
    radius = np.hypot(x - p[1], y - p[2])
    if (
        x.min() > p[1] - 3 * major
        or x.max() < p[1] + 3 * major
        or y.min() > p[2] - 3 * major
        or y.max() < p[2] + 3 * major
    ):
        raise ValueError("template footprint exceeds map")
    unit = p.copy()
    unit[0] = 1.0
    unit[6] = 0.0
    roi = radius <= 3 * major
    template = np.where(roi, gaussian(unit, x, y), 0.0)
    weights = template / np.sum(template**2)
    origin = np.hypot(x, y)
    return dict(
        x=x,
        y=y,
        disk=origin <= 40,
        background=(origin >= 45) & (origin <= 60),
        template=weights,
        apertures=np.array([radius <= f * major for f in (1.0, 1.5, 2.0)]),
        profiles=np.array(
            [(radius >= lo) & (radius < lo + 2) for lo in range(0, 60, 2)]
        ),
        parameters=p,
        major=np.array(major),
    )


def embed(raster, g):
    """Exact 2 arcsecond containing-pixel mapping, independently of native code."""
    if np.shape(raster) != (41, 41):
        raise ValueError("unexpected recovery raster shape")
    col = np.floor(g["x"] / -2 + 20.5).astype(int)
    row = np.floor(g["y"] / 2 + 20.5).astype(int)
    use = (col >= 0) & (col < 41) & (row >= 0) & (row < 41)
    result = np.zeros(col.shape)
    result[use] = raster[row[use], col[use]]
    return result


def statistics(values):
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        return dict(available=False)
    med = np.median(values)
    return dict(
        available=True,
        pixels=int(values.size),
        mean=float(values.mean()),
        std=float(values.std()),
        median=float(med),
        mad=float(np.median(np.abs(values - med))),
    )


def measurements(signal, support, g, guard=None):
    """Saved total maps or signed differences; never fit a difference here."""
    from point_empirical import aperture_measurements

    good = support & np.isfinite(signal)
    p = g["parameters"]
    bg = None
    fit = dict(status="not_requested")
    if guard is not None:
        fit = dict(guard=guard, parameters=guard.get("accepted_parameters"))
        if fit["parameters"] is not None:
            q = np.asarray(fit["parameters"])
            bg = float(q[6])
            c, s = np.cos(q[5]), np.sin(q[5])
            rotation = np.array([[c, -s], [s, c]])
            fit.update(
                major_fwhm=float(max(q[3:5]) * np.sqrt(8 * np.log(2))),
                minor_fwhm=float(min(q[3:5]) * np.sqrt(8 * np.log(2))),
                ellipse_arcsec2=(rotation @ np.diag(q[3:5] ** 2) @ rotation.T).tolist(),
            )
    template_roi = g["template"] != 0
    projection = (
        float(np.sum(signal[template_roi] * g["template"][template_roi]))
        if good[template_roi].all()
        else None
    )
    apertures = aperture_measurements(
        signal, support, g["x"], g["y"], p[1:3], float(g["major"]), 1.0, 4.0
    )
    for ap in apertures:
        raw = ap["value_mJy_per_beam_arcsec2"]
        constant = bg * ap["expected_pixels"] * 4 if bg is not None else None
        ap.update(
            constant_contribution=constant,
            background_subtracted=raw - constant
            if raw is not None and constant is not None
            else None,
        )
    profiles = [
        statistics(signal[m]) if good[m].all() else dict(available=False)
        for m in g["profiles"]
    ]
    m = g["background"]
    background = statistics(signal[m]) if good[m].all() else dict(available=False)
    return dict(
        fit=fit,
        template_raw=projection,
        template_constant=bg * float(g["template"].sum()) if bg is not None else None,
        template_background_subtracted=projection - bg * float(g["template"].sum())
        if projection is not None and bg is not None
        else None,
        apertures=apertures,
        profiles=profiles,
        background=background,
    )


def comparison(signal, support, reference, ref_support, truth, g):
    disk = g["disk"]
    delta = signal - reference
    result = response(delta[disk], truth[disk], (support & ref_support)[disk])
    result.update(
        lost_benchmark_pixels=int(np.sum(disk & ~(support & ref_support))),
        gained_map_pixels=int(np.sum(support & ~ref_support)),
        lost_map_pixels=int(np.sum(ref_support & ~support)),
        signed_difference=measurements(delta, support & ref_support, g),
    )
    return result, delta
