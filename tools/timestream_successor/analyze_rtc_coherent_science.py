#!/usr/bin/env python3
"""Scientific error comparison in explicit units, conditional diagnostic only."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import optimize, special
from run_rtc_notch_recovery import write_json, binding
from run_rtc_coherent_science import ASEC

ARMS = ("lowpass", "finite", "coherent", "reject")


def image(folder, arm, inj, coordinate, support, chunks, pixels):
    numerator = np.zeros((pixels, pixels))
    weight = np.zeros_like(numerator)
    for chunk in chunks:
        p = folder / "downstream" / f"{arm}-{inj}-{chunk}-{coordinate}"
        numerator += np.fromfile(p / (support + "-sum.f64"), "<f8").reshape(
            pixels, pixels
        )
        weight += np.fromfile(p / (support + "-weight.f64"), "<f8").reshape(
            pixels, pixels
        )
    m = np.full_like(numerator, np.nan)
    np.divide(numerator, weight, out=m, where=weight > 0)
    return m, weight


def airy(radius):
    u = np.pi * 50 / (299792458 / 150e9) * radius / ASEC
    z = np.ones_like(u)
    nz = u != 0
    z[nz] = (2 * special.j1(u[nz]) / u[nz]) ** 2
    return z


def coordinates(design):
    axis = (np.arange(design["map_pixels"]) - (design["map_pixels"] - 1) / 2) * design[
        "pixel_arcsec"
    ]
    return np.meshgrid(axis, axis)


def quantities(m, weight, design):
    xx, yy = coordinates(design)
    res = []
    for i, cross in enumerate(design["crossings"]):
        cx, cy = cross["center_arcsec"]
        radius = np.hypot(xx - cx, yy - cy)
        select = (radius < 2 * 8.483936) & np.isfinite(m) & (weight > 0)
        record = dict(
            crossing=i,
            kind=cross["kind"],
            pixels=int(select.sum()),
            aperture_fraction=float(
                select.sum() / np.count_nonzero(radius < 2 * 8.483936)
            ),
        )
        if select.sum() < 12:
            record.update(
                available=False, reason="insufficient actual aperture coverage"
            )
            res.append(record)
            continue
        x = xx[select] - cx
        y = yy[select] - cy
        z = m[select]

        def residual(p):
            amp, dx, dy, offset, sx, sy = p
            return amp * airy(np.hypot(x - dx, y - dy)) + offset + sx * x + sy * y - z

        fit = optimize.least_squares(
            residual,
            [100, 0, 0, float(np.median(z)), 0, 0],
            bounds=(
                [-np.inf, -8.483936, -8.483936, -np.inf, -np.inf, -np.inf],
                [np.inf, 8.483936, 8.483936, np.inf, np.inf, np.inf],
            ),
            max_nfev=300,
        )
        amp, dx, dy, *_ = fit.x
        record.update(
            available=bool(fit.success and abs(dx) < 8.48 and abs(dy) < 8.48),
            peak_mJy_per_beam=float(amp),
            centroid_x_arcsec=float(dx),
            centroid_y_arcsec=float(dy),
            fit_residual_rms_mJy_per_beam=float(np.sqrt(np.mean(fit.fun**2))),
            reason=None if fit.success else fit.message,
        )
        res.append(record)
    cx, cy = design["extended_center_arcsec"]
    sigma = design["extended_apparent_FWHM_arcsec"] / np.sqrt(8 * np.log(2))
    radius = np.hypot(xx - cx, yy - cy)
    template = np.exp(-0.5 * radius**2 / sigma**2)
    sel = (radius < 3 * sigma) & np.isfinite(m) & (weight > 0)
    extended = dict(available=False, pixels=int(sel.sum()))
    if sel.sum() > 12:
        x = xx[sel] - cx
        y = yy[sel] - cy
        h = np.column_stack([template[sel], np.ones(sel.sum()), x, y])
        coeff = np.linalg.lstsq(h, m[sel], rcond=None)[0]
        extended.update(
            available=True,
            peak_mJy_per_beam=float(coeff[0]),
            aperture_fraction=float(sel.sum() / np.count_nonzero(radius < 3 * sigma)),
        )
    off = np.isfinite(m) & (weight > 0) & (radius > 4 * sigma)
    for cross in design["crossings"]:
        cx, cy = cross["center_arcsec"]
        off &= np.hypot(xx - cx, yy - cy) > 4 * 8.483936
    vals = m[off]
    artifacts = (
        dict(
            pixels=int(off.sum()),
            rms_mJy_per_beam=float(np.sqrt(np.mean(vals**2))),
            min_mJy_per_beam=float(vals.min()),
            max_mJy_per_beam=float(vals.max()),
            q01_q50_q99_mJy_per_beam=np.quantile(vals, [0.01, 0.5, 0.99]).tolist(),
        )
        if len(vals)
        else dict(pixels=0)
    )
    # Actual pixel-pair covariance at fixed separations, not an independence assumption.
    for lag in (1, 4, 12):
        pair = off[:, lag:] & off[:, :-lag]
        a = m[:, lag:][pair]
        b = m[:, :-lag][pair]
        artifacts[f"covariance_{lag * design['pixel_arcsec']}arcsec_mJy2_per_beam2"] = (
            float(np.mean((a - a.mean()) * (b - b.mean()))) if len(a) > 1 else None
        )
    return dict(
        compact=res,
        extended=extended,
        off_source=artifacts,
        covered_pixels=int((weight > 0).sum()),
    )


def moments(values, truth):
    values = np.asarray(values, float)
    errors = values - truth
    return dict(
        n=len(values),
        truth=truth,
        mean=float(values.mean()),
        bias=float(errors.mean()),
        uncertainty_sd=float(values.std(ddof=1)) if len(values) > 1 else None,
        variance_population=float(values.var()),
        MSE=float(np.mean(errors**2)),
        bias_squared=float(errors.mean() ** 2),
    )


def analyze(a):
    a.output.mkdir(parents=True, exist_ok=False)
    design = json.loads((a.prepare / "science-design.json").read_text())
    n = design["map_pixels"]
    chunks = design["chunks"]
    records = []
    scenarios = [a.real / "real"] + sorted(a.noise.glob("noise-*"))
    for folder in scenarios:
        for arm in ARMS:
            for support in ("own", "common"):
                for coord in [0, 1] if folder.name == "real" else [0]:
                    baseline, w = image(folder, arm, "none", coord, support, chunks, n)
                    injected, _ = image(folder, arm, "sky", coord, support, chunks, n)
                    delta = injected - baseline
                    for name, m in [
                        ("without_sky", baseline),
                        ("with_sky", injected),
                        ("paired_response", delta),
                    ]:
                        record = dict(
                            scenario=folder.name,
                            arm=arm,
                            support=support,
                            coordinate="x" if coord == 0 else "r_proxy",
                            product=name,
                            quantities=quantities(m, w, design),
                        )
                        records.append(record)
                    if folder.name == "real" and coord == 0:
                        for name, m in [
                            ("without_sky", baseline),
                            ("with_sky", injected),
                            ("paired_response", delta),
                        ]:
                            np.save(a.output / f"{arm}-{support}-{name}.npy", m)
                        np.save(a.output / f"{arm}-{support}-weight.npy", w)
                        if arm == "coherent":
                            for response in ("template", "projection"):
                                m, _ = image(
                                    folder, arm, response, coord, support, chunks, n
                                )
                                records.append(
                                    dict(
                                        scenario="real",
                                        arm=arm,
                                        support=support,
                                        coordinate="x",
                                        product=response + "-paired-response",
                                        quantities=quantities(m - baseline, w, design),
                                    )
                                )
                                np.save(
                                    a.output
                                    / f"coherent-{support}-{response}-response.npy",
                                    m - baseline,
                                )
    write_json(a.output / "measurements.json", records)
    comparison = []
    for arm in ARMS:
        for support in ("own", "common"):
            chosen = [
                r
                for r in records
                if r["scenario"].startswith("noise-")
                and r["arm"] == arm
                and r["support"] == support
                and r["product"] == "with_sky"
            ]
            row = dict(arm=arm, support=support, compact=[], extended=None)
            for i, cross in enumerate(design["crossings"]):
                found = [r["quantities"]["compact"][i] for r in chosen]
                ok = [r for r in found if r["available"]]
                row["compact"].append(
                    dict(
                        crossing=i,
                        kind=cross["kind"],
                        available=len(ok),
                        missing=len(found) - len(ok),
                        coverage_fraction=[r["aperture_fraction"] for r in found],
                        peak_mJy_per_beam=moments(
                            [r["peak_mJy_per_beam"] for r in ok], 100.0
                        )
                        if ok
                        else None,
                        centroid_x_arcsec=moments(
                            [r["centroid_x_arcsec"] for r in ok], 0.0
                        )
                        if ok
                        else None,
                        centroid_y_arcsec=moments(
                            [r["centroid_y_arcsec"] for r in ok], 0.0
                        )
                        if ok
                        else None,
                    )
                )
            vals = [
                r["quantities"]["extended"]["peak_mJy_per_beam"]
                for r in chosen
                if r["quantities"]["extended"]["available"]
            ]
            row["extended"] = moments(vals, 25.0) if vals else None
            row["off_source"] = {
                k: moments([r["quantities"]["off_source"][k] for r in chosen], 0.0)
                for k in ["rms_mJy_per_beam", "min_mJy_per_beam", "max_mJy_per_beam"]
            }
            comparison.append(row)
    write_json(
        a.output / "comparison.json",
        dict(
            design=binding(a.prepare / "science-design.json"),
            reference="conditional existing10-mode Cleaner/ordinary naive projection; matched-APT units/static weights; no JINC/fullPTC/CAL/FRUIT qualification",
            truth="declared physical source injections; correlated stationary empirical backgrounds plus known added line; no cleaned-data or sideband truth",
            independent_noise_realizations=len(scenarios) - 1,
            results=comparison,
        ),
    )
    # Fixed central-lobe support test: do not count retained tails as recovered crossing.
    t = np.fromfile(a.evidence / "native-full/native-time.f64", "<f8")
    xy = (
        np.fromfile(a.evidence / "geometry-02/pointing-269.f64", "<f8").reshape(-1, 2)
        * ASEC
    )
    support_rows = []
    for c in design["crossings"]:
        center = np.asarray(c["center_arcsec"])
        inside = np.linalg.norm(xy - center, axis=1) < (
            3.8317059702075125 * (299792458 / 150e9) / (np.pi * 50) * ASEC
        )
        i = c["native_row"]
        lo = i
        hi = i + 1
        while lo > 0 and inside[lo - 1]:
            lo -= 1
        while hi < len(t) and inside[hi]:
            hi += 1
        record = dict(
            crossing=c,
            central_lobe_rows=[lo, hi],
            duration_seconds=float(t[hi - 1] - t[lo]),
            arms={},
        )
        for arm in ARMS:
            m = np.memmap(
                a.prepare / (arm + "-good.u8"), np.bool_, "r", shape=(len(t), 491)
            )
            rr = np.arange(lo + (lo % 2), hi, 2)
            count = int(m[rr, 269].sum())
            record["arms"][arm] = dict(
                retained=count, total=len(rr), whole_central_lobe=bool(count == len(rr))
            )
        support_rows.append(record)
    write_json(a.output / "crossing-support.json", support_rows)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["evidence", "prepare", "real", "noise", "output"]:
        p.add_argument("--" + k, type=Path, required=True)
    analyze(p.parse_args())
