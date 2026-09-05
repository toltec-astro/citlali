#!/usr/bin/env python3
"""Prospective EL-F12 measurements and exact historical/control gates.

This analyzer never feeds a selector. Regions, bounds and comparisons are the
owner-approved EL-F12 design. Missing values fail rather than disappear.
"""
from __future__ import annotations

import csv
import math
import json
import shlex
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.modeling import fitting, models
from netCDF4 import Dataset

from tools.fruit_loops.compare_injected_source_pair import (
    ARRAYS, FWHM_FACTOR, file_record, gaussian_center_for_map_world_offset, gaussian_fit,
    image, iteration_dirs, kernel_projection_metrics, product_path,
)
from tools.fruit_loops.analyze_prospective_influence_persistence import require_equal_netcdf
from tools.fruit_loops.edit_restart_checkpoint_penalty import values_equal
from tools.fruit_loops.analyze_penalty_placement import scalar_text

SCIENCE_PLANES = ("signal_I", "kernel_I", "weight_I", "weight_formal_I")
WCS_KEYS = tuple(f"{stem}{axis}" for axis in (1, 2) for stem in
                 ("CTYPE", "CUNIT", "CRPIX", "CRVAL", "CDELT")) + ("BUNIT",)
SOURCE = (0.0, -60.0)
NEPTUNE = (12.53903, -5.334553)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def grid(path: Path) -> dict:
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul["signal_I"]
        shape = np.asarray(hdu.data).squeeze().shape
        header = {key: hdu.header[key] for key in WCS_KEYS}
    if (header["CTYPE1"], header["CTYPE2"]) != ("AZOFFSET", "ELOFFSET"):
        raise ValueError("EL-F12 requires the registered AZ/EL grid")
    if header["CUNIT1"] != "arcsec" or header["CUNIT2"] != "arcsec":
        raise ValueError("EL-F12 grid units changed")
    if header["CDELT1"] != -1.0 or header["CDELT2"] != 1.0:
        raise ValueError("EL-F12 registered one-arcsec pixel axes changed")
    return {"shape": list(shape), "header": header}


def regions(grid_record: dict) -> dict[str, np.ndarray]:
    y, x = np.indices(grid_record["shape"], dtype=float)
    header = grid_record["header"]
    x = (x + 1 - header["CRPIX1"]) * header["CDELT1"] + header["CRVAL1"]
    y = (y + 1 - header["CRPIX2"]) * header["CDELT2"] + header["CRVAL2"]
    source_radius = np.hypot(x - SOURCE[0], y - SOURCE[1])
    neptune_radius = np.hypot(x - NEPTUNE[0], y - NEPTUNE[1])
    return {"full": np.ones(x.shape, dtype=bool), "source": source_radius <= 20,
            "neptune": neptune_radius <= 20,
            "annulus": (source_radius >= 40) & (source_radius <= 120) & (neptune_radius > 25),
            "source_radius": source_radius}


def support(signal: np.ndarray, weight: np.ndarray, cut: float = 0.1) -> np.ndarray:
    positive = np.sort(weight[np.isfinite(weight) & (weight > 0)])
    if not positive.size:
        raise ValueError("empty scientific support")
    index = (int(np.floor(0.75 * positive.size)) + positive.size) // 2
    return np.isfinite(signal) & np.isfinite(weight) & (weight > 0) & (weight >= cut * positive[index])


def signed_summary(values: np.ndarray, mask: np.ndarray) -> dict:
    chosen = values[mask]
    if not chosen.size or not np.isfinite(chosen).all():
        raise ValueError("missing/nonfinite required measurement support")
    return {"pixels": int(chosen.size), "rms": float(np.sqrt(np.mean(chosen ** 2))),
            "positive_peak": float(max(0, chosen.max())), "negative_peak": float(min(0, chosen.min())),
            "positive_sum": float(chosen[chosen > 0].sum()), "negative_sum": float(chosen[chosen < 0].sum())}


def require_map_identity(expected: Path, actual: Path, planes=SCIENCE_PLANES) -> dict:
    if grid(expected) != grid(actual):
        raise ValueError("map/grid identity failed")
    for extension in planes:
        with fits.open(expected, memmap=True) as left, fits.open(actual, memmap=True) as right:
            a, b = left[extension].data, right[extension].data
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError(f"bitwise map identity failed: {actual}:{extension}")
    return {"expected": str(expected), "actual": str(actual), "bitwise_planes": list(planes)}


def require_checkpoint_identity(expected: Path, actual: Path, *, audit_added: bool) -> dict:
    """Only declared source-path provenance and H audit addition may differ."""
    observed = []
    with Dataset(expected) as left, Dataset(actual) as right:
        if set(left.ncattrs()) != set(right.ncattrs()):
            raise ValueError("checkpoint attributes changed")
        for name in left.ncattrs():
            if not values_equal(left.getncattr(name), right.getncattr(name)):
                raise ValueError(f"checkpoint attribute changed: {name}")
        if set(left.dimensions) != set(right.dimensions):
            raise ValueError("checkpoint dimensions changed")
        for name, dimension in left.dimensions.items():
            other = right.dimensions[name]
            if len(dimension) != len(other) or dimension.isunlimited() != other.isunlimited():
                raise ValueError(f"checkpoint dimension changed: {name}")
        extra = {"fruit_response_state"} if audit_added else set()
        if set(right.variables) != set(left.variables) | extra or extra & set(left.variables):
            raise ValueError("checkpoint variable census changed")
        if audit_added:
            state = scalar_text(right["fruit_response_state"][...])
            if state.splitlines()[:1] != ["SCI-FRUIT-EL-F12-STATE-R0.1+CAP-001"] or not state.splitlines()[1].startswith("H "):
                raise ValueError("H checkpoint audit state missing/wrong arm")
        for name, variable in left.variables.items():
            other = right[name]
            if variable.dimensions != other.dimensions or variable.dtype != other.dtype or variable.ncattrs() != other.ncattrs():
                raise ValueError(f"checkpoint variable structure changed: {name}")
            for attribute in variable.ncattrs():
                if not values_equal(variable.getncattr(attribute), other.getncattr(attribute)):
                    raise ValueError(f"checkpoint variable metadata changed: {name}/{attribute}")
            a, b = variable[...], other[...]
            if values_equal(a, b):
                continue
            if name == "learning_policy_yaml" and audit_added:
                a, b = yaml.safe_load(scalar_text(a)), yaml.safe_load(scalar_text(b))
                if b.pop("fruit_response_arm", None) != "H" or a != b:
                    raise ValueError("H learning policy changes more than audit arm")
                observed.append(name)
                continue
            raise ValueError(f"scientific checkpoint changed: {name}")
    return {"provenance_differences": observed, "audit_added": audit_added,
            "D19_and_all_other_state": "exact"}


def iteration_rows(path: Path, iteration: int) -> tuple:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        iteration_name = "iteration" if "iteration" in (reader.fieldnames or []) else "iter"
        if iteration_name not in (reader.fieldnames or []):
            raise ValueError(f"learning file lacks iteration identity: {path}")
        rows = [tuple(row[name] for name in reader.fieldnames) for row in reader if int(row[iteration_name]) == iteration]
        return tuple(reader.fieldnames), rows


def decision_assignments(path: Path) -> list[dict]:
    with Dataset(path) as file:
        text = scalar_text(file["fruit_response_state"][...])
    lines = text.splitlines()
    if lines[0] != "SCI-FRUIT-EL-F12-STATE-R0.1+CAP-001":
        raise ValueError("unknown operational decision schema")
    count = int(lines[2])
    if count < 0 or count > 64:
        raise ValueError("operational assignment cap exceeded")
    result = []
    for line in lines[3:3 + count]:
        observation, array, uid, scan, source, first = shlex.split(line)
        result.append({"observation": observation, "array": int(array), "uid": int(uid),
                       "scan": int(scan), "source_iteration": int(source), "first_application": int(first)})
    if len(result) != count:
        raise ValueError("missing operational assignments")
    return result


def compare_trajectories(expected_root: Path, actual_root: Path, *, audit_added=False, maps_only=False) -> dict:
    expected, actual = iteration_dirs(expected_root, 123424), iteration_dirs(actual_root, 123424)
    if not set(actual) <= set(expected):
        raise ValueError("comparison absolute iterations are absent")
    result = {}
    for iteration in sorted(actual):
        a, b = expected[iteration], actual[iteration]
        for array in ARRAYS:
            require_map_identity(product_path(a, 123424, array), product_path(b, 123424, array))
        if maps_only:
            result[iteration] = {"maps": "bitwise"}
            continue
        checkpoint = require_checkpoint_identity(a / "citlali_restart_checkpoint.nc", b / "citlali_restart_checkpoint.nc", audit_added=audit_added)
        filenames = {p.name for p in a.glob("learning*.csv")}
        if filenames != {p.name for p in b.glob("learning*.csv")}:
            raise ValueError("learning file population changed")
        for name in filenames:
            if iteration_rows(a / name, iteration) != iteration_rows(b / name, iteration):
                raise ValueError(f"ordered per-iteration learning records differ: {name}")
        relative = Path("123424/raw/toltec_commissioning_pointing_123424_mapdiag.nc")
        require_equal_netcdf(a / relative, b / relative)
        result[iteration] = {"maps": "bitwise", "checkpoint": checkpoint, "learning": "exact ordered iteration rows", "mapdiag": "exact"}
    return result


def measure_pair(uninjected: Path, injected: Path, common: np.ndarray, masks: dict) -> tuple[dict, dict]:
    u, i = image(uninjected, "signal_I"), image(injected, "signal_I")
    kernel = image(injected, "kernel_I")
    transfer = i - u
    center = gaussian_center_for_map_world_offset(injected, "signal_I", *SOURCE)
    planet_center = gaussian_center_for_map_world_offset(uninjected, "signal_I", *NEPTUNE)
    def masked(data):
        return np.where(common, data, np.nan)
    source_fit = gaussian_fit(masked(transfer), 1.0, center, 25.0)
    kernel_fit = gaussian_fit(masked(kernel), 1.0, center, 25.0)
    planet_fit, planet_model = gaussian_source_model(masked(u), 1.0, planet_center, 25.0)
    projection = kernel_projection_metrics(masked(transfer), masked(kernel), 100.0)
    residual = transfer - projection["scale_mjy_beam"] * kernel
    planet_residual = u - planet_model
    if any(not np.isfinite(v) for fit in (source_fit, kernel_fit, planet_fit) for v in fit.values()):
        raise ValueError("nonfinite required fit")
    if min(source_fit["amplitude"], kernel_fit["amplitude"], planet_fit["amplitude"]) <= 0:
        raise ValueError("undefined source or kernel fit")
    metrics = {"central_recovery": source_fit["amplitude"] / (100.0 * kernel_fit["amplitude"]),
               "whole_kernel_recovery": projection["recovery_fraction"],
               "major_ratio": source_fit["major_fwhm_arcsec"] / kernel_fit["major_fwhm_arcsec"],
               "minor_ratio": source_fit["minor_fwhm_arcsec"] / kernel_fit["minor_fwhm_arcsec"],
               "centroid_error": float(np.hypot(source_fit["x_arcsec"] - kernel_fit["x_arcsec"], source_fit["y_arcsec"] - kernel_fit["y_arcsec"])),
               "source_fit": source_fit, "kernel_fit": kernel_fit, "neptune_fit": planet_fit}
    summaries = {}
    for name in ("full", "source", "neptune", "annulus"):
        domain = common & masks[name]
        summaries[name] = {"transfer": signed_summary(transfer, domain), "residual": signed_summary(residual, domain),
                           "uninjected_residual": signed_summary(planet_residual, domain)}
    metrics["regions"] = summaries
    # Arcsec² pixel area divided by the processed Gaussian beam solid angle;
    # retained descriptive aperture flux, not a separate protection endpoint.
    beam_area = np.pi * kernel_fit["major_fwhm_arcsec"] * kernel_fit["minor_fwhm_arcsec"] / (4 * np.log(2))
    metrics["aperture_flux_mjy"] = float(transfer[common & masks["source"]].sum() / beam_area)
    metrics["radial_summaries"] = []
    for low in range(0, 180, 10):
        domain = common & (masks["source_radius"] >= low) & (masks["source_radius"] < low + 10)
        metrics["radial_summaries"].append({"inner_arcsec": low, "outer_arcsec": low + 10,
             "measurement": signed_summary(transfer, domain) if domain.any() else None})
    return metrics, {"transfer": transfer, "residual": residual, "uninjected_residual": planet_residual}


def rms_bound(reference: float) -> float:
    return max(1.10 * reference, reference + 0.1)


def protections(h: dict, c: dict, iteration: int, lost: int) -> dict[str, bool]:
    checks = {"support": lost == 0}
    for name in ("central_recovery", "whole_kernel_recovery", "major_ratio", "minor_ratio"):
        checks[name] = abs(c[name] - 1) <= abs(h[name] - 1) + .01
        if iteration == 6:
            checks[name + "_terminal"] = abs(c[name] - 1) <= (.05 if "recovery" in name else .03)
    checks["centroid"] = c["centroid_error"] <= h["centroid_error"] + .02
    if iteration == 6:
        checks["centroid_terminal"] = c["centroid_error"] <= .1
    for region, quantity in (("full", "residual"), ("source", "residual"), ("annulus", "transfer"),
                             ("neptune", "transfer"), ("full", "uninjected_residual"), ("annulus", "uninjected_residual")):
        checks[f"{region}_{quantity}"] = c["regions"][region][quantity]["rms"] <= rms_bound(h["regions"][region][quantity]["rms"])
    for name in ("amplitude", "major_fwhm_arcsec", "minor_fwhm_arcsec"):
        checks["neptune_" + name] = abs(c["neptune_fit"][name] / h["neptune_fit"][name] - 1) <= .01
    checks["neptune_centroid"] = float(np.hypot(c["neptune_fit"]["x_arcsec"] - h["neptune_fit"]["x_arcsec"],
                                               c["neptune_fit"]["y_arcsec"] - h["neptune_fit"]["y_arcsec"])) <= .02
    return checks


def analyze_candidate(root: Path, arm: str, output: Path) -> dict:
    trees = {label: iteration_dirs(root / name / injection / "reduced", 123424)
             for label, name, injection in (("Hu", "H", "uninjected"), ("Hi", "H", "injected"),
                                           ("Cu", arm, "uninjected"), ("Ci", arm, "injected"))}
    if any(set(tree) != set(range(7)) for tree in trees.values()):
        raise ValueError("incomplete seven-iteration paired population")
    rows = []
    prior = {}
    all_checks = []
    promising_endpoints = []
    for iteration in range(1, 7):
        for array in ARRAYS:
            paths = {key: product_path(tree[iteration], 123424, array) for key, tree in trees.items()}
            grids = {key: grid(path) for key, path in paths.items()}
            if any(value != grids["Hu"] for value in grids.values()):
                raise ValueError("four-map grid differs; no remapping allowed")
            masks = regions(grids["Hu"])
            supports = {key: support(image(path, "signal_I"), image(path, "weight_I")) for key, path in paths.items()}
            hs, cs = supports["Hu"] & supports["Hi"], supports["Cu"] & supports["Ci"]
            common = hs & cs
            try:
                h, hm = measure_pair(paths["Hu"], paths["Hi"], common, masks)
                c, cm = measure_pair(paths["Cu"], paths["Ci"], common, masks)
            except (ValueError, RuntimeError, ZeroDivisionError) as error:
                rows.append({"array": array, "iteration": iteration, "status": "unavailable",
                             "failure": f"{type(error).__name__}: {error}"})
                all_checks.append(False)
                prior.pop(array, None)
                continue
            lost, gained = int((hs & ~cs).sum()), int((cs & ~hs).sum())
            checks = protections(h, c, iteration, lost)
            support_record = {"H_pixels": int(hs.sum()), "candidate_pixels": int(cs.sum()), "common_pixels": int(common.sum()),
                              "lost": lost, "gained": gained, "union_pixels": int((hs | cs).sum()),
                              "branch_pixels": {key: int(mask.sum()) for key, mask in supports.items()}}
            convergence = {}
            if array in prior:
                ph, pc, phm, pcm, previous_common = prior[array]
                domain = common & previous_common
                for region in ("full", "annulus"):
                    hv = signed_summary(hm["transfer"] - phm["transfer"], domain & masks[region])["rms"]
                    cv = signed_summary(cm["transfer"] - pcm["transfer"], domain & masks[region])["rms"]
                    convergence[region] = {"H": hv, "candidate": cv, "pixels": int((domain & masks[region]).sum())}
                    if iteration == 6:
                        checks["convergence_" + region] = cv <= rms_bound(hv)
                for name in ("central_recovery", "whole_kernel_recovery"):
                    hv, cv = h[name] - ph[name], c[name] - pc[name]
                    convergence[name] = {"H_signed_change": hv, "candidate_signed_change": cv}
                    if iteration == 6:
                        checks["convergence_" + name] = abs(cv) <= max(1.1 * abs(hv), .01)
            prior[array] = (h, c, hm, cm, common)
            if iteration in (5, 6) and array == "a1400":
                for region in ("annulus", "neptune"):
                    hv, cv = h["regions"][region]["transfer"]["rms"], c["regions"][region]["transfer"]["rms"]
                    promising_endpoints.append(cv <= .8 * hv and hv - cv >= .1)
            output.mkdir(parents=True, exist_ok=True)
            artifact = output / f"{arm}_{array}_iter{iteration}.npz"
            if artifact.exists():
                raise ValueError("refusing to overwrite analysis artifact")
            np.savez_compressed(artifact, H_support=hs, candidate_support=cs, common_support=common,
                                lost_support=hs & ~cs, gained_support=cs & ~hs,
                                **{f"H_{key}": value for key, value in hm.items()},
                                **{f"candidate_{key}": value for key, value in cm.items()})
            rows.append({"array": array, "iteration": iteration, "H": h, "candidate": c,
                         "support": support_record, "convergence": convergence, "protections": checks,
                         "artifact": file_record(artifact)})
            all_checks.extend(checks.values())
    assigned = {key: decision_assignments(tree[6] / "citlali_restart_checkpoint.nc")
                for key, tree in trees.items() if key in ("Cu", "Ci")}
    return {"arm": arm, "rows": rows, "all_protections_pass": all(all_checks),
            "selected_opportunities": sum(len(value) for value in assigned.values()), "assignments": assigned,
            "prioritized_leakage_pass": len(promising_endpoints) == 4 and all(promising_endpoints),
            "promising_before_required_restarts": all(all_checks) and all(promising_endpoints),
            "claim": "exposed development only; independent-pointing replication requires separate approval",
            "false_action_rate": "unavailable: no independent detector truth labels", "terminal_iteration": "hard-cap censored"}


# Same fitting definition as compare_injected_source_pair.gaussian_fit;
# also return the fitted Gaussian component for uninjected residual maps.
def gaussian_source_model(
    values: np.ndarray,
    pixel_size_arcsec: float,
    expected_center_arcsec: tuple[float, float] | None = None,
    search_radius_arcsec: float = 25.0,
) -> tuple[dict, np.ndarray]:
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("cannot fit an entirely non-finite map")
    map_yy, map_xx = np.indices(values.shape, dtype=float)
    map_xx = (
        map_xx - (values.shape[1] - 1) / 2.0
    ) * pixel_size_arcsec
    map_yy = (
        map_yy - (values.shape[0] - 1) / 2.0
    ) * pixel_size_arcsec
    peak_candidates = finite
    if expected_center_arcsec is not None:
        expected_x, expected_y = expected_center_arcsec
        peak_candidates = peak_candidates & (
            np.hypot(map_xx - expected_x, map_yy - expected_y)
            <= search_radius_arcsec
        )
        if not peak_candidates.any():
            raise ValueError("no finite samples near the expected source center")
    peak_flat = np.nanargmax(
        np.where(peak_candidates, values, np.nan)
    )
    peak_y, peak_x = np.unravel_index(peak_flat, values.shape)
    radius_px = max(4, int(math.ceil(25.0 / pixel_size_arcsec)))
    y0, y1 = max(0, peak_y - radius_px), min(values.shape[0], peak_y + radius_px + 1)
    x0, x1 = max(0, peak_x - radius_px), min(values.shape[1], peak_x + radius_px + 1)
    cutout = values[y0:y1, x0:x1]
    yy, xx = np.indices(cutout.shape, dtype=float)
    xx = (xx + x0 - (values.shape[1] - 1) / 2.0) * pixel_size_arcsec
    yy = (yy + y0 - (values.shape[0] - 1) / 2.0) * pixel_size_arcsec
    good = np.isfinite(cutout)
    background = float(np.nanmedian(cutout))
    amplitude = float(np.nanmax(cutout) - background)
    initial_sigma = max(2.0 * pixel_size_arcsec, 3.0)
    mean_bounds = {}
    if expected_center_arcsec is not None:
        expected_x, expected_y = expected_center_arcsec
        mean_bounds = {
            "x_mean": (
                expected_x - search_radius_arcsec,
                expected_x + search_radius_arcsec,
            ),
            "y_mean": (
                expected_y - search_radius_arcsec,
                expected_y + search_radius_arcsec,
            ),
        }
    model = models.Const2D(amplitude=background) + models.Gaussian2D(
        amplitude=max(amplitude, np.finfo(float).eps),
        x_mean=float(xx[peak_y - y0, peak_x - x0]),
        y_mean=float(yy[peak_y - y0, peak_x - x0]),
        x_stddev=initial_sigma,
        y_stddev=initial_sigma,
        theta=0.0,
        bounds={
            "amplitude": (0.0, None),
            "x_stddev": (pixel_size_arcsec / 4.0, 30.0),
            "y_stddev": (pixel_size_arcsec / 4.0, 30.0),
            **mean_bounds,
        },
    )
    fitted = fitting.TRFLSQFitter()(
        model, xx[good], yy[good], cutout[good], maxiter=1000
    )
    gaussian = fitted[1]
    x_fwhm = abs(float(gaussian.x_stddev.value)) * FWHM_FACTOR
    y_fwhm = abs(float(gaussian.y_stddev.value)) * FWHM_FACTOR
    result = {
        "amplitude": float(gaussian.amplitude.value),
        "x_arcsec": float(gaussian.x_mean.value),
        "y_arcsec": float(gaussian.y_mean.value),
        "major_fwhm_arcsec": max(x_fwhm, y_fwhm),
        "minor_fwhm_arcsec": min(x_fwhm, y_fwhm),
    }


    return result, np.asarray(gaussian(map_xx, map_yy), dtype=float)
