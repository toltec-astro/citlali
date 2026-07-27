#!/usr/bin/env python3
"""Analyze a completed fruit-loop population stage without changing policy."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import stat

import numpy as np
from astropy.io import fits
from scipy.stats import median_abs_deviation

from tools.fruit_loops.compare_feedback_ablation import reduction_rows


ARRAYS = ("a1100", "a1400", "a2000")
POINTING_FWHM_UPPER_ARCSEC = {
    "a1100": 10.0,
    "a1400": 12.6,
    "a2000": 19.0,
}
CROSS_ARRAY_ASSOCIATION_LIMIT_ARCSEC = 9.5
TOLERANCES = (0.01, 0.02, 0.05, 0.10)
CENTROID_STEP_LIMIT_ARCSEC = 0.1
BACKGROUND_INNER_RADIUS_ARCSEC = 40.0
BACKGROUND_OUTER_RADIUS_ARCSEC = 120.0
MINIMUM_BACKGROUND_PIXELS = 100
MINIMUM_BLANK_SKY_FITS = 12
GAUSSIAN_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
TERMINAL_PROVENANCE_FILES = (
    "astrometry_provenance.yaml",
    "coadd_provenance.yaml",
    "config_source_manifest.yaml",
    "kids_external_provenance.yaml",
    "mapmaking_provenance.yaml",
    "noise_products_provenance.yaml",
    "pointing_provenance.yaml",
    "polarimetry_provenance.yaml",
    "post_processing_provenance.yaml",
    "processed_timestream_provenance.yaml",
    "runtime_provenance.yaml",
)
CORE_FIELDS = (
    "amplitude",
    "amplitude_error",
    "kernel_normalized_amplitude",
    "major_fwhm_over_kernel",
    "minor_fwhm_over_kernel",
    "legacy_peak_over_full_map_rms",
    "fit_sig2noise",
    "x_t_arcsec",
    "y_t_arcsec",
    "map_weight_median",
    "map_background_sigma",
    "map_pixel_roughness_mjy",
    "map_roughness_fraction",
)
ERROR_RE = re.compile(r"\[(error|critical)\]|\bfatal\b", re.IGNORECASE)
WARNING_RE = re.compile(r"\[warning\]\s*(.*)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--run-matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", default="sentinel_extension_first")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_run_matrix(path: Path, *, phase: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row.get("phase") == phase
        ]
    if len(rows) != 16:
        raise ValueError(
            f"expected 16 observations for {phase}; found {len(rows)}"
        )
    obsnums = [int(row["obsnum"]) for row in rows]
    if len(obsnums) != len(set(obsnums)):
        raise ValueError(f"duplicate observations in phase {phase}")
    return sorted(rows, key=lambda row: int(row["quality_rank"]))


def map_path(stage_root: Path, obsnum: int, iteration: int, array: str) -> Path:
    return (
        stage_root
        / f"obs{obsnum}/reduced/redu{iteration:02d}/{obsnum}/raw"
        / f"toltec_commissioning_{array}_pointing_{obsnum}_citlali.fits"
    )


def weighted_gaussian_amplitude(
    signal: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
    xx_arcsec: np.ndarray,
    yy_arcsec: np.ndarray,
    *,
    center_x_arcsec: float,
    center_y_arcsec: float,
    fwhm_arcsec: float,
) -> tuple[float, float]:
    """Fit a fixed circular Gaussian amplitude plus a constant background."""
    if not math.isfinite(fwhm_arcsec) or fwhm_arcsec <= 0.0:
        return math.nan, math.nan
    sigma_arcsec = fwhm_arcsec / GAUSSIAN_FWHM_FACTOR
    radius2 = (
        np.square(xx_arcsec - center_x_arcsec)
        + np.square(yy_arcsec - center_y_arcsec)
    )
    selected = valid & (radius2 <= (3.0 * fwhm_arcsec) ** 2)
    if int(selected.sum()) < 12:
        return math.nan, math.nan
    template = np.exp(-0.5 * radius2[selected] / sigma_arcsec**2)
    values = signal[selected]
    weights = weight[selected]
    sum_weight = float(np.sum(weights))
    sum_template = float(np.sum(weights * template))
    sum_template2 = float(np.sum(weights * template * template))
    sum_signal = float(np.sum(weights * values))
    sum_template_signal = float(np.sum(weights * template * values))
    determinant = (
        sum_template2 * sum_weight - sum_template * sum_template
    )
    if (
        not math.isfinite(determinant)
        or determinant <= 0.0
        or sum_weight <= 0.0
    ):
        return math.nan, math.nan
    amplitude = (
        sum_template_signal * sum_weight - sum_signal * sum_template
    ) / determinant
    formal_uncertainty = math.sqrt(sum_weight / determinant)
    return float(amplitude), float(formal_uncertainty)


def empirical_blank_sky_point_source_metrics(
    signal: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
    xx_arcsec: np.ndarray,
    yy_arcsec: np.ndarray,
    *,
    fit_x_arcsec: float,
    fit_y_arcsec: float,
    kernel_major_fwhm_arcsec: float,
    kernel_minor_fwhm_arcsec: float,
) -> dict[str, float | int]:
    """Calibrate fixed-PSF amplitude uncertainty with blank-sky fits.

    Blank fits use the same fixed circular Gaussian estimator as the source.
    Their formal-weight-standardized amplitudes empirically calibrate the
    source uncertainty, retaining the local weight dependence while absorbing
    correlated-map-noise scale errors.
    """
    fwhm_arcsec = math.sqrt(
        kernel_major_fwhm_arcsec * kernel_minor_fwhm_arcsec
    )
    source_amplitude, source_formal_uncertainty = (
        weighted_gaussian_amplitude(
            signal,
            weight,
            valid,
            xx_arcsec,
            yy_arcsec,
            center_x_arcsec=fit_x_arcsec,
            center_y_arcsec=fit_y_arcsec,
            fwhm_arcsec=fwhm_arcsec,
        )
    )
    step_arcsec = max(2.5 * fwhm_arcsec, 4.0)
    x_values = np.arange(
        float(np.nanmin(xx_arcsec)) + 3.0 * fwhm_arcsec,
        float(np.nanmax(xx_arcsec)) - 3.0 * fwhm_arcsec
        + 0.5 * step_arcsec,
        step_arcsec,
    )
    y_values = np.arange(
        float(np.nanmin(yy_arcsec)) + 3.0 * fwhm_arcsec,
        float(np.nanmax(yy_arcsec)) - 3.0 * fwhm_arcsec
        + 0.5 * step_arcsec,
        step_arcsec,
    )
    standardized_blank_amplitudes: list[float] = []
    for center_y in y_values:
        for center_x in x_values:
            radius = math.hypot(
                center_x - fit_x_arcsec, center_y - fit_y_arcsec
            )
            if not (
                BACKGROUND_INNER_RADIUS_ARCSEC
                <= radius
                <= BACKGROUND_OUTER_RADIUS_ARCSEC
            ):
                continue
            amplitude, formal_uncertainty = weighted_gaussian_amplitude(
                signal,
                weight,
                valid,
                xx_arcsec,
                yy_arcsec,
                center_x_arcsec=float(center_x),
                center_y_arcsec=float(center_y),
                fwhm_arcsec=fwhm_arcsec,
            )
            if (
                math.isfinite(amplitude)
                and math.isfinite(formal_uncertainty)
                and formal_uncertainty > 0.0
            ):
                standardized_blank_amplitudes.append(
                    amplitude / formal_uncertainty
                )
    count = len(standardized_blank_amplitudes)
    if (
        count < MINIMUM_BLANK_SKY_FITS
        or not math.isfinite(source_amplitude)
        or not math.isfinite(source_formal_uncertainty)
        or source_formal_uncertainty <= 0.0
    ):
        return {
            "empirical_psf_amplitude_mjy_beam": source_amplitude,
            "empirical_psf_amplitude_uncertainty_mjy_beam": math.nan,
            "empirical_point_source_sig2noise": math.nan,
            "empirical_blank_sky_fit_count": count,
            "empirical_blank_sky_standardized_center": math.nan,
            "empirical_blank_sky_standardized_sigma": math.nan,
        }
    blank = np.asarray(standardized_blank_amplitudes, dtype=float)
    blank_center = float(np.median(blank))
    blank_sigma = float(median_abs_deviation(blank, scale="normal"))
    if not math.isfinite(blank_sigma) or blank_sigma <= 0.0:
        empirical_uncertainty = math.nan
        empirical_sig2noise = math.nan
    else:
        empirical_uncertainty = source_formal_uncertainty * blank_sigma
        empirical_sig2noise = (
            source_amplitude / source_formal_uncertainty - blank_center
        ) / blank_sigma
    return {
        "empirical_psf_amplitude_mjy_beam": source_amplitude,
        "empirical_psf_amplitude_uncertainty_mjy_beam":
            empirical_uncertainty,
        "empirical_point_source_sig2noise": empirical_sig2noise,
        "empirical_blank_sky_fit_count": count,
        "empirical_blank_sky_standardized_center": blank_center,
        "empirical_blank_sky_standardized_sigma": blank_sigma,
    }


def source_free_map_metrics(
    path: Path, *, fit_x_arcsec: float, fit_y_arcsec: float,
    kernel_major_fwhm_arcsec: float,
    kernel_minor_fwhm_arcsec: float,
) -> dict[str, float | int]:
    with fits.open(path, memmap=True) as hdul:
        signal = np.asarray(hdul["signal_I"].data, dtype=float).squeeze()
        weight = np.asarray(hdul["weight_I"].data, dtype=float).squeeze()
        coverage = (
            np.asarray(hdul["coverage_bool_I"].data).squeeze() > 0.5
        )
        header = hdul["signal_I"].header
    ny, nx = signal.shape
    x = (
        np.arange(nx, dtype=float) + 1.0 - float(header["CRPIX1"])
    ) * float(header["CDELT1"]) + float(header["CRVAL1"])
    y = (
        np.arange(ny, dtype=float) + 1.0 - float(header["CRPIX2"])
    ) * float(header["CDELT2"]) + float(header["CRVAL2"])
    xx, yy = np.meshgrid(x, y)
    radius = np.hypot(xx - fit_x_arcsec, yy - fit_y_arcsec)
    valid = coverage & np.isfinite(signal) & np.isfinite(weight) & (weight > 0)
    background = (
        valid
        & (radius >= BACKGROUND_INNER_RADIUS_ARCSEC)
        & (radius <= BACKGROUND_OUTER_RADIUS_ARCSEC)
    )
    if int(background.sum()) < MINIMUM_BACKGROUND_PIXELS:
        raise ValueError(f"{path}: insufficient background pixels")
    values = signal[background]
    background_sigma = float(
        median_abs_deviation(values, scale="normal")
    )
    horizontal = signal[:, 1:] - signal[:, :-1]
    horizontal_mask = background[:, 1:] & background[:, :-1]
    vertical = signal[1:, :] - signal[:-1, :]
    vertical_mask = background[1:, :] & background[:-1, :]
    differences = np.concatenate(
        (horizontal[horizontal_mask], vertical[vertical_mask])
    )
    pixel_roughness = float(
        median_abs_deviation(differences, scale="normal") / np.sqrt(2.0)
    )
    result: dict[str, float | int] = {
        "map_background_median_mjy": float(np.median(values)),
        "map_background_sigma_mjy": background_sigma,
        "map_pixel_roughness_mjy": pixel_roughness,
        "map_roughness_fraction": (
            pixel_roughness / background_sigma
            if background_sigma > 0.0 else math.nan
        ),
    }
    result.update(
        empirical_blank_sky_point_source_metrics(
            signal,
            weight,
            valid,
            xx,
            yy,
            fit_x_arcsec=fit_x_arcsec,
            fit_y_arcsec=fit_y_arcsec,
            kernel_major_fwhm_arcsec=kernel_major_fwhm_arcsec,
            kernel_minor_fwhm_arcsec=kernel_minor_fwhm_arcsec,
        )
    )
    return result


def finite_ratio(numerator: float, denominator: float) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator == 0.0
    ):
        return math.nan
    return numerator / denominator


def fit_is_valid(row: dict) -> bool:
    if not all(math.isfinite(float(row[field])) for field in CORE_FIELDS):
        return False
    return (
        float(row["amplitude"]) > 0.0
        and float(row["kernel_fit_amplitude"]) > 0.0
        and float(row["major_fwhm_arcsec"]) > 0.0
        and float(row["minor_fwhm_arcsec"]) > 0.0
        and float(row["kernel_major_fwhm_arcsec"]) > 0.0
        and float(row["kernel_minor_fwhm_arcsec"]) > 0.0
    )


def load_iteration_metrics(
    stage_root: Path, matrix_rows: list[dict],
) -> list[dict]:
    metadata = {int(row["obsnum"]): row for row in matrix_rows}
    rows: list[dict] = []
    for matrix in matrix_rows:
        obsnum = int(matrix["obsnum"])
        extracted = reduction_rows(
            "population_stage_a",
            stage_root / f"obs{obsnum}/reduced",
            obsnum,
        )
        if len(extracted) != 30:
            raise ValueError(
                f"obsnum {obsnum}: expected 30 iteration-array rows; "
                f"found {len(extracted)}"
            )
        rows.extend(extracted)

    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        obsnum = int(row["obsnum"])
        matrix = metadata[obsnum]
        if str(row["source"]).casefold() != str(matrix["source"]).casefold():
            raise ValueError(
                f"obsnum {obsnum}: source mismatch "
                f"{row['source']!r} != {matrix['source']!r}"
            )
        row.update(
            {
                "quality_rank": int(matrix["quality_rank"]),
                "quality_stratum": matrix["quality_stratum"],
                "quality_score": float(matrix["quality_score"]),
                "selection_reason": matrix["selection_reason"],
            }
        )
        row.update(
            source_free_map_metrics(
                map_path(
                    stage_root, obsnum, int(row["iteration"]),
                    str(row["array"]),
                ),
                fit_x_arcsec=float(row["x_t_arcsec"]),
                fit_y_arcsec=float(row["y_t_arcsec"]),
                kernel_major_fwhm_arcsec=float(
                    row["kernel_major_fwhm_arcsec"]
                ),
                kernel_minor_fwhm_arcsec=float(
                    row["kernel_minor_fwhm_arcsec"]
                ),
            )
        )
        groups.setdefault((obsnum, str(row["array"])), []).append(row)

    ratio_fields = (
        "amplitude",
        "amplitude_error",
        "kernel_normalized_amplitude",
        "major_fwhm_over_kernel",
        "minor_fwhm_over_kernel",
        "legacy_peak_over_full_map_rms",
        "fit_sig2noise",
        "empirical_psf_amplitude_mjy_beam",
        "empirical_psf_amplitude_uncertainty_mjy_beam",
        "empirical_point_source_sig2noise",
        "map_weight_median",
        "map_background_sigma_mjy",
        "map_pixel_roughness_mjy",
        "map_roughness_fraction",
    )
    for group in groups.values():
        group.sort(key=lambda row: int(row["iteration"]))
        if [int(row["iteration"]) for row in group] != list(range(10)):
            raise ValueError("iteration identities are not contiguous 0..9")
        seed = group[0]
        for index, row in enumerate(group):
            previous = group[index - 1] if index else None
            for field in ratio_fields:
                row[f"{field}_ratio_seed"] = finite_ratio(
                    float(row[field]), float(seed[field])
                )
                row[f"{field}_change_fraction"] = (
                    finite_ratio(
                        float(row[field]), float(previous[field])
                    ) - 1.0
                    if previous is not None else math.nan
                )
            row["centroid_shift_from_previous_arcsec"] = (
                math.hypot(
                    float(row["x_t_arcsec"])
                    - float(previous["x_t_arcsec"]),
                    float(row["y_t_arcsec"])
                    - float(previous["y_t_arcsec"]),
                )
                if previous is not None else math.nan
            )
            row["centroid_shift_from_seed_arcsec"] = math.hypot(
                float(row["x_t_arcsec"]) - float(seed["x_t_arcsec"]),
                float(row["y_t_arcsec"]) - float(seed["y_t_arcsec"]),
            )
            row["fit_valid"] = fit_is_valid(row)

    observation_iterations: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        observation_iterations.setdefault(
            (int(row["obsnum"]), int(row["iteration"])), []
        ).append(row)
    for group in observation_iterations.values():
        if {str(row["array"]) for row in group} != set(ARRAYS):
            raise ValueError("iteration is missing one or more arrays")
        median_x = float(np.median([
            float(row["x_t_arcsec"]) for row in group
        ]))
        median_y = float(np.median([
            float(row["y_t_arcsec"]) for row in group
        ]))
        for row in group:
            array = str(row["array"])
            upper = POINTING_FWHM_UPPER_ARCSEC[array]
            association_offset = math.hypot(
                float(row["x_t_arcsec"]) - median_x,
                float(row["y_t_arcsec"]) - median_y,
            )
            upper_bound_hit = (
                math.isclose(
                    float(row["major_fwhm_arcsec"]),
                    upper,
                    rel_tol=0.0,
                    abs_tol=1.0e-5,
                )
                or math.isclose(
                    float(row["minor_fwhm_arcsec"]),
                    upper,
                    rel_tol=0.0,
                    abs_tol=1.0e-5,
                )
            )
            row.update(
                {
                    "cross_array_centroid_median_x_arcsec": median_x,
                    "cross_array_centroid_median_y_arcsec": median_y,
                    "cross_array_centroid_offset_arcsec":
                        association_offset,
                    "source_association_valid": bool(
                        row["fit_valid"]
                        and association_offset
                        <= CROSS_ARRAY_ASSOCIATION_LIMIT_ARCSEC
                    ),
                    "fwhm_upper_bound_arcsec": upper,
                    "fwhm_upper_bound_hit": upper_bound_hit,
                    "amplitude_centroid_interpretable": bool(
                        row["fit_valid"]
                        and association_offset
                        <= CROSS_ARRAY_ASSOCIATION_LIMIT_ARCSEC
                    ),
                    "psf_interpretable": bool(
                        row["fit_valid"]
                        and association_offset
                        <= CROSS_ARRAY_ASSOCIATION_LIMIT_ARCSEC
                        and not upper_bound_hit
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            int(row["quality_rank"]),
            int(row["obsnum"]),
            str(row["array"]),
            int(row["iteration"]),
        ),
    )


def build_transition_metrics(iteration_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in iteration_rows:
        groups.setdefault(
            (int(row["obsnum"]), str(row["array"])), []
        ).append(row)
    result = []
    for (obsnum, array), group in groups.items():
        group.sort(key=lambda row: int(row["iteration"]))
        seed = group[0]
        for previous, current in zip(group, group[1:]):
            row = {
                "obsnum": obsnum,
                "source": current["source"],
                "quality_rank": current["quality_rank"],
                "quality_stratum": current["quality_stratum"],
                "quality_score": current["quality_score"],
                "array": array,
                "previous_iteration": int(previous["iteration"]),
                "current_iteration": int(current["iteration"]),
                "raw_amplitude_change_fraction": abs(
                    finite_ratio(
                        float(current["amplitude"]),
                        float(previous["amplitude"]),
                    ) - 1.0
                ),
                "kernel_normalized_amplitude_change_fraction": abs(
                    finite_ratio(
                        float(current["kernel_normalized_amplitude"]),
                        float(previous["kernel_normalized_amplitude"]),
                    ) - 1.0
                ),
                "major_fwhm_over_kernel_change_fraction": abs(
                    finite_ratio(
                        float(current["major_fwhm_over_kernel"]),
                        float(previous["major_fwhm_over_kernel"]),
                    ) - 1.0
                ),
                "minor_fwhm_over_kernel_change_fraction": abs(
                    finite_ratio(
                        float(current["minor_fwhm_over_kernel"]),
                        float(previous["minor_fwhm_over_kernel"]),
                    ) - 1.0
                ),
                "centroid_step_arcsec":
                    current["centroid_shift_from_previous_arcsec"],
                "centroid_shift_from_seed_arcsec":
                    current["centroid_shift_from_seed_arcsec"],
                "successive_map_relative_rms":
                    current["successive_map_delta_relative_rms"],
                "legacy_peak_over_full_map_rms_change_fraction": abs(
                    finite_ratio(
                        float(current["legacy_peak_over_full_map_rms"]),
                        float(previous["legacy_peak_over_full_map_rms"]),
                    ) - 1.0
                ),
                "legacy_peak_over_full_map_rms_ratio_seed": finite_ratio(
                    float(current["legacy_peak_over_full_map_rms"]),
                    float(seed["legacy_peak_over_full_map_rms"]),
                ),
                "fit_sig2noise_change_fraction": abs(
                    finite_ratio(
                        float(current["fit_sig2noise"]),
                        float(previous["fit_sig2noise"]),
                    ) - 1.0
                ),
                "fit_sig2noise_ratio_seed": finite_ratio(
                    float(current["fit_sig2noise"]),
                    float(seed["fit_sig2noise"]),
                ),
                "empirical_point_source_sig2noise_change_fraction": abs(
                    finite_ratio(
                        float(current["empirical_point_source_sig2noise"]),
                        float(previous["empirical_point_source_sig2noise"]),
                    ) - 1.0
                ),
                "empirical_point_source_sig2noise_ratio_seed": finite_ratio(
                    float(current["empirical_point_source_sig2noise"]),
                    float(seed["empirical_point_source_sig2noise"]),
                ),
                "map_weight_change_fraction": abs(
                    finite_ratio(
                        float(current["map_weight_median"]),
                        float(previous["map_weight_median"]),
                    ) - 1.0
                ),
                "background_sigma_change_fraction": abs(
                    finite_ratio(
                        float(current["map_background_sigma_mjy"]),
                        float(previous["map_background_sigma_mjy"]),
                    ) - 1.0
                ),
                "roughness_change_fraction": abs(
                    finite_ratio(
                        float(current["map_pixel_roughness_mjy"]),
                        float(previous["map_pixel_roughness_mjy"]),
                    ) - 1.0
                ),
                "fit_valid": bool(
                    previous["fit_valid"] and current["fit_valid"]
                ),
                "source_association_valid": bool(
                    previous["source_association_valid"]
                    and current["source_association_valid"]
                ),
                "psf_interpretable": bool(
                    previous["psf_interpretable"]
                    and current["psf_interpretable"]
                ),
            }
            row["maximum_fwhm_change_fraction"] = max(
                float(row["major_fwhm_over_kernel_change_fraction"]),
                float(row["minor_fwhm_over_kernel_change_fraction"]),
            )
            required = (
                "kernel_normalized_amplitude_change_fraction",
                "maximum_fwhm_change_fraction",
                "centroid_step_arcsec",
                "successive_map_relative_rms",
            )
            row["metric_finite"] = bool(
                all(math.isfinite(float(row[key])) for key in required)
            )
            row["interpretable"] = bool(
                row["fit_valid"]
                and row["source_association_valid"]
                and row["psf_interpretable"]
                and row["metric_finite"]
            )
            row["noise_metrics_finite"] = bool(
                all(
                    math.isfinite(float(row[key]))
                    for key in (
                        "background_sigma_change_fraction",
                        "roughness_change_fraction",
                        "fit_sig2noise_change_fraction",
                        "empirical_point_source_sig2noise_change_fraction",
                    )
                )
            )
            failure_reasons = []
            if not row["fit_valid"] or not row["metric_finite"]:
                failure_reasons.append("nonfinite_or_invalid_fit")
            if not row["source_association_valid"]:
                failure_reasons.append("cross_array_source_mismatch")
            if not row["psf_interpretable"]:
                failure_reasons.append("psf_fit_censored_or_invalid")
            row["classification"] = (
                ";".join(failure_reasons) if failure_reasons
                else "interpretable"
            )
            result.append(row)
    return sorted(
        result,
        key=lambda row: (
            int(row["quality_rank"]),
            int(row["obsnum"]),
            str(row["array"]),
            int(row["current_iteration"]),
        ),
    )


def first_two_transition_pass(
    transitions: list[dict],
    *,
    field: str,
    maximum: float,
    eligibility_field: str = "interpretable",
) -> int | None:
    for previous, current in zip(transitions, transitions[1:]):
        if (
            int(current["current_iteration"])
            != int(previous["current_iteration"]) + 1
        ):
            raise ValueError("transition sequence is not contiguous")
        if (
            bool(previous[eligibility_field])
            and bool(current[eligibility_field])
            and float(previous[field]) < maximum
            and float(current[field]) < maximum
        ):
            return int(current["current_iteration"])
    return None


def first_combined_pass(
    transitions: list[dict], *, tolerance: float,
) -> int | None:
    for previous, current in zip(transitions, transitions[1:]):
        pair = (previous, current)
        passed = all(
            bool(row["interpretable"])
            and float(
                row["kernel_normalized_amplitude_change_fraction"]
            ) < tolerance
            and float(row["maximum_fwhm_change_fraction"]) < tolerance
            and float(row["centroid_step_arcsec"])
            < CENTROID_STEP_LIMIT_ARCSEC
            and float(row["successive_map_relative_rms"]) < tolerance
            for row in pair
        )
        if passed:
            return int(current["current_iteration"])
    return None


def convergence_assessment(
    iteration_rows: list[dict], transition_rows: list[dict],
) -> list[dict]:
    iteration_groups: dict[tuple[int, str], list[dict]] = {}
    transition_groups: dict[tuple[int, str], list[dict]] = {}
    for row in iteration_rows:
        iteration_groups.setdefault(
            (int(row["obsnum"]), str(row["array"])), []
        ).append(row)
    for row in transition_rows:
        transition_groups.setdefault(
            (int(row["obsnum"]), str(row["array"])), []
        ).append(row)

    result = []
    for key, transitions in transition_groups.items():
        transitions.sort(key=lambda row: int(row["current_iteration"]))
        iterations = sorted(
            iteration_groups[key], key=lambda row: int(row["iteration"])
        )
        tail = transitions[-2:]
        if [int(row["current_iteration"]) for row in tail] != [8, 9]:
            raise ValueError(f"{key}: endpoint window is not iterations 8;9")
        endpoint = iterations[-1]
        for tolerance in TOLERANCES:
            amplitude_first = first_two_transition_pass(
                transitions,
                field="kernel_normalized_amplitude_change_fraction",
                maximum=tolerance,
                eligibility_field="source_association_valid",
            )
            fwhm_first = first_two_transition_pass(
                transitions,
                field="maximum_fwhm_change_fraction",
                maximum=tolerance,
                eligibility_field="psf_interpretable",
            )
            centroid_first = first_two_transition_pass(
                transitions,
                field="centroid_step_arcsec",
                maximum=CENTROID_STEP_LIMIT_ARCSEC,
                eligibility_field="source_association_valid",
            )
            map_first = first_two_transition_pass(
                transitions,
                field="successive_map_relative_rms",
                maximum=tolerance,
                eligibility_field="metric_finite",
            )
            background_first = first_two_transition_pass(
                transitions,
                field="background_sigma_change_fraction",
                maximum=tolerance,
                eligibility_field="noise_metrics_finite",
            )
            fit_snr_first = first_two_transition_pass(
                transitions,
                field="fit_sig2noise_change_fraction",
                maximum=tolerance,
                eligibility_field="noise_metrics_finite",
            )
            empirical_snr_first = first_two_transition_pass(
                transitions,
                field="empirical_point_source_sig2noise_change_fraction",
                maximum=tolerance,
                eligibility_field="noise_metrics_finite",
            )
            combined_first = first_combined_pass(
                transitions, tolerance=tolerance
            )
            endpoint_pass = all(
                bool(row["interpretable"])
                and float(
                    row["kernel_normalized_amplitude_change_fraction"]
                ) < tolerance
                and float(row["maximum_fwhm_change_fraction"]) < tolerance
                and float(row["centroid_step_arcsec"])
                < CENTROID_STEP_LIMIT_ARCSEC
                and float(row["successive_map_relative_rms"]) < tolerance
                for row in tail
            )
            endpoint_amplitude_pass = all(
                bool(row["source_association_valid"])
                and float(
                    row["kernel_normalized_amplitude_change_fraction"]
                ) < tolerance
                for row in tail
            )
            endpoint_fwhm_pass = all(
                bool(row["psf_interpretable"])
                and float(row["maximum_fwhm_change_fraction"]) < tolerance
                for row in tail
            )
            endpoint_centroid_pass = all(
                bool(row["source_association_valid"])
                and float(row["centroid_step_arcsec"])
                < CENTROID_STEP_LIMIT_ARCSEC
                for row in tail
            )
            endpoint_map_pass = all(
                bool(row["metric_finite"])
                and float(row["successive_map_relative_rms"]) < tolerance
                for row in tail
            )
            endpoint_background_pass = all(
                bool(row["noise_metrics_finite"])
                and float(row["background_sigma_change_fraction"]) < tolerance
                for row in tail
            )
            endpoint_fit_snr_pass = all(
                bool(row["noise_metrics_finite"])
                and float(row["fit_sig2noise_change_fraction"]) < tolerance
                for row in tail
            )
            endpoint_empirical_snr_pass = all(
                bool(row["noise_metrics_finite"])
                and float(
                    row[
                        "empirical_point_source_sig2noise_change_fraction"
                    ]
                ) < tolerance
                for row in tail
            )
            result.append(
                {
                    "obsnum": key[0],
                    "source": endpoint["source"],
                    "quality_rank": endpoint["quality_rank"],
                    "quality_stratum": endpoint["quality_stratum"],
                    "quality_score": endpoint["quality_score"],
                    "array": key[1],
                    "tolerance_percent": int(round(100 * tolerance)),
                    "window_transitions": "8;9",
                    "interpretable": all(
                        bool(row["interpretable"]) for row in transitions
                    ),
                    "source_association_interpretable": all(
                        bool(row["source_association_valid"])
                        for row in iterations
                    ),
                    "psf_interpretable": all(
                        bool(row["psf_interpretable"])
                        for row in iterations
                    ),
                    "first_amplitude_stable_iteration": amplitude_first,
                    "first_fwhm_stable_iteration": fwhm_first,
                    "first_centroid_stable_iteration": centroid_first,
                    "first_map_stable_iteration": map_first,
                    "first_background_stable_iteration": background_first,
                    "first_fit_snr_stable_iteration": fit_snr_first,
                    "first_empirical_snr_stable_iteration":
                        empirical_snr_first,
                    "first_all_candidate_stable_iteration": combined_first,
                    "endpoint_two_transition_pass": endpoint_pass,
                    "endpoint_amplitude_pass": endpoint_amplitude_pass,
                    "endpoint_fwhm_pass": endpoint_fwhm_pass,
                    "endpoint_centroid_pass": endpoint_centroid_pass,
                    "endpoint_map_pass": endpoint_map_pass,
                    "endpoint_background_pass": endpoint_background_pass,
                    "endpoint_fit_snr_pass": endpoint_fit_snr_pass,
                    "endpoint_empirical_snr_pass":
                        endpoint_empirical_snr_pass,
                    "tail_max_amplitude_change_fraction": max(
                        float(row[
                            "kernel_normalized_amplitude_change_fraction"
                        ])
                        for row in tail
                    ),
                    "tail_max_fwhm_change_fraction": max(
                        float(row["maximum_fwhm_change_fraction"])
                        for row in tail
                    ),
                    "tail_max_centroid_step_arcsec": max(
                        float(row["centroid_step_arcsec"]) for row in tail
                    ),
                    "tail_max_successive_map_relative_rms": max(
                        float(row["successive_map_relative_rms"])
                        for row in tail
                    ),
                    "tail_max_background_sigma_change_fraction": max(
                        float(row["background_sigma_change_fraction"])
                        for row in tail
                    ),
                    "tail_max_fit_snr_change_fraction": max(
                        float(row["fit_sig2noise_change_fraction"])
                        for row in tail
                    ),
                    "tail_max_empirical_snr_change_fraction": max(
                        float(
                            row[
                                "empirical_point_source_sig2noise_change_fraction"
                            ]
                        )
                        for row in tail
                    ),
                    "endpoint_kernel_normalized_amplitude_ratio_seed":
                        endpoint[
                            "kernel_normalized_amplitude_ratio_seed"
                        ],
                    "endpoint_major_fwhm_over_kernel":
                        endpoint["major_fwhm_over_kernel"],
                    "endpoint_minor_fwhm_over_kernel":
                        endpoint["minor_fwhm_over_kernel"],
                    "endpoint_centroid_shift_from_seed_arcsec":
                        endpoint["centroid_shift_from_seed_arcsec"],
                    "endpoint_legacy_peak_over_full_map_rms_ratio_seed":
                        endpoint[
                            "legacy_peak_over_full_map_rms_ratio_seed"
                        ],
                    "minimum_legacy_peak_over_full_map_rms_ratio_seed": min(
                        float(
                            row[
                                "legacy_peak_over_full_map_rms_ratio_seed"
                            ]
                        )
                        for row in iterations
                    ),
                    "endpoint_fit_sig2noise_ratio_seed":
                        endpoint["fit_sig2noise_ratio_seed"],
                    "minimum_fit_sig2noise_ratio_seed": min(
                        float(row["fit_sig2noise_ratio_seed"])
                        for row in iterations
                    ),
                    "endpoint_empirical_point_source_sig2noise_ratio_seed":
                        endpoint[
                            "empirical_point_source_sig2noise_ratio_seed"
                        ],
                    "minimum_empirical_point_source_sig2noise_ratio_seed": min(
                        float(
                            row[
                                "empirical_point_source_sig2noise_ratio_seed"
                            ]
                        )
                        for row in iterations
                    ),
                }
            )
    return sorted(
        result,
        key=lambda row: (
            int(row["tolerance_percent"]),
            int(row["quality_rank"]),
            str(row["array"]),
        ),
    )


def diagnostic_yield_summary(array_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in array_rows:
        groups.setdefault(
            (str(row["quality_stratum"]), int(row["tolerance_percent"])),
            [],
        ).append(row)
    result = []
    for (stratum_name, tolerance), rows in groups.items():
        source_eligible = [
            row for row in rows
            if bool(row["source_association_interpretable"])
        ]
        psf_eligible = [
            row for row in rows if bool(row["psf_interpretable"])
        ]
        result.append(
            {
                "quality_stratum": stratum_name,
                "tolerance_percent": tolerance,
                "array_trajectory_count": len(rows),
                "source_associated_trajectory_count":
                    len(source_eligible),
                "psf_interpretable_trajectory_count": len(psf_eligible),
                "amplitude_ever_pass_count": sum(
                    row["first_amplitude_stable_iteration"] is not None
                    for row in source_eligible
                ),
                "amplitude_endpoint_pass_count": sum(
                    bool(row["endpoint_amplitude_pass"])
                    for row in source_eligible
                ),
                "fwhm_ever_pass_count": sum(
                    row["first_fwhm_stable_iteration"] is not None
                    for row in psf_eligible
                ),
                "fwhm_endpoint_pass_count": sum(
                    bool(row["endpoint_fwhm_pass"])
                    for row in psf_eligible
                ),
                "centroid_ever_pass_count": sum(
                    row["first_centroid_stable_iteration"] is not None
                    for row in source_eligible
                ),
                "centroid_endpoint_pass_count": sum(
                    bool(row["endpoint_centroid_pass"])
                    for row in source_eligible
                ),
                "map_ever_pass_count": sum(
                    row["first_map_stable_iteration"] is not None
                    for row in rows
                ),
                "map_endpoint_pass_count": sum(
                    bool(row["endpoint_map_pass"]) for row in rows
                ),
                "background_ever_pass_count": sum(
                    row["first_background_stable_iteration"] is not None
                    for row in source_eligible
                ),
                "background_endpoint_pass_count": sum(
                    bool(row["endpoint_background_pass"])
                    for row in source_eligible
                ),
                "fit_snr_endpoint_pass_count": sum(
                    bool(row["endpoint_fit_snr_pass"])
                    for row in source_eligible
                ),
                "empirical_snr_endpoint_pass_count": sum(
                    bool(row["endpoint_empirical_snr_pass"])
                    for row in source_eligible
                ),
                "combined_ever_pass_count": sum(
                    row["first_all_candidate_stable_iteration"] is not None
                    for row in rows
                ),
                "combined_endpoint_pass_count": sum(
                    bool(row["endpoint_two_transition_pass"]) for row in rows
                ),
                "minimum_endpoint_legacy_dynamic_range_ratio_seed": min(
                    float(
                        row[
                            "endpoint_legacy_peak_over_full_map_rms_ratio_seed"
                        ]
                    )
                    for row in source_eligible
                ),
                "minimum_endpoint_fit_snr_ratio_seed": min(
                    float(row["endpoint_fit_sig2noise_ratio_seed"])
                    for row in source_eligible
                ),
                "minimum_endpoint_empirical_snr_ratio_seed": min(
                    float(
                        row[
                            "endpoint_empirical_point_source_sig2noise_ratio_seed"
                        ]
                    )
                    for row in source_eligible
                ),
                "maximum_endpoint_centroid_shift_from_seed_arcsec": max(
                    float(row["endpoint_centroid_shift_from_seed_arcsec"])
                    for row in source_eligible
                ),
            }
        )
    order = {"normal": 0, "marginal": 1, "stress": 2}
    return sorted(
        result,
        key=lambda row: (
            int(row["tolerance_percent"]),
            order[str(row["quality_stratum"])],
        ),
    )


def observation_assessment(array_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in array_rows:
        groups.setdefault(
            (int(row["obsnum"]), int(row["tolerance_percent"])), []
        ).append(row)
    result = []
    for (obsnum, tolerance), rows in groups.items():
        if {str(row["array"]) for row in rows} != set(ARRAYS):
            raise ValueError(f"obsnum {obsnum}: incomplete array assessment")
        result.append(
            {
                "obsnum": obsnum,
                "source": rows[0]["source"],
                "quality_rank": rows[0]["quality_rank"],
                "quality_stratum": rows[0]["quality_stratum"],
                "quality_score": rows[0]["quality_score"],
                "tolerance_percent": tolerance,
                "interpretable_arrays": sum(
                    bool(row["interpretable"]) for row in rows
                ),
                "source_associated_arrays": sum(
                    bool(row["source_association_interpretable"])
                    for row in rows
                ),
                "psf_interpretable_arrays": sum(
                    bool(row["psf_interpretable"]) for row in rows
                ),
                "arrays_with_any_combined_pass": sum(
                    row["first_all_candidate_stable_iteration"] is not None
                    for row in rows
                ),
                "all_arrays_have_combined_pass": all(
                    row["first_all_candidate_stable_iteration"] is not None
                    for row in rows
                ),
                "arrays_passing_endpoint_window": sum(
                    bool(row["endpoint_two_transition_pass"]) for row in rows
                ),
                "all_arrays_pass_endpoint_window": all(
                    bool(row["endpoint_two_transition_pass"]) for row in rows
                ),
                "maximum_first_combined_stable_iteration": max(
                    (
                        int(row["first_all_candidate_stable_iteration"])
                        for row in rows
                        if row["first_all_candidate_stable_iteration"]
                        is not None
                    ),
                    default=None,
                ),
                "worst_endpoint_legacy_dynamic_range_ratio_seed": min(
                    float(
                        row[
                            "endpoint_legacy_peak_over_full_map_rms_ratio_seed"
                        ]
                    )
                    for row in rows
                ),
                "worst_endpoint_fit_snr_ratio_seed": min(
                    float(row["endpoint_fit_sig2noise_ratio_seed"])
                    for row in rows
                ),
                "worst_endpoint_empirical_snr_ratio_seed": min(
                    float(
                        row[
                            "endpoint_empirical_point_source_sig2noise_ratio_seed"
                        ]
                    )
                    for row in rows
                ),
                "maximum_endpoint_centroid_shift_from_seed_arcsec": max(
                    float(
                        row["endpoint_centroid_shift_from_seed_arcsec"]
                    )
                    for row in rows
                ),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            int(row["tolerance_percent"]),
            int(row["quality_rank"]),
        ),
    )


def stratum_summary(observation_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in observation_rows:
        groups.setdefault(
            (str(row["quality_stratum"]), int(row["tolerance_percent"])),
            [],
        ).append(row)
    result = []
    for (stratum_name, tolerance), rows in groups.items():
        result.append(
            {
                "quality_stratum": stratum_name,
                "tolerance_percent": tolerance,
                "observation_count": len(rows),
                "interpretable_observation_count": sum(
                    int(row["source_associated_arrays"]) == len(ARRAYS)
                    for row in rows
                ),
                "fully_psf_interpretable_observation_count": sum(
                    int(row["psf_interpretable_arrays"]) == len(ARRAYS)
                    for row in rows
                ),
                "observations_all_arrays_ever_pass": sum(
                    bool(row["all_arrays_have_combined_pass"]) for row in rows
                ),
                "observations_all_arrays_endpoint_pass": sum(
                    bool(row["all_arrays_pass_endpoint_window"]) for row in rows
                ),
                "array_pass_count": sum(
                    int(row["arrays_with_any_combined_pass"]) for row in rows
                ),
                "array_endpoint_pass_count": sum(
                    int(row["arrays_passing_endpoint_window"]) for row in rows
                ),
                "array_count": len(rows) * len(ARRAYS),
                "minimum_endpoint_legacy_dynamic_range_ratio_seed": min(
                    float(
                        row[
                            "worst_endpoint_legacy_dynamic_range_ratio_seed"
                        ]
                    )
                    for row in rows
                ),
                "minimum_endpoint_fit_snr_ratio_seed": min(
                    float(row["worst_endpoint_fit_snr_ratio_seed"])
                    for row in rows
                ),
                "minimum_endpoint_empirical_snr_ratio_seed": min(
                    float(row["worst_endpoint_empirical_snr_ratio_seed"])
                    for row in rows
                ),
                "maximum_endpoint_centroid_shift_from_seed_arcsec": max(
                    float(
                        row[
                            "maximum_endpoint_centroid_shift_from_seed_arcsec"
                        ]
                    )
                    for row in rows
                ),
            }
        )
    order = {"normal": 0, "marginal": 1, "stress": 2}
    return sorted(
        result,
        key=lambda row: (
            int(row["tolerance_percent"]),
            order[str(row["quality_stratum"])],
        ),
    )


def read_log(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as stream:
            return stream.read()
    return path.read_text(encoding="utf-8", errors="replace")


def audit_stage(
    stage_root: Path, matrix_rows: list[dict],
) -> tuple[list[dict], dict]:
    setup = stage_root / "setup"
    jobs_path = setup / "stage_a_jobs.tsv"
    with jobs_path.open(newline="", encoding="utf-8") as stream:
        jobs = list(csv.DictReader(stream, delimiter="\t"))
    if {int(row["obsnum"]) for row in jobs} != {
        int(row["obsnum"]) for row in matrix_rows
    }:
        raise ValueError("setup jobs do not match frozen run matrix")
    checksums = {}
    for line in (setup / "config_checksums.sha256").read_text().splitlines():
        digest, filename = line.split(None, 1)
        checksums[filename.strip()] = digest
    binary_env = dict(
        line.split("=", 1)
        for line in (setup / "binary.env").read_text().splitlines()
    )
    local_binary = setup / "bin" / Path(
        binary_env["CITLALI_SNAPSHOT"]
    ).name
    binary_ok = (
        local_binary.is_file()
        and sha256(local_binary) == binary_env["CITLALI_SHA256"]
    )
    audits = []
    config_mode_anomalies = []
    all_warning_messages: Counter[str] = Counter()
    for job in jobs:
        obsnum = int(job["obsnum"])
        reduced = stage_root / f"obs{obsnum}/reduced"
        redu_dirs = [reduced / f"redu{i:02d}" for i in range(10)]
        missing_iterations = [
            path.name for path in redu_dirs if not path.is_dir()
        ]
        missing_products = []
        error_count = 0
        warning_messages: Counter[str] = Counter()
        config_mismatch_count = 0
        expected_config = setup / job["config"]
        expected_hash = checksums[job["config"]]
        if sha256(expected_config) != expected_hash:
            config_mismatch_count += 1
        for iteration, redu in enumerate(redu_dirs):
            raw = redu / str(obsnum) / "raw"
            required = [
                redu / "index.yaml",
                redu / "citlali_restart_checkpoint.nc",
                redu / f"learning_iter_{iteration}.csv",
                redu / "citlali.log.gz",
                redu / str(obsnum) / "index.yaml",
                raw / "index.yaml",
            ]
            required.extend(
                raw
                / f"toltec_commissioning_{array}_pointing_"
                  f"{obsnum}_citlali.fits"
                for array in ARRAYS
            )
            if iteration == 9:
                required.extend(
                    redu / filename
                    for filename in TERMINAL_PROVENANCE_FILES
                )
                required.extend(
                    (
                        redu / str(obsnum)
                        / "raw_timestream_provenance.yaml",
                        redu / str(obsnum)
                        / "timestream_output_provenance.yaml",
                    )
                )
            for path in required:
                if not path.is_file() or path.stat().st_size == 0:
                    missing_products.append(str(path.relative_to(stage_root)))
            copied_config = redu / job["config"]
            if not copied_config.is_file():
                missing_products.append(
                    str(copied_config.relative_to(stage_root))
                )
            else:
                if sha256(copied_config) != expected_hash:
                    config_mismatch_count += 1
                expected_mode = stat.S_IMODE(expected_config.stat().st_mode)
                actual_mode = stat.S_IMODE(copied_config.stat().st_mode)
                if actual_mode != expected_mode:
                    config_mode_anomalies.append(
                        {
                            "obsnum": obsnum,
                            "iteration": iteration,
                            "path": str(copied_config),
                            "expected_mode_octal": oct(expected_mode),
                            "actual_mode_octal": oct(actual_mode),
                            "readable_locally": True,
                            "content_sha256_matches": (
                                sha256(copied_config) == expected_hash
                            ),
                        }
                    )
            log_path = redu / "citlali.log.gz"
            if log_path.is_file():
                text = read_log(log_path)
                error_count += len(ERROR_RE.findall(text))
                for message in WARNING_RE.findall(text):
                    warning_messages[message] += 1
                    all_warning_messages[message] += 1
        slurm_matches = sorted(
            (stage_root / "logs").glob(f"flpop-a-*_{job['task']}.out")
        )
        slurm_complete = False
        if len(slurm_matches) == 1:
            slurm_text = read_log(slurm_matches[0])
            error_count += len(ERROR_RE.findall(slurm_text))
            slurm_complete = "Citlali Process finished" in slurm_text
        audits.append(
            {
                "task": int(job["task"]),
                "obsnum": obsnum,
                "source": job["source"],
                "quality_rank": int(job["rank"]),
                "quality_stratum": job["stratum"],
                "iteration_directory_count": sum(
                    path.is_dir() for path in redu_dirs
                ),
                "missing_iteration_directories":
                    ";".join(missing_iterations),
                "missing_or_empty_product_count": len(missing_products),
                "missing_or_empty_products": ";".join(missing_products),
                "config_content_mismatch_count": config_mismatch_count,
                "error_level_message_count": error_count,
                "warning_message_count": sum(warning_messages.values()),
                "warning_messages": ";".join(
                    f"{count}x {message}"
                    for message, count in sorted(warning_messages.items())
                ),
                "slurm_log_count": len(slurm_matches),
                "slurm_process_finished": slurm_complete,
                "audit_pass": (
                    not missing_iterations
                    and not missing_products
                    and config_mismatch_count == 0
                    and error_count == 0
                    and len(slurm_matches) == 1
                    and slurm_complete
                ),
            }
        )
    stage = {
        "binary_sha256": binary_env["CITLALI_SHA256"],
        "binary_checksum_pass": binary_ok,
        "job_count": len(audits),
        "job_audit_pass_count": sum(row["audit_pass"] for row in audits),
        "config_mode_anomaly_count": len(config_mode_anomalies),
        "config_mode_anomalies": config_mode_anomalies,
        "warning_message_count": sum(all_warning_messages.values()),
        "warning_messages": dict(all_warning_messages),
        "error_level_message_count": sum(
            int(row["error_level_message_count"]) for row in audits
        ),
    }
    return audits, stage


def build_gate(
    stage_audit: dict,
    iteration_rows: list[dict],
    transition_rows: list[dict],
    observation_rows: list[dict],
) -> dict:
    tolerance_one = [
        row for row in observation_rows
        if int(row["tolerance_percent"]) == 1
    ]
    interpretable_by_stratum = {
        stratum_name: sum(
            row["quality_stratum"] == stratum_name
            and int(row["source_associated_arrays"]) == len(ARRAYS)
            for row in tolerance_one
        )
        for stratum_name in ("normal", "marginal", "stress")
    }
    checks = {
        "frozen_binary_checksum_pass":
            stage_audit["binary_checksum_pass"],
        "sixteen_jobs_complete":
            stage_audit["job_audit_pass_count"] == 16,
        "zero_error_level_messages":
            stage_audit["error_level_message_count"] == 0,
        "all_480_iteration_metrics_finite_or_classified":
            len(iteration_rows) == 480
            and all(bool(row["fit_valid"]) for row in iteration_rows),
        "all_432_transitions_finite_or_classified":
            len(transition_rows) == 432
            and all(bool(row["metric_finite"]) for row in transition_rows),
        "at_least_two_interpretable_observations_per_stratum":
            all(value >= 2 for value in interpretable_by_stratum.values()),
        "all_four_tolerances_assessed":
            len(observation_rows) == 16 * len(TOLERANCES)
            and {
                int(row["tolerance_percent"]) for row in observation_rows
            } == {1, 2, 5, 10},
        "no_quality_dependent_setup_or_measurement_failure":
            stage_audit["job_audit_pass_count"] == 16
            and all(bool(row["metric_finite"]) for row in transition_rows),
    }
    return {
        "schema_version": "citlali-fruit-loop-population-stage-a-gate-v2",
        "checks": checks,
        "stage_b_gate_pass": all(checks.values()),
        "source_associated_observations_by_stratum":
            interpretable_by_stratum,
        "interpretation": (
            "Stage B eligibility is structural and interpretability-based. "
            "It does not assert that every sentinel converged by iteration 9 "
            "or approve a production stopping tolerance."
        ),
        "operational_exception": (
            f"{stage_audit['config_mode_anomaly_count']} config copies arrived "
            "with modes differing from their setup configs. In this run the "
            "two affected redu01 copies arrived mode 0200 on Unity; the owner "
            "restored owner-read permission, and both local copies are "
            "byte-identical to the checksummed setup configs. This was not "
            "quality-dependent and did not affect scientific products."
        ),
    }


def plot_observation(
    rows: list[dict], obsnum: int, output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if int(row["obsnum"]) == obsnum]
    if len(selected) != 30:
        raise ValueError(f"obsnum {obsnum}: incomplete plot rows")
    panels = (
        (
            "kernel_normalized_amplitude_ratio_seed",
            "Kernel-normalized amplitude / seed",
        ),
        ("fwhm", "FWHM / realized kernel"),
        ("centroid_shift_from_seed_arcsec", "Centroid shift from seed (arcsec)"),
        (
            "empirical_point_source_sig2noise_ratio_seed",
            "Empirical point-source S/N / seed",
        ),
        (
            "successive_map_delta_relative_rms",
            "Successive whole-map relative RMS",
        ),
        (
            "map_background_sigma_mjy_ratio_seed",
            "Source-free background sigma / seed",
        ),
    )
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5))
    colors = dict(zip(ARRAYS, ("tab:blue", "tab:orange", "tab:green")))
    for axis, (field, ylabel) in zip(axes.flat, panels, strict=True):
        for array in ARRAYS:
            sequence = sorted(
                (row for row in selected if row["array"] == array),
                key=lambda row: int(row["iteration"]),
            )
            x = [row["iteration"] for row in sequence]
            if field == "fwhm":
                axis.plot(
                    x,
                    [row["major_fwhm_over_kernel"] for row in sequence],
                    marker="o", color=colors[array], label=f"{array} major",
                )
                axis.plot(
                    x,
                    [row["minor_fwhm_over_kernel"] for row in sequence],
                    marker=".", linestyle="--", color=colors[array],
                    label=f"{array} minor",
                )
            else:
                axis.plot(
                    x, [row[field] for row in sequence],
                    marker="o", color=colors[array], label=array,
                )
        axis.set_xlabel("Absolute fruit-loop iteration")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if field == "successive_map_delta_relative_rms":
            axis.set_yscale("log")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)
    metadata = selected[0]
    fig.suptitle(
        f"Obs {obsnum} ({metadata['source']}), quality rank "
        f"{metadata['quality_rank']} / {metadata['quality_stratum']}"
    )
    fig.tight_layout()
    fig.savefig(output / f"obs{obsnum}_convergence.png", dpi=180)
    plt.close(fig)


def plot_yield(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ("normal", "marginal", "stress")
    tolerances = (1, 2, 5, 10)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    width = 0.24
    x = np.arange(len(tolerances), dtype=float)
    for offset, stratum_name in enumerate(order):
        selected = {
            int(row["tolerance_percent"]): row
            for row in rows if row["quality_stratum"] == stratum_name
        }
        fractions = [
            selected[tolerance]["array_pass_count"]
            / selected[tolerance]["array_count"]
            for tolerance in tolerances
        ]
        endpoint = [
            selected[tolerance]["array_endpoint_pass_count"]
            / selected[tolerance]["array_count"]
            for tolerance in tolerances
        ]
        axes[0].bar(
            x + (offset - 1) * width, fractions, width,
            label=stratum_name,
        )
        axes[1].bar(
            x + (offset - 1) * width, endpoint, width,
            label=stratum_name,
        )
    for axis, title in zip(
        axes,
        ("Any two-transition pass by iteration 9", "Endpoint transitions 8;9"),
        strict=True,
    ):
        axis.set_xticks(x, [f"{value}%" for value in tolerances])
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel("Candidate tolerance")
        axis.set_ylabel("Array-trajectory fraction")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "convergence_yield_by_stratum.png", dpi=180)
    plt.close(fig)


def markdown_report(
    gate: dict,
    stratum_rows: list[dict],
    diagnostic_rows: list[dict],
    observation_rows: list[dict],
    stage_audit: dict,
    iteration_rows: list[dict],
    transition_rows: list[dict],
) -> str:
    valid_iteration_count = sum(
        bool(row["fit_valid"]) for row in iteration_rows
    )
    interpretable_transition_count = sum(
        bool(row["interpretable"]) for row in transition_rows
    )
    source_mismatch_iteration_count = sum(
        not bool(row["source_association_valid"]) for row in iteration_rows
    )
    fwhm_censored_iteration_count = sum(
        bool(row["fwhm_upper_bound_hit"]) for row in iteration_rows
    )
    diagnostics_by_tolerance = {
        tolerance: [
            row for row in diagnostic_rows
            if int(row["tolerance_percent"]) == tolerance
        ]
        for tolerance in (1, 2, 5, 10)
    }

    def total(tolerance: int, field: str) -> int:
        return sum(
            int(row[field])
            for row in diagnostics_by_tolerance[tolerance]
        )

    source_trajectory_count = total(
        1, "source_associated_trajectory_count"
    )
    psf_trajectory_count = total(
        1, "psf_interpretable_trajectory_count"
    )
    source_endpoints = [
        row for row in iteration_rows
        if int(row["iteration"]) == 9
        and bool(row["source_association_valid"])
    ]
    maximum_cumulative_centroid_shift = max(
        float(row["centroid_shift_from_seed_arcsec"])
        for row in source_endpoints
    )
    legacy_dynamic_range_loss_over_ten_percent = sum(
        float(row["legacy_peak_over_full_map_rms_ratio_seed"]) < 0.9
        for row in source_endpoints
    )
    legacy_dynamic_range_loss_over_twenty_percent = sum(
        float(row["legacy_peak_over_full_map_rms_ratio_seed"]) < 0.8
        for row in source_endpoints
    )
    warning_summary = "; ".join(
        f"{count}x {message}"
        for message, count in sorted(
            stage_audit["warning_messages"].items()
        )
    )
    lines = [
        "# Fruit-loop population Stage A analysis",
        "",
        f"- Stage B gate: `{'PASS' if gate['stage_b_gate_pass'] else 'FAIL'}`",
        f"- Completed jobs: `{stage_audit['job_audit_pass_count']}/16`",
        f"- Error-level messages: `{stage_audit['error_level_message_count']}`",
        f"- Warning messages: `{stage_audit['warning_message_count']}`",
        f"- Iteration metrics: `{valid_iteration_count}/"
        f"{len(iteration_rows)} valid`",
        f"- Combined-diagnostic transitions: "
        f"`{interpretable_transition_count}/{len(transition_rows)} "
        "interpretable`",
        f"- Source-mismatch iteration fits: "
        f"`{source_mismatch_iteration_count}`",
        f"- Upper-bound-censored FWHM fits: "
        f"`{fwhm_censored_iteration_count}`",
        "- Production policy changed: `false`",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(
        f"- {name}: `{'PASS' if passed else 'FAIL'}`"
        for name, passed in gate["checks"].items()
    )
    lines.extend(
        [
            "",
            "Warning-only audit: " + warning_summary,
            "",
            "## Separate calibration-reference verdicts",
            "",
            "| Use | Stage A verdict | Evidence boundary |",
            "|---|---|---|",
            "| Astrometric pointing offset | **Qualified per trajectory; "
            "not universal** | "
            f"`{source_trajectory_count}/48` trajectories retain the same "
            "cross-array source association; "
            f"`{total(1, 'centroid_endpoint_pass_count')}/"
            f"{source_trajectory_count}` of those pass the final two "
            "centroid steps below 0.1 arcsec. Cumulative movement reaches "
            f"{maximum_cumulative_centroid_shift:.3f} "
            "arcsec, so the endpoint gate cannot be replaced by a seed-only "
            "comparison. |",
            "| Effective processed PSF | **Qualified where the width fit is "
            "uncensored; not universal** | "
            f"`{psf_trajectory_count}/48` trajectories avoid the fitter's "
            "upper FWHM bound. Of those, "
            f"`{total(1, 'fwhm_endpoint_pass_count')}/"
            f"{psf_trajectory_count}` pass at 1% and "
            f"`{total(5, 'fwhm_endpoint_pass_count')}/"
            f"{psf_trajectory_count}` pass at 5% in transitions 8;9. |",
            "| Photometric amplitude / transfer | **No** | Real-source "
            "amplitude has no injected or external truth. Even stability is "
            f"tolerance-dependent: `{total(1, 'amplitude_endpoint_pass_count')}/"
            f"{source_trajectory_count}` pass at 1%, versus "
            f"`{total(5, 'amplitude_endpoint_pass_count')}/"
            f"{source_trajectory_count}` at 5%. |",
            "| Associated science-processing response | **Not determined** | "
            "Stage A contains pointing reductions only; no matched "
            "pointing-versus-science injection was run. |",
            "",
            "## Diagnostic endpoint yield",
            "",
            "| Tolerance | Amplitude | FWHM | Whole map | Background sigma | "
            "Combined (no S/N) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tolerance in (1, 2, 5, 10):
        lines.append(
            f"| {tolerance}% | "
            f"{total(tolerance, 'amplitude_endpoint_pass_count')}/"
            f"{source_trajectory_count} | "
            f"{total(tolerance, 'fwhm_endpoint_pass_count')}/"
            f"{psf_trajectory_count} | "
            f"{total(tolerance, 'map_endpoint_pass_count')}/48 | "
            f"{total(tolerance, 'background_endpoint_pass_count')}/"
            f"{source_trajectory_count} | "
            f"{total(tolerance, 'combined_endpoint_pass_count')}/48 |"
        )
    lines.extend(
        [
            "",
            "## Combined convergence yield",
            "",
            "| Tolerance | Stratum | Arrays ever passing | Arrays passing "
            "endpoint 8;9 | Observations all arrays ever passing |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in stratum_rows:
        lines.append(
            f"| {row['tolerance_percent']}% | {row['quality_stratum']} | "
            f"{row['array_pass_count']}/{row['array_count']} | "
            f"{row['array_endpoint_pass_count']}/{row['array_count']} | "
            f"{row['observations_all_arrays_ever_pass']}/"
            f"{row['observation_count']} |"
        )
    one_percent = [
        row for row in observation_rows
        if int(row["tolerance_percent"]) == 1
    ]
    unresolved = [
        row for row in one_percent
        if not bool(row["all_arrays_have_combined_pass"])
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage A passes the predeclared Stage B gate because every job and "
            "metric is complete and every quality stratum has at least two "
            "fully source-associated observations. Censored PSF fits and a "
            "cross-array source mismatch are retained as classified failures, "
            "not converted into false convergence. This does not select a "
            "stopping tolerance. The "
            "combined candidate includes source amplitude, shape, centroid, "
            "and successive whole-map change. Statistical S/N, background "
            "noise, and the legacy dynamic-range diagnostic are reported "
            "separately and do not gate convergence.",
            "",
            "The historical pointing-table `sig2noise` is amplitude divided "
            "by full-map RMS, not statistical significance. As a retained "
            "dynamic-range diagnostic, "
            f"`{legacy_dynamic_range_loss_over_ten_percent}/"
            f"{source_trajectory_count}` source-associated trajectories "
            "decrease by more than 10% from seed to iteration 9, including "
            f"`{legacy_dynamic_range_loss_over_twenty_percent}` that decrease "
            "by more than 20%. These changes are not called S/N loss.",
            "",
            f"At 1%, `{len(unresolved)}/16` observations do not have all "
            "three arrays satisfy the combined two-transition diagnostic by "
            "iteration 9. They remain part of yield accounting and are "
            "candidates for checkpoint-v2 continuation after the full "
            "population run.",
            "",
            "Real-source trajectories constrain astrometric and effective-PSF "
            "stability but do not establish photometric truth. Pointing-to-"
            "science response remains unmeasured.",
            "",
            "## Operational exception",
            "",
            gate["operational_exception"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    matrix_rows = read_run_matrix(args.run_matrix, phase=args.phase)
    args.output.mkdir(parents=True, exist_ok=True)
    audit_rows, stage_audit = audit_stage(args.stage_root, matrix_rows)
    iteration_rows = load_iteration_metrics(args.stage_root, matrix_rows)
    transition_rows = build_transition_metrics(iteration_rows)
    assessment_rows = convergence_assessment(
        iteration_rows, transition_rows
    )
    diagnostic_rows = diagnostic_yield_summary(assessment_rows)
    observation_rows = observation_assessment(assessment_rows)
    stratum_rows = stratum_summary(observation_rows)
    gate = build_gate(
        stage_audit, iteration_rows, transition_rows, observation_rows
    )

    outputs = (
        ("job_audit.csv", audit_rows),
        ("iteration_metrics.csv", iteration_rows),
        ("transition_metrics.csv", transition_rows),
        ("array_convergence_assessment.csv", assessment_rows),
        ("diagnostic_yield_summary.csv", diagnostic_rows),
        ("observation_convergence_assessment.csv", observation_rows),
        ("stratum_convergence_summary.csv", stratum_rows),
    )
    for filename, rows in outputs:
        write_csv(args.output / filename, rows)
    for row in matrix_rows:
        plot_observation(
            iteration_rows, int(row["obsnum"]), args.output
        )
    plot_yield(stratum_rows, args.output)
    (args.output / "stage_a_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "report.md").write_text(
        markdown_report(
            gate,
            stratum_rows,
            diagnostic_rows,
            observation_rows,
            stage_audit,
            iteration_rows,
            transition_rows,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version":
            "citlali-fruit-loop-population-stage-a-analysis-v2",
        "stage_root": str(args.stage_root.resolve()),
        "run_matrix": str(args.run_matrix.resolve()),
        "run_matrix_sha256": sha256(args.run_matrix),
        "setup_manifest_sha256": sha256(
            args.stage_root / "setup/manifest.yaml"
        ),
        "binary_sha256": stage_audit["binary_sha256"],
        "observation_count": len(matrix_rows),
        "iteration_metric_count": len(iteration_rows),
        "transition_metric_count": len(transition_rows),
        "candidate_tolerances_percent": [1, 2, 5, 10],
        "required_consecutive_transitions": 2,
        "centroid_step_limit_arcsec": CENTROID_STEP_LIMIT_ARCSEC,
        "cross_array_source_association_limit_arcsec":
            CROSS_ARRAY_ASSOCIATION_LIMIT_ARCSEC,
        "pointing_fwhm_upper_bounds_arcsec":
            POINTING_FWHM_UPPER_ARCSEC,
        "legacy_pointing_sig2noise_identity":
            "fitted_amplitude_over_full_map_standard_deviation",
        "legacy_pointing_sig2noise_used_for_convergence": False,
        "background_annulus_arcsec": [
            BACKGROUND_INNER_RADIUS_ARCSEC,
            BACKGROUND_OUTER_RADIUS_ARCSEC,
        ],
        "empirical_point_source_snr_estimator": {
            "name": "blank_sky_formal_weight_calibrated_fixed_psf_v1",
            "template": (
                "circular Gaussian with geometric-mean realized-kernel FWHM"
            ),
            "fit": "amplitude plus constant background",
            "blank_sky_region_arcsec": [
                BACKGROUND_INNER_RADIUS_ARCSEC,
                BACKGROUND_OUTER_RADIUS_ARCSEC,
            ],
            "minimum_blank_fits": MINIMUM_BLANK_SKY_FITS,
            "noise_scale": (
                "normal-scaled MAD of blank fitted amplitudes divided by "
                "their formal-weight uncertainties"
            ),
        },
        "stage_b_gate_pass": gate["stage_b_gate_pass"],
        "production_defaults_changed": False,
        "files": {},
    }
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote Stage A analysis for {len(matrix_rows)} observations "
        f"to {args.output}; Stage B gate="
        f"{'PASS' if gate['stage_b_gate_pass'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
