#!/usr/bin/env python3
"""Analyze the registered SCI-FRUIT EL-F11 persistence replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from netCDF4 import Dataset
from scipy.stats import rankdata

from tools.fruit_loops.analyze_jinc_accounting import (
    RECEIPT_PLANES,
    bitwise_equal,
    distribution,
    finalize_jinc,
    receipt_plane,
    region_masks,
    scalar,
    sha256,
    validate_registered_files,
)
from tools.fruit_loops.analyze_off_source_penalty_counterfactual import (
    require_exact_maps,
)
from tools.fruit_loops.analyze_penalty_placement import (
    require_compatible_checkpoint,
)
from tools.fruit_loops.analyze_shared_start_response import world_axes
from tools.fruit_loops.compare_injected_source_pair import (
    ARRAYS,
    product_path,
    rms,
)
from tools.fruit_loops.edit_restart_checkpoint_penalty import values_equal


MECHANISMS = (
    "signed_leverage",
    "signal_contrast",
    "absolute_coefficient_mass_share",
    "quadratic_support_share",
    "total_signed_cancellation",
    "target_signed_cancellation",
    "total_unique_detector_count",
)
ALLOWED_LEDGER_REASONS = {
    "admitted",
    "final_flagged",
    "nonfinite_signal",
    "analysis_coefficient_unavailable",
    "center_outside_map",
}
WCS_KEYS = (
    "NAXIS1",
    "NAXIS2",
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CDELT1",
    "CDELT2",
    "BUNIT",
)


def image(path: Path, extension: str) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        values = np.asarray(hdul[extension].data, dtype=np.float64).squeeze()
    if values.ndim != 2:
        raise ValueError(f"expected 2-D image: {path}:{extension}")
    return values


def require_equal_netcdf(expected_path: Path, actual_path: Path) -> dict:
    """Compare NetCDF meaning while ignoring container byte layout."""
    with Dataset(expected_path) as expected, Dataset(actual_path) as actual:
        if expected.ncattrs() != actual.ncattrs():
            raise ValueError("NetCDF global attribute names differ")
        for name in expected.ncattrs():
            if not values_equal(
                expected.getncattr(name), actual.getncattr(name)
            ):
                raise ValueError(f"NetCDF global attribute differs: {name}")
        if set(expected.dimensions) != set(actual.dimensions):
            raise ValueError("NetCDF dimensions differ")
        for name, dimension in expected.dimensions.items():
            other = actual.dimensions[name]
            if len(dimension) != len(other) or dimension.isunlimited() != other.isunlimited():
                raise ValueError(f"NetCDF dimension differs: {name}")
        if set(expected.variables) != set(actual.variables):
            raise ValueError("NetCDF variables differ")
        for name, expected_variable in expected.variables.items():
            actual_variable = actual.variables[name]
            if (
                expected_variable.dimensions != actual_variable.dimensions
                or expected_variable.dtype != actual_variable.dtype
                or expected_variable.ncattrs() != actual_variable.ncattrs()
            ):
                raise ValueError(f"NetCDF variable structure differs: {name}")
            for attribute in expected_variable.ncattrs():
                if not values_equal(
                    expected_variable.getncattr(attribute),
                    actual_variable.getncattr(attribute),
                ):
                    raise ValueError(
                        f"NetCDF variable attribute differs: {name}:{attribute}"
                    )
            if not values_equal(expected_variable[...], actual_variable[...]):
                raise ValueError(f"NetCDF variable value differs: {name}")
    return {
        "structure_attributes_and_values_identical": True,
        "whole_file_hash_identical": sha256(expected_path) == sha256(actual_path),
    }


def fits_grid_identity(left_path: Path, right_path: Path) -> dict:
    with fits.open(left_path, memmap=True) as left, fits.open(
        right_path, memmap=True
    ) as right:
        left_header = left["signal_I"].header
        right_header = right["signal_I"].header
        observed = {}
        for key in WCS_KEYS:
            left_value = left_header.get(key)
            right_value = right_header.get(key)
            if left_value != right_value:
                raise ValueError(
                    f"iteration-4/5 FITS identity differs: {key}: "
                    f"{left_value!r} != {right_value!r}"
                )
            observed[key] = left_value
    left_x, left_y = world_axes(left_path, "signal_I")
    right_x, right_y = world_axes(right_path, "signal_I")
    if not np.array_equal(left_x, right_x) or not np.array_equal(left_y, right_y):
        raise ValueError("iteration-4/5 world axes differ")
    return {
        "fits_header_identity": observed,
        "world_axes_bitwise_identical": True,
    }


def read_receipt(path: Path, registration: dict) -> tuple[dict, dict[str, np.ndarray]]:
    with Dataset(path) as receipt:
        if scalar(receipt, "schema_identity") != registration["receipt_schema"]:
            raise ValueError("unexpected accounting receipt schema")
        if int(scalar(receipt, "diagnostic_only")) != 1:
            raise ValueError("accounting receipt is not marked diagnostic")
        expected_scalars = {
            "fruit_iteration": registration["execution"]["completed_iteration"],
            "array_name": registration["target"]["array"],
            "array_id": registration["target"]["array_id"],
            "target_uid": registration["target"]["uid"],
            "target_scan_index": registration["target"]["scan_index_zero_based"],
        }
        observed_scalars = {}
        for name, expected in expected_scalars.items():
            observed = scalar(receipt, name)
            if observed != expected:
                raise ValueError(
                    f"unexpected receipt identity {name}: {observed!r}"
                )
            observed_scalars[name] = observed
        for name in RECEIPT_PLANES:
            if name not in receipt.variables:
                raise ValueError(f"required receipt plane absent: {name}")
        planes = {name: receipt_plane(receipt, name) for name in RECEIPT_PLANES}
        observed_scalars.update(
            coverage_cut=float(scalar(receipt, "coverage_cut")),
            normalization_threshold=float(
                scalar(receipt, "normalization_threshold")
            ),
            empirical_coefficient_scale=float(
                scalar(receipt, "empirical_coefficient_scale")
            ),
            signal_unit=str(scalar(receipt, "signal_unit")),
        )
    return observed_scalars, planes


def check_ledger(path: Path, registration: dict) -> dict:
    samples = Table.read(path, format="ascii.ecsv")
    required = {
        "scan_index",
        "sample_index",
        "array_id",
        "uid",
        "admitted",
        "reason",
        "contributed_pixel_count",
    }
    if not required.issubset(samples.colnames):
        raise ValueError("target ledger lacks required columns")
    sample_indices = np.asarray(samples["sample_index"], dtype=np.int64)
    if np.unique(sample_indices).size != len(samples):
        raise ValueError("target ledger sample indices are not unique")
    target = registration["target"]
    for column, expected in (
        ("scan_index", target["scan_index_zero_based"]),
        ("array_id", target["array_id"]),
        ("uid", target["uid"]),
    ):
        if not np.all(np.asarray(samples[column], dtype=np.int64) == int(expected)):
            raise ValueError(f"target ledger identity differs: {column}")
    reasons = np.asarray(samples["reason"], dtype=str)
    unknown = sorted(set(reasons) - ALLOWED_LEDGER_REASONS)
    if unknown:
        raise ValueError(f"target ledger has unknown reasons: {unknown}")
    admitted = np.asarray(samples["admitted"], dtype=np.int64).astype(bool)
    if not np.all(reasons[admitted] == "admitted"):
        raise ValueError("admitted ledger rows have inconsistent reasons")
    if np.any(reasons[~admitted] == "admitted"):
        raise ValueError("unavailable ledger rows are labelled admitted")
    contributed = np.asarray(samples["contributed_pixel_count"], dtype=np.int64)
    if np.any(contributed[admitted] <= 0):
        raise ValueError("admitted target samples lack JINC contributions")
    unique_reasons, counts = np.unique(reasons, return_counts=True)
    return {
        "rows": int(len(samples)),
        "unique_sample_indices": int(np.unique(sample_indices).size),
        "admitted": int(np.count_nonzero(admitted)),
        "unavailable": int(np.count_nonzero(~admitted)),
        "reason_counts": dict(
            zip(unique_reasons.tolist(), counts.astype(int).tolist())
        ),
        "noise_only_pass_rows": 0,
    }


def derive_iteration_maps(
    planes: dict[str, np.ndarray], coverage_cut: float
) -> tuple[dict[str, np.ndarray], dict]:
    total = finalize_jinc(
        planes["total_N"], planes["total_C"], planes["total_Q"], coverage_cut
    )
    n_minus = planes["total_N"] - planes["target_N"]
    c_minus = planes["total_C"] - planes["target_C"]
    q_minus = planes["total_Q"] - planes["target_Q"]
    without = finalize_jinc(n_minus, c_minus, q_minus, coverage_cut)
    total_signal = np.divide(
        planes["total_N"],
        planes["total_C"],
        out=np.full_like(planes["total_N"], np.nan),
        where=np.abs(planes["total_C"]) > 1e-8,
    )
    target_signal = np.divide(
        planes["target_N"],
        planes["target_C"],
        out=np.full_like(planes["target_N"], np.nan),
        where=np.abs(planes["target_C"]) > 1e-8,
    )
    without_signal = np.divide(
        n_minus,
        c_minus,
        out=np.full_like(n_minus, np.nan),
        where=np.abs(c_minus) > 1e-8,
    )
    signed_leverage = np.divide(
        planes["target_C"],
        planes["total_C"],
        out=np.full_like(planes["target_C"], np.nan),
        where=np.abs(planes["total_C"]) > 1e-8,
    )
    signal_contrast = without_signal - target_signal
    deletion_response = without_signal - total_signal
    predicted = signed_leverage * signal_contrast
    conditioned = (
        np.isfinite(total_signal)
        & np.isfinite(target_signal)
        & np.isfinite(without_signal)
        & (np.abs(planes["total_C"]) > 1e-8)
        & (np.abs(planes["target_C"]) > 1e-8)
        & (np.abs(c_minus) > 1e-8)
    )
    maps = {
        "total_signal": total_signal,
        "target_signal": target_signal,
        "without_target_signal": without_signal,
        "signed_leverage": signed_leverage,
        "signal_contrast": signal_contrast,
        "deletion_response": deletion_response,
        "predicted_deletion": predicted,
        "deletion_identity_residual": deletion_response - predicted,
        "absolute_coefficient_mass_share": np.divide(
            planes["target_abs_C_terms"],
            planes["total_abs_C_terms"],
            out=np.full_like(planes["target_abs_C_terms"], np.nan),
            where=planes["total_abs_C_terms"] > 0.0,
        ),
        "quadratic_support_share": np.divide(
            planes["target_Q"],
            planes["total_Q"],
            out=np.full_like(planes["target_Q"], np.nan),
            where=planes["total_Q"] > 0.0,
        ),
        "total_signed_cancellation": np.divide(
            np.abs(planes["total_C"]),
            planes["total_abs_C_terms"],
            out=np.full_like(planes["total_C"], np.nan),
            where=planes["total_abs_C_terms"] > 0.0,
        ),
        "target_signed_cancellation": np.divide(
            np.abs(planes["target_C"]),
            planes["target_abs_C_terms"],
            out=np.full_like(planes["target_C"], np.nan),
            where=planes["target_abs_C_terms"] > 0.0,
        ),
        "total_unique_detector_count": planes[
            "total_unique_detector_count"
        ].astype(np.float64),
        "conditioned": conditioned,
    }
    return maps, {"total": total, "without": without}


def require_total_closure(
    finalized: dict, planes: dict[str, np.ndarray], receipt_scalars: dict,
    fits_path: Path,
) -> dict:
    signal = image(fits_path, "signal_I")
    formal = image(fits_path, "weight_formal_I")
    empirical = image(fits_path, "weight_I")
    checks = {
        "signal_from_total_accumulators": bitwise_equal(
            finalized["total"]["signal"], signal
        ),
        "formal_from_total_accumulators": bitwise_equal(
            finalized["total"]["coefficient"], formal
        ),
        "captured_formal_matches_fits": bitwise_equal(
            planes["formal_coefficient"], formal
        ),
        "captured_empirical_matches_fits": bitwise_equal(
            planes["empirical_coefficient"], empirical
        ),
        "normalization_support_matches": np.array_equal(
            finalized["total"]["support"],
            planes["normalization_support"].astype(bool),
        ),
        "normalization_threshold_matches": (
            finalized["total"]["threshold"]
            == receipt_scalars["normalization_threshold"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"iteration-4 total accounting closure failed: {checks}")
    return checks


def require_identity_closure(
    maps: dict[str, np.ndarray], registration: dict
) -> dict:
    selected = maps["conditioned"]
    scale = np.maximum.reduce(
        [
            np.ones_like(maps["deletion_response"]),
            np.abs(maps["deletion_response"]),
            np.abs(maps["predicted_deletion"]),
            np.abs(maps["total_signal"]),
            np.abs(maps["target_signal"]),
            np.abs(maps["without_target_signal"]),
        ]
    )
    bound = (
        float(registration["forward_error"]["identity_safety_factor"])
        * float(registration["forward_error"]["unit_roundoff"])
        * scale
    )
    residual = np.abs(maps["deletion_identity_residual"])
    violations = selected & (residual > bound)
    if np.any(violations):
        raise ValueError("iteration-4 deletion identity exceeds frozen bound")
    return {
        "conditioned_pixels": int(np.count_nonzero(selected)),
        "max_abs_residual_mjy_beam": float(np.max(residual[selected])),
        "max_bound_mjy_beam": float(np.max(bound[selected])),
        "violating_pixels": int(np.count_nonzero(violations)),
    }


def load_component_maps(path: Path) -> dict[str, np.ndarray]:
    with Dataset(path) as dataset:
        if int(dataset.getncattr("diagnostic_only")) != 1:
            raise ValueError("EL-F10 component maps are not diagnostic")
        result = {
            name: np.asarray(dataset.variables[name][...]).squeeze()
            for name in (
                "deletion_response",
                "signed_leverage",
                "signal_contrast",
                "absolute_coefficient_mass_share",
                "quadratic_support_share",
                "total_signed_cancellation",
                "target_signed_cancellation",
                "total_unique_detector_count",
                "conditioned",
            )
        }
    result["conditioned"] = result["conditioned"].astype(bool)
    return result


def finite_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return finite_number(np.corrcoef(left, right)[0, 1])


def persistence_summary(
    d4: np.ndarray, d5: np.ndarray, selected: np.ndarray
) -> dict:
    left = d4[selected]
    right = d5[selected]
    if left.size == 0:
        return {"count": 0}
    dot = float(np.dot(left, right))
    left_energy = float(np.dot(left, left))
    right_energy = float(np.dot(right, right))
    left_norm = math.sqrt(left_energy)
    right_norm = math.sqrt(right_energy)
    beta = dot / left_energy if left_energy > 0.0 else math.nan
    residual = right - beta * left if math.isfinite(beta) else right * math.nan
    nonzero = (left != 0.0) & (right != 0.0)
    difference = right - left
    return {
        "count": int(left.size),
        "d4_rms_mjy_beam": rms(left),
        "d5_rms_mjy_beam": rms(right),
        "difference_rms_mjy_beam": rms(difference),
        "d4_max_abs_mjy_beam": float(np.max(np.abs(left))),
        "d5_max_abs_mjy_beam": float(np.max(np.abs(right))),
        "difference_max_abs_mjy_beam": float(np.max(np.abs(difference))),
        "d4_signed_sum_mjy_beam": float(np.sum(left)),
        "d5_signed_sum_mjy_beam": float(np.sum(right)),
        "d4_squared_norm": left_energy,
        "d5_squared_norm": right_energy,
        "twice_cross_term": 2.0 * dot,
        "difference_squared_norm": float(np.dot(difference, difference)),
        "energy_identity_residual": float(
            np.dot(difference, difference)
            - (left_energy + right_energy - 2.0 * dot)
        ),
        "normalized_inner_product": finite_number(
            dot / (left_norm * right_norm)
            if left_norm > 0.0 and right_norm > 0.0
            else math.nan
        ),
        "pearson_correlation": correlation(left, right),
        "spearman_rank_correlation": correlation(
            rankdata(left, method="average"),
            rankdata(right, method="average"),
        ),
        "beta_d5_on_d4": finite_number(beta),
        "scaled_residual_fraction": finite_number(
            np.linalg.norm(residual) / right_norm
            if right_norm > 0.0
            else math.nan
        ),
        "both_nonzero_count": int(np.count_nonzero(nonzero)),
        "sign_agreement_fraction": finite_number(
            np.mean(np.sign(left[nonzero]) == np.sign(right[nonzero]))
            if np.any(nonzero)
            else math.nan
        ),
    }


def stable_top_indices(values: np.ndarray, selected: np.ndarray, count: int) -> np.ndarray:
    indices = np.flatnonzero(selected)
    order = np.argsort(-np.abs(values.ravel()[indices]), kind="stable")
    return indices[order[:count]]


def top_response_summary(
    d4: np.ndarray, d5: np.ndarray, selected: np.ndarray,
    fractions: list[float],
) -> list[dict]:
    population = int(np.count_nonzero(selected))
    total_d5_energy = float(np.sum(np.square(d5[selected])))
    rows = []
    for fraction in fractions:
        count = max(1, math.ceil(fraction * population))
        top4 = stable_top_indices(d4, selected, count)
        top5 = stable_top_indices(d5, selected, count)
        intersection = np.intersect1d(top4, top5, assume_unique=True)
        union = np.union1d(top4, top5)
        captured = float(np.sum(np.square(d5.ravel()[top4])))
        rows.append(
            {
                "fraction": fraction,
                "population": population,
                "selected_count": count,
                "intersection_count": int(intersection.size),
                "overlap_fraction_each_set": intersection.size / count,
                "jaccard_fraction": intersection.size / union.size,
                "d4_abs_threshold_mjy_beam": float(
                    np.min(np.abs(d4.ravel()[top4]))
                ),
                "d5_abs_threshold_mjy_beam": float(
                    np.min(np.abs(d5.ravel()[top5]))
                ),
                "d5_squared_response_captured_by_d4_selection": (
                    captured / total_d5_energy if total_d5_energy > 0.0 else None
                ),
            }
        )
    return rows


def write_maps(
    path: Path, d4: dict[str, np.ndarray], d5: dict[str, np.ndarray],
    supports: dict[str, np.ndarray],
) -> None:
    with Dataset(path, "w", format="NETCDF4") as dataset:
        shape = d4["deletion_response"].shape
        dataset.createDimension("map_row", shape[0])
        dataset.createDimension("map_col", shape[1])
        dataset.setncattr("diagnostic_only", 1)
        dataset.setncattr("not_science_product", 1)
        dataset.setncattr("oracle_targeted", 1)
        planes: dict[str, np.ndarray] = {
            "D4_deletion_response": d4["deletion_response"],
            "D5_deletion_response": d5["deletion_response"],
            "D5_minus_D4": d5["deletion_response"] - d4["deletion_response"],
            **{f"D4_{name}": d4[name] for name in MECHANISMS},
            **{f"D5_{name}": d5[name] for name in MECHANISMS},
            **supports,
        }
        for name, values in planes.items():
            dtype = "i1" if values.dtype == bool else "f8"
            variable = dataset.createVariable(
                name, dtype, ("map_row", "map_col"), zlib=True, complevel=1
            )
            variable[:] = values


def write_plot(
    path: Path, d4: np.ndarray, d5: np.ndarray, support: np.ndarray,
    fits_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    x, y = world_axes(fits_path, "signal_I")
    maps = (d4, d5, d5 - d4)
    titles = ("Iteration 4 $D_4$", "Iteration 5 $D_5$", "$D_5-D_4$")
    finite = support & np.isfinite(d4) & np.isfinite(d5)
    limit = float(
        np.percentile(
            np.concatenate([np.abs(d4[finite]), np.abs(d5[finite])]), 99
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for axis, values, title in zip(axes, maps, titles):
        shown = np.where(support, values, np.nan)
        artist = axis.imshow(
            shown,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("AZOFFSET (arcsec)")
    axes[0].set_ylabel("ELOFFSET (arcsec)")
    figure.colorbar(artist, ax=axes, label="Deletion response (mJy/beam)")
    figure.suptitle("EL-F11 oracle-targeted UID 4460 scan-5 persistence")
    figure.subplots_adjust(left=0.06, right=0.92, bottom=0.12, top=0.86, wspace=0.08)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def count_log_errors(path: Path) -> dict:
    text = path.read_text(errors="replace")
    error_lines = [
        line for line in text.splitlines()
        if re.search(r"\]\s+\[(?:error|critical)\]", line, re.I)
    ]
    if error_lines:
        raise ValueError(f"replay log contains error/critical records: {error_lines[:3]}")
    return {"error_or_critical_records": 0}


def analyze(registration: dict, output_dir: Path) -> dict:
    checked = validate_registered_files(registration)
    paths = {name: Path(value) for name, value in registration["paths"].items()}
    obsnum = int(registration["target"]["obsnum"])
    reference_redu = paths["iteration_4_reference"]
    replay_redu = paths["diagnostic_iteration_4"]

    science_planes = require_exact_maps(reference_redu, replay_redu, obsnum)
    formal_planes = 0
    for array_name in ARRAYS:
        expected = image(product_path(reference_redu, obsnum, array_name), "weight_formal_I")
        actual = image(product_path(replay_redu, obsnum, array_name), "weight_formal_I")
        if not bitwise_equal(expected, actual):
            raise ValueError(f"formal coefficient changed: {array_name}")
        formal_planes += 1
    if paths["iteration_4_reference_learning"].read_bytes() != paths[
        "diagnostic_iteration_4_learning"
    ].read_bytes():
        raise ValueError("iteration-4 learning output is not byte-identical")
    mapdiag = require_equal_netcdf(
        paths["iteration_4_reference_mapdiag"],
        paths["diagnostic_iteration_4_mapdiag"],
    )
    checkpoint = require_compatible_checkpoint(
        paths["iteration_4_reference_checkpoint"],
        paths["diagnostic_iteration_4_checkpoint"],
        set(registration["neutrality"]["checkpoint_allowed_differences"]),
    )
    if checkpoint["observed_allowed_differences"] != sorted(
        registration["neutrality"]["checkpoint_required_observed_differences"]
    ):
        raise ValueError(
            "checkpoint observed difference set differs from registration"
        )
    log_result = count_log_errors(paths["replay_log"])

    receipt_scalars, planes = read_receipt(paths["accounting_receipt"], registration)
    d4, finalized = derive_iteration_maps(planes, receipt_scalars["coverage_cut"])
    total_closure = require_total_closure(
        finalized,
        planes,
        receipt_scalars,
        product_path(replay_redu, obsnum, "a1400"),
    )
    ledger = check_ledger(paths["target_samples"], registration)
    identity = require_identity_closure(d4, registration)

    d5 = load_component_maps(paths["d5_component_maps"])
    if d4["deletion_response"].shape != d5["deletion_response"].shape:
        raise ValueError("iteration-4/5 component map shapes differ")
    grid = fits_grid_identity(
        product_path(replay_redu, obsnum, "a1400"), paths["d5_science_fits"]
    )
    common = d4["conditioned"] & d5["conditioned"]
    for name in ("deletion_response", "signed_leverage", "signal_contrast"):
        common &= np.isfinite(d4[name]) & np.isfinite(d5[name])
    if not np.any(common):
        raise ValueError("iteration-4/5 common conditioned target support is empty")
    supports = {
        "D4_conditioned": d4["conditioned"],
        "D5_conditioned": d5["conditioned"],
        "common_conditioned": common,
        "conditioned_union": d4["conditioned"] | d5["conditioned"],
    }
    support_counts = {
        "iteration_4": int(np.count_nonzero(d4["conditioned"])),
        "iteration_5": int(np.count_nonzero(d5["conditioned"])),
        "intersection": int(np.count_nonzero(common)),
        "union": int(np.count_nonzero(supports["conditioned_union"])),
        "iteration_4_only": int(np.count_nonzero(d4["conditioned"] & ~d5["conditioned"])),
        "iteration_5_only": int(np.count_nonzero(d5["conditioned"] & ~d4["conditioned"])),
    }

    spatial = region_masks(product_path(replay_redu, obsnum, "a1400"), registration)
    regions = {
        "complete_common_conditioned": common,
        "neptune_r20": common & spatial["neptune_r20"],
        "injected_source_r20": common & spatial["injected_source_r20"],
        "annulus_r40_120_excluding_neptune_r25": (
            common & spatial["annulus_r40_120_excluding_neptune_r25"]
        ),
    }
    persistence = {
        name: persistence_summary(
            d4["deletion_response"], d5["deletion_response"], selected
        )
        for name, selected in regions.items()
    }
    top_rows = top_response_summary(
        d4["deletion_response"],
        d5["deletion_response"],
        common,
        [float(value) for value in registration["top_response_fractions"]],
    )

    mechanisms = {}
    for region_name, selected in regions.items():
        mechanisms[region_name] = {
            "iteration_4": {
                name: distribution(d4[name], selected) for name in MECHANISMS
            },
            "iteration_5": {
                name: distribution(d5[name], selected) for name in MECHANISMS
            },
        }

    d5_result = json.loads(paths["d5_analysis_result"].read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    write_maps(output_dir / "PERSISTENCE_COMPONENT_MAPS_R0.1.nc", d4, d5, supports)
    write_plot(
        output_dir / "PERSISTENCE_RESPONSE_R0.1.png",
        d4["deletion_response"],
        d5["deletion_response"],
        common,
        product_path(replay_redu, obsnum, "a1400"),
    )
    with (output_dir / "PERSISTENCE_METRICS_R0.1.csv").open(
        "w", newline=""
    ) as stream:
        fields = ["region", *next(iter(persistence.values())).keys()]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for region_name, values in persistence.items():
            writer.writerow({"region": region_name, **values})
    with (output_dir / "TOP_RESPONSE_OVERLAP_R0.1.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(top_rows[0]))
        writer.writeheader()
        writer.writerows(top_rows)

    result = {
        "schema_version": "sci-fruit-el-f11-result-v1",
        "test_id": registration["test_id"],
        "oracle_targeted": True,
        "registered_files_validated": checked,
        "compatibility": {
            "nine_science_planes_bitwise_identical": science_planes == 9,
            "three_formal_planes_bitwise_identical": formal_planes == 3,
            "learning_output_byte_identical": True,
            "mapdiag": mapdiag,
            "checkpoint": checkpoint,
            "grid": grid,
            "log": log_result,
        },
        "receipt_identity": receipt_scalars,
        "total_accumulator_exact_closure": total_closure,
        "target_sample_ledger": ledger,
        "deletion_identity": identity,
        "support_counts": support_counts,
        "persistence_metrics": persistence,
        "top_response_overlap": top_rows,
        "mechanism_distributions": mechanisms,
        "iteration_4_occurrence_distributions": {
            region_name: {
                "total_occurrence_pixel_count": distribution(
                    planes["total_occurrence_pixel_count"], selected
                ),
                "target_occurrence_pixel_count": distribution(
                    planes["target_occurrence_pixel_count"], selected
                ),
            }
            for region_name, selected in regions.items()
        },
        "retained_iteration_5_region_metrics": d5_result["region_metrics"],
        "claim_limits": registration["claim_limits"],
    }
    (output_dir / "ANALYSIS_RESULT_R0.1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(args.registration.read_text())
    result = analyze(registration, args.output_dir)
    print(json.dumps(result["persistence_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
