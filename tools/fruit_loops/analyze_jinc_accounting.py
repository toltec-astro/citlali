#!/usr/bin/env python3
"""Analyze the registered SCI-FRUIT EL-F10 JINC accounting replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from netCDF4 import Dataset

from tools.fruit_loops.analyze_off_source_penalty_counterfactual import (
    require_exact_maps,
)
from tools.fruit_loops.analyze_penalty_placement import (
    require_compatible_checkpoint,
)
from tools.fruit_loops.analyze_shared_start_response import world_axes
from tools.fruit_loops.compare_injected_source_pair import ARRAYS, product_path, rms


REPOSITORY = Path(__file__).resolve().parents[2]
RECEIPT_PLANES = (
    "total_N",
    "total_C",
    "total_Q",
    "target_N",
    "target_C",
    "target_Q",
    "total_abs_N_terms",
    "total_abs_C_terms",
    "target_abs_N_terms",
    "target_abs_C_terms",
    "total_occurrence_pixel_count",
    "target_occurrence_pixel_count",
    "total_unique_detector_count",
    "target_unique_detector_count",
    "formal_coefficient",
    "empirical_coefficient",
    "normalization_support",
    "science_policy_support",
)
REGIONS = (
    "complete_map",
    "injected_source_r20",
    "neptune_r20",
    "annulus_r40_120_excluding_neptune_r25",
    "target_scan_footprint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def validate_registered_files(registration: dict) -> list[dict]:
    checked = []
    for entry in registration["registered_files"]:
        path = resolve_path(entry["path"])
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(entry["size_bytes"]) or digest != entry["sha256"]:
            raise ValueError(f"registered file identity changed: {path}")
        checked.append({"role": entry["role"], "path": str(path),
                        "size_bytes": size, "sha256": digest})
    return checked


def image(path: Path, extension: str) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        values = np.asarray(hdul[extension].data, dtype=np.float64).squeeze()
    if values.ndim != 2:
        raise ValueError(f"expected 2-D image: {path}:{extension}")
    return values


def bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def scalar(dataset: Dataset, name: str) -> object:
    value = np.asarray(dataset.variables[name][...]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "item"):
        return value.item()
    return value


def positive_order_threshold(values: np.ndarray, cut: float) -> tuple[float, int, int]:
    positive = np.sort(values[np.isfinite(values) & (values > 0.0)])
    if positive.size == 0:
        return 0.0, 0, 0
    upper_quartile_floor = math.floor(0.75 * positive.size)
    index = (upper_quartile_floor + positive.size) // 2
    return float(positive[index] * cut), int(positive.size), int(index)


def finalize_jinc(
    numerator: np.ndarray,
    denominator: np.ndarray,
    variance: np.ndarray,
    coverage_cut: float,
) -> dict[str, np.ndarray | float | int]:
    denominator_valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
    variance_valid = np.isfinite(variance) & (variance > 0.0)
    valid = denominator_valid & variance_valid
    safe_denominator = np.where(denominator_valid, denominator, 1.0)
    raw_coefficient = (
        np.square(denominator) / np.maximum(variance, 1e-30) * valid
    )
    threshold, count, index = positive_order_threshold(
        raw_coefficient, coverage_cut / 10.0
    )
    support = (
        np.isfinite(raw_coefficient)
        & (raw_coefficient > 0.0)
        & (raw_coefficient >= threshold)
    )
    signal = np.where(support, numerator / safe_denominator, 0.0)
    coefficient = np.where(support, raw_coefficient, 0.0)
    return {
        "signal": signal,
        "coefficient": coefficient,
        "raw_coefficient": raw_coefficient,
        "support": support,
        "threshold": threshold,
        "positive_count": count,
        "selected_index": index,
    }


def gamma(count: np.ndarray, unit_roundoff: float) -> np.ndarray:
    operations = np.maximum(np.asarray(count, dtype=np.float64), 1.0)
    product = operations * unit_roundoff
    return np.where(product < 1.0, product / (1.0 - product), np.inf)


def accumulator_difference_bound(
    total: np.ndarray,
    target: np.ndarray,
    total_abs: np.ndarray,
    target_abs: np.ndarray,
    total_count: np.ndarray,
    target_count: np.ndarray,
    unit_roundoff: float,
) -> np.ndarray:
    without_count = np.maximum(total_count - target_count, 0)
    without_abs = np.maximum(total_abs - target_abs, 0.0)
    subtraction = unit_roundoff * (np.abs(total) + np.abs(target))
    return (
        gamma(total_count, unit_roundoff) * total_abs
        + gamma(target_count, unit_roundoff) * target_abs
        + gamma(without_count, unit_roundoff) * without_abs
        + subtraction
    )


def reconstruction_bounds(planes: dict[str, np.ndarray], registration: dict) -> dict:
    unit_roundoff = float(registration["forward_error"]["unit_roundoff"])
    safety = float(registration["forward_error"]["finalization_safety_factor"])
    hits = planes["total_occurrence_pixel_count"]
    target_hits = planes["target_occurrence_pixel_count"]
    n_bound = accumulator_difference_bound(
        planes["total_N"], planes["target_N"],
        planes["total_abs_N_terms"], planes["target_abs_N_terms"],
        hits, target_hits, unit_roundoff,
    )
    c_bound = accumulator_difference_bound(
        planes["total_C"], planes["target_C"],
        planes["total_abs_C_terms"], planes["target_abs_C_terms"],
        hits, target_hits, unit_roundoff,
    )
    q_bound = accumulator_difference_bound(
        planes["total_Q"], planes["target_Q"],
        np.abs(planes["total_Q"]), np.abs(planes["target_Q"]),
        hits, target_hits, unit_roundoff,
    )
    n_minus = planes["total_N"] - planes["target_N"]
    c_minus = planes["total_C"] - planes["target_C"]
    q_minus = planes["total_Q"] - planes["target_Q"]
    c_margin = np.abs(c_minus) - c_bound
    q_margin = q_minus - q_bound
    conditioned = (c_margin > 1e-8) & (q_margin > 0.0)
    reconstructed_signal = np.divide(
        n_minus, c_minus, out=np.zeros_like(n_minus), where=c_minus != 0.0
    )
    signal_bound = np.full_like(n_minus, np.inf)
    signal_bound[conditioned] = safety * (
        (
            n_bound[conditioned]
            + np.abs(reconstructed_signal[conditioned]) * c_bound[conditioned]
        )
        / c_margin[conditioned]
        + unit_roundoff * np.maximum(
            1.0, np.abs(reconstructed_signal[conditioned])
        )
    )
    reconstructed_coefficient = np.divide(
        np.square(c_minus), q_minus, out=np.zeros_like(c_minus), where=q_minus > 0.0
    )
    coefficient_bound = np.full_like(c_minus, np.inf)
    coefficient_bound[conditioned] = safety * (
        (
            2.0 * np.abs(c_minus[conditioned]) * c_bound[conditioned]
            + np.square(c_bound[conditioned])
        )
        / q_margin[conditioned]
        + (
            np.square(np.abs(c_minus[conditioned]) + c_bound[conditioned])
            * q_bound[conditioned]
        )
        / (q_minus[conditioned] * q_margin[conditioned])
        + unit_roundoff
        * np.maximum(1.0, np.abs(reconstructed_coefficient[conditioned]))
    )
    return {
        "N": n_bound,
        "C": c_bound,
        "Q": q_bound,
        "signal": signal_bound,
        "coefficient": coefficient_bound,
        "conditioned": conditioned,
    }


def region_masks(path: Path, registration: dict) -> dict[str, np.ndarray]:
    x, y = world_axes(path, "signal_I")
    xx, yy = np.meshgrid(x, y)
    injection = registration["regions"]["injection_world_arcsec"]
    neptune = registration["regions"]["neptune_world_arcsec"]
    injection_radius = np.hypot(xx - injection[0], yy - injection[1])
    neptune_radius = np.hypot(xx - neptune[0], yy - neptune[1])
    return {
        "complete_map": np.ones(xx.shape, dtype=bool),
        "injected_source_r20": injection_radius <= 20.0,
        "neptune_r20": neptune_radius <= 20.0,
        "annulus_r40_120_excluding_neptune_r25": (
            (injection_radius >= 40.0)
            & (injection_radius <= 120.0)
            & (neptune_radius > 25.0)
        ),
    }


def distribution(values: np.ndarray, mask: np.ndarray) -> dict:
    selected = values[mask & np.isfinite(values)]
    if selected.size == 0:
        return {"count": 0}
    return {
        "count": int(selected.size),
        "min": float(np.min(selected)),
        "p05": float(np.percentile(selected, 5)),
        "p25": float(np.percentile(selected, 25)),
        "median": float(np.median(selected)),
        "p75": float(np.percentile(selected, 75)),
        "p95": float(np.percentile(selected, 95)),
        "max": float(np.max(selected)),
        "mean": float(np.mean(selected)),
        "rms": rms(selected),
    }


def binned_response(
    predictor: np.ndarray, response: np.ndarray, support: np.ndarray,
    bins: int,
) -> list[dict]:
    valid = support & np.isfinite(predictor) & np.isfinite(response)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []
    ordered = indices[np.argsort(predictor.ravel()[indices], kind="stable")]
    rows = []
    for bin_index, selected in enumerate(np.array_split(ordered, bins)):
        if selected.size == 0:
            continue
        x = predictor.ravel()[selected]
        y = response.ravel()[selected]
        rows.append({
            "bin": bin_index,
            "count": int(selected.size),
            "predictor_min": float(np.min(x)),
            "predictor_median": float(np.median(x)),
            "predictor_max": float(np.max(x)),
            "response_rms_mjy_beam": rms(y),
            "response_median_mjy_beam": float(np.median(y)),
        })
    return rows


def write_derived_maps(path: Path, maps: dict[str, np.ndarray]) -> None:
    with Dataset(path, "w", format="NETCDF4") as dataset:
        first = next(iter(maps.values()))
        dataset.createDimension("map_row", first.shape[0])
        dataset.createDimension("map_col", first.shape[1])
        dataset.setncattr("diagnostic_only", 1)
        dataset.setncattr("not_science_product", 1)
        for name, values in maps.items():
            dtype = "i1" if values.dtype == bool else "f8"
            var = dataset.createVariable(
                name, dtype, ("map_row", "map_col"), zlib=True,
                complevel=1,
            )
            var[:] = values


def write_plot(path: Path, maps: dict[str, np.ndarray], fits_path: Path) -> None:
    import matplotlib.pyplot as plt

    x, y = world_axes(fits_path, "signal_I")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    entries = (
        ("deletion_response", "without-target minus total", "RdBu_r"),
        ("signed_leverage", "signed leverage $C_t/C$", "RdBu_r"),
        ("signal_contrast", "$m_{-t}-m_t$", "RdBu_r"),
        ("predicted_deletion", "leverage × contrast", "RdBu_r"),
        ("deletion_identity_residual", "deletion identity residual", "RdBu_r"),
        ("total_unique_detector_count", "unique detectors", "viridis"),
    )
    for axis, (name, title, cmap) in zip(axes.ravel(), entries):
        shown = maps[name]
        finite = np.isfinite(shown)
        kwargs = {}
        if cmap == "RdBu_r" and finite.any():
            limit = float(np.percentile(np.abs(shown[finite]), 99))
            if limit > 0.0:
                kwargs = {"vmin": -limit, "vmax": limit}
        image_artist = axis.imshow(
            shown, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
            cmap=cmap, aspect="equal", **kwargs,
        )
        axis.set_title(title)
        figure.colorbar(image_artist, ax=axis, shrink=0.8)
    for axis in axes[-1, :]:
        axis.set_xlabel("AZOFFSET (arcsec)")
    for axis in axes[:, 0]:
        axis.set_ylabel("ELOFFSET (arcsec)")
    figure.suptitle("EL-F10 targeted JINC accounting — a1400 UID 4460 scan 5")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(registration: dict, output_dir: Path) -> dict:
    checked = validate_registered_files(registration)
    run_redu = Path(registration["paths"]["diagnostic_iteration_5"])
    n5_redu = Path(registration["paths"]["n5_reference"])
    a5_redu = Path(registration["paths"]["a5_map_reference"])
    obsnum = int(registration["target"]["obsnum"])
    receipt_path = Path(registration["paths"]["accounting_receipt"])
    samples_path = Path(registration["paths"]["target_samples"])

    neutrality_planes = require_exact_maps(n5_redu, run_redu, obsnum)
    formal_exact = 0
    for array_name in ARRAYS:
        left = image(product_path(n5_redu, obsnum, array_name), "weight_formal_I")
        right = image(product_path(run_redu, obsnum, array_name), "weight_formal_I")
        if not bitwise_equal(left, right):
            raise ValueError(f"formal coefficient changed: {array_name}")
        formal_exact += 1
    checkpoint = require_compatible_checkpoint(
        n5_redu / "citlali_restart_checkpoint.nc",
        run_redu / "citlali_restart_checkpoint.nc",
        set(registration["neutrality"]["checkpoint_allowed_differences"]),
    )

    with Dataset(receipt_path) as receipt:
        if scalar(receipt, "schema_identity") != registration["receipt_schema"]:
            raise ValueError("unexpected accounting receipt schema")
        if int(scalar(receipt, "diagnostic_only")) != 1:
            raise ValueError("accounting receipt is not marked diagnostic")
        for name in RECEIPT_PLANES:
            if name not in receipt.variables:
                raise ValueError(f"required receipt plane is absent: {name}")
        planes = {
            name: np.asarray(receipt.variables[name][...]).squeeze()
            for name in RECEIPT_PLANES
        }
        coverage_cut = float(scalar(receipt, "coverage_cut"))
        normalization_threshold = float(
            scalar(receipt, "normalization_threshold")
        )
        empirical_scale = float(scalar(receipt, "empirical_coefficient_scale"))

    diagnostic_fits = product_path(run_redu, obsnum, "a1400")
    diagnostic_signal = image(diagnostic_fits, "signal_I")
    diagnostic_formal = image(diagnostic_fits, "weight_formal_I")
    diagnostic_empirical = image(diagnostic_fits, "weight_I")
    total_finalized = finalize_jinc(
        planes["total_N"], planes["total_C"], planes["total_Q"], coverage_cut
    )
    exact_checks = {
        "signal_from_total_accumulators": bitwise_equal(
            total_finalized["signal"], diagnostic_signal
        ),
        "formal_from_total_accumulators": bitwise_equal(
            total_finalized["coefficient"], diagnostic_formal
        ),
        "captured_formal_matches_fits": bitwise_equal(
            planes["formal_coefficient"], diagnostic_formal
        ),
        "captured_empirical_matches_fits": bitwise_equal(
            planes["empirical_coefficient"], diagnostic_empirical
        ),
        "normalization_support_matches": np.array_equal(
            total_finalized["support"],
            planes["normalization_support"].astype(bool),
        ),
        "normalization_threshold_matches": (
            total_finalized["threshold"] == normalization_threshold
        ),
    }
    if not all(exact_checks.values()):
        raise ValueError(f"total-accumulator exact closure failed: {exact_checks}")

    samples = Table.read(samples_path, format="ascii.ecsv")
    reasons, reason_counts = np.unique(
        np.asarray(samples["reason"], dtype=str), return_counts=True
    )
    ledger = dict(zip(reasons.tolist(), reason_counts.astype(int).tolist()))
    proposed = len(samples)
    admitted = int(np.count_nonzero(np.asarray(samples["admitted"], dtype=int)))
    unavailable = proposed - admitted
    expected = registration["target_sample_ledger"]
    if (
        proposed != int(expected["proposed"])
        or admitted != int(expected["otherwise_admitted"])
        or unavailable != int(expected["already_unavailable"])
    ):
        raise ValueError(
            f"target sample ledger differs: proposed={proposed}, "
            f"admitted={admitted}, unavailable={unavailable}, reasons={ledger}"
        )

    n_minus = planes["total_N"] - planes["target_N"]
    c_minus = planes["total_C"] - planes["target_C"]
    q_minus = planes["total_Q"] - planes["target_Q"]
    reconstructed = finalize_jinc(n_minus, c_minus, q_minus, coverage_cut)
    a5_fits = product_path(a5_redu, obsnum, "a1400")
    a5_signal = image(a5_fits, "signal_I")
    a5_formal = image(a5_fits, "weight_formal_I")
    bounds = reconstruction_bounds(planes, registration)
    common = reconstructed["support"] & (a5_formal > 0.0) & bounds["conditioned"]
    signal_difference = reconstructed["signal"] - a5_signal
    coefficient_difference = reconstructed["coefficient"] - a5_formal
    signal_within = np.abs(signal_difference) <= bounds["signal"]
    coefficient_within = np.abs(coefficient_difference) <= bounds["coefficient"]
    if not np.all(signal_within[common]):
        raise ValueError("without-target signal exceeds registered forward-error bound")
    if not np.all(coefficient_within[common]):
        raise ValueError("without-target coefficient exceeds registered forward-error bound")

    reconstructed_support = reconstructed["support"]
    a5_support = a5_formal > 0.0
    changed_support = reconstructed_support ^ a5_support
    finite_coefficient_bounds = bounds["coefficient"][
        np.isfinite(bounds["coefficient"])
    ]
    threshold_bound = (
        coverage_cut
        / 10.0
        * float(np.max(finite_coefficient_bounds))
        if finite_coefficient_bounds.size
        else 0.0
    )
    support_edge_explained = np.abs(
        reconstructed["raw_coefficient"] - reconstructed["threshold"]
    ) <= (bounds["coefficient"] + threshold_bound)
    unexplained_support = changed_support & ~support_edge_explained
    if np.any(unexplained_support):
        raise ValueError(
            "without-target support differs outside registered threshold-edge bound"
        )

    conditioned = (
        (np.abs(planes["total_C"]) > 1e-8)
        & (np.abs(planes["target_C"]) > 1e-8)
        & (np.abs(c_minus) > 1e-8)
    )
    total_signal = np.divide(
        planes["total_N"], planes["total_C"],
        out=np.full_like(planes["total_N"], np.nan),
        where=np.abs(planes["total_C"]) > 1e-8,
    )
    target_signal = np.divide(
        planes["target_N"], planes["target_C"],
        out=np.full_like(planes["target_N"], np.nan),
        where=np.abs(planes["target_C"]) > 1e-8,
    )
    without_signal = np.divide(
        n_minus, c_minus, out=np.full_like(n_minus, np.nan),
        where=np.abs(c_minus) > 1e-8,
    )
    signed_leverage = np.divide(
        planes["target_C"], planes["total_C"],
        out=np.full_like(planes["target_C"], np.nan),
        where=np.abs(planes["total_C"]) > 1e-8,
    )
    signal_contrast = without_signal - target_signal
    deletion_response = without_signal - total_signal
    predicted_deletion = signed_leverage * signal_contrast
    identity_residual = deletion_response - predicted_deletion
    abs_mass_share = np.divide(
        planes["target_abs_C_terms"], planes["total_abs_C_terms"],
        out=np.full_like(planes["target_abs_C_terms"], np.nan),
        where=planes["total_abs_C_terms"] > 0.0,
    )
    q_share = np.divide(
        planes["target_Q"], planes["total_Q"],
        out=np.full_like(planes["target_Q"], np.nan),
        where=planes["total_Q"] > 0.0,
    )
    total_cancellation = np.divide(
        np.abs(planes["total_C"]), planes["total_abs_C_terms"],
        out=np.full_like(planes["total_C"], np.nan),
        where=planes["total_abs_C_terms"] > 0.0,
    )
    target_cancellation = np.divide(
        np.abs(planes["target_C"]), planes["target_abs_C_terms"],
        out=np.full_like(planes["target_C"], np.nan),
        where=planes["target_abs_C_terms"] > 0.0,
    )
    derived_maps = {
        "total_signal": total_signal,
        "target_signal": target_signal,
        "without_target_signal": without_signal,
        "signed_leverage": signed_leverage,
        "signal_contrast": signal_contrast,
        "deletion_response": deletion_response,
        "predicted_deletion": predicted_deletion,
        "deletion_identity_residual": identity_residual,
        "absolute_coefficient_mass_share": abs_mass_share,
        "quadratic_support_share": q_share,
        "total_signed_cancellation": total_cancellation,
        "target_signed_cancellation": target_cancellation,
        "coefficient_square_total": np.square(planes["total_C"]),
        "coefficient_cross_term": -2.0 * planes["total_C"] * planes["target_C"],
        "coefficient_square_target": np.square(planes["target_C"]),
        "total_unique_detector_count": planes["total_unique_detector_count"].astype(float),
        "conditioned": conditioned,
    }
    regions = region_masks(diagnostic_fits, registration)
    regions["target_scan_footprint"] = (
        planes["target_occurrence_pixel_count"] > 0
    )
    metric_arrays = {
        "signed_leverage_Ct_over_C": signed_leverage,
        "absolute_mass_share_Bt_over_B": abs_mass_share,
        "quadratic_share_Qt_over_Q": q_share,
        "total_cancellation_abs_C_over_B": total_cancellation,
        "target_cancellation_abs_Ct_over_Bt": target_cancellation,
        "total_occurrence_pixel_count": planes["total_occurrence_pixel_count"],
        "target_occurrence_pixel_count": planes["target_occurrence_pixel_count"],
        "total_unique_detector_count": planes["total_unique_detector_count"],
        "target_signal_mjy_beam": target_signal,
        "without_target_signal_mjy_beam": without_signal,
        "signal_contrast_mjy_beam": signal_contrast,
        "deletion_response_mjy_beam": deletion_response,
        "identity_residual_mjy_beam": identity_residual,
    }
    metrics = {}
    diagnostic_support = planes["normalization_support"].astype(bool)
    for region_name, region in regions.items():
        selected = region & diagnostic_support
        metrics[region_name] = {
            name: distribution(values, selected & conditioned)
            for name, values in metric_arrays.items()
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_derived_maps(output_dir / "JINC_COMPONENT_MAPS_R0.1.nc", derived_maps)
    predictors = {
        "signed_leverage": signed_leverage,
        "absolute_mass_share": abs_mass_share,
        "quadratic_share": q_share,
        "total_cancellation": total_cancellation,
        "unique_detector_count": planes["total_unique_detector_count"],
        "signal_contrast_abs": np.abs(signal_contrast),
    }
    bin_rows = []
    for name, values in predictors.items():
        for row in binned_response(
            values, deletion_response,
            conditioned & diagnostic_support,
            int(registration["descriptive_binning"]["equal_count_bins"]),
        ):
            bin_rows.append({"predictor": name, **row})
    with (output_dir / "BINNED_RESPONSE_R0.1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bin_rows[0]))
        writer.writeheader()
        writer.writerows(bin_rows)

    trigger_rows = []
    for trigger in registration["trigger_pixels"]:
        row = int(trigger["row"])
        col = int(trigger["col"])
        trigger_rows.append({
            "row": row,
            "col": col,
            **{name: float(values[row, col]) for name, values in metric_arrays.items()},
        })
    with (output_dir / "TRIGGER_ACCOUNTING_R0.1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trigger_rows[0]))
        writer.writeheader()
        writer.writerows(trigger_rows)
    write_plot(output_dir / "JINC_ACCOUNTING_R0.1.png", derived_maps, diagnostic_fits)

    result = {
        "schema_version": "sci-fruit-el-f10-result-v1",
        "test_id": registration["test_id"],
        "registered_files_validated": checked,
        "neutrality": {
            "nine_science_planes_bitwise_identical": neutrality_planes == 9,
            "formal_planes_bitwise_identical": formal_exact == 3,
            "checkpoint": checkpoint,
        },
        "total_accumulator_exact_closure": exact_checks,
        "target_sample_ledger": {
            "proposed": proposed,
            "admitted": admitted,
            "already_unavailable": unavailable,
            "reasons": ledger,
        },
        "without_target_reconstruction": {
            "common_conditioned_support_pixels": int(np.count_nonzero(common)),
            "signal_max_abs_difference": float(np.max(np.abs(signal_difference[common]))),
            "signal_max_bound": float(np.max(bounds["signal"][common])),
            "formal_coefficient_max_abs_difference": float(
                np.max(np.abs(coefficient_difference[common]))
            ),
            "formal_coefficient_max_bound": float(
                np.max(bounds["coefficient"][common])
            ),
            "gained_support_pixels": int(np.count_nonzero(reconstructed_support & ~a5_support)),
            "lost_support_pixels": int(np.count_nonzero(a5_support & ~reconstructed_support)),
            "threshold_edge_explained_changes": int(np.count_nonzero(changed_support & support_edge_explained)),
            "unexplained_support_changes": int(np.count_nonzero(unexplained_support)),
            "registered_derived_threshold_bound": threshold_bound,
            "diagnostic_empirical_scale": empirical_scale,
        },
        "deletion_identity": {
            "conditioned_pixels": int(np.count_nonzero(conditioned)),
            "max_abs_residual_mjy_beam": float(
                np.nanmax(np.abs(identity_residual[conditioned]))
            ),
        },
        "region_metrics": metrics,
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
    print(json.dumps(result["without_target_reconstruction"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
