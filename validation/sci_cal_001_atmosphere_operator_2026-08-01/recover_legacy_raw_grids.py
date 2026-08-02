#!/usr/bin/env python3
"""Recover the local q25/q50/q75 raw-grid lineage for SCI-CAL-001.

The source NPZ files remain read-only in toltec_beammap.  This script verifies
their complete-file digests, extracts the exact monochromatic samples used by
Citlali's legacy fit, reproduces the coefficients, and evaluates raw-anchor and
post-hoc q50 leave-one-model-out correction errors.  It never retrieves the
missing q95 file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_SHA = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
REPAIR_LINE_HEAD = "ae99be1cef8c390d0e7490835ffca1f31da7ebc0"
DEFAULT_SOURCE_DIR = Path("/Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity")
SOURCE_REPOSITORY = "git@github.com:toltec-astro/toltec_beammap.git"
SOURCE_REPOSITORY_HEAD = "958a2a15f43189846a24556a63ef908da789c7b8"
VENDOR_COMMIT = "cedd5345e8d29546f4103f149527e09a9c68a412"
TOLTECA_ORIGIN_MAIN = "2791e6a1e6349ad1d3ac549a648f41cbc51b98c7"
TOLTECA_LMT_LOADER_SHA256 = (
    "f2fbf70dff7a355e70188e11e97f50e059c8104a8fb29953d24de4f1a23235d5"
)
LMT_ATMOSPHERE_SOURCE_SHA256 = (
    "66f580b85ccbfff9152519ec644df363e4571b9263fe06849dc89aa1858e52d0"
)
DETECTOR_SOURCE_SHA256 = (
    "82105317865ae1182d88d0874ed96c36a2b8c79c56d7fc6bb1990f008bd81d1a"
)
MODELED_PASSBAND_SHA256 = (
    "861e6ce7af55b18c14a800defaf0b9a11099a16c307da08e391e1d8f79a39765"
)
MODELED_PASSBAND_MD5 = "c8cae1089964f1a90ecfee36267d1fcd"
PHASE0_SCRIPT_REL = Path(
    "validation/sci_cal_001_phase0_2026-07-31/generate_q_model_continuity.py"
)
PHASE0_SCRIPT_SHA256 = (
    "a46211c007bdc1fa11d1408c6db4c4a68264ca00cd383806fd421ba978fffe78"
)
CALIBRATE_REL = Path("include/citlali/core/timestream/rtc/calibrate.h")
CALIBRATE_SHA256 = "d70a55278227b43cdd7de19bc67e4ddb332524d40e1455c5fa7a80ae5e2d11ee"

RAW_SOURCES = {
    "am_q25": {
        "filename": "amLMT25.npz",
        "sha256": ("6ddffcd2c68bbc0f6d8f6470eba0d1aa81457dcc2f348fd2d7e44c9dfe48c87b"),
        "md5": "008d7fa69aff187a9edf419f3d961b4c",
        "tolteca_datafile_id": "454",
    },
    "am_q50": {
        "filename": "amLMT50.npz",
        "sha256": ("1fe6dd2ab7a4d65f445e20c5a8f438eb42884836e7932d86f80c30e235710f81"),
        "md5": "6ec393672be8af4dfa06a3f4cf9aa32e",
        "tolteca_datafile_id": "455",
    },
    "am_q75": {
        "filename": "amLMT75.npz",
        "sha256": ("adbb8eb974c4e2744c3efb0f627708565f954c4029d9345e4f434699e8843f8e"),
        "md5": "d6cf4bb27008179ec491864388deac58",
        "tolteca_datafile_id": "456",
    },
}
MISSING_Q95 = {
    "model": "am_q95",
    "tolteca_datafile_id": "461",
    "expected_md5": "0ca7b331823237767d26016d19bffb3d",
    "retrieval_status": "not_present_locally_not_contacted",
}

BAND_FREQUENCIES_GHZ = {
    "a1100": 272.73,
    "a1400": 214.29,
    "a2000": 150.00,
}
REFERENCE_FREQUENCY_GHZ = 225.00
EXPECTED_ELEVATIONS_DEG = np.arange(20.0, 82.0, 2.0, dtype=np.float64)
EXPECTED_FREQUENCY_GHZ = np.arange(0, 50001, dtype=np.float64) / 100.0

MANIFEST_NAME = "recovered_raw_grid_manifest.json"
RAW_SUBSET_NAME = "recovered_raw_nominal_grid.csv"
COEFFICIENT_NAME = "recovered_fit_coefficients.csv"
FIT_METRICS_NAME = "raw_anchor_fit_metrics.csv"
OPERATOR_METRICS_NAME = "raw_anchor_operator_metrics.csv"
Q50_HOLDOUT_NAME = "raw_q50_holdout_metrics.csv"
Q50_OPERATOR_HOLDOUT_NAME = "raw_q50_operator_holdout_metrics.csv"
PHYSICAL_METRICS_NAME = "raw_grid_physical_metrics.csv"
REPORT_NAME = "RAW_GRID_RECOVERY_REPORT.md"
MONOTONICITY_TOLERANCE = 1.0e-12


def digest_path(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    return digest_path(path, "sha256")


def f64(value: float) -> str:
    return format(float(value), ".17e")


def render_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("cannot render an empty CSV artifact")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(rows[0]),
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def load_phase0_module(repo_root: Path):
    expected = {
        PHASE0_SCRIPT_REL: PHASE0_SCRIPT_SHA256,
        CALIBRATE_REL: CALIBRATE_SHA256,
    }
    for relative, digest in expected.items():
        actual = sha256_path(repo_root / relative)
        if actual != digest:
            raise RuntimeError(
                f"frozen digest mismatch for {relative}: {actual} != {digest}"
            )
    module_path = repo_root / PHASE0_SCRIPT_REL
    spec = importlib.util.spec_from_file_location(
        "sci_cal_001_phase0_raw_recovery", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_source_files(source_dir: Path) -> None:
    for metadata in RAW_SOURCES.values():
        path = source_dir / metadata["filename"]
        if sha256_path(path) != metadata["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")
        if digest_path(path, "md5") != metadata["md5"]:
            raise RuntimeError(f"MD5 mismatch for {path}")

    supporting = {
        "LMTAtmosphere.py": LMT_ATMOSPHERE_SOURCE_SHA256,
        "Detector.py": DETECTOR_SOURCE_SHA256,
        "model_passbands.npz": MODELED_PASSBAND_SHA256,
    }
    for filename, expected in supporting.items():
        path = source_dir / filename
        if sha256_path(path) != expected:
            raise RuntimeError(f"supporting-source SHA-256 mismatch for {path}")
    if digest_path(source_dir / "model_passbands.npz", "md5") != MODELED_PASSBAND_MD5:
        raise RuntimeError("modeled passband MD5 mismatch")


def load_raw_sources(source_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    validate_source_files(source_dir)
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for model_name, metadata in RAW_SOURCES.items():
        path = source_dir / metadata["filename"]
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["el", "atmFreq", "atmTRJ", "atmTtx"]:
                raise RuntimeError(f"unexpected NPZ members in {path}")
            data = {name: archive[name].copy() for name in archive.files}
        if data["el"].shape != (31,):
            raise RuntimeError(f"unexpected elevation shape in {path}")
        if data["atmFreq"].shape != (50001, 31):
            raise RuntimeError(f"unexpected frequency shape in {path}")
        if data["atmTRJ"].shape != (50001, 31):
            raise RuntimeError(f"unexpected atmosphere-temperature shape in {path}")
        if data["atmTtx"].shape != (50001, 31):
            raise RuntimeError(f"unexpected transmission shape in {path}")
        if not np.array_equal(data["el"], EXPECTED_ELEVATIONS_DEG):
            raise RuntimeError(f"unexpected elevation values in {path}")
        if not np.all(data["atmFreq"] == data["atmFreq"][:, [0]]):
            raise RuntimeError(f"frequency columns differ in {path}")
        if not np.array_equal(data["atmFreq"][:, 0], EXPECTED_FREQUENCY_GHZ):
            raise RuntimeError(f"unexpected frequency grid in {path}")
        if not (
            np.all(np.isfinite(data["atmTtx"]))
            and np.all(data["atmTtx"] > 0.0)
            and np.all(data["atmTtx"] <= 1.0)
        ):
            raise RuntimeError(f"invalid transmission domain in {path}")
        loaded[model_name] = data
    return loaded


def source_order_polynomial(
    coefficients: np.ndarray, elevation_rad: np.ndarray
) -> np.ndarray:
    terms = [
        coefficients[index] * np.power(elevation_rad, 6 - index) for index in range(7)
    ]
    result = terms[0].copy()
    for term in terms[1:]:
        result += term
    return result


def frequency_index(frequency_ghz: float) -> int:
    index = round(frequency_ghz * 100.0)
    if EXPECTED_FREQUENCY_GHZ[index] != frequency_ghz:
        raise RuntimeError(f"frequency is not on the raw grid: {frequency_ghz}")
    return index


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="linear"))


def build_recovery(repo_root: Path, source_dir: Path) -> dict[str, Any]:
    phase0 = load_phase0_module(repo_root)
    source_model = phase0.parse_source(repo_root)
    _, thresholds = phase0.build_rows(source_model)
    raw = load_raw_sources(source_dir)
    elevation_deg = EXPECTED_ELEVATIONS_DEG
    elevation_rad = (
        elevation_deg
        * float(source_model.pi_literal)
        / float(source_model.degree_divisor_literal)
    )
    cosine_zenith = np.cos(float(source_model.pi_literal) / 2.0 - elevation_rad)
    secant_zenith = 1.0 / cosine_zenith
    airmass = secant_zenith * (
        1.0
        - float(source_model.airmass_correction_literal)
        * (np.square(secant_zenith) - 1.0)
    )

    raw_rows: list[dict[str, str]] = []
    coefficient_rows: list[dict[str, str]] = []
    fit_metric_rows: list[dict[str, str]] = []
    operator_metric_rows: list[dict[str, str]] = []
    per_model_band: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    reference_index = frequency_index(REFERENCE_FREQUENCY_GHZ)
    for model_name in RAW_SOURCES:
        data = raw[model_name]
        tx225 = data["atmTtx"][reference_index]
        if tx225[-1] != float(source_model.transmissions[model_name]):
            raise RuntimeError(f"80-degree 225-GHz anchor mismatch for {model_name}")
        for band, band_frequency_ghz in BAND_FREQUENCIES_GHZ.items():
            band_index = frequency_index(band_frequency_ghz)
            band_tx = data["atmTtx"][band_index]
            ratio = band_tx / tx225
            recovered_coefficients = np.polyfit(elevation_rad, ratio, 6)
            rounded_coefficients = np.round(recovered_coefficients, 8)
            source_literals = source_model.coefficients[model_name][band]
            source_coefficients = np.array(
                [float(value) for value in source_literals], dtype=np.float64
            )
            if not np.array_equal(rounded_coefficients, source_coefficients):
                raise RuntimeError(
                    f"rounded coefficient reconstruction failed for {model_name}/{band}"
                )
            fitted_ratio = source_order_polynomial(source_coefficients, elevation_rad)
            ratio_fit_tx = fitted_ratio * tx225
            operator_tx = fitted_ratio * np.exp(-airmass * thresholds[model_name])
            raw_los_tau = -np.log(band_tx)
            ratio_fit_los_tau = -np.log(ratio_fit_tx)
            operator_los_tau = -np.log(operator_tx)
            signed_ratio_fit_correction_error = np.expm1(
                ratio_fit_los_tau - raw_los_tau
            )
            absolute_ratio_fit_correction_error = np.abs(
                signed_ratio_fit_correction_error
            )
            signed_operator_correction_error = np.expm1(operator_los_tau - raw_los_tau)
            absolute_operator_correction_error = np.abs(
                signed_operator_correction_error
            )
            ratio_fit_maximum_index = int(
                np.argmax(absolute_ratio_fit_correction_error)
            )
            operator_maximum_index = int(np.argmax(absolute_operator_correction_error))

            for coefficient_index, source_literal in enumerate(source_literals):
                coefficient_rows.append(
                    {
                        "source_sha": SOURCE_SHA,
                        "model": model_name,
                        "band": band,
                        "degree_power": str(6 - coefficient_index),
                        "source_literal": source_literal,
                        "recovered_unrounded_binary64": f64(
                            recovered_coefficients[coefficient_index]
                        ),
                        "recovered_rounded_8_decimals": f64(
                            rounded_coefficients[coefficient_index]
                        ),
                        "absolute_unrounded_to_source_difference": f64(
                            abs(
                                recovered_coefficients[coefficient_index]
                                - source_coefficients[coefficient_index]
                            )
                        ),
                        "exact_after_8_decimal_rounding": "true",
                    }
                )

            for elevation_index, elevation in enumerate(elevation_deg):
                raw_rows.append(
                    {
                        "source_sha": SOURCE_SHA,
                        "source_npz": RAW_SOURCES[model_name]["filename"],
                        "source_npz_sha256": RAW_SOURCES[model_name]["sha256"],
                        "model": model_name,
                        "band": band,
                        "band_frequency_ghz": f64(band_frequency_ghz),
                        "reference_frequency_ghz": f64(REFERENCE_FREQUENCY_GHZ),
                        "elevation_deg": f64(elevation),
                        "elevation_rad_binary64": f64(elevation_rad[elevation_index]),
                        "raw_225ghz_transmission": f64(tx225[elevation_index]),
                        "raw_band_transmission": f64(band_tx[elevation_index]),
                        "raw_band_to_225_transmission_ratio": f64(
                            ratio[elevation_index]
                        ),
                        "source_polynomial_ratio": f64(fitted_ratio[elevation_index]),
                        "raw_line_of_sight_tau": f64(raw_los_tau[elevation_index]),
                        "ratio_fit_line_of_sight_tau": f64(
                            ratio_fit_los_tau[elevation_index]
                        ),
                        "full_airmass_anchor_line_of_sight_tau": f64(
                            operator_los_tau[elevation_index]
                        ),
                        "signed_ratio_fit_fractional_correction_error": f64(
                            signed_ratio_fit_correction_error[elevation_index]
                        ),
                        "absolute_ratio_fit_fractional_correction_error": f64(
                            absolute_ratio_fit_correction_error[elevation_index]
                        ),
                        "signed_operator_fractional_correction_error": f64(
                            signed_operator_correction_error[elevation_index]
                        ),
                        "absolute_operator_fractional_correction_error": f64(
                            absolute_operator_correction_error[elevation_index]
                        ),
                    }
                )

            fit_metric_rows.append(
                {
                    "model": model_name,
                    "band": band,
                    "truth_kind": "recovered_raw_am_grid_ratio_fit_at_nominal_frequency",
                    "raw_elevation_min_deg": f64(elevation_deg[0]),
                    "raw_elevation_max_deg": f64(elevation_deg[-1]),
                    "raw_elevation_step_deg": f64(2.0),
                    "raw_elevation_count": str(elevation_deg.size),
                    "max_abs_fractional_correction_error": f64(
                        float(np.max(absolute_ratio_fit_correction_error))
                    ),
                    "p95_abs_fractional_correction_error": f64(
                        percentile(absolute_ratio_fit_correction_error, 0.95)
                    ),
                    "median_abs_fractional_correction_error": f64(
                        percentile(absolute_ratio_fit_correction_error, 0.5)
                    ),
                    "rms_fractional_correction_error": f64(
                        float(
                            np.sqrt(
                                np.mean(np.square(signed_ratio_fit_correction_error))
                            )
                        )
                    ),
                    "max_error_elevation_deg": f64(
                        elevation_deg[ratio_fit_maximum_index]
                    ),
                    "passes_provisional_1pct_at_raw_anchor_nodes": str(
                        bool(np.max(absolute_ratio_fit_correction_error) <= 0.01)
                    ).lower(),
                    "successor_evaluation_disposition": "historical_generic_anchor_diagnostic_only_independent_am12_intermediate_runs_missing",
                }
            )
            operator_metric_rows.append(
                {
                    "model": model_name,
                    "band": band,
                    "candidate": "owner_approved_full_airmass_anchor_reconstruction_v0",
                    "truth_kind": "recovered_raw_am_grid_at_nominal_frequency",
                    "raw_elevation_min_deg": f64(elevation_deg[0]),
                    "raw_elevation_max_deg": f64(elevation_deg[-1]),
                    "raw_elevation_step_deg": f64(2.0),
                    "raw_elevation_count": str(elevation_deg.size),
                    "selector_tau225_binary64": f64(thresholds[model_name]),
                    "max_abs_fractional_correction_error": f64(
                        float(np.max(absolute_operator_correction_error))
                    ),
                    "p95_abs_fractional_correction_error": f64(
                        percentile(absolute_operator_correction_error, 0.95)
                    ),
                    "median_abs_fractional_correction_error": f64(
                        percentile(absolute_operator_correction_error, 0.5)
                    ),
                    "rms_fractional_correction_error": f64(
                        float(
                            np.sqrt(
                                np.mean(np.square(signed_operator_correction_error))
                            )
                        )
                    ),
                    "min_signed_fractional_correction_error": f64(
                        float(np.min(signed_operator_correction_error))
                    ),
                    "max_signed_fractional_correction_error": f64(
                        float(np.max(signed_operator_correction_error))
                    ),
                    "max_error_elevation_deg": f64(
                        elevation_deg[operator_maximum_index]
                    ),
                    "passes_provisional_1pct_at_raw_anchor_nodes": str(
                        bool(np.max(absolute_operator_correction_error) <= 0.01)
                    ).lower(),
                    "successor_evaluation_disposition": "historical_generic_anchor_diagnostic_only_independent_am12_intermediate_runs_missing",
                }
            )
            per_model_band[model_name, band] = {
                "los_tau": raw_los_tau,
                "operator_los_tau": operator_los_tau,
                "transmission": band_tx,
            }

    q50_holdout_rows: list[dict[str, str]] = []
    q50_operator_holdout_rows: list[dict[str, str]] = []
    interpolation_fraction = (thresholds["am_q50"] - thresholds["am_q25"]) / (
        thresholds["am_q75"] - thresholds["am_q25"]
    )
    for band in BAND_FREQUENCIES_GHZ:
        q25_los = per_model_band["am_q25", band]["los_tau"]
        q50_los = per_model_band["am_q50", band]["los_tau"]
        q75_los = per_model_band["am_q75", band]["los_tau"]
        predicted = (
            1.0 - interpolation_fraction
        ) * q25_los + interpolation_fraction * q75_los
        signed_error = np.expm1(predicted - q50_los)
        absolute_error = np.abs(signed_error)
        maximum_index = int(np.argmax(absolute_error))
        q50_holdout_rows.append(
            {
                "candidate": "piecewise_linear_los_tau_raw_anchor_holdout_v0",
                "withheld_raw_model": "am_q50",
                "training_raw_models": "am_q25;am_q75",
                "band": band,
                "truth_kind": "recovered_raw_am_grid_at_nominal_frequency",
                "tau225_q25_binary64": f64(thresholds["am_q25"]),
                "tau225_q50_binary64": f64(thresholds["am_q50"]),
                "tau225_q75_binary64": f64(thresholds["am_q75"]),
                "interpolation_fraction_binary64": f64(interpolation_fraction),
                "max_abs_fractional_correction_error": f64(
                    float(np.max(absolute_error))
                ),
                "p95_abs_fractional_correction_error": f64(
                    percentile(absolute_error, 0.95)
                ),
                "median_abs_fractional_correction_error": f64(
                    percentile(absolute_error, 0.5)
                ),
                "rms_fractional_correction_error": f64(
                    float(np.sqrt(np.mean(np.square(signed_error))))
                ),
                "min_signed_fractional_correction_error": f64(
                    float(np.min(signed_error))
                ),
                "max_signed_fractional_correction_error": f64(
                    float(np.max(signed_error))
                ),
                "max_error_elevation_deg": f64(elevation_deg[maximum_index]),
                "passes_provisional_1pct_on_this_holdout": str(
                    bool(np.max(absolute_error) <= 0.01)
                ).lower(),
                "successor_evaluation_disposition": "historical_post_hoc_single_holdout_only_not_successor_gate",
            }
        )
        q25_operator_los = per_model_band["am_q25", band]["operator_los_tau"]
        q75_operator_los = per_model_band["am_q75", band]["operator_los_tau"]
        operator_prediction = (
            1.0 - interpolation_fraction
        ) * q25_operator_los + interpolation_fraction * q75_operator_los
        operator_signed_error = np.expm1(operator_prediction - q50_los)
        operator_absolute_error = np.abs(operator_signed_error)
        operator_maximum_index = int(np.argmax(operator_absolute_error))
        q50_operator_holdout_rows.append(
            {
                "candidate": "piecewise_linear_full_airmass_anchor_los_tau_raw_holdout_v0",
                "withheld_raw_model": "am_q50",
                "training_operator_models": "am_q25;am_q75",
                "band": band,
                "truth_kind": "recovered_raw_am_grid_at_nominal_frequency",
                "tau225_q25_binary64": f64(thresholds["am_q25"]),
                "tau225_q50_binary64": f64(thresholds["am_q50"]),
                "tau225_q75_binary64": f64(thresholds["am_q75"]),
                "interpolation_fraction_binary64": f64(interpolation_fraction),
                "max_abs_fractional_correction_error": f64(
                    float(np.max(operator_absolute_error))
                ),
                "p95_abs_fractional_correction_error": f64(
                    percentile(operator_absolute_error, 0.95)
                ),
                "median_abs_fractional_correction_error": f64(
                    percentile(operator_absolute_error, 0.5)
                ),
                "rms_fractional_correction_error": f64(
                    float(np.sqrt(np.mean(np.square(operator_signed_error))))
                ),
                "min_signed_fractional_correction_error": f64(
                    float(np.min(operator_signed_error))
                ),
                "max_signed_fractional_correction_error": f64(
                    float(np.max(operator_signed_error))
                ),
                "max_error_elevation_deg": f64(elevation_deg[operator_maximum_index]),
                "passes_provisional_1pct_on_this_holdout": str(
                    bool(np.max(operator_absolute_error) <= 0.01)
                ).lower(),
                "successor_evaluation_disposition": "historical_post_hoc_single_holdout_only_not_successor_gate",
            }
        )

    raw_physical_rows: list[dict[str, str]] = []
    for band in BAND_FREQUENCIES_GHZ:
        los_surface = np.stack(
            [per_model_band[model_name, band]["los_tau"] for model_name in RAW_SOURCES]
        )
        opacity_diff = np.diff(los_surface, axis=0)
        elevation_diff = np.diff(los_surface, axis=1)
        correction = np.exp(los_surface)
        running_minimum = np.minimum.accumulate(correction, axis=1)
        wrong_way_excursion = correction / running_minimum - 1.0
        raw_physical_rows.append(
            {
                "band": band,
                "models": ";".join(RAW_SOURCES),
                "raw_elevation_min_deg": f64(elevation_deg[0]),
                "raw_elevation_max_deg": f64(elevation_deg[-1]),
                "finite_nonnegative_los_tau": str(
                    bool(
                        np.all(np.isfinite(los_surface)) and np.all(los_surface >= 0.0)
                    )
                ).lower(),
                "opacity_monotonicity_violation_cells": str(
                    int(np.count_nonzero(opacity_diff < -MONOTONICITY_TOLERANCE))
                ),
                "minimum_opacity_los_tau_step": f64(float(np.min(opacity_diff))),
                "elevation_monotonicity_violation_cells": str(
                    int(np.count_nonzero(elevation_diff > MONOTONICITY_TOLERANCE))
                ),
                "maximum_wrong_way_elevation_los_tau_step": f64(
                    max(0.0, float(np.max(elevation_diff)))
                ),
                "maximum_wrong_way_elevation_correction_excursion_fraction": f64(
                    float(np.max(wrong_way_excursion))
                ),
            }
        )

    return {
        "phase0": phase0,
        "source_model": source_model,
        "thresholds": thresholds,
        "raw": raw,
        "raw_rows": raw_rows,
        "coefficient_rows": coefficient_rows,
        "fit_metric_rows": fit_metric_rows,
        "operator_metric_rows": operator_metric_rows,
        "q50_holdout_rows": q50_holdout_rows,
        "q50_operator_holdout_rows": q50_operator_holdout_rows,
        "raw_physical_rows": raw_physical_rows,
    }


def build_manifest(recovery: dict[str, Any]) -> bytes:
    raw = recovery["raw"]
    manifest = {
        "schema_version": "sci-cal-001-recovered-raw-grid-manifest-v1",
        "identity": {
            "package": "SCI-CAL-001",
            "repair_base_sha": SOURCE_SHA,
            "repair_line_evidence_head": REPAIR_LINE_HEAD,
            "status": "historical_generic_diagnostic_raw_recovery_q25_q50_q75",
            "owner_direction": "versioned_am12_successor_evaluation_only",
            "adoption_status": "evaluation_only_not_adopted",
            "q95_operational_disposition": "excluded_historical_diagnostic_only",
            "successor_study_status": "pending_results",
            "study_artifact_binding_status": "unbound_pending_study_results",
            "operator_authorization": "none",
            "operational_domain_authorization": "none",
        },
        "source_repository": {
            "remote": SOURCE_REPOSITORY,
            "observed_head": SOURCE_REPOSITORY_HEAD,
            "vendor_commit": VENDOR_COMMIT,
            "source_directory_semantics": (
                "runtime --source-dir is not part of artifact identity; "
                "files are located by filename and digest"
            ),
            "files_are_read_only_inputs": True,
        },
        "raw_sources": [
            {
                "model": model_name,
                "path_within_source_directory": metadata["filename"],
                "filename": metadata["filename"],
                "sha256": metadata["sha256"],
                "md5": metadata["md5"],
                "tolteca_datafile_id": metadata["tolteca_datafile_id"],
                "npz_members": {
                    key: {
                        "shape": list(raw[model_name][key].shape),
                        "dtype": str(raw[model_name][key].dtype),
                    }
                    for key in ("el", "atmFreq", "atmTRJ", "atmTtx")
                },
            }
            for model_name, metadata in RAW_SOURCES.items()
        ],
        "missing_q95_request": MISSING_Q95,
        "tolteca_registry": {
            "repository_ref": TOLTECA_ORIGIN_MAIN,
            "path": "tolteca/simu/lmt/__init__.py",
            "sha256": TOLTECA_LMT_LOADER_SHA256,
        },
        "supporting_sources": [
            {
                "path_within_source_directory": "LMTAtmosphere.py",
                "sha256": LMT_ATMOSPHERE_SOURCE_SHA256,
                "role": "states am origin and 20--80 degree NPZ support",
            },
            {
                "path_within_source_directory": "Detector.py",
                "sha256": DETECTOR_SOURCE_SHA256,
                "role": "documents separate modeled passband consumer",
            },
            {
                "path_within_source_directory": "model_passbands.npz",
                "sha256": MODELED_PASSBAND_SHA256,
                "md5": MODELED_PASSBAND_MD5,
                "role": "available modeled passbands not used by recovered Citlali fit",
            },
        ],
        "raw_grid": {
            "elevation_deg": {
                "minimum": f64(20.0),
                "maximum": f64(80.0),
                "step": f64(2.0),
                "count": 31,
            },
            "frequency_ghz": {
                "minimum": f64(0.0),
                "maximum": f64(500.0),
                "step": f64(0.01),
                "count": 50001,
            },
            "transmission_identity": (
                "dimensionless atmospheric transmission stored as atmTtx; "
                "exact reference-plane wording is not embedded"
            ),
            "atmosphere_temperature_identity": "rayleigh_jeans_kelvin",
        },
        "recovered_legacy_fit": {
            "reference_frequency_ghz": f64(REFERENCE_FREQUENCY_GHZ),
            "band_frequencies_ghz": {
                band: f64(frequency) for band, frequency in BAND_FREQUENCIES_GHZ.items()
            },
            "ratio": "atmTtx(band_frequency,elevation)/atmTtx(225GHz,elevation)",
            "elevation_coordinate": "radians",
            "polynomial_degree": 6,
            "fit": "numpy.polyfit over all 31 raw elevation nodes",
            "coefficient_rounding_decimal_places": 8,
            "passband_integration": "none_monochromatic_sampling",
        },
        "numeric_environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "float_radix": sys.float_info.radix,
            "float_mantissa_bits": sys.float_info.mant_dig,
        },
        "unresolved_generation_provenance": [
            "exact am version or executable digest",
            "am command and configuration",
            "atmospheric profile files and percentile construction",
            "site and slant-geometry directives used during generation",
            "origin of the raw frequency/elevation output directives",
            "q95 raw NPZ bytes",
        ],
    }
    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def render_report(recovery: dict[str, Any]) -> bytes:
    thresholds = recovery["thresholds"]
    fit_rows = recovery["fit_metric_rows"]
    operator_rows = recovery["operator_metric_rows"]
    holdout_rows = recovery["q50_holdout_rows"]
    operator_holdout_rows = recovery["q50_operator_holdout_rows"]
    maximum_fit = max(
        fit_rows,
        key=lambda row: float(row["max_abs_fractional_correction_error"]),
    )
    maximum_holdout = max(
        holdout_rows,
        key=lambda row: float(row["max_abs_fractional_correction_error"]),
    )
    maximum_operator = max(
        operator_rows,
        key=lambda row: float(row["max_abs_fractional_correction_error"]),
    )
    maximum_operator_holdout = max(
        operator_holdout_rows,
        key=lambda row: float(row["max_abs_fractional_correction_error"]),
    )
    raw_physical_rows = recovery["raw_physical_rows"]
    opacity_violations = sum(
        int(row["opacity_monotonicity_violation_cells"]) for row in raw_physical_rows
    )
    elevation_violations = sum(
        int(row["elevation_monotonicity_violation_cells"]) for row in raw_physical_rows
    )
    lines = [
        "# SCI-CAL-001 recovered raw atmosphere-grid evidence",
        "",
        "## Recovery result",
        "",
        "The complete local q25, q50, and q75 NPZ grids are identified by both SHA-256 and the MD5 values in TolTECA's static-data registry. The q95 bytes are not local. No remote endpoint or Unity system was contacted.",
        "",
        "Each recovered NPZ has 31 elevations from 20 through 80 degrees in 2-degree steps and 50,001 frequency samples from 0 through 500 GHz in 0.01-GHz steps. It contains dimensionless transmission (`atmTtx`) and Rayleigh-Jeans atmosphere temperature (`atmTRJ`).",
        "",
        "## Exact legacy-fit reconstruction",
        "",
        "For every q25/q50/q75 model and TolTEC band, the current Citlali coefficients are reproduced exactly after eight-decimal rounding by fitting degree six in elevation radians to:",
        "",
        "```text",
        "atmTtx(nu_band, elevation) / atmTtx(225.00 GHz, elevation)",
        "```",
        "",
        "with `nu_band = 272.73, 214.29, 150.00 GHz` for a1100, a1400, and a2000. All 31 raw elevation nodes participate in `numpy.polyfit`. The legacy operator is therefore monochromatic at recovered nominal-wavelength frequencies; no TolTEC passband is integrated in this lineage.",
        "",
        "The exact 80-degree 225-GHz raw transmissions and source-derived selector tau225 values are:",
        "",
        "The selector coordinate is zenith optical depth: `tau225 = -log(T225 at 80 deg) / X(80 deg)` with the repair-base modified-secant airmass, not the unscaled 80-degree slant optical depth.",
        "",
        "| Model | T225 at 80 deg | selector tau225 |",
        "| --- | ---: | ---: |",
    ]
    source_model = recovery["source_model"]
    for model_name in RAW_SOURCES:
        lines.append(
            f"| `{model_name}` | `{source_model.transmissions[model_name]}` | `{f64(thresholds[model_name])}` |"
        )
    lines.extend(
        [
            "",
            "The unrounded-to-source coefficient differences are all below half of the final decimal unit, and every rounded coefficient is exactly equal to the repair-base literal. `recovered_fit_coefficients.csv` preserves all 63 comparisons.",
            "",
            "## Raw-node representation fidelity",
            "",
            "`raw_anchor_fit_metrics.csv` isolates the degree-six transmission-ratio fit from the 225-GHz slant-path reconstruction. Its worst fractional correction error is `"
            + f"{100.0 * float(maximum_fit['max_abs_fractional_correction_error']):.6f}%`"
            + " for `"
            + maximum_fit["model"]
            + "/"
            + maximum_fit["band"]
            + "`. `raw_anchor_operator_metrics.csv` evaluates the owner-required top-of-atmosphere-pivot, full-sample-airmass anchor reconstruction using the repair-base coefficients. It is not the current application correction, whose missing sample-airmass factor remains separate mandatory repair scope. Its worst raw-anchor correction error is `"
            + f"{100.0 * float(maximum_operator['max_abs_fractional_correction_error']):.6f}%`"
            + " for `"
            + maximum_operator["model"]
            + "/"
            + maximum_operator["band"]
            + "`. These are real q25--q75 historical generic-anchor diagnostics, not the separately versioned AM 12.2 successor-adoption gate.",
            "",
            "A post-hoc raw leave-one-model-out check is possible at q50: interpolate raw LOS optical depth between raw q25 and q75 using the exact selector tau225 coordinates, then compare with the recovered raw q50 calculation. q50 was already inspected during provenance recovery, so this is not a preregistered or blinded holdout. Its worst correction error is `"
            + f"{100.0 * float(maximum_holdout['max_abs_fractional_correction_error']):.6f}%`"
            + " in `"
            + maximum_holdout["band"]
            + "`. Interpolating the full-airmass q25/q75 anchor reconstructions instead and comparing with raw q50 gives worst error `"
            + f"{100.0 * float(maximum_operator_holdout['max_abs_fractional_correction_error']):.6f}%`"
            + " in `"
            + maximum_operator_holdout["band"]
            + "`. Both pass one percent in this single post-hoc q50 check only. They do not provide preregistered AM 12.2 intermediate profiles across the selected q95-excluding successor study or declare an operational domain.",
            "",
            "Across the recovered q25/q50/q75 nominal-frequency raw surfaces, `raw_grid_physical_metrics.csv` records "
            + str(opacity_violations)
            + " increasing-opacity violations and "
            + str(elevation_violations)
            + " increasing-elevation wrong-way cells at tolerance `1e-12`.",
            "",
            "## Provenance still missing",
            "",
            "The local evidence names Scott Paine's `am` model and historical LMT percentile grids, but it does not preserve the exact generic-product `am` executable/version, atmosphere-profile files, percentile construction, generation command, or site/geometry directives. The historical generic q95 request is TolTECA datafile ID `461`, expected MD5 `0ca7b331823237767d26016d19bffb3d`; those bytes remain required only for faithful historical-lineage closure. They are not a gate for the selected q95-excluding AM 12.2 successor evaluation.",
            "",
            "The nearby modeled passband artifact and TolTECA's versioned passband tables are not inputs to the recovered Citlali coefficients. A band-integrated successor would be a new, explicitly approved spectral convention, not a faithful rerun of this monochromatic lineage.",
            "",
            "## Disposition",
            "",
            "This partial recovery materially narrows the historical provenance record. The owner has selected evaluation of a separately versioned AM 12.2 successor with generic q95 retained as diagnostic-only evidence, but no final atmosphere operator or operational domain is selected or authorized. Successor-specific profile construction, independent intermediate-opacity runs, spectral and warning/grid policy, exact domain endpoints, and aligned-elevation eligibility remain required.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def expected_artifacts(script_path: Path, source_dir: Path) -> dict[str, bytes]:
    output_dir = script_path.parent
    repo_root = output_dir.parents[1]
    recovery = build_recovery(repo_root, source_dir)
    return {
        MANIFEST_NAME: build_manifest(recovery),
        RAW_SUBSET_NAME: render_csv(recovery["raw_rows"]),
        COEFFICIENT_NAME: render_csv(recovery["coefficient_rows"]),
        FIT_METRICS_NAME: render_csv(recovery["fit_metric_rows"]),
        OPERATOR_METRICS_NAME: render_csv(recovery["operator_metric_rows"]),
        Q50_HOLDOUT_NAME: render_csv(recovery["q50_holdout_rows"]),
        Q50_OPERATOR_HOLDOUT_NAME: render_csv(recovery["q50_operator_holdout_rows"]),
        PHYSICAL_METRICS_NAME: render_csv(recovery["raw_physical_rows"]),
        REPORT_NAME: render_report(recovery),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="read-only directory containing the frozen atmosphere NPZ files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in artifacts instead of rewriting them",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    artifacts = expected_artifacts(script_path, args.source_dir.resolve())
    failed = False
    for name, expected in artifacts.items():
        path = script_path.parent / name
        if args.check:
            if not path.exists() or path.read_bytes() != expected:
                print(f"stale or missing recovered artifact: {path}", file=sys.stderr)
                failed = True
        else:
            path.write_bytes(expected)
            print(f"wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
