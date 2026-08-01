#!/usr/bin/env python3
"""Generate the deterministic SCI-CAL-001 legacy-anchor operator analysis.

This is deliberately not an atmosphere-model regeneration.  It verifies and
parses the exact repair-base q-model literals, treats those fitted surfaces as
surrogate evidence, and compares continuous line-of-sight-optical-depth
representations.  The companion recovery script evaluates the locally
recovered q25/q50/q75 grids; full-domain raw-model fidelity remains
unevaluable until q95 and intermediate-opacity model calculations are supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.interpolate import BarycentricInterpolator, PchipInterpolator


SOURCE_SHA = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
REPAIR_LINE_HEAD = "ae99be1cef8c390d0e7490835ffca1f31da7ebc0"
CALIBRATE_REL = Path("include/citlali/core/timestream/rtc/calibrate.h")
SELECTOR_REL = Path("include/citlali/core/timestream/extinction_model_selection.h")
PHASE0_SCRIPT_REL = Path(
    "validation/sci_cal_001_phase0_2026-07-31/generate_q_model_continuity.py"
)
EXPECTED_DIGESTS = {
    CALIBRATE_REL: ("d70a55278227b43cdd7de19bc67e4ddb332524d40e1455c5fa7a80ae5e2d11ee"),
    SELECTOR_REL: ("45cf86bbb2318c22514411f6d2a0e0371e22e9e355e61b293d93c628d9f3469d"),
    PHASE0_SCRIPT_REL: (
        "a46211c007bdc1fa11d1408c6db4c4a68264ca00cd383806fd421ba978fffe78"
    ),
}

MODELS = ("am_q0", "am_q25", "am_q50", "am_q75", "am_q95")
BANDS = ("a1100", "a1400", "a2000")
CANDIDATES = (
    "piecewise_linear_los_tau_v0",
    "pchip_los_tau_v0",
    "cubic_through_anchors_los_tau_v0",
)
ELEVATION_MIN_DEG = 30.0
ELEVATION_MAX_DEG = 80.0
ELEVATION_STEP_DEG = 0.1
TAU_GRID_COUNT = 1001
MONOTONICITY_TOLERANCE = 1.0e-12

MANIFEST_NAME = "legacy_anchor_manifest.json"
SURFACE_NAME = "legacy_anchor_surface.csv"
ANCHOR_METRICS_NAME = "legacy_anchor_metrics.csv"
CANDIDATE_METRICS_NAME = "candidate_surface_metrics.csv"
LOAO_NAME = "leave_one_anchor_out_metrics.csv"
DISAGREEMENT_NAME = "candidate_disagreement_metrics.csv"
REPORT_NAME = "CONTINUOUS_OPERATOR_EVALUATION.md"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def f64(value: float) -> str:
    return format(float(value), ".17e")


def load_phase0_module(repo_root: Path):
    for relative, expected in EXPECTED_DIGESTS.items():
        actual = sha256_path(repo_root / relative)
        if actual != expected:
            raise RuntimeError(
                f"frozen digest mismatch for {relative}: {actual} != {expected}"
            )

    module_path = repo_root / PHASE0_SCRIPT_REL
    spec = importlib.util.spec_from_file_location(
        "sci_cal_001_phase0_generator", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load phase-0 parser from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_order_polynomial(
    coefficient_literals: tuple[str, ...], elevation_rad: np.ndarray
) -> np.ndarray:
    coefficients = tuple(float(value) for value in coefficient_literals)
    terms = [
        coefficient * np.power(elevation_rad, 6 - index)
        for index, coefficient in enumerate(coefficients)
    ]
    result = terms[0].copy()
    for term in terms[1:]:
        result += term
    return result


def modified_secant_airmass(
    elevation_rad: np.ndarray, pi_value: float, correction: float
) -> np.ndarray:
    cosine_zenith = np.cos(pi_value / 2.0 - elevation_rad)
    secant_zenith = 1.0 / cosine_zenith
    return secant_zenith * (1.0 - correction * (np.power(secant_zenith, 2) - 1.0))


def build_anchor_state(repo_root: Path) -> dict[str, Any]:
    phase0 = load_phase0_module(repo_root)
    model = phase0.parse_source(repo_root)
    _, thresholds_by_name = phase0.build_rows(model)

    elevation_deg = np.linspace(
        ELEVATION_MIN_DEG,
        ELEVATION_MAX_DEG,
        round((ELEVATION_MAX_DEG - ELEVATION_MIN_DEG) / ELEVATION_STEP_DEG) + 1,
        dtype=np.float64,
    )
    pi_value = float(model.pi_literal)
    elevation_rad = elevation_deg * pi_value / float(model.degree_divisor_literal)
    correction = float(model.airmass_correction_literal)
    airmass = modified_secant_airmass(elevation_rad, pi_value, correction)

    thresholds = np.array(
        [thresholds_by_name[name] for name in MODELS], dtype=np.float64
    )
    thresholds[0] = 0.0
    thresholds_by_name["am_q0"] = 0.0
    anchor_los_tau: dict[str, np.ndarray] = {}
    anchor_transmission: dict[str, np.ndarray] = {}
    for band in BANDS:
        los_rows = [np.zeros_like(elevation_rad)]
        transmission_rows = [np.ones_like(elevation_rad)]
        for model_name in MODELS[1:]:
            polynomial = source_order_polynomial(
                model.coefficients[model_name][band], elevation_rad
            )
            transmission = polynomial * np.exp(
                -airmass * thresholds_by_name[model_name]
            )
            if not (np.all(np.isfinite(transmission)) and np.all(transmission > 0.0)):
                raise RuntimeError(
                    f"invalid legacy surrogate transmission for {model_name}/{band}"
                )
            transmission_rows.append(transmission)
            los_rows.append(-np.log(transmission))
        anchor_transmission[band] = np.stack(transmission_rows)
        anchor_los_tau[band] = np.stack(los_rows)

    return {
        "phase0": phase0,
        "model": model,
        "thresholds_by_name": thresholds_by_name,
        "thresholds": thresholds,
        "elevation_deg": elevation_deg,
        "elevation_rad": elevation_rad,
        "airmass": airmass,
        "anchor_los_tau": anchor_los_tau,
        "anchor_transmission": anchor_transmission,
    }


def interpolate_linear(
    x_nodes: np.ndarray, y_nodes: np.ndarray, x_query: np.ndarray
) -> np.ndarray:
    output = np.empty((x_query.size, y_nodes.shape[1]), dtype=np.float64)
    for index, value in enumerate(x_query):
        if value < x_nodes[0] or value > x_nodes[-1]:
            raise ValueError("linear interpolation request is out of support")
        if value == x_nodes[-1]:
            output[index] = y_nodes[-1]
            continue
        upper = int(np.searchsorted(x_nodes, value, side="right"))
        lower = max(0, upper - 1)
        fraction = (value - x_nodes[lower]) / (x_nodes[upper] - x_nodes[lower])
        output[index] = (1.0 - fraction) * y_nodes[lower] + fraction * y_nodes[upper]
    return output


def interpolate_above_q25(
    candidate: str,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    if candidate == "piecewise_linear_los_tau_v0":
        return interpolate_linear(x_nodes, y_nodes, x_query)
    if candidate == "pchip_los_tau_v0":
        result = PchipInterpolator(x_nodes, y_nodes, axis=0, extrapolate=False)(x_query)
        return np.asarray(result, dtype=np.float64).reshape(
            x_query.size, y_nodes.shape[1]
        )
    if candidate == "cubic_through_anchors_los_tau_v0":
        result = BarycentricInterpolator(x_nodes, y_nodes, axis=0)(x_query)
        return np.asarray(result, dtype=np.float64).reshape(
            x_query.size, y_nodes.shape[1]
        )
    raise ValueError(f"unknown candidate: {candidate}")


def evaluate_candidate(
    candidate: str,
    thresholds: np.ndarray,
    anchor_los_tau: np.ndarray,
    tau_query: np.ndarray,
) -> np.ndarray:
    if np.any(tau_query < 0.0) or np.any(tau_query > thresholds[-1]):
        raise ValueError("candidate request is outside diagnostic support")

    output = np.empty((tau_query.size, anchor_los_tau.shape[1]), dtype=np.float64)
    q25 = thresholds[1]
    low = tau_query <= q25
    output[low] = (tau_query[low, np.newaxis] / q25) * anchor_los_tau[1]
    if np.any(~low):
        output[~low] = interpolate_above_q25(
            candidate,
            thresholds[1:],
            anchor_los_tau[1:],
            tau_query[~low],
        )
    return output


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


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="linear"))


def build_manifest(state: dict[str, Any]) -> bytes:
    model = state["model"]
    thresholds_by_name = state["thresholds_by_name"]
    manifest = {
        "schema_version": "sci-cal-001-legacy-anchor-manifest-v1",
        "identity": {
            "package": "SCI-CAL-001",
            "repair_base_sha": SOURCE_SHA,
            "repair_line_evidence_head": REPAIR_LINE_HEAD,
            "evidence_status": ("legacy_polynomial_surrogate_not_raw_atmosphere_model"),
            "operator_authorization": "none",
        },
        "frozen_inputs": [
            {
                "path": str(relative),
                "sha256": digest,
            }
            for relative, digest in EXPECTED_DIGESTS.items()
        ],
        "source_constants": {
            "pi_literal": model.pi_literal,
            "degree_divisor_literal": model.degree_divisor_literal,
            "reference_elevation_deg_literal": (model.reference_elevation_deg_literal),
            "modified_secant_correction_literal": (model.airmass_correction_literal),
        },
        "anchors": [
            {
                "model": name,
                "tau225_selector_anchor_binary64": f64(thresholds_by_name[name]),
                "tau225_selector_anchor_hex": float(thresholds_by_name[name]).hex(),
                "reference_225ghz_transmission_literal": (model.transmissions[name]),
                "coefficients_by_band_source_literals": {
                    band: list(model.coefficients[name][band]) for band in BANDS
                },
            }
            for name in MODELS
        ],
        "diagnostic_domain_not_operational": {
            "tau225_min": f64(0.0),
            "tau225_max": f64(state["thresholds"][-1]),
            "elevation_min_deg": f64(ELEVATION_MIN_DEG),
            "elevation_max_deg": f64(ELEVATION_MAX_DEG),
            "elevation_step_deg": f64(ELEVATION_STEP_DEG),
            "tau_dense_grid_base_count": TAU_GRID_COUNT,
            "monotonicity_tolerance_los_tau": f64(MONOTONICITY_TOLERANCE),
        },
        "candidate_definitions": {
            "common": (
                "For 0<=tau225<=tau_q25 use the owner-approved exact linear "
                "interpolation in LOS optical depth. For tau>tau_q25 evaluate "
                "the exact legacy fitted anchor surface at the requested "
                "elevation, then interpolate LOS optical depth in tau225. "
                "Outside [0,tau_q95] fail closed."
            ),
            "piecewise_linear_los_tau_v0": (
                "Piecewise affine interpolation in tau225 between adjacent "
                "q-anchor LOS optical depths; C0 at every anchor."
            ),
            "pchip_los_tau_v0": (
                "SciPy PCHIP in tau225 through q25/q50/q75/q95 LOS optical "
                "depths; shape preserving in opacity when anchor values are "
                "monotone; C0 with the separately fixed low-opacity segment."
            ),
            "cubic_through_anchors_los_tau_v0": (
                "One barycentric cubic in tau225 through q25/q50/q75/q95 LOS "
                "optical depths; an exact but non-shape-constrained stress "
                "candidate; C0 with the fixed low-opacity segment."
            ),
        },
        "numeric_environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "float_radix": sys.float_info.radix,
            "float_mantissa_bits": sys.float_info.mant_dig,
        },
        "limitations": [
            "Raw q25/q50/q75 calculations are evaluated by the companion recovery script, not this legacy-anchor analysis.",
            "Raw q95 and above-q75/intermediate-opacity atmosphere calculations are not present.",
            "Selector-derived tau225 anchors may not equal original profile inputs.",
            "Legacy elevation polynomials are fitted evidence, not approved raw anchors.",
            "The diagnostic 30--80 degree range is not an operational domain.",
            "The full-domain provisional one-percent raw-grid correction-error gate is not evaluated.",
        ],
    }
    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def build_surface_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    thresholds = state["thresholds"]
    model = state["model"]
    for model_index, model_name in enumerate(MODELS):
        for band in BANDS:
            transmissions = state["anchor_transmission"][band][model_index]
            los_values = state["anchor_los_tau"][band][model_index]
            for elevation_index, elevation_deg in enumerate(state["elevation_deg"]):
                rows.append(
                    {
                        "source_sha": SOURCE_SHA,
                        "evidence_kind": ("legacy_polynomial_surrogate_not_raw_model"),
                        "model": model_name,
                        "band": band,
                        "tau225_selector_anchor_binary64": f64(thresholds[model_index]),
                        "reference_225ghz_transmission_literal": (
                            model.transmissions[model_name]
                        ),
                        "elevation_deg": f64(elevation_deg),
                        "elevation_rad_binary64": f64(
                            state["elevation_rad"][elevation_index]
                        ),
                        "airmass_binary64": f64(state["airmass"][elevation_index]),
                        "transmission_binary64": f64(transmissions[elevation_index]),
                        "line_of_sight_tau_binary64": f64(los_values[elevation_index]),
                        "extinction_correction_binary64": f64(
                            math.exp(los_values[elevation_index])
                        ),
                    }
                )
    return rows


def build_anchor_metric_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model_index, model_name in enumerate(MODELS):
        for band in BANDS:
            los_values = state["anchor_los_tau"][band][model_index]
            elevation_diff = np.diff(los_values)
            correction_values = np.exp(los_values)
            running_minimum = np.minimum.accumulate(correction_values)
            wrong_way_excursion = correction_values / running_minimum - 1.0
            maximum_index = int(np.argmax(wrong_way_excursion))
            rows.append(
                {
                    "model": model_name,
                    "band": band,
                    "tau225_selector_anchor_binary64": f64(
                        state["thresholds"][model_index]
                    ),
                    "finite_positive_transmission": str(
                        bool(
                            np.all(
                                np.isfinite(
                                    state["anchor_transmission"][band][model_index]
                                )
                            )
                            and np.all(
                                state["anchor_transmission"][band][model_index] > 0.0
                            )
                        )
                    ).lower(),
                    "elevation_monotonicity_violation_steps": str(
                        int(np.count_nonzero(elevation_diff > MONOTONICITY_TOLERANCE))
                    ),
                    "max_wrong_way_elevation_los_tau_step": f64(
                        max(0.0, float(np.max(elevation_diff)))
                    ),
                    "max_wrong_way_elevation_correction_excursion_fraction": f64(
                        float(np.max(wrong_way_excursion))
                    ),
                    "max_excursion_elevation_deg": f64(
                        state["elevation_deg"][maximum_index]
                    ),
                    "minimum_los_tau": f64(float(np.min(los_values))),
                    "maximum_los_tau": f64(float(np.max(los_values))),
                }
            )
    return rows


def dense_tau_grid(thresholds: np.ndarray) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.0,
                    thresholds[-1],
                    TAU_GRID_COUNT,
                    dtype=np.float64,
                ),
                thresholds,
            ]
        )
    )


def build_candidate_metric_rows(
    state: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, str]] = []
    thresholds = state["thresholds"]
    tau_grid = dense_tau_grid(thresholds)
    evaluated: dict[str, dict[str, np.ndarray]] = {
        candidate: {} for candidate in CANDIDATES
    }

    for candidate in CANDIDATES:
        for band in BANDS:
            anchors = state["anchor_los_tau"][band]
            los_surface = evaluate_candidate(candidate, thresholds, anchors, tau_grid)
            evaluated[candidate][band] = los_surface
            if not np.all(np.isfinite(los_surface)):
                raise RuntimeError(f"non-finite candidate surface: {candidate}/{band}")

            anchor_errors = []
            continuity_errors = []
            for threshold_index, threshold in enumerate(thresholds):
                at_anchor = evaluate_candidate(
                    candidate,
                    thresholds,
                    anchors,
                    np.array([threshold], dtype=np.float64),
                )[0]
                anchor_errors.append(
                    float(np.max(np.abs(at_anchor - anchors[threshold_index])))
                )
                if 0 < threshold_index < len(thresholds) - 1:
                    around = evaluate_candidate(
                        candidate,
                        thresholds,
                        anchors,
                        np.array(
                            [
                                np.nextafter(threshold, -math.inf),
                                threshold,
                                np.nextafter(threshold, math.inf),
                            ],
                            dtype=np.float64,
                        ),
                    )
                    continuity_errors.append(
                        float(
                            max(
                                np.max(np.abs(around[1] - around[0])),
                                np.max(np.abs(around[2] - around[1])),
                            )
                        )
                    )

            low_tau = np.linspace(0.0, thresholds[1], 101)
            low_surface = evaluate_candidate(candidate, thresholds, anchors, low_tau)
            approved_low = (low_tau[:, np.newaxis] / thresholds[1]) * anchors[1]
            opacity_diff = np.diff(los_surface, axis=0)
            elevation_diff = np.diff(los_surface, axis=1)
            transmission = np.exp(-los_surface)
            correction = np.exp(los_surface)
            running_minimum = np.minimum.accumulate(correction, axis=1)
            wrong_way_excursion = correction / running_minimum - 1.0

            rows.append(
                {
                    "candidate": candidate,
                    "band": band,
                    "diagnostic_tau225_min": f64(0.0),
                    "diagnostic_tau225_max": f64(thresholds[-1]),
                    "diagnostic_elevation_min_deg": f64(ELEVATION_MIN_DEG),
                    "diagnostic_elevation_max_deg": f64(ELEVATION_MAX_DEG),
                    "tau_grid_points": str(tau_grid.size),
                    "elevation_grid_points": str(state["elevation_deg"].size),
                    "max_exact_anchor_abs_los_tau_error": f64(max(anchor_errors)),
                    "max_low_opacity_identity_abs_los_tau_error": f64(
                        float(np.max(np.abs(low_surface - approved_low)))
                    ),
                    "max_nextafter_anchor_abs_los_tau_step": f64(
                        max(continuity_errors, default=0.0)
                    ),
                    "finite_surface": str(
                        bool(np.all(np.isfinite(los_surface)))
                    ).lower(),
                    "positive_transmission": str(
                        bool(np.all(transmission > 0.0))
                    ).lower(),
                    "transmission_not_above_one": str(
                        bool(np.all(transmission <= 1.0 + MONOTONICITY_TOLERANCE))
                    ).lower(),
                    "extinction_correction_at_least_one": str(
                        bool(np.all(correction >= 1.0 - MONOTONICITY_TOLERANCE))
                    ).lower(),
                    "minimum_los_tau": f64(float(np.min(los_surface))),
                    "maximum_los_tau": f64(float(np.max(los_surface))),
                    "minimum_transmission": f64(float(np.min(transmission))),
                    "negative_los_tau_cells": str(
                        int(np.count_nonzero(los_surface < -MONOTONICITY_TOLERANCE))
                    ),
                    "opacity_monotonicity_violation_cells": str(
                        int(np.count_nonzero(opacity_diff < -MONOTONICITY_TOLERANCE))
                    ),
                    "minimum_opacity_los_tau_step": f64(float(np.min(opacity_diff))),
                    "elevation_monotonicity_violation_cells": str(
                        int(np.count_nonzero(elevation_diff > MONOTONICITY_TOLERANCE))
                    ),
                    "max_wrong_way_elevation_los_tau_step": f64(
                        max(0.0, float(np.max(elevation_diff)))
                    ),
                    "max_wrong_way_elevation_correction_excursion_fraction": f64(
                        float(np.max(wrong_way_excursion))
                    ),
                    "raw_grid_fractional_correction_error_status": (
                        "not_evaluable_full_domain_q95_and_intermediate_raw_runs_missing"
                    ),
                }
            )
    return rows, evaluated


def build_leave_one_out_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    thresholds = state["thresholds"]
    for withheld_index in (2, 3):
        withheld_model = MODELS[withheld_index]
        retained = np.array(
            [index for index in range(1, len(MODELS)) if index != withheld_index]
        )
        x_nodes = thresholds[retained]
        x_query = np.array([thresholds[withheld_index]], dtype=np.float64)
        for candidate in CANDIDATES:
            for band in BANDS:
                anchors = state["anchor_los_tau"][band]
                predicted = interpolate_above_q25(
                    candidate,
                    x_nodes,
                    anchors[retained],
                    x_query,
                )[0]
                truth = anchors[withheld_index]
                signed_fractional_error = np.expm1(predicted - truth)
                absolute_fractional_error = np.abs(signed_fractional_error)
                maximum_index = int(np.argmax(absolute_fractional_error))
                rows.append(
                    {
                        "candidate": candidate,
                        "withheld_legacy_anchor": withheld_model,
                        "band": band,
                        "truth_kind": ("legacy_fitted_anchor_not_raw_model_run"),
                        "tau225_binary64": f64(thresholds[withheld_index]),
                        "max_abs_fractional_correction_error": f64(
                            float(np.max(absolute_fractional_error))
                        ),
                        "p95_abs_fractional_correction_error": f64(
                            percentile(absolute_fractional_error, 0.95)
                        ),
                        "median_abs_fractional_correction_error": f64(
                            percentile(absolute_fractional_error, 0.5)
                        ),
                        "rms_fractional_correction_error": f64(
                            float(np.sqrt(np.mean(np.square(signed_fractional_error))))
                        ),
                        "min_signed_fractional_correction_error": f64(
                            float(np.min(signed_fractional_error))
                        ),
                        "max_signed_fractional_correction_error": f64(
                            float(np.max(signed_fractional_error))
                        ),
                        "max_error_elevation_deg": f64(
                            state["elevation_deg"][maximum_index]
                        ),
                        "raw_model_holdout": "false",
                    }
                )
    return rows


def build_disagreement_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    thresholds = state["thresholds"]
    midpoint_specs = [
        (
            MODELS[index],
            MODELS[index + 1],
            (thresholds[index] + thresholds[index + 1]) / 2.0,
        )
        for index in range(1, len(MODELS) - 1)
    ]
    for left_name, right_name, midpoint in midpoint_specs:
        query = np.array([midpoint], dtype=np.float64)
        for band in BANDS:
            candidate_values = {
                candidate: evaluate_candidate(
                    candidate,
                    thresholds,
                    state["anchor_los_tau"][band],
                    query,
                )[0]
                for candidate in CANDIDATES
            }
            for candidate_a, candidate_b in itertools.combinations(CANDIDATES, 2):
                absolute_los_difference = np.abs(
                    candidate_values[candidate_a] - candidate_values[candidate_b]
                )
                symmetric_fractional_correction_difference = np.expm1(
                    absolute_los_difference
                )
                maximum_index = int(
                    np.argmax(symmetric_fractional_correction_difference)
                )
                rows.append(
                    {
                        "left_anchor": left_name,
                        "right_anchor": right_name,
                        "tau225_midpoint_binary64": f64(midpoint),
                        "band": band,
                        "candidate_a": candidate_a,
                        "candidate_b": candidate_b,
                        "max_symmetric_fractional_correction_difference": f64(
                            float(np.max(symmetric_fractional_correction_difference))
                        ),
                        "p95_symmetric_fractional_correction_difference": f64(
                            percentile(symmetric_fractional_correction_difference, 0.95)
                        ),
                        "median_symmetric_fractional_correction_difference": f64(
                            percentile(symmetric_fractional_correction_difference, 0.5)
                        ),
                        "max_difference_elevation_deg": f64(
                            state["elevation_deg"][maximum_index]
                        ),
                        "truth_available": "false",
                    }
                )
    return rows


def render_report(
    state: dict[str, Any],
    anchor_metrics: list[dict[str, str]],
    candidate_metrics: list[dict[str, str]],
    leave_one_out: list[dict[str, str]],
    disagreement: list[dict[str, str]],
) -> bytes:
    thresholds = state["thresholds_by_name"]
    candidate_summary = {}
    for candidate in CANDIDATES:
        selected = [row for row in candidate_metrics if row["candidate"] == candidate]
        heldout = [row for row in leave_one_out if row["candidate"] == candidate]
        candidate_summary[candidate] = {
            "anchor_error": max(
                float(row["max_exact_anchor_abs_los_tau_error"]) for row in selected
            ),
            "low_error": max(
                float(row["max_low_opacity_identity_abs_los_tau_error"])
                for row in selected
            ),
            "opacity_violations": sum(
                int(row["opacity_monotonicity_violation_cells"]) for row in selected
            ),
            "elevation_violations": sum(
                int(row["elevation_monotonicity_violation_cells"]) for row in selected
            ),
            "elevation_excursion": max(
                float(row["max_wrong_way_elevation_correction_excursion_fraction"])
                for row in selected
            ),
            "heldout_error": max(
                float(row["max_abs_fractional_correction_error"]) for row in heldout
            ),
        }

    q95_a2000 = next(
        row
        for row in anchor_metrics
        if row["model"] == "am_q95" and row["band"] == "a2000"
    )
    worst_disagreement = max(
        disagreement,
        key=lambda row: float(row["max_symmetric_fractional_correction_difference"]),
    )

    lines = [
        "# SCI-CAL-001 continuous-operator candidate evaluation",
        "",
        "## Status",
        "",
        "**No successor operator or operational domain is selected.** Exact q25/q50/q75 raw grids and the legacy monochromatic fit have been recovered, while q95, intermediate-opacity model runs, and the original `am` execution/profile provenance remain missing. This report deliberately uses only the exact repair-base legacy q-model polynomials as surrogate evidence; `RAW_GRID_RECOVERY_REPORT.md` contains the bounded raw-grid checks. The full-domain provisional one-percent fractional extinction-correction fidelity gate cannot yet be evaluated.",
        "",
        f"The evidence is bound to repair base `{SOURCE_SHA}` and repair-line evidence head `{REPAIR_LINE_HEAD}`. Frozen input digests are recorded in `{MANIFEST_NAME}`.",
        "",
        "## Diagnostic support",
        "",
        f"The analysis spans tau225 `0` through the source-derived q95 selector anchor `{f64(thresholds['am_q95'])}` and elevation `{ELEVATION_MIN_DEG:.1f}` through `{ELEVATION_MAX_DEG:.1f}` degrees at `{ELEVATION_STEP_DEG:.1f}`-degree spacing. This is the phase-0 diagnostic range, not an approved operational domain. Values outside the tau range are undefined and would fail closed.",
        "",
        "Source-derived selector anchors are:",
        "",
        "| Model | tau225 binary64 |",
        "| --- | ---: |",
    ]
    for model_name in MODELS:
        lines.append(f"| `{model_name}` | `{f64(thresholds[model_name])}` |")

    lines.extend(
        [
            "",
            "## Candidate definitions",
            "",
            "All candidates preserve the owner-approved zero-to-q25 operator exactly: LOS optical depth is `(tau225/tau_q25) * LOS_tau_q25(elevation)`. Above q25 they interpolate the exact legacy fitted LOS-optical-depth anchor surfaces in tau225. The candidates are diagnostic `v0` surfaces, not implementation contracts.",
            "",
            "- `piecewise_linear_los_tau_v0`: piecewise affine in tau225; minimal and C0.",
            "- `pchip_los_tau_v0`: shape-preserving PCHIP in tau225 through q25--q95; C0 where it meets the fixed low-opacity segment.",
            "- `cubic_through_anchors_los_tau_v0`: one unconstrained barycentric cubic through q25--q95; included as an exact-anchor stress candidate.",
            "",
            "Every surface is evaluated as `T = exp(-LOS_tau)` and correction `C = exp(LOS_tau)`. Fractional correction error against a truth value is `abs(exp(LOS_tau_candidate - LOS_tau_truth) - 1)`.",
            "",
            "## Structural and surrogate checks",
            "",
            "| Candidate | Max anchor LOS-tau error | Max low-opacity LOS-tau error | Opacity violations | Elevation violations | Max wrong-way correction excursion | Worst q50/q75 leave-one-out correction error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in CANDIDATES:
        summary = candidate_summary[candidate]
        lines.append(
            f"| `{candidate}` | `{f64(summary['anchor_error'])}` | `{f64(summary['low_error'])}` | {summary['opacity_violations']} | {summary['elevation_violations']} | `{100.0 * summary['elevation_excursion']:.6f}%` | `{100.0 * summary['heldout_error']:.6f}%` |"
        )

    lines.extend(
        [
            "",
            "All candidates are finite, have `0<T<=1` and correction `C>=1` on the diagnostic grid, preserve the exact q anchors to binary64 evaluation precision, and preserve the approved low-opacity identity. The detailed per-band results, nextafter continuity checks, minima, and violation counts are in `candidate_surface_metrics.csv`.",
            "",
            "The legacy q anchors are monotone with increasing opacity throughout the diagnostic elevation grid. The unconstrained cubic is retained only to expose possible between-anchor behavior; exact anchors alone do not establish a valid interpolant.",
            "",
            "Every exact-anchor candidate necessarily inherits the legacy q95/a2000 elevation feature. On the 0.1-degree grid that anchor has "
            + q95_a2000["elevation_monotonicity_violation_steps"]
            + " wrong-way steps and a maximum running-minimum-to-later correction excursion of `"
            + f"{100.0 * float(q95_a2000['max_wrong_way_elevation_correction_excursion_fraction']):.6f}%`"
            + ". This is the owner-identified sub-percent diagnostic; it is not converted here into an absolute-photometry claim.",
            "",
            "## Surrogate holdout and intermediate-opacity evidence",
            "",
            "The leave-one-anchor-out table withholds q50 or q75 and predicts its legacy fitted surface from the other q anchors. This tests interpolation sensitivity, not an independent atmosphere run. It must not be used as the provisional one-percent raw-grid fidelity result.",
            "",
            "The largest pairwise candidate difference at the arithmetic midpoint of an above-q25 interval is `"
            + f"{100.0 * float(worst_disagreement['max_symmetric_fractional_correction_difference']):.6f}%`"
            + " for `"
            + worst_disagreement["band"]
            + "` between `"
            + worst_disagreement["left_anchor"]
            + "` and `"
            + worst_disagreement["right_anchor"]
            + "` (`"
            + worst_disagreement["candidate_a"]
            + "` versus `"
            + worst_disagreement["candidate_b"]
            + "`). No truth value exists locally at those midpoints.",
            "",
            "## Decision disposition",
            "",
            "The evidence is insufficient to choose a versioned successor operator or declare an operational opacity/elevation domain. An exact-anchor surface built from the legacy fits cannot simultaneously remove the q95/a2000 elevation feature, and candidate agreement, legacy-fit leave-one-out performance, or the post-hoc raw q50 leave-one-model-out check is not a substitute for preregistered full-domain raw-model fidelity.",
            "",
            (
                "After the requested raw grid is supplied, evaluate at least the "
                "piecewise-linear LOS-tau baseline and monotone PCHIP against "
                "preregistered withheld tau/elevation model nodes. Select the "
                "simplest candidate that preserves exact approved anchors, "
                "positivity, continuity, opacity monotonicity, fail-closed support, "
                "and no more than one-percent fractional correction error over the "
                "owner-declared domain. Elevation monotonicity must either pass or "
                "receive an explicit owner scientific disposition supported by "
                "recovered raw q95 and independent model evidence. The 0.839827% "
                "q95/a2000 feature is diagnostic rather than automatically "
                "release-blocking, but it may not be silently waived. Observational "
                "5--10% absolute accuracy and approximately 5% repeatability remain "
                "separate later gates."
            ),
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def expected_artifacts(script_path: Path) -> dict[str, bytes]:
    output_dir = script_path.parent
    repo_root = output_dir.parents[1]
    state = build_anchor_state(repo_root)
    surface_rows = build_surface_rows(state)
    anchor_metrics = build_anchor_metric_rows(state)
    candidate_metrics, _ = build_candidate_metric_rows(state)
    leave_one_out = build_leave_one_out_rows(state)
    disagreement = build_disagreement_rows(state)
    return {
        MANIFEST_NAME: build_manifest(state),
        SURFACE_NAME: render_csv(surface_rows),
        ANCHOR_METRICS_NAME: render_csv(anchor_metrics),
        CANDIDATE_METRICS_NAME: render_csv(candidate_metrics),
        LOAO_NAME: render_csv(leave_one_out),
        DISAGREEMENT_NAME: render_csv(disagreement),
        REPORT_NAME: render_report(
            state,
            anchor_metrics,
            candidate_metrics,
            leave_one_out,
            disagreement,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in artifacts instead of rewriting them",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    artifacts = expected_artifacts(script_path)
    failed = False
    for name, expected in artifacts.items():
        path = script_path.parent / name
        if args.check:
            if not path.exists() or path.read_bytes() != expected:
                print(f"stale or missing generated artifact: {path}", file=sys.stderr)
                failed = True
        else:
            path.write_bytes(expected)
            print(f"wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
