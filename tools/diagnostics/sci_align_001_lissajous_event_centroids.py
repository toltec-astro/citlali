#!/usr/bin/env python3
"""Spatial event qualification and robust centroid models for SCI-ALIGN-001."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table
from scipy.optimize import lsq_linear

import analyze_sci_align_001_lissajous_timestream as analysis


class EventCentroidError(RuntimeError):
    """The frozen event-centroid contract is violated."""


MODEL_NAMES = ("constant", "lag", "hysteresis", "joint")


def load_event_centroid_protocol(
    path: Path, crossing_protocol_path: Path,
) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != (
        "sci-align-001-lissajous-event-centroid-protocol-v1"
    ):
        raise EventCentroidError("unsupported event-centroid protocol schema")
    expected = document["crossing_authority"]["sha256"]
    if analysis.sha256_file(crossing_protocol_path) != expected:
        raise EventCentroidError("crossing protocol identity changed")
    grid = document["source_quality"][
        "trajectory_center_grid_effective_fwhm"
    ]
    if int(grid["count"]) < 3 or int(grid["count"]) % 2 != 1:
        raise EventCentroidError("event-centroid grid must be odd and nontrivial")
    if float(grid["minimum"]) != -float(grid["maximum"]):
        raise EventCentroidError("event-centroid grid is not symmetric")
    thresholds = list(map(
        float, document["source_quality"][
            "correlation_sensitivity_thresholds"
        ],
    ))
    primary = float(document["source_quality"][
        "primary_minimum_correlation"
    ])
    if thresholds != sorted(set(thresholds)) or primary not in thresholds:
        raise EventCentroidError("correlation thresholds are invalid")
    return document


def effective_fwhm_arcsec(
    beam: analysis.BeamGeometry, ux: float, uy: float,
) -> float:
    angle = math.atan2(uy, ux) - float(beam.angle_rad)
    inverse_square = (
        (math.cos(angle) / float(beam.major_fwhm_arcsec)) ** 2
        + (math.sin(angle) / float(beam.minor_fwhm_arcsec)) ** 2
    )
    if not math.isfinite(inverse_square) or inverse_square <= 0.0:
        raise EventCentroidError("effective event FWHM is invalid")
    return 1.0 / math.sqrt(inverse_square)


def _profile_template(
    data: np.ndarray, template: np.ndarray,
) -> tuple[float, float, float]:
    data_centered = data - float(np.mean(data))
    template_centered = template - float(np.mean(template))
    data_square = float(np.dot(data_centered, data_centered))
    template_square = float(np.dot(template_centered, template_centered))
    if data_square <= 0.0 or template_square <= 1.0e-16:
        return math.nan, math.nan, math.nan
    covariance = float(np.dot(data_centered, template_centered))
    amplitude = covariance / template_square
    correlation = covariance / math.sqrt(data_square * template_square)
    intercept = float(np.mean(data) - amplitude * np.mean(template))
    return correlation, amplitude, intercept


def effective_event_windows(events: Table) -> dict[str, tuple[int, int]]:
    """Partition overlapping accepted windows within each detector-scan."""
    accepted = events[np.asarray(events["accepted"], dtype=bool)]
    grouped: dict[tuple[int, int], list[Any]] = {}
    for event in accepted:
        grouped.setdefault(
            (int(event["scan_row"]), int(event["uid"])), []
        ).append(event)
    result: dict[str, tuple[int, int]] = {}
    for members in grouped.values():
        ordered = sorted(members, key=lambda row: int(row["closest_sample"]))
        boundaries = [
            (int(left["closest_sample"]) + int(right["closest_sample"])) // 2 + 1
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ]
        for index, event in enumerate(ordered):
            start = int(event["fit_window_start"])
            stop = int(event["fit_window_stop_exclusive"])
            if index:
                start = max(start, boundaries[index - 1])
            if index < len(boundaries):
                stop = min(stop, boundaries[index])
            if not start < stop:
                raise EventCentroidError("event-window partition is empty")
            result[str(event["event_id"])] = (start, stop)
    return result


def catalog_event_centroids(
    observation: analysis.PreparedObservation,
    events: Table,
    protocol: dict[str, Any],
) -> Table:
    """Profile one symmetric spatial matched filter for every complete event."""
    quality = protocol["source_quality"]
    grid_spec = quality["trajectory_center_grid_effective_fwhm"]
    q_grid = np.linspace(
        float(grid_spec["minimum"]),
        float(grid_spec["maximum"]),
        int(grid_spec["count"]),
    )
    minimum = int(quality["minimum_scored_samples_per_event"])
    primary = float(quality["primary_minimum_correlation"])
    rows: list[dict[str, Any]] = []
    accepted = events[np.asarray(events["accepted"], dtype=bool)]
    windows = effective_event_windows(events)
    for scan in observation.scans:
        scan_events = accepted[
            np.asarray(accepted["scan_row"], dtype=int) == int(scan.scan_row)
        ]
        if not len(scan_events):
            continue
        x, y, _, _ = scan.coordinates(0.0)
        uid_to_detector = {
            int(uid): index for index, uid in enumerate(scan.detector_uid)
        }
        signal = scan.residual_by_baseline["constant"]
        for event in scan_events:
            uid = int(event["uid"])
            detector = uid_to_detector[uid]
            start, stop = windows[str(event["event_id"])]
            indices = np.arange(start, stop, dtype=int)
            indices = indices[scan.score_mask[indices, detector]]
            vx = float(event["velocity_x_arcsec_per_sec"])
            vy = float(event["velocity_y_arcsec_per_sec"])
            speed = math.hypot(vx, vy)
            ux = vx / speed
            uy = vy / speed
            fwhm = effective_fwhm_arcsec(observation.beam, ux, uy)
            base = {
                "event_id": str(event["event_id"]),
                "scan_row": int(scan.scan_row),
                "output_scan_index": int(scan.output_scan_index),
                "uid": uid,
                "network": int(event["network"]),
                "event_index": int(event["detector_event_index"]),
                "effective_window_start": start,
                "effective_window_stop_exclusive": stop,
                "geometric_window_sample_count": int(
                    int(event["fit_window_stop_exclusive"])
                    - int(event["fit_window_start"])
                ),
                "partitioned_window_sample_count": int(stop - start),
                "scored_sample_count": int(indices.size),
                "velocity_x_arcsec_per_sec": vx,
                "velocity_y_arcsec_per_sec": vy,
                "speed_arcsec_per_sec": speed,
                "unit_x": ux,
                "unit_y": uy,
                "effective_fwhm_arcsec": fwhm,
            }
            if indices.size < minimum:
                rows.append({
                    **base,
                    "peak_grid_index": -1,
                    "peak_shift_effective_fwhm": math.nan,
                    "peak_shift_arcsec": math.nan,
                    "peak_correlation": math.nan,
                    "profiled_amplitude_native": math.nan,
                    "profiled_intercept_native": math.nan,
                    "peak_at_grid_boundary": False,
                    "quality_qualified": False,
                    "quality_disposition": "insufficient_event_samples",
                })
                continue
            data = np.asarray(signal[indices, detector], dtype=float)
            correlations = np.full(q_grid.size, np.nan)
            amplitudes = np.full(q_grid.size, np.nan)
            intercepts = np.full(q_grid.size, np.nan)
            xy = np.ix_(indices, [detector])
            for grid_index, q_value in enumerate(q_grid):
                shift = float(q_value) * fwhm
                center_x = float(observation.ppt_x_arcsec) + shift * ux
                center_y = float(observation.ppt_y_arcsec) + shift * uy
                template = analysis.gaussian_beam(
                    x[xy], y[xy],
                    np.full(indices.size, center_x)[:, None],
                    np.full(indices.size, center_y)[:, None],
                    observation.beam,
                )[:, 0]
                (
                    correlations[grid_index],
                    amplitudes[grid_index],
                    intercepts[grid_index],
                ) = _profile_template(data, template)
            if not np.any(np.isfinite(correlations)):
                disposition = "nonfinite_profile"
                peak = -1
                q_peak = math.nan
            else:
                peak = int(np.nanargmax(correlations))
                q_peak = float(q_grid[peak])
                if 0 < peak < q_grid.size - 1:
                    triplet = correlations[peak - 1:peak + 2]
                    curvature = float(triplet[0] - 2.0 * triplet[1] + triplet[2])
                    if np.all(np.isfinite(triplet)) and curvature < 0.0:
                        q_peak += float(
                            0.5 * (triplet[0] - triplet[2]) / curvature
                            * (q_grid[1] - q_grid[0])
                        )
                boundary = peak in {0, q_grid.size - 1}
                if boundary:
                    disposition = "matched_filter_peak_not_bracketed"
                elif not math.isfinite(float(amplitudes[peak])) or (
                    float(amplitudes[peak]) <= 0.0
                ):
                    disposition = "nonpositive_profiled_amplitude"
                elif float(correlations[peak]) < primary:
                    disposition = "correlation_below_primary_threshold"
                else:
                    disposition = "qualified"
            boundary = peak in {0, q_grid.size - 1}
            rows.append({
                **base,
                "peak_grid_index": peak,
                "peak_shift_effective_fwhm": q_peak,
                "peak_shift_arcsec": q_peak * fwhm,
                "peak_correlation": (
                    float(correlations[peak]) if peak >= 0 else math.nan
                ),
                "profiled_amplitude_native": (
                    float(amplitudes[peak]) if peak >= 0 else math.nan
                ),
                "profiled_intercept_native": (
                    float(intercepts[peak]) if peak >= 0 else math.nan
                ),
                "peak_at_grid_boundary": boundary,
                "quality_qualified": disposition == "qualified",
                "quality_disposition": disposition,
            })
    if not rows:
        raise EventCentroidError("event-centroid profiler produced no row")
    return Table(rows=rows)


def qualified_mask(
    rows: Table, threshold: float, minimum_scored_samples: int = 8,
) -> np.ndarray:
    return (
        (np.asarray(rows["scored_sample_count"], dtype=int)
         >= minimum_scored_samples)
        & ~np.asarray(rows["peak_at_grid_boundary"], dtype=bool)
        & np.isfinite(np.asarray(rows["peak_correlation"], dtype=float))
        & (np.asarray(rows["peak_correlation"], dtype=float) >= threshold)
        & np.isfinite(np.asarray(rows["profiled_amplitude_native"], dtype=float))
        & (np.asarray(rows["profiled_amplitude_native"], dtype=float) > 0.0)
    )


def centroid_census(rows: Table, protocol: dict[str, Any]) -> dict[str, Any]:
    primary = float(protocol["source_quality"]["primary_minimum_correlation"])
    minimum = int(protocol["source_quality"][
        "minimum_scored_samples_per_event"
    ])
    qualified = qualified_mask(rows, primary, minimum)
    dispositions: dict[str, int] = {}
    for value in rows["quality_disposition"]:
        key = str(value)
        dispositions[key] = dispositions.get(key, 0) + 1
    sensitivity = {}
    for threshold in protocol["source_quality"][
        "correlation_sensitivity_thresholds"
    ]:
        mask = qualified_mask(rows, float(threshold), minimum)
        sensitivity[f"{float(threshold):.1f}"] = {
            "event_count": int(np.count_nonzero(mask)),
            "detector_count": len(set(map(int, rows[mask]["uid"]))),
        }
    return {
        "assessed_event_count": len(rows),
        "primary_qualified_event_count": int(np.count_nonzero(qualified)),
        "primary_rejected_event_count": int(np.count_nonzero(~qualified)),
        "primary_qualified_detector_count": len(set(map(
            int, rows[qualified]["uid"]
        ))),
        "quality_disposition_counts": dict(sorted(dispositions.items())),
        "threshold_sensitivity": sensitivity,
    }


def _parameter_names(model: str) -> tuple[str, ...]:
    if model == "constant":
        return ("x0_arcsec", "y0_arcsec")
    if model == "lag":
        return ("x0_arcsec", "y0_arcsec", "tau_ms")
    if model == "hysteresis":
        return ("x0_arcsec", "y0_arcsec", "h_az_arcsec", "h_el_arcsec")
    if model == "joint":
        return (
            "x0_arcsec", "y0_arcsec", "tau_ms",
            "h_az_arcsec", "h_el_arcsec",
        )
    raise EventCentroidError(f"unsupported centroid model: {model}")


def _design(rows: Table, model: str) -> np.ndarray:
    ux = np.asarray(rows["unit_x"], dtype=float)
    uy = np.asarray(rows["unit_y"], dtype=float)
    columns = [ux, uy]
    if model in {"lag", "joint"}:
        columns.append(-np.asarray(rows["speed_arcsec_per_sec"], dtype=float) / 1000.0)
    if model in {"hysteresis", "joint"}:
        vx = np.asarray(rows["velocity_x_arcsec_per_sec"], dtype=float)
        vy = np.asarray(rows["velocity_y_arcsec_per_sec"], dtype=float)
        columns.extend([ux * np.sign(vx), uy * np.sign(vy)])
    return np.column_stack(columns)


def _bounds(model: str, protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    spec = protocol["centroid_estimator"]["bounds"]
    bounds = []
    for name in _parameter_names(model):
        if name in {"x0_arcsec", "y0_arcsec"}:
            bounds.append(spec["x0_y0_arcsec"])
        elif name == "tau_ms":
            bounds.append(spec["tau_ms"])
        else:
            bounds.append(spec["h_az_h_el_arcsec"])
    return (
        np.asarray([value[0] for value in bounds], dtype=float),
        np.asarray([value[1] for value in bounds], dtype=float),
    )


def _base_weights(rows: Table) -> np.ndarray:
    counts = Counter(map(int, rows["uid"]))
    return np.asarray([1.0 / counts[int(uid)] for uid in rows["uid"]])


def _huber_loss(residual: np.ndarray, cutoff: float) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(
        absolute <= cutoff,
        0.5 * residual * residual,
        cutoff * (absolute - 0.5 * cutoff),
    )


def _linear_prediction(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Explicit reduction avoids a platform BLAS warning observed for these
    # tiny, moderately conditioned design matrices while preserving arithmetic.
    return np.sum(design * values[None, :], axis=1)


def _bounded_weighted_solution(
    design: np.ndarray,
    measured: np.ndarray,
    weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Solve the tiny weighted linear system without an avoidable TRF path."""
    root = np.sqrt(weights)
    weighted_design = design * root[:, None]
    weighted_measured = measured * root
    solution, _, _, _ = np.linalg.lstsq(
        weighted_design, weighted_measured, rcond=None
    )
    if np.all(solution >= lower) and np.all(solution <= upper):
        return np.asarray(solution, dtype=float)
    result = lsq_linear(
        weighted_design, weighted_measured,
        bounds=(lower, upper), method="trf", tol=1.0e-12,
        max_iter=500,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise EventCentroidError("bounded centroid linear solve failed")
    return np.asarray(result.x, dtype=float)


def robust_centroid_fit(
    rows: Table,
    model: str,
    protocol: dict[str, Any],
    scale_arcsec: float,
    fixed_tau_ms: float | None = None,
) -> dict[str, Any]:
    names = list(_parameter_names(model))
    design = _design(rows, model)
    measured = np.asarray(rows["peak_shift_arcsec"], dtype=float)
    if fixed_tau_ms is not None:
        if "tau_ms" not in names:
            raise EventCentroidError("fixed tau requested for a no-tau model")
        position = names.index("tau_ms")
        measured = measured - design[:, position] * float(fixed_tau_ms)
        design = np.delete(design, position, axis=1)
        names.pop(position)
    lower, upper = _bounds(model, protocol)
    if fixed_tau_ms is not None:
        position = _parameter_names(model).index("tau_ms")
        lower = np.delete(lower, position)
        upper = np.delete(upper, position)
    base = _base_weights(rows)
    weighted_design = design * np.sqrt(base)[:, None]
    design_rank = int(np.linalg.matrix_rank(weighted_design))
    if design_rank != design.shape[1]:
        raise EventCentroidError(f"centroid {model} design is rank deficient")
    design_condition = float(np.linalg.cond(weighted_design))
    if not math.isfinite(design_condition):
        raise EventCentroidError(f"centroid {model} design condition is invalid")
    weights = base.copy()
    cutoff = float(
        protocol["centroid_estimator"]["huber_tuning"]
    ) * scale_arcsec
    solution = None
    iterations = 0
    for iterations in range(1, 101):
        current = _bounded_weighted_solution(
            design, measured, weights, lower, upper
        )
        residual = measured - _linear_prediction(design, current)
        ratio = np.abs(residual) / max(cutoff, 1.0e-12)
        huber = np.ones_like(ratio)
        large = ratio > 1.0
        huber[large] = 1.0 / ratio[large]
        updated = base * huber
        if solution is not None and np.max(np.abs(current - solution)) < 1.0e-10:
            weights = updated
            solution = current
            break
        weights = updated
        solution = current
    if solution is None:
        raise EventCentroidError("centroid fit produced no solution")
    residual = measured - _linear_prediction(design, solution)
    objective = float(np.sum(base * _huber_loss(residual, cutoff)) / np.sum(base))
    parameters = {name: float(value) for name, value in zip(names, solution, strict=True)}
    if fixed_tau_ms is not None:
        parameters["tau_ms"] = float(fixed_tau_ms)
    full_names = _parameter_names(model)
    parameters = {name: parameters[name] for name in full_names}
    return {
        "status": "success",
        "model": model,
        "objective": objective,
        "parameters": parameters,
        "tau_ms": float(parameters.get("tau_ms", 0.0)),
        "robust_scale_arcsec": float(scale_arcsec),
        "huber_cutoff_arcsec": float(cutoff),
        "event_count": len(rows),
        "detector_count": len(set(map(int, rows["uid"]))),
        "effective_base_weight": float(np.sum(base)),
        "final_irls_weight": float(np.sum(weights)),
        "iterations": iterations,
        "design_parameter_count": int(design.shape[1]),
        "design_rank": design_rank,
        "design_condition_number": design_condition,
        "residual_median_arcsec": float(np.median(residual)),
        "residual_mad_scale_arcsec": float(
            1.4826 * np.median(np.abs(residual - np.median(residual)))
        ),
        "residual_rms_arcsec": float(np.sqrt(np.mean(residual * residual))),
        "boundary": bool(np.any(
            np.isclose(solution, lower, atol=1.0e-7)
            | np.isclose(solution, upper, atol=1.0e-7)
        )),
    }


def fit_centroid_models(
    centroid_rows: Table,
    protocol: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    minimum = int(protocol["source_quality"][
        "minimum_scored_samples_per_event"
    ])
    selected = centroid_rows[qualified_mask(
        centroid_rows, threshold, minimum
    )]
    spec = protocol["centroid_estimator"]
    if len(selected) < int(spec["minimum_qualified_events"]):
        raise EventCentroidError("too few qualified centroid events")
    if len(set(map(int, selected["uid"]))) < int(
        spec["minimum_qualified_detectors"]
    ):
        raise EventCentroidError("too few qualified centroid detectors")
    design = _design(selected, "constant")
    measured = np.asarray(selected["peak_shift_arcsec"], dtype=float)
    initial = _bounded_weighted_solution(
        design, measured, _base_weights(selected),
        *_bounds("constant", protocol),
    )
    residual = measured - _linear_prediction(design, initial)
    scale = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    if not math.isfinite(scale) or scale <= 1.0e-6:
        raise EventCentroidError("constant-model robust scale is invalid")
    return {
        "threshold": float(threshold),
        "qualified_event_count": len(selected),
        "qualified_detector_count": len(set(map(int, selected["uid"]))),
        "common_robust_scale_arcsec": scale,
        "models": {
            model: robust_centroid_fit(
                selected, model, protocol, scale
            )
            for model in MODEL_NAMES
        },
    }


def robust_tau_profile(
    centroid_rows: Table,
    protocol: dict[str, Any],
    fit_result: dict[str, Any],
) -> list[dict[str, float]]:
    threshold = float(fit_result["threshold"])
    minimum = int(protocol["source_quality"][
        "minimum_scored_samples_per_event"
    ])
    selected = centroid_rows[qualified_mask(
        centroid_rows, threshold, minimum
    )]
    scale = float(fit_result["common_robust_scale_arcsec"])
    rows = []
    for tau_ms in np.linspace(-50.0, 50.0, 41):
        result = robust_centroid_fit(
            selected, "lag", protocol, scale, fixed_tau_ms=float(tau_ms)
        )
        rows.append({"tau_ms": float(tau_ms), "objective": result["objective"]})
    return rows


def centroid_prediction(rows: Table, fit: dict[str, Any]) -> np.ndarray:
    """Evaluate one persisted centroid model on catalog rows."""
    model = str(fit["model"])
    names = _parameter_names(model)
    parameters = fit["parameters"]
    values = np.asarray([float(parameters[name]) for name in names])
    return _linear_prediction(_design(rows, model), values)


def event_profile_samples(
    observation: analysis.PreparedObservation,
    event: Any,
    centroid: Any,
) -> dict[str, np.ndarray | float | int]:
    """Reconstruct one event's matched-filter data and best local template."""
    matches = [
        scan for scan in observation.scans
        if int(scan.scan_row) == int(event["scan_row"])
    ]
    if len(matches) != 1:
        raise EventCentroidError("event scan identity is not unique")
    scan = matches[0]
    detector_matches = np.flatnonzero(
        np.asarray(scan.detector_uid, dtype=int) == int(event["uid"])
    )
    if detector_matches.size != 1:
        raise EventCentroidError("event detector identity is not unique")
    detector = int(detector_matches[0])
    start = int(centroid["effective_window_start"])
    stop = int(centroid["effective_window_stop_exclusive"])
    context = np.arange(max(0, start - 25), min(
        scan.recorded_time.size, stop + 25
    ))
    scored = np.arange(start, stop, dtype=int)
    scored = scored[scan.score_mask[scored, detector]]
    x, y, _, _ = scan.coordinates(0.0)
    ux = float(centroid["unit_x"])
    uy = float(centroid["unit_y"])
    shift = float(centroid["peak_shift_arcsec"])
    center_x = float(observation.ppt_x_arcsec) + shift * ux
    center_y = float(observation.ppt_y_arcsec) + shift * uy
    template_context = analysis.gaussian_beam(
        x[context, detector], y[context, detector],
        np.full(context.size, center_x), np.full(context.size, center_y),
        observation.beam,
    )
    model_context = (
        float(centroid["profiled_intercept_native"])
        + float(centroid["profiled_amplitude_native"]) * template_context
    )
    closest = int(event["closest_sample"])
    time_ms = 1000.0 * (
        np.asarray(scan.recorded_time[context], dtype=float)
        - float(scan.recorded_time[closest])
    )
    along = (
        (x[context, detector] - float(observation.ppt_x_arcsec)) * ux
        + (y[context, detector] - float(observation.ppt_y_arcsec)) * uy
    )
    return {
        "scan_row": int(scan.scan_row),
        "detector_index": detector,
        "context_indices": context,
        "scored_indices": scored,
        "time_ms": time_ms,
        "along_arcsec": along,
        "data": np.asarray(
            scan.residual_by_baseline["constant"][context, detector],
            dtype=float,
        ),
        "model": np.asarray(model_context, dtype=float),
        "score_mask": np.isin(context, scored),
        "center_x_arcsec": center_x,
        "center_y_arcsec": center_y,
    }
