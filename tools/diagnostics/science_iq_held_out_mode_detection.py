#!/usr/bin/env python3
"""Test transfer of raw-I/Q response modes without catalog-timed detection.

This forensic diagnostic reconstructs the established stable-UID raw-I/Q
projection for the affected networks and summarizes it in fixed-width time
bins.  A three-state network-vector HMM is trained on the first half of one
event-rich observation without consulting the event catalog.  The frozen
model is then applied to the held-out half, the other science observations,
and the quiet science controls.

Catalog events are not supplied to training or decoding.  They are used only
after detection for one-to-one timing matches, direction checks, and
circular-shift null comparisons.  The test is therefore independent of
catalog event times at detection time, but it remains informed by the
previously learned stable-UID projection.

The fitted states are phenomenological response patterns, not physical
hardware-state identities.  This script never changes Citlali flags, weights,
learning state, or reduction products.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "citlali-iq-held-out-mpl-cache"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from tools.diagnostics.science_iq_continuous_event_morphology import (  # noqa: E402
    DEFAULT_EVENT_RICH_OBSNUMS,
    DEFAULT_NETWORKS,
    EVENT_VECTOR_SCHEMA,
    TEMPLATE_SCHEMA,
    Projection,
    _array_name,
    _file_identity,
    _find_raw_files,
    _load_template,
    _project_network,
    _rack,
    _robust_sigma,
)
from tools.diagnostics.science_iq_hidden_state_analysis import (  # noqa: E402
    CATALOG_SCHEMA,
    HmmFit,
    _forward_backward,
    _log_emission,
    _viterbi,
    _fit_gaussian_hmm,
)


SCHEMA_VERSION = "citlali-science-iq-held-out-mode-detection-v1"
HIDDEN_STATE_SCHEMA = "citlali-science-iq-hidden-state-v1"
DEFAULT_SCIENCE_OBSNUMS = (152390, 152392, 152419, 152431, 152433)


@dataclass(frozen=True)
class SegmentNormalization:
    """Unsupervised per-segment location/trend with frozen applied scales."""

    reference_unix_sec: float
    slope_rad_per_rms_loading_per_hour: np.ndarray
    trend_intercept_rad_per_rms_loading: np.ndarray
    residual_center_rad_per_rms_loading: np.ndarray
    intrinsic_residual_scale_rad_per_rms_loading: np.ndarray
    applied_scale_rad_per_rms_loading: np.ndarray


@dataclass(frozen=True)
class TimeMatch:
    """One-to-one prediction/catalog time match."""

    prediction_index: int
    catalog_index: int
    residual_sec: float


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _finite_median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixed_bin_measurements(
    projections: dict[int, Projection],
    *,
    networks: list[int],
    bin_width_sec: float,
    minimum_samples_per_network: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return common fixed-bin centers, levels, counts, and long-form rows."""
    if not projections:
        raise ValueError("fixed-bin measurement requires projections")
    if bin_width_sec <= 0.0:
        raise ValueError("bin width must be positive")
    start = max(
        float(projections[network].time_unix_sec[0])
        for network in networks
    )
    end = min(
        float(projections[network].time_unix_sec[-1])
        for network in networks
    )
    count = int(math.floor((end - start) / float(bin_width_sec)))
    if count < 2:
        raise ValueError("observation is too short for fixed-bin analysis")
    edges = start + np.arange(count + 1, dtype=float) * bin_width_sec
    centers = 0.5 * (edges[:-1] + edges[1:])
    levels = np.full((count, len(networks)), np.nan, dtype=float)
    sample_counts = np.zeros((count, len(networks)), dtype=int)
    rows: list[dict[str, Any]] = []
    obsnum = int(next(iter(projections.values())).obsnum)
    for column, network in enumerate(networks):
        projection = projections[network]
        if int(projection.obsnum) != obsnum:
            raise ValueError("fixed-bin projections mix observations")
        left = np.searchsorted(
            projection.time_unix_sec,
            edges[:-1],
            side="left",
        )
        right = np.searchsorted(
            projection.time_unix_sec,
            edges[1:],
            side="left",
        )
        for index, (first, stop) in enumerate(
            zip(left, right, strict=True)
        ):
            values = projection.projected_phase_rad[first:stop]
            finite = values[np.isfinite(values)]
            sample_counts[index, column] = int(finite.size)
            status = (
                "measured"
                if finite.size >= int(minimum_samples_per_network)
                else "insufficient_samples"
            )
            if status == "measured":
                levels[index, column] = float(np.median(finite))
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "obsnum": obsnum,
                    "bin_index_zero_based": index,
                    "bin_start_unix_sec": float(edges[index]),
                    "bin_end_unix_sec": float(edges[index + 1]),
                    "bin_center_unix_sec": float(centers[index]),
                    "bin_width_sec": float(bin_width_sec),
                    "network": network,
                    "array": _array_name(network),
                    "rack": _rack(network),
                    "sample_count": int(finite.size),
                    "measurement_status": status,
                    "projected_phase_level_rad_per_rms_loading": (
                        float(levels[index, column])
                        if np.isfinite(levels[index, column])
                        else None
                    ),
                }
            )
    valid = np.all(np.isfinite(levels), axis=1)
    if np.count_nonzero(valid) < 20:
        raise ValueError(
            f"obs {obsnum}: too few complete fixed-bin measurements"
        )
    return centers[valid], levels[valid], sample_counts[valid], [
        row
        for row in rows
        if valid[int(row["bin_index_zero_based"])]
    ]


def _iteratively_clipped_line(
    time_hour: np.ndarray,
    values: np.ndarray,
    *,
    iterations: int = 4,
    clip_sigma: float = 4.0,
) -> tuple[float, float]:
    time_hour = np.asarray(time_hour, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(time_hour) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        raise ValueError("linear trend requires at least two finite samples")
    selected = finite.copy()
    slope, intercept = np.polyfit(
        time_hour[selected],
        values[selected],
        deg=1,
    )
    for _ in range(int(iterations)):
        residual = values - (slope * time_hour + intercept)
        center = float(np.median(residual[selected]))
        scale = _robust_sigma(residual[selected])
        if not np.isfinite(scale) or scale <= 0.0:
            break
        updated = finite & (
            np.abs(residual - center) <= float(clip_sigma) * scale
        )
        if np.count_nonzero(updated) < 2 or np.array_equal(
            updated,
            selected,
        ):
            break
        selected = updated
        slope, intercept = np.polyfit(
            time_hour[selected],
            values[selected],
            deg=1,
        )
    return float(slope), float(intercept)


def _normalize_segment(
    time_unix_sec: np.ndarray,
    levels: np.ndarray,
    *,
    applied_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, SegmentNormalization]:
    """Detrend and center without labels; optionally apply frozen scales."""
    time = np.asarray(time_unix_sec, dtype=float)
    values = np.asarray(levels, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(time):
        raise ValueError("normalization requires time-by-network levels")
    reference = float(np.median(time))
    relative_hour = (time - reference) / 3600.0
    slopes = np.empty(values.shape[1], dtype=float)
    intercepts = np.empty_like(slopes)
    centers = np.empty_like(slopes)
    intrinsic = np.empty_like(slopes)
    residual = np.empty_like(values)
    for coordinate in range(values.shape[1]):
        slope, intercept = _iteratively_clipped_line(
            relative_hour,
            values[:, coordinate],
        )
        trend_removed = values[:, coordinate] - (
            slope * relative_hour + intercept
        )
        center = float(np.median(trend_removed))
        scale = _robust_sigma(trend_removed)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.std(trend_removed))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        slopes[coordinate] = slope
        intercepts[coordinate] = intercept
        centers[coordinate] = center
        intrinsic[coordinate] = scale
        residual[:, coordinate] = trend_removed - center
    if applied_scales is None:
        applied = intrinsic.copy()
    else:
        applied = np.asarray(applied_scales, dtype=float)
        if applied.shape != intrinsic.shape:
            raise ValueError("applied normalization scale has wrong shape")
        if np.any(~np.isfinite(applied)) or np.any(applied <= 0.0):
            raise ValueError("applied normalization scales must be positive")
    standardized = residual / applied[None, :]
    return standardized, SegmentNormalization(
        reference_unix_sec=reference,
        slope_rad_per_rms_loading_per_hour=slopes,
        trend_intercept_rad_per_rms_loading=intercepts,
        residual_center_rad_per_rms_loading=centers,
        intrinsic_residual_scale_rad_per_rms_loading=intrinsic,
        applied_scale_rad_per_rms_loading=applied,
    )


def _reorder_fit(fit: HmmFit, order: list[int]) -> HmmFit:
    order_array = np.asarray(order, dtype=int)
    if sorted(order) != list(range(fit.n_states)):
        raise ValueError("HMM state order is not a permutation")
    inverse = np.empty_like(order_array)
    inverse[order_array] = np.arange(len(order_array))
    return HmmFit(
        n_states=fit.n_states,
        means=fit.means[order_array],
        variances=fit.variances[order_array],
        transition=fit.transition[np.ix_(order_array, order_array)],
        initial=fit.initial[order_array],
        posterior=fit.posterior[:, order_array],
        decoded=inverse[fit.decoded],
        log_likelihood=fit.log_likelihood,
        bic=fit.bic,
        aic=fit.aic,
        parameter_count=fit.parameter_count,
        iterations=fit.iterations,
        converged=fit.converged,
        minimum_center_separation_sigma=(
            fit.minimum_center_separation_sigma
        ),
        minimum_posterior_occupancy_fraction=(
            fit.minimum_posterior_occupancy_fraction
        ),
        selection_eligible=fit.selection_eligible,
        ineligibility_reason=fit.ineligibility_reason,
    )


def _canonicalize_two_mode_fit(
    fit: HmmFit,
    *,
    networks: list[int],
) -> tuple[HmmFit, list[str], dict[str, Any]]:
    """Order states as transition hub, group-129 mode, group-348 mode."""
    if fit.n_states != 3:
        raise ValueError("two-mode canonicalization requires three states")
    decoded_count = np.zeros((3, 3), dtype=int)
    for before, after in zip(
        fit.decoded[:-1],
        fit.decoded[1:],
        strict=True,
    ):
        decoded_count[int(before), int(after)] += 1
    undirected = decoded_count + decoded_count.T
    neighbor_count = np.count_nonzero(
        undirected - np.diag(np.diag(undirected)),
        axis=1,
    )
    off_diagonal_flow = (
        np.sum(undirected, axis=1) - np.diag(undirected)
    )
    occupancy = np.bincount(fit.decoded, minlength=3)
    baseline = max(
        range(3),
        key=lambda state: (
            int(neighbor_count[state]),
            int(off_diagonal_flow[state]),
            int(occupancy[state]),
            -float(np.linalg.norm(fit.means[state])),
        ),
    )
    others = [state for state in range(3) if state != baseline]
    group_129 = np.asarray(
        [network in {1, 2, 9} for network in networks],
        dtype=float,
    )
    group_348 = np.asarray(
        [network in {3, 4, 8} for network in networks],
        dtype=float,
    )
    contrast = group_129 / np.sum(group_129) - (
        group_348 / np.sum(group_348)
    )
    scores = {
        state: float(np.dot(fit.means[state] - fit.means[baseline], contrast))
        for state in others
    }
    mode_129 = max(others, key=lambda state: scores[state])
    mode_348 = next(state for state in others if state != mode_129)
    order = [baseline, mode_129, mode_348]
    result = _reorder_fit(fit, order)
    return (
        result,
        ["baseline_hub", "mode_129_relative", "mode_348_relative"],
        {
            "original_state_order": order,
            "baseline_neighbor_count": int(neighbor_count[baseline]),
            "baseline_off_diagonal_flow": int(
                off_diagonal_flow[baseline]
            ),
            "mode_129_contrast": scores[mode_129],
            "mode_348_contrast": scores[mode_348],
        },
    )


def _decode_frozen_model(
    values: np.ndarray,
    fit: HmmFit,
) -> tuple[np.ndarray, np.ndarray, float]:
    emission = _log_emission(values, fit.means, fit.variances)
    posterior, _, likelihood = _forward_backward(
        emission,
        fit.initial,
        fit.transition,
    )
    decoded = _viterbi(emission, fit.initial, fit.transition)
    return decoded, posterior, likelihood


def _one_to_one_time_matches(
    predicted_times: np.ndarray,
    catalog_times: np.ndarray,
    *,
    tolerance_sec: float,
) -> list[TimeMatch]:
    """Maximize sorted one-to-one matches, then minimize timing residual."""
    predicted = np.asarray(predicted_times, dtype=float)
    catalog = np.asarray(catalog_times, dtype=float)
    n_predicted = len(predicted)
    n_catalog = len(catalog)
    match_count = np.zeros((n_predicted + 1, n_catalog + 1), dtype=int)
    total_cost = np.zeros((n_predicted + 1, n_catalog + 1), dtype=float)
    action = np.zeros((n_predicted + 1, n_catalog + 1), dtype=np.int8)
    for first in range(1, n_predicted + 1):
        action[first, 0] = 1
    for second in range(1, n_catalog + 1):
        action[0, second] = 2
    for first in range(1, n_predicted + 1):
        for second in range(1, n_catalog + 1):
            candidates = [
                (
                    int(match_count[first - 1, second]),
                    float(total_cost[first - 1, second]),
                    1,
                ),
                (
                    int(match_count[first, second - 1]),
                    float(total_cost[first, second - 1]),
                    2,
                ),
            ]
            residual = predicted[first - 1] - catalog[second - 1]
            if abs(residual) <= float(tolerance_sec):
                candidates.append(
                    (
                        int(match_count[first - 1, second - 1] + 1),
                        float(
                            total_cost[first - 1, second - 1]
                            + abs(residual)
                        ),
                        3,
                    )
                )
            best = min(
                candidates,
                key=lambda row: (-row[0], row[1], row[2]),
            )
            match_count[first, second] = best[0]
            total_cost[first, second] = best[1]
            action[first, second] = best[2]
    matches: list[TimeMatch] = []
    first = n_predicted
    second = n_catalog
    while first > 0 or second > 0:
        selected = int(action[first, second])
        if selected == 3:
            matches.append(
                TimeMatch(
                    prediction_index=first - 1,
                    catalog_index=second - 1,
                    residual_sec=float(
                        predicted[first - 1] - catalog[second - 1]
                    ),
                )
            )
            first -= 1
            second -= 1
        elif selected == 1:
            first -= 1
        elif selected == 2:
            second -= 1
        else:
            break
    matches.reverse()
    return matches


def _normalization_rows(
    *,
    segment_id: str,
    obsnum: int,
    networks: list[int],
    normalization: SegmentNormalization,
    scale_source: str,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "segment_id": segment_id,
            "obsnum": obsnum,
            "network": network,
            "array": _array_name(network),
            "rack": _rack(network),
            "normalization_uses_catalog_events": False,
            "trend_method": "iteratively_clipped_linear_fit",
            "reference_unix_sec": normalization.reference_unix_sec,
            "linear_slope_rad_per_rms_loading_per_hour": float(
                normalization.slope_rad_per_rms_loading_per_hour[column]
            ),
            "linear_intercept_rad_per_rms_loading": float(
                normalization.trend_intercept_rad_per_rms_loading[column]
            ),
            "residual_center_rad_per_rms_loading": float(
                normalization.residual_center_rad_per_rms_loading[column]
            ),
            "intrinsic_residual_scale_rad_per_rms_loading": float(
                normalization.intrinsic_residual_scale_rad_per_rms_loading[
                    column
                ]
            ),
            "applied_scale_rad_per_rms_loading": float(
                normalization.applied_scale_rad_per_rms_loading[column]
            ),
            "applied_scale_source": scale_source,
            "intrinsic_to_applied_scale_ratio": float(
                normalization.intrinsic_residual_scale_rad_per_rms_loading[
                    column
                ]
                / normalization.applied_scale_rad_per_rms_loading[column]
            ),
        }
        for column, network in enumerate(networks)
    ]


def _event_conditioned_labels(
    time_unix_sec: np.ndarray,
    *,
    obsnum: int,
    assignments: pd.DataFrame,
) -> np.ndarray:
    selected = assignments[assignments["obsnum"] == int(obsnum)].sort_values(
        "interval_start_unix_sec"
    )
    if selected.empty:
        return np.full(len(time_unix_sec), -1, dtype=int)
    start = selected["interval_start_unix_sec"].to_numpy(dtype=float)
    end = selected["interval_end_unix_sec"].to_numpy(dtype=float)
    state = selected[
        "decoded_state_ordinal_sorted_by_network8_center"
    ].to_numpy(dtype=int)
    index = np.searchsorted(end, time_unix_sec, side="right")
    result = np.full(len(time_unix_sec), -1, dtype=int)
    valid = index < len(state)
    valid &= time_unix_sec >= start[np.minimum(index, len(state) - 1)]
    result[valid] = state[index[valid]]
    return result


def _predicted_transition_rows(
    *,
    segment_id: str,
    obsnum: int,
    role: str,
    time_unix_sec: np.ndarray,
    decoded: np.ndarray,
    posterior: np.ndarray,
    fit: HmmFit,
    state_names: list[str],
    networks: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transition_index = 0
    for index in np.flatnonzero(decoded[1:] != decoded[:-1]):
        before = int(decoded[index])
        after = int(decoded[index + 1])
        center_change = fit.means[after] - fit.means[before]
        dominant = float(np.median(center_change))
        sign = int(np.sign(dominant))
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "prediction_id": (
                    f"{segment_id}_transition_{transition_index:05d}"
                ),
                "segment_id": segment_id,
                "obsnum": obsnum,
                "evaluation_role": role,
                "prediction_uses_catalog_event_times": False,
                "transition_index_zero_based": transition_index,
                "transition_time_unix_sec": float(
                    0.5 * (time_unix_sec[index] + time_unix_sec[index + 1])
                ),
                "bin_before_index_within_segment_zero_based": int(index),
                "bin_after_index_within_segment_zero_based": int(index + 1),
                "state_before": before,
                "state_before_name": state_names[before],
                "state_after": after,
                "state_after_name": state_names[after],
                "dominant_network_median_center_change_standardized": (
                    dominant
                ),
                "predicted_dominant_sign": (
                    "positive"
                    if sign > 0
                    else "negative"
                    if sign < 0
                    else "none"
                ),
                "pre_state_posterior": float(
                    posterior[index, before]
                ),
                "post_state_posterior": float(
                    posterior[index + 1, after]
                ),
                "minimum_adjacent_state_posterior": float(
                    min(
                        posterior[index, before],
                        posterior[index + 1, after],
                    )
                ),
                **{
                    f"nw{network}_center_change_standardized": float(
                        center_change[column]
                    )
                    for column, network in enumerate(networks)
                },
            }
        )
        transition_index += 1
    return rows


def _decoded_bin_rows(
    *,
    segment_id: str,
    obsnum: int,
    role: str,
    time_unix_sec: np.ndarray,
    raw_levels: np.ndarray,
    standardized: np.ndarray,
    decoded: np.ndarray,
    posterior: np.ndarray,
    state_names: list[str],
    networks: list[int],
    event_conditioned: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, time_value in enumerate(time_unix_sec):
        state = int(decoded[index])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "segment_id": segment_id,
                "obsnum": obsnum,
                "evaluation_role": role,
                "prediction_uses_catalog_event_times": False,
                "bin_index_within_segment_zero_based": index,
                "bin_center_unix_sec": float(time_value),
                "decoded_state": state,
                "decoded_state_name": state_names[state],
                "decoded_state_posterior": float(
                    posterior[index, state]
                ),
                "maximum_state_posterior": float(
                    np.max(posterior[index])
                ),
                "event_conditioned_joint_state": (
                    int(event_conditioned[index])
                    if event_conditioned[index] >= 0
                    else None
                ),
                **{
                    f"nw{network}_raw_level_rad_per_rms_loading": float(
                        raw_levels[index, column]
                    )
                    for column, network in enumerate(networks)
                },
                **{
                    f"nw{network}_standardized_level": float(
                        standardized[index, column]
                    )
                    for column, network in enumerate(networks)
                },
            }
        )
    return rows


def _shift_null_rows(
    *,
    segment_id: str,
    obsnum: int,
    prediction_times: np.ndarray,
    catalog_times: np.ndarray,
    segment_start_unix_sec: float,
    segment_end_unix_sec: float,
    tolerance_sec: float,
    permutations: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    duration = float(segment_end_unix_sec - segment_start_unix_sec)
    if duration <= 0.0:
        raise ValueError("shift-null segment duration must be positive")
    generator = np.random.default_rng(random_seed)
    prediction_offset = (
        np.asarray(prediction_times, dtype=float) - segment_start_unix_sec
    )
    rows: list[dict[str, Any]] = []
    for permutation in range(int(permutations)):
        shift = float(generator.uniform(0.05 * duration, 0.95 * duration))
        shifted = (
            segment_start_unix_sec
            + np.mod(prediction_offset + shift, duration)
        )
        shifted.sort()
        match_count = len(
            _one_to_one_time_matches(
                shifted,
                catalog_times,
                tolerance_sec=tolerance_sec,
            )
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "segment_id": segment_id,
                "obsnum": obsnum,
                "permutation_index_zero_based": permutation,
                "circular_shift_sec": shift,
                "predicted_transition_count": int(len(prediction_times)),
                "catalog_event_count": int(len(catalog_times)),
                "matched_count": int(match_count),
                "catalog_recall": (
                    float(match_count / len(catalog_times))
                    if len(catalog_times)
                    else None
                ),
                "prediction_precision": (
                    float(match_count / len(prediction_times))
                    if len(prediction_times)
                    else None
                ),
            }
        )
    return rows


def _catalog_match_rows(
    *,
    segment_id: str,
    obsnum: int,
    role: str,
    catalog_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    matches: list[TimeMatch],
) -> list[dict[str, Any]]:
    by_catalog = {match.catalog_index: match for match in matches}
    rows: list[dict[str, Any]] = []
    for catalog_index, event in enumerate(catalog_rows):
        match = by_catalog.get(catalog_index)
        prediction = (
            transition_rows[match.prediction_index]
            if match is not None
            else None
        )
        catalog_sign = str(event["dominant_projected_step_sign"])
        predicted_sign = (
            str(prediction["predicted_dominant_sign"])
            if prediction is not None
            else None
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "segment_id": segment_id,
                "obsnum": obsnum,
                "evaluation_role": role,
                "catalog_used_only_after_detection": True,
                "catalog_event_index_within_segment_zero_based": (
                    catalog_index
                ),
                "event_id": event["event_id"],
                "event_time_unix_sec": float(
                    event["refined_event_time_unix_sec"]
                ),
                "catalog_quality_tier": event["quality_tier"],
                "catalog_network_count": int(event["network_count"]),
                "catalog_networks": event["networks"],
                "catalog_dominant_projected_step_sign": catalog_sign,
                "matched_prediction": match is not None,
                "prediction_id": (
                    prediction["prediction_id"]
                    if prediction is not None
                    else None
                ),
                "predicted_transition_time_unix_sec": (
                    prediction["transition_time_unix_sec"]
                    if prediction is not None
                    else None
                ),
                "prediction_minus_catalog_time_sec": (
                    match.residual_sec if match is not None else None
                ),
                "absolute_timing_residual_sec": (
                    abs(match.residual_sec)
                    if match is not None
                    else None
                ),
                "predicted_dominant_sign": predicted_sign,
                "predicted_direction_matches_catalog_sign": (
                    predicted_sign == catalog_sign
                    if prediction is not None
                    and predicted_sign in {"positive", "negative"}
                    else None
                ),
                "predicted_state_before": (
                    prediction["state_before"]
                    if prediction is not None
                    else None
                ),
                "predicted_state_before_name": (
                    prediction["state_before_name"]
                    if prediction is not None
                    else None
                ),
                "predicted_state_after": (
                    prediction["state_after"]
                    if prediction is not None
                    else None
                ),
                "predicted_state_after_name": (
                    prediction["state_after_name"]
                    if prediction is not None
                    else None
                ),
                "minimum_adjacent_state_posterior": (
                    prediction["minimum_adjacent_state_posterior"]
                    if prediction is not None
                    else None
                ),
            }
        )
    return rows


def _segment_outputs(
    *,
    segment_id: str,
    obsnum: int,
    role: str,
    scale_policy: str,
    time_unix_sec: np.ndarray,
    raw_levels: np.ndarray,
    standardized: np.ndarray,
    normalization: SegmentNormalization,
    fit: HmmFit,
    state_names: list[str],
    networks: list[int],
    catalog: pd.DataFrame,
    assignments: pd.DataFrame,
    match_tolerance_sec: float,
    shift_null_permutations: int,
    random_seed: int,
    bin_width_sec: float,
) -> dict[str, Any]:
    decoded, posterior, likelihood = _decode_frozen_model(
        standardized,
        fit,
    )
    event_conditioned = _event_conditioned_labels(
        time_unix_sec,
        obsnum=obsnum,
        assignments=assignments,
    )
    bin_rows = _decoded_bin_rows(
        segment_id=segment_id,
        obsnum=obsnum,
        role=role,
        time_unix_sec=time_unix_sec,
        raw_levels=raw_levels,
        standardized=standardized,
        decoded=decoded,
        posterior=posterior,
        state_names=state_names,
        networks=networks,
        event_conditioned=event_conditioned,
    )
    transition_rows = _predicted_transition_rows(
        segment_id=segment_id,
        obsnum=obsnum,
        role=role,
        time_unix_sec=time_unix_sec,
        decoded=decoded,
        posterior=posterior,
        fit=fit,
        state_names=state_names,
        networks=networks,
    )
    segment_start = float(time_unix_sec[0] - 0.5 * bin_width_sec)
    segment_end = float(time_unix_sec[-1] + 0.5 * bin_width_sec)
    selected_catalog = catalog[
        (catalog["obsnum"] == obsnum)
        & catalog["primary_event_candidate"].astype(bool)
        & (catalog["refined_event_time_unix_sec"] >= segment_start)
        & (catalog["refined_event_time_unix_sec"] < segment_end)
    ].sort_values("refined_event_time_unix_sec")
    catalog_rows = selected_catalog.to_dict("records")
    prediction_times = np.asarray(
        [row["transition_time_unix_sec"] for row in transition_rows],
        dtype=float,
    )
    catalog_times = selected_catalog[
        "refined_event_time_unix_sec"
    ].to_numpy(dtype=float)
    matches = _one_to_one_time_matches(
        prediction_times,
        catalog_times,
        tolerance_sec=match_tolerance_sec,
    )
    match_rows = _catalog_match_rows(
        segment_id=segment_id,
        obsnum=obsnum,
        role=role,
        catalog_rows=catalog_rows,
        transition_rows=transition_rows,
        matches=matches,
    )
    null_rows = _shift_null_rows(
        segment_id=segment_id,
        obsnum=obsnum,
        prediction_times=prediction_times,
        catalog_times=catalog_times,
        segment_start_unix_sec=segment_start,
        segment_end_unix_sec=segment_end,
        tolerance_sec=match_tolerance_sec,
        permutations=shift_null_permutations,
        random_seed=random_seed,
    )
    null_count = np.asarray(
        [row["matched_count"] for row in null_rows],
        dtype=int,
    )
    matched_count = len(matches)
    catalog_count = len(catalog_times)
    prediction_count = len(prediction_times)
    valid_conditioned = event_conditioned >= 0
    conditioned_state_count = (
        int(len(np.unique(event_conditioned[valid_conditioned])))
        if np.any(valid_conditioned)
        else 0
    )
    ari = (
        float(
            adjusted_rand_score(
                event_conditioned[valid_conditioned],
                decoded[valid_conditioned],
            )
        )
        if conditioned_state_count > 1
        else None
    )
    nmi = (
        float(
            normalized_mutual_info_score(
                event_conditioned[valid_conditioned],
                decoded[valid_conditioned],
            )
        )
        if conditioned_state_count > 1
        else None
    )
    direction_rows = [
        row
        for row in match_rows
        if row["predicted_direction_matches_catalog_sign"] is not None
    ]
    duration_min = (segment_end - segment_start) / 60.0
    summary = {
        "schema_version": SCHEMA_VERSION,
        "segment_id": segment_id,
        "obsnum": obsnum,
        "evaluation_role": role,
        "target_scale_policy": scale_policy,
        "segment_start_unix_sec": segment_start,
        "segment_end_unix_sec": segment_end,
        "segment_duration_sec": segment_end - segment_start,
        "bin_width_sec": float(bin_width_sec),
        "complete_bin_count": int(len(time_unix_sec)),
        "detector_uses_catalog_event_times": False,
        "catalog_used_only_for_post_detection_scoring": True,
        "frozen_training_model_log_likelihood": float(likelihood),
        "predicted_transition_count": prediction_count,
        "predicted_transition_rate_per_min": float(
            prediction_count / duration_min
        ),
        "catalog_primary_event_count": catalog_count,
        "catalog_primary_event_rate_per_min": float(
            catalog_count / duration_min
        ),
        "one_to_one_matched_count": matched_count,
        "catalog_recall": (
            float(matched_count / catalog_count)
            if catalog_count
            else None
        ),
        "prediction_precision": (
            float(matched_count / prediction_count)
            if prediction_count
            else None
        ),
        "f1": (
            float(
                2.0
                * matched_count
                / (catalog_count + prediction_count)
            )
            if catalog_count + prediction_count
            else None
        ),
        "matched_direction_agreement_fraction": (
            float(
                np.mean(
                    [
                        row[
                            "predicted_direction_matches_catalog_sign"
                        ]
                        for row in direction_rows
                    ]
                )
            )
            if direction_rows
            else None
        ),
        "median_absolute_timing_residual_sec": _finite_median(
            abs(match.residual_sec) for match in matches
        ),
        "shift_null_matched_count_median": float(
            np.median(null_count)
        ),
        "shift_null_matched_count_p95": float(
            np.quantile(null_count, 0.95)
        ),
        "observed_match_count_exceeds_all_shift_nulls": bool(
            matched_count > int(np.max(null_count))
        ),
        "shift_null_empirical_p_greater_equal": float(
            (1 + np.count_nonzero(null_count >= matched_count))
            / (1 + len(null_count))
        ),
        "event_conditioned_state_valid_bin_count": int(
            np.count_nonzero(valid_conditioned)
        ),
        "event_conditioned_state_count": conditioned_state_count,
        "event_conditioned_state_adjusted_rand_index": ari,
        "event_conditioned_state_normalized_mutual_information": nmi,
        **{
            f"{state_names[state]}_decoded_bin_occupancy_fraction": float(
                np.mean(decoded == state)
            )
            for state in range(fit.n_states)
        },
        **{
            f"nw{network}_intrinsic_to_training_scale_ratio": float(
                normalization.intrinsic_residual_scale_rad_per_rms_loading[
                    column
                ]
                / normalization.applied_scale_rad_per_rms_loading[column]
            )
            for column, network in enumerate(networks)
        },
    }
    return {
        "decoded": decoded,
        "posterior": posterior,
        "bin_rows": bin_rows,
        "transition_rows": transition_rows,
        "match_rows": match_rows,
        "null_rows": null_rows,
        "summary": summary,
    }


def _template_parameter_rows(
    *,
    fit: HmmFit,
    state_names: list[str],
    networks: list[int],
    training_normalization: SegmentNormalization,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in range(fit.n_states):
        for column, network in enumerate(networks):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "state": state,
                    "state_name": state_names[state],
                    "network": network,
                    "array": _array_name(network),
                    "rack": _rack(network),
                    "state_center_standardized": float(
                        fit.means[state, column]
                    ),
                    "emission_sigma_standardized": float(
                        math.sqrt(fit.variances[state, column])
                    ),
                    "state_center_relative_rad_per_rms_loading": float(
                        fit.means[state, column]
                        * training_normalization
                        .applied_scale_rad_per_rms_loading[column]
                    ),
                    "emission_sigma_rad_per_rms_loading": float(
                        math.sqrt(fit.variances[state, column])
                        * training_normalization
                        .applied_scale_rad_per_rms_loading[column]
                    ),
                    "training_scale_rad_per_rms_loading": float(
                        training_normalization
                        .applied_scale_rad_per_rms_loading[column]
                    ),
                    "training_posterior_bin_occupancy_fraction": float(
                        np.mean(fit.posterior[:, state])
                    ),
                    "training_decoded_bin_occupancy_fraction": float(
                        np.mean(fit.decoded == state)
                    ),
                }
            )
    return rows


def _template_transition_rows(
    *,
    fit: HmmFit,
    state_names: list[str],
) -> list[dict[str, Any]]:
    decoded_count = np.zeros(
        (fit.n_states, fit.n_states),
        dtype=int,
    )
    for before, after in zip(
        fit.decoded[:-1],
        fit.decoded[1:],
        strict=True,
    ):
        decoded_count[int(before), int(after)] += 1
    rows: list[dict[str, Any]] = []
    for before in range(fit.n_states):
        total = int(np.sum(decoded_count[before]))
        for after in range(fit.n_states):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "state_before": before,
                    "state_before_name": state_names[before],
                    "state_after": after,
                    "state_after_name": state_names[after],
                    "fitted_transition_probability": float(
                        fit.transition[before, after]
                    ),
                    "training_decoded_transition_count": int(
                        decoded_count[before, after]
                    ),
                    "training_decoded_transition_fraction_from_before": (
                        float(decoded_count[before, after] / total)
                        if total
                        else None
                    ),
                }
            )
    return rows


def _template_figure(
    path: Path,
    parameter_rows: list[dict[str, Any]],
    *,
    networks: list[int],
) -> None:
    frame = pd.DataFrame(parameter_rows)
    centers = frame.pivot(
        index="state_name",
        columns="network",
        values="state_center_standardized",
    ).loc[
        ["baseline_hub", "mode_129_relative", "mode_348_relative"],
        networks,
    ]
    delta = centers.iloc[1:, :] - centers.iloc[0, :].to_numpy()[None, :]
    limit = max(
        1.0,
        float(
            np.nanmax(
                np.abs(
                    np.concatenate(
                        [centers.to_numpy(), delta.to_numpy()],
                        axis=0,
                    )
                )
            )
        ),
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.3),
        constrained_layout=True,
    )
    for axis, matrix, title in (
        (
            axes[0],
            centers,
            "Frozen training-state centers",
        ),
        (
            axes[1],
            delta,
            "Response-mode center minus baseline center",
        ),
    ):
        image = axis.imshow(
            matrix.to_numpy(dtype=float),
            vmin=-limit,
            vmax=limit,
            aspect="auto",
            cmap="coolwarm",
        )
        axis.set_title(title)
        axis.set_xticks(
            range(len(networks)),
            [f"nw{network}" for network in networks],
        )
        axis.set_yticks(
            range(len(matrix.index)),
            [str(value) for value in matrix.index],
        )
        for row in range(len(matrix.index)):
            for column in range(len(networks)):
                value = float(matrix.iloc[row, column])
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
    figure.colorbar(
        image,
        ax=axes,
        label="standardized projected-phase coordinate",
    )
    figure.suptitle(
        "Catalog-time-independent template: obs 152431 first half"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _timeline_figure(
    path: Path,
    bin_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    bins = pd.DataFrame(bin_rows)
    matches = pd.DataFrame(match_rows)
    summary = pd.DataFrame(summary_rows)
    display_summary = summary[
        (summary["evaluation_role"] == "training_in_sample")
        | (
            summary["target_scale_policy"]
            == "target_intrinsic_scale"
        )
    ].copy()
    segment_ids = display_summary["segment_id"].astype(str).tolist()
    figure, axes = plt.subplots(
        len(segment_ids),
        1,
        figsize=(13.0, 2.1 * len(segment_ids)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for axis, segment_id in zip(axes, segment_ids, strict=True):
        selected = bins[bins["segment_id"] == segment_id].copy()
        start = float(selected["bin_center_unix_sec"].min())
        relative_min = (
            selected["bin_center_unix_sec"].to_numpy(dtype=float) - start
        ) / 60.0
        decoded = selected["decoded_state"].to_numpy(dtype=int)
        axis.step(
            relative_min,
            decoded,
            where="mid",
            color="0.2",
            linewidth=0.8,
        )
        for state in range(3):
            mask = decoded == state
            axis.scatter(
                relative_min[mask],
                decoded[mask],
                s=4,
                color=colors[state],
                alpha=0.7,
            )
        selected_matches = matches[matches["segment_id"] == segment_id]
        for row in selected_matches.to_dict("records"):
            event_min = (float(row["event_time_unix_sec"]) - start) / 60.0
            axis.axvline(
                event_min,
                color=(
                    "tab:red"
                    if bool(row["matched_prediction"])
                    else "0.7"
                ),
                alpha=0.3,
                linewidth=0.6,
            )
        row = display_summary[
            display_summary["segment_id"] == segment_id
        ].iloc[0]
        axis.set_title(
            f"{segment_id}: transitions={int(row.predicted_transition_count)}, "
            f"catalog={int(row.catalog_primary_event_count)}, "
            f"matched={int(row.one_to_one_matched_count)}"
        )
        axis.set_yticks(
            [0, 1, 2],
            ["baseline", "mode 129", "mode 348"],
        )
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("time from segment start (min)")
    figure.suptitle(
        "Shape-transfer decoded states; "
        "red catalog lines matched after detection"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _performance_figure(
    path: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(summary_rows)
    labels: list[str] = []
    for row in frame.to_dict("records"):
        obsnum = int(row["obsnum"])
        role = str(row["evaluation_role"])
        policy = str(row["target_scale_policy"])
        if role == "training_in_sample":
            suffix = "train"
        elif role == "within_observation_held_out":
            suffix = (
                "held/frozen"
                if policy == "frozen_training_scale"
                else "held/shape"
            )
        else:
            suffix = (
                "frozen"
                if policy == "frozen_training_scale"
                else "shape"
            )
        labels.append(f"{obsnum}\n{suffix}")
    x = np.arange(len(frame), dtype=float)
    observed = frame["one_to_one_matched_count"].to_numpy(dtype=float)
    null_p95 = frame[
        "shift_null_matched_count_p95"
    ].to_numpy(dtype=float)
    recall = frame["catalog_recall"].to_numpy(dtype=float)
    precision = frame["prediction_precision"].to_numpy(dtype=float)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.0),
        constrained_layout=True,
    )
    width = 0.36
    axes[0].bar(
        x - 0.5 * width,
        observed,
        width=width,
        label="observed one-to-one matches",
    )
    axes[0].bar(
        x + 0.5 * width,
        null_p95,
        width=width,
        label="circular-shift null p95",
    )
    axes[0].set_ylabel("matched events")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_xticks(x, [])
    axes[1].plot(
        x,
        recall,
        "o-",
        label="catalog recall",
    )
    axes[1].plot(
        x,
        precision,
        "s-",
        label="prediction precision",
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("fraction")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    figure.suptitle(
        "Held-out/transfer detector timing performance "
        "(catalog used only for scoring)"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--apt-root", type=Path, required=True)
    parser.add_argument("--event-vector-dir", type=Path, required=True)
    parser.add_argument("--tone-analysis-dir", type=Path, required=True)
    parser.add_argument("--continuous-analysis-dir", type=Path, required=True)
    parser.add_argument("--hidden-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--obsnums",
        nargs="+",
        type=int,
        default=list(DEFAULT_SCIENCE_OBSNUMS),
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        type=int,
        default=list(DEFAULT_NETWORKS),
    )
    parser.add_argument(
        "--event-rich-obsnums",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVENT_RICH_OBSNUMS),
    )
    parser.add_argument("--training-obsnum", type=int, default=152431)
    parser.add_argument("--training-fraction", type=float, default=0.5)
    parser.add_argument("--bin-width-sec", type=float, default=0.5)
    parser.add_argument("--minimum-bin-samples", type=int, default=16)
    parser.add_argument("--hmm-initializations", type=int, default=8)
    parser.add_argument("--hmm-maximum-iterations", type=int, default=200)
    parser.add_argument(
        "--hmm-convergence-tolerance",
        type=float,
        default=1.0e-7,
    )
    parser.add_argument(
        "--minimum-state-occupancy-fraction",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--minimum-center-separation-sigma",
        type=float,
        default=1.0,
    )
    parser.add_argument("--match-tolerance-sec", type=float, default=0.75)
    parser.add_argument("--shift-null-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260730)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    networks = [int(value) for value in args.networks]
    obsnums = [int(value) for value in args.obsnums]
    event_rich_obsnums = [
        int(value) for value in args.event_rich_obsnums
    ]
    if len(networks) != 6:
        raise ValueError("the two-mode detector requires six networks")
    if int(args.training_obsnum) not in obsnums:
        raise ValueError("training observation is not in --obsnums")
    if not 0.2 <= float(args.training_fraction) <= 0.8:
        raise ValueError("--training-fraction must be between 0.2 and 0.8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tone_path = args.event_vector_dir / "science_event_tone_vectors.csv"
    template_path = (
        args.tone_analysis_dir / "science_tone_rank_one_modes.csv"
    )
    catalog_path = (
        args.continuous_analysis_dir / "continuous_event_catalog.csv"
    )
    continuous_manifest_path = (
        args.continuous_analysis_dir / "manifest.json"
    )
    assignment_path = (
        args.hidden_state_dir / "joint_interval_state_assignments.csv"
    )
    hidden_manifest_path = args.hidden_state_dir / "manifest.json"
    event_tones = pd.read_csv(tone_path)
    fixed_templates = pd.read_csv(template_path)
    catalog = pd.read_csv(catalog_path)
    assignments = pd.read_csv(assignment_path)
    continuous_manifest = json.loads(
        continuous_manifest_path.read_text(encoding="utf-8")
    )
    hidden_manifest = json.loads(
        hidden_manifest_path.read_text(encoding="utf-8")
    )
    if set(event_tones["schema_version"].astype(str)) != {
        EVENT_VECTOR_SCHEMA
    }:
        raise ValueError("event-tone input has the wrong schema")
    if set(fixed_templates["schema_version"].astype(str)) != {
        TEMPLATE_SCHEMA
    }:
        raise ValueError("fixed-template input has the wrong schema")
    if set(catalog["schema_version"].astype(str)) != {CATALOG_SCHEMA}:
        raise ValueError("continuous event catalog has the wrong schema")
    if continuous_manifest["schema_version"] != CATALOG_SCHEMA:
        raise ValueError("continuous manifest has the wrong schema")
    if set(assignments["schema_version"].astype(str)) != {
        HIDDEN_STATE_SCHEMA
    }:
        raise ValueError("hidden-state assignments have the wrong schema")
    if hidden_manifest["schema_version"] != HIDDEN_STATE_SCHEMA:
        raise ValueError("hidden-state manifest has the wrong schema")

    raw_by_observation = _find_raw_files(
        args.data_root,
        networks=networks,
    )
    missing_raw = sorted(set(obsnums) - set(raw_by_observation))
    if missing_raw:
        raise FileNotFoundError(
            f"requested raw observations are unavailable: {missing_raw}"
        )
    binned: dict[int, dict[str, np.ndarray]] = {}
    all_fixed_bin_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for obsnum in obsnums:
        apt_path = args.apt_root / f"apt_{obsnum}_matched.ecsv"
        if not apt_path.is_file():
            raise FileNotFoundError(
                f"obs {obsnum} lacks its exact matched APT: {apt_path}"
            )
        projections: dict[int, Projection] = {}
        for network in networks:
            template = _load_template(
                obsnum=obsnum,
                network=network,
                event_tones=event_tones,
                fixed_templates=fixed_templates,
                event_rich_obsnums=event_rich_obsnums,
            )
            projection = _project_network(
                obsnum=obsnum,
                network=network,
                raw_path=raw_by_observation[obsnum][network],
                apt_path=apt_path,
                template=template,
                step_window_sec=0.20,
                step_guard_sec=0.05,
            )
            projections[network] = projection
            input_rows.append(
                {
                    "obsnum": obsnum,
                    "network": network,
                    "raw": _file_identity(projection.raw_path),
                    "apt": _file_identity(projection.apt_path),
                    "projection_template_source": (
                        projection.template_source
                    ),
                    "projection_template_training_obsnums": list(
                        projection.template_training_obsnums
                    ),
                    "projection_template_tone_count": (
                        projection.template_tone_count
                    ),
                }
            )
        centers, levels, counts, fixed_rows = _fixed_bin_measurements(
            projections,
            networks=networks,
            bin_width_sec=float(args.bin_width_sec),
            minimum_samples_per_network=int(args.minimum_bin_samples),
        )
        binned[obsnum] = {
            "time": centers,
            "levels": levels,
            "counts": counts,
        }
        all_fixed_bin_rows.extend(fixed_rows)
        print(
            f"obs {obsnum}: complete fixed bins={len(centers)} "
            f"duration_min={len(centers) * args.bin_width_sec / 60.0:.2f}"
        )
        del projections
        gc.collect()

    training = binned[int(args.training_obsnum)]
    full_time = training["time"]
    split_time = float(
        full_time[0]
        + float(args.training_fraction) * (full_time[-1] - full_time[0])
    )
    training_mask = full_time < split_time
    held_out_mask = ~training_mask
    if np.count_nonzero(training_mask) < 100:
        raise ValueError("training segment has too few fixed bins")
    training_standardized, training_normalization = _normalize_segment(
        full_time[training_mask],
        training["levels"][training_mask],
    )
    fit = _fit_gaussian_hmm(
        training_standardized,
        n_states=3,
        n_initializations=int(args.hmm_initializations),
        random_seed=int(args.random_seed),
        maximum_iterations=int(args.hmm_maximum_iterations),
        convergence_tolerance=float(args.hmm_convergence_tolerance),
        minimum_occupancy_fraction=float(
            args.minimum_state_occupancy_fraction
        ),
        minimum_center_separation_sigma=float(
            args.minimum_center_separation_sigma
        ),
        sort_coordinate=0,
    )
    if not fit.selection_eligible:
        raise ValueError(
            "training three-state HMM failed eligibility: "
            f"{fit.ineligibility_reason}"
        )
    fit, state_names, canonicalization = _canonicalize_two_mode_fit(
        fit,
        networks=networks,
    )

    parameter_rows = _template_parameter_rows(
        fit=fit,
        state_names=state_names,
        networks=networks,
        training_normalization=training_normalization,
    )
    template_transition_rows = _template_transition_rows(
        fit=fit,
        state_names=state_names,
    )
    normalization_rows: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []
    all_transition_rows: list[dict[str, Any]] = []
    all_match_rows: list[dict[str, Any]] = []
    all_null_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    segment_specs: list[dict[str, Any]] = [
        {
            "segment_id": (
                f"obs{int(args.training_obsnum)}_first_half_training"
            ),
            "obsnum": int(args.training_obsnum),
            "role": "training_in_sample",
            "mask": training_mask,
        },
        {
            "segment_id": (
                f"obs{int(args.training_obsnum)}_second_half_held_out"
            ),
            "obsnum": int(args.training_obsnum),
            "role": "within_observation_held_out",
            "mask": held_out_mask,
        },
    ]
    for obsnum in obsnums:
        if obsnum == int(args.training_obsnum):
            continue
        segment_specs.append(
            {
                "segment_id": f"obs{obsnum}_full_transfer",
                "obsnum": obsnum,
                "role": (
                    "cross_observation_transfer_event_rich"
                    if obsnum in event_rich_obsnums
                    else "cross_observation_quiet_control"
                ),
                "mask": np.ones(len(binned[obsnum]["time"]), dtype=bool),
            }
        )

    for segment_index, spec in enumerate(segment_specs):
        obsnum = int(spec["obsnum"])
        mask = np.asarray(spec["mask"], dtype=bool)
        time = binned[obsnum]["time"][mask]
        levels = binned[obsnum]["levels"][mask]
        if spec["role"] == "training_in_sample":
            policies = [
                {
                    "name": "training_intrinsic_scale",
                    "standardized": training_standardized,
                    "normalization": training_normalization,
                    "scale_source": "training_segment_intrinsic",
                }
            ]
        else:
            frozen_standardized, frozen_normalization = _normalize_segment(
                time,
                levels,
                applied_scales=training_normalization
                .applied_scale_rad_per_rms_loading,
            )
            intrinsic_standardized, intrinsic_normalization = (
                _normalize_segment(
                    time,
                    levels,
                )
            )
            policies = [
                {
                    "name": "frozen_training_scale",
                    "standardized": frozen_standardized,
                    "normalization": frozen_normalization,
                    "scale_source": (
                        f"obs{int(args.training_obsnum)}_first_half_training"
                    ),
                },
                {
                    "name": "target_intrinsic_scale",
                    "standardized": intrinsic_standardized,
                    "normalization": intrinsic_normalization,
                    "scale_source": "target_segment_intrinsic",
                },
            ]
        for policy_index, policy in enumerate(policies):
            segment_id = (
                str(spec["segment_id"])
                if spec["role"] == "training_in_sample"
                else f"{spec['segment_id']}_{policy['name']}"
            )
            normalization = policy["normalization"]
            normalization_rows.extend(
                _normalization_rows(
                    segment_id=segment_id,
                    obsnum=obsnum,
                    networks=networks,
                    normalization=normalization,
                    scale_source=str(policy["scale_source"]),
                )
            )
            outputs = _segment_outputs(
                segment_id=segment_id,
                obsnum=obsnum,
                role=str(spec["role"]),
                scale_policy=str(policy["name"]),
                time_unix_sec=time,
                raw_levels=levels,
                standardized=np.asarray(
                    policy["standardized"],
                    dtype=float,
                ),
                normalization=normalization,
                fit=fit,
                state_names=state_names,
                networks=networks,
                catalog=catalog,
                assignments=assignments,
                match_tolerance_sec=float(args.match_tolerance_sec),
                shift_null_permutations=int(
                    args.shift_null_permutations
                ),
                random_seed=(
                    int(args.random_seed)
                    + 1000 * segment_index
                    + 100 * policy_index
                ),
                bin_width_sec=float(args.bin_width_sec),
            )
            all_bin_rows.extend(outputs["bin_rows"])
            all_transition_rows.extend(outputs["transition_rows"])
            all_match_rows.extend(outputs["match_rows"])
            all_null_rows.extend(outputs["null_rows"])
            summary_rows.append(outputs["summary"])
            row = outputs["summary"]
            print(
                f"{segment_id}: transitions="
                f"{row['predicted_transition_count']} "
                f"catalog={row['catalog_primary_event_count']} "
                f"matched={row['one_to_one_matched_count']} "
                f"recall={row['catalog_recall']:.3f} "
                f"precision={row['prediction_precision']:.3f} "
                f"null_p95={row['shift_null_matched_count_p95']:.1f}"
            )

    model_payload = {
        "schema_version": SCHEMA_VERSION,
        "template_id": (
            f"ngc4449-held-out-two-mode-obs{int(args.training_obsnum)}"
        ),
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "lifecycle_state": "forensic_observe_only",
        "detector_uses_catalog_event_times": False,
        "projection_is_prior_event_mode_informed": True,
        "training": {
            "obsnum": int(args.training_obsnum),
            "fraction": float(args.training_fraction),
            "segment_start_unix_sec": float(
                full_time[training_mask][0]
                - 0.5 * float(args.bin_width_sec)
            ),
            "segment_end_unix_sec": float(
                full_time[training_mask][-1]
                + 0.5 * float(args.bin_width_sec)
            ),
            "bin_width_sec": float(args.bin_width_sec),
            "bin_count": int(np.count_nonzero(training_mask)),
            "method": (
                "fixed-bin medians; unlabeled robust linear detrend and "
                "centering; frozen three-state diagonal-Gaussian HMM"
            ),
        },
        "identity": {
            "networks": networks,
            "network_order": networks,
            "state_names": state_names,
            "projected_phase_unit": (
                "rad per RMS-normalized stable-UID loading"
            ),
        },
        "normalization": {
            "reference_unix_sec": (
                training_normalization.reference_unix_sec
            ),
            "slope_rad_per_rms_loading_per_hour": (
                training_normalization
                .slope_rad_per_rms_loading_per_hour.tolist()
            ),
            "trend_intercept_rad_per_rms_loading": (
                training_normalization
                .trend_intercept_rad_per_rms_loading.tolist()
            ),
            "residual_center_rad_per_rms_loading": (
                training_normalization
                .residual_center_rad_per_rms_loading.tolist()
            ),
            "scale_rad_per_rms_loading": (
                training_normalization
                .applied_scale_rad_per_rms_loading.tolist()
            ),
        },
        "hmm": {
            "means_standardized": fit.means.tolist(),
            "variances_standardized_squared": fit.variances.tolist(),
            "transition_probability": fit.transition.tolist(),
            "initial_probability": fit.initial.tolist(),
            "training_log_likelihood": fit.log_likelihood,
            "training_bic_descriptive": fit.bic,
            "training_aic_descriptive": fit.aic,
            "iterations": fit.iterations,
            "converged": fit.converged,
            "minimum_center_separation_sigma": (
                fit.minimum_center_separation_sigma
            ),
            "minimum_posterior_occupancy_fraction": (
                fit.minimum_posterior_occupancy_fraction
            ),
        },
        "canonicalization": canonicalization,
    }

    output_names = {
        "fixed_bins": "fixed_bin_network_levels.csv",
        "normalization": "segment_normalization.csv",
        "template_parameters": "detector_template_state_parameters.csv",
        "template_transitions": "detector_template_transitions.csv",
        "decoded_bins": "decoded_fixed_bins.csv",
        "predicted_transitions": "predicted_state_transitions.csv",
        "catalog_matches": "catalog_event_matches.csv",
        "shift_null": "circular_shift_null.csv",
        "segment_summary": "segment_detection_summary.csv",
        "template_model": "detector_template.json",
        "template_figure": "detector_template_patterns.png",
        "timeline_figure": "decoded_state_timelines.png",
        "performance_figure": "held_out_transfer_performance.png",
    }
    tables: dict[str, list[dict[str, Any]]] = {
        "fixed_bins": all_fixed_bin_rows,
        "normalization": normalization_rows,
        "template_parameters": parameter_rows,
        "template_transitions": template_transition_rows,
        "decoded_bins": all_bin_rows,
        "predicted_transitions": all_transition_rows,
        "catalog_matches": all_match_rows,
        "shift_null": all_null_rows,
        "segment_summary": summary_rows,
    }
    for key, rows in tables.items():
        _write_csv(args.output_dir / output_names[key], rows)
    template_json = json.dumps(model_payload, indent=2) + "\n"
    (
        args.output_dir / output_names["template_model"]
    ).write_text(template_json, encoding="utf-8")
    _template_figure(
        args.output_dir / output_names["template_figure"],
        parameter_rows,
        networks=networks,
    )
    _timeline_figure(
        args.output_dir / output_names["timeline_figure"],
        all_bin_rows,
        all_match_rows,
        summary_rows,
    )
    _performance_figure(
        args.output_dir / output_names["performance_figure"],
        summary_rows,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "description": (
            "Held-out and cross-observation transfer of fixed-bin raw-I/Q "
            "network response modes"
        ),
        "semantics": {
            "detection_event_time_dependency": (
                "none; catalog events are not supplied to training or "
                "decoding and are used only afterward for validation"
            ),
            "projection_dependency": (
                "the stable-UID network projection was learned from the "
                "prior science event corpus; detection is event-time "
                "independent but mode-template informed"
            ),
            "training": (
                "first half of the configured training observation; fixed "
                "0.5 s bins; no catalog boundaries or event labels"
            ),
            "target_preprocessing": (
                "target-local unlabeled linear detrend and residual median; "
                "paired inference uses either frozen training scales to "
                "preserve absolute severity or target-intrinsic scales to "
                "test response-pattern transfer; both are scored and quiet "
                "controls adjudicate shape-normalized false positives"
            ),
            "state_identity": (
                "phenomenological response pattern; not a physical hardware "
                "state"
            ),
            "catalog_matching": (
                "sorted one-to-one maximum-cardinality timing match followed "
                "by minimum residual; circularly shifted predicted paths "
                "form the timing null"
            ),
        },
        "parameters": {
            "obsnums": obsnums,
            "networks": networks,
            "event_rich_obsnums": event_rich_obsnums,
            "training_obsnum": int(args.training_obsnum),
            "training_fraction": float(args.training_fraction),
            "bin_width_sec": float(args.bin_width_sec),
            "minimum_bin_samples": int(args.minimum_bin_samples),
            "hmm_initializations": int(args.hmm_initializations),
            "hmm_maximum_iterations": int(args.hmm_maximum_iterations),
            "hmm_convergence_tolerance": float(
                args.hmm_convergence_tolerance
            ),
            "minimum_state_occupancy_fraction": float(
                args.minimum_state_occupancy_fraction
            ),
            "minimum_center_separation_sigma": float(
                args.minimum_center_separation_sigma
            ),
            "match_tolerance_sec": float(args.match_tolerance_sec),
            "shift_null_permutations": int(
                args.shift_null_permutations
            ),
            "random_seed": int(args.random_seed),
        },
        "inputs": {
            "event_tones_projection_training": _file_identity(tone_path),
            "fixed_projection_templates": _file_identity(template_path),
            "continuous_catalog_validation_only": _file_identity(
                catalog_path
            ),
            "continuous_manifest": _file_identity(
                continuous_manifest_path
            ),
            "event_conditioned_assignments_validation_only": _file_identity(
                assignment_path
            ),
            "hidden_state_manifest": _file_identity(hidden_manifest_path),
            "raw_and_apt_files": input_rows,
        },
        "counts": {key: len(rows) for key, rows in tables.items()},
        "outputs": output_names,
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        payload.encode()
    ).hexdigest()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
