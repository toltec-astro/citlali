#!/usr/bin/env python3
"""Infer bounded hidden states in the learned raw-I/Q projection.

This diagnostic consumes the full-duration cross-rack event catalog and
reconstructs the same stable-UID projected phase from raw I/Q.  Primary event
times define candidate change-point intervals.  Robust interval medians, not
the 122 Hz samples, are the emissions of one-, two-, and three-state Gaussian
hidden Markov models.

The model is deliberately forensic:

* detector identity requires the exact observation-specific matched APT;
* event-rich observations retain the leave-one-observation-out UID template;
* a robust linear trend is separated from the interval-level state model;
* model selection uses a recorded parsimonious BIC rule plus minimum
  occupancy and center-separation gates;
* state labels are observation-local ordinal levels, never detector IDs or
  calibrated physical states; and
* catalog boundaries are supplied to the state model, so this is not an
  independent event-rate measurement.

The outputs measure state levels, dwell runs, transition direction,
event/state agreement, and cross-network correspondence without assuming
exponential recovery.
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
    str(Path(tempfile.gettempdir()) / "citlali-iq-state-mpl-cache"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import special, stats  # noqa: E402
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


SCHEMA_VERSION = "citlali-science-iq-hidden-state-v1"
CATALOG_SCHEMA = "citlali-science-iq-continuous-event-morphology-v1"
DEFAULT_SCIENCE_OBSNUMS = (152390, 152392, 152419, 152431, 152433)


@dataclass
class HmmFit:
    """Diagonal-Gaussian HMM result on interval-level emissions."""

    n_states: int
    means: np.ndarray
    variances: np.ndarray
    transition: np.ndarray
    initial: np.ndarray
    posterior: np.ndarray
    decoded: np.ndarray
    log_likelihood: float
    bic: float
    aic: float
    parameter_count: int
    iterations: int
    converged: bool
    minimum_center_separation_sigma: float
    minimum_posterior_occupancy_fraction: float
    selection_eligible: bool
    ineligibility_reason: str | None


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


def _log_emission(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if values.ndim != 2:
        raise ValueError("HMM emissions must be a time-by-coordinate matrix")
    if means.ndim != 2 or variances.shape != means.shape:
        raise ValueError("HMM means/variances have incompatible shapes")
    if values.shape[1] != means.shape[1]:
        raise ValueError("HMM coordinate count differs from emissions")
    if np.any(~np.isfinite(means)) or np.any(~np.isfinite(variances)):
        raise ValueError("HMM parameters must be finite")
    if np.any(variances <= 0.0):
        raise ValueError("HMM variances must be positive")
    finite = np.isfinite(values)
    difference = values[:, None, :] - means[None, :, :]
    term = np.log(2.0 * np.pi * variances)[None, :, :] + (
        difference**2 / variances[None, :, :]
    )
    term = np.where(finite[:, None, :], term, 0.0)
    return -0.5 * np.sum(term, axis=2)


def _forward_backward(
    log_emission: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    log_emission = np.asarray(log_emission, dtype=float)
    initial = np.asarray(initial, dtype=float)
    transition = np.asarray(transition, dtype=float)
    n_time, n_states = log_emission.shape
    if initial.shape != (n_states,):
        raise ValueError("initial HMM probability has the wrong shape")
    if transition.shape != (n_states, n_states):
        raise ValueError("transition matrix has the wrong shape")
    if np.any(initial <= 0.0) or np.any(transition <= 0.0):
        raise ValueError("HMM probabilities must be strictly positive")
    log_initial = np.log(initial)
    log_transition = np.log(transition)
    alpha = np.empty((n_time, n_states), dtype=float)
    alpha[0] = log_initial + log_emission[0]
    for index in range(1, n_time):
        alpha[index] = log_emission[index] + special.logsumexp(
            alpha[index - 1][:, None] + log_transition,
            axis=0,
        )
    log_likelihood = float(special.logsumexp(alpha[-1]))
    beta = np.zeros((n_time, n_states), dtype=float)
    for index in range(n_time - 2, -1, -1):
        beta[index] = special.logsumexp(
            log_transition
            + log_emission[index + 1][None, :]
            + beta[index + 1][None, :],
            axis=1,
        )
    log_gamma = alpha + beta - log_likelihood
    posterior = np.exp(log_gamma)
    posterior /= np.sum(posterior, axis=1, keepdims=True)
    transition_sum = np.zeros((n_states, n_states), dtype=float)
    for index in range(n_time - 1):
        log_xi = (
            alpha[index][:, None]
            + log_transition
            + log_emission[index + 1][None, :]
            + beta[index + 1][None, :]
            - log_likelihood
        )
        transition_sum += np.exp(log_xi)
    return posterior, transition_sum, log_likelihood


def _viterbi(
    log_emission: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    log_emission = np.asarray(log_emission, dtype=float)
    n_time, n_states = log_emission.shape
    log_transition = np.log(np.asarray(transition, dtype=float))
    score = np.empty((n_time, n_states), dtype=float)
    back = np.zeros((n_time, n_states), dtype=int)
    score[0] = np.log(np.asarray(initial, dtype=float)) + log_emission[0]
    for index in range(1, n_time):
        candidate = score[index - 1][:, None] + log_transition
        back[index] = np.argmax(candidate, axis=0)
        score[index] = (
            candidate[back[index], np.arange(n_states)]
            + log_emission[index]
        )
    decoded = np.empty(n_time, dtype=int)
    decoded[-1] = int(np.argmax(score[-1]))
    for index in range(n_time - 2, -1, -1):
        decoded[index] = back[index + 1, decoded[index + 1]]
    return decoded


def _initial_hmm_parameters(
    values: np.ndarray,
    *,
    n_states: int,
    random_seed: int,
    variance_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n_time, n_coordinate = values.shape
    imputed = values.copy()
    for coordinate in range(n_coordinate):
        finite = np.isfinite(imputed[:, coordinate])
        if not np.any(finite):
            raise ValueError("HMM coordinate has no finite interval levels")
        imputed[~finite, coordinate] = np.median(
            imputed[finite, coordinate]
        )
    if n_states == 1:
        labels = np.zeros(n_time, dtype=int)
    else:
        center = np.median(imputed, axis=0)
        scale = np.array(
            [_robust_sigma(imputed[:, index]) for index in range(n_coordinate)],
            dtype=float,
        )
        fallback_scale = np.std(imputed, axis=0)
        scale = np.where(
            np.isfinite(scale) & (scale > 0.0),
            scale,
            fallback_scale,
        )
        scale = np.where(
            np.isfinite(scale) & (scale > 0.0),
            scale,
            1.0,
        )
        standardized = (imputed - center) / scale
        generator = np.random.default_rng(random_seed)
        cluster_centers = np.empty(
            (n_states, n_coordinate),
            dtype=float,
        )
        cluster_centers[0] = standardized[
            int(generator.integers(0, n_time))
        ]
        nearest_distance_squared = np.sum(
            (standardized - cluster_centers[0]) ** 2,
            axis=1,
        )
        for state in range(1, n_states):
            total = float(np.sum(nearest_distance_squared))
            if not np.isfinite(total) or total <= 0.0:
                selected_index = int(
                    generator.integers(0, n_time)
                )
            else:
                selected_index = int(
                    generator.choice(
                        n_time,
                        p=nearest_distance_squared / total,
                    )
                )
            cluster_centers[state] = standardized[selected_index]
            distance_squared = np.sum(
                (standardized - cluster_centers[state]) ** 2,
                axis=1,
            )
            nearest_distance_squared = np.minimum(
                nearest_distance_squared,
                distance_squared,
            )
        labels = np.zeros(n_time, dtype=int)
        for _ in range(25):
            distance_squared = np.sum(
                (
                    standardized[:, None, :]
                    - cluster_centers[None, :, :]
                )
                ** 2,
                axis=2,
            )
            new_labels = np.argmin(distance_squared, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for state in range(n_states):
                selected = labels == state
                if np.any(selected):
                    cluster_centers[state] = np.mean(
                        standardized[selected],
                        axis=0,
                    )
                else:
                    nearest = np.min(distance_squared, axis=1)
                    cluster_centers[state] = standardized[
                        int(np.argmax(nearest))
                    ]
    means = np.empty((n_states, n_coordinate), dtype=float)
    variances = np.empty_like(means)
    global_variance = np.var(imputed, axis=0)
    global_variance = np.maximum(global_variance, variance_floor)
    for state in range(n_states):
        selected = labels == state
        if not np.any(selected):
            means[state] = np.quantile(
                imputed,
                (state + 0.5) / n_states,
                axis=0,
            )
            variances[state] = global_variance
        else:
            means[state] = np.mean(imputed[selected], axis=0)
            variances[state] = np.maximum(
                np.var(imputed[selected], axis=0),
                variance_floor,
            )
    transition = np.full((n_states, n_states), 0.25, dtype=float)
    for before, after in zip(labels[:-1], labels[1:], strict=True):
        transition[before, after] += 1.0
    transition[np.diag_indices(n_states)] += 1.0
    transition /= np.sum(transition, axis=1, keepdims=True)
    initial = np.full(n_states, 0.1, dtype=float)
    initial[labels[0]] += 1.0
    initial /= np.sum(initial)
    return means, variances, transition, initial


def _minimum_center_separation(
    means: np.ndarray,
    variances: np.ndarray,
) -> float:
    if len(means) <= 1:
        return math.inf
    separations: list[float] = []
    for first in range(len(means)):
        for second in range(first + 1, len(means)):
            pooled = 0.5 * (variances[first] + variances[second])
            distance = math.sqrt(
                float(np.sum((means[first] - means[second]) ** 2 / pooled))
            )
            separations.append(distance)
    return float(min(separations))


def _sort_hmm_states(
    fit: HmmFit,
    *,
    sort_coordinate: int,
) -> HmmFit:
    order = np.argsort(fit.means[:, sort_coordinate])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return HmmFit(
        n_states=fit.n_states,
        means=fit.means[order],
        variances=fit.variances[order],
        transition=fit.transition[np.ix_(order, order)],
        initial=fit.initial[order],
        posterior=fit.posterior[:, order],
        decoded=inverse[fit.decoded],
        log_likelihood=fit.log_likelihood,
        bic=fit.bic,
        aic=fit.aic,
        parameter_count=fit.parameter_count,
        iterations=fit.iterations,
        converged=fit.converged,
        minimum_center_separation_sigma=fit.minimum_center_separation_sigma,
        minimum_posterior_occupancy_fraction=(
            fit.minimum_posterior_occupancy_fraction
        ),
        selection_eligible=fit.selection_eligible,
        ineligibility_reason=fit.ineligibility_reason,
    )


def _fit_gaussian_hmm(
    values: np.ndarray,
    *,
    n_states: int,
    n_initializations: int,
    random_seed: int,
    maximum_iterations: int,
    convergence_tolerance: float,
    minimum_occupancy_fraction: float,
    minimum_center_separation_sigma: float,
    sort_coordinate: int = 0,
) -> HmmFit:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("HMM requires at least two interval emissions")
    if not np.all(np.any(np.isfinite(values), axis=0)):
        raise ValueError("every HMM coordinate requires finite emissions")
    global_variance = np.nanvar(values, axis=0)
    variance_floor = np.maximum(global_variance * 1.0e-4, 1.0e-10)
    best: HmmFit | None = None
    for initialization in range(int(n_initializations)):
        means, variances, transition, initial = _initial_hmm_parameters(
            values,
            n_states=n_states,
            random_seed=random_seed + initialization,
            variance_floor=variance_floor,
        )
        previous = -math.inf
        converged = False
        iteration_count = 0
        for iteration_count in range(1, int(maximum_iterations) + 1):
            emission = _log_emission(values, means, variances)
            posterior, transition_sum, log_likelihood = _forward_backward(
                emission,
                initial,
                transition,
            )
            initial = posterior[0] + 0.05
            initial /= np.sum(initial)
            transition = transition_sum + 0.05
            transition[np.diag_indices(n_states)] += 0.25
            transition /= np.sum(transition, axis=1, keepdims=True)
            for state in range(n_states):
                for coordinate in range(values.shape[1]):
                    finite = np.isfinite(values[:, coordinate])
                    weight = posterior[finite, state]
                    denominator = float(np.sum(weight))
                    if denominator <= 1.0e-8:
                        continue
                    coordinate_values = values[finite, coordinate]
                    mean = float(
                        np.sum(weight * coordinate_values) / denominator
                    )
                    variance = float(
                        np.sum(weight * (coordinate_values - mean) ** 2)
                        / denominator
                    )
                    means[state, coordinate] = mean
                    variances[state, coordinate] = max(
                        variance,
                        float(variance_floor[coordinate]),
                    )
            if (
                np.isfinite(previous)
                and abs(log_likelihood - previous)
                <= convergence_tolerance
                * max(1.0, abs(previous))
            ):
                converged = True
                break
            previous = log_likelihood
        emission = _log_emission(values, means, variances)
        posterior, _, log_likelihood = _forward_backward(
            emission,
            initial,
            transition,
        )
        decoded = _viterbi(emission, initial, transition)
        parameter_count = int(
            2 * n_states * values.shape[1]
            + n_states * (n_states - 1)
            + (n_states - 1)
        )
        sample_count = int(np.count_nonzero(np.any(np.isfinite(values), axis=1)))
        bic = float(
            -2.0 * log_likelihood
            + parameter_count * math.log(max(2, sample_count))
        )
        aic = float(-2.0 * log_likelihood + 2.0 * parameter_count)
        separation = _minimum_center_separation(means, variances)
        occupancy = np.mean(posterior, axis=0)
        minimum_occupancy = float(np.min(occupancy))
        eligible = (
            converged
            and minimum_occupancy >= minimum_occupancy_fraction
            and (
                n_states == 1
                or separation >= minimum_center_separation_sigma
            )
        )
        reasons: list[str] = []
        if not converged:
            reasons.append("not_converged")
        if minimum_occupancy < minimum_occupancy_fraction:
            reasons.append("state_occupancy_below_minimum")
        if (
            n_states > 1
            and separation < minimum_center_separation_sigma
        ):
            reasons.append("state_centers_not_separated")
        candidate = HmmFit(
            n_states=n_states,
            means=means.copy(),
            variances=variances.copy(),
            transition=transition.copy(),
            initial=initial.copy(),
            posterior=posterior.copy(),
            decoded=decoded.copy(),
            log_likelihood=log_likelihood,
            bic=bic,
            aic=aic,
            parameter_count=parameter_count,
            iterations=iteration_count,
            converged=converged,
            minimum_center_separation_sigma=separation,
            minimum_posterior_occupancy_fraction=minimum_occupancy,
            selection_eligible=eligible,
            ineligibility_reason=";".join(reasons) if reasons else None,
        )
        if best is None or candidate.log_likelihood > best.log_likelihood:
            best = candidate
    if best is None:
        raise RuntimeError("HMM initialization produced no candidate")
    return _sort_hmm_states(best, sort_coordinate=sort_coordinate)


def _select_hmm_model(
    fits: list[HmmFit],
    *,
    bic_parsimony_tolerance: float,
) -> HmmFit:
    eligible = [fit for fit in fits if fit.selection_eligible]
    if not eligible:
        raise ValueError("no HMM candidate passed the selection gates")
    minimum_bic = min(fit.bic for fit in eligible)
    acceptable = [
        fit
        for fit in eligible
        if fit.bic <= minimum_bic + bic_parsimony_tolerance
    ]
    return min(acceptable, key=lambda fit: fit.n_states)


def _robust_linear_detrend(
    time_unix_sec: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    time = np.asarray(time_unix_sec, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(time) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        reference = _finite_median(time)
        intercept = _finite_median(values)
        return (
            np.full_like(values, np.nan),
            0.0,
            float(intercept if intercept is not None else math.nan),
            float(reference if reference is not None else math.nan),
        )
    reference = float(np.median(time[finite]))
    relative_hour = (time[finite] - reference) / 3600.0
    if np.count_nonzero(finite) >= 3:
        slope, intercept, _, _ = stats.theilslopes(
            values[finite],
            relative_hour,
        )
    else:
        slope, intercept = np.polyfit(
            relative_hour,
            values[finite],
            deg=1,
        )
    detrended = np.full_like(values, np.nan)
    detrended[finite] = values[finite] - (
        float(intercept) + float(slope) * relative_hour
    )
    return detrended, float(slope), float(intercept), reference


def _interval_measurements(
    *,
    obsnum: int,
    projections: dict[int, Projection],
    event_rows: list[dict[str, Any]],
    transition_guard_sec: float,
    minimum_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not projections:
        raise ValueError("interval measurement requires projections")
    start = max(
        float(projection.time_unix_sec[0])
        for projection in projections.values()
    )
    end = min(
        float(projection.time_unix_sec[-1])
        for projection in projections.values()
    )
    primary = [
        row
        for row in event_rows
        if bool(row["primary_event_candidate"])
        and start < float(row["refined_event_time_unix_sec"]) < end
    ]
    primary.sort(key=lambda row: row["refined_event_time_unix_sec"])
    boundaries = np.asarray(
        [
            start,
            *[
                float(row["refined_event_time_unix_sec"])
                for row in primary
            ],
            end,
        ],
        dtype=float,
    )
    interval_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    for interval_index, (lower, upper) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        duration = float(upper - lower)
        guard = min(float(transition_guard_sec), 0.25 * duration)
        interval_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "interval_index_zero_based": interval_index,
                "interval_start_unix_sec": float(lower),
                "interval_end_unix_sec": float(upper),
                "interval_midpoint_unix_sec": float(0.5 * (lower + upper)),
                "interval_duration_sec": duration,
                "effective_transition_guard_sec": guard,
                "left_censored_by_observation_start": interval_index == 0,
                "right_censored_by_observation_end": (
                    interval_index == len(boundaries) - 2
                ),
                "preceding_event_id": (
                    primary[interval_index - 1]["event_id"]
                    if interval_index > 0
                    else None
                ),
                "following_event_id": (
                    primary[interval_index]["event_id"]
                    if interval_index < len(primary)
                    else None
                ),
            }
        )
        for network, projection in sorted(projections.items()):
            selected = (
                (projection.time_unix_sec >= lower + guard)
                & (projection.time_unix_sec <= upper - guard)
            )
            sample_count = int(np.count_nonzero(selected))
            if sample_count >= int(minimum_samples):
                time = projection.time_unix_sec[selected]
                value = projection.projected_phase_rad[selected]
                finite = np.isfinite(time) & np.isfinite(value)
                sample_count = int(np.count_nonzero(finite))
            else:
                time = np.asarray([], dtype=float)
                value = np.asarray([], dtype=float)
                finite = np.asarray([], dtype=bool)
            if sample_count >= int(minimum_samples):
                time = time[finite]
                value = value[finite]
                midpoint = float(np.median(time))
                level = float(np.median(value))
                sigma = _robust_sigma(value)
                slope = float(
                    np.polyfit(time - midpoint, value, deg=1)[0]
                )
                status = "valid"
            else:
                level = math.nan
                sigma = math.nan
                slope = math.nan
                status = "insufficient_samples"
            measurement_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "obsnum": obsnum,
                    "interval_index_zero_based": interval_index,
                    "network": network,
                    "array": _array_name(network),
                    "rack": _rack(network),
                    "interval_start_unix_sec": float(lower),
                    "interval_end_unix_sec": float(upper),
                    "interval_midpoint_unix_sec": float(
                        0.5 * (lower + upper)
                    ),
                    "interval_duration_sec": duration,
                    "effective_transition_guard_sec": guard,
                    "sample_count": sample_count,
                    "measurement_status": status,
                    "projected_phase_level_rad_per_rms_loading": (
                        _finite_or_none(level)
                    ),
                    "within_interval_projected_phase_sigma_rad": (
                        _finite_or_none(sigma)
                    ),
                    "within_interval_linear_slope_rad_per_sec": (
                        _finite_or_none(slope)
                    ),
                }
            )
    return interval_rows, measurement_rows


def _dwell_run_rows(
    *,
    obsnum: int,
    model_scope: str,
    network: int | None,
    interval_rows: list[dict[str, Any]],
    decoded: np.ndarray,
    posterior: np.ndarray,
) -> list[dict[str, Any]]:
    decoded = np.asarray(decoded, dtype=int)
    if len(interval_rows) != len(decoded):
        raise ValueError("state sequence and interval count differ")
    boundaries = np.concatenate(
        (
            np.asarray([0]),
            np.flatnonzero(decoded[1:] != decoded[:-1]) + 1,
            np.asarray([len(decoded)]),
        )
    )
    rows: list[dict[str, Any]] = []
    for run_index, (first, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        last = int(stop - 1)
        state = int(decoded[first])
        start = float(interval_rows[first]["interval_start_unix_sec"])
        end = float(interval_rows[last]["interval_end_unix_sec"])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "model_scope": model_scope,
                "network": network,
                "run_index_zero_based": run_index,
                "state_ordinal_low_to_high": state,
                "first_interval_index_zero_based": int(first),
                "last_interval_index_zero_based": last,
                "interval_count": int(stop - first),
                "run_start_unix_sec": start,
                "run_end_unix_sec": end,
                "dwell_duration_sec": float(end - start),
                "left_censored_by_observation_start": int(first) == 0,
                "right_censored_by_observation_end": last == len(decoded) - 1,
                "minimum_decoded_state_posterior": float(
                    np.min(posterior[first:stop, state])
                ),
                "median_decoded_state_posterior": float(
                    np.median(posterior[first:stop, state])
                ),
            }
        )
    return rows


def _candidate_state_counts(
    *,
    interval_count: int,
    maximum_states: int,
    minimum_model_intervals: int,
) -> list[int]:
    if interval_count < int(minimum_model_intervals):
        return [1]
    return list(range(1, min(int(maximum_states), interval_count // 5) + 1))


def _fit_candidate_models(
    values: np.ndarray,
    *,
    candidate_counts: list[int],
    args: argparse.Namespace,
    random_seed: int,
    sort_coordinate: int = 0,
) -> tuple[list[HmmFit], HmmFit]:
    fits = [
        _fit_gaussian_hmm(
            values,
            n_states=count,
            n_initializations=int(args.hmm_initializations),
            random_seed=random_seed + 1000 * count,
            maximum_iterations=int(args.hmm_maximum_iterations),
            convergence_tolerance=float(args.hmm_convergence_tolerance),
            minimum_occupancy_fraction=float(
                args.minimum_state_occupancy_fraction
            ),
            minimum_center_separation_sigma=float(
                args.minimum_center_separation_sigma
            ),
            sort_coordinate=sort_coordinate,
        )
        for count in candidate_counts
    ]
    selected = _select_hmm_model(
        fits,
        bic_parsimony_tolerance=float(args.bic_parsimony_tolerance),
    )
    return fits, selected


def _model_comparison_rows(
    *,
    obsnum: int,
    model_scope: str,
    network: int | None,
    interval_count: int,
    fits: list[HmmFit],
    selected: HmmFit,
) -> list[dict[str, Any]]:
    minimum_bic = min(fit.bic for fit in fits)
    bic_by_count = {fit.n_states: fit.bic for fit in fits}
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "obsnum": obsnum,
            "model_scope": model_scope,
            "network": network,
            "interval_count": interval_count,
            "n_states": fit.n_states,
            "log_likelihood": fit.log_likelihood,
            "parameter_count": fit.parameter_count,
            "aic": fit.aic,
            "bic": fit.bic,
            "delta_bic_from_best_candidate": float(fit.bic - minimum_bic),
            "delta_bic_from_one_state": (
                float(fit.bic - bic_by_count[1])
                if 1 in bic_by_count
                else None
            ),
            "candidate_is_bic_minimum": fit.bic == minimum_bic,
            "candidate_is_selected": fit.n_states == selected.n_states,
            "converged": fit.converged,
            "em_iterations": fit.iterations,
            "minimum_center_separation_sigma": _finite_or_none(
                fit.minimum_center_separation_sigma
            ),
            "minimum_posterior_occupancy_fraction": (
                fit.minimum_posterior_occupancy_fraction
            ),
            "selection_eligible": fit.selection_eligible,
            "ineligibility_reason": fit.ineligibility_reason,
        }
        for fit in fits
    ]


def _transition_matrix_rows(
    *,
    obsnum: int,
    model_scope: str,
    network: int | None,
    fit: HmmFit,
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
                    "obsnum": obsnum,
                    "model_scope": model_scope,
                    "network": network,
                    "selected_n_states": fit.n_states,
                    "state_before_ordinal_low_to_high": before,
                    "state_after_ordinal_low_to_high": after,
                    "fitted_transition_probability": float(
                        fit.transition[before, after]
                    ),
                    "decoded_transition_count": int(
                        decoded_count[before, after]
                    ),
                    "decoded_transition_fraction_from_before": (
                        float(decoded_count[before, after] / total)
                        if total
                        else None
                    ),
                }
            )
    return rows


def _network_state_outputs(
    *,
    obsnum: int,
    network: int,
    interval_rows: list[dict[str, Any]],
    measurement_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        [
            row
            for row in measurement_rows
            if int(row["network"]) == network
        ]
    ).sort_values("interval_index_zero_based")
    if len(frame) != len(interval_rows):
        raise ValueError(
            f"obs {obsnum} nw{network}: interval measurement count differs"
        )
    time = frame["interval_midpoint_unix_sec"].to_numpy(dtype=float)
    level = frame[
        "projected_phase_level_rad_per_rms_loading"
    ].to_numpy(dtype=float)
    detrended, slope, intercept, reference = _robust_linear_detrend(
        time,
        level,
    )
    finite = np.isfinite(detrended)
    if not np.all(finite):
        raise ValueError(
            f"obs {obsnum} nw{network}: state intervals are incomplete"
        )
    counts = _candidate_state_counts(
        interval_count=len(detrended),
        maximum_states=int(args.maximum_states),
        minimum_model_intervals=int(args.minimum_model_intervals),
    )
    fits, selected = _fit_candidate_models(
        detrended,
        candidate_counts=counts,
        args=args,
        random_seed=int(args.random_seed) + 100 * obsnum + network,
    )
    duration = frame["interval_duration_sec"].to_numpy(dtype=float)
    total_duration = float(np.sum(duration))
    parameter_rows: list[dict[str, Any]] = []
    for state in range(selected.n_states):
        decoded = selected.decoded == state
        parameter_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "network": network,
                "array": _array_name(network),
                "rack": _rack(network),
                "selected_n_states": selected.n_states,
                "state_ordinal_low_to_high": state,
                "detrended_state_center_rad_per_rms_loading": float(
                    selected.means[state, 0]
                ),
                "emission_sigma_rad_per_rms_loading": float(
                    math.sqrt(selected.variances[state, 0])
                ),
                "posterior_interval_occupancy_fraction": float(
                    np.mean(selected.posterior[:, state])
                ),
                "decoded_interval_occupancy_fraction": float(
                    np.mean(decoded)
                ),
                "decoded_time_occupancy_fraction": float(
                    np.sum(duration[decoded]) / total_duration
                ),
                "linear_trend_slope_rad_per_rms_loading_per_hour": slope,
                "linear_trend_intercept_rad_per_rms_loading": intercept,
                "linear_trend_reference_unix_sec": reference,
            }
        )
    assignment_rows: list[dict[str, Any]] = []
    for index, (interval, measurement) in enumerate(
        zip(interval_rows, frame.to_dict("records"), strict=True)
    ):
        state = int(selected.decoded[index])
        assignment_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "network": network,
                "interval_index_zero_based": index,
                "interval_start_unix_sec": interval[
                    "interval_start_unix_sec"
                ],
                "interval_end_unix_sec": interval["interval_end_unix_sec"],
                "interval_duration_sec": interval["interval_duration_sec"],
                "projected_phase_level_rad_per_rms_loading": measurement[
                    "projected_phase_level_rad_per_rms_loading"
                ],
                "detrended_projected_phase_level_rad_per_rms_loading": float(
                    detrended[index]
                ),
                "selected_n_states": selected.n_states,
                "decoded_state_ordinal_low_to_high": state,
                "decoded_state_posterior": float(
                    selected.posterior[index, state]
                ),
                "maximum_state_posterior": float(
                    np.max(selected.posterior[index])
                ),
            }
        )
    return {
        "fits": fits,
        "selected": selected,
        "comparison_rows": _model_comparison_rows(
            obsnum=obsnum,
            model_scope="network",
            network=network,
            interval_count=len(interval_rows),
            fits=fits,
            selected=selected,
        ),
        "parameter_rows": parameter_rows,
        "assignment_rows": assignment_rows,
        "dwell_rows": _dwell_run_rows(
            obsnum=obsnum,
            model_scope="network",
            network=network,
            interval_rows=interval_rows,
            decoded=selected.decoded,
            posterior=selected.posterior,
        ),
        "transition_rows": _transition_matrix_rows(
            obsnum=obsnum,
            model_scope="network",
            network=network,
            fit=selected,
        ),
        "detrended": detrended,
        "trend_slope": slope,
        "trend_intercept": intercept,
        "trend_reference": reference,
    }


def _network_event_state_rows(
    *,
    obsnum: int,
    network: int,
    event_rows: list[dict[str, Any]],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    primary = [
        row for row in event_rows if bool(row["primary_event_candidate"])
    ]
    primary.sort(key=lambda row: row["refined_event_time_unix_sec"])
    fit: HmmFit = model["selected"]
    detrended = np.asarray(model["detrended"], dtype=float)
    if len(primary) + 1 != len(fit.decoded):
        raise ValueError(
            f"obs {obsnum}: event count does not define state intervals"
        )
    sign_value = {"positive": 1, "negative": -1}
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(primary):
        before = int(fit.decoded[index])
        after = int(fit.decoded[index + 1])
        changed = before != after
        center_change = float(
            fit.means[after, 0] - fit.means[before, 0]
        )
        direction = int(np.sign(center_change)) if changed else 0
        catalog_direction = sign_value.get(
            str(event["dominant_projected_step_sign"]),
            0,
        )
        participating = network in {
            int(value) for value in str(event["networks"]).split()
        }
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": event["event_id"],
                "obsnum": obsnum,
                "network": network,
                "network_participated_in_catalog_event": participating,
                "event_time_unix_sec": event[
                    "refined_event_time_unix_sec"
                ],
                "catalog_dominant_projected_step_sign": event[
                    "dominant_projected_step_sign"
                ],
                "catalog_quality_tier": event["quality_tier"],
                "selected_n_states": fit.n_states,
                "state_before_ordinal_low_to_high": before,
                "state_after_ordinal_low_to_high": after,
                "decoded_state_changed": changed,
                "decoded_state_center_change_rad_per_rms_loading": (
                    center_change
                ),
                "decoded_state_change_direction": (
                    "positive"
                    if direction > 0
                    else "negative"
                    if direction < 0
                    else "none"
                ),
                "state_direction_matches_catalog_sign": (
                    bool(direction == catalog_direction)
                    if changed and catalog_direction
                    else None
                ),
                "measured_adjacent_interval_level_change_rad_per_rms_loading": (
                    float(detrended[index + 1] - detrended[index])
                ),
                "pre_interval_state_posterior": float(
                    fit.posterior[index, before]
                ),
                "post_interval_state_posterior": float(
                    fit.posterior[index + 1, after]
                ),
            }
        )
    return rows


def _robust_coordinate_scales(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("coordinate-scale input must be two-dimensional")
    scales = np.asarray(
        [
            _robust_sigma(values[:, index])
            for index in range(values.shape[1])
        ],
        dtype=float,
    )
    fallback = np.nanstd(values, axis=0)
    scales = np.where(
        np.isfinite(scales) & (scales > 0.0),
        scales,
        fallback,
    )
    return np.where(
        np.isfinite(scales) & (scales > 0.0),
        scales,
        1.0,
    )


def _joint_state_outputs(
    *,
    obsnum: int,
    networks: list[int],
    interval_rows: list[dict[str, Any]],
    network_models: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    detrended = np.column_stack(
        [network_models[network]["detrended"] for network in networks]
    )
    scales = _robust_coordinate_scales(detrended)
    standardized = detrended / scales[None, :]
    counts = _candidate_state_counts(
        interval_count=len(interval_rows),
        maximum_states=int(args.maximum_states),
        minimum_model_intervals=int(args.minimum_model_intervals),
    )
    sort_coordinate = (
        networks.index(8) if 8 in networks else 0
    )
    fits, selected = _fit_candidate_models(
        standardized,
        candidate_counts=counts,
        args=args,
        random_seed=int(args.random_seed) + 100 * obsnum + 99,
        sort_coordinate=sort_coordinate,
    )
    duration = np.asarray(
        [row["interval_duration_sec"] for row in interval_rows],
        dtype=float,
    )
    total_duration = float(np.sum(duration))
    parameter_rows: list[dict[str, Any]] = []
    for state in range(selected.n_states):
        decoded = selected.decoded == state
        for coordinate, network in enumerate(networks):
            parameter_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "obsnum": obsnum,
                    "selected_n_states": selected.n_states,
                    "state_ordinal_sorted_by_network8_center": state,
                    "network": network,
                    "array": _array_name(network),
                    "rack": _rack(network),
                    "standardized_state_center": float(
                        selected.means[state, coordinate]
                    ),
                    "detrended_state_center_rad_per_rms_loading": float(
                        selected.means[state, coordinate]
                        * scales[coordinate]
                    ),
                    "standardized_emission_sigma": float(
                        math.sqrt(selected.variances[state, coordinate])
                    ),
                    "emission_sigma_rad_per_rms_loading": float(
                        math.sqrt(selected.variances[state, coordinate])
                        * scales[coordinate]
                    ),
                    "coordinate_robust_scale_rad_per_rms_loading": float(
                        scales[coordinate]
                    ),
                    "posterior_interval_occupancy_fraction": float(
                        np.mean(selected.posterior[:, state])
                    ),
                    "decoded_interval_occupancy_fraction": float(
                        np.mean(decoded)
                    ),
                    "decoded_time_occupancy_fraction": float(
                        np.sum(duration[decoded]) / total_duration
                    ),
                }
            )
    assignment_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "obsnum": obsnum,
            "interval_index_zero_based": index,
            "interval_start_unix_sec": interval[
                "interval_start_unix_sec"
            ],
            "interval_end_unix_sec": interval["interval_end_unix_sec"],
            "interval_duration_sec": interval["interval_duration_sec"],
            "selected_n_states": selected.n_states,
            "decoded_state_ordinal_sorted_by_network8_center": int(
                selected.decoded[index]
            ),
            "decoded_state_posterior": float(
                selected.posterior[index, selected.decoded[index]]
            ),
            "maximum_state_posterior": float(
                np.max(selected.posterior[index])
            ),
        }
        for index, interval in enumerate(interval_rows)
    ]
    return {
        "fits": fits,
        "selected": selected,
        "comparison_rows": _model_comparison_rows(
            obsnum=obsnum,
            model_scope="joint",
            network=None,
            interval_count=len(interval_rows),
            fits=fits,
            selected=selected,
        ),
        "parameter_rows": parameter_rows,
        "assignment_rows": assignment_rows,
        "dwell_rows": _dwell_run_rows(
            obsnum=obsnum,
            model_scope="joint",
            network=None,
            interval_rows=interval_rows,
            decoded=selected.decoded,
            posterior=selected.posterior,
        ),
        "transition_rows": _transition_matrix_rows(
            obsnum=obsnum,
            model_scope="joint",
            network=None,
            fit=selected,
        ),
        "standardized": standardized,
        "scales": scales,
    }


def _joint_event_state_rows(
    *,
    obsnum: int,
    networks: list[int],
    event_rows: list[dict[str, Any]],
    joint_model: dict[str, Any],
) -> list[dict[str, Any]]:
    primary = [
        row for row in event_rows if bool(row["primary_event_candidate"])
    ]
    primary.sort(key=lambda row: row["refined_event_time_unix_sec"])
    fit: HmmFit = joint_model["selected"]
    if len(primary) + 1 != len(fit.decoded):
        raise ValueError("joint state sequence differs from event intervals")
    sign_value = {"positive": 1, "negative": -1}
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(primary):
        before = int(fit.decoded[index])
        after = int(fit.decoded[index + 1])
        changed = before != after
        participating = {
            int(value) for value in str(event["networks"]).split()
        }
        catalog_direction = sign_value.get(
            str(event["dominant_projected_step_sign"]),
            0,
        )
        direction_match: list[bool] = []
        if changed and catalog_direction:
            for coordinate, network in enumerate(networks):
                if network not in participating:
                    continue
                direction = int(
                    np.sign(
                        fit.means[after, coordinate]
                        - fit.means[before, coordinate]
                    )
                )
                direction_match.append(direction == catalog_direction)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": event["event_id"],
                "obsnum": obsnum,
                "event_time_unix_sec": event[
                    "refined_event_time_unix_sec"
                ],
                "catalog_quality_tier": event["quality_tier"],
                "catalog_dominant_projected_step_sign": event[
                    "dominant_projected_step_sign"
                ],
                "catalog_participating_network_count": int(
                    event["network_count"]
                ),
                "selected_joint_n_states": fit.n_states,
                "joint_state_before": before,
                "joint_state_after": after,
                "decoded_joint_state_changed": changed,
                "participating_network_center_direction_match_fraction": (
                    float(np.mean(direction_match))
                    if direction_match
                    else None
                ),
                "all_participating_network_center_directions_match": (
                    bool(all(direction_match))
                    if direction_match
                    else None
                ),
                "pre_interval_joint_state_posterior": float(
                    fit.posterior[index, before]
                ),
                "post_interval_joint_state_posterior": float(
                    fit.posterior[index + 1, after]
                ),
            }
        )
    return rows


def _cross_network_rows(
    *,
    obsnum: int,
    networks: list[int],
    network_models: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(networks):
        first_fit: HmmFit = network_models[first]["selected"]
        first_level = np.asarray(
            network_models[first]["detrended"],
            dtype=float,
        )
        first_change = first_fit.decoded[1:] != first_fit.decoded[:-1]
        for second in networks[first_index + 1 :]:
            second_fit: HmmFit = network_models[second]["selected"]
            second_level = np.asarray(
                network_models[second]["detrended"],
                dtype=float,
            )
            second_change = (
                second_fit.decoded[1:] != second_fit.decoded[:-1]
            )
            union = np.count_nonzero(first_change | second_change)
            intersection = np.count_nonzero(first_change & second_change)
            if (
                np.nanstd(first_level) > 0.0
                and np.nanstd(second_level) > 0.0
            ):
                spearman = stats.spearmanr(first_level, second_level)
                spearman_r = _finite_or_none(spearman.statistic)
                spearman_p = _finite_or_none(spearman.pvalue)
            else:
                spearman_r = None
                spearman_p = None
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "obsnum": obsnum,
                    "network_first": first,
                    "network_second": second,
                    "selected_n_states_first": first_fit.n_states,
                    "selected_n_states_second": second_fit.n_states,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(
                            first_fit.decoded,
                            second_fit.decoded,
                        )
                    ),
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(
                            first_fit.decoded,
                            second_fit.decoded,
                        )
                    ),
                    "detrended_interval_level_spearman_r": spearman_r,
                    "detrended_interval_level_spearman_p_approximate": (
                        spearman_p
                    ),
                    "decoded_change_boundary_jaccard": (
                        float(intersection / union)
                        if union
                        else None
                    ),
                    "decoded_change_boundary_intersection_count": int(
                        intersection
                    ),
                    "decoded_change_boundary_union_count": int(union),
                }
            )
    return rows


def _observation_summary_row(
    *,
    obsnum: int,
    event_rows: list[dict[str, Any]],
    network_models: dict[int, dict[str, Any]],
    network_event_rows: list[dict[str, Any]],
    joint_model: dict[str, Any],
    joint_event_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
    minimum_model_intervals: int,
) -> dict[str, Any]:
    primary = [
        row for row in event_rows if bool(row["primary_event_candidate"])
    ]
    network_event = pd.DataFrame(network_event_rows)
    participating = network_event[
        network_event["network_participated_in_catalog_event"].astype(bool)
    ]
    changed = participating[
        participating["decoded_state_changed"].astype(bool)
    ]
    joint_event = pd.DataFrame(joint_event_rows)
    joint_fit: HmmFit = joint_model["selected"]
    joint_dwell = pd.DataFrame(joint_model["dwell_rows"])
    uncensored = joint_dwell[
        ~joint_dwell["left_censored_by_observation_start"].astype(bool)
        & ~joint_dwell["right_censored_by_observation_end"].astype(bool)
    ]
    joint_bic = {
        fit.n_states: fit.bic for fit in joint_model["fits"]
    }
    interval_count = len(primary) + 1
    model_supported = interval_count >= int(minimum_model_intervals)
    return {
        "schema_version": SCHEMA_VERSION,
        "obsnum": obsnum,
        "primary_catalog_event_count": int(len(primary)),
        "catalog_interval_count": interval_count,
        "state_model_status": (
            "modeled"
            if model_supported
            else "insufficient_catalog_intervals"
        ),
        "minimum_catalog_intervals_for_state_selection": int(
            minimum_model_intervals
        ),
        "network_selected_state_counts": " ".join(
            f"nw{network}:{network_models[network]['selected'].n_states}"
            for network in sorted(network_models)
        ),
        "selected_joint_n_states": joint_fit.n_states,
        "joint_delta_bic_selected_minus_one_state": (
            float(joint_fit.bic - joint_bic[1])
            if model_supported
            else None
        ),
        "joint_minimum_center_separation_sigma": _finite_or_none(
            joint_fit.minimum_center_separation_sigma
        ),
        "joint_catalog_boundary_state_change_fraction": (
            float(joint_event["decoded_joint_state_changed"].mean())
            if model_supported and len(joint_event)
            else None
        ),
        "participating_network_catalog_boundary_state_change_fraction": (
            float(participating["decoded_state_changed"].mean())
            if model_supported and len(participating)
            else None
        ),
        "changed_participating_network_direction_match_fraction": (
            float(
                changed[
                    "state_direction_matches_catalog_sign"
                ].dropna().mean()
            )
            if model_supported and len(changed)
            else None
        ),
        "median_uncensored_joint_state_dwell_sec": (
            float(uncensored["dwell_duration_sec"].median())
            if len(uncensored)
            else None
        ),
        "joint_state_run_count": int(len(joint_dwell)),
        "median_pairwise_adjusted_rand_index": _finite_median(
            row["adjusted_rand_index"] for row in cross_rows
        )
        if model_supported
        else None,
        "median_pairwise_normalized_mutual_information": _finite_median(
            row["normalized_mutual_information"] for row in cross_rows
        )
        if model_supported
        else None,
        "median_pairwise_level_spearman_r": _finite_median(
            row["detrended_interval_level_spearman_r"]
            for row in cross_rows
        )
        if model_supported
        else None,
        "median_pairwise_change_boundary_jaccard": _finite_median(
            row["decoded_change_boundary_jaccard"]
            for row in cross_rows
            if row["decoded_change_boundary_jaccard"] is not None
        )
        if model_supported
        else None,
    }


def _trajectory_example_candidate(
    *,
    obsnum: int,
    networks: list[int],
    projections: dict[int, Projection],
    event_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    network_models: dict[int, dict[str, Any]],
    window_sec: float,
    padding_sec: float,
    output_sample_rate_hz: float,
) -> dict[str, Any] | None:
    primary = [
        row for row in event_rows if bool(row["primary_event_candidate"])
    ]
    primary.sort(key=lambda row: row["refined_event_time_unix_sec"])
    if not primary:
        return None
    event_time = np.asarray(
        [row["refined_event_time_unix_sec"] for row in primary],
        dtype=float,
    )
    counts = np.asarray(
        [
            np.count_nonzero(
                (event_time >= start) & (event_time <= start + window_sec)
            )
            for start in event_time
        ],
        dtype=int,
    )
    window_start = float(event_time[int(np.argmax(counts))])
    window_end = window_start + float(window_sec)
    selected_events = [
        row
        for row in primary
        if window_start
        <= float(row["refined_event_time_unix_sec"])
        <= window_end
    ]
    sample_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for network in networks:
        projection = projections[network]
        model = network_models[network]
        selected_fit: HmmFit = model["selected"]
        scale = float(
            _robust_coordinate_scales(
                np.asarray(model["detrended"], dtype=float)[:, None]
            )[0]
        )
        lower = window_start - padding_sec
        upper = window_end + padding_sec
        selected = (
            (projection.time_unix_sec >= lower)
            & (projection.time_unix_sec <= upper)
        )
        indices = np.flatnonzero(selected)
        decimation = max(
            1,
            int(
                round(
                    projection.sample_frequency_hz
                    / float(output_sample_rate_hz)
                )
            ),
        )
        indices = indices[::decimation]
        time = projection.time_unix_sec[indices]
        trend = (
            model["trend_intercept"]
            + model["trend_slope"]
            * (time - model["trend_reference"])
            / 3600.0
        )
        detrended = projection.projected_phase_rad[indices] - trend
        for absolute_time, raw_value, residual in zip(
            time,
            projection.projected_phase_rad[indices],
            detrended,
            strict=True,
        ):
            sample_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "obsnum": obsnum,
                    "network": network,
                    "time_from_selected_window_start_sec": float(
                        absolute_time - window_start
                    ),
                    "projected_phase_rad_per_rms_loading": float(raw_value),
                    "detrended_projected_phase_rad_per_rms_loading": float(
                        residual
                    ),
                    "standardized_detrended_projected_phase": float(
                        residual / scale
                    ),
                    "coordinate_robust_scale_rad_per_rms_loading": float(
                        scale
                    ),
                }
            )
        for interval_index, interval in enumerate(interval_rows):
            if (
                float(interval["interval_end_unix_sec"]) < lower
                or float(interval["interval_start_unix_sec"]) > upper
            ):
                continue
            state = int(selected_fit.decoded[interval_index])
            state_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "obsnum": obsnum,
                    "network": network,
                    "interval_index_zero_based": interval_index,
                    "interval_start_from_selected_window_sec": float(
                        interval["interval_start_unix_sec"] - window_start
                    ),
                    "interval_end_from_selected_window_sec": float(
                        interval["interval_end_unix_sec"] - window_start
                    ),
                    "decoded_state_ordinal_low_to_high": state,
                    "state_center_standardized_detrended_projected_phase": (
                        float(selected_fit.means[state, 0] / scale)
                    ),
                    "state_posterior": float(
                        selected_fit.posterior[interval_index, state]
                    ),
                }
            )
    marker_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "event_id": row["event_id"],
            "obsnum": obsnum,
            "event_time_from_selected_window_start_sec": float(
                row["refined_event_time_unix_sec"] - window_start
            ),
            "catalog_dominant_projected_step_sign": row[
                "dominant_projected_step_sign"
            ],
            "quality_tier": row["quality_tier"],
            "network_count": int(row["network_count"]),
        }
        for row in selected_events
    ]
    return {
        "obsnum": obsnum,
        "event_count": len(selected_events),
        "window_start_unix_sec": window_start,
        "window_sec": float(window_sec),
        "padding_sec": float(padding_sec),
        "sample_rows": sample_rows,
        "state_rows": state_rows,
        "marker_rows": marker_rows,
    }


def _model_selection_figure(
    path: Path,
    comparison_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    networks: list[int],
) -> None:
    comparison = pd.DataFrame(comparison_rows)
    summary = pd.DataFrame(summary_rows).sort_values("obsnum")
    obsnums = summary["obsnum"].astype(int).tolist()
    labels = [f"nw{network}" for network in networks] + ["joint"]
    selected = np.full((len(obsnums), len(labels)), np.nan, dtype=float)
    improvement = np.full_like(selected, np.nan)
    for row_index, obsnum in enumerate(obsnums):
        status = summary.loc[
            summary["obsnum"] == obsnum,
            "state_model_status",
        ].iloc[0]
        if status != "modeled":
            continue
        for column, network in enumerate(networks):
            rows = comparison[
                (comparison["obsnum"] == obsnum)
                & comparison["model_scope"].eq("network")
                & (comparison["network"] == network)
            ]
            chosen = rows[rows["candidate_is_selected"].astype(bool)]
            selected[row_index, column] = int(chosen.iloc[0]["n_states"])
            improvement[row_index, column] = float(
                chosen.iloc[0]["delta_bic_from_one_state"]
            )
        rows = comparison[
            (comparison["obsnum"] == obsnum)
            & comparison["model_scope"].eq("joint")
        ]
        chosen = rows[rows["candidate_is_selected"].astype(bool)]
        selected[row_index, -1] = int(chosen.iloc[0]["n_states"])
        improvement[row_index, -1] = float(
            chosen.iloc[0]["delta_bic_from_one_state"]
        )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.5),
        constrained_layout=True,
    )
    state_cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    state_cmap.set_bad("#d9d9d9")
    image = axes[0].imshow(
        selected,
        aspect="auto",
        vmin=1,
        vmax=3,
        cmap=state_cmap,
    )
    for row in range(selected.shape[0]):
        for column in range(selected.shape[1]):
            if not np.isfinite(selected[row, column]):
                axes[0].text(
                    column,
                    row,
                    "baseline\nonly",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=7,
                )
                continue
            axes[0].text(
                column,
                row,
                str(int(selected[row, column])),
                ha="center",
                va="center",
                color="white" if selected[row, column] < 2.5 else "black",
            )
    axes[0].set_title(
        "Selected hidden-state count "
        "(sparse observations are baseline-only)"
    )
    axes[0].set_xticks(range(len(labels)), labels)
    axes[0].set_yticks(
        range(len(obsnums)),
        [str(value) for value in obsnums],
    )
    axes[0].set_ylabel("obsnum")
    figure.colorbar(image, ax=axes[0], label="states", ticks=[1, 2, 3])
    finite = np.abs(improvement[np.isfinite(improvement)])
    limit = float(np.quantile(finite, 0.95)) if finite.size else 1.0
    limit = max(limit, 1.0)
    bic_cmap = matplotlib.colormaps.get_cmap("coolwarm").copy()
    bic_cmap.set_bad("#d9d9d9")
    image = axes[1].imshow(
        np.clip(improvement, -limit, limit),
        aspect="auto",
        vmin=-limit,
        vmax=limit,
        cmap=bic_cmap,
    )
    axes[1].set_title(
        "Selected-model BIC minus one-state BIC "
        "(negative favors multiple states)"
    )
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set_yticks(
        range(len(obsnums)),
        [str(value) for value in obsnums],
    )
    axes[1].set_ylabel("obsnum")
    figure.colorbar(image, ax=axes[1], label="ΔBIC")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _cross_network_figure(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    networks: list[int],
) -> None:
    frame = pd.DataFrame(rows)
    stateful = frame[
        (frame["selected_n_states_first"] > 1)
        | (frame["selected_n_states_second"] > 1)
    ]
    obsnums = sorted(int(value) for value in stateful["obsnum"].unique())
    if not obsnums:
        obsnums = sorted(int(value) for value in frame["obsnum"].unique())
    obsnums = obsnums[-3:]
    figure, axes = plt.subplots(
        1,
        len(obsnums),
        figsize=(5.0 * len(obsnums), 4.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, obsnum in zip(axes, obsnums, strict=True):
        matrix = np.full(
            (len(networks), len(networks)),
            np.nan,
            dtype=float,
        )
        np.fill_diagonal(matrix, 1.0)
        selected = frame[frame["obsnum"] == obsnum]
        for row in selected.to_dict("records"):
            first = networks.index(int(row["network_first"]))
            second = networks.index(int(row["network_second"]))
            if (
                int(row["selected_n_states_first"]) <= 1
                or int(row["selected_n_states_second"]) <= 1
            ):
                continue
            value = float(row["adjusted_rand_index"])
            matrix[first, second] = value
            matrix[second, first] = value
        ari_cmap = matplotlib.colormaps.get_cmap("magma").copy()
        ari_cmap.set_bad("#d9d9d9")
        image = axis.imshow(
            matrix,
            vmin=0.0,
            vmax=1.0,
            cmap=ari_cmap,
        )
        axis.set_title(f"{obsnum}: independent-state ARI")
        axis.set_xticks(
            range(len(networks)),
            [f"nw{value}" for value in networks],
            rotation=45,
        )
        axis.set_yticks(
            range(len(networks)),
            [f"nw{value}" for value in networks],
        )
        for row in range(len(networks)):
            for column in range(len(networks)):
                if not np.isfinite(matrix[row, column]):
                    axis.text(
                        column,
                        row,
                        "—",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
                    continue
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix[row, column] < 0.65
                        else "black"
                    ),
                    fontsize=8,
                )
    figure.colorbar(image, ax=axes, label="adjusted Rand index")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _dwell_figure(
    path: Path,
    dwell_rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(dwell_rows)
    selected = frame[
        frame["model_scope"].eq("joint")
        & ~frame["left_censored_by_observation_start"].astype(bool)
        & ~frame["right_censored_by_observation_end"].astype(bool)
    ]
    obsnums = sorted(
        int(value)
        for value in selected.groupby("obsnum").size().nlargest(3).index
    )
    figure, axis = plt.subplots(
        figsize=(8.5, 5.0),
        constrained_layout=True,
    )
    for obsnum in obsnums:
        duration = np.sort(
            selected[selected["obsnum"] == obsnum][
                "dwell_duration_sec"
            ].to_numpy(dtype=float)
        )
        cumulative = np.arange(1, len(duration) + 1) / len(duration)
        axis.step(
            duration,
            cumulative,
            where="post",
            label=f"{obsnum} (n={len(duration)})",
        )
    axis.set_xscale("log")
    axis.set_xlabel("uncensored joint-state dwell duration (s)")
    axis.set_ylabel("empirical cumulative fraction")
    axis.set_title("Joint hidden-state dwell distributions")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _joint_center_pattern_figure(
    path: Path,
    joint_parameter_rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(joint_parameter_rows)
    frame = frame[frame["selected_n_states"] > 1]
    obsnums = sorted(int(value) for value in frame["obsnum"].unique())
    if not obsnums:
        raise ValueError("joint-center figure requires a multistate observation")
    figure, axes = plt.subplots(
        1,
        len(obsnums),
        figsize=(4.6 * len(obsnums), 3.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    maximum = float(
        np.nanmax(np.abs(frame["standardized_state_center"].to_numpy()))
    )
    maximum = max(1.0, maximum)
    image = None
    for axis, obsnum in zip(axes, obsnums, strict=True):
        selected = frame[frame["obsnum"] == obsnum]
        matrix = selected.pivot(
            index="state_ordinal_sorted_by_network8_center",
            columns="network",
            values="standardized_state_center",
        )
        image = axis.imshow(
            matrix.to_numpy(dtype=float),
            vmin=-maximum,
            vmax=maximum,
            aspect="auto",
            cmap="coolwarm",
        )
        axis.set_title(str(obsnum))
        axis.set_xticks(
            range(len(matrix.columns)),
            [f"nw{int(value)}" for value in matrix.columns],
            rotation=45,
        )
        axis.set_yticks(
            range(len(matrix.index)),
            [f"state {int(value)}" for value in matrix.index],
        )
        for row in range(len(matrix.index)):
            for column in range(len(matrix.columns)):
                value = float(matrix.iloc[row, column])
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black" if abs(value) < 0.9 * maximum else "white",
                    fontsize=8,
                )
    if image is None:
        raise RuntimeError("joint-center figure did not render")
    figure.suptitle(
        "Joint-state center patterns by network "
        "(robustly standardized projected phase)"
    )
    figure.colorbar(
        image,
        ax=axes,
        label="state center / network robust scale",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _trajectory_figure(
    path: Path,
    example: dict[str, Any],
    *,
    networks: list[int],
) -> None:
    samples = pd.DataFrame(example["sample_rows"])
    states = pd.DataFrame(example["state_rows"])
    markers = pd.DataFrame(example["marker_rows"])
    figure, axes = plt.subplots(
        len(networks),
        1,
        figsize=(13.0, 2.0 * len(networks)),
        sharex=True,
        constrained_layout=True,
    )
    colors = plt.get_cmap("viridis")
    for axis, network in zip(axes, networks, strict=True):
        network_samples = samples[samples["network"] == network]
        axis.plot(
            network_samples["time_from_selected_window_start_sec"],
            network_samples[
                "standardized_detrended_projected_phase"
            ],
            color="0.25",
            linewidth=0.8,
        )
        network_states = states[states["network"] == network]
        state_count = int(
            network_states[
                "decoded_state_ordinal_low_to_high"
            ].max()
            + 1
        )
        for row in network_states.to_dict("records"):
            state = int(row["decoded_state_ordinal_low_to_high"])
            axis.hlines(
                row[
                    "state_center_standardized_detrended_projected_phase"
                ],
                row["interval_start_from_selected_window_sec"],
                row["interval_end_from_selected_window_sec"],
                color=colors(
                    state / max(1, state_count - 1)
                ),
                linewidth=2.2,
            )
        for row in markers.to_dict("records"):
            axis.axvline(
                row["event_time_from_selected_window_start_sec"],
                color=(
                    "tab:red"
                    if row["catalog_dominant_projected_step_sign"]
                    == "positive"
                    else "tab:blue"
                ),
                alpha=0.25,
                linewidth=0.8,
            )
        axis.set_ylabel(f"nw{network}\nstandardized")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("time from selected window start (s)")
    figure.suptitle(
        f"obs {example['obsnum']}: densest {example['window_sec']:.0f} s "
        f"window, {example['event_count']} catalog events; "
        "colored lines are decoded state centers",
        fontsize=15,
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
    parser.add_argument("--transition-guard-sec", type=float, default=0.35)
    parser.add_argument("--minimum-interval-samples", type=int, default=8)
    parser.add_argument("--minimum-model-intervals", type=int, default=20)
    parser.add_argument("--maximum-states", type=int, default=3)
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
    parser.add_argument(
        "--bic-parsimony-tolerance",
        type=float,
        default=6.0,
    )
    parser.add_argument("--random-seed", type=int, default=20260730)
    parser.add_argument(
        "--trajectory-example-window-sec",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--trajectory-example-padding-sec",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--trajectory-example-sample-rate-hz",
        type=float,
        default=20.0,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    networks = [int(value) for value in args.networks]
    obsnums = [int(value) for value in args.obsnums]
    event_rich_obsnums = [
        int(value) for value in args.event_rich_obsnums
    ]
    if len(networks) != 6:
        raise ValueError("the standard state figures require six networks")
    if not 1 <= int(args.maximum_states) <= 3:
        raise ValueError("--maximum-states must be between one and three")
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
    event_tones = pd.read_csv(tone_path)
    fixed_templates = pd.read_csv(template_path)
    catalog = pd.read_csv(catalog_path)
    continuous_manifest = json.loads(
        continuous_manifest_path.read_text(encoding="utf-8")
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

    raw_by_observation = _find_raw_files(
        args.data_root,
        networks=networks,
    )
    missing_raw = sorted(set(obsnums) - set(raw_by_observation))
    if missing_raw:
        raise FileNotFoundError(
            f"requested raw observations are unavailable: {missing_raw}"
        )
    for obsnum in obsnums:
        apt_path = args.apt_root / f"apt_{obsnum}_matched.ecsv"
        if not apt_path.is_file():
            raise FileNotFoundError(
                f"obs {obsnum} lacks its exact matched APT: {apt_path}"
            )

    all_interval_rows: list[dict[str, Any]] = []
    all_measurement_rows: list[dict[str, Any]] = []
    all_comparison_rows: list[dict[str, Any]] = []
    all_network_parameter_rows: list[dict[str, Any]] = []
    all_network_assignment_rows: list[dict[str, Any]] = []
    all_joint_parameter_rows: list[dict[str, Any]] = []
    all_joint_assignment_rows: list[dict[str, Any]] = []
    all_dwell_rows: list[dict[str, Any]] = []
    all_transition_rows: list[dict[str, Any]] = []
    all_network_event_rows: list[dict[str, Any]] = []
    all_joint_event_rows: list[dict[str, Any]] = []
    all_cross_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    best_trajectory: dict[str, Any] | None = None

    for obsnum in obsnums:
        observation_catalog = catalog[catalog["obsnum"] == obsnum]
        event_rows = observation_catalog.to_dict("records")
        raw_paths = raw_by_observation[obsnum]
        apt_path = args.apt_root / f"apt_{obsnum}_matched.ecsv"
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
                raw_path=raw_paths[network],
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
                    "raw": _file_identity(raw_paths[network]),
                    "apt": _file_identity(apt_path),
                }
            )
        interval_rows, measurement_rows = _interval_measurements(
            obsnum=obsnum,
            projections=projections,
            event_rows=event_rows,
            transition_guard_sec=float(args.transition_guard_sec),
            minimum_samples=int(args.minimum_interval_samples),
        )
        network_models: dict[int, dict[str, Any]] = {}
        observation_network_event_rows: list[dict[str, Any]] = []
        for network in networks:
            model = _network_state_outputs(
                obsnum=obsnum,
                network=network,
                interval_rows=interval_rows,
                measurement_rows=measurement_rows,
                args=args,
            )
            network_models[network] = model
            event_state_rows = _network_event_state_rows(
                obsnum=obsnum,
                network=network,
                event_rows=event_rows,
                model=model,
            )
            observation_network_event_rows.extend(event_state_rows)
            all_comparison_rows.extend(model["comparison_rows"])
            all_network_parameter_rows.extend(model["parameter_rows"])
            all_network_assignment_rows.extend(model["assignment_rows"])
            all_dwell_rows.extend(model["dwell_rows"])
            all_transition_rows.extend(model["transition_rows"])
        joint_model = _joint_state_outputs(
            obsnum=obsnum,
            networks=networks,
            interval_rows=interval_rows,
            network_models=network_models,
            args=args,
        )
        joint_event_rows = _joint_event_state_rows(
            obsnum=obsnum,
            networks=networks,
            event_rows=event_rows,
            joint_model=joint_model,
        )
        cross_rows = _cross_network_rows(
            obsnum=obsnum,
            networks=networks,
            network_models=network_models,
        )
        summary_rows.append(
            _observation_summary_row(
                obsnum=obsnum,
                event_rows=event_rows,
                network_models=network_models,
                network_event_rows=observation_network_event_rows,
                joint_model=joint_model,
                joint_event_rows=joint_event_rows,
                cross_rows=cross_rows,
                minimum_model_intervals=int(args.minimum_model_intervals),
            )
        )
        trajectory = _trajectory_example_candidate(
            obsnum=obsnum,
            networks=networks,
            projections=projections,
            event_rows=event_rows,
            interval_rows=interval_rows,
            network_models=network_models,
            window_sec=float(args.trajectory_example_window_sec),
            padding_sec=float(args.trajectory_example_padding_sec),
            output_sample_rate_hz=float(
                args.trajectory_example_sample_rate_hz
            ),
        )
        if trajectory is not None and (
            best_trajectory is None
            or trajectory["event_count"] > best_trajectory["event_count"]
        ):
            best_trajectory = trajectory
        all_interval_rows.extend(interval_rows)
        all_measurement_rows.extend(measurement_rows)
        all_comparison_rows.extend(joint_model["comparison_rows"])
        all_joint_parameter_rows.extend(joint_model["parameter_rows"])
        all_joint_assignment_rows.extend(joint_model["assignment_rows"])
        all_dwell_rows.extend(joint_model["dwell_rows"])
        all_transition_rows.extend(joint_model["transition_rows"])
        all_network_event_rows.extend(observation_network_event_rows)
        all_joint_event_rows.extend(joint_event_rows)
        all_cross_rows.extend(cross_rows)
        primary_count = int(
            observation_catalog[
                observation_catalog["primary_event_candidate"].astype(bool)
            ].shape[0]
        )
        print(
            f"obs {obsnum}: intervals={len(interval_rows)} "
            f"events={primary_count} "
            f"joint_states={joint_model['selected'].n_states} "
            "network_states="
            + " ".join(
                f"nw{network}:{network_models[network]['selected'].n_states}"
                for network in networks
            )
        )
        del projections
        gc.collect()

    if best_trajectory is None:
        raise ValueError("no trajectory example could be selected")
    output_names = {
        "intervals": "state_intervals.csv",
        "interval_measurements": "state_interval_measurements.csv",
        "model_comparison": "state_model_comparison.csv",
        "network_parameters": "network_state_parameters.csv",
        "network_assignments": "network_interval_state_assignments.csv",
        "joint_parameters": "joint_state_parameters.csv",
        "joint_assignments": "joint_interval_state_assignments.csv",
        "dwell_runs": "state_dwell_runs.csv",
        "transition_matrices": "state_transition_matrices.csv",
        "network_event_audit": "network_event_state_audit.csv",
        "joint_event_audit": "joint_event_state_audit.csv",
        "cross_network": "cross_network_state_correspondence.csv",
        "observation_summary": "observation_state_summary.csv",
        "trajectory_samples": "state_trajectory_example_samples.csv",
        "trajectory_states": "state_trajectory_example_states.csv",
        "trajectory_events": "state_trajectory_example_events.csv",
        "model_figure": "state_model_selection.png",
        "cross_network_figure": "cross_network_state_correspondence.png",
        "joint_center_figure": "joint_state_center_patterns.png",
        "dwell_figure": "joint_state_dwell_distributions.png",
        "trajectory_figure": "state_trajectory_example.png",
    }
    tables: dict[str, list[dict[str, Any]]] = {
        "intervals": all_interval_rows,
        "interval_measurements": all_measurement_rows,
        "model_comparison": all_comparison_rows,
        "network_parameters": all_network_parameter_rows,
        "network_assignments": all_network_assignment_rows,
        "joint_parameters": all_joint_parameter_rows,
        "joint_assignments": all_joint_assignment_rows,
        "dwell_runs": all_dwell_rows,
        "transition_matrices": all_transition_rows,
        "network_event_audit": all_network_event_rows,
        "joint_event_audit": all_joint_event_rows,
        "cross_network": all_cross_rows,
        "observation_summary": summary_rows,
        "trajectory_samples": best_trajectory["sample_rows"],
        "trajectory_states": best_trajectory["state_rows"],
        "trajectory_events": best_trajectory["marker_rows"],
    }
    for key, rows in tables.items():
        _write_csv(args.output_dir / output_names[key], rows)
    _model_selection_figure(
        args.output_dir / output_names["model_figure"],
        all_comparison_rows,
        summary_rows,
        networks=networks,
    )
    _cross_network_figure(
        args.output_dir / output_names["cross_network_figure"],
        all_cross_rows,
        networks=networks,
    )
    _joint_center_pattern_figure(
        args.output_dir / output_names["joint_center_figure"],
        all_joint_parameter_rows,
    )
    _dwell_figure(
        args.output_dir / output_names["dwell_figure"],
        all_dwell_rows,
    )
    _trajectory_figure(
        args.output_dir / output_names["trajectory_figure"],
        best_trajectory,
        networks=networks,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "description": (
            "Catalog-interval hidden-state analysis of stable-UID raw-I/Q "
            "projected phase"
        ),
        "semantics": {
            "change_points": (
                "primary cross-rack boundaries from the validated continuous "
                "event catalog; this state analysis is not an independent "
                "event-rate measurement"
            ),
            "emission": (
                "robust projected-phase median in the guarded interval "
                "between catalog boundaries"
            ),
            "projected_phase_unit": (
                "radians per RMS-normalized UID loading; operational "
                "coordinate, not calibrated detector phase"
            ),
            "state_label": (
                "observation-local ordinal level; not a detector identity or "
                "hardware-state identity"
            ),
            "trend": (
                "Theil-Sen linear trend separated per observation/network "
                "before state fitting"
            ),
            "model": (
                "diagonal-Gaussian HMM on interval medians; one to three "
                "states; no exponential-recovery assumption"
            ),
            "selection": (
                "smallest eligible state count within the configured BIC "
                "tolerance of the minimum; eligibility requires convergence, "
                "minimum posterior occupancy, and minimum center separation"
            ),
            "joint_model": (
                "six-network HMM on robustly standardized, detrended interval "
                "levels; constant sparse-baseline coordinates use unit scale "
                "and carry no state discrimination; state ordering follows "
                "the network-8 center"
            ),
        },
        "parameters": {
            "obsnums": obsnums,
            "networks": networks,
            "event_rich_obsnums": event_rich_obsnums,
            "transition_guard_sec": float(args.transition_guard_sec),
            "minimum_interval_samples": int(
                args.minimum_interval_samples
            ),
            "minimum_model_intervals": int(args.minimum_model_intervals),
            "maximum_states": int(args.maximum_states),
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
            "bic_parsimony_tolerance": float(
                args.bic_parsimony_tolerance
            ),
            "random_seed": int(args.random_seed),
            "trajectory_example_window_sec": float(
                args.trajectory_example_window_sec
            ),
            "trajectory_example_padding_sec": float(
                args.trajectory_example_padding_sec
            ),
            "trajectory_example_sample_rate_hz": float(
                args.trajectory_example_sample_rate_hz
            ),
        },
        "inputs": {
            "event_tones": _file_identity(tone_path),
            "fixed_templates": _file_identity(template_path),
            "continuous_event_catalog": _file_identity(catalog_path),
            "continuous_analysis_manifest": _file_identity(
                continuous_manifest_path
            ),
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
