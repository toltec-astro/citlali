#!/usr/bin/env python3
"""Test whether science I/Q event susceptibility follows UID or readout band.

This tool consumes the complete, model-valid tone table produced by
``science_iq_event_vector_analysis.py`` schema v2. Detector identity is joined
by APT ``uid``. Tone slot is retained as a separate, observation-local readout
coordinate and is never used as detector identity.

The analysis compares three descriptions:

* detector-fixed susceptibility, measured by repeated response of the same
  UID across events;
* readout-band susceptibility, measured by response-rate structure in ordered
  signed digital-tone-offset bins; and
* event-wide network response, measured through event-level coefficients and
  cross-network coupling.

Permutation tests shuffle responsive labels only among model-valid tones
within each event. They preserve event severity and the availability mask.
The resulting p-values are operational model checks, not claims of independent
detector-level statistical significance.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "citlali-tone-susceptibility-mpl-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


SCHEMA_VERSION = "citlali-science-iq-tone-susceptibility-v1"
REQUIRED_TONE_SCHEMA = "citlali-science-iq-event-vector-v2"
DEFAULT_NETWORKS = (1, 2, 3, 4, 8, 9)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _median_or_nan(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(finite) < 3:
        return math.nan
    if (
        np.nanmax(first[finite]) == np.nanmin(first[finite])
        or np.nanmax(second[finite]) == np.nanmin(second[finite])
    ):
        return math.nan
    return float(stats.spearmanr(first[finite], second[finite]).statistic)


def _equal_count_bins(
    frequency_hz: np.ndarray,
    *,
    n_bins: int,
) -> np.ndarray:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    finite = np.flatnonzero(np.isfinite(frequency_hz))
    if finite.size < n_bins:
        raise ValueError(
            f"need at least {n_bins} finite tones, found {finite.size}"
        )
    ordered = finite[np.argsort(frequency_hz[finite])]
    labels = np.full(frequency_hz.size, -1, dtype=int)
    for bin_index, indices in enumerate(np.array_split(ordered, n_bins)):
        labels[indices] = bin_index
    return labels


def _top_fraction_overlap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    fraction: float,
) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    indices = np.flatnonzero(finite)
    if indices.size < 3:
        return math.nan
    count = max(1, int(math.ceil(float(fraction) * indices.size)))
    first_top = set(
        indices[np.argsort(first[indices], kind="stable")[-count:]].tolist()
    )
    second_top = set(
        indices[np.argsort(second[indices], kind="stable")[-count:]].tolist()
    )
    return len(first_top & second_top) / count


def _response_metrics(
    response: np.ndarray,
    opportunity: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    frequency_bins: int,
    minimum_opportunities: int,
    top_fraction: float,
) -> dict[str, Any]:
    response = np.asarray(response, dtype=bool)
    opportunity = np.asarray(opportunity, dtype=bool)
    if response.shape != opportunity.shape or response.ndim != 2:
        raise ValueError("response and opportunity must be matching 2-D arrays")
    if response.shape[1] != len(frequency_hz):
        raise ValueError("frequency count does not match tone matrix")

    opportunities = np.count_nonzero(opportunity, axis=0)
    keep = (
        (opportunities >= int(minimum_opportunities))
        & np.isfinite(frequency_hz)
    )
    if np.count_nonzero(keep) < frequency_bins:
        raise ValueError("too few repeatedly available tones for analysis")
    response_count = np.count_nonzero(response & opportunity, axis=0)
    rates = np.full(opportunities.shape, np.nan, dtype=float)
    rates[keep] = response_count[keep] / opportunities[keep]

    first_rows = np.arange(response.shape[0]) < response.shape[0] // 2
    second_rows = ~first_rows
    first_opp = np.count_nonzero(opportunity[first_rows, :], axis=0)
    second_opp = np.count_nonzero(opportunity[second_rows, :], axis=0)
    first_rate = np.full(rates.shape, np.nan)
    second_rate = np.full(rates.shape, np.nan)
    first_valid = keep & (first_opp > 0)
    second_valid = keep & (second_opp > 0)
    first_rate[first_valid] = (
        np.count_nonzero(
            response[first_rows, :] & opportunity[first_rows, :],
            axis=0,
        )[first_valid]
        / first_opp[first_valid]
    )
    second_rate[second_valid] = (
        np.count_nonzero(
            response[second_rows, :] & opportunity[second_rows, :],
            axis=0,
        )[second_valid]
        / second_opp[second_valid]
    )

    labels = _equal_count_bins(
        np.where(keep, frequency_hz, np.nan),
        n_bins=frequency_bins,
    )
    bin_rates = np.full(frequency_bins, np.nan)
    for bin_index in range(frequency_bins):
        columns = labels == bin_index
        denominator = np.count_nonzero(opportunity[:, columns])
        if denominator:
            bin_rates[bin_index] = (
                np.count_nonzero(response[:, columns] & opportunity[:, columns])
                / denominator
            )
    return {
        "keep": keep,
        "tone_response_rate": rates,
        "frequency_bin_labels": labels,
        "frequency_bin_rates": bin_rates,
        "tone_response_rate_variance": float(np.nanvar(rates[keep])),
        "frequency_bin_rate_variance": float(np.nanvar(bin_rates)),
        "response_rate_frequency_spearman": _safe_spearman(
            rates[keep],
            np.asarray(frequency_hz)[keep],
        ),
        "split_half_response_rate_spearman": _safe_spearman(
            first_rate[keep],
            second_rate[keep],
        ),
        "top_fraction_split_half_overlap": _top_fraction_overlap(
            first_rate[keep],
            second_rate[keep],
            fraction=top_fraction,
        ),
        "first_half_rate": first_rate,
        "second_half_rate": second_rate,
    }


def _permutation_summary(
    response: np.ndarray,
    opportunity: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    frequency_bins: int,
    minimum_opportunities: int,
    top_fraction: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    observed = _response_metrics(
        response,
        opportunity,
        frequency_hz,
        frequency_bins=frequency_bins,
        minimum_opportunities=minimum_opportunities,
        top_fraction=top_fraction,
    )
    names = (
        "tone_response_rate_variance",
        "frequency_bin_rate_variance",
        "response_rate_frequency_spearman",
        "split_half_response_rate_spearman",
        "top_fraction_split_half_overlap",
    )
    null = {name: np.full(n_permutations, np.nan) for name in names}
    permuted = np.zeros_like(response, dtype=bool)
    for permutation in range(n_permutations):
        permuted.fill(False)
        for event_index in range(response.shape[0]):
            available = np.flatnonzero(opportunity[event_index, :])
            if available.size == 0:
                continue
            values = response[event_index, available].copy()
            rng.shuffle(values)
            permuted[event_index, available] = values
        metrics = _response_metrics(
            permuted,
            opportunity,
            frequency_hz,
            frequency_bins=frequency_bins,
            minimum_opportunities=minimum_opportunities,
            top_fraction=top_fraction,
        )
        for name in names:
            null[name][permutation] = metrics[name]

    summaries: dict[str, dict[str, float]] = {}
    for name in names:
        values = null[name]
        values = values[np.isfinite(values)]
        value = float(observed[name])
        null_mean = float(np.mean(values)) if values.size else math.nan
        null_std = float(np.std(values, ddof=1)) if values.size > 1 else math.nan
        comparison_value = abs(value) if "spearman" in name else value
        comparison_null = np.abs(values) if "spearman" in name else values
        p_value = (
            (1.0 + np.count_nonzero(comparison_null >= comparison_value))
            / (values.size + 1.0)
            if values.size and np.isfinite(comparison_value)
            else math.nan
        )
        z_score = (
            (value - null_mean) / null_std
            if np.isfinite(null_std) and null_std > 0.0
            else math.nan
        )
        summaries[name] = {
            "observed": value,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z_score,
            "permutation_p": float(p_value),
        }
    return observed, summaries


def _matrix_for_network(
    tone_rows: pd.DataFrame,
    cluster_rows: pd.DataFrame,
    *,
    network: int,
) -> dict[str, Any]:
    rows = tone_rows[tone_rows["network"] == int(network)].copy()
    if rows.empty:
        raise ValueError(f"no tone rows for nw{network}")
    duplicated = rows.duplicated(["event_cluster_id", "uid"], keep=False)
    if duplicated.any():
        raise ValueError(f"nw{network}: duplicate event/UID tone rows")
    event_order = (
        cluster_rows[["event_cluster_id", "cluster_time_unix_sec"]]
        .drop_duplicates()
        .sort_values("cluster_time_unix_sec")
    )
    events = [
        value
        for value in event_order["event_cluster_id"].tolist()
        if value in set(rows["event_cluster_id"])
    ]
    uid_summary = (
        rows.groupby("uid", as_index=False)
        .agg(
            lo_center_frequency_hz=("lo_center_frequency_hz", "median"),
            tone_offset_frequency_hz=(
                "tone_offset_frequency_hz",
                "median",
            ),
            probe_frequency_hz=("probe_frequency_hz", "median"),
            apt_tone_frequency_hz=("apt_tone_frequency_hz", "median"),
            tone_slot_zero_based=("tone_slot_zero_based", "median"),
            tone_slot_count=("tone_slot_zero_based", "nunique"),
        )
        .sort_values(["tone_offset_frequency_hz", "uid"])
    )
    uids = uid_summary["uid"].astype(int).to_numpy()
    event_index = {value: index for index, value in enumerate(events)}
    uid_index = {int(value): index for index, value in enumerate(uids)}
    opportunity = np.zeros((len(events), len(uids)), dtype=bool)
    response = np.zeros_like(opportunity)
    phase_mrad = np.full(opportunity.shape, np.nan)
    threshold_mrad = np.full(opportunity.shape, np.nan)
    phase_slope_abs_per_hz = np.full(opportunity.shape, np.nan)
    complex_slope_abs_per_hz = np.full(opportunity.shape, np.nan)
    for row in rows.itertuples(index=False):
        event = event_index.get(row.event_cluster_id)
        tone = uid_index.get(int(row.uid))
        if event is None or tone is None:
            continue
        opportunity[event, tone] = True
        response[event, tone] = bool(row.phase_responsive)
        phase_mrad[event, tone] = float(row.phase_change_mrad)
        threshold_mrad[event, tone] = float(row.phase_threshold_mrad)
        phase_slope_abs_per_hz[event, tone] = abs(
            float(row.frequency_direction_imag_per_hz)
        )
        complex_slope_abs_per_hz[event, tone] = math.hypot(
            float(row.frequency_direction_real_per_hz),
            float(row.frequency_direction_imag_per_hz),
        )
    return {
        "rows": rows,
        "events": events,
        "uids": uids,
        "uid_summary": uid_summary,
        "opportunity": opportunity,
        "response": response,
        "phase_mrad": phase_mrad,
        "threshold_mrad": threshold_mrad,
        "phase_slope_abs_per_hz": phase_slope_abs_per_hz,
        "complex_slope_abs_per_hz": complex_slope_abs_per_hz,
    }


def _tone_summary_rows(
    matrix: dict[str, Any],
    metrics: dict[str, Any],
    *,
    network: int,
) -> list[dict[str, Any]]:
    response = matrix["response"]
    opportunity = matrix["opportunity"]
    phase_mrad = matrix["phase_mrad"]
    threshold_mrad = matrix["threshold_mrad"]
    phase_slope_abs_per_hz = matrix["phase_slope_abs_per_hz"]
    complex_slope_abs_per_hz = matrix["complex_slope_abs_per_hz"]
    uid_summary = matrix["uid_summary"].reset_index(drop=True)
    absolute_phase = np.abs(phase_mrad)
    event_median_absolute_phase = np.nanmedian(
        np.where(opportunity, absolute_phase, np.nan),
        axis=1,
    )
    relative_absolute_phase = np.full(absolute_phase.shape, np.nan)
    np.divide(
        absolute_phase,
        event_median_absolute_phase[:, None],
        out=relative_absolute_phase,
        where=(
            opportunity
            & np.isfinite(event_median_absolute_phase[:, None])
            & (event_median_absolute_phase[:, None] > 0.0)
        ),
    )
    first_rows = np.arange(response.shape[0]) < response.shape[0] // 2
    second_rows = ~first_rows
    event_sign = np.sign(np.nanmedian(
        np.where(response, phase_mrad, np.nan),
        axis=1,
    ))
    rows: list[dict[str, Any]] = []
    for tone_index, identity in uid_summary.iterrows():
        available = opportunity[:, tone_index]
        responsive = response[:, tone_index] & available
        phase = phase_mrad[:, tone_index]
        signs = np.sign(phase[responsive])
        aligned = signs == event_sign[responsive]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(network),
                "uid": int(identity["uid"]),
                "tone_slot_zero_based": int(
                    round(identity["tone_slot_zero_based"])
                ),
                "tone_slot_count_across_events": int(
                    identity["tone_slot_count"]
                ),
                "probe_frequency_hz": float(
                    identity["probe_frequency_hz"]
                ),
                "lo_center_frequency_hz": float(
                    identity["lo_center_frequency_hz"]
                ),
                "tone_offset_frequency_hz": float(
                    identity["tone_offset_frequency_hz"]
                ),
                "apt_tone_frequency_hz": float(
                    identity["apt_tone_frequency_hz"]
                ),
                "event_opportunities": int(np.count_nonzero(available)),
                "responsive_events": int(np.count_nonzero(responsive)),
                "response_rate": _finite_or_none(
                    metrics["tone_response_rate"][tone_index]
                ),
                "first_half_response_rate": _finite_or_none(
                    metrics["first_half_rate"][tone_index]
                ),
                "second_half_response_rate": _finite_or_none(
                    metrics["second_half_rate"][tone_index]
                ),
                "frequency_bin_zero_based": int(
                    metrics["frequency_bin_labels"][tone_index]
                ),
                "median_phase_threshold_mrad": _finite_or_none(
                    _median_or_nan(threshold_mrad[available, tone_index])
                ),
                "median_abs_phase_slope_per_hz": _finite_or_none(
                    _median_or_nan(
                        phase_slope_abs_per_hz[available, tone_index]
                    )
                ),
                "median_abs_complex_slope_per_hz": _finite_or_none(
                    _median_or_nan(
                        complex_slope_abs_per_hz[available, tone_index]
                    )
                ),
                "median_abs_phase_change_mrad_all_events": _finite_or_none(
                    _median_or_nan(absolute_phase[available, tone_index])
                ),
                "median_event_normalized_abs_phase_all_events": (
                    _finite_or_none(
                        _median_or_nan(
                            relative_absolute_phase[available, tone_index]
                        )
                    )
                ),
                "first_half_median_event_normalized_abs_phase": (
                    _finite_or_none(
                        _median_or_nan(
                            relative_absolute_phase[
                                first_rows & available,
                                tone_index,
                            ]
                        )
                    )
                ),
                "second_half_median_event_normalized_abs_phase": (
                    _finite_or_none(
                        _median_or_nan(
                            relative_absolute_phase[
                                second_rows & available,
                                tone_index,
                            ]
                        )
                    )
                ),
                "median_abs_phase_change_mrad_when_responsive": (
                    _finite_or_none(
                        _median_or_nan(np.abs(phase[responsive]))
                    )
                    if np.any(responsive)
                    else None
                ),
                "dominant_phase_sign_fraction_when_responsive": (
                    float(
                        max(
                            np.count_nonzero(signs > 0),
                            np.count_nonzero(signs < 0),
                        )
                        / signs.size
                    )
                    if signs.size
                    else None
                ),
                "network_event_sign_alignment_fraction": (
                    float(np.mean(aligned)) if aligned.size else None
                ),
                "included_in_repeatability_test": bool(
                    metrics["keep"][tone_index]
                ),
            }
        )
    return rows


def _frequency_bin_rows(
    tone_rows: list[dict[str, Any]],
    *,
    network: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(tone_rows)
    frame = frame[frame["frequency_bin_zero_based"] >= 0]
    rows: list[dict[str, Any]] = []
    for bin_index, group in frame.groupby("frequency_bin_zero_based"):
        opportunities = int(group["event_opportunities"].sum())
        responsive = int(group["responsive_events"].sum())
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(network),
                "frequency_bin_zero_based": int(bin_index),
                "n_uids": int(len(group)),
                "tone_offset_min_hz": float(
                    group["tone_offset_frequency_hz"].min()
                ),
                "tone_offset_median_hz": float(
                    group["tone_offset_frequency_hz"].median()
                ),
                "tone_offset_max_hz": float(
                    group["tone_offset_frequency_hz"].max()
                ),
                "probe_frequency_median_hz": float(
                    group["probe_frequency_hz"].median()
                ),
                "event_tone_opportunities": opportunities,
                "responsive_event_tones": responsive,
                "response_rate": (
                    responsive / opportunities if opportunities else None
                ),
                "median_phase_threshold_mrad": float(
                    group["median_phase_threshold_mrad"].median()
                ),
                "median_abs_phase_slope_per_hz": float(
                    group["median_abs_phase_slope_per_hz"].median()
                ),
                "median_abs_complex_slope_per_hz": float(
                    group["median_abs_complex_slope_per_hz"].median()
                ),
                "median_abs_phase_change_mrad_all_events": float(
                    group[
                        "median_abs_phase_change_mrad_all_events"
                    ].median()
                ),
                "median_event_normalized_abs_phase_all_events": float(
                    group[
                        "median_event_normalized_abs_phase_all_events"
                    ].median()
                ),
            }
        )
    return rows


def _residual_rows(
    tone_rows: pd.DataFrame,
    fit_rows: pd.DataFrame,
    *,
    networks: list[int],
    frequency_bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coefficients = fit_rows[
        [
            "event_cluster_id",
            "network",
            "fit_status",
            "combined_gain_fraction",
            "combined_phase_rad",
            "combined_frequency_shift_hz",
        ]
    ]
    merged = tone_rows.merge(
        coefficients,
        on=["event_cluster_id", "network"],
        how="left",
        validate="many_to_one",
    )
    merged = merged[
        merged["network"].isin(networks)
        & merged["fit_status"].eq("fit")
        & merged["phase_responsive"].astype(bool)
    ].copy()
    observed = (
        merged["fractional_change_real"].to_numpy()
        + 1j * merged["fractional_change_imag"].to_numpy()
    )
    direction = (
        merged["frequency_direction_real_per_hz"].to_numpy()
        + 1j * merged["frequency_direction_imag_per_hz"].to_numpy()
    )
    prediction = (
        merged["combined_gain_fraction"].to_numpy()
        + 1j * merged["combined_phase_rad"].to_numpy()
        + merged["combined_frequency_shift_hz"].to_numpy() * direction
    )
    residual = observed - prediction
    merged["observed_energy"] = np.abs(observed) ** 2
    merged["residual_energy"] = np.abs(residual) ** 2
    merged["residual_real"] = residual.real
    merged["residual_imag"] = residual.imag
    polarity = np.sign(merged["combined_frequency_shift_hz"].to_numpy())
    polarity[polarity == 0] = 1.0
    merged["aligned_residual_real"] = residual.real * polarity
    merged["aligned_residual_imag"] = residual.imag * polarity

    tone_output: list[dict[str, Any]] = []
    bin_output: list[dict[str, Any]] = []
    for network in networks:
        selected = merged[merged["network"] == network].copy()
        if selected.empty:
            continue
        identity = (
            selected.groupby("uid", as_index=False)
            .agg(
                tone_offset_frequency_hz=(
                    "tone_offset_frequency_hz",
                    "median",
                ),
                probe_frequency_hz=("probe_frequency_hz", "median"),
                tone_slot_zero_based=("tone_slot_zero_based", "median"),
            )
            .sort_values("tone_offset_frequency_hz")
        )
        labels = _equal_count_bins(
            identity["tone_offset_frequency_hz"].to_numpy(),
            n_bins=frequency_bins,
        )
        identity["frequency_bin_zero_based"] = labels
        selected = selected.merge(
            identity[["uid", "frequency_bin_zero_based"]],
            on="uid",
            how="left",
            validate="many_to_one",
        )
        for uid, group in selected.groupby("uid"):
            observed_energy = float(group["observed_energy"].sum())
            residual_energy = float(group["residual_energy"].sum())
            identity_row = identity[identity["uid"] == uid].iloc[0]
            tone_output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "network": int(network),
                    "uid": int(uid),
                    "tone_slot_zero_based": int(
                        round(identity_row["tone_slot_zero_based"])
                    ),
                    "probe_frequency_hz": float(
                        identity_row["probe_frequency_hz"]
                    ),
                    "tone_offset_frequency_hz": float(
                        identity_row["tone_offset_frequency_hz"]
                    ),
                    "frequency_bin_zero_based": int(
                        identity_row["frequency_bin_zero_based"]
                    ),
                    "model_fit_event_count": int(
                        group["event_cluster_id"].nunique()
                    ),
                    "observed_energy": observed_energy,
                    "residual_energy": residual_energy,
                    "per_tone_zero_baseline_r2": (
                        1.0 - residual_energy / observed_energy
                        if observed_energy > 0.0
                        else None
                    ),
                    "mean_aligned_residual_real": float(
                        group["aligned_residual_real"].mean()
                    ),
                    "mean_aligned_residual_imag": float(
                        group["aligned_residual_imag"].mean()
                    ),
                    "rms_residual_fraction": float(
                        math.sqrt(group["residual_energy"].mean())
                    ),
                }
            )
        for bin_index, group in selected.groupby(
            "frequency_bin_zero_based"
        ):
            observed_energy = float(group["observed_energy"].sum())
            residual_energy = float(group["residual_energy"].sum())
            bin_output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "network": int(network),
                    "frequency_bin_zero_based": int(bin_index),
                    "n_uids": int(group["uid"].nunique()),
                    "tone_offset_median_hz": float(
                        group["tone_offset_frequency_hz"].median()
                    ),
                    "probe_frequency_median_hz": float(
                        group["probe_frequency_hz"].median()
                    ),
                    "event_tone_rows": int(len(group)),
                    "zero_baseline_r2": (
                        1.0 - residual_energy / observed_energy
                        if observed_energy > 0.0
                        else None
                    ),
                    "mean_aligned_residual_real": float(
                        group["aligned_residual_real"].mean()
                    ),
                    "mean_aligned_residual_imag": float(
                        group["aligned_residual_imag"].mean()
                    ),
                    "rms_residual_fraction": float(
                        math.sqrt(group["residual_energy"].mean())
                    ),
                }
            )
    return tone_output, bin_output


def _delay_model_rows(
    tone_rows: pd.DataFrame,
    *,
    networks: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = tone_rows[tone_rows["network"].isin(networks)]
    for (event_cluster_id, network), event in selected.groupby(
        ["event_cluster_id", "network"]
    ):
        for population, group in (
            ("all_model_valid", event),
            (
                "phase_responsive",
                event[event["phase_responsive"].astype(bool)],
            ),
        ):
            finite = (
                np.isfinite(group["tone_offset_frequency_hz"])
                & np.isfinite(group["phase_change_mrad"])
            )
            group = group[finite]
            if len(group) < 8:
                continue
            offset_hz = group["tone_offset_frequency_hz"].to_numpy()
            phase_rad = 1.0e-3 * group["phase_change_mrad"].to_numpy()
            design = np.column_stack(
                [
                    np.ones(len(group)),
                    2.0 * np.pi * offset_hz,
                ]
            )
            coefficients, _, _, _ = np.linalg.lstsq(
                design,
                phase_rad,
                rcond=None,
            )
            prediction = design @ coefficients
            denominator = float(np.sum(phase_rad**2))
            delay_r2 = (
                1.0
                - float(np.sum((phase_rad - prediction) ** 2))
                / denominator
                if denominator > 0.0
                else math.nan
            )
            common_phase = float(np.mean(phase_rad))
            common_phase_r2 = (
                1.0
                - float(np.sum((phase_rad - common_phase) ** 2))
                / denominator
                if denominator > 0.0
                else math.nan
            )
            below = group[group["tone_offset_frequency_hz"] < -25.0e6]
            above = group[group["tone_offset_frequency_hz"] > 25.0e6]
            center = group[
                np.abs(group["tone_offset_frequency_hz"]) < 25.0e6
            ]
            edges = group[
                np.abs(group["tone_offset_frequency_hz"]) > 150.0e6
            ]
            below_phase = _median_or_nan(
                below["phase_change_mrad"].to_numpy()
            )
            above_phase = _median_or_nan(
                above["phase_change_mrad"].to_numpy()
            )
            center_amplitude = _median_or_nan(
                np.abs(center["phase_change_mrad"].to_numpy())
            )
            edge_amplitude = _median_or_nan(
                np.abs(edges["phase_change_mrad"].to_numpy())
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_cluster_id": event_cluster_id,
                    "network": int(network),
                    "population": population,
                    "n_tones": int(len(group)),
                    "common_phase_rad": common_phase,
                    "delay_equivalent_sec": float(coefficients[1]),
                    "delay_equivalent_ps": float(
                        coefficients[1] * 1.0e12
                    ),
                    "common_phase_zero_baseline_r2": _finite_or_none(
                        common_phase_r2
                    ),
                    "phase_plus_delay_zero_baseline_r2": _finite_or_none(
                        delay_r2
                    ),
                    "delay_incremental_r2_beyond_common_phase": (
                        _finite_or_none(delay_r2 - common_phase_r2)
                    ),
                    "negative_offset_median_phase_mrad": _finite_or_none(
                        below_phase
                    ),
                    "positive_offset_median_phase_mrad": _finite_or_none(
                        above_phase
                    ),
                    "opposite_median_phase_sign_across_lo": (
                        bool(below_phase * above_phase < 0.0)
                        if np.isfinite(below_phase)
                        and np.isfinite(above_phase)
                        else None
                    ),
                    "edge_to_center_abs_phase_ratio": (
                        _finite_or_none(edge_amplitude / center_amplitude)
                        if np.isfinite(center_amplitude)
                        and center_amplitude > 0.0
                        else None
                    ),
                }
            )
    return rows


def _rank_one_mode_rows(
    tone_rows: pd.DataFrame,
    *,
    networks: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    tone_mode_rows: list[dict[str, Any]] = []
    selected = tone_rows[tone_rows["network"].isin(networks)]
    for network in networks:
        rows = selected[selected["network"] == network]
        event_count = int(rows["event_cluster_id"].nunique())
        uid_counts = rows.groupby("uid")["event_cluster_id"].nunique()
        complete_uids = uid_counts[uid_counts == event_count].index
        complete = rows[rows["uid"].isin(complete_uids)]
        identity = (
            complete.groupby("uid", as_index=False)
            .agg(
                tone_offset_frequency_hz=(
                    "tone_offset_frequency_hz",
                    "median",
                ),
                probe_frequency_hz=("probe_frequency_hz", "median"),
                tone_slot_zero_based=("tone_slot_zero_based", "median"),
            )
            .sort_values("tone_offset_frequency_hz")
        )
        uids = identity["uid"].astype(int).tolist()
        phase = (
            complete.pivot(
                index="event_cluster_id",
                columns="uid",
                values="phase_change_mrad",
            )[uids]
            .sort_index()
            .to_numpy()
            * 1.0e-3
        )
        real = (
            complete.pivot(
                index="event_cluster_id",
                columns="uid",
                values="fractional_change_real",
            )[uids]
            .sort_index()
            .to_numpy()
        )
        imaginary = (
            complete.pivot(
                index="event_cluster_id",
                columns="uid",
                values="fractional_change_imag",
            )[uids]
            .sort_index()
            .to_numpy()
        )
        if not (
            np.all(np.isfinite(phase))
            and np.all(np.isfinite(real))
            and np.all(np.isfinite(imaginary))
        ):
            raise ValueError(f"nw{network}: complete UID matrix is non-finite")

        _, phase_singular, phase_vh = np.linalg.svd(
            phase,
            full_matrices=False,
        )
        complex_matrix = np.column_stack([real, imaginary])
        _, complex_singular, complex_vh = np.linalg.svd(
            complex_matrix,
            full_matrices=False,
        )
        phase_energy = phase_singular**2
        complex_energy = complex_singular**2
        half = phase.shape[0] // 2
        _, _, first_vh = np.linalg.svd(
            phase[:half, :],
            full_matrices=False,
        )
        _, _, second_vh = np.linalg.svd(
            phase[half:, :],
            full_matrices=False,
        )
        split_cosine = float(
            abs(
                np.dot(first_vh[0], second_vh[0])
                / (
                    np.linalg.norm(first_vh[0])
                    * np.linalg.norm(second_vh[0])
                )
            )
        )
        loading = phase_vh[0].copy()
        median_phase = np.median(phase, axis=0)
        if np.dot(loading, median_phase) < 0.0:
            loading *= -1.0
        loading /= math.sqrt(float(np.mean(loading**2)))
        summary_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(network),
                "event_count": event_count,
                "complete_uid_count": int(len(uids)),
                "phase_rank1_energy_fraction": float(
                    phase_energy[0] / np.sum(phase_energy)
                ),
                "phase_rank2_cumulative_energy_fraction": float(
                    np.sum(phase_energy[:2]) / np.sum(phase_energy)
                ),
                "complex_rank1_energy_fraction": float(
                    complex_energy[0] / np.sum(complex_energy)
                ),
                "complex_rank2_cumulative_energy_fraction": float(
                    np.sum(complex_energy[:2]) / np.sum(complex_energy)
                ),
                "phase_rank1_split_half_loading_cosine": split_cosine,
            }
        )
        for tone_index, identity_row in identity.reset_index(
            drop=True
        ).iterrows():
            tone_mode_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "network": int(network),
                    "uid": int(identity_row["uid"]),
                    "tone_slot_zero_based": int(
                        round(identity_row["tone_slot_zero_based"])
                    ),
                    "tone_offset_frequency_hz": float(
                        identity_row["tone_offset_frequency_hz"]
                    ),
                    "probe_frequency_hz": float(
                        identity_row["probe_frequency_hz"]
                    ),
                    "phase_rank1_loading_rms_normalized": float(
                        loading[tone_index]
                    ),
                }
            )
    return summary_rows, tone_mode_rows


def _pair_coupling_rows(
    fit_rows: pd.DataFrame,
    *,
    networks: list[int],
) -> list[dict[str, Any]]:
    selected = fit_rows[
        fit_rows["network"].isin(networks)
        & fit_rows["fit_status"].eq("fit")
    ]
    rows: list[dict[str, Any]] = []
    fields = {
        "frequency_like": "combined_frequency_shift_hz",
        "phase": "combined_phase_rad",
        "gain": "combined_gain_fraction",
    }
    for first, second in itertools.combinations(networks, 2):
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "network_first": int(first),
            "network_second": int(second),
        }
        for label, field in fields.items():
            pivot = (
                selected[selected["network"].isin([first, second])]
                .pivot(
                    index="event_cluster_id",
                    columns="network",
                    values=field,
                )
                .dropna()
            )
            if first not in pivot or second not in pivot:
                count = 0
                correlation = math.nan
                opposite = math.nan
            else:
                count = len(pivot)
                correlation = _safe_spearman(
                    pivot[first].to_numpy(),
                    pivot[second].to_numpy(),
                )
                opposite = float(
                    np.mean(
                        pivot[first].to_numpy()
                        * pivot[second].to_numpy()
                        < 0.0
                    )
                )
            record[f"{label}_paired_event_count"] = int(count)
            record[f"{label}_spearman"] = _finite_or_none(correlation)
            record[f"{label}_opposite_sign_fraction"] = _finite_or_none(
                opposite
            )
        rows.append(record)
    return rows


def _make_susceptibility_figure(
    path: Path,
    *,
    tone_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    network_rows: list[dict[str, Any]],
    networks: list[int],
) -> None:
    tones = pd.DataFrame(tone_rows)
    bins = pd.DataFrame(bin_rows)
    summaries = pd.DataFrame(network_rows).set_index("network")
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        constrained_layout=True,
        sharey=True,
    )
    for ax, network in zip(axes.flat, networks):
        selected = tones[
            (tones["network"] == network)
            & tones["included_in_repeatability_test"]
        ]
        binned = bins[bins["network"] == network].sort_values(
            "frequency_bin_zero_based"
        )
        ax.scatter(
            selected["tone_offset_frequency_hz"] / 1.0e6,
            selected["response_rate"],
            s=9,
            alpha=0.35,
            color="0.25",
            label="UID",
        )
        ax.plot(
            binned["tone_offset_median_hz"] / 1.0e6,
            binned["response_rate"],
            "o-",
            color="C3",
            linewidth=2,
            label="equal-count frequency bin",
        )
        row = summaries.loc[network]
        ax.text(
            0.02,
            0.97,
            (
                f"fixed-tone p={row['tone_heterogeneity_permutation_p']:.3g}\n"
                f"band p={row['tone_offset_banding_permutation_p']:.3g}\n"
                f"split rho={row['split_half_response_rate_spearman']:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.set_title(f"nw{network}")
        ax.set_xlabel("digital tone offset from LO (MHz)")
        ax.set_ylabel("event response rate")
        ax.set_ylim(-0.03, 1.03)
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Repeated tone susceptibility versus digital tone offset",
        fontsize=16,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_response_heatmap(
    path: Path,
    *,
    matrices: dict[int, dict[str, Any]],
    networks: list[int],
) -> None:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        constrained_layout=True,
    )
    cmap = ListedColormap(["#202020", "#f28e2b"])
    cmap.set_bad("#d9d9d9")
    for ax, network in zip(axes.flat, networks):
        matrix = matrices[network]
        opportunity = matrix["opportunity"]
        response = matrix["response"]
        image = np.where(opportunity, response.astype(float), np.nan)
        ax.imshow(
            np.ma.masked_invalid(image),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        frequency = matrix["uid_summary"][
            "tone_offset_frequency_hz"
        ].to_numpy()
        tick_positions = np.linspace(0, len(frequency) - 1, 5)
        tick_indices = np.clip(
            np.rint(tick_positions).astype(int),
            0,
            len(frequency) - 1,
        )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [f"{frequency[index] / 1.0e6:.0f}" for index in tick_indices]
        )
        ax.set_title(f"nw{network}")
        ax.set_xlabel("digital tone offset from LO (MHz)")
        ax.set_ylabel("chronological event index")
    fig.suptitle(
        "Tone response masks: orange responsive, black quiet, gray unavailable",
        fontsize=16,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_amplitude_figure(
    path: Path,
    *,
    tone_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    network_rows: list[dict[str, Any]],
    networks: list[int],
) -> None:
    tones = pd.DataFrame(tone_rows)
    bins = pd.DataFrame(bin_rows)
    summaries = pd.DataFrame(network_rows).set_index("network")
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        constrained_layout=True,
    )
    for ax, network in zip(axes.flat, networks):
        selected = tones[
            (tones["network"] == network)
            & tones["included_in_repeatability_test"]
        ]
        binned = bins[bins["network"] == network].sort_values(
            "frequency_bin_zero_based"
        )
        ax.scatter(
            selected["tone_offset_frequency_hz"] / 1.0e6,
            selected["median_event_normalized_abs_phase_all_events"],
            s=9,
            alpha=0.35,
            color="0.25",
            label="UID",
        )
        ax.plot(
            binned["tone_offset_median_hz"] / 1.0e6,
            binned["median_event_normalized_abs_phase_all_events"],
            "o-",
            color="C0",
            linewidth=2,
            label="equal-count frequency bin",
        )
        row = summaries.loc[network]
        ax.text(
            0.02,
            0.97,
            (
                f"offset rho="
                f"{row['relative_amplitude_tone_offset_spearman']:.2f}\n"
                f"split rho="
                f"{row['relative_amplitude_split_half_spearman']:.2f}\n"
                f"rate-threshold rho="
                f"{row['response_rate_threshold_spearman']:.2f}\n"
                f"amplitude-sweep-slope rho="
                f"{row['relative_amplitude_phase_slope_spearman']:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.axhline(1.0, color="0.6", linewidth=1, linestyle=":")
        ax.set_title(f"nw{network}")
        ax.set_xlabel("digital tone offset from LO (MHz)")
        ax.set_ylabel("median |phase| / event network median")
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Threshold-independent relative raw-phase response",
        fontsize=16,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_pair_coupling_figure(
    path: Path,
    *,
    pair_rows: list[dict[str, Any]],
    networks: list[int],
) -> None:
    count = len(networks)
    correlation = np.full((count, count), np.nan)
    opposite = np.full((count, count), np.nan)
    np.fill_diagonal(correlation, 1.0)
    np.fill_diagonal(opposite, 0.0)
    index = {network: offset for offset, network in enumerate(networks)}
    for row in pair_rows:
        first = index[int(row["network_first"])]
        second = index[int(row["network_second"])]
        correlation[first, second] = correlation[second, first] = float(
            row["frequency_like_spearman"]
        )
        opposite[first, second] = opposite[second, first] = float(
            row["frequency_like_opposite_sign_fraction"]
        )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )
    for ax, values, title, limits, cmap in (
        (
            axes[0],
            correlation,
            "Equivalent-frequency coefficient Spearman",
            (-1.0, 1.0),
            "coolwarm",
        ),
        (
            axes[1],
            opposite,
            "Opposite-sign event fraction",
            (0.0, 1.0),
            "viridis",
        ),
    ):
        image = ax.imshow(values, vmin=limits[0], vmax=limits[1], cmap=cmap)
        ax.set_xticks(range(count), networks)
        ax.set_yticks(range(count), networks)
        ax.set_xlabel("network")
        ax.set_ylabel("network")
        ax.set_title(title)
        for row_index in range(count):
            for column_index in range(count):
                if np.isfinite(values[row_index, column_index]):
                    ax.text(
                        column_index,
                        row_index,
                        f"{values[row_index, column_index]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=(
                            "white"
                            if abs(values[row_index, column_index]) > 0.55
                            else "black"
                        ),
                    )
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_rank_one_figure(
    path: Path,
    *,
    tone_mode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    networks: list[int],
) -> None:
    tones = pd.DataFrame(tone_mode_rows)
    summaries = pd.DataFrame(summary_rows).set_index("network")
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        constrained_layout=True,
    )
    for ax, network in zip(axes.flat, networks):
        selected = tones[tones["network"] == network].sort_values(
            "tone_offset_frequency_hz"
        )
        ax.scatter(
            selected["tone_offset_frequency_hz"] / 1.0e6,
            selected["phase_rank1_loading_rms_normalized"],
            s=9,
            alpha=0.45,
        )
        row = summaries.loc[network]
        ax.text(
            0.02,
            0.97,
            (
                f"phase rank-1={row['phase_rank1_energy_fraction']:.2f}\n"
                f"complex rank-1={row['complex_rank1_energy_fraction']:.2f}\n"
                f"split cosine="
                f"{row['phase_rank1_split_half_loading_cosine']:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.axhline(0.0, color="0.6", linewidth=1)
        ax.set_title(f"nw{network}")
        ax.set_xlabel("digital tone offset from LO (MHz)")
        ax.set_ylabel("phase rank-1 loading (RMS normalized)")
    fig.suptitle(
        "Dominant stable tone-transfer mode across events",
        fontsize=16,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_residual_figure(
    path: Path,
    *,
    residual_bin_rows: list[dict[str, Any]],
    networks: list[int],
) -> None:
    data = pd.DataFrame(residual_bin_rows)
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        constrained_layout=True,
    )
    for ax, network in zip(axes.flat, networks):
        selected = data[data["network"] == network].sort_values(
            "frequency_bin_zero_based"
        )
        x = selected["tone_offset_median_hz"] / 1.0e6
        ax.plot(
            x,
            100.0 * selected["mean_aligned_residual_real"],
            "o-",
            label="real",
        )
        ax.plot(
            x,
            100.0 * selected["mean_aligned_residual_imag"],
            "o-",
            label="imaginary",
        )
        ax.axhline(0.0, color="0.5", linewidth=1)
        ax.set_title(f"nw{network}")
        ax.set_xlabel("digital tone offset from LO (MHz)")
        ax.set_ylabel("polarity-aligned mean residual (%)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Residual after per-event combined gain, phase, and frequency fit",
        fontsize=16,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--networks",
        type=int,
        nargs="+",
        default=list(DEFAULT_NETWORKS),
    )
    parser.add_argument("--frequency-bins", type=int, default=12)
    parser.add_argument("--minimum-opportunities", type=int, default=20)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=4449)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frequency_bins < 3:
        raise ValueError("--frequency-bins must be at least three")
    if args.minimum_opportunities < 2:
        raise ValueError("--minimum-opportunities must be at least two")
    if not 0.0 < args.top_fraction <= 0.5:
        raise ValueError("--top-fraction must be in (0, 0.5]")
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be positive")
    networks = [int(value) for value in args.networks]
    if len(set(networks)) != len(networks):
        raise ValueError("--networks contains duplicates")

    tone_path = args.input_dir / "science_event_tone_vectors.csv"
    fit_path = args.input_dir / "science_event_vector_fits.csv"
    cluster_path = args.input_dir / "science_raw_event_clusters.csv"
    tone_data = pd.read_csv(tone_path)
    fit_data = pd.read_csv(fit_path)
    cluster_data = pd.read_csv(cluster_path)
    schemas = set(tone_data["schema_version"].dropna().astype(str))
    if schemas != {REQUIRED_TONE_SCHEMA}:
        raise ValueError(
            f"tone table requires schema {REQUIRED_TONE_SCHEMA}, found "
            f"{sorted(schemas)}"
        )
    responsive_values = set(tone_data["phase_responsive"].astype(bool))
    if responsive_values != {False, True}:
        raise ValueError(
            "tone table lacks both responsive and quiet model-valid tones; "
            "rerun science_iq_event_vector_analysis.py schema v2"
        )
    delay_rows = _delay_model_rows(tone_data, networks=networks)
    delay_frame = pd.DataFrame(delay_rows)
    rank_summary_rows, rank_tone_rows = _rank_one_mode_rows(
        tone_data,
        networks=networks,
    )
    rank_summary = pd.DataFrame(rank_summary_rows).set_index("network")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.random_seed))
    matrices: dict[int, dict[str, Any]] = {}
    tone_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    for network in networks:
        matrix = _matrix_for_network(
            tone_data,
            cluster_data,
            network=network,
        )
        matrices[network] = matrix
        frequency = matrix["uid_summary"][
            "tone_offset_frequency_hz"
        ].to_numpy()
        metrics, permutation = _permutation_summary(
            matrix["response"],
            matrix["opportunity"],
            frequency,
            frequency_bins=int(args.frequency_bins),
            minimum_opportunities=int(args.minimum_opportunities),
            top_fraction=float(args.top_fraction),
            n_permutations=int(args.n_permutations),
            rng=rng,
        )
        network_tones = _tone_summary_rows(
            matrix,
            metrics,
            network=network,
        )
        tone_rows.extend(network_tones)
        bin_rows.extend(
            _frequency_bin_rows(network_tones, network=network)
        )
        event_response_fraction = np.count_nonzero(
            matrix["response"], axis=1
        ) / np.maximum(
            1,
            np.count_nonzero(matrix["opportunity"], axis=1),
        )
        tone_frame = pd.DataFrame(network_tones)
        repeated = tone_frame[
            tone_frame["included_in_repeatability_test"]
        ]
        network_delay = delay_frame[
            (delay_frame["network"] == network)
            & delay_frame["population"].eq("all_model_valid")
        ]
        network_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": network,
                "event_count": len(matrix["events"]),
                "uid_count": len(matrix["uids"]),
                "median_event_response_fraction": float(
                    np.median(event_response_fraction)
                ),
                "tone_response_rate_variance": metrics[
                    "tone_response_rate_variance"
                ],
                "tone_heterogeneity_null_mean": permutation[
                    "tone_response_rate_variance"
                ]["null_mean"],
                "tone_heterogeneity_z_score": permutation[
                    "tone_response_rate_variance"
                ]["z_score"],
                "tone_heterogeneity_permutation_p": permutation[
                    "tone_response_rate_variance"
                ]["permutation_p"],
                "frequency_bin_rate_variance": metrics[
                    "frequency_bin_rate_variance"
                ],
                "tone_offset_banding_null_mean": permutation[
                    "frequency_bin_rate_variance"
                ]["null_mean"],
                "tone_offset_banding_z_score": permutation[
                    "frequency_bin_rate_variance"
                ]["z_score"],
                "tone_offset_banding_permutation_p": permutation[
                    "frequency_bin_rate_variance"
                ]["permutation_p"],
                "response_rate_tone_offset_spearman": metrics[
                    "response_rate_frequency_spearman"
                ],
                "tone_offset_spearman_permutation_p": permutation[
                    "response_rate_frequency_spearman"
                ]["permutation_p"],
                "split_half_response_rate_spearman": metrics[
                    "split_half_response_rate_spearman"
                ],
                "split_half_spearman_permutation_p": permutation[
                    "split_half_response_rate_spearman"
                ]["permutation_p"],
                "top_fraction": float(args.top_fraction),
                "top_fraction_split_half_overlap": metrics[
                    "top_fraction_split_half_overlap"
                ],
                "top_overlap_permutation_p": permutation[
                    "top_fraction_split_half_overlap"
                ]["permutation_p"],
                "response_rate_threshold_spearman": _safe_spearman(
                    repeated["response_rate"].to_numpy(),
                    repeated["median_phase_threshold_mrad"].to_numpy(),
                ),
                "relative_amplitude_tone_offset_spearman": _safe_spearman(
                    repeated[
                        "median_event_normalized_abs_phase_all_events"
                    ].to_numpy(),
                    repeated["tone_offset_frequency_hz"].to_numpy(),
                ),
                "relative_amplitude_split_half_spearman": _safe_spearman(
                    repeated[
                        "first_half_median_event_normalized_abs_phase"
                    ].to_numpy(),
                    repeated[
                        "second_half_median_event_normalized_abs_phase"
                    ].to_numpy(),
                ),
                "response_rate_phase_slope_spearman": _safe_spearman(
                    repeated["response_rate"].to_numpy(),
                    repeated["median_abs_phase_slope_per_hz"].to_numpy(),
                ),
                "relative_amplitude_phase_slope_spearman": _safe_spearman(
                    repeated[
                        "median_event_normalized_abs_phase_all_events"
                    ].to_numpy(),
                    repeated["median_abs_phase_slope_per_hz"].to_numpy(),
                ),
                "median_phase_plus_delay_zero_baseline_r2": float(
                    network_delay[
                        "phase_plus_delay_zero_baseline_r2"
                    ].median()
                ),
                "median_delay_incremental_r2_beyond_common_phase": float(
                    network_delay[
                        "delay_incremental_r2_beyond_common_phase"
                    ].median()
                ),
                "median_edge_to_center_abs_phase_ratio": float(
                    network_delay[
                        "edge_to_center_abs_phase_ratio"
                    ].median()
                ),
                "opposite_phase_sign_across_lo_event_fraction": float(
                    network_delay[
                        "opposite_median_phase_sign_across_lo"
                    ].mean()
                ),
                "phase_rank1_energy_fraction": float(
                    rank_summary.loc[
                        network,
                        "phase_rank1_energy_fraction",
                    ]
                ),
                "phase_rank2_cumulative_energy_fraction": float(
                    rank_summary.loc[
                        network,
                        "phase_rank2_cumulative_energy_fraction",
                    ]
                ),
                "complex_rank1_energy_fraction": float(
                    rank_summary.loc[
                        network,
                        "complex_rank1_energy_fraction",
                    ]
                ),
                "complex_rank2_cumulative_energy_fraction": float(
                    rank_summary.loc[
                        network,
                        "complex_rank2_cumulative_energy_fraction",
                    ]
                ),
                "phase_rank1_split_half_loading_cosine": float(
                    rank_summary.loc[
                        network,
                        "phase_rank1_split_half_loading_cosine",
                    ]
                ),
            }
        )
        print(
            f"nw{network}: events={len(matrix['events'])} "
            f"uids={len(matrix['uids'])} "
            f"fixed_p={network_rows[-1]['tone_heterogeneity_permutation_p']:.4g} "
            f"band_p={network_rows[-1]['tone_offset_banding_permutation_p']:.4g} "
            f"split_rho={network_rows[-1]['split_half_response_rate_spearman']:.3f}",
            flush=True,
        )

    residual_tones, residual_bins = _residual_rows(
        tone_data,
        fit_data,
        networks=networks,
        frequency_bins=int(args.frequency_bins),
    )
    pair_rows = _pair_coupling_rows(fit_data, networks=networks)
    outputs = {
        "tone_susceptibility": "science_tone_susceptibility.csv",
        "network_summary": "science_network_susceptibility_summary.csv",
        "frequency_bins": "science_tone_offset_bin_susceptibility.csv",
        "network_pair_coupling": "science_network_pair_coupling.csv",
        "tone_offset_delay_fits": "science_tone_offset_delay_fits.csv",
        "rank_one_summary": "science_tone_rank_one_summary.csv",
        "rank_one_tones": "science_tone_rank_one_modes.csv",
        "tone_model_residuals": "science_tone_model_residuals.csv",
        "residual_frequency_bins": "science_residual_tone_offset_bins.csv",
        "susceptibility_figure": (
            "science_tone_susceptibility_vs_offset.png"
        ),
        "response_heatmap": "science_tone_response_heatmap.png",
        "relative_amplitude_figure": (
            "science_tone_relative_phase_vs_offset.png"
        ),
        "pair_coupling_figure": "science_network_pair_coupling.png",
        "rank_one_figure": "science_tone_rank_one_modes.png",
        "residual_figure": "science_tone_model_residuals.png",
    }
    _write_csv(args.output_dir / outputs["tone_susceptibility"], tone_rows)
    _write_csv(args.output_dir / outputs["network_summary"], network_rows)
    _write_csv(args.output_dir / outputs["frequency_bins"], bin_rows)
    _write_csv(args.output_dir / outputs["network_pair_coupling"], pair_rows)
    _write_csv(args.output_dir / outputs["tone_offset_delay_fits"], delay_rows)
    _write_csv(
        args.output_dir / outputs["rank_one_summary"],
        rank_summary_rows,
    )
    _write_csv(
        args.output_dir / outputs["rank_one_tones"],
        rank_tone_rows,
    )
    _write_csv(
        args.output_dir / outputs["tone_model_residuals"],
        residual_tones,
    )
    _write_csv(
        args.output_dir / outputs["residual_frequency_bins"],
        residual_bins,
    )
    _make_susceptibility_figure(
        args.output_dir / outputs["susceptibility_figure"],
        tone_rows=tone_rows,
        bin_rows=bin_rows,
        network_rows=network_rows,
        networks=networks,
    )
    _make_response_heatmap(
        args.output_dir / outputs["response_heatmap"],
        matrices=matrices,
        networks=networks,
    )
    _make_amplitude_figure(
        args.output_dir / outputs["relative_amplitude_figure"],
        tone_rows=tone_rows,
        bin_rows=bin_rows,
        network_rows=network_rows,
        networks=networks,
    )
    _make_pair_coupling_figure(
        args.output_dir / outputs["pair_coupling_figure"],
        pair_rows=pair_rows,
        networks=networks,
    )
    _make_rank_one_figure(
        args.output_dir / outputs["rank_one_figure"],
        tone_mode_rows=rank_tone_rows,
        summary_rows=rank_summary_rows,
        networks=networks,
    )
    _make_residual_figure(
        args.output_dir / outputs["residual_figure"],
        residual_bin_rows=residual_bins,
        networks=networks,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "description": (
            "UID repeatability, digital-tone-offset banding, residual "
            "spectrum, and cross-network event coupling"
        ),
        "semantics": {
            "detector_identity": (
                "APT uid; tone slot is retained only as a readout coordinate"
            ),
            "tone_offset_frequency_hz": (
                "Median signed digital tone frequency relative to the "
                "network LO center; the banding coordinate"
            ),
            "probe_frequency_hz": (
                "Median LO-center plus signed tone-offset frequency for the "
                "UID"
            ),
            "response": (
                "Phase change exceeds the event-vector analysis per-tone "
                "noise threshold"
            ),
            "delay_model": (
                "Raw phase = common phase + 2*pi*tone_offset*delay; "
                "tested on all model-valid tones and separately on the "
                "responsive fit population"
            ),
            "rank_one_model": (
                "Uncentered SVD on UIDs available in every event; energy "
                "fractions test event amplitude times a stable tone-transfer "
                "mode"
            ),
            "permutation": (
                "Responsive labels shuffled among model-valid tones within "
                "each event; event severity and availability preserved"
            ),
            "p_values": (
                "Operational permutation diagnostics, not independent "
                "detector-level significance claims"
            ),
        },
        "inputs": {
            "tone_vectors": str(tone_path),
            "event_vector_fits": str(fit_path),
            "event_clusters": str(cluster_path),
        },
        "parameters": {
            "networks": networks,
            "frequency_bins": int(args.frequency_bins),
            "minimum_opportunities": int(args.minimum_opportunities),
            "top_fraction": float(args.top_fraction),
            "n_permutations": int(args.n_permutations),
            "random_seed": int(args.random_seed),
        },
        "counts": {
            "tone_rows": len(tone_rows),
            "network_rows": len(network_rows),
            "frequency_bin_rows": len(bin_rows),
            "pair_rows": len(pair_rows),
            "delay_fit_rows": len(delay_rows),
            "rank_summary_rows": len(rank_summary_rows),
            "rank_tone_rows": len(rank_tone_rows),
            "residual_tone_rows": len(residual_tones),
            "residual_bin_rows": len(residual_bins),
        },
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
