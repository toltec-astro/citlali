#!/usr/bin/env python3
"""Classify synchronized science raw-I/Q events by complex-vector geometry.

This diagnostic does not score target sweeps as good or bad tunes. It uses the
processed sweep only as an empirical measurement of the local complex
derivative ``d(I+iQ)/df`` for each raw tone slot.

Candidate science chunks come from persisted Citlali RTC diagnostics. Within
those chunks, event times and complex changes are measured independently from
raw ``I + iQ``. Cross-network raw candidates are clustered, and every network
is then evaluated at the shared cluster epoch against three real-coefficient
models:

* common gain: ``delta_z / z = a``;
* common phase rotation: ``delta_z / z = i phi``; and
* frequency-like motion: ``delta_z / z = delta_f (d z / d f) / z``.

Candidate times are operational change-point estimates, not exact physical
onsets. UTC/PPS and sample-modulus tests are exploratory timing diagnostics.
"""

from __future__ import annotations

import argparse
import csv
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
    str(Path(tempfile.gettempdir()) / "citlali-iq-event-vector-mpl-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402

from tools.diagnostics import pointing_iq_event_coherence as iq_tool  # noqa: E402
from tools.diagnostics.science_iq_temperature_survey import (  # noqa: E402
    _scan_intervals_from_times,
)


SCHEMA_VERSION = "citlali-science-iq-event-vector-v2"
DEFAULT_OBSNUMS = (152390, 152392, 152419, 152431, 152433)
DEFAULT_AFFECTED_NETWORKS = (1, 2, 3, 4, 8, 9)
DEFAULT_CONTROL_NETWORKS = (0, 5, 7, 11, 12)
DEFAULT_SAMPLE_MODULI = (8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class SweepModel:
    network: int
    path: Path
    lo_center_frequency_hz: float
    tone_offset_frequency_hz: np.ndarray
    probe_frequency_hz: np.ndarray
    z_at_probe: np.ndarray
    dz_df: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class RawCandidate:
    event_sec: float
    event_row: int
    coherent_same_sign_fraction: float
    strong_phase_fraction: float
    n_apt_usable_tones: int
    n_strong_phase_tones: int


@dataclass(frozen=True)
class EventVector:
    fractional_change: np.ndarray
    phase_change_rad: np.ndarray
    phase_threshold_rad: np.ndarray
    responsive: np.ndarray
    valid: np.ndarray


def _utc_iso(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=UTC).isoformat()


def _array_name(network: int) -> str:
    if 0 <= int(network) <= 6:
        return "a1100"
    if 7 <= int(network) <= 10:
        return "a1400"
    if 11 <= int(network) <= 12:
        return "a2000"
    raise ValueError(f"network {network} has no TolTEC array mapping")


def _rack(network: int) -> str:
    return "RACKA" if int(network) <= 6 else "RACKO"


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        names = ", ".join(path.name for path in paths)
        raise FileNotFoundError(
            f"expected one file matching {root / pattern}, "
            f"found {len(paths)}: {names}"
        )
    return paths[0]


def _science_rtc_path(reduction_root: Path, obsnum: int) -> Path:
    path = (
        reduction_root
        / str(obsnum)
        / "raw"
        / f"toltec_commissioning_science_{obsnum}_rtcdiag.nc"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _interval_means(
    prefix: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    counts = ends - starts
    if np.any(counts <= 0):
        raise ValueError("event comparison interval contains no samples")
    return (prefix[ends, :] - prefix[starts, :]) / counts[:, None]


def _raw_candidate(
    data: iq_tool.NetworkData,
    *,
    sigma_threshold: float,
    min_phase_mrad: float,
    pre_window_sec: float,
    guard_window_sec: float,
    post_window_sec: float,
) -> RawCandidate:
    time_sec = np.asarray(data.time_sec, dtype=float)
    complex_iq = np.asarray(data.complex_iq, dtype=complex)
    if time_sec.size < 32 or np.any(np.diff(time_sec) <= 0):
        raise ValueError(f"nw{data.network}: invalid raw sample time")

    amplitude = np.abs(complex_iq)
    phase = np.unwrap(np.angle(complex_iq), axis=0)
    phase_sigma = iq_tool._robust_sigma(
        np.diff(phase, axis=0), axis=0
    ) / math.sqrt(2.0)
    phase_threshold = np.maximum(
        float(sigma_threshold) * phase_sigma,
        float(min_phase_mrad) * 1.0e-3,
    )
    valid = (
        data.apt_usable
        & np.isfinite(phase_threshold)
        & (np.nanmedian(amplitude, axis=0) > 0.0)
    )
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError(f"nw{data.network}: no APT-usable raw tones")

    unit_phase = np.full(complex_iq.shape, np.nan + 1j * np.nan)
    np.divide(complex_iq, amplitude, out=unit_phase, where=amplitude > 0.0)
    prefix = np.vstack(
        [
            np.zeros((1, complex_iq.shape[1]), dtype=complex),
            np.nancumsum(unit_phase, axis=0),
        ]
    )
    minimum_time = (
        float(time_sec[0]) + float(pre_window_sec) + float(guard_window_sec)
    )
    maximum_time = (
        float(time_sec[-1])
        - float(post_window_sec)
        - float(guard_window_sec)
    )
    candidate_rows = np.flatnonzero(
        (time_sec >= minimum_time) & (time_sec <= maximum_time)
    )
    if candidate_rows.size == 0:
        raise ValueError(f"nw{data.network}: no complete event window")
    candidate_time = time_sec[candidate_rows]
    pre_start = np.searchsorted(
        time_sec,
        candidate_time - guard_window_sec - pre_window_sec,
    )
    pre_end = np.searchsorted(
        time_sec,
        candidate_time - guard_window_sec,
    )
    post_start = np.searchsorted(
        time_sec,
        candidate_time + guard_window_sec,
    )
    post_end = np.searchsorted(
        time_sec,
        candidate_time + guard_window_sec + post_window_sec,
    )
    phase_pre = _interval_means(prefix, pre_start, pre_end)
    phase_post = _interval_means(prefix, post_start, post_end)
    phase_shift = np.angle(phase_post / phase_pre)
    strong = np.abs(phase_shift) > phase_threshold[None, :]
    strong[:, ~valid] = False
    positive = np.count_nonzero(strong & (phase_shift > 0.0), axis=1)
    negative = np.count_nonzero(strong & (phase_shift < 0.0), axis=1)
    coherent_fraction = np.maximum(positive, negative) / n_valid
    strong_fraction = np.count_nonzero(strong, axis=1) / n_valid
    best = int(np.nanargmax(coherent_fraction))
    return RawCandidate(
        event_sec=float(time_sec[candidate_rows[best]]),
        event_row=int(candidate_rows[best]),
        coherent_same_sign_fraction=float(coherent_fraction[best]),
        strong_phase_fraction=float(strong_fraction[best]),
        n_apt_usable_tones=n_valid,
        n_strong_phase_tones=int(np.count_nonzero(strong[best])),
    )


def _extract_event_vector(
    data: iq_tool.NetworkData,
    *,
    event_sec: float,
    sigma_threshold: float,
    min_phase_mrad: float,
    pre_window_sec: float,
    guard_window_sec: float,
    post_window_sec: float,
) -> EventVector:
    time_sec = np.asarray(data.time_sec, dtype=float)
    complex_iq = np.asarray(data.complex_iq, dtype=complex)
    phase = np.unwrap(np.angle(complex_iq), axis=0)
    phase_sigma = iq_tool._robust_sigma(
        np.diff(phase, axis=0), axis=0
    ) / math.sqrt(2.0)
    threshold = np.maximum(
        float(sigma_threshold) * phase_sigma,
        float(min_phase_mrad) * 1.0e-3,
    )
    pre = (
        (time_sec >= event_sec - guard_window_sec - pre_window_sec)
        & (time_sec < event_sec - guard_window_sec)
    )
    post = (
        (time_sec > event_sec + guard_window_sec)
        & (time_sec <= event_sec + guard_window_sec + post_window_sec)
    )
    if np.count_nonzero(pre) < 4 or np.count_nonzero(post) < 4:
        raise ValueError(
            f"nw{data.network}: shared event window is incomplete"
        )
    z_pre = np.nanmean(complex_iq[pre, :], axis=0)
    z_post = np.nanmean(complex_iq[post, :], axis=0)
    fractional_change = np.full(z_pre.shape, np.nan + 1j * np.nan)
    np.divide(
        z_post,
        z_pre,
        out=fractional_change,
        where=np.abs(z_pre) > 0.0,
    )
    fractional_change -= 1.0
    phase_change = np.angle(z_post / z_pre)
    valid = (
        data.apt_usable
        & np.isfinite(fractional_change.real)
        & np.isfinite(fractional_change.imag)
        & np.isfinite(threshold)
        & (np.abs(z_pre) > 0.0)
    )
    responsive = valid & (np.abs(phase_change) > threshold)
    return EventVector(
        fractional_change=fractional_change,
        phase_change_rad=phase_change,
        phase_threshold_rad=threshold,
        responsive=responsive,
        valid=valid,
    )


def _local_complex_derivative(
    frequency_hz: np.ndarray,
    complex_iq: np.ndarray,
    probe_frequency_hz: np.ndarray,
    *,
    half_window_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    complex_iq = np.asarray(complex_iq, dtype=complex)
    probe_frequency_hz = np.asarray(probe_frequency_hz, dtype=float)
    if frequency_hz.shape != complex_iq.shape:
        raise ValueError("sweep frequency and I/Q shapes differ")
    if frequency_hz.shape[1] != probe_frequency_hz.size:
        raise ValueError("sweep tone count and probe-frequency count differ")
    if half_window_steps < 1:
        raise ValueError("half_window_steps must be positive")

    n_tones = probe_frequency_hz.size
    z_at_probe = np.full(n_tones, np.nan + 1j * np.nan)
    derivative = np.full(n_tones, np.nan + 1j * np.nan)
    valid = np.zeros(n_tones, dtype=bool)
    for tone in range(n_tones):
        x_all = frequency_hz[:, tone] - probe_frequency_hz[tone]
        z_all = complex_iq[:, tone]
        finite = (
            np.isfinite(x_all)
            & np.isfinite(z_all.real)
            & np.isfinite(z_all.imag)
        )
        indices = np.flatnonzero(finite)
        if indices.size < 3:
            continue
        center = int(indices[np.argmin(np.abs(x_all[indices]))])
        lo = max(0, center - half_window_steps)
        hi = min(frequency_hz.shape[0], center + half_window_steps + 1)
        selected = np.arange(lo, hi)
        selected = selected[finite[selected]]
        if selected.size < 3:
            continue
        x = x_all[selected]
        z = z_all[selected]
        x_mean = float(np.mean(x))
        z_mean = np.mean(z)
        denominator = float(np.sum((x - x_mean) ** 2))
        if not np.isfinite(denominator) or denominator <= 0.0:
            continue
        slope = np.sum((x - x_mean) * (z - z_mean)) / denominator
        intercept = z_mean - slope * x_mean
        if (
            np.isfinite(intercept.real)
            and np.isfinite(intercept.imag)
            and np.isfinite(slope.real)
            and np.isfinite(slope.imag)
            and abs(intercept) > 0.0
            and abs(slope) > 0.0
        ):
            z_at_probe[tone] = intercept
            derivative[tone] = slope
            valid[tone] = True
    return z_at_probe, derivative, valid


def _load_sweep_model(
    *,
    data_root: Path,
    raw_path: Path,
    obsnum: int,
    network: int,
    half_window_steps: int,
) -> SweepModel:
    sweep_path = _find_one(
        data_root,
        f"toltec{network}_{obsnum:06d}_000_0001_*_tune_processed.nc",
    )
    with netCDF4.Dataset(raw_path) as raw:
        raw_tone = np.asarray(
            raw.variables["Header.Toltec.ToneFreq"][0, :],
            dtype=float,
        )
        raw_lo = float(raw.variables["Header.Toltec.LoCenterFreq"][...])
    with netCDF4.Dataset(sweep_path) as sweep:
        sweep_tone = np.asarray(
            sweep.variables["Header.Toltec.ToneFreq"][:],
            dtype=float,
        )
        sweep_lo = float(sweep.variables["Header.Toltec.LoCenterFreq"][...])
        frequency = np.asarray(sweep.variables["Data.Kids.fs"][:], dtype=float)
        complex_iq = np.asarray(
            sweep.variables["Data.Kids.Is"][:], dtype=float
        ) + 1j * np.asarray(sweep.variables["Data.Kids.Qs"][:], dtype=float)
    if raw_tone.shape != sweep_tone.shape:
        raise ValueError(
            f"obs {obsnum} nw{network}: raw and sweep tone counts differ"
        )
    if not np.allclose(raw_tone, sweep_tone, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"obs {obsnum} nw{network}: raw and sweep tone slots differ"
        )
    if not math.isclose(raw_lo, sweep_lo, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(
            f"obs {obsnum} nw{network}: raw and sweep LO frequencies differ"
        )
    probe = raw_lo + raw_tone
    z_at_probe, derivative, valid = _local_complex_derivative(
        frequency,
        complex_iq,
        probe,
        half_window_steps=half_window_steps,
    )
    return SweepModel(
        network=int(network),
        path=sweep_path,
        lo_center_frequency_hz=raw_lo,
        tone_offset_frequency_hz=raw_tone,
        probe_frequency_hz=probe,
        z_at_probe=z_at_probe,
        dz_df=derivative,
        valid=valid,
    )


def _fit_real_complex_columns(
    y: np.ndarray,
    columns: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y, dtype=complex)
    prepared = [
        np.broadcast_to(np.asarray(column, dtype=complex), y.shape)
        for column in columns
    ]
    finite = np.isfinite(y.real) & np.isfinite(y.imag)
    for column in prepared:
        finite &= np.isfinite(column.real) & np.isfinite(column.imag)
    if np.count_nonzero(finite) < max(3, len(columns) + 1):
        return (
            np.full(len(columns), np.nan),
            np.full(y.shape, np.nan + 1j * np.nan),
            math.nan,
        )
    y_selected = y[finite]
    matrix = np.column_stack(
        [
            np.concatenate([column[finite].real, column[finite].imag])
            for column in prepared
        ]
    )
    target = np.concatenate([y_selected.real, y_selected.imag])
    coefficients, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    prediction = np.zeros(y.shape, dtype=complex)
    for coefficient, column in zip(coefficients, prepared):
        prediction += float(coefficient) * column
    denominator = float(np.sum(np.abs(y_selected) ** 2))
    residual = y_selected - prediction[finite]
    r_squared = (
        1.0 - float(np.sum(np.abs(residual) ** 2)) / denominator
        if denominator > 0.0
        else math.nan
    )
    return coefficients, prediction, r_squared


def _fit_event_modes(
    fractional_change: np.ndarray,
    frequency_direction_per_hz: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(fractional_change, dtype=complex)[mask]
    frequency = np.asarray(frequency_direction_per_hz, dtype=complex)[mask]
    one = np.ones(y.shape, dtype=complex)
    imaginary = 1j * one
    fits: dict[str, tuple[np.ndarray, np.ndarray, float]] = {
        "gain": _fit_real_complex_columns(y, [one]),
        "phase": _fit_real_complex_columns(y, [imaginary]),
        "frequency": _fit_real_complex_columns(y, [frequency]),
        "common_complex": _fit_real_complex_columns(y, [one, imaginary]),
        "combined": _fit_real_complex_columns(
            y,
            [one, imaginary, frequency],
        ),
    }
    single = {
        name: fit[2]
        for name, fit in fits.items()
        if name in {"gain", "phase", "frequency"}
    }
    finite_single = {
        name: value for name, value in single.items() if np.isfinite(value)
    }
    best_single = (
        max(finite_single, key=finite_single.get)
        if finite_single
        else "unavailable"
    )
    combined = fits["combined"]
    common = fits["common_complex"]
    frequency_fit = fits["frequency"]
    return {
        "n_fit_tones": int(y.size),
        "gain_r2": _finite_or_none(fits["gain"][2]),
        "phase_r2": _finite_or_none(fits["phase"][2]),
        "frequency_r2": _finite_or_none(frequency_fit[2]),
        "common_complex_r2": _finite_or_none(common[2]),
        "combined_r2": _finite_or_none(combined[2]),
        "best_single_mode": best_single,
        "gain_fraction_gain_only": _finite_or_none(fits["gain"][0][0]),
        "phase_rad_phase_only": _finite_or_none(fits["phase"][0][0]),
        "frequency_shift_hz_frequency_only": _finite_or_none(
            frequency_fit[0][0]
        ),
        "combined_gain_fraction": _finite_or_none(combined[0][0]),
        "combined_phase_rad": _finite_or_none(combined[0][1]),
        "combined_frequency_shift_hz": _finite_or_none(combined[0][2]),
        "frequency_incremental_r2_beyond_common": _finite_or_none(
            combined[2] - common[2]
        ),
        "common_incremental_r2_beyond_frequency": _finite_or_none(
            combined[2] - frequency_fit[2]
        ),
    }


def _cluster_candidates(
    rows: list[dict[str, Any]],
    *,
    affected_networks: set[int],
    minimum_fraction: float,
    tolerance_sec: float,
    minimum_affected_networks: int,
) -> list[list[dict[str, Any]]]:
    selected = sorted(
        [
            row
            for row in rows
            if float(row["raw_coherent_same_sign_fraction"])
            >= float(minimum_fraction)
        ],
        key=lambda row: float(row["raw_event_time_unix_sec"]),
    )
    groups: list[list[dict[str, Any]]] = []
    for row in selected:
        if not groups:
            groups.append([row])
            continue
        group_start = min(
            float(item["raw_event_time_unix_sec"]) for item in groups[-1]
        )
        if (
            float(row["raw_event_time_unix_sec"]) - group_start
            <= float(tolerance_sec)
        ):
            groups[-1].append(row)
        else:
            groups.append([row])
    return [
        group
        for group in groups
        if len(
            {
                int(row["network"])
                for row in group
                if int(row["network"]) in affected_networks
            }
        )
        >= int(minimum_affected_networks)
    ]


def _rayleigh_summary(values: Iterable[float], *, period: float) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "period": float(period),
            "mean_phase": None,
            "resultant_length": None,
            "rayleigh_p_approx": None,
        }
    angle = 2.0 * np.pi * np.mod(array, period) / period
    vector = np.mean(np.exp(1j * angle))
    resultant = float(abs(vector))
    z_value = float(array.size * resultant**2)
    p_value = math.exp(-z_value)
    if array.size >= 5:
        p_value *= 1.0 + (
            (2.0 * z_value - z_value**2) / (4.0 * array.size)
            - (
                24.0 * z_value
                - 132.0 * z_value**2
                + 76.0 * z_value**3
                - 9.0 * z_value**4
            )
            / (288.0 * array.size**2)
        )
    mean_phase = float(np.mod(np.angle(vector), 2.0 * np.pi))
    return {
        "n": int(array.size),
        "period": float(period),
        "mean_phase": mean_phase * period / (2.0 * np.pi),
        "resultant_length": resultant,
        "rayleigh_p_approx": min(1.0, max(0.0, float(p_value))),
    }


def _select_scan_rows(
    *,
    fractions: np.ndarray,
    alignments: np.ndarray,
    networks: np.ndarray,
    affected_networks: set[int],
    control_networks: set[int],
    minimum_fraction: float,
    minimum_alignment: float,
    minimum_affected_networks: int,
    max_scans: int,
    null_scans: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    affected_columns = np.flatnonzero(
        np.isin(networks, sorted(affected_networks))
    )
    control_columns = np.flatnonzero(
        np.isin(networks, sorted(control_networks))
    )
    records: list[dict[str, Any]] = []
    for scan_row in range(fractions.shape[0]):
        affected_values = fractions[scan_row, affected_columns]
        controls = fractions[scan_row, control_columns]
        qualifying = (
            np.isfinite(affected_values)
            & (affected_values >= minimum_fraction)
            & np.isfinite(alignments[scan_row, affected_columns])
            & (alignments[scan_row, affected_columns] >= minimum_alignment)
        )
        affected_median = float(np.nanmedian(affected_values))
        control_median = float(np.nanmedian(controls))
        records.append(
            {
                "scan_row_zero_based": int(scan_row),
                "n_affected_rtc_candidates": int(np.count_nonzero(qualifying)),
                "affected_median_step_detector_fraction": affected_median,
                "control_median_step_detector_fraction": control_median,
                "affected_minus_control_step_fraction": (
                    affected_median - control_median
                ),
            }
        )
    qualifying_records = [
        row
        for row in records
        if row["n_affected_rtc_candidates"] >= minimum_affected_networks
    ]
    if qualifying_records:
        ordered = sorted(
            qualifying_records,
            key=lambda row: (
                int(row["n_affected_rtc_candidates"]),
                float(row["affected_minus_control_step_fraction"]),
            ),
            reverse=True,
        )
        chosen = ordered[:max_scans]
    else:
        chosen = sorted(
            records,
            key=lambda row: float(
                row["affected_minus_control_step_fraction"]
            ),
            reverse=True,
        )[:null_scans]
    selected_rows = {
        int(row["scan_row_zero_based"]) for row in chosen
    }
    for row in records:
        row["selected_for_raw_analysis"] = (
            int(row["scan_row_zero_based"]) in selected_rows
        )
    return sorted(selected_rows), records


def _read_observation_scan_inputs(
    *,
    rtc_path: Path,
    telescope_path: Path,
) -> dict[str, Any]:
    with netCDF4.Dataset(telescope_path) as telescope:
        telescope_time = np.asarray(
            telescope.variables["Data.TelescopeBackend.TelTime"][:],
            dtype=float,
        )
    with netCDF4.Dataset(rtc_path) as rtc:
        output_scan = np.asarray(
            rtc.variables["output_scan_index"][:], dtype=int
        )
        durations = np.asarray(
            rtc.variables["scan_duration_s"][:], dtype=float
        )
        networks = np.asarray(
            rtc.variables["rtc_diag_network_ids"][:], dtype=int
        )
        fractions = np.asarray(
            rtc.variables["rtc_network_step_det_frac"][:], dtype=float
        )
        alignments = np.asarray(
            rtc.variables["rtc_network_step_alignment_frac"][:],
            dtype=float,
        )
    intervals = _scan_intervals_from_times(telescope_time, durations)
    return {
        "output_scan": output_scan,
        "durations": durations,
        "networks": networks,
        "fractions": fractions,
        "alignments": alignments,
        "intervals": intervals,
    }


def _global_sample_index(
    receive_time_sec: np.ndarray,
    event_time_sec: float,
) -> int:
    index = int(np.searchsorted(receive_time_sec, event_time_sec))
    if index <= 0:
        return 0
    if index >= receive_time_sec.size:
        return int(receive_time_sec.size - 1)
    if abs(receive_time_sec[index - 1] - event_time_sec) < abs(
        receive_time_sec[index] - event_time_sec
    ):
        return index - 1
    return index


def _make_timing_figure(
    path: Path,
    *,
    candidate_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        constrained_layout=True,
    )
    ax_raster, ax_pps, ax_lag, ax_modulus = axes.flat
    qualifying = [
        row
        for row in candidate_rows
        if row["event_cluster_id"] is not None
    ]
    obsnums = sorted({int(row["obsnum"]) for row in qualifying})
    for obs_index, obsnum in enumerate(obsnums):
        selected = [row for row in qualifying if int(row["obsnum"]) == obsnum]
        start = min(
            float(row["observation_start_time_unix_sec"])
            for row in selected
        )
        for row in selected:
            ax_raster.scatter(
                (
                    float(row["raw_event_time_unix_sec"]) - start
                )
                / 60.0,
                int(row["network"]) + 0.08 * obs_index,
                color=f"C{obs_index}",
                s=16,
                alpha=0.75,
            )
        ax_raster.plot([], [], color=f"C{obs_index}", label=str(obsnum))
    ax_raster.set_xlabel("elapsed observation time (min)")
    ax_raster.set_ylabel("network ID")
    ax_raster.set_title("Raw candidates assigned to cross-network clusters")
    ax_raster.legend(title="obsnum", fontsize=8)

    phases = [
        float(row["cluster_time_modulo_1s_sec"]) for row in cluster_rows
    ]
    ax_pps.hist(phases, bins=np.linspace(0.0, 1.0, 17), color="C0", alpha=0.8)
    ax_pps.set_xlabel("cluster candidate time modulo 1 s (s)")
    ax_pps.set_ylabel("cluster count")
    ax_pps.set_title("Exploratory PPS-phase test")

    residual_ms = [
        1.0e3 * float(row["candidate_minus_cluster_time_sec"])
        for row in qualifying
    ]
    ax_lag.hist(residual_ms, bins=25, color="C1", alpha=0.8)
    ax_lag.set_xlabel("network candidate minus cluster median (ms)")
    ax_lag.set_ylabel("network-candidate count")
    ax_lag.set_title("Cross-network candidate-time residuals")

    modulus = [
        row
        for row in timing_rows
        if row["test_kind"] == "global_sample_index_modulus"
    ]
    ax_modulus.plot(
        [int(row["modulus_samples"]) for row in modulus],
        [float(row["resultant_length"]) for row in modulus],
        "o-",
    )
    ax_modulus.set_xscale("log", base=2)
    ax_modulus.set_ylim(0.0, 1.0)
    ax_modulus.set_xlabel("sample-index modulus")
    ax_modulus.set_ylabel("circular resultant length")
    ax_modulus.set_title("Exploratory sample-boundary concentration")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_mode_figure(
    path: Path,
    *,
    fit_rows: list[dict[str, Any]],
    affected_networks: set[int],
) -> None:
    valid = [
        row
        for row in fit_rows
        if row["fit_status"] == "fit"
        and row["phase_r2"] is not None
        and row["frequency_r2"] is not None
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        constrained_layout=True,
    )
    ax_scatter, ax_network = axes
    for row in valid:
        network = int(row["network"])
        marker = "o" if network in affected_networks else "s"
        ax_scatter.scatter(
            float(row["phase_r2"]),
            float(row["frequency_r2"]),
            c=[network],
            cmap="turbo",
            vmin=0,
            vmax=12,
            marker=marker,
            alpha=0.7,
        )
    ax_scatter.plot([0, 1], [0, 1], color="0.5", linestyle=":")
    ax_scatter.set_xlim(-0.03, 1.03)
    ax_scatter.set_ylim(-0.03, 1.03)
    ax_scatter.set_xlabel("common phase-rotation R²")
    ax_scatter.set_ylabel("frequency-like R²")
    ax_scatter.set_title("Single-mode competition")

    networks = sorted({int(row["network"]) for row in valid})
    med_phase = []
    med_frequency = []
    med_gain = []
    for network in networks:
        selected = [row for row in valid if int(row["network"]) == network]
        med_phase.append(float(np.median([row["phase_r2"] for row in selected])))
        med_frequency.append(
            float(np.median([row["frequency_r2"] for row in selected]))
        )
        med_gain.append(float(np.median([row["gain_r2"] for row in selected])))
    positions = np.arange(len(networks))
    width = 0.25
    ax_network.bar(positions - width, med_gain, width, label="gain")
    ax_network.bar(positions, med_phase, width, label="phase")
    ax_network.bar(
        positions + width,
        med_frequency,
        width,
        label="frequency-like",
    )
    ax_network.set_xticks(positions, networks)
    ax_network.set_ylim(0.0, 1.0)
    ax_network.set_xlabel("network ID")
    ax_network.set_ylabel("median event R²")
    ax_network.set_title("Model support by network")
    ax_network.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reduction-root", type=Path, required=True)
    parser.add_argument("--apt-root", type=Path, required=True)
    parser.add_argument(
        "--obsnums",
        type=int,
        nargs="+",
        default=list(DEFAULT_OBSNUMS),
    )
    parser.add_argument(
        "--affected-networks",
        type=int,
        nargs="+",
        default=list(DEFAULT_AFFECTED_NETWORKS),
    )
    parser.add_argument(
        "--control-networks",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTROL_NETWORKS),
    )
    parser.add_argument("--subobsnum", type=int, default=0)
    parser.add_argument("--raw-file-scan", type=int, default=2)
    parser.add_argument("--max-scans-per-observation", type=int, default=18)
    parser.add_argument("--null-scans-per-observation", type=int, default=2)
    parser.add_argument("--rtc-minimum-fraction", type=float, default=0.10)
    parser.add_argument("--rtc-minimum-alignment", type=float, default=0.50)
    parser.add_argument("--rtc-minimum-affected-networks", type=int, default=3)
    parser.add_argument("--raw-event-fraction", type=float, default=0.10)
    parser.add_argument("--cluster-tolerance-sec", type=float, default=0.35)
    parser.add_argument("--cluster-minimum-affected-networks", type=int, default=3)
    parser.add_argument("--sigma-threshold", type=float, default=8.0)
    parser.add_argument("--min-phase-mrad", type=float, default=5.0)
    parser.add_argument("--pre-window-sec", type=float, default=0.20)
    parser.add_argument("--guard-window-sec", type=float, default=0.05)
    parser.add_argument("--post-window-sec", type=float, default=0.20)
    parser.add_argument("--sweep-half-window-steps", type=int, default=3)
    parser.add_argument(
        "--sample-moduli",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_MODULI),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    affected = {int(value) for value in args.affected_networks}
    controls = {int(value) for value in args.control_networks}
    if affected & controls:
        raise ValueError("affected and control network sets overlap")
    if args.max_scans_per_observation <= 0:
        raise ValueError("--max-scans-per-observation must be positive")
    if args.null_scans_per_observation < 0:
        raise ValueError("--null-scans-per-observation cannot be negative")
    if any(int(value) <= 1 for value in args.sample_moduli):
        raise ValueError("--sample-moduli entries must exceed one")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    tone_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    sweep_cache: dict[tuple[int, int], SweepModel] = {}
    raw_time_cache: dict[Path, np.ndarray] = {}
    cluster_serial = 0

    for obsnum_value in args.obsnums:
        obsnum = int(obsnum_value)
        rtc_path = _science_rtc_path(args.reduction_root, obsnum)
        telescope_path = _find_one(
            args.data_root,
            f"tel_toltec_*_{obsnum}_00_0002.nc",
        )
        apt_path = args.apt_root / f"apt_{obsnum}_matched.ecsv"
        if not apt_path.is_file():
            raise FileNotFoundError(apt_path)
        apt = Table.read(apt_path)
        scan_inputs = _read_observation_scan_inputs(
            rtc_path=rtc_path,
            telescope_path=telescope_path,
        )
        networks = np.asarray(scan_inputs["networks"], dtype=int)
        selected_scan_rows, obs_selection = _select_scan_rows(
            fractions=np.asarray(scan_inputs["fractions"], dtype=float),
            alignments=np.asarray(scan_inputs["alignments"], dtype=float),
            networks=networks,
            affected_networks=affected,
            control_networks=controls,
            minimum_fraction=float(args.rtc_minimum_fraction),
            minimum_alignment=float(args.rtc_minimum_alignment),
            minimum_affected_networks=int(
                args.rtc_minimum_affected_networks
            ),
            max_scans=int(args.max_scans_per_observation),
            null_scans=int(args.null_scans_per_observation),
        )
        for row in obs_selection:
            selection_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "obsnum": obsnum,
                    "citlali_scan_one_based": int(
                        scan_inputs["output_scan"][
                            row["scan_row_zero_based"]
                        ]
                    ),
                    **row,
                }
            )
        raw_paths: dict[int, Path] = {}
        for network_value in networks:
            network = int(network_value)
            raw_path = iq_tool._find_raw_file(
                args.data_root,
                network=network,
                obsnum=obsnum,
                subobsnum=int(args.subobsnum),
                raw_file_scan=int(args.raw_file_scan),
            )
            if raw_path is None:
                raise FileNotFoundError(
                    f"obs {obsnum}: no raw file for nw{network}"
                )
            raw_paths[network] = raw_path
            if raw_path not in raw_time_cache:
                with netCDF4.Dataset(raw_path) as raw:
                    raw_time_cache[raw_path] = np.asarray(
                        raw.variables["Data.Toltec.RecvTime"][:],
                        dtype=float,
                    )
            sweep_cache[(obsnum, network)] = _load_sweep_model(
                data_root=args.data_root,
                raw_path=raw_path,
                obsnum=obsnum,
                network=network,
                half_window_steps=int(args.sweep_half_window_steps),
            )
        input_rows.append(
            {
                "obsnum": obsnum,
                "rtc_path": str(rtc_path),
                "telescope_path": str(telescope_path),
                "apt_path": str(apt_path),
                "raw_paths": {
                    str(network): str(path)
                    for network, path in raw_paths.items()
                },
                "sweep_paths": {
                    str(network): str(sweep_cache[(obsnum, network)].path)
                    for network in networks
                },
            }
        )
        print(
            f"obs {obsnum}: selected {len(selected_scan_rows)} "
            f"of {len(scan_inputs['intervals'])} chunks",
            flush=True,
        )

        for scan_number_index, scan_row in enumerate(selected_scan_rows, start=1):
            interval = scan_inputs["intervals"][scan_row]
            scan_start = float(interval["start_time_unix_sec"])
            scan_end = float(interval["end_time_unix_sec"])
            citlali_scan = int(scan_inputs["output_scan"][scan_row])
            network_data: dict[int, iq_tool.NetworkData] = {}
            scan_candidates: list[dict[str, Any]] = []
            for network_value in networks:
                network = int(network_value)
                raw_path = raw_paths[network]
                data = iq_tool._load_network(
                    raw_path,
                    network=network,
                    scan_start_sec=scan_start,
                    scan_end_sec=scan_end,
                    apt=apt,
                )
                network_data[network] = data
                candidate = _raw_candidate(
                    data,
                    sigma_threshold=float(args.sigma_threshold),
                    min_phase_mrad=float(args.min_phase_mrad),
                    pre_window_sec=float(args.pre_window_sec),
                    guard_window_sec=float(args.guard_window_sec),
                    post_window_sec=float(args.post_window_sec),
                )
                event_time = scan_start + candidate.event_sec
                global_index = _global_sample_index(
                    raw_time_cache[raw_path],
                    event_time,
                )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "obsnum": obsnum,
                    "citlali_scan_one_based": citlali_scan,
                    "scan_row_zero_based": int(scan_row),
                    "network": network,
                    "array": _array_name(network),
                    "rack": _rack(network),
                    "observation_start_time_unix_sec": float(
                        raw_time_cache[raw_path][0]
                    ),
                    "scan_start_time_unix_sec": scan_start,
                    "scan_start_time_utc": _utc_iso(scan_start),
                    "raw_event_sec_within_scan": candidate.event_sec,
                    "raw_event_time_unix_sec": event_time,
                    "raw_event_time_utc": _utc_iso(event_time),
                    "raw_event_time_modulo_1s_sec": float(
                        np.mod(event_time, 1.0)
                    ),
                    "raw_event_sample_within_chunk": candidate.event_row,
                    "raw_event_global_sample_index": global_index,
                    "raw_coherent_same_sign_fraction": (
                        candidate.coherent_same_sign_fraction
                    ),
                    "raw_strong_phase_fraction": (
                        candidate.strong_phase_fraction
                    ),
                    "n_apt_usable_tones": candidate.n_apt_usable_tones,
                    "n_strong_phase_tones": candidate.n_strong_phase_tones,
                    "event_cluster_id": None,
                    "candidate_minus_cluster_time_sec": None,
                    "raw_path": str(raw_path),
                }
                scan_candidates.append(record)
                candidate_rows.append(record)

            groups = _cluster_candidates(
                scan_candidates,
                affected_networks=affected,
                minimum_fraction=float(args.raw_event_fraction),
                tolerance_sec=float(args.cluster_tolerance_sec),
                minimum_affected_networks=int(
                    args.cluster_minimum_affected_networks
                ),
            )
            for group in groups:
                cluster_serial += 1
                cluster_id = f"c{cluster_serial:04d}"
                times = np.asarray(
                    [row["raw_event_time_unix_sec"] for row in group],
                    dtype=float,
                )
                cluster_time = float(np.median(times))
                participant_networks = sorted(
                    {int(row["network"]) for row in group}
                )
                participant_racks = sorted(
                    {_rack(network) for network in participant_networks}
                )
                for row in group:
                    row["event_cluster_id"] = cluster_id
                    row["candidate_minus_cluster_time_sec"] = (
                        float(row["raw_event_time_unix_sec"]) - cluster_time
                    )
                cluster_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "event_cluster_id": cluster_id,
                        "obsnum": obsnum,
                        "citlali_scan_one_based": citlali_scan,
                        "scan_row_zero_based": int(scan_row),
                        "cluster_time_unix_sec": cluster_time,
                        "cluster_time_utc": _utc_iso(cluster_time),
                        "cluster_time_modulo_1s_sec": float(
                            np.mod(cluster_time, 1.0)
                        ),
                        "network_count": len(participant_networks),
                        "affected_network_count": len(
                            set(participant_networks) & affected
                        ),
                        "control_network_count": len(
                            set(participant_networks) & controls
                        ),
                        "networks": " ".join(
                            str(value) for value in participant_networks
                        ),
                        "racks": " ".join(participant_racks),
                        "cross_rack": len(participant_racks) > 1,
                        "candidate_span_sec": float(np.max(times) - np.min(times)),
                    }
                )

                shared_event_sec = cluster_time - scan_start
                for network_value in networks:
                    network = int(network_value)
                    data = network_data[network]
                    sweep = sweep_cache[(obsnum, network)]
                    try:
                        event_vector = _extract_event_vector(
                            data,
                            event_sec=shared_event_sec,
                            sigma_threshold=float(args.sigma_threshold),
                            min_phase_mrad=float(args.min_phase_mrad),
                            pre_window_sec=float(args.pre_window_sec),
                            guard_window_sec=float(args.guard_window_sec),
                            post_window_sec=float(args.post_window_sec),
                        )
                        frequency_direction = np.full(
                            event_vector.fractional_change.shape,
                            np.nan + 1j * np.nan,
                        )
                        z_pre_equivalent = sweep.z_at_probe
                        np.divide(
                            sweep.dz_df,
                            z_pre_equivalent,
                            out=frequency_direction,
                            where=np.abs(z_pre_equivalent) > 0.0,
                        )
                        model_valid = (
                            event_vector.valid
                            & sweep.valid
                            & np.isfinite(frequency_direction.real)
                            & np.isfinite(frequency_direction.imag)
                        )
                        fit_mask = event_vector.responsive & model_valid
                        n_valid_event = int(
                            np.count_nonzero(event_vector.valid)
                        )
                        positive_fraction = (
                            np.count_nonzero(
                                event_vector.responsive
                                & (event_vector.phase_change_rad > 0.0)
                            )
                            / n_valid_event
                        )
                        negative_fraction = (
                            np.count_nonzero(
                                event_vector.responsive
                                & (event_vector.phase_change_rad < 0.0)
                            )
                            / n_valid_event
                        )
                        shared_coherent_fraction = float(
                            max(positive_fraction, negative_fraction)
                        )
                        if (
                            np.count_nonzero(fit_mask) < 8
                            or shared_coherent_fraction
                            < float(args.raw_event_fraction)
                        ):
                            fit_population = "phase_responsive"
                            modes = {}
                            fit_status = "insufficient_coherent_response"
                        else:
                            fit_population = "phase_responsive"
                            modes = _fit_event_modes(
                                event_vector.fractional_change,
                                frequency_direction,
                                fit_mask,
                            )
                            fit_status = "fit"
                        error_message = None
                    except (ValueError, FloatingPointError) as error:
                        event_vector = None
                        frequency_direction = np.asarray([], dtype=complex)
                        model_valid = np.asarray([], dtype=bool)
                        fit_mask = np.asarray([], dtype=bool)
                        fit_population = "unavailable"
                        modes = {}
                        fit_status = "unavailable"
                        error_message = str(error)
                        shared_coherent_fraction = math.nan
                    fit_record = {
                        "schema_version": SCHEMA_VERSION,
                        "event_cluster_id": cluster_id,
                        "obsnum": obsnum,
                        "citlali_scan_one_based": citlali_scan,
                        "network": network,
                        "array": _array_name(network),
                        "rack": _rack(network),
                        "network_group": (
                            "affected"
                            if network in affected
                            else "control"
                            if network in controls
                            else "unclassified"
                        ),
                        "cluster_time_unix_sec": cluster_time,
                        "cluster_time_utc": _utc_iso(cluster_time),
                        "fit_status": fit_status,
                        "fit_population": fit_population,
                        "n_apt_usable_tones": int(
                            np.count_nonzero(data.apt_usable)
                        ),
                        "n_phase_responsive_tones": (
                            int(np.count_nonzero(event_vector.responsive))
                            if event_vector is not None
                            else 0
                        ),
                        "phase_responsive_fraction": (
                            float(
                                np.count_nonzero(event_vector.responsive)
                                / np.count_nonzero(event_vector.valid)
                            )
                            if event_vector is not None
                            and np.count_nonzero(event_vector.valid) > 0
                            else None
                        ),
                        "phase_coherent_same_sign_fraction": _finite_or_none(
                            shared_coherent_fraction
                        ),
                        **modes,
                        "error_message": error_message,
                        "raw_path": str(data.path),
                        "sweep_path": str(sweep.path),
                    }
                    fit_rows.append(fit_record)
                    if event_vector is not None:
                        for tone in np.flatnonzero(model_valid):
                            tone_rows.append(
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "event_cluster_id": cluster_id,
                                    "obsnum": obsnum,
                                    "citlali_scan_one_based": citlali_scan,
                                    "network": network,
                                    "tone_slot_zero_based": int(tone),
                                    "uid": int(data.uid[tone]),
                                    "lo_center_frequency_hz": float(
                                        sweep.lo_center_frequency_hz
                                    ),
                                    "tone_offset_frequency_hz": float(
                                        sweep.tone_offset_frequency_hz[tone]
                                    ),
                                    "probe_frequency_hz": float(
                                        sweep.probe_frequency_hz[tone]
                                    ),
                                    "apt_tone_frequency_hz": float(
                                        data.tone_frequency_hz[tone]
                                    ),
                                    "network_fit_status": fit_status,
                                    "phase_responsive": bool(
                                        event_vector.responsive[tone]
                                    ),
                                    "fractional_change_real": float(
                                        event_vector.fractional_change[tone].real
                                    ),
                                    "fractional_change_imag": float(
                                        event_vector.fractional_change[tone].imag
                                    ),
                                    "phase_change_mrad": float(
                                        1.0e3
                                        * event_vector.phase_change_rad[tone]
                                    ),
                                    "phase_threshold_mrad": float(
                                        1.0e3
                                        * event_vector.phase_threshold_rad[tone]
                                    ),
                                    "frequency_direction_real_per_hz": float(
                                        frequency_direction[tone].real
                                    ),
                                    "frequency_direction_imag_per_hz": float(
                                        frequency_direction[tone].imag
                                    ),
                                }
                            )
            print(
                f"  chunk {scan_number_index}/{len(selected_scan_rows)} "
                f"(Citlali scan {citlali_scan}): {len(groups)} clusters",
                flush=True,
            )

    timing_rows: list[dict[str, Any]] = []
    cluster_times = [
        float(row["cluster_time_unix_sec"]) for row in cluster_rows
    ]
    pps = _rayleigh_summary(cluster_times, period=1.0)
    timing_rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "test_kind": "cluster_time_modulo_1s",
            "modulus_samples": None,
            **pps,
            "interpretation": "exploratory_operational_candidate_times",
        }
    )
    clustered_candidates = [
        row
        for row in candidate_rows
        if row["event_cluster_id"] is not None
    ]
    sample_indices = []
    for cluster in cluster_rows:
        cluster_id = cluster["event_cluster_id"]
        indices = [
            int(row["raw_event_global_sample_index"])
            for row in clustered_candidates
            if row["event_cluster_id"] == cluster_id
        ]
        if indices:
            sample_indices.append(float(np.median(indices)))
    for modulus in args.sample_moduli:
        summary = _rayleigh_summary(sample_indices, period=float(modulus))
        timing_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "test_kind": "global_sample_index_modulus",
                "modulus_samples": int(modulus),
                **summary,
                "interpretation": (
                    "exploratory_multiple_moduli_no_hardware_boundary_assumed"
                ),
            }
        )

    outputs = {
        "scan_selection": "science_event_scan_selection.csv",
        "raw_candidates": "science_raw_event_candidates.csv",
        "event_clusters": "science_raw_event_clusters.csv",
        "event_vector_fits": "science_event_vector_fits.csv",
        "tone_vectors": "science_event_tone_vectors.csv",
        "timing_tests": "science_event_timing_tests.csv",
        "timing_figure": "science_event_timing.png",
        "mode_figure": "science_event_vector_modes.png",
    }
    _write_csv(args.output_dir / outputs["scan_selection"], selection_rows)
    _write_csv(args.output_dir / outputs["raw_candidates"], candidate_rows)
    if cluster_rows:
        _write_csv(args.output_dir / outputs["event_clusters"], cluster_rows)
    else:
        outputs.pop("event_clusters")
    if fit_rows:
        _write_csv(args.output_dir / outputs["event_vector_fits"], fit_rows)
    else:
        outputs.pop("event_vector_fits")
    if tone_rows:
        _write_csv(args.output_dir / outputs["tone_vectors"], tone_rows)
    else:
        outputs.pop("tone_vectors")
    _write_csv(args.output_dir / outputs["timing_tests"], timing_rows)
    if cluster_rows and fit_rows:
        _make_timing_figure(
            args.output_dir / outputs["timing_figure"],
            candidate_rows=candidate_rows,
            cluster_rows=cluster_rows,
            timing_rows=timing_rows,
        )
        _make_mode_figure(
            args.output_dir / outputs["mode_figure"],
            fit_rows=fit_rows,
            affected_networks=affected,
        )
    else:
        outputs.pop("timing_figure")
        outputs.pop("mode_figure")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "description": (
            "Raw-I/Q cross-network candidate timing and empirical "
            "complex-vector mode decomposition"
        ),
        "obsnums": [int(value) for value in args.obsnums],
        "affected_networks": sorted(affected),
        "control_networks": sorted(controls),
        "thresholds": {
            "max_scans_per_observation": int(
                args.max_scans_per_observation
            ),
            "null_scans_per_observation": int(
                args.null_scans_per_observation
            ),
            "rtc_minimum_fraction": float(args.rtc_minimum_fraction),
            "rtc_minimum_alignment": float(args.rtc_minimum_alignment),
            "rtc_minimum_affected_networks": int(
                args.rtc_minimum_affected_networks
            ),
            "raw_event_fraction": float(args.raw_event_fraction),
            "cluster_tolerance_sec": float(args.cluster_tolerance_sec),
            "cluster_minimum_affected_networks": int(
                args.cluster_minimum_affected_networks
            ),
            "sigma_threshold": float(args.sigma_threshold),
            "min_phase_mrad": float(args.min_phase_mrad),
            "pre_window_sec": float(args.pre_window_sec),
            "guard_window_sec": float(args.guard_window_sec),
            "post_window_sec": float(args.post_window_sec),
            "sweep_half_window_steps": int(args.sweep_half_window_steps),
            "sample_moduli": [int(value) for value in args.sample_moduli],
        },
        "semantics": {
            "target_sweep_role": (
                "Empirical d(I+iQ)/df basis only; no tune-quality or "
                "tune-causality inference"
            ),
            "candidate_time": (
                "Operational maximum coherent pre/post raw-phase change; "
                "not an exact physical onset"
            ),
            "model_coefficients": (
                "Real least-squares coefficients on fractional complex "
                "change for gain, phase rotation, and frequency-like motion"
            ),
            "model_r2": (
                "Zero-baseline explained complex-change energy on the stated "
                "fit population; descriptive, not statistical significance"
            ),
            "tone_vectors": (
                "One row per event/network/model-valid APT tone; "
                "phase_responsive marks membership in the thresholded fit "
                "population; tone_offset_frequency_hz is the signed digital "
                "tone offset from lo_center_frequency_hz"
            ),
            "timing_tests": (
                "Exploratory circular concentration tests. PPS telemetry and "
                "hardware block-boundary metadata are unavailable."
            ),
        },
        "inputs": input_rows,
        "counts": {
            "selected_raw_candidate_rows": len(candidate_rows),
            "event_clusters": len(cluster_rows),
            "event_vector_fit_rows": len(fit_rows),
            "tone_vector_rows": len(tone_rows),
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
