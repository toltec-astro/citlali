#!/usr/bin/env python3
"""Diagnose raw TolTEC I/Q coherence around a pointing-timestream event.

This tool deliberately operates on raw network files and a persisted Citlali
RTC timestream.  It preserves network IDs and joins raw tone slots to detector
identity through ``kids_tone`` and ``uid`` in the supplied APT.

The diagnostic distinguishes:

* a network-coherent complex rotation or gain change;
* detector-specific changes with dispersed directions or onset times; and
* a large processed RTC event whose raw I/Q counterpart is small.

The reported "detector onset" is an operational threshold crossing in the raw
phase residual.  It is not the RTC dominant-step sample and is not presented as
an exact physical transition time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "citlali-pointing-iq-mpl-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402


RAD_TO_ARCSEC = 206_264.806_247_096_36


@dataclass(frozen=True)
class EventIdentity:
    obsnum: int
    subobsnum: int
    raw_file_scan: int
    citlali_scan: int
    network: int
    uid: int
    tone: int
    scan_start_sec: float
    scan_end_sec: float
    rtc_event_sec: float
    rtc_step_score: float


@dataclass
class NetworkData:
    network: int
    path: Path
    time_sec: np.ndarray
    complex_iq: np.ndarray
    uid: np.ndarray
    tone_frequency_hz: np.ndarray
    apt_usable: np.ndarray
    phase_residual_rad: np.ndarray | None = None
    amplitude_fraction: np.ndarray | None = None
    phase_sigma_rad: np.ndarray | None = None
    phase_threshold_rad: np.ndarray | None = None
    tone_onset_sec: np.ndarray | None = None
    event_phase_shift_rad: np.ndarray | None = None
    event_amplitude_fraction: np.ndarray | None = None
    event_complex_change: np.ndarray | None = None
    strong_event_tone: np.ndarray | None = None


def _as_scalar(ds: netCDF4.Dataset, name: str) -> int:
    if name not in ds.variables:
        raise KeyError(f"missing required scalar variable {name!r}")
    return int(np.asarray(ds.variables[name][...]).item())


def _robust_sigma(values: np.ndarray, *, axis: int = 0) -> np.ndarray:
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis)
    return 1.482_602_218_505_602 * mad


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _find_scan_row(ds: netCDF4.Dataset, citlali_scan: int) -> int:
    if "output_scan_index" not in ds.variables:
        raise KeyError("RTC timestream is missing output_scan_index")
    output_scan = np.asarray(ds.variables["output_scan_index"][:], dtype=int)
    rows = np.flatnonzero(output_scan == int(citlali_scan))
    if rows.size != 1:
        raise ValueError(
            f"expected one RTC row for Citlali scan {citlali_scan}, found {rows.size}"
        )
    return int(rows[0])


def _load_event_identity(
    rtc_path: Path,
    *,
    obsnum: int,
    subobsnum: int,
    raw_file_scan: int,
    citlali_scan: int,
    network: int,
    uid: int,
    tone: int,
) -> EventIdentity:
    with netCDF4.Dataset(rtc_path) as ds:
        scan_row = _find_scan_row(ds, citlali_scan)
        i0, i1 = np.asarray(ds.variables["scan_indices"][scan_row], dtype=int)
        tel_time = np.asarray(ds.variables["TelTime"][:], dtype=float)
        apt_uid = np.asarray(ds.variables["apt_uid"][:], dtype=int)
        uid_rows = np.flatnonzero(apt_uid == int(uid))
        if uid_rows.size != 1:
            raise ValueError(f"expected exactly one RTC detector with uid={uid}")
        detector_row = int(uid_rows[0])
        step_sample = int(ds.variables["rtc_step_sample"][scan_row, detector_row])
        step_score = float(ds.variables["rtc_step_score"][scan_row, detector_row])
        n_scan_samples = int(i1 - i0 + 1)
        if not 0 <= step_sample < n_scan_samples:
            raise ValueError(
                f"RTC step sample {step_sample} lies outside scan length "
                f"{n_scan_samples}"
            )
        scan_start = float(tel_time[i0])
        scan_end = float(tel_time[i1])
        event_sec = float(tel_time[i0 + step_sample] - scan_start)
    return EventIdentity(
        obsnum=int(obsnum),
        subobsnum=int(subobsnum),
        raw_file_scan=int(raw_file_scan),
        citlali_scan=int(citlali_scan),
        network=int(network),
        uid=int(uid),
        tone=int(tone),
        scan_start_sec=scan_start,
        scan_end_sec=scan_end,
        rtc_event_sec=event_sec,
        rtc_step_score=step_score,
    )


def _find_raw_file(
    data_root: Path,
    *,
    network: int,
    obsnum: int,
    subobsnum: int,
    raw_file_scan: int,
) -> Path | None:
    pattern = (
        f"toltec{network}_{obsnum:06d}_{subobsnum:03d}_"
        f"{raw_file_scan:04d}_*.nc"
    )
    candidates = sorted(data_root.glob(pattern))
    matches: list[Path] = []
    for path in candidates:
        with netCDF4.Dataset(path) as ds:
            if (
                _as_scalar(ds, "Header.Toltec.RoachIndex") == network
                and _as_scalar(ds, "Header.Toltec.ObsNum") == obsnum
                and _as_scalar(ds, "Header.Toltec.SubObsNum") == subobsnum
                and _as_scalar(ds, "Header.Toltec.ScanNum") == raw_file_scan
            ):
                matches.append(path)
    if not matches:
        return None
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"expected one raw file for nw{network}, found {len(matches)}: {names}"
        )
    return matches[0]


def _apt_arrays(
    apt: Table,
    *,
    network: int,
    n_tones: int,
    raw_tone_frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uid = np.full(n_tones, -1, dtype=int)
    frequency_hz = np.asarray(raw_tone_frequency_hz, dtype=float).copy()
    usable = np.zeros(n_tones, dtype=bool)
    rows = apt[np.asarray(apt["nw"], dtype=int) == int(network)]
    for row in rows:
        tone_value = float(row["kids_tone"])
        if not np.isfinite(tone_value):
            continue
        tone = int(tone_value)
        if tone < 0 or tone >= n_tones:
            continue
        uid_value = float(row["uid"])
        if np.isfinite(uid_value):
            uid[tone] = int(uid_value)
        if "tone_freq" in rows.colnames:
            value = float(row["tone_freq"])
            if np.isfinite(value):
                frequency_hz[tone] = value
        kids_flag = float(row["kids_flag"]) if "kids_flag" in rows.colnames else 0.0
        map_flag = float(row["flag"]) if "flag" in rows.colnames else 0.0
        usable[tone] = (
            np.isfinite(kids_flag)
            and np.isfinite(map_flag)
            and kids_flag == 0.0
            and map_flag == 0.0
        )
    return uid, frequency_hz, usable


def _load_network(
    path: Path,
    *,
    network: int,
    scan_start_sec: float,
    scan_end_sec: float,
    apt: Table,
) -> NetworkData:
    with netCDF4.Dataset(path) as ds:
        recv_time = np.asarray(ds.variables["Data.Toltec.RecvTime"][:], dtype=float)
        in_scan = (recv_time >= scan_start_sec) & (recv_time <= scan_end_sec)
        if np.count_nonzero(in_scan) < 32:
            raise ValueError(f"{path.name}: too few raw samples in RTC scan interval")
        i_data = np.asarray(ds.variables["Data.Toltec.Is"][in_scan, :], dtype=float)
        q_data = np.asarray(ds.variables["Data.Toltec.Qs"][in_scan, :], dtype=float)
        raw_frequency = np.asarray(
            ds.variables["Header.Toltec.ToneFreq"][0, :], dtype=float
        )
    if i_data.shape != q_data.shape:
        raise ValueError(f"{path.name}: I and Q shapes differ")
    uid, frequency_hz, usable = _apt_arrays(
        apt,
        network=network,
        n_tones=i_data.shape[1],
        raw_tone_frequency_hz=raw_frequency,
    )
    return NetworkData(
        network=int(network),
        path=path,
        time_sec=recv_time[in_scan] - float(scan_start_sec),
        complex_iq=i_data + 1j * q_data,
        uid=uid,
        tone_frequency_hz=frequency_hz,
        apt_usable=usable,
    )


def _first_sustained_crossing(
    time_sec: np.ndarray,
    residual: np.ndarray,
    threshold: np.ndarray,
    *,
    search_start_sec: float,
    search_end_sec: float,
    sustain_samples: int,
) -> np.ndarray:
    search = np.flatnonzero(
        (time_sec >= float(search_start_sec))
        & (time_sec <= float(search_end_sec))
    )
    onset = np.full(residual.shape[1], np.nan, dtype=float)
    if search.size < sustain_samples:
        return onset
    kernel = np.ones(int(sustain_samples), dtype=int)
    for tone in range(residual.shape[1]):
        hit = np.abs(residual[search, tone]) > threshold[tone]
        sustained = np.convolve(hit.astype(int), kernel, mode="valid")
        found = np.flatnonzero(sustained >= sustain_samples)
        if found.size:
            onset[tone] = float(time_sec[search[int(found[0])]])
    return onset


def _analyze_network(
    data: NetworkData,
    *,
    rtc_event_sec: float,
    sigma_threshold: float,
    min_phase_mrad: float,
    sustain_samples: int,
) -> None:
    time_sec = data.time_sec
    phase = np.unwrap(np.angle(data.complex_iq), axis=0)
    amplitude = np.abs(data.complex_iq)

    baseline_start = max(float(time_sec[0]), 0.25)
    baseline_end = min(float(rtc_event_sec) - 0.70, float(time_sec[-1]))
    baseline = (time_sec >= baseline_start) & (time_sec <= baseline_end)
    if np.count_nonzero(baseline) < 32:
        raise ValueError(
            f"nw{data.network}: baseline interval has fewer than 32 samples"
        )

    x = time_sec[baseline]
    x_mean = float(np.mean(x))
    y = phase[baseline, :]
    y_mean = np.mean(y, axis=0)
    denominator = float(np.sum((x - x_mean) ** 2))
    slope = np.sum((x - x_mean)[:, None] * (y - y_mean), axis=0) / denominator
    phase_model = y_mean[None, :] + (
        time_sec[:, None] - x_mean
    ) * slope[None, :]
    phase_residual = phase - phase_model

    phase_sigma = _robust_sigma(
        np.diff(phase_residual[baseline, :], axis=0), axis=0
    ) / math.sqrt(2.0)
    min_phase_rad = float(min_phase_mrad) * 1.0e-3
    phase_threshold = np.maximum(
        float(sigma_threshold) * phase_sigma, min_phase_rad
    )

    amplitude_baseline = np.nanmedian(amplitude[baseline, :], axis=0)
    amplitude_fraction = amplitude / amplitude_baseline[None, :] - 1.0

    pre = (
        (time_sec >= float(rtc_event_sec) - 0.82)
        & (time_sec <= float(rtc_event_sec) - 0.50)
    )
    event = (
        (time_sec >= float(rtc_event_sec) - 0.14)
        & (time_sec <= float(rtc_event_sec) - 0.04)
    )
    if np.count_nonzero(pre) < 8 or np.count_nonzero(event) < 4:
        raise ValueError(f"nw{data.network}: event comparison windows are incomplete")
    z_pre = np.nanmedian(data.complex_iq[pre, :], axis=0)
    z_event = np.nanmedian(data.complex_iq[event, :], axis=0)
    complex_change = z_event / z_pre - 1.0
    phase_shift = np.angle(z_event / z_pre)
    amplitude_shift = np.abs(z_event) / np.abs(z_pre) - 1.0

    finite = (
        np.isfinite(phase_shift)
        & np.isfinite(amplitude_shift)
        & np.isfinite(phase_threshold)
        & (np.abs(z_pre) > 0)
    )
    strong = (
        finite
        & data.apt_usable
        & (np.abs(phase_shift) > phase_threshold)
    )
    onset = _first_sustained_crossing(
        time_sec,
        phase_residual,
        phase_threshold,
        search_start_sec=float(rtc_event_sec) - 0.90,
        search_end_sec=float(rtc_event_sec),
        sustain_samples=int(sustain_samples),
    )

    data.phase_residual_rad = phase_residual
    data.amplitude_fraction = amplitude_fraction
    data.phase_sigma_rad = phase_sigma
    data.phase_threshold_rad = phase_threshold
    data.tone_onset_sec = onset
    data.event_phase_shift_rad = phase_shift
    data.event_amplitude_fraction = amplitude_shift
    data.event_complex_change = complex_change
    data.strong_event_tone = strong


def _network_summary(
    data: NetworkData,
    *,
    rtc_event_sec: float,
    sustain_samples: int,
) -> dict[str, Any]:
    assert data.event_phase_shift_rad is not None
    assert data.event_amplitude_fraction is not None
    assert data.event_complex_change is not None
    assert data.strong_event_tone is not None
    assert data.phase_residual_rad is not None
    assert data.phase_threshold_rad is not None

    valid = (
        data.apt_usable
        & np.isfinite(data.event_phase_shift_rad)
        & np.isfinite(data.event_amplitude_fraction)
        & np.isfinite(data.event_complex_change.real)
        & np.isfinite(data.event_complex_change.imag)
    )
    strong = data.strong_event_tone
    n_valid = int(np.count_nonzero(valid))
    n_strong = int(np.count_nonzero(strong))
    phase_shift = data.event_phase_shift_rad
    amplitude_shift = data.event_amplitude_fraction
    complex_change = data.event_complex_change
    if n_valid:
        complex_coherence = float(
            np.abs(np.mean(complex_change[valid]))
            / np.mean(np.abs(complex_change[valid]))
        )
    else:
        complex_coherence = math.nan
    if n_strong:
        direction_coherence = float(
            np.abs(np.mean(np.exp(1j * np.angle(complex_change[strong]))))
        )
        reference_sign = np.sign(np.nanmedian(phase_shift[strong]))
        same_phase_sign_fraction = float(
            np.mean(np.sign(phase_shift[strong]) == reference_sign)
        )
    else:
        direction_coherence = math.nan
        same_phase_sign_fraction = math.nan

    event_interval = (
        (data.time_sec >= 3.5)
        & (data.time_sec <= min(4.4, float(data.time_sec[-1])))
    )
    phase_for_pca = data.phase_residual_rad[event_interval, :][:, valid]
    pca1_fraction = math.nan
    if phase_for_pca.shape[0] >= 2 and phase_for_pca.shape[1] >= 2:
        phase_for_pca = phase_for_pca - np.mean(phase_for_pca, axis=0)
        singular = np.linalg.svd(phase_for_pca, compute_uv=False)
        denominator = float(np.sum(singular**2))
        if denominator > 0:
            pca1_fraction = float(singular[0] ** 2 / denominator)

    active = np.abs(data.phase_residual_rad[:, valid]) > (
        data.phase_threshold_rad[valid][None, :]
    )
    active_fraction = (
        np.mean(active, axis=1)
        if active.shape[1]
        else np.full(data.time_sec.shape, np.nan)
    )
    onset_search = np.flatnonzero(
        (data.time_sec >= float(rtc_event_sec) - 0.90)
        & (data.time_sec <= min(float(rtc_event_sec) + 0.30, data.time_sec[-1]))
    )
    network_onset_sec = math.nan
    if onset_search.size >= sustain_samples:
        above_ten_percent = active_fraction[onset_search] >= 0.10
        sustained = np.convolve(
            above_ten_percent.astype(int),
            np.ones(int(sustain_samples), dtype=int),
            mode="valid",
        )
        found = np.flatnonzero(sustained >= sustain_samples)
        if found.size:
            network_onset_sec = float(
                data.time_sec[onset_search[int(found[0])]]
            )

    median_phase_curve = (
        np.nanmedian(data.phase_residual_rad[:, valid], axis=1) * 1.0e3
        if n_valid
        else np.full(data.time_sec.shape, np.nan)
    )
    event_curve_window = (
        (data.time_sec >= float(rtc_event_sec) - 0.90)
        & (data.time_sec <= min(float(rtc_event_sec) + 0.30, data.time_sec[-1]))
    )
    median_phase_extremum_mrad = math.nan
    median_phase_extremum_sec = math.nan
    if np.any(event_curve_window):
        curve_rows = np.flatnonzero(event_curve_window)
        local = median_phase_curve[event_curve_window]
        if np.any(np.isfinite(local)):
            extremum_row = int(np.nanargmax(np.abs(local)))
            median_phase_extremum_mrad = float(local[extremum_row])
            median_phase_extremum_sec = float(
                data.time_sec[curve_rows[extremum_row]]
            )

    return {
        "network": int(data.network),
        "raw_file": str(data.path),
        "n_raw_tones": int(data.complex_iq.shape[1]),
        "n_apt_usable_tones": n_valid,
        "n_strong_phase_tones": n_strong,
        "strong_phase_fraction": (
            float(n_strong / n_valid) if n_valid else None
        ),
        "median_phase_shift_mrad": _finite_or_none(
            np.nanmedian(phase_shift[valid]) * 1.0e3 if n_valid else math.nan
        ),
        "median_amplitude_shift_percent": _finite_or_none(
            np.nanmedian(amplitude_shift[valid]) * 100.0
            if n_valid
            else math.nan
        ),
        "complex_change_coherence": _finite_or_none(complex_coherence),
        "strong_change_direction_coherence": _finite_or_none(
            direction_coherence
        ),
        "strong_same_phase_sign_fraction": _finite_or_none(
            same_phase_sign_fraction
        ),
        "phase_pca1_variance_fraction": _finite_or_none(pca1_fraction),
        "network_ten_percent_onset_sec": _finite_or_none(network_onset_sec),
        "maximum_simultaneous_strong_fraction": _finite_or_none(
            np.nanmax(active_fraction)
        ),
        "median_phase_extremum_mrad": _finite_or_none(
            median_phase_extremum_mrad
        ),
        "median_phase_extremum_sec": _finite_or_none(
            median_phase_extremum_sec
        ),
    }


def _tone_rows(
    data: NetworkData,
    *,
    identity: EventIdentity,
) -> list[dict[str, Any]]:
    assert data.phase_sigma_rad is not None
    assert data.phase_threshold_rad is not None
    assert data.tone_onset_sec is not None
    assert data.event_phase_shift_rad is not None
    assert data.event_amplitude_fraction is not None
    assert data.event_complex_change is not None
    assert data.strong_event_tone is not None
    rows: list[dict[str, Any]] = []
    for tone in range(data.complex_iq.shape[1]):
        change = data.event_complex_change[tone]
        rows.append(
            {
                "obsnum": identity.obsnum,
                "citlali_scan": identity.citlali_scan,
                "network": data.network,
                "tone": tone,
                "uid": int(data.uid[tone]),
                "tone_frequency_hz": float(data.tone_frequency_hz[tone]),
                "apt_usable": bool(data.apt_usable[tone]),
                "phase_noise_mrad": _finite_or_none(
                    data.phase_sigma_rad[tone] * 1.0e3
                ),
                "phase_threshold_mrad": _finite_or_none(
                    data.phase_threshold_rad[tone] * 1.0e3
                ),
                "onset_sec": _finite_or_none(data.tone_onset_sec[tone]),
                "phase_shift_mrad": _finite_or_none(
                    data.event_phase_shift_rad[tone] * 1.0e3
                ),
                "amplitude_shift_percent": _finite_or_none(
                    data.event_amplitude_fraction[tone] * 100.0
                ),
                "complex_change_real": _finite_or_none(change.real),
                "complex_change_imag": _finite_or_none(change.imag),
                "strong_phase_event": bool(data.strong_event_tone[tone]),
            }
        )
    return rows


def _plot_heatmap(
    ax: plt.Axes,
    data: NetworkData,
    *,
    event_sec: float,
    phase_limit_mrad: float,
) -> Any:
    assert data.phase_residual_rad is not None
    usable = data.apt_usable & np.isfinite(data.tone_frequency_hz)
    order = np.flatnonzero(usable)
    order = order[np.argsort(data.tone_frequency_hz[order])]
    image = data.phase_residual_rad[:, order].T * 1.0e3
    extent = [
        float(data.time_sec[0]),
        float(data.time_sec[-1]),
        float(data.tone_frequency_hz[order[0]] / 1.0e9),
        float(data.tone_frequency_hz[order[-1]] / 1.0e9),
    ]
    artist = ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap="coolwarm",
        vmin=-float(phase_limit_mrad),
        vmax=float(phase_limit_mrad),
    )
    ax.axvline(event_sec, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(f"nw{data.network}: phase residual by readout frequency")
    ax.set_ylabel("tone frequency (GHz)")
    return artist


def _make_figure(
    output_path: Path,
    *,
    identity: EventIdentity,
    event_data: NetworkData,
    selected_networks: list[NetworkData],
    summaries: list[dict[str, Any]],
    onset_sec: float,
    phase_limit_mrad: float,
) -> None:
    assert event_data.phase_residual_rad is not None
    assert event_data.amplitude_fraction is not None
    tone = identity.tone
    time_sec = event_data.time_sec
    complex_iq = event_data.complex_iq[:, tone]
    baseline = time_sec <= max(onset_sec - 0.25, float(time_sec[0]))
    z0 = np.nanmedian(complex_iq[baseline])
    iq_norm = (complex_iq - z0) / np.abs(z0)

    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.1])

    ax_iq_time = fig.add_subplot(grid[0, 0])
    ax_phase = fig.add_subplot(grid[0, 1])
    ax_iq_plane = fig.add_subplot(grid[0, 2])
    ax_common = fig.add_subplot(grid[1, :2])
    ax_summary = fig.add_subplot(grid[1, 2])

    ax_iq_time.plot(time_sec, iq_norm.real * 100.0, label="I")
    ax_iq_time.plot(time_sec, iq_norm.imag * 100.0, label="Q")
    ax_iq_time.set_ylabel("change / baseline |I+iQ| (%)")
    ax_iq_time.set_xlabel("time within Citlali scan (s)")
    ax_iq_time.legend()

    ax_phase.plot(
        time_sec,
        event_data.phase_residual_rad[:, tone] * 1.0e3,
        label="phase",
        color="tab:blue",
    )
    ax_phase.set_ylabel("phase residual (mrad)")
    ax_phase.set_xlabel("time within Citlali scan (s)")
    ax_amp = ax_phase.twinx()
    ax_amp.plot(
        time_sec,
        event_data.amplitude_fraction[:, tone] * 100.0,
        label="amplitude",
        color="tab:orange",
        alpha=0.75,
    )
    ax_amp.set_ylabel("amplitude change (%)")

    event_mask = (
        (time_sec >= max(onset_sec - 0.5, float(time_sec[0])))
        & (time_sec <= min(identity.rtc_event_sec + 0.5, float(time_sec[-1])))
    )
    scatter = ax_iq_plane.scatter(
        iq_norm.real[event_mask] * 100.0,
        iq_norm.imag[event_mask] * 100.0,
        c=time_sec[event_mask],
        cmap="viridis",
        s=15,
    )
    ax_iq_plane.scatter(
        np.interp(onset_sec, time_sec, iq_norm.real) * 100.0,
        np.interp(onset_sec, time_sec, iq_norm.imag) * 100.0,
        marker="o",
        color="black",
        label="raw onset",
    )
    ax_iq_plane.scatter(
        np.interp(identity.rtc_event_sec, time_sec, iq_norm.real) * 100.0,
        np.interp(identity.rtc_event_sec, time_sec, iq_norm.imag) * 100.0,
        marker="x",
        color="black",
        label="RTC sample",
    )
    ax_iq_plane.set_xlabel("normalized ΔI (%)")
    ax_iq_plane.set_ylabel("normalized ΔQ (%)")
    ax_iq_plane.legend()
    fig.colorbar(scatter, ax=ax_iq_plane, label="scan time (s)")

    for data in selected_networks:
        assert data.phase_residual_rad is not None
        usable = data.apt_usable
        median_phase = np.nanmedian(
            data.phase_residual_rad[:, usable], axis=1
        ) * 1.0e3
        ax_common.plot(data.time_sec, median_phase, label=f"nw{data.network}")
    ax_common.set_ylabel("median phase residual (mrad)")
    ax_common.set_xlabel("time within Citlali scan (s)")
    ax_common.legend(ncol=max(1, len(selected_networks)))

    network_ids = np.asarray([int(row["network"]) for row in summaries], dtype=int)
    strong_fraction = np.asarray(
        [
            (
                float(row["strong_phase_fraction"])
                if row["strong_phase_fraction"] is not None
                else np.nan
            )
            for row in summaries
        ],
        dtype=float,
    )
    median_phase = np.asarray(
        [
            (
                float(row["median_phase_shift_mrad"])
                if row["median_phase_shift_mrad"] is not None
                else np.nan
            )
            for row in summaries
        ],
        dtype=float,
    )
    ax_summary.bar(network_ids, strong_fraction, color="0.65")
    ax_summary.set_ylim(0.0, 1.0)
    ax_summary.set_ylabel("strong phase fraction")
    ax_summary.set_xlabel("network ID")
    ax_summary.set_xticks(network_ids)
    ax_summary_phase = ax_summary.twinx()
    ax_summary_phase.plot(
        network_ids, median_phase, marker="o", color="tab:red", linewidth=1.0
    )
    ax_summary_phase.set_ylabel("median phase shift (mrad)")

    heatmap_axes: list[plt.Axes] = []
    heatmap_artist = None
    for column, data in enumerate(selected_networks[:3]):
        ax = fig.add_subplot(grid[2, column])
        heatmap_axes.append(ax)
        heatmap_artist = _plot_heatmap(
            ax,
            data,
            event_sec=identity.rtc_event_sec,
            phase_limit_mrad=phase_limit_mrad,
        )
        ax.set_xlabel("time within Citlali scan (s)")
    if heatmap_artist is not None:
        fig.colorbar(
            heatmap_artist,
            ax=heatmap_axes,
            label="phase residual (mrad)",
            shrink=0.9,
        )

    ax_fraction = fig.add_subplot(grid[3, :])
    for data in selected_networks:
        assert data.phase_residual_rad is not None
        assert data.phase_threshold_rad is not None
        usable = data.apt_usable
        active = np.abs(data.phase_residual_rad[:, usable]) > (
            data.phase_threshold_rad[usable][None, :]
        )
        fraction = np.mean(active, axis=1)
        ax_fraction.plot(data.time_sec, fraction, label=f"nw{data.network}")
    ax_fraction.set_ylim(0.0, 1.0)
    ax_fraction.set_ylabel("fraction above phase threshold")
    ax_fraction.set_xlabel("time within Citlali scan (s)")
    ax_fraction.legend(ncol=max(1, len(selected_networks)))

    for ax in (ax_iq_time, ax_phase, ax_common, ax_fraction):
        ax.axvline(onset_sec, color="black", linewidth=1.0, label="_raw onset")
        ax.axvline(
            identity.rtc_event_sec,
            color="black",
            linewidth=1.0,
            linestyle="--",
            label="_RTC sample",
        )
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"TolTEC raw I/Q event coherence: obs {identity.obsnum}, "
        f"Citlali scan {identity.citlali_scan}, nw{identity.network}, "
        f"uid {identity.uid}, tone {identity.tone}"
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _classification(
    summary: dict[str, Any],
    *,
    event_phase_mrad: float,
) -> str:
    strong_fraction = float(summary["strong_phase_fraction"])
    same_sign = float(summary["strong_same_phase_sign_fraction"])
    direction = float(summary["strong_change_direction_coherence"])
    median_amplitude = abs(float(summary["median_amplitude_shift_percent"]))
    if (
        strong_fraction >= 0.30
        and same_sign >= 0.90
        and direction >= 0.80
        and abs(float(event_phase_mrad)) >= 5.0
        and median_amplitude < 2.0
    ):
        return "network_coherent_phase_dominant_complex_rotation"
    if strong_fraction >= 0.30 and median_amplitude >= 2.0:
        return "network_coherent_gain_or_compression_candidate"
    if strong_fraction < 0.10:
        return "detector_local_or_raw_iq_weak"
    return "mixed_or_unclassified"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rtc-file", type=Path, required=True)
    parser.add_argument("--apt-file", type=Path, required=True)
    parser.add_argument("--obsnum", type=int, required=True)
    parser.add_argument("--subobsnum", type=int, default=0)
    parser.add_argument("--raw-file-scan", type=int, default=2)
    parser.add_argument("--citlali-scan", type=int, required=True)
    parser.add_argument("--event-network", type=int, required=True)
    parser.add_argument("--event-uid", type=int, required=True)
    parser.add_argument(
        "--event-tone",
        type=int,
        default=None,
        help="raw tone slot; by default resolve event UID through the APT",
    )
    parser.add_argument(
        "--control-networks",
        type=int,
        nargs="*",
        default=[0, 5],
    )
    parser.add_argument(
        "--summary-networks",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12",
        help="comma-separated network IDs included in the event-window summary",
    )
    parser.add_argument("--sigma-threshold", type=float, default=5.0)
    parser.add_argument("--min-phase-mrad", type=float, default=5.0)
    parser.add_argument("--sustain-samples", type=int, default=4)
    parser.add_argument("--phase-limit-mrad", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    apt = Table.read(args.apt_file)

    event_rows = apt[
        (np.asarray(apt["nw"], dtype=int) == int(args.event_network))
        & (np.asarray(apt["uid"], dtype=int) == int(args.event_uid))
    ]
    if len(event_rows) != 1:
        raise ValueError(
            f"expected one APT row for nw{args.event_network}, "
            f"uid={args.event_uid}; found {len(event_rows)}"
        )
    apt_tone = int(float(event_rows["kids_tone"][0]))
    event_tone = apt_tone if args.event_tone is None else int(args.event_tone)
    if event_tone != apt_tone:
        raise ValueError(
            f"explicit event tone {event_tone} disagrees with APT kids_tone "
            f"{apt_tone} for uid={args.event_uid}"
        )

    identity = _load_event_identity(
        args.rtc_file,
        obsnum=args.obsnum,
        subobsnum=args.subobsnum,
        raw_file_scan=args.raw_file_scan,
        citlali_scan=args.citlali_scan,
        network=args.event_network,
        uid=args.event_uid,
        tone=event_tone,
    )

    requested_summary = [
        int(value.strip())
        for value in str(args.summary_networks).split(",")
        if value.strip()
    ]
    networks_to_load: list[int] = []
    for network in [
        int(args.event_network),
        *[int(value) for value in args.control_networks],
        *requested_summary,
    ]:
        if network not in networks_to_load:
            networks_to_load.append(network)

    loaded: dict[int, NetworkData] = {}
    for network in networks_to_load:
        raw_path = _find_raw_file(
            args.data_root,
            network=network,
            obsnum=args.obsnum,
            subobsnum=args.subobsnum,
            raw_file_scan=args.raw_file_scan,
        )
        if raw_path is None:
            continue
        data = _load_network(
            raw_path,
            network=network,
            scan_start_sec=identity.scan_start_sec,
            scan_end_sec=identity.scan_end_sec,
            apt=apt,
        )
        _analyze_network(
            data,
            rtc_event_sec=identity.rtc_event_sec,
            sigma_threshold=args.sigma_threshold,
            min_phase_mrad=args.min_phase_mrad,
            sustain_samples=args.sustain_samples,
        )
        loaded[network] = data

    if identity.network not in loaded:
        raise RuntimeError(f"raw event network nw{identity.network} is unavailable")
    event_data = loaded[identity.network]
    if not 0 <= identity.tone < event_data.complex_iq.shape[1]:
        raise ValueError(f"event tone {identity.tone} is outside the raw tone axis")
    if int(event_data.uid[identity.tone]) != identity.uid:
        raise ValueError(
            f"APT join mismatch: nw{identity.network} tone {identity.tone} "
            f"maps to uid={event_data.uid[identity.tone]}, expected {identity.uid}"
        )

    assert event_data.tone_onset_sec is not None
    onset_sec = float(event_data.tone_onset_sec[identity.tone])
    if not np.isfinite(onset_sec):
        raise RuntimeError(
            f"no sustained raw phase onset found for uid={identity.uid}"
        )

    summary_rows = [
        _network_summary(
            loaded[network],
            rtc_event_sec=identity.rtc_event_sec,
            sustain_samples=args.sustain_samples,
        )
        for network in requested_summary
        if network in loaded
    ]
    selected_ids: list[int] = []
    for network in [
        int(args.event_network),
        *[int(value) for value in args.control_networks],
    ]:
        if network in loaded and network not in selected_ids:
            selected_ids.append(network)
    selected_data = [loaded[network] for network in selected_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or (
        f"o{identity.obsnum}_cs{identity.citlali_scan}_"
        f"nw{identity.network}_uid{identity.uid}"
    )
    summary_path = args.output_dir / f"{prefix}_summary.json"
    tones_path = args.output_dir / f"{prefix}_tone_metrics.csv"
    figure_path = args.output_dir / f"{prefix}_diagnostic.png"

    event_summary = next(
        row for row in summary_rows if int(row["network"]) == identity.network
    )
    assert event_data.event_phase_shift_rad is not None
    assert event_data.event_amplitude_fraction is not None
    event_phase_mrad = float(
        event_data.event_phase_shift_rad[identity.tone] * 1.0e3
    )
    event_amplitude_percent = float(
        event_data.event_amplitude_fraction[identity.tone] * 100.0
    )
    result = {
        "schema_version": 1,
        "identity": {
            **identity.__dict__,
            "rtc_file": str(args.rtc_file),
            "apt_file": str(args.apt_file),
            "data_root": str(args.data_root),
        },
        "analysis_policy": {
            "baseline_start_sec": 0.25,
            "baseline_end_before_rtc_sec": 0.70,
            "onset_search_lookback_sec": 0.90,
            "sigma_threshold": float(args.sigma_threshold),
            "minimum_phase_threshold_mrad": float(args.min_phase_mrad),
            "sustain_samples": int(args.sustain_samples),
            "event_pre_window_sec_relative_rtc": [-0.82, -0.50],
            "event_measurement_window_sec_relative_rtc": [-0.14, -0.04],
        },
        "event_detector": {
            "operational_raw_onset_sec": onset_sec,
            "onset_leads_rtc_sample_sec": float(identity.rtc_event_sec - onset_sec),
            "phase_shift_mrad": event_phase_mrad,
            "amplitude_shift_percent": event_amplitude_percent,
        },
        "event_network_classification": _classification(
            event_summary, event_phase_mrad=event_phase_mrad
        ),
        "network_summary": summary_rows,
        "products": {
            "summary_json": str(summary_path),
            "tone_metrics_csv": str(tones_path),
            "diagnostic_png": str(figure_path),
        },
    }
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    tone_rows: list[dict[str, Any]] = []
    for network in requested_summary:
        if network in loaded:
            tone_rows.extend(_tone_rows(loaded[network], identity=identity))
    if tone_rows:
        with tones_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tone_rows[0]))
            writer.writeheader()
            writer.writerows(tone_rows)

    _make_figure(
        figure_path,
        identity=identity,
        event_data=event_data,
        selected_networks=selected_data,
        summaries=summary_rows,
        onset_sec=onset_sec,
        phase_limit_mrad=args.phase_limit_mrad,
    )

    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
