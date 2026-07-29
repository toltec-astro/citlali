#!/usr/bin/env python3
"""Survey coherent raw-I/Q phase events across pointing observations.

For each persisted Citlali scan and each available readout network, this tool
searches the raw ``I + iQ`` streams for the time that maximizes the fraction of
APT-usable tones undergoing a same-sign phase change.  Detector identity is
joined through the supplied matched APT; network IDs are preserved explicitly.

The reported raw event time is an operational classifier result.  It is not a
physical onset time and is not forced to equal Citlali's RTC dominant-step
sample.  The two times and their offset are persisted separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from astropy.table import Table

from tools.diagnostics import pointing_iq_event_coherence as event_tool

import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OBSNUMS = (152420, 152432, 152434)
RTC_MISSING_SAMPLE = -2_147_483_647


@dataclass(frozen=True)
class ScanMetadata:
    scan_row: int
    citlali_scan: int
    sample_start: int
    sample_end: int
    start_time_sec: float
    end_time_sec: float


@dataclass(frozen=True)
class RtcNetworkMetadata:
    network: int
    dominant_sample: int | None
    event_sec: float | None
    step_detector_fraction: float | None
    step_alignment_fraction: float | None
    step_score_max: float | None


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _utc_iso(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=UTC).isoformat()


def _scan_metadata(ds: netCDF4.Dataset) -> list[ScanMetadata]:
    required = ("output_scan_index", "scan_indices", "TelTime")
    missing = [name for name in required if name not in ds.variables]
    if missing:
        raise KeyError(f"RTC timestream is missing variables: {missing}")
    output_scan = np.asarray(ds.variables["output_scan_index"][:], dtype=int)
    scan_indices = np.asarray(ds.variables["scan_indices"][:], dtype=int)
    tel_time = np.asarray(ds.variables["TelTime"][:], dtype=float)
    scans: list[ScanMetadata] = []
    for row, citlali_scan in enumerate(output_scan):
        sample_start, sample_end = scan_indices[row]
        if not 0 <= sample_start <= sample_end < tel_time.size:
            raise ValueError(
                f"invalid scan bounds [{sample_start}, {sample_end}] "
                f"for RTC sample count {tel_time.size}"
            )
        scans.append(
            ScanMetadata(
                scan_row=int(row),
                citlali_scan=int(citlali_scan),
                sample_start=int(sample_start),
                sample_end=int(sample_end),
                start_time_sec=float(tel_time[sample_start]),
                end_time_sec=float(tel_time[sample_end]),
            )
        )
    return scans


def _rtc_network_metadata(
    ds: netCDF4.Dataset,
    *,
    scan: ScanMetadata,
) -> dict[int, RtcNetworkMetadata]:
    required = (
        "rtc_diag_network_ids",
        "rtc_network_step_dominant_sample",
        "rtc_network_step_det_frac",
        "rtc_network_step_alignment_frac",
        "rtc_network_step_score_max",
        "TelTime",
    )
    missing = [name for name in required if name not in ds.variables]
    if missing:
        raise KeyError(f"RTC timestream is missing variables: {missing}")
    network_ids = np.asarray(ds.variables["rtc_diag_network_ids"][:], dtype=int)
    dominant = np.asarray(
        ds.variables["rtc_network_step_dominant_sample"][scan.scan_row, :],
        dtype=int,
    )
    detector_fraction = np.asarray(
        ds.variables["rtc_network_step_det_frac"][scan.scan_row, :],
        dtype=float,
    )
    alignment_fraction = np.asarray(
        ds.variables["rtc_network_step_alignment_frac"][scan.scan_row, :],
        dtype=float,
    )
    score_max = np.asarray(
        ds.variables["rtc_network_step_score_max"][scan.scan_row, :],
        dtype=float,
    )
    tel_time = np.asarray(ds.variables["TelTime"][:], dtype=float)
    n_scan_samples = scan.sample_end - scan.sample_start + 1
    result: dict[int, RtcNetworkMetadata] = {}
    for column, network in enumerate(network_ids):
        sample_value = int(dominant[column])
        sample = (
            sample_value
            if sample_value != RTC_MISSING_SAMPLE
            and 0 <= sample_value < n_scan_samples
            else None
        )
        event_sec = (
            float(
                tel_time[scan.sample_start + sample] - scan.start_time_sec
            )
            if sample is not None
            else None
        )
        result[int(network)] = RtcNetworkMetadata(
            network=int(network),
            dominant_sample=sample,
            event_sec=event_sec,
            step_detector_fraction=_finite_or_none(detector_fraction[column]),
            step_alignment_fraction=_finite_or_none(
                alignment_fraction[column]
            ),
            step_score_max=_finite_or_none(score_max[column]),
        )
    return result


def _interval_means(
    prefix: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    counts = ends - starts
    if np.any(counts <= 0):
        raise ValueError("event comparison interval contains no samples")
    return (prefix[ends, :] - prefix[starts, :]) / counts[:, None]


def _classify_raw_event(
    data: event_tool.NetworkData,
    *,
    sigma_threshold: float,
    min_phase_mrad: float,
    pre_window_sec: float,
    guard_window_sec: float,
    post_window_sec: float,
) -> dict[str, Any]:
    time_sec = np.asarray(data.time_sec, dtype=float)
    complex_iq = np.asarray(data.complex_iq, dtype=complex)
    if time_sec.size < 32:
        raise ValueError(f"nw{data.network}: fewer than 32 raw samples")
    if np.any(np.diff(time_sec) <= 0):
        raise ValueError(f"nw{data.network}: raw sample time is not increasing")

    amplitude = np.abs(complex_iq)
    phase = np.unwrap(np.angle(complex_iq), axis=0)
    phase_sigma = event_tool._robust_sigma(
        np.diff(phase, axis=0), axis=0
    ) / math.sqrt(2.0)
    phase_threshold = np.maximum(
        float(sigma_threshold) * phase_sigma,
        float(min_phase_mrad) * 1.0e-3,
    )
    valid = (
        data.apt_usable
        & np.isfinite(phase_threshold)
        & (np.nanmedian(amplitude, axis=0) > 0)
    )
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError(f"nw{data.network}: no APT-usable raw tones")

    unit_phase = np.full(complex_iq.shape, np.nan + 1j * np.nan, dtype=complex)
    np.divide(
        complex_iq,
        amplitude,
        out=unit_phase,
        where=amplitude > 0,
    )
    phase_prefix = np.vstack(
        [
            np.zeros((1, complex_iq.shape[1]), dtype=complex),
            np.nancumsum(unit_phase, axis=0),
        ]
    )
    amplitude_prefix = np.vstack(
        [
            np.zeros((1, complex_iq.shape[1]), dtype=float),
            np.nancumsum(amplitude, axis=0),
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
        raise ValueError(f"nw{data.network}: no complete event comparison window")

    candidate_time = time_sec[candidate_rows]
    pre_start = np.searchsorted(
        time_sec, candidate_time - guard_window_sec - pre_window_sec
    )
    pre_end = np.searchsorted(
        time_sec, candidate_time - guard_window_sec
    )
    post_start = np.searchsorted(
        time_sec, candidate_time + guard_window_sec
    )
    post_end = np.searchsorted(
        time_sec, candidate_time + guard_window_sec + post_window_sec
    )
    phase_pre = _interval_means(phase_prefix, pre_start, pre_end)
    phase_post = _interval_means(phase_prefix, post_start, post_end)
    phase_shift = np.angle(phase_post / phase_pre)
    strong = np.abs(phase_shift) > phase_threshold[None, :]
    strong[:, ~valid] = False
    positive_fraction = np.count_nonzero(
        strong & (phase_shift > 0), axis=1
    ) / n_valid
    negative_fraction = np.count_nonzero(
        strong & (phase_shift < 0), axis=1
    ) / n_valid
    coherent_fraction = np.maximum(positive_fraction, negative_fraction)
    strong_fraction = np.count_nonzero(strong, axis=1) / n_valid
    best = int(np.nanargmax(coherent_fraction))
    raw_row = int(candidate_rows[best])
    raw_event_sec = float(time_sec[raw_row])
    selected_shift = phase_shift[best, :]
    selected_strong = strong[best, :]

    amplitude_pre = _interval_means(
        amplitude_prefix, pre_start, pre_end
    )[best, :]
    amplitude_post = _interval_means(
        amplitude_prefix, post_start, post_end
    )[best, :]
    amplitude_shift = amplitude_post / amplitude_pre - 1.0
    complex_change = phase_post[best, :] / phase_pre[best, :] - 1.0
    n_strong = int(np.count_nonzero(selected_strong))
    reference_sign = (
        float(np.sign(np.nanmedian(selected_shift[selected_strong])))
        if n_strong
        else math.nan
    )
    same_sign_fraction = (
        float(
            np.mean(
                np.sign(selected_shift[selected_strong]) == reference_sign
            )
        )
        if n_strong
        else math.nan
    )
    direction_coherence = (
        float(
            np.abs(
                np.mean(
                    np.exp(1j * np.angle(complex_change[selected_strong]))
                )
            )
        )
        if n_strong
        else math.nan
    )
    return {
        "n_raw_tones": int(complex_iq.shape[1]),
        "n_apt_usable_tones": n_valid,
        "n_strong_phase_tones": n_strong,
        "raw_event_sec": raw_event_sec,
        "raw_event_sample": raw_row,
        "coherent_same_sign_fraction": float(coherent_fraction[best]),
        "strong_phase_fraction": float(strong_fraction[best]),
        "same_phase_sign_fraction": _finite_or_none(same_sign_fraction),
        "strong_change_direction_coherence": _finite_or_none(
            direction_coherence
        ),
        "median_phase_shift_mrad": _finite_or_none(
            np.nanmedian(selected_shift[valid]) * 1.0e3
        ),
        "median_strong_phase_shift_mrad": _finite_or_none(
            np.nanmedian(selected_shift[selected_strong]) * 1.0e3
            if n_strong
            else math.nan
        ),
        "median_amplitude_shift_percent": _finite_or_none(
            np.nanmedian(amplitude_shift[valid]) * 100.0
        ),
    }


def _rack(network: int) -> str:
    return "RACKA" if int(network) <= 6 else "RACKO"


def _array_name(network: int) -> str:
    if 0 <= int(network) <= 6:
        return "a1100"
    if 7 <= int(network) <= 10:
        return "a1400"
    if 11 <= int(network) <= 12:
        return "a2000"
    raise ValueError(f"network {network} has no TolTEC array mapping")


def _cluster_events(
    rows: list[dict[str, Any]],
    *,
    minimum_fraction: float,
    tolerance_sec: float,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    by_scan: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        row["event_cluster_id"] = None
        if float(row["coherent_same_sign_fraction"]) < minimum_fraction:
            continue
        key = (int(row["obsnum"]), int(row["citlali_scan"]))
        by_scan.setdefault(key, []).append(row)

    for (obsnum, citlali_scan), scan_rows in sorted(by_scan.items()):
        scan_rows.sort(key=lambda row: float(row["raw_event_sec"]))
        groups: list[list[dict[str, Any]]] = []
        for row in scan_rows:
            if not groups:
                groups.append([row])
                continue
            group_times = [
                float(item["raw_event_sec"]) for item in groups[-1]
            ]
            if (
                float(row["raw_event_sec"]) - min(group_times)
                <= float(tolerance_sec)
            ):
                groups[-1].append(row)
            else:
                groups.append([row])
        for group_index, group in enumerate(groups, start=1):
            cluster_id = (
                f"o{obsnum}_s{citlali_scan:02d}_c{group_index:02d}"
            )
            for row in group:
                row["event_cluster_id"] = cluster_id
            times = np.asarray(
                [float(row["raw_event_sec"]) for row in group], dtype=float
            )
            networks = sorted(int(row["network"]) for row in group)
            racks = sorted({_rack(network) for network in networks})
            clusters.append(
                {
                    "event_cluster_id": cluster_id,
                    "obsnum": obsnum,
                    "citlali_scan": citlali_scan,
                    "network_count": len(networks),
                    "networks": networks,
                    "racks": racks,
                    "cross_rack": len(racks) > 1,
                    "median_event_sec": float(np.median(times)),
                    "event_span_sec": float(np.max(times) - np.min(times)),
                    "maximum_coherent_fraction": float(
                        max(
                            float(row["coherent_same_sign_fraction"])
                            for row in group
                        )
                    ),
                }
            )
    return clusters


def _population_summary(
    rows: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    *,
    network_event_fraction: float,
) -> dict[str, Any]:
    network_rows: list[dict[str, Any]] = []
    keys = sorted(
        {(int(row["obsnum"]), int(row["network"])) for row in rows}
    )
    for obsnum, network in keys:
        selected = [
            row
            for row in rows
            if int(row["obsnum"]) == obsnum
            and int(row["network"]) == network
        ]
        fractions = np.asarray(
            [float(row["coherent_same_sign_fraction"]) for row in selected],
            dtype=float,
        )
        network_rows.append(
            {
                "obsnum": obsnum,
                "network": network,
                "array": _array_name(network),
                "rack": _rack(network),
                "n_scans": len(selected),
                "n_event_scans": int(
                    np.count_nonzero(fractions >= network_event_fraction)
                ),
                "n_severe_scans": int(np.count_nonzero(fractions >= 0.30)),
                "median_coherent_fraction": float(np.median(fractions)),
                "maximum_coherent_fraction": float(np.max(fractions)),
            }
        )

    observation_rows: list[dict[str, Any]] = []
    for obsnum in sorted({int(row["obsnum"]) for row in rows}):
        selected = [row for row in rows if int(row["obsnum"]) == obsnum]
        fractions = np.asarray(
            [float(row["coherent_same_sign_fraction"]) for row in selected],
            dtype=float,
        )
        obs_clusters = [
            cluster for cluster in clusters if int(cluster["obsnum"]) == obsnum
        ]
        observation_rows.append(
            {
                "obsnum": obsnum,
                "n_scan_network_rows": len(selected),
                "n_network_events": int(
                    np.count_nonzero(fractions >= network_event_fraction)
                ),
                "n_severe_network_events": int(
                    np.count_nonzero(fractions >= 0.30)
                ),
                "maximum_coherent_fraction": float(np.max(fractions)),
                "n_event_clusters": len(obs_clusters),
                "n_cross_rack_clusters": sum(
                    bool(cluster["cross_rack"]) for cluster in obs_clusters
                ),
            }
        )
    return {
        "by_observation_network": network_rows,
        "by_observation": observation_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_figure(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    obsnums: list[int],
    network_event_fraction: float,
) -> None:
    networks = sorted({int(row["network"]) for row in rows})
    scans = sorted({int(row["citlali_scan"]) for row in rows})
    fig, axes = plt.subplots(
        len(obsnums),
        2,
        figsize=(15, 3.8 * len(obsnums)),
        constrained_layout=True,
        squeeze=False,
    )
    fraction_image = None
    time_image = None
    for row_index, obsnum in enumerate(obsnums):
        fraction = np.full((len(networks), len(scans)), np.nan)
        event_time = np.full_like(fraction, np.nan)
        for row in rows:
            if int(row["obsnum"]) != obsnum:
                continue
            network_row = networks.index(int(row["network"]))
            scan_column = scans.index(int(row["citlali_scan"]))
            value = float(row["coherent_same_sign_fraction"])
            fraction[network_row, scan_column] = value
            if value >= network_event_fraction:
                event_time[network_row, scan_column] = float(
                    row["raw_event_sec"]
                )

        ax_fraction, ax_time = axes[row_index]
        fraction_image = ax_fraction.imshow(
            fraction,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        time_image = ax_time.imshow(
            event_time,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=5.0,
            cmap="viridis",
        )
        ax_fraction.set_title(
            f"obs {obsnum}: maximum same-sign phase fraction"
        )
        ax_time.set_title(
            f"obs {obsnum}: raw event time "
            f"(fraction ≥ {network_event_fraction:.0%})"
        )
        for ax in (ax_fraction, ax_time):
            ax.set_xticks(range(len(scans)), scans)
            ax.set_yticks(range(len(networks)), networks)
            ax.set_xlabel("Citlali scan (one-based output_scan_index)")
            ax.set_ylabel("network ID")
            if 5 in networks and 7 in networks:
                split = (networks.index(5) + networks.index(7)) / 2.0
                ax.axhline(split, color="white", linewidth=0.8)
    if fraction_image is not None:
        fig.colorbar(
            fraction_image,
            ax=axes[:, 0],
            label="fraction of APT-usable tones",
            shrink=0.9,
        )
    if time_image is not None:
        fig.colorbar(
            time_image,
            ax=axes[:, 1],
            label="seconds within Citlali scan",
            shrink=0.9,
        )
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
    parser.add_argument("--subobsnum", type=int, default=0)
    parser.add_argument("--raw-file-scan", type=int, default=2)
    parser.add_argument("--sigma-threshold", type=float, default=8.0)
    parser.add_argument("--min-phase-mrad", type=float, default=5.0)
    parser.add_argument("--pre-window-sec", type=float, default=0.20)
    parser.add_argument("--guard-window-sec", type=float, default=0.05)
    parser.add_argument("--post-window-sec", type=float, default=0.20)
    parser.add_argument(
        "--network-event-fraction", type=float, default=0.10
    )
    parser.add_argument("--cluster-tolerance-sec", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.network_event_fraction <= 1.0:
        raise ValueError("--network-event-fraction must lie in (0, 1]")
    for name in (
        "pre_window_sec",
        "guard_window_sec",
        "post_window_sec",
        "sigma_threshold",
        "min_phase_mrad",
        "cluster_tolerance_sec",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    for obsnum in args.obsnums:
        rtc_path = (
            args.reduction_root
            / str(obsnum)
            / "raw"
            / f"toltec_commissioning_pointing_{obsnum}_rtc_timestream.nc"
        )
        apt_path = args.apt_root / f"apt_{obsnum}_matched.ecsv"
        if not rtc_path.is_file():
            raise FileNotFoundError(rtc_path)
        if not apt_path.is_file():
            raise FileNotFoundError(apt_path)
        apt = Table.read(apt_path)
        with netCDF4.Dataset(rtc_path) as rtc:
            scans = _scan_metadata(rtc)
            network_ids = np.asarray(
                rtc.variables["rtc_diag_network_ids"][:], dtype=int
            )
            raw_paths = {
                int(network): event_tool._find_raw_file(
                    args.data_root,
                    network=int(network),
                    obsnum=int(obsnum),
                    subobsnum=int(args.subobsnum),
                    raw_file_scan=int(args.raw_file_scan),
                )
                for network in network_ids
            }
            missing_networks = [
                network
                for network, path in raw_paths.items()
                if path is None
            ]
            if missing_networks:
                raise FileNotFoundError(
                    f"obs {obsnum}: no raw file for networks "
                    f"{missing_networks}"
                )
            input_records.append(
                {
                    "obsnum": int(obsnum),
                    "rtc_path": str(rtc_path),
                    "apt_path": str(apt_path),
                    "raw_paths": {
                        str(network): str(path)
                        for network, path in raw_paths.items()
                    },
                }
            )
            for scan in scans:
                rtc_networks = _rtc_network_metadata(rtc, scan=scan)
                for network in network_ids:
                    network = int(network)
                    raw_path = raw_paths[network]
                    assert raw_path is not None
                    data = event_tool._load_network(
                        raw_path,
                        network=network,
                        scan_start_sec=scan.start_time_sec,
                        scan_end_sec=scan.end_time_sec,
                        apt=apt,
                    )
                    raw = _classify_raw_event(
                        data,
                        sigma_threshold=float(args.sigma_threshold),
                        min_phase_mrad=float(args.min_phase_mrad),
                        pre_window_sec=float(args.pre_window_sec),
                        guard_window_sec=float(args.guard_window_sec),
                        post_window_sec=float(args.post_window_sec),
                    )
                    rtc_meta = rtc_networks[network]
                    raw_absolute_time_sec = (
                        scan.start_time_sec + float(raw["raw_event_sec"])
                    )
                    rows.append(
                        {
                            "obsnum": int(obsnum),
                            "citlali_scan": scan.citlali_scan,
                            "network": network,
                            "array": _array_name(network),
                            "rack": _rack(network),
                            "scan_start_absolute_sec": scan.start_time_sec,
                            "scan_start_utc": _utc_iso(
                                scan.start_time_sec
                            ),
                            "raw_event_sec": raw["raw_event_sec"],
                            "raw_event_absolute_sec": raw_absolute_time_sec,
                            "raw_event_utc": _utc_iso(
                                raw_absolute_time_sec
                            ),
                            "raw_event_sample": raw["raw_event_sample"],
                            "n_raw_tones": raw["n_raw_tones"],
                            "n_apt_usable_tones": raw[
                                "n_apt_usable_tones"
                            ],
                            "n_strong_phase_tones": raw[
                                "n_strong_phase_tones"
                            ],
                            "coherent_same_sign_fraction": raw[
                                "coherent_same_sign_fraction"
                            ],
                            "strong_phase_fraction": raw[
                                "strong_phase_fraction"
                            ],
                            "same_phase_sign_fraction": raw[
                                "same_phase_sign_fraction"
                            ],
                            "strong_change_direction_coherence": raw[
                                "strong_change_direction_coherence"
                            ],
                            "median_phase_shift_mrad": raw[
                                "median_phase_shift_mrad"
                            ],
                            "median_strong_phase_shift_mrad": raw[
                                "median_strong_phase_shift_mrad"
                            ],
                            "median_amplitude_shift_percent": raw[
                                "median_amplitude_shift_percent"
                            ],
                            "rtc_dominant_sample": (
                                rtc_meta.dominant_sample
                            ),
                            "rtc_event_sec": rtc_meta.event_sec,
                            "rtc_minus_raw_event_sec": (
                                rtc_meta.event_sec
                                - float(raw["raw_event_sec"])
                                if rtc_meta.event_sec is not None
                                else None
                            ),
                            "rtc_step_detector_fraction": (
                                rtc_meta.step_detector_fraction
                            ),
                            "rtc_step_alignment_fraction": (
                                rtc_meta.step_alignment_fraction
                            ),
                            "rtc_step_score_max": rtc_meta.step_score_max,
                            "event_cluster_id": None,
                        }
                    )

    clusters = _cluster_events(
        rows,
        minimum_fraction=float(args.network_event_fraction),
        tolerance_sec=float(args.cluster_tolerance_sec),
    )
    population = _population_summary(
        rows,
        clusters,
        network_event_fraction=float(args.network_event_fraction),
    )
    manifest = {
        "schema": "citlali-pointing-raw-iq-event-survey-v1",
        "semantics": {
            "citlali_scan": (
                "one-based persisted output_scan_index; internal scan rows "
                "remain zero-based"
            ),
            "raw_event_sec": (
                "operational maximum same-sign raw phase-change time within "
                "the Citlali scan; not a physical onset time"
            ),
            "rtc_event_sec": (
                "persisted RTC network dominant-step sample time within the "
                "Citlali scan; unavailable is null"
            ),
            "fractions": "dimensionless fractions of APT-usable raw tones",
            "phase": "milliradians",
            "amplitude": "percent",
            "missing": "semantic unavailability is encoded as null or blank",
        },
        "parameters": {
            "sigma_threshold": float(args.sigma_threshold),
            "min_phase_mrad": float(args.min_phase_mrad),
            "pre_window_sec": float(args.pre_window_sec),
            "guard_window_sec": float(args.guard_window_sec),
            "post_window_sec": float(args.post_window_sec),
            "network_event_fraction": float(args.network_event_fraction),
            "cluster_tolerance_sec": float(args.cluster_tolerance_sec),
        },
        "inputs": input_records,
        "rows": rows,
        "clusters": clusters,
        "population_summary": population,
    }

    stem = "pointing_raw_iq_event_survey"
    json_path = args.output_dir / f"{stem}.json"
    row_csv_path = args.output_dir / f"{stem}_scan_network.csv"
    cluster_csv_path = args.output_dir / f"{stem}_clusters.csv"
    population_csv_path = args.output_dir / f"{stem}_population.csv"
    figure_path = args.output_dir / f"{stem}.png"
    json_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(row_csv_path, rows)
    if clusters:
        _write_csv(cluster_csv_path, clusters)
    _write_csv(
        population_csv_path,
        population["by_observation_network"],
    )
    _make_figure(
        figure_path,
        rows=rows,
        obsnums=[int(obsnum) for obsnum in args.obsnums],
        network_event_fraction=float(args.network_event_fraction),
    )
    print(json_path)
    print(row_csv_path)
    print(cluster_csv_path if clusters else "no event clusters")
    print(population_csv_path)
    print(figure_path)


if __name__ == "__main__":
    main()
