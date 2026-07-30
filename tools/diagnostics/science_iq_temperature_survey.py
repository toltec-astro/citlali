#!/usr/bin/env python3
"""Relate science-observation readout events to slow cryostat thermometry.

This diagnostic uses persisted Citlali RTC per-chunk metrics to survey every
approximately 10-second science chunk.  It joins those metrics to the nearest
recorded housekeeping sample and independently checks selected event-rich and
event-poor chunks in the raw ``I + iQ`` data.

The housekeeping association is intentionally observation-scale evidence.
TolTEC housekeeping is sampled approximately once per minute and cannot
establish the cause or onset of a subsecond readout event.
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
    str(Path(tempfile.gettempdir()) / "citlali-science-iq-hk-mpl-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from tools.diagnostics import pointing_iq_event_coherence as iq_tool  # noqa: E402
from tools.diagnostics.pointing_iq_event_survey import (  # noqa: E402
    _classify_raw_event,
)


SCHEMA_VERSION = "citlali-science-iq-temperature-survey-v1"
DEFAULT_OBSNUMS = (152390, 152392, 152419, 152431, 152433)
DEFAULT_POINTING_OBSNUMS = (
    152389,
    152391,
    152393,
    152418,
    152420,
    152430,
    152432,
    152434,
)
DEFAULT_AFFECTED_NETWORKS = (1, 2, 3, 4, 8, 9)
DEFAULT_CONTROL_NETWORKS = (0, 5, 7, 11, 12)
MISSING_INT = -2_147_483_647


@dataclass(frozen=True)
class HousekeepingChannel:
    group: str
    channel_id: str
    name: str
    time_variable: str
    value_variable: str
    unit: str = "K"


HOUSEKEEPING_CHANNELS = (
    HousekeepingChannel(
        "toltec_thermometry",
        "Temperature4",
        "4K central busbar",
        "Data.ToltecThermetry.Time4",
        "Data.ToltecThermetry.Temperature4",
    ),
    HousekeepingChannel(
        "toltec_thermometry",
        "Temperature5",
        "1.1_1_top",
        "Data.ToltecThermetry.Time5",
        "Data.ToltecThermetry.Temperature5",
    ),
    HousekeepingChannel(
        "toltec_thermometry",
        "Temperature6",
        "2.0_1_foot",
        "Data.ToltecThermetry.Time6",
        "Data.ToltecThermetry.Temperature6",
    ),
    HousekeepingChannel(
        "toltec_thermometry",
        "Temperature9",
        "1.1_0.1_top",
        "Data.ToltecThermetry.Time9",
        "Data.ToltecThermetry.Temperature9",
    ),
    HousekeepingChannel(
        "toltec_thermometry",
        "Temperature10",
        "2.0_0.1_top",
        "Data.ToltecThermetry.Time10",
        "Data.ToltecThermetry.Temperature10",
    ),
    HousekeepingChannel(
        "toltec_thermometry",
        "Temperature11",
        "1.4_0.1_top",
        "Data.ToltecThermetry.Time11",
        "Data.ToltecThermetry.Temperature11",
    ),
    HousekeepingChannel(
        "toltec_thermometry",
        "Temperature13",
        "LS_front",
        "Data.ToltecThermetry.Time13",
        "Data.ToltecThermetry.Temperature13",
    ),
    HousekeepingChannel(
        "dilution_fridge",
        "T1",
        "PT2 Head",
        "Data.ToltecDilutionFridge.SampleTime",
        "Data.ToltecDilutionFridge.StsDevT1TempSigTemp",
    ),
    HousekeepingChannel(
        "dilution_fridge",
        "T2",
        "PT2 Plate",
        "Data.ToltecDilutionFridge.SampleTime",
        "Data.ToltecDilutionFridge.StsDevT2TempSigTemp",
    ),
    HousekeepingChannel(
        "dilution_fridge",
        "T3",
        "Still Plate",
        "Data.ToltecDilutionFridge.SampleTime",
        "Data.ToltecDilutionFridge.StsDevT3TempSigTemp",
    ),
    HousekeepingChannel(
        "dilution_fridge",
        "T4",
        "Cold Plate",
        "Data.ToltecDilutionFridge.SampleTime",
        "Data.ToltecDilutionFridge.StsDevT4TempSigTemp",
    ),
    HousekeepingChannel(
        "dilution_fridge",
        "T8",
        "MC Plate",
        "Data.ToltecDilutionFridge.SampleTime",
        "Data.ToltecDilutionFridge.StsDevT8TempSigTemp",
    ),
    HousekeepingChannel(
        "dilution_fridge",
        "T12",
        "MC Bar",
        "Data.ToltecDilutionFridge.SampleTime",
        "Data.ToltecDilutionFridge.StsDevT12TempSigTemp",
    ),
)


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


def _scan_intervals_from_times(
    telescope_time_sec: np.ndarray,
    scan_duration_sec: np.ndarray,
) -> list[dict[str, float | int]]:
    """Reconstruct forced-duration Citlali chunk times from telescope time."""
    telescope_time_sec = np.asarray(telescope_time_sec, dtype=float)
    scan_duration_sec = np.asarray(scan_duration_sec, dtype=float)
    if telescope_time_sec.size < 4 or scan_duration_sec.size == 0:
        raise ValueError("telescope time or scan duration is empty")
    if np.any(~np.isfinite(telescope_time_sec)):
        raise ValueError("telescope time contains non-finite values")
    if np.any(np.diff(telescope_time_sec) <= 0):
        raise ValueError("telescope time is not strictly increasing")
    if np.any(~np.isfinite(scan_duration_sec)) or np.any(scan_duration_sec <= 0):
        raise ValueError("scan duration contains invalid values")

    dt = float(np.median(np.diff(telescope_time_sec)))
    nominal_duration = float(np.median(scan_duration_sec))
    period_samples = int(round((nominal_duration + dt) / dt))
    if period_samples < 2:
        raise ValueError("derived chunk period has fewer than two samples")
    required = scan_duration_sec.size * period_samples
    if required > telescope_time_sec.size:
        raise ValueError(
            f"{scan_duration_sec.size} chunks of {period_samples} telescope "
            f"samples require {required} rows, but only "
            f"{telescope_time_sec.size} are available"
        )

    intervals: list[dict[str, float | int]] = []
    for scan_row, duration in enumerate(scan_duration_sec):
        start_index = scan_row * period_samples
        end_index = start_index + period_samples - 1
        start = float(telescope_time_sec[start_index])
        end = float(telescope_time_sec[end_index])
        nominal = end - start
        trim = max(0.0, nominal - float(duration))
        if scan_row == 0:
            start += trim
        elif scan_row == scan_duration_sec.size - 1:
            end -= trim
        elif abs(nominal - float(duration)) > max(0.05, 3.0 * dt):
            raise ValueError(
                f"interior scan {scan_row} duration {duration} differs from "
                f"nominal telescope duration {nominal}"
            )
        intervals.append(
            {
                "scan_row_zero_based": int(scan_row),
                "start_time_unix_sec": start,
                "end_time_unix_sec": end,
                "center_time_unix_sec": 0.5 * (start + end),
                "duration_sec": float(duration),
            }
        )
    return intervals


def _nearest_sample_indices(
    query_time_sec: np.ndarray,
    sample_time_sec: np.ndarray,
    *,
    max_age_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query_time_sec, dtype=float)
    sample = np.asarray(sample_time_sec, dtype=float)
    valid = np.flatnonzero(np.isfinite(sample) & (sample > 0.0))
    indices = np.full(query.shape, -1, dtype=int)
    ages = np.full(query.shape, np.nan, dtype=float)
    if valid.size == 0:
        return indices, ages
    ordered = valid[np.argsort(sample[valid])]
    ordered_time = sample[ordered]
    insert = np.searchsorted(ordered_time, query)
    for position, value in np.ndenumerate(query):
        if not np.isfinite(value):
            continue
        slot = int(insert[position])
        candidates: list[int] = []
        if slot > 0:
            candidates.append(int(ordered[slot - 1]))
        if slot < ordered.size:
            candidates.append(int(ordered[slot]))
        if not candidates:
            continue
        nearest = min(candidates, key=lambda i: abs(sample[i] - value))
        age = abs(float(sample[nearest] - value))
        if age <= float(max_age_sec):
            indices[position] = nearest
            ages[position] = age
    return indices, ages


def _read_housekeeping(
    path: Path,
    *,
    obsnum: int,
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with netCDF4.Dataset(path) as ds:
        for channel in HOUSEKEEPING_CHANNELS:
            if (
                channel.time_variable not in ds.variables
                or channel.value_variable not in ds.variables
            ):
                arrays[channel.channel_id] = (
                    np.asarray([], dtype=float),
                    np.asarray([], dtype=float),
                )
                continue
            times = np.asarray(
                ds.variables[channel.time_variable][:], dtype=float
            )
            values = np.asarray(
                ds.variables[channel.value_variable][:], dtype=float
            )
            if times.shape != values.shape:
                raise ValueError(
                    f"{path.name}: {channel.channel_id} time/value shapes differ"
                )
            arrays[channel.channel_id] = (times, values)
            for sample_index, (time_sec, value) in enumerate(zip(times, values)):
                valid_time = np.isfinite(time_sec) and time_sec > 0.0
                valid_value = np.isfinite(value) and value > 0.0
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "obsnum": int(obsnum),
                        "hk_sample_index_zero_based": int(sample_index),
                        "channel_group": channel.group,
                        "channel_id": channel.channel_id,
                        "channel_name": channel.name,
                        "unit": channel.unit,
                        "sample_time_unix_sec": (
                            float(time_sec) if valid_time else None
                        ),
                        "sample_time_utc": (
                            _utc_iso(float(time_sec)) if valid_time else None
                        ),
                        "value": float(value) if valid_value else None,
                        "status": (
                            "valid"
                            if valid_time and valid_value
                            else "invalid_or_unavailable"
                        ),
                        "source_path": str(path),
                    }
                )
    return rows, arrays


def _config_scalar(
    ds: netCDF4.Dataset,
    name: str,
    default: float,
) -> float:
    if name not in ds.variables:
        return float(default)
    value = np.asarray(ds.variables[name][...]).reshape(-1)
    return float(value[0]) if value.size else float(default)


def _read_scan_network_metrics(
    *,
    obsnum: int,
    rtc_path: Path,
    telescope_path: Path,
    hk_reference_time: np.ndarray,
    max_hk_age_sec: float,
    night_start_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        intervals = _scan_intervals_from_times(telescope_time, durations)
        centers = np.asarray(
            [float(item["center_time_unix_sec"]) for item in intervals]
        )
        hk_indices, hk_ages = _nearest_sample_indices(
            centers,
            hk_reference_time,
            max_age_sec=max_hk_age_sec,
        )
        networks = np.asarray(
            rtc.variables["rtc_diag_network_ids"][:], dtype=int
        )
        rtc_rate = _config_scalar(rtc, "RTC_SAMPRATE", 122.0703125)
        step_min_used = int(
            _config_scalar(rtc, "CONFIG.RTC.STEP_MASK.MIN_DET_USED", 32)
        )
        step_min_fraction = _config_scalar(
            rtc, "CONFIG.RTC.STEP_MASK.MIN_STEP_DET_FRAC", 0.10
        )
        step_min_alignment = _config_scalar(
            rtc, "CONFIG.RTC.STEP_MASK.MIN_ALIGNMENT_FRAC", 0.50
        )
        impulsive_min_used = int(
            _config_scalar(
                rtc,
                "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_USED",
                32,
            )
        )
        impulsive_min_fraction = _config_scalar(
            rtc,
            "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_FRAC",
            0.05,
        )
        impulsive_min_alignment = _config_scalar(
            rtc,
            "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_ALIGNMENT_FRAC",
            0.50,
        )

        variables = {
            name: np.asarray(rtc.variables[name][:])
            for name in (
                "rtc_network_n_det_input",
                "rtc_network_n_det_used",
                "rtc_network_step_score_median",
                "rtc_network_step_score_max",
                "rtc_network_step_det_frac",
                "rtc_network_step_alignment_frac",
                "rtc_network_step_dominant_sample",
                "rtc_network_step_mask_applied",
                "rtc_network_impulsive_n_det_used",
                "rtc_network_impulsive_score_median",
                "rtc_network_impulsive_score_max",
                "rtc_network_impulsive_det_frac",
                "rtc_network_impulsive_alignment_frac",
                "rtc_network_impulsive_dominant_sample",
                "rtc_network_impulsive_mask_applied",
            )
        }

    rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    for scan_row, (scan_number, interval) in enumerate(
        zip(output_scan, intervals)
    ):
        scan_record = {
            "obsnum": int(obsnum),
            "citlali_scan_one_based": int(scan_number),
            **interval,
            "center_time_utc": _utc_iso(
                float(interval["center_time_unix_sec"])
            ),
            "elapsed_observation_sec": (
                float(interval["center_time_unix_sec"])
                - float(intervals[0]["start_time_unix_sec"])
            ),
            "elapsed_night_sec": (
                float(interval["center_time_unix_sec"]) - night_start_sec
            ),
            "hk_sample_index_zero_based": (
                int(hk_indices[scan_row])
                if hk_indices[scan_row] >= 0
                else None
            ),
            "hk_sample_age_sec": _finite_or_none(hk_ages[scan_row]),
        }
        scan_rows.append(scan_record)
        for network_column, network in enumerate(networks):
            step_sample = int(
                variables["rtc_network_step_dominant_sample"][
                    scan_row, network_column
                ]
            )
            impulsive_sample = int(
                variables["rtc_network_impulsive_dominant_sample"][
                    scan_row, network_column
                ]
            )
            step_used = int(
                variables["rtc_network_n_det_used"][scan_row, network_column]
            )
            step_fraction = float(
                variables["rtc_network_step_det_frac"][
                    scan_row, network_column
                ]
            )
            step_alignment = float(
                variables["rtc_network_step_alignment_frac"][
                    scan_row, network_column
                ]
            )
            impulsive_used = int(
                variables["rtc_network_impulsive_n_det_used"][
                    scan_row, network_column
                ]
            )
            impulsive_fraction = float(
                variables["rtc_network_impulsive_det_frac"][
                    scan_row, network_column
                ]
            )
            impulsive_alignment = float(
                variables["rtc_network_impulsive_alignment_frac"][
                    scan_row, network_column
                ]
            )
            scan_start = float(interval["start_time_unix_sec"])
            step_time = (
                scan_start + step_sample / rtc_rate
                if 0 <= step_sample < int(math.ceil(rtc_rate * 20.0))
                else None
            )
            impulsive_time = (
                scan_start + impulsive_sample / rtc_rate
                if 0 <= impulsive_sample < int(math.ceil(rtc_rate * 20.0))
                else None
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    **scan_record,
                    "network": int(network),
                    "array": _array_name(int(network)),
                    "rack": _rack(int(network)),
                    "n_det_input": int(
                        variables["rtc_network_n_det_input"][
                            scan_row, network_column
                        ]
                    ),
                    "n_det_step_used": step_used,
                    "step_score_median": _finite_or_none(
                        variables["rtc_network_step_score_median"][
                            scan_row, network_column
                        ]
                    ),
                    "step_score_max": _finite_or_none(
                        variables["rtc_network_step_score_max"][
                            scan_row, network_column
                        ]
                    ),
                    "step_detector_fraction": _finite_or_none(step_fraction),
                    "step_alignment_fraction": _finite_or_none(step_alignment),
                    "step_dominant_sample": (
                        step_sample if step_sample != MISSING_INT else None
                    ),
                    "step_event_time_unix_sec": step_time,
                    "step_event_time_utc": (
                        _utc_iso(step_time) if step_time is not None else None
                    ),
                    "step_local_candidate": bool(
                        step_used >= step_min_used
                        and np.isfinite(step_fraction)
                        and step_fraction >= step_min_fraction
                        and np.isfinite(step_alignment)
                        and step_alignment >= step_min_alignment
                    ),
                    "step_mask_applied": bool(
                        variables["rtc_network_step_mask_applied"][
                            scan_row, network_column
                        ]
                    ),
                    "n_det_impulsive_used": impulsive_used,
                    "impulsive_score_median": _finite_or_none(
                        variables["rtc_network_impulsive_score_median"][
                            scan_row, network_column
                        ]
                    ),
                    "impulsive_score_max": _finite_or_none(
                        variables["rtc_network_impulsive_score_max"][
                            scan_row, network_column
                        ]
                    ),
                    "impulsive_detector_fraction": _finite_or_none(
                        impulsive_fraction
                    ),
                    "impulsive_alignment_fraction": _finite_or_none(
                        impulsive_alignment
                    ),
                    "impulsive_dominant_sample": (
                        impulsive_sample
                        if impulsive_sample != MISSING_INT
                        else None
                    ),
                    "impulsive_event_time_unix_sec": impulsive_time,
                    "impulsive_event_time_utc": (
                        _utc_iso(impulsive_time)
                        if impulsive_time is not None
                        else None
                    ),
                    "impulsive_local_candidate": bool(
                        impulsive_used >= impulsive_min_used
                        and np.isfinite(impulsive_fraction)
                        and impulsive_fraction >= impulsive_min_fraction
                        and np.isfinite(impulsive_alignment)
                        and impulsive_alignment >= impulsive_min_alignment
                    ),
                    "impulsive_mask_applied": bool(
                        variables["rtc_network_impulsive_mask_applied"][
                            scan_row, network_column
                        ]
                    ),
                    "rtc_path": str(rtc_path),
                    "telescope_path": str(telescope_path),
                }
            )
    return rows, scan_rows


def _median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else None


def _maximum(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else None


def _aggregate_hk_network_bins(
    scan_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in scan_rows:
        hk_index = row["hk_sample_index_zero_based"]
        if hk_index is None:
            continue
        key = (int(row["obsnum"]), int(hk_index), int(row["network"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (obsnum, hk_index, network), selected in sorted(grouped.items()):
        exposure = sum(float(row["duration_sec"]) for row in selected)
        step_count = sum(bool(row["step_local_candidate"]) for row in selected)
        step_applied = sum(bool(row["step_mask_applied"]) for row in selected)
        impulsive_count = sum(
            bool(row["impulsive_local_candidate"]) for row in selected
        )
        impulsive_applied = sum(
            bool(row["impulsive_mask_applied"]) for row in selected
        )
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "hk_sample_index_zero_based": hk_index,
                "network": network,
                "array": _array_name(network),
                "rack": _rack(network),
                "n_scans": len(selected),
                "exposure_sec": exposure,
                "median_step_detector_fraction": _median(
                    row["step_detector_fraction"] for row in selected
                ),
                "maximum_step_detector_fraction": _maximum(
                    row["step_detector_fraction"] for row in selected
                ),
                "step_local_candidate_count": step_count,
                "step_local_candidate_rate_per_min": (
                    60.0 * step_count / exposure if exposure > 0.0 else None
                ),
                "step_mask_applied_count": step_applied,
                "step_mask_applied_rate_per_min": (
                    60.0 * step_applied / exposure if exposure > 0.0 else None
                ),
                "median_impulsive_detector_fraction": _median(
                    row["impulsive_detector_fraction"] for row in selected
                ),
                "maximum_impulsive_detector_fraction": _maximum(
                    row["impulsive_detector_fraction"] for row in selected
                ),
                "impulsive_local_candidate_count": impulsive_count,
                "impulsive_local_candidate_rate_per_min": (
                    60.0 * impulsive_count / exposure
                    if exposure > 0.0
                    else None
                ),
                "impulsive_mask_applied_count": impulsive_applied,
                "impulsive_mask_applied_rate_per_min": (
                    60.0 * impulsive_applied / exposure
                    if exposure > 0.0
                    else None
                ),
                "median_hk_sample_age_sec": _median(
                    row["hk_sample_age_sec"] for row in selected
                ),
            }
        )
    return output


def _hk_wide(
    hk_rows: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    output: dict[tuple[int, int], dict[str, Any]] = {}
    for row in hk_rows:
        key = (
            int(row["obsnum"]),
            int(row["hk_sample_index_zero_based"]),
        )
        record = output.setdefault(
            key,
            {
                "obsnum": key[0],
                "hk_sample_index_zero_based": key[1],
                "hk_time_unix_sec": None,
                "hk_time_utc": None,
            },
        )
        if row["channel_id"] == "T8":
            record["hk_time_unix_sec"] = row["sample_time_unix_sec"]
            record["hk_time_utc"] = row["sample_time_utc"]
        record[str(row["channel_id"])] = row["value"]
    return output


def _group_contrast_rows(
    bins: list[dict[str, Any]],
    hk_rows: list[dict[str, Any]],
    *,
    affected_networks: set[int],
    control_networks: set[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in bins:
        key = (
            int(row["obsnum"]),
            int(row["hk_sample_index_zero_based"]),
        )
        grouped.setdefault(key, []).append(row)
    hk_values = _hk_wide(hk_rows)
    output: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items()):
        affected = [
            row for row in selected if int(row["network"]) in affected_networks
        ]
        controls = [
            row for row in selected if int(row["network"]) in control_networks
        ]
        if not affected or not controls:
            continue
        affected_step = _median(
            row["median_step_detector_fraction"] for row in affected
        )
        control_step = _median(
            row["median_step_detector_fraction"] for row in controls
        )
        affected_rate = _median(
            row["step_local_candidate_rate_per_min"] for row in affected
        )
        control_rate = _median(
            row["step_local_candidate_rate_per_min"] for row in controls
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            **hk_values.get(key, {}),
            "n_affected_networks": len(affected),
            "n_control_networks": len(controls),
            "affected_median_step_detector_fraction": affected_step,
            "control_median_step_detector_fraction": control_step,
            "affected_minus_control_step_fraction": (
                affected_step - control_step
                if affected_step is not None and control_step is not None
                else None
            ),
            "affected_median_step_event_rate_per_min": affected_rate,
            "control_median_step_event_rate_per_min": control_rate,
            "affected_minus_control_event_rate_per_min": (
                affected_rate - control_rate
                if affected_rate is not None and control_rate is not None
                else None
            ),
        }
        output.append(record)
    return output


def _benjamini_hochberg(p_values: Iterable[float | None]) -> list[float | None]:
    values = list(p_values)
    finite = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and np.isfinite(value)
    ]
    result: list[float | None] = [None] * len(values)
    if not finite:
        return result
    ordered = sorted(finite, key=lambda item: item[1])
    running = 1.0
    count = len(ordered)
    for reverse_rank, (index, value) in enumerate(reversed(ordered), start=1):
        rank = count - reverse_rank + 1
        running = min(running, value * count / rank)
        result[index] = min(1.0, running)
    return result


def _temperature_associations(
    contrast_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "affected_median_step_detector_fraction",
        "control_median_step_detector_fraction",
        "affected_minus_control_step_fraction",
    )
    associations: list[dict[str, Any]] = []
    obsnums = sorted({int(row["obsnum"]) for row in contrast_rows})
    scopes: list[tuple[str, int | None, list[dict[str, Any]]]] = [
        (
            "all_observations_level_confounded",
            None,
            list(contrast_rows),
        )
    ]
    scopes.extend(
        (
            "within_observation",
            obsnum,
            [row for row in contrast_rows if int(row["obsnum"]) == obsnum],
        )
        for obsnum in obsnums
    )
    for scope, obsnum, selected in scopes:
        for channel in HOUSEKEEPING_CHANNELS:
            for metric in metrics:
                x = np.asarray(
                    [
                        row.get(channel.channel_id, math.nan)
                        if row.get(channel.channel_id) is not None
                        else math.nan
                        for row in selected
                    ],
                    dtype=float,
                )
                y = np.asarray(
                    [
                        row.get(metric, math.nan)
                        if row.get(metric) is not None
                        else math.nan
                        for row in selected
                    ],
                    dtype=float,
                )
                valid = (
                    np.isfinite(x)
                    & (x > 0.0)
                    & np.isfinite(y)
                )
                if (
                    np.count_nonzero(valid) >= 5
                    and np.unique(x[valid]).size > 1
                    and np.unique(y[valid]).size > 1
                ):
                    rho, p_value = spearmanr(x[valid], y[valid])
                    rho_value = _finite_or_none(rho)
                    p_value_value = _finite_or_none(p_value)
                else:
                    rho_value = None
                    p_value_value = None
                associations.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "scope": scope,
                        "obsnum": obsnum,
                        "channel_group": channel.group,
                        "channel_id": channel.channel_id,
                        "channel_name": channel.name,
                        "metric": metric,
                        "n_samples": int(np.count_nonzero(valid)),
                        "spearman_rho": rho_value,
                        "p_value_uncorrected": p_value_value,
                        "q_value_bh": None,
                        "interpretation": (
                            "descriptive_only_autocorrelated_and_time_confounded"
                        ),
                    }
                )
    q_values = _benjamini_hochberg(
        row["p_value_uncorrected"] for row in associations
    )
    for row, q_value in zip(associations, q_values):
        row["q_value_bh"] = q_value
    return associations


def _observation_network_summary(
    scan_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in scan_rows:
        key = (int(row["obsnum"]), int(row["network"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (obsnum, network), selected in sorted(grouped.items()):
        exposure = sum(float(row["duration_sec"]) for row in selected)
        step_count = sum(bool(row["step_local_candidate"]) for row in selected)
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "network": network,
                "array": _array_name(network),
                "rack": _rack(network),
                "n_scans": len(selected),
                "exposure_sec": exposure,
                "median_step_detector_fraction": _median(
                    row["step_detector_fraction"] for row in selected
                ),
                "maximum_step_detector_fraction": _maximum(
                    row["step_detector_fraction"] for row in selected
                ),
                "step_local_candidate_count": step_count,
                "step_local_candidate_rate_per_min": (
                    60.0 * step_count / exposure if exposure > 0.0 else None
                ),
                "step_mask_applied_count": sum(
                    bool(row["step_mask_applied"]) for row in selected
                ),
                "median_impulsive_detector_fraction": _median(
                    row["impulsive_detector_fraction"] for row in selected
                ),
                "impulsive_mask_applied_count": sum(
                    bool(row["impulsive_mask_applied"]) for row in selected
                ),
            }
        )
    return output


def _read_pointing_scan_network_metrics(
    *,
    obsnum: int,
    pointing_reduction_root: Path,
) -> tuple[list[dict[str, Any]], Path]:
    rtc_path = (
        pointing_reduction_root
        / str(obsnum)
        / "raw"
        / f"toltec_commissioning_pointing_{obsnum}_rtcdiag.nc"
    )
    if not rtc_path.is_file():
        raise FileNotFoundError(rtc_path)
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
    if fractions.shape != (output_scan.size, networks.size):
        raise ValueError(
            f"{rtc_path.name}: step fraction shape {fractions.shape} "
            f"does not match {(output_scan.size, networks.size)}"
        )
    rows: list[dict[str, Any]] = []
    for scan_row, scan_number in enumerate(output_scan):
        for network_column, network in enumerate(networks):
            rows.append(
                {
                    "obsnum": int(obsnum),
                    "citlali_scan_one_based": int(scan_number),
                    "scan_row_zero_based": int(scan_row),
                    "network": int(network),
                    "array": _array_name(int(network)),
                    "rack": _rack(int(network)),
                    "duration_sec": float(durations[scan_row]),
                    "step_detector_fraction": _finite_or_none(
                        fractions[scan_row, network_column]
                    ),
                    "rtc_path": str(rtc_path),
                }
            )
    return rows, rtc_path


def _night_chronology_rows(
    *,
    science_rows: list[dict[str, Any]],
    pointing_rows: list[dict[str, Any]],
    hk_rows: list[dict[str, Any]],
    affected_networks: set[int],
    control_networks: set[int],
) -> list[dict[str, Any]]:
    typed_rows = [
        ("science", row) for row in science_rows
    ] + [("pointing", row) for row in pointing_rows]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for observation_type, row in typed_rows:
        grouped.setdefault(
            (observation_type, int(row["obsnum"])), []
        ).append(row)

    hk_grouped: dict[int, list[dict[str, Any]]] = {}
    for row in hk_rows:
        hk_grouped.setdefault(int(row["obsnum"]), []).append(row)

    output: list[dict[str, Any]] = []
    for (observation_type, obsnum), selected in grouped.items():
        affected = [
            float(row["step_detector_fraction"])
            for row in selected
            if int(row["network"]) in affected_networks
            and row["step_detector_fraction"] is not None
        ]
        controls = [
            float(row["step_detector_fraction"])
            for row in selected
            if int(row["network"]) in control_networks
            and row["step_detector_fraction"] is not None
        ]
        if not affected or not controls:
            continue
        affected_median = float(np.median(affected))
        control_median = float(np.median(controls))
        observation_hk = hk_grouped.get(obsnum, [])
        valid_times = [
            float(row["sample_time_unix_sec"])
            for row in observation_hk
            if row["channel_id"] == "T8"
            and row["sample_time_unix_sec"] is not None
        ]
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "obsnum": obsnum,
            "observation_type": observation_type,
            "start_time_unix_sec": min(valid_times) if valid_times else None,
            "start_time_utc": (
                _utc_iso(min(valid_times)) if valid_times else None
            ),
            "n_scan_network_cells": len(selected),
            "affected_median_step_detector_fraction": affected_median,
            "control_median_step_detector_fraction": control_median,
            "affected_minus_control_step_fraction": (
                affected_median - control_median
            ),
        }
        for channel in HOUSEKEEPING_CHANNELS:
            values = [
                float(row["value"])
                for row in observation_hk
                if row["channel_id"] == channel.channel_id
                and row["value"] is not None
            ]
            record[channel.channel_id] = (
                float(np.median(values)) if values else None
            )
        output.append(record)
    return sorted(
        output,
        key=lambda row: (
            float(row["start_time_unix_sec"])
            if row["start_time_unix_sec"] is not None
            else math.inf
        ),
    )


def _select_raw_validation_rows(
    scan_rows: list[dict[str, Any]],
    *,
    per_class: int,
) -> list[dict[str, Any]]:
    if per_class <= 0:
        return []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in scan_rows:
        key = (int(row["obsnum"]), int(row["network"]))
        grouped.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    for _, rows in sorted(grouped.items()):
        interior = [
            row
            for row in rows
            if int(row["scan_row_zero_based"]) not in {0, len(rows) - 1}
            and row["step_detector_fraction"] is not None
        ]
        ordered = sorted(
            interior,
            key=lambda row: float(row["step_detector_fraction"]),
        )
        for row in ordered[:per_class]:
            selected.append({**row, "selection_class": "rtc_quiet"})
        for row in ordered[-per_class:]:
            selected.append({**row, "selection_class": "rtc_event_rich"})
    return selected


def _raw_validation(
    selected_rows: list[dict[str, Any]],
    *,
    data_root: Path,
    apt_root: Path,
    subobsnum: int,
    raw_file_scan: int,
    sigma_threshold: float,
    min_phase_mrad: float,
    pre_window_sec: float,
    guard_window_sec: float,
    post_window_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    apt_cache: dict[int, Table] = {}
    raw_cache: dict[tuple[int, int], Path] = {}
    for selected in selected_rows:
        obsnum = int(selected["obsnum"])
        network = int(selected["network"])
        if obsnum not in apt_cache:
            apt_path = apt_root / f"apt_{obsnum}_matched.ecsv"
            if not apt_path.is_file():
                raise FileNotFoundError(apt_path)
            apt_cache[obsnum] = Table.read(apt_path)
        key = (obsnum, network)
        if key not in raw_cache:
            raw_path = iq_tool._find_raw_file(
                data_root,
                network=network,
                obsnum=obsnum,
                subobsnum=subobsnum,
                raw_file_scan=raw_file_scan,
            )
            if raw_path is None:
                raise FileNotFoundError(
                    f"no raw science file for obs {obsnum}, nw{network}"
                )
            raw_cache[key] = raw_path
            inputs.append(
                {
                    "obsnum": obsnum,
                    "network": network,
                    "raw_path": str(raw_path),
                    "apt_path": str(
                        apt_root / f"apt_{obsnum}_matched.ecsv"
                    ),
                }
            )
        raw_path = raw_cache[key]
        data = iq_tool._load_network(
            raw_path,
            network=network,
            scan_start_sec=float(selected["start_time_unix_sec"]),
            scan_end_sec=float(selected["end_time_unix_sec"]),
            apt=apt_cache[obsnum],
        )
        classified = _classify_raw_event(
            data,
            sigma_threshold=sigma_threshold,
            min_phase_mrad=min_phase_mrad,
            pre_window_sec=pre_window_sec,
            guard_window_sec=guard_window_sec,
            post_window_sec=post_window_sec,
        )
        raw_event_time = (
            float(selected["start_time_unix_sec"])
            + float(classified["raw_event_sec"])
        )
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "network": network,
                "array": _array_name(network),
                "citlali_scan_one_based": int(
                    selected["citlali_scan_one_based"]
                ),
                "selection_class": selected["selection_class"],
                "rtc_step_detector_fraction": selected[
                    "step_detector_fraction"
                ],
                "rtc_step_alignment_fraction": selected[
                    "step_alignment_fraction"
                ],
                "rtc_step_score_max": selected["step_score_max"],
                "raw_coherent_same_sign_fraction": classified[
                    "coherent_same_sign_fraction"
                ],
                "raw_strong_phase_fraction": classified[
                    "strong_phase_fraction"
                ],
                "raw_same_phase_sign_fraction": classified[
                    "same_phase_sign_fraction"
                ],
                "raw_n_apt_usable_tones": classified[
                    "n_apt_usable_tones"
                ],
                "raw_n_strong_phase_tones": classified[
                    "n_strong_phase_tones"
                ],
                "raw_event_time_unix_sec": raw_event_time,
                "raw_event_time_utc": _utc_iso(raw_event_time),
                "raw_path": str(raw_path),
            }
        )
    return output, inputs


def _make_overview_figure(
    path: Path,
    *,
    scan_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    obsnums: list[int],
) -> None:
    networks = sorted({int(row["network"]) for row in scan_rows})
    fig, axes = plt.subplots(
        len(obsnums),
        2,
        figsize=(16, 3.3 * len(obsnums)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row_index, obsnum in enumerate(obsnums):
        selected = [
            row for row in scan_rows if int(row["obsnum"]) == obsnum
        ]
        scans = sorted(
            {int(row["citlali_scan_one_based"]) for row in selected}
        )
        fraction = np.full((len(networks), len(scans)), np.nan)
        for row in selected:
            network_row = networks.index(int(row["network"]))
            scan_column = scans.index(int(row["citlali_scan_one_based"]))
            value = row["step_detector_fraction"]
            fraction[network_row, scan_column] = (
                float(value) if value is not None else math.nan
            )
        ax_heat, ax_trend = axes[row_index]
        image = ax_heat.imshow(
            fraction,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=0.8,
            cmap="magma",
            extent=(0.0, len(scans) * 10.0 / 60.0, -0.5, len(networks) - 0.5),
        )
        ax_heat.set_yticks(range(len(networks)), networks)
        ax_heat.set_xlabel("elapsed observation time (min)")
        ax_heat.set_ylabel("network ID")
        ax_heat.set_title(f"obs {obsnum}: RTC step-detector fraction")

        contrast = sorted(
            [
                row
                for row in contrast_rows
                if int(row["obsnum"]) == obsnum
            ],
            key=lambda row: float(row["hk_time_unix_sec"]),
        )
        if contrast:
            first_time = float(contrast[0]["hk_time_unix_sec"])
            elapsed = np.asarray(
                [
                    (float(row["hk_time_unix_sec"]) - first_time) / 60.0
                    for row in contrast
                ]
            )
            affected = np.asarray(
                [
                    float(row["affected_median_step_detector_fraction"])
                    for row in contrast
                ]
            )
            control = np.asarray(
                [
                    float(row["control_median_step_detector_fraction"])
                    for row in contrast
                ]
            )
            ax_trend.plot(elapsed, affected, "o-", label="affected networks")
            ax_trend.plot(elapsed, control, "o-", label="control networks")
            ax_trend.set_ylim(-0.03, 0.85)
            ax_trend.set_ylabel("median RTC step-detector fraction")
            ax_temperature = ax_trend.twinx()
            for channel_id, color in (("Temperature5", "C2"), ("T8", "C3")):
                values = np.asarray(
                    [
                        float(row[channel_id])
                        if row.get(channel_id) is not None
                        else math.nan
                        for row in contrast
                    ]
                )
                baseline = float(np.nanmedian(values))
                ax_temperature.plot(
                    elapsed,
                    1.0e3 * (values - baseline),
                    ".--",
                    color=color,
                    alpha=0.75,
                    label=f"{channel_id} ΔT",
                )
            ax_temperature.set_ylabel("temperature change from median (mK)")
            lines = ax_trend.lines + ax_temperature.lines
            ax_trend.legend(
                lines,
                [line.get_label() for line in lines],
                loc="upper left",
                fontsize=8,
                ncol=2,
            )
        ax_trend.set_xlabel("elapsed HK time (min)")
        ax_trend.set_title(
            f"obs {obsnum}: network contrast and slow thermometry"
        )
    if image is not None:
        fig.colorbar(
            image,
            ax=axes[:, 0],
            label="fraction of detectors",
            shrink=0.9,
        )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_raw_validation_figure(
    path: Path,
    *,
    rows: list[dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if rows:
        for selection, marker in (
            ("rtc_quiet", "o"),
            ("rtc_event_rich", "^"),
        ):
            selected = [
                row for row in rows if row["selection_class"] == selection
            ]
            x = [
                float(row["rtc_step_detector_fraction"]) for row in selected
            ]
            y = [
                float(row["raw_coherent_same_sign_fraction"])
                for row in selected
            ]
            colors = [int(row["network"]) for row in selected]
            scatter = ax.scatter(
                x,
                y,
                c=colors,
                cmap="turbo",
                vmin=0,
                vmax=12,
                marker=marker,
                alpha=0.8,
                label=selection,
            )
        fig.colorbar(scatter, ax=ax, label="network ID")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle=":", label="one-to-one")
    ax.axhline(0.10, color="0.7", linestyle="--", linewidth=0.8)
    ax.axvline(0.10, color="0.7", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Citlali RTC step-detector fraction")
    ax.set_ylabel("independent raw-I/Q coherent phase fraction")
    ax.set_title("Stratified raw-I/Q validation of science chunks")
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_night_chronology_figure(
    path: Path,
    *,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError("night chronology is empty")
    ordered = sorted(rows, key=lambda row: float(row["start_time_unix_sec"]))
    first_time = float(ordered[0]["start_time_unix_sec"])
    elapsed_hour = np.asarray(
        [
            (float(row["start_time_unix_sec"]) - first_time) / 3600.0
            for row in ordered
        ]
    )
    markers = [
        "o" if row["observation_type"] == "science" else "s"
        for row in ordered
    ]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        sharex=True,
        constrained_layout=True,
    )

    affected = np.asarray(
        [float(row["affected_median_step_detector_fraction"]) for row in ordered]
    )
    controls = np.asarray(
        [float(row["control_median_step_detector_fraction"]) for row in ordered]
    )
    contrast = affected - controls
    for values, color, label in (
        (affected, "C0", "affected networks"),
        (controls, "C1", "control networks"),
        (contrast, "C3", "affected minus control"),
    ):
        axes[0].plot(elapsed_hour, values, color=color, alpha=0.65)
        for x_value, y_value, marker in zip(elapsed_hour, values, markers):
            axes[0].scatter(
                x_value,
                y_value,
                color=color,
                marker=marker,
                s=45,
            )
        axes[0].plot([], [], color=color, marker="o", label=label)
    axes[0].axhline(0.0, color="0.7", linewidth=0.8)
    axes[0].set_ylabel("median RTC step-detector fraction")
    axes[0].set_title("Night chronology: network-selective transition")
    axes[0].legend(loc="upper left", ncol=3)

    for ax, channels, title in (
        (
            axes[1],
            (("Temperature4", "4K busbar"), ("Temperature13", "LS front")),
            "warm-stage thermometry",
        ),
        (
            axes[2],
            (("T8", "MC plate"), ("T12", "MC bar")),
            "cold-stage thermometry",
        ),
    ):
        for color_index, (channel_id, label) in enumerate(channels):
            values = np.asarray(
                [
                    float(row[channel_id])
                    if row.get(channel_id) is not None
                    else math.nan
                    for row in ordered
                ]
            )
            delta_mk = 1.0e3 * (values - values[0])
            color = f"C{color_index + 2}"
            ax.plot(
                elapsed_hour,
                delta_mk,
                color=color,
                alpha=0.65,
                label=label,
            )
            for x_value, y_value, marker in zip(
                elapsed_hour, delta_mk, markers
            ):
                ax.scatter(
                    x_value,
                    y_value,
                    color=color,
                    marker=marker,
                    s=45,
                )
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.set_ylabel("change from first observation (mK)")
        ax.set_title(title)
        ax.legend(loc="upper left", ncol=2)

    transition = next(
        (
            index
            for index, row in enumerate(ordered)
            if int(row["obsnum"]) == 152419
        ),
        None,
    )
    if transition is not None and transition > 0:
        transition_hour = 0.5 * (
            elapsed_hour[transition - 1] + elapsed_hour[transition]
        )
        for ax in axes:
            ax.axvline(
                transition_hour,
                color="0.25",
                linestyle=":",
                linewidth=1.2,
            )
        axes[0].text(
            transition_hour,
            0.98,
            "transition bracketed by 152418 / 152419",
            transform=axes[0].get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=9,
        )

    axes[2].set_xticks(
        elapsed_hour,
        [
            f"{int(row['obsnum'])}\n"
            f"{'S' if row['observation_type'] == 'science' else 'P'}"
            for row in ordered
        ],
        rotation=45,
        ha="right",
    )
    axes[2].set_xlabel("observation number (S=science, P=pointing)")
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
    parser.add_argument("--pointing-reduction-root", type=Path)
    parser.add_argument(
        "--pointing-obsnums",
        type=int,
        nargs="+",
        default=list(DEFAULT_POINTING_OBSNUMS),
    )
    parser.add_argument("--subobsnum", type=int, default=0)
    parser.add_argument("--raw-file-scan", type=int, default=2)
    parser.add_argument("--max-hk-age-sec", type=float, default=35.0)
    parser.add_argument("--raw-validation-per-class", type=int, default=1)
    parser.add_argument("--sigma-threshold", type=float, default=8.0)
    parser.add_argument("--min-phase-mrad", type=float, default=5.0)
    parser.add_argument("--pre-window-sec", type=float, default=0.20)
    parser.add_argument("--guard-window-sec", type=float, default=0.05)
    parser.add_argument("--post-window-sec", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_hk_age_sec <= 0.0:
        raise ValueError("--max-hk-age-sec must be positive")
    if args.raw_validation_per_class < 0:
        raise ValueError("--raw-validation-per-class cannot be negative")
    affected = {int(network) for network in args.affected_networks}
    controls = {int(network) for network in args.control_networks}
    overlap = sorted(affected & controls)
    if overlap:
        raise ValueError(f"affected/control network sets overlap: {overlap}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_telescope = _find_one(
        args.data_root,
        f"tel_toltec_*_{int(args.obsnums[0])}_00_0002.nc",
    )
    with netCDF4.Dataset(first_telescope) as ds:
        night_start_sec = float(
            ds.variables["Data.TelescopeBackend.TelTime"][0]
        )

    hk_rows: list[dict[str, Any]] = []
    scan_network_rows: list[dict[str, Any]] = []
    pointing_scan_network_rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    for obsnum in args.obsnums:
        obsnum = int(obsnum)
        rtc_path = _science_rtc_path(args.reduction_root, obsnum)
        telescope_path = _find_one(
            args.data_root,
            f"tel_toltec_*_{obsnum}_00_0002.nc",
        )
        hk_path = _find_one(
            args.data_root,
            f"toltec_hk_*_{obsnum}_00_0002.nc",
        )
        obs_hk_rows, hk_arrays = _read_housekeeping(
            hk_path,
            obsnum=obsnum,
        )
        hk_rows.extend(obs_hk_rows)
        reference_time = hk_arrays["T8"][0]
        obs_scan_rows, _ = _read_scan_network_metrics(
            obsnum=obsnum,
            rtc_path=rtc_path,
            telescope_path=telescope_path,
            hk_reference_time=reference_time,
            max_hk_age_sec=float(args.max_hk_age_sec),
            night_start_sec=night_start_sec,
        )
        scan_network_rows.extend(obs_scan_rows)
        input_records.append(
            {
                "obsnum": obsnum,
                "rtc_path": str(rtc_path),
                "telescope_path": str(telescope_path),
                "housekeeping_path": str(hk_path),
            }
        )

    pointing_input_records: list[dict[str, Any]] = []
    if args.pointing_reduction_root is not None:
        for obsnum in args.pointing_obsnums:
            obsnum = int(obsnum)
            obs_pointing_rows, rtc_path = (
                _read_pointing_scan_network_metrics(
                    obsnum=obsnum,
                    pointing_reduction_root=args.pointing_reduction_root,
                )
            )
            pointing_scan_network_rows.extend(obs_pointing_rows)
            hk_path = _find_one(
                args.data_root,
                f"toltec_hk_*_{obsnum}_00_0002.nc",
            )
            obs_hk_rows, _ = _read_housekeeping(hk_path, obsnum=obsnum)
            hk_rows.extend(obs_hk_rows)
            pointing_input_records.append(
                {
                    "obsnum": obsnum,
                    "rtc_path": str(rtc_path),
                    "housekeeping_path": str(hk_path),
                }
            )

    hk_bins = _aggregate_hk_network_bins(scan_network_rows)
    contrast_rows = _group_contrast_rows(
        hk_bins,
        hk_rows,
        affected_networks=affected,
        control_networks=controls,
    )
    association_rows = _temperature_associations(contrast_rows)
    observation_rows = _observation_network_summary(scan_network_rows)
    chronology_rows = _night_chronology_rows(
        science_rows=scan_network_rows,
        pointing_rows=pointing_scan_network_rows,
        hk_rows=hk_rows,
        affected_networks=affected,
        control_networks=controls,
    )
    selected_raw_rows = _select_raw_validation_rows(
        scan_network_rows,
        per_class=int(args.raw_validation_per_class),
    )
    raw_rows, raw_inputs = _raw_validation(
        selected_raw_rows,
        data_root=args.data_root,
        apt_root=args.apt_root,
        subobsnum=int(args.subobsnum),
        raw_file_scan=int(args.raw_file_scan),
        sigma_threshold=float(args.sigma_threshold),
        min_phase_mrad=float(args.min_phase_mrad),
        pre_window_sec=float(args.pre_window_sec),
        guard_window_sec=float(args.guard_window_sec),
        post_window_sec=float(args.post_window_sec),
    )

    outputs = {
        "scan_network_metrics": "science_scan_network_metrics.csv",
        "housekeeping_samples": "science_housekeeping_samples.csv",
        "hk_network_bins": "science_hk_network_bins.csv",
        "hk_group_contrast": "science_hk_group_contrast.csv",
        "temperature_associations": "science_temperature_associations.csv",
        "observation_network_summary": (
            "science_observation_network_summary.csv"
        ),
        "raw_validation": "science_raw_iq_validation.csv",
        "overview_figure": "science_iq_temperature_overview.png",
        "raw_validation_figure": "science_raw_iq_validation.png",
    }
    if chronology_rows:
        outputs["night_chronology"] = "night_observation_chronology.csv"
        outputs["night_chronology_figure"] = (
            "night_event_temperature_chronology.png"
        )
    if not raw_rows:
        outputs.pop("raw_validation")
    _write_csv(
        args.output_dir / outputs["scan_network_metrics"],
        scan_network_rows,
    )
    _write_csv(
        args.output_dir / outputs["housekeeping_samples"],
        hk_rows,
    )
    _write_csv(args.output_dir / outputs["hk_network_bins"], hk_bins)
    _write_csv(
        args.output_dir / outputs["hk_group_contrast"],
        contrast_rows,
    )
    _write_csv(
        args.output_dir / outputs["temperature_associations"],
        association_rows,
    )
    _write_csv(
        args.output_dir / outputs["observation_network_summary"],
        observation_rows,
    )
    if raw_rows:
        _write_csv(
            args.output_dir / outputs["raw_validation"],
            raw_rows,
        )
    if chronology_rows:
        _write_csv(
            args.output_dir / outputs["night_chronology"],
            chronology_rows,
        )
    _make_overview_figure(
        args.output_dir / outputs["overview_figure"],
        scan_rows=scan_network_rows,
        contrast_rows=contrast_rows,
        obsnums=[int(value) for value in args.obsnums],
    )
    _make_raw_validation_figure(
        args.output_dir / outputs["raw_validation_figure"],
        rows=raw_rows,
    )
    if chronology_rows:
        _make_night_chronology_figure(
            args.output_dir / outputs["night_chronology_figure"],
            rows=chronology_rows,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "description": (
            "Science RTC event population, slow housekeeping association, "
            "and stratified independent raw-I/Q validation"
        ),
        "obsnums": [int(value) for value in args.obsnums],
        "affected_networks": sorted(affected),
        "control_networks": sorted(controls),
        "thresholds": {
            "max_hk_sample_age_sec": float(args.max_hk_age_sec),
            "raw_validation_per_class": int(
                args.raw_validation_per_class
            ),
            "raw_sigma_threshold": float(args.sigma_threshold),
            "raw_min_phase_mrad": float(args.min_phase_mrad),
            "raw_pre_window_sec": float(args.pre_window_sec),
            "raw_guard_window_sec": float(args.guard_window_sec),
            "raw_post_window_sec": float(args.post_window_sec),
        },
        "semantics": {
            "rtc_chunking": (
                "Persisted Citlali forced-duration chunks, approximately 10 s"
            ),
            "step_detector_fraction": (
                "Fraction published by Citlali RTC diagnostics; not a "
                "statistical significance or probability"
            ),
            "raw_coherent_same_sign_fraction": (
                "Maximum fraction of matched-APT usable raw tones with a "
                "same-sign phase change in the selected chunk"
            ),
            "housekeeping_join": (
                "Nearest measured T8 housekeeping epoch within the configured "
                "maximum age; other channels retain their measured sample "
                "values. No subsecond trigger time is inferred."
            ),
            "temperature_associations": (
                "Descriptive Spearman associations. Samples are "
                "autocorrelated and temperature covaries with elapsed night "
                "time; p/q values do not establish causality."
            ),
        },
        "inputs": input_records,
        "pointing_inputs": pointing_input_records,
        "raw_validation_inputs": raw_inputs,
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
