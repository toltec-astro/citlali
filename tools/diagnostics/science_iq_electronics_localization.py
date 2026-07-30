#!/usr/bin/env python3
"""Localize the stable raw-I/Q event mode in recorded electronics coordinates.

This diagnostic follows the schema-v2 event vectors produced by
``science_iq_event_vector_analysis.py``.  It deliberately separates:

* detector identity (APT ``uid``);
* raw tone-list slot (an observation-local array index);
* signed digital tone offset from the network LO;
* absolute probe/RF frequency; and
* coordinates that are absent from the data (PFB channel, hardware lane,
  subband, and authoritative firmware version).

The analysis uses leave-one-observation-out prediction to compare coordinate
ownership models.  It also evaluates the science-derived mode on independent
pointing events and on fixed epochs in clean pointing observations.  Every raw
tone is retained in the electronics inventory, and every APT-usable tone is
retained in event and null vectors; responders are never the denominator.

The optional DAC-comb FFT bin is explicitly provisional.  It follows a dated
software implementation using a 2**21 waveform and 512 MHz sample rate.  It is
not labeled as an authoritative FPGA/PFB coordinate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "citlali-iq-localization-mpl-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.table import Table  # noqa: E402
from scipy import stats  # noqa: E402

from tools.diagnostics import pointing_iq_event_coherence as iq_tool  # noqa: E402


SCHEMA_VERSION = "citlali-science-iq-electronics-localization-v1"
REQUIRED_EVENT_SCHEMA = "citlali-science-iq-event-vector-v2"
DEFAULT_NETWORKS = (1, 2, 3, 4, 8, 9)
DEFAULT_EVENT_OBSNUMS = (152419, 152431, 152433)
DEFAULT_POINTING_EVENT_OBSNUMS = (152420, 152432, 152434)
DEFAULT_NULL_OBSNUMS = (152390, 152418)
RAW_PATTERN = re.compile(
    r"toltec(?P<network>\d+)_(?P<obsnum>\d{6})_000_0002_.*\.nc$"
)


@dataclass(frozen=True)
class CoordinateMode:
    coordinate: np.ndarray
    loading: np.ndarray
    singular_values: np.ndarray


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else math.nan


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


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(finite) < 2:
        return math.nan
    denominator = float(
        np.linalg.norm(first[finite]) * np.linalg.norm(second[finite])
    )
    if denominator <= 0.0:
        return math.nan
    return float(abs(np.dot(first[finite], second[finite])) / denominator)


def _array_name(network: int) -> str:
    if 0 <= int(network) <= 6:
        return "a1100"
    if 7 <= int(network) <= 10:
        return "a1400"
    if 11 <= int(network) <= 12:
        return "a2000"
    raise ValueError(f"invalid TolTEC network {network}")


def _rack(network: int) -> str:
    return "RACKA" if int(network) <= 6 else "RACKO"


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


def _decode_chars(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("character header is not one-dimensional")
    if array.dtype.kind == "S":
        pieces = array.tolist()
        terminator = next(
            (
                index
                for index, piece in enumerate(pieces)
                if piece in {b"", b"\0"}
            ),
            len(pieces),
        )
        raw = b"".join(pieces[:terminator])
        return raw.split(b"\0", 1)[0].decode(errors="replace")
    pieces = array.astype("U1").tolist()
    terminator = next(
        (
            index
            for index, piece in enumerate(pieces)
            if piece in {"", "\0"}
        ),
        len(pieces),
    )
    return "".join(pieces[:terminator]).split("\0", 1)[0]


def _scalar(dataset: netCDF4.Dataset, name: str) -> Any:
    if name not in dataset.variables:
        raise KeyError(f"missing required raw header {name}")
    return np.asarray(dataset.variables[name][...]).item()


def _optional_scalar(
    dataset: netCDF4.Dataset,
    name: str,
    default: Any = None,
) -> Any:
    if name not in dataset.variables:
        return default
    return np.asarray(dataset.variables[name][...]).item()


def _optional_chars(dataset: netCDF4.Dataset, name: str) -> str | None:
    if name not in dataset.variables:
        return None
    return _decode_chars(np.asarray(dataset.variables[name][...]))


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_unix_sec": float(stat.st_mtime),
    }


def _find_one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        names = ", ".join(path.name for path in paths)
        raise FileNotFoundError(
            f"expected one match for {root / pattern}, "
            f"found {len(paths)}: {names}"
        )
    return paths[0]


def _table_float(row: Any, field: str) -> float | None:
    if field not in row.colnames:
        return None
    try:
        value = float(row[field])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _apt_tone_map(
    apt_path: Path | None,
    *,
    network: int,
) -> dict[int, dict[str, Any]]:
    if apt_path is None:
        return {}
    apt = Table.read(apt_path, format="ascii.ecsv")
    rows = apt[np.asarray(apt["nw"], dtype=int) == int(network)]
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        tone_value = _table_float(row, "kids_tone")
        if tone_value is None:
            continue
        tone = int(tone_value)
        if tone in result:
            raise ValueError(
                f"{apt_path.name} nw{network}: duplicate kids_tone {tone}"
            )
        kids_flag = _table_float(row, "kids_flag")
        map_flag = _table_float(row, "flag")
        result[tone] = {
            "apt_uid": _table_float(row, "uid"),
            "apt_det_id": _table_float(row, "det_id"),
            "apt_kids_tone": tone,
            "apt_kids_fp_hz": _table_float(row, "kids_fp"),
            "apt_kids_fr_hz": _table_float(row, "kids_fr"),
            "apt_tone_frequency_hz": _table_float(row, "tone_freq"),
            "apt_frequency_group": _table_float(row, "fg"),
            "apt_polarization_group": _table_float(row, "pg"),
            "apt_location": _table_float(row, "loc"),
            "apt_orientation": _table_float(row, "ori"),
            "apt_kids_flag": kids_flag,
            "apt_map_flag": map_flag,
            "apt_usable": bool(kids_flag == 0.0 and map_flag == 0.0),
        }
    return result


def _authority_rows() -> list[dict[str, Any]]:
    raw = "TolTEC raw NetCDF header"
    apt = "matched APT joined by nw and kids_tone; UID is detector identity"
    return [
        {
            "coordinate": "observation/tune identity",
            "status": "authoritative",
            "source": (
                "Header.Toltec.ObsNum/SubObsNum/ScanNum and "
                "TargSweepObsNum/SubObsNum/ScanNum"
            ),
            "limitation": "",
        },
        {
            "coordinate": "detector UID",
            "status": "authoritative_when_apt_available",
            "source": apt,
            "limitation": "unavailable for raw observations lacking an APT",
        },
        {
            "coordinate": "APT identity",
            "status": "authoritative",
            "source": "exact matched-APT path and table row fields",
            "limitation": "",
        },
        {
            "coordinate": "signed digital tone offset",
            "status": "authoritative_recorded",
            "source": f"{raw} Header.Toltec.ToneFreq",
            "limitation": "name is historical; values are offsets from LO",
        },
        {
            "coordinate": "LO center",
            "status": "authoritative_recorded",
            "source": f"{raw} Header.Toltec.LoCenterFreq",
            "limitation": "",
        },
        {
            "coordinate": "probe/RF frequency",
            "status": "derived_from_authoritative_headers",
            "source": "LoCenterFreq + ToneFreq",
            "limitation": "",
        },
        {
            "coordinate": "raw tone-list slot",
            "status": "authoritative_array_index",
            "source": "zero-based index in Header.Toltec.ToneFreq",
            "limitation": (
                "observation-local list index; not identified as a PFB bin"
            ),
        },
        {
            "coordinate": "sideband",
            "status": "derived_from_authoritative_header",
            "source": "sign of Header.Toltec.ToneFreq",
            "limitation": "zero is represented separately",
        },
        {
            "coordinate": "provisional DAC-comb FFT bin",
            "status": "provisional_derived_not_firmware_authority",
            "source": (
                "tone offset with 2**21 and 512 MHz constants from "
                "tolteca_web toltecTonePowerViewer and taco_recipes "
                "toltec_tone_power2.py"
            ),
            "limitation": (
                "software calculation is dated and does not establish the "
                "deployed FPGA/PFB channel assignment"
            ),
        },
        {
            "coordinate": "FFT/PFB channel and within-bin position",
            "status": "unavailable",
            "source": "not present in inspected NetCDF/APT fields",
            "limitation": "requires deployed firmware/channel-map authority",
        },
        {
            "coordinate": "DAC lane/channel",
            "status": "unavailable",
            "source": "not present in inspected NetCDF/APT fields",
            "limitation": "requires hardware/firmware mapping",
        },
        {
            "coordinate": "ADC lane/channel",
            "status": "unavailable",
            "source": "not present in inspected NetCDF/APT fields",
            "limitation": "requires hardware/firmware mapping",
        },
        {
            "coordinate": "subband/channelizer path",
            "status": "unavailable",
            "source": "not present in inspected NetCDF/APT fields",
            "limitation": "requires hardware/firmware mapping",
        },
        {
            "coordinate": "ROACH/readout board",
            "status": "authoritative_recorded",
            "source": (
                "Header.Toltec.RoachIndex and RoachKatcpMacAddr"
            ),
            "limitation": "network identifies the board-level chain",
        },
        {
            "coordinate": "rack",
            "status": "study_topology",
            "source": "project-owner topology: networks 0-6 A, 7-12 O",
            "limitation": "not encoded in the raw NetCDF",
        },
        {
            "coordinate": "synthesizer/attenuator identity",
            "status": "authoritative_recorded_string",
            "source": (
                "Header.Toltec.SynthSerialNum and AttenSerialNum"
            ),
            "limitation": "attenuator string may contain null placeholders",
        },
        {
            "coordinate": "firmware/configuration version",
            "status": "partially_available",
            "source": "Header.Toltec.CompileTime and SelectedMask",
            "limitation": (
                "CompileTime semantics and deployed firmware version are not "
                "documented by the inspected products"
            ),
        },
    ]


def _trigger_inventory_rows() -> list[dict[str, Any]]:
    return [
        {
            "telemetry": "cryogenic temperatures",
            "availability": "time_aligned_slow_housekeeping",
            "result": "previous survey found no event-onset temperature spike",
            "classification": "trigger_negative_evidence",
        },
        {
            "telemetry": "LO frequency",
            "availability": "time_aligned_raw_Data.Toltec.LoFreq",
            "result": "tested here for min/max/unique state per observation",
            "classification": "recorded_setup_and_state",
        },
        {
            "telemetry": "drive/sense attenuation",
            "availability": "observation_header_only",
            "result": "inventory records constancy; no event-time samples",
            "classification": "setup_only",
        },
        {
            "telemetry": "ADC waveform",
            "availability": "begin/end 4096-sample snapblocks only",
            "result": "not aligned to interior events",
            "classification": "endpoint_only",
        },
        {
            "telemetry": "telescope trajectory and scan boundaries",
            "availability": "time_aligned",
            "result": (
                "previous causal analysis found no convincing motion trigger"
            ),
            "classification": "trigger_negative_evidence",
        },
        {
            "telemetry": "LNA-bias voltage/current",
            "availability": "unavailable",
            "result": "cannot test shared trigger",
            "classification": "decisive_missing_telemetry",
        },
        {
            "telemetry": "per-network electronics temperature",
            "availability": "unavailable",
            "result": "cannot test progressive warm-electronics susceptibility",
            "classification": "decisive_missing_telemetry",
        },
        {
            "telemetry": "ROACH/PFB status and control registers",
            "availability": "unavailable",
            "result": "cannot test event-time digital state changes",
            "classification": "decisive_missing_telemetry",
        },
        {
            "telemetry": "10 MHz/PPS lock state",
            "availability": "unavailable",
            "result": (
                "event-time modulus tests found no robust boundary locking, "
                "but lock excursions cannot be tested"
            ),
            "classification": "decisive_missing_telemetry",
        },
        {
            "telemetry": "packet/timing counters",
            "availability": "raw timestamp fields without authoritative map",
            "result": "no documented counter semantics for localization",
            "classification": "uninterpretable_present_field",
        },
        {
            "telemetry": "mid-observation TolTEC commands",
            "availability": "not ordinary observing behavior",
            "result": (
                "LMTMC commands telescope only after setup; no command trigger "
                "without explicit operator/script evidence"
            ),
            "classification": "operational_negative_evidence",
        },
    ]


def _inventory(
    *,
    data_root: Path,
    apt_root: Path,
    networks: list[int],
    dac_fft_size: int,
    dac_sample_rate_hz: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_files: dict[tuple[int, int], Path] = {}
    for path in sorted(data_root.glob("toltec*_000_0002_*.nc")):
        match = RAW_PATTERN.fullmatch(path.name)
        if not match:
            continue
        network = int(match.group("network"))
        obsnum = int(match.group("obsnum"))
        if network not in networks:
            continue
        key = (obsnum, network)
        if key in raw_files:
            raise ValueError(f"duplicate raw observation/network file for {key}")
        raw_files[key] = path
    if not raw_files:
        raise FileNotFoundError(f"no raw files found under {data_root}")

    tone_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for (obsnum, network), raw_path in sorted(raw_files.items()):
        apt_path_candidate = apt_root / f"apt_{obsnum}_matched.ecsv"
        apt_path = apt_path_candidate if apt_path_candidate.is_file() else None
        apt_map = _apt_tone_map(apt_path, network=network)
        tune_paths = sorted(
            data_root.glob(
                f"toltec{network}_{obsnum:06d}_000_0001_"
                "*_tune_processed.nc"
            )
        )
        tune_path = tune_paths[0] if len(tune_paths) == 1 else None
        with netCDF4.Dataset(raw_path) as raw:
            header_network = int(_scalar(raw, "Header.Toltec.RoachIndex"))
            header_obsnum = int(_scalar(raw, "Header.Toltec.ObsNum"))
            if header_network != network or header_obsnum != obsnum:
                raise ValueError(f"{raw_path.name}: filename/header mismatch")
            lo_hz = float(_scalar(raw, "Header.Toltec.LoCenterFreq"))
            tone_offset = np.asarray(
                raw.variables["Header.Toltec.ToneFreq"][0, :],
                dtype=float,
            )
            tone_amplitude = np.asarray(
                raw.variables["Header.Toltec.ToneAmp"][0, :],
                dtype=float,
            )
            tone_phase = np.asarray(
                raw.variables["Header.Toltec.TonePhase"][0, :],
                dtype=float,
            )
            tone_mask = np.asarray(
                raw.variables["Header.Toltec.ToneMask"][0, :],
                dtype=float,
            )
            lo_series = np.asarray(
                raw.variables["Data.Toltec.LoFreq"][:],
                dtype=float,
            )
            sample_type = np.asarray(
                raw.variables["Data.Toltec.SampleType"][:],
                dtype=int,
            )
            roach_mac = _optional_chars(
                raw, "Header.Toltec.RoachKatcpMacAddr"
            )
            synth_serial = _optional_chars(
                raw, "Header.Toltec.SynthSerialNum"
            )
            atten_serial = _optional_chars(
                raw, "Header.Toltec.AttenSerialNum"
            )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "subobsnum": int(_scalar(raw, "Header.Toltec.SubObsNum")),
                "raw_scan_num": int(_scalar(raw, "Header.Toltec.ScanNum")),
                "network": network,
                "array": _array_name(network),
                "rack": _rack(network),
                "targ_sweep_obsnum": int(
                    _scalar(raw, "Header.Toltec.TargSweepObsNum")
                ),
                "targ_sweep_subobsnum": int(
                    _scalar(raw, "Header.Toltec.TargSweepSubObsNum")
                ),
                "targ_sweep_scan_num": int(
                    _scalar(raw, "Header.Toltec.TargSweepScanNum")
                ),
                "obs_start_unix_sec": int(
                    _scalar(raw, "Header.Toltec.ObsStartTime")
                ),
                "obs_end_unix_sec": int(
                    _scalar(raw, "Header.Toltec.ObsEndTime")
                ),
                "fpga_frequency_hz": float(
                    _scalar(raw, "Header.Toltec.FpgaFreq")
                ),
                "accumulation_length": int(
                    _scalar(raw, "Header.Toltec.AccumLen")
                ),
                "sample_frequency_hz": float(
                    _scalar(raw, "Header.Toltec.SampleFreq")
                ),
                "lo_center_frequency_hz": lo_hz,
                "lo_data_min_hz": float(np.nanmin(lo_series)),
                "lo_data_max_hz": float(np.nanmax(lo_series)),
                "lo_data_unique_count": int(np.unique(lo_series).size),
                "sample_type_unique_values": ",".join(
                    str(value) for value in np.unique(sample_type)
                ),
                "sense_attenuation_db": float(
                    _scalar(raw, "Header.Toltec.SenseAtten")
                ),
                "drive_attenuation_db": float(
                    _scalar(raw, "Header.Toltec.DriveAtten")
                ),
                "num_tones_header": int(
                    _scalar(raw, "Header.Toltec.NumKidsActual")
                ),
                "compile_time_raw": int(
                    _scalar(raw, "Header.Toltec.CompileTime")
                ),
                "selected_mask_raw": int(
                    _scalar(raw, "Header.Toltec.SelectedMask")
                ),
                "roach_katcp_mac": roach_mac,
                "synth_serial": synth_serial,
                "attenuator_serial": atten_serial,
                "raw_path": str(raw_path),
                "apt_path": str(apt_path) if apt_path else None,
                "processed_tune_path": (
                    str(tune_path) if tune_path is not None else None
                ),
            }
        metadata_rows.append(metadata)
        input_rows.append(
            {
                "obsnum": obsnum,
                "network": network,
                "raw": _file_identity(raw_path),
                "apt": _file_identity(apt_path) if apt_path else None,
                "processed_tune": (
                    _file_identity(tune_path) if tune_path else None
                ),
            }
        )
        bin_width_hz = float(dac_sample_rate_hz) / int(dac_fft_size)
        for tone_slot, offset_hz in enumerate(tone_offset):
            apt_values = apt_map.get(tone_slot, {})
            signed_bin = int(round(float(offset_hz) / bin_width_hz))
            record = {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "network": network,
                "array": _array_name(network),
                "rack": _rack(network),
                "tone_slot_zero_based": int(tone_slot),
                "lo_center_frequency_hz": lo_hz,
                "tone_offset_frequency_hz": float(offset_hz),
                "probe_frequency_hz": float(lo_hz + offset_hz),
                "sideband_from_offset": (
                    "negative"
                    if offset_hz < 0.0
                    else "positive"
                    if offset_hz > 0.0
                    else "zero"
                ),
                "raw_tone_amplitude_normalized": float(
                    tone_amplitude[tone_slot]
                ),
                "raw_tone_phase_rad": float(tone_phase[tone_slot]),
                "raw_tone_mask": float(tone_mask[tone_slot]),
                "provisional_dac_fft_bin_signed": signed_bin,
                "provisional_dac_fft_bin_wrapped": int(
                    signed_bin % int(dac_fft_size)
                ),
                "provisional_dac_fft_bin_residual_hz": float(
                    offset_hz - signed_bin * bin_width_hz
                ),
                "pfb_bin": None,
                "pfb_position_within_bin": None,
                "dac_lane": None,
                "adc_lane": None,
                "channelizer_path": None,
                "apt_row_available": bool(apt_values),
                **apt_values,
                "raw_path": str(raw_path),
                "apt_path": str(apt_path) if apt_path else None,
                "processed_tune_path": (
                    str(tune_path) if tune_path is not None else None
                ),
                "roach_katcp_mac": metadata["roach_katcp_mac"],
                "synth_serial": metadata["synth_serial"],
                "attenuator_serial": metadata["attenuator_serial"],
                "compile_time_raw": metadata["compile_time_raw"],
                "selected_mask_raw": metadata["selected_mask_raw"],
            }
            tone_rows.append(record)
    return tone_rows, metadata_rows, input_rows


def _mode_for_observation(
    rows: pd.DataFrame,
    *,
    coordinate: str = "uid",
    n_modes: int = 2,
) -> CoordinateMode:
    event_count = int(rows["event_cluster_id"].nunique())
    coordinate_count = rows.groupby(coordinate)["event_cluster_id"].nunique()
    complete = coordinate_count[coordinate_count == event_count].index
    selected = rows[rows[coordinate].isin(complete)]
    matrix = (
        selected.pivot(
            index="event_cluster_id",
            columns=coordinate,
            values="phase_change_mrad",
        )[complete]
        .sort_index()
        .to_numpy()
        * 1.0e-3
    )
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("too few events or coordinates for an empirical mode")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("complete event-coordinate matrix is non-finite")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    loading = vh[: int(n_modes), :]
    median = np.median(matrix, axis=0)
    if np.dot(loading[0], median) < 0.0:
        loading[0] *= -1.0
    return CoordinateMode(
        coordinate=np.asarray(complete),
        loading=loading,
        singular_values=singular,
    )


def _fit_loading(
    phase_rad: np.ndarray,
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    phase_rad = np.asarray(phase_rad, dtype=float)
    design = np.asarray(design, dtype=float)
    finite = np.isfinite(phase_rad) & np.all(np.isfinite(design), axis=1)
    phase_rad = phase_rad[finite]
    design = design[finite]
    if len(phase_rad) <= design.shape[1]:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            math.nan,
            math.nan,
        )
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        phase_rad,
        rcond=None,
    )
    prediction = design @ coefficients
    denominator = float(np.sum(phase_rad**2))
    r2 = (
        1.0 - float(np.sum((phase_rad - prediction) ** 2)) / denominator
        if denominator > 0.0
        else math.nan
    )
    return (
        coefficients,
        prediction,
        r2,
        _safe_spearman(prediction, phase_rad),
    )


def _exact_mode_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    coordinate: str,
    rank: int,
    model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode = _mode_for_observation(
        train,
        coordinate=coordinate,
        n_modes=rank,
    )
    coordinate_index = {
        value: index for index, value in enumerate(mode.coordinate)
    }
    event_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for event_id, event in test.groupby("event_cluster_id"):
        selected = event[event[coordinate].isin(coordinate_index)].copy()
        indices = np.asarray(
            [coordinate_index[value] for value in selected[coordinate]],
            dtype=int,
        )
        y = selected["phase_change_mrad"].to_numpy() * 1.0e-3
        design = mode.loading[:rank, indices].T
        coefficients, prediction, r2, rho = _fit_loading(y, design)
        event_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(event["network"].iloc[0]),
                "held_out_obsnum": int(event["obsnum"].iloc[0]),
                "event_cluster_id": event_id,
                "model": model_name,
                "event_coefficient_count": int(rank),
                "test_tone_count": int(len(selected)),
                "zero_baseline_r2": _finite_or_none(r2),
                "predicted_measured_spearman": _finite_or_none(rho),
            }
        )
        if model_name in {
            "empirical_uid_rank1",
            "common_phase_plus_delay",
        }:
            for row, measured, predicted in zip(
                selected.to_dict("records"),
                y,
                prediction,
                strict=True,
            ):
                residual_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "network": int(row["network"]),
                        "held_out_obsnum": int(row["obsnum"]),
                        "event_cluster_id": event_id,
                        "model": model_name,
                        "uid": int(row["uid"]),
                        "tone_slot_zero_based": int(
                            row["tone_slot_zero_based"]
                        ),
                        "tone_offset_frequency_hz": float(
                            row["tone_offset_frequency_hz"]
                        ),
                        "probe_frequency_hz": float(
                            row["probe_frequency_hz"]
                        ),
                        "measured_phase_rad": float(measured),
                        "predicted_phase_rad": float(predicted),
                        "residual_phase_rad": float(measured - predicted),
                    }
                )
    return event_rows, residual_rows


def _paired_uid_slot_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    rank: int,
) -> list[dict[str, Any]]:
    """Score UID and list-slot modes on exactly the same held-out tones."""
    uid_mode = _mode_for_observation(
        train,
        coordinate="uid",
        n_modes=rank,
    )
    slot_mode = _mode_for_observation(
        train,
        coordinate="tone_slot_zero_based",
        n_modes=rank,
    )
    uid_index = {
        int(value): index for index, value in enumerate(uid_mode.coordinate)
    }
    slot_index = {
        int(value): index for index, value in enumerate(slot_mode.coordinate)
    }
    rows: list[dict[str, Any]] = []
    for event_id, event in test.groupby("event_cluster_id"):
        selected = event[
            event["uid"].isin(uid_index)
            & event["tone_slot_zero_based"].isin(slot_index)
        ].copy()
        y = selected["phase_change_mrad"].to_numpy() * 1.0e-3
        designs = {
            f"empirical_uid_rank{rank}_shared_tones": (
                uid_mode.loading[
                    :rank,
                    np.asarray(
                        [uid_index[int(value)] for value in selected["uid"]],
                        dtype=int,
                    ),
                ].T
            ),
            f"tone_slot_rank{rank}_shared_tones": (
                slot_mode.loading[
                    :rank,
                    np.asarray(
                        [
                            slot_index[int(value)]
                            for value in selected["tone_slot_zero_based"]
                        ],
                        dtype=int,
                    ),
                ].T
            ),
        }
        for model_name, design in designs.items():
            _, _, r2, rho = _fit_loading(y, design)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "network": int(event["network"].iloc[0]),
                    "held_out_obsnum": int(event["obsnum"].iloc[0]),
                    "event_cluster_id": event_id,
                    "model": model_name,
                    "event_coefficient_count": int(rank),
                    "test_tone_count": int(len(selected)),
                    "zero_baseline_r2": _finite_or_none(r2),
                    "predicted_measured_spearman": _finite_or_none(rho),
                }
            )
    return rows


def _direct_model_predictions(
    test: pd.DataFrame,
    *,
    delay: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = "common_phase_plus_delay" if delay else "common_phase"
    event_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for event_id, event in test.groupby("event_cluster_id"):
        y = event["phase_change_mrad"].to_numpy() * 1.0e-3
        if delay:
            offset = (
                event["tone_offset_frequency_hz"].to_numpy() / 1.0e8
            )
            design = np.column_stack([np.ones(len(event)), offset])
        else:
            design = np.ones((len(event), 1))
        coefficients, prediction, r2, rho = _fit_loading(y, design)
        event_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(event["network"].iloc[0]),
                "held_out_obsnum": int(event["obsnum"].iloc[0]),
                "event_cluster_id": event_id,
                "model": model,
                "event_coefficient_count": int(design.shape[1]),
                "test_tone_count": int(len(event)),
                "zero_baseline_r2": _finite_or_none(r2),
                "predicted_measured_spearman": _finite_or_none(rho),
            }
        )
        if delay:
            for row, measured, predicted in zip(
                event.to_dict("records"),
                y,
                prediction,
                strict=True,
            ):
                residual_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "network": int(row["network"]),
                        "held_out_obsnum": int(row["obsnum"]),
                        "event_cluster_id": event_id,
                        "model": model,
                        "uid": int(row["uid"]),
                        "tone_slot_zero_based": int(
                            row["tone_slot_zero_based"]
                        ),
                        "tone_offset_frequency_hz": float(
                            row["tone_offset_frequency_hz"]
                        ),
                        "probe_frequency_hz": float(
                            row["probe_frequency_hz"]
                        ),
                        "measured_phase_rad": float(measured),
                        "predicted_phase_rad": float(predicted),
                        "residual_phase_rad": float(measured - predicted),
                    }
                )
    return event_rows, residual_rows


def _binned_mode_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    field: str,
    n_bins: int,
    model_name: str,
) -> list[dict[str, Any]]:
    values = train[field].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError(f"{field}: too few distinct bin edges")
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    train_copy = train.copy()
    test_copy = test.copy()
    train_copy["_coordinate_bin"] = np.clip(
        np.digitize(train_copy[field], edges) - 1,
        0,
        len(edges) - 2,
    )
    test_copy["_coordinate_bin"] = np.clip(
        np.digitize(test_copy[field], edges) - 1,
        0,
        len(edges) - 2,
    )
    aggregate = (
        train_copy.groupby(
            ["event_cluster_id", "_coordinate_bin"],
            as_index=False,
        )["phase_change_mrad"]
        .mean()
    )
    mode = _mode_for_observation(
        aggregate,
        coordinate="_coordinate_bin",
        n_modes=1,
    )
    coordinate_index = {
        int(value): index for index, value in enumerate(mode.coordinate)
    }
    rows: list[dict[str, Any]] = []
    for event_id, event in test_copy.groupby("event_cluster_id"):
        selected = event[
            event["_coordinate_bin"].isin(coordinate_index)
        ]
        indices = np.asarray(
            [
                coordinate_index[int(value)]
                for value in selected["_coordinate_bin"]
            ],
            dtype=int,
        )
        y = selected["phase_change_mrad"].to_numpy() * 1.0e-3
        design = mode.loading[0, indices, None]
        _, _, r2, rho = _fit_loading(y, design)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(event["network"].iloc[0]),
                "held_out_obsnum": int(event["obsnum"].iloc[0]),
                "event_cluster_id": event_id,
                "model": model_name,
                "event_coefficient_count": 1,
                "test_tone_count": int(len(selected)),
                "zero_baseline_r2": _finite_or_none(r2),
                "predicted_measured_spearman": _finite_or_none(rho),
            }
        )
    return rows


def _sideband_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict[str, Any]]:
    train_copy = train.copy()
    test_copy = test.copy()
    train_copy["_sideband"] = np.sign(
        train_copy["tone_offset_frequency_hz"]
    ).astype(int)
    test_copy["_sideband"] = np.sign(
        test_copy["tone_offset_frequency_hz"]
    ).astype(int)
    aggregate = (
        train_copy.groupby(
            ["event_cluster_id", "_sideband"],
            as_index=False,
        )["phase_change_mrad"]
        .mean()
    )
    mode = _mode_for_observation(
        aggregate,
        coordinate="_sideband",
        n_modes=1,
    )
    coordinate_index = {
        int(value): index for index, value in enumerate(mode.coordinate)
    }
    rows: list[dict[str, Any]] = []
    for event_id, event in test_copy.groupby("event_cluster_id"):
        selected = event[event["_sideband"].isin(coordinate_index)]
        indices = np.asarray(
            [coordinate_index[int(value)] for value in selected["_sideband"]],
            dtype=int,
        )
        y = selected["phase_change_mrad"].to_numpy() * 1.0e-3
        design = mode.loading[0, indices, None]
        _, _, r2, rho = _fit_loading(y, design)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(event["network"].iloc[0]),
                "held_out_obsnum": int(event["obsnum"].iloc[0]),
                "event_cluster_id": event_id,
                "model": "sideband_rank1",
                "event_coefficient_count": 1,
                "test_tone_count": int(len(selected)),
                "zero_baseline_r2": _finite_or_none(r2),
                "predicted_measured_spearman": _finite_or_none(rho),
            }
        )
    return rows


def _cross_validated_models(
    event_tones: pd.DataFrame,
    *,
    networks: list[int],
    event_obsnums: list[int],
    coordinate_bins: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    event_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for network in networks:
        network_rows = event_tones[event_tones["network"] == network]
        for held_out in event_obsnums:
            train = network_rows[network_rows["obsnum"] != held_out]
            test = network_rows[network_rows["obsnum"] == held_out]
            if train.empty or test.empty:
                continue
            for delay in (False, True):
                rows, residual = _direct_model_predictions(
                    test,
                    delay=delay,
                )
                event_rows.extend(rows)
                residual_rows.extend(residual)
            for coordinate, base_name in (
                ("uid", "empirical_uid"),
                ("tone_slot_zero_based", "tone_slot"),
            ):
                for rank in (1, 2):
                    rows, residual = _exact_mode_predictions(
                        train,
                        test,
                        coordinate=coordinate,
                        rank=rank,
                        model_name=f"{base_name}_rank{rank}",
                    )
                    event_rows.extend(rows)
                    residual_rows.extend(residual)
            for rank in (1, 2):
                event_rows.extend(
                    _paired_uid_slot_predictions(
                        train,
                        test,
                        rank=rank,
                    )
                )
            event_rows.extend(
                _binned_mode_predictions(
                    train,
                    test,
                    field="tone_offset_frequency_hz",
                    n_bins=coordinate_bins,
                    model_name=f"tone_offset_{coordinate_bins}bin_rank1",
                )
            )
            event_rows.extend(
                _binned_mode_predictions(
                    train,
                    test,
                    field="probe_frequency_hz",
                    n_bins=coordinate_bins,
                    model_name=f"absolute_rf_{coordinate_bins}bin_rank1",
                )
            )
            event_rows.extend(_sideband_predictions(train, test))
    frame = pd.DataFrame(event_rows)
    summary_rows: list[dict[str, Any]] = []
    for (network, model), group in frame.groupby(["network", "model"]):
        summary_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(network),
                "model": model,
                "held_out_event_count": int(len(group)),
                "held_out_observation_count": int(
                    group["held_out_obsnum"].nunique()
                ),
                "median_zero_baseline_r2": float(
                    group["zero_baseline_r2"].median()
                ),
                "mean_zero_baseline_r2": float(
                    group["zero_baseline_r2"].mean()
                ),
                "median_predicted_measured_spearman": float(
                    _finite_median(
                        group["predicted_measured_spearman"].to_numpy()
                    )
                ),
                "median_test_tone_count": float(
                    group["test_tone_count"].median()
                ),
            }
        )
    return event_rows, summary_rows, residual_rows


def _observation_modes(
    event_tones: pd.DataFrame,
    *,
    networks: list[int],
    event_obsnums: list[int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summaries: list[dict[str, Any]] = []
    tone_rows: list[dict[str, Any]] = []
    modes: dict[tuple[int, int], pd.DataFrame] = {}
    for network in networks:
        for obsnum in event_obsnums:
            rows = event_tones[
                (event_tones["network"] == network)
                & (event_tones["obsnum"] == obsnum)
            ]
            if rows.empty:
                continue
            mode = _mode_for_observation(rows, coordinate="uid", n_modes=2)
            singular = mode.singular_values
            event_count = int(rows["event_cluster_id"].nunique())
            identity = (
                rows[rows["uid"].isin(mode.coordinate)]
                .groupby("uid", as_index=False)
                .agg(
                    tone_slot_zero_based=(
                        "tone_slot_zero_based",
                        "median",
                    ),
                    tone_offset_frequency_hz=(
                        "tone_offset_frequency_hz",
                        "median",
                    ),
                    probe_frequency_hz=("probe_frequency_hz", "median"),
                )
                .set_index("uid")
                .loc[mode.coordinate]
                .reset_index()
            )
            loading = mode.loading[0].copy()
            loading /= math.sqrt(float(np.mean(loading**2)))
            identity["loading"] = loading
            modes[(network, obsnum)] = identity
            summaries.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "network": network,
                    "obsnum": obsnum,
                    "event_count": event_count,
                    "complete_uid_count": int(len(mode.coordinate)),
                    "phase_rank1_energy_fraction": float(
                        singular[0] ** 2 / np.sum(singular**2)
                    ),
                    "phase_rank2_cumulative_energy_fraction": float(
                        np.sum(singular[:2] ** 2) / np.sum(singular**2)
                    ),
                }
            )
            for row in identity.to_dict("records"):
                tone_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "network": network,
                        "obsnum": obsnum,
                        "uid": int(row["uid"]),
                        "tone_slot_zero_based": int(
                            round(row["tone_slot_zero_based"])
                        ),
                        "tone_offset_frequency_hz": float(
                            row["tone_offset_frequency_hz"]
                        ),
                        "probe_frequency_hz": float(
                            row["probe_frequency_hz"]
                        ),
                        "phase_rank1_loading_rms_normalized": float(
                            row["loading"]
                        ),
                    }
                )

    pair_rows: list[dict[str, Any]] = []
    for network in networks:
        available = [
            obsnum
            for obsnum in event_obsnums
            if (network, obsnum) in modes
        ]
        for first_index, first_obs in enumerate(available):
            for second_obs in available[first_index + 1 :]:
                first = modes[(network, first_obs)]
                second = modes[(network, second_obs)]
                for coordinate, mapping in (
                    ("uid", "exact_uid"),
                    ("tone_slot_zero_based", "exact_tone_slot"),
                ):
                    joined = first.merge(
                        second,
                        on=coordinate,
                        suffixes=("_first", "_second"),
                    )
                    if mapping == "exact_uid":
                        subset = joined[
                            joined["tone_slot_zero_based_first"]
                            != joined["tone_slot_zero_based_second"]
                        ]
                        changed_cosine = _cosine(
                            subset["loading_first"],
                            subset["loading_second"],
                        )
                        changed_rho = _safe_spearman(
                            subset["loading_first"],
                            subset["loading_second"],
                        )
                        changed_count = int(len(subset))
                    else:
                        changed_cosine = math.nan
                        changed_rho = math.nan
                        changed_count = 0
                    pair_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "network": network,
                            "first_obsnum": first_obs,
                            "second_obsnum": second_obs,
                            "mapping": mapping,
                            "matched_tone_count": int(len(joined)),
                            "loading_cosine": _cosine(
                                joined["loading_first"],
                                joined["loading_second"],
                            ),
                            "loading_spearman": _safe_spearman(
                                joined["loading_first"],
                                joined["loading_second"],
                            ),
                            "slot_changed_uid_count": changed_count,
                            "slot_changed_uid_loading_cosine": (
                                _finite_or_none(changed_cosine)
                            ),
                            "slot_changed_uid_loading_spearman": (
                                _finite_or_none(changed_rho)
                            ),
                        }
                    )
                ordered = first.sort_values("tone_offset_frequency_hz")
                in_range = second[
                    (
                        second["tone_offset_frequency_hz"]
                        >= ordered["tone_offset_frequency_hz"].min()
                    )
                    & (
                        second["tone_offset_frequency_hz"]
                        <= ordered["tone_offset_frequency_hz"].max()
                    )
                ]
                prediction = np.interp(
                    in_range["tone_offset_frequency_hz"],
                    ordered["tone_offset_frequency_hz"],
                    ordered["loading"],
                )
                pair_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "network": network,
                        "first_obsnum": first_obs,
                        "second_obsnum": second_obs,
                        "mapping": "interpolated_tone_offset",
                        "matched_tone_count": int(len(in_range)),
                        "loading_cosine": _cosine(
                            prediction,
                            in_range["loading"].to_numpy(),
                        ),
                        "loading_spearman": _safe_spearman(
                            prediction,
                            in_range["loading"].to_numpy(),
                        ),
                        "slot_changed_uid_count": None,
                        "slot_changed_uid_loading_cosine": None,
                        "slot_changed_uid_loading_spearman": None,
                    }
                )
    return summaries, tone_rows, pair_rows


def _identifiability_rows(
    event_tones: pd.DataFrame,
    *,
    networks: list[int],
    event_obsnums: list[int],
    coordinate_bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for network in networks:
        selected = event_tones[event_tones["network"] == network]
        identity = selected.drop_duplicates(["obsnum", "uid"])
        all_offsets = identity["tone_offset_frequency_hz"].to_numpy()
        bin_edges = np.unique(
            np.quantile(
                all_offsets,
                np.linspace(0.0, 1.0, coordinate_bins + 1),
            )
        )
        bin_edges[0] = np.nextafter(bin_edges[0], -np.inf)
        bin_edges[-1] = np.nextafter(bin_edges[-1], np.inf)
        for first_index, first_obs in enumerate(event_obsnums):
            for second_obs in event_obsnums[first_index + 1 :]:
                first = identity[identity["obsnum"] == first_obs]
                second = identity[identity["obsnum"] == second_obs]
                joined = first.merge(
                    second,
                    on="uid",
                    suffixes=("_first", "_second"),
                )
                first_bin = np.digitize(
                    joined["tone_offset_frequency_hz_first"],
                    bin_edges,
                )
                second_bin = np.digitize(
                    joined["tone_offset_frequency_hz_second"],
                    bin_edges,
                )
                by_slot = first.merge(
                    second,
                    on="tone_slot_zero_based",
                    suffixes=("_first", "_second"),
                )
                offset_delta = (
                    joined["tone_offset_frequency_hz_second"]
                    - joined["tone_offset_frequency_hz_first"]
                )
                rf_delta = (
                    joined["probe_frequency_hz_second"]
                    - joined["probe_frequency_hz_first"]
                )
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "network": network,
                        "first_obsnum": first_obs,
                        "second_obsnum": second_obs,
                        "common_uid_count": int(len(joined)),
                        "lo_first_hz": float(
                            joined["lo_center_frequency_hz_first"].iloc[0]
                        ),
                        "lo_second_hz": float(
                            joined["lo_center_frequency_hz_second"].iloc[0]
                        ),
                        "lo_shift_hz": float(
                            joined["lo_center_frequency_hz_second"].iloc[0]
                            - joined["lo_center_frequency_hz_first"].iloc[0]
                        ),
                        "uid_tone_slot_changed_fraction": float(
                            np.mean(
                                joined["tone_slot_zero_based_first"]
                                != joined["tone_slot_zero_based_second"]
                            )
                        ),
                        "median_abs_tone_offset_shift_hz": float(
                            np.median(np.abs(offset_delta))
                        ),
                        "maximum_abs_tone_offset_shift_hz": float(
                            np.max(np.abs(offset_delta))
                        ),
                        "median_abs_probe_rf_shift_hz": float(
                            np.median(np.abs(rf_delta))
                        ),
                        "maximum_abs_probe_rf_shift_hz": float(
                            np.max(np.abs(rf_delta))
                        ),
                        "uid_offset_bin_changed_fraction": float(
                            np.mean(first_bin != second_bin)
                        ),
                        "uid_sideband_changed_fraction": float(
                            np.mean(
                                np.sign(
                                    joined[
                                        "tone_offset_frequency_hz_first"
                                    ]
                                )
                                != np.sign(
                                    joined[
                                        "tone_offset_frequency_hz_second"
                                    ]
                                )
                            )
                        ),
                        "common_tone_slot_count": int(len(by_slot)),
                        "tone_slot_different_uid_fraction": float(
                            np.mean(
                                by_slot["uid_first"] != by_slot["uid_second"]
                            )
                        ),
                    }
                )
    return rows


def _low_rank_rows(
    event_tones: pd.DataFrame,
    *,
    networks: list[int],
    sample_frequency_hz: float,
    pre_window_sec: float,
    post_window_sec: float,
    sigma_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for network in networks:
        rows = event_tones[event_tones["network"] == network]
        mode = _mode_for_observation(rows, coordinate="uid", n_modes=2)
        event_count = int(rows["event_cluster_id"].nunique())
        selected = rows[rows["uid"].isin(mode.coordinate)]
        phase = (
            selected.pivot(
                index="event_cluster_id",
                columns="uid",
                values="phase_change_mrad",
            )[mode.coordinate]
            .sort_index()
            .to_numpy()
            * 1.0e-3
        )
        threshold = (
            selected.pivot(
                index="event_cluster_id",
                columns="uid",
                values="phase_threshold_mrad",
            )[mode.coordinate]
            .sort_index()
            .to_numpy()
            * 1.0e-3
        )
        singular = mode.singular_values
        total_energy = float(np.sum(phase**2))
        rank1 = float(singular[0] ** 2 / np.sum(singular**2))
        rank2 = float(np.sum(singular[:2] ** 2) / np.sum(singular**2))
        pre_count = float(sample_frequency_hz) * float(pre_window_sec)
        post_count = float(sample_frequency_hz) * float(post_window_sec)
        noise_upper = float(
            np.sum(
                (threshold / float(sigma_threshold)) ** 2
                * (1.0 / pre_count + 1.0 / post_count)
            )
            / total_energy
        )
        summary_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": network,
                "event_count": event_count,
                "complete_uid_count": int(len(mode.coordinate)),
                "phase_rank1_energy_fraction": rank1,
                "phase_rank2_cumulative_energy_fraction": rank2,
                "rank1_residual_energy_fraction": 1.0 - rank1,
                "rank2_residual_energy_fraction": 1.0 - rank2,
                "second_mode_incremental_energy_fraction": rank2 - rank1,
                "second_mode_fraction_of_rank1_residual": (
                    (rank2 - rank1) / (1.0 - rank1)
                ),
                "measurement_noise_energy_upper_bound_fraction": noise_upper,
                "rank1_residual_minus_noise_upper_bound_fraction": max(
                    0.0, 1.0 - rank1 - noise_upper
                ),
            }
        )
        loading = mode.loading[0].copy()
        loading /= np.linalg.norm(loading)
        # Avoid platform BLAS warnings observed for a very short, wide
        # matrix-vector product.  This is the same dot product without
        # dispatching through the accelerated matmul implementation.
        event_amplitude = np.sum(phase * loading[None, :], axis=1)
        amplitude_median = float(np.median(np.abs(event_amplitude)))
        groups = {
            "positive_amplitude": event_amplitude >= 0.0,
            "negative_amplitude": event_amplitude < 0.0,
            "low_absolute_amplitude": (
                np.abs(event_amplitude) <= amplitude_median
            ),
            "high_absolute_amplitude": (
                np.abs(event_amplitude) > amplitude_median
            ),
        }
        for label, keep in groups.items():
            group = phase[keep, :]
            if group.shape[0] < 2:
                continue
            _, split_singular, split_vh = np.linalg.svd(
                group,
                full_matrices=False,
            )
            split_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "network": network,
                    "split": label,
                    "event_count": int(group.shape[0]),
                    "rank1_energy_fraction": float(
                        split_singular[0] ** 2
                        / np.sum(split_singular**2)
                    ),
                    "loading_cosine_to_all_events": _cosine(
                        split_vh[0],
                        loading,
                    ),
                }
            )
    return summary_rows, split_rows


def _select_separated_events(
    rows: pd.DataFrame,
    *,
    minimum_separation_sec: float,
) -> pd.DataFrame:
    rows = rows.sort_values("event_absolute_sec")
    keep: list[int] = []
    latest = -math.inf
    for index, row in rows.iterrows():
        event_time = float(row["event_absolute_sec"])
        if event_time - latest >= float(minimum_separation_sec):
            keep.append(index)
            latest = event_time
    return rows.loc[keep].copy()


def _extract_raw_vectors(
    *,
    raw_path: Path,
    apt_path: Path,
    network: int,
    epochs_unix_sec: Iterable[float],
    vector_ids: Iterable[str],
    population: str,
    sigma_threshold: float,
    min_phase_mrad: float,
    pre_window_sec: float,
    guard_window_sec: float,
    post_window_sec: float,
) -> list[dict[str, Any]]:
    apt = Table.read(apt_path, format="ascii.ecsv")
    with netCDF4.Dataset(raw_path) as raw:
        time_unix = np.asarray(
            raw.variables["Data.Toltec.RecvTime"][:],
            dtype=float,
        )
        i_data = np.ma.filled(
            np.ma.asarray(
                raw.variables["Data.Toltec.Is"][:],
                dtype=float,
            ),
            np.nan,
        )
        q_data = np.ma.filled(
            np.ma.asarray(
                raw.variables["Data.Toltec.Qs"][:],
                dtype=float,
            ),
            np.nan,
        )
        tone_offset = np.asarray(
            raw.variables["Header.Toltec.ToneFreq"][0, :],
            dtype=float,
        )
        lo_hz = float(_scalar(raw, "Header.Toltec.LoCenterFreq"))
        obsnum = int(_scalar(raw, "Header.Toltec.ObsNum"))
    uid, apt_frequency, usable = iq_tool._apt_arrays(
        apt,
        network=network,
        n_tones=i_data.shape[1],
        raw_tone_frequency_hz=tone_offset,
    )
    complex_iq = i_data + 1j * q_data
    phase = np.unwrap(np.angle(complex_iq), axis=0)
    phase_difference = np.diff(phase, axis=0)
    finite_difference_count = np.sum(
        np.isfinite(phase_difference),
        axis=0,
    )
    sigma_eligible = finite_difference_count >= 4
    phase_sigma = np.full(phase_difference.shape[1], np.nan)
    phase_sigma[sigma_eligible] = (
        iq_tool._robust_sigma(
            phase_difference[:, sigma_eligible],
            axis=0,
        )
        / math.sqrt(2.0)
    )
    threshold = np.maximum(
        float(sigma_threshold) * phase_sigma,
        float(min_phase_mrad) * 1.0e-3,
    )
    rows: list[dict[str, Any]] = []
    for epoch, vector_id in zip(
        epochs_unix_sec,
        vector_ids,
        strict=True,
    ):
        epoch = float(epoch)
        pre = (
            (
                time_unix
                >= epoch - float(guard_window_sec) - float(pre_window_sec)
            )
            & (time_unix < epoch - float(guard_window_sec))
        )
        post = (
            (time_unix > epoch + float(guard_window_sec))
            & (
                time_unix
                <= epoch
                + float(guard_window_sec)
                + float(post_window_sec)
            )
        )
        if np.count_nonzero(pre) < 4 or np.count_nonzero(post) < 4:
            continue
        pre_values = complex_iq[pre, :]
        post_values = complex_iq[post, :]
        pre_finite = np.isfinite(pre_values)
        post_finite = np.isfinite(post_values)
        z_pre = np.full(pre_values.shape[1], np.nan + 1j * np.nan)
        z_post = np.full(post_values.shape[1], np.nan + 1j * np.nan)
        np.divide(
            np.nansum(pre_values, axis=0),
            np.sum(pre_finite, axis=0),
            out=z_pre,
            where=np.sum(pre_finite, axis=0) > 0,
        )
        np.divide(
            np.nansum(post_values, axis=0),
            np.sum(post_finite, axis=0),
            out=z_post,
            where=np.sum(post_finite, axis=0) > 0,
        )
        fractional = np.full(z_pre.shape, np.nan + 1j * np.nan)
        np.divide(
            z_post,
            z_pre,
            out=fractional,
            where=np.abs(z_pre) > 0.0,
        )
        fractional -= 1.0
        phase_change = np.angle(z_post / z_pre)
        valid = (
            usable
            & (uid >= 0)
            & np.isfinite(phase_change)
            & np.isfinite(fractional.real)
            & np.isfinite(fractional.imag)
            & np.isfinite(threshold)
            & (np.abs(z_pre) > 0.0)
        )
        responsive = valid & (np.abs(phase_change) > threshold)
        for tone in np.flatnonzero(valid):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "population": population,
                    "vector_id": vector_id,
                    "obsnum": obsnum,
                    "network": network,
                    "event_time_unix_sec": epoch,
                    "tone_slot_zero_based": int(tone),
                    "uid": int(uid[tone]),
                    "lo_center_frequency_hz": lo_hz,
                    "tone_offset_frequency_hz": float(tone_offset[tone]),
                    "probe_frequency_hz": float(
                        lo_hz + tone_offset[tone]
                    ),
                    "apt_tone_frequency_hz": float(apt_frequency[tone]),
                    "phase_responsive": bool(responsive[tone]),
                    "phase_threshold_mrad": float(1.0e3 * threshold[tone]),
                    "phase_change_mrad": float(
                        1.0e3 * phase_change[tone]
                    ),
                    "fractional_change_real": float(fractional[tone].real),
                    "fractional_change_imag": float(fractional[tone].imag),
                    "raw_path": str(raw_path),
                    "apt_path": str(apt_path),
                }
            )
    return rows


def _pointing_vectors(
    *,
    data_root: Path,
    apt_root: Path,
    consensus_root: Path,
    networks: list[int],
    event_obsnums: list[int],
    null_obsnums: list[int],
    null_epochs_per_observation: int,
    minimum_event_separation_sec: float,
    sigma_threshold: float,
    min_phase_mrad: float,
    pre_window_sec: float,
    guard_window_sec: float,
    post_window_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for obsnum in event_obsnums:
        consensus_path = (
            consensus_root
            / f"obs{obsnum}"
            / f"o{obsnum}_multinetwork_level_shift_consensus_events.csv"
        )
        consensus = pd.read_csv(consensus_path)
        selected = consensus[
            consensus["classification"].eq("corroborated")
            & (consensus["normal_response_network_count"] >= 3)
        ]
        selected = _select_separated_events(
            selected,
            minimum_separation_sec=minimum_event_separation_sec,
        )
        epochs = selected["event_absolute_sec"].astype(float).tolist()
        vector_ids = [
            f"pointing-{obsnum}-event-{int(value):04d}"
            for value in selected["consensus_event_id"]
        ]
        for vector_id, event_row in zip(
            vector_ids,
            selected.to_dict("records"),
            strict=True,
        ):
            selection_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "population": "pointing_event",
                    "vector_id": vector_id,
                    "obsnum": obsnum,
                    "event_time_unix_sec": float(
                        event_row["event_absolute_sec"]
                    ),
                    "selection_source": str(consensus_path),
                    "selection_classification": event_row["classification"],
                    "normal_response_network_count": int(
                        event_row["normal_response_network_count"]
                    ),
                }
            )
        apt_path = apt_root / f"apt_{obsnum}_matched.ecsv"
        for network in networks:
            raw_path = _find_one(
                data_root,
                f"toltec{network}_{obsnum:06d}_000_0002_*.nc",
            )
            rows.extend(
                _extract_raw_vectors(
                    raw_path=raw_path,
                    apt_path=apt_path,
                    network=network,
                    epochs_unix_sec=epochs,
                    vector_ids=vector_ids,
                    population="pointing_event",
                    sigma_threshold=sigma_threshold,
                    min_phase_mrad=min_phase_mrad,
                    pre_window_sec=pre_window_sec,
                    guard_window_sec=guard_window_sec,
                    post_window_sec=post_window_sec,
                )
            )
    for obsnum in null_obsnums:
        apt_path = apt_root / f"apt_{obsnum}_matched.ecsv"
        reference_path = _find_one(
            data_root,
            f"toltec{networks[0]}_{obsnum:06d}_000_0002_*.nc",
        )
        with netCDF4.Dataset(reference_path) as raw:
            time_unix = np.asarray(
                raw.variables["Data.Toltec.RecvTime"][:],
                dtype=float,
            )
        margin = (
            float(pre_window_sec)
            + float(guard_window_sec)
            + 0.25
        )
        epochs = np.linspace(
            time_unix[0] + margin,
            time_unix[-1] - margin,
            int(null_epochs_per_observation),
        )
        vector_ids = [
            f"pointing-{obsnum}-null-{index:04d}"
            for index in range(len(epochs))
        ]
        for vector_id, epoch in zip(vector_ids, epochs, strict=True):
            selection_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "population": "clean_pointing_fixed_epoch",
                    "vector_id": vector_id,
                    "obsnum": obsnum,
                    "event_time_unix_sec": float(epoch),
                    "selection_source": (
                        "fixed grid in previously classified clean pointing"
                    ),
                    "selection_classification": "null_epoch",
                    "normal_response_network_count": 0,
                }
            )
        for network in networks:
            raw_path = _find_one(
                data_root,
                f"toltec{network}_{obsnum:06d}_000_0002_*.nc",
            )
            rows.extend(
                _extract_raw_vectors(
                    raw_path=raw_path,
                    apt_path=apt_path,
                    network=network,
                    epochs_unix_sec=epochs,
                    vector_ids=vector_ids,
                    population="clean_pointing_fixed_epoch",
                    sigma_threshold=sigma_threshold,
                    min_phase_mrad=min_phase_mrad,
                    pre_window_sec=pre_window_sec,
                    guard_window_sec=guard_window_sec,
                    post_window_sec=post_window_sec,
                )
            )
    return rows, selection_rows


def _template_projection_rows(
    vectors: pd.DataFrame,
    template_tones: pd.DataFrame,
    *,
    networks: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for network in networks:
        template = (
            template_tones[template_tones["network"] == network]
            .set_index("uid")["phase_rank1_loading_rms_normalized"]
        )
        selected = vectors[vectors["network"] == network]
        for vector_id, vector in selected.groupby("vector_id"):
            usable = vector[vector["uid"].isin(template.index)]
            y = usable["phase_change_mrad"].to_numpy() * 1.0e-3
            loading = template.loc[usable["uid"]].to_numpy(dtype=float)
            design = loading[:, None]
            coefficient, _, r2, rho = _fit_loading(y, design)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "population": vector["population"].iloc[0],
                    "vector_id": vector_id,
                    "obsnum": int(vector["obsnum"].iloc[0]),
                    "network": network,
                    "matched_uid_count": int(len(usable)),
                    "template_amplitude_rad_per_rms_loading": (
                        float(coefficient[0])
                        if coefficient.size
                        else None
                    ),
                    "template_zero_baseline_r2": _finite_or_none(r2),
                    "template_phase_cosine": (
                        _finite_or_none(math.sqrt(max(0.0, r2)))
                        if np.isfinite(r2)
                        else None
                    ),
                    "template_predicted_measured_spearman": (
                        _finite_or_none(rho)
                    ),
                    "phase_rms_mrad": float(
                        1.0e3 * math.sqrt(float(np.mean(y**2)))
                    ),
                    "responsive_tone_fraction": float(
                        usable["phase_responsive"].mean()
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    for (network, population, obsnum), group in frame.groupby(
        ["network", "population", "obsnum"]
    ):
        summary.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(network),
                "population": population,
                "obsnum": int(obsnum),
                "vector_count": int(len(group)),
                "median_template_zero_baseline_r2": float(
                    group["template_zero_baseline_r2"].median()
                ),
                "median_template_phase_cosine": float(
                    group["template_phase_cosine"].median()
                ),
                "median_abs_template_amplitude_mrad": float(
                    1.0e3
                    * group[
                        "template_amplitude_rad_per_rms_loading"
                    ].abs().median()
                ),
                "median_phase_rms_mrad": float(
                    group["phase_rms_mrad"].median()
                ),
                "median_responsive_tone_fraction": float(
                    group["responsive_tone_fraction"].median()
                ),
            }
        )
    return rows, summary


def _pointing_population_comparison_rows(
    projection_rows: list[dict[str, Any]],
    *,
    networks: list[int],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(projection_rows)
    rows: list[dict[str, Any]] = []
    for network in networks:
        selected = frame[frame["network"] == network]
        events = selected[
            selected["population"] == "pointing_event"
        ]
        nulls = selected[
            selected["population"] == "clean_pointing_fixed_epoch"
        ]
        event_score = events["template_zero_baseline_r2"].to_numpy(
            dtype=float
        )
        null_score = nulls["template_zero_baseline_r2"].to_numpy(dtype=float)
        pairwise = event_score[:, None] - null_score[None, :]
        auc = float(
            np.mean(pairwise > 0.0) + 0.5 * np.mean(pairwise == 0.0)
        )
        event_observation_medians = events.groupby("obsnum")[
            "template_zero_baseline_r2"
        ].median()
        null_observation_medians = nulls.groupby("obsnum")[
            "template_zero_baseline_r2"
        ].median()
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(network),
                "event_vector_count": int(len(events)),
                "null_vector_count": int(len(nulls)),
                "event_observation_count": int(events["obsnum"].nunique()),
                "null_observation_count": int(nulls["obsnum"].nunique()),
                "median_event_template_zero_baseline_r2": float(
                    np.median(event_score)
                ),
                "median_null_template_zero_baseline_r2": float(
                    np.median(null_score)
                ),
                "minimum_event_observation_median_r2": float(
                    event_observation_medians.min()
                ),
                "maximum_event_observation_median_r2": float(
                    event_observation_medians.max()
                ),
                "minimum_null_observation_median_r2": float(
                    null_observation_medians.min()
                ),
                "maximum_null_observation_median_r2": float(
                    null_observation_medians.max()
                ),
                "event_null_pairwise_auc": auc,
                "event_fraction_above_maximum_null": float(
                    np.mean(event_score > np.max(null_score))
                ),
            }
        )
    return rows


def _pointing_mode_rows(
    vectors: pd.DataFrame,
    template_tones: pd.DataFrame,
    *,
    networks: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    events = vectors[vectors["population"].eq("pointing_event")]
    for (network, obsnum), group in events.groupby(["network", "obsnum"]):
        if int(network) not in networks:
            continue
        renamed = group.rename(columns={"vector_id": "event_cluster_id"})
        mode = _mode_for_observation(
            renamed,
            coordinate="uid",
            n_modes=2,
        )
        template = (
            template_tones[
                template_tones["network"] == int(network)
            ]
            .set_index("uid")["phase_rank1_loading_rms_normalized"]
        )
        common = [
            int(uid) for uid in mode.coordinate if int(uid) in template.index
        ]
        index = {int(uid): idx for idx, uid in enumerate(mode.coordinate)}
        loading = np.asarray(
            [mode.loading[0, index[uid]] for uid in common],
            dtype=float,
        )
        reference = template.loc[common].to_numpy(dtype=float)
        singular = mode.singular_values
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "network": int(network),
                "obsnum": int(obsnum),
                "event_count": int(group["vector_id"].nunique()),
                "complete_uid_count": int(len(mode.coordinate)),
                "science_template_common_uid_count": int(len(common)),
                "pointing_rank1_energy_fraction": float(
                    singular[0] ** 2 / np.sum(singular**2)
                ),
                "pointing_rank2_cumulative_energy_fraction": float(
                    np.sum(singular[:2] ** 2) / np.sum(singular**2)
                ),
                "pointing_mode_science_template_cosine": _cosine(
                    loading,
                    reference,
                ),
                "pointing_mode_science_template_spearman": _safe_spearman(
                    loading,
                    reference,
                ),
            }
        )
    return rows


def _residual_bin_rows(
    residual_rows: list[dict[str, Any]],
    *,
    coordinate_bins: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(residual_rows)
    rows: list[dict[str, Any]] = []
    for (network, model), group in frame.groupby(["network", "model"]):
        event_rms = (
            group.groupby("event_cluster_id")["measured_phase_rad"]
            .apply(lambda value: math.sqrt(float(np.mean(value**2))))
            .rename("_event_rms")
        )
        group = group.join(event_rms, on="event_cluster_id")
        group["_normalized_abs_residual"] = (
            group["residual_phase_rad"].abs() / group["_event_rms"]
        )
        for coordinate, coordinate_name in (
            ("tone_offset_frequency_hz", "tone_offset"),
            ("tone_slot_zero_based", "tone_slot"),
        ):
            values = group[coordinate].to_numpy(dtype=float)
            edges = np.unique(
                np.quantile(
                    values[np.isfinite(values)],
                    np.linspace(0.0, 1.0, coordinate_bins + 1),
                )
            )
            edges[0] = np.nextafter(edges[0], -np.inf)
            edges[-1] = np.nextafter(edges[-1], np.inf)
            labels = np.clip(
                np.digitize(values, edges) - 1,
                0,
                len(edges) - 2,
            )
            work = group.copy()
            work["_bin"] = labels
            for bin_index, selected in work.groupby("_bin"):
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "network": int(network),
                        "model": model,
                        "coordinate": coordinate_name,
                        "coordinate_bin_zero_based": int(bin_index),
                        "coordinate_median": float(
                            selected[coordinate].median()
                        ),
                        "tone_event_count": int(len(selected)),
                        "median_normalized_abs_residual": float(
                            selected["_normalized_abs_residual"].median()
                        ),
                        "median_abs_residual_mrad": float(
                            1.0e3
                            * selected["residual_phase_rad"].abs().median()
                        ),
                    }
                )
    return rows


def _model_figure(
    path: Path,
    summary_rows: list[dict[str, Any]],
    *,
    networks: list[int],
) -> None:
    frame = pd.DataFrame(summary_rows)
    models = [
        "common_phase",
        "common_phase_plus_delay",
        "tone_offset_12bin_rank1",
        "tone_slot_rank1_shared_tones",
        "empirical_uid_rank1_shared_tones",
        "empirical_uid_rank2_shared_tones",
    ]
    labels = [
        "phase",
        "phase+delay",
        "offset (12 bin)",
        "tone slot (shared)",
        "UID mode 1 (shared)",
        "UID modes 1+2 (shared)",
    ]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 8.5),
        sharey=True,
        constrained_layout=True,
    )
    for axis, network in zip(axes.flat, networks, strict=True):
        selected = frame[frame["network"] == network].set_index("model")
        values = [
            selected.loc[model, "median_zero_baseline_r2"]
            if model in selected.index
            else np.nan
            for model in models
        ]
        axis.bar(np.arange(len(models)), values, color="#4c78a8")
        axis.set_title(f"nw{network}")
        axis.set_xticks(np.arange(len(models)), labels, rotation=35, ha="right")
        axis.axhline(0.0, color="0.4", linewidth=0.8)
        axis.set_ylim(-0.05, 1.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("median held-out phase energy explained")
    axes[1, 0].set_ylabel("median held-out phase energy explained")
    figure.suptitle("Leave-one-observation-out transfer-model comparison")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _mapping_figure(
    path: Path,
    pair_rows: list[dict[str, Any]],
    *,
    networks: list[int],
) -> None:
    frame = pd.DataFrame(pair_rows)
    mappings = [
        "exact_uid",
        "interpolated_tone_offset",
        "exact_tone_slot",
    ]
    labels = ["UID", "tone offset", "tone-list slot"]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.5, 8.0),
        sharey=True,
        constrained_layout=True,
    )
    for axis, network in zip(axes.flat, networks, strict=True):
        selected = frame[frame["network"] == network]
        data = [
            selected[selected["mapping"] == mapping]["loading_cosine"].to_numpy()
            for mapping in mappings
        ]
        axis.boxplot(data, tick_labels=labels, showmeans=True)
        axis.set_title(f"nw{network}")
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("cross-observation loading cosine")
    axes[1, 0].set_ylabel("cross-observation loading cosine")
    figure.suptitle(
        "Observation modes retain UID/offset identity, not tone-list slot"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _projection_figure(
    path: Path,
    projection_rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(projection_rows)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    for axis, network in zip(axes, (8, 9), strict=True):
        selected = frame[frame["network"] == network]
        event = selected[selected["population"] == "pointing_event"][
            "template_zero_baseline_r2"
        ].to_numpy()
        null = selected[
            selected["population"] == "clean_pointing_fixed_epoch"
        ]["template_zero_baseline_r2"].to_numpy()
        axis.boxplot(
            [event, null],
            tick_labels=["independent events", "clean fixed epochs"],
            showmeans=True,
        )
        axis.set_title(f"nw{network}")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("science-template phase energy explained")
    figure.suptitle("Science transfer mode generalizes to pointing events")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _residual_figure(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 8.0),
        constrained_layout=True,
    )
    for column, network in enumerate((8, 9)):
        for row_index, coordinate in enumerate(("tone_offset", "tone_slot")):
            axis = axes[row_index, column]
            selected = frame[
                (frame["network"] == network)
                & (frame["coordinate"] == coordinate)
            ]
            for model, label, color in (
                (
                    "common_phase_plus_delay",
                    "phase+delay",
                    "#e45756",
                ),
                ("empirical_uid_rank1", "held-out UID mode", "#4c78a8"),
            ):
                values = selected[selected["model"] == model].sort_values(
                    "coordinate_median"
                )
                x = values["coordinate_median"].to_numpy(dtype=float)
                if coordinate == "tone_offset":
                    x /= 1.0e6
                axis.plot(
                    x,
                    values["median_normalized_abs_residual"],
                    marker="o",
                    label=label,
                    color=color,
                )
            axis.set_title(f"nw{network}: {coordinate.replace('_', ' ')}")
            axis.grid(alpha=0.25)
            if coordinate == "tone_offset":
                axis.set_xlabel("digital tone offset from LO (MHz)")
            else:
                axis.set_xlabel("observation-local tone-list slot")
            if column == 0:
                axis.set_ylabel("median |residual| / event phase RMS")
    axes[0, 0].legend()
    figure.suptitle("Held-out residuals versus recorded hardware coordinates")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _coordinate_movement_figure(
    path: Path,
    identifiability_rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(identifiability_rows)
    networks = sorted(frame["network"].unique())
    slot = frame.groupby("network")[
        "uid_tone_slot_changed_fraction"
    ].median()
    offset = (
        frame.groupby("network")["maximum_abs_tone_offset_shift_hz"].max()
        / 1.0e6
    )
    lo = (
        frame.groupby("network")["lo_shift_hz"].apply(
            lambda value: float(np.max(np.abs(value)))
        )
        / 1.0e6
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.3),
        constrained_layout=True,
    )
    axes[0].bar([str(value) for value in networks], slot.loc[networks])
    axes[0].set_xlabel("network")
    axes[0].set_ylabel("median fraction of UIDs changing list slot")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", alpha=0.25)
    x = np.arange(len(networks))
    axes[1].bar(x - 0.18, offset.loc[networks], 0.36, label="max tone shift")
    axes[1].bar(x + 0.18, lo.loc[networks], 0.36, label="max LO shift")
    axes[1].set_xticks(x, [str(value) for value in networks])
    axes[1].set_xlabel("network")
    axes[1].set_ylabel("maximum shift (MHz)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.suptitle(
        "The dataset changes tone-list slots but not LO or broad tone offset"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--apt-root", type=Path, required=True)
    parser.add_argument("--event-vector-dir", type=Path, required=True)
    parser.add_argument("--tone-analysis-dir", type=Path, required=True)
    parser.add_argument("--pointing-consensus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--networks",
        nargs="+",
        type=int,
        default=list(DEFAULT_NETWORKS),
    )
    parser.add_argument(
        "--event-obsnums",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVENT_OBSNUMS),
    )
    parser.add_argument(
        "--pointing-event-obsnums",
        nargs="+",
        type=int,
        default=list(DEFAULT_POINTING_EVENT_OBSNUMS),
    )
    parser.add_argument(
        "--null-obsnums",
        nargs="+",
        type=int,
        default=list(DEFAULT_NULL_OBSNUMS),
    )
    parser.add_argument("--coordinate-bins", type=int, default=12)
    parser.add_argument("--null-epochs-per-observation", type=int, default=20)
    parser.add_argument("--minimum-event-separation-sec", type=float, default=0.6)
    parser.add_argument("--sigma-threshold", type=float, default=8.0)
    parser.add_argument("--min-phase-mrad", type=float, default=5.0)
    parser.add_argument("--pre-window-sec", type=float, default=0.20)
    parser.add_argument("--guard-window-sec", type=float, default=0.05)
    parser.add_argument("--post-window-sec", type=float, default=0.20)
    parser.add_argument("--dac-fft-size", type=int, default=2**21)
    parser.add_argument("--dac-sample-rate-hz", type=float, default=512.0e6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    networks = [int(value) for value in args.networks]
    event_obsnums = [int(value) for value in args.event_obsnums]
    pointing_event_obsnums = [
        int(value) for value in args.pointing_event_obsnums
    ]
    null_obsnums = [int(value) for value in args.null_obsnums]
    if len(networks) != 6:
        raise ValueError("the standard figure layout requires six networks")
    if args.coordinate_bins < 3:
        raise ValueError("--coordinate-bins must be at least three")
    if args.null_epochs_per_observation < 2:
        raise ValueError("--null-epochs-per-observation must be at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tone_path = args.event_vector_dir / "science_event_tone_vectors.csv"
    event_tones = pd.read_csv(tone_path)
    schemas = set(event_tones["schema_version"].dropna().astype(str))
    if schemas != {REQUIRED_EVENT_SCHEMA}:
        raise ValueError(
            f"requires {REQUIRED_EVENT_SCHEMA}, found {sorted(schemas)}"
        )
    event_tones = event_tones[
        event_tones["network"].isin(networks)
        & event_tones["obsnum"].isin(event_obsnums)
    ].copy()
    template_path = (
        args.tone_analysis_dir / "science_tone_rank_one_modes.csv"
    )
    template_tones = pd.read_csv(template_path)

    coordinate_rows, metadata_rows, raw_inputs = _inventory(
        data_root=args.data_root,
        apt_root=args.apt_root,
        networks=networks,
        dac_fft_size=int(args.dac_fft_size),
        dac_sample_rate_hz=float(args.dac_sample_rate_hz),
    )
    identifiability_rows = _identifiability_rows(
        event_tones,
        networks=networks,
        event_obsnums=event_obsnums,
        coordinate_bins=int(args.coordinate_bins),
    )
    observation_summary, observation_tones, mapping_rows = _observation_modes(
        event_tones,
        networks=networks,
        event_obsnums=event_obsnums,
    )
    cv_events, cv_summary, residual_tones = _cross_validated_models(
        event_tones,
        networks=networks,
        event_obsnums=event_obsnums,
        coordinate_bins=int(args.coordinate_bins),
    )
    residual_bins = _residual_bin_rows(
        residual_tones,
        coordinate_bins=int(args.coordinate_bins),
    )
    sample_frequency = float(
        pd.DataFrame(metadata_rows)[
            pd.DataFrame(metadata_rows)["obsnum"].isin(event_obsnums)
        ]["sample_frequency_hz"].median()
    )
    low_rank_rows, split_rows = _low_rank_rows(
        event_tones,
        networks=networks,
        sample_frequency_hz=sample_frequency,
        pre_window_sec=float(args.pre_window_sec),
        post_window_sec=float(args.post_window_sec),
        sigma_threshold=float(args.sigma_threshold),
    )
    pointing_tone_rows, pointing_selection_rows = _pointing_vectors(
        data_root=args.data_root,
        apt_root=args.apt_root,
        consensus_root=args.pointing_consensus_root,
        networks=networks,
        event_obsnums=pointing_event_obsnums,
        null_obsnums=null_obsnums,
        null_epochs_per_observation=int(args.null_epochs_per_observation),
        minimum_event_separation_sec=float(
            args.minimum_event_separation_sec
        ),
        sigma_threshold=float(args.sigma_threshold),
        min_phase_mrad=float(args.min_phase_mrad),
        pre_window_sec=float(args.pre_window_sec),
        guard_window_sec=float(args.guard_window_sec),
        post_window_sec=float(args.post_window_sec),
    )
    pointing_frame = pd.DataFrame(pointing_tone_rows)
    projection_rows, projection_summary = _template_projection_rows(
        pointing_frame,
        template_tones,
        networks=networks,
    )
    projection_comparison = _pointing_population_comparison_rows(
        projection_rows,
        networks=networks,
    )
    pointing_mode_rows = _pointing_mode_rows(
        pointing_frame,
        template_tones,
        networks=networks,
    )

    output_names = {
        "coordinate_authority": "electronics_coordinate_authority.csv",
        "tone_coordinate_inventory": "tone_electronics_coordinates.csv",
        "raw_metadata": "observation_network_electronics_metadata.csv",
        "identifiability": "coordinate_identifiability.csv",
        "observation_modes": "observation_mode_summary.csv",
        "observation_mode_tones": "observation_mode_tones.csv",
        "mode_mapping_stability": "mode_mapping_stability.csv",
        "cv_events": "heldout_model_event_scores.csv",
        "cv_summary": "heldout_model_summary.csv",
        "residual_tones": "heldout_model_residual_tones.csv",
        "residual_bins": "heldout_residual_coordinate_bins.csv",
        "low_rank": "low_rank_decomposition.csv",
        "mode_splits": "mode_sign_amplitude_stability.csv",
        "pointing_selection": "independent_pointing_vector_selection.csv",
        "pointing_tones": "independent_pointing_tone_vectors.csv",
        "pointing_projection": "independent_pointing_template_scores.csv",
        "pointing_projection_summary": (
            "independent_pointing_template_summary.csv"
        ),
        "pointing_projection_comparison": (
            "independent_pointing_population_comparison.csv"
        ),
        "pointing_modes": "independent_pointing_mode_summary.csv",
        "trigger_inventory": "trigger_telemetry_inventory.csv",
        "model_figure": "heldout_model_comparison.png",
        "mapping_figure": "mode_mapping_stability.png",
        "projection_figure": "independent_pointing_template_projection.png",
        "residual_figure": "heldout_residual_vs_coordinate.png",
        "coordinate_figure": "coordinate_movement_audit.png",
    }
    tables: dict[str, list[dict[str, Any]]] = {
        "coordinate_authority": _authority_rows(),
        "tone_coordinate_inventory": coordinate_rows,
        "raw_metadata": metadata_rows,
        "identifiability": identifiability_rows,
        "observation_modes": observation_summary,
        "observation_mode_tones": observation_tones,
        "mode_mapping_stability": mapping_rows,
        "cv_events": cv_events,
        "cv_summary": cv_summary,
        "residual_tones": residual_tones,
        "residual_bins": residual_bins,
        "low_rank": low_rank_rows,
        "mode_splits": split_rows,
        "pointing_selection": pointing_selection_rows,
        "pointing_tones": pointing_tone_rows,
        "pointing_projection": projection_rows,
        "pointing_projection_summary": projection_summary,
        "pointing_projection_comparison": projection_comparison,
        "pointing_modes": pointing_mode_rows,
        "trigger_inventory": _trigger_inventory_rows(),
    }
    for key, rows in tables.items():
        _write_csv(args.output_dir / output_names[key], rows)
    _model_figure(
        args.output_dir / output_names["model_figure"],
        cv_summary,
        networks=networks,
    )
    _mapping_figure(
        args.output_dir / output_names["mapping_figure"],
        mapping_rows,
        networks=networks,
    )
    _projection_figure(
        args.output_dir / output_names["projection_figure"],
        projection_rows,
    )
    _residual_figure(
        args.output_dir / output_names["residual_figure"],
        residual_bins,
    )
    _coordinate_movement_figure(
        args.output_dir / output_names["coordinate_figure"],
        identifiability_rows,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "description": (
            "Electronics-coordinate inventory, held-out ownership tests, "
            "low-rank stability, and independent pointing/null validation"
        ),
        "semantics": {
            "detector_identity": "APT uid",
            "tone_slot": (
                "zero-based observation-local Header.Toltec.ToneFreq index; "
                "not a PFB-bin claim"
            ),
            "all_tone_denominator": (
                "inventory retains every raw tone; event vectors retain every "
                "APT-usable finite tone, whether responsive or not"
            ),
            "heldout_r2": (
                "zero-baseline held-out phase energy explained after fitting "
                "only the stated event coefficients"
            ),
            "provisional_dac_bin": (
                "derived from a dated software formula; not authoritative "
                "firmware/PFB metadata"
            ),
            "null_population": (
                "fixed epochs in pointings 152390 and 152418, previously "
                "classified clean; not responder-selected"
            ),
            "event_null_auc": (
                "descriptive probability that one selected pointing-event "
                "score exceeds one fixed clean-epoch score; vectors within "
                "an observation are not claimed independent"
            ),
        },
        "parameters": {
            "networks": networks,
            "event_obsnums": event_obsnums,
            "pointing_event_obsnums": pointing_event_obsnums,
            "null_obsnums": null_obsnums,
            "coordinate_bins": int(args.coordinate_bins),
            "null_epochs_per_observation": int(
                args.null_epochs_per_observation
            ),
            "minimum_event_separation_sec": float(
                args.minimum_event_separation_sec
            ),
            "sigma_threshold": float(args.sigma_threshold),
            "minimum_phase_mrad": float(args.min_phase_mrad),
            "pre_window_sec": float(args.pre_window_sec),
            "guard_window_sec": float(args.guard_window_sec),
            "post_window_sec": float(args.post_window_sec),
            "provisional_dac_fft_size": int(args.dac_fft_size),
            "provisional_dac_sample_rate_hz": float(
                args.dac_sample_rate_hz
            ),
        },
        "inputs": {
            "event_tones": _file_identity(tone_path),
            "science_template": _file_identity(template_path),
            "raw_apt_tune_files": raw_inputs,
            "pointing_consensus_root": str(args.pointing_consensus_root),
        },
        "counts": {key: len(rows) for key, rows in tables.items()},
        "outputs": output_names,
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        manifest_text.encode()
    ).hexdigest()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
