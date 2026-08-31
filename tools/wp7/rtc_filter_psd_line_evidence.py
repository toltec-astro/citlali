#!/usr/bin/env python3
"""Build one network-native WP-7 D2 PSD and line evidence artifact.

This is an offline measurement tool.  It does not select a downsampling factor,
design a filter, or publish a scientific timestream product.  The line finder
and masked Welch estimator are the established implementations from
``tools/blank_sky/rtc_line_audit.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.blank_sky.rtc_line_audit import (
    _cluster_peak_rows,
    _common_mode_from_centered,
    _find_line_peaks,
    _masked_welch_psd,
)


INPUT_SCHEMA = "citlali-wp7-rtc-filter-psd-line-input-v1"
RESULT_SCHEMA = "citlali-wp7-rtc-filter-psd-line-evidence-v1"
ESTABLISHED_LINE_STRATEGY = "citlali-rtc-line-audit-v1"
NATIVE_TIMING_DOMAIN = "network_native"
DISCOVERY_TIMING_DOMAIN = "legacy_rectangular_discovery"
SUPPORTED_TIMING_DOMAINS = {NATIVE_TIMING_DOMAIN, DISCOVERY_TIMING_DOMAIN}
SUPPORTED_STREAM_STAGES = {
    "native_prefilter",
    "native_post_cleaning_residual",
    "legacy_rtc_output",
    "legacy_ptc_output",
}


@dataclass(frozen=True)
class EstimatorConfig:
    segment_sec: float = 4.0
    min_segment_sec: float = 2.0
    overlap_frac: float = 0.5
    min_windows: int = 2
    line_min_hz: float = 1.0
    line_max_hz: float = 60.0
    prominence_thresh: float = 8.0
    common_mode_prominence_thresh: float = 6.0
    continuum_radius_bins: int = 8
    cluster_tol_hz: float = 0.15


@dataclass(frozen=True)
class EvidenceInput:
    manifest_path: Path
    manifest: dict[str, Any]
    occurrence_id: np.ndarray
    time_sec: np.ndarray
    physical_run_id: np.ndarray
    detector_id: np.ndarray
    signal: np.ndarray
    valid: np.ndarray
    source_excluded: np.ndarray
    input_hashes: dict[str, str]


def _validate_estimator_config(config: EstimatorConfig) -> None:
    if (
        not np.isfinite(config.segment_sec)
        or config.segment_sec <= 0
        or not np.isfinite(config.min_segment_sec)
        or config.min_segment_sec <= 0
        or config.segment_sec < config.min_segment_sec
    ):
        raise RuntimeError("PSD segment durations are invalid")
    if (
        not np.isfinite(config.overlap_frac)
        or config.overlap_frac < 0
        or config.overlap_frac >= 1
    ):
        raise RuntimeError("PSD overlap fraction is invalid")
    if config.min_windows < 1:
        raise RuntimeError("PSD minimum window count is invalid")
    if (
        not np.isfinite(config.line_min_hz)
        or config.line_min_hz < 0
        or not np.isfinite(config.line_max_hz)
        or config.line_max_hz <= config.line_min_hz
    ):
        raise RuntimeError("line search frequency range is invalid")
    if (
        not np.isfinite(config.prominence_thresh)
        or config.prominence_thresh <= 1
        or not np.isfinite(config.common_mode_prominence_thresh)
        or config.common_mode_prominence_thresh <= 1
        or config.continuum_radius_bins < 1
        or not np.isfinite(config.cluster_tol_hz)
        or config.cluster_tol_hz <= 0
    ):
        raise RuntimeError("line search estimator controls are invalid")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _load_array(
    manifest_path: Path,
    arrays: dict[str, Any],
    name: str,
) -> tuple[np.ndarray, str]:
    declaration = arrays.get(name)
    if not isinstance(declaration, str) or not declaration:
        raise RuntimeError(f"array path {name!r} is missing")
    path = (manifest_path.parent / declaration).resolve()
    if not path.is_file():
        raise RuntimeError(f"array {name!r} is absent: {path}")
    return np.load(path, allow_pickle=False), sha256_file(path)


def _broadcast_sample_detector(
    value: np.ndarray,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    value = np.asarray(value)
    if value.shape == (shape[0],):
        value = np.broadcast_to(value[:, None], shape)
    if value.shape != shape:
        raise RuntimeError(
            f"{label} shape {value.shape} does not match sample/detector shape {shape}"
        )
    return np.asarray(value, dtype=bool)


def _validate_run_identity(run_id: np.ndarray) -> None:
    if np.any(run_id < 0):
        raise RuntimeError("physical_run_id values must be nonnegative")
    seen: set[int] = set()
    previous: int | None = None
    for raw_value in run_id:
        value = int(raw_value)
        if value != previous:
            if value in seen:
                raise RuntimeError("physical_run_id is not contiguous")
            seen.add(value)
            previous = value


def load_input(manifest_path: Path) -> EvidenceInput:
    manifest_path = manifest_path.resolve()
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != INPUT_SCHEMA:
        raise RuntimeError("PSD/line input manifest schema is not supported")

    identity = _require_mapping(manifest.get("identity"), "identity")
    required_identity = (
        "case_id",
        "route_family",
        "observation",
        "subobservation",
        "scan",
        "network",
        "array",
        "stream_stage",
        "timing_domain",
        "signal_units",
        "cadence_domain_id",
    )
    missing = [name for name in required_identity if name not in identity]
    if missing:
        raise RuntimeError(f"identity fields are missing: {', '.join(missing)}")
    if identity["stream_stage"] not in SUPPORTED_STREAM_STAGES:
        raise RuntimeError("stream_stage is not supported")
    if identity["timing_domain"] not in SUPPORTED_TIMING_DOMAINS:
        raise RuntimeError("timing_domain is not supported")
    legacy_stage = str(identity["stream_stage"]).startswith("legacy_")
    if legacy_stage != (identity["timing_domain"] == DISCOVERY_TIMING_DOMAIN):
        raise RuntimeError("stream_stage and timing_domain classifications disagree")

    source_mask = _require_mapping(manifest.get("source_mask"), "source_mask")
    if not isinstance(source_mask.get("policy_id"), str) or not source_mask["policy_id"]:
        raise RuntimeError("source_mask.policy_id is required")
    source_mask_status = source_mask.get("status")
    if source_mask_status not in {
        "applied",
        "approved_not_applicable",
        "absent_discovery",
    }:
        raise RuntimeError("source_mask.status is missing or unsupported")
    if identity["timing_domain"] == NATIVE_TIMING_DOMAIN:
        if source_mask_status == "absent_discovery":
            raise RuntimeError("native evidence lacks an approved source-mask disposition")
    elif source_mask_status != "absent_discovery":
        raise RuntimeError("legacy discovery cannot claim an approved source mask")
    line_mask = _require_mapping(manifest.get("line_mask"), "line_mask")
    if not isinstance(line_mask.get("policy_id"), str) or not line_mask["policy_id"]:
        raise RuntimeError("line_mask.policy_id is required")
    if line_mask.get("strategy_id") != ESTABLISHED_LINE_STRATEGY:
        raise RuntimeError("line_mask.strategy_id is not the established strategy")
    if not isinstance(line_mask.get("intervals_hz"), list):
        raise RuntimeError("line_mask.intervals_hz must be an array")
    line_mask_status = line_mask.get("status")
    if line_mask_status not in {"applied", "complete_no_lines", "pending"}:
        raise RuntimeError("line_mask.status is missing or unsupported")
    if line_mask_status == "complete_no_lines" and line_mask["intervals_hz"]:
        raise RuntimeError("complete_no_lines line mask contains intervals")
    if line_mask_status == "pending" and line_mask["intervals_hz"]:
        raise RuntimeError("pending line mask contains credited intervals")
    cadence_domain = _require_mapping(manifest.get("cadence_domain"), "cadence_domain")
    nominal_interval_sec = float(cadence_domain.get("nominal_interval_sec", float("nan")))
    maximum_fractional_deviation = float(
        cadence_domain.get("maximum_fractional_deviation", float("nan"))
    )
    if (
        not np.isfinite(nominal_interval_sec)
        or nominal_interval_sec <= 0
        or not np.isfinite(maximum_fractional_deviation)
        or maximum_fractional_deviation < 0
    ):
        raise RuntimeError("cadence_domain bounds are missing or invalid")

    arrays = _require_mapping(manifest.get("arrays"), "arrays")
    loaded: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for name in (
        "occurrence_id",
        "time_sec",
        "physical_run_id",
        "detector_id",
        "signal",
        "valid",
        "source_excluded",
    ):
        loaded[name], hashes[name] = _load_array(manifest_path, arrays, name)

    occurrence_id = np.asarray(loaded["occurrence_id"], dtype=np.int64).reshape(-1)
    time_sec = np.asarray(loaded["time_sec"], dtype=np.float64).reshape(-1)
    run_id = np.asarray(loaded["physical_run_id"], dtype=np.int64).reshape(-1)
    detector_id = np.asarray(loaded["detector_id"], dtype=np.int64).reshape(-1)
    signal = np.asarray(loaded["signal"], dtype=np.float64)
    if signal.ndim != 2:
        raise RuntimeError("signal must be a sample-by-detector plane")
    if signal.shape != (occurrence_id.size, detector_id.size):
        raise RuntimeError("signal axes do not match occurrence and detector axes")
    if time_sec.size != occurrence_id.size or run_id.size != occurrence_id.size:
        raise RuntimeError("native occurrence, time, and run axes differ in length")
    if occurrence_id.size < 16 or detector_id.size == 0:
        raise RuntimeError("input contains too few samples or no detectors")
    if np.unique(occurrence_id).size != occurrence_id.size:
        raise RuntimeError("native occurrence identities are not unique")
    if np.any(np.diff(occurrence_id) <= 0):
        raise RuntimeError("native occurrence identities are not strictly increasing")
    if not np.all(np.isfinite(time_sec)) or np.any(np.diff(time_sec) <= 0):
        raise RuntimeError("native times must be finite and strictly increasing")
    if np.unique(detector_id).size != detector_id.size:
        raise RuntimeError("detector identities are not unique")
    _validate_run_identity(run_id)
    same_run = run_id[1:] == run_id[:-1]
    measured_intervals = np.diff(time_sec)[same_run]
    if measured_intervals.size == 0:
        raise RuntimeError("no within-run cadence intervals are available")
    measured_deviation = np.abs(measured_intervals - nominal_interval_sec) / nominal_interval_sec
    if np.any(measured_deviation > maximum_fractional_deviation):
        raise RuntimeError("native timing lies outside the declared cadence domain")

    valid = _broadcast_sample_detector(loaded["valid"], signal.shape, "valid")
    source_excluded = _broadcast_sample_detector(
        loaded["source_excluded"], signal.shape, "source_excluded"
    )
    return EvidenceInput(
        manifest_path=manifest_path,
        manifest=manifest,
        occurrence_id=occurrence_id,
        time_sec=time_sec,
        physical_run_id=run_id,
        detector_id=detector_id,
        signal=signal,
        valid=valid,
        source_excluded=source_excluded,
        input_hashes=hashes,
    )


def _cadence_summary(time_sec: np.ndarray, run_id: np.ndarray) -> dict[str, float | int]:
    same_run = run_id[1:] == run_id[:-1]
    intervals = np.diff(time_sec)[same_run]
    if intervals.size == 0:
        raise RuntimeError("no within-run cadence intervals are available")
    dt = float(np.median(intervals))
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("native cadence estimate is invalid")
    return {
        "sample_count": int(time_sec.size),
        "physical_run_count": int(np.unique(run_id).size),
        "within_run_interval_count": int(intervals.size),
        "median_interval_sec": dt,
        "sample_rate_hz": 1.0 / dt,
        "minimum_interval_sec": float(np.min(intervals)),
        "maximum_interval_sec": float(np.max(intervals)),
        "maximum_fractional_interval_deviation": float(
            np.max(np.abs(intervals - dt)) / dt
        ),
        "first_time_sec": float(time_sec[0]),
        "last_time_sec": float(time_sec[-1]),
    }


def _insert_run_separators(
    signal: np.ndarray,
    valid: np.ndarray,
    run_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert invalid sentinels so Welch windows cannot cross physical runs."""
    boundaries = np.where(run_id[1:] != run_id[:-1])[0] + 1
    if boundaries.size == 0:
        return signal, valid
    pieces_x: list[np.ndarray] = []
    pieces_valid: list[np.ndarray] = []
    start = 0
    for end in (*boundaries.tolist(), signal.size):
        pieces_x.append(signal[start:end])
        pieces_valid.append(valid[start:end])
        if end != signal.size:
            pieces_x.append(np.asarray([0.0]))
            pieces_valid.append(np.asarray([False]))
        start = end
    return np.concatenate(pieces_x), np.concatenate(pieces_valid)


def _center_signal(signal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    centered = np.zeros_like(signal, dtype=np.float64)
    for detector in range(signal.shape[1]):
        use = valid[:, detector]
        if np.any(use):
            centered[:, detector] = signal[:, detector] - float(
                np.median(signal[use, detector])
            )
    return centered


def _fixed_frequency_psds(
    data: EvidenceInput,
    dt_sec: float,
    config: EstimatorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    analysis_valid = (
        data.valid & ~data.source_excluded & np.isfinite(data.signal)
    )
    centered = _center_signal(data.signal, analysis_valid)
    expected_nperseg = max(16, int(round(config.segment_sec / dt_sec)))
    minimum_nperseg = max(16, int(round(config.min_segment_sec / dt_sec)))
    expected_nperseg = max(expected_nperseg, minimum_nperseg)
    expected_frequency = np.fft.rfftfreq(expected_nperseg, d=dt_sec)
    psd = np.full(
        (data.detector_id.size, expected_frequency.size), np.nan, dtype=np.float64
    )
    n_windows = np.zeros(data.detector_id.size, dtype=np.int64)
    accepted = np.zeros(data.detector_id.size, dtype=bool)

    for detector in range(data.detector_id.size):
        signal, valid = _insert_run_separators(
            centered[:, detector], analysis_valid[:, detector], data.physical_run_id
        )
        frequency, detector_psd, used = _masked_welch_psd(
            signal,
            valid,
            dt_sec,
            segment_sec=config.segment_sec,
            min_segment_sec=config.min_segment_sec,
            overlap_frac=config.overlap_frac,
            # Preserve one fixed grid across physical runs.  The established
            # helper's min_windows control sizes a segment against the single
            # longest run; D2 instead pools independent windows from every
            # declared run and enforces the total immediately below.
            min_windows=1,
        )
        if frequency is None or detector_psd is None or used < config.min_windows:
            continue
        if frequency.shape != expected_frequency.shape or not np.array_equal(
            frequency, expected_frequency
        ):
            continue
        psd[detector] = detector_psd
        n_windows[detector] = used
        accepted[detector] = True

    if not np.any(accepted):
        raise RuntimeError("no detector has enough contiguous support for the fixed PSD grid")
    return expected_frequency, psd, n_windows, accepted, analysis_valid


def _line_inventory(
    data: EvidenceInput,
    frequency: np.ndarray,
    psd: np.ndarray,
    accepted: np.ndarray,
    analysis_valid: np.ndarray,
    config: EstimatorConfig,
) -> list[dict[str, Any]]:
    detector_rows: list[dict[str, object]] = []
    for detector in np.where(accepted)[0]:
        peaks = _find_line_peaks(
            frequency,
            psd[detector],
            fmin=config.line_min_hz,
            fmax=config.line_max_hz,
            prominence_thresh=config.prominence_thresh,
            continuum_radius_bins=config.continuum_radius_bins,
        )
        for peak in peaks:
            detector_rows.append(
                {
                    "detector_id": int(data.detector_id[detector]),
                    **peak,
                }
            )

    centered = _center_signal(data.signal, analysis_valid)
    common_mode = _common_mode_from_centered(centered[:, accepted], analysis_valid[:, accepted])
    common_valid = np.sum(analysis_valid[:, accepted], axis=1) >= max(
        1, int(math.ceil(0.25 * np.count_nonzero(accepted)))
    )
    common_signal, common_valid = _insert_run_separators(
        common_mode, common_valid, data.physical_run_id
    )
    same_run = data.physical_run_id[1:] == data.physical_run_id[:-1]
    dt_sec = float(np.median(np.diff(data.time_sec)[same_run]))
    common_frequency, common_psd, common_window_count = _masked_welch_psd(
        common_signal,
        common_valid,
        dt_sec,
        segment_sec=config.segment_sec,
        min_segment_sec=config.min_segment_sec,
        overlap_frac=config.overlap_frac,
        min_windows=1,
    )
    common_peaks: list[dict[str, float]] = []
    if (
        common_frequency is not None
        and common_psd is not None
        and common_window_count >= config.min_windows
        and common_frequency.shape == frequency.shape
        and np.array_equal(common_frequency, frequency)
    ):
        common_peaks = _find_line_peaks(
            common_frequency,
            common_psd,
            fmin=config.line_min_hz,
            fmax=config.line_max_hz,
            prominence_thresh=config.common_mode_prominence_thresh,
            continuum_radius_bins=config.continuum_radius_bins,
        )

    inventory: list[dict[str, Any]] = []
    for index, cluster in enumerate(
        _cluster_peak_rows(detector_rows, config.cluster_tol_hz)
    ):
        frequencies = np.asarray([float(row["freq_hz"]) for row in cluster])
        widths = np.asarray([float(row["width_hz"]) for row in cluster])
        prominences = np.asarray([float(row["prominence"]) for row in cluster])
        detector_ids = sorted({int(row["detector_id"]) for row in cluster})
        center = float(np.median(frequencies))
        common_match = [
            peak
            for peak in common_peaks
            if abs(float(peak["freq_hz"]) - center) <= config.cluster_tol_hz
        ]
        inventory.append(
            {
                "line_id": f"line-{index:04d}",
                "center_hz": center,
                "support_low_hz": float(np.min(frequencies - 0.5 * widths)),
                "support_high_hz": float(np.max(frequencies + 0.5 * widths)),
                "detector_count": len(detector_ids),
                "detector_fraction": len(detector_ids) / int(np.count_nonzero(accepted)),
                "detector_ids": detector_ids,
                "median_prominence": float(np.median(prominences)),
                "maximum_prominence": float(np.max(prominences)),
                "common_mode_prominence": (
                    float(common_match[0]["prominence"])
                    if common_match
                    else None
                ),
            }
        )
    return inventory


def _validated_intervals(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    line_mask = _require_mapping(manifest.get("line_mask"), "line_mask")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(line_mask["intervals_hz"]):
        row = _require_mapping(item, f"line_mask.intervals_hz[{index}]")
        low = float(row.get("low_hz", float("nan")))
        high = float(row.get("high_hz", float("nan")))
        if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high <= low:
            raise RuntimeError(f"line mask interval {index} is invalid")
        effective = row.get("effective_before_decimation")
        if not isinstance(effective, bool):
            raise RuntimeError(
                f"line mask interval {index} lacks effective_before_decimation"
            )
        operator_evidence_id = row.get("operator_evidence_id")
        if effective and (
            not isinstance(operator_evidence_id, str) or not operator_evidence_id
        ):
            raise RuntimeError(
                f"line mask interval {index} lacks pre-decimation operator evidence"
            )
        result.append(
            {
                "interval_id": str(row.get("interval_id", f"interval-{index:04d}")),
                "low_hz": low,
                "high_hz": high,
                "effective_before_decimation": effective,
                "operator_evidence_id": operator_evidence_id,
            }
        )
    return result


def _line_fold_inventory(
    lines: list[dict[str, Any]],
    protections: list[dict[str, Any]],
    sample_rate_hz: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for factor in range(1, 257):
        nyquist = sample_rate_hz / (2.0 * factor)
        foldable = 0
        unprotected = 0
        for line in lines:
            is_foldable = factor > 1 and float(line["support_high_hz"]) > nyquist
            protected_by = [
                item["interval_id"]
                for item in protections
                if item["effective_before_decimation"]
                and item["low_hz"] <= float(line["support_low_hz"])
                and item["high_hz"] >= float(line["support_high_hz"])
            ]
            is_protected = bool(protected_by)
            center = float(line["center_hz"])
            folded = abs(((center + nyquist) % (2.0 * nyquist)) - nyquist)
            if is_foldable:
                foldable += 1
                if not is_protected:
                    unprotected += 1
            rows.append(
                {
                    "factor": factor,
                    "output_nyquist_hz": nyquist,
                    "line_id": line["line_id"],
                    "line_center_hz": center,
                    "folded_center_hz": folded,
                    "foldable": is_foldable,
                    "protected_before_decimation": is_protected,
                    "protection_interval_ids": protected_by,
                }
            )
        summaries.append(
            {
                "factor": factor,
                "output_nyquist_hz": nyquist,
                "foldable_line_count": foldable,
                "unprotected_foldable_line_count": unprotected,
                "line_gate": "withhold" if unprotected else "not_blocked_by_inventory",
            }
        )
    return rows, summaries


def _broadband_mask(
    frequency: np.ndarray, protections: list[dict[str, Any]]
) -> np.ndarray:
    eligible = np.ones(frequency.shape, dtype=bool)
    for item in protections:
        eligible &= ~(
            (frequency >= float(item["low_hz"]))
            & (frequency <= float(item["high_hz"]))
        )
    return eligible


def _aggregate_psd(psd: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    values = psd[accepted]
    return np.vstack(
        (
            np.nanmedian(values, axis=0),
            np.nanquantile(values, 0.90, axis=0),
            np.nanquantile(values, 0.95, axis=0),
            np.nanquantile(values, 0.99, axis=0),
            np.nanmax(values, axis=0),
        )
    )


def _write_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, value, allow_pickle=False)
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def build_evidence(
    data: EvidenceInput,
    output_dir: Path,
    config: EstimatorConfig = EstimatorConfig(),
) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    _validate_estimator_config(config)

    cadence = _cadence_summary(data.time_sec, data.physical_run_id)
    dt_sec = float(cadence["median_interval_sec"])
    frequency, psd, n_windows, accepted, analysis_valid = _fixed_frequency_psds(
        data, dt_sec, config
    )
    lines = _line_inventory(
        data, frequency, psd, accepted, analysis_valid, config
    )
    protections = _validated_intervals(data.manifest)
    native_nyquist_hz = 0.5 * float(cadence["sample_rate_hz"])
    if any(float(item["high_hz"]) > native_nyquist_hz for item in protections):
        raise RuntimeError("line mask interval exceeds the measured native Nyquist")
    stage = data.manifest["identity"]["stream_stage"]
    if stage == "native_prefilter":
        fold_rows, fold_summary = _line_fold_inventory(
            lines, protections, float(cadence["sample_rate_hz"])
        )
        ordering_relevance = "predecimation_line_gate_candidate"
    else:
        fold_rows, fold_summary = [], []
        ordering_relevance = "diagnostic_only_postcleaning_stream"
    eligible_frequency = _broadband_mask(frequency, protections)
    aggregate = _aggregate_psd(psd, accepted)

    output_dir.mkdir(parents=True)
    arrays = {
        "frequency_hz": _write_array(output_dir / "frequency_hz.npy", frequency),
        "detector_id": _write_array(output_dir / "detector_id.npy", data.detector_id),
        "psd": _write_array(output_dir / "psd.npy", psd),
        "psd_window_count": _write_array(
            output_dir / "psd_window_count.npy", n_windows
        ),
        "detector_accepted": _write_array(
            output_dir / "detector_accepted.npy", accepted
        ),
        "aggregate_psd": _write_array(
            output_dir / "aggregate_psd.npy", aggregate
        ),
        "broadband_frequency_eligible": _write_array(
            output_dir / "broadband_frequency_eligible.npy", eligible_frequency
        ),
    }

    identity = data.manifest["identity"]
    timing_domain = identity["timing_domain"]
    if timing_domain != NATIVE_TIMING_DOMAIN:
        disposition = "discovery_only_non_native_timing"
    elif (
        stage == "native_post_cleaning_residual"
        and data.manifest["line_mask"]["status"] == "pending"
    ):
        disposition = "measurement_complete_envelope_pending_line_mask"
    elif stage == "native_post_cleaning_residual":
        disposition = "residual_psd_envelope_candidate"
    else:
        disposition = "native_prefilter_line_inventory_candidate"

    nperseg = (frequency.size - 1) * 2
    window = np.hanning(nperseg)
    enbw_hz = float(
        cadence["sample_rate_hz"]
        * np.sum(window * window)
        / (np.sum(window) ** 2)
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "evidence_only_no_factor_or_filter_selected",
        "disposition": disposition,
        "identity": identity,
        "source_mask": data.manifest["source_mask"],
        "cadence_domain": data.manifest["cadence_domain"],
        "line_mask": {
            "policy_id": data.manifest["line_mask"]["policy_id"],
            "strategy_id": data.manifest["line_mask"]["strategy_id"],
            "status": data.manifest["line_mask"]["status"],
            "intervals_hz": protections,
        },
        "input_array_sha256": data.input_hashes,
        "input_manifest_sha256": sha256_file(data.manifest_path),
        "native_axis": {
            **cadence,
            "first_occurrence_id": int(data.occurrence_id[0]),
            "last_occurrence_id": int(data.occurrence_id[-1]),
        },
        "estimator": {
            **asdict(config),
            "implementation": ESTABLISHED_LINE_STRATEGY,
            "window": "numpy.hanning",
            "detrend": "per-window median",
            "sidedness": "one-sided real FFT; interior bins doubled",
            "psd_units": f"{identity['signal_units']}^2/Hz",
            "nperseg": nperseg,
            "equivalent_noise_bandwidth_hz": enbw_hz,
            "physical_run_crossing": "forbidden_by_inserted_invalid_sentinel",
        },
        "detector_summary": {
            "declared_count": int(data.detector_id.size),
            "accepted_count": int(np.count_nonzero(accepted)),
            "rejected_insufficient_contiguous_support_count": int(
                accepted.size - np.count_nonzero(accepted)
            ),
            "aggregate_row_order": ["median", "q90", "q95", "q99", "maximum"],
        },
        "support_summary": {
            "cell_count": int(data.valid.size),
            "original_valid_cell_count": int(np.count_nonzero(data.valid)),
            "source_excluded_original_valid_cell_count": int(
                np.count_nonzero(data.valid & data.source_excluded)
            ),
            "finite_unexcluded_valid_cell_count": int(
                np.count_nonzero(analysis_valid)
            ),
        },
        "line_inventory": {
            "strategy_id": ESTABLISHED_LINE_STRATEGY,
            "ordering_relevance": ordering_relevance,
            "line_count": len(lines),
            "lines": lines,
            "factor_summary": fold_summary,
            "factor_line_rows": fold_rows,
        },
        "artifacts": arrays,
        "limitations": [
            "This artifact does not select a PSD envelope aggregation rule.",
            "This artifact does not select a downsampling factor or design a filter.",
            (
                "Line protection is credited only when explicitly declared "
                "effective before decimation."
            ),
        ],
    }
    report_path = output_dir / "evidence.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment-sec", type=float, default=4.0)
    parser.add_argument("--min-segment-sec", type=float, default=2.0)
    parser.add_argument("--overlap-frac", type=float, default=0.5)
    parser.add_argument("--min-windows", type=int, default=2)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = EstimatorConfig(
        segment_sec=args.segment_sec,
        min_segment_sec=args.min_segment_sec,
        overlap_frac=args.overlap_frac,
        min_windows=args.min_windows,
    )
    result = build_evidence(
        load_input(args.input_manifest), args.output_dir.resolve(), config
    )
    print(json.dumps({"disposition": result["disposition"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
