#!/usr/bin/env python3
"""Probe dynamic RTC notch selection on beammap source-crossing TOD.

This is an offline diagnostic for the beammap `source_crossing_tod` products.
It intentionally mirrors the RTC line-audit peak finder used by Citlali, then
reports which shared-line clusters would be selected from the persisted RTC
TOD.  That lets us distinguish "the notch filter does not work" from "the
line selector is not finding the modes that dominate the current plots".
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
from scipy import signal as scipy_signal


ROOT = Path(__file__).resolve().parents[2]
BLANK_SKY_TOOLS = ROOT / "tools" / "blank_sky"
if str(BLANK_SKY_TOOLS) not in sys.path:
    sys.path.insert(0, str(BLANK_SKY_TOOLS))

from rtc_line_audit import (  # noqa: E402
    _cluster_peak_rows,
    _common_mode_from_centered,
    _find_line_peaks,
    _masked_welch_psd,
    _network_signal_valid,
)


FILL_DOUBLE = -999_999.0
FILL_INT = -2147483647


@dataclass(frozen=True)
class AuditConfig:
    line_min_hz: float
    line_max_hz: float
    segment_sec: float
    min_segment_sec: float
    overlap_frac: float
    continuum_radius_bins: int
    prominence_thresh: float
    cm_prominence_thresh: float
    min_good_frac: float
    min_windows: int
    max_peaks_per_detector: int
    max_det: int
    min_det_for_network: int
    cluster_tol_hz: float
    notch_min_detector_frac: float
    notch_min_detectors: int
    notch_min_cm_prominence: float
    apply_min_support_networks: int
    apply_min_detector_frac: float
    apply_min_cm_prominence: float
    apply_width_scale: float
    apply_min_width_hz: float
    apply_max_width_hz: float
    apply_max_notches: int
    apply_cluster_tol_hz: float


def _scalar(ds: netCDF4.Dataset, name: str, default: float | int) -> float:
    if name not in ds.variables:
        return float(default)
    value = np.asarray(ds.variables[name][:]).reshape(-1)
    if value.size == 0:
        return float(default)
    return float(value[0])


def _config_from_nc(ds: netCDF4.Dataset) -> AuditConfig:
    # The post-filter stage is the one we care about for source-crossing TOD.
    line_min = _scalar(ds, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MIN_HZ", math.nan)
    if not np.isfinite(line_min):
        line_min = _scalar(ds, "CONFIG.RTC.LINE_AUDIT.LINE_MIN_HZ", 1.0)
    line_max = _scalar(ds, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MAX_HZ", math.nan)
    if not np.isfinite(line_max):
        line_max = _scalar(ds, "CONFIG.RTC.LINE_AUDIT.LINE_MAX_HZ", 30.0)
    return AuditConfig(
        line_min_hz=float(line_min),
        line_max_hz=float(line_max),
        segment_sec=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.SEGMENT_SEC", 4.0),
        min_segment_sec=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.MIN_SEGMENT_SEC", 2.0),
        overlap_frac=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.OVERLAP_FRAC", 0.5),
        continuum_radius_bins=int(_scalar(ds, "CONFIG.RTC.LINE_AUDIT.CONTINUUM_RADIUS_BINS", 8)),
        prominence_thresh=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.PROMINENCE_THRESH", 8.0),
        cm_prominence_thresh=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.CM_PROMINENCE_THRESH", 6.0),
        min_good_frac=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.MIN_GOOD_FRAC", 0.8),
        min_windows=int(_scalar(ds, "CONFIG.RTC.LINE_AUDIT.MIN_WINDOWS", 2)),
        max_peaks_per_detector=int(_scalar(ds, "CONFIG.RTC.LINE_AUDIT.MAX_PEAKS_PER_DETECTOR", 6)),
        max_det=int(_scalar(ds, "CONFIG.RTC.LINE_AUDIT.MAX_DET", 0)),
        min_det_for_network=int(_scalar(ds, "CONFIG.RTC.LINE_AUDIT.MIN_DET_FOR_NETWORK", 16)),
        cluster_tol_hz=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.CLUSTER_TOL_HZ", 0.15),
        notch_min_detector_frac=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTOR_FRAC", 0.02),
        notch_min_detectors=int(_scalar(ds, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTORS", 8)),
        notch_min_cm_prominence=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_CM_PROMINENCE", 10.0),
        apply_min_support_networks=int(_scalar(ds, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_SUPPORT_NETWORKS", 1)),
        apply_min_detector_frac=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_DETECTOR_FRAC", 0.9),
        apply_min_cm_prominence=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_CM_PROMINENCE", 150.0),
        apply_width_scale=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.APPLY_WIDTH_SCALE", 1.5),
        apply_min_width_hz=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_WIDTH_HZ", 0.25),
        apply_max_width_hz=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_WIDTH_HZ", 1.5),
        apply_max_notches=int(_scalar(ds, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_NOTCHES", 8)),
        apply_cluster_tol_hz=_scalar(ds, "CONFIG.RTC.LINE_AUDIT.APPLY_CLUSTER_TOL_HZ", 0.25),
    )


def _parse_list(value: str) -> list[int] | None:
    value = str(value).strip()
    if value.lower() == "all":
        return None
    return [int(tok.strip()) for tok in value.split(",") if tok.strip()]


def _find_tod_file(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.glob("*/raw/source_crossing_tod/*_rtc_timestream.nc"))
    if not matches:
        matches = sorted(path.glob("source_crossing_tod/*_rtc_timestream.nc"))
    if not matches:
        matches = sorted(path.glob("*_rtc_timestream.nc"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one RTC source-crossing TOD file under {path}, found {len(matches)}")
    return matches[0]


def _find_diag_file(tod_file: Path) -> Path | None:
    diag = tod_file.with_name(tod_file.name.replace("_rtc_timestream.nc", "_rtcdiag.nc"))
    return diag if diag.exists() else None


def _scan_slice(ds: netCDF4.Dataset, scan_row: int) -> slice:
    i0, i1 = np.asarray(ds.variables["scan_indices"][scan_row], dtype=int).tolist()
    # Existing diagnostics use i1 + 1; keep that convention for one-to-one comparison.
    return slice(int(i0), int(i1) + 1)


def _dt_sec(ds: netCDF4.Dataset, sl: slice) -> float:
    for name in ("TelTime", "TelUTC", "PpsTime"):
        if name not in ds.variables:
            continue
        t = np.asarray(ds.variables[name][sl], dtype=float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            return float(np.median(dt))
    return 1.0


def _audit_scan_networks(
    ds: netCDF4.Dataset,
    *,
    scan_row: int,
    cfg: AuditConfig,
    networks: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sl = _scan_slice(ds, scan_row)
    dt_sec = _dt_sec(ds, sl)
    fs_hz = 1.0 / dt_sec if np.isfinite(dt_sec) and dt_sec > 0 else float("nan")
    output_scan_index = int(np.asarray(ds.variables["output_scan_index"][scan_row]).item())
    signal_scan = np.asarray(ds.variables["signal"][sl, :], dtype=float)
    flags_scan = np.asarray(ds.variables["flags"][sl, :], dtype=np.int8)
    nw_all = np.asarray(ds.variables["apt_nw"][:], dtype=int)
    uid_all = np.asarray(ds.variables["apt_uid"][:], dtype=int)

    scan_rows: list[dict[str, object]] = []
    detector_rows: list[dict[str, object]] = []
    for nw in networks:
        det_sel = np.where(nw_all == int(nw))[0]
        if det_sel.size == 0:
            continue
        x_centered, valid, det_map_idx, _ = _network_signal_valid(
            signal_scan[:, det_sel],
            flags_scan[:, det_sel],
            min_good_frac=cfg.min_good_frac,
            max_det=cfg.max_det,
        )
        if x_centered.shape[1] < max(4, cfg.min_det_for_network):
            continue
        uids = uid_all[det_sel][det_map_idx]

        detector_peak_rows: list[dict[str, object]] = []
        for j in range(x_centered.shape[1]):
            freq, psd, n_win = _masked_welch_psd(
                x_centered[:, j],
                valid[:, j],
                dt_sec,
                segment_sec=cfg.segment_sec,
                min_segment_sec=cfg.min_segment_sec,
                overlap_frac=cfg.overlap_frac,
                min_windows=cfg.min_windows,
            )
            if freq is None or psd is None or n_win < cfg.min_windows:
                continue
            peaks = _find_line_peaks(
                freq,
                psd,
                fmin=cfg.line_min_hz,
                fmax=cfg.line_max_hz,
                prominence_thresh=cfg.prominence_thresh,
                continuum_radius_bins=cfg.continuum_radius_bins,
            )
            for peak in peaks[: cfg.max_peaks_per_detector]:
                row = {
                    "scan_row": int(scan_row),
                    "output_scan_index": output_scan_index,
                    "network": int(nw),
                    "uid": int(uids[j]),
                    "freq_hz": float(peak["freq_hz"]),
                    "prominence": float(peak["prominence"]),
                    "width_hz": float(peak["width_hz"]),
                    "line_power_frac": float(peak["line_power_frac"]),
                    "n_psd_windows": int(n_win),
                }
                detector_peak_rows.append(row)
                detector_rows.append(row)

        cm = _common_mode_from_centered(x_centered, valid)
        cm_valid = np.sum(valid, axis=1) >= max(4, int(0.25 * x_centered.shape[1]))
        cm_freq, cm_psd, cm_nwin = _masked_welch_psd(
            cm,
            cm_valid,
            dt_sec,
            segment_sec=cfg.segment_sec,
            min_segment_sec=cfg.min_segment_sec,
            overlap_frac=cfg.overlap_frac,
            min_windows=cfg.min_windows,
        )
        cm_peaks = []
        if cm_freq is not None and cm_psd is not None and cm_nwin >= cfg.min_windows:
            cm_peaks = _find_line_peaks(
                cm_freq,
                cm_psd,
                fmin=cfg.line_min_hz,
                fmax=cfg.line_max_hz,
                prominence_thresh=cfg.cm_prominence_thresh,
                continuum_radius_bins=cfg.continuum_radius_bins,
            )

        if not detector_peak_rows:
            continue
        tol_hz = max(cfg.cluster_tol_hz, 2.0 * fs_hz / max(float(x_centered.shape[0]), 1.0))
        for cluster in _cluster_peak_rows(detector_peak_rows, tol_hz):
            freqs = np.asarray([float(row["freq_hz"]) for row in cluster], dtype=float)
            proms = np.asarray([float(row["prominence"]) for row in cluster], dtype=float)
            widths = np.asarray([float(row["width_hz"]) for row in cluster], dtype=float)
            pfracs = np.asarray([float(row["line_power_frac"]) for row in cluster], dtype=float)
            uid_vals = sorted({int(row["uid"]) for row in cluster})
            center = float(np.median(freqs))
            freq_min = float(np.min(freqs))
            freq_max = float(np.max(freqs))
            median_width = float(np.median(widths))
            half_span = max(abs(center - freq_min), abs(freq_max - center))
            notch_width = max(median_width, median_width + 2.0 * half_span)
            cm_match_freq = float("nan")
            cm_match_prom = float("nan")
            for peak in cm_peaks:
                if abs(float(peak["freq_hz"]) - center) <= tol_hz:
                    cm_match_freq = float(peak["freq_hz"])
                    cm_match_prom = float(peak["prominence"])
                    break
            det_frac = float(len(uid_vals) / max(x_centered.shape[1], 1))
            median_prom = float(np.median(proms))
            recommend = (
                det_frac >= cfg.notch_min_detector_frac
                or (
                    np.isfinite(cm_match_prom)
                    and cm_match_prom >= cfg.notch_min_cm_prominence
                    and len(uid_vals) >= cfg.notch_min_detectors
                )
            )
            scan_rows.append(
                {
                    "scan_row": int(scan_row),
                    "output_scan_index": output_scan_index,
                    "network": int(nw),
                    "fs_hz": float(fs_hz),
                    "n_det_used": int(x_centered.shape[1]),
                    "cluster_freq_hz": center,
                    "detector_count": int(len(uid_vals)),
                    "detector_frac": det_frac,
                    "median_prominence": median_prom,
                    "max_prominence": float(np.max(proms)),
                    "median_width_hz": median_width,
                    "notch_width_hz": notch_width,
                    "freq_min_hz": freq_min,
                    "freq_max_hz": freq_max,
                    "median_line_power_frac": float(np.nanmedian(pfracs)),
                    "common_mode_freq_hz": cm_match_freq,
                    "common_mode_prominence": cm_match_prom,
                    "notch_score": det_frac * median_prom,
                    "recommend_notch": int(recommend),
                }
            )
    return scan_rows, detector_rows


def _select_global_clusters(rows: list[dict[str, object]], cfg: AuditConfig) -> list[dict[str, object]]:
    candidates = [row for row in rows if int(row["recommend_notch"]) != 0]
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda row: float(row["cluster_freq_hz"]))
    tol_hz = max(cfg.cluster_tol_hz, cfg.apply_cluster_tol_hz)
    clusters: list[dict[str, object]] = []
    i = 0
    while i < len(candidates):
        j = i + 1
        while j < len(candidates) and abs(float(candidates[j]["cluster_freq_hz"]) - float(candidates[j - 1]["cluster_freq_hz"])) <= tol_hz:
            j += 1
        group = candidates[i:j]
        networks = sorted({int(row["network"]) for row in group})
        freqs = np.asarray([float(row["cluster_freq_hz"]) for row in group], dtype=float)
        widths = np.asarray([float(row.get("notch_width_hz", row["median_width_hz"])) for row in group], dtype=float)
        freq_los = []
        freq_his = []
        for row in group:
            f0 = float(row["cluster_freq_hz"])
            width = float(row.get("notch_width_hz", row["median_width_hz"]))
            freq_los.append(float(row.get("freq_min_hz", f0 - 0.5 * width)))
            freq_his.append(float(row.get("freq_max_hz", f0 + 0.5 * width)))
        scores = np.asarray([float(row["notch_score"]) for row in group], dtype=float)
        max_det_frac = max(float(row["detector_frac"]) for row in group)
        cm_proms = np.asarray([float(row["common_mode_prominence"]) for row in group], dtype=float)
        finite_cm = cm_proms[np.isfinite(cm_proms)]
        max_cm = float(np.max(finite_cm)) if finite_cm.size else float("nan")
        enough_networks = len(networks) >= cfg.apply_min_support_networks
        strong_cm = (
            np.isfinite(max_cm)
            and max_cm >= cfg.apply_min_cm_prominence
            and max_det_frac >= cfg.apply_min_detector_frac
        )
        if enough_networks or strong_cm:
            width_hz = float(np.nanmedian(widths))
            span_hz = float(np.nanmax(freq_his) - np.nanmin(freq_los))
            if not np.isfinite(width_hz) or width_hz <= 0:
                width_hz = cfg.apply_min_width_hz
            if np.isfinite(span_hz) and span_hz > width_hz:
                width_hz = span_hz
            width_hz *= cfg.apply_width_scale
            width_hz = max(width_hz, cfg.apply_min_width_hz)
            width_hz = min(width_hz, cfg.apply_max_width_hz)
            center = float(np.median(freqs))
            width_hz = min(width_hz, max(0.05, 0.5 * center))
            clusters.append(
                {
                    "center_hz": center,
                    "width_hz": width_hz,
                    "support_networks": int(len(networks)),
                    "networks": ",".join(str(nw) for nw in networks),
                    "max_detector_frac": float(max_det_frac),
                    "max_common_mode_prominence": max_cm,
                    "median_notch_score": float(np.nanmedian(scores)),
                }
            )
        i = j

    clusters.sort(
        key=lambda row: (
            -int(row["support_networks"]),
            -float(row["max_detector_frac"]),
            -(float(row["max_common_mode_prominence"]) if np.isfinite(float(row["max_common_mode_prominence"])) else -1.0),
            -float(row["median_notch_score"]),
            float(row["center_hz"]),
        )
    )
    if cfg.apply_max_notches > 0:
        clusters = clusters[: cfg.apply_max_notches]
    return clusters


def _detector_peaks(
    ds: netCDF4.Dataset,
    *,
    scan_row: int,
    uid: int,
    cfg: AuditConfig,
    min_prominence: float = 1.0,
) -> list[dict[str, float]]:
    uid_all = np.asarray(ds.variables["apt_uid"][:], dtype=int)
    matches = np.where(uid_all == int(uid))[0]
    if matches.size == 0:
        raise RuntimeError(f"uid {uid} not found")
    det = int(matches[0])
    sl = _scan_slice(ds, scan_row)
    dt = _dt_sec(ds, sl)
    y = np.asarray(ds.variables["signal"][sl, det], dtype=float)
    flags = np.asarray(ds.variables["flags"][sl, det], dtype=np.int8)
    valid = np.isfinite(y) & (flags == 0)
    if np.sum(valid) < 16:
        return []
    y = y - float(np.median(y[valid]))
    freq, psd, nwin = _masked_welch_psd(
        y,
        valid,
        dt,
        segment_sec=cfg.segment_sec,
        min_segment_sec=cfg.min_segment_sec,
        overlap_frac=cfg.overlap_frac,
        min_windows=cfg.min_windows,
    )
    if freq is None or psd is None or nwin < cfg.min_windows:
        return []
    peaks = _find_line_peaks(
        freq,
        psd,
        fmin=cfg.line_min_hz,
        fmax=cfg.line_max_hz,
        prominence_thresh=min_prominence,
        continuum_radius_bins=cfg.continuum_radius_bins,
    )
    for peak in peaks:
        peak["n_psd_windows"] = float(nwin)
    return peaks


def _apply_notches_to_detector(
    ds: netCDF4.Dataset,
    *,
    scan_row: int,
    uid: int,
    clusters: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uid_all = np.asarray(ds.variables["apt_uid"][:], dtype=int)
    det = int(np.where(uid_all == int(uid))[0][0])
    sl = _scan_slice(ds, scan_row)
    dt = _dt_sec(ds, sl)
    fs = 1.0 / dt
    y = np.asarray(ds.variables["signal"][sl, det], dtype=float)
    flags = np.asarray(ds.variables["flags"][sl, det], dtype=np.int8)
    valid = np.isfinite(y) & (flags == 0)
    baseline = float(np.median(y[valid])) if np.any(valid) else 0.0
    filtered = y - baseline
    for cluster in clusters:
        f0 = float(cluster["center_hz"])
        width = float(cluster["width_hz"])
        if not np.isfinite(f0) or not np.isfinite(width) or f0 <= 0 or width <= 0 or f0 >= 0.5 * fs:
            continue
        b, a = scipy_signal.iirnotch(w0=f0, Q=f0 / width, fs=fs)
        filtered = scipy_signal.filtfilt(b, a, filtered, method="pad")
    return y - baseline, filtered, valid


def _runtime_diag_rows(diag_file: Path | None, output_scans: list[int]) -> list[dict[str, object]]:
    if diag_file is None:
        return []
    rows: list[dict[str, object]] = []
    with netCDF4.Dataset(diag_file) as ds:
        outputs = np.asarray(ds.variables["output_scan_index"][:], dtype=int)
        nw_ids = np.asarray(ds.variables["rtc_diag_network_ids"][:], dtype=int)
        for output_scan in output_scans:
            match = np.where(outputs == int(output_scan))[0]
            if match.size == 0:
                continue
            s = int(match[0])
            for j, nw in enumerate(nw_ids):
                rows.append(
                    {
                        "output_scan_index": int(output_scan),
                        "network": int(nw),
                        "post_shared_freq_hz": _clean_float(ds.variables["rtc_network_post_line_audit_shared_freq_hz"][s, j]),
                        "post_detector_frac": _clean_float(ds.variables["rtc_network_post_line_audit_shared_detector_frac"][s, j]),
                        "post_median_prominence": _clean_float(ds.variables["rtc_network_post_line_audit_shared_median_prominence"][s, j]),
                        "post_applied_freq_hz": _clean_float(ds.variables["rtc_network_post_line_audit_shared_applied_freq_hz"][s, j]),
                        "post_n_applied_notches": _clean_int(ds.variables["rtc_network_post_line_audit_n_applied_notches"][s, j]),
                    }
                )
    return rows


def _clean_float(value: object) -> float:
    v = float(np.asarray(value).item())
    return float("nan") if v == FILL_DOUBLE else v


def _clean_int(value: object) -> int:
    v = int(np.asarray(value).item())
    return 0 if v == FILL_INT else v


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="redu dir, source_crossing_tod dir, or RTC timestream nc file")
    parser.add_argument("--output-scans", default="5", help="comma list of output_scan_index values, or all")
    parser.add_argument("--networks", default="all", help="comma list of networks, or all")
    parser.add_argument("--uids", default="", help="comma list of detector uids to inspect")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--line-min-hz", type=float, default=None)
    parser.add_argument("--line-max-hz", type=float, default=None)
    parser.add_argument("--max-notches", type=int, default=None)
    args = parser.parse_args()

    tod_file = _find_tod_file(args.path)
    diag_file = _find_diag_file(tod_file)
    outdir = args.outdir or (tod_file.parent / "python_dynamic_notch_probe")
    outdir.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(tod_file) as ds:
        cfg = _config_from_nc(ds)
        if args.line_min_hz is not None:
            cfg = AuditConfig(**{**cfg.__dict__, "line_min_hz": float(args.line_min_hz)})
        if args.line_max_hz is not None:
            cfg = AuditConfig(**{**cfg.__dict__, "line_max_hz": float(args.line_max_hz)})
        if args.max_notches is not None:
            cfg = AuditConfig(**{**cfg.__dict__, "apply_max_notches": int(args.max_notches)})

        outputs = np.asarray(ds.variables["output_scan_index"][:], dtype=int)
        requested_scans = _parse_list(args.output_scans)
        scan_rows_to_run = list(range(outputs.size)) if requested_scans is None else [
            int(np.where(outputs == scan)[0][0]) for scan in requested_scans if np.any(outputs == scan)
        ]
        output_scans = [int(outputs[idx]) for idx in scan_rows_to_run]
        nw_all = np.asarray(ds.variables["apt_nw"][:], dtype=int)
        requested_networks = _parse_list(args.networks)
        networks = sorted(set(int(v) for v in nw_all)) if requested_networks is None else requested_networks
        uids = _parse_list(args.uids) or []
        uid_all = np.asarray(ds.variables["apt_uid"][:], dtype=int)
        uid_nw = {int(uid): int(nw_all[np.where(uid_all == int(uid))[0][0]]) for uid in uids if np.any(uid_all == int(uid))}

        all_cluster_rows: list[dict[str, object]] = []
        detector_peak_rows: list[dict[str, object]] = []
        selected_rows: list[dict[str, object]] = []
        uid_rows: list[dict[str, object]] = []
        for scan_row in scan_rows_to_run:
            clusters, det_peaks = _audit_scan_networks(ds, scan_row=scan_row, cfg=cfg, networks=networks)
            all_cluster_rows.extend(clusters)
            detector_peak_rows.extend(det_peaks)
            selected = _select_global_clusters(clusters, cfg)
            for rank, row in enumerate(selected, start=1):
                selected_rows.append({"output_scan_index": int(outputs[scan_row]), "rank": rank, **row})

            for uid in uids:
                peaks_before = _detector_peaks(ds, scan_row=scan_row, uid=uid, cfg=cfg, min_prominence=1.0)
                raw, filtered, valid = _apply_notches_to_detector(ds, scan_row=scan_row, uid=uid, clusters=selected)
                dt = _dt_sec(ds, _scan_slice(ds, scan_row))
                std_before = float(np.std(raw[valid], ddof=1)) if np.sum(valid) > 1 else float("nan")
                std_after = float(np.std(filtered[valid], ddof=1)) if np.sum(valid) > 1 else float("nan")
                for rank, peak in enumerate(peaks_before[:10], start=1):
                    uid_rows.append(
                        {
                            "output_scan_index": int(outputs[scan_row]),
                            "uid": int(uid),
                            "network": uid_nw.get(int(uid), FILL_INT),
                            "rank": rank,
                            "freq_hz": float(peak["freq_hz"]),
                            "prominence": float(peak["prominence"]),
                            "width_hz": float(peak["width_hz"]),
                            "line_power_frac": float(peak["line_power_frac"]),
                            "n_psd_windows": int(peak["n_psd_windows"]),
                            "std_before": std_before,
                            "std_after_python_selected_notches": std_after,
                            "fs_hz": float(1.0 / dt),
                        }
                    )

    runtime_rows = _runtime_diag_rows(diag_file, output_scans)
    _write_csv(outdir / "python_residual_clusters.csv", all_cluster_rows)
    _write_csv(outdir / "python_selected_global_notches.csv", selected_rows)
    _write_csv(outdir / "python_detector_peaks.csv", detector_peak_rows)
    _write_csv(outdir / "python_uid_peak_probe.csv", uid_rows)
    _write_csv(outdir / "citlali_runtime_post_line_audit.csv", runtime_rows)

    print(f"RTC TOD: {tod_file}")
    print(f"Runtime diag: {diag_file if diag_file else 'not found'}")
    print(f"Wrote {outdir / 'python_residual_clusters.csv'}")
    print(f"Wrote {outdir / 'python_selected_global_notches.csv'}")
    print(f"Wrote {outdir / 'python_uid_peak_probe.csv'}")


if __name__ == "__main__":
    main()
