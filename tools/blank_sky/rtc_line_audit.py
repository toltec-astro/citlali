#!/usr/bin/env python3
"""Audit RTC timestreams for narrowband periodic contamination.

This is an offline diagnostic tool intended for RTC outputs taken after
despiking/flagging and before later filter/downsample stages. It computes
masked Welch-style PSDs from contiguous good segments and emits:

- scan/network-level narrowband line candidates that may justify a notch filter
- detector-level recurrent narrowband candidates that may justify bad-detector
  flagging instead of a global notch
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import netCDF4
import numpy as np

try:
    from .blank_sky_null_audit import _parse_networks, _parse_scans, _resolve_obsnum, _write_csv
    from .mp_mode_estimator import _infer_dt_sec
except ImportError:
    from blank_sky_null_audit import _parse_networks, _parse_scans, _resolve_obsnum, _write_csv
    from mp_mode_estimator import _infer_dt_sec


def _sample_det_indices(n: int, max_n: int) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_n, dtype=int)


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _rolling_median(x: np.ndarray, radius: int) -> np.ndarray:
    vals = np.asarray(x, dtype=float)
    if vals.size == 0:
        return vals.copy()
    radius = max(1, int(radius))
    out = np.empty_like(vals, dtype=float)
    for i in range(vals.size):
        j0 = max(0, i - radius)
        j1 = min(vals.size, i + radius + 1)
        out[i] = float(np.median(vals[j0:j1]))
    return out


def _masked_welch_psd(
    x: np.ndarray,
    valid: np.ndarray,
    dt_sec: float,
    *,
    segment_sec: float,
    min_segment_sec: float,
    overlap_frac: float,
    min_windows: int = 2,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    x = np.asarray(x, dtype=float).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if x.size < 16 or valid.size != x.size or not np.isfinite(dt_sec) or dt_sec <= 0:
        return None, None, 0

    fs = 1.0 / dt_sec
    runs = _contiguous_runs(valid)
    longest_run = max((i1 - i0 for i0, i1 in runs), default=0)
    nperseg = max(16, int(round(float(segment_sec) * fs)))
    min_seg_n = max(16, int(round(float(min_segment_sec) * fs)))
    if nperseg < min_seg_n:
        nperseg = min_seg_n
    if longest_run < min_seg_n:
        return None, None, 0
    hop_frac = max(0.05, 1.0 - float(overlap_frac))
    if int(round(longest_run)) > 0:
        denom = 1.0 + hop_frac * max(0, int(min_windows) - 1)
        max_nperseg_for_windows = int(np.floor(float(longest_run) / denom)) if denom > 0 else nperseg
        if max_nperseg_for_windows >= min_seg_n and nperseg > max_nperseg_for_windows:
            nperseg = max_nperseg_for_windows
    nperseg = min(nperseg, int(longest_run))
    hop = max(1, int(round(nperseg * hop_frac)))

    win = np.hanning(nperseg)
    win_norm = float(fs * np.sum(win * win))
    if not np.isfinite(win_norm) or win_norm <= 0:
        return None, None, 0

    acc = None
    n_used = 0
    for i0, i1 in runs:
        seg_len = i1 - i0
        if seg_len < min_seg_n:
            continue
        if seg_len < nperseg:
            starts = [i0]
        else:
            starts = list(range(i0, i1 - nperseg + 1, hop))
            if starts and starts[-1] != (i1 - nperseg):
                starts.append(i1 - nperseg)
        for s in starts:
            e = min(i1, s + nperseg)
            chunk = np.asarray(x[s:e], dtype=float)
            if chunk.size < min_seg_n:
                continue
            if chunk.size < nperseg:
                padded = np.zeros(nperseg, dtype=float)
                padded[: chunk.size] = chunk - float(np.median(chunk))
                chunk = padded
            else:
                chunk = chunk[:nperseg]
                chunk = chunk - float(np.median(chunk))

            spec = np.fft.rfft(chunk * win)
            psd = (np.abs(spec) ** 2) / win_norm
            if psd.size > 2:
                psd[1:-1] *= 2.0
            if acc is None:
                acc = np.zeros_like(psd, dtype=float)
            acc += psd
            n_used += 1

    if acc is None or n_used == 0:
        return None, None, 0
    freq = np.fft.rfftfreq(nperseg, d=dt_sec)
    return freq, acc / float(n_used), n_used


def _find_line_peaks(
    freq: np.ndarray,
    psd: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    prominence_thresh: float,
    continuum_radius_bins: int,
) -> list[dict[str, float]]:
    freq = np.asarray(freq, dtype=float)
    psd = np.asarray(psd, dtype=float)
    good = np.isfinite(freq) & np.isfinite(psd) & (psd > 0)
    if fmin > 0:
        good &= freq >= fmin
    if fmax > 0:
        good &= freq <= fmax
    idx = np.where(good)[0]
    if idx.size < 8:
        return []

    f = freq[idx]
    p = psd[idx]
    continuum = _rolling_median(p, continuum_radius_bins)
    continuum = np.where(np.isfinite(continuum) & (continuum > 0), continuum, np.nanmedian(p))
    if not np.isfinite(continuum).any():
        return []
    prominence = p / continuum

    peaks: list[dict[str, float]] = []
    for i in range(1, f.size - 1):
        if not np.isfinite(prominence[i]) or prominence[i] < prominence_thresh:
            continue
        if prominence[i] < prominence[i - 1] or prominence[i] < prominence[i + 1]:
            continue
        target = 1.0 + 0.5 * max(prominence[i] - 1.0, 0.0)
        j0 = i
        while j0 > 0 and prominence[j0 - 1] >= target:
            j0 -= 1
        j1 = i
        while j1 + 1 < f.size and prominence[j1 + 1] >= target:
            j1 += 1
        width_hz = float(max(f[j1] - f[j0], f[1] - f[0]))
        local_excess = np.maximum(p[j0 : j1 + 1] - continuum[j0 : j1 + 1], 0.0)
        total_power = float(np.trapezoid(p, f))
        line_power = float(np.trapezoid(local_excess, f[j0 : j1 + 1]))
        line_power_frac = line_power / total_power if total_power > 0 else float("nan")
        peaks.append(
            {
                "freq_hz": float(f[i]),
                "prominence": float(prominence[i]),
                "width_hz": width_hz,
                "line_power_frac": float(line_power_frac),
            }
        )
    peaks.sort(key=lambda row: (-row["prominence"], row["freq_hz"]))
    return peaks


def _cluster_peak_rows(rows: list[dict[str, object]], tol_hz: float) -> list[list[dict[str, object]]]:
    if not rows:
        return []
    tol_hz = max(float(tol_hz), 1.0e-6)
    rows = sorted(rows, key=lambda row: float(row["freq_hz"]))
    clusters: list[list[dict[str, object]]] = [[rows[0]]]
    for row in rows[1:]:
        prev = clusters[-1][-1]
        if abs(float(row["freq_hz"]) - float(prev["freq_hz"])) <= tol_hz:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    return clusters


def _common_mode_from_centered(x_centered: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x_centered = np.asarray(x_centered, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    cm = np.zeros(x_centered.shape[0], dtype=float)
    for i in range(x_centered.shape[0]):
        good = valid[i]
        if np.any(good):
            cm[i] = float(np.mean(x_centered[i, good]))
    return cm


def _network_signal_valid(
    signal: np.ndarray,
    flags: np.ndarray,
    *,
    min_good_frac: float,
    max_det: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    flags = np.asarray(flags)
    valid = np.isfinite(signal) & (flags == 0)
    good = np.mean(valid, axis=0) >= float(min_good_frac)
    det_idx = np.where(good)[0]
    if det_idx.size == 0:
        return (
            np.zeros((signal.shape[0], 0), dtype=float),
            np.zeros((signal.shape[0], 0), dtype=bool),
            det_idx,
            np.zeros((signal.shape[0], 0), dtype=float),
        )
    det_idx = det_idx[_sample_det_indices(det_idx.size, max_det)]
    signal = signal[:, det_idx]
    valid = valid[:, det_idx]
    x_centered = np.zeros_like(signal, dtype=float)
    keep = np.zeros(signal.shape[1], dtype=bool)
    for j in range(signal.shape[1]):
        vals = signal[valid[:, j], j]
        if vals.size < 16:
            continue
        med = float(np.median(vals))
        resid = vals - med
        sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.std(resid, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        keep[j] = True
        x_centered[:, j] = signal[:, j] - med
    return x_centered[:, keep], valid[:, keep], det_idx[keep], signal[:, keep]


def _audit_file(nc_file: Path, outdir: Path, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scan_rows: list[dict[str, object]] = []
    det_rows: list[dict[str, object]] = []

    with netCDF4.Dataset(nc_file) as ds:
        obsnum = _resolve_obsnum(nc_file, args.obsnum)
        scan_indices = np.asarray(ds.variables["scan_indices"][:], dtype=int)
        output_scan_index = np.asarray(ds.variables["output_scan_index"][:], dtype=int)
        nw_all = np.asarray(ds.variables["apt_nw"][:], dtype=int)
        uid_all = np.asarray(ds.variables["apt_uid"][:], dtype=int)
        scans = _parse_scans(args.scans, scan_indices.shape[0])
        requested_output_scans = None
        if str(args.output_scans).lower() != "all":
            requested_output_scans = set(_parse_int_list(str(args.output_scans)))
        networks = _parse_networks(args.networks, nw_all)

        for scan in scans:
            if requested_output_scans is not None and int(output_scan_index[scan]) not in requested_output_scans:
                continue
            i0 = int(scan_indices[scan, 0])
            i1 = int(scan_indices[scan, 1])
            dt_sec = _infer_dt_sec(ds, i0, i1)
            fs_hz = 1.0 / dt_sec if np.isfinite(dt_sec) and dt_sec > 0 else float("nan")
            signal_scan = np.asarray(ds.variables["signal"][i0 : i1 + 1, :], dtype=float)
            flags_scan = np.asarray(ds.variables["flags"][i0 : i1 + 1, :], dtype=np.int8)

            for nw in networks:
                det_sel = np.where(nw_all == nw)[0]
                if det_sel.size == 0:
                    continue
                x_centered, valid, det_map_idx, signal_used = _network_signal_valid(
                    signal_scan[:, det_sel],
                    flags_scan[:, det_sel],
                    min_good_frac=float(args.min_good_frac),
                    max_det=int(args.max_det),
                )
                if x_centered.shape[1] < max(4, int(args.min_det_for_network)):
                    continue

                uids = uid_all[det_sel][det_map_idx]
                detector_peak_rows: list[dict[str, object]] = []
                for j in range(x_centered.shape[1]):
                    freq, psd, n_win = _masked_welch_psd(
                        x_centered[:, j],
                        valid[:, j],
                        dt_sec,
                        segment_sec=float(args.segment_sec),
                        min_segment_sec=float(args.min_segment_sec),
                        overlap_frac=float(args.overlap_frac),
                        min_windows=int(args.min_windows),
                    )
                    if freq is None or psd is None or n_win < int(args.min_windows):
                        continue
                    peaks = _find_line_peaks(
                        freq,
                        psd,
                        fmin=float(args.line_min_hz),
                        fmax=float(args.line_max_hz),
                        prominence_thresh=float(args.prominence_thresh),
                        continuum_radius_bins=int(args.continuum_radius_bins),
                    )
                    for peak in peaks[: int(args.max_peaks_per_detector)]:
                        row = {
                            "obsnum": obsnum,
                            "scan": int(scan),
                            "output_scan_index": int(output_scan_index[scan]),
                            "network": int(nw),
                            "uid": int(uids[j]),
                            "freq_hz": float(peak["freq_hz"]),
                            "prominence": float(peak["prominence"]),
                            "width_hz": float(peak["width_hz"]),
                            "line_power_frac": float(peak["line_power_frac"]),
                            "n_psd_windows": int(n_win),
                        }
                        detector_peak_rows.append(row)
                        det_rows.append(row.copy())

                cm = _common_mode_from_centered(x_centered, valid)
                cm_valid = np.sum(valid, axis=1) >= max(4, int(0.25 * x_centered.shape[1]))
                cm_freq, cm_psd, cm_nwin = _masked_welch_psd(
                    cm,
                    cm_valid,
                    dt_sec,
                    segment_sec=float(args.segment_sec),
                    min_segment_sec=float(args.min_segment_sec),
                    overlap_frac=float(args.overlap_frac),
                    min_windows=int(args.min_windows),
                )
                cm_peaks = []
                if cm_freq is not None and cm_psd is not None and cm_nwin >= int(args.min_windows):
                    cm_peaks = _find_line_peaks(
                        cm_freq,
                        cm_psd,
                        fmin=float(args.line_min_hz),
                        fmax=float(args.line_max_hz),
                        prominence_thresh=float(args.cm_prominence_thresh),
                        continuum_radius_bins=int(args.continuum_radius_bins),
                    )

                if not detector_peak_rows and not cm_peaks:
                    continue

                tol_hz = max(float(args.cluster_tol_hz), 2.0 * float(fs_hz) / max(float(x_centered.shape[0]), 1.0))
                for cluster in _cluster_peak_rows(detector_peak_rows, tol_hz):
                    freqs = np.asarray([float(row["freq_hz"]) for row in cluster], dtype=float)
                    proms = np.asarray([float(row["prominence"]) for row in cluster], dtype=float)
                    widths = np.asarray([float(row["width_hz"]) for row in cluster], dtype=float)
                    line_power_fracs = np.asarray([float(row["line_power_frac"]) for row in cluster], dtype=float)
                    uid_vals = sorted({int(row["uid"]) for row in cluster})
                    center = float(np.median(freqs))
                    cm_match_prom = float("nan")
                    cm_match_freq = float("nan")
                    for peak in cm_peaks:
                        if abs(float(peak["freq_hz"]) - center) <= tol_hz:
                            cm_match_prom = float(peak["prominence"])
                            cm_match_freq = float(peak["freq_hz"])
                            break
                    det_frac = float(len(uid_vals) / max(x_centered.shape[1], 1))
                    median_prom = float(np.median(proms))
                    score = det_frac * median_prom
                    recommend_notch = (
                        det_frac >= float(args.notch_min_detector_frac)
                        or (
                            np.isfinite(cm_match_prom)
                            and cm_match_prom >= float(args.notch_min_cm_prominence)
                            and len(uid_vals) >= int(args.notch_min_detectors)
                        )
                    )
                    scan_rows.append(
                        {
                            "obsnum": obsnum,
                            "scan": int(scan),
                            "output_scan_index": int(output_scan_index[scan]),
                            "network": int(nw),
                            "fs_hz": float(fs_hz),
                            "n_det_used": int(x_centered.shape[1]),
                            "cluster_freq_hz": center,
                            "detector_count": int(len(uid_vals)),
                            "detector_frac": det_frac,
                            "median_prominence": median_prom,
                            "max_prominence": float(np.max(proms)),
                            "median_width_hz": float(np.median(widths)),
                            "median_line_power_frac": float(np.median(line_power_fracs)),
                            "common_mode_freq_hz": cm_match_freq,
                            "common_mode_prominence": cm_match_prom,
                            "notch_score": score,
                            "recommend_notch": int(recommend_notch),
                            "uids": ",".join(str(uid) for uid in uid_vals[:20]),
                        }
                    )

    scan_csv = outdir / "rtc_line_audit_scan_network.csv"
    det_csv = outdir / "rtc_line_audit_detector_peaks.csv"
    _write_csv(scan_csv, scan_rows)
    _write_csv(det_csv, det_rows)
    return scan_rows, det_rows


def _annotate_detector_rows_with_cluster_context(
    det_rows: list[dict[str, object]],
    scan_rows: list[dict[str, object]],
    *,
    cluster_tol_hz: float,
) -> None:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in scan_rows:
        key = (str(row["obsnum"]), int(row["output_scan_index"]), int(row["network"]))
        grouped.setdefault(key, []).append(row)
    for det_row in det_rows:
        key = (str(det_row["obsnum"]), int(det_row["output_scan_index"]), int(det_row["network"]))
        candidates = grouped.get(key, [])
        det_freq = float(det_row["freq_hz"])
        best = None
        best_delta = float("inf")
        for row in candidates:
            delta = abs(float(row["cluster_freq_hz"]) - det_freq)
            if delta <= float(cluster_tol_hz) and delta < best_delta:
                best = row
                best_delta = delta
        if best is None:
            det_row["cluster_detector_frac"] = float("nan")
            det_row["cluster_recommend_notch"] = 0
        else:
            det_row["cluster_detector_frac"] = float(best["detector_frac"])
            det_row["cluster_recommend_notch"] = int(best["recommend_notch"])


def _make_detector_candidates(
    det_rows: list[dict[str, object]],
    *,
    min_scan_hits: int,
    min_median_prominence: float,
    min_median_line_power_frac: float,
    cluster_tol_hz: float,
    bad_detector_max_cluster_frac: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in det_rows:
        key = (str(row["obsnum"]), int(row["network"]), int(row["uid"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for (obsnum, nw, uid), rows in grouped.items():
        for cluster in _cluster_peak_rows(rows, cluster_tol_hz):
            freqs = np.asarray([float(row["freq_hz"]) for row in cluster], dtype=float)
            proms = np.asarray([float(row["prominence"]) for row in cluster], dtype=float)
            pfrac = np.asarray([float(row["line_power_frac"]) for row in cluster], dtype=float)
            scan_set = sorted({int(row["output_scan_index"]) for row in cluster})
            med_freq = float(np.median(freqs))
            med_prom = float(np.median(proms))
            med_pfrac = float(np.median(pfrac))
            cluster_det_frac = np.asarray(
                [float(row.get("cluster_detector_frac", float("nan"))) for row in cluster],
                dtype=float,
            )
            cluster_det_frac = cluster_det_frac[np.isfinite(cluster_det_frac)]
            median_cluster_det_frac = float(np.median(cluster_det_frac)) if cluster_det_frac.size else float("nan")
            notch_support_frac = float(
                np.mean([int(row.get("cluster_recommend_notch", 0)) != 0 for row in cluster])
            )
            candidate = (
                len(scan_set) >= int(min_scan_hits)
                and med_prom >= float(min_median_prominence)
                and med_pfrac >= float(min_median_line_power_frac)
                and (not np.isfinite(median_cluster_det_frac) or median_cluster_det_frac <= float(bad_detector_max_cluster_frac))
                and notch_support_frac < 0.5
            )
            out.append(
                {
                    "obsnum": obsnum,
                    "network": int(nw),
                    "uid": int(uid),
                    "dominant_freq_hz": med_freq,
                    "n_peak_hits": int(len(cluster)),
                    "n_scan_hits": int(len(scan_set)),
                    "scan_hits": ",".join(str(v) for v in scan_set[:20]),
                    "median_prominence": med_prom,
                    "max_prominence": float(np.max(proms)),
                    "median_line_power_frac": med_pfrac,
                    "median_cluster_detector_frac": median_cluster_det_frac,
                    "notch_support_frac": notch_support_frac,
                    "recommend_bad_detector": int(candidate),
                }
            )
    out.sort(key=lambda row: (-int(row["recommend_bad_detector"]), -float(row["median_prominence"]), row["dominant_freq_hz"]))
    return out


def _write_report(
    path: Path,
    rtc_files: list[Path],
    scan_rows: list[dict[str, object]],
    det_candidates: list[dict[str, object]],
) -> None:
    notch_rows = [row for row in scan_rows if int(row.get("recommend_notch", 0)) != 0]
    lines = [
        "# RTC Line Audit",
        "",
        "This report summarizes narrowband periodic candidates measured from RTC",
        "timestream outputs using masked Welch PSDs computed only from contiguous",
        "good segments (`flags == 0`).",
        "",
        "Outputs:",
        "",
        "- `rtc_line_audit_scan_network.csv`: scan/network line clusters that may justify a notch",
        "- `rtc_line_audit_detector_peaks.csv`: detector-level peak detections",
        "- `rtc_line_audit_bad_detectors.csv`: recurrent detector-level line candidates",
        "",
        "RTC files audited:",
        "",
    ]
    lines.extend(f"- `{p}`" for p in rtc_files)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `recommend_notch = 1` means the line had broad enough detector support or matched a strong common-mode line.",
            "- `recommend_bad_detector = 1` means one detector repeatedly showed the same strong narrow line across scans.",
            "- Notch candidates and bad-detector candidates are intentionally separate outcomes.",
            "",
            "## Top Notch Candidates",
            "",
        ]
    )
    if not notch_rows:
        lines.append("No notch candidates crossed the current conservative thresholds.")
    else:
        notch_rows = sorted(notch_rows, key=lambda row: (-float(row["notch_score"]), -float(row["median_prominence"])))
        lines.extend(
            [
                "| obsnum | scan | nw | freq [Hz] | det frac | median prom | cm prom | width [Hz] |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in notch_rows[:20]:
            lines.append(
                f"| {row['obsnum']} | {row['output_scan_index']} | {row['network']} | "
                f"{float(row['cluster_freq_hz']):.3f} | {float(row['detector_frac']):.3f} | "
                f"{float(row['median_prominence']):.2f} | {float(row['common_mode_prominence']):.2f} | "
                f"{float(row['median_width_hz']):.3f} |"
            )

    lines.extend(["", "## Top Bad-Detector Candidates", ""])
    bad_rows = [row for row in det_candidates if int(row.get("recommend_bad_detector", 0)) != 0]
    if not bad_rows:
        lines.append("No detector-level line candidates crossed the current conservative thresholds.")
    else:
        lines.extend(
            [
                "| obsnum | nw | uid | freq [Hz] | scan hits | median prom | line power frac | median cluster det frac |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in bad_rows[:20]:
            lines.append(
                f"| {row['obsnum']} | {row['network']} | {row['uid']} | "
                f"{float(row['dominant_freq_hz']):.3f} | {row['n_scan_hits']} | "
                f"{float(row['median_prominence']):.2f} | {float(row['median_line_power_frac']):.3f} | "
                f"{float(row['median_cluster_detector_frac']):.3f} |"
            )

    path.write_text("\n".join(lines) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redu-dir", type=Path, required=True, help="Reduction directory containing RTC timestream files.")
    parser.add_argument("--obsnum", default=None, help="Optional obsnum override for single-file use.")
    parser.add_argument("--scans", default="all", help="Comma-separated scan indices or 'all'.")
    parser.add_argument("--output-scans", default="all", help="Comma-separated output_scan_index values or 'all'.")
    parser.add_argument("--networks", default="all", help="Comma-separated networks or 'all'.")
    parser.add_argument("--line-min-hz", type=float, default=1.0)
    parser.add_argument("--line-max-hz", type=float, default=60.0)
    parser.add_argument("--segment-sec", type=float, default=4.0)
    parser.add_argument("--min-segment-sec", type=float, default=2.0)
    parser.add_argument("--overlap-frac", type=float, default=0.5)
    parser.add_argument("--continuum-radius-bins", type=int, default=8)
    parser.add_argument("--prominence-thresh", type=float, default=8.0)
    parser.add_argument("--cm-prominence-thresh", type=float, default=6.0)
    parser.add_argument("--min-good-frac", type=float, default=0.8)
    parser.add_argument("--min-windows", type=int, default=2)
    parser.add_argument("--max-peaks-per-detector", type=int, default=3)
    parser.add_argument("--max-det", type=int, default=128, help="Maximum detectors per network to analyze; 0 means all.")
    parser.add_argument("--min-det-for-network", type=int, default=16)
    parser.add_argument("--cluster-tol-hz", type=float, default=0.15)
    parser.add_argument("--notch-min-detector-frac", type=float, default=0.10)
    parser.add_argument("--notch-min-detectors", type=int, default=8)
    parser.add_argument("--notch-min-cm-prominence", type=float, default=10.0)
    parser.add_argument("--detector-min-scan-hits", type=int, default=2)
    parser.add_argument("--detector-min-median-prominence", type=float, default=12.0)
    parser.add_argument("--detector-min-line-power-frac", type=float, default=0.10)
    parser.add_argument("--bad-detector-max-cluster-frac", type=float, default=0.10)
    parser.add_argument("--outdir", type=Path, default=None, help="Optional output directory. Defaults to <redu-dir>/rtc_line_audit.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    redu_dir = Path(args.redu_dir)
    rtc_files = sorted(redu_dir.glob("*/raw/*_rtc_timestream.nc"))
    if not rtc_files:
        raise RuntimeError(f"no RTC timestream files found under {redu_dir}")

    outdir = args.outdir if args.outdir is not None else (redu_dir / "rtc_line_audit")
    outdir.mkdir(parents=True, exist_ok=True)

    scan_rows: list[dict[str, object]] = []
    det_rows: list[dict[str, object]] = []
    for rtc_file in rtc_files:
        file_scan_rows, file_det_rows = _audit_file(rtc_file, outdir, args)
        scan_rows.extend(file_scan_rows)
        det_rows.extend(file_det_rows)

    scan_rows.sort(key=lambda row: (str(row["obsnum"]), int(row["output_scan_index"]), int(row["network"]), float(row["cluster_freq_hz"])))
    det_rows.sort(key=lambda row: (str(row["obsnum"]), int(row["output_scan_index"]), int(row["network"]), int(row["uid"]), float(row["freq_hz"])))

    _annotate_detector_rows_with_cluster_context(
        det_rows,
        scan_rows,
        cluster_tol_hz=float(args.cluster_tol_hz),
    )

    det_candidates = _make_detector_candidates(
        det_rows,
        min_scan_hits=int(args.detector_min_scan_hits),
        min_median_prominence=float(args.detector_min_median_prominence),
        min_median_line_power_frac=float(args.detector_min_line_power_frac),
        cluster_tol_hz=float(args.cluster_tol_hz),
        bad_detector_max_cluster_frac=float(args.bad_detector_max_cluster_frac),
    )

    _write_csv(outdir / "rtc_line_audit_bad_detectors.csv", det_candidates)
    _write_report(outdir / "RTC_LINE_AUDIT.md", rtc_files, scan_rows, det_candidates)

    print(f"Wrote {outdir / 'rtc_line_audit_scan_network.csv'}")
    print(f"Wrote {outdir / 'rtc_line_audit_detector_peaks.csv'}")
    print(f"Wrote {outdir / 'rtc_line_audit_bad_detectors.csv'}")
    print(f"Wrote {outdir / 'RTC_LINE_AUDIT.md'}")


if __name__ == "__main__":
    main()
