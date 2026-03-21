#!/usr/bin/env python3
"""Prototype a detector-local second-pass PTC residual deglitcher.

This is an offline design/iteration tool for blank-sky PTC timestreams.
It mirrors the current RTC local-residual compact-event logic closely enough
to answer the practical question:

- if we run a conservative second pass after PCA cleaning, what would it flag?

The prototype works on mini or full `*_ptc_timestream.nc` files and writes:
- a detailed accepted-event CSV
- a scan/network summary CSV
- a short markdown report highlighting the strongest candidate events
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np

try:
    from .blank_sky_null_audit import _parse_networks, _parse_scans, _write_csv
except ImportError:
    from blank_sky_null_audit import _parse_networks, _parse_scans, _write_csv


DEFAULT_REDU_DIR = Path(
    "/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu64"
)


@dataclass
class Options:
    min_spike_sigma: float = 8.0
    min_good_frac: float = 0.5
    baseline_window_sec: float = 0.25
    sigma_scale: float = 0.75
    delta_sigma_scale: float = 0.75
    raw_candidate_rel_sigma_scale: float = 1.0
    raw_window_sec: float = 0.18
    raw_half_peak_frac: float = 0.5
    raw_max_width_sec: float = 0.18
    delta_window_sec: float = 0.12
    delta_half_peak_frac: float = 0.5
    delta_max_width_sec: float = 0.10
    max_step_shift_z: float = 3.0
    merge_within_detector_sec: float = 0.08
    cluster_events_sec: float = 0.08
    min_cluster_detectors: int = 3
    high_score_cluster_override: float = 9.0
    max_auto_flag_clusters_per_network: int = 3


def _filled(var: netCDF4.Variable, fill: float | int | None = None) -> np.ndarray:
    data = var[:]
    if np.ma.isMaskedArray(data):
        if fill is None:
            dtype = np.asarray(data).dtype
            if np.issubdtype(dtype, np.floating):
                fill = float("nan")
            else:
                fill = -2147483647
        data = np.ma.filled(data, fill_value=fill)
    return np.asarray(data)


def _dt_from_dataset(ds: netCDF4.Dataset) -> float:
    for name in ("TelTime", "PpsTime"):
        if name in ds.variables:
            t = np.asarray(ds.variables[name][:], dtype=float).reshape(-1)
            if t.size >= 2:
                dt = float(np.nanmedian(np.diff(t)))
                if np.isfinite(dt) and dt > 0:
                    return dt
    return 1.0


def _robust_center_scale(x: np.ndarray, good: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    good = np.asarray(good, dtype=bool).reshape(-1)
    vals = x[good & np.isfinite(x)]
    if vals.size < 8:
        vals = x[np.isfinite(x)]
    if vals.size < 8:
        return float("nan"), float("nan")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(vals, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return med, float("nan")
    return med, sigma


def _edge_truncate_smooth(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n == 0:
        return x.copy()
    window = int(max(3, window))
    if (window % 2) == 0:
        window += 1
    kernel = np.ones(window, dtype=float)
    sums = np.convolve(x, kernel, mode="same")
    counts = np.convolve(np.ones(n, dtype=float), kernel, mode="same")
    return sums / counts


@dataclass
class Event:
    kind: str
    sample: int
    start_sample: int
    end_sample: int
    score: float
    width_samples: int
    baseline_shift_z: float
    peak_abs_z: float
    peak_delta_abs_z: float
    accepted: bool


def _characterize_event(
    resid: np.ndarray,
    metric_abs_z: np.ndarray,
    base_flags: np.ndarray,
    metric_peak_index: int,
    peak_sample: int,
    gate_half_window: int,
    max_width_samples: int,
    half_peak_frac: float,
    resid_sigma: float,
    max_step_shift_z: float,
    kind: str,
    metric_is_delta: bool,
) -> Event:
    invalid = Event(
        kind=kind,
        sample=-2147483647,
        start_sample=-2147483647,
        end_sample=-2147483647,
        score=float("nan"),
        width_samples=-2147483647,
        baseline_shift_z=float("nan"),
        peak_abs_z=float("nan"),
        peak_delta_abs_z=float("nan"),
        accepted=False,
    )
    if not (np.isfinite(resid_sigma) and resid_sigma > 0):
        return invalid
    if metric_peak_index < 0 or metric_peak_index >= metric_abs_z.size:
        return invalid
    if peak_sample < 0 or peak_sample >= resid.size:
        return invalid
    peak_z = float(metric_abs_z[metric_peak_index])
    if not np.isfinite(peak_z) or peak_z <= 0:
        return invalid

    left_bound = max(0, metric_peak_index - gate_half_window)
    right_bound = min(metric_abs_z.size - 1, metric_peak_index + gate_half_window)
    width_thresh = max(half_peak_frac * peak_z, min(peak_z, 1.5))

    left = metric_peak_index
    while left - 1 >= left_bound and np.isfinite(metric_abs_z[left - 1]) and metric_abs_z[left - 1] >= width_thresh:
        left -= 1
    right = metric_peak_index
    while right + 1 <= right_bound and np.isfinite(metric_abs_z[right + 1]) and metric_abs_z[right + 1] >= width_thresh:
        right += 1

    event_start = max(0, left)
    event_end = min(resid.size - 1, right + 1 if metric_is_delta else right)
    width_samples = max(0, event_end - event_start + 1)

    pre_lo = max(0, peak_sample - gate_half_window)
    pre_hi = max(pre_lo, peak_sample - (2 if metric_is_delta else 1))
    post_lo = min(resid.size, peak_sample + 2)
    post_hi = min(resid.size, peak_sample + gate_half_window + 1)

    pre_vals = resid[pre_lo:pre_hi][(~base_flags[pre_lo:pre_hi]) & np.isfinite(resid[pre_lo:pre_hi])]
    post_vals = resid[post_lo:post_hi][(~base_flags[post_lo:post_hi]) & np.isfinite(resid[post_lo:post_hi])]
    baseline_shift_z = float("nan")
    if pre_vals.size >= 4 and post_vals.size >= 4:
        baseline_shift_z = float(abs(np.median(post_vals) - np.median(pre_vals)) / resid_sigma)

    accepted = bool(
        width_samples <= max_width_samples
        and np.isfinite(baseline_shift_z)
        and baseline_shift_z <= max_step_shift_z
    )
    return Event(
        kind=kind,
        sample=int(peak_sample),
        start_sample=int(event_start),
        end_sample=int(event_end),
        score=float(peak_z),
        width_samples=int(width_samples),
        baseline_shift_z=baseline_shift_z,
        peak_abs_z=float(peak_z) if kind == "raw_like" else float("nan"),
        peak_delta_abs_z=float(peak_z) if kind == "delta_like" else float("nan"),
        accepted=accepted,
    )


def _cluster_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    lo = hi = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx <= hi + 1:
            hi = idx
        else:
            runs.append((lo, hi))
            lo = hi = idx
    runs.append((lo, hi))
    return runs


def _merge_detector_event_rows(
    rows: list[dict[str, object]],
    merge_samples: int,
) -> list[dict[str, object]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: (int(row["uid"]), int(row["sample"])))
    merged: list[dict[str, object]] = []
    group: list[dict[str, object]] = [rows[0]]

    def flush(current: list[dict[str, object]]) -> None:
        best = max(current, key=lambda row: float(row["score"]))
        out = dict(best)
        out["merged_event_count"] = len(current)
        out["merged_kinds"] = ",".join(sorted(set(str(row["kind"]) for row in current)))
        out["start_sample"] = min(int(row["start_sample"]) for row in current)
        out["end_sample"] = max(int(row["end_sample"]) for row in current)
        out["sample"] = int(round(float(np.median([int(row["sample"]) for row in current]))))
        out["time_sec"] = float(out["sample"]) * float(out["dt_sec"])
        out["width_samples"] = int(out["end_sample"]) - int(out["start_sample"]) + 1
        out["width_sec"] = float(out["width_samples"]) * float(out["dt_sec"])
        merged.append(out)

    for row in rows[1:]:
        if (
            int(row["uid"]) == int(group[-1]["uid"])
            and int(row["sample"]) <= int(group[-1]["sample"]) + merge_samples
        ):
            group.append(row)
        else:
            flush(group)
            group = [row]
    flush(group)
    return merged


def _cluster_network_event_rows(
    rows: list[dict[str, object]],
    cluster_samples: int,
) -> list[dict[str, object]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: int(row["sample"]))
    clusters: list[dict[str, object]] = []
    group: list[dict[str, object]] = [rows[0]]

    def flush(current: list[dict[str, object]]) -> None:
        best = max(current, key=lambda row: float(row["score"]))
        samples = [int(row["sample"]) for row in current]
        start_sample = min(int(row["start_sample"]) for row in current)
        end_sample = max(int(row["end_sample"]) for row in current)
        uids = sorted(set(int(row["uid"]) for row in current))
        clusters.append(
            {
                "sample": int(round(float(np.median(samples)))),
                "start_sample": start_sample,
                "end_sample": end_sample,
                "peak_score": float(best["score"]),
                "top_uid": int(best["uid"]),
                "top_kind": str(best["kind"]),
                "n_detector_events": len(current),
                "n_detectors": len(uids),
                "uids": uids,
                "rows": list(current),
            }
        )

    for row in rows[1:]:
        if int(row["sample"]) <= max(int(rr["sample"]) for rr in group) + cluster_samples:
            group.append(row)
        else:
            flush(group)
            group = [row]
    flush(group)
    return clusters


def _iter_ptc_files(redu_dir: Path, obsnums: list[str] | None = None) -> list[Path]:
    files = sorted(redu_dir.glob("*/raw/*_ptc_timestream.nc"))
    if obsnums:
        wanted = {str(obs) for obs in obsnums}
        files = [path for path in files if path.name.split("_")[3] in wanted]
    if not files:
        raise FileNotFoundError(f"no *_ptc_timestream.nc files found under {redu_dir}")
    return files


def _analyze_detector(
    signal: np.ndarray,
    flags: np.ndarray,
    dt_sec: float,
    opts: Options,
) -> tuple[list[Event], np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    flags = np.asarray(flags, dtype=bool).reshape(-1)
    n_pts = signal.size
    base_flags = flags.copy()

    good_frac = float(np.mean((~base_flags) & np.isfinite(signal)))
    if good_frac < opts.min_good_frac:
        return [], np.zeros(n_pts, dtype=bool), np.full(n_pts, np.nan, dtype=float)

    med, sigma = _robust_center_scale(signal, ~base_flags)
    if not np.isfinite(sigma) or sigma <= 0:
        return [], np.zeros(n_pts, dtype=bool), np.full(n_pts, np.nan, dtype=float)

    smooth_window = int(round(opts.baseline_window_sec / dt_sec))
    smooth_window = max(3, smooth_window)
    if (smooth_window % 2) == 0:
        smooth_window += 1

    baseline_input = signal.copy()
    bad = base_flags | ~np.isfinite(baseline_input)
    baseline_input[bad] = med
    smooth = _edge_truncate_smooth(baseline_input, smooth_window)
    resid = signal - smooth

    resid_med, resid_sigma = _robust_center_scale(resid, ~base_flags)
    if not np.isfinite(resid_sigma) or resid_sigma <= 0:
        return [], np.zeros(n_pts, dtype=bool), resid

    abs_dev = np.abs(resid - resid_med)
    local_abs_z = abs_dev / resid_sigma
    raw_candidate_z = opts.raw_candidate_rel_sigma_scale * opts.sigma_scale * opts.min_spike_sigma

    raw_flags = np.zeros(n_pts, dtype=bool)
    events: list[Event] = []

    candidate_samples = [
        i
        for i in range(n_pts)
        if (not base_flags[i]) and np.isfinite(local_abs_z[i]) and local_abs_z[i] > raw_candidate_z
    ]
    raw_gate_half_window = max(4, int(round(opts.raw_window_sec / dt_sec)))
    raw_max_width_samples = max(1, int(round(opts.raw_max_width_sec / dt_sec)))
    for lo, hi in _cluster_runs(candidate_samples):
        best_sample = max(range(lo, hi + 1), key=lambda i: local_abs_z[i])
        event = _characterize_event(
            resid,
            local_abs_z,
            base_flags,
            best_sample,
            best_sample,
            raw_gate_half_window,
            raw_max_width_samples,
            opts.raw_half_peak_frac,
            resid_sigma,
            opts.max_step_shift_z,
            "raw_like",
            False,
        )
        if event.accepted:
            raw_flags[event.start_sample : event.end_sample + 1] = True
            events.append(event)

    local_delta_vals = []
    local_delta_edges = []
    for i in range(n_pts - 1):
        if base_flags[i] or base_flags[i + 1] or raw_flags[i] or raw_flags[i + 1]:
            continue
        if not (np.isfinite(resid[i]) and np.isfinite(resid[i + 1])):
            continue
        local_delta_vals.append(resid[i + 1] - resid[i])
        local_delta_edges.append(i)

    final_flags = raw_flags.copy()
    if len(local_delta_vals) >= 8:
        delta_arr = np.asarray(local_delta_vals, dtype=float)
        delta_med = float(np.median(delta_arr))
        delta_sigma = 1.4826 * float(np.median(np.abs(delta_arr - delta_med)))
        if not np.isfinite(delta_sigma) or delta_sigma <= 0:
            delta_sigma = float(np.std(delta_arr, ddof=1))
        if np.isfinite(delta_sigma) and delta_sigma > 0:
            local_delta_abs_z = np.full(max(n_pts - 1, 0), np.nan, dtype=float)
            candidate_edges: list[int] = []
            local_delta_cutoff = opts.delta_sigma_scale * opts.min_spike_sigma * delta_sigma
            for edge in local_delta_edges:
                abs_delta = abs((resid[edge + 1] - resid[edge]) - delta_med)
                local_delta_abs_z[edge] = abs_delta / delta_sigma
                if abs_delta > local_delta_cutoff:
                    candidate_edges.append(edge)
            delta_gate_half_window = max(4, int(round(opts.delta_window_sec / dt_sec)))
            delta_max_width_samples = max(1, int(round(opts.delta_max_width_sec / dt_sec)))
            for lo, hi in _cluster_runs(candidate_edges):
                best_edge = max(range(lo, hi + 1), key=lambda i: local_delta_abs_z[i])
                event = _characterize_event(
                    resid,
                    local_delta_abs_z,
                    base_flags,
                    best_edge,
                    best_edge + 1,
                    delta_gate_half_window,
                    delta_max_width_samples,
                    opts.delta_half_peak_frac,
                    resid_sigma,
                    opts.max_step_shift_z,
                    "delta_like",
                    True,
                )
                if event.accepted:
                    final_flags[best_edge] = True
                    if best_edge + 1 < n_pts:
                        final_flags[best_edge + 1] = True
                    events.append(event)

    return events, final_flags, resid / resid_sigma


def _analyze_file(
    ptc_path: Path,
    outdir: Path,
    opts: Options,
    scan_spec: str,
    networks_spec: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    with netCDF4.Dataset(ptc_path) as ds:
        output_scan_index = _filled(ds.variables["output_scan_index"], fill=-1).astype(int)
        scan_indices = _filled(ds.variables["scan_indices"], fill=-1).astype(int)
        apt_nw = _filled(ds.variables["apt_nw"], fill=-1).astype(int)
        apt_uid = _filled(ds.variables["apt_uid"], fill=-1).astype(int)
        apt_flag = _filled(ds.variables["apt_flag"], fill=0).astype(int) if "apt_flag" in ds.variables else np.zeros_like(apt_uid)
        all_networks = _parse_networks(networks_spec, apt_nw)
        scan_rows = _parse_scans(scan_spec, output_scan_index.size)
        dt_sec = _dt_from_dataset(ds)
        merge_samples = max(1, int(round(opts.merge_within_detector_sec / dt_sec)))
        cluster_samples = max(1, int(round(opts.cluster_events_sec / dt_sec)))

        obsnum = ptc_path.name.split("_")[3]
        for scan_row in scan_rows:
            scan_num = int(output_scan_index[scan_row])
            if scan_num < 0:
                continue
            start, end = [int(v) for v in scan_indices[scan_row, :2]]
            signal = _filled(ds.variables["signal"][start : end + 1, :], fill=np.nan).astype(float)
            flags = _filled(ds.variables["flags"][start : end + 1, :], fill=0).astype(int) != 0

            for nw in all_networks:
                det_cols = np.where(apt_nw == nw)[0]
                if det_cols.size == 0:
                    continue
                accepted_events: list[dict[str, object]] = []
                proposed_flags = np.zeros((signal.shape[0], det_cols.size), dtype=bool)
                residual_peak = float("nan")
                residual_peak_uid = -1
                local_det_lookup = {int(det_col): local_j for local_j, det_col in enumerate(det_cols)}

                for local_j, det_col in enumerate(det_cols):
                    if apt_flag[det_col] != 0:
                        continue
                    events, det_prop_flags, det_resid_z = _analyze_detector(
                        signal[:, det_col], flags[:, det_col], dt_sec, opts
                    )
                    proposed_flags[:, local_j] = det_prop_flags
                    if np.isfinite(det_resid_z).any():
                        det_peak = float(np.nanmax(np.abs(det_resid_z[~flags[:, det_col]])))
                        if not np.isfinite(residual_peak) or det_peak > residual_peak:
                            residual_peak = det_peak
                            residual_peak_uid = int(apt_uid[det_col])
                    for event in events:
                        accepted_events.append(
                            {
                                "obsnum": obsnum,
                                "scan": scan_num,
                                "network": int(nw),
                                "uid": int(apt_uid[det_col]),
                                "det_index": int(det_col),
                                "kind": event.kind,
                                "sample": int(event.sample),
                                "time_sec": float(event.sample * dt_sec),
                                "score": float(event.score),
                                "width_samples": int(event.width_samples),
                                "width_sec": float(event.width_samples * dt_sec),
                                "baseline_shift_z": float(event.baseline_shift_z),
                                "start_sample": int(event.start_sample),
                                "end_sample": int(event.end_sample),
                                "dt_sec": float(dt_sec),
                            }
                        )

                merged_events = _merge_detector_event_rows(accepted_events, merge_samples)
                merged_events.sort(key=lambda row: (-float(row["score"]), row["sample"], row["uid"]))
                clusters = _cluster_network_event_rows(merged_events, cluster_samples)
                clusters.sort(key=lambda row: (-float(row["peak_score"]), row["sample"], row["top_uid"]))
                candidate_clusters = [
                    cluster
                    for cluster in clusters
                    if cluster["n_detectors"] >= opts.min_cluster_detectors
                    or cluster["peak_score"] >= opts.high_score_cluster_override
                ]
                candidate_clusters.sort(key=lambda row: (-float(row["peak_score"]), row["sample"], row["top_uid"]))
                busy_network_vetoed = len(candidate_clusters) > opts.max_auto_flag_clusters_per_network
                accepted_clusters = (
                    []
                    if busy_network_vetoed
                    else candidate_clusters
                )

                accepted_cluster_events: list[dict[str, object]] = []
                accepted_proposed_flags = np.zeros_like(proposed_flags)
                for cluster_id, cluster in enumerate(accepted_clusters):
                    for row in cluster["rows"]:
                        row_out = dict(row)
                        row_out["cluster_id"] = cluster_id
                        row_out["cluster_sample"] = int(cluster["sample"])
                        row_out["cluster_peak_score"] = float(cluster["peak_score"])
                        row_out["cluster_n_detectors"] = int(cluster["n_detectors"])
                        row_out["cluster_n_detector_events"] = int(cluster["n_detector_events"])
                        accepted_cluster_events.append(row_out)
                        local_j = local_det_lookup.get(int(row["det_index"]))
                        if local_j is not None:
                            accepted_proposed_flags[
                                int(row["start_sample"]) : int(row["end_sample"]) + 1,
                                local_j,
                            ] = True

                n_det = int(det_cols.size)
                n_pts = int(signal.shape[0])
                existing_fraction = float(np.mean(flags[:, det_cols])) if det_cols.size else float("nan")
                proposed_fraction = float(np.mean(accepted_proposed_flags)) if accepted_proposed_flags.size else float("nan")
                new_fraction = (
                    float(np.mean(accepted_proposed_flags & ~flags[:, det_cols]))
                    if det_cols.size
                    else float("nan")
                )
                top = accepted_cluster_events[0] if accepted_cluster_events else None
                top_cluster = accepted_clusters[0] if accepted_clusters else None
                top_candidate_cluster = candidate_clusters[0] if candidate_clusters else None

                summary_rows.append(
                    {
                        "obsnum": obsnum,
                        "scan": scan_num,
                        "network": int(nw),
                        "n_det": n_det,
                        "n_pts": n_pts,
                        "n_merged_events_total": len(merged_events),
                        "n_clusters_total": len(clusters),
                        "busy_network_vetoed": int(busy_network_vetoed),
                        "n_candidate_clusters": len(candidate_clusters),
                        "n_candidate_events": int(sum(int(cluster["n_detector_events"]) for cluster in candidate_clusters)),
                        "n_accepted_events": len(accepted_cluster_events),
                        "n_accepted_clusters": len(accepted_clusters),
                        "n_det_with_proposed_flags": int(np.sum(np.any(accepted_proposed_flags, axis=0))) if accepted_proposed_flags.size else 0,
                        "existing_flagged_fraction": existing_fraction,
                        "proposed_flagged_fraction": proposed_fraction,
                        "newly_flagged_fraction": new_fraction,
                        "max_unflagged_residual_z": residual_peak,
                        "max_unflagged_residual_uid": residual_peak_uid,
                        "top_kind": top["kind"] if top else "",
                        "top_uid": int(top["uid"]) if top else -1,
                        "top_score": float(top["score"]) if top else float("nan"),
                        "top_sample": int(top["sample"]) if top else -1,
                        "top_candidate_cluster_sample": int(top_candidate_cluster["sample"]) if top_candidate_cluster else -1,
                        "top_candidate_cluster_peak_score": float(top_candidate_cluster["peak_score"]) if top_candidate_cluster else float("nan"),
                        "top_candidate_cluster_n_detectors": int(top_candidate_cluster["n_detectors"]) if top_candidate_cluster else 0,
                        "top_candidate_cluster_n_events": int(top_candidate_cluster["n_detector_events"]) if top_candidate_cluster else 0,
                        "top_cluster_sample": int(top_cluster["sample"]) if top_cluster else -1,
                        "top_cluster_peak_score": float(top_cluster["peak_score"]) if top_cluster else float("nan"),
                        "top_cluster_n_detectors": int(top_cluster["n_detectors"]) if top_cluster else 0,
                        "top_cluster_n_events": int(top_cluster["n_detector_events"]) if top_cluster else 0,
                    }
                )
                detailed_rows.extend(accepted_cluster_events)

    return detailed_rows, summary_rows


def _write_report(outpath: Path, detailed: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    lines = [
        "# PTC Second-Pass Prototype",
        "",
        "Offline detector-local compact-event audit for post-PCA PTC timestreams.",
        "",
    ]
    if not detailed:
        lines.extend(["No accepted auto-flag events found.", ""])
    else:
        top_events = sorted(detailed, key=lambda row: (-float(row["score"]), str(row["obsnum"]), int(row["scan"])))[:15]
        lines.extend(["## Top Accepted Events", ""])
        for row in top_events:
            lines.append(
                f"- obs `{row['obsnum']}` scan `{row['scan']}` nw `{row['network']}` "
                f"uid `{row['uid']}` `{row['kind']}` score `{float(row['score']):.2f}` "
                f"sample `{row['sample']}` width `{float(row['width_sec']):.3f} s` "
                f"baseline_shift_z `{float(row['baseline_shift_z']):.2f}`"
            )
        lines.append("")

    lines.extend(["", "## Most Affected Scan/Networks", ""])
    top_rows = sorted(
        summary,
        key=lambda row: (
            -int(row["busy_network_vetoed"]),
            -float(row["newly_flagged_fraction"]),
            -int(row["n_candidate_clusters"]),
        ),
    )[:15]
    for row in top_rows:
        lines.append(
            f"- obs `{row['obsnum']}` scan `{row['scan']}` nw `{row['network']}` "
            f"candidate_events `{row['n_candidate_events']}` candidate_clusters `{row['n_candidate_clusters']}` "
            f"accepted_events `{row['n_accepted_events']}` accepted_clusters `{row['n_accepted_clusters']}` "
            f"(total merged `{row['n_merged_events_total']}` total clusters `{row['n_clusters_total']}`) "
            f"busy_network_vetoed `{int(row['busy_network_vetoed'])}` "
            f"newly_flagged_fraction `{float(row['newly_flagged_fraction']):.4f}` "
            f"top_candidate_cluster_peak_score `{float(row['top_candidate_cluster_peak_score']):.2f}` "
            f"top_candidate_cluster_n_detectors `{int(row['top_candidate_cluster_n_detectors'])}`"
        )
    outpath.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redu-dir", type=Path, default=DEFAULT_REDU_DIR)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--obsnum",
        action="append",
        default=[],
        help="Obsnum to analyze. Can be repeated. Default: all PTC files under redu-dir.",
    )
    parser.add_argument("--scans", default="all", help="0-based scan-row indices within each PTC file, or 'all'")
    parser.add_argument("--networks", default="all")
    parser.add_argument("--min-spike-sigma", type=float, default=8.0)
    parser.add_argument("--min-good-frac", type=float, default=0.5)
    parser.add_argument("--baseline-window-sec", type=float, default=0.25)
    parser.add_argument("--sigma-scale", type=float, default=0.75)
    parser.add_argument("--delta-sigma-scale", type=float, default=0.75)
    parser.add_argument("--raw-candidate-rel-sigma-scale", type=float, default=1.0)
    parser.add_argument("--raw-window-sec", type=float, default=0.18)
    parser.add_argument("--raw-half-peak-frac", type=float, default=0.5)
    parser.add_argument("--raw-max-width-sec", type=float, default=0.18)
    parser.add_argument("--delta-window-sec", type=float, default=0.12)
    parser.add_argument("--delta-half-peak-frac", type=float, default=0.5)
    parser.add_argument("--delta-max-width-sec", type=float, default=0.10)
    parser.add_argument("--max-step-shift-z", type=float, default=3.0)
    parser.add_argument("--merge-within-detector-sec", type=float, default=0.08)
    parser.add_argument("--cluster-events-sec", type=float, default=0.08)
    parser.add_argument("--min-cluster-detectors", type=int, default=3)
    parser.add_argument("--high-score-cluster-override", type=float, default=9.0)
    parser.add_argument("--max-auto-flag-clusters-per-network", type=int, default=3)
    args = parser.parse_args()

    outdir = args.outdir if args.outdir else (args.redu_dir / "ptc_second_pass_prototype")
    outdir.mkdir(parents=True, exist_ok=True)

    opts = Options(
        min_spike_sigma=args.min_spike_sigma,
        min_good_frac=args.min_good_frac,
        baseline_window_sec=args.baseline_window_sec,
        sigma_scale=args.sigma_scale,
        delta_sigma_scale=args.delta_sigma_scale,
        raw_candidate_rel_sigma_scale=args.raw_candidate_rel_sigma_scale,
        raw_window_sec=args.raw_window_sec,
        raw_half_peak_frac=args.raw_half_peak_frac,
        raw_max_width_sec=args.raw_max_width_sec,
        delta_window_sec=args.delta_window_sec,
        delta_half_peak_frac=args.delta_half_peak_frac,
        delta_max_width_sec=args.delta_max_width_sec,
        max_step_shift_z=args.max_step_shift_z,
        merge_within_detector_sec=args.merge_within_detector_sec,
        cluster_events_sec=args.cluster_events_sec,
        min_cluster_detectors=args.min_cluster_detectors,
        high_score_cluster_override=args.high_score_cluster_override,
        max_auto_flag_clusters_per_network=args.max_auto_flag_clusters_per_network,
    )

    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for ptc_path in _iter_ptc_files(args.redu_dir, obsnums=args.obsnum):
        det_rows, sum_rows = _analyze_file(ptc_path, outdir, opts, args.scans, args.networks)
        detailed_rows.extend(det_rows)
        summary_rows.extend(sum_rows)

    if detailed_rows:
        _write_csv(outdir / "ptc_second_pass_detailed.csv", detailed_rows)
    else:
        (outdir / "ptc_second_pass_detailed.csv").write_text("obsnum\n")
    if summary_rows:
        _write_csv(outdir / "ptc_second_pass_summary_by_scan_network.csv", summary_rows)
    else:
        (outdir / "ptc_second_pass_summary_by_scan_network.csv").write_text("obsnum\n")
    _write_report(outdir / "PTC_SECOND_PASS_PROTOTYPE.md", detailed_rows, summary_rows)
    print(f"Wrote {outdir / 'ptc_second_pass_detailed.csv'}")
    print(f"Wrote {outdir / 'ptc_second_pass_summary_by_scan_network.csv'}")
    print(f"Wrote {outdir / 'PTC_SECOND_PASS_PROTOTYPE.md'}")


if __name__ == "__main__":
    main()
