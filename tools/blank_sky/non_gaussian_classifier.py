#!/usr/bin/env python3
"""Classify non-Gaussian detector-network behavior in Citlali timestreams.

This is an offline diagnostic tool for RTC/PTC netCDF products. It is meant to
answer a more specific question than the blank-sky null audit:

- does a bad network/scan look impulsive?
- step-like or level-shift-like?
- narrowband / line-like?
- broadly coherent / common-mode-like?

The output is intentionally explicit about the underlying metrics so that the
most useful pieces can later be promoted into Citlali itself if they hold up.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import netCDF4
import numpy as np

from blank_sky_null_audit import (
    _common_mode_spectrum_metrics,
    _corr_abs,
    _downsample_indices,
    _downsample_time_indices,
    _eigen_metrics,
    _get_scan_templates,
    _parse_networks,
    _parse_scans,
    _resolve_obsnum,
    _robust_center_scale,
    _sample_pair_corr_metrics,
    _shape_metrics,
    _tail_metrics,
    _write_csv,
)
from mp_mode_estimator import _infer_dt_sec


def _safe_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _safe_max(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr))


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return np.asarray([], dtype=int)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return (ends - starts).astype(int)


def _dominant_cluster(values: np.ndarray, tol: float) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), 0.0
    if vals.size == 1 or tol <= 0:
        return float(vals[0]), 1.0
    vals.sort()
    best_i = 0
    best_j = 1
    j = 0
    for i in range(vals.size):
        if j < i:
            j = i
        while j + 1 < vals.size and (vals[j + 1] - vals[i]) <= tol:
            j += 1
        if (j - i) > (best_j - best_i):
            best_i = i
            best_j = j
    cluster = vals[best_i : best_j + 1]
    return float(np.median(cluster)), float(cluster.size / vals.size)


def _score01(value: float, ref: float, span: float) -> float:
    if not np.isfinite(value) or span <= 0:
        return 0.0
    return float(np.clip((value - ref) / span, 0.0, 1.0))


def _prepare_detector_matrix(
    signal: np.ndarray,
    flags: np.ndarray,
    dt_sec: float,
    *,
    min_good_frac: float,
    max_det: int | None,
    max_time: int | None,
    clip_z: float,
) -> dict[str, float | int | np.ndarray] | None:
    signal = np.asarray(signal, dtype=float)
    flags = np.asarray(flags)
    if signal.ndim == 1:
        signal = signal[:, None]
        flags = flags[:, None]
    if signal.ndim != 2 or signal.shape[1] < 2:
        return None

    t_idx, time_stride = _downsample_time_indices(signal.shape[0], max_time)
    signal = signal[t_idx, :]
    flags = flags[t_idx, :]

    valid = np.isfinite(signal) & (flags == 0)
    det_good = np.mean(valid, axis=0) >= float(min_good_frac)
    det_idx = np.where(det_good)[0]
    if det_idx.size < 6:
        return None

    det_idx = det_idx[_downsample_indices(det_idx.size, max_det)]
    signal = signal[:, det_idx]
    flags = flags[:, det_idx]
    valid = valid[:, det_idx]

    keep_cols: list[int] = []
    centers: list[float] = []
    scales: list[float] = []
    for j in range(signal.shape[1]):
        xj = signal[valid[:, j], j]
        center, scale = _robust_center_scale(xj)
        if np.isfinite(scale) and scale > 0:
            keep_cols.append(j)
            centers.append(center)
            scales.append(scale)
    if len(keep_cols) < 6:
        return None

    signal = signal[:, keep_cols]
    flags = flags[:, keep_cols]
    valid = valid[:, keep_cols]
    centers_arr = np.asarray(centers, dtype=float)
    scales_arr = np.asarray(scales, dtype=float)

    filled = np.where(valid, signal, centers_arr[None, :])
    x_centered = filled - centers_arr[None, :]
    z = np.where(valid, x_centered / scales_arr[None, :], 0.0)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if np.isfinite(clip_z) and clip_z > 0:
        z = np.clip(z, -clip_z, clip_z)

    return {
        "x_centered": x_centered,
        "z": z,
        "valid": valid,
        "flags": flags,
        "n_time": int(z.shape[0]),
        "n_det": int(z.shape[1]),
        "valid_frac": float(np.mean(valid)),
        "time_stride": int(time_stride),
        "dt_sec": float(dt_sec * time_stride),
    }


def _step_metric(z: np.ndarray, valid: np.ndarray, window: int) -> tuple[float, float]:
    z = np.asarray(z, dtype=float).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    n = z.size
    if n < 16:
        return float("nan"), float("nan")
    w = int(np.clip(window, 4, max(4, n // 4)))
    if n < (2 * w + 2):
        return float("nan"), float("nan")

    x = np.where(valid, z, 0.0)
    good = valid.astype(float)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    gcum = np.concatenate(([0.0], np.cumsum(good)))

    centers = np.arange(w, n - w, dtype=int)
    left_sum = csum[centers] - csum[centers - w]
    left_n = gcum[centers] - gcum[centers - w]
    right_sum = csum[centers + w] - csum[centers]
    right_n = gcum[centers + w] - gcum[centers]

    enough = (left_n >= max(4.0, 0.5 * w)) & (right_n >= max(4.0, 0.5 * w))
    if not np.any(enough):
        return float("nan"), float("nan")
    delta = np.full(centers.size, np.nan, dtype=float)
    delta[enough] = np.abs(right_sum[enough] / right_n[enough] - left_sum[enough] / left_n[enough])
    idx = int(np.nanargmax(delta))
    return float(delta[idx]), float(centers[idx])


def _line_metric(x: np.ndarray, valid: np.ndarray, dt_sec: float, fmin: float, fmax: float) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if x.size < 32 or not np.isfinite(dt_sec) or dt_sec <= 0:
        return float("nan"), float("nan")
    filled = np.where(valid, x, 0.0)
    filled = filled - np.mean(filled)
    if not np.any(np.isfinite(filled)):
        return float("nan"), float("nan")

    win = np.hanning(filled.size)
    spec = np.fft.rfft(filled * win)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(filled.size, d=dt_sec)

    mask = np.isfinite(power) & np.isfinite(freqs)
    if fmin > 0:
        mask &= freqs >= fmin
    if fmax > 0:
        mask &= freqs <= fmax
    if np.sum(mask) < 8:
        return float("nan"), float("nan")

    p = power[mask]
    f = freqs[mask]
    idx = int(np.argmax(p))
    peak = float(p[idx])
    med = float(np.median(p))
    prom = peak / med if np.isfinite(med) and med > 0 else float("nan")
    return prom, float(f[idx])


def _classify_row(row: dict[str, object]) -> dict[str, object]:
    max_template_corr = max(
        float(row.get("cm_corr_el", float("nan"))),
        float(row.get("cm_corr_az", float("nan"))),
        float(row.get("cm_corr_t", float("nan"))),
    )
    impulsive = (
        0.35 * _score01(float(row.get("tail4_binom_z", float("nan"))), 1.0, 4.0)
        + 0.20 * _score01(float(row.get("z_excess_kurtosis", float("nan"))), 0.05, 0.20)
        + 0.15 * _score01(float(row.get("median_det_tail6_frac", float("nan"))), 1.0e-4, 8.0e-4)
        + 0.15 * _score01(float(row.get("median_det_flag_regions_per_ksamp", float("nan"))), 0.2, 1.0)
        + 0.15 * _score01(abs(float(row.get("tail4_pos_neg_asym", float("nan")))), 0.15, 0.45)
    )
    step_like = (
        0.45 * _score01(float(row.get("max_det_step_score", float("nan"))), 3.0, 6.0)
        + 0.35 * _score01(float(row.get("step_det_frac", float("nan"))), 0.10, 0.45)
        + 0.20 * _score01(float(row.get("step_alignment_frac", float("nan"))), 0.25, 0.50)
    )
    line_consensus = _score01(float(row.get("line_consensus_frac", float("nan"))), 0.20, 0.40)
    line_strength = _score01(float(row.get("median_det_line_prom", float("nan"))), 12.0, 30.0)
    cm_line_strength = _score01(float(row.get("cm_line_prominence", float("nan"))), 10.0, 20.0)
    narrowband = (
        line_consensus * (0.70 + 0.30 * line_strength)
        + 0.20 * cm_line_strength
    )
    coherent = (
        0.25 * _score01(float(row.get("med_abs_corr", float("nan"))), 0.03, 0.03)
        + 0.30 * _score01(float(row.get("top_mode_frac", float("nan"))), 0.03, 0.05)
        + 0.25 * _score01(float(row.get("cm_low_mid_ratio", float("nan"))), 1.2, 1.5)
        + 0.20 * _score01(max_template_corr, 0.02, 0.10)
    )

    scores = {
        "impulsive": float(np.clip(impulsive, 0.0, 1.0)),
        "step_like": float(np.clip(step_like, 0.0, 1.0)),
        "narrowband": float(np.clip(narrowband, 0.0, 1.0)),
        "coherent": float(np.clip(coherent, 0.0, 1.0)),
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary = ranked[0][0] if ranked[0][1] >= 0.20 else "weak_mixed"
    secondary = ranked[1][0] if ranked[1][1] >= 0.15 else ""
    return {
        "score_impulsive": scores["impulsive"],
        "score_step_like": scores["step_like"],
        "score_narrowband": scores["narrowband"],
        "score_coherent": scores["coherent"],
        "primary_signature": primary,
        "secondary_signature": secondary,
    }


def _row_value(row: dict[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _make_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    networks = sorted(set(int(row["network"]) for row in rows))
    summary: list[dict[str, object]] = []
    for nw in networks:
        rr = [row for row in rows if int(row["network"]) == nw]

        def med(key: str) -> float:
            vals = np.asarray([_row_value(row, key) for row in rr], dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(np.median(vals)) if vals.size else float("nan")

        tmp = {
            "score_impulsive": med("score_impulsive"),
            "score_step_like": med("score_step_like"),
            "score_narrowband": med("score_narrowband"),
            "score_coherent": med("score_coherent"),
        }
        ranked = sorted(tmp.items(), key=lambda item: item[1], reverse=True)
        primary = ranked[0][0].replace("score_", "") if ranked[0][1] >= 0.20 else "weak_mixed"
        secondary = ranked[1][0].replace("score_", "") if ranked[1][1] >= 0.15 else ""

        summary.append(
            {
                "network": nw,
                "n_rows": len(rr),
                "primary_signature": primary,
                "secondary_signature": secondary,
                "median_valid_frac": med("valid_frac"),
                "median_tail4_binom_z": med("tail4_binom_z"),
                "median_excess_kurtosis": med("z_excess_kurtosis"),
                "median_det_step_score": med("median_det_step_score"),
                "median_det_line_prom": med("median_det_line_prom"),
                "median_med_abs_corr": med("med_abs_corr"),
                "median_top_mode_frac": med("top_mode_frac"),
                "median_cm_low_mid_ratio": med("cm_low_mid_ratio"),
                "median_score_impulsive": tmp["score_impulsive"],
                "median_score_step_like": tmp["score_step_like"],
                "median_score_narrowband": tmp["score_narrowband"],
                "median_score_coherent": tmp["score_coherent"],
            }
        )
    return summary


def _top_rows(rows: list[dict[str, object]], key: str, n: int = 8) -> list[dict[str, object]]:
    ranked = []
    for row in rows:
        val = _row_value(row, key)
        if np.isfinite(val):
            ranked.append((val, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in ranked[:n]]


def _emit_ranked_section(lines: list[str], title: str, rows: list[dict[str, object]], key: str) -> None:
    lines.extend([f"## {title}", ""])
    if not rows:
        lines.extend(["- none", ""])
        return
    for row in rows:
        lines.append(
            "- scan={scan} output_scan={output_scan_index} nw={network} "
            "{key}={value:.2f} primary={primary_signature} secondary={secondary_signature}".format(
                key=key,
                value=float(row[key]),
                **row,
            )
        )
    lines.append("")


def _write_report(
    outpath: Path,
    nc_file: Path,
    array: str,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    top_impulsive = _top_rows(rows, "score_impulsive")
    top_step = _top_rows(rows, "score_step_like")
    top_line = _top_rows(rows, "score_narrowband")
    top_coherent = _top_rows(rows, "score_coherent")

    lines = [
        f"# Non-Gaussian Classifier: {nc_file.name}",
        "",
        f"- Input file: `{nc_file}`",
        f"- Array selection: `{array}`",
        f"- Rows analyzed: `{len(rows)}`",
        "",
        "## Network Summary",
        "",
        "| nw | n_rows | primary | secondary | med impulsive | med step | med line | med coherent | med step score | med line prom | med top-mode | med low/mid |",
        "|---:|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {network} | {n_rows} | {primary_signature} | {secondary_signature} | "
            "{median_score_impulsive:.2f} | {median_score_step_like:.2f} | "
            "{median_score_narrowband:.2f} | {median_score_coherent:.2f} | "
            "{median_det_step_score:.2f} | {median_det_line_prom:.2f} | "
            "{median_top_mode_frac:.3f} | {median_cm_low_mid_ratio:.2f} |".format(**row)
        )
    lines.append("")

    _emit_ranked_section(lines, "Top Impulsive-Like Rows", top_impulsive, "score_impulsive")
    _emit_ranked_section(lines, "Top Step-Like Rows", top_step, "score_step_like")
    _emit_ranked_section(lines, "Top Narrowband-Like Rows", top_line, "score_narrowband")
    _emit_ranked_section(lines, "Top Coherent-Like Rows", top_coherent, "score_coherent")

    lines.extend(
        [
            "## Notes",
            "",
            "- `impulsive` is driven by heavy tails, kurtosis, and flagged-event density.",
            "- `step_like` is driven by abrupt pre/post window jumps and detector alignment in time.",
            "- `narrowband` is driven by detector/common-mode line prominence and frequency consensus.",
            "- `coherent` is driven by pair correlation, top-mode fraction, low/mid common-mode power, and telescope/common-mode coupling.",
            "- These scores are heuristic. The raw metrics in the CSV files are the main product.",
            "",
        ]
    )

    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc-file", required=True, help="RTC or PTC Citlali timestream netCDF")
    ap.add_argument("--array", default="a1100", help="Array selection, e.g. a1100")
    ap.add_argument("--networks", default="all", help="Comma list or 'all'")
    ap.add_argument("--scans", default="all", help="Comma list or 'all'")
    ap.add_argument("--obsnum", default=None, help="Optional obsnum override")
    ap.add_argument("--utils-root", required=True, help="Path containing toltec_dp_utils")
    ap.add_argument("--outdir", default=None, help="Default: <nc parent>/non_gaussian_classifier")
    ap.add_argument("--min-good-frac", type=float, default=0.80, help="Minimum good-sample fraction per detector")
    ap.add_argument("--max-det", type=int, default=None, help="Maximum detectors per scan/network after subsampling")
    ap.add_argument("--max-time", type=int, default=20000, help="Maximum time samples per scan after subsampling")
    ap.add_argument("--clip-z", type=float, default=12.0, help="Clip standardized samples to +/- this value")
    ap.add_argument("--n-pairs", type=int, default=4000, help="Sampled detector pairs for correlation metric")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed for sampled pair metric")
    ap.add_argument("--step-window-sec", type=float, default=0.5, help="Window for step-like detector jumps")
    ap.add_argument("--step-score-thresh", type=float, default=2.5, help="Threshold for counting a detector as step-like")
    ap.add_argument("--line-min-hz", type=float, default=2.0, help="Minimum frequency for detector line metric")
    ap.add_argument("--line-max-hz", type=float, default=20.0, help="Maximum frequency for detector line metric")
    ap.add_argument("--line-prom-thresh", type=float, default=3.0, help="Prominence threshold for line consensus")
    args = ap.parse_args()

    nc_file = Path(os.path.expanduser(args.nc_file)).resolve()
    if not nc_file.exists():
        raise FileNotFoundError(f"missing nc file: {nc_file}")
    outdir = (
        Path(os.path.expanduser(args.outdir)).resolve()
        if args.outdir
        else (nc_file.parent / "non_gaussian_classifier")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    utils_root = Path(os.path.expanduser(args.utils_root)).resolve()
    if str(utils_root) not in sys.path:
        sys.path.insert(0, str(utils_root))
    from toltec_dp_utils.ToltecCitlaliTimestream import ToltecCitlaliTimestream  # pylint: disable=import-error

    tcs = ToltecCitlaliTimestream(ncFile=str(nc_file), array=args.array, load_data=False, interactive=False)
    obsnum = _resolve_obsnum(nc_file, args.obsnum)
    rng = np.random.default_rng(args.seed)

    if tcs.scan_indices is None:
        raise ValueError("scan_indices missing from file; cannot use scan-based analysis")

    with netCDF4.Dataset(str(nc_file)) as ds:
        output_scan_index = (
            np.asarray(ds.variables["output_scan_index"][:], dtype=int)
            if "output_scan_index" in ds.variables
            else np.arange(1, int(tcs.scan_indices.shape[0]) + 1, dtype=int)
        )
        apt_nw = np.asarray(ds.variables["apt_nw"][:], dtype=int)
        selected_global = np.asarray(tcs._det_index, dtype=int)  # noqa: SLF001
        selected_networks = apt_nw[selected_global]
        scans = _parse_scans(args.scans, int(tcs.scan_indices.shape[0]))
        networks = _parse_networks(args.networks, selected_networks)

        rows: list[dict[str, object]] = []

        for scan in scans:
            scan_idx = tcs.getScanIndices(scan)
            dt_sec = _infer_dt_sec(ds, int(scan_idx[0]), int(scan_idx[1]))
            tel_el, tel_az, d_el, d_az, t = _get_scan_templates(ds, scan_idx)

            for nw in networks:
                signal, _ = tcs._get_signal(scan_idx, network=nw)  # noqa: SLF001
                flags, _ = tcs._get_flags(scan_idx, network=nw)  # noqa: SLF001
                prepared = _prepare_detector_matrix(
                    signal=signal,
                    flags=flags,
                    dt_sec=dt_sec,
                    min_good_frac=float(args.min_good_frac),
                    max_det=args.max_det,
                    max_time=args.max_time,
                    clip_z=float(args.clip_z),
                )
                if prepared is None:
                    continue

                x_centered = np.asarray(prepared["x_centered"], dtype=float)
                z = np.asarray(prepared["z"], dtype=float)
                valid = np.asarray(prepared["valid"], dtype=bool)
                flags_used = np.asarray(prepared["flags"])
                z_valid = z[valid]
                if z_valid.size < 32:
                    continue

                dt_eff = float(prepared["dt_sec"])
                fs_eff = 1.0 / dt_eff if np.isfinite(dt_eff) and dt_eff > 0 else float("nan")
                dt_for_step = dt_eff if np.isfinite(dt_eff) and dt_eff > 0 else 1.0e-6
                step_window = max(4, int(round(float(args.step_window_sec) / dt_for_step)))

                med_abs_corr, p95_abs_corr = _sample_pair_corr_metrics(z, int(args.n_pairs), rng)
                top_mode_frac, k90 = _eigen_metrics(z)
                common_mode = np.median(x_centered, axis=1)
                spec = _common_mode_spectrum_metrics(common_mode, fs_eff)
                skew, excess_kurtosis = _shape_metrics(z_valid)
                tail4 = _tail_metrics(z_valid, 4.0)
                tail6 = _tail_metrics(z_valid, 6.0)
                pos4 = int(np.sum(z_valid > 4.0))
                neg4 = int(np.sum(z_valid < -4.0))
                pn_den = pos4 + neg4
                tail4_pos_neg_asym = float((pos4 - neg4) / pn_den) if pn_den > 0 else float("nan")

                det_tail4_frac: list[float] = []
                det_tail6_frac: list[float] = []
                det_flag_frac: list[float] = []
                det_flag_med_run: list[float] = []
                det_flag_max_run: list[float] = []
                det_flag_regions_per_ksamp: list[float] = []
                det_step_score: list[float] = []
                det_step_index: list[float] = []
                det_line_prom: list[float] = []
                det_line_freq: list[float] = []

                for j in range(z.shape[1]):
                    valid_j = valid[:, j]
                    if np.sum(valid_j) < 8:
                        continue
                    z_j = z[:, j]
                    x_j = x_centered[:, j]
                    flags_j = np.asarray(flags_used[:, j] != 0, dtype=bool)
                    z_j_valid = z_j[valid_j]

                    det_tail4_frac.append(float(np.mean(np.abs(z_j_valid) > 4.0)))
                    det_tail6_frac.append(float(np.mean(np.abs(z_j_valid) > 6.0)))

                    det_flag_frac.append(float(np.mean(flags_j)))
                    runs = _run_lengths(flags_j)
                    det_flag_med_run.append(float(np.median(runs)) if runs.size else 0.0)
                    det_flag_max_run.append(float(np.max(runs)) if runs.size else 0.0)
                    det_flag_regions_per_ksamp.append(
                        1000.0 * float(runs.size) / max(float(flags_j.size), 1.0)
                    )

                    step_score, step_index = _step_metric(z_j, valid_j, step_window)
                    det_step_score.append(step_score)
                    det_step_index.append(step_index)

                    line_prom, line_freq = _line_metric(
                        x_j,
                        valid_j,
                        dt_eff,
                        float(args.line_min_hz),
                        float(args.line_max_hz),
                    )
                    det_line_prom.append(line_prom)
                    det_line_freq.append(line_freq)

                step_scores_arr = np.asarray(det_step_score, dtype=float)
                step_idx_arr = np.asarray(det_step_index, dtype=float)
                step_active = np.isfinite(step_scores_arr) & (step_scores_arr >= float(args.step_score_thresh))
                step_center, step_align_frac = _dominant_cluster(
                    step_idx_arr[step_active],
                    tol=max(2.0, 0.5 * step_window),
                )
                step_det_frac = float(np.mean(step_active)) if step_scores_arr.size else float("nan")

                line_prom_arr = np.asarray(det_line_prom, dtype=float)
                line_freq_arr = np.asarray(det_line_freq, dtype=float)
                line_active = np.isfinite(line_prom_arr) & np.isfinite(line_freq_arr) & (
                    line_prom_arr >= float(args.line_prom_thresh)
                )
                freq_res = fs_eff / max(float(prepared["n_time"]), 1.0) if np.isfinite(fs_eff) else float("nan")
                line_center, line_consensus_frac = _dominant_cluster(
                    line_freq_arr[line_active],
                    tol=max(0.05, 2.0 * freq_res) if np.isfinite(freq_res) and freq_res > 0 else 0.05,
                )

                row = {
                    "obsnum": obsnum,
                    "scan": int(scan),
                    "output_scan_index": int(output_scan_index[scan]) if scan < output_scan_index.size else -1,
                    "network": int(nw),
                    "n_det_used": int(prepared["n_det"]),
                    "n_time_used": int(prepared["n_time"]),
                    "valid_frac": float(prepared["valid_frac"]),
                    "fs_hz": float(fs_eff),
                    "z_skew": skew,
                    "z_excess_kurtosis": excess_kurtosis,
                    "tail4_frac": tail4["frac"],
                    "tail4_ratio": tail4["ratio"],
                    "tail4_binom_z": tail4["binom_z"],
                    "tail6_frac": tail6["frac"],
                    "tail6_ratio": tail6["ratio"],
                    "tail6_binom_z": tail6["binom_z"],
                    "tail4_pos_neg_asym": tail4_pos_neg_asym,
                    "median_det_tail4_frac": _safe_median(det_tail4_frac),
                    "max_det_tail4_frac": _safe_max(det_tail4_frac),
                    "median_det_tail6_frac": _safe_median(det_tail6_frac),
                    "median_det_flag_frac": _safe_median(det_flag_frac),
                    "median_det_flag_run_len": _safe_median(det_flag_med_run),
                    "max_det_flag_run_len": _safe_max(det_flag_max_run),
                    "median_det_flag_regions_per_ksamp": _safe_median(det_flag_regions_per_ksamp),
                    "med_abs_corr": med_abs_corr,
                    "p95_abs_corr": p95_abs_corr,
                    "top_mode_frac": top_mode_frac,
                    "k90_corr_modes": k90,
                    "cm_corr_el": _corr_abs(common_mode, tel_el),
                    "cm_corr_del": _corr_abs(common_mode, d_el),
                    "cm_corr_az": _corr_abs(common_mode, tel_az),
                    "cm_corr_daz": _corr_abs(common_mode, d_az),
                    "cm_corr_t": _corr_abs(common_mode, t),
                    "cm_bp_low": spec["cm_bp_low"],
                    "cm_bp_mid": spec["cm_bp_mid"],
                    "cm_bp_high": spec["cm_bp_high"],
                    "cm_low_mid_ratio": spec["cm_low_mid_ratio"],
                    "cm_high_mid_ratio": spec["cm_high_mid_ratio"],
                    "cm_peak_freq_hz": spec["cm_peak_freq_hz"],
                    "cm_peak_prominence": spec["cm_peak_prominence"],
                    "cm_line_prominence": (
                        spec["cm_peak_prominence"]
                        if np.isfinite(spec["cm_peak_freq_hz"]) and spec["cm_peak_freq_hz"] >= float(args.line_min_hz)
                        else float("nan")
                    ),
                    "median_det_step_score": _safe_median(det_step_score),
                    "max_det_step_score": _safe_max(det_step_score),
                    "step_det_frac": step_det_frac,
                    "step_alignment_frac": step_align_frac,
                    "dominant_step_sample": step_center,
                    "dominant_step_time_sec": step_center * dt_eff if np.isfinite(step_center) else float("nan"),
                    "median_det_line_prom": _safe_median(det_line_prom),
                    "max_det_line_prom": _safe_max(det_line_prom),
                    "dominant_line_freq_hz": line_center,
                    "line_consensus_frac": line_consensus_frac,
                }
                row.update(_classify_row(row))
                rows.append(row)

    if not rows:
        raise RuntimeError(f"no rows generated for {nc_file}")

    detailed_csv = outdir / "non_gaussian_classifier_detailed.csv"
    _write_csv(detailed_csv, rows)

    summary_rows = _make_summary_rows(rows)
    summary_csv = outdir / "non_gaussian_classifier_summary_by_network.csv"
    _write_csv(summary_csv, summary_rows)

    report_md = outdir / "NON_GAUSSIAN_CLASSIFIER.md"
    _write_report(report_md, nc_file, args.array, rows, summary_rows)

    print(f"Wrote {detailed_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
