#!/usr/bin/env python3
"""Estimate adaptive PCA mode counts from a fitted Marchenko-Pastur bulk model.

This is a prototype analysis tool for detector covariance spectra in Citlali
RTC/PTC timestream products. The intended use case is per-scan, per-network
mode counting for atmospheric/common-mode cleaning:

- robustly whiten detectors within a scan/network
- optionally band-limit the timestream to an atmosphere-dominated band
- compute the detector covariance eigenspectrum
- fit the Marchenko-Pastur bulk using trimmed eigenvalue quantiles
- count modes above the fitted upper edge lambda_plus

The resulting `k_mp` is a candidate adaptive cut depth for coherent modes. It
is most relevant for broadband/coherent contamination and is not a replacement
for despiking or other heavy-tail mitigation.
"""

from __future__ import annotations

import argparse
import csv
import math
from functools import lru_cache
from pathlib import Path

import netCDF4
import numpy as np


MP_FIT_PROBS = (0.1, 0.5, 0.9)


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _parse_scans(spec: str, n_scans: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(n_scans))
    scans = _parse_int_list(spec)
    for scan in scans:
        if scan < 0 or scan >= n_scans:
            raise ValueError(f"scan {scan} out of range [0, {n_scans})")
    return scans


def _parse_networks(spec: str, available: np.ndarray) -> list[int]:
    have = sorted(int(v) for v in np.unique(np.asarray(available, dtype=int)))
    if spec.lower() == "all":
        return have
    requested = _parse_int_list(spec)
    bad = [nw for nw in requested if nw not in have]
    if bad:
        raise ValueError(f"requested network(s) not present: {bad}; available={have}")
    return requested


def _resolve_obsnum(nc_file: Path, fallback: str | None) -> str:
    if fallback:
        return str(fallback)
    parts = [tok for tok in nc_file.name.split("_") if tok.isdigit()]
    return parts[0] if parts else "unknown"


def _downsample_time_indices(n: int, max_n: int) -> tuple[np.ndarray, int]:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=int), 1
    stride = int(math.ceil(n / max_n))
    idx = np.arange(0, n, stride, dtype=int)
    if idx.size > max_n:
        idx = idx[:max_n]
    return idx, stride


def _robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 0:
        sigma = 1.4826 * mad
    else:
        sigma = float(np.std(x, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return med, float("nan")
    return med, sigma


def _infer_dt_sec(ds: netCDF4.Dataset, i0: int, i1: int) -> float:
    for name in ("TelTime", "TelUTC", "PpsTime"):
        if name not in ds.variables:
            continue
        t = np.asarray(ds.variables[name][i0 : i1 + 1], dtype=float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            continue
        return float(np.median(dt))
    return 1.0


def _bandpass_fft(
    x: np.ndarray,
    valid: np.ndarray,
    dt_sec: float,
    band_low_hz: float,
    band_high_hz: float,
) -> np.ndarray:
    if dt_sec <= 0 or (band_low_hz <= 0 and band_high_hz <= 0):
        return x
    freqs = np.fft.rfftfreq(x.shape[0], d=dt_sec)
    keep = np.ones_like(freqs, dtype=bool)
    if band_low_hz > 0:
        keep &= freqs >= band_low_hz
    if band_high_hz > 0:
        keep &= freqs <= band_high_hz
    if not np.any(keep):
        return x
    filled = np.where(valid, x, 0.0)
    spec = np.fft.rfft(filled, axis=0)
    spec[~keep, :] = 0.0
    return np.fft.irfft(spec, n=x.shape[0], axis=0)


def _prepare_detector_matrix(
    signal: np.ndarray,
    flags: np.ndarray,
    dt_sec: float,
    *,
    min_good_frac: float,
    max_det: int,
    max_time: int,
    band_low_hz: float,
    band_high_hz: float,
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
    dt_eff = float(dt_sec * time_stride)

    valid = np.isfinite(signal) & (flags == 0)
    det_good = np.mean(valid, axis=0) >= float(min_good_frac)
    det_idx = np.where(det_good)[0]
    if det_idx.size < 6:
        return None
    if max_det > 0 and det_idx.size > max_det:
        pick = np.linspace(0, det_idx.size - 1, max_det, dtype=int)
        det_idx = det_idx[pick]

    signal = signal[:, det_idx]
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
    valid = valid[:, keep_cols]
    centers_arr = np.asarray(centers, dtype=float)
    scales_arr = np.asarray(scales, dtype=float)

    filled = np.where(valid, signal, centers_arr[None, :])
    x_centered = filled - centers_arr[None, :]

    if band_low_hz > 0 or band_high_hz > 0:
        x_centered = _bandpass_fft(x_centered, valid, dt_eff, band_low_hz, band_high_hz)

    keep_cols2: list[int] = []
    scales2: list[float] = []
    for j in range(x_centered.shape[1]):
        xj = x_centered[valid[:, j], j]
        _, scale = _robust_center_scale(xj)
        if np.isfinite(scale) and scale > 0:
            keep_cols2.append(j)
            scales2.append(scale)
    if len(keep_cols2) < 6:
        return None

    x_centered = x_centered[:, keep_cols2]
    valid = valid[:, keep_cols2]
    scales2_arr = np.asarray(scales2, dtype=float)
    z = np.where(valid, x_centered / scales2_arr[None, :], 0.0)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if np.isfinite(clip_z) and clip_z > 0:
        z = np.clip(z, -clip_z, clip_z)

    return {
        "z": z,
        "valid": valid,
        "n_time": int(z.shape[0]),
        "n_det": int(z.shape[1]),
        "valid_frac": float(np.mean(valid)),
        "time_stride": int(time_stride),
        "dt_sec": float(dt_eff),
    }


def _cov_eigenvalues(z: np.ndarray) -> np.ndarray:
    n_t = z.shape[0]
    zz = np.nan_to_num(np.asarray(z, dtype=float), nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cov = (zz.T @ zz) / max(n_t - 1, 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(np.asarray(evals, dtype=float))[::-1]
    return evals[np.isfinite(evals) & (evals > 0)]


@lru_cache(maxsize=512)
def _mp_quantiles(q_round: float, grid_n: int = 4096) -> tuple[float, float, float]:
    q = float(q_round)
    q = min(max(q, 1e-4), 0.9999)
    lam_minus = (1.0 - math.sqrt(q)) ** 2
    lam_plus = (1.0 + math.sqrt(q)) ** 2
    x = np.linspace(lam_minus, lam_plus, grid_n, dtype=float)
    y = np.sqrt(np.maximum((lam_plus - x) * (x - lam_minus), 0.0))
    pdf = y / np.maximum(2.0 * math.pi * q * x, 1e-12)
    cdf = np.zeros_like(x)
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))
    if cdf[-1] <= 0:
        return (lam_minus, 0.5 * (lam_minus + lam_plus), lam_plus)
    cdf /= cdf[-1]
    return tuple(float(np.interp(p, cdf, x)) for p in MP_FIT_PROBS)


def _fit_mp_bulk(
    evals_desc: np.ndarray,
    n_time: int,
    *,
    bulk_keep_frac: float,
    q_grid_size: int,
) -> dict[str, float]:
    evals_desc = np.asarray(evals_desc, dtype=float)
    evals_desc = evals_desc[np.isfinite(evals_desc) & (evals_desc > 0)]
    if evals_desc.size < 8:
        return {
            "k_mp": 0,
            "n_bulk": int(evals_desc.size),
            "q_fit": float("nan"),
            "n_eff_fit": float("nan"),
            "sigma2_fit": float("nan"),
            "lambda_minus": float("nan"),
            "lambda_plus": float("nan"),
            "top_over_edge": float("nan"),
            "fit_err": float("nan"),
        }

    n_det = int(evals_desc.size)
    rel_floor = max(float(evals_desc[0]) * 1e-10, np.finfo(float).tiny)
    positive_bulk = evals_desc[evals_desc > rel_floor]
    if positive_bulk.size < 8:
        positive_bulk = evals_desc

    n_bulk = max(6, int(math.floor(positive_bulk.size * bulk_keep_frac)))
    n_bulk = min(n_bulk, positive_bulk.size)
    bulk = np.sort(positive_bulk)[0:n_bulk]
    emp_q10, emp_q50, emp_q90 = (float(np.quantile(bulk, p)) for p in MP_FIT_PROBS)
    if not (np.isfinite(emp_q10) and np.isfinite(emp_q50) and np.isfinite(emp_q90) and emp_q10 > 0 and emp_q50 > 0 and emp_q90 > 0):
        return {
            "k_mp": 0,
            "n_bulk": int(n_bulk),
            "q_fit": float("nan"),
            "n_eff_fit": float("nan"),
            "sigma2_fit": float("nan"),
            "lambda_minus": float("nan"),
            "lambda_plus": float("nan"),
            "top_over_edge": float("nan"),
            "fit_err": float("nan"),
        }

    q_min = max(n_det / max(n_time, 1), 1e-3)
    q_max = max(4.0 * q_min, 8.0)
    if q_max / max(q_min, 1e-6) > 2.0:
        q_candidates = np.geomspace(q_min, q_max, max(q_grid_size, 8), dtype=float)
    else:
        q_candidates = np.linspace(q_min, q_max, max(q_grid_size, 8), dtype=float)

    best: dict[str, float] | None = None
    for q in q_candidates:
        mp_q10, mp_q50, mp_q90 = _mp_quantiles(round(float(q), 5))
        if mp_q50 <= 0 or mp_q10 <= 0 or mp_q90 <= 0:
            continue
        sigma2 = emp_q50 / mp_q50
        pred_q10 = sigma2 * mp_q10
        pred_q90 = sigma2 * mp_q90
        err = (math.log(pred_q10 / emp_q10) ** 2) + (math.log(pred_q90 / emp_q90) ** 2)
        cur = {
            "q_fit": float(q),
            "n_eff_fit": float(n_det / q),
            "sigma2_fit": float(sigma2),
            "lambda_minus": float(sigma2 * (1.0 - math.sqrt(q)) ** 2),
            "lambda_plus": float(sigma2 * (1.0 + math.sqrt(q)) ** 2),
            "fit_err": float(err),
        }
        if best is None or cur["fit_err"] < best["fit_err"]:
            best = cur

    if best is None:
        best = {
            "q_fit": float("nan"),
            "n_eff_fit": float("nan"),
            "sigma2_fit": float("nan"),
            "lambda_minus": float("nan"),
            "lambda_plus": float("nan"),
            "fit_err": float("nan"),
        }

    lam_plus = best["lambda_plus"]
    if not np.isfinite(lam_plus) or lam_plus <= 0:
        k_mp = 0
        top_over_edge = float("nan")
    else:
        k_mp = int(np.sum(evals_desc > lam_plus))
        top_over_edge = float(evals_desc[0] / lam_plus)

    best.update(
        {
            "k_mp": int(k_mp),
            "n_bulk": int(n_bulk),
            "top_over_edge": float(top_over_edge),
            "max_eval": float(evals_desc[0]),
            "median_eval": float(np.median(evals_desc)),
        }
    )
    return best


def _write_csv(path: Path, rows: list[dict[str, object]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _nanmedian(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _summarize_by_network(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(int(row["network"]), []).append(row)

    out: list[dict[str, object]] = []
    for nw in sorted(buckets):
        rr = buckets[nw]
        out.append(
            {
                "network": nw,
                "n_rows": len(rr),
                "median_k_mp": _nanmedian([float(r["k_mp"]) for r in rr]),
                "max_k_mp": max(int(r["k_mp"]) for r in rr),
                "median_k_frac": _nanmedian([float(r["k_frac"]) for r in rr]),
                "median_lambda_plus": _nanmedian([float(r["lambda_plus"]) for r in rr]),
                "median_q_fit": _nanmedian([float(r["q_fit"]) for r in rr]),
                "median_n_eff_fit": _nanmedian([float(r["n_eff_fit"]) for r in rr]),
                "median_top_over_edge": _nanmedian([float(r["top_over_edge"]) for r in rr]),
                "median_fit_err": _nanmedian([float(r["fit_err"]) for r in rr]),
                "median_valid_frac": _nanmedian([float(r["valid_frac"]) for r in rr]),
            }
        )
    return out


def _write_report(
    outpath: Path,
    nc_file: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    *,
    array: str,
    networks: list[int],
    band_low_hz: float,
    band_high_hz: float,
    configured_k: int,
) -> None:
    with outpath.open("w") as f:
        f.write(f"# MP Mode Estimate: {nc_file.name}\n\n")
        f.write(f"- Input file: `{nc_file}`\n")
        f.write(f"- Array selection: `{array}`\n")
        f.write(f"- Networks: `{','.join(str(nw) for nw in networks)}`\n")
        if band_low_hz > 0 or band_high_hz > 0:
            f.write(f"- Band-pass for covariance: `{band_low_hz:.3f} - {band_high_hz:.3f} Hz`\n")
        else:
            f.write("- Band-pass for covariance: `full band`\n")
        if configured_k >= 0:
            f.write(f"- Configured comparison cut: `{configured_k}`\n")
        f.write(f"- Rows analyzed: `{len(rows)}`\n\n")

        f.write("## Network Summary\n\n")
        f.write("| nw | n_rows | med k_mp | max k_mp | med k_frac | med q_fit | med n_eff | med top/edge |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            f.write(
                f"| {row['network']} | {row['n_rows']} | "
                f"{row['median_k_mp']:.2f} | {row['max_k_mp']} | "
                f"{row['median_k_frac']:.3f} | {row['median_q_fit']:.3f} | "
                f"{row['median_n_eff_fit']:.1f} | {row['median_top_over_edge']:.2f} |\n"
            )

        ranked = sorted(
            rows,
            key=lambda r: (float(r["k_mp"]), float(r["top_over_edge"])),
            reverse=True,
        )[:8]
        f.write("\n## Highest MP Mode Counts\n\n")
        for row in ranked:
            msg = (
                f"- scan={row['scan']} output_scan={row['output_scan']} nw={row['network']} "
                f"k_mp={row['k_mp']} lambda_plus={float(row['lambda_plus']):.3f} "
                f"top/edge={float(row['top_over_edge']):.2f}"
            )
            if configured_k >= 0:
                msg += f" delta_vs_cfg={int(row['k_mp']) - configured_k:+d}"
            f.write(msg + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc-file", required=True, help="Path to RTC/PTC timestream netCDF")
    ap.add_argument("--obsnum", help="Obsnum label override")
    ap.add_argument("--array", default="all", choices=["all", "a1100", "a1400", "a2000"])
    ap.add_argument("--scans", default="all", help="Comma list of internal scan indices or 'all'")
    ap.add_argument("--networks", default="all", help="Comma list of network IDs or 'all'")
    ap.add_argument("--min-good-frac", type=float, default=0.9)
    ap.add_argument("--max-det", type=int, default=0, help="Cap detectors per row; 0 uses all")
    ap.add_argument("--max-time", type=int, default=0, help="Cap time samples per row; 0 uses all")
    ap.add_argument("--band-low-hz", type=float, default=0.0)
    ap.add_argument("--band-high-hz", type=float, default=0.0)
    ap.add_argument("--clip-z", type=float, default=12.0)
    ap.add_argument("--bulk-keep-frac", type=float, default=0.8, help="Fraction of smallest eigenvalues treated as MP bulk")
    ap.add_argument("--q-grid-size", type=int, default=64)
    ap.add_argument("--configured-k", type=int, default=-1, help="Configured PCA cut for comparison; negative disables")
    ap.add_argument("--outdir", help="Default: <nc parent>/mp_mode_estimate")
    args = ap.parse_args()

    nc_file = Path(args.nc_file).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else nc_file.parent / "mp_mode_estimate"
    outdir.mkdir(parents=True, exist_ok=True)

    array_name_to_id = {"a1100": 0, "a1400": 1, "a2000": 2}
    obsnum = _resolve_obsnum(nc_file, args.obsnum)

    with netCDF4.Dataset(nc_file) as ds:
        signal = np.asarray(ds.variables["signal"][:], dtype=float)
        flags = np.asarray(ds.variables["flags"][:]) != 0
        apt_nw = np.asarray(ds.variables["apt_nw"][:], dtype=int)
        apt_flag = np.asarray(ds.variables["apt_flag"][:], dtype=int)
        apt_array = np.asarray(ds.variables["apt_array"][:], dtype=int)
        scan_indices = np.asarray(ds.variables["scan_indices"][:], dtype=int) if "scan_indices" in ds.variables else None
        output_scan_index = (
            np.asarray(ds.variables["output_scan_index"][:], dtype=int)
            if "output_scan_index" in ds.variables
            else None
        )

        if scan_indices is None or len(scan_indices) == 0:
            scan_ranges = [(0, 0, signal.shape[0] - 1)]
        else:
            scan_ranges = [(i, int(s[0]), int(s[1])) for i, s in enumerate(scan_indices)]
        selected_scans = _parse_scans(args.scans, len(scan_ranges))
        selected_networks = _parse_networks(args.networks, apt_nw)

        rows: list[dict[str, object]] = []
        for scan_id in selected_scans:
            _, i0, i1 = scan_ranges[scan_id]
            output_scan = int(output_scan_index[scan_id]) if output_scan_index is not None else scan_id
            dt_sec = _infer_dt_sec(ds, i0, i1)

            sig_s = signal[i0 : i1 + 1, :]
            flg_s = flags[i0 : i1 + 1, :]

            for nw in selected_networks:
                det_mask = (apt_nw == nw) & (apt_flag == 0)
                if args.array != "all":
                    det_mask &= apt_array == array_name_to_id[args.array]
                if np.sum(det_mask) < 6:
                    continue

                arr_vals = apt_array[det_mask]
                array_id = int(np.bincount(arr_vals).argmax()) if arr_vals.size else -1

                prep = _prepare_detector_matrix(
                    sig_s[:, det_mask],
                    flg_s[:, det_mask],
                    dt_sec,
                    min_good_frac=float(args.min_good_frac),
                    max_det=int(args.max_det),
                    max_time=int(args.max_time),
                    band_low_hz=float(args.band_low_hz),
                    band_high_hz=float(args.band_high_hz),
                    clip_z=float(args.clip_z),
                )
                if prep is None:
                    continue

                z = np.asarray(prep["z"], dtype=float)
                evals = _cov_eigenvalues(z)
                fit = _fit_mp_bulk(
                    evals,
                    int(prep["n_time"]),
                    bulk_keep_frac=float(args.bulk_keep_frac),
                    q_grid_size=int(args.q_grid_size),
                )
                rows.append(
                    {
                        "obsnum": obsnum,
                        "scan": int(scan_id),
                        "output_scan": int(output_scan),
                        "network": int(nw),
                        "array": int(array_id),
                        "n_det": int(prep["n_det"]),
                        "n_time": int(prep["n_time"]),
                        "dt_sec": float(prep["dt_sec"]),
                        "valid_frac": float(prep["valid_frac"]),
                        "time_stride": int(prep["time_stride"]),
                        "k_mp": int(fit["k_mp"]),
                        "k_frac": float(fit["k_mp"] / max(int(prep["n_det"]), 1)),
                        "n_bulk": int(fit["n_bulk"]),
                        "q_fit": float(fit["q_fit"]),
                        "n_eff_fit": float(fit["n_eff_fit"]),
                        "sigma2_fit": float(fit["sigma2_fit"]),
                        "lambda_minus": float(fit["lambda_minus"]),
                        "lambda_plus": float(fit["lambda_plus"]),
                        "max_eval": float(fit["max_eval"]),
                        "median_eval": float(fit["median_eval"]),
                        "top_over_edge": float(fit["top_over_edge"]),
                        "fit_err": float(fit["fit_err"]),
                        "configured_k": int(args.configured_k),
                        "delta_vs_configured_k": int(fit["k_mp"] - args.configured_k) if args.configured_k >= 0 else "",
                        "band_low_hz": float(args.band_low_hz),
                        "band_high_hz": float(args.band_high_hz),
                    }
                )

    detailed_cols = [
        "obsnum",
        "scan",
        "output_scan",
        "network",
        "array",
        "n_det",
        "n_time",
        "dt_sec",
        "valid_frac",
        "time_stride",
        "k_mp",
        "k_frac",
        "n_bulk",
        "q_fit",
        "n_eff_fit",
        "sigma2_fit",
        "lambda_minus",
        "lambda_plus",
        "max_eval",
        "median_eval",
        "top_over_edge",
        "fit_err",
        "configured_k",
        "delta_vs_configured_k",
        "band_low_hz",
        "band_high_hz",
    ]
    summary_rows = _summarize_by_network(rows)
    summary_cols = [
        "network",
        "n_rows",
        "median_k_mp",
        "max_k_mp",
        "median_k_frac",
        "median_lambda_plus",
        "median_q_fit",
        "median_n_eff_fit",
        "median_top_over_edge",
        "median_fit_err",
        "median_valid_frac",
    ]

    detailed_csv = outdir / "mp_mode_estimate_detailed.csv"
    summary_csv = outdir / "mp_mode_estimate_summary_by_network.csv"
    report_md = outdir / "MP_MODE_ESTIMATE.md"
    _write_csv(detailed_csv, rows, detailed_cols)
    _write_csv(summary_csv, summary_rows, summary_cols)
    _write_report(
        report_md,
        nc_file,
        rows,
        summary_rows,
        array=args.array,
        networks=selected_networks,
        band_low_hz=float(args.band_low_hz),
        band_high_hz=float(args.band_high_hz),
        configured_k=int(args.configured_k),
    )

    print(f"Wrote {detailed_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
