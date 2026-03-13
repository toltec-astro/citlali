#!/usr/bin/env python3
"""Audit cleaned blank-sky timestreams against a Gaussian null model.

The intended use case is a cleaned Citlali timestream product, usually
`*_ptc_timestream.nc`, from a field where per-sample sky signal is negligible.
Under that null, residuals should be approximately independent Gaussian noise.

This script writes:
- a detailed per-scan, per-network CSV
- a summary CSV aggregated by network
- a short markdown report highlighting the strongest null failures
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import netCDF4
import numpy as np


TAIL_THRESHOLDS = (3.0, 4.0, 5.0)
COMMON_MODE_BANDS = (
    ("low", 0.05, 0.5),
    ("mid", 0.5, 2.0),
    ("high", 2.0, 10.0),
)


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
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


def _downsample_indices(n: int, max_n: int | None) -> np.ndarray:
    if max_n is None or max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, int(max_n), dtype=int)


def _downsample_time_indices(n: int, max_n: int | None) -> tuple[np.ndarray, int]:
    if max_n is None or max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=int), 1
    stride = int(math.ceil(n / max_n))
    idx = np.arange(0, n, stride, dtype=int)
    if idx.size > max_n:
        idx = idx[:max_n]
    return idx, stride


def _resolve_obsnum(nc_file: Path, fallback: str | None) -> str:
    if fallback:
        return str(fallback)
    parts = [tok for tok in nc_file.name.split("_") if tok.isdigit()]
    return parts[0] if parts else "unknown"


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def _nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _nanstd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 8:
        return float("nan")
    x = np.asarray(a[valid], dtype=float)
    y = np.asarray(b[valid], dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if den <= 0:
        return float("nan")
    return float(abs(np.sum(x * y) / den))


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


def _prepare_detector_matrix(
    signal: np.ndarray,
    flags: np.ndarray,
    min_good_frac: float,
    max_det: int | None,
    max_time: int | None,
    clip_z: float,
) -> dict[str, np.ndarray | float | int] | None:
    signal = np.asarray(signal, dtype=float)
    flags = np.asarray(flags)
    if signal.ndim == 1:
        signal = signal[:, None]
        flags = flags[:, None]
    if signal.ndim != 2 or signal.shape[1] < 2:
        return None

    t_idx, stride = _downsample_time_indices(signal.shape[0], max_time)
    signal = signal[t_idx, :]
    flags = flags[t_idx, :]

    valid = np.isfinite(signal) & (flags == 0)
    det_good = np.mean(valid, axis=0) >= float(min_good_frac)
    det_idx = np.where(det_good)[0]
    if det_idx.size < 6:
        return None
    det_idx = det_idx[_downsample_indices(det_idx.size, max_det)]
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
    z = x_centered / scales_arr[None, :]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if np.isfinite(clip_z) and clip_z > 0:
        z = np.clip(z, -clip_z, clip_z)

    return {
        "x_centered": x_centered,
        "z": z,
        "valid": valid,
        "n_time": int(z.shape[0]),
        "n_det": int(z.shape[1]),
        "valid_frac": float(np.mean(valid)),
        "time_stride": int(stride),
        "scales": scales_arr,
    }


def _sample_pair_corr_metrics(z: np.ndarray, n_pairs: int, rng: np.random.Generator) -> tuple[float, float]:
    n_t, n_d = z.shape
    if n_t < 8 or n_d < 2:
        return float("nan"), float("nan")
    n_possible = n_d * (n_d - 1) // 2
    n_use = min(max(1, n_pairs), n_possible)
    i = rng.integers(0, n_d, size=n_use * 2, endpoint=False)
    j = rng.integers(0, n_d, size=n_use * 2, endpoint=False)
    keep = i < j
    i = i[keep][:n_use]
    j = j[keep][:n_use]
    if i.size == 0:
        return float("nan"), float("nan")
    dots = np.sum(z[:, i] * z[:, j], axis=0) / max(n_t - 1, 1)
    abs_corr = np.abs(dots)
    return float(np.median(abs_corr)), float(np.percentile(abs_corr, 95.0))


def _eigen_metrics(z: np.ndarray) -> tuple[float, float]:
    n_t, n_d = z.shape
    if n_t < 8 or n_d < 2:
        return float("nan"), float("nan")
    zz = np.nan_to_num(np.asarray(z, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cov = (zz.T @ zz) / max(n_t - 1, 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    evals = evals[np.isfinite(evals) & (evals > 0)]
    if evals.size == 0:
        return float("nan"), float("nan")
    frac = evals / np.sum(evals)
    top_mode_frac = float(frac[0])
    k90 = float(np.searchsorted(np.cumsum(frac), 0.90) + 1)
    return top_mode_frac, k90


def _gaussian_tail_prob(threshold: float) -> float:
    return float(math.erfc(threshold / math.sqrt(2.0)))


def _tail_metrics(z_valid: np.ndarray, threshold: float) -> dict[str, float]:
    n = int(z_valid.size)
    if n < 1:
        return {
            "frac": float("nan"),
            "count": float("nan"),
            "expected_frac": _gaussian_tail_prob(threshold),
            "expected_count": float("nan"),
            "ratio": float("nan"),
            "binom_z": float("nan"),
        }
    abs_z = np.abs(z_valid)
    count = int(np.sum(abs_z > threshold))
    frac = float(count / n)
    expected_frac = _gaussian_tail_prob(threshold)
    expected_count = expected_frac * n
    var = n * expected_frac * max(1.0 - expected_frac, 0.0)
    binom_z = float((count - expected_count) / math.sqrt(var)) if var > 0 else float("nan")
    ratio = _safe_ratio(frac, expected_frac)
    return {
        "frac": frac,
        "count": float(count),
        "expected_frac": expected_frac,
        "expected_count": expected_count,
        "ratio": ratio,
        "binom_z": binom_z,
    }


def _shape_metrics(z_valid: np.ndarray) -> tuple[float, float]:
    if z_valid.size < 8:
        return float("nan"), float("nan")
    mean = float(np.mean(z_valid))
    centered = z_valid - mean
    var = float(np.mean(centered * centered))
    if not np.isfinite(var) or var <= 0:
        return float("nan"), float("nan")
    skew = float(np.mean(centered ** 3) / (var ** 1.5))
    kurt = float(np.mean(centered ** 4) / (var ** 2) - 3.0)
    return skew, kurt


def _quartile_metrics(z: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    n_t = z.shape[0]
    if n_t < 16:
        return float("nan"), float("nan")
    q_edges = np.linspace(0, n_t, 5, dtype=int)
    q_sigmas: list[float] = []
    q_tail4: list[float] = []
    for q0, q1 in zip(q_edges[:-1], q_edges[1:]):
        vals = z[q0:q1, :][valid[q0:q1, :]]
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 8:
            continue
        q_sigmas.append(float(np.std(vals, ddof=1)))
        q_tail4.append(float(np.mean(np.abs(vals) > 4.0)))
    if len(q_sigmas) < 2 or len(q_tail4) < 2:
        return float("nan"), float("nan")
    sigma_ratio = _safe_ratio(max(q_sigmas), min(q_sigmas))
    tail4_range = float(max(q_tail4) - min(q_tail4))
    return sigma_ratio, tail4_range


def _get_scan_templates(ds: netCDF4.Dataset, scan_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tel_el = np.asarray(ds.variables["TelElAct"][scan_idx], dtype=float)
    tel_az = np.unwrap(np.asarray(ds.variables["TelAzAct"][scan_idx], dtype=float))
    d_el = np.gradient(tel_el)
    d_az = np.gradient(tel_az)
    t = np.linspace(-1.0, 1.0, scan_idx.size, dtype=float)
    return tel_el, tel_az, d_el, d_az, t


def _common_mode_spectrum_metrics(common_mode: np.ndarray, fs: float) -> dict[str, float]:
    common_mode = np.asarray(common_mode, dtype=float)
    common_mode = common_mode[np.isfinite(common_mode)]
    if common_mode.size < 16 or not np.isfinite(fs) or fs <= 0:
        return {
            "cm_bp_low": float("nan"),
            "cm_bp_mid": float("nan"),
            "cm_bp_high": float("nan"),
            "cm_low_mid_ratio": float("nan"),
            "cm_high_mid_ratio": float("nan"),
            "cm_peak_freq_hz": float("nan"),
            "cm_peak_prominence": float("nan"),
        }
    x = common_mode - np.mean(common_mode)
    win = np.hanning(x.size)
    spec = np.fft.rfft(x * win)
    power = np.abs(spec) ** 2
    freq = np.fft.rfftfreq(x.size, d=1.0 / fs)

    band_power: dict[str, float] = {}
    for name, f0, f1 in COMMON_MODE_BANDS:
        mask = (freq >= f0) & (freq < f1)
        if not np.any(mask):
            band_power[name] = float("nan")
        else:
            band_power[name] = float(np.median(power[mask]))

    peak_mask = (freq >= 0.05) & (freq <= min(16.0, float(np.max(freq))))
    if np.any(peak_mask):
        local_power = power[peak_mask]
        local_freq = freq[peak_mask]
        idx = int(np.argmax(local_power))
        peak_power = float(local_power[idx])
        peak_freq = float(local_freq[idx])
        peak_prominence = _safe_ratio(peak_power, float(np.median(local_power)))
    else:
        peak_freq = float("nan")
        peak_prominence = float("nan")

    return {
        "cm_bp_low": band_power["low"],
        "cm_bp_mid": band_power["mid"],
        "cm_bp_high": band_power["high"],
        "cm_low_mid_ratio": _safe_ratio(band_power["low"], band_power["mid"]),
        "cm_high_mid_ratio": _safe_ratio(band_power["high"], band_power["mid"]),
        "cm_peak_freq_hz": peak_freq,
        "cm_peak_prominence": peak_prominence,
    }


def _surrogate_metrics(
    z: np.ndarray,
    n_pairs: int,
    n_surrogates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_t, n_d = z.shape
    if n_surrogates <= 0 or n_t < 8 or n_d < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    med_abs_corr = np.zeros(n_surrogates, dtype=float)
    top_mode = np.zeros(n_surrogates, dtype=float)
    for i in range(n_surrogates):
        zs = np.empty_like(z)
        shifts = rng.integers(0, n_t, size=n_d, endpoint=False)
        for j in range(n_d):
            zs[:, j] = np.roll(z[:, j], int(shifts[j]))
        med_abs_corr[i], _ = _sample_pair_corr_metrics(zs, n_pairs, rng)
        top_mode[i], _ = _eigen_metrics(zs)
    return med_abs_corr, top_mode


def _row_value(row: dict[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols = sorted(set().union(*[row.keys() for row in rows]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    networks = sorted(set(int(row["network"]) for row in rows))
    summary: list[dict[str, object]] = []
    for nw in networks:
        rr = [row for row in rows if int(row["network"]) == nw]

        def med(key: str) -> float:
            vals = np.asarray([_row_value(row, key) for row in rr], dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(np.median(vals)) if vals.size else float("nan")

        def maxv(key: str) -> float:
            vals = np.asarray([_row_value(row, key) for row in rr], dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(np.max(vals)) if vals.size else float("nan")

        summary.append(
            {
                "network": nw,
                "n_rows": len(rr),
                "median_valid_frac": med("valid_frac"),
                "median_tail4_binom_z": med("tail4_binom_z"),
                "max_tail4_binom_z": maxv("tail4_binom_z"),
                "median_excess_kurtosis": med("z_excess_kurtosis"),
                "median_med_abs_corr": med("med_abs_corr"),
                "median_med_abs_corr_surr_z": med("med_abs_corr_surr_z"),
                "median_top_mode_frac": med("top_mode_frac"),
                "median_top_mode_surr_z": med("top_mode_surr_z"),
                "median_cm_low_mid_ratio": med("cm_low_mid_ratio"),
                "median_cm_peak_prominence": med("cm_peak_prominence"),
                "median_quartile_sigma_ratio": med("quartile_sigma_ratio"),
                "median_cm_corr_t": med("cm_corr_t"),
                "max_cm_peak_prominence": maxv("cm_peak_prominence"),
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


def _write_report(
    outpath: Path,
    nc_file: Path,
    array: str,
    networks: list[int],
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    n_surrogates: int,
) -> None:
    top_tail = _top_rows(rows, "tail4_binom_z")
    top_corr = _top_rows(rows, "med_abs_corr_surr_z")
    top_low = _top_rows(rows, "cm_low_mid_ratio")

    lines = [
        f"# Blank-Sky Null Audit: {nc_file.name}",
        "",
        f"- Input file: `{nc_file}`",
        f"- Array selection: `{array}`",
        f"- Networks: `{','.join(str(nw) for nw in networks)}`",
        f"- Rows analyzed: {len(rows)}",
        f"- Surrogates per row: {n_surrogates}",
        "",
        "## Network Summary",
        "",
        "| nw | n_rows | med tail4 z | max tail4 z | med kurt | med corr surr z | med top-mode surr z | med low/mid | med line prom |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {network} | {n_rows} | {median_tail4_binom_z:.2f} | {max_tail4_binom_z:.2f} | "
            "{median_excess_kurtosis:.2f} | {median_med_abs_corr_surr_z:.2f} | "
            "{median_top_mode_surr_z:.2f} | {median_cm_low_mid_ratio:.2f} | "
            "{median_cm_peak_prominence:.2f} |".format(**row)
        )

    def emit_rows(title: str, subset: list[dict[str, object]], key: str) -> None:
        lines.extend(["", f"## {title}", ""])
        if not subset:
            lines.append("- none")
            return
        for row in subset:
            lines.append(
                "- scan={scan} output_scan={output_scan_index} nw={network} {key}={value:.2f}".format(
                    scan=row["scan"],
                    output_scan_index=row["output_scan_index"],
                    network=row["network"],
                    key=key,
                    value=_row_value(row, key),
                )
            )

    emit_rows("Top Tail-Excess Rows", top_tail, "tail4_binom_z")
    emit_rows("Top Coherence-Null Failures", top_corr, "med_abs_corr_surr_z")
    emit_rows("Top Low-Frequency Leakage Rows", top_low, "cm_low_mid_ratio")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Large positive tail/kurtosis values suggest impulsive or non-Gaussian contamination.",
            "- Large surrogate-null deviations suggest coherent detector contamination beyond each detector's own PSD.",
            "- Large low/mid band ratios or common-mode template coupling suggest low-frequency or scan-synchronous leakage.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc-file", required=True, help="Path to cleaned Citlali timestream file, usually *_ptc_timestream.nc")
    ap.add_argument("--obsnum", default=None, help="Obsnum label override")
    ap.add_argument("--array", default="all", choices=["all", "a1100", "a1400", "a2000"])
    ap.add_argument("--scans", default="all", help="Comma list of internal scan indices or 'all'")
    ap.add_argument("--networks", default="all", help="Comma list of network IDs or 'all'")
    ap.add_argument("--utils-root", default="~/GitHub/toltec-data-product-utilities",
                    help="Path that contains toltec_dp_utils")
    ap.add_argument("--min-good-frac", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=180, help="Cap detectors per scan/network")
    ap.add_argument("--max-time", type=int, default=2048, help="Cap time samples per scan/network")
    ap.add_argument("--n-pairs", type=int, default=4000, help="Sampled detector pairs for corr metrics")
    ap.add_argument("--n-surrogates", type=int, default=8, help="Circular-shift surrogates per row")
    ap.add_argument("--clip-z", type=float, default=50.0, help="Clip standardized residuals before coherence metrics")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=None, help="Default: <nc parent>/blank_sky_null_audit")
    args = ap.parse_args()

    nc_file = Path(os.path.expanduser(args.nc_file)).resolve()
    if not nc_file.exists():
        raise FileNotFoundError(f"missing nc file: {nc_file}")
    outdir = Path(os.path.expanduser(args.outdir)).resolve() if args.outdir else (nc_file.parent / "blank_sky_null_audit")
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
        fs_native = float(tcs.sampleRate)

        rows: list[dict[str, object]] = []

        for scan in scans:
            scan_idx = tcs.getScanIndices(scan)
            tel_el, tel_az, d_el, d_az, t = _get_scan_templates(ds, scan_idx)

            for nw in networks:
                signal, _ = tcs._get_signal(scan_idx, network=nw)  # noqa: SLF001
                flags, _ = tcs._get_flags(scan_idx, network=nw)  # noqa: SLF001
                prepared = _prepare_detector_matrix(
                    signal=signal,
                    flags=flags,
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
                z_valid = z[valid]
                if z_valid.size < 32:
                    continue

                med_abs_corr, p95_abs_corr = _sample_pair_corr_metrics(z, int(args.n_pairs), rng)
                top_mode_frac, k90 = _eigen_metrics(z)
                surr_corr, surr_top = _surrogate_metrics(z, int(args.n_pairs), int(args.n_surrogates), rng)

                common_mode = np.median(x_centered, axis=1)
                fs_eff = fs_native / max(int(prepared["time_stride"]), 1)
                spec = _common_mode_spectrum_metrics(common_mode, fs_eff)
                quartile_sigma_ratio, quartile_tail4_range = _quartile_metrics(z, valid)
                skew, excess_kurtosis = _shape_metrics(z_valid)
                tail3 = _tail_metrics(z_valid, 3.0)
                tail4 = _tail_metrics(z_valid, 4.0)
                tail5 = _tail_metrics(z_valid, 5.0)
                pos4 = int(np.sum(z_valid > 4.0))
                neg4 = int(np.sum(z_valid < -4.0))
                pn_den = pos4 + neg4
                tail4_pos_neg_asym = float((pos4 - neg4) / pn_den) if pn_den > 0 else float("nan")

                rows.append(
                    {
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
                        "tail3_frac": tail3["frac"],
                        "tail3_ratio": tail3["ratio"],
                        "tail3_binom_z": tail3["binom_z"],
                        "tail4_frac": tail4["frac"],
                        "tail4_ratio": tail4["ratio"],
                        "tail4_binom_z": tail4["binom_z"],
                        "tail5_frac": tail5["frac"],
                        "tail5_ratio": tail5["ratio"],
                        "tail5_binom_z": tail5["binom_z"],
                        "tail4_pos_neg_asym": tail4_pos_neg_asym,
                        "med_abs_corr": med_abs_corr,
                        "p95_abs_corr": p95_abs_corr,
                        "top_mode_frac": top_mode_frac,
                        "k90_corr_modes": k90,
                        "med_abs_corr_surr_median": _nanmedian(surr_corr),
                        "med_abs_corr_surr_std": _nanstd(surr_corr),
                        "med_abs_corr_over_surr_median": _safe_ratio(med_abs_corr, _nanmedian(surr_corr)),
                        "med_abs_corr_surr_z": _safe_ratio(med_abs_corr - _nanmedian(surr_corr), _nanstd(surr_corr)),
                        "top_mode_surr_median": _nanmedian(surr_top),
                        "top_mode_surr_std": _nanstd(surr_top),
                        "top_mode_over_surr_median": _safe_ratio(top_mode_frac, _nanmedian(surr_top)),
                        "top_mode_surr_z": _safe_ratio(top_mode_frac - _nanmedian(surr_top), _nanstd(surr_top)),
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
                        "quartile_sigma_ratio": quartile_sigma_ratio,
                        "quartile_tail4_range": quartile_tail4_range,
                    }
                )

    if not rows:
        raise RuntimeError(f"no rows generated for {nc_file}")

    detailed_csv = outdir / "blank_sky_null_audit_detailed.csv"
    _write_csv(detailed_csv, rows)

    summary_rows = _make_summary_rows(rows)
    summary_csv = outdir / "blank_sky_null_audit_summary_by_network.csv"
    _write_csv(summary_csv, summary_rows)

    report_md = outdir / "BLANK_SKY_NULL_AUDIT.md"
    _write_report(report_md, nc_file, args.array, networks, rows, summary_rows, int(args.n_surrogates))

    print(f"Wrote {detailed_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
