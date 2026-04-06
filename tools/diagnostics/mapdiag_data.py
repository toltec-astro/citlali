#!/usr/bin/env python3
"""Shared mapdiag loading helpers for engineering plots and reports."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
from astropy.io import fits


def _parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def filled(var: netCDF4.Variable, fill: float | int | None = None) -> np.ndarray:
    data = var[:]
    if np.ma.isMaskedArray(data):
        if fill is None:
            dtype = np.asarray(data).dtype
            if np.issubdtype(dtype, np.floating):
                fill = float("nan")
            else:
                fill = np.iinfo(np.int32).min
        data = np.ma.filled(data, fill_value=fill)
    return np.asarray(data)


def _string_list(var: netCDF4.Variable) -> list[str]:
    data = var[:]
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data, fill_value="")
    arr = np.asarray(data, dtype=object).reshape(-1)
    return [str(item) for item in arr.tolist()]


def _scalar_string(ds: netCDF4.Dataset, name: str, default: str = "") -> str:
    var = ds.variables.get(name)
    if var is None:
        return default
    value = var[:]
    if np.ma.isMaskedArray(value):
        value = np.ma.filled(value, fill_value=default)
    arr = np.asarray(value, dtype=object).reshape(-1)
    return str(arr[0]) if arr.size else default


def collect_products(redu_dir: Path) -> list[Path]:
    return sorted(redu_dir.rglob("*_mapdiag*.nc"))


def _find_map_fits(source_nc: Path, array: str) -> Path | None:
    candidates = [
        path
        for path in sorted(source_nc.parent.glob(f"*{array}*_citlali.fits"))
        if "noise" not in path.name
    ]
    return candidates[0] if candidates else None


def _compute_core_peak_sig2noise(
    fits_path: Path | None,
    stokes: str,
    weight_threshold: float,
) -> float:
    if fits_path is None or not fits_path.is_file() or not np.isfinite(weight_threshold):
        return float("nan")
    signal_hdu = f"signal_{stokes}"
    weight_hdu = f"weight_{stokes}"
    try:
        with fits.open(fits_path) as hdul:
            signal = np.asarray(hdul[signal_hdu].data[0, 0], dtype=float)
            weight = np.asarray(hdul[weight_hdu].data[0, 0], dtype=float)
    except Exception:
        return float("nan")
    core_mask = np.isfinite(weight) & (weight > 0.0) & (weight >= weight_threshold)
    if not np.any(core_mask):
        return float("nan")
    sig2noise = np.abs(signal * np.sqrt(np.maximum(weight, 0.0)))
    core_sig2noise = np.where(core_mask, sig2noise, np.nan)
    if not np.isfinite(core_sig2noise).any():
        return float("nan")
    return float(np.nanmax(core_sig2noise))


def load_reduction_tables(
    redu_dir: Path,
    array: str,
    obsnums_spec: str = "all",
) -> dict[str, object]:
    redu_dir = Path(redu_dir).expanduser().resolve()
    if not redu_dir.is_dir():
        raise NotADirectoryError(redu_dir)

    obsnum_filter = None if obsnums_spec == "all" else set(str(v) for v in _parse_int_list(obsnums_spec))
    products = collect_products(redu_dir)
    if not products:
        raise FileNotFoundError(f"no mapdiag products found under {redu_dir}")

    map_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    used_files = 0

    for nc_file in products:
        fits_cache: dict[tuple[str, str], float] = {}
        with netCDF4.Dataset(nc_file) as ds:
            stage = _scalar_string(ds, "MAP_STAGE", default="unknown")
            map_regime = _scalar_string(ds, "MAP_REGIME", default="unknown")
            source_name = _scalar_string(ds, "SOURCE", default="")
            project_id = _scalar_string(ds, "PROJID", default="")
            obs_goal = _scalar_string(ds, "OBSGOAL", default="")
            array_names = _string_list(ds.variables["map_array_name"])
            stokes_names = _string_list(ds.variables["map_stokes"])
            map_names = _string_list(ds.variables["map_name"])
            obsnum_strings = _string_list(ds.variables["coadd_obsnum"])
            dateobs_strings = _string_list(ds.variables["coadd_dateobs"])
            file_obsnum = int(filled(ds.variables["obsnum"], fill=-2147483647).reshape(-1)[0])

            peak_s2n = filled(ds.variables["map_peak_abs_sig2noise"], fill=float("nan"))
            core_peak_s2n = filled(ds.variables["map_core_peak_abs_sig2noise"], fill=float("nan")) if "map_core_peak_abs_sig2noise" in ds.variables else None
            peak_signal = filled(ds.variables["map_peak_signal"], fill=float("nan"))
            median_err = filled(ds.variables["map_median_err"], fill=float("nan"))
            median_rms = filled(ds.variables["map_median_rms"], fill=float("nan"))
            empirical_to_formal = filled(ds.variables["map_empirical_to_formal_noise_ratio"], fill=float("nan")) if "map_empirical_to_formal_noise_ratio" in ds.variables else None
            weight_threshold = filled(ds.variables["map_weight_threshold"], fill=float("nan"))
            core_weight_sum = filled(ds.variables["map_core_weight_sum"], fill=float("nan"))
            n_core_pixels = filled(ds.variables["map_n_core_pixels"], fill=-2147483647)
            n_valid_pixels = filled(ds.variables["map_n_valid_pixels"], fill=-2147483647)
            coverage_median = filled(ds.variables["map_core_coverage_median"], fill=float("nan"))
            noise_rms_p16 = filled(ds.variables["map_noise_rms_p16"], fill=float("nan")) if "map_noise_rms_p16" in ds.variables else None
            noise_rms_p84 = filled(ds.variables["map_noise_rms_p84"], fill=float("nan")) if "map_noise_rms_p84" in ds.variables else None
            core_tail_excess_abs = filled(ds.variables["map_core_tail_excess_abs_gt3"], fill=float("nan")) if "map_core_tail_excess_abs_gt3" in ds.variables else None
            core_tail_excess_pos = filled(ds.variables["map_core_tail_excess_pos_gt3"], fill=float("nan")) if "map_core_tail_excess_pos_gt3" in ds.variables else None
            core_tail_excess_neg = filled(ds.variables["map_core_tail_excess_neg_lt3"], fill=float("nan")) if "map_core_tail_excess_neg_lt3" in ds.variables else None
            core_sig2noise_skew = filled(ds.variables["map_core_sig2noise_skew"], fill=float("nan")) if "map_core_sig2noise_skew" in ds.variables else None
            noise_tail_excess_abs = filled(ds.variables["map_noise_tail_excess_abs_gt3"], fill=float("nan")) if "map_noise_tail_excess_abs_gt3" in ds.variables else None
            noise_tail_excess_pos = filled(ds.variables["map_noise_tail_excess_pos_gt3"], fill=float("nan")) if "map_noise_tail_excess_pos_gt3" in ds.variables else None
            noise_tail_excess_neg = filled(ds.variables["map_noise_tail_excess_neg_lt3"], fill=float("nan")) if "map_noise_tail_excess_neg_lt3" in ds.variables else None
            noise_sig2noise_skew = filled(ds.variables["map_noise_sig2noise_skew"], fill=float("nan")) if "map_noise_sig2noise_skew" in ds.variables else None
            contrib_core_frac = filled(ds.variables["coadd_obs_core_weight_frac"], fill=float("nan"))
            contrib_weight_frac = filled(ds.variables["coadd_obs_weight_frac"], fill=float("nan"))
            contrib_core_sum = filled(ds.variables["coadd_obs_core_weight_sum"], fill=float("nan"))

            file_used = False
            for map_idx, map_array_name in enumerate(array_names):
                if map_array_name != array:
                    continue
                obs_context = "coadd" if file_obsnum < 0 else str(file_obsnum)
                if obsnum_filter is not None and obs_context != "coadd" and obs_context not in obsnum_filter:
                    continue
                file_used = True
                map_selector = f"{map_names[map_idx]} {stokes_names[map_idx]}".strip()
                fits_path = _find_map_fits(nc_file, array)
                threshold_value = float(weight_threshold[map_idx])
                cache_key = (stokes_names[map_idx], f"{threshold_value:.17g}")
                core_peak = fits_cache.get(cache_key)
                if core_peak is None:
                    core_peak = float("nan")
                    if core_peak_s2n is not None:
                        core_peak = float(core_peak_s2n[map_idx])
                    if not np.isfinite(core_peak):
                        core_peak = _compute_core_peak_sig2noise(fits_path, stokes_names[map_idx], threshold_value)
                    fits_cache[cache_key] = core_peak
                row = {
                    "source_file": str(nc_file),
                    "obs_context": obs_context,
                    "is_coadd": int(file_obsnum < 0),
                    "stage": stage,
                    "map_regime": map_regime,
                    "source_name": source_name,
                    "project_id": project_id,
                    "obs_goal": obs_goal,
                    "array": array,
                    "map_name": map_names[map_idx],
                    "stokes": stokes_names[map_idx],
                    "map_selector": map_selector,
                    "peak_abs_sig2noise": float(peak_s2n[map_idx]),
                    "core_peak_abs_sig2noise": float(core_peak),
                    "peak_signal": float(peak_signal[map_idx]),
                    "median_err": float(median_err[map_idx]),
                    "median_rms": float(median_rms[map_idx]),
                    "empirical_to_formal_noise_ratio": float(empirical_to_formal[map_idx]) if empirical_to_formal is not None else float("nan"),
                    "weight_threshold": float(weight_threshold[map_idx]),
                    "core_weight_sum": float(core_weight_sum[map_idx]),
                    "n_core_pixels": int(n_core_pixels[map_idx]),
                    "n_valid_pixels": int(n_valid_pixels[map_idx]),
                    "core_coverage_median": float(coverage_median[map_idx]),
                    "noise_rms_p16": float(noise_rms_p16[map_idx]) if noise_rms_p16 is not None else float("nan"),
                    "noise_rms_p84": float(noise_rms_p84[map_idx]) if noise_rms_p84 is not None else float("nan"),
                    "core_tail_excess_abs_gt3": float(core_tail_excess_abs[map_idx]) if core_tail_excess_abs is not None else float("nan"),
                    "core_tail_excess_pos_gt3": float(core_tail_excess_pos[map_idx]) if core_tail_excess_pos is not None else float("nan"),
                    "core_tail_excess_neg_lt3": float(core_tail_excess_neg[map_idx]) if core_tail_excess_neg is not None else float("nan"),
                    "core_sig2noise_skew": float(core_sig2noise_skew[map_idx]) if core_sig2noise_skew is not None else float("nan"),
                    "noise_tail_excess_abs_gt3": float(noise_tail_excess_abs[map_idx]) if noise_tail_excess_abs is not None else float("nan"),
                    "noise_tail_excess_pos_gt3": float(noise_tail_excess_pos[map_idx]) if noise_tail_excess_pos is not None else float("nan"),
                    "noise_tail_excess_neg_lt3": float(noise_tail_excess_neg[map_idx]) if noise_tail_excess_neg is not None else float("nan"),
                    "noise_sig2noise_skew": float(noise_sig2noise_skew[map_idx]) if noise_sig2noise_skew is not None else float("nan"),
                }
                map_rows.append(row)
                if file_obsnum < 0:
                    for obs_idx, contrib_obsnum in enumerate(obsnum_strings):
                        contribution_rows.append(
                            {
                                "source_file": str(nc_file),
                                "stage": stage,
                                "array": array,
                                "map_selector": map_selector,
                                "map_name": map_names[map_idx],
                                "stokes": stokes_names[map_idx],
                                "contrib_obsnum": str(contrib_obsnum),
                                "dateobs": dateobs_strings[obs_idx] if obs_idx < len(dateobs_strings) else "",
                                "core_weight_frac": float(contrib_core_frac[map_idx, obs_idx]),
                                "weight_frac": float(contrib_weight_frac[map_idx, obs_idx]),
                                "core_weight_sum": float(contrib_core_sum[map_idx, obs_idx]),
                            }
                        )
            if file_used:
                used_files += 1

    if not map_rows:
        raise FileNotFoundError(f"no usable mapdiag rows found for array={array} under {redu_dir}")

    return {
        "array": array,
        "n_mapdiag": used_files,
        "map_rows": map_rows,
        "contribution_rows": contribution_rows,
    }
