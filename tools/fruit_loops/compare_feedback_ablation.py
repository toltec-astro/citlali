#!/usr/bin/env python3
"""Compare saved pointing fruit-loop iterations across feedback ablations."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table

from tools.fruit_loops.compare_injected_source_pair import gaussian_fit


ARRAY_NAMES = {0: "a1100", 1: "a1400", 2: "a2000"}


def scalar_image(path: Path, extension: str) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        return np.asarray(hdul[extension].data, dtype=float).squeeze()


def rms(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return math.nan
    return float(np.sqrt(np.mean(np.square(values[finite]))))


def robust_background_sigma(
    values: np.ndarray, center_y: float, center_x: float,
    pixel_size_arcsec: float, inner_arcsec: float = 40.0,
    outer_arcsec: float = 100.0,
) -> float:
    rows, cols = np.indices(values.shape, dtype=float)
    radius_arcsec = np.hypot(
        rows - center_y, cols - center_x,
    ) * pixel_size_arcsec
    selected = (
        np.isfinite(values)
        & (radius_arcsec >= inner_arcsec)
        & (radius_arcsec <= outer_arcsec)
    )
    samples = values[selected]
    if samples.size < 10:
        samples = values[np.isfinite(values)]
    if samples.size == 0:
        return math.nan
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    return 1.4826 * mad


def iteration_config(reduction_dir: Path, fallback: Path | None = None) -> dict:
    candidates = sorted(
        path
        for path in reduction_dir.glob("citlali*.yaml")
        if path.name != "citlali_merged_config.yaml"
    )
    if not candidates and fallback is not None:
        candidates = [fallback]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one low-level config in {reduction_dir}, "
            f"found {[path.name for path in candidates]}"
        )
    config = yaml.safe_load(candidates[0].read_text())
    if not isinstance(config, dict):
        raise ValueError(f"invalid low-level config: {candidates[0]}")
    return config


def feedback_support_metrics(
    signal: np.ndarray, weight: np.ndarray, flux_limit: float,
    feedback_config: dict, pixel_size_arcsec: float,
) -> dict[str, float | int]:
    gain = np.ones_like(signal)
    if feedback_config.get("enabled", False):
        positive_weight = weight[np.isfinite(weight) & (weight > 0.0)]
        if positive_weight.size:
            reference_name = str(feedback_config.get("reference", "p95"))
            quantiles = {
                "median": 0.50,
                "p50": 0.50,
                "p90": 0.90,
                "p95": 0.95,
                "p99": 0.99,
            }
            if reference_name in {"max", "peak"}:
                reference = float(np.max(positive_weight))
            else:
                reference = float(
                    np.quantile(
                        positive_weight,
                        quantiles.get(reference_name, 0.95),
                        method="nearest",
                    )
                )
            low = float(feedback_config.get("low_relative_weight", 0.02))
            high = float(feedback_config.get("high_relative_weight", 0.10))
            gain = np.clip((weight / reference - low) / (high - low), 0.0, 1.0)
    selected = np.isfinite(signal) & (signal >= flux_limit)
    active = selected & np.isfinite(gain) & (gain > 0.0)
    if not active.any():
        return {
            "flux_selected_active_pixels": 0,
            "flux_selected_outside_40arcsec_fraction": math.nan,
            "flux_selected_signal_outside_40arcsec_fraction": math.nan,
            "flux_selected_taper_signal_loss_fraction": math.nan,
        }
    peak = np.unravel_index(np.nanargmax(signal), signal.shape)
    rows, cols = np.indices(signal.shape)
    radius_pixels = 40.0 / pixel_size_arcsec
    outside = active & (
        np.square(rows - peak[0]) + np.square(cols - peak[1])
        > radius_pixels**2
    )
    weighted_signal = signal * gain
    selected_signal_sum = float(np.sum(signal[selected]))
    active_signal_sum = float(np.sum(weighted_signal[active]))
    return {
        "flux_selected_active_pixels": int(active.sum()),
        "flux_selected_outside_40arcsec_fraction": float(
            outside.sum() / active.sum()
        ),
        "flux_selected_signal_outside_40arcsec_fraction": float(
            np.sum(weighted_signal[outside]) / active_signal_sum
        ),
        "flux_selected_taper_signal_loss_fraction": float(
            1.0 - active_signal_sum / selected_signal_sum
        ),
    }


def reduction_rows(label: str, reduced: Path, obsnum: int) -> list[dict]:
    rows: list[dict] = []
    previous_maps: dict[str, np.ndarray] = {}
    reduction_dirs = sorted(
        (
            path
            for path in reduced.glob("redu[0-9][0-9]*")
            if path.is_dir() and path.name[4:].isdigit()
        ),
        key=lambda path: int(path.name[4:]),
    )
    if not reduction_dirs:
        raise ValueError(f"no saved fruit-loop iterations found in {reduced}")
    iterations = [int(path.name[4:]) for path in reduction_dirs]
    expected = list(range(len(reduction_dirs)))
    if iterations != expected:
        raise ValueError(
            f"saved fruit-loop iterations must be contiguous from zero in "
            f"{reduced}; found {iterations}"
        )
    fallback_configs = [
        path
        for reduction_dir in reduction_dirs
        for path in reduction_dir.glob("citlali*.yaml")
        if path.name != "citlali_merged_config.yaml"
    ]
    fallback_config = fallback_configs[0] if fallback_configs else None
    for iteration, reduction_dir in zip(iterations, reduction_dirs):
        raw = reduction_dir / str(obsnum) / "raw"
        config = iteration_config(reduction_dir, fallback_config)
        fruit_config = config["timestream"]["fruit_loops"]
        flux_limits = fruit_config["array_flux_limit"]
        feedback_config = fruit_config.get("weight_feedback", {})
        pixel_size_arcsec = float(config["mapmaking"]["pixel_size_arcsec"])
        if not math.isfinite(pixel_size_arcsec) or pixel_size_arcsec <= 0.0:
            raise ValueError(
                f"invalid mapmaking.pixel_size_arcsec in {reduction_dir}"
            )
        table_path = raw / f"ppt_commissioning_pointing_{obsnum}_citlali.ecsv"
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        table = Table.read(table_path, format="ascii.ecsv")
        for fit in table:
            array_id = int(fit["array"])
            array_name = ARRAY_NAMES[array_id]
            map_path = (
                raw
                / f"toltec_commissioning_{array_name}_pointing_{obsnum}_citlali.fits"
            )
            signal = scalar_image(map_path, "signal_I")
            kernel = scalar_image(map_path, "kernel_I")
            weight = scalar_image(map_path, "weight_I")
            with fits.open(map_path, memmap=True) as hdul:
                absolute_iteration = int(
                    hdul[0].header.get("FRUITLOOPS_ITER", iteration)
                )
                source = str(hdul[0].header.get("SOURCE", "unknown"))
            finite_weight = weight[np.isfinite(weight) & (weight > 0.0)]
            delta = (
                signal - previous_maps[array_name]
                if array_name in previous_maps
                else np.full_like(signal, np.nan)
            )
            previous_rms = (
                rms(previous_maps[array_name])
                if array_name in previous_maps
                else math.nan
            )
            signal_rms = rms(signal)
            kernel_fit = gaussian_fit(kernel, pixel_size_arcsec)
            fitted_a_fwhm = abs(float(fit["a_fwhm"]))
            fitted_b_fwhm = abs(float(fit["b_fwhm"]))
            signal_peak = np.unravel_index(
                np.nanargmax(signal), signal.shape
            )
            background_sigma = robust_background_sigma(
                signal, float(signal_peak[0]), float(signal_peak[1]),
                pixel_size_arcsec,
            )
            kernel_amplitude = kernel_fit["amplitude"]
            amplitude = float(fit["amp"])
            amplitude_error = float(fit["amp_err"])
            legacy_peak_over_full_map_rms = float(fit["sig2noise"])
            fit_sig2noise = (
                amplitude / amplitude_error
                if math.isfinite(amplitude)
                and math.isfinite(amplitude_error)
                and amplitude_error > 0.0
                else math.nan
            )
            rows.append(
                {
                    "variant": label,
                    "obsnum": obsnum,
                    "source": source,
                    "iteration": absolute_iteration,
                    "array": array_name,
                    "amplitude": amplitude,
                    "amplitude_error": amplitude_error,
                    "a_fwhm_arcsec": fitted_a_fwhm,
                    "b_fwhm_arcsec": fitted_b_fwhm,
                    "major_fwhm_arcsec": max(
                        fitted_a_fwhm, fitted_b_fwhm
                    ),
                    "minor_fwhm_arcsec": min(
                        fitted_a_fwhm, fitted_b_fwhm
                    ),
                    # Retain the historical field for reproducibility. The
                    # pointing writer defines it as amplitude divided by the
                    # full-map standard deviation, so it is a dynamic-range
                    # diagnostic rather than statistical significance.
                    "sig2noise": legacy_peak_over_full_map_rms,
                    "legacy_peak_over_full_map_rms":
                        legacy_peak_over_full_map_rms,
                    "fit_sig2noise": fit_sig2noise,
                    "x_t_arcsec": float(fit["x_t"]),
                    "y_t_arcsec": float(fit["y_t"]),
                    "kernel_peak": float(np.nanmax(kernel)),
                    "kernel_fit_amplitude": kernel_amplitude,
                    "kernel_major_fwhm_arcsec":
                        kernel_fit["major_fwhm_arcsec"],
                    "kernel_minor_fwhm_arcsec":
                        kernel_fit["minor_fwhm_arcsec"],
                    "kernel_x_arcsec": kernel_fit["x_arcsec"],
                    "kernel_y_arcsec": kernel_fit["y_arcsec"],
                    "kernel_normalized_amplitude":
                        float(fit["amp"]) / kernel_amplitude
                        if kernel_amplitude > 0.0 else math.nan,
                    "major_fwhm_over_kernel":
                        max(fitted_a_fwhm, fitted_b_fwhm)
                        / kernel_fit["major_fwhm_arcsec"],
                    "minor_fwhm_over_kernel":
                        min(fitted_a_fwhm, fitted_b_fwhm)
                        / kernel_fit["minor_fwhm_arcsec"],
                    "map_weight_median": (
                        float(np.median(finite_weight))
                        if finite_weight.size
                        else math.nan
                    ),
                    "map_weight_mean": (
                        float(np.mean(finite_weight))
                        if finite_weight.size
                        else math.nan
                    ),
                    "map_rms": signal_rms,
                    "map_background_sigma": background_sigma,
                    "successive_map_delta_rms": rms(delta),
                    "successive_map_delta_relative_rms": (
                        rms(delta) / previous_rms
                        if previous_rms > 0.0
                        else math.nan
                    ),
                    **feedback_support_metrics(
                        signal, weight, float(flux_limits[array_id]),
                        feedback_config, pixel_size_arcsec,
                    ),
                }
            )
            previous_maps[array_name] = signal
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=REDUCED_DIR",
        help="May be repeated; REDUCED_DIR must contain redu00 through redu04",
    )
    parser.add_argument("--obsnum", type=int, default=133410)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for item in args.run:
        if "=" not in item:
            parser.error(f"invalid --run {item!r}; expected LABEL=REDUCED_DIR")
        label, path = item.split("=", 1)
        rows.extend(reduction_rows(label, Path(path), args.obsnum))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
