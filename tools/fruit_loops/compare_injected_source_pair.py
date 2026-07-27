#!/usr/bin/env python3
"""Measure a known source from injected-minus-control fruit-loop maps."""

from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.table import Table


ARRAYS = ("a1100", "a1400", "a2000")
FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))


def redu_dirs(root: Path) -> dict[int, Path]:
    result = {}
    for path in root.glob("redu[0-9]*"):
        suffix = path.name[4:]
        if path.is_dir() and suffix.isdigit():
            result[int(suffix)] = path
    if not result:
        raise ValueError(f"no reduNN directories found in {root}")
    return result


def product_path(redu: Path, obsnum: int, array: str) -> Path:
    return (
        redu
        / str(obsnum)
        / "raw"
        / f"toltec_commissioning_{array}_pointing_{obsnum}_citlali.fits"
    )


def iteration_dirs(root: Path, obsnum: int) -> dict[int, Path]:
    result = {}
    for redu in redu_dirs(root).values():
        path = product_path(redu, obsnum, ARRAYS[0])
        with fits.open(path, memmap=True) as hdul:
            iteration = int(hdul[0].header["FRUITLOOPS_ITER"])
        if iteration in result:
            raise ValueError(
                f"duplicate fruit-loop iteration {iteration} in "
                f"{result[iteration]} and {redu}"
            )
        result[iteration] = redu
    return result


def image(path: Path, extension: str) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        return np.asarray(hdul[extension].data, dtype=float).squeeze()


def rms(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return math.nan
    return float(np.sqrt(np.mean(np.square(values[finite]))))


def kernel_projection_metrics(
    transfer: np.ndarray, kernel: np.ndarray, truth: float,
) -> dict[str, float]:
    finite = np.isfinite(transfer) & np.isfinite(kernel)
    if not finite.any():
        raise ValueError("cannot compare entirely non-finite transfer/kernel")
    transfer_values = transfer[finite]
    kernel_values = kernel[finite]
    kernel_power = float(np.dot(kernel_values, kernel_values))
    if kernel_power <= 0.0:
        raise ValueError("cannot project onto a zero kernel")
    scale = float(np.dot(kernel_values, transfer_values) / kernel_power)
    residual = transfer_values - scale * kernel_values
    transfer_rms = rms(transfer_values)
    return {
        "scale_mjy_beam": scale,
        "recovery_fraction": scale / truth if truth > 0.0 else math.nan,
        "residual_relative_rms":
            rms(residual) / transfer_rms
            if transfer_rms > 0.0 else math.nan,
    }


def gaussian_fit(
    values: np.ndarray, pixel_size_arcsec: float,
) -> dict[str, float]:
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("cannot fit an entirely non-finite map")
    peak_flat = np.nanargmax(np.where(finite, values, np.nan))
    peak_y, peak_x = np.unravel_index(peak_flat, values.shape)
    radius_px = max(4, int(math.ceil(25.0 / pixel_size_arcsec)))
    y0, y1 = max(0, peak_y - radius_px), min(values.shape[0], peak_y + radius_px + 1)
    x0, x1 = max(0, peak_x - radius_px), min(values.shape[1], peak_x + radius_px + 1)
    cutout = values[y0:y1, x0:x1]
    yy, xx = np.indices(cutout.shape, dtype=float)
    xx = (xx + x0 - (values.shape[1] - 1) / 2.0) * pixel_size_arcsec
    yy = (yy + y0 - (values.shape[0] - 1) / 2.0) * pixel_size_arcsec
    good = np.isfinite(cutout)
    background = float(np.nanmedian(cutout))
    amplitude = float(np.nanmax(cutout) - background)
    initial_sigma = max(2.0 * pixel_size_arcsec, 3.0)
    model = models.Const2D(amplitude=background) + models.Gaussian2D(
        amplitude=max(amplitude, np.finfo(float).eps),
        x_mean=float(xx[peak_y - y0, peak_x - x0]),
        y_mean=float(yy[peak_y - y0, peak_x - x0]),
        x_stddev=initial_sigma,
        y_stddev=initial_sigma,
        theta=0.0,
        bounds={
            "amplitude": (0.0, None),
            "x_stddev": (pixel_size_arcsec / 4.0, 30.0),
            "y_stddev": (pixel_size_arcsec / 4.0, 30.0),
        },
    )
    fitted = fitting.TRFLSQFitter()(
        model, xx[good], yy[good], cutout[good], maxiter=1000
    )
    gaussian = fitted[1]
    x_fwhm = abs(float(gaussian.x_stddev.value)) * FWHM_FACTOR
    y_fwhm = abs(float(gaussian.y_stddev.value)) * FWHM_FACTOR
    return {
        "amplitude": float(gaussian.amplitude.value),
        "x_arcsec": float(gaussian.x_mean.value),
        "y_arcsec": float(gaussian.y_mean.value),
        "major_fwhm_arcsec": max(x_fwhm, y_fwhm),
        "minor_fwhm_arcsec": min(x_fwhm, y_fwhm),
    }


def low_level_config(redu: Path) -> dict:
    candidates = sorted(
        path
        for path in redu.glob("citlali*.yaml")
        if path.name != "citlali_merged_config.yaml"
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one low-level config in {redu}, found {candidates}"
        )
    result = yaml.safe_load(candidates[0].read_text())
    if not isinstance(result, dict):
        raise ValueError(f"invalid config {candidates[0]}")
    return result


def normalized_pair_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["runtime"]["output_dir"] = "<paired-output-root>"
    result["timestream"]["fruit_loops"]["injected_source_test"]["enabled"] = (
        "<paired-enabled-state>"
    )
    return result


def require_pair_config(
    control: dict, injected: dict, manifest: dict,
) -> None:
    if normalized_pair_config(control) != normalized_pair_config(injected):
        raise ValueError(
            "control/injected low-level configs differ beyond output_dir and "
            "injected_source_test.enabled"
        )
    expected = {
        "start_iteration": int(manifest["start_iteration"]),
        "array_amplitude_mjy_beam": [
            float(value)
            for value in manifest["array_amplitude_mjy_beam"]
        ],
    }
    control_test = control["timestream"]["fruit_loops"][
        "injected_source_test"
    ]
    injected_test = injected["timestream"]["fruit_loops"][
        "injected_source_test"
    ]
    if control_test["enabled"] is not False:
        raise ValueError("control config has injected-source test enabled")
    if injected_test["enabled"] is not True:
        raise ValueError("injected config has injected-source test disabled")
    for key, value in expected.items():
        if control_test[key] != value or injected_test[key] != value:
            raise ValueError(
                f"paired config {key} does not match manifest value {value}"
            )


def fit_table(redu: Path, obsnum: int) -> Table:
    return Table.read(
        redu
        / str(obsnum)
        / "raw"
        / f"ppt_commissioning_pointing_{obsnum}_citlali.ecsv",
        format="ascii.ecsv",
    )


def require_exact_restart_control(
    reference: Path, control: Path, obsnum: int, iteration: int,
) -> None:
    reference_product = product_path(reference, obsnum, ARRAYS[0])
    with fits.open(reference_product, memmap=True) as hdul:
        reference_iteration = int(hdul[0].header["FRUITLOOPS_ITER"])
    if reference_iteration != iteration:
        raise ValueError(
            "uninterrupted continuation reference has iteration "
            f"{reference_iteration}, expected {iteration}"
        )

    for array in ARRAYS:
        reference_path = product_path(reference, obsnum, array)
        control_path = product_path(control, obsnum, array)
        for extension in ("signal_I", "kernel_I", "weight_I"):
            expected = image(reference_path, extension)
            actual = image(control_path, extension)
            if np.array_equal(expected, actual, equal_nan=True):
                continue
            difference = actual - expected
            expected_rms = rms(expected)
            relative_rms = (
                rms(difference) / expected_rms
                if expected_rms > 0.0 else math.nan
            )
            raise ValueError(
                "restarted control differs from uninterrupted continuation: "
                f"iteration={iteration} array={array} "
                f"extension={extension} relative_rms={relative_rms:.8g}"
            )


def comparison_rows(
    control_root: Path,
    injected_root: Path,
    manifest_path: Path,
    continuation_reference: Path,
    obsnum: int,
) -> list[dict[str, float | int | str]]:
    manifest = yaml.safe_load(manifest_path.read_text())
    amplitudes = dict(
        zip(manifest["array_order"], manifest["array_amplitude_mjy_beam"])
    )
    control_dirs = iteration_dirs(control_root, obsnum)
    injected_dirs = iteration_dirs(injected_root, obsnum)
    iterations = sorted(set(control_dirs) & set(injected_dirs))
    expected = list(
        range(
            int(manifest["start_iteration"]),
            int(manifest["stop_iteration_exclusive"]),
        )
    )
    if iterations != expected:
        raise ValueError(
            f"paired iterations differ from manifest: expected={expected} "
            f"actual={iterations}"
        )
    require_exact_restart_control(
        continuation_reference,
        control_dirs[iterations[0]],
        obsnum,
        iterations[0],
    )

    rows: list[dict[str, float | int | str]] = []
    previous_transfer: dict[str, np.ndarray] = {}
    for iteration in iterations:
        control = control_dirs[iteration]
        injected = injected_dirs[iteration]
        control_config = low_level_config(control)
        injected_config = low_level_config(injected)
        require_pair_config(control_config, injected_config, manifest)
        pixel_size = float(injected_config["mapmaking"]["pixel_size_arcsec"])
        if control_config["mapmaking"]["pixel_size_arcsec"] != pixel_size:
            raise ValueError("control/injected pixel sizes differ")
        control_table = fit_table(control, obsnum)
        injected_table = fit_table(injected, obsnum)

        for array_index, array in enumerate(ARRAYS):
            control_map = image(
                product_path(control, obsnum, array), "signal_I"
            )
            injected_map = image(
                product_path(injected, obsnum, array), "signal_I"
            )
            control_kernel = image(
                product_path(control, obsnum, array), "kernel_I"
            )
            injected_kernel = image(
                product_path(injected, obsnum, array), "kernel_I"
            )
            control_weight = image(
                product_path(control, obsnum, array), "weight_I"
            )
            injected_weight = image(
                product_path(injected, obsnum, array), "weight_I"
            )
            truth = float(amplitudes[array])
            transfer = injected_map - control_map
            transfer_fit = gaussian_fit(transfer, pixel_size)
            kernel_fit = gaussian_fit(injected_kernel, pixel_size)
            kernel_projection = kernel_projection_metrics(
                transfer, injected_kernel, truth,
            )
            previous = previous_transfer.get(array)
            successive_rms = (
                rms(transfer - previous) if previous is not None else math.nan
            )
            previous_rms = rms(previous) if previous is not None else math.nan
            control_fit = control_table[control_table["array"] == array_index][0]
            injected_fit = injected_table[
                injected_table["array"] == array_index
            ][0]
            positive_control_weight = control_weight[
                np.isfinite(control_weight) & (control_weight > 0.0)
            ]
            positive_injected_weight = injected_weight[
                np.isfinite(injected_weight) & (injected_weight > 0.0)
            ]
            centroid_error = math.hypot(
                transfer_fit["x_arcsec"] - kernel_fit["x_arcsec"],
                transfer_fit["y_arcsec"] - kernel_fit["y_arcsec"],
            )
            rows.append(
                {
                    "iteration": iteration,
                    "array": array,
                    "injected_amplitude_mjy_beam": truth,
                    "recovered_transfer_amplitude_mjy_beam":
                        transfer_fit["amplitude"],
                    "amplitude_recovery_fraction":
                        transfer_fit["amplitude"] / truth
                        if truth > 0.0 else math.nan,
                    "kernel_fit_amplitude": kernel_fit["amplitude"],
                    "kernel_normalized_amplitude_recovery_fraction":
                        transfer_fit["amplitude"]
                        / (truth * kernel_fit["amplitude"])
                        if truth > 0.0
                        and kernel_fit["amplitude"] > 0.0
                        else math.nan,
                    "kernel_projection_recovery_fraction":
                        kernel_projection["recovery_fraction"],
                    "kernel_projection_residual_relative_rms":
                        kernel_projection["residual_relative_rms"],
                    "transfer_major_fwhm_arcsec":
                        transfer_fit["major_fwhm_arcsec"],
                    "transfer_minor_fwhm_arcsec":
                        transfer_fit["minor_fwhm_arcsec"],
                    "kernel_major_fwhm_arcsec":
                        kernel_fit["major_fwhm_arcsec"],
                    "kernel_minor_fwhm_arcsec":
                        kernel_fit["minor_fwhm_arcsec"],
                    "major_fwhm_over_kernel":
                        transfer_fit["major_fwhm_arcsec"]
                        / kernel_fit["major_fwhm_arcsec"],
                    "minor_fwhm_over_kernel":
                        transfer_fit["minor_fwhm_arcsec"]
                        / kernel_fit["minor_fwhm_arcsec"],
                    "centroid_error_arcsec": centroid_error,
                    "successive_transfer_delta_rms": successive_rms,
                    "successive_transfer_delta_relative_rms":
                        successive_rms / previous_rms
                        if previous_rms > 0.0 else math.nan,
                    "kernel_control_injected_delta_rms":
                        rms(injected_kernel - control_kernel),
                    "weight_control_injected_delta_rms":
                        rms(injected_weight - control_weight),
                    "control_weight_median":
                        float(np.median(positive_control_weight))
                        if positive_control_weight.size else math.nan,
                    "injected_weight_median":
                        float(np.median(positive_injected_weight))
                        if positive_injected_weight.size else math.nan,
                    "control_fit_amplitude_mjy_beam":
                        float(control_fit["amp"]),
                    "injected_fit_amplitude_mjy_beam":
                        float(injected_fit["amp"]),
                    "control_legacy_peak_over_full_map_rms":
                        float(control_fit["sig2noise"]),
                    "injected_legacy_peak_over_full_map_rms":
                        float(injected_fit["sig2noise"]),
                    "control_fit_sig2noise":
                        float(control_fit["amp"])
                        / float(control_fit["amp_err"]),
                    "injected_fit_sig2noise":
                        float(injected_fit["amp"])
                        / float(injected_fit["amp_err"]),
                }
            )
            previous_transfer[array] = transfer
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--injected", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--continuation-reference",
        required=True,
        type=Path,
        help="uninterrupted reduNN directory for the first paired iteration",
    )
    parser.add_argument("--obsnum", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = comparison_rows(
        args.control,
        args.injected,
        args.manifest,
        args.continuation_reference,
        args.obsnum,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    report_path = args.output.with_suffix(".md")
    lines = [
        "# Fruit-loop Injected-source Transfer",
        "",
        "Exact restart control: PASS",
        "",
        "| Iter | Array | Raw amp recovery | Kernel-normalized amp | "
        "Kernel projection | FWHM/kernel major | "
        "Centroid error (arcsec) | Successive relative RMS |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['iteration']} | {row['array']} | "
            f"{row['amplitude_recovery_fraction']:.6f} | "
            f"{row['kernel_normalized_amplitude_recovery_fraction']:.6f} | "
            f"{row['kernel_projection_recovery_fraction']:.6f} | "
            f"{row['major_fwhm_over_kernel']:.6f} | "
            f"{row['centroid_error_arcsec']:.6f} | "
            f"{row['successive_transfer_delta_relative_rms']:.6g} |"
        )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
