#!/usr/bin/env python3
"""Measure a known source from injected-minus-control fruit-loop maps."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
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


def gaussian_center_for_map_world_offset(
    path: Path,
    extension: str,
    az_offset_arcsec: float,
    el_offset_arcsec: float,
) -> tuple[float, float]:
    """Convert FITS AZ/EL world offsets to gaussian_fit coordinates."""
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[extension]
        shape = np.asarray(hdu.data).squeeze().shape
        header = hdu.header
    if len(shape) != 2:
        raise ValueError(f"expected a 2-D squeezed map in {path}:{extension}")
    if header.get("CTYPE1") != "AZOFFSET" or header.get("CTYPE2") != "ELOFFSET":
        raise ValueError(
            f"expected AZOFFSET/ELOFFSET WCS in {path}:{extension}"
        )

    def arcsec_scale(unit: str) -> float:
        normalized = unit.strip().lower()
        if normalized in {"arcsec", "arcsecond", "arcseconds"}:
            return 1.0
        if normalized in {"deg", "degree", "degrees"}:
            return 3600.0
        raise ValueError(f"unsupported offset-map WCS unit {unit!r}")

    scale_x = arcsec_scale(str(header["CUNIT1"]))
    scale_y = arcsec_scale(str(header["CUNIT2"]))
    cdelt_x = float(header["CDELT1"]) * scale_x
    cdelt_y = float(header["CDELT2"]) * scale_y
    if cdelt_x == 0.0 or cdelt_y == 0.0:
        raise ValueError(f"zero offset-map WCS increment in {path}:{extension}")
    pixel_x = (
        (az_offset_arcsec - float(header["CRVAL1"]) * scale_x) / cdelt_x
        + float(header["CRPIX1"])
        - 1.0
    )
    pixel_y = (
        (el_offset_arcsec - float(header["CRVAL2"]) * scale_y) / cdelt_y
        + float(header["CRPIX2"])
        - 1.0
    )
    pixel_size = abs(cdelt_x)
    if not math.isclose(abs(cdelt_y), pixel_size, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"non-square offset-map pixels in {path}:{extension}")
    return (
        (pixel_x - (shape[1] - 1) / 2.0) * pixel_size,
        (pixel_y - (shape[0] - 1) / 2.0) * pixel_size,
    )


def rms(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return math.nan
    return float(np.sqrt(np.mean(np.square(values[finite]))))


def file_record(path: Path, relative_to: Path | None = None) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    recorded_path = path
    if relative_to is not None:
        try:
            recorded_path = path.relative_to(relative_to)
        except ValueError:
            pass
    return {
        "path": str(recorded_path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


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
    values: np.ndarray,
    pixel_size_arcsec: float,
    expected_center_arcsec: tuple[float, float] | None = None,
    search_radius_arcsec: float = 25.0,
) -> dict[str, float]:
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("cannot fit an entirely non-finite map")
    map_yy, map_xx = np.indices(values.shape, dtype=float)
    map_xx = (
        map_xx - (values.shape[1] - 1) / 2.0
    ) * pixel_size_arcsec
    map_yy = (
        map_yy - (values.shape[0] - 1) / 2.0
    ) * pixel_size_arcsec
    peak_candidates = finite
    if expected_center_arcsec is not None:
        expected_x, expected_y = expected_center_arcsec
        peak_candidates = peak_candidates & (
            np.hypot(map_xx - expected_x, map_yy - expected_y)
            <= search_radius_arcsec
        )
        if not peak_candidates.any():
            raise ValueError("no finite samples near the expected source center")
    peak_flat = np.nanargmax(
        np.where(peak_candidates, values, np.nan)
    )
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
    mean_bounds = {}
    if expected_center_arcsec is not None:
        expected_x, expected_y = expected_center_arcsec
        mean_bounds = {
            "x_mean": (
                expected_x - search_radius_arcsec,
                expected_x + search_radius_arcsec,
            ),
            "y_mean": (
                expected_y - search_radius_arcsec,
                expected_y + search_radius_arcsec,
            ),
        }
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
            **mean_bounds,
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
        "az_offset_arcsec": float(manifest.get("az_offset_arcsec", 0.0)),
        "el_offset_arcsec": float(manifest.get("el_offset_arcsec", 0.0)),
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
        missing_default = 0.0 if key in {
            "az_offset_arcsec", "el_offset_arcsec"
        } else None
        control_value = control_test.get(key, missing_default)
        injected_value = injected_test.get(key, missing_default)
        if control_value != value or injected_value != value:
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
    az_offset_arcsec = float(manifest.get("az_offset_arcsec", 0.0))
    el_offset_arcsec = float(manifest.get("el_offset_arcsec", 0.0))
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
            # A global peak search can lock onto an unrelated paired-map
            # subtraction artifact. Convert the declared FITS map-world
            # position into this fitter's signed pixel-axis coordinates.
            expected_center = gaussian_center_for_map_world_offset(
                product_path(injected, obsnum, array),
                "signal_I",
                az_offset_arcsec,
                el_offset_arcsec,
            )
            transfer_fit = gaussian_fit(
                transfer,
                pixel_size,
                expected_center_arcsec=expected_center,
            )
            kernel_fit = gaussian_fit(
                injected_kernel,
                pixel_size,
                expected_center_arcsec=expected_center,
            )
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
                    "injected_az_offset_arcsec": az_offset_arcsec,
                    "injected_el_offset_arcsec": el_offset_arcsec,
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


def write_plot(
    rows: list[dict[str, float | int | str]], path: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "a1100": "#1f77b4",
        "a1400": "#ff7f0e",
        "a2000": "#2ca02c",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    for array in ARRAYS:
        selected = sorted(
            (row for row in rows if row["array"] == array),
            key=lambda row: int(row["iteration"]),
        )
        iterations = [int(row["iteration"]) for row in selected]
        color = colors[array]
        axes[0, 0].plot(
            iterations,
            [
                float(row["kernel_normalized_amplitude_recovery_fraction"])
                for row in selected
            ],
            marker="o",
            color=color,
            label=f"{array} Gaussian",
        )
        axes[0, 0].plot(
            iterations,
            [
                float(row["kernel_projection_recovery_fraction"])
                for row in selected
            ],
            marker="s",
            linestyle="--",
            color=color,
            label=f"{array} projection",
        )
        axes[0, 1].plot(
            iterations,
            [float(row["major_fwhm_over_kernel"]) for row in selected],
            marker="o",
            color=color,
            label=f"{array} major",
        )
        axes[0, 1].plot(
            iterations,
            [float(row["minor_fwhm_over_kernel"]) for row in selected],
            marker="s",
            linestyle="--",
            color=color,
            label=f"{array} minor",
        )
        axes[1, 0].plot(
            iterations,
            [float(row["centroid_error_arcsec"]) for row in selected],
            marker="o",
            color=color,
            label=array,
        )
        axes[1, 1].plot(
            iterations,
            [
                float(row["successive_transfer_delta_relative_rms"])
                for row in selected
            ],
            marker="o",
            color=color,
            label=array,
        )

    axes[0, 0].axhline(1.0, color="0.4", linewidth=1.0)
    axes[0, 0].set_ylabel("Recovered / injected")
    axes[0, 0].set_title("Amplitude response")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].axhline(1.0, color="0.4", linewidth=1.0)
    axes[0, 1].set_ylabel("Transfer FWHM / kernel FWHM")
    axes[0, 1].set_title("Shape response")
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[1, 0].set_ylabel("Transfer–kernel centroid (arcsec)")
    axes[1, 0].set_title("Centroid agreement")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_ylabel("RMS(iteration delta) / RMS(previous)")
    axes[1, 1].set_title("Remaining iteration-to-iteration change")
    axes[1, 1].legend(fontsize=8)
    for axis in axes[1, :]:
        axis.set_xlabel("Absolute FRUIT iteration")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xticks(sorted({int(row["iteration"]) for row in rows}))
    figure.suptitle(
        "Pointing 152389: centered 100 mJy/beam injection\n"
        "development-only diagnostic; no qualification threshold applied"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_provenance_manifest(
    path: Path,
    *,
    control_root: Path,
    injected_root: Path,
    pair_manifest_path: Path,
    continuation_reference: Path,
    obsnum: int,
    output_csv: Path,
    output_report: Path,
    output_plot: Path | None,
    executable: Path,
    software_id: str,
    test_id: str,
) -> None:
    pair_manifest = yaml.safe_load(pair_manifest_path.read_text())
    input_paths: set[Path] = {
        Path(__file__).resolve(),
        executable.resolve(),
        pair_manifest_path.resolve(),
        Path(pair_manifest["source_config"]).resolve(),
        (
            Path(pair_manifest["restart_path"])
            / "citlali_restart_checkpoint.nc"
        ).resolve(),
    }
    for variant in pair_manifest["variants"].values():
        input_paths.add(
            (pair_manifest_path.parent / variant["config"]).resolve()
        )
    for array in ARRAYS:
        input_paths.add(
            product_path(continuation_reference, obsnum, array).resolve()
        )
    for root in (control_root, injected_root):
        for redu in iteration_dirs(root, obsnum).values():
            for array in ARRAYS:
                input_paths.add(product_path(redu, obsnum, array).resolve())
            input_paths.add(
                (
                    redu
                    / str(obsnum)
                    / "raw"
                    / f"ppt_commissioning_pointing_{obsnum}_citlali.ecsv"
                ).resolve()
            )
            for name in (
                "LOCAL_IO_SUPPRESSION.yaml",
                "config_source_manifest.yaml",
                "runtime_provenance.yaml",
            ):
                candidate = (redu / name).resolve()
                if candidate.is_file():
                    input_paths.add(candidate)

    output_paths = [output_csv, output_report]
    if output_plot is not None:
        output_paths.append(output_plot)
    payload = {
        "schema_version": "sci-fruit-injected-source-development-v1",
        "test_id": test_id,
        "role": "exploratory-development-only",
        "qualification_use_authorized": False,
        "obsnum": obsnum,
        "software": {
            "reported_version": software_id,
            **file_record(executable.resolve()),
        },
        "pair": {
            "start_iteration": int(pair_manifest["start_iteration"]),
            "stop_iteration_exclusive": int(
                pair_manifest["stop_iteration_exclusive"]
            ),
            "array_order": list(pair_manifest["array_order"]),
            "array_amplitude_mjy_beam": [
                float(value)
                for value in pair_manifest["array_amplitude_mjy_beam"]
            ],
            "az_offset_arcsec": float(
                pair_manifest.get("az_offset_arcsec", 0.0)
            ),
            "el_offset_arcsec": float(
                pair_manifest.get("el_offset_arcsec", 0.0)
            ),
            "exact_restart_control": "PASS",
        },
        "measurement": {
            "map_response": "injected signal_I minus control signal_I",
            "source_location": {
                "frame": "FITS map world",
                "axes": ["AZOFFSET", "ELOFFSET"],
                "unit": "arcsec",
                "az_offset_arcsec": float(
                    pair_manifest.get("az_offset_arcsec", 0.0)
                ),
                "el_offset_arcsec": float(
                    pair_manifest.get("el_offset_arcsec", 0.0)
                ),
            },
            "gaussian_search_radius_arcsec": 25.0,
            "kernel_comparator": "same-iteration injected kernel_I",
        },
        "inputs": [
            file_record(input_path)
            for input_path in sorted(input_paths, key=str)
        ],
        "outputs": [
            file_record(output_path.resolve(), relative_to=path.parent.resolve())
            for output_path in output_paths
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
    parser.add_argument(
        "--plot",
        type=Path,
        help="optional path for a development-diagnostic summary plot",
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help="optional path for a hashed development-evidence manifest",
    )
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--software-id")
    parser.add_argument("--test-id")
    args = parser.parse_args()

    provenance_values = (
        args.provenance_output,
        args.executable,
        args.software_id,
        args.test_id,
    )
    if any(value is not None for value in provenance_values) and not all(
        value is not None for value in provenance_values
    ):
        parser.error(
            "--provenance-output, --executable, --software-id, and --test-id "
            "must be supplied together"
        )

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
    if args.plot is not None:
        write_plot(rows, args.plot)
    if args.provenance_output is not None:
        write_provenance_manifest(
            args.provenance_output,
            control_root=args.control,
            injected_root=args.injected,
            pair_manifest_path=args.manifest,
            continuation_reference=args.continuation_reference,
            obsnum=args.obsnum,
            output_csv=args.output,
            output_report=report_path,
            output_plot=args.plot,
            executable=args.executable,
            software_id=args.software_id,
            test_id=args.test_id,
        )
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote report to {report_path}")
    if args.plot is not None:
        print(f"wrote plot to {args.plot}")
    if args.provenance_output is not None:
        print(f"wrote provenance to {args.provenance_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
