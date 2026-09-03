#!/usr/bin/env python3
"""Analyze the registered SCI-FRUIT EL-F7 shared-start decomposition."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from netCDF4 import Dataset

from tools.fruit_loops.analyze_compact_relaxation_screen import (
    IMAGE_EXTENSIONS,
    common_support,
    load_image,
)
from tools.fruit_loops.analyze_off_source_penalty_counterfactual import (
    require_exact_checkpoint,
    require_exact_maps,
    target_penalty_present,
)
from tools.fruit_loops.compare_injected_source_pair import (
    ARRAYS,
    file_record,
    gaussian_center_for_map_world_offset,
    gaussian_fit,
    iteration_dirs,
    product_path,
    rms,
)


COMPONENTS = (
    "T5_total_adaptive",
    "S5_shared_start",
    "H5_other_history",
    "D4460_5",
)
DECOMPOSITION_COMPONENTS = COMPONENTS[1:]
REGIONS = (
    "complete_map",
    "injected_source_r20",
    "neptune_r20",
    "annulus_r40_120_excluding_neptune_r25",
)
FITS_EXTENSION_NAMES = {
    "T5_total_adaptive": "T5_TOTAL",
    "S5_shared_start": "S5_SHARE",
    "H5_other_history": "H5_HIST",
    "D4460_5": "D5_UID",
}


def iteration_dir(root: Path, obsnum: int, iteration: int) -> Path:
    """Return the unique output directory for an absolute FRUIT iteration."""
    found = iteration_dirs(root, obsnum)
    if iteration not in found:
        raise ValueError(
            f"absolute iteration {iteration} is absent from {root}: "
            f"found={sorted(found)}"
        )
    return found[iteration]


def fruit_iteration(path: Path) -> int:
    with fits.open(path, memmap=True) as hdul:
        return int(hdul[0].header["FRUITLOOPS_ITER"])


def read_merged_config(redu: Path) -> dict:
    path = redu / "citlali_merged_config.yaml"
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"invalid merged config: {path}")
    return config


def normalized_paired_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["runtime"]["output_dir"] = "<paired-output>"
    fruit = result["timestream"]["fruit_loops"]
    fruit["restart_path"] = "<paired-restart>"
    fruit["injected_source_test"]["enabled"] = "<paired-enabled>"
    return result


def require_paired_configs(
    sham_redu: Path, probe_redu: Path, manifest: dict,
) -> None:
    """Require identical realized settings outside registered paired fields."""
    sham = read_merged_config(sham_redu)
    probe = read_merged_config(probe_redu)
    if normalized_paired_config(sham) != normalized_paired_config(probe):
        raise ValueError(
            "EL-F7 merged configs differ beyond output/restart paths and "
            "injection enabled state"
        )
    expected_common = {
        "start_iteration": int(manifest["iteration"]),
        "array_amplitude_mjy_beam": [
            float(value) for value in manifest["amplitudes_mjy_beam"]
        ],
        "az_offset_arcsec": float(
            manifest["injection_position_fits_world_arcsec"][0]
        ),
        "el_offset_arcsec": float(
            manifest["injection_position_fits_world_arcsec"][1]
        ),
    }
    sham_test = sham["timestream"]["fruit_loops"]["injected_source_test"]
    probe_test = probe["timestream"]["fruit_loops"]["injected_source_test"]
    if sham_test["enabled"] is not False or probe_test["enabled"] is not True:
        raise ValueError("EL-F7 realized injection enabled states differ")
    for key, expected in expected_common.items():
        if sham_test[key] != expected or probe_test[key] != expected:
            raise ValueError(f"EL-F7 realized injection field differs: {key}")


def validate_sham(manifest: dict) -> dict:
    """Apply the mandatory exact sham gate without writing result artifacts."""
    obsnum = int(manifest["obsnum"])
    iteration = int(manifest["iteration"])
    reference = Path(manifest["existing_control_iteration_5"])
    actual = iteration_dir(Path(manifest["control_sham_root"]), obsnum, iteration)
    planes = require_exact_maps(reference, actual, obsnum)
    require_exact_checkpoint(
        reference / "citlali_restart_checkpoint.nc",
        actual / "citlali_restart_checkpoint.nc",
    )
    return {
        "sham_replay_exact": True,
        "sham_exact_planes": planes,
        "sham_checkpoint_all_variables_value_identical": True,
        "sham_iteration_directory": str(actual.resolve()),
    }


def roundoff_bound(arrays: list[np.ndarray], factor: float) -> float:
    finite_maxima = []
    for values in arrays:
        finite = np.isfinite(values)
        if finite.any():
            finite_maxima.append(float(np.max(np.abs(values[finite]))))
    scale = max([1.0, *finite_maxima])
    return factor * np.finfo(np.float64).eps * scale


def closure_metrics(
    total: np.ndarray,
    shared: np.ndarray,
    history: np.ndarray,
    penalty: np.ndarray,
    support: np.ndarray,
    factor: float,
) -> dict[str, float | bool]:
    """Evaluate the registered telescoping identity on common support."""
    residual = total - (shared + history + penalty)
    maximum = float(np.max(np.abs(residual[support])))
    bound = roundoff_bound([total, shared, history, penalty], factor)
    return {
        "maximum_absolute_residual_mjy_beam": maximum,
        "roundoff_bound_mjy_beam": bound,
        "closure_pass": bool(maximum <= bound),
    }


def coordinate_grid(shape: tuple[int, int], pixel_size: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(shape, dtype=float)
    xx = (xx - (shape[1] - 1) / 2.0) * pixel_size
    yy = (yy - (shape[0] - 1) / 2.0) * pixel_size
    return xx, yy


def circular_mask(
    xx: np.ndarray, yy: np.ndarray, center: tuple[float, float], radius: float,
) -> np.ndarray:
    return np.hypot(xx - center[0], yy - center[1]) <= radius


def make_region_masks(
    shape: tuple[int, int],
    pixel_size: float,
    injection_center: tuple[float, float],
    neptune_center: tuple[float, float],
    manifest: dict,
) -> dict[str, np.ndarray]:
    xx, yy = coordinate_grid(shape, pixel_size)
    injection_radius = np.hypot(
        xx - injection_center[0], yy - injection_center[1]
    )
    neptune_radius = np.hypot(
        xx - neptune_center[0], yy - neptune_center[1]
    )
    return {
        "complete_map": np.ones(shape, dtype=bool),
        "injected_source_r20": (
            injection_radius <= float(manifest["injected_source_radius_arcsec"])
        ),
        "neptune_r20": (
            neptune_radius <= float(manifest["neptune_radius_arcsec"])
        ),
        "annulus_r40_120_excluding_neptune_r25": (
            (injection_radius >= float(manifest["annulus_inner_arcsec"]))
            & (injection_radius <= float(manifest["annulus_outer_arcsec"]))
            & (
                neptune_radius
                > float(manifest["annulus_neptune_exclusion_radius_arcsec"])
            )
        ),
    }


def projection(
    component: np.ndarray,
    kernel: np.ndarray,
    support: np.ndarray,
    truth: float,
) -> tuple[dict[str, float], np.ndarray]:
    values = component[support]
    template = kernel[support]
    power = float(np.dot(template, template))
    if power <= 0.0:
        raise ValueError("fixed decomposition kernel has zero power")
    scale = float(np.dot(template, values) / power)
    residual = component - scale * kernel
    component_rms = rms(values)
    metrics = {
        "fixed_kernel_scale_mjy_beam": scale,
        "fixed_kernel_recovery_fraction": scale / truth,
        "fixed_kernel_residual_relative_rms": (
            rms(residual[support]) / component_rms
            if component_rms > 0.0 else math.nan
        ),
    }
    return metrics, residual


def aperture_fraction(
    component: np.ndarray,
    kernel: np.ndarray,
    support: np.ndarray,
    aperture: np.ndarray,
    truth: float,
) -> float:
    selected = support & aperture
    denominator = truth * float(np.sum(kernel[selected]))
    if denominator == 0.0:
        return math.nan
    return float(np.sum(component[selected])) / denominator


def native_compact_metrics(
    component: np.ndarray,
    native_kernel: np.ndarray,
    pixel_size: float,
    center: tuple[float, float],
    search_radius: float,
    truth: float,
) -> dict[str, float]:
    source_fit = gaussian_fit(
        component,
        pixel_size,
        expected_center_arcsec=center,
        search_radius_arcsec=search_radius,
    )
    kernel_fit = gaussian_fit(
        native_kernel,
        pixel_size,
        expected_center_arcsec=center,
        search_radius_arcsec=search_radius,
    )
    return {
        "fitted_amplitude_mjy_beam": float(source_fit["amplitude"]),
        "kernel_normalized_central_recovery": (
            float(source_fit["amplitude"])
            / (truth * float(kernel_fit["amplitude"]))
        ),
        "fitted_major_fwhm_arcsec": float(source_fit["major_fwhm_arcsec"]),
        "fitted_minor_fwhm_arcsec": float(source_fit["minor_fwhm_arcsec"]),
        "kernel_major_fwhm_arcsec": float(kernel_fit["major_fwhm_arcsec"]),
        "kernel_minor_fwhm_arcsec": float(kernel_fit["minor_fwhm_arcsec"]),
        "major_fwhm_over_kernel": (
            float(source_fit["major_fwhm_arcsec"])
            / float(kernel_fit["major_fwhm_arcsec"])
        ),
        "minor_fwhm_over_kernel": (
            float(source_fit["minor_fwhm_arcsec"])
            / float(kernel_fit["minor_fwhm_arcsec"])
        ),
        "centroid_separation_from_kernel_arcsec": math.hypot(
            float(source_fit["x_arcsec"]) - float(kernel_fit["x_arcsec"]),
            float(source_fit["y_arcsec"]) - float(kernel_fit["y_arcsec"]),
        ),
    }


def component_metrics(
    array: str,
    components: dict[str, np.ndarray],
    fixed_kernel: np.ndarray,
    native_kernels: dict[str, np.ndarray],
    support: np.ndarray,
    regions: dict[str, np.ndarray],
    injection_center: tuple[float, float],
    manifest: dict,
    truth: float,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = []
    residuals = {}
    aperture = regions["injected_source_r20"]
    for name in COMPONENTS:
        values = components[name]
        projected, residual = projection(values, fixed_kernel, support, truth)
        residuals[name] = residual
        row: dict[str, float | int | str] = {
            "array": array,
            "component": name,
            "common_valid_pixels": int(np.count_nonzero(support)),
            "complete_map_rms_mjy_beam": rms(values[support]),
            "aperture_integrated_response_fraction": aperture_fraction(
                values, fixed_kernel, support, aperture, truth
            ),
            **projected,
        }
        for region_name in REGIONS:
            selected = support & regions[region_name]
            if not selected.any():
                raise ValueError(f"empty EL-F7 region: {array}:{region_name}")
            row[f"{region_name}_pixels"] = int(np.count_nonzero(selected))
            row[f"{region_name}_rms_mjy_beam"] = rms(values[selected])
            row[f"{region_name}_residual_rms_mjy_beam"] = rms(
                residual[selected]
            )
        native = {
            "fitted_amplitude_mjy_beam": math.nan,
            "kernel_normalized_central_recovery": math.nan,
            "fitted_major_fwhm_arcsec": math.nan,
            "fitted_minor_fwhm_arcsec": math.nan,
            "kernel_major_fwhm_arcsec": math.nan,
            "kernel_minor_fwhm_arcsec": math.nan,
            "major_fwhm_over_kernel": math.nan,
            "minor_fwhm_over_kernel": math.nan,
            "centroid_separation_from_kernel_arcsec": math.nan,
        }
        if name in native_kernels:
            native = native_compact_metrics(
                values,
                native_kernels[name],
                float(manifest["pixel_size_arcsec"]),
                injection_center,
                float(manifest["gaussian_search_radius_arcsec"]),
                truth,
            )
        row.update(native)
        rows.append(row)
    return rows, residuals


def cross_term_rows(
    array: str,
    components: dict[str, np.ndarray],
    support: np.ndarray,
    regions: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for left, right in itertools.combinations(COMPONENTS, 2):
        for region_name in REGIONS:
            selected = support & regions[region_name]
            left_values = components[left][selected]
            right_values = components[right][selected]
            inner = float(np.mean(left_values * right_values))
            left_norm = float(np.sqrt(np.mean(np.square(left_values))))
            right_norm = float(np.sqrt(np.mean(np.square(right_values))))
            cosine = (
                inner / (left_norm * right_norm)
                if left_norm > 0.0 and right_norm > 0.0 else math.nan
            )
            rows.append(
                {
                    "array": array,
                    "left_component": left,
                    "right_component": right,
                    "region": region_name,
                    "pixels": int(np.count_nonzero(selected)),
                    "mean_product_mjy2_beam2": inner,
                    "two_mean_product_mjy2_beam2": 2.0 * inner,
                    "cosine": cosine,
                }
            )
    return rows


def netcdf_value_summary(left: object, right: object) -> dict:
    lhs = np.ma.asarray(left)
    rhs = np.ma.asarray(right)
    result: dict[str, object] = {
        "left_shape": list(lhs.shape),
        "right_shape": list(rhs.shape),
        "left_dtype": str(lhs.dtype),
        "right_dtype": str(rhs.dtype),
    }
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        result["equal"] = False
        return result
    left_mask = np.ma.getmaskarray(lhs)
    right_mask = np.ma.getmaskarray(rhs)
    if not np.array_equal(left_mask, right_mask):
        result.update(
            {
                "equal": False,
                "different_mask_elements": int(
                    np.count_nonzero(left_mask != right_mask)
                ),
            }
        )
        return result
    left_data = np.asarray(lhs.data)
    right_data = np.asarray(rhs.data)
    if left_data.dtype.kind in "fc":
        equal_elements = (left_data == right_data) | (
            np.isnan(left_data) & np.isnan(right_data)
        )
        finite = np.isfinite(left_data) & np.isfinite(right_data)
        result.update(
            {
                "equal": bool(np.all(equal_elements)),
                "different_elements": int(np.count_nonzero(~equal_elements)),
                "maximum_absolute_difference": (
                    float(np.max(np.abs(left_data[finite] - right_data[finite])))
                    if finite.any() else math.nan
                ),
            }
        )
    else:
        equal_elements = left_data == right_data
        result.update(
            {
                "equal": bool(np.all(equal_elements)),
                "different_elements": int(np.count_nonzero(~equal_elements)),
            }
        )
    return result


def checkpoint_difference(
    control_path: Path, probe_path: Path,
) -> dict[str, object]:
    """Describe every variable changed during the shared-start transition."""
    with Dataset(control_path) as control, Dataset(probe_path) as probe:
        control_names = set(control.variables)
        probe_names = set(probe.variables)
        summaries = {}
        for name in sorted(control_names & probe_names):
            summary = netcdf_value_summary(
                control.variables[name][...], probe.variables[name][...]
            )
            if not bool(summary["equal"]):
                summaries[name] = summary
        dimension_differences = {}
        for name in sorted(set(control.dimensions) | set(probe.dimensions)):
            left = len(control.dimensions[name]) if name in control.dimensions else None
            right = len(probe.dimensions[name]) if name in probe.dimensions else None
            if left != right:
                dimension_differences[name] = {"control": left, "probe": right}
    return {
        "control_checkpoint": file_record(control_path.resolve()),
        "probe_checkpoint": file_record(probe_path.resolve()),
        "control_only_variables": sorted(control_names - probe_names),
        "probe_only_variables": sorted(probe_names - control_names),
        "dimension_differences": dimension_differences,
        "differing_variable_count": len(summaries),
        "differing_variables": summaries,
    }


def read_execution(log_dir: Path) -> list[dict]:
    def time_value(text: str, label: str) -> float | None:
        for pattern in (
            rf"^\s*([0-9.]+)\s+{label}\s*$",
            rf"^\s*{label}\s+([0-9.]+)\s*$",
        ):
            match = re.search(pattern, text, re.M)
            if match is not None:
                return float(match.group(1))
        return None

    rows = []
    error_pattern = re.compile(r"\[(?:error|critical)\]|(?:error|critical):", re.I)
    for label in ("control-sham", "shared-start-probe"):
        path = log_dir / f"{label}.log"
        text = path.read_text(encoding="utf-8")
        wall = time_value(text, "real")
        user = time_value(text, "user")
        system = time_value(text, "sys")
        rss = re.search(r"^\s*([0-9]+)\s+maximum resident set size$", text, re.M)
        errors = sum(bool(error_pattern.search(line)) for line in text.splitlines())
        if (
            wall is None or user is None or system is None or rss is None
            or "citlali is done!" not in text or errors
        ):
            raise ValueError(f"incomplete or unsuccessful execution log: {path}")
        rows.append(
            {
                "trajectory": label,
                "status": "completed",
                "wall_seconds": wall,
                "user_seconds": user,
                "system_seconds": system,
                "maximum_resident_bytes": int(rss.group(1)),
                "error_or_critical_messages": errors,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_component_maps(
    output_dir: Path,
    obsnum: int,
    array: str,
    source_path: Path,
    components: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    manifest: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with fits.open(source_path, memmap=True) as hdul:
        header = hdul["signal_I"].header.copy()
    for key in ("CHECKSUM", "DATASUM"):
        if key in header:
            del header[key]
    primary = fits.PrimaryHDU()
    primary.header["HIERARCH SCI.TESTID"] = manifest["test_id"]
    primary.header["HIERARCH SCI.OBSNUM"] = obsnum
    primary.header["HIERARCH SCI.FRUIT_ITER"] = int(manifest["iteration"])
    primary.header["HIERARCH SCI.ARRAY"] = array
    primary.header["HIERARCH SCI.LINEAR"] = False
    primary.header["HIERARCH SCI.CALIBRATED"] = False
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [primary]
    for name in COMPONENTS:
        extension = FITS_EXTENSION_NAMES[name]
        component_header = header.copy()
        component_header["BUNIT"] = "mJy/beam"
        component_header["HIERARCH SCI.COMPONENT"] = name
        hdus.append(
            fits.ImageHDU(
                components[name].astype("float64"),
                header=component_header,
                name=extension,
            )
        )
        residual_header = header.copy()
        residual_header["BUNIT"] = "mJy/beam"
        residual_header["HIERARCH SCI.RESIDUAL_OF"] = name
        hdus.append(
            fits.ImageHDU(
                residuals[name].astype("float64"),
                header=residual_header,
                name=extension.replace("_", "")[:5] + "RES",
            )
        )
    output = output_dir / f"point_{obsnum}_{array}_el_f7_components.fits"
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=False)
    return output


def world_axes(path: Path, extension: str) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[extension]
        shape = np.asarray(hdu.data).squeeze().shape
        header = hdu.header
    units = (str(header["CUNIT1"]).lower(), str(header["CUNIT2"]).lower())
    scales = tuple(3600.0 if unit.startswith("deg") else 1.0 for unit in units)
    x = (
        float(header["CRVAL1"]) * scales[0]
        + (np.arange(shape[1]) + 1.0 - float(header["CRPIX1"]))
        * float(header["CDELT1"]) * scales[0]
    )
    y = (
        float(header["CRVAL2"]) * scales[1]
        + (np.arange(shape[0]) + 1.0 - float(header["CRPIX2"]))
        * float(header["CDELT2"]) * scales[1]
    )
    return x, y


def write_plot(
    path: Path,
    array_components: dict[str, dict[str, np.ndarray]],
    source_paths: dict[str, Path],
    manifest: dict,
) -> None:
    import matplotlib.pyplot as plt

    injection = np.asarray(manifest["injection_position_fits_world_arcsec"], dtype=float)
    neptune = np.asarray(manifest["neptune_position_fits_world_arcsec"], dtype=float)
    figure, axes = plt.subplots(3, 4, figsize=(15.5, 11.0), sharex=True, sharey=True)
    titles = {
        "T5_total_adaptive": "T5 total adaptive",
        "S5_shared_start": "S5 shared start",
        "H5_other_history": "H5 other history",
        "D4460_5": "D5 UID 4460",
    }
    for row, array in enumerate(ARRAYS):
        x, y = world_axes(source_paths[array], "signal_I")
        x = x - injection[0]
        y = y - injection[1]
        for column, component in enumerate(COMPONENTS):
            values = array_components[array][component]
            if x[0] > x[-1]:
                shown = values[:, ::-1]
                shown_x = x[::-1]
            else:
                shown = values
                shown_x = x
            if y[0] > y[-1]:
                shown = shown[::-1, :]
                shown_y = y[::-1]
            else:
                shown_y = y
            xx, yy = np.meshgrid(shown_x, shown_y)
            radial = np.hypot(xx, yy) <= 120.0
            finite = np.isfinite(shown) & radial
            limit = float(np.nanpercentile(np.abs(shown[finite]), 99.0))
            if not math.isfinite(limit) or limit == 0.0:
                limit = 1.0
            dx = float(abs(shown_x[1] - shown_x[0]))
            dy = float(abs(shown_y[1] - shown_y[0]))
            image = axes[row, column].imshow(
                shown,
                origin="lower",
                extent=(
                    float(shown_x[0] - dx / 2.0),
                    float(shown_x[-1] + dx / 2.0),
                    float(shown_y[0] - dy / 2.0),
                    float(shown_y[-1] + dy / 2.0),
                ),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            axes[row, column].add_patch(
                plt.Circle((0.0, 0.0), 20.0, fill=False, color="black", linewidth=0.8)
            )
            axes[row, column].add_patch(
                plt.Circle(
                    tuple(neptune - injection),
                    20.0,
                    fill=False,
                    color="#c58b00",
                    linewidth=0.8,
                )
            )
            axes[row, column].set_xlim(-120.0, 120.0)
            axes[row, column].set_ylim(-120.0, 120.0)
            axes[row, column].grid(alpha=0.18)
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
            if row == 0:
                axes[row, column].set_title(titles[component])
            if column == 0:
                axes[row, column].set_ylabel(f"{array}\nEL offset (arcsec)")
            if row == 2:
                axes[row, column].set_xlabel("AZ offset (arcsec)")
    figure.suptitle(
        "EL-F7 iteration-5 response decomposition\n"
        "coordinates relative to injected source; panel scales are independent"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze(manifest: dict) -> tuple[list[dict], list[dict], dict, dict, dict]:
    obsnum = int(manifest["obsnum"])
    iteration = int(manifest["iteration"])
    truth_by_array = dict(
        zip(
            manifest["array_order"],
            (float(value) for value in manifest["amplitudes_mjy_beam"]),
        )
    )
    sham = validate_sham(manifest)
    sham_redu = Path(sham["sham_iteration_directory"])
    probe_redu = iteration_dir(
        Path(manifest["shared_start_probe_root"]), obsnum, iteration
    )
    require_paired_configs(sham_redu, probe_redu, manifest)

    reductions = {
        "C5": Path(manifest["existing_control_iteration_5"]),
        "A5": Path(manifest["existing_adaptive_iteration_5"]),
        "N5": Path(manifest["existing_without_uid4460_iteration_5"]),
        "P5": probe_redu,
    }
    metrics = []
    cross_terms = []
    closure = {}
    array_components: dict[str, dict[str, np.ndarray]] = {}
    array_residuals: dict[str, dict[str, np.ndarray]] = {}
    source_paths: dict[str, Path] = {}
    for array in ARRAYS:
        paths = {
            label: product_path(redu, obsnum, array)
            for label, redu in reductions.items()
        }
        if any(fruit_iteration(path) != iteration for path in paths.values()):
            raise ValueError(f"EL-F7 product iteration differs for {array}")
        loaded: dict[str, np.ndarray] = {}
        grids: dict[str, tuple] = {}
        supports = []
        for extension in IMAGE_EXTENSIONS:
            for label, product in paths.items():
                values, grid = load_image(product, extension)
                loaded[f"{label}_{extension}"] = values
                previous = grids.setdefault(extension, grid)
                if grid != previous:
                    raise ValueError(
                        f"EL-F7 WCS/grid differs: {array}:{extension}:{label}"
                    )
            extension_values = {
                label: loaded[f"{label}_{extension}"] for label in reductions
            }
            supports.append(
                common_support(
                    extension_values,
                    context=f"EL-F7 array={array} extension={extension}",
                )
            )
        support = np.logical_and.reduce(supports)
        if not support.any():
            raise ValueError(f"EL-F7 common support is empty for {array}")
        components = {
            "T5_total_adaptive": loaded["A5_signal_I"] - loaded["C5_signal_I"],
            "S5_shared_start": loaded["P5_signal_I"] - loaded["C5_signal_I"],
            "H5_other_history": loaded["N5_signal_I"] - loaded["P5_signal_I"],
            "D4460_5": loaded["A5_signal_I"] - loaded["N5_signal_I"],
        }
        closure[array] = closure_metrics(
            components["T5_total_adaptive"],
            components["S5_shared_start"],
            components["H5_other_history"],
            components["D4460_5"],
            support,
            float(manifest["closure_roundoff_factor"]),
        )
        if not bool(closure[array]["closure_pass"]):
            raise ValueError(f"EL-F7 telescoping closure failed for {array}")
        injection_center = gaussian_center_for_map_world_offset(
            paths["P5"],
            "signal_I",
            *[float(value) for value in manifest["injection_position_fits_world_arcsec"]],
        )
        neptune_center = gaussian_center_for_map_world_offset(
            paths["P5"],
            "signal_I",
            *[float(value) for value in manifest["neptune_position_fits_world_arcsec"]],
        )
        regions = make_region_masks(
            components["T5_total_adaptive"].shape,
            float(manifest["pixel_size_arcsec"]),
            injection_center,
            neptune_center,
            manifest,
        )
        component_rows, residuals = component_metrics(
            array,
            components,
            loaded["P5_kernel_I"],
            {
                "T5_total_adaptive": loaded["A5_kernel_I"],
                "S5_shared_start": loaded["P5_kernel_I"],
            },
            support,
            regions,
            injection_center,
            manifest,
            truth_by_array[array],
        )
        metrics.extend(component_rows)
        cross_terms.extend(cross_term_rows(array, components, support, regions))
        array_components[array] = components
        array_residuals[array] = residuals
        source_paths[array] = paths["P5"]

    checkpoint = checkpoint_difference(
        reductions["C5"] / "citlali_restart_checkpoint.nc",
        reductions["P5"] / "citlali_restart_checkpoint.nc",
    )
    rediscovered = target_penalty_present(
        reductions["P5"] / "citlali_restart_checkpoint.nc",
        manifest["target_penalty"],
        iteration,
    )
    execution = read_execution(Path(manifest["execution_log_dir"]))
    result = {
        "test_id": manifest["test_id"],
        "valid_decomposition": True,
        **sham,
        "paired_realized_configuration_check": "PASS",
        "common_units_wcs_grid_normalization_and_finite_support": "PASS",
        "closure": closure,
        "shared_start_probe_uid4460_learned_at_end_iteration_5": rediscovered,
        "checkpoint_differing_variable_count": checkpoint[
            "differing_variable_count"
        ],
        "execution": {
            "trajectory_count": len(execution),
            "aggregate_wall_seconds": sum(
                float(row["wall_seconds"]) for row in execution
            ),
            "maximum_resident_bytes": max(
                int(row["maximum_resident_bytes"]) for row in execution
            ),
        },
        "response_identity": {
            "T5_total_adaptive": "A5-C5",
            "S5_shared_start": "P5-C5",
            "H5_other_history": "N5-P5",
            "D4460_5": "A5-N5",
        },
        "claim_scope": (
            "one shared-start transition in observation 123424; descriptive "
            "development evidence only"
        ),
        "fully_matched_operator_transfer_established": False,
        "method_or_safeguard_selected": False,
    }
    auxiliary = {
        "execution": execution,
        "array_components": array_components,
        "array_residuals": array_residuals,
        "source_paths": source_paths,
    }
    return metrics, cross_terms, result, checkpoint, auxiliary


def metric_row(rows: list[dict], array: str, component: str) -> dict:
    return next(
        row for row in rows
        if row["array"] == array and row["component"] == component
    )


def report_text(metrics: list[dict], result: dict) -> str:
    lines = [
        "# FRUIT EL-F7 shared-start response result",
        "",
        "Result: **valid shared-start decomposition; descriptive development "
        "evidence only**",
        "",
        f"Test ID: `{result['test_id']}`",
        "",
        "## Validity",
        "",
        f"- exact sham image planes: `{result['sham_exact_planes']}`;",
        "- complete sham checkpoint: value-identical;",
        "- paired realized configuration check: `PASS`;",
        "- common units, WCS/grid, normalization, and finite support: `PASS`;",
        "- telescoping closure: `PASS` in all arrays; and",
        "- unexpected error/critical messages: zero.",
        "",
        "## Compact response",
        "",
        "| Array | T5 central recovery | S5 central recovery | T5 width major/minor | S5 width major/minor |",
        "|---|---:|---:|---:|---:|",
    ]
    for array in ARRAYS:
        total = metric_row(metrics, array, "T5_total_adaptive")
        shared = metric_row(metrics, array, "S5_shared_start")
        lines.append(
            f"| {array} | {float(total['kernel_normalized_central_recovery']):.6f} "
            f"| {float(shared['kernel_normalized_central_recovery']):.6f} "
            f"| {float(total['major_fwhm_over_kernel']):.4f} / "
            f"{float(total['minor_fwhm_over_kernel']):.4f} "
            f"| {float(shared['major_fwhm_over_kernel']):.4f} / "
            f"{float(shared['minor_fwhm_over_kernel']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Where the response structure appears",
            "",
            "The table reports RMS in mJy/beam. These are continuous descriptive "
            "measurements; no dominance threshold was registered.",
            "",
            "| Array | Region | S5 shared start | H5 other history | D5 UID 4460 | T5 total |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    region_labels = {
        "injected_source_r20": "injected source r<20",
        "neptune_r20": "Neptune r<20",
        "annulus_r40_120_excluding_neptune_r25": "annulus 40-120 excl. Neptune",
    }
    for array in ARRAYS:
        by_component = {
            name: metric_row(metrics, array, name) for name in COMPONENTS
        }
        for region, label in region_labels.items():
            key = f"{region}_rms_mjy_beam"
            lines.append(
                f"| {array} | {label} "
                f"| {float(by_component['S5_shared_start'][key]):.6g} "
                f"| {float(by_component['H5_other_history'][key]):.6g} "
                f"| {float(by_component['D4460_5'][key]):.6g} "
                f"| {float(by_component['T5_total_adaptive'][key]):.6g} |"
            )
    lines.extend(
        [
            "",
            "Pairwise inner products, cross terms, complete maps, fixed-kernel "
            "residuals, and all checkpoint differences are retained in the "
            "machine-readable result artifacts.",
            "",
            "## Interpretation boundary",
            "",
            "`S5` is the cleanest source response available in this observation "
            "because both branches entered iteration 5 with identical state. It "
            "is still a shared-incoming-state one-step response, not a fully "
            "matched-operator transfer function: the injected data may change "
            "processing within iteration 5.",
            "",
            "`H5` groups all earlier injected-history state other than the removed "
            "UID 4460 record, and `D5` is the EL-F6 penalty intervention effect. "
            "The exact sum does not imply independent calibration, orthogonality, "
            "linearity, or interchangeable intervention order.",
            "",
            "This result does not select a safeguard, penalty policy, recurrence, "
            "method, stopping rule, qualification route, or production default.",
            "",
        ]
    )
    return "\n".join(lines)


def write_provenance(
    path: Path,
    manifest_path: Path,
    manifest: dict,
    result_paths: list[Path],
    auxiliary: dict,
) -> None:
    obsnum = int(manifest["obsnum"])
    iteration = int(manifest["iteration"])
    input_paths = {
        Path(__file__).resolve(),
        manifest_path.resolve(),
        Path(manifest["registration"]).resolve()
        if Path(manifest["registration"]).is_absolute()
        else (manifest_path.parent / manifest["registration"]).resolve(),
    }
    for key in (
        "source_control_iteration_4",
        "existing_control_iteration_5",
        "existing_adaptive_iteration_5",
        "existing_without_uid4460_iteration_5",
    ):
        redu = Path(manifest[key])
        input_paths.add((redu / "citlali_restart_checkpoint.nc").resolve())
        for array in ARRAYS:
            input_paths.add(product_path(redu, obsnum, array).resolve())
    for root_key in ("control_sham_root", "shared_start_probe_root"):
        redu = iteration_dir(Path(manifest[root_key]), obsnum, iteration)
        input_paths.add((redu / "citlali_restart_checkpoint.nc").resolve())
        input_paths.add((redu / "citlali_merged_config.yaml").resolve())
        for array in ARRAYS:
            input_paths.add(product_path(redu, obsnum, array).resolve())
    for log in Path(manifest["execution_log_dir"]).glob("*.log"):
        input_paths.add(log.resolve())
    payload = {
        "schema_version": "sci-fruit-el-f7-provenance-v1",
        "test_id": manifest["test_id"],
        "role": "exploratory-development-response-clarification-only",
        "qualification_use_authorized": False,
        "inputs": [
            file_record(item) for item in sorted(input_paths, key=str)
        ],
        "outputs": [
            file_record(item.resolve(), relative_to=path.parent.resolve())
            for item in sorted(result_paths, key=str)
        ],
        "execution": auxiliary["execution"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-sham-only", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    if args.validate_sham_only:
        print(json.dumps(validate_sham(manifest), indent=2, sort_keys=True))
        return 0

    metrics, cross_terms, result, checkpoint, auxiliary = analyze(manifest)
    output_dir = Path(manifest["analysis_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "COMPONENT_METRICS_R0.1.csv"
    cross_path = output_dir / "CROSS_TERMS_R0.1.csv"
    execution_path = output_dir / "PRIMARY_EXECUTION_R0.1.csv"
    checkpoint_path = output_dir / "CHECKPOINT_DIFFERENCES_R0.1.json"
    result_path = output_dir / "DECOMPOSITION_RESULT_R0.1.json"
    report_path = output_dir / "EXECUTION_RESULT_R0.1.md"
    plot_path = output_dir / "RESPONSE_DECOMPOSITION_R0.1.png"
    map_dir = output_dir / "component-maps"

    write_csv(metrics_path, metrics)
    write_csv(cross_path, cross_terms)
    write_csv(execution_path, auxiliary["execution"])
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
    )
    component_products = []
    for array in ARRAYS:
        component_products.append(
            write_component_maps(
                map_dir,
                int(manifest["obsnum"]),
                array,
                auxiliary["source_paths"][array],
                auxiliary["array_components"][array],
                auxiliary["array_residuals"][array],
                manifest,
            )
        )
    write_plot(
        plot_path,
        auxiliary["array_components"],
        auxiliary["source_paths"],
        manifest,
    )
    result["complete_component_products"] = [
        file_record(item.resolve()) for item in component_products
    ]
    result["component_metrics"] = [file_record(metrics_path.resolve())]
    result["cross_terms"] = [file_record(cross_path.resolve())]
    result["checkpoint_differences"] = file_record(checkpoint_path.resolve())
    result["response_figure"] = file_record(plot_path.resolve())
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path.write_text(report_text(metrics, result))
    provenance_path = output_dir / "ANALYSIS_PROVENANCE_R0.1.yaml"
    result_paths = [
        metrics_path,
        cross_path,
        execution_path,
        checkpoint_path,
        result_path,
        report_path,
        plot_path,
        *component_products,
    ]
    write_provenance(
        provenance_path,
        manifest_path,
        manifest,
        result_paths,
        auxiliary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
