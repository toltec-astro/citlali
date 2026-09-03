#!/usr/bin/env python3
"""Analyze the registered SCI-FRUIT EL-F6 off-source causal replay."""

from __future__ import annotations

import argparse
import csv
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
    annulus_mask,
    common_support,
    load_image,
)
from tools.fruit_loops.compare_injected_source_pair import (
    ARRAYS,
    file_record,
    gaussian_center_for_map_world_offset,
    gaussian_fit,
    kernel_projection_metrics,
    product_path,
    rms,
)
from tools.fruit_loops.edit_restart_checkpoint_penalty import values_equal


def require_exact_checkpoint(expected_path: Path, actual_path: Path) -> None:
    """Require NetCDF structure, attributes, and every value to match."""
    with Dataset(expected_path) as expected, Dataset(actual_path) as actual:
        if expected.ncattrs() != actual.ncattrs():
            raise ValueError("sham checkpoint global attribute names differ")
        for attribute in expected.ncattrs():
            if expected.getncattr(attribute) != actual.getncattr(attribute):
                raise ValueError(
                    f"sham checkpoint global attribute differs: {attribute}"
                )
        if set(expected.dimensions) != set(actual.dimensions):
            raise ValueError("sham checkpoint dimension names differ")
        for name, dimension in expected.dimensions.items():
            if len(dimension) != len(actual.dimensions[name]):
                raise ValueError(
                    f"sham checkpoint dimension length differs: {name}"
                )
        if set(expected.variables) != set(actual.variables):
            raise ValueError("sham checkpoint variable names differ")
        for name, expected_variable in expected.variables.items():
            actual_variable = actual.variables[name]
            if expected_variable.dimensions != actual_variable.dimensions:
                raise ValueError(
                    f"sham checkpoint variable dimensions differ: {name}"
                )
            if expected_variable.dtype != actual_variable.dtype:
                raise ValueError(f"sham checkpoint type differs: {name}")
            if expected_variable.ncattrs() != actual_variable.ncattrs():
                raise ValueError(
                    f"sham checkpoint attribute names differ: {name}"
                )
            for attribute in expected_variable.ncattrs():
                if expected_variable.getncattr(
                    attribute
                ) != actual_variable.getncattr(attribute):
                    raise ValueError(
                        "sham checkpoint variable attribute differs: "
                        f"{name}:{attribute}"
                    )
            if not values_equal(expected_variable[...], actual_variable[...]):
                raise ValueError(
                    f"sham checkpoint variable values differ: {name}"
                )


def require_exact_maps(
    expected_redu: Path,
    actual_redu: Path,
    obsnum: int,
    arrays: tuple[str, ...] = ARRAYS,
) -> int:
    """Require bitwise image planes and matching grids for selected arrays."""
    compared = 0
    for array in arrays:
        expected_path = product_path(expected_redu, obsnum, array)
        actual_path = product_path(actual_redu, obsnum, array)
        for extension in IMAGE_EXTENSIONS:
            _, expected_grid = load_image(expected_path, extension)
            _, actual_grid = load_image(actual_path, extension)
            if expected_grid != actual_grid:
                raise ValueError(
                    f"map grid differs: {array}:{extension}"
                )
            with fits.open(expected_path, memmap=True) as expected_hdul, fits.open(
                actual_path, memmap=True
            ) as actual_hdul:
                expected = np.asarray(expected_hdul[extension].data).squeeze()
                actual = np.asarray(actual_hdul[extension].data).squeeze()
            if (
                expected.shape != actual.shape
                or expected.dtype != actual.dtype
                or expected.tobytes() != actual.tobytes()
            ):
                raise ValueError(
                    f"map values are not bitwise equal: {array}:{extension}"
                )
            compared += 1
    return compared


def fruit_iteration(path: Path) -> int:
    with fits.open(path, memmap=True) as hdul:
        return int(hdul[0].header["FRUITLOOPS_ITER"])


def measure_pair(
    control_redu: Path,
    injected_redu: Path,
    manifest: dict,
    *,
    iteration: int,
    label: str,
) -> list[dict]:
    """Measure one off-source injected-minus-control response."""
    obsnum = int(manifest["obsnum"])
    amplitudes = dict(
        zip(
            manifest["array_order"],
            (float(value) for value in manifest["amplitudes_mjy_beam"]),
        )
    )
    pixel_size = float(manifest["pixel_size_arcsec"])
    az_offset = float(manifest["az_offset_arcsec"])
    el_offset = float(manifest["el_offset_arcsec"])
    search_radius = float(manifest["gaussian_search_radius_arcsec"])
    annulus_inner = float(manifest["annulus_inner_arcsec"])
    annulus_outer = float(manifest["annulus_outer_arcsec"])
    rows = []
    for array in ARRAYS:
        control_path = product_path(control_redu, obsnum, array)
        injected_path = product_path(injected_redu, obsnum, array)
        if (
            fruit_iteration(control_path) != iteration
            or fruit_iteration(injected_path) != iteration
        ):
            raise ValueError(
                f"{label} {array} product is not absolute iteration {iteration}"
            )
        loaded: dict[str, np.ndarray] = {}
        grids: dict[str, tuple] = {}
        for variant, path in (("control", control_path), ("injected", injected_path)):
            for extension in IMAGE_EXTENSIONS:
                values, grid = load_image(path, extension)
                loaded[f"{variant}_{extension}"] = values
                previous = grids.setdefault(extension, grid)
                if grid != previous:
                    raise ValueError(
                        f"paired grid differs: {label}:{array}:{extension}"
                    )
        support = common_support(
            loaded, context=f"EL-F6 {label} iteration={iteration} array={array}"
        )
        transfer = loaded["injected_signal_I"] - loaded["control_signal_I"]
        kernel = loaded["injected_kernel_I"]
        expected_center = gaussian_center_for_map_world_offset(
            injected_path, "signal_I", az_offset, el_offset
        )
        transfer_fit = gaussian_fit(
            transfer,
            pixel_size,
            expected_center_arcsec=expected_center,
            search_radius_arcsec=search_radius,
        )
        kernel_fit = gaussian_fit(
            kernel,
            pixel_size,
            expected_center_arcsec=expected_center,
            search_radius_arcsec=search_radius,
        )
        truth = amplitudes[array]
        projection = kernel_projection_metrics(transfer, kernel, truth)
        residual = transfer - float(projection["scale_mjy_beam"]) * kernel
        annulus = annulus_mask(
            transfer.shape,
            pixel_size,
            annulus_inner,
            annulus_outer,
            center_arcsec=expected_center,
        )
        annulus_support = support & annulus
        if not annulus_support.any():
            raise ValueError(f"empty EL-F6 annulus for {label}:{array}")
        central_recovery = transfer_fit["amplitude"] / (
            truth * kernel_fit["amplitude"]
        )
        rows.append(
            {
                "variant": label,
                "iteration": iteration,
                "array": array,
                "kernel_normalized_central_recovery": central_recovery,
                "central_recovery_absolute_error": abs(central_recovery - 1.0),
                "full_kernel_recovery": projection["recovery_fraction"],
                "full_kernel_recovery_absolute_error": abs(
                    projection["recovery_fraction"] - 1.0
                ),
                "major_fwhm_over_kernel": (
                    transfer_fit["major_fwhm_arcsec"]
                    / kernel_fit["major_fwhm_arcsec"]
                ),
                "minor_fwhm_over_kernel": (
                    transfer_fit["minor_fwhm_arcsec"]
                    / kernel_fit["minor_fwhm_arcsec"]
                ),
                "centroid_error_arcsec": math.hypot(
                    transfer_fit["x_arcsec"] - kernel_fit["x_arcsec"],
                    transfer_fit["y_arcsec"] - kernel_fit["y_arcsec"],
                ),
                "kernel_residual_relative_rms": projection[
                    "residual_relative_rms"
                ],
                "annular_residual_over_truth": (
                    rms(residual[annulus_support]) / truth
                ),
                "common_valid_pixels": int(np.count_nonzero(support)),
                "total_pixels": int(support.size),
            }
        )
    return rows


def reversal_fraction(
    counterfactual: float, preceding: float, failed: float
) -> float:
    denominator = failed - preceding
    if denominator <= 0.0:
        raise ValueError("registered original values do not define a loss")
    return (failed - counterfactual) / denominator


def classify_effect(
    kernel_fraction: float, annular_fraction: float, minimum: float = 0.5
) -> str:
    if kernel_fraction >= minimum and annular_fraction >= minimum:
        return "substantial_causal_contribution"
    if kernel_fraction > 0.0 and annular_fraction > 0.0:
        return "partial_causal_contribution"
    if kernel_fraction <= 0.0 and annular_fraction <= 0.0:
        return "no_support_for_causal_contribution"
    return "mixed_effect"


def read_execution(log_dir: Path) -> list[dict]:
    rows = []
    error_pattern = re.compile(r"\[(?:error|critical)\]|(?:error|critical):", re.I)
    for label in ("untouched-injected-sham", "injected-without-uid4460"):
        path = log_dir / f"{label}.log"
        text = path.read_text(encoding="utf-8")
        wall = re.search(r"^\s*([0-9.]+) real\s", text, re.M)
        user = re.search(r"^\s*([0-9.]+) user\s", text, re.M)
        system = re.search(r"^\s*([0-9.]+) sys\s", text, re.M)
        rss = re.search(r"^\s*([0-9]+)\s+maximum resident set size$", text, re.M)
        errors = sum(bool(error_pattern.search(line)) for line in text.splitlines())
        if (
            wall is None
            or user is None
            or system is None
            or rss is None
            or "citlali is done!" not in text
            or errors
        ):
            raise ValueError(f"incomplete or unsuccessful execution log: {path}")
        rows.append(
            {
                "trajectory": label,
                "status": "completed",
                "wall_seconds": float(wall.group(1)),
                "user_seconds": float(user.group(1)),
                "system_seconds": float(system.group(1)),
                "maximum_resident_bytes": int(rss.group(1)),
                "error_or_critical_messages": errors,
            }
        )
    return rows


def target_penalty_present(checkpoint: Path, expected: dict, iteration: int) -> bool:
    selector = dict(expected)
    selector["iteration"] = iteration
    variable_names = {
        "producer": "penalty_producer",
        "reason": "penalty_reason",
        "iteration": "penalty_iteration",
        "scan": "penalty_scan",
        "uid": "penalty_uid",
        "network": "penalty_network",
        "array": "penalty_array",
        "factor": "penalty_factor",
        "score": "penalty_score",
        "scan_local": "penalty_scan_local",
    }
    with Dataset(checkpoint) as dataset:
        count = len(dataset.dimensions["effective_detector_penalty"])
        for index in range(count):
            matched = True
            for key, variable in variable_names.items():
                actual = dataset.variables[variable][index]
                actual = actual.item() if hasattr(actual, "item") else actual
                expected_value = selector[key]
                if key in {"factor", "score"}:
                    matched &= math.isclose(
                        float(actual), float(expected_value), rel_tol=0.0, abs_tol=1e-12
                    )
                elif key == "scan_local":
                    matched &= bool(actual) is bool(expected_value)
                else:
                    matched &= actual == expected_value
            if matched:
                return True
    return False


def write_response_maps(
    control_redu: Path,
    injected_redu: Path,
    output_dir: Path,
    manifest: dict,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    obsnum = int(manifest["obsnum"])
    records = []
    for array in ARRAYS:
        control_path = product_path(control_redu, obsnum, array)
        injected_path = product_path(injected_redu, obsnum, array)
        with fits.open(control_path, memmap=True) as control, fits.open(
            injected_path, memmap=True
        ) as injected:
            response = injected["signal_I"].data.astype("float64") - control[
                "signal_I"
            ].data.astype("float64")
            header = injected["signal_I"].header.copy()
            for key in ("CHECKSUM", "DATASUM"):
                if key in header:
                    del header[key]
        primary = fits.PrimaryHDU()
        primary.header["HIERARCH SCI.TESTID"] = manifest["test_id"]
        primary.header["HIERARCH SCI.RESPONSE"] = "counterfactual-control"
        primary.header["HIERARCH SCI.OBSNUM"] = obsnum
        primary.header["HIERARCH SCI.FRUIT_ITER"] = 5
        primary.header["HIERARCH SCI.ARRAY"] = array
        response_hdu = fits.ImageHDU(response, header=header, name="RESPONSE_I")
        output = output_dir / f"point_{obsnum}_{array}_fruit_iter_05_response.fits"
        fits.HDUList([primary, response_hdu]).writeto(
            output, overwrite=True, checksum=False
        )
        records.append(file_record(output.resolve()))
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(manifest: dict, response_dir: Path) -> tuple[list[dict], list[dict], dict]:
    obsnum = int(manifest["obsnum"])
    original_injected_5 = Path(manifest["original_injected_iteration_5"])
    sham_5 = Path(manifest["sham_iteration_5"])
    candidate_5 = Path(manifest["counterfactual_iteration_5"])
    control_4 = Path(manifest["control_iteration_4"])
    control_5 = Path(manifest["control_iteration_5"])
    original_injected_4 = Path(manifest["original_injected_iteration_4"])

    sham_planes = require_exact_maps(original_injected_5, sham_5, obsnum)
    require_exact_checkpoint(
        original_injected_5 / "citlali_restart_checkpoint.nc",
        sham_5 / "citlali_restart_checkpoint.nc",
    )
    unchanged_planes = require_exact_maps(
        original_injected_5, candidate_5, obsnum, ("a1100", "a2000")
    )

    audit = json.loads(Path(manifest["intervention_audit"]).read_text())
    removed = audit["transformation"]["removed_record"]
    for key, value in manifest["expected_removed_penalty"].items():
        if removed[key] != value:
            raise ValueError(
                f"intervention audit differs for {key}: expected={value} actual={removed[key]}"
            )
    if not audit["transformation"]["all_other_values_verified_equal"]:
        raise ValueError("intervention audit did not verify all other values")

    metrics = []
    metrics.extend(
        measure_pair(
            control_4,
            original_injected_4,
            manifest,
            iteration=4,
            label="original_iteration_4",
        )
    )
    metrics.extend(
        measure_pair(
            control_5,
            original_injected_5,
            manifest,
            iteration=5,
            label="original_iteration_5",
        )
    )
    metrics.extend(
        measure_pair(
            control_5,
            candidate_5,
            manifest,
            iteration=5,
            label="counterfactual_iteration_5",
        )
    )
    a1400 = {
        row["variant"]: row
        for row in metrics
        if row["array"] == "a1400"
    }
    frozen = manifest["original_a1400"]
    original_checks = {
        "iteration_4_kernel_residual_relative_rms": a1400[
            "original_iteration_4"
        ]["kernel_residual_relative_rms"],
        "iteration_5_kernel_residual_relative_rms": a1400[
            "original_iteration_5"
        ]["kernel_residual_relative_rms"],
        "iteration_4_annular_residual_over_truth": a1400[
            "original_iteration_4"
        ]["annular_residual_over_truth"],
        "iteration_5_annular_residual_over_truth": a1400[
            "original_iteration_5"
        ]["annular_residual_over_truth"],
        "iteration_4_central_recovery": a1400["original_iteration_4"][
            "kernel_normalized_central_recovery"
        ],
        "iteration_5_central_recovery": a1400["original_iteration_5"][
            "kernel_normalized_central_recovery"
        ],
        "iteration_4_full_kernel_recovery": a1400["original_iteration_4"][
            "full_kernel_recovery"
        ],
        "iteration_5_full_kernel_recovery": a1400["original_iteration_5"][
            "full_kernel_recovery"
        ],
    }
    for key, actual in original_checks.items():
        if not math.isclose(
            float(actual), float(frozen[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"frozen original metric differs: {key}")

    candidate = a1400["counterfactual_iteration_5"]
    kernel_reversal = reversal_fraction(
        float(candidate["kernel_residual_relative_rms"]),
        float(frozen["iteration_4_kernel_residual_relative_rms"]),
        float(frozen["iteration_5_kernel_residual_relative_rms"]),
    )
    annular_reversal = reversal_fraction(
        float(candidate["annular_residual_over_truth"]),
        float(frozen["iteration_4_annular_residual_over_truth"]),
        float(frozen["iteration_5_annular_residual_over_truth"]),
    )
    minimum = float(manifest["minimum_reversal_fraction"])
    full = float(manifest["full_reversal_fraction"])
    execution = read_execution(Path(manifest["execution_log_dir"]))
    response_products = write_response_maps(
        control_5, candidate_5, response_dir, manifest
    )
    rediscovered = target_penalty_present(
        candidate_5 / "citlali_restart_checkpoint.nc",
        manifest["expected_removed_penalty"],
        5,
    )
    result = {
        "test_id": manifest["test_id"],
        "valid_counterfactual": True,
        "sham_replay_exact": True,
        "sham_exact_planes": sham_planes,
        "sham_checkpoint_all_variables_value_identical": True,
        "intervention_audit_pass": True,
        "unchanged_nontarget_planes": unchanged_planes,
        "a1400_kernel_residual_reversal_fraction": kernel_reversal,
        "a1400_annular_residual_reversal_fraction": annular_reversal,
        "full_reversal": kernel_reversal >= full and annular_reversal >= full,
        "mechanism_classification": classify_effect(
            kernel_reversal, annular_reversal, minimum
        ),
        "a1400_counterfactual_iteration_5": candidate,
        "a1400_central_recovery_error_change_from_original_iteration_5": (
            float(candidate["central_recovery_absolute_error"])
            - float(
                a1400["original_iteration_5"][
                    "central_recovery_absolute_error"
                ]
            )
        ),
        "a1400_full_kernel_recovery_error_change_from_original_iteration_5": (
            float(candidate["full_kernel_recovery_absolute_error"])
            - float(
                a1400["original_iteration_5"][
                    "full_kernel_recovery_absolute_error"
                ]
            )
        ),
        "target_penalty_rediscovered_at_iteration_5": rediscovered,
        "complete_counterfactual_response_products": response_products,
        "execution": {
            "trajectory_count": len(execution),
            "aggregate_wall_seconds": sum(row["wall_seconds"] for row in execution),
            "maximum_resident_bytes": max(
                row["maximum_resident_bytes"] for row in execution
            ),
        },
        "claim_scope": "one off-source checkpoint intervention in observation 123424",
    }
    return metrics, execution, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--response-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    metrics, execution, result = analyze(manifest, args.response_dir)
    write_csv(args.metrics, metrics)
    write_csv(args.execution, execution)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
