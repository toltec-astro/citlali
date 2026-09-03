#!/usr/bin/env python3
"""Analyze the registered SCI-FRUIT EL-F3 checkpoint intervention."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    gaussian_fit,
    kernel_projection_metrics,
    product_path,
    rms,
)
from tools.fruit_loops.edit_restart_checkpoint_penalty import values_equal


def read_metrics(path: Path) -> dict[tuple[float, int, str], dict[str, float]]:
    result: dict[tuple[float, int, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (float(row["alpha"]), int(row["iteration"]), row["array"])
            result[key] = {
                name: float(row[name])
                for name in (
                    "kernel_normalized_central_recovery",
                    "full_kernel_recovery",
                    "major_fwhm_over_kernel",
                    "minor_fwhm_over_kernel",
                    "centroid_error_arcsec",
                    "kernel_residual_relative_rms",
                    "annular_residual_over_truth",
                )
            }
    return result


def require_exact_checkpoint(expected_path: Path, actual_path: Path) -> None:
    with Dataset(expected_path) as expected, Dataset(actual_path) as actual:
        if expected.ncattrs() != actual.ncattrs():
            raise ValueError("control checkpoint global attribute names differ")
        for attribute in expected.ncattrs():
            if expected.getncattr(attribute) != actual.getncattr(attribute):
                raise ValueError(
                    f"control checkpoint global attribute differs: {attribute}"
                )
        if set(expected.dimensions) != set(actual.dimensions):
            raise ValueError("control checkpoint dimension names differ")
        for name, dimension in expected.dimensions.items():
            if len(dimension) != len(actual.dimensions[name]):
                raise ValueError(
                    f"control checkpoint dimension length differs: {name}"
                )
        if set(expected.variables) != set(actual.variables):
            raise ValueError("control checkpoint variable names differ")
        for name, expected_variable in expected.variables.items():
            actual_variable = actual.variables[name]
            if expected_variable.dimensions != actual_variable.dimensions:
                raise ValueError(
                    f"control checkpoint variable dimensions differ: {name}"
                )
            if expected_variable.dtype != actual_variable.dtype:
                raise ValueError(f"control checkpoint type differs: {name}")
            if expected_variable.ncattrs() != actual_variable.ncattrs():
                raise ValueError(
                    f"control checkpoint attribute names differ: {name}"
                )
            for attribute in expected_variable.ncattrs():
                if expected_variable.getncattr(
                    attribute
                ) != actual_variable.getncattr(attribute):
                    raise ValueError(
                        "control checkpoint variable attribute differs: "
                        f"{name}:{attribute}"
                    )
            if not values_equal(expected_variable[...], actual_variable[...]):
                raise ValueError(
                    f"control checkpoint variable values differ: {name}"
                )


def require_exact_control_maps(
    expected_redu: Path,
    actual_redu: Path,
    obsnum: int,
) -> None:
    for array in ARRAYS:
        expected_path = product_path(expected_redu, obsnum, array)
        actual_path = product_path(actual_redu, obsnum, array)
        for extension in IMAGE_EXTENSIONS:
            expected, expected_grid = load_image(expected_path, extension)
            actual, actual_grid = load_image(actual_path, extension)
            if expected_grid != actual_grid:
                raise ValueError(
                    f"control map grid differs: {array}:{extension}"
                )
            if not np.array_equal(expected, actual, equal_nan=True):
                raise ValueError(
                    f"control map values differ: {array}:{extension}"
                )


def fruit_iteration(path: Path) -> int:
    with fits.open(path, memmap=True) as hdul:
        return int(hdul[0].header["FRUITLOOPS_ITER"])


def measure_pair(
    control_redu: Path,
    injected_redu: Path,
    obsnum: int,
    amplitudes: dict[str, float],
    annulus_inner_arcsec: float,
    annulus_outer_arcsec: float,
    pixel_size_arcsec: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for array in ARRAYS:
        control_path = product_path(control_redu, obsnum, array)
        injected_path = product_path(injected_redu, obsnum, array)
        if fruit_iteration(control_path) != 5 or fruit_iteration(injected_path) != 5:
            raise ValueError(f"counterfactual {array} product is not iteration 5")
        loaded: dict[str, np.ndarray] = {}
        for variant, path in (
            ("control", control_path),
            ("injected", injected_path),
        ):
            for extension in IMAGE_EXTENSIONS:
                values, grid = load_image(path, extension)
                loaded[f"{variant}_{extension}"] = values
                if variant == "control":
                    loaded[f"control_grid_{extension}"] = grid
                elif grid != loaded[f"control_grid_{extension}"]:
                    raise ValueError(
                        f"paired grid differs: {array}:{extension}"
                    )

        support = common_support(
            {
                name: values
                for name, values in loaded.items()
                if not name.startswith("control_grid_")
            },
            context=f"EL-F3 iteration=5 array={array}",
        )
        transfer = loaded["injected_signal_I"] - loaded["control_signal_I"]
        kernel = loaded["injected_kernel_I"]
        transfer_fit = gaussian_fit(
            transfer,
            pixel_size_arcsec,
            expected_center_arcsec=(0.0, 0.0),
        )
        kernel_fit = gaussian_fit(
            kernel,
            pixel_size_arcsec,
            expected_center_arcsec=(0.0, 0.0),
        )
        truth = amplitudes[array]
        projection = kernel_projection_metrics(transfer, kernel, truth)
        beta = float(projection["scale_mjy_beam"])
        residual = transfer - beta * kernel
        annulus = annulus_mask(
            transfer.shape,
            pixel_size_arcsec,
            annulus_inner_arcsec,
            annulus_outer_arcsec,
        )
        annulus_support = support & annulus
        if not annulus_support.any():
            raise ValueError(f"empty EL-F3 annulus for {array}")
        rows.append(
            {
                "array": array,
                "kernel_normalized_central_recovery": (
                    transfer_fit["amplitude"]
                    / (truth * kernel_fit["amplitude"])
                ),
                "full_kernel_recovery": projection["recovery_fraction"],
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
    counterfactual: float,
    original_iteration_4: float,
    original_iteration_5: float,
    improvement_direction: int,
) -> float:
    denominator = improvement_direction * (
        original_iteration_4 - original_iteration_5
    )
    numerator = improvement_direction * (
        counterfactual - original_iteration_5
    )
    if denominator <= 0.0:
        raise ValueError("registered original values do not define a loss")
    return numerator / denominator


def classify_effect(recovery_fraction: float, annular_fraction: float) -> str:
    if recovery_fraction >= 0.5 and annular_fraction >= 0.5:
        return "substantial_causal_contribution"
    if recovery_fraction > 0.0 and annular_fraction > 0.0:
        return "partial_causal_contribution"
    if recovery_fraction <= 0.0 and annular_fraction <= 0.0:
        return "no_support_for_causal_contribution"
    return "mixed_effect"


def screen_pass(
    candidate: dict[str, float | str],
    reference: dict[str, float],
    screen: dict,
) -> bool:
    return all(
        (
            abs(float(candidate["kernel_normalized_central_recovery"]) - 1.0)
            <= abs(reference["kernel_normalized_central_recovery"] - 1.0)
            + float(screen["max_absolute_recovery_error_degradation"]),
            abs(float(candidate["major_fwhm_over_kernel"]) - 1.0)
            <= float(screen["max_width_fractional_error"]),
            abs(float(candidate["minor_fwhm_over_kernel"]) - 1.0)
            <= float(screen["max_width_fractional_error"]),
            float(candidate["centroid_error_arcsec"])
            <= float(screen["max_centroid_error_arcsec"]),
            float(candidate["annular_residual_over_truth"])
            <= float(screen["max_residual_ratio_to_reference"])
            * reference["annular_residual_over_truth"],
            float(candidate["kernel_residual_relative_rms"])
            <= float(screen["max_residual_ratio_to_reference"])
            * reference["kernel_residual_relative_rms"],
        )
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze(manifest: dict) -> tuple[list[dict], dict]:
    obsnum = int(manifest["obsnum"])
    original_control = Path(manifest["original_control_iteration_5"])
    replay_control = Path(manifest["replay_control_iteration_5"])
    counterfactual_injected = Path(
        manifest["counterfactual_injected_iteration_5"]
    )
    require_exact_control_maps(original_control, replay_control, obsnum)
    require_exact_checkpoint(
        original_control / "citlali_restart_checkpoint.nc",
        replay_control / "citlali_restart_checkpoint.nc",
    )

    audit = json.loads(Path(manifest["intervention_audit"]).read_text())
    removed = audit["transformation"]["removed_record"]
    expected_removed = manifest["expected_removed_penalty"]
    for key, value in expected_removed.items():
        if removed[key] != value:
            raise ValueError(
                f"intervention audit differs for {key}: "
                f"expected={value} actual={removed[key]}"
            )
    if not audit["transformation"]["all_other_values_verified_equal"]:
        raise ValueError("intervention audit did not verify other values")

    amplitudes = dict(
        zip(ARRAYS, (float(value) for value in manifest["amplitudes_mjy_beam"]))
    )
    measured = measure_pair(
        replay_control,
        counterfactual_injected,
        obsnum,
        amplitudes,
        float(manifest["annulus_inner_arcsec"]),
        float(manifest["annulus_outer_arcsec"]),
        float(manifest["pixel_size_arcsec"]),
    )
    original_metrics = read_metrics(Path(manifest["original_metrics"]))
    rows: list[dict] = []
    for counterfactual in measured:
        array = str(counterfactual["array"])
        original_5 = original_metrics[(1.25, 5, array)]
        reference = original_metrics[(1.0, 6, array)]
        row = dict(counterfactual)
        row.update(
            {
                "original_iteration_5_recovery": original_5[
                    "kernel_normalized_central_recovery"
                ],
                "recovery_change_from_original_iteration_5": (
                    float(counterfactual["kernel_normalized_central_recovery"])
                    - original_5["kernel_normalized_central_recovery"]
                ),
                "original_iteration_5_annular_residual": original_5[
                    "annular_residual_over_truth"
                ],
                "annular_residual_change_from_original_iteration_5": (
                    float(counterfactual["annular_residual_over_truth"])
                    - original_5["annular_residual_over_truth"]
                ),
                "inherited_array_screen_pass": screen_pass(
                    counterfactual, reference, manifest["screen"]
                ),
            }
        )
        rows.append(row)

    a1400 = next(row for row in rows if row["array"] == "a1400")
    original = manifest["original_a1400"]
    recovery_reversal = reversal_fraction(
        float(a1400["kernel_normalized_central_recovery"]),
        float(original["iteration_4_recovery"]),
        float(original["iteration_5_recovery"]),
        1,
    )
    annular_reversal = reversal_fraction(
        float(a1400["annular_residual_over_truth"]),
        float(original["iteration_4_annular_residual"]),
        float(original["iteration_5_annular_residual"]),
        -1,
    )
    result = {
        "test_id": manifest["test_id"],
        "valid_counterfactual": True,
        "control_replay_exact": True,
        "intervention_audit_pass": True,
        "a1400_recovery_reversal_fraction": recovery_reversal,
        "a1400_annular_residual_reversal_fraction": annular_reversal,
        "full_reversal": recovery_reversal >= 1.0 and annular_reversal >= 1.0,
        "mechanism_classification": classify_effect(
            recovery_reversal, annular_reversal
        ),
        "all_array_inherited_screen_pass": all(
            bool(row["inherited_array_screen_pass"]) for row in rows
        ),
        "claim_scope": "one exposed development checkpoint intervention",
    }
    return rows, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    rows, result = analyze(manifest)
    write_csv(args.output, rows)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote result to {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
