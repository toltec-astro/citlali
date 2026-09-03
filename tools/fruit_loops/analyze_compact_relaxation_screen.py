#!/usr/bin/env python3
"""Analyze the prospective SCI-FRUIT EL-F1 compact-relaxation screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from tools.fruit_loops.compare_injected_source_pair import (
    ARRAYS,
    gaussian_center_for_map_world_offset,
    gaussian_fit,
    iteration_dirs,
    kernel_projection_metrics,
    low_level_config,
    product_path,
    require_pair_config,
    rms,
)


IMAGE_EXTENSIONS = ("signal_I", "kernel_I", "weight_I")
WCS_PREFIXES = (
    "CTYPE",
    "CUNIT",
    "CRPIX",
    "CRVAL",
    "CDELT",
    "CD",
    "PC",
    "CROTA",
)
WCS_KEYS = {
    "WCSAXES",
    "RADESYS",
    "EQUINOX",
    "LONPOLE",
    "LATPOLE",
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
}


def load_image(path: Path, extension: str) -> tuple[np.ndarray, tuple]:
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[extension]
        values = np.asarray(hdu.data, dtype=float).squeeze()
        signature = tuple(
            sorted(
                (key, hdu.header[key])
                for key in hdu.header
                if key in WCS_KEYS or key.startswith(WCS_PREFIXES)
            )
        )
    return values, (values.shape, signature)


def require_exact_images(
    expected_root: Path,
    actual_root: Path,
    obsnum: int,
    iteration: int,
) -> None:
    for array in ARRAYS:
        expected_path = product_path(expected_root, obsnum, array)
        actual_path = product_path(actual_root, obsnum, array)
        for extension in IMAGE_EXTENSIONS:
            expected, expected_grid = load_image(expected_path, extension)
            actual, actual_grid = load_image(actual_path, extension)
            if expected_grid != actual_grid:
                raise ValueError(
                    "pre-injection pair grid differs: "
                    f"iteration={iteration} array={array} "
                    f"extension={extension}"
                )
            if not np.array_equal(expected, actual, equal_nan=True):
                raise ValueError(
                    "pre-injection pair is not bitwise identical: "
                    f"iteration={iteration} array={array} "
                    f"extension={extension}"
                )


def common_support(
    named_images: dict[str, np.ndarray],
    *,
    context: str,
) -> np.ndarray:
    reference_name, reference = next(iter(named_images.items()))
    reference_support = np.isfinite(reference)
    for name, values in named_images.items():
        if values.shape != reference.shape:
            raise ValueError(
                f"shape mismatch for {context}: {reference_name}="
                f"{reference.shape} {name}={values.shape}"
            )
        support = np.isfinite(values)
        if not np.array_equal(reference_support, support):
            raise ValueError(
                f"finite-support mismatch for {context}: "
                f"{reference_name} versus {name}"
            )
    if not reference_support.any():
        raise ValueError(f"empty finite support for {context}")
    return reference_support


def annulus_mask(
    shape: tuple[int, ...],
    pixel_size_arcsec: float,
    inner_arcsec: float,
    outer_arcsec: float,
    center_arcsec: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError(f"expected a 2-D map, got shape {shape}")
    yy, xx = np.indices(shape, dtype=float)
    xx = (xx - (shape[1] - 1) / 2.0) * pixel_size_arcsec
    yy = (yy - (shape[0] - 1) / 2.0) * pixel_size_arcsec
    center_x, center_y = center_arcsec
    radius = np.hypot(xx - center_x, yy - center_y)
    return (radius >= inner_arcsec) & (radius <= outer_arcsec)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty EL-F1 metrics table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def trajectory_config(iteration_roots: dict[int, Path]) -> dict:
    """Load the authoritative realized config for one complete trajectory."""
    merged_snapshots = sorted(
        root / "citlali_merged_config.yaml"
        for root in iteration_roots.values()
        if (root / "citlali_merged_config.yaml").is_file()
    )
    if merged_snapshots:
        loaded = [yaml.safe_load(path.read_text()) for path in merged_snapshots]
        if not all(isinstance(config, dict) for config in loaded):
            raise ValueError(
                f"invalid merged trajectory config in {merged_snapshots}"
            )
        reference = loaded[0]
        if any(config != reference for config in loaded[1:]):
            raise ValueError(
                "merged trajectory configs differ across iterations: "
                f"{merged_snapshots}"
            )
        return reference

    # Compatibility fallback for older retained outputs that predate the
    # authoritative merged-config snapshot.
    loaded = [low_level_config(root) for root in iteration_roots.values()]
    reference = loaded[0]
    if any(config != reference for config in loaded[1:]):
        raise ValueError("low-level trajectory configs differ across iterations")
    return reference


def analyze_pair(
    *,
    alpha_label: str,
    alpha: float,
    control_root: Path,
    injected_root: Path,
    manifest: dict,
    stop_iteration_exclusive: int | None = None,
) -> list[dict]:
    obsnum = int(manifest["obsnum"])
    trajectory_start = int(manifest["trajectory_start_iteration"])
    injection_start = int(manifest["injection_start_iteration"])
    stop_iteration = (
        int(manifest["stop_iteration_exclusive"])
        if stop_iteration_exclusive is None
        else int(stop_iteration_exclusive)
    )
    expected_iterations = list(range(trajectory_start, stop_iteration))
    control_dirs = iteration_dirs(control_root, obsnum)
    injected_dirs = iteration_dirs(injected_root, obsnum)
    if sorted(control_dirs) != expected_iterations:
        raise ValueError(
            f"alpha={alpha_label} control iterations differ: "
            f"expected={expected_iterations} actual={sorted(control_dirs)}"
        )
    if sorted(injected_dirs) != expected_iterations:
        raise ValueError(
            f"alpha={alpha_label} injected iterations differ: "
            f"expected={expected_iterations} actual={sorted(injected_dirs)}"
        )

    pair_contract = {
        "start_iteration": injection_start,
        "array_amplitude_mjy_beam": manifest[
            "array_amplitude_mjy_beam"
        ],
        "az_offset_arcsec": float(
            manifest.get("az_offset_arcsec", 0.0)
        ),
        "el_offset_arcsec": float(
            manifest.get("el_offset_arcsec", 0.0)
        ),
    }
    control_config = trajectory_config(control_dirs)
    injected_config = trajectory_config(injected_dirs)
    require_pair_config(control_config, injected_config, pair_contract)
    for name, config in (
        ("control", control_config),
        ("injected", injected_config),
    ):
        realized = float(
            config["timestream"]["fruit_loops"]["relaxation_alpha"]
        )
        if realized != alpha:
            raise ValueError(
                f"alpha={alpha_label} {name} realized alpha={realized}"
            )

    for iteration in range(trajectory_start, injection_start):
        require_exact_images(
            control_dirs[iteration], injected_dirs[iteration], obsnum, iteration
        )

    amplitudes = dict(
        zip(manifest["array_order"], manifest["array_amplitude_mjy_beam"])
    )
    annulus_inner = float(manifest["annulus_inner_arcsec"])
    annulus_outer = float(manifest["annulus_outer_arcsec"])
    search_radius = float(manifest["gaussian_search_radius_arcsec"])
    reference_grids: dict[tuple[str, str], tuple] = {}
    previous_kernels: dict[str, np.ndarray] = {}
    previous_transfers: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    for iteration in expected_iterations:
        control_redu = control_dirs[iteration]
        injected_redu = injected_dirs[iteration]
        pixel_size = float(control_config["mapmaking"]["pixel_size_arcsec"])
        for array in ARRAYS:
            control_path = product_path(control_redu, obsnum, array)
            injected_path = product_path(injected_redu, obsnum, array)
            loaded: dict[str, np.ndarray] = {}
            for variant, path in (
                ("control", control_path),
                ("injected", injected_path),
            ):
                for extension in IMAGE_EXTENSIONS:
                    values, grid = load_image(path, extension)
                    grid_key = (variant, f"{array}:{extension}")
                    previous_grid = reference_grids.setdefault(grid_key, grid)
                    if grid != previous_grid:
                        raise ValueError(
                            "grid/WCS changed across iterations: "
                            f"alpha={alpha_label} iteration={iteration} "
                            f"variant={variant} array={array} "
                            f"extension={extension}"
                        )
                    loaded[f"{variant}_{extension}"] = values

            for extension in IMAGE_EXTENSIONS:
                _, control_grid = load_image(control_path, extension)
                _, injected_grid = load_image(injected_path, extension)
                if control_grid != injected_grid:
                    raise ValueError(
                        "paired grid/WCS mismatch: "
                        f"alpha={alpha_label} iteration={iteration} "
                        f"array={array} extension={extension}"
                    )

            support = common_support(
                loaded,
                context=(
                    f"alpha={alpha_label} iteration={iteration} array={array}"
                ),
            )
            injected_kernel = loaded["injected_kernel_I"]
            if iteration < injection_start:
                previous_kernels[array] = injected_kernel.copy()
                continue

            truth = float(amplitudes[array])
            transfer = (
                loaded["injected_signal_I"] - loaded["control_signal_I"]
            )
            expected_center = gaussian_center_for_map_world_offset(
                injected_path,
                "signal_I",
                float(manifest.get("az_offset_arcsec", 0.0)),
                float(manifest.get("el_offset_arcsec", 0.0)),
            )
            transfer_fit = gaussian_fit(
                transfer,
                pixel_size,
                expected_center_arcsec=expected_center,
                search_radius_arcsec=search_radius,
            )
            kernel_fit = gaussian_fit(
                injected_kernel,
                pixel_size,
                expected_center_arcsec=expected_center,
                search_radius_arcsec=search_radius,
            )
            projection = kernel_projection_metrics(
                transfer, injected_kernel, truth
            )
            beta = float(projection["scale_mjy_beam"])
            residual = transfer - beta * injected_kernel
            annulus = annulus_mask(
                transfer.shape, pixel_size, annulus_inner, annulus_outer,
                center_arcsec=expected_center,
            )
            annulus_support = support & annulus
            if not annulus_support.any():
                raise ValueError(
                    f"empty analysis annulus for alpha={alpha_label} "
                    f"iteration={iteration} array={array}"
                )
            previous_kernel = previous_kernels.get(array)
            if previous_kernel is None:
                raise ValueError(
                    f"missing preceding kernel for iteration {iteration}"
                )
            kernel_denominator = rms(previous_kernel[support])
            response_change = (
                rms((injected_kernel - previous_kernel)[support])
                / kernel_denominator
                if kernel_denominator > 0.0
                else math.nan
            )
            previous_transfer = previous_transfers.get(array)
            transfer_denominator = (
                rms(previous_transfer[support])
                if previous_transfer is not None
                else math.nan
            )
            successive_change = (
                rms((transfer - previous_transfer)[support])
                / transfer_denominator
                if previous_transfer is not None
                and transfer_denominator > 0.0
                else math.nan
            )
            rows.append(
                {
                    "alpha": alpha,
                    "iteration": iteration,
                    "array": array,
                    "injected_amplitude_mjy_beam": truth,
                    "injected_az_offset_arcsec": float(
                        manifest.get("az_offset_arcsec", 0.0)
                    ),
                    "injected_el_offset_arcsec": float(
                        manifest.get("el_offset_arcsec", 0.0)
                    ),
                    "kernel_normalized_central_recovery":
                        transfer_fit["amplitude"]
                        / (truth * kernel_fit["amplitude"]),
                    "full_kernel_recovery": projection["recovery_fraction"],
                    "major_fwhm_over_kernel":
                        transfer_fit["major_fwhm_arcsec"]
                        / kernel_fit["major_fwhm_arcsec"],
                    "minor_fwhm_over_kernel":
                        transfer_fit["minor_fwhm_arcsec"]
                        / kernel_fit["minor_fwhm_arcsec"],
                    "centroid_error_arcsec": math.hypot(
                        transfer_fit["x_arcsec"] - kernel_fit["x_arcsec"],
                        transfer_fit["y_arcsec"] - kernel_fit["y_arcsec"],
                    ),
                    "kernel_residual_relative_rms":
                        projection["residual_relative_rms"],
                    "annular_residual_over_truth":
                        rms(residual[annulus_support]) / truth,
                    "response_change_relative_rms": response_change,
                    "successive_transfer_change_relative_rms":
                        successive_change,
                    "common_valid_pixels": int(np.count_nonzero(support)),
                    "total_pixels": int(support.size),
                }
            )
            previous_kernels[array] = injected_kernel.copy()
            previous_transfers[array] = transfer.copy()
    return rows


def select_row(
    rows: list[dict], alpha: float, iteration: int, array: str
) -> dict:
    matches = [
        row
        for row in rows
        if row["alpha"] == alpha
        and row["iteration"] == iteration
        and row["array"] == array
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one row for alpha={alpha} iteration={iteration} "
            f"array={array}; found {len(matches)}"
        )
    return matches[0]


def classify(rows: list[dict], manifest: dict) -> dict:
    screen = manifest["screen"]
    reference_alpha = float(screen["reference_alpha"])
    target_iteration = int(screen["reference_target_iteration"])
    deadline = int(screen["candidate_target_deadline_iteration"])
    final_iteration = int(screen["final_iteration"])
    max_error_degradation = float(
        screen["max_absolute_recovery_error_degradation"]
    )
    max_width_error = float(screen["max_width_fractional_error"])
    max_centroid = float(screen["max_centroid_error_arcsec"])
    max_residual_ratio = float(screen["max_residual_ratio_to_control"])
    outcomes: dict[str, dict] = {}

    for alpha in sorted({float(row["alpha"]) for row in rows}):
        if alpha == reference_alpha:
            continue
        checks: dict[str, dict[str, bool]] = {}
        for array in ARRAYS:
            reference_target = select_row(
                rows, reference_alpha, target_iteration, array
            )["kernel_normalized_central_recovery"]
            early_rows = [
                row
                for row in rows
                if row["alpha"] == alpha
                and row["array"] == array
                and row["iteration"] <= deadline
            ]
            final = select_row(rows, alpha, final_iteration, array)
            control_final = select_row(
                rows, reference_alpha, final_iteration, array
            )
            checks[array] = {
                "reaches_control_iteration_5_by_iteration_4": any(
                    row["kernel_normalized_central_recovery"]
                    >= reference_target
                    for row in early_rows
                ),
                "final_recovery_error_within_allowance": abs(
                    final["kernel_normalized_central_recovery"] - 1.0
                )
                <= abs(
                    control_final["kernel_normalized_central_recovery"] - 1.0
                )
                + max_error_degradation,
                "final_major_width_within_limit": abs(
                    final["major_fwhm_over_kernel"] - 1.0
                )
                <= max_width_error,
                "final_minor_width_within_limit": abs(
                    final["minor_fwhm_over_kernel"] - 1.0
                )
                <= max_width_error,
                "final_centroid_within_limit":
                    final["centroid_error_arcsec"] <= max_centroid,
                "final_annular_residual_within_limit":
                    final["annular_residual_over_truth"]
                    <= max_residual_ratio
                    * control_final["annular_residual_over_truth"],
                "final_kernel_residual_within_limit":
                    final["kernel_residual_relative_rms"]
                    <= max_residual_ratio
                    * control_final["kernel_residual_relative_rms"],
            }
        all_checks_pass = all(
            passed
            for array_checks in checks.values()
            for passed in array_checks.values()
        )
        outcomes[f"{alpha:.2f}"] = {
            "array_checks": checks,
            "scientifically_promising_before_restart": all_checks_pass,
            "restart_status": "required" if all_checks_pass else "not_required",
        }

    promising = [
        alpha
        for alpha, outcome in outcomes.items()
        if outcome["scientifically_promising_before_restart"]
    ]
    return {
        "test_id": manifest["test_id"],
        "valid_primary_screen": True,
        "candidate_outcomes": outcomes,
        "promising_candidates_requiring_restart": promising,
        "classification": (
            "restart_pending_for_promising_candidate"
            if promising
            else "not_promising_on_this_compact_case"
        ),
    }


def write_report(path: Path, rows: list[dict], result: dict) -> None:
    lines = [
        "# SCI-FRUIT EL-F1 compact-relaxation screen result",
        "",
        f"Primary classification: **{result['classification']}**",
        "",
        "This is development evidence only and is not a method qualification.",
        "",
        "## Iteration metrics",
        "",
        "| Alpha | Iter | Array | Central recovery | Full-kernel recovery | "
        "Major/kernel | Minor/kernel | Centroid (arcsec) | Annular residual | "
        "Kernel residual |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['alpha']:.2f} | {row['iteration']} | {row['array']} | "
            f"{row['kernel_normalized_central_recovery']:.6f} | "
            f"{row['full_kernel_recovery']:.6f} | "
            f"{row['major_fwhm_over_kernel']:.6f} | "
            f"{row['minor_fwhm_over_kernel']:.6f} | "
            f"{row['centroid_error_arcsec']:.6f} | "
            f"{row['annular_residual_over_truth']:.8g} | "
            f"{row['kernel_residual_relative_rms']:.8g} |"
        )
    lines.extend(["", "## Frozen-screen checks", ""])
    for alpha, outcome in result["candidate_outcomes"].items():
        lines.append(
            f"### Alpha {alpha}: "
            + (
                "scientifically promising; exact restart pending"
                if outcome["scientifically_promising_before_restart"]
                else "not promising on this compact case"
            )
        )
        lines.append("")
        for array, checks in outcome["array_checks"].items():
            failed = [name for name, passed in checks.items() if not passed]
            lines.append(
                f"- `{array}`: "
                + ("PASS" if not failed else "FAIL — " + ", ".join(failed))
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    rows: list[dict] = []
    for alpha_label, pair in manifest["alphas"].items():
        rows.extend(
            analyze_pair(
                alpha_label=alpha_label,
                alpha=float(alpha_label),
                control_root=Path(pair["control"]),
                injected_root=Path(pair["injected"]),
                manifest=manifest,
            )
        )
    rows.sort(key=lambda row: (row["alpha"], row["iteration"], row["array"]))
    write_csv(args.output, rows)
    result = classify(rows, manifest)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.result.with_suffix(".md"), rows, result)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote screen result to {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
