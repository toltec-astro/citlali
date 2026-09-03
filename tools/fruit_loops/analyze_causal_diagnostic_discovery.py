#!/usr/bin/env python3
"""Compare truth labels with diagnostics available to one FRUIT trajectory.

This is an exploratory development tool. Injected-minus-control measurements
label scientific outcomes, but candidate warning signals are calculated from
one trajectory at a time. It does not define or qualify a stopping rule.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from netCDF4 import Dataset


ARRAYS = ("a1100", "a1400", "a2000")


def finite_rms(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask & np.isfinite(values)]
    if selected.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(selected))))


def finite_cosine(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = mask & np.isfinite(left) & np.isfinite(right)
    left_values = left[selected]
    right_values = right[selected]
    left_norm = float(np.linalg.norm(left_values))
    right_norm = float(np.linalg.norm(right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return math.nan
    return float(np.dot(left_values, right_values) / (left_norm * right_norm))


def scalar(dataset: Dataset, name: str, cast: type = float):
    return cast(np.asarray(dataset.variables[name][:]).reshape(-1)[0])


def value_at(variable, index: int):
    value = variable[index]
    return value.item() if hasattr(value, "item") else value


def penalty_identities(dataset: Dataset) -> set[tuple]:
    count = scalar(dataset, "effective_detector_penalty_count", int)
    fields = (
        "penalty_producer",
        "penalty_reason",
        "penalty_iteration",
        "penalty_scan",
        "penalty_uid",
        "penalty_array",
    )
    return {
        tuple(value_at(dataset.variables[field], index) for field in fields)
        for index in range(count)
    }


def new_penalty_counts(
    previous: set[tuple],
    current: set[tuple],
    array_order: tuple[str, ...] = ARRAYS,
) -> dict[str, int]:
    counts = {array: 0 for array in array_order}
    for identity in current - previous:
        array_index = int(identity[-1])
        if 0 <= array_index < len(array_order):
            counts[array_order[array_index]] += 1
    return counts


def iteration_directories(root: Path, obsnum: int) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for directory in root.glob("redu[0-9]*"):
        checkpoint = directory / "citlali_restart_checkpoint.nc"
        if not checkpoint.is_file():
            continue
        with Dataset(checkpoint) as dataset:
            iteration = scalar(dataset, "completed_iteration", int)
        if iteration in result:
            raise ValueError(f"duplicate iteration {iteration} under {root}")
        result[iteration] = directory
    if not result:
        raise ValueError(f"no checkpoint-bearing iterations under {root}")
    expected = list(range(min(result), max(result) + 1))
    if sorted(result) != expected:
        raise ValueError(
            f"non-contiguous iterations under {root}: {sorted(result)}"
        )
    return result


def annular_mask_from_checkpoint(
    dataset: Dataset,
    inner_radius_arcsec: float,
    outer_radius_arcsec: float,
) -> np.ndarray:
    n_rows = scalar(dataset, "fruit_feedback_n_rows", int)
    n_cols = scalar(dataset, "fruit_feedback_n_cols", int)
    cdelt = np.asarray(dataset.variables["fruit_feedback_wcs_cdelt"][:], float)
    crpix = np.asarray(dataset.variables["fruit_feedback_wcs_crpix"][:], float)
    yy, xx = np.indices((n_rows, n_cols), dtype=float)
    radius = np.hypot(
        (xx - crpix[0]) * cdelt[0],
        (yy - crpix[1]) * cdelt[1],
    )
    return (radius >= inner_radius_arcsec) & (radius <= outer_radius_arcsec)


def feedback_planes(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    map_count = scalar(dataset, "fruit_feedback_map_count", int)
    n_rows = scalar(dataset, "fruit_feedback_n_rows", int)
    n_cols = scalar(dataset, "fruit_feedback_n_cols", int)
    shape = (map_count, n_rows, n_cols)
    signal = np.asarray(dataset.variables["fruit_feedback_signal"][:], float)
    kernel = np.asarray(dataset.variables["fruit_feedback_kernel"][:], float)
    return signal.reshape(shape), kernel.reshape(shape)


def product_path(directory: Path, obsnum: int, array: str) -> Path:
    return (
        directory
        / str(obsnum)
        / "raw"
        / f"toltec_commissioning_{array}_pointing_{obsnum}_citlali.fits"
    )


def signal_image(
    path: Path,
    inner_radius_arcsec: float,
    outer_radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        values = np.asarray(hdul["signal_I"].data, dtype=float).squeeze().copy()
        header = hdul["signal_I"].header
    yy, xx = np.indices(values.shape, dtype=float)
    radius = np.hypot(
        (xx + 1.0 - float(header["CRPIX1"])) * float(header["CDELT1"]),
        (yy + 1.0 - float(header["CRPIX2"])) * float(header["CDELT2"]),
    )
    return values, (
        (radius >= inner_radius_arcsec) & (radius <= outer_radius_arcsec)
    )


def pointing_rows(directory: Path, obsnum: int) -> dict[str, dict[str, float]]:
    raw = directory / str(obsnum) / "raw"
    table = Table.read(
        raw / f"ppt_commissioning_pointing_{obsnum}_citlali.ecsv",
        format="ascii.ecsv",
    )
    result: dict[str, dict[str, float]] = {}
    for index, array in enumerate(ARRAYS):
        row = table[np.asarray(table["array"], dtype=int) == index][0]
        result[array] = {
            "fit_amplitude": float(row["amp"]),
            "fit_signal_to_noise": float(row["fit_sig2noise"]),
        }
    return result


def map_diagnostic_rows(
    directory: Path,
    obsnum: int,
) -> dict[str, dict[str, float]]:
    path = (
        directory
        / str(obsnum)
        / "raw"
        / f"toltec_commissioning_pointing_{obsnum}_mapdiag.nc"
    )
    result: dict[str, dict[str, float]] = {}
    with Dataset(path) as dataset:
        for index, array in enumerate(ARRAYS):
            result[array] = {
                "map_median_rms": float(dataset.variables["map_median_rms"][index]),
                "map_core_tail_excess": float(
                    dataset.variables["map_core_tail_excess_abs_gt3"][index]
                ),
                "map_noise_tail_excess": float(
                    dataset.variables["map_noise_tail_excess_abs_gt3"][index]
                ),
            }
    return result


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def trajectory_metrics(
    root: Path,
    obsnum: int,
    expected_alpha: float,
    annulus_inner_arcsec: float,
    annulus_outer_arcsec: float,
) -> dict[tuple[int, str], dict[str, float | int]]:
    directories = iteration_directories(root, obsnum)
    result: dict[tuple[int, str], dict[str, float | int]] = {}
    previous_feedback: np.ndarray | None = None
    previous_feedback_delta: np.ndarray | None = None
    previous_output: dict[str, np.ndarray] = {}
    previous_output_delta: dict[str, np.ndarray] = {}
    previous_penalties: set[tuple] = set()
    previous_fit: dict[str, dict[str, float]] = {}
    previous_mapdiag: dict[str, dict[str, float]] = {}

    for iteration, directory in sorted(directories.items()):
        checkpoint_path = directory / "citlali_restart_checkpoint.nc"
        with Dataset(checkpoint_path) as checkpoint:
            alpha = scalar(checkpoint, "fruit_feedback_alpha", float)
            if not math.isclose(alpha, expected_alpha, rel_tol=0.0, abs_tol=1e-7):
                raise ValueError(
                    f"checkpoint alpha {alpha} differs from {expected_alpha}: "
                    f"{checkpoint_path}"
                )
            feedback, _ = feedback_planes(checkpoint)
            feedback_annulus = annular_mask_from_checkpoint(
                checkpoint,
                annulus_inner_arcsec,
                annulus_outer_arcsec,
            )
            penalties = penalty_identities(checkpoint)
        if feedback.shape[0] != len(ARRAYS):
            raise ValueError(
                f"feedback map count {feedback.shape[0]} is not {len(ARRAYS)}"
            )
        staged_penalties = new_penalty_counts(previous_penalties, penalties)
        fits = pointing_rows(directory, obsnum)
        mapdiag = map_diagnostic_rows(directory, obsnum)

        feedback_delta = (
            feedback - previous_feedback
            if previous_feedback is not None
            else np.full_like(feedback, np.nan)
        )
        for index, array in enumerate(ARRAYS):
            output, output_annulus = signal_image(
                product_path(directory, obsnum, array),
                annulus_inner_arcsec,
                annulus_outer_arcsec,
            )
            output_delta = (
                output - previous_output[array]
                if array in previous_output
                else np.full_like(output, np.nan)
            )
            feedback_update_rms = finite_rms(
                feedback_delta[index], feedback_annulus
            )
            prior_feedback_update_rms = (
                finite_rms(previous_feedback_delta[index], feedback_annulus)
                if previous_feedback_delta is not None
                else math.nan
            )
            output_update_rms = finite_rms(output_delta, output_annulus)
            prior_output_update_rms = (
                finite_rms(previous_output_delta[array], output_annulus)
                if array in previous_output_delta
                else math.nan
            )
            metrics: dict[str, float | int] = {
                "feedback_update_annular_rms": feedback_update_rms,
                "feedback_update_growth": safe_ratio(
                    feedback_update_rms, prior_feedback_update_rms
                ),
                "feedback_update_previous_cosine": (
                    finite_cosine(
                        feedback_delta[index],
                        previous_feedback_delta[index],
                        feedback_annulus,
                    )
                    if previous_feedback_delta is not None
                    else math.nan
                ),
                "output_update_annular_rms": output_update_rms,
                "output_update_growth": safe_ratio(
                    output_update_rms, prior_output_update_rms
                ),
                "output_update_previous_cosine": (
                    finite_cosine(
                        output_delta,
                        previous_output_delta[array],
                        output_annulus,
                    )
                    if array in previous_output_delta
                    else math.nan
                ),
                "new_penalties_staged": staged_penalties[array],
                "fit_amplitude": fits[array]["fit_amplitude"],
                "fit_amplitude_fractional_change": (
                    safe_ratio(
                        fits[array]["fit_amplitude"]
                        - previous_fit[array]["fit_amplitude"],
                        abs(previous_fit[array]["fit_amplitude"]),
                    )
                    if array in previous_fit
                    else math.nan
                ),
                "fit_signal_to_noise": fits[array]["fit_signal_to_noise"],
                "map_median_rms": mapdiag[array]["map_median_rms"],
                "map_median_rms_fractional_change": (
                    safe_ratio(
                        mapdiag[array]["map_median_rms"]
                        - previous_mapdiag[array]["map_median_rms"],
                        abs(previous_mapdiag[array]["map_median_rms"]),
                    )
                    if array in previous_mapdiag
                    else math.nan
                ),
                "map_core_tail_excess": mapdiag[array]["map_core_tail_excess"],
                "map_noise_tail_excess": mapdiag[array]["map_noise_tail_excess"],
            }
            result[(iteration, array)] = metrics
            previous_output[array] = output
            previous_output_delta[array] = output_delta

        previous_feedback = feedback
        previous_feedback_delta = feedback_delta
        previous_penalties = penalties
        previous_fit = fits
        previous_mapdiag = mapdiag

    # A penalty staged in checkpoint k is part of the state seen by iteration
    # k+1. Make that causal alignment explicit in every transition row.
    for iteration in sorted(directories):
        for array in ARRAYS:
            result[(iteration, array)]["new_penalties_applied_for_update"] = (
                int(result[(iteration - 1, array)]["new_penalties_staged"])
                if (iteration - 1, array) in result
                else 0
            )
    return result


def read_truth(path: Path) -> dict[tuple[float, int, str], dict[str, float]]:
    result: dict[tuple[float, int, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (float(row["alpha"]), int(row["iteration"]), row["array"])
            result[key] = {
                name: float(row[name])
                for name in (
                    "kernel_normalized_central_recovery",
                    "major_fwhm_over_kernel",
                    "minor_fwhm_over_kernel",
                    "centroid_error_arcsec",
                    "kernel_residual_relative_rms",
                    "annular_residual_over_truth",
                )
            }
    return result


def array_screen_pass(
    candidate: dict[str, float],
    reference: dict[str, float],
    screen: dict,
) -> bool:
    return all(
        (
            abs(candidate["kernel_normalized_central_recovery"] - 1.0)
            <= abs(reference["kernel_normalized_central_recovery"] - 1.0)
            + float(screen["max_absolute_recovery_error_degradation"]),
            abs(candidate["major_fwhm_over_kernel"] - 1.0)
            <= float(screen["max_width_fractional_error"]),
            abs(candidate["minor_fwhm_over_kernel"] - 1.0)
            <= float(screen["max_width_fractional_error"]),
            candidate["centroid_error_arcsec"]
            <= float(screen["max_centroid_error_arcsec"]),
            candidate["annular_residual_over_truth"]
            <= float(screen["max_residual_ratio_to_reference"])
            * reference["annular_residual_over_truth"],
            candidate["kernel_residual_relative_rms"]
            <= float(screen["max_residual_ratio_to_reference"])
            * reference["kernel_residual_relative_rms"],
        )
    )


def resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def analyze(manifest: dict, manifest_path: Path) -> list[dict]:
    array_order = tuple(str(value) for value in manifest["array_order"])
    if array_order != ARRAYS:
        raise ValueError(
            f"array_order must be {list(ARRAYS)}, received {list(array_order)}"
        )
    annulus_inner_arcsec = float(manifest["annulus_inner_arcsec"])
    annulus_outer_arcsec = float(manifest["annulus_outer_arcsec"])
    if not 0.0 <= annulus_inner_arcsec < annulus_outer_arcsec:
        raise ValueError(
            "annulus radii must satisfy 0 <= inner < outer, received "
            f"{annulus_inner_arcsec}, {annulus_outer_arcsec}"
        )
    rows: list[dict] = []
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        obsnum = int(case["obsnum"])
        truth = read_truth(resolve_path(case["truth_metrics"], manifest_path))
        reference_alpha = float(case["reference_alpha"])
        reference_iteration = int(case["reference_terminal_iteration"])
        references = {
            array: truth[(reference_alpha, reference_iteration, array)]
            for array in ARRAYS
        }
        for candidate in case["candidates"]:
            alpha = float(candidate["alpha"])
            root = Path(candidate["root"])
            metrics = {
                variant: trajectory_metrics(
                    root / variant / "reduced",
                    obsnum,
                    alpha,
                    annulus_inner_arcsec,
                    annulus_outer_arcsec,
                )
                for variant in ("control", "injected")
            }
            keys = sorted(set(metrics["control"]) & set(metrics["injected"]))
            prior_truth: dict[str, dict[str, float]] = {}
            for iteration, array in keys:
                truth_key = (alpha, iteration, array)
                if truth_key not in truth:
                    continue
                outcome = truth[truth_key]
                row: dict[str, object] = {
                    "case_id": case_id,
                    "obsnum": obsnum,
                    "alpha": alpha,
                    "iteration": iteration,
                    "array": array,
                    "truth_recovery": outcome[
                        "kernel_normalized_central_recovery"
                    ],
                    "truth_recovery_change": (
                        outcome["kernel_normalized_central_recovery"]
                        - prior_truth[array]["kernel_normalized_central_recovery"]
                        if array in prior_truth
                        else math.nan
                    ),
                    "truth_annular_residual": outcome[
                        "annular_residual_over_truth"
                    ],
                    "truth_annular_residual_growth": (
                        safe_ratio(
                            outcome["annular_residual_over_truth"],
                            prior_truth[array]["annular_residual_over_truth"],
                        )
                        if array in prior_truth
                        else math.nan
                    ),
                    "array_screen_pass_against_reference_terminal": (
                        array_screen_pass(
                            outcome, references[array], manifest["screen"]
                        )
                    ),
                }
                for variant in ("control", "injected"):
                    for name, value in metrics[variant][(iteration, array)].items():
                        row[f"{variant}_{name}"] = value
                rows.append(row)
                prior_truth[array] = outcome

    pass_by_iteration: dict[tuple[str, float, int], bool] = {}
    for row in rows:
        key = (str(row["case_id"]), float(row["alpha"]), int(row["iteration"]))
        pass_by_iteration[key] = pass_by_iteration.get(key, True) and bool(
            row["array_screen_pass_against_reference_terminal"]
        )
    for row in rows:
        key = (str(row["case_id"]), float(row["alpha"]), int(row["iteration"]))
        row["all_array_screen_pass_against_reference_terminal"] = (
            pass_by_iteration[key]
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rows = analyze(manifest, manifest_path)
    if not rows:
        raise ValueError("analysis produced no rows")
    write_csv(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
