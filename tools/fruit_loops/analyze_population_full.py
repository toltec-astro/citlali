#!/usr/bin/env python3
"""Analyze the complete 108-observation fruit-loop population."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import json
import math
from pathlib import Path
import re
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from tools.fruit_loops.analyze_population_stage import (
    ARRAYS,
    CENTROID_STEP_LIMIT_ARCSEC,
    audit_stage,
    build_transition_metrics,
    finite_ratio,
    load_iteration_metrics,
    map_path,
    sha256,
    write_csv,
)
from tools.fruit_loops.source_morphology import morphology_template_metrics


AMPLITUDE_TOLERANCES = (0.02, 0.025, 0.03, 0.04, 0.05)
MINIMUM_STOP_ITERATION = 6
AMPLITUDE_LIMIT = 0.03
FWHM_LIMIT = 0.05
MAP_CHANGE_LIMIT = 0.05
WEIGHT_CHANGE_LIMIT = 0.05
VALID_SUPPORT_CHANGE_LIMIT = 0.01
BACKGROUND_SEED_LIMIT = 1.10
PLANET_SOURCES = {"uranus", "neptune"}
LEARNING_RE = re.compile(
    r"reduction learning finalize:.*?"
    r"iter=(?P<iteration>\d+)\s+"
    r"phase=(?P<phase>\S+).*?"
    r"effective_sample_mask_intervals=(?P<masks>\d+).*?"
    r"effective_detector_penalties=(?P<penalties>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-root", required=True, type=Path)
    parser.add_argument("--stage-b-root", required=True, type=Path)
    parser.add_argument("--run-matrix", required=True, type=Path)
    parser.add_argument("--planet-ephemerides", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip per-observation convergence plots.",
    )
    parser.add_argument(
        "--skip-empirical-point-source-snr",
        action="store_true",
        help="Skip the slower blank-sky point-source S/N diagnostic.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze present observations but mark the population incomplete.",
    )
    return parser.parse_args()


def read_population_matrix(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 108:
        raise ValueError(f"{path}: expected 108 observations; found {len(rows)}")
    obsnums = [int(row["obsnum"]) for row in rows]
    if len(obsnums) != len(set(obsnums)):
        raise ValueError(f"{path}: duplicate observation identities")
    phases = {row["phase"] for row in rows}
    expected_phases = {
        "sentinel_extension_first",
        "population_after_sentinel_gate",
    }
    if phases != expected_phases:
        raise ValueError(f"{path}: unexpected phases {sorted(phases)}")
    return sorted(rows, key=lambda row: int(row["quality_rank"]))


def read_planet_ephemerides(path: Path) -> dict[int, dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 33:
        raise ValueError(f"{path}: expected 33 planet rows; found {len(rows)}")
    result = {}
    for row in rows:
        obsnum = int(row["obsnum"])
        if obsnum in result:
            raise ValueError(f"{path}: duplicate obsnum {obsnum}")
        diameter = float(row["angular_diameter_arcsec"])
        if not math.isfinite(diameter) or diameter <= 0.0:
            raise ValueError(f"{path}: invalid diameter for obsnum {obsnum}")
        result[obsnum] = {
            "source": row["source"],
            "mjd": float(row["mjd"]),
            "target_id": row["target_id"],
            "angular_diameter_arcsec": diameter,
            "observer_range_au": float(row["observer_range_au"]),
        }
    return result


def available_matrix_rows(
    rows: list[dict],
    stage_a_root: Path,
    stage_b_root: Path,
) -> tuple[list[dict], list[int]]:
    available = []
    missing = []
    for row in rows:
        root = (
            stage_a_root
            if row["phase"] == "sentinel_extension_first"
            else stage_b_root
        )
        obsnum = int(row["obsnum"])
        if (root / f"obs{obsnum}/reduced").is_dir():
            available.append(row)
        else:
            missing.append(obsnum)
    return available, missing


def read_learning_state(reduction_root: Path, iteration: int) -> dict:
    path = reduction_root / f"redu{iteration:02d}/citlali.log.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with gzip.open(path, "rt", errors="replace") as stream:
        text = stream.read()
    matches = [match.groupdict() for match in LEARNING_RE.finditer(text)]
    selected = [
        item for item in matches if int(item["iteration"]) == iteration
    ]
    if len(selected) != 1:
        raise ValueError(
            f"{path}: expected one learning-state record for iteration "
            f"{iteration}; found {len(selected)}"
        )
    item = selected[0]
    return {
        "learning_phase": item["phase"],
        "effective_sample_mask_interval_count": int(item["masks"]),
        "effective_detector_penalty_count": int(item["penalties"]),
    }


def morphology_metrics_for_map(
    path: Path,
    *,
    center_x_arcsec: float,
    center_y_arcsec: float,
    diameter_arcsec: float,
) -> dict:
    with fits.open(path, memmap=True) as hdul:
        signal = np.asarray(hdul["signal_I"].data, dtype=float).squeeze()
        weight = np.asarray(hdul["weight_I"].data, dtype=float).squeeze()
        coverage = (
            np.asarray(hdul["coverage_bool_I"].data).squeeze() > 0.5
        )
        kernel = np.asarray(hdul["kernel_I"].data, dtype=float).squeeze()
        header = hdul["signal_I"].header
    ny, nx = signal.shape
    x_axis = (
        np.arange(nx, dtype=float) + 1.0 - float(header["CRPIX1"])
    ) * float(header["CDELT1"]) + float(header["CRVAL1"])
    y_axis = (
        np.arange(ny, dtype=float) + 1.0 - float(header["CRPIX2"])
    ) * float(header["CDELT2"]) + float(header["CRVAL2"])
    result = morphology_template_metrics(
        signal,
        weight,
        coverage,
        kernel,
        x_axis,
        y_axis,
        center_x_arcsec=center_x_arcsec,
        center_y_arcsec=center_y_arcsec,
        source_angular_diameter_arcsec=diameter_arcsec,
    )
    valid_mask = (
        coverage
        & np.isfinite(signal)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    result["map_valid_pixel_count"] = int(np.count_nonzero(valid_mask))
    result["_map_valid_mask"] = valid_mask
    return result


def valid_support_change_fraction(
    previous: np.ndarray,
    current: np.ndarray,
) -> float:
    if previous.shape != current.shape:
        return 1.0
    union_count = int(np.count_nonzero(previous | current))
    if union_count == 0:
        return 0.0
    return float(np.count_nonzero(previous ^ current) / union_count)


def write_observation_plots(
    output_dir: Path,
    iteration_rows: list[dict],
    transition_rows: list[dict],
) -> int:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    iterations_by_obs: dict[int, list[dict]] = {}
    transitions_by_obs: dict[int, list[dict]] = {}
    for row in iteration_rows:
        iterations_by_obs.setdefault(int(row["obsnum"]), []).append(row)
    for row in transition_rows:
        transitions_by_obs.setdefault(int(row["obsnum"]), []).append(row)
    colors = {
        "a1100": "tab:blue",
        "a1400": "tab:orange",
        "a2000": "tab:green",
    }
    for obsnum, obs_rows in iterations_by_obs.items():
        obs_transitions = transitions_by_obs[obsnum]
        first = obs_rows[0]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for array in ARRAYS:
            color = colors[array]
            array_rows = sorted(
                (row for row in obs_rows if row["array"] == array),
                key=lambda row: int(row["iteration"]),
            )
            array_transitions = sorted(
                (
                    row for row in obs_transitions
                    if row["array"] == array
                ),
                key=lambda row: int(row["current_iteration"]),
            )
            iterations = [int(row["iteration"]) for row in array_rows]
            axes[0, 0].plot(
                iterations,
                [
                    float(row[
                        "morphology_template_amplitude_scale_ratio_seed"
                    ])
                    for row in array_rows
                ],
                marker="o",
                color=color,
                label=array,
            )
            axes[0, 1].plot(
                iterations,
                [
                    float(row["major_fwhm_over_morphology_template"])
                    for row in array_rows
                ],
                marker="o",
                color=color,
                label=f"{array} major",
            )
            axes[0, 1].plot(
                iterations,
                [
                    float(row["minor_fwhm_over_morphology_template"])
                    for row in array_rows
                ],
                marker=".",
                linestyle="--",
                color=color,
                alpha=0.8,
                label=f"{array} minor",
            )
            axes[1, 0].plot(
                iterations,
                [
                    float(row["centroid_shift_from_seed_arcsec"])
                    for row in array_rows
                ],
                marker="o",
                color=color,
                label=array,
            )
            transition_iterations = [
                int(row["current_iteration"]) for row in array_transitions
            ]
            axes[1, 1].plot(
                transition_iterations,
                [
                    100.0 * float(
                        row["morphology_amplitude_change_fraction"]
                    )
                    for row in array_transitions
                ],
                marker="o",
                color=color,
                label=f"{array} amplitude",
            )
            axes[1, 1].plot(
                transition_iterations,
                [
                    100.0 * float(row["successive_map_relative_rms"])
                    for row in array_transitions
                ],
                marker=".",
                linestyle="--",
                color=color,
                alpha=0.8,
                label=f"{array} map",
            )
        axes[0, 0].set(
            xlabel="Iteration",
            ylabel="Morphology-template amplitude / seed",
        )
        axes[0, 1].set(
            xlabel="Iteration",
            ylabel="FWHM / morphology template",
        )
        axes[1, 0].set(
            xlabel="Iteration",
            ylabel="Centroid shift from seed (arcsec)",
        )
        axes[1, 1].set(
            xlabel="Current iteration",
            ylabel="Successive absolute change (%)",
            yscale="log",
        )
        axes[1, 1].axhline(
            100.0 * AMPLITUDE_LIMIT,
            color="0.35",
            linestyle=":",
            linewidth=1.0,
        )
        for axis in axes.flat:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
        fig.suptitle(
            f"obs{obsnum} — {first['source']} — "
            f"{first['source_morphology']} — "
            f"quality {first['quality_rank']} ({first['quality_stratum']})"
        )
        fig.savefig(plot_dir / f"obs{obsnum}_convergence.png", dpi=140)
        plt.close(fig)
    return len(iterations_by_obs)


def load_observation(
    stage_root_text: str,
    matrix_row: dict,
    planet_ephemeris: dict | None,
    include_empirical_point_source_snr: bool,
) -> list[dict]:
    stage_root = Path(stage_root_text)
    obsnum = int(matrix_row["obsnum"])
    source = str(matrix_row["source"])
    is_planet = source.casefold() in PLANET_SOURCES
    if is_planet and planet_ephemeris is None:
        raise ValueError(f"obsnum {obsnum}: missing planet ephemeris")
    if planet_ephemeris is not None:
        if source.casefold() != str(
            planet_ephemeris["source"]
        ).casefold():
            raise ValueError(f"obsnum {obsnum}: ephemeris source mismatch")
    diameter = (
        float(planet_ephemeris["angular_diameter_arcsec"])
        if planet_ephemeris is not None else 0.0
    )
    morphology = "planetary_disk" if is_planet else "unresolved"
    rows = load_iteration_metrics(
        stage_root,
        [matrix_row],
        include_empirical_point_source_snr=
            include_empirical_point_source_snr,
    )
    learning = {
        iteration: read_learning_state(
            stage_root / f"obs{obsnum}/reduced", iteration
        )
        for iteration in range(10)
    }
    for row in rows:
        iteration = int(row["iteration"])
        metrics = morphology_metrics_for_map(
            map_path(stage_root, obsnum, iteration, str(row["array"])),
            center_x_arcsec=float(row["x_t_arcsec"]),
            center_y_arcsec=float(row["y_t_arcsec"]),
            diameter_arcsec=diameter,
        )
        row.update(metrics)
        row.update(learning[iteration])
        row["source_morphology"] = morphology
        row["source_angular_diameter_arcsec"] = diameter
        row["major_fwhm_over_morphology_template"] = finite_ratio(
            float(row["major_fwhm_arcsec"]),
            float(row["morphology_template_major_fwhm_arcsec"]),
        )
        row["minor_fwhm_over_morphology_template"] = finite_ratio(
            float(row["minor_fwhm_arcsec"]),
            float(row["morphology_template_minor_fwhm_arcsec"]),
        )
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["array"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["iteration"]))
        seed = group[0]
        for index, row in enumerate(group):
            previous = group[index - 1] if index else None
            for field in (
                "morphology_template_amplitude_scale",
                "morphology_template_amplitude_uncertainty",
                "morphology_template_formal_sig2noise",
                "major_fwhm_over_morphology_template",
                "minor_fwhm_over_morphology_template",
            ):
                row[f"{field}_ratio_seed"] = finite_ratio(
                    float(row[field]), float(seed[field])
                )
                row[f"{field}_change_fraction"] = (
                    finite_ratio(
                        float(row[field]), float(previous[field])
                    ) - 1.0
                    if previous is not None else math.nan
                )
            row["map_valid_support_change_fraction"] = (
                valid_support_change_fraction(
                    previous["_map_valid_mask"],
                    row["_map_valid_mask"],
                )
                if previous is not None else math.nan
            )
    for row in rows:
        del row["_map_valid_mask"]
    return rows


def load_population_metrics(
    stage_a_root: Path,
    stage_b_root: Path,
    matrix_rows: list[dict],
    ephemerides: dict[int, dict],
    *,
    workers: int,
    include_empirical_point_source_snr: bool,
) -> list[dict]:
    jobs = []
    for row in matrix_rows:
        root = (
            stage_a_root
            if row["phase"] == "sentinel_extension_first"
            else stage_b_root
        )
        obsnum = int(row["obsnum"])
        jobs.append((
            str(root),
            row,
            ephemerides.get(obsnum),
            include_empirical_point_source_snr,
        ))
    result: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(load_observation, *job): int(job[1]["obsnum"])
            for job in jobs
        }
        completed = 0
        for future in as_completed(pending):
            obsnum = pending[future]
            result.extend(future.result())
            completed += 1
            print(
                f"loaded obsnum {obsnum} "
                f"({completed}/{len(pending)} observations)",
                flush=True,
            )
    return sorted(
        result,
        key=lambda row: (
            int(row["quality_rank"]),
            int(row["obsnum"]),
            str(row["array"]),
            int(row["iteration"]),
        ),
    )


def build_full_transition_metrics(iteration_rows: list[dict]) -> list[dict]:
    rows = build_transition_metrics(iteration_rows)
    indexed = {
        (int(row["obsnum"]), str(row["array"]), int(row["iteration"])): row
        for row in iteration_rows
    }
    for row in rows:
        obsnum = int(row["obsnum"])
        array = str(row["array"])
        current_iteration = int(row["current_iteration"])
        previous = indexed[(obsnum, array, current_iteration - 1)]
        current = indexed[(obsnum, array, current_iteration)]
        row["source_morphology"] = current["source_morphology"]
        row["source_angular_diameter_arcsec"] = current[
            "source_angular_diameter_arcsec"
        ]
        row["morphology_amplitude_signed_change_fraction"] = (
            finite_ratio(
                float(current["morphology_template_amplitude_scale"]),
                float(previous["morphology_template_amplitude_scale"]),
            ) - 1.0
        )
        row["morphology_amplitude_change_fraction"] = abs(
            float(row["morphology_amplitude_signed_change_fraction"])
        )
        major_change = abs(
            finite_ratio(
                float(current["major_fwhm_over_morphology_template"]),
                float(previous["major_fwhm_over_morphology_template"]),
            ) - 1.0
        )
        minor_change = abs(
            finite_ratio(
                float(current["minor_fwhm_over_morphology_template"]),
                float(previous["minor_fwhm_over_morphology_template"]),
            ) - 1.0
        )
        row["morphology_major_fwhm_change_fraction"] = major_change
        row["morphology_minor_fwhm_change_fraction"] = minor_change
        row["morphology_maximum_fwhm_change_fraction"] = max(
            major_change, minor_change
        )
        row["previous_learning_phase"] = previous["learning_phase"]
        row["current_learning_phase"] = current["learning_phase"]
        row["learning_phase_apply"] = bool(
            previous["learning_phase"] == "apply"
            and current["learning_phase"] == "apply"
        )
        row["learning_counts_stable"] = bool(
            int(previous["effective_sample_mask_interval_count"])
            == int(current["effective_sample_mask_interval_count"])
            and int(previous["effective_detector_penalty_count"])
            == int(current["effective_detector_penalty_count"])
        )
        row["map_valid_support_change_fraction"] = current[
            "map_valid_support_change_fraction"
        ]
        row["map_valid_support_stable"] = bool(
            math.isfinite(
                float(row["map_valid_support_change_fraction"])
            )
            and float(row["map_valid_support_change_fraction"])
            < VALID_SUPPORT_CHANGE_LIMIT
        )
        row["background_health_guard"] = bool(
            math.isfinite(
                float(current["map_background_sigma_mjy_ratio_seed"])
            )
            and float(current["map_background_sigma_mjy_ratio_seed"])
            <= BACKGROUND_SEED_LIMIT
        )
        row["morphology_metric_finite"] = bool(
            all(
                math.isfinite(float(row[field]))
                for field in (
                    "morphology_amplitude_change_fraction",
                    "morphology_maximum_fwhm_change_fraction",
                    "centroid_step_arcsec",
                    "successive_map_relative_rms",
                    "map_weight_change_fraction",
                )
            )
        )
        row["candidate_amplitude_transition_pass"] = bool(
            row["morphology_metric_finite"]
            and row["morphology_amplitude_change_fraction"]
            < AMPLITUDE_LIMIT
        )
        row["candidate_fwhm_transition_pass"] = bool(
            row["morphology_metric_finite"]
            and row["morphology_maximum_fwhm_change_fraction"] < FWHM_LIMIT
        )
        row["candidate_centroid_transition_pass"] = bool(
            row["morphology_metric_finite"]
            and row["centroid_step_arcsec"] < CENTROID_STEP_LIMIT_ARCSEC
        )
        row["candidate_map_transition_pass"] = bool(
            row["morphology_metric_finite"]
            and row["successive_map_relative_rms"] < MAP_CHANGE_LIMIT
        )
        row["candidate_weight_transition_pass"] = bool(
            row["morphology_metric_finite"]
            and row["map_weight_change_fraction"] < WEIGHT_CHANGE_LIMIT
        )
        row["candidate_support_transition_pass"] = bool(
            row["map_valid_support_stable"]
        )
        row["candidate_learning_phase_transition_pass"] = bool(
            row["learning_phase_apply"]
        )
        row["candidate_background_transition_pass"] = bool(
            row["background_health_guard"]
        )
        row["candidate_source_association_transition_pass"] = bool(
            row["source_association_valid"]
        )
        row["candidate_psf_transition_pass"] = bool(
            row["psf_interpretable"]
        )
        components = (
            "candidate_amplitude_transition_pass",
            "candidate_fwhm_transition_pass",
            "candidate_centroid_transition_pass",
            "candidate_map_transition_pass",
            "candidate_weight_transition_pass",
            "candidate_support_transition_pass",
            "candidate_learning_phase_transition_pass",
            "candidate_background_transition_pass",
            "candidate_source_association_transition_pass",
            "candidate_psf_transition_pass",
        )
        common = all(bool(row[field]) for field in components)
        row["candidate_core_transition_pass"] = common
        row["candidate_strict_state_transition_pass"] = bool(
            common and row["learning_counts_stable"]
        )
    return rows


def first_two_pass(
    transitions: list[dict],
    *,
    field: str,
    maximum: float | None = None,
) -> int | None:
    by_iteration = {
        int(row["current_iteration"]): row for row in transitions
    }
    for stop in range(MINIMUM_STOP_ITERATION, 10):
        pair = [by_iteration.get(stop - 1), by_iteration.get(stop)]
        if any(row is None for row in pair):
            continue
        if maximum is None:
            passed = all(bool(row[field]) for row in pair)
        else:
            passed = all(
                bool(row["source_association_valid"])
                and math.isfinite(float(row[field]))
                and float(row[field]) < maximum
                for row in pair
            )
        if passed:
            return stop
    return None


def amplitude_stop_simulations(
    iteration_rows: list[dict],
    transition_rows: list[dict],
) -> list[dict]:
    iteration_groups: dict[tuple[int, str], list[dict]] = {}
    transition_groups: dict[tuple[int, str], list[dict]] = {}
    for row in iteration_rows:
        iteration_groups.setdefault(
            (int(row["obsnum"]), str(row["array"])), []
        ).append(row)
    for row in transition_rows:
        transition_groups.setdefault(
            (int(row["obsnum"]), str(row["array"])), []
        ).append(row)
    result = []
    for key, transitions in transition_groups.items():
        iterations = {
            int(row["iteration"]): row for row in iteration_groups[key]
        }
        final = iterations[9]
        for tolerance in AMPLITUDE_TOLERANCES:
            stop = first_two_pass(
                transitions,
                field="morphology_amplitude_change_fraction",
                maximum=tolerance,
            )
            stopped = iterations[stop] if stop is not None else None
            signed_residual = (
                finite_ratio(
                    float(stopped["morphology_template_amplitude_scale"]),
                    float(final["morphology_template_amplitude_scale"]),
                ) - 1.0
                if stopped is not None else math.nan
            )
            result.append(
                {
                    "obsnum": key[0],
                    "source": final["source"],
                    "source_morphology": final["source_morphology"],
                    "quality_rank": final["quality_rank"],
                    "quality_stratum": final["quality_stratum"],
                    "array": key[1],
                    "tolerance_percent": 100.0 * tolerance,
                    "first_stable_iteration": stop,
                    "resolved_by_iteration_9": stop is not None,
                    "signed_amplitude_residual_to_iteration_9":
                        signed_residual,
                    "absolute_amplitude_residual_to_iteration_9": abs(
                        signed_residual
                    ),
                }
            )
    return result


def percentile(values: list[float], quantile: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return (
        float(np.percentile(np.asarray(finite), quantile))
        if finite else math.nan
    )


def summarize_amplitude_simulations(rows: list[dict]) -> list[dict]:
    result = []
    group_specs = []
    morphologies = sorted({str(row["source_morphology"]) for row in rows})
    for morphology in ("all", *morphologies):
        for array in ("all", *ARRAYS):
            for tolerance in sorted({
                float(row["tolerance_percent"]) for row in rows
            }):
                selected = [
                    row for row in rows
                    if float(row["tolerance_percent"]) == tolerance
                    and (
                        morphology == "all"
                        or row["source_morphology"] == morphology
                    )
                    and (array == "all" or row["array"] == array)
                ]
                if selected:
                    group_specs.append(
                        (morphology, array, tolerance, selected)
                    )
    for morphology, array, tolerance, selected in group_specs:
        resolved = [
            row for row in selected if row["resolved_by_iteration_9"]
        ]
        residuals = [
            float(row["absolute_amplitude_residual_to_iteration_9"])
            for row in resolved
        ]
        signed = [
            float(row["signed_amplitude_residual_to_iteration_9"])
            for row in resolved
        ]
        stops = [
            int(row["first_stable_iteration"]) for row in resolved
        ]
        result.append(
            {
                "source_morphology": morphology,
                "array": array,
                "tolerance_percent": tolerance,
                "trajectory_count": len(selected),
                "resolved_count": len(resolved),
                "resolved_fraction": len(resolved) / len(selected),
                "median_stop_iteration":
                    median(stops) if stops else math.nan,
                "median_signed_residual_percent":
                    100.0 * median(signed) if signed else math.nan,
                "median_absolute_residual_percent":
                    100.0 * median(residuals) if residuals else math.nan,
                "p90_absolute_residual_percent":
                    100.0 * percentile(residuals, 90.0),
                "maximum_absolute_residual_percent":
                    100.0 * max(residuals) if residuals else math.nan,
            }
        )
    return result


def trajectory_behavior_summary(iteration_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[list[dict]]] = {}
    trajectories: dict[tuple[int, str], list[dict]] = {}
    for row in iteration_rows:
        trajectories.setdefault(
            (int(row["obsnum"]), str(row["array"])), []
        ).append(row)
    for rows in trajectories.values():
        rows.sort(key=lambda row: int(row["iteration"]))
        key = (str(rows[0]["source_morphology"]), str(rows[0]["array"]))
        groups.setdefault(key, []).append(rows)
    result = []
    for (morphology, array), selected in sorted(groups.items()):
        amplitude_monotonic = []
        background_monotonic = []
        endpoint_amplitude_ratios = []
        endpoint_background_ratios = []
        endpoint_formal_snr_ratios = []
        endpoint_empirical_snr_ratios = []
        endpoint_legacy_dynamic_range_ratios = []
        tail_signed_changes = []
        for rows in selected:
            amplitudes = [
                float(row["morphology_template_amplitude_scale"])
                for row in rows
            ]
            background = [
                float(row["map_background_sigma_mjy"]) for row in rows
            ]
            amplitude_monotonic.append(all(
                current >= previous
                for previous, current in zip(amplitudes, amplitudes[1:])
            ))
            background_monotonic.append(all(
                current >= previous
                for previous, current in zip(background, background[1:])
            ))
            endpoint = rows[-1]
            endpoint_amplitude_ratios.append(float(
                endpoint[
                    "morphology_template_amplitude_scale_ratio_seed"
                ]
            ))
            endpoint_background_ratios.append(float(
                endpoint["map_background_sigma_mjy_ratio_seed"]
            ))
            endpoint_formal_snr_ratios.append(float(
                endpoint[
                    "morphology_template_formal_sig2noise_ratio_seed"
                ]
            ))
            endpoint_empirical_snr_ratios.append(float(
                endpoint[
                    "empirical_point_source_sig2noise_ratio_seed"
                ]
            ))
            endpoint_legacy_dynamic_range_ratios.append(float(
                endpoint["legacy_peak_over_full_map_rms_ratio_seed"]
            ))
            tail_signed_changes.extend(
                float(row[
                    "morphology_template_amplitude_scale_change_fraction"
                ])
                for row in rows if int(row["iteration"]) >= 6
            )
        result.append(
            {
                "source_morphology": morphology,
                "array": array,
                "trajectory_count": len(selected),
                "amplitude_monotonic_trajectory_count":
                    sum(amplitude_monotonic),
                "amplitude_monotonic_trajectory_fraction":
                    sum(amplitude_monotonic) / len(selected),
                "tail_positive_amplitude_step_fraction":
                    sum(value > 0.0 for value in tail_signed_changes)
                    / len(tail_signed_changes),
                "endpoint_amplitude_ratio_seed_median":
                    median(endpoint_amplitude_ratios),
                "background_monotonic_trajectory_count":
                    sum(background_monotonic),
                "background_monotonic_trajectory_fraction":
                    sum(background_monotonic) / len(selected),
                "endpoint_background_sigma_ratio_seed_median":
                    median(endpoint_background_ratios),
                "endpoint_formal_snr_ratio_seed_median":
                    median(endpoint_formal_snr_ratios),
                "endpoint_empirical_snr_ratio_seed_median":
                    percentile(endpoint_empirical_snr_ratios, 50.0),
                "endpoint_legacy_dynamic_range_ratio_seed_median":
                    median(endpoint_legacy_dynamic_range_ratios),
            }
        )
    return result


def planet_morphology_effect_summary(
    iteration_rows: list[dict],
    transition_rows: list[dict],
) -> list[dict]:
    result = []
    for array in ARRAYS:
        iterations = [
            row for row in iteration_rows
            if row["source_morphology"] == "planetary_disk"
            and row["array"] == array
        ]
        transitions = [
            row for row in transition_rows
            if row["source_morphology"] == "planetary_disk"
            and row["array"] == array
            and int(row["current_iteration"]) >= 6
        ]
        major = [
            finite_ratio(
                float(row["morphology_template_major_fwhm_arcsec"]),
                float(row["kernel_major_fwhm_arcsec"]),
            ) - 1.0
            for row in iterations
        ]
        minor = [
            finite_ratio(
                float(row["morphology_template_minor_fwhm_arcsec"]),
                float(row["kernel_minor_fwhm_arcsec"]),
            ) - 1.0
            for row in iterations
        ]
        amplitude_difference = [
            abs(
                float(row["morphology_amplitude_change_fraction"])
                - float(
                    row["kernel_normalized_amplitude_change_fraction"]
                )
            )
            for row in transitions
        ]
        result.append(
            {
                "array": array,
                "iteration_metric_count": len(iterations),
                "median_major_fwhm_broadening_percent":
                    100.0 * median(major),
                "maximum_major_fwhm_broadening_percent":
                    100.0 * max(major),
                "median_minor_fwhm_broadening_percent":
                    100.0 * median(minor),
                "maximum_minor_fwhm_broadening_percent":
                    100.0 * max(minor),
                "p90_absolute_tail_amplitude_metric_difference_percent":
                    100.0 * percentile(amplitude_difference, 90.0),
                "maximum_absolute_tail_amplitude_metric_difference_percent":
                    100.0 * max(amplitude_difference),
            }
        )
    return result


def first_observation_pass(
    transitions: list[dict], *, field: str,
) -> int | None:
    indexed = {
        (int(row["current_iteration"]), str(row["array"])): row
        for row in transitions
    }
    for stop in range(MINIMUM_STOP_ITERATION, 10):
        passed = all(
            bool(indexed[(iteration, array)][field])
            for iteration in (stop - 1, stop)
            for array in ARRAYS
            if (iteration, array) in indexed
        )
        complete = all(
            (iteration, array) in indexed
            for iteration in (stop - 1, stop)
            for array in ARRAYS
        )
        if complete and passed:
            return stop
    return None


def observation_stop_simulations(
    iteration_rows: list[dict],
    transition_rows: list[dict],
) -> list[dict]:
    transition_groups: dict[int, list[dict]] = {}
    endpoint_by_obs: dict[int, dict] = {}
    for row in transition_rows:
        transition_groups.setdefault(int(row["obsnum"]), []).append(row)
    for row in iteration_rows:
        if int(row["iteration"]) == 9:
            endpoint_by_obs.setdefault(int(row["obsnum"]), row)
    result = []
    for obsnum, transitions in transition_groups.items():
        endpoint = endpoint_by_obs[obsnum]
        result.append(
            {
                "obsnum": obsnum,
                "source": endpoint["source"],
                "source_morphology": endpoint["source_morphology"],
                "quality_rank": endpoint["quality_rank"],
                "quality_stratum": endpoint["quality_stratum"],
                "core_first_stable_iteration": first_observation_pass(
                    transitions, field="candidate_core_transition_pass"
                ),
                "strict_state_first_stable_iteration":
                    first_observation_pass(
                        transitions,
                        field="candidate_strict_state_transition_pass",
                    ),
            }
        )
    return sorted(result, key=lambda row: int(row["quality_rank"]))


def observation_criterion_status(
    observation_rows: list[dict],
    transition_rows: list[dict],
) -> list[dict]:
    transition_groups: dict[int, list[dict]] = {}
    for row in transition_rows:
        transition_groups.setdefault(int(row["obsnum"]), []).append(row)
    criteria = {
        "amplitude": "candidate_amplitude_transition_pass",
        "fwhm": "candidate_fwhm_transition_pass",
        "centroid": "candidate_centroid_transition_pass",
        "whole_map": "candidate_map_transition_pass",
        "weight": "candidate_weight_transition_pass",
        "valid_support": "candidate_support_transition_pass",
        "learning_phase": "candidate_learning_phase_transition_pass",
        "background": "candidate_background_transition_pass",
        "source_association":
            "candidate_source_association_transition_pass",
        "psf_interpretability": "candidate_psf_transition_pass",
    }
    result = []
    for observation in observation_rows:
        obsnum = int(observation["obsnum"])
        transitions = transition_groups[obsnum]
        row = dict(observation)
        failed = []
        for name, field in criteria.items():
            stop = first_observation_pass(transitions, field=field)
            row[f"{name}_first_stable_iteration"] = stop
            if stop is None:
                failed.append(name)
        row["individually_unresolved_criteria"] = ";".join(failed)
        row["continuation_class"] = (
            "candidate_converged"
            if observation["core_first_stable_iteration"] is not None
            else (
                "measurement_limited"
                if {
                    "source_association", "psf_interpretability"
                } & set(failed)
                else "trajectory_unresolved"
            )
        )
        result.append(row)
    return result


def map_residual_metrics(
    stop_path: Path,
    final_path: Path,
    *,
    center_x_arcsec: float,
    center_y_arcsec: float,
    aperture_radius_arcsec: float,
    background_sigma: float,
) -> dict[str, float]:
    with fits.open(stop_path, memmap=True) as hdul:
        stopped = np.asarray(
            hdul["signal_I"].data, dtype=float
        ).squeeze()
    with fits.open(final_path, memmap=True) as hdul:
        final = np.asarray(hdul["signal_I"].data, dtype=float).squeeze()
        header = hdul["signal_I"].header
    finite = np.isfinite(stopped) & np.isfinite(final)
    if not np.any(finite):
        return {
            "whole_map_relative_rms_to_iteration_9": math.nan,
            "source_aperture_relative_rms_to_iteration_9": math.nan,
            "source_aperture_delta_rms_over_background_sigma": math.nan,
        }
    delta = stopped - final
    whole_denominator = float(np.sqrt(np.mean(np.square(final[finite]))))
    whole_delta = float(np.sqrt(np.mean(np.square(delta[finite]))))
    ny, nx = final.shape
    x_axis = (
        np.arange(nx, dtype=float) + 1.0 - float(header["CRPIX1"])
    ) * float(header["CDELT1"]) + float(header["CRVAL1"])
    y_axis = (
        np.arange(ny, dtype=float) + 1.0 - float(header["CRPIX2"])
    ) * float(header["CDELT2"]) + float(header["CRVAL2"])
    xx, yy = np.meshgrid(x_axis, y_axis)
    aperture = (
        finite
        & (
            np.hypot(
                xx - center_x_arcsec,
                yy - center_y_arcsec,
            ) <= aperture_radius_arcsec
        )
    )
    if not np.any(aperture):
        aperture_delta = math.nan
        aperture_denominator = math.nan
    else:
        aperture_delta = float(
            np.sqrt(np.mean(np.square(delta[aperture])))
        )
        aperture_denominator = float(
            np.sqrt(np.mean(np.square(final[aperture])))
        )
    return {
        "whole_map_relative_rms_to_iteration_9": (
            whole_delta / whole_denominator
            if whole_denominator > 0.0 else math.nan
        ),
        "source_aperture_relative_rms_to_iteration_9": (
            aperture_delta / aperture_denominator
            if math.isfinite(aperture_delta)
            and math.isfinite(aperture_denominator)
            and aperture_denominator > 0.0 else math.nan
        ),
        "source_aperture_delta_rms_over_background_sigma": (
            aperture_delta / background_sigma
            if math.isfinite(aperture_delta)
            and math.isfinite(background_sigma)
            and background_sigma > 0.0 else math.nan
        ),
    }


def candidate_residuals(
    iteration_rows: list[dict],
    observation_rows: list[dict],
    stage_roots: dict[int, Path],
) -> list[dict]:
    indexed = {
        (int(row["obsnum"]), str(row["array"]), int(row["iteration"])): row
        for row in iteration_rows
    }
    result = []
    for observation in observation_rows:
        obsnum = int(observation["obsnum"])
        for mode, stop_field in (
            ("core", "core_first_stable_iteration"),
            ("strict_state", "strict_state_first_stable_iteration"),
        ):
            stop = observation[stop_field]
            if stop is None:
                continue
            for array in ARRAYS:
                stopped = indexed[(obsnum, array, int(stop))]
                final = indexed[(obsnum, array, 9)]
                amplitude = finite_ratio(
                    float(stopped["morphology_template_amplitude_scale"]),
                    float(final["morphology_template_amplitude_scale"]),
                ) - 1.0
                major = finite_ratio(
                    float(stopped["major_fwhm_over_morphology_template"]),
                    float(final["major_fwhm_over_morphology_template"]),
                ) - 1.0
                minor = finite_ratio(
                    float(stopped["minor_fwhm_over_morphology_template"]),
                    float(final["minor_fwhm_over_morphology_template"]),
                ) - 1.0
                root = stage_roots[obsnum]
                residual = {
                        "mode": mode,
                        "obsnum": obsnum,
                        "source": final["source"],
                        "source_morphology": final["source_morphology"],
                        "quality_rank": final["quality_rank"],
                        "quality_stratum": final["quality_stratum"],
                        "array": array,
                        "stop_iteration": stop,
                        "signed_amplitude_residual_fraction": amplitude,
                        "absolute_amplitude_residual_fraction":
                            abs(amplitude),
                        "centroid_residual_arcsec": math.hypot(
                            float(stopped["x_t_arcsec"])
                            - float(final["x_t_arcsec"]),
                            float(stopped["y_t_arcsec"])
                            - float(final["y_t_arcsec"]),
                        ),
                        "maximum_fwhm_residual_fraction": max(
                            abs(major), abs(minor)
                        ),
                    }
                residual.update(
                    map_residual_metrics(
                        map_path(root, obsnum, int(stop), array),
                        map_path(root, obsnum, 9, array),
                        center_x_arcsec=float(final["x_t_arcsec"]),
                        center_y_arcsec=float(final["y_t_arcsec"]),
                        aperture_radius_arcsec=float(
                            final["morphology_template_fit_radius_arcsec"]
                        ),
                        background_sigma=float(
                            final["map_background_sigma_mjy"]
                        ),
                    )
                )
                result.append(residual)
    return result


def summarize_candidate(
    observation_rows: list[dict],
    residual_rows: list[dict],
) -> list[dict]:
    result = []
    morphologies = sorted({
        str(row["source_morphology"]) for row in observation_rows
    })
    for mode, stop_field in (
        ("core", "core_first_stable_iteration"),
        ("strict_state", "strict_state_first_stable_iteration"),
    ):
        for morphology in ("all", *morphologies):
            observations = [
                row for row in observation_rows
                if morphology == "all"
                or row["source_morphology"] == morphology
            ]
            selected_residuals = [
                row for row in residual_rows
                if row["mode"] == mode
                and (
                    morphology == "all"
                    or row["source_morphology"] == morphology
                )
            ]
            stops = [
                int(row[stop_field])
                for row in observations if row[stop_field] is not None
            ]
            amplitudes = [
                float(row["absolute_amplitude_residual_fraction"])
                for row in selected_residuals
            ]
            signed = [
                float(row["signed_amplitude_residual_fraction"])
                for row in selected_residuals
            ]
            centroids = [
                float(row["centroid_residual_arcsec"])
                for row in selected_residuals
            ]
            fwhm = [
                float(row["maximum_fwhm_residual_fraction"])
                for row in selected_residuals
            ]
            map_rms = [
                float(row["whole_map_relative_rms_to_iteration_9"])
                for row in selected_residuals
            ]
            aperture_rms = [
                float(row["source_aperture_relative_rms_to_iteration_9"])
                for row in selected_residuals
            ]
            aperture_background = [
                float(row[
                    "source_aperture_delta_rms_over_background_sigma"
                ])
                for row in selected_residuals
            ]
            result.append(
                {
                    "mode": mode,
                    "source_morphology": morphology,
                    "observation_count": len(observations),
                    "resolved_observation_count": len(stops),
                    "resolved_observation_fraction":
                        len(stops) / len(observations),
                    "median_stop_iteration":
                        median(stops) if stops else math.nan,
                    "median_signed_amplitude_residual_percent":
                        100.0 * median(signed) if signed else math.nan,
                    "p90_absolute_amplitude_residual_percent":
                        100.0 * percentile(amplitudes, 90.0),
                    "maximum_absolute_amplitude_residual_percent":
                        100.0 * max(amplitudes) if amplitudes else math.nan,
                    "p90_centroid_residual_arcsec":
                        percentile(centroids, 90.0),
                    "maximum_centroid_residual_arcsec":
                        max(centroids) if centroids else math.nan,
                    "p90_fwhm_residual_percent":
                        100.0 * percentile(fwhm, 90.0),
                    "maximum_fwhm_residual_percent":
                        100.0 * max(fwhm) if fwhm else math.nan,
                    "p90_whole_map_relative_rms_percent":
                        100.0 * percentile(map_rms, 90.0),
                    "maximum_whole_map_relative_rms_percent":
                        100.0 * max(map_rms) if map_rms else math.nan,
                    "p90_source_aperture_relative_rms_percent":
                        100.0 * percentile(aperture_rms, 90.0),
                    "maximum_source_aperture_relative_rms_percent":
                        100.0 * max(aperture_rms)
                        if aperture_rms else math.nan,
                    "p90_source_aperture_delta_rms_over_background_sigma":
                        percentile(aperture_background, 90.0),
                    "maximum_source_aperture_delta_rms_over_background_sigma":
                        max(aperture_background)
                        if aperture_background else math.nan,
                }
            )
    return result


def candidate_yield_breakdown(
    observation_rows: list[dict],
) -> list[dict]:
    result = []
    for mode, stop_field in (
        ("core", "core_first_stable_iteration"),
        ("strict_state", "strict_state_first_stable_iteration"),
    ):
        for dimension in ("quality_stratum", "source"):
            values = sorted({
                str(row[dimension]) for row in observation_rows
            })
            for value in values:
                selected = [
                    row for row in observation_rows
                    if str(row[dimension]) == value
                ]
                stops = [
                    int(row[stop_field])
                    for row in selected if row[stop_field] is not None
                ]
                result.append(
                    {
                        "mode": mode,
                        "breakdown": dimension,
                        "group": value,
                        "observation_count": len(selected),
                        "resolved_observation_count": len(stops),
                        "resolved_observation_fraction":
                            len(stops) / len(selected),
                        "median_stop_iteration":
                            median(stops) if stops else math.nan,
                    }
                )
    return result


def continuation_candidates(
    criterion_rows: list[dict],
    residual_rows: list[dict],
) -> list[dict]:
    residuals_by_obs: dict[int, list[dict]] = {}
    for row in residual_rows:
        if row["mode"] == "core":
            residuals_by_obs.setdefault(int(row["obsnum"]), []).append(row)
    result = []
    for row in criterion_rows:
        obsnum = int(row["obsnum"])
        reasons = []
        if row["core_first_stable_iteration"] is None:
            reasons.append("no_core_stop_by_iteration_9")
            reasons.extend(
                f"{name}_not_individually_stable"
                for name in str(
                    row["individually_unresolved_criteria"]
                ).split(";")
                if name
            )
        for residual in residuals_by_obs.get(obsnum, []):
            if float(residual["absolute_amplitude_residual_fraction"]) > 0.05:
                reasons.append(f"{residual['array']}_amplitude_residual_gt_5pct")
            if float(residual["centroid_residual_arcsec"]) > 0.1:
                reasons.append(f"{residual['array']}_centroid_residual_gt_0p1arcsec")
            if float(residual["maximum_fwhm_residual_fraction"]) > 0.05:
                reasons.append(f"{residual['array']}_fwhm_residual_gt_5pct")
            if (
                float(
                    residual[
                        "source_aperture_relative_rms_to_iteration_9"
                    ]
                ) > 0.05
            ):
                reasons.append(
                    f"{residual['array']}_source_aperture_residual_gt_5pct"
                )
        if reasons:
            result.append(
                {
                    "obsnum": obsnum,
                    "source": row["source"],
                    "source_morphology": row["source_morphology"],
                    "quality_rank": row["quality_rank"],
                    "quality_stratum": row["quality_stratum"],
                    "core_first_stable_iteration":
                        row["core_first_stable_iteration"],
                    "continuation_class": row["continuation_class"],
                    "individually_unresolved_criteria":
                        row["individually_unresolved_criteria"],
                    "reasons": ";".join(sorted(set(reasons))),
                }
            )
    return result


def markdown_report(
    *,
    expected_count: int,
    analyzed_count: int,
    missing_obsnums: list[int],
    stage_a_audit: dict,
    stage_b_audit: dict,
    amplitude_summary: list[dict],
    behavior_summary: list[dict],
    morphology_effect_summary: list[dict],
    candidate_summary: list[dict],
    yield_breakdown: list[dict],
    continuation_rows: list[dict],
    plot_count: int,
    empirical_point_source_snr_included: bool,
) -> str:
    amplitude_rows = [
        row for row in amplitude_summary
        if row["source_morphology"] in {
            "all", "unresolved", "planetary_disk"
        }
        and row["array"] == "all"
        and float(row["tolerance_percent"]) in {2.0, 2.5, 3.0, 4.0, 5.0}
    ]
    candidate_rows = [
        row for row in candidate_summary
        if row["mode"] == "core"
    ]
    quality_rows = [
        row for row in yield_breakdown
        if row["mode"] == "core"
        and row["breakdown"] == "quality_stratum"
    ]
    source_rows = [
        row for row in yield_breakdown
        if row["mode"] == "core"
        and row["breakdown"] == "source"
    ]
    continuation_classes = {
        name: sum(
            row["continuation_class"] == name
            for row in continuation_rows
        )
        for name in ("measurement_limited", "trajectory_unresolved")
    }
    lines = [
        "# Full Fruit-Loop Population Analysis",
        "",
        f"- Expected observations: `{expected_count}`",
        f"- Analyzed observations: `{analyzed_count}`",
        f"- Missing observations: `{';'.join(map(str, missing_obsnums)) or 'none'}`",
        f"- Stage A audited jobs: "
        f"`{stage_a_audit['job_audit_pass_count']}/{stage_a_audit['job_count']}`",
        f"- Stage B audited jobs: "
        f"`{stage_b_audit['job_audit_pass_count']}/{stage_b_audit['job_count']}`",
        f"- Per-observation convergence plots: `{plot_count}`",
        f"- Empirical blank-sky point-source S/N included: "
        f"`{str(empirical_point_source_snr_included).lower()}`",
        "- Production stopping policy changed: `false`",
        "",
        "## Morphology-aware amplitude-only simulation",
        "",
        "| Morphology | Tolerance | Resolved | Median stop | "
        "Median absolute residual | P90 residual | Maximum residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in amplitude_rows:
        lines.append(
            f"| {row['source_morphology']} | "
            f"{float(row['tolerance_percent']):g}% | "
            f"{row['resolved_count']}/{row['trajectory_count']} | "
            f"{row['median_stop_iteration']:.1f} | "
            f"{row['median_absolute_residual_percent']:.2f}% | "
            f"{row['p90_absolute_residual_percent']:.2f}% | "
            f"{row['maximum_absolute_residual_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Trajectory behavior",
            "",
            "| Morphology | Array | Amplitude monotonic | Tail positive steps | "
            "Endpoint amplitude / seed | Background monotonic | "
            "Endpoint background / seed | Formal S/N / seed | "
            "Empirical S/N / seed | Legacy dynamic range / seed |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in behavior_summary:
        lines.append(
            f"| {row['source_morphology']} | {row['array']} | "
            f"{row['amplitude_monotonic_trajectory_count']}/"
            f"{row['trajectory_count']} | "
            f"{100.0 * row['tail_positive_amplitude_step_fraction']:.1f}% | "
            f"{row['endpoint_amplitude_ratio_seed_median']:.2f} | "
            f"{row['background_monotonic_trajectory_count']}/"
            f"{row['trajectory_count']} | "
            f"{row['endpoint_background_sigma_ratio_seed_median']:.2f} | "
            f"{row['endpoint_formal_snr_ratio_seed_median']:.2f} | "
            f"{row['endpoint_empirical_snr_ratio_seed_median']:.2f} | "
            f"{row['endpoint_legacy_dynamic_range_ratio_seed_median']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The formal template-fit S/N rises while the historical "
            "peak/full-map-RMS dynamic range can fall. Background sigma is "
            "not monotonically increasing; the historical apparent S/N loss "
            "is therefore not evidence of worsening source-free noise.",
            "",
            "## Planet-disk correction",
            "",
            "| Array | Median/max major-axis broadening | "
            "Median/max minor-axis broadening | "
            "P90/max tail amplitude-metric difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in morphology_effect_summary:
        lines.append(
            f"| {row['array']} | "
            f"{row['median_major_fwhm_broadening_percent']:.2f}% / "
            f"{row['maximum_major_fwhm_broadening_percent']:.2f}% | "
            f"{row['median_minor_fwhm_broadening_percent']:.2f}% / "
            f"{row['maximum_minor_fwhm_broadening_percent']:.2f}% | "
            f"{row['p90_absolute_tail_amplitude_metric_difference_percent']:.2f}% / "
            f"{row['maximum_absolute_tail_amplitude_metric_difference_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Candidate V0 all-array simulation",
            "",
            "The core rule uses 3% morphology-aware amplitude, 5% "
            "morphology-aware FWHM, 0.1 arcsec centroid, 5% successive map "
            "change, 5% weight change, less than 1% valid-mask symmetric "
            "difference, apply-phase learning, and a background sigma no "
            "more than 10% above seed. "
            "The strict-state variant additionally requires unchanged "
            "effective learning mask/penalty counts.",
            "",
            "| Morphology | Resolved observations | Median stop | "
            "Amplitude P90/max | Centroid P90/max | FWHM P90/max | "
            "Source-aperture residual P90/max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidate_rows:
        lines.append(
            f"| {row['source_morphology']} | "
            f"{row['resolved_observation_count']}/"
            f"{row['observation_count']} | "
            f"{row['median_stop_iteration']:.1f} | "
            f"{row['p90_absolute_amplitude_residual_percent']:.2f}% / "
            f"{row['maximum_absolute_amplitude_residual_percent']:.2f}% | "
            f"{row['p90_centroid_residual_arcsec']:.3f} / "
            f"{row['maximum_centroid_residual_arcsec']:.3f} arcsec | "
            f"{row['p90_fwhm_residual_percent']:.2f}% / "
            f"{row['maximum_fwhm_residual_percent']:.2f}% | "
            f"{row['p90_source_aperture_relative_rms_percent']:.2f}% / "
            f"{row['maximum_source_aperture_relative_rms_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "### Core-rule yield by frozen quality stratum",
            "",
            "| Stratum | Resolved observations | Median stop |",
            "|---|---:|---:|",
        ]
    )
    for row in quality_rows:
        lines.append(
            f"| {row['group']} | {row['resolved_observation_count']}/"
            f"{row['observation_count']} | "
            f"{row['median_stop_iteration']:.1f} |"
        )
    lines.extend(
        [
            "",
            "### Core-rule yield by source",
            "",
            "| Source | Resolved observations | Median stop |",
            "|---|---:|---:|",
        ]
    )
    for row in source_rows:
        lines.append(
            f"| {row['group']} | {row['resolved_observation_count']}/"
            f"{row['observation_count']} | "
            f"{row['median_stop_iteration']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Continuation review",
            "",
            f"`{len(continuation_rows)}` observations require continuation "
            "or trajectory review under the provisional residual guards.",
            "",
            f"- Measurement-limited: "
            f"`{continuation_classes['measurement_limited']}`",
            f"- Trajectory unresolved with individually measurable criteria: "
            f"`{continuation_classes['trajectory_unresolved']}`",
            "",
            "Radio sources and planetary disks are assessed independently. "
            "Planet amplitudes use each realized kernel convolved with the "
            "epoch-specific JPL Horizons uniform disk; planet widths are "
            "normalized by that disk-convolved template rather than the bare "
            "point-source kernel.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    matrix_rows = read_population_matrix(args.run_matrix)
    ephemerides = read_planet_ephemerides(args.planet_ephemerides)
    available, missing = available_matrix_rows(
        matrix_rows, args.stage_a_root, args.stage_b_root
    )
    if missing and not args.allow_incomplete:
        raise ValueError(
            f"population download incomplete; missing obsnums {missing}"
        )
    planet_obsnums = {
        int(row["obsnum"]) for row in matrix_rows
        if str(row["source"]).casefold() in PLANET_SOURCES
    }
    if set(ephemerides) != planet_obsnums:
        raise ValueError(
            "planet ephemeris identities do not match population matrix"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    stage_a_rows = [
        row for row in matrix_rows
        if row["phase"] == "sentinel_extension_first"
    ]
    stage_b_rows = [
        row for row in matrix_rows
        if row["phase"] == "population_after_sentinel_gate"
    ]
    audit_a_rows, audit_a = audit_stage(args.stage_a_root, stage_a_rows)
    audit_b_rows, audit_b = audit_stage(args.stage_b_root, stage_b_rows)
    audit_complete = bool(
        audit_a["job_audit_pass_count"] == audit_a["job_count"]
        and audit_b["job_audit_pass_count"] == audit_b["job_count"]
    )
    if not audit_complete and not args.allow_incomplete:
        raise ValueError(
            "population audit failed; use --allow-incomplete only for an "
            "explicitly provisional pass"
        )
    iteration_rows = load_population_metrics(
        args.stage_a_root,
        args.stage_b_root,
        available,
        ephemerides,
        workers=args.workers,
        include_empirical_point_source_snr=
            not args.skip_empirical_point_source_snr,
    )
    transition_rows = build_full_transition_metrics(iteration_rows)
    amplitude_rows = amplitude_stop_simulations(
        iteration_rows, transition_rows
    )
    amplitude_summary = summarize_amplitude_simulations(amplitude_rows)
    behavior_summary = trajectory_behavior_summary(iteration_rows)
    morphology_effect_summary = planet_morphology_effect_summary(
        iteration_rows, transition_rows
    )
    observation_rows = observation_stop_simulations(
        iteration_rows, transition_rows
    )
    criterion_rows = observation_criterion_status(
        observation_rows, transition_rows
    )
    trajectory_continuation_rows = [
        row for row in criterion_rows
        if row["continuation_class"] == "trajectory_unresolved"
    ]
    measurement_limited_rows = [
        row for row in criterion_rows
        if row["continuation_class"] == "measurement_limited"
    ]
    stage_roots = {
        int(row["obsnum"]): (
            args.stage_a_root
            if row["phase"] == "sentinel_extension_first"
            else args.stage_b_root
        )
        for row in available
    }
    residual_rows = candidate_residuals(
        iteration_rows, observation_rows, stage_roots
    )
    candidate_summary = summarize_candidate(
        observation_rows, residual_rows
    )
    yield_breakdown = candidate_yield_breakdown(observation_rows)
    continuation_rows = continuation_candidates(
        criterion_rows, residual_rows
    )

    outputs = (
        ("stage_a_job_audit.csv", audit_a_rows),
        ("stage_b_job_audit.csv", audit_b_rows),
        ("iteration_metrics.csv", iteration_rows),
        ("transition_metrics.csv", transition_rows),
        ("amplitude_stop_simulation.csv", amplitude_rows),
        ("amplitude_stop_summary.csv", amplitude_summary),
        ("trajectory_behavior_summary.csv", behavior_summary),
        ("planet_morphology_effect_summary.csv", morphology_effect_summary),
        ("observation_candidate_stop.csv", observation_rows),
        ("observation_criterion_status.csv", criterion_rows),
        (
            "trajectory_continuation_candidates.csv",
            trajectory_continuation_rows,
        ),
        ("measurement_limited_observations.csv", measurement_limited_rows),
        ("candidate_stop_residuals.csv", residual_rows),
        ("candidate_stop_summary.csv", candidate_summary),
        ("candidate_yield_breakdown.csv", yield_breakdown),
        ("continuation_candidates.csv", continuation_rows),
    )
    for filename, rows in outputs:
        if rows:
            write_csv(args.output / filename, rows)
    plot_count = (
        0
        if args.skip_plots else write_observation_plots(
            args.output, iteration_rows, transition_rows
        )
    )
    report = markdown_report(
        expected_count=len(matrix_rows),
        analyzed_count=len(available),
        missing_obsnums=missing,
        stage_a_audit=audit_a,
        stage_b_audit=audit_b,
        amplitude_summary=amplitude_summary,
        behavior_summary=behavior_summary,
        morphology_effect_summary=morphology_effect_summary,
        candidate_summary=candidate_summary,
        yield_breakdown=yield_breakdown,
        continuation_rows=continuation_rows,
        plot_count=plot_count,
        empirical_point_source_snr_included=
            not args.skip_empirical_point_source_snr,
    )
    (args.output / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version":
            "citlali-fruit-loop-full-population-analysis-v1",
        "population_complete": not missing and audit_complete,
        "expected_observation_count": len(matrix_rows),
        "analyzed_observation_count": len(available),
        "missing_obsnums": missing,
        "stage_a_root": str(args.stage_a_root.resolve()),
        "stage_b_root": str(args.stage_b_root.resolve()),
        "run_matrix": str(args.run_matrix.resolve()),
        "run_matrix_sha256": sha256(args.run_matrix),
        "planet_ephemerides": str(args.planet_ephemerides.resolve()),
        "planet_ephemerides_sha256": sha256(args.planet_ephemerides),
        "source_morphology": {
            "unresolved": "realized point-source kernel",
            "planetary_disk": (
                "realized kernel convolved with epoch-specific uniform disk"
            ),
        },
        "empirical_point_source_snr_included":
            not args.skip_empirical_point_source_snr,
        "candidate_v0": {
            "minimum_stop_iteration": MINIMUM_STOP_ITERATION,
            "required_consecutive_transitions": 2,
            "amplitude_limit_fraction": AMPLITUDE_LIMIT,
            "fwhm_limit_fraction": FWHM_LIMIT,
            "centroid_limit_arcsec": CENTROID_STEP_LIMIT_ARCSEC,
            "successive_map_limit_fraction": MAP_CHANGE_LIMIT,
            "map_weight_limit_fraction": WEIGHT_CHANGE_LIMIT,
            "valid_support_change_limit_fraction":
                VALID_SUPPORT_CHANGE_LIMIT,
            "background_sigma_seed_ratio_maximum": BACKGROUND_SEED_LIMIT,
            "all_arrays_required": True,
            "strict_state_variant": (
                "also requires unchanged effective mask/penalty counts"
            ),
        },
        "production_stopping_policy_changed": False,
        "files": {},
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][str(path.relative_to(args.output))] = sha256(
                path
            )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote full-population analysis for {len(available)}/"
        f"{len(matrix_rows)} observations to {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
