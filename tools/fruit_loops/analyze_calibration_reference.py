#!/usr/bin/env python3
"""Build the point-source fruit-loop calibration-reference evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from netCDF4 import Dataset

from tools.fruit_loops.compare_feedback_ablation import reduction_rows
from tools.fruit_loops.compare_injected_source_pair import comparison_rows


ARRAYS = ("a1100", "a1400", "a2000")
OBSNUMS = (133410, 144176, 148434, 151718, 153481)
TOLERANCES = (0.01, 0.02, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.home()
        / "work_toltec/local_data/"
        "2026-ENG-hero-multiyear-pointings-v1",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--legacy-metrics",
        type=Path,
        help="Optional existing TolAPT iteration_metrics.csv to hash.",
    )
    parser.add_argument(
        "--legacy-reproduction",
        type=Path,
        help=(
            "Optional regenerated iteration_metrics.csv; when supplied with "
            "--legacy-metrics, require byte-for-byte equality."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def first_map_path(redu: Path, obsnum: int, array: str = "a1100") -> Path:
    return (
        redu
        / str(obsnum)
        / "raw"
        / f"toltec_commissioning_{array}_pointing_{obsnum}_citlali.fits"
    )


def config_path(redu: Path) -> Path:
    candidates = sorted(
        path
        for path in redu.glob("citlali*.yaml")
        if path.name != "citlali_merged_config.yaml"
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one low-level config in {redu}; found {candidates}"
        )
    return candidates[0]


def scalar_checkpoint_value(path: Path, name: str) -> str:
    if not path.is_file():
        return ""
    with Dataset(path) as dataset:
        if name not in dataset.variables:
            return ""
        value = np.asarray(dataset.variables[name][...]).squeeze()
        if value.dtype.kind in {"S", "U"}:
            if value.ndim == 0:
                return str(value.item())
            return "".join(
                item.decode() if isinstance(item, bytes) else str(item)
                for item in value.flat
            ).rstrip("\x00")
        return str(value.item())


def redu_directories(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.glob("redu[0-9]*")
            if path.is_dir() and path.name[4:].isdigit()
        ),
        key=lambda path: int(path.name[4:]),
    )


def run_record(
    family: str,
    run_id: str,
    root: Path,
    obsnum: int,
    *,
    disposition: str,
    exact_restart: str = "not_applicable",
) -> dict:
    reductions = redu_directories(root)
    if not reductions:
        raise ValueError(f"no iteration products in {root}")
    iterations = []
    versions = set()
    config_hashes = set()
    for redu in reductions:
        path = first_map_path(redu, obsnum)
        with fits.open(path, memmap=True) as hdul:
            iterations.append(int(hdul[0].header["FRUITLOOPS_ITER"]))
        index_path = redu / "index.yaml"
        if index_path.is_file():
            index = yaml.safe_load(index_path.read_text())
            versions.update(
                str(value) for value in index.get("citlali_version", [])
            )
        candidates = [
            path
            for path in redu.glob("citlali*.yaml")
            if path.name != "citlali_merged_config.yaml"
        ]
        if len(candidates) == 1:
            config_hashes.add(sha256(candidates[0]))
    checkpoint = next(
        (
            redu / "citlali_restart_checkpoint.nc"
            for redu in reversed(reductions)
            if (redu / "citlali_restart_checkpoint.nc").is_file()
        ),
        reductions[-1] / "citlali_restart_checkpoint.nc",
    )
    return {
        "family": family,
        "run_id": run_id,
        "obsnum": obsnum,
        "root": str(root),
        "n_iteration_products": len(reductions),
        "first_absolute_iteration": min(iterations),
        "last_absolute_iteration": max(iterations),
        "iterations_contiguous": iterations
        == list(range(min(iterations), max(iterations) + 1)),
        "citlali_versions": ";".join(sorted(versions)),
        "config_sha256_values": ";".join(sorted(config_hashes)),
        "checkpoint_schema": scalar_checkpoint_value(
            checkpoint, "schema_version"
        ),
        "exact_restart_gate": exact_restart,
        "disposition": disposition,
    }


def run_inventory(project_root: Path) -> list[dict]:
    diagnostics = project_root / "diagnostics"
    rows = []
    real = diagnostics / "fruitloops5_rc1"
    for obsnum in OBSNUMS:
        rows.append(
            run_record(
                "real_source_five_iteration",
                f"obs{obsnum}",
                real / f"obs{obsnum}/reduced",
                obsnum,
                disposition="valid_existing_evidence",
            )
        )

    ablation = diagnostics / "fruitloops5_rc1_ablation/obs133410"
    excluded = {
        "bin", "config", "logs", "logs_followup", "setup_followup",
    }
    for path in sorted(ablation.iterdir()):
        reduced = path / "reduced"
        if path.name in excluded or not reduced.is_dir():
            continue
        rows.append(
            run_record(
                "ablation",
                path.name,
                reduced,
                133410,
                disposition=(
                    "intentional_no_op_after_iteration_1"
                    if path.name == "snr_only_s200"
                    else "valid_existing_evidence"
                ),
            )
        )

    v1 = diagnostics / "fruitloops5_rc1_injected_source/obs133410"
    for label in ("control", "injected"):
        rows.append(
            run_record(
                "injected_source_schema_v1",
                label,
                v1 / f"{label}/reduced",
                133410,
                disposition="quarantined_checkpoint_v1_restart_mismatch",
                exact_restart="FAIL",
            )
        )

    v2 = diagnostics / "fruitloops5_rc1_injected_source_v2/obs133410"
    rows.append(
        run_record(
            "injected_source_schema_v2",
            "uninterrupted_reference",
            v2 / "reference/reduced",
            133410,
            disposition="valid_existing_evidence",
            exact_restart="reference",
        )
    )
    for label in ("control", "injected"):
        rows.append(
            run_record(
                "injected_source_schema_v2",
                label,
                v2 / f"{label}/reduced",
                133410,
                disposition="valid_existing_evidence",
                exact_restart="PASS",
            )
        )
        rows.append(
            run_record(
                "injected_source_schema_v2_extended",
                label,
                v2 / f"extended_9_18/{label}/reduced",
                133410,
                disposition="valid_existing_evidence",
                exact_restart="PASS",
            )
        )
    return rows


def configured_beammap_fluxes(config: dict) -> dict[str, float]:
    values = {}
    for item in config["inputs"][0]["cal_items"]:
        if item.get("type") != "photometry":
            continue
        for flux in item["beammap_source"]["fluxes"]:
            values[str(flux["array_name"])] = float(flux["value_mJy"])
    return values


def observation_inventory(project_root: Path) -> list[dict]:
    rows = []
    real = project_root / "diagnostics/fruitloops5_rc1"
    for obsnum in OBSNUMS:
        redu = real / f"obs{obsnum}/reduced/redu00"
        table = Table.read(
            redu
            / str(obsnum)
            / "raw"
            / f"ppt_commissioning_pointing_{obsnum}_citlali.ecsv"
        )
        map_path = first_map_path(redu, obsnum)
        header = fits.getheader(map_path)
        config = yaml.safe_load(config_path(redu).read_text())
        apt_path_string = next(
            item["filepath"]
            for item in config["inputs"][0]["cal_items"]
            if item.get("type") == "array_prop_table"
        )
        apt_path = (
            project_root / "apts/hero_rc1" / Path(apt_path_string).name
        )
        apt_meta = Table.read(apt_path).meta
        source = str(table.meta["source"])
        apt_source = str(apt_meta.get("source", "unknown"))
        source_matches = source.casefold() == apt_source.casefold()
        configured = configured_beammap_fluxes(config)
        row = {
            "obsnum": obsnum,
            "pointing_source": source,
            "observation_date_utc": str(table.meta.get("date", "")),
            "observation_mjd": float(table.meta.get("mjd", math.nan)),
            "mean_source_elevation_deg": float(
                table.meta.get("MEAN_SOURCE_EL", math.nan)
            ),
            "radiometer_tau": float(
                header.get("HEADER.RADIOMETER.TAU", math.nan)
            ),
            "weather_temperature_c": float(
                header.get("HEADER.WEATHER.TEMPERATURE", math.nan)
            ),
            "weather_humidity_percent": float(
                header.get("HEADER.WEATHER.HUMIDITY", math.nan)
            ),
            "weather_wind_speed": float(
                header.get("HEADER.WEATHER.WINDSPEED1", math.nan)
            ),
            "matched_apt_path": str(apt_path),
            "matched_apt_sha256": sha256(apt_path),
            "matched_apt_obsnum": int(
                apt_meta.get("obsnum_matched", -1)
            ),
            "matched_apt_source": apt_source,
            "matched_apt_date_utc": str(apt_meta.get("date", "")),
            "matched_apt_source_matches_pointing": source_matches,
            "apt_a1100_flux_mjy_beam": float(
                apt_meta["a1100_flux"][0]
            ),
            "apt_a1400_flux_mjy_beam": float(
                apt_meta["a1400_flux"][0]
            ),
            "apt_a2000_flux_mjy_beam": float(
                apt_meta["a2000_flux"][0]
            ),
            "configured_beammap_a1100_mjy": configured.get(
                "a1100", math.nan
            ),
            "configured_beammap_a1400_mjy": configured.get(
                "a1400", math.nan
            ),
            "configured_beammap_a2000_mjy": configured.get(
                "a2000", math.nan
            ),
            "configured_beammap_flux_authoritative": False,
            "external_flux_authority": (
                "ALMA calibrator flux service, date/frequency matched"
                if source.casefold() in {"3c273", "3c279"}
                else "epoch- and bandpass-integrated planetary brightness model"
            ),
            "local_absolute_flux_status": (
                "matched_APT_reference_available_but_not_injected_truth"
                if source_matches
                else "matched_APT_is_for_a_different_source"
            ),
            "associated_science_observations": "",
            "associated_science_status":
                "none_recorded_in_local_project_or_discovered_configs",
        }
        rows.append(row)
    return rows


def add_real_step_metrics(rows: list[dict]) -> None:
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["variant"]), int(row["obsnum"]), str(row["array"])),
            [],
        ).append(row)
    ratio_fields = (
        "amplitude", "kernel_normalized_amplitude", "major_fwhm_arcsec",
        "minor_fwhm_arcsec", "sig2noise", "map_background_sigma",
        "map_weight_median",
    )
    for group in groups.values():
        group.sort(key=lambda item: int(item["iteration"]))
        seed = group[0]
        for index, row in enumerate(group):
            previous = group[index - 1] if index else None
            for field in ratio_fields:
                value = float(row[field])
                seed_value = float(seed[field])
                row[f"{field}_ratio_seed"] = (
                    value / seed_value if seed_value != 0.0 else math.nan
                )
                row[f"{field}_change_fraction"] = (
                    value / float(previous[field]) - 1.0
                    if previous is not None
                    and float(previous[field]) != 0.0
                    else math.nan
                )
            row["centroid_shift_from_previous_arcsec"] = (
                math.hypot(
                    float(row["x_t_arcsec"])
                    - float(previous["x_t_arcsec"]),
                    float(row["y_t_arcsec"])
                    - float(previous["y_t_arcsec"]),
                )
                if previous is not None else math.nan
            )
            row["centroid_shift_from_seed_arcsec"] = math.hypot(
                float(row["x_t_arcsec"]) - float(seed["x_t_arcsec"]),
                float(row["y_t_arcsec"]) - float(seed["y_t_arcsec"]),
            )


def real_iteration_metrics(project_root: Path) -> list[dict]:
    rows = []
    root = project_root / "diagnostics/fruitloops5_rc1"
    for obsnum in OBSNUMS:
        rows.extend(
            reduction_rows(
                "five_observation_sequence",
                root / f"obs{obsnum}/reduced",
                obsnum,
            )
        )
    rows.extend(
        reduction_rows(
            "obs133410_full_policy_10_iters",
            project_root
            / "diagnostics/fruitloops5_rc1_ablation/obs133410/"
            "full_policy_10_iters/reduced",
            133410,
        )
    )
    add_real_step_metrics(rows)
    return rows


def injected_metrics(project_root: Path) -> list[dict]:
    root = (
        project_root
        / "diagnostics/fruitloops5_rc1_injected_source_v2/obs133410"
    )
    return comparison_rows(
        root / "extended_9_18/control/reduced",
        root / "extended_9_18/injected/reduced",
        root / "extended_9_18/setup_pair/manifest.yaml",
        root / "reference/reduced/redu09",
        133410,
    )


def trajectory_summary(rows: list[dict]) -> list[dict]:
    result = []
    for array in ARRAYS:
        sequence = sorted(
            (row for row in rows if row["array"] == array),
            key=lambda row: int(row["iteration"]),
        )
        recovery = np.asarray(
            [
                row["kernel_normalized_amplitude_recovery_fraction"]
                for row in sequence
            ],
            dtype=float,
        )
        increments = np.diff(recovery)
        result.append(
            {
                "array": array,
                "first_iteration": sequence[0]["iteration"],
                "last_iteration": sequence[-1]["iteration"],
                "monotonic_non_decreasing": bool(
                    np.all(increments >= 0.0)
                ),
                "all_positive_increments": bool(np.all(increments > 0.0)),
                "first_increment": float(increments[0]),
                "penultimate_increment": float(increments[-2]),
                "last_increment": float(increments[-1]),
                "increment_contraction_first_to_last":
                    float(increments[-1] / increments[0]),
                "last_recovery_fraction": float(recovery[-1]),
                "remaining_attenuation_fraction": float(1.0 - recovery[-1]),
                "last_three_iteration_span": float(
                    np.max(recovery[-3:]) - np.min(recovery[-3:])
                ),
                "last_major_fwhm_over_kernel":
                    float(sequence[-1]["major_fwhm_over_kernel"]),
                "last_minor_fwhm_over_kernel":
                    float(sequence[-1]["minor_fwhm_over_kernel"]),
                "last_centroid_error_arcsec":
                    float(sequence[-1]["centroid_error_arcsec"]),
                "last_successive_map_relative_rms":
                    float(
                        sequence[-1][
                            "successive_transfer_delta_relative_rms"
                        ]
                    ),
                "classification":
                    "shape_and_amplitude_plateau_with_residual_attenuation;"
                    "whole_map_not_yet_stable_at_1_percent",
            }
        )
    return result


def threshold_assessment(rows: list[dict]) -> list[dict]:
    result = []
    for array in ARRAYS:
        sequence = sorted(
            (row for row in rows if row["array"] == array),
            key=lambda row: int(row["iteration"]),
        )
        transitions = []
        for previous, current in zip(sequence, sequence[1:]):
            transitions.append(
                {
                    "iteration": int(current["iteration"]),
                    "amplitude": abs(
                        float(
                            current[
                                "kernel_normalized_amplitude_recovery_fraction"
                            ]
                        )
                        / float(
                            previous[
                                "kernel_normalized_amplitude_recovery_fraction"
                            ]
                        )
                        - 1.0
                    ),
                    "major_fwhm": abs(
                        float(current["major_fwhm_over_kernel"])
                        / float(previous["major_fwhm_over_kernel"])
                        - 1.0
                    ),
                    "minor_fwhm": abs(
                        float(current["minor_fwhm_over_kernel"])
                        / float(previous["minor_fwhm_over_kernel"])
                        - 1.0
                    ),
                    "map": float(
                        current["successive_transfer_delta_relative_rms"]
                    ),
                    "s2n": abs(
                        float(current["injected_fit_s2n"])
                        / float(previous["injected_fit_s2n"])
                        - 1.0
                    ),
                    "centroid": float(current["centroid_error_arcsec"]),
                }
            )
        tail = transitions[-2:]
        for tolerance in TOLERANCES:
            amplitude_pass = all(
                item["amplitude"] < tolerance for item in tail
            )
            fwhm_pass = all(
                max(item["major_fwhm"], item["minor_fwhm"]) < tolerance
                for item in tail
            )
            map_pass = all(item["map"] < tolerance for item in tail)
            snr_pass = all(item["s2n"] < tolerance for item in tail)
            centroid_pass = all(
                item["centroid"] < 0.1 for item in tail
            )
            result.append(
                {
                    "array": array,
                    "endpoint_iteration": sequence[-1]["iteration"],
                    "window_transitions": ";".join(
                        str(item["iteration"]) for item in tail
                    ),
                    "tolerance_percent": int(round(100 * tolerance)),
                    "amplitude_two_step_pass": amplitude_pass,
                    "fwhm_two_step_pass": fwhm_pass,
                    "centroid_below_0p1_arcsec_pass": centroid_pass,
                    "successive_map_two_step_pass": map_pass,
                    "s2n_two_step_pass": snr_pass,
                    "all_candidate_diagnostics_pass":
                        amplitude_pass
                        and fwhm_pass
                        and centroid_pass
                        and map_pass
                        and snr_pass,
                    "maximum_amplitude_change_fraction": max(
                        item["amplitude"] for item in tail
                    ),
                    "maximum_fwhm_change_fraction": max(
                        max(item["major_fwhm"], item["minor_fwhm"])
                        for item in tail
                    ),
                    "maximum_centroid_error_arcsec": max(
                        item["centroid"] for item in tail
                    ),
                    "maximum_successive_map_relative_rms": max(
                        item["map"] for item in tail
                    ),
                    "maximum_s2n_change_fraction": max(
                        item["s2n"] for item in tail
                    ),
                }
            )
    return result


def transfer_comparison(rows: list[dict]) -> list[dict]:
    endpoint = {
        str(row["array"]): row
        for row in rows
        if int(row["iteration"]) == max(
            int(item["iteration"]) for item in rows
        )
    }
    result = []
    for obsnum in OBSNUMS:
        for array in ARRAYS:
            point = endpoint.get(array) if obsnum == 133410 else None
            result.append(
                {
                    "pointing_obsnum": obsnum,
                    "science_obsnum": "",
                    "array": array,
                    "pointing_kernel_normalized_recovery_fraction":
                        (
                            point[
                                "kernel_normalized_amplitude_recovery_fraction"
                            ]
                            if point is not None else math.nan
                        ),
                    "science_recovery_fraction": math.nan,
                    "science_minus_pointing_recovery_fraction": math.nan,
                    "prediction_tolerance_pass": "",
                    "status":
                        "not_measured_no_associated_science_products_and_"
                        "science_injection_not_implemented",
                }
            )
    return result


def planned_unity_runs() -> list[dict]:
    return [
        {
            "run_id": "C0",
            "readiness": "ready_after_owner_selects_tolerance",
            "question": "Does the iteration-18 transfer satisfy a sustained "
                        "plateau criterion?",
            "obsnum": 133410,
            "mode": "pointing",
            "branches": "control;injected",
            "amplitude_scale": "existing_representative_1x",
            "source_position": "map_center",
            "checkpoint": "existing_exact-gated_v2_iteration_18_lineage",
            "block_iterations": 3,
            "minimum_unity_jobs": 2,
            "gate": "evaluate after each block; continue only while a "
                    "candidate diagnostic fails and change remains material",
            "blocker": "scientific tolerance not selected",
        },
        {
            "run_id": "L0_faint",
            "readiness": "ready",
            "question": "Is point-source transfer linear at 0.1x amplitude?",
            "obsnum": 133410,
            "mode": "pointing",
            "branches": "control;injected",
            "amplitude_scale": "0.1x_existing_truth",
            "source_position": "map_center",
            "checkpoint": "v2_reference_iteration_8",
            "block_iterations": 10,
            "minimum_unity_jobs": 2,
            "gate": "exact restarted control equals uninterrupted iteration 9",
            "blocker": "",
        },
        {
            "run_id": "L0_bright",
            "readiness": "ready",
            "question": "Is point-source transfer linear at 3x amplitude?",
            "obsnum": 133410,
            "mode": "pointing",
            "branches": "control;injected",
            "amplitude_scale": "3x_existing_truth",
            "source_position": "map_center",
            "checkpoint": "v2_reference_iteration_8",
            "block_iterations": 10,
            "minimum_unity_jobs": 2,
            "gate": "exact restarted control equals uninterrupted iteration 9",
            "blocker": "",
        },
        {
            "run_id": "P0",
            "readiness": "blocked_pending_design_approval",
            "question": "Does transfer depend on map position or projection?",
            "obsnum": 133410,
            "mode": "pointing",
            "branches": "control;injected",
            "amplitude_scale": "existing_representative_1x",
            "source_position": "one_frozen_off-center_well-covered_position",
            "checkpoint": "v2_reference_iteration_8",
            "block_iterations": 10,
            "minimum_unity_jobs": 2,
            "gate": "same exact-restart and pair-isolation gates as L0",
            "blocker": "current diagnostic has no position parameter",
        },
        {
            "run_id": "R0_dry",
            "readiness": "ready_after_fresh_v2_reference",
            "question": "Does transfer generalize to a very dry planet "
                        "pointing?",
            "obsnum": 144176,
            "mode": "pointing",
            "branches": "reference;control;injected",
            "amplitude_scale": "synthetic_representative_per-array",
            "source_position": "map_center",
            "checkpoint": "fresh_v2_reference_iteration_8",
            "block_iterations": 5,
            "minimum_unity_jobs": 3,
            "gate": "exact restarted control equals uninterrupted iteration 9",
            "blocker": "",
        },
        {
            "run_id": "R0_high_tau",
            "readiness": "ready_after_fresh_v2_reference",
            "question": "Does transfer generalize to the highest-tau "
                        "pointing?",
            "obsnum": 151718,
            "mode": "pointing",
            "branches": "reference;control;injected",
            "amplitude_scale": "synthetic_representative_per-array",
            "source_position": "map_center",
            "checkpoint": "fresh_v2_reference_iteration_8",
            "block_iterations": 5,
            "minimum_unity_jobs": 3,
            "gate": "exact restarted control equals uninterrupted iteration 9",
            "blocker": "",
        },
        {
            "run_id": "S0",
            "readiness": "blocked_pending_design_approval_and_association",
            "question": "Does pointing-derived transfer predict the same "
                        "known source in associated science processing?",
            "obsnum": "",
            "mode": "matched_pointing_and_science",
            "branches": "pointing_control;pointing_injected;"
                        "science_control;science_injected",
            "amplitude_scale": "same_truth_in_both_modes",
            "source_position": "same_sky_coordinate_in_both_modes",
            "checkpoint": "mode-matched_v2_checkpoints",
            "block_iterations": 5,
            "minimum_unity_jobs": 4,
            "gate": "science recovery predicted by pointing within selected "
                    "tolerance in every array",
            "blocker": "no local pointing-science association; science "
                       "injection seam is not implemented",
        },
    ]


def plot_real_observation(
    rows: list[dict], obsnum: int, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = [
        row for row in rows
        if int(row["obsnum"]) == obsnum
    ]
    variants = sorted(
        {str(row["variant"]) for row in candidates},
        key=lambda name: sum(
            str(row["variant"]) == name for row in candidates
        ),
        reverse=True,
    )
    selected = [
        row for row in candidates if row["variant"] == variants[0]
    ]
    panels = (
        ("kernel_normalized_amplitude_ratio_seed", "Kernel-normalized amp / seed"),
        ("major_fwhm_over_kernel", "Major FWHM / kernel"),
        ("centroid_shift_from_seed_arcsec", "Centroid shift from seed (arcsec)"),
        ("sig2noise_ratio_seed", "S/N / seed"),
        ("successive_map_delta_relative_rms", "Successive whole-map relative RMS"),
        ("map_background_sigma_ratio_seed", "Background sigma / seed"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5))
    for axis, (field, ylabel) in zip(axes.flat, panels, strict=True):
        for array in ARRAYS:
            sequence = sorted(
                (row for row in selected if row["array"] == array),
                key=lambda row: int(row["iteration"]),
            )
            axis.plot(
                [row["iteration"] for row in sequence],
                [row[field] for row in sequence],
                marker="o",
                label=array,
            )
        axis.set_xlabel("Absolute fruit-loop iteration")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Real pointing observation {obsnum}: {variants[0]}")
    fig.tight_layout()
    fig.savefig(output / f"real_obs{obsnum}_convergence.png", dpi=180)
    plt.close(fig)


def plot_injected(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        (
            "kernel_normalized_amplitude_recovery_fraction",
            "Kernel-normalized amplitude recovery",
        ),
        ("kernel_projection_recovery_fraction", "Full-map kernel projection"),
        ("major_fwhm_over_kernel", "Major FWHM / kernel"),
        ("centroid_error_arcsec", "Centroid error (arcsec)"),
        (
            "successive_transfer_delta_relative_rms",
            "Successive transfer-map relative RMS",
        ),
        ("injected_fit_s2n", "Ordinary injected-map fitted S/N"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5))
    for axis, (field, ylabel) in zip(axes.flat, panels, strict=True):
        for array in ARRAYS:
            sequence = sorted(
                (row for row in rows if row["array"] == array),
                key=lambda row: int(row["iteration"]),
            )
            axis.plot(
                [row["iteration"] for row in sequence],
                [row[field] for row in sequence],
                marker="o",
                label=array,
            )
        axis.set_xlabel("Absolute fruit-loop iteration")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Obs 133410 injected-minus-control transfer (checkpoint v2)"
    )
    fig.tight_layout()
    fig.savefig(output / "injected_obs133410_convergence.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.legacy_reproduction and not args.legacy_metrics:
        raise ValueError(
            "--legacy-reproduction requires --legacy-metrics"
        )
    legacy_reproduced = False
    if args.legacy_metrics and args.legacy_reproduction:
        legacy_reproduced = (
            args.legacy_metrics.read_bytes()
            == args.legacy_reproduction.read_bytes()
        )
        if not legacy_reproduced:
            raise ValueError(
                "regenerated legacy metrics differ from the archived metrics"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    real_rows = real_iteration_metrics(args.project_root)
    injected_rows = injected_metrics(args.project_root)
    inventories = observation_inventory(args.project_root)
    runs = run_inventory(args.project_root)
    trajectory = trajectory_summary(injected_rows)
    thresholds = threshold_assessment(injected_rows)
    transfer = transfer_comparison(injected_rows)
    planned_runs = planned_unity_runs()

    for name, rows in (
        ("observation_inventory.csv", inventories),
        ("run_inventory.csv", runs),
        ("real_iteration_metrics.csv", real_rows),
        ("injected_source_iteration_metrics.csv", injected_rows),
        ("injected_trajectory_summary.csv", trajectory),
        ("convergence_threshold_assessment.csv", thresholds),
        ("pointing_science_transfer_comparison.csv", transfer),
        ("planned_unity_run_matrix.csv", planned_runs),
    ):
        write_csv(args.output / name, rows)
    for obsnum in OBSNUMS:
        plot_real_observation(real_rows, obsnum, args.output)
    plot_injected(injected_rows, args.output)

    binaries = [
        args.project_root
        / "diagnostics/fruitloops5_rc1_ablation/obs133410/bin/"
        "5eba09a081bb4a4d0a6b97eb1eebb99477be1470cb3a994e674c2e8733255d3e/"
        "citlali",
        args.project_root
        / "diagnostics/fruitloops5_rc1_injected_source/obs133410/bin/"
        "citlali-a5fcad296",
        args.project_root
        / "diagnostics/fruitloops5_rc1_injected_source_v2/obs133410/bin/"
        "citlali-7d8fd23f6",
    ]
    manifest = {
        "schema_version":
            "citlali-fruit-loop-calibration-reference-analysis-v1",
        "project_root": str(args.project_root),
        "real_metric_rows": len(real_rows),
        "injected_metric_rows": len(injected_rows),
        "exact_restart_gate": "PASS",
        "legacy_metrics_reproduced_byte_for_byte": legacy_reproduced,
        "legacy_metrics_path":
            str(args.legacy_metrics) if args.legacy_metrics else "",
        "legacy_reproduction_path":
            (
                str(args.legacy_reproduction)
                if args.legacy_reproduction else ""
            ),
        "legacy_reproduction_sha256":
            (
                sha256(args.legacy_reproduction)
                if args.legacy_reproduction
                and args.legacy_reproduction.is_file()
                else ""
            ),
        "legacy_metrics_sha256":
            (
                sha256(args.legacy_metrics)
                if args.legacy_metrics and args.legacy_metrics.is_file()
                else ""
            ),
        "retained_binary_sha256": {
            str(path): sha256(path) for path in binaries if path.is_file()
        },
        "production_defaults_changed": False,
        "science_transfer_status":
            "not_measured; requires approved science injection seam and "
            "associated science observation",
        "files": sorted(
            {path.name for path in args.output.iterdir()}
            | {"manifest.json"}
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote calibration-reference evidence to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
