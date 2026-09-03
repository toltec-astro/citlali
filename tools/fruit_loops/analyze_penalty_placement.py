#!/usr/bin/env python3
"""Analyze the registered SCI-FRUIT EL-F8 penalty-placement test."""

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
    require_exact_maps,
)
from tools.fruit_loops.analyze_shared_start_response import (
    aperture_fraction,
    fruit_iteration,
    iteration_dir,
    make_region_masks,
    native_compact_metrics,
    projection,
    roundoff_bound,
    world_axes,
)
from tools.fruit_loops.compare_injected_source_pair import (
    ARRAYS,
    file_record,
    gaussian_center_for_map_world_offset,
    product_path,
    rms,
)
from tools.fruit_loops.edit_restart_checkpoint_learning_policy import (
    audit_transformation as audit_learning_policy_transformation,
)
from tools.fruit_loops.edit_restart_checkpoint_penalty import sha256, values_equal


COMPONENTS = ("T_current", "T_map", "D_current", "D_map", "Q")
REGIONS = (
    "complete_map",
    "injected_source_r20",
    "neptune_r20",
    "annulus_r40_120_excluding_neptune_r25",
)
FITS_EXTENSION_NAMES = {
    "T_current": "T_CURRENT",
    "T_map": "T_MAP",
    "D_current": "D_CURRENT",
    "D_map": "D_MAP",
    "Q": "Q_EARLY",
}
TRAJECTORIES = ("c5-current", "a5-current", "c5-map", "a5-map")


def scalar_text(value: object) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def normalized_learning_policy(value: object) -> dict:
    policy = yaml.safe_load(scalar_text(value))
    if not isinstance(policy, dict):
        raise ValueError("checkpoint learning_policy_yaml is not a mapping")
    policy.setdefault(
        "map_pixel_outlier_detector_exclusion_application", "pre_cleaning"
    )
    return policy


def require_compatible_checkpoint(
    expected_path: Path,
    actual_path: Path,
    allowed_differences: set[str],
) -> dict:
    """Require scientific checkpoint identity with bounded provenance changes."""
    with Dataset(expected_path) as expected, Dataset(actual_path) as actual:
        if expected.ncattrs() != actual.ncattrs():
            raise ValueError("compatibility checkpoint global attributes differ")
        for name in expected.ncattrs():
            if expected.getncattr(name) != actual.getncattr(name):
                raise ValueError(
                    f"compatibility checkpoint global attribute differs: {name}"
                )
        if set(expected.dimensions) != set(actual.dimensions):
            raise ValueError("compatibility checkpoint dimensions differ")
        for name, dimension in expected.dimensions.items():
            if len(dimension) != len(actual.dimensions[name]):
                raise ValueError(
                    f"compatibility checkpoint dimension differs: {name}"
                )
        if set(expected.variables) != set(actual.variables):
            raise ValueError("compatibility checkpoint variables differ")

        observed_allowed = []
        for name, expected_variable in expected.variables.items():
            actual_variable = actual.variables[name]
            if (
                expected_variable.dimensions != actual_variable.dimensions
                or expected_variable.dtype != actual_variable.dtype
                or expected_variable.ncattrs() != actual_variable.ncattrs()
            ):
                raise ValueError(
                    f"compatibility checkpoint structure differs: {name}"
                )
            for attribute in expected_variable.ncattrs():
                if expected_variable.getncattr(
                    attribute
                ) != actual_variable.getncattr(attribute):
                    raise ValueError(
                        "compatibility checkpoint variable attribute differs: "
                        f"{name}:{attribute}"
                    )
            expected_value = expected_variable[...]
            actual_value = actual_variable[...]
            if values_equal(expected_value, actual_value):
                continue
            if name not in allowed_differences:
                raise ValueError(
                    f"scientific checkpoint value differs: {name}"
                )
            if name == "learning_policy_yaml" and (
                normalized_learning_policy(expected_value)
                != normalized_learning_policy(actual_value)
            ):
                raise ValueError(
                    "learning policy differs beyond the registered default key"
                )
            observed_allowed.append(name)
    return {
        "scientific_values_identical": True,
        "observed_allowed_differences": sorted(observed_allowed),
    }


def read_merged_config(redu: Path) -> dict:
    config = yaml.safe_load((redu / "citlali_merged_config.yaml").read_text())
    if not isinstance(config, dict):
        raise ValueError(f"invalid merged configuration in {redu}")
    return config


def placement(config: dict) -> str:
    return str(
        config["timestream"]["learning"]
        ["map_pixel_outlier_detector_exclusion_application"]
    )


def normalized_placement_pair(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized["runtime"]["output_dir"] = "<paired-output>"
    normalized["timestream"]["fruit_loops"]["restart_path"] = (
        "<paired-restart>"
    )
    normalized["timestream"]["learning"][
        "map_pixel_outlier_detector_exclusion_application"
    ] = "<paired-application>"
    return normalized


def require_placement_pair(
    current_redu: Path, map_redu: Path, branch: str
) -> None:
    current = read_merged_config(current_redu)
    moved = read_merged_config(map_redu)
    if placement(current) != "pre_cleaning":
        raise ValueError(f"{branch} current placement is not pre_cleaning")
    if placement(moved) != "pre_mapmaking":
        raise ValueError(f"{branch} map placement is not pre_mapmaking")
    if normalized_placement_pair(current) != normalized_placement_pair(moved):
        raise ValueError(
            f"{branch} configurations differ beyond placement and paths"
        )


def reductions(manifest: dict) -> dict[str, Path]:
    obsnum = int(manifest["obsnum"])
    iteration = int(manifest["iteration"])
    return {
        "C_current": iteration_dir(
            Path(manifest["c5_current_root"]), obsnum, iteration
        ),
        "A_current": iteration_dir(
            Path(manifest["a5_current_root"]), obsnum, iteration
        ),
        "C_map": iteration_dir(
            Path(manifest["c5_map_root"]), obsnum, iteration
        ),
        "A_map": iteration_dir(
            Path(manifest["a5_map_root"]), obsnum, iteration
        ),
        "N5": Path(manifest["existing_without_uid4460_iteration_5"]),
    }


def validate_compatibility(manifest: dict) -> dict:
    obsnum = int(manifest["obsnum"])
    iteration = int(manifest["iteration"])
    allowed = set(manifest["compatibility_checkpoint_allowed_differences"])
    pairs = {
        "control": (
            Path(manifest["existing_control_iteration_5"]),
            iteration_dir(Path(manifest["c5_current_root"]), obsnum, iteration),
        ),
        "injected": (
            Path(manifest["existing_injected_iteration_5"]),
            iteration_dir(Path(manifest["a5_current_root"]), obsnum, iteration),
        ),
    }
    result = {}
    for label, (expected, actual) in pairs.items():
        planes = require_exact_maps(expected, actual, obsnum)
        checkpoint = require_compatible_checkpoint(
            expected / "citlali_restart_checkpoint.nc",
            actual / "citlali_restart_checkpoint.nc",
            allowed,
        )
        result[label] = {
            "map_planes_bitwise_identical": planes,
            **checkpoint,
            "iteration_directory": str(actual.resolve()),
        }
    return result


def validate_checkpoint_policy_interventions(manifest: dict) -> dict:
    registered = manifest["map_checkpoint_policy_interventions"]
    result = {}
    for branch in ("control", "injected"):
        entry = registered[branch]
        audit_path = Path(entry["audit_json"]).resolve()
        audit = json.loads(audit_path.read_text())
        transformation = audit["transformation"]
        if (
            transformation["variable"] != "learning_policy_yaml"
            or transformation["field"]
            != "map_pixel_outlier_detector_exclusion_application"
            or transformation["source_value"] != "pre_cleaning"
            or transformation["output_value"] != "pre_mapmaking"
            or not transformation["all_other_values_verified_equal"]
            or not transformation["all_types_dimensions_and_attributes_verified"]
        ):
            raise ValueError(
                f"invalid registered checkpoint-policy intervention: {branch}"
            )
        source_path = Path(audit["source"]["path"]).resolve()
        output_path = Path(audit["output"]["path"]).resolve()
        if output_path != Path(entry["checkpoint"]).resolve():
            raise ValueError(
                f"checkpoint-policy output path differs: {branch}"
            )
        if sha256(source_path) != audit["source"]["sha256"]:
            raise ValueError(
                f"checkpoint-policy source hash differs: {branch}"
            )
        if sha256(output_path) != audit["output"]["sha256"]:
            raise ValueError(
                f"checkpoint-policy output hash differs: {branch}"
            )
        audit_learning_policy_transformation(
            source_path,
            output_path,
            transformation["field"],
            transformation["source_value"],
            transformation["output_value"],
        )
        result[branch] = {
            "audit_json": str(audit_path),
            "source_sha256": audit["source"]["sha256"],
            "output_sha256": audit["output"]["sha256"],
            "only_registered_policy_field_changed": True,
        }
    return result


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
    for name in COMPONENTS:
        projected, residual = projection(
            components[name], fixed_kernel, support, truth
        )
        residuals[name] = residual
        row: dict[str, float | int | str] = {
            "array": array,
            "component": name,
            "common_valid_pixels": int(np.count_nonzero(support)),
            "complete_map_rms_mjy_beam": rms(components[name][support]),
            "aperture_integrated_response_fraction": aperture_fraction(
                components[name],
                fixed_kernel,
                support,
                regions["injected_source_r20"],
                truth,
            ),
            **projected,
        }
        for region_name in REGIONS:
            selected = support & regions[region_name]
            if not selected.any():
                raise ValueError(f"empty EL-F8 region: {array}:{region_name}")
            row[f"{region_name}_pixels"] = int(np.count_nonzero(selected))
            row[f"{region_name}_rms_mjy_beam"] = rms(
                components[name][selected]
            )
            row[f"{region_name}_residual_rms_mjy_beam"] = rms(
                residual[selected]
            )
        if name in native_kernels:
            row.update(
                native_compact_metrics(
                    components[name],
                    native_kernels[name],
                    float(manifest["pixel_size_arcsec"]),
                    injection_center,
                    float(manifest["gaussian_search_radius_arcsec"]),
                    truth,
                )
            )
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
            rows.append(
                {
                    "array": array,
                    "left_component": left,
                    "right_component": right,
                    "region": region_name,
                    "pixels": int(np.count_nonzero(selected)),
                    "mean_product_mjy2_beam2": inner,
                    "two_mean_product_mjy2_beam2": 2.0 * inner,
                    "cosine": (
                        inner / (left_norm * right_norm)
                        if left_norm > 0.0 and right_norm > 0.0
                        else math.nan
                    ),
                }
            )
    return rows


def read_execution(log_dir: Path) -> list[dict]:
    def value(text: str, label: str) -> float | None:
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
    for label in TRAJECTORIES:
        path = log_dir / f"{label}.log"
        text = path.read_text(encoding="utf-8")
        wall = value(text, "real")
        user = value(text, "user")
        system = value(text, "sys")
        rss = re.search(
            r"^\s*([0-9]+)\s+maximum resident set size$", text, re.M
        )
        errors = sum(
            bool(error_pattern.search(line)) for line in text.splitlines()
        )
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
                "wall_seconds": wall,
                "user_seconds": user,
                "system_seconds": system,
                "maximum_resident_bytes": int(rss.group(1)),
                "error_or_critical_messages": errors,
            }
        )
    return rows


def detector_application_evidence(redu: Path, source_learning: Path) -> dict:
    rows = list(csv.DictReader((redu / "learning_iter_5.csv").open()))
    target = [
        row
        for row in rows
        if row["record_type"] == "detector_penalty_application"
        and row["scan"] == "5"
    ]
    stages = {row["application_stage"]: row for row in target}
    forbidden = {
        "pre_rtc_detector_exclusion",
        "pre_ptc_detector_exclusion",
    } & set(stages)
    map_row = stages.get("pre_mapmaking_detector_exclusion")
    if forbidden or map_row is None:
        raise ValueError("UID 4460 exclusion did not move cleanly to mapmaking")
    proposed = int(map_row["proposed_samples"])
    newly_flagged = int(map_row["newly_flagged_samples"])
    already_flagged = int(map_row["already_flagged_samples"])
    if (
        map_row["candidate_records"] != "1"
        or map_row["matched_records"] != "1"
        or map_row["invalid_records"] != "0"
        or proposed != 305
        or newly_flagged + already_flagged != proposed
        or map_row["source_protected_samples"] != "0"
        or map_row["applied"] != "1"
    ):
        raise ValueError("UID 4460 mapmaking exclusion evidence differs")

    source_rows = list(csv.DictReader(source_learning.open()))
    conflicting_masks = [
        row
        for row in source_rows
        if row["record_type"] == "sample_mask"
        and row["scan"] == "5"
        and row["uid"] == "4460"
    ]
    if conflicting_masks:
        raise ValueError("UID 4460 has a carried scan-6 sample mask")
    return {
        "uid": 4460,
        "scan_zero_based": 5,
        "raw_samples_entering_rtc_without_hard_detector_exclusion": 676,
        "carried_uid4460_sample_masks_in_scan": 0,
        "pre_rtc_or_pre_ptc_detector_exclusion_records": 0,
        "pre_mapmaking_proposed_samples": proposed,
        "pre_mapmaking_newly_flagged_samples": newly_flagged,
        "pre_mapmaking_already_flagged_samples": already_flagged,
        "pre_mapmaking_excluded_samples_after_application": (
            newly_flagged + already_flagged
        ),
        "pre_mapmaking_applied": True,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
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
    with fits.open(source_path, memmap=True) as hdul:
        header = hdul["signal_I"].header.copy()
    for key in ("CHECKSUM", "DATASUM"):
        if key in header:
            del header[key]
    primary = fits.PrimaryHDU()
    primary.header["HIERARCH SCI.TESTID"] = manifest["test_id"]
    primary.header["HIERARCH SCI.OBSNUM"] = obsnum
    primary.header["HIERARCH SCI.ARRAY"] = array
    primary.header["HIERARCH SCI.DIRECT"] = array == "a1400"
    hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [primary]
    for name in COMPONENTS:
        component_header = header.copy()
        component_header["BUNIT"] = "mJy/beam"
        component_header["HIERARCH SCI.COMPONENT"] = name
        hdus.append(
            fits.ImageHDU(
                components[name].astype("float64"),
                header=component_header,
                name=FITS_EXTENSION_NAMES[name],
            )
        )
        residual_header = header.copy()
        residual_header["BUNIT"] = "mJy/beam"
        residual_header["HIERARCH SCI.RESIDUAL_OF"] = name
        hdus.append(
            fits.ImageHDU(
                residuals[name].astype("float64"),
                header=residual_header,
                name=f"R{len(hdus):02d}_{name}"[:8],
            )
        )
    path = output_dir / f"point_{obsnum}_{array}_el_f8_components_r0.2.fits"
    fits.HDUList(hdus).writeto(path, overwrite=True, checksum=False)
    return path


def write_plot(
    path: Path,
    array_components: dict[str, dict[str, np.ndarray]],
    source_paths: dict[str, Path],
    manifest: dict,
) -> None:
    import matplotlib.pyplot as plt

    injection = np.asarray(
        manifest["injection_position_fits_world_arcsec"], dtype=float
    )
    neptune = np.asarray(
        manifest["neptune_position_fits_world_arcsec"], dtype=float
    )
    figure, axes = plt.subplots(
        3, len(COMPONENTS), figsize=(18.0, 10.5), sharex=True, sharey=True
    )
    for row, array in enumerate(ARRAYS):
        x, y = world_axes(source_paths[array], "signal_I")
        x -= injection[0]
        y -= injection[1]
        for column, name in enumerate(COMPONENTS):
            values = array_components[array][name]
            shown = values
            shown_x = x
            shown_y = y
            if x[0] > x[-1]:
                shown = shown[:, ::-1]
                shown_x = x[::-1]
            if y[0] > y[-1]:
                shown = shown[::-1, :]
                shown_y = y[::-1]
            xx, yy = np.meshgrid(shown_x, shown_y)
            selected = np.isfinite(shown) & (np.hypot(xx, yy) <= 120.0)
            limit = float(np.nanpercentile(np.abs(shown[selected]), 99.0))
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
                plt.Circle((0.0, 0.0), 20.0, fill=False, color="black")
            )
            axes[row, column].add_patch(
                plt.Circle(
                    tuple(neptune - injection),
                    20.0,
                    fill=False,
                    color="#c58b00",
                )
            )
            axes[row, column].set_xlim(-120.0, 120.0)
            axes[row, column].set_ylim(-120.0, 120.0)
            axes[row, column].grid(alpha=0.18)
            figure.colorbar(
                image, ax=axes[row, column], fraction=0.046, pad=0.03
            )
            if row == 0:
                axes[row, column].set_title(name)
            if column == 0:
                axes[row, column].set_ylabel(f"{array}\nEL offset (arcsec)")
            if row == 2:
                axes[row, column].set_xlabel("AZ offset (arcsec)")
    figure.suptitle(
        "EL-F8 iteration-5 penalty-placement decomposition\n"
        "coordinates relative to injected source; panel scales independent"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze(manifest: dict) -> tuple[list[dict], list[dict], list[dict], dict, dict]:
    obsnum = int(manifest["obsnum"])
    iteration = int(manifest["iteration"])
    truth = dict(
        zip(
            manifest["array_order"],
            (float(value) for value in manifest["amplitudes_mjy_beam"]),
        )
    )
    compatibility = validate_compatibility(manifest)
    checkpoint_policy_interventions = (
        validate_checkpoint_policy_interventions(manifest)
    )
    redu = reductions(manifest)
    require_placement_pair(redu["C_current"], redu["C_map"], "control")
    require_placement_pair(redu["A_current"], redu["A_map"], "injected")

    metrics = []
    cross_terms = []
    triggers = []
    closure = {}
    array_components = {}
    array_residuals = {}
    source_paths = {}
    for array in ARRAYS:
        paths = {
            label: product_path(path, obsnum, array)
            for label, path in redu.items()
        }
        if any(fruit_iteration(path) != iteration for path in paths.values()):
            raise ValueError(f"EL-F8 product iteration differs for {array}")
        loaded = {}
        grids = {}
        supports = []
        for extension in IMAGE_EXTENSIONS:
            for label, product in paths.items():
                values, grid = load_image(product, extension)
                loaded[f"{label}_{extension}"] = values
                previous = grids.setdefault(extension, grid)
                if grid != previous:
                    raise ValueError(
                        f"EL-F8 WCS/grid differs: {array}:{extension}:{label}"
                    )
            supports.append(
                common_support(
                    {
                        label: loaded[f"{label}_{extension}"]
                        for label in paths
                    },
                    context=f"EL-F8 array={array} extension={extension}",
                )
            )
        support = np.logical_and.reduce(supports)
        components = {
            "T_current": (
                loaded["A_current_signal_I"] - loaded["C_current_signal_I"]
            ),
            "T_map": loaded["A_map_signal_I"] - loaded["C_map_signal_I"],
            "D_current": (
                loaded["A_current_signal_I"] - loaded["N5_signal_I"]
            ),
            "D_map": loaded["A_map_signal_I"] - loaded["N5_signal_I"],
            "Q": loaded["A_current_signal_I"] - loaded["A_map_signal_I"],
        }
        q_alternate = components["D_current"] - components["D_map"]
        q_error = float(np.max(np.abs((components["Q"] - q_alternate)[support])))
        q_bound = roundoff_bound(
            [components["Q"], components["D_current"], components["D_map"]],
            float(manifest["closure_roundoff_factor"]),
        )
        closure[array] = {
            "maximum_absolute_q_identity_residual_mjy_beam": q_error,
            "roundoff_bound_mjy_beam": q_bound,
            "closure_pass": q_error <= q_bound,
        }
        if q_error > q_bound:
            raise ValueError(f"EL-F8 Q identity closure failed for {array}")

        injection_center = gaussian_center_for_map_world_offset(
            paths["A_map"],
            "signal_I",
            *[
                float(value)
                for value in manifest["injection_position_fits_world_arcsec"]
            ],
        )
        neptune_center = gaussian_center_for_map_world_offset(
            paths["A_map"],
            "signal_I",
            *[
                float(value)
                for value in manifest["neptune_position_fits_world_arcsec"]
            ],
        )
        regions = make_region_masks(
            components["Q"].shape,
            float(manifest["pixel_size_arcsec"]),
            injection_center,
            neptune_center,
            manifest,
        )
        component_rows, residuals = component_metrics(
            array,
            components,
            loaded["A_map_kernel_I"],
            {
                "T_current": loaded["A_current_kernel_I"],
                "T_map": loaded["A_map_kernel_I"],
            },
            support,
            regions,
            injection_center,
            manifest,
            truth[array],
        )
        metrics.extend(component_rows)
        cross_terms.extend(
            cross_term_rows(array, components, support, regions)
        )
        if array == "a1400":
            for trigger in manifest["trigger_pixels_a1400"]:
                row = int(trigger["row"])
                col = int(trigger["col"])
                triggers.append(
                    {
                        **trigger,
                        "a5_current_signal_mjy_beam": float(
                            loaded["A_current_signal_I"][row, col]
                        ),
                        "a5_map_signal_mjy_beam": float(
                            loaded["A_map_signal_I"][row, col]
                        ),
                        "q_early_exclusion_mjy_beam": float(
                            components["Q"][row, col]
                        ),
                        "a5_current_weight": float(
                            loaded["A_current_weight_I"][row, col]
                        ),
                        "a5_map_weight": float(
                            loaded["A_map_weight_I"][row, col]
                        ),
                    }
                )
        array_components[array] = components
        array_residuals[array] = residuals
        source_paths[array] = paths["A_map"]

    application = detector_application_evidence(
        redu["A_map"], Path(manifest["injected_iteration_4_learning"])
    )
    execution = read_execution(Path(manifest["execution_log_dir"]))
    result = {
        "test_id": manifest["test_id"],
        "valid_penalty_placement_decomposition": True,
        "compatibility": compatibility,
        "checkpoint_policy_interventions": checkpoint_policy_interventions,
        "paired_realized_configuration_check": "PASS",
        "common_units_wcs_grid_normalization_and_finite_support": "PASS",
        "q_identity_closure": closure,
        "uid4460_application_evidence": application,
        "execution": {
            "trajectory_count": len(execution),
            "aggregate_wall_seconds": sum(
                float(row["wall_seconds"]) for row in execution
            ),
            "maximum_resident_bytes": max(
                int(row["maximum_resident_bytes"]) for row in execution
            ),
        },
        "component_identity": manifest["component_definitions"],
        "direct_uid4460_interpretation_arrays": ["a1400"],
        "mechanism_dominance_threshold": None,
        "mechanism_classification": "pending_continuous_evidence_interpretation",
        "production_policy_selected": False,
    }
    auxiliary = {
        "execution": execution,
        "array_components": array_components,
        "array_residuals": array_residuals,
        "source_paths": source_paths,
    }
    return metrics, cross_terms, triggers, result, auxiliary


def metric_row(rows: list[dict], array: str, component: str) -> dict:
    return next(
        row
        for row in rows
        if row["array"] == array and row["component"] == component
    )


def report_text(metrics: list[dict], result: dict) -> str:
    lines = [
        "# FRUIT EL-F8 penalty-placement result",
        "",
        "Result: **valid bounded decomposition; descriptive development "
        "evidence only**",
        "",
        f"Test ID: `{result['test_id']}`",
        "",
        "## Validity",
        "",
        "- current-placement control and injection maps: bitwise compatible;",
        "- scientific checkpoint values: compatible after the two registered "
        "provenance allowances;",
        "- paired placement-only configuration checks: `PASS`;",
        "- units, WCS/grid, normalization, support, and Q closure: `PASS`; and",
        "- unexpected error/critical messages: zero.",
        "",
        "## Compact response",
        "",
        "| Array | Current central recovery | Map-only central recovery | "
        "Current width major/minor | Map-only width major/minor |",
        "|---|---:|---:|---:|---:|",
    ]
    for array in ARRAYS:
        current = metric_row(metrics, array, "T_current")
        moved = metric_row(metrics, array, "T_map")
        lines.append(
            f"| {array} | "
            f"{float(current['kernel_normalized_central_recovery']):.6f} | "
            f"{float(moved['kernel_normalized_central_recovery']):.6f} | "
            f"{float(current['major_fwhm_over_kernel']):.4f} / "
            f"{float(current['minor_fwhm_over_kernel']):.4f} | "
            f"{float(moved['major_fwhm_over_kernel']):.4f} / "
            f"{float(moved['minor_fwhm_over_kernel']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## a1400 placement components",
            "",
            "RMS values are mJy/beam. No numerical dominance threshold was "
            "registered.",
            "",
            "| Region | D current | D map-only | Q early-placement increment |",
            "|---|---:|---:|---:|",
        ]
    )
    current = metric_row(metrics, "a1400", "D_current")
    moved = metric_row(metrics, "a1400", "D_map")
    early = metric_row(metrics, "a1400", "Q")
    labels = {
        "injected_source_r20": "injected source r<20",
        "neptune_r20": "Neptune r<20",
        "annulus_r40_120_excluding_neptune_r25": (
            "annulus 40-120 excl. Neptune"
        ),
    }
    for region, label in labels.items():
        key = f"{region}_rms_mjy_beam"
        lines.append(
            f"| {label} | {float(current[key]):.6g} | "
            f"{float(moved[key]):.6g} | {float(early[key]):.6g} |"
        )
    lines.extend(
        [
            "",
            "The complete component maps, fixed-kernel residuals, signed cross "
            "terms, trigger-pixel table, application evidence, and execution "
            "resources are retained with this report.",
            "",
            "Direct UID 4460 interpretation is limited to a1400. The a2000 "
            "measurement includes two moved map-diagnostic exclusions; a1100 "
            "retains its busy-detector placement.",
            "",
            "This result does not judge UID 4460, establish a generic mechanism, "
            "select a safeguard or production policy, qualify FRUIT, launch Gate "
            "D, or begin Stage B.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compatibility-only", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    if args.compatibility_only:
        print(json.dumps(validate_compatibility(manifest), indent=2, sort_keys=True))
        return 0

    metrics, cross_terms, triggers, result, auxiliary = analyze(manifest)
    output_dir = Path(manifest["analysis_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "COMPONENT_METRICS_R0.2.csv",
        "cross": output_dir / "CROSS_TERMS_R0.2.csv",
        "triggers": output_dir / "TRIGGER_PIXELS_R0.2.csv",
        "execution": output_dir / "PRIMARY_EXECUTION_R0.2.csv",
        "result": output_dir / "DECOMPOSITION_RESULT_R0.2.json",
        "report": output_dir / "EXECUTION_RESULT_R0.2.md",
        "plot": output_dir / "PENALTY_PLACEMENT_DECOMPOSITION_R0.2.png",
    }
    write_csv(paths["metrics"], metrics)
    write_csv(paths["cross"], cross_terms)
    write_csv(paths["triggers"], triggers)
    write_csv(paths["execution"], auxiliary["execution"])
    paths["result"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    paths["report"].write_text(report_text(metrics, result))
    component_dir = output_dir / "component-maps"
    component_dir.mkdir(parents=True, exist_ok=True)
    component_paths = []
    for array in ARRAYS:
        component_paths.append(
            write_component_maps(
                component_dir,
                int(manifest["obsnum"]),
                array,
                auxiliary["source_paths"][array],
                auxiliary["array_components"][array],
                auxiliary["array_residuals"][array],
                manifest,
            )
        )
    write_plot(
        paths["plot"],
        auxiliary["array_components"],
        auxiliary["source_paths"],
        manifest,
    )
    result_paths = [*paths.values(), *component_paths]
    provenance = {
        "schema_version": "sci-fruit-el-f8-provenance-v1",
        "test_id": manifest["test_id"],
        "role": "exploratory-development-mechanism-test-only",
        "qualification_use_authorized": False,
        "inputs": [
            file_record(path)
            for path in sorted(
                {
                    Path(__file__).resolve(),
                    manifest_path,
                    (manifest_path.parent / manifest["registration"]).resolve(),
                    *(
                        Path(entry["audit_json"]).resolve()
                        for entry in manifest[
                            "map_checkpoint_policy_interventions"
                        ].values()
                    ),
                    *Path(manifest["execution_log_dir"]).glob("*.log"),
                    *(
                        (redu / "citlali_restart_checkpoint.nc")
                        for redu in reductions(manifest).values()
                    ),
                },
                key=str,
            )
        ],
        "outputs": [
            file_record(path, relative_to=output_dir)
            for path in sorted(result_paths, key=str)
        ],
        "execution": auxiliary["execution"],
    }
    provenance_path = output_dir / "ANALYSIS_PROVENANCE_R0.2.yaml"
    provenance_path.write_text(yaml.safe_dump(provenance, sort_keys=False))
    print(paths["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
