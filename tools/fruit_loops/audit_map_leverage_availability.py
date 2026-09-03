#!/usr/bin/env python3
"""Complete the read-only SCI-FRUIT EL-F9 availability/flagging audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from netCDF4 import Dataset


REPOSITORY = Path(__file__).resolve().parents[2]
STAT_FIELDS = ("rms", "stddev", "median", "flagged_frac", "weights")
PTC_FIELDS = (
    "ptc_detector_weight",
    "ptc_detector_rms",
    "ptc_detector_stddev",
    "ptc_detector_median",
    "ptc_detector_flagged_fraction",
    "ptc_invvar_window_median",
    "ptc_invvar_window_flagged_frac_median",
    "ptc_invvar_window_flagged_frac_max",
    "ptc_invvar_window_heavy_flagged_fraction",
)
RTC_FIELDS = (
    "rtc_despike_local_flagged_sample_count",
    "rtc_final_flagged_frac",
    "rtc_invvar_window_median",
    "rtc_detector_notch_n_applied",
)
EXPECTED_PIXEL_PRODUCTS = (
    "jinc_grid_denominator_G",
    "jinc_variance_accumulator_V",
    "jinc_signal_numerator_S",
    "uid4460_grid_denominator_G",
    "uid4460_variance_accumulator_V",
    "uid4460_signal_numerator_S",
    "map_hit_count",
    "map_unique_detector_count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_registered_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def registered_files(registration: dict) -> list[dict]:
    return [
        *registration.get("inputs", []),
        *registration.get("implementation_evidence", []),
        *registration.get("additional_inputs", []),
    ]


def validate_registered_files(registration: dict) -> list[dict]:
    result = []
    for entry in registered_files(registration):
        path = resolve_registered_path(entry["path"])
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != int(entry["size_bytes"]):
            raise ValueError(f"registered size mismatch: {path}")
        if actual_hash != entry["sha256"]:
            raise ValueError(f"registered hash mismatch: {path}")
        result.append(
            {
                "role": entry.get("role", "implementation_evidence"),
                "path": str(path.resolve()),
                "size_bytes": actual_size,
                "sha256": actual_hash,
            }
        )
    return result


def validate_registration_chain(registration_dir: Path, registration: dict) -> dict:
    checks = {
        "superseded_registration": registration["supersedes"],
        "base_input_roster": {
            "path": registration["inherited_registration"]["base_input_roster"],
            "sha256": registration["inherited_registration"][
                "base_input_roster_sha256"
            ],
        },
        "controlling_definition": {
            "path": registration["inherited_registration"][
                "controlling_definition"
            ],
            "sha256": registration["inherited_registration"][
                "controlling_definition_sha256"
            ],
        },
        "controlling_semantics": {
            "path": registration["inherited_registration"][
                "controlling_semantics"
            ],
            "sha256": registration["inherited_registration"][
                "controlling_semantics_sha256"
            ],
        },
    }
    for label, entry in checks.items():
        path = registration_dir / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"registration-chain hash mismatch: {label}")
        if "size_bytes" in entry and path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"registration-chain size mismatch: {label}")
    return {"validated": True, "objects": list(checks)}


def role_paths(registration: dict) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for entry in registered_files(registration):
        if "role" in entry:
            result[entry["role"]] = resolve_registered_path(entry["path"])
    return result


def image(hdul: fits.HDUList, extension: str) -> np.ndarray:
    values = np.asarray(hdul[extension].data, dtype=np.float64).squeeze()
    if values.ndim != 2:
        raise ValueError(f"{extension} is not two-dimensional after squeeze")
    return values


def fits_plane_inventory(path: Path) -> list[str]:
    with fits.open(path, memmap=True) as hdul:
        return [hdu.name for hdu in hdul if hdu.data is not None]


def midpoint_percentile(values: np.ndarray, target: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not np.isfinite(target) or finite.size == 0:
        return float("nan")
    below = np.count_nonzero(finite < target)
    equal = np.count_nonzero(finite == target)
    return float(100.0 * (below + 0.5 * equal) / finite.size)


def jinc_formal_coefficient(grid_weight: float, variance: float) -> float:
    if not np.isfinite(grid_weight) or not np.isfinite(variance) or variance <= 0:
        return float("nan")
    return float(grid_weight * grid_weight / variance)


def jinc_nonadditivity_examples() -> dict:
    without = jinc_formal_coefficient(1.0, 1.0)
    with_negative_lobe = jinc_formal_coefficient(0.9, 1.01)
    with_positive_lobe = jinc_formal_coefficient(1.1, 1.01)
    return {
        "without_uid": without,
        "with_negative_kernel_lobe": with_negative_lobe,
        "all_minus_without_negative": with_negative_lobe - without,
        "with_positive_kernel_lobe": with_positive_lobe,
        "all_minus_without_positive": with_positive_lobe - without,
        "demonstrates_signed_final_coefficient_difference": bool(
            with_negative_lobe - without < 0 < with_positive_lobe - without
        ),
    }


def require_compatible_maps(all_path: Path, without_path: Path) -> dict:
    with fits.open(all_path, memmap=True) as all_hdul, fits.open(
        without_path, memmap=True
    ) as without_hdul:
        all_primary = all_hdul[0].header
        without_primary = without_hdul[0].header
        if all_primary.get("METHOD", "").lower() != "jinc":
            raise ValueError("N5 map is not a JINC product")
        if without_primary.get("METHOD", "").lower() != "jinc":
            raise ValueError("A5-map map is not a JINC product")
        for extension in ("SIGNAL_I", "WEIGHT_I", "WEIGHT_FORMAL_I"):
            left = all_hdul[extension]
            right = without_hdul[extension]
            if left.data.shape != right.data.shape:
                raise ValueError(f"paired map shape differs: {extension}")
            if left.header.get("BUNIT") != right.header.get("BUNIT"):
                raise ValueError(f"paired map unit differs: {extension}")
            for key in (
                "NAXIS1",
                "NAXIS2",
                "CRPIX1",
                "CRPIX2",
                "CRVAL1",
                "CRVAL2",
                "CDELT1",
                "CDELT2",
                "CUNIT1",
                "CUNIT2",
                "CTYPE1",
                "CTYPE2",
            ):
                if left.header.get(key) != right.header.get(key):
                    raise ValueError(f"paired map WCS differs: {extension}:{key}")
        return {
            "method": "jinc",
            "shape": list(image(all_hdul, "SIGNAL_I").shape),
            "signal_unit": all_hdul["SIGNAL_I"].header["BUNIT"],
            "formal_coefficient_unit": all_hdul["WEIGHT_FORMAL_I"].header[
                "BUNIT"
            ],
            "wcs_grid_identical": True,
        }


def analyze_final_coefficients(all_path: Path, without_path: Path) -> tuple[dict, dict]:
    with fits.open(all_path, memmap=True) as all_hdul, fits.open(
        without_path, memmap=True
    ) as without_hdul:
        all_formal = image(all_hdul, "WEIGHT_FORMAL_I")
        without_formal = image(without_hdul, "WEIGHT_FORMAL_I")
        all_empirical = image(all_hdul, "WEIGHT_I")
        without_empirical = image(without_hdul, "WEIGHT_I")
        difference = all_formal - without_formal
        finite = np.isfinite(all_formal) & np.isfinite(without_formal)
        scale = max(
            1.0,
            float(np.max(np.abs(all_formal[finite]))),
            float(np.max(np.abs(without_formal[finite]))),
        )
        bound = float(64.0 * np.finfo(np.float64).eps * scale)
        positive = difference > bound
        negative = difference < -bound
        roundoff = finite & ~(positive | negative)
        all_positive = finite & (all_formal > 0)
        without_positive = finite & (without_formal > 0)
        common_positive = all_positive & without_positive
        all_ratio = all_empirical[all_positive] / all_formal[all_positive]
        without_ratio = (
            without_empirical[without_positive]
            / without_formal[without_positive]
        )
        result = {
            "classification_bound": bound,
            "finite_pixels": int(finite.sum()),
            "n5_positive_formal_pixels": int(all_positive.sum()),
            "a5_map_positive_formal_pixels": int(without_positive.sum()),
            "common_positive_formal_pixels": int(common_positive.sum()),
            "n5_positive_a5_map_zero_support_loss_pixels": int(
                (all_positive & ~without_positive).sum()
            ),
            "a5_map_positive_n5_zero_support_gain_pixels": int(
                (without_positive & ~all_positive).sum()
            ),
            "materially_negative_difference_pixels": int(negative.sum()),
            "materially_positive_difference_pixels": int(positive.sum()),
            "roundoff_or_zero_difference_pixels": int(roundoff.sum()),
            "difference_min": float(np.min(difference[finite])),
            "difference_max": float(np.max(difference[finite])),
            "n5_empirical_to_formal_ratio_median": float(np.median(all_ratio)),
            "a5_map_empirical_to_formal_ratio_median": float(
                np.median(without_ratio)
            ),
            "exact_uid_leverage_available": False,
            "reason": (
                "published JINC weight_formal_I is G^2/V; G and V are not "
                "published and the paired final-coefficient difference is signed"
            ),
        }
        arrays = {
            "all_formal": all_formal,
            "without_formal": without_formal,
            "difference": difference,
        }
        return result, arrays


def accepted_a1400_context(stats: Dataset, uid: int) -> tuple[int, np.ndarray]:
    uids = np.asarray(stats.variables["apt_uid"][:], dtype=np.int64)
    found = np.flatnonzero(uids == uid)
    if found.size != 1:
        raise ValueError(f"UID {uid} is not unique in detector statistics")
    arrays = np.asarray(stats.variables["apt_array"][:], dtype=np.int64)
    apt_flags = np.asarray(stats.variables["apt_flag"][:], dtype=np.float64)
    accepted = (arrays == 1) & (apt_flags == 0)
    return int(found[0]), accepted


def compare_scan_statistics(
    all_path: Path, without_path: Path, scan: int, accepted: np.ndarray
) -> dict:
    differing: dict[str, int] = {}
    with Dataset(all_path) as all_stats, Dataset(without_path) as without_stats:
        for field in STAT_FIELDS:
            left = np.asarray(all_stats.variables[field][scan, accepted])
            right = np.asarray(without_stats.variables[field][scan, accepted])
            differing[field] = int(np.count_nonzero(~np.equal(left, right)))
    return {
        "scan_zero_based": scan,
        "accepted_a1400_detector_count": int(accepted.sum()),
        "differing_values_by_field": differing,
        "all_compared_values_identical": all(value == 0 for value in differing.values()),
    }


def detector_rows(
    stats_path: Path,
    ptc_path: Path,
    rtc_path: Path,
    apt_path: Path,
    uid: int,
    scan: int,
) -> tuple[list[dict], dict, np.ndarray]:
    rows: list[dict] = []
    with Dataset(stats_path) as stats:
        detector_index, accepted = accepted_a1400_context(stats, uid)
        for field in STAT_FIELDS:
            values = np.asarray(stats.variables[field][scan, :], dtype=np.float64)
            target = float(values[detector_index])
            rows.append(
                {
                    "stage": "final_statistics",
                    "metric": field,
                    "value": target,
                    "percentile_from_low_end": midpoint_percentile(
                        values[accepted], target
                    ),
                    "cohort_n_finite": int(
                        np.isfinite(values[accepted]).sum()
                    ),
                }
            )
    with Dataset(ptc_path) as ptc:
        for field in PTC_FIELDS:
            values = np.asarray(ptc.variables[field][scan, :], dtype=np.float64)
            target = float(values[detector_index])
            rows.append(
                {
                    "stage": "ptc",
                    "metric": field,
                    "value": target,
                    "percentile_from_low_end": midpoint_percentile(
                        values[accepted], target
                    ),
                    "cohort_n_finite": int(
                        np.isfinite(values[accepted]).sum()
                    ),
                }
            )
    with Dataset(rtc_path) as rtc:
        for field in RTC_FIELDS:
            values = np.asarray(rtc.variables[field][scan, :], dtype=np.float64)
            target = float(values[detector_index])
            rows.append(
                {
                    "stage": "rtc",
                    "metric": field,
                    "value": target,
                    "percentile_from_low_end": midpoint_percentile(
                        values[accepted], target
                    ),
                    "cohort_n_finite": int(
                        np.isfinite(values[accepted]).sum()
                    ),
                }
            )

    apt = Table.read(apt_path)
    matches = apt[np.asarray(apt["uid"], dtype=np.int64) == uid]
    if len(matches) != 1:
        raise ValueError(f"UID {uid} is not unique in APT")
    source = matches[0]
    identity = {
        "detector_index": detector_index,
        "uid": uid,
        "array_id": int(source["array"]),
        "array_name": "a1400",
        "network": int(source["nw"]),
        "kids_flag": int(source["kids_flag"]),
        "apt_flag": int(source["flag"]),
        "apt_flag2": int(source["flag2"]),
        "a_fwhm_arcsec": float(source["a_fwhm"]),
        "b_fwhm_arcsec": float(source["b_fwhm"]),
        "apt_signal_to_noise": float(source["sig2noise"]),
        "x_t_arcsec": float(source["x_t"]),
        "y_t_arcsec": float(source["y_t"]),
        "accepted_a1400_cohort_n": int(accepted.sum()),
    }
    return rows, identity, accepted


def read_learning_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def learning_evidence(
    source_path: Path, application_path: Path, uid: int, scan: int
) -> tuple[dict, list[dict]]:
    source = read_learning_rows(source_path)
    applications = read_learning_rows(application_path)
    penalty = [
        row
        for row in source
        if row["record_type"] == "detector_penalty"
        and int(row["iter"]) == 4
        and int(row["uid"]) == uid
        and int(row["scan"]) == scan
        and row["reason"] == "map_pixel_outlier_detector_dominance"
    ]
    triggers = [
        row
        for row in source
        if row["record_type"] == "map_pixel_outlier"
        and int(row["iter"]) == 4
        and int(row["uid"]) == uid
        and int(row["scan"]) == scan
    ]
    application = [
        row
        for row in applications
        if row["record_type"] == "detector_penalty_application"
        and int(row["iter"]) == 5
        and int(row["scan"]) == scan
        and row["application_stage"] == "pre_mapmaking_detector_exclusion"
    ]
    if len(penalty) != 1 or len(triggers) != 4 or len(application) != 1:
        raise ValueError("registered UID 4460 learning evidence is incomplete")
    p = penalty[0]
    a = application[0]
    summary = {
        "penalty_iter": int(p["iter"]),
        "penalty_factor": float(p["factor"]),
        "penalty_score": float(p["score"]),
        "penalty_scan_local": bool(int(p["scan_local"])),
        "trigger_pixel_count": len(triggers),
        "application_iter": int(a["iter"]),
        "application_stage": a["application_stage"],
        "proposed_samples": int(a["proposed_samples"]),
        "newly_flagged_samples": int(a["newly_flagged_samples"]),
        "already_flagged_samples": int(a["already_flagged_samples"]),
        "source_protected_samples": int(a["source_protected_samples"]),
        "applied": bool(int(a["applied"])),
    }
    parsed_triggers = [
        {
            "row": int(row["row"]),
            "col": int(row["col"]),
            "sample": int(row["sample"]),
            "map_value_mjy_beam": float(row["value"]),
            "n_eff": float(row["n_eff"]),
            "leave_one_out_z": float(row["leave_one_out_z"]),
            "source_distance_arcsec": float(row["source_distance_arcsec"]),
        }
        for row in triggers
    ]
    return summary, parsed_triggers


def historical_uid_sample_masks(path: Path, uid: int) -> dict:
    rows = [
        row
        for row in read_learning_rows(path)
        if row["record_type"] == "sample_mask" and int(row["uid"]) == uid
    ]
    unique = {
        (int(row["scan"]), int(row["raw_start"]), int(row["raw_stop"]))
        for row in rows
    }
    return {
        "unique_mask_count": len(unique),
        "scan_zero_based_values": sorted({entry[0] for entry in unique}),
        "scan_5_mask_count": sum(entry[0] == 5 for entry in unique),
        "records_repeat_across_iterations": len(rows) > len(unique),
    }


def trigger_response_rows(
    triggers: list[dict],
    all_map_path: Path,
    without_map_path: Path,
    component_path: Path,
) -> list[dict]:
    with fits.open(all_map_path, memmap=True) as all_hdul, fits.open(
        without_map_path, memmap=True
    ) as without_hdul, fits.open(component_path, memmap=True) as components:
        all_signal = image(all_hdul, "SIGNAL_I")
        without_signal = image(without_hdul, "SIGNAL_I")
        all_formal = image(all_hdul, "WEIGHT_FORMAL_I")
        without_formal = image(without_hdul, "WEIGHT_FORMAL_I")
        direct = image(components, "D_MAP")
        rows = []
        for trigger in triggers:
            row = int(trigger["row"])
            col = int(trigger["col"])
            rows.append(
                {
                    **trigger,
                    "n5_signal_mjy_beam": float(all_signal[row, col]),
                    "a5_map_signal_mjy_beam": float(without_signal[row, col]),
                    "direct_map_effect_mjy_beam": float(direct[row, col]),
                    "n5_formal_coefficient": float(all_formal[row, col]),
                    "a5_map_formal_coefficient": float(without_formal[row, col]),
                    "formal_coefficient_difference_not_leverage": float(
                        all_formal[row, col] - without_formal[row, col]
                    ),
                }
            )
    return rows


def inventory_retained_products(paths: dict[str, Path]) -> dict:
    fits_planes = {
        role: fits_plane_inventory(path)
        for role, path in paths.items()
        if role in {"n5_a1400_map", "a5_map_a1400_map"}
    }
    with Dataset(paths["a5_map_map_diagnostics"]) as mapdiag:
        mapdiag_variables = sorted(mapdiag.variables)
    result = {
        "fits_planes": fits_planes,
        "mapdiag_variables": mapdiag_variables,
        "exact_pixel_products_requested": list(EXPECTED_PIXEL_PRODUCTS),
        "exact_pixel_products_found": [],
        "exact_pixel_products_unavailable": list(EXPECTED_PIXEL_PRODUCTS),
        "mapdiag_is_aggregate_not_pixel_resolved": True,
        "processed_sample_position_value_flag_cube_retained": False,
        "contribution_diagnostics_persisted_completely": False,
    }
    return result


def scalar_netcdf_value(dataset: Dataset, name: str) -> object:
    value = np.asarray(dataset.variables[name][:]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def learning_configuration(path: Path) -> dict:
    names = {
        "map_pixel_outlier_detector_exclusion_enabled": (
            "CONFIG.LEARNING.MAP_PIXEL_OUTLIER_DETECTOR_EXCLUSION_ENABLED"
        ),
        "map_pixel_outlier_detector_exclusion_min_pixels": (
            "CONFIG.LEARNING.MAP_PIXEL_OUTLIER_DETECTOR_EXCLUSION_MIN_PIXELS"
        ),
        "map_pixel_outlier_detector_exclusion_application": (
            "CONFIG.LEARNING.MAP_PIXEL_OUTLIER_DETECTOR_EXCLUSION_APPLICATION"
        ),
    }
    with Dataset(path) as dataset:
        missing = [source for source in names.values() if source not in dataset.variables]
        if missing:
            raise ValueError(f"PTC diagnostic lacks learning config: {missing}")
        return {
            key: scalar_netcdf_value(dataset, source)
            for key, source in names.items()
        } | {
            "other_map_pixel_selection_fields_persisted_here": False,
        }


def world_axes(header: fits.Header, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    x = (
        np.arange(shape[1], dtype=float) + 1.0 - float(header["CRPIX1"])
    ) * float(header["CDELT1"]) + float(header["CRVAL1"])
    y = (
        np.arange(shape[0], dtype=float) + 1.0 - float(header["CRPIX2"])
    ) * float(header["CDELT2"]) + float(header["CRVAL2"])
    return x, y


def symmetric_limit(values: np.ndarray, percentile: float = 99.5) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return float(np.percentile(finite, percentile)) if finite.size else 1.0


def make_plot(
    output: Path,
    component_path: Path,
    coefficient_arrays: dict,
    triggers: list[dict],
    injection_position: tuple[float, float],
) -> None:
    with fits.open(component_path, memmap=True) as components:
        direct = image(components, "D_MAP")
        header = components["D_MAP"].header
    x, y = world_axes(header, direct.shape)
    x = x - injection_position[0]
    y = y - injection_position[1]
    extent = [x[0], x[-1], y[0], y[-1]]
    coefficient_difference = coefficient_arrays["difference"]
    coefficient_fraction = np.full_like(coefficient_difference, np.nan)
    positive = coefficient_arrays["all_formal"] > 0
    coefficient_fraction[positive] = (
        coefficient_difference[positive]
        / coefficient_arrays["all_formal"][positive]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), constrained_layout=True)
    panels = (
        (
            direct,
            "Direct map effect: A5-map − N5",
            "mJy/beam",
            symmetric_limit(direct),
        ),
        (
            coefficient_difference,
            "Signed formal-coefficient difference\n(N5 − A5-map; not leverage)",
            r"1/(mJy/beam)$^2$",
            symmetric_limit(coefficient_difference),
        ),
        (
            coefficient_fraction,
            "Signed coefficient difference / N5\n(not fractional UID leverage)",
            "dimensionless",
            symmetric_limit(coefficient_fraction),
        ),
    )
    for axis, (values, title, label, limit) in zip(axes, panels):
        shown = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        trigger_x = [x[item["col"]] for item in triggers]
        trigger_y = [y[item["row"]] for item in triggers]
        axis.scatter(
            trigger_x,
            trigger_y,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            s=45,
            label="iteration-4 trigger pixels",
        )
        axis.axhline(0.0, color="0.45", linewidth=0.5)
        axis.axvline(0.0, color="0.45", linewidth=0.5)
        axis.set_xlim(-120.0, 120.0)
        axis.set_ylim(-120.0, 120.0)
        axis.set_title(title)
        axis.set_xlabel("Az offset from injected source (arcsec)")
        axis.set_aspect("equal")
        colorbar = fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_label(label)
    axes[0].set_ylabel("El offset from injected source (arcsec)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("EL-F9 a1400 map response and published JINC coefficient diagnostic")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def normalized_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): normalized_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized_json(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def run(registration_dir: Path, output_dir: Path) -> dict:
    registration_r01 = yaml.safe_load(
        (registration_dir / "REGISTRATION_R0.1.yaml").read_text()
    )
    registration_r03 = yaml.safe_load(
        (registration_dir / "REGISTRATION_R0.3.yaml").read_text()
    )
    validated = [
        *validate_registered_files(registration_r01),
        *validate_registered_files(registration_r03),
    ]
    registration_chain = validate_registration_chain(
        registration_dir, registration_r03
    )
    paths = {
        **role_paths(registration_r01),
        **role_paths(registration_r03),
    }
    fixed = registration_r01["fixed_entities"]
    uid = int(fixed["apt_uid"])
    scan = int(fixed["scan_zero_based"])

    compatibility = require_compatible_maps(
        paths["n5_a1400_map"], paths["a5_map_a1400_map"]
    )
    coefficient_result, coefficient_arrays = analyze_final_coefficients(
        paths["n5_a1400_map"], paths["a5_map_a1400_map"]
    )
    detector_metrics, detector_identity, accepted = detector_rows(
        paths["a5_map_detector_statistics"],
        paths["a5_map_ptc_diagnostics"],
        paths["a5_map_rtc_diagnostics"],
        paths["matched_apt"],
        uid,
        scan,
    )
    paired_stats = compare_scan_statistics(
        paths["n5_detector_statistics"],
        paths["a5_map_detector_statistics"],
        scan,
        accepted,
    )
    learning, triggers = learning_evidence(
        paths["injected_iteration_4_learning_evidence"],
        paths["a5_map_iteration_5_learning_evidence"],
        uid,
        scan,
    )
    sample_masks = historical_uid_sample_masks(
        paths["injected_iteration_4_learning_evidence"], uid
    )
    trigger_rows = trigger_response_rows(
        triggers,
        paths["n5_a1400_map"],
        paths["a5_map_a1400_map"],
        paths["el_f8_a1400_component_maps"],
    )
    inventory = inventory_retained_products(paths)
    realized_learning_configuration = learning_configuration(
        paths["a5_map_ptc_diagnostics"]
    )
    direct_at_triggers = np.asarray(
        [row["direct_map_effect_mjy_beam"] for row in trigger_rows]
    )
    trigger_response = {
        "all_four_direct_map_effects_exactly_zero": bool(
            np.array_equal(direct_at_triggers, np.zeros(4))
        ),
        "interpretation": (
            "the four trigger pixels select the scan-local record, but the "
            "next-iteration direct response is elsewhere after all proposed "
            "UID 4460 mapmaking samples are excluded"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "DETECTOR_METRICS_R0.1.csv", detector_metrics)
    write_csv(output_dir / "TRIGGER_RESPONSE_R0.1.csv", trigger_rows)
    make_plot(
        output_dir / "MAP_RESPONSE_AND_COEFFICIENT_DIAGNOSTIC_R0.1.png",
        paths["el_f8_a1400_component_maps"],
        coefficient_arrays,
        triggers,
        tuple(float(item) for item in fixed["injected_source_offset_arcsec"]),
    )

    result = {
        "schema_version": "sci-fruit-el-f9-result-v1",
        "test_id": registration_r01["test_id"],
        "result_revision": "r0.1",
        "scope": "read-only retained-product audit",
        "registered_inputs_validated": True,
        "registration_chain": registration_chain,
        "validated_file_count": len(validated),
        "map_compatibility": compatibility,
        "jinc_nonadditivity_demonstration": jinc_nonadditivity_examples(),
        "formal_coefficient_diagnostic": coefficient_result,
        "detector_identity": detector_identity,
        "paired_scan_5_a1400_statistics": paired_stats,
        "learning_evidence": learning,
        "realized_learning_configuration": realized_learning_configuration,
        "historical_uid_sample_masks": sample_masks,
        "trigger_response": trigger_response,
        "retained_product_inventory": inventory,
        "availability_disposition": {
            "exact_fractional_map_leverage": "unavailable",
            "exact_withheld_contribution_map": "unavailable",
            "exact_local_hit_and_unique_detector_counts": "unavailable",
            "flagging_and_weighting_trace": "completed",
            "new_reduction_run": False,
            "proxy_leverage_substituted": False,
        },
        "claim_limit": (
            "single registered observation/array/detector/scan and exact "
            "iteration-4-to-5 pair; no generic detector or policy conclusion"
        ),
    }
    result = normalized_json(result)
    (output_dir / "AVAILABILITY_RESULT_R0.1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    provenance = {
        "schema_version": "sci-fruit-el-f9-analysis-provenance-v1",
        "test_id": registration_r01["test_id"],
        "result_revision": "r0.1",
        "registration_files": [
            {
                "path": str((registration_dir / name).resolve()),
                "sha256": sha256(registration_dir / name),
            }
            for name in ("REGISTRATION_R0.1.yaml", "REGISTRATION_R0.3.yaml")
        ],
        "registered_files": validated,
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "external_inputs_modified": False,
        "citlali_run": False,
        "unity_activity": False,
    }
    (output_dir / "ANALYSIS_PROVENANCE_R0.1.yaml").write_text(
        yaml.safe_dump(provenance, sort_keys=False)
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registration_dir = args.registration_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else registration_dir
    )
    result = run(registration_dir, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
