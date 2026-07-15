#!/usr/bin/env python3
"""Audit a completed Citlali reduction directory without reading large arrays.

This is a fast preflight tool for validation runs.  It answers questions like
"did this run actually write under the refactor path?", "did it finish?", and
"which coarse stages consumed time?" using only the low-level config, logs, and
file inventory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - depends on validation environment
    yaml = None  # type: ignore[assignment]


REDU_RE = re.compile(r"^redu(\d+)$")
TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")
VALIDATION_PATH_RE = re.compile(r"/2026-refactor/(?P<mode>[^/]+)/(?P<label>[^/]+)/reduced/?")
PRODUCT_SUFFIXES = {".fits", ".fit", ".nc", ".nc4", ".cdf", ".csv", ".ecsv"}
PROFILE_SIDECAR_NAMES = {"citlali_profile.ecsv"}
PROVENANCE_SIDECARS = {
    "kids_external": {
        "filename": "kids_external_provenance.yaml",
        "schema_version": "citlali-kids-external-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("authority",),
            ("config_schema",),
            ("data_schema",),
            ("dependency", "name"),
            ("dependency", "version"),
            ("supported_tod_types",),
            ("selected_tod_type",),
            ("requested", "values"),
            ("requested", "solver_extra_output_present"),
            ("effective", "values"),
            ("effective", "resolution"),
        ),
        "allow_multiple": False,
    },
    "config_source_manifest": {
        "filename": "config_source_manifest.yaml",
        "schema_version": "citlali-config-source-manifest-v1",
        "required_paths": (
            ("merge_authority",),
            ("merge_semantics",),
            ("upstream", "authority"),
            ("upstream", "ordered_sources_provided"),
            ("sources",),
            ("merged", "snapshot_filename"),
            ("merged", "serialization"),
            ("merged", "size_bytes"),
            ("merged", "sha256"),
        ),
        "allow_multiple": False,
    },
    "runtime": {
        "filename": "runtime_provenance.yaml",
        "schema_version": "citlali-runtime-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("requested",),
            ("effective",),
            ("realized",),
        ),
        "allow_multiple": False,
    },
    "mapmaking": {
        "filename": "mapmaking_provenance.yaml",
        "schema_version": "citlali-mapmaking-provenance-v2",
        "accepted_schema_versions": (
            "citlali-mapmaking-provenance-v1",
            "citlali-mapmaking-provenance-v2",
        ),
        "required_paths": (
            ("initialized",),
            ("requested",),
            ("effective", "config"),
            ("effective", "resolution"),
            ("observations",),
            ("coadd",),
            ("realized", "reduction_completed"),
            ("realized", "mapmaking_executed"),
        ),
        "required_paths_by_schema": {
            "citlali-mapmaking-provenance-v1": (
                ("initialized",),
                ("requested",),
                ("effective", "config"),
                ("effective", "resolution"),
                ("observation",),
                ("realized", "reduction_completed"),
                ("realized", "mapmaking_executed"),
            ),
        },
        "allow_multiple": False,
    },
    "coadd": {
        "filename": "coadd_provenance.yaml",
        "schema_version": "citlali-coadd-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("requested", "enabled"),
            ("effective", "config", "enabled"),
            ("effective", "resolution"),
            ("realized", "reduction_completed"),
            ("realized", "coadd_executed"),
            ("realized", "map_count"),
            ("realized", "required_map_write_count"),
            ("realized", "outputs_completed"),
        ),
        "allow_multiple": False,
    },
    "noise_products": {
        "filename": "noise_products_provenance.yaml",
        "schema_version": "citlali-noise-products-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("requested", "enabled"),
            ("requested", "n_noise_maps"),
            ("requested", "randomize_dets"),
            ("requested", "write_realizations"),
            ("requested", "products", "enabled"),
            ("requested", "products", "apply_empirical_weights"),
            ("effective", "config"),
            ("effective", "resolution"),
            ("realized", "reduction_completed"),
            ("realized", "generation_executed"),
            ("realized", "outputs_completed"),
        ),
        "allow_multiple": False,
    },
    "pointing": {
        "filename": "pointing_provenance.yaml",
        "schema_version": "citlali-pointing-provenance-v2",
        "accepted_schema_versions": (
            "citlali-pointing-provenance-v1",
            "citlali-pointing-provenance-v2",
        ),
        "required_paths": (
            ("initialized",),
            ("requested",),
            ("effective", "config"),
            ("effective", "resolution"),
            ("observations",),
            ("realized", "reduction_completed"),
            ("realized", "pointing_executed"),
            ("realized", "completed_observation_count"),
            ("realized", "scientific_map_count"),
            ("realized", "raw_fit_attempt_count"),
            ("realized", "raw_valid_fit_count"),
            ("realized", "filtered_fit_attempt_count"),
            ("realized", "filtered_valid_fit_count"),
            ("realized", "outputs_completed"),
        ),
        "required_paths_by_schema": {
            "citlali-pointing-provenance-v1": (
                ("initialized",),
                ("requested",),
                ("effective", "config"),
                ("effective", "resolution"),
                ("observations",),
                ("realized", "reduction_completed"),
                ("realized", "pointing_executed"),
                ("realized", "completed_observation_count"),
                ("realized", "scientific_map_count"),
                ("realized", "fit_attempt_count"),
                ("realized", "valid_fit_count"),
                ("realized", "outputs_completed"),
            ),
        },
        "allow_multiple": False,
    },
    "beammap": {
        "filename": "beammap_provenance.yaml",
        "schema_version": "citlali-beammap-provenance-v2",
        "accepted_schema_versions": (
            "citlali-beammap-provenance-v1",
            "citlali-beammap-provenance-v2",
        ),
        "required_paths": (
            ("initialized",),
            ("requested",),
            ("effective", "config"),
            ("effective", "resolution"),
            ("observations",),
            ("realized", "reduction_completed"),
            ("realized", "beammap_executed"),
            ("realized", "completed_observation_count"),
            ("realized", "completed_iteration_count"),
            ("realized", "outputs_completed"),
        ),
        "allow_multiple": False,
    },
    "post_processing": {
        "filename": "post_processing_provenance.yaml",
        "schema_version": "citlali-post-processing-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("requested",),
            ("effective", "values"),
            ("effective", "resolution"),
            ("realized", "reduction_completed"),
            ("realized", "observation"),
            ("realized", "coadd"),
            ("realized", "pointing_fits", "raw"),
            ("realized", "pointing_fits", "filtered"),
            ("realized", "beammap_fits"),
            ("realized", "outputs_completed"),
        ),
        "allow_multiple": False,
    },
    "timestream_output": {
        "filename": "timestream_output_provenance.yaml",
        "schema_version": "citlali-timestream-output-provenance-v1",
        "required_paths": (
            ("requested",),
            ("effective",),
            ("realized",),
        ),
        "allow_multiple": True,
    },
    "processed_timestream": {
        "filename": "processed_timestream_provenance.yaml",
        "schema_version": "citlali-processed-timestream-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("requested",),
            ("effective", "config"),
            ("effective", "resolutions"),
            ("realized",),
        ),
        "allow_multiple": False,
    },
    "raw_timestream": {
        "filename": "raw_timestream_provenance.yaml",
        "schema_version": "citlali-raw-timestream-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("requested",),
            ("effective", "config"),
            ("effective", "resolutions"),
            ("observation",),
            ("realized", "execution_completed"),
            ("realized", "completed_scan_count"),
            ("realized", "flagged_sample_count"),
            ("realized", "dynamic_notch_count"),
            ("realized", "required_timestream_write_count"),
        ),
        "allow_multiple": True,
    },
}
LOG_MARKERS = (
    ("start", "reduction-local compressed log"),
    ("version", "citlali version:"),
    ("setup", "pipeline setup"),
    ("running_pipeline", "running pipeline"),
    ("first_mapmaking_start", "starting mapmaking"),
    ("first_mapmaking_run", "running mapmaking"),
    ("max_iteration", "max iteration reached"),
    ("ptcdiag_start", "writing ptc diagnostics sidecar chunks"),
    ("apt_start", "writing apt table"),
    ("apt_done", "done writing apt table"),
    ("fitqc_start", "writing beammap fit qc table"),
    ("fitqc_done", "done writing beammap fit qc table"),
    ("split_flag0_done", "beammap split maps (flag=0) have been written"),
    ("split_flag1_done", "beammap split maps (flag=1) have been written"),
    ("index_start", "making index files"),
    ("done", "citlali is done"),
)
INTERVALS = (
    ("total_log", "start", "done"),
    ("startup_to_setup", "start", "setup"),
    ("pipeline_to_first_mapmaking", "running_pipeline", "first_mapmaking_start"),
    ("mapmaking_to_max_iteration", "first_mapmaking_start", "max_iteration"),
    ("ptcdiag_to_apt_start", "ptcdiag_start", "apt_start"),
    ("apt_write", "apt_start", "apt_done"),
    ("fitqc_write", "fitqc_start", "fitqc_done"),
    ("fitqc_to_split_flag0_done", "fitqc_done", "split_flag0_done"),
    ("split_flag1_write", "split_flag0_done", "split_flag1_done"),
    ("index_to_done", "index_start", "done"),
)


@dataclass(frozen=True)
class TimedLine:
    timestamp: datetime
    line: str


def redu_number(path: Path) -> int | None:
    match = REDU_RE.match(path.name)
    return int(match.group(1)) if match else None


def find_latest_redu(root: Path) -> Path:
    candidates = [child for child in root.iterdir() if child.is_dir() and redu_number(child) is not None]
    if not candidates:
        raise FileNotFoundError(f"no reduNN directories under {root}")
    return max(candidates, key=lambda path: redu_number(path) or -1)


def resolve_redu_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    if redu_number(path) is not None:
        return path
    return find_latest_redu(path)


def open_text(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield from handle


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")


def selected_log_line(line: str) -> bool:
    lowered = line.lower()
    return any(text in line for _, text in LOG_MARKERS) or any(
        token in lowered for token in ("fatal", "critical", "error", "traceback")
    )


def collect_labels_from_text(text: str) -> list[dict[str, str]]:
    result = []
    for match in VALIDATION_PATH_RE.finditer(text):
        result.append({"mode": match.group("mode"), "label": match.group("label"), "path": match.group(0)})
    return result


def load_yaml(path: Path) -> Any:
    if yaml is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_nested_key(value: Any, key: str) -> list[Any]:
    if isinstance(value, dict):
        found = []
        for child_key, child_value in value.items():
            if child_key == key:
                found.append(child_value)
            found.extend(find_nested_key(child_value, key))
        return found
    if isinstance(value, list):
        found = []
        for child in value:
            found.extend(find_nested_key(child, key))
        return found
    return []


def has_nested_path(value: Any, path: tuple[str, ...]) -> bool:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def nested_value(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def processed_provenance_semantic_errors(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_records = {
        "effective.cleaner_mode":
            ("effective", "resolutions", "cleaner_mode"),
        "effective.weighting_source_mask":
            ("effective", "resolutions", "weighting_source_mask"),
        "effective.weighting_dependencies":
            ("effective", "resolutions", "weighting_dependencies"),
        "effective.fruit_loop_iterations":
            ("effective", "resolutions", "fruit_loop_iterations"),
        "effective.fruit_loop_interpolation":
            ("effective", "resolutions", "fruit_loop_interpolation"),
        "realized.source_protection":
            ("realized", "source_protection"),
        "realized.fruit_loop_iterations_completed":
            ("realized", "fruit_loop_iterations_completed"),
        "realized.fruit_loops_converged":
            ("realized", "fruit_loops_converged"),
    }
    try:
        for label, path in required_records.items():
            if nested_value(data, path).get("available") is not True:
                errors.append(f"{label} is unavailable")
        if errors:
            return errors

        requested = nested_value(data, ("requested",))
        effective = nested_value(data, ("effective", "config"))
        resolutions = nested_value(data, ("effective", "resolutions"))
        realized = nested_value(data, ("realized",))

        cleaner = resolutions["cleaner_mode"]["value"]
        if effective["processed_time_chunk"]["clean"]["active"] != cleaner["effective"]:
            errors.append("cleaner resolution does not match effective clean.active")

        source_mask = resolutions["weighting_source_mask"]["value"]
        requested_weighting = requested["processed_time_chunk"]["weighting"]
        effective_weighting = effective["processed_time_chunk"]["weighting"]
        if effective_weighting["source_mask_radius_arcsec"] != source_mask["effective"]:
            errors.append("source-mask resolution does not match effective weighting")
        if source_mask["requested_present"]:
            if requested_weighting["source_mask_radius_arcsec"] != source_mask.get("requested"):
                errors.append("source-mask resolution does not match requested weighting")
        elif not source_mask["inherited_from_cleaning"]:
            errors.append("absent source mask is not marked as inherited")

        weighting = resolutions["weighting_dependencies"]["value"]
        requested_validation = requested_weighting["validation"]["enabled"]
        effective_validation = effective_weighting["validation"]["enabled"]
        expected_validation = bool(
            requested_validation
            or weighting["validation_forced_by_weighting_type"]
        )
        if effective_validation != expected_validation:
            errors.append("weight-validation resolution does not match effective config")
        requested_busy = requested_weighting["busy_row_suppression"]["enabled"]
        effective_busy = effective_weighting["busy_row_suppression"]["enabled"]
        expected_busy = bool(
            requested_busy
            and not weighting["busy_row_disabled_without_second_pass"]
        )
        if effective_busy != expected_busy:
            errors.append("busy-row resolution does not match effective config")

        fruit = resolutions["fruit_loop_iterations"]["value"]
        effective_fruit = effective["fruit_loops"]
        if effective_fruit["max_iters"] != fruit["effective_max_iters"]:
            errors.append("iteration resolution does not match effective max_iters")
        if effective_fruit["save_all_iters"] != fruit["effective_save_all_iters"]:
            errors.append("iteration resolution does not match effective save_all_iters")
        if fruit["forced_single_iteration_while_disabled"] != (
            not requested["fruit_loops"]["enabled"]
        ):
            errors.append("disabled fruit-loop iteration decision is inconsistent")

        source = realized["source_protection"]["value"]
        requested_second_pass = requested["processed_time_chunk"]["flagging"]["second_pass_local"]
        effective_second_pass = effective["processed_time_chunk"]["flagging"]["second_pass_local"]
        expected_activation_request = bool(
            requested_second_pass["enabled"]
            and requested_second_pass["source_protection"]["enabled"]
        )
        if source["processed_activation_requested"] != expected_activation_request:
            errors.append("source-protection request record is inconsistent")
        expected_active = bool(
            expected_activation_request and source["source_aware_reduction"]
        )
        if source["processed_active"] != expected_active:
            errors.append("source-protection realization is inconsistent")
        if effective_second_pass["source_protection"]["active"] != source["processed_active"]:
            errors.append("source-protection realization does not match effective config")

        completed = realized["fruit_loop_iterations_completed"]["value"]
        if not isinstance(completed, int) or completed < 1:
            errors.append("completed iteration count must be a positive integer")
        elif completed > effective_fruit["max_iters"]:
            errors.append("completed iteration count exceeds effective max_iters")
        if not isinstance(realized["fruit_loops_converged"]["value"], bool):
            errors.append("fruit-loop convergence realization must be boolean")
    except (KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate processed provenance semantics: {exc}")
    return errors


def raw_provenance_semantic_errors(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("raw execution plan is not initialized")

        observation = data["observation"]
        if observation.get("available") is not True:
            errors.append("resolved observation state is unavailable")
        elif not isinstance(observation.get("value"), dict):
            errors.append("resolved observation state value is not a mapping")
        else:
            value = observation["value"]
            resolved: dict[str, Any] = {}
            for name in (
                "native_sample_rate_hz",
                "effective_sample_rate_hz",
                "downsample_factor",
                "filter_edge_guard_samples",
                "filter_outer_context_samples",
                "source_protection_active",
                "extinction_active",
                "extinction_model",
            ):
                record = value.get(name)
                if (
                    not isinstance(record, dict)
                    or record.get("available") is not True
                ):
                    errors.append(f"observation {name} is unavailable")
                else:
                    resolved[name] = record.get("value")

            for name in ("native_sample_rate_hz", "effective_sample_rate_hz"):
                if name not in resolved:
                    continue
                number = resolved.get(name)
                if (
                    isinstance(number, bool)
                    or not isinstance(number, (int, float))
                    or not math.isfinite(float(number))
                    or number <= 0
                ):
                    errors.append(f"observation {name} must be finite and positive")
            for name, minimum in (
                ("downsample_factor", 1),
                ("filter_edge_guard_samples", 0),
                ("filter_outer_context_samples", 0),
            ):
                if name not in resolved:
                    continue
                number = resolved.get(name)
                if type(number) is not int or number < minimum:
                    errors.append(
                        f"observation {name} must be an integer >= {minimum}"
                    )
            for name in ("source_protection_active", "extinction_active"):
                if name not in resolved:
                    continue
                if type(resolved.get(name)) is not bool:
                    errors.append(f"observation {name} must be boolean")
            if (
                "extinction_model" in resolved
                and not isinstance(resolved.get("extinction_model"), str)
            ):
                errors.append("observation extinction_model must be a string")
            if type(value.get("filter_edge_guard_parity_deferred")) is not bool:
                errors.append(
                    "observation filter_edge_guard_parity_deferred must be boolean"
                )

            native = resolved.get("native_sample_rate_hz")
            effective = resolved.get("effective_sample_rate_hz")
            factor = resolved.get("downsample_factor")
            if (
                isinstance(native, (int, float))
                and not isinstance(native, bool)
                and isinstance(effective, (int, float))
                and not isinstance(effective, bool)
                and type(factor) is int
                and factor >= 1
                and math.isfinite(float(native))
                and math.isfinite(float(effective))
                and not math.isclose(
                    float(effective), float(native) / factor,
                    rel_tol=1.0e-9, abs_tol=1.0e-12,
                )
            ):
                errors.append(
                    "effective sample rate does not match native rate/downsample factor"
                )

        realized = data["realized"]
        if realized["execution_completed"] is not True:
            errors.append("raw observation execution is not complete")
        for name in (
            "completed_scan_count",
            "required_timestream_write_count",
        ):
            record = realized[name]
            if record.get("available") is not True:
                errors.append(f"realized {name} is unavailable")
                continue
            value = record.get("value")
            if type(value) is not int or value < 0:
                errors.append(
                    f"realized {name} must be a nonnegative integer"
                )
    except (AttributeError, KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate raw provenance semantics: {exc}")
    return errors


def mapmaking_provenance_semantic_errors(
    data: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("mapmaking execution plan is not initialized")

        requested = data["requested"]
        effective = data["effective"]["config"]
        resolution = data["effective"]["resolution"]
        realized = data["realized"]

        if effective["grouping"] != resolution["effective_grouping"]:
            errors.append(
                "grouping resolution does not match effective config"
            )
        if requested["grouping"] != resolution["requested_grouping"]:
            errors.append(
                "grouping resolution does not match requested config"
            )
        if effective["cunit"] != resolution["effective_unit"]:
            errors.append("unit resolution does not match effective config")
        if requested["cunit"] != resolution["requested_unit"]:
            errors.append("unit resolution does not match requested config")

        expected_automatic = requested["grouping"] == "auto"
        if resolution["automatic_grouping_resolved"] != expected_automatic:
            errors.append("automatic-grouping resolution is inconsistent")

        expected_fallback = bool(
            requested["grouping"] == "detector"
            and effective["grouping"] == "array"
        )
        if (
            resolution["detector_grouping_fell_back_to_array"]
            != expected_fallback
        ):
            errors.append("detector-grouping fallback is inconsistent")

        expected_substitution = requested["cunit"] != effective["cunit"]
        if (
            resolution["uncalibrated_unit_substituted"]
            != expected_substitution
        ):
            errors.append("unit-substitution resolution is inconsistent")

        if realized["reduction_completed"] is not True:
            errors.append("mapmaking reduction is not complete")
        if realized["mapmaking_executed"] != effective["enabled"]:
            errors.append(
                "mapmaking execution record does not match effective config"
            )
        if data.get("schema_version") == "citlali-mapmaking-provenance-v2":
            errors.extend(mapmaking_cardinality_semantic_errors(data))
    except (KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate mapmaking provenance semantics: {exc}")
    return errors


def available_count(record: Any, field: str) -> int:
    if not isinstance(record, dict) or record.get("available") is not True:
        raise ValueError(f"{field} is unavailable")
    value = record.get("value")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def valid_map_product_cardinality(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    map_count = record.get("map_count")
    write_count = record.get("required_map_write_count")
    return bool(
        isinstance(map_count, int)
        and not isinstance(map_count, bool)
        and map_count > 0
        and isinstance(write_count, int)
        and not isinstance(write_count, bool)
        and write_count >= map_count
        and write_count % map_count == 0
    )


def normalized_mapmaking_obsnum(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        numeric = int(value)
        return str(numeric) if numeric > 0 else None
    return None


def mapmaking_cardinality_semantic_errors(
    data: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        observations = data["observations"]
        coadd = data["coadd"]
        realized = data["realized"]
        effective = data["effective"]["config"]
        if not isinstance(observations, list):
            return ["mapmaking observations must be a sequence"]

        seen_obsnums: set[str] = set()
        for expected_index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                errors.append(
                    f"mapmaking observation {expected_index} is not a mapping"
                )
                continue
            if observation.get("observation_index") != expected_index:
                errors.append(
                    f"mapmaking observation {expected_index} has inconsistent index"
                )
            obsnum = normalized_mapmaking_obsnum(observation.get("obsnum"))
            if obsnum is None:
                errors.append(
                    f"mapmaking observation {expected_index} has invalid obsnum"
                )
            elif obsnum in seen_obsnums:
                errors.append(f"duplicate mapmaking obsnum: {obsnum}")
            else:
                seen_obsnums.add(obsnum)
            if not valid_map_product_cardinality(observation):
                errors.append(
                    f"mapmaking observation {expected_index} has invalid product cardinality"
                )
            pixel_size = observation.get("effective_pixel_size_rad")
            if (
                not isinstance(pixel_size, (int, float))
                or isinstance(pixel_size, bool)
                or not math.isfinite(pixel_size)
                or pixel_size <= 0.0
            ):
                errors.append(
                    f"mapmaking observation {expected_index} has invalid pixel size"
                )
            if observation.get("outputs_completed") is not True:
                errors.append(
                    f"mapmaking observation {expected_index} outputs are incomplete"
                )

        completed_observations = available_count(
            realized["completed_observation_count"],
            "completed_observation_count",
        )
        if completed_observations != len(observations):
            errors.append(
                "completed observation count does not match observations"
            )

        if not isinstance(coadd, dict) or not isinstance(
            coadd.get("available"), bool
        ):
            errors.append("mapmaking coadd availability is invalid")
            expected_coadds = 0
        elif coadd["available"]:
            expected_coadds = 1
            if not valid_map_product_cardinality(coadd):
                errors.append("mapmaking coadd has invalid product cardinality")
            if coadd.get("outputs_completed") is not True:
                errors.append("mapmaking coadd outputs are incomplete")
        else:
            expected_coadds = 0

        completed_coadds = available_count(
            realized["completed_coadd_count"], "completed_coadd_count"
        )
        if completed_coadds != expected_coadds:
            errors.append("completed coadd count does not match coadd state")

        if effective["enabled"] and not observations:
            errors.append("enabled mapmaking has no completed observations")
        if not effective["enabled"] and (observations or expected_coadds):
            errors.append("disabled mapmaking records map products")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot evaluate mapmaking cardinality: {exc}")
    return errors


def coadd_provenance_semantic_errors(
    data: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("coadd execution plan is not initialized")

        requested = data["requested"]["enabled"]
        effective = data["effective"]["config"]["enabled"]
        resolution = data["effective"]["resolution"]
        realized = data["realized"]
        if not isinstance(requested, bool) or not isinstance(effective, bool):
            errors.append("coadd activation values must be boolean")
            return errors

        mapmaking_enabled = resolution["mapmaking_enabled"]
        resolution_values = (
            mapmaking_enabled,
            resolution["requested_enabled"],
            resolution["effective_enabled"],
            resolution["disabled_by_mapmaking"],
        )
        if not all(isinstance(value, bool) for value in resolution_values):
            errors.append("coadd resolution values must be boolean")
            return errors
        realized_values = (
            realized["reduction_completed"],
            realized["coadd_executed"],
            realized["outputs_completed"],
        )
        if not all(isinstance(value, bool) for value in realized_values):
            errors.append("coadd realized-state values must be boolean")
            return errors

        expected_effective = requested and mapmaking_enabled
        if resolution["requested_enabled"] != requested:
            errors.append("coadd requested resolution is inconsistent")
        if resolution["effective_enabled"] != effective:
            errors.append("coadd effective resolution is inconsistent")
        if effective != expected_effective:
            errors.append("coadd activation does not follow mapmaking policy")
        if resolution["disabled_by_mapmaking"] != (
            requested and not mapmaking_enabled
        ):
            errors.append("coadd mapmaking-disable resolution is inconsistent")

        if realized["reduction_completed"] is not True:
            errors.append("coadd reduction is not complete")
        if realized["coadd_executed"] != effective:
            errors.append("coadd execution does not match effective policy")

        map_count_record = realized["map_count"]
        write_count_record = realized["required_map_write_count"]
        if effective:
            map_count = available_count(map_count_record, "coadd map_count")
            write_count = available_count(
                write_count_record, "coadd required_map_write_count"
            )
            if map_count <= 0 or write_count < map_count or (
                write_count % map_count
            ):
                errors.append("coadd realized cardinality is invalid")
            if realized["outputs_completed"] is not True:
                errors.append("coadd outputs are incomplete")
        else:
            if map_count_record.get("available") is not False or (
                write_count_record.get("available") is not False
            ):
                errors.append("disabled coadd records product cardinality")
            if realized["outputs_completed"] is not False:
                errors.append("disabled coadd records completed outputs")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot evaluate coadd provenance semantics: {exc}")
    return errors


def valid_noise_config(config: Any, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(config, dict):
        return [f"{label} noise config must be a mapping"]
    for path in (
        ("enabled",),
        ("randomize_dets",),
        ("write_realizations",),
        ("products", "enabled"),
        ("products", "apply_empirical_weights"),
    ):
        try:
            value = nested_value(config, path)
        except KeyError:
            errors.append(
                f"{label} noise config is missing {'.'.join(path)}"
            )
            continue
        if type(value) is not bool:
            errors.append(
                f"{label} noise config {'.'.join(path)} must be boolean"
            )
    count = config.get("n_noise_maps")
    if type(count) is not int or count < 0:
        errors.append(
            f"{label} noise config n_noise_maps must be a nonnegative integer"
        )
    return errors


def unavailable_count(record: Any, field: str) -> bool:
    if not isinstance(record, dict) or record.get("available") is not False:
        return False
    return "value" not in record


def noise_provenance_semantic_errors(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("noise-products execution plan is not initialized")

        requested = data["requested"]
        effective = data["effective"]["config"]
        resolution = data["effective"]["resolution"]
        realized = data["realized"]
        errors.extend(valid_noise_config(requested, "requested"))
        errors.extend(valid_noise_config(effective, "effective"))
        if errors:
            return errors

        for name in (
            "mapmaking_enabled",
            "requested_enabled",
            "effective_enabled",
            "disabled_by_mapmaking",
            "count_zeroed_while_disabled",
        ):
            if type(resolution.get(name)) is not bool:
                errors.append(f"noise resolution {name} must be boolean")
        for name in (
            "requested_n_noise_maps",
            "effective_n_noise_maps",
        ):
            value = resolution.get(name)
            if type(value) is not int or value < 0:
                errors.append(
                    f"noise resolution {name} must be a nonnegative integer"
                )
        if errors:
            return errors

        requested_enabled = requested["enabled"]
        mapmaking_enabled = resolution["mapmaking_enabled"]
        expected_effective = requested_enabled and mapmaking_enabled
        expected_count = (
            requested["n_noise_maps"] if expected_effective else 0
        )
        if resolution["requested_enabled"] != requested_enabled:
            errors.append("noise requested activation resolution is inconsistent")
        if resolution["effective_enabled"] != effective["enabled"]:
            errors.append("noise effective activation resolution is inconsistent")
        if effective["enabled"] != expected_effective:
            errors.append("noise activation does not follow mapmaking policy")
        if resolution["disabled_by_mapmaking"] != (
            requested_enabled and not mapmaking_enabled
        ):
            errors.append("noise mapmaking-disable resolution is inconsistent")
        if resolution["requested_n_noise_maps"] != requested["n_noise_maps"]:
            errors.append("noise requested count resolution is inconsistent")
        if resolution["effective_n_noise_maps"] != effective["n_noise_maps"]:
            errors.append("noise effective count resolution is inconsistent")
        if effective["n_noise_maps"] != expected_count:
            errors.append("noise effective count does not follow activation policy")
        if resolution["count_zeroed_while_disabled"] != (
            requested["n_noise_maps"] != effective["n_noise_maps"]
        ):
            errors.append("noise count-zeroing resolution is inconsistent")

        for name in (
            "randomize_dets",
            "write_realizations",
        ):
            if effective[name] != requested[name]:
                errors.append(f"noise effective {name} differs from request")
        for name in ("enabled", "apply_empirical_weights"):
            if effective["products"][name] != requested["products"][name]:
                errors.append(
                    f"noise effective products.{name} differs from request"
                )

        randomization = resolution.get("randomization")
        expected_randomization = {
            "engine": "boost::random::mt19937",
            "seed": 5489,
            "seed_policy": "fixed_internal_default",
            "generator_scope": "reduction_pipeline_invocation",
        }
        if not isinstance(randomization, dict) or any(
            randomization.get(name) != value
            for name, value in expected_randomization.items()
        ):
            errors.append("noise randomization identity is inconsistent")

        for name in (
            "reduction_completed",
            "generation_executed",
            "outputs_completed",
        ):
            if type(realized.get(name)) is not bool:
                errors.append(f"noise realized {name} must be boolean")
        if errors:
            return errors
        if realized["reduction_completed"] is not True:
            errors.append("noise-products reduction is not complete")

        count_names = (
            "noise_maps_per_scientific_map",
            "observation_scientific_map_count",
            "observation_noise_realization_count",
            "coadd_scientific_map_count",
            "coadd_noise_realization_count",
            "total_noise_realization_count",
            "empirical_product_map_count",
            "realization_image_write_count",
        )
        if not effective["enabled"]:
            if realized["generation_executed"]:
                errors.append("disabled noise-products records generation")
            if realized["outputs_completed"]:
                errors.append("disabled noise-products records completed outputs")
            for name in count_names:
                if not unavailable_count(realized.get(name), name):
                    errors.append(
                        f"disabled noise-products records {name}"
                    )
            return errors

        counts: dict[str, int] = {}
        for name in count_names:
            try:
                counts[name] = available_count(realized[name], name)
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            return errors
        if counts["noise_maps_per_scientific_map"] != effective["n_noise_maps"]:
            errors.append("noise realized per-map count differs from effective config")
        if realized["generation_executed"] != (
            effective["n_noise_maps"] > 0
        ):
            errors.append("noise generation record differs from effective count")
        expected_observation_realizations = (
            counts["observation_scientific_map_count"]
            * effective["n_noise_maps"]
        )
        expected_coadd_realizations = (
            counts["coadd_scientific_map_count"]
            * effective["n_noise_maps"]
        )
        if counts["observation_noise_realization_count"] != (
            expected_observation_realizations
        ):
            errors.append("noise observation realization count is inconsistent")
        if counts["coadd_noise_realization_count"] != (
            expected_coadd_realizations
        ):
            errors.append("noise coadd realization count is inconsistent")
        if counts["total_noise_realization_count"] != (
            expected_observation_realizations + expected_coadd_realizations
        ):
            errors.append("noise total realization count is inconsistent")

        active_product_maps = (
            counts["observation_scientific_map_count"]
            + counts["coadd_scientific_map_count"]
        )
        product_count = counts["empirical_product_map_count"]
        if not effective["products"]["enabled"]:
            if product_count != 0:
                errors.append("disabled empirical products have nonzero count")
        elif active_product_maps == 0 and product_count != 0:
            errors.append("empirical products exist without scientific maps")
        elif active_product_maps and product_count not in (
            active_product_maps, 2 * active_product_maps
        ):
            errors.append("empirical product count has invalid output-stage cardinality")

        output_realizations = (
            expected_coadd_realizations
            if counts["coadd_scientific_map_count"] > 0
            else expected_observation_realizations
        )
        write_count = counts["realization_image_write_count"]
        if not effective["write_realizations"]:
            if write_count != 0:
                errors.append("disabled realization outputs have nonzero count")
        elif output_realizations == 0 and write_count != 0:
            errors.append("realization outputs exist without realizations")
        elif output_realizations and write_count not in (
            output_realizations, 2 * output_realizations
        ):
            errors.append("realization write count has invalid output-stage cardinality")
        if realized["outputs_completed"] is not True:
            errors.append("enabled noise-products outputs are incomplete")
    except (AttributeError, KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate noise-products provenance semantics: {exc}")
    return errors


def noise_mapmaking_cross_check_errors(
    noise: dict[str, Any], mapmaking: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    try:
        if mapmaking.get("schema_version") != (
            "citlali-mapmaking-provenance-v2"
        ):
            return ["noise-products cross-check requires mapmaking provenance v2"]
        noise_resolution = noise["effective"]["resolution"]
        noise_effective = noise["effective"]["config"]
        noise_realized = noise["realized"]
        mapmaking_effective = mapmaking["effective"]["config"]
        if noise_resolution["mapmaking_enabled"] != (
            mapmaking_effective["enabled"]
        ):
            errors.append(
                "noise mapmaking activation differs from mapmaking provenance"
            )
        if not noise_effective["enabled"]:
            return errors

        observation_map_count = sum(
            observation["map_count"]
            for observation in mapmaking["observations"]
        )
        coadd = mapmaking["coadd"]
        coadd_available = coadd["available"]
        coadd_map_count = coadd["map_count"] if coadd_available else 0
        observation_noise_generated = (
            not coadd_available or mapmaking_effective["method"] == "jinc"
        )
        expected_observation_maps = (
            observation_map_count if observation_noise_generated else 0
        )
        realized_observation_maps = available_count(
            noise_realized["observation_scientific_map_count"],
            "observation_scientific_map_count",
        )
        realized_coadd_maps = available_count(
            noise_realized["coadd_scientific_map_count"],
            "coadd_scientific_map_count",
        )
        if realized_observation_maps != expected_observation_maps:
            errors.append(
                "noise observation map count differs from mapmaking provenance"
            )
        if realized_coadd_maps != coadd_map_count:
            errors.append(
                "noise coadd map count differs from mapmaking provenance"
            )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot cross-check noise and mapmaking provenance: {exc}")
    return errors


def valid_pointing_config(config: Any, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(config, dict):
        return [f"{label} pointing config must be a mapping"]
    strategy = config.get("source_strategy")
    if strategy not in ("standard", "psf_preserve"):
        errors.append(f"{label} pointing source_strategy is invalid")
    center_mode = config.get("fruitloops_center_mode")
    if center_mode not in ("auto", "header", "peak", "map_center"):
        errors.append(f"{label} pointing fruitloops_center_mode is invalid")
    for name in ("fit_gaussian", "header_require_coverage"):
        if type(config.get(name)) is not bool:
            errors.append(f"{label} pointing {name} must be boolean")
    radius = config.get("header_max_radius_arcsec")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(radius)
        or radius < 0
    ):
        errors.append(
            f"{label} pointing header_max_radius_arcsec must be finite and nonnegative"
        )
    return errors


def pointing_provenance_semantic_errors(
    data: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("pointing execution plan is not initialized")

        requested = data["requested"]
        effective = data["effective"]["config"]
        resolution = data["effective"]["resolution"]
        observations = data["observations"]
        realized = data["realized"]
        errors.extend(valid_pointing_config(requested, "requested"))
        errors.extend(valid_pointing_config(effective, "effective"))
        if not isinstance(observations, list):
            errors.append("pointing observations must be a sequence")
        if errors:
            return errors

        explicit = resolution.get("explicit_request")
        if not isinstance(explicit, dict) or any(
            type(explicit.get(name)) is not bool
            for name in (
                "source_strategy",
                "fit_gaussian",
                "fruitloops_center_mode",
                "header_max_radius_arcsec",
                "header_require_coverage",
            )
        ):
            errors.append("pointing explicit-request record is invalid")
            return errors
        for name in (
            "mapmaking_enabled",
            "map_filter_enabled",
            "coadd_enabled",
            "fit_output_path_available",
            "fit_disabled_by_mapmaking",
            "fit_disabled_by_output_policy",
            "header_max_radius_defaulted",
        ):
            if type(resolution.get(name)) is not bool:
                errors.append(f"pointing resolution {name} must be boolean")
        default_radius = resolution.get("default_header_max_radius_arcsec")
        if (
            isinstance(default_radius, bool)
            or not isinstance(default_radius, (int, float))
            or not math.isfinite(default_radius)
            or default_radius < 0
        ):
            errors.append("pointing default header radius is invalid")
        for name in (
            "reduction_completed",
            "pointing_executed",
            "outputs_completed",
        ):
            if type(realized.get(name)) is not bool:
                errors.append(f"pointing realized {name} must be boolean")
        schema_version = data.get("schema_version")
        fit_count_names = (
            ("fit_attempt_count", "valid_fit_count")
            if schema_version == "citlali-pointing-provenance-v1"
            else (
                "raw_fit_attempt_count",
                "raw_valid_fit_count",
                "filtered_fit_attempt_count",
                "filtered_valid_fit_count",
            )
        )
        count_names = (
            "completed_observation_count",
            "scientific_map_count",
            *fit_count_names,
        )
        for name in count_names:
            if type(realized.get(name)) is not int or realized[name] < 0:
                errors.append(
                    f"pointing realized {name} must be a nonnegative integer"
                )
        if errors:
            return errors

        mapmaking_enabled = resolution["mapmaking_enabled"]
        # Raw pointing fits consume normalized observation maps before the
        # optional filtering and coadd stages.
        fit_output_path_available = mapmaking_enabled
        expected_fit = (
            requested["fit_gaussian"] and fit_output_path_available
        )
        if effective["source_strategy"] != requested["source_strategy"]:
            errors.append("pointing effective source strategy differs from request")
        if effective["fruitloops_center_mode"] != requested[
            "fruitloops_center_mode"
        ]:
            errors.append("pointing effective center mode differs from request")
        if effective["header_require_coverage"] != requested[
            "header_require_coverage"
        ]:
            errors.append("pointing effective coverage policy differs from request")
        if effective["fit_gaussian"] != expected_fit:
            errors.append("pointing fit activation does not follow mapmaking policy")
        if resolution["fit_output_path_available"] != (
            fit_output_path_available
        ):
            errors.append("pointing fit-output resolution is inconsistent")
        if resolution["fit_disabled_by_mapmaking"] != (
            requested["fit_gaussian"] and not mapmaking_enabled
        ):
            errors.append("pointing fit-disable resolution is inconsistent")
        if resolution["fit_disabled_by_output_policy"] != (
            requested["fit_gaussian"]
            and mapmaking_enabled
            and not fit_output_path_available
        ):
            errors.append(
                "pointing output-policy fit-disable resolution is inconsistent"
            )
        radius_defaulted = not explicit["header_max_radius_arcsec"]
        if resolution["header_max_radius_defaulted"] != radius_defaulted:
            errors.append("pointing radius-default resolution is inconsistent")
        expected_radius = (
            default_radius
            if radius_defaulted
            else requested["header_max_radius_arcsec"]
        )
        if effective["header_max_radius_arcsec"] != expected_radius:
            errors.append("pointing effective header radius is inconsistent")

        if realized["reduction_completed"] is not True:
            errors.append("pointing reduction is not complete")
        if realized["pointing_executed"] != mapmaking_enabled:
            errors.append("pointing execution differs from mapmaking policy")
        if realized["outputs_completed"] is not True:
            errors.append("pointing outputs are incomplete")

        totals = {
            "completed_observation_count": 0,
            "scientific_map_count": 0,
            **{name: 0 for name in fit_count_names},
        }
        seen_obsnums: set[str] = set()
        for expected_index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                errors.append(
                    f"pointing observation {expected_index} must be a mapping"
                )
                continue
            if observation.get("observation_index") != expected_index:
                errors.append("pointing observation indices are not contiguous")
            obsnum = normalized_mapmaking_obsnum(observation.get("obsnum"))
            if obsnum is None:
                errors.append(
                    f"pointing observation {expected_index} has invalid obsnum"
                )
            elif obsnum in seen_obsnums:
                errors.append(f"duplicate pointing obsnum: {obsnum}")
            else:
                seen_obsnums.add(obsnum)
            map_count = observation.get("map_count")
            if type(map_count) is not int or map_count < 0:
                errors.append(
                    f"pointing observation {expected_index} has invalid cardinality"
                )
                continue
            if map_count == 0:
                errors.append(
                    f"pointing observation {expected_index} has no scientific maps"
                )
            if schema_version == "citlali-pointing-provenance-v1":
                stage_specs = (("", True),)
            else:
                stage_specs = (
                    ("raw_", True),
                    (
                        "filtered_",
                        resolution["map_filter_enabled"]
                        and not resolution["coadd_enabled"],
                    ),
                )
            for prefix, stage_expected in stage_specs:
                attempts_name = f"{prefix}fit_attempt_count"
                valid_name = f"{prefix}valid_fit_count"
                recorded_name = f"{prefix}fit_results_recorded"
                attempts = observation.get(attempts_name)
                valid = observation.get(valid_name)
                if any(
                    type(value) is not int or value < 0
                    for value in (attempts, valid)
                ):
                    errors.append(
                        f"pointing observation {expected_index} "
                        f"{prefix}fit cardinality is invalid"
                    )
                    continue
                expected_attempts = (
                    map_count
                    if stage_expected and effective["fit_gaussian"]
                    else 0
                )
                if attempts != expected_attempts:
                    errors.append(
                        f"pointing observation {expected_index} "
                        f"{prefix}fit attempts are inconsistent"
                    )
                if valid > attempts:
                    errors.append(
                        f"pointing observation {expected_index} "
                        f"{prefix}valid fits exceed attempts"
                    )
                if observation.get(recorded_name) is not stage_expected:
                    errors.append(
                        f"pointing observation {expected_index} "
                        f"{prefix}fit stage record is inconsistent"
                    )
                totals[attempts_name] += attempts
                totals[valid_name] += valid
            if observation.get("outputs_completed") is not True:
                errors.append(
                    f"pointing observation {expected_index} outputs are incomplete"
                )
            totals["completed_observation_count"] += 1
            totals["scientific_map_count"] += map_count

        if mapmaking_enabled and not observations:
            errors.append("enabled pointing has no completed observations")
        if not mapmaking_enabled and observations:
            errors.append("disabled pointing records observation products")
        for name, expected in totals.items():
            if realized[name] != expected:
                errors.append(f"pointing realized {name} is inconsistent")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot evaluate pointing provenance semantics: {exc}")
    return errors


def pointing_mapmaking_cross_check_errors(
    pointing: dict[str, Any], mapmaking: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    try:
        if mapmaking.get("schema_version") != (
            "citlali-mapmaking-provenance-v2"
        ):
            return ["pointing cross-check requires mapmaking provenance v2"]
        pointing_resolution = pointing["effective"]["resolution"]
        mapmaking_effective = mapmaking["effective"]["config"]
        if pointing_resolution["mapmaking_enabled"] != (
            mapmaking_effective["enabled"]
        ):
            errors.append(
                "pointing mapmaking activation differs from mapmaking provenance"
            )
        pointing_observations = pointing["observations"]
        mapmaking_observations = mapmaking["observations"]
        if len(pointing_observations) != len(mapmaking_observations):
            errors.append(
                "pointing observation count differs from mapmaking provenance"
            )
            return errors
        for index, (pointing_obs, mapmaking_obs) in enumerate(
            zip(pointing_observations, mapmaking_observations)
        ):
            if (
                pointing_obs["observation_index"]
                != mapmaking_obs["observation_index"]
                or normalized_mapmaking_obsnum(pointing_obs["obsnum"])
                != normalized_mapmaking_obsnum(mapmaking_obs["obsnum"])
                or pointing_obs["map_count"] != mapmaking_obs["map_count"]
                or pointing_obs["outputs_completed"]
                != mapmaking_obs["outputs_completed"]
            ):
                errors.append(
                    f"pointing observation {index} differs from mapmaking provenance"
                )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot cross-check pointing and mapmaking provenance: {exc}")
    return errors


def beammap_provenance_semantic_errors(
    data: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("beammap execution plan is not initialized")

        requested = data["requested"]
        effective = data["effective"]["config"]
        resolution = data["effective"]["resolution"]
        observations = data["observations"]
        realized = data["realized"]
        if not all(isinstance(value, dict) for value in (requested, effective)):
            return ["beammap requested/effective config must be mappings"]
        if not isinstance(observations, list):
            return ["beammap observations must be a sequence"]

        requested_iterations = requested.get("iter_max")
        effective_iterations = effective.get("iter_max")
        if any(
            type(value) is not int or value <= 0
            for value in (requested_iterations, effective_iterations)
        ):
            errors.append("beammap iteration limits must be positive integers")
        if resolution.get("requested_max_iterations") != requested_iterations:
            errors.append("beammap requested iteration resolution is inconsistent")
        if resolution.get("effective_max_iterations") != effective_iterations:
            errors.append("beammap effective iteration resolution is inconsistent")

        mapmaking_enabled = resolution.get("mapmaking_enabled")
        if type(mapmaking_enabled) is not bool:
            errors.append("beammap mapmaking activation must be boolean")
        detector_tod_config = effective.get("detector_tod_output")
        if not isinstance(detector_tod_config, dict) or type(
            detector_tod_config.get("enabled")
        ) is not bool:
            errors.append("beammap detector-TOD activation must be boolean")
            detector_tod_enabled = False
        else:
            detector_tod_enabled = detector_tod_config["enabled"]

        for name in (
            "reduction_completed",
            "beammap_executed",
            "outputs_completed",
        ):
            if type(realized.get(name)) is not bool:
                errors.append(f"beammap realized {name} must be boolean")
        completed_iterations = realized.get("completed_iteration_count")
        if type(completed_iterations) is not int or completed_iterations < 0:
            errors.append(
                "beammap completed iteration count must be a nonnegative integer"
            )
            completed_iterations = 0

        total_iterations = 0
        seen_obsnums: set[str] = set()
        for expected_index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                errors.append(
                    f"beammap observation {expected_index} must be a mapping"
                )
                continue
            if observation.get("observation_index") != expected_index:
                errors.append("beammap observation indices are not contiguous")
            obsnum = normalized_mapmaking_obsnum(observation.get("obsnum"))
            if obsnum is None:
                errors.append(
                    f"beammap observation {expected_index} has invalid obsnum"
                )
            elif obsnum in seen_obsnums:
                errors.append(f"duplicate beammap obsnum: {obsnum}")
            else:
                seen_obsnums.add(obsnum)

            if data.get("schema_version") == (
                "citlali-beammap-provenance-v2"
            ):
                observation_label = f"beammap observation {expected_index}"
                if (
                    observation.get("source_identity_authority")
                    != "telescope_data"
                ):
                    errors.append(
                        f"{observation_label} source identity authority is invalid"
                    )
                photometry = observation.get("photometry")
                photometry_label = f"{observation_label} photometry"
                if not isinstance(photometry, dict):
                    errors.append(f"{photometry_label} must be a mapping")
                else:
                    if (
                        photometry.get("calibrator_flux_authority")
                        != "tolproj"
                    ):
                        errors.append(
                            f"{photometry_label} calibrator flux authority is invalid"
                        )
                    if (
                        photometry.get("flux_input_path")
                        != "beammap_source.fluxes"
                    ):
                        errors.append(
                            f"{photometry_label} flux input path is invalid"
                        )
                    if (
                        photometry.get("required_flux_policy")
                        != "fail_reduction"
                    ):
                        errors.append(
                            f"{photometry_label} required flux policy is invalid"
                        )
                    fluxes = photometry.get("fluxes")
                    if not isinstance(fluxes, list) or not fluxes:
                        errors.append(
                            f"{photometry_label} fluxes must be a sequence"
                        )
                    else:
                        for flux_index, flux in enumerate(fluxes):
                            flux_label = (
                                f"{photometry_label} flux {flux_index}"
                            )
                            if not isinstance(flux, dict):
                                errors.append(f"{flux_label} must be a mapping")
                                continue
                            if not isinstance(flux.get("array_name"), str) or not flux[
                                "array_name"
                            ]:
                                errors.append(
                                    f"{flux_label} array name must not be empty"
                                )
                            value = flux.get("value_mJy")
                            uncertainty = flux.get("uncertainty_mJy")
                            if (
                                isinstance(value, bool)
                                or not isinstance(value, (int, float))
                                or not math.isfinite(value)
                                or value <= 0.0
                            ):
                                errors.append(
                                    f"{flux_label} value must be positive and finite"
                                )
                            if (
                                isinstance(uncertainty, bool)
                                or not isinstance(uncertainty, (int, float))
                                or not math.isfinite(uncertainty)
                                or uncertainty < 0.0
                            ):
                                errors.append(
                                    f"{flux_label} uncertainty must be nonnegative and finite"
                                )

            count_names = ("detector_count", "map_count", "scan_count")
            counts = {name: observation.get(name) for name in count_names}
            if any(type(value) is not int or value <= 0 for value in counts.values()):
                errors.append(
                    f"beammap observation {expected_index} has invalid cardinality"
                )
                continue
            map_count = counts["map_count"]
            iterations = observation.get("iterations")
            if not isinstance(iterations, list) or not iterations:
                errors.append(
                    f"beammap observation {expected_index} has no iterations"
                )
                continue

            previous_converged = 0
            for iteration_index, iteration in enumerate(iterations):
                label = (
                    f"beammap observation {expected_index} iteration "
                    f"{iteration_index}"
                )
                if not isinstance(iteration, dict):
                    errors.append(f"{label} must be a mapping")
                    continue
                if iteration.get("iteration_index") != iteration_index:
                    errors.append(f"{label} has inconsistent index")
                if iteration.get("phase") not in (
                    "legacy",
                    "locator",
                    "pre_measurement",
                    "measurement_start",
                    "measurement",
                ):
                    errors.append(f"{label} has invalid phase")
                active_maps = iteration.get("active_map_count")
                pass_count = iteration.get("mapmaking_pass_count")
                if (
                    type(active_maps) is not int
                    or active_maps <= 0
                    or active_maps > map_count
                ):
                    errors.append(f"{label} has invalid active-map count")
                if type(pass_count) is not int or pass_count <= 0:
                    errors.append(f"{label} has no completed mapmaking pass")
                rtc_rerun = iteration.get("source_aware_rtc_rerun")
                if (
                    not isinstance(rtc_rerun, dict)
                    or rtc_rerun.get("available") is not True
                    or type(rtc_rerun.get("value")) is not bool
                ):
                    errors.append(f"{label} has no RTC-rerun decision")
                if iteration.get("fitting_completed") is not True:
                    errors.append(f"{label} fitting is incomplete")
                if iteration.get("completed") is not True:
                    errors.append(f"{label} lifecycle is incomplete")

                newly_converged = iteration.get("newly_converged_map_count")
                total_converged = iteration.get("total_converged_map_count")
                if any(
                    type(value) is not int or value < 0
                    for value in (newly_converged, total_converged)
                ):
                    errors.append(f"{label} has invalid convergence counts")
                elif (
                    total_converged < previous_converged
                    or total_converged > map_count
                    or newly_converged
                    != total_converged - previous_converged
                ):
                    errors.append(f"{label} convergence counts are inconsistent")
                else:
                    previous_converged = total_converged

                reason = iteration.get("termination_reason")
                is_terminal = iteration_index == len(iterations) - 1
                if reason not in (
                    "none",
                    "maximum_iterations",
                    "all_maps_converged",
                ):
                    errors.append(f"{label} has invalid termination reason")
                elif is_terminal and reason == "none":
                    errors.append(f"{label} lacks terminal state")
                elif not is_terminal and reason != "none":
                    errors.append(f"{label} terminates before the final iteration")
                elif reason == "maximum_iterations" and (
                    type(effective_iterations) is int
                    and len(iterations) != effective_iterations
                ):
                    errors.append(f"{label} maximum-iteration state is inconsistent")
                elif reason == "all_maps_converged" and (
                    total_converged != map_count
                ):
                    errors.append(f"{label} convergence termination is incomplete")

            terminal = observation.get("terminal_iteration")
            terminal_reason = observation.get("termination_reason")
            if (
                not isinstance(terminal, dict)
                or terminal.get("available") is not True
                or terminal.get("value") != len(iterations) - 1
            ):
                errors.append(
                    f"beammap observation {expected_index} terminal iteration is inconsistent"
                )
            if terminal_reason != iterations[-1].get("termination_reason"):
                errors.append(
                    f"beammap observation {expected_index} termination reason is inconsistent"
                )

            detector_tod = observation.get("detector_tod")
            if not isinstance(detector_tod, dict):
                errors.append(
                    f"beammap observation {expected_index} detector TOD is missing"
                )
            else:
                required = detector_tod.get("required")
                expected_writes = 1 if detector_tod_enabled else 0
                if required is not detector_tod_enabled:
                    errors.append(
                        f"beammap observation {expected_index} detector-TOD policy differs from effective config"
                    )
                if detector_tod.get("completed_write_count") != expected_writes:
                    errors.append(
                        f"beammap observation {expected_index} detector-TOD write cardinality is inconsistent"
                    )
                if expected_writes:
                    try:
                        output_iteration = available_count(
                            detector_tod["output_iteration"],
                            "detector_tod.output_iteration",
                        )
                        detector_count = available_count(
                            detector_tod["detector_count"],
                            "detector_tod.detector_count",
                        )
                        slot_count = available_count(
                            detector_tod["slot_count"],
                            "detector_tod.slot_count",
                        )
                        sample_count = available_count(
                            detector_tod["maximum_sample_count"],
                            "detector_tod.maximum_sample_count",
                        )
                        if output_iteration >= len(iterations):
                            errors.append(
                                f"beammap observation {expected_index} detector-TOD iteration is invalid"
                            )
                        if detector_count != counts["detector_count"]:
                            errors.append(
                                f"beammap observation {expected_index} detector-TOD detector count is inconsistent"
                            )
                        if slot_count == 0 or sample_count == 0:
                            errors.append(
                                f"beammap observation {expected_index} detector-TOD shape is empty"
                            )
                    except (KeyError, ValueError) as exc:
                        errors.append(
                            f"beammap observation {expected_index} detector TOD is incomplete: {exc}"
                        )
            if observation.get("outputs_completed") is not True:
                errors.append(
                    f"beammap observation {expected_index} outputs are incomplete"
                )
            total_iterations += len(iterations)

        try:
            completed_observations = available_count(
                realized["completed_observation_count"],
                "completed_observation_count",
            )
            if completed_observations != len(observations):
                errors.append(
                    "beammap completed observation count is inconsistent"
                )
        except (KeyError, ValueError) as exc:
            errors.append(f"beammap completed observation count is invalid: {exc}")
        if completed_iterations != total_iterations:
            errors.append("beammap completed iteration count is inconsistent")
        if realized.get("reduction_completed") is not True:
            errors.append("beammap reduction is not complete")
        if realized.get("outputs_completed") is not True:
            errors.append("beammap outputs are incomplete")
        if realized.get("beammap_executed") != mapmaking_enabled:
            errors.append("beammap execution differs from mapmaking policy")
        if mapmaking_enabled and not observations:
            errors.append("enabled beammap has no completed observations")
        if mapmaking_enabled is False and observations:
            errors.append("disabled beammap records observation products")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot evaluate beammap provenance semantics: {exc}")
    return errors


def beammap_mapmaking_cross_check_errors(
    beammap: dict[str, Any], mapmaking: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    try:
        if mapmaking.get("schema_version") != (
            "citlali-mapmaking-provenance-v2"
        ):
            return ["beammap cross-check requires mapmaking provenance v2"]
        beammap_resolution = beammap["effective"]["resolution"]
        mapmaking_effective = mapmaking["effective"]["config"]
        if beammap_resolution["mapmaking_enabled"] != mapmaking_effective[
            "enabled"
        ]:
            errors.append(
                "beammap activation differs from mapmaking provenance"
            )
        beammap_observations = beammap["observations"]
        mapmaking_observations = mapmaking["observations"]
        if len(beammap_observations) != len(mapmaking_observations):
            errors.append(
                "beammap observation count differs from mapmaking provenance"
            )
            return errors
        for index, (beammap_obs, mapmaking_obs) in enumerate(
            zip(beammap_observations, mapmaking_observations)
        ):
            if (
                beammap_obs["observation_index"]
                != mapmaking_obs["observation_index"]
                or normalized_mapmaking_obsnum(beammap_obs["obsnum"])
                != normalized_mapmaking_obsnum(mapmaking_obs["obsnum"])
                or beammap_obs["map_count"] != mapmaking_obs["map_count"]
                or beammap_obs["outputs_completed"]
                != mapmaking_obs["outputs_completed"]
            ):
                errors.append(
                    f"beammap observation {index} differs from mapmaking provenance"
                )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(
            f"cannot cross-check beammap and mapmaking provenance: {exc}"
        )
    return errors


def beammap_post_processing_cross_check_errors(
    beammap: dict[str, Any], post_processing: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    try:
        resolution = post_processing["effective"]["resolution"]
        if resolution["reduction_type"] != "beammap":
            errors.append("beammap provenance is paired with non-beammap post-processing")
        if resolution["mapmaking_enabled"] != beammap["effective"][
            "resolution"
        ]["mapmaking_enabled"]:
            errors.append(
                "beammap activation differs from post-processing provenance"
            )
        if post_processing["realized"]["beammap_fits"][
            "context_count"
        ] != beammap["realized"]["completed_iteration_count"]:
            errors.append(
                "beammap iteration count differs from post-processing fit contexts"
            )
    except (KeyError, TypeError) as exc:
        errors.append(
            f"cannot cross-check beammap and post-processing provenance: {exc}"
        )
    return errors


def post_processing_fit_cardinality_errors(
    cardinality: Any, label: str,
) -> list[str]:
    if not isinstance(cardinality, dict):
        return [f"{label} must be a mapping"]
    errors: list[str] = []
    counts: dict[str, int] = {}
    for name in ("context_count", "attempt_count", "valid_count"):
        value = cardinality.get(name)
        if type(value) is not int or value < 0:
            errors.append(f"{label}.{name} must be a nonnegative integer")
        else:
            counts[name] = value
    if not errors and counts["valid_count"] > counts["attempt_count"]:
        errors.append(f"{label} valid fits exceed attempted fits")
    return errors


def post_processing_map_context_errors(
    state: Any, label: str, source_finding_enabled: bool,
) -> list[str]:
    if not isinstance(state, dict):
        return [f"{label} must be a mapping"]
    errors: list[str] = []
    counts: dict[str, int] = {}
    for name in (
        "filter_context_count",
        "filtered_map_count",
        "source_finding_context_count",
        "detected_source_count",
        "source_table_write_count",
        "source_table_row_count",
    ):
        value = state.get(name)
        if type(value) is not int or value < 0:
            errors.append(f"{label}.{name} must be a nonnegative integer")
        else:
            counts[name] = value
    errors.extend(
        post_processing_fit_cardinality_errors(
            state.get("catalog_fits"), f"{label}.catalog_fits"
        )
    )
    if errors:
        return errors
    fits = state["catalog_fits"]
    expected_source_contexts = (
        counts["filter_context_count"] if source_finding_enabled else 0
    )
    if counts["source_finding_context_count"] != expected_source_contexts:
        errors.append(f"{label} source-finding context count is inconsistent")
    if counts["source_table_write_count"] != expected_source_contexts:
        errors.append(f"{label} source-table write count is inconsistent")
    if fits["context_count"] != expected_source_contexts:
        errors.append(f"{label} catalog-fit context count is inconsistent")
    if fits["attempt_count"] != counts["detected_source_count"]:
        errors.append(f"{label} detected-source count is inconsistent")
    if counts["source_table_row_count"] != counts["detected_source_count"]:
        errors.append(f"{label} source-table row count is inconsistent")
    return errors


def post_processing_provenance_semantic_errors(
    data: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("post-processing execution plan is not initialized")
        requested = data["requested"]
        effective = data["effective"]["values"]
        resolution = data["effective"]["resolution"]
        realized = data["realized"]
        if not all(isinstance(value, dict) for value in (requested, effective)):
            return ["post-processing requested/effective config must be mappings"]

        requested_filter = requested["map_filtering"]["enabled"]
        effective_filter = effective["map_filtering"]["enabled"]
        requested_source = requested["source_finding"]["enabled"]
        effective_source = effective["source_finding"]["enabled"]
        effective_fitting = effective["source_fitting"]["active"]
        bool_values = (
            requested_filter,
            effective_filter,
            requested_source,
            effective_source,
            effective_fitting,
            resolution["mapmaking_enabled"],
            resolution["coadd_enabled"],
            resolution["map_filtering_requested"],
            resolution["map_filtering_effective"],
            resolution["map_filtering_disabled_by_mapmaking"],
            resolution["source_finding_requested"],
            resolution["source_finding_effective"],
            resolution["source_finding_disabled_by_mapmaking"],
            resolution["source_fitting_required_by_reduction"],
            resolution["source_fitting_required_by_map_filtering"],
            resolution["source_fitting_required_by_source_finding"],
            resolution["source_fitting_effective"],
            resolution["source_fitting_disabled_by_mapmaking"],
        )
        if not all(type(value) is bool for value in bool_values):
            return ["post-processing activation values must be boolean"]
        if requested_source and not requested_filter:
            errors.append("source finding is requested without map filtering")
        if resolution["map_filtering_requested"] != requested_filter:
            errors.append("map-filter request resolution is inconsistent")
        if resolution["map_filtering_effective"] != effective_filter:
            errors.append("map-filter effective resolution is inconsistent")
        if resolution["source_finding_requested"] != requested_source:
            errors.append("source-finding request resolution is inconsistent")
        if resolution["source_finding_effective"] != effective_source:
            errors.append("source-finding effective resolution is inconsistent")
        if resolution["source_fitting_effective"] != effective_fitting:
            errors.append("source-fitting effective resolution is inconsistent")
        mapmaking_enabled = resolution["mapmaking_enabled"]
        if effective_filter != (requested_filter and mapmaking_enabled):
            errors.append("map-filter activation does not follow mapmaking policy")
        if effective_source != (requested_source and mapmaking_enabled):
            errors.append("source-finding activation does not follow mapmaking policy")
        if effective_source and not effective_filter:
            errors.append("effective source finding lacks effective map filtering")
        if resolution["map_filtering_disabled_by_mapmaking"] != (
            requested_filter and not mapmaking_enabled
        ):
            errors.append("map-filter disable resolution is inconsistent")
        if resolution["source_finding_disabled_by_mapmaking"] != (
            requested_source and not mapmaking_enabled
        ):
            errors.append("source-finding disable resolution is inconsistent")

        reduction_type = resolution["reduction_type"]
        if reduction_type not in ("science", "pointing", "beammap"):
            errors.append("post-processing reduction type is invalid")
            return errors
        required_by_reduction = reduction_type in ("pointing", "beammap")
        fitting_required = (
            required_by_reduction or requested_filter or requested_source
        )
        if resolution["source_fitting_required_by_reduction"] != (
            required_by_reduction
        ):
            errors.append("source-fitting reduction requirement is inconsistent")
        if resolution["source_fitting_required_by_map_filtering"] != (
            requested_filter
        ):
            errors.append("source-fitting filter requirement is inconsistent")
        if resolution["source_fitting_required_by_source_finding"] != (
            requested_source
        ):
            errors.append("source-fitting finder requirement is inconsistent")
        if effective_fitting != (mapmaking_enabled and fitting_required):
            errors.append("source-fitting activation resolution is inconsistent")
        if resolution["source_fitting_disabled_by_mapmaking"] != (
            fitting_required and not mapmaking_enabled
        ):
            errors.append("source-fitting disable resolution is inconsistent")

        for name in ("reduction_completed", "outputs_completed"):
            if realized.get(name) is not True:
                errors.append(f"post-processing {name} is not true")
        errors.extend(
            post_processing_map_context_errors(
                realized["observation"], "observation", effective_source
            )
        )
        errors.extend(
            post_processing_map_context_errors(
                realized["coadd"], "coadd", effective_source
            )
        )
        for name, cardinality in (
            ("pointing raw fits", realized["pointing_fits"]["raw"]),
            ("pointing filtered fits", realized["pointing_fits"]["filtered"]),
            ("beammap fits", realized["beammap_fits"]),
        ):
            errors.extend(
                post_processing_fit_cardinality_errors(cardinality, name)
            )

        raw_contexts = realized["pointing_fits"]["raw"]["context_count"]
        filtered_contexts = realized["pointing_fits"]["filtered"]["context_count"]
        beammap_contexts = realized["beammap_fits"]["context_count"]
        if reduction_type == "pointing" and mapmaking_enabled:
            if raw_contexts == 0:
                errors.append("pointing reduction records no raw fit context")
            expected_filtered = realized["observation"]["filter_context_count"]
            if filtered_contexts != expected_filtered:
                errors.append("pointing filtered-fit context count is inconsistent")
        elif raw_contexts or filtered_contexts:
            errors.append("non-pointing reduction records pointing fits")
        if reduction_type == "beammap" and mapmaking_enabled:
            if beammap_contexts == 0:
                errors.append("beammap reduction records no fit context")
        elif beammap_contexts:
            errors.append("non-beammap reduction records beammap fits")
    except (KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate post-processing provenance semantics: {exc}")
    return errors


def post_processing_mapmaking_cross_check_errors(
    post_processing: dict[str, Any], mapmaking: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if mapmaking.get("schema_version") != (
            "citlali-mapmaking-provenance-v2"
        ):
            return [
                "post-processing cross-check requires mapmaking provenance v2"
            ]
        resolution = post_processing["effective"]["resolution"]
        realized = post_processing["realized"]
        mapmaking_effective = mapmaking["effective"]["config"]
        if resolution["mapmaking_enabled"] != mapmaking_effective["enabled"]:
            errors.append("post-processing mapmaking activation differs from mapmaking provenance")
        if not resolution["map_filtering_effective"]:
            expected_observation_contexts = 0
            expected_observation_maps = 0
            expected_coadd_contexts = 0
            expected_coadd_maps = 0
        elif resolution["coadd_enabled"]:
            coadd = mapmaking["coadd"]
            expected_observation_contexts = 0
            expected_observation_maps = 0
            expected_coadd_contexts = 1
            expected_coadd_maps = (
                coadd["map_count"] if coadd["available"] else -1
            )
        else:
            observations = mapmaking["observations"]
            expected_observation_contexts = len(observations)
            expected_observation_maps = sum(
                item["map_count"] for item in observations
            )
            expected_coadd_contexts = 0
            expected_coadd_maps = 0
        expected = (
            (
                realized["observation"],
                expected_observation_contexts,
                expected_observation_maps,
            ),
            (realized["coadd"], expected_coadd_contexts, expected_coadd_maps),
        )
        for state, contexts, maps in expected:
            if (
                state["filter_context_count"] != contexts
                or state["filtered_map_count"] != maps
            ):
                errors.append(
                    "post-processing filter cardinality differs from "
                    "mapmaking provenance"
                )
                break
    except (KeyError, TypeError) as exc:
        errors.append(
            "cannot cross-check post-processing and mapmaking provenance: "
            f"{exc}"
        )
    return errors


def post_processing_pointing_cross_check_errors(
    post_processing: dict[str, Any], pointing: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if pointing.get("schema_version") != (
            "citlali-pointing-provenance-v2"
        ):
            return [
                "post-processing cross-check requires pointing provenance v2"
            ]
        pp_fits = post_processing["realized"]["pointing_fits"]
        pointing_realized = pointing["realized"]
        pairs = (
            ("raw", "raw_fit_attempt_count", "raw_valid_fit_count"),
            ("filtered", "filtered_fit_attempt_count", "filtered_valid_fit_count"),
        )
        for stage, attempt_name, valid_name in pairs:
            if (
                pp_fits[stage]["attempt_count"] != pointing_realized[attempt_name]
                or pp_fits[stage]["valid_count"] != pointing_realized[valid_name]
            ):
                errors.append(
                    f"post-processing {stage} fits differ from pointing "
                    "provenance"
                )
    except (KeyError, TypeError) as exc:
        errors.append(
            "cannot cross-check post-processing and pointing provenance: "
            f"{exc}"
        )
    return errors


def kids_external_provenance_semantic_errors(
    data: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("KIDs external config plan is not initialized")
        if data["authority"] != "kidscpp":
            errors.append("KIDs external authority must be kidscpp")
        if data["config_schema"] != "citlali-kidscpp-bridge-v1":
            errors.append("KIDs external config schema is not recognized")
        if not isinstance(data["data_schema"], str) or not data["data_schema"]:
            errors.append("KIDs external data schema is empty")
        dependency = data["dependency"]
        if dependency["name"] != "kidscpp":
            errors.append("KIDs dependency name must be kidscpp")
        if (
            not isinstance(dependency["version"], str)
            or not dependency["version"]
        ):
            errors.append("KIDs dependency version is empty")
        supported = data["supported_tod_types"]
        if supported != ["xs", "rs", "is", "qs"]:
            errors.append("KIDs supported TOD types must be xs, rs, is, qs")
        if data["selected_tod_type"] not in supported:
            errors.append("selected KIDs TOD type is not supported")
        requested = data["requested"]["values"]
        effective = data["effective"]["values"]
        if not isinstance(
            data["requested"]["solver_extra_output_present"], bool
        ):
            errors.append("KIDs solver extra-output presence must be boolean")
        for label, identity in (
            ("requested", requested),
            ("effective", effective),
        ):
            if not isinstance(identity["fitter"]["modelspec"], str):
                errors.append(f"KIDs {label} fitter modelspec must be a string")
            if not isinstance(
                identity["fitter"]["weight_window"]["type"], str
            ):
                errors.append(
                    f"KIDs {label} weight-window type must be a string"
                )
            fwhm_hz = identity["fitter"]["weight_window"]["fwhm_Hz"]
            if not isinstance(fwhm_hz, (int, float)) or not math.isfinite(
                fwhm_hz
            ):
                errors.append(f"KIDs {label} weight-window FWHM must be finite")
            if not isinstance(identity["solver"]["extra_output"], bool):
                errors.append(f"KIDs {label} solver extra_output must be boolean")
            if not isinstance(identity["solver"]["fitreportdir"], str):
                errors.append(
                    f"KIDs {label} solver fitreportdir must be a string"
                )
            if not isinstance(identity["solver"]["parallel_policy"], str):
                errors.append(
                    f"KIDs {label} solver parallel policy must be a string"
                )
        if requested["fitter"] != effective["fitter"]:
            errors.append("KIDs effective fitter identity differs from request")
        for name in ("fitreportdir", "parallel_policy"):
            if requested["solver"][name] != effective["solver"][name]:
                errors.append(
                    f"KIDs effective solver {name} differs from request"
                )
        if effective["solver"]["extra_output"] is not False:
            errors.append(
                "KIDs effective solver extra_output must remain disabled"
            )
        forced = data["effective"]["resolution"][
            "solver_extra_output_forced_disabled"
        ]
        if not isinstance(forced, bool):
            errors.append("KIDs solver extra-output resolution must be boolean")
        expected_forced = bool(requested["solver"]["extra_output"])
        if forced != expected_forced:
            errors.append("KIDs solver extra-output resolution is inconsistent")
    except (KeyError, TypeError) as exc:
        errors.append(
            f"cannot evaluate KIDs external provenance semantics: {exc}"
        )
    return errors


def config_source_manifest_semantic_errors(
    data: dict[str, Any], manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        if data["merge_authority"] != "citlali_cli":
            errors.append("config merge authority must be citlali_cli")
        if data["merge_semantics"] != "ordered_later_sources_override":
            errors.append("config merge semantics are not recognized")
        upstream = data["upstream"]
        if upstream["authority"] != "tolteca":
            errors.append("upstream config authority must be tolteca")
        if upstream["ordered_sources_provided"] is not False:
            errors.append("TolTECA ordered-source availability must be false")

        sources = data["sources"]
        if not isinstance(sources, list) or not sources:
            return errors + ["config source manifest has no input sources"]
        copied_filenames: list[str] = []
        for index, source in enumerate(sources):
            label = f"config source {index}"
            if source["precedence"] != index:
                errors.append(f"{label} precedence is not contiguous")
            if source["role"] != "citlali_cli_config":
                errors.append(f"{label} role is not recognized")
            copied_filename = source["copied_filename"]
            if not isinstance(copied_filename, str) or not copied_filename:
                errors.append(f"{label} copied filename is empty")
                continue
            copied_filenames.append(copied_filename)
            copied_path = manifest_path.parent / copied_filename
            if not copied_path.is_file():
                errors.append(f"{label} copied file is missing")
                continue
            if source["size_bytes"] != copied_path.stat().st_size:
                errors.append(f"{label} copied file size differs")
            if source["sha256"] != sha256_file(copied_path):
                errors.append(f"{label} copied file SHA-256 differs")
        if len(copied_filenames) != len(set(copied_filenames)):
            errors.append("config source copied filenames are not unique")

        merged = data["merged"]
        if merged["serialization"] != "yaml_cpp_dump":
            errors.append("merged config serialization is not recognized")
        snapshot_filename = merged["snapshot_filename"]
        if snapshot_filename != "citlali_merged_config.yaml":
            errors.append("merged config snapshot filename is not recognized")
        else:
            snapshot_path = manifest_path.parent / snapshot_filename
            if not snapshot_path.is_file():
                errors.append("merged config snapshot is missing")
            else:
                if merged["size_bytes"] != snapshot_path.stat().st_size:
                    errors.append("merged config snapshot size differs")
                if merged["sha256"] != sha256_file(snapshot_path):
                    errors.append("merged config snapshot SHA-256 differs")
    except (KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate config source manifest semantics: {exc}")
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_provenance_files(redu: Path, filename: str) -> list[Path]:
    return sorted(redu.rglob(filename))


def audit_provenance_sidecars(
    redu: Path, require_processed: bool = False,
    require_raw: bool = False,
    require_mapmaking: bool = False,
    require_coadd: bool = False,
    require_noise_products: bool = False,
    require_pointing: bool = False,
    require_post_processing: bool = False,
    require_beammap: bool = False,
    require_kids_external: bool = False,
    require_config_source_manifest: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in PROVENANCE_SIDECARS.items():
        required = bool(
            (name == "processed_timestream" and require_processed)
            or (name == "raw_timestream" and require_raw)
            or (name == "mapmaking" and require_mapmaking)
            or (name == "coadd" and require_coadd)
            or (name == "noise_products" and require_noise_products)
            or (name == "pointing" and require_pointing)
            or (name == "post_processing" and require_post_processing)
            or (name == "beammap" and require_beammap)
            or (name == "kids_external" and require_kids_external)
            or (
                name == "config_source_manifest"
                and require_config_source_manifest
            )
        )
        paths = find_provenance_files(redu, str(spec["filename"]))
        record: dict[str, Any] = {
            "paths": [str(path) for path in paths],
            "count": len(paths),
            "present": bool(paths),
            "required": required,
            "valid": not required,
        }
        if not paths:
            result[name] = record
            continue
        file_records = []
        for path in paths:
            item: dict[str, Any] = {"path": str(path), "valid": False}
            try:
                data = load_yaml(path)
                if not isinstance(data, dict):
                    raise ValueError("provenance root must be a mapping")
                schema_version = data.get("schema_version")
                required_paths = spec.get(
                    "required_paths_by_schema", {}
                ).get(schema_version, spec["required_paths"])
                missing_paths = [
                    ".".join(section)
                    for section in required_paths
                    if not has_nested_path(data, section)
                ]
                accepted_schema_versions = spec.get(
                    "accepted_schema_versions", (spec["schema_version"],)
                )
                initialized_ok = data.get("initialized") is not False
                item.update(
                    {
                        "schema_version": schema_version,
                        "schema_ok": schema_version in accepted_schema_versions,
                        "missing_paths": missing_paths,
                        "initialized_ok": initialized_ok,
                        "sha256": sha256_file(path),
                    }
                )
                semantic_errors = []
                if not missing_paths:
                    if name == "kids_external":
                        semantic_errors = (
                            kids_external_provenance_semantic_errors(data)
                        )
                    elif name == "config_source_manifest":
                        semantic_errors = (
                            config_source_manifest_semantic_errors(data, path)
                        )
                    elif name == "processed_timestream":
                        semantic_errors = (
                            processed_provenance_semantic_errors(data)
                        )
                    elif name == "raw_timestream":
                        semantic_errors = raw_provenance_semantic_errors(data)
                    elif name == "mapmaking":
                        semantic_errors = (
                            mapmaking_provenance_semantic_errors(data)
                        )
                    elif name == "coadd":
                        semantic_errors = coadd_provenance_semantic_errors(
                            data
                        )
                    elif name == "noise_products":
                        semantic_errors = (
                            noise_provenance_semantic_errors(data)
                        )
                    elif name == "pointing":
                        semantic_errors = (
                            pointing_provenance_semantic_errors(data)
                        )
                    elif name == "post_processing":
                        semantic_errors = (
                            post_processing_provenance_semantic_errors(data)
                        )
                    elif name == "beammap":
                        semantic_errors = beammap_provenance_semantic_errors(
                            data
                        )
                item["semantic_errors"] = semantic_errors
                item["valid"] = bool(
                    item["schema_ok"]
                    and not missing_paths
                    and initialized_ok
                    and not semantic_errors
                )
            except Exception as exc:
                item["error"] = str(exc)
            file_records.append(item)
        cardinality_ok = bool(spec["allow_multiple"] or len(paths) == 1)
        record.update(
            {
                "files": file_records,
                "cardinality_ok": cardinality_ok,
                "schema_version": file_records[0].get("schema_version"),
                "schema_ok": all(
                    bool(item.get("schema_ok")) for item in file_records
                ),
                "initialized_ok": all(
                    bool(item.get("initialized_ok"))
                    for item in file_records
                ),
                "missing_paths": file_records[0].get("missing_paths", [])
                    if len(file_records) == 1 else {
                        item["path"]: item.get("missing_paths", [])
                        for item in file_records
                        if item.get("missing_paths")
                    },
                "sha256": file_records[0].get("sha256", "")
                    if len(file_records) == 1 else {
                        item["path"]: item.get("sha256", "")
                        for item in file_records
                    },
            }
        )
        record["valid"] = bool(
            cardinality_ok
            and all(bool(item.get("valid")) for item in file_records)
        )
        result[name] = record

    raw = result["raw_timestream"]
    output = result["timestream_output"]
    if raw["present"] or raw["required"]:
        output_by_dir = {
            str(Path(path).parent): path for path in output.get("paths", [])
        }
        raw_by_dir = {
            str(Path(path).parent): path for path in raw.get("paths", [])
        }
        missing_dirs = sorted(set(output_by_dir) - set(raw_by_dir))
        unexpected_dirs = sorted(set(raw_by_dir) - set(output_by_dir))
        coverage_ok = bool(
            output_by_dir and not missing_dirs and not unexpected_dirs
        )
        raw.update(
            {
                "observation_coverage_ok": coverage_ok,
                "missing_observation_dirs": missing_dirs,
                "unexpected_observation_dirs": unexpected_dirs,
            }
        )

        raw_items = {
            str(Path(item["path"]).parent): item
            for item in raw.get("files", [])
        }
        output_items = {
            str(Path(item["path"]).parent): item
            for item in output.get("files", [])
        }
        for observation_dir in sorted(set(output_by_dir) & set(raw_by_dir)):
            item = raw_items[observation_dir]
            if (
                not item.get("valid")
                or not output_items[observation_dir].get("valid")
            ):
                continue
            try:
                raw_data = load_yaml(Path(raw_by_dir[observation_dir]))
                output_data = load_yaml(Path(output_by_dir[observation_dir]))
                raw_scans = nested_value(
                    raw_data,
                    ("realized", "completed_scan_count", "value"),
                )
                output_scans = nested_value(
                    output_data, ("realized", "n_scans")
                )
                if raw_scans != output_scans:
                    item["semantic_errors"].append(
                        "completed scan count does not match "
                        "timestream-output provenance"
                    )
                    item["valid"] = False
            except Exception as exc:
                item["semantic_errors"].append(
                    f"cannot cross-check observation provenance: {exc}"
                )
                item["valid"] = False

        raw["valid"] = bool(
            raw["valid"]
            and coverage_ok
            and all(item.get("valid") for item in raw.get("files", []))
        )

    mapmaking = result["mapmaking"]
    coadd = result["coadd"]
    if mapmaking["present"] and coadd["present"] and (
        mapmaking["valid"] and coadd["valid"]
    ):
        try:
            mapmaking_data = load_yaml(Path(mapmaking["paths"][0]))
            coadd_data = load_yaml(Path(coadd["paths"][0]))
            mapmaking_coadd = mapmaking_data["coadd"]
            coadd_realized = coadd_data["realized"]
            available = mapmaking_coadd["available"]
            if available != coadd_realized["coadd_executed"]:
                raise ValueError(
                    "coadd execution differs from mapmaking provenance"
                )
            if available:
                if (
                    mapmaking_coadd["map_count"]
                    != coadd_realized["map_count"]["value"]
                    or mapmaking_coadd["required_map_write_count"]
                    != coadd_realized["required_map_write_count"]["value"]
                    or mapmaking_coadd["outputs_completed"]
                    != coadd_realized["outputs_completed"]
                ):
                    raise ValueError(
                        "coadd cardinality differs from mapmaking provenance"
                    )
        except Exception as exc:
            coadd["valid"] = False
            coadd.setdefault("cross_check_errors", []).append(str(exc))

    noise = result["noise_products"]
    if mapmaking["present"] and noise["present"] and (
        mapmaking["valid"] and noise["valid"]
    ):
        try:
            mapmaking_data = load_yaml(Path(mapmaking["paths"][0]))
            noise_data = load_yaml(Path(noise["paths"][0]))
            cross_check_errors = noise_mapmaking_cross_check_errors(
                noise_data, mapmaking_data
            )
            if cross_check_errors:
                noise["valid"] = False
                noise["cross_check_errors"] = cross_check_errors
        except Exception as exc:
            noise["valid"] = False
            noise.setdefault("cross_check_errors", []).append(str(exc))

    pointing = result["pointing"]
    if mapmaking["present"] and pointing["present"] and (
        mapmaking["valid"] and pointing["valid"]
    ):
        try:
            mapmaking_data = load_yaml(Path(mapmaking["paths"][0]))
            pointing_data = load_yaml(Path(pointing["paths"][0]))
            cross_check_errors = pointing_mapmaking_cross_check_errors(
                pointing_data, mapmaking_data
            )
            if cross_check_errors:
                pointing["valid"] = False
                pointing["cross_check_errors"] = cross_check_errors
        except Exception as exc:
            pointing["valid"] = False
            pointing.setdefault("cross_check_errors", []).append(str(exc))

    beammap = result["beammap"]
    if mapmaking["present"] and beammap["present"] and (
        mapmaking["valid"] and beammap["valid"]
    ):
        try:
            mapmaking_data = load_yaml(Path(mapmaking["paths"][0]))
            beammap_data = load_yaml(Path(beammap["paths"][0]))
            cross_check_errors = beammap_mapmaking_cross_check_errors(
                beammap_data, mapmaking_data
            )
            if cross_check_errors:
                beammap["valid"] = False
                beammap["cross_check_errors"] = cross_check_errors
        except Exception as exc:
            beammap["valid"] = False
            beammap.setdefault("cross_check_errors", []).append(str(exc))

    post_processing = result["post_processing"]
    if mapmaking["present"] and post_processing["present"] and (
        mapmaking["valid"] and post_processing["valid"]
    ):
        try:
            mapmaking_data = load_yaml(Path(mapmaking["paths"][0]))
            post_processing_data = load_yaml(
                Path(post_processing["paths"][0])
            )
            cross_check_errors = (
                post_processing_mapmaking_cross_check_errors(
                    post_processing_data, mapmaking_data
                )
            )
            if cross_check_errors:
                post_processing["valid"] = False
                post_processing["cross_check_errors"] = cross_check_errors
        except Exception as exc:
            post_processing["valid"] = False
            post_processing.setdefault("cross_check_errors", []).append(
                str(exc)
            )
    if pointing["present"] and post_processing["present"] and (
        pointing["valid"] and post_processing["valid"]
    ):
        try:
            pointing_data = load_yaml(Path(pointing["paths"][0]))
            post_processing_data = load_yaml(
                Path(post_processing["paths"][0])
            )
            cross_check_errors = (
                post_processing_pointing_cross_check_errors(
                    post_processing_data, pointing_data
                )
            )
            if cross_check_errors:
                post_processing["valid"] = False
                post_processing.setdefault(
                    "cross_check_errors", []
                ).extend(cross_check_errors)
        except Exception as exc:
            post_processing["valid"] = False
            post_processing.setdefault("cross_check_errors", []).append(
                str(exc)
            )
    if beammap["present"] and post_processing["present"] and (
        beammap["valid"] and post_processing["valid"]
    ):
        try:
            beammap_data = load_yaml(Path(beammap["paths"][0]))
            post_processing_data = load_yaml(
                Path(post_processing["paths"][0])
            )
            cross_check_errors = (
                beammap_post_processing_cross_check_errors(
                    beammap_data, post_processing_data
                )
            )
            if cross_check_errors:
                beammap["valid"] = False
                beammap.setdefault("cross_check_errors", []).extend(
                    cross_check_errors
                )
        except Exception as exc:
            beammap["valid"] = False
            beammap.setdefault("cross_check_errors", []).append(str(exc))
    return result


def provenance_ok(audit: dict[str, Any]) -> bool:
    return all(
        bool(record.get("valid"))
        for record in audit.get("provenance", {}).values()
    )


def provenance_hash_summary(record: dict[str, Any]) -> str:
    hashes = record.get("sha256", "")
    if isinstance(hashes, str):
        return hashes
    if not isinstance(hashes, dict):
        return ""
    unique = sorted({str(value) for value in hashes.values() if value})
    prefixes = ", ".join(value[:12] for value in unique)
    return f"{len(hashes)} files; {len(unique)} unique: {prefixes}"


def find_config(path: Path) -> Path | None:
    configs = sorted(path.glob("citlali_o*.yaml"))
    return configs[0] if configs else None


def find_log(path: Path) -> Path | None:
    candidates = sorted(path.glob("citlali.log*"))
    return candidates[0] if candidates else None


def audit_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "error": "no citlali_o*.yaml found"}
    result: dict[str, Any] = {"path": str(path)}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        result["labels"] = collect_labels_from_text(text)
        data = load_yaml(path)
        if data is not None:
            output_dirs = [str(value) for value in find_nested_key(data, "output_dir")]
            result["output_dirs"] = output_dirs
            result["n_threads"] = find_nested_key(data, "n_threads")
            result["parallel_policy"] = find_nested_key(data, "parallel_policy")
            result["reduction_type"] = find_nested_key(data, "reduction_type")
    except Exception as exc:
        result["error"] = str(exc)
    return result


def audit_log(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "error": "no citlali.log found"}
    markers: dict[str, TimedLine] = {}
    selected: list[str] = []
    labels: list[dict[str, str]] = []
    counts = Counter()
    mapmaking_starts: list[str] = []
    mapmaking_runs: list[str] = []
    ptc_chunks = 0
    first_ptc_chunk: datetime | None = None
    last_ptc_chunk: datetime | None = None
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    try:
        for line in open_text(path):
            stripped = line.rstrip("\n")
            ts = parse_timestamp(stripped)
            if ts is not None:
                first_ts = first_ts or ts
                last_ts = ts
            labels.extend(collect_labels_from_text(stripped))
            lowered = stripped.lower()
            if "fatal" in lowered:
                counts["fatal"] += 1
            if "critical" in lowered:
                counts["critical"] += 1
            if "error" in lowered:
                counts["error"] += 1
            if "traceback" in lowered:
                counts["traceback"] += 1
            if selected_log_line(stripped):
                selected.append(stripped)
            if ts is None:
                continue
            for key, text in LOG_MARKERS:
                if text in stripped:
                    markers.setdefault(key, TimedLine(ts, stripped))
                    if key == "first_mapmaking_start":
                        mapmaking_starts.append(ts.isoformat(sep=" "))
                    if key == "first_mapmaking_run":
                        mapmaking_runs.append(ts.isoformat(sep=" "))
            if "ptc diagnostics sidecar chunk written" in stripped:
                ptc_chunks += 1
                first_ptc_chunk = first_ptc_chunk or ts
                last_ptc_chunk = ts
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}

    intervals: dict[str, float] = {}
    for name, start_key, end_key in INTERVALS:
        start = markers.get(start_key)
        end = markers.get(end_key)
        if start is not None and end is not None:
            intervals[name] = (end.timestamp - start.timestamp).total_seconds()
    if ptc_chunks and first_ptc_chunk is not None and last_ptc_chunk is not None:
        intervals["ptc_first_to_last_chunk"] = (last_ptc_chunk - first_ptc_chunk).total_seconds()
        intervals["ptc_avg_chunk_spacing"] = (
            intervals["ptc_first_to_last_chunk"] / (ptc_chunks - 1) if ptc_chunks > 1 else 0.0
        )

    return {
        "path": str(path),
        "first_timestamp": first_ts.isoformat(sep=" ") if first_ts else None,
        "last_timestamp": last_ts.isoformat(sep=" ") if last_ts else None,
        "markers": {key: {"timestamp": value.timestamp.isoformat(sep=" "), "line": value.line} for key, value in markers.items()},
        "interval_seconds": intervals,
        "mapmaking_starts": mapmaking_starts,
        "mapmaking_runs": mapmaking_runs,
        "ptc_chunk_count": ptc_chunks,
        "issue_counts": dict(sorted(counts.items())),
        "labels": labels,
        "selected_lines": selected[:500],
    }


def product_kind(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        return "fits"
    if suffix in {".nc", ".nc4", ".cdf"}:
        return "netcdf"
    if suffix == ".ecsv":
        return "ecsv"
    if suffix == ".csv":
        return "csv"
    if suffix in {".log", ".gz"} and "log" in name:
        return "log"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    return "other"


def audit_products(path: Path, top: int) -> dict[str, Any]:
    files = [child for child in path.rglob("*") if child.is_file()]
    by_kind = Counter(product_kind(child) for child in files)
    comparable = [
        {
            "path": child.relative_to(path).as_posix(),
            "kind": product_kind(child),
            "size_bytes": child.stat().st_size,
        }
        for child in files
        if child.suffix.lower() in PRODUCT_SUFFIXES or product_kind(child) in {"fits", "netcdf", "ecsv", "csv"}
    ]
    comparable.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
    stable_comparable = [
        row for row in comparable
        if Path(str(row["path"])).name not in PROFILE_SIDECAR_NAMES
    ]
    stable_by_kind = Counter(str(row["kind"]) for row in stable_comparable)
    profile_sidecars = [
        row for row in comparable
        if Path(str(row["path"])).name in PROFILE_SIDECAR_NAMES
    ]
    return {
        "file_count": len(files),
        "counts_by_kind": dict(sorted(by_kind.items())),
        "comparable_count": len(comparable),
        "stable_counts_by_kind": dict(sorted(stable_by_kind.items())),
        "stable_comparable_count": len(stable_comparable),
        "profile_sidecars": profile_sidecars,
        "largest_comparable": comparable[:top],
    }


def unique_labels(*sections: dict[str, Any]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for section in sections:
        for item in section.get("labels", []):
            key = (str(item.get("mode", "")), str(item.get("label", "")))
            if key in seen:
                continue
            seen.add(key)
            result.append({"mode": key[0], "label": key[1]})
    return result


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    redu = resolve_redu_path(Path(args.reduction))
    config = audit_config(find_config(redu))
    log = audit_log(find_log(redu))
    labels = unique_labels(config, log)
    result = {
        "reduction": str(redu),
        "expected_label": args.expected_label,
        "expected_mode": args.expected_mode,
        "labels": labels,
        "label_ok": None,
        "mode_ok": None,
        "config": config,
        "log": log,
        "provenance": audit_provenance_sidecars(
            redu,
            getattr(args, "require_processed_provenance", False),
            getattr(args, "require_raw_provenance", False),
            getattr(args, "require_mapmaking_provenance", False),
            getattr(args, "require_coadd_provenance", False),
            getattr(args, "require_noise_products_provenance", False),
            getattr(args, "require_pointing_provenance", False),
            getattr(args, "require_post_processing_provenance", False),
            getattr(args, "require_beammap_provenance", False),
            getattr(args, "require_kids_external_provenance", False),
            getattr(args, "require_config_source_manifest", False),
        ),
        "products": audit_products(redu, args.top),
    }
    if args.expected_label:
        result["label_ok"] = any(item["label"] == args.expected_label for item in labels)
    if args.expected_mode:
        result["mode_ok"] = any(item["mode"] == args.expected_mode for item in labels)
    return result


def fmt_seconds(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Citlali Reduction Run Audit",
        "",
        f"- Reduction: `{result['reduction']}`",
        f"- Expected mode: `{result['expected_mode'] or ''}`",
        f"- Expected label: `{result['expected_label'] or ''}`",
        f"- Mode OK: `{result['mode_ok']}`",
        f"- Label OK: `{result['label_ok']}`",
        "",
        "## Identity",
        "",
    ]
    labels = result["labels"]
    if labels:
        for item in labels:
            lines.append(f"- `{item['mode']}/{item['label']}`")
    else:
        lines.append("- No validation path labels found in config or log.")

    config = result["config"]
    lines.extend(["", "## Config", ""])
    lines.append(f"- Config: `{config.get('path')}`")
    for key in ("reduction_type", "parallel_policy", "n_threads", "output_dirs"):
        if key in config:
            lines.append(f"- {key}: `{config[key]}`")
    if "error" in config:
        lines.append(f"- error: `{config['error']}`")

    log = result["log"]
    lines.extend(["", "## Log", ""])
    lines.append(f"- Log: `{log.get('path')}`")
    lines.append(f"- First timestamp: `{log.get('first_timestamp')}`")
    lines.append(f"- Last timestamp: `{log.get('last_timestamp')}`")
    lines.append(f"- PTC chunks: `{log.get('ptc_chunk_count')}`")
    lines.append(f"- Issue counts: `{log.get('issue_counts')}`")
    if "error" in log:
        lines.append(f"- error: `{log['error']}`")

    intervals = log.get("interval_seconds", {})
    if intervals:
        lines.extend(["", "## Timing", "", "| Interval | Seconds |", "| --- | ---: |"])
        for key, value in intervals.items():
            lines.append(f"| `{key}` | {fmt_seconds(value)} |")

    provenance = result["provenance"]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "| Sidecar | Present | Required | Valid | Schema | SHA-256 |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for name, record in provenance.items():
        lines.append(
            f"| `{name}` | `{record['present']}` | `{record['required']}` | "
            f"`{record['valid']}` | `{record.get('schema_version', '')}` | "
            f"`{provenance_hash_summary(record)}` |"
        )
        if record.get("missing_paths"):
            lines.append(
                f"\nMissing `{name}` paths: `"
                + "`, `".join(record["missing_paths"])
                + "`"
            )
        if record.get("error"):
            lines.append(f"\n`{name}` error: `{record['error']}`")
        for item in record.get("files", []):
            if item.get("semantic_errors"):
                lines.append(
                    f"\n`{name}` semantic errors for `{item['path']}`: `"
                    + "`; `".join(item["semantic_errors"])
                    + "`"
                )
        if record.get("cross_check_errors"):
            lines.append(
                f"\n`{name}` cross-check errors: `"
                + "`; `".join(record["cross_check_errors"])
                + "`"
            )
        if record.get("missing_observation_dirs"):
            lines.append(
                f"\n`{name}` missing observation directories: `"
                + "`, `".join(record["missing_observation_dirs"])
                + "`"
            )
        if record.get("unexpected_observation_dirs"):
            lines.append(
                f"\n`{name}` unexpected observation directories: `"
                + "`, `".join(record["unexpected_observation_dirs"])
                + "`"
            )

    products = result["products"]
    lines.extend(["", "## Products", ""])
    lines.append(f"- Files: `{products['file_count']}`")
    lines.append(f"- Comparable products: `{products['comparable_count']}`")
    lines.append(f"- Stable comparable products: `{products['stable_comparable_count']}`")
    lines.append(f"- Counts by kind: `{products['counts_by_kind']}`")
    lines.append(f"- Stable counts by kind: `{products['stable_counts_by_kind']}`")
    if products["profile_sidecars"]:
        lines.append(
            "- Profile sidecars: `" +
            ", ".join(str(row["path"]) for row in products["profile_sidecars"]) +
            "`"
        )
    if products["largest_comparable"]:
        lines.extend(["", "Largest comparable products:", ""])
        for row in products["largest_comparable"]:
            lines.append(f"- `{row['path']}` `{row['kind']}` {row['size_bytes']} bytes")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reduction", help="A reduNN directory or a reduced root containing reduNN children.")
    parser.add_argument("--expected-mode", default="", help="Expected validation mode, e.g. beammap.")
    parser.add_argument("--expected-label", default="", help="Expected validation label, e.g. refactor or citlali.")
    parser.add_argument("--top", type=int, default=12, help="Number of largest products to list.")
    parser.add_argument(
        "--require-processed-provenance",
        action="store_true",
        help="Fail unless processed_timestream_provenance.yaml is present and valid.",
    )
    parser.add_argument(
        "--require-raw-provenance",
        action="store_true",
        help="Fail unless every raw_timestream_provenance.yaml is valid.",
    )
    parser.add_argument(
        "--require-mapmaking-provenance",
        action="store_true",
        help="Fail unless mapmaking_provenance.yaml is present and valid.",
    )
    parser.add_argument(
        "--require-coadd-provenance",
        action="store_true",
        help="Fail unless coadd_provenance.yaml is present and valid.",
    )
    parser.add_argument(
        "--require-noise-products-provenance",
        action="store_true",
        help=(
            "Fail unless noise_products_provenance.yaml is present and valid."
        ),
    )
    parser.add_argument(
        "--require-pointing-provenance",
        action="store_true",
        help="Fail unless pointing_provenance.yaml is present and valid.",
    )
    parser.add_argument(
        "--require-post-processing-provenance",
        action="store_true",
        help=(
            "Fail unless post_processing_provenance.yaml is present and valid."
        ),
    )
    parser.add_argument(
        "--require-beammap-provenance",
        action="store_true",
        help="Fail unless beammap_provenance.yaml is present and valid.",
    )
    parser.add_argument(
        "--require-kids-external-provenance",
        action="store_true",
        help=(
            "Fail unless kids_external_provenance.yaml identifies a valid "
            "kidscpp bridge."
        ),
    )
    parser.add_argument(
        "--require-config-source-manifest",
        action="store_true",
        help=(
            "Fail unless the ordered config-source manifest and its copied "
            "inputs match their recorded hashes."
        ),
    )
    parser.add_argument("--json-out", default="", help="Optional path for machine-readable JSON.")
    parser.add_argument("--report-out", default="", help="Optional path for Markdown output.")
    return parser.parse_args(argv)


def write_text(path: str, text: str) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = build_audit(args)
    report = render_markdown(result)
    if args.json_out:
        write_text(args.json_out, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.report_out:
        write_text(args.report_out, report)
    print(report, end="")
    if result["label_ok"] is False or result["mode_ok"] is False:
        return 2
    if not provenance_ok(result):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
