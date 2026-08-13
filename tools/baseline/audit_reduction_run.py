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
import os
import re
import stat
import struct
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
    "astrometry": {
        "filename": "astrometry_provenance.yaml",
        "schema_version": "citlali-astrometry-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("authority", "calibration_selection"),
            ("authority", "application"),
            ("authority", "support_origin_metadata_available"),
            ("authority", "configured_values_origin"),
            ("identity", "axes"),
            ("identity", "offset_unit"),
            ("identity", "time_support"),
            ("identity", "algorithm"),
            ("contract", "upstream_selection_owner"),
            ("contract", "one_configured_value"),
            ("contract", "two_values_without_positive_mjd_pair"),
            ("contract", "two_values_with_positive_mjd_pair"),
            ("contract", "explicit_mjd_requires_observation_bracketing"),
            ("contract", "extrapolation"),
            ("expected_observation_count",),
            ("observations",),
            ("reduction_completed",),
        ),
        "allow_multiple": False,
    },
    "polarimetry": {
        "filename": "polarimetry_provenance.yaml",
        "schema_version": "citlali-polarimetry-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("capability", "status"),
            ("capability", "enabled_supported"),
            ("capability", "reason"),
            ("capability", "exit_condition"),
            ("requested",),
            ("effective", "config"),
            ("effective", "capability_resolution"),
            ("realized", "reduction_completed"),
            ("realized", "polarimetry_executed"),
            ("realized", "hwpr_loaded"),
        ),
        "allow_multiple": False,
    },
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
        "schema_version": "citlali-runtime-provenance-v2",
        "accepted_schema_versions": (
            "citlali-runtime-provenance-v1",
            "citlali-runtime-provenance-v2",
        ),
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
            ("expected", "initialized"),
            ("realized", "reduction_completed"),
            ("realized", "generation_executed"),
            ("realized", "actual_completion_valid"),
            ("realized", "completed_count_matches_effective"),
            ("realized", "completion_basis"),
            ("realized", "outputs_completed"),
            ("package", "package_id"),
            ("package", "provenance_id"),
            ("package", "product_contract_version"),
            ("package", "authority"),
            ("package", "detached_product_status"),
            ("package", "product_contract_inventory"),
            ("package", "member_files"),
            ("package", "member_count"),
            ("package", "member_inventory_digest"),
            ("package", "member_inventory_digest_kind"),
            ("package", "member_inventory_preimage_encoding"),
            ("package", "publication_state"),
            ("package", "complete"),
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
        "schema_version": "citlali-processed-timestream-provenance-v2",
        "accepted_schema_versions": (
            "citlali-processed-timestream-provenance-v1",
            "citlali-processed-timestream-provenance-v2",
        ),
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
        "schema_version": "citlali-raw-timestream-provenance-v4",
        "accepted_schema_versions": (
            "citlali-raw-timestream-provenance-v1",
            "citlali-raw-timestream-provenance-v2",
            "citlali-raw-timestream-provenance-v3",
            "citlali-raw-timestream-provenance-v4",
        ),
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

        if data.get("schema_version") == "citlali-processed-timestream-provenance-v2":
            restart_record = resolutions.get("fruit_loop_restart")
            if not isinstance(restart_record, dict):
                errors.append("effective.fruit_loop_restart is missing")
            else:
                restart_path = requested["fruit_loops"].get("restart_path")
                restart_requested = restart_path not in (None, "", "null")
                if restart_record.get("available") is not restart_requested:
                    errors.append(
                        "fruit-loop restart availability does not match requested restart_path"
                    )
                elif restart_requested:
                    restart = restart_record.get("value", {})
                    if restart.get("source_reduction_dir") != restart_path:
                        errors.append(
                            "fruit-loop restart source does not match requested restart_path"
                        )
                    if restart.get("next_iteration") != (
                        restart.get("completed_iteration", -2) + 1
                    ):
                        errors.append(
                            "fruit-loop restart iteration identity is inconsistent"
                        )

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


CANONICAL_CALIBRATION_LINEAGE_SCHEMA = (
    "sci-cal-001-canonical-calibration-lineage-v1"
)
CANONICAL_CALIBRATION_COMPONENTS = {
    "selected_apt_sha256",
    "selected_apt_row_association_sha256",
    "raw_acquisition_binding_sha256",
    "admitted_factor_state_sha256",
    "tolapt_manifest_association_sha256",
}
SELECTED_CALIBRATION_APT_FILENAME = "selected_calibration_apt.ecsv"
V4_CALIBRATION_STATE_FIELDS = (
    "reduced_observation_identity",
    "calibration_validity_detail",
    "calibration_product_schema",
    "calibration_target_unit",
    "calibration_photometry_policy",
    "calibration_factor_composition",
    "calibration_factor_provenance",
    "calibration_compatibility_fcf_semantics",
    "calibration_weight_recipient_semantics",
    "calibration_compact_covariance_state",
    "observation_flxscale_correction_applied",
    "applied_observation_flxscale_correction",
    "observation_flxscale_correction_state",
    "observation_flxscale_correction_source_identity",
    "observation_flxscale_correction_recipient_identity",
    "calibration_apt_artifact_sha256",
    "calibration_acquisition_binding_sha256",
    "calibration_identity",
    "calibration_package_identity",
    "calibration_factor_state_sha256",
    "calibration_raw_observation_identity",
    "calibration_acquisition_binding_mode",
    "calibration_acquisition_key_schema",
    "calibration_response_identity",
    "calibration_conditional_variance_transfer",
    "calibration_conditional_inverse_variance_transfer",
    "calibration_precision_limitation",
    "calibration_nuisance_states",
    "calibration_minimum_total_multiplier",
    "calibration_maximum_total_multiplier",
)


def canonical_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def calibration_identity_field(name: str, value: str) -> str:
    return f"|{len(name)}:{name}={len(value)}:{value}"


def recompute_calibration_package_identity(
    calibration_identity: str,
    selected_apt_sha256: str,
    acquisition_binding_sha256: str,
) -> str:
    preimage = "sci-cal-001-calibration-package-v2"
    for name, value in (
        ("calibration_identity", calibration_identity),
        ("package_local_apt_path", SELECTED_CALIBRATION_APT_FILENAME),
        ("package_local_apt_sha256", selected_apt_sha256),
        ("acquisition_binding_sha256", acquisition_binding_sha256),
    ):
        preimage += calibration_identity_field(name, value)
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def typed_calibration_identity_field(name: str, kind: str, value: str) -> str:
    return (
        f"|{len(name)}:{name}:{len(kind)}:{kind}:"
        f"{len(value)}:{value}"
    )


def cxx_hexfloat(value: float) -> str:
    """Return the finite binary64 spelling emitted by C++ std::hexfloat."""
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    sign = "-" if bits >> 63 else ""
    exponent_bits = (bits >> 52) & 0x7FF
    fraction = bits & ((1 << 52) - 1)
    if exponent_bits == 0x7FF:
        raise ValueError("non-finite canonical calibration value")
    if exponent_bits == 0 and fraction == 0:
        return f"{sign}0x0p+0"
    if exponent_bits:
        exponent = exponent_bits - 1023
        fraction_hex = f"{fraction:013x}".rstrip("0")
        significand = "1" + (f".{fraction_hex}" if fraction_hex else "")
    else:
        leading_bit = fraction.bit_length() - 1
        exponent = leading_bit - 1074
        remainder = fraction - (1 << leading_bit)
        shifted = remainder << (52 - leading_bit)
        fraction_hex = f"{shifted:013x}".rstrip("0")
        significand = "1" + (f".{fraction_hex}" if fraction_hex else "")
    exponent_text = f"+{exponent}" if exponent >= 0 else str(exponent)
    return f"{sign}0x{significand}p{exponent_text}"


def calibration_vector_identity_from_basis(
    basis: Any, label: str,
) -> tuple[str, list[float]]:
    if not isinstance(basis, dict):
        raise ValueError(f"{label} identity basis is not a mapping")
    if basis.get("schema_version") != "calibration-vector-hexfloat-v1":
        raise ValueError(f"{label} vector schema is invalid")
    count = basis.get("count")
    values = basis.get("values")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{label} vector count is invalid")
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{label} vector cardinality is inconsistent")
    parsed: list[float] = []
    preimage = f"calibration-vector-hexfloat-v1|count={count}"
    for index, encoded in enumerate(values):
        if not isinstance(encoded, str):
            raise ValueError(f"{label} vector value {index} is not text")
        try:
            value = float.fromhex(encoded)
        except ValueError as error:
            raise ValueError(
                f"{label} vector value {index} is not canonical hexfloat"
            ) from error
        if not math.isfinite(value) or cxx_hexfloat(value) != encoded:
            raise ValueError(
                f"{label} vector value {index} is not canonical hexfloat"
            )
        parsed.append(value)
        preimage += f"|{index}={encoded}"
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()
    if basis.get("sha256") != digest:
        raise ValueError(f"{label} vector digest does not recompute")
    return digest, parsed


def recompute_applied_extinction_state(
    state: Any,
) -> tuple[str, bool]:
    if not isinstance(state, dict):
        raise ValueError("applied extinction identity basis is not a mapping")
    if state.get("schema_version") != \
            "sci-cal-001-applied-extinction-state-basis-v1":
        raise ValueError("applied extinction identity basis schema is invalid")
    if state.get("available") is not True or type(state.get("active")) is not bool:
        raise ValueError("applied extinction identity basis is unavailable")
    active = state["active"]
    if not active:
        if set(state) != {
            "schema_version", "available", "active", "sha256",
        }:
            raise ValueError("inactive extinction identity basis is not empty")
        digest = hashlib.sha256(
            b"sci-cal-001-applied-extinction-state-v1|active=false"
        ).hexdigest()
    else:
        sample_digest, samples = calibration_vector_identity_from_basis(
            state.get("sample_elevation_rad"), "sample elevation"
        )
        arrays = state.get("los_tau_by_array")
        if not isinstance(arrays, list) or not arrays:
            raise ValueError("active extinction LOS-tau basis is incomplete")
        preimage = "sci-cal-001-applied-extinction-state-v1|active=true"
        preimage += calibration_identity_field(
            "sample_elevation_sha256", sample_digest
        )
        observed_indices: list[int] = []
        for item in arrays:
            if not isinstance(item, dict) or set(item) != {"array_index", "los_tau"}:
                raise ValueError("active extinction LOS-tau entry is invalid")
            array_index = item["array_index"]
            if isinstance(array_index, bool) or not isinstance(array_index, int):
                raise ValueError("active extinction array identity is invalid")
            los_digest, los_values = calibration_vector_identity_from_basis(
                item["los_tau"], f"array {array_index} LOS tau"
            )
            if len(los_values) != len(samples):
                raise ValueError("active extinction vector cardinalities differ")
            observed_indices.append(array_index)
            preimage += calibration_identity_field(
                f"array_{array_index}_los_tau_sha256", los_digest
            )
        if observed_indices != [0, 1, 2]:
            raise ValueError("active extinction arrays are not exactly ordered 0,1,2")
        digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()
    if state.get("sha256") != digest:
        raise ValueError("applied extinction state digest does not recompute")
    return digest, active


def recompute_admitted_factor_state(factors: Any) -> str:
    if not isinstance(factors, dict):
        raise ValueError("v4 factor-operator lineage is not a mapping")
    basis = factors.get("identity_basis")
    if not isinstance(basis, dict) or basis.get("schema_version") != \
            "sci-cal-001-admitted-factor-identity-basis-v1":
        raise ValueError("admitted-factor identity basis schema is invalid")
    target_digest, target = calibration_vector_identity_from_basis(
        basis.get("target_unit_factor"), "target-unit factor"
    )
    flxscale_digest, flxscale = calibration_vector_identity_from_basis(
        basis.get("detector_flxscale"), "detector flxscale"
    )
    minimum_digest, minimum = calibration_vector_identity_from_basis(
        basis.get("minimum_extinction_correction"),
        "minimum extinction correction",
    )
    maximum_digest, maximum = calibration_vector_identity_from_basis(
        basis.get("maximum_extinction_correction"),
        "maximum extinction correction",
    )
    cardinalities = {len(target), len(flxscale), len(minimum), len(maximum)}
    if cardinalities == {0} or len(cardinalities) != 1:
        raise ValueError("admitted-factor vector cardinalities differ")
    if any(
        not math.isfinite(value) or value <= 0.0
        for vector in (target, flxscale, minimum, maximum)
        for value in vector
    ):
        raise ValueError("admitted-factor identity basis is non-finite or non-positive")
    extinction_digest, extinction_active = recompute_applied_extinction_state(
        basis.get("applied_sample_extinction_state")
    )
    applied = factors.get("observation_flxscale_correction_applied")
    correction = factors.get("applied_observation_flxscale_correction")
    if type(applied) is not bool or isinstance(correction, bool) or \
            not isinstance(correction, (int, float)) or \
            not math.isfinite(float(correction)) or float(correction) <= 0.0:
        raise ValueError("observation correction factor state is invalid")
    if not extinction_active and any(value != 1.0 for value in minimum + maximum):
        raise ValueError("inactive extinction and factor vectors conflict")
    preimage = "sci-cal-001-admitted-factor-state-v1"
    for name, value in (
        ("target_unit_factor_sha256", target_digest),
        ("observation_flxscale_correction_applied", "true" if applied else "false"),
        ("applied_observation_flxscale_correction", cxx_hexfloat(float(correction))),
        ("observation_flxscale_correction_state", factors.get(
            "observation_flxscale_correction_state"
        )),
        ("observation_flxscale_correction_source_identity", factors.get(
            "observation_flxscale_correction_source_identity"
        )),
        ("observation_flxscale_correction_recipient_identity", factors.get(
            "observation_flxscale_correction_recipient_identity"
        )),
        ("detector_flxscale_sha256", flxscale_digest),
        ("minimum_extinction_correction_sha256", minimum_digest),
        ("maximum_extinction_correction_sha256", maximum_digest),
        ("applied_sample_extinction_state_sha256", extinction_digest),
    ):
        if not isinstance(value, str):
            raise ValueError(f"admitted-factor field {name} is invalid")
        preimage += calibration_identity_field(name, value)
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()
    if factors.get("factor_state_sha256") != digest:
        raise ValueError("v4 admitted-factor state identity does not recompute")
    return digest


def validate_selected_apt_factor_binding(
    path: Path, stable_joins: Any, factors: Any,
) -> None:
    from astropy.table import Table

    if not isinstance(stable_joins, dict) or not isinstance(factors, dict):
        raise ValueError("v4 selected-APT factor binding is incomplete")
    try:
        table = Table.read(path, format="ascii.ecsv")
    except Exception as error:
        raise ValueError("v4 selected APT sibling is not valid ECSV") from error
    required_columns = {"uid", "nw", "tone_freq", "flag", "flxscale"}
    if not required_columns.issubset(table.colnames):
        raise ValueError("v4 selected APT sibling columns are incomplete")
    rows = stable_joins.get("ordered_detector_apt_rows")
    basis = factors.get("identity_basis")
    if not isinstance(rows, list) or not isinstance(basis, dict):
        raise ValueError("v4 selected-APT factor binding is incomplete")
    _, flxscale = calibration_vector_identity_from_basis(
        basis.get("detector_flxscale"), "detector flxscale"
    )
    if len(rows) != len(flxscale):
        raise ValueError("v4 selected-APT/factor detector cardinalities differ")
    source_indices: set[int] = set()
    for detector_index, row in enumerate(rows):
        source_index = row.get("selected_apt_source_row_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or \
                not 0 <= source_index < len(table) or \
                source_index in source_indices:
            raise ValueError("v4 selected-APT source row binding is invalid")
        source_indices.add(source_index)
        apt_row = table[source_index]
        try:
            uid = str(apt_row["uid"])
            network = int(apt_row["nw"])
            tone_frequency = float(apt_row["tone_freq"])
            flag = int(apt_row["flag"])
            apt_flxscale = float(apt_row["flxscale"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "v4 selected APT sibling row values are invalid"
            ) from error
        if not math.isfinite(tone_frequency) or not math.isfinite(apt_flxscale):
            raise ValueError("v4 selected APT sibling row values are non-finite")
        retained = {
            field.get("name"): field.get("value")
            for field in row.get("retained_fields", [])
            if isinstance(field, dict)
        }
        if uid != row.get("uid") or uid != retained.get("uid") or \
                str(flag) != retained.get("flag") or \
                network != row.get("raw_network") or \
                tone_frequency != row.get("absolute_tone_frequency_hz") or \
                (flag == 0) != row.get("eligible"):
            raise ValueError(
                "v4 selected APT sibling row differs from serialized detector join"
            )
        local_tone = sum(
            int(table[index]["nw"]) == network
            for index in range(source_index)
        )
        if local_tone != row.get("raw_network_local_tone"):
            raise ValueError(
                "v4 selected APT sibling row order differs from detector join"
            )
        if apt_flxscale != flxscale[detector_index]:
            raise ValueError(
                "v4 selected APT flxscale differs from admitted factor state"
            )
    if source_indices != set(range(len(table))):
        raise ValueError(
            "v4 selected-APT source row coverage is incomplete"
        )


def yaml_values_exactly_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if len(left) != len(right):
            return False
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            for index, (right_key, right_value) in enumerate(unmatched):
                if yaml_values_exactly_equal(left_key, right_key):
                    if not yaml_values_exactly_equal(left_value, right_value):
                        return False
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    if isinstance(left, list):
        return len(left) == len(right) and all(
            yaml_values_exactly_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def validate_requested_response_preimage(
    data: dict[str, Any], response: Any,
) -> str:
    if not isinstance(response, dict):
        raise ValueError("v4 response-basis provenance is unavailable")
    preimage = response.get("requested_config_preimage")
    if not isinstance(preimage, dict) or set(preimage) != {
        "serialization", "value", "sha256",
    } or preimage.get("serialization") != "yaml-request-node-v1" or \
            not isinstance(preimage.get("value"), str) or \
            not canonical_sha256(preimage.get("sha256")):
        raise ValueError("v4 requested-config preimage is incomplete")
    serialized = preimage["value"]
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if preimage["sha256"] != digest:
        raise ValueError("v4 requested-config preimage digest does not recompute")
    if yaml is None:
        raise ValueError("v4 requested-config preimage YAML parser is unavailable")
    try:
        requested_from_preimage = yaml.safe_load(serialized)
    except Exception as error:
        raise ValueError("v4 requested-config preimage is not valid YAML") from error
    requested = data.get("requested")
    if not isinstance(requested, dict):
        raise ValueError("v4 requested raw config is unavailable")
    requested_raw = dict(requested)
    requested_raw.pop("calibration", None)
    requested_raw.pop("interface_sync_offset", None)
    if not yaml_values_exactly_equal(requested_from_preimage, requested_raw):
        raise ValueError(
            "v4 requested-config preimage differs from requested raw config"
        )
    provenance = response.get("provenance")
    if not isinstance(provenance, str):
        raise ValueError("v4 response-basis provenance is unavailable")
    matches = re.findall(r"(?:^|;)requested_state_sha256=([^;]+)", provenance)
    sources = re.findall(r"(?:^|;)requested_state_source=([^;]+)", provenance)
    if matches != [digest] or sources != ["raw_timestream_plan.requested"]:
        raise ValueError(
            "v4 requested-response identity does not match requested-config preimage"
        )
    return digest


def validate_v4_calibration_state_joins(
    data: dict[str, Any], value: dict[str, Any], provenance_path: Path | None,
) -> None:
    observation = data.get("observation", {}).get("value", {})
    realized = data.get("realized", {})
    for field in V4_CALIBRATION_STATE_FIELDS:
        observation_record = observation.get(field)
        realized_record = realized.get(field)
        if not isinstance(observation_record, dict) or \
                observation_record.get("available") is not True or \
                not isinstance(realized_record, dict) or \
                realized_record.get("available") is not True:
            raise ValueError(
                f"v4 observation/realized calibration {field} is unavailable"
            )
        if observation_record.get("value") != realized_record.get("value"):
            raise ValueError(
                f"v4 observation/realized calibration {field} differs"
            )
    package_observation = value.get("package_observation_identity")
    if not isinstance(package_observation, str) or not package_observation:
        raise ValueError("v4 package observation identity is unavailable")
    if observation["reduced_observation_identity"]["value"] != \
            package_observation:
        raise ValueError(
            "v4 package observation identity differs from observation state"
        )
    if provenance_path is None:
        raise ValueError(
            "v4 calibrated provenance path is unavailable for observation binding"
        )
    if provenance_path.parent.name != package_observation:
        raise ValueError(
            "v4 package observation identity differs from owning directory"
        )
    selected = value["selected_apt"]
    acquisition = value["raw_acquisition"]
    factors = value["factor_operator_state"]
    response = value["response_basis"]
    expected = {
        "calibration_product_schema": factors.get("product_schema"),
        "calibration_target_unit": factors.get("target_unit"),
        "calibration_photometry_policy": factors.get("photometry_policy"),
        "calibration_factor_composition": factors.get("factor_composition"),
        "calibration_factor_provenance": factors.get("factor_provenance"),
        "calibration_compatibility_fcf_semantics": factors.get(
            "compatibility_fcf_semantics"
        ),
        "calibration_weight_recipient_semantics": factors.get(
            "weight_recipient_semantics"
        ),
        "calibration_compact_covariance_state": factors.get(
            "compact_covariance_state"
        ),
        "observation_flxscale_correction_applied": factors.get(
            "observation_flxscale_correction_applied"
        ),
        "applied_observation_flxscale_correction": factors.get(
            "applied_observation_flxscale_correction"
        ),
        "observation_flxscale_correction_state": factors.get(
            "observation_flxscale_correction_state"
        ),
        "observation_flxscale_correction_source_identity": factors.get(
            "observation_flxscale_correction_source_identity"
        ),
        "observation_flxscale_correction_recipient_identity": factors.get(
            "observation_flxscale_correction_recipient_identity"
        ),
        "calibration_apt_artifact_sha256": selected.get("package_local_sha256"),
        "calibration_acquisition_binding_sha256": acquisition.get(
            "binding_sha256"
        ),
        "calibration_identity": value.get("calibration_identity"),
        "calibration_package_identity": value.get("package_identity"),
        "calibration_factor_state_sha256": factors.get("factor_state_sha256"),
        "calibration_raw_observation_identity": acquisition.get(
            "raw_observation_identity"
        ),
        "calibration_acquisition_binding_mode": acquisition.get("binding_mode"),
        "calibration_acquisition_key_schema": acquisition.get("key_schema"),
        "calibration_response_identity": response.get("provenance"),
        "calibration_conditional_variance_transfer": factors.get(
            "conditional_variance_transfer"
        ),
        "calibration_conditional_inverse_variance_transfer": factors.get(
            "conditional_inverse_variance_transfer"
        ),
        "calibration_precision_limitation": value.get("precision_limitation"),
        "calibration_nuisance_states": value.get("nuisance_states"),
        "calibration_minimum_total_multiplier": factors.get(
            "minimum_total_multiplier"
        ),
        "calibration_maximum_total_multiplier": factors.get(
            "maximum_total_multiplier"
        ),
    }
    for field, expected_value in expected.items():
        if observation[field]["value"] != expected_value:
            raise ValueError(
                f"v4 observation calibration {field} does not join canonical lineage"
            )


def recompute_ordered_row_association(
    stable_joins: Any, selected_apt_sha256: str,
) -> str:
    if not isinstance(stable_joins, dict):
        raise ValueError("v4 stable-join lineage is not a mapping")
    if stable_joins.get("apt_row_order_authoritative") is not False:
        raise ValueError("v4 selected APT row order is incorrectly authoritative")
    rows = stable_joins.get("ordered_detector_apt_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("v4 ordered detector/APT rows are incomplete")
    preimage = "selected-apt-row-association-v2"
    preimage += typed_calibration_identity_field(
        "apt_sha256", "sha256", selected_apt_sha256
    )
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("v4 ordered detector/APT row is not a mapping")
        ordered_index = row.get("ordered_detector_index")
        source_index = row.get("selected_apt_source_row_index")
        if ordered_index != expected_index or isinstance(source_index, bool) or \
                not isinstance(source_index, int) or source_index < 0:
            raise ValueError("v4 ordered detector/APT row indices are invalid")
        retained = row.get("retained_fields")
        if not isinstance(retained, list) or not retained:
            raise ValueError("v4 ordered detector/APT retained fields are incomplete")
        stable = "selected-apt-ordered-row-v2"
        stable += typed_calibration_identity_field(
            "ordered_detector_index", "index", str(ordered_index)
        )
        stable += typed_calibration_identity_field(
            "selected_source_row_index", "index", str(source_index)
        )
        retained_names: list[str] = []
        for field in retained:
            if not isinstance(field, dict) or set(field) != {
                "name", "ecsv_datatype", "value",
            }:
                raise ValueError("v4 retained APT row field is invalid")
            name = field["name"]
            datatype = field["ecsv_datatype"]
            value = field["value"]
            if not all(isinstance(item, str) and item for item in (
                name, datatype, value,
            )):
                raise ValueError("v4 retained APT row field is incomplete")
            retained_names.append(name)
            stable += typed_calibration_identity_field(name, datatype, value)
        if len(retained_names) != len(set(retained_names)) or \
                "uid" not in retained_names or "flag" not in retained_names:
            raise ValueError("v4 retained APT row fields are conflicting")
        eligible = row.get("eligible")
        validity_basis = row.get("validity_basis")
        if type(eligible) is not bool or not isinstance(validity_basis, str) or \
                not validity_basis:
            raise ValueError("v4 ordered detector/APT row validity is invalid")
        stable += typed_calibration_identity_field(
            "eligible", "bool", "true" if eligible else "false"
        )
        stable += typed_calibration_identity_field(
            "validity_basis", "string", validity_basis
        )
        if row.get("stable_association") != stable:
            raise ValueError("v4 ordered-row stable association does not recompute")
        preimage += typed_calibration_identity_field(
            "ordered_row", "typed_row_association", stable
        )
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()
    if stable_joins.get("ordered_row_association_sha256") != digest:
        raise ValueError("v4 ordered-row association identity does not recompute")
    return digest


def recompute_raw_acquisition_binding(
    raw_acquisition: Any, stable_joins: dict[str, Any],
    selected_apt_sha256: str, row_digest: str,
) -> tuple[str, str]:
    if not isinstance(raw_acquisition, dict):
        raise ValueError("v4 raw-acquisition lineage is not a mapping")
    artifacts = raw_acquisition.get("artifacts")
    rows = stable_joins.get("ordered_detector_apt_rows")
    if not isinstance(artifacts, list) or not artifacts or not isinstance(rows, list):
        raise ValueError("v4 raw-acquisition payload is incomplete")
    raw_identity = "raw-observation-acquisition-identity-v2"
    tone_count = 0
    networks: list[int] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("v4 raw acquisition artifact is invalid")
        path = artifact.get("path")
        digest = artifact.get("sha256")
        interface = artifact.get("interface")
        network = artifact.get("roach_index")
        tones = artifact.get("absolute_tone_frequency_hz")
        if not isinstance(path, str) or not path or not canonical_sha256(digest) or \
                not isinstance(interface, str) or not interface or \
                isinstance(network, bool) or not isinstance(network, int) or \
                network < 0 or not isinstance(tones, list) or not tones:
            raise ValueError("v4 raw acquisition artifact is incomplete")
        if interface != f"toltec{network}":
            raise ValueError("v4 raw acquisition interface/network identity conflicts")
        artifact_identity = "raw-artifact-v1"
        for name, kind, value in (
            ("path", "string", path),
            ("sha256", "sha256", digest),
            ("interface", "string", interface),
            ("network", "int", str(network)),
        ):
            artifact_identity += typed_calibration_identity_field(name, kind, value)
        for tone in tones:
            if isinstance(tone, bool) or not isinstance(tone, (int, float)) or \
                    not math.isfinite(float(tone)):
                raise ValueError("v4 raw acquisition tone frequency is invalid")
            artifact_identity += typed_calibration_identity_field(
                "absolute_tone_frequency_hz", "float64", format(float(tone), ".17g")
            )
            tone_count += 1
        raw_identity += typed_calibration_identity_field(
            "artifact", "typed_raw_artifact", artifact_identity
        )
        networks.append(network)
    if networks != sorted(set(networks)) or tone_count != len(rows):
        raise ValueError("v4 raw acquisition artifact cardinality/order is invalid")
    if raw_acquisition.get("raw_observation_identity") != raw_identity:
        raise ValueError("v4 raw observation identity does not recompute")
    binding = "apt-acquisition-binding-v2"
    binding += typed_calibration_identity_field(
        "apt_sha256", "sha256", selected_apt_sha256
    )
    binding += typed_calibration_identity_field(
        "raw_identity", "typed_raw_identity", raw_identity
    )
    binding += typed_calibration_identity_field(
        "selected_row_association_sha256", "sha256", row_digest
    )
    for row in rows:
        join = "apt-raw-ordered-join-v1"
        for name, kind, value in (
            ("network", "int", str(row.get("raw_network"))),
            ("network_local_tone", "index", str(row.get("raw_network_local_tone"))),
            ("absolute_tone_frequency_hz", "float64", format(
                float(row.get("absolute_tone_frequency_hz")), ".17g"
            )),
            ("uid", "int64", str(row.get("uid"))),
        ):
            join += typed_calibration_identity_field(name, kind, value)
        binding += typed_calibration_identity_field(
            "ordered_join", "typed_join", join
        )
    digest = hashlib.sha256(binding.encode("utf-8")).hexdigest()
    if raw_acquisition.get("binding_sha256") != digest:
        raise ValueError("v4 raw-acquisition binding identity does not recompute")
    return digest, raw_identity


def recompute_tolapt_manifest_association(
    selected: dict[str, Any], selected_apt_sha256: str,
) -> str:
    manifest = selected.get("tolapt_manifest")
    if not isinstance(manifest, dict) or type(manifest.get("available")) is not bool:
        raise ValueError("v4 TolAPT-manifest lineage is invalid")
    if not manifest["available"]:
        if set(manifest) != {"available"}:
            raise ValueError("unavailable TolAPT-manifest lineage is not empty")
        return ""
    value = manifest.get("value")
    if not isinstance(value, dict):
        raise ValueError("available TolAPT-manifest lineage is incomplete")
    association = "tolapt-selected-output-association-v2"
    for name, kind, item in (
        ("manifest_sha256", "sha256", value.get("sha256")),
        ("contract_version", "string", value.get("contract_version")),
        ("run_id", "string", value.get("run_id")),
    ):
        if not isinstance(item, str) or not item:
            raise ValueError("available TolAPT-manifest lineage is incomplete")
        association += typed_calibration_identity_field(name, kind, item)
    inputs = value.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("available TolAPT-manifest input lineage is incomplete")
    for key in ("design_apt", "measured_apt"):
        record = inputs.get(key)
        if not isinstance(record, dict):
            raise ValueError("available TolAPT-manifest input lineage is incomplete")
        prefix = f"inputs.{key}"
        for suffix, kind in (
            ("path", "string"), ("sha256", "sha256"),
            ("bytes", "uint64"), ("mtime_utc", "utc_timestamp"),
        ):
            item = record.get(suffix)
            if suffix == "bytes":
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    raise ValueError("available TolAPT-manifest input lineage is invalid")
                item = str(item)
            if not isinstance(item, str) or not item:
                raise ValueError("available TolAPT-manifest input lineage is incomplete")
            association += typed_calibration_identity_field(
                f"{prefix}.{suffix}", kind, item
            )
    for name, kind, item in (
        ("output_key", "string", value.get("selected_output_key")),
        ("output_path", "run_relative_path", value.get("selected_output_path")),
        ("selected_output_sha256", "sha256", selected_apt_sha256),
    ):
        if not isinstance(item, str) or not item:
            raise ValueError("available TolAPT-manifest output lineage is incomplete")
        association += typed_calibration_identity_field(name, kind, item)
    digest = hashlib.sha256(association.encode("utf-8")).hexdigest()
    if value.get("association_sha256") != digest:
        raise ValueError("v4 TolAPT-manifest association identity does not recompute")
    return digest


def recompute_admitted_calibration_identity(
    value: dict[str, Any], selected_digest: str, row_digest: str,
    acquisition_digest: str, factor_digest: str, manifest_digest: str,
) -> str:
    selected = value["selected_apt"]
    acquisition = value["raw_acquisition"]
    factors = value["factor_operator_state"]
    response = value.get("response_basis")
    if not isinstance(response, dict) or not isinstance(response.get("provenance"), str) or \
            not response["provenance"]:
        raise ValueError("v4 response-basis provenance is unavailable")
    alpha = factors.get("reference_spectral_index_alpha")
    default_applied = factors.get("reference_spectral_index_default_applied")
    tau225 = factors.get("tau225")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in (alpha, tau225)
    ) or type(default_applied) is not bool:
        raise ValueError("v4 reference spectral-index/tau state is invalid")
    reference_state = (
        f"{cxx_hexfloat(float(alpha))}"
        f";default={'true' if default_applied else 'false'}"
        f";tau225={cxx_hexfloat(float(tau225))}"
    )
    preimage = "sci-cal-001-canonical-calibration-identity-v1"
    fields = (
        ("selected_apt_source_path", selected.get("source_path")),
        ("selected_apt_sha256", selected_digest),
        ("apt_row_association_sha256", row_digest),
        ("apt_observation_identity", selected.get("observation_identity")),
        ("apt_matched_observation_identity", selected.get("matched_observation_identity")),
        ("apt_selected_source", selected.get("selected_source")),
        ("tolapt_manifest_association_sha256", manifest_digest),
        ("acquisition_binding_sha256", acquisition_digest),
        ("raw_observation_identity", acquisition.get("raw_observation_identity")),
        ("target_unit", factors.get("target_unit")),
        ("factor_composition", factors.get("factor_composition")),
        ("factor_provenance", factors.get("factor_provenance")),
        ("factor_state_sha256", factor_digest),
        ("atmosphere_operator_id", factors.get("atmosphere_operator_id")),
        ("atmosphere_operator_contract_sha256", factors.get("atmosphere_operator_contract_sha256")),
        ("atmosphere_node_table_sha256", factors.get("atmosphere_node_table_sha256")),
        ("passband_set_id", factors.get("passband_set_id")),
        ("reference_profile_id", factors.get("reference_profile_id")),
        ("reference_and_tau_state", reference_state),
        ("response_basis_provenance", response["provenance"]),
        ("validity", "valid_complete_product"),
    )
    for name, item in fields:
        if not isinstance(item, str):
            raise ValueError(f"v4 calibration identity field {name} is invalid")
        preimage += calibration_identity_field(name, item)
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()
    if value.get("calibration_identity") != digest:
        raise ValueError("v4 calibration identity does not recompute")
    return digest


def raw_provenance_semantic_errors(
    data: dict[str, Any], provenance_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("raw execution plan is not initialized")

        if data.get("schema_version") in {
            "citlali-raw-timestream-provenance-v2",
            "citlali-raw-timestream-provenance-v3",
            "citlali-raw-timestream-provenance-v4",
        }:
            requested_offsets = data["requested"]["interface_sync_offset"]
            effective_offsets = data["effective"]["config"][
                "interface_sync_offset"
            ]
            expected_keys = {
                *(f"toltec{index}" for index in range(13)),
                "hwpr",
            }
            for label, record in (
                ("requested", requested_offsets),
                ("effective", effective_offsets),
            ):
                if record.get("unit") != "s":
                    errors.append(f"{label} interface-sync unit is not seconds")
                offsets = record.get("offsets")
                if not isinstance(offsets, dict) or set(offsets) != expected_keys:
                    errors.append(
                        f"{label} interface-sync offsets are incomplete"
                    )
                    continue
                for key, value in offsets.items():
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        errors.append(
                            f"{label} interface-sync offset {key} is not finite"
                        )
            if requested_offsets != effective_offsets:
                errors.append("interface-sync requested/effective values differ")

        if data.get("schema_version") in {
            "citlali-raw-timestream-provenance-v3",
            "citlali-raw-timestream-provenance-v4",
        }:
            expected_identity = {
                "atmosphere_operator_id":
                    "am12_fixed_djf25_piecewise_linear_los_tau_v1",
                "atmosphere_operator_contract_sha256":
                    "7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a",
                "atmosphere_node_table_sha256":
                    "fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f",
                "passband_set_id":
                    "toltec-passband-set-v1:sha256:5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433",
                "reference_profile_id":
                    "LMT_DJF_25.amc:sha256:aeeeeb48bef422f2d9392b5d7a3d62ab1887fd9e7c10322d5246d914841ba866",
            }
            requested_calibration = data["requested"]["calibration"]
            effective_calibration = data["effective"]["config"]["calibration"]
            requested_alpha = requested_calibration[
                "reference_spectral_index_alpha"
            ]
            effective_alpha = effective_calibration[
                "reference_spectral_index_alpha"
            ]
            default_applied = effective_calibration[
                "reference_spectral_index_default_applied"
            ]
            if effective_alpha not in {-1, 0, 2, 4}:
                errors.append("effective calibration alpha is unsupported")
            if type(default_applied) is not bool:
                errors.append("calibration alpha default status is not boolean")
            if requested_alpha.get("available") is True:
                if requested_alpha.get("value") != effective_alpha:
                    errors.append("requested/effective calibration alpha differs")
                if default_applied is not False:
                    errors.append("explicit calibration alpha is marked defaulted")
            else:
                if effective_alpha != 0 or default_applied is not True:
                    errors.append("omitted calibration alpha did not default to zero")

            observation_calibration = data["observation"]["value"]
            realized_calibration = data["realized"]
            for name in (
                "tau225",
                "reference_spectral_index_alpha",
                "reference_spectral_index_default_applied",
                *expected_identity,
                "calibration_quality_regime",
                "calibration_valid",
                "calibration_validity_reason",
            ):
                for label, section in (
                    ("observation", observation_calibration),
                    ("realized", realized_calibration),
                ):
                    record = section.get(name)
                    if not isinstance(record, dict) or record.get("available") is not True:
                        errors.append(f"{label} calibration {name} is unavailable")

            def calibration_value(section: dict[str, Any], name: str) -> Any:
                record = section.get(name)
                return record.get("value") if isinstance(record, dict) else None

            for name, expected in expected_identity.items():
                for label, section in (
                    ("observation", observation_calibration),
                    ("realized", realized_calibration),
                ):
                    if calibration_value(section, name) != expected:
                        errors.append(f"{label} calibration {name} is not approved")

            for name in (
                "tau225",
                "reference_spectral_index_alpha",
                "reference_spectral_index_default_applied",
                *expected_identity,
                "calibration_quality_regime",
                "calibration_valid",
                "calibration_validity_reason",
            ):
                if calibration_value(observation_calibration, name) != calibration_value(
                    realized_calibration, name
                ):
                    errors.append(f"observation/realized calibration {name} differs")

            realized_alpha = calibration_value(
                realized_calibration, "reference_spectral_index_alpha"
            )
            realized_default = calibration_value(
                realized_calibration, "reference_spectral_index_default_applied"
            )
            if realized_alpha != effective_alpha or realized_default != default_applied:
                errors.append("effective/realized calibration alpha differs")
            tau225 = calibration_value(realized_calibration, "tau225")
            if (
                isinstance(tau225, bool)
                or not isinstance(tau225, (int, float))
                or not math.isfinite(float(tau225))
                or not 0.0 <= float(tau225) <= 0.25
            ):
                errors.append("realized calibration tau225 is outside support")
            elif calibration_value(realized_calibration, "calibration_valid") is True:
                expected_regime = (
                    "science_qualification_regime"
                    if float(tau225) <= 0.15
                    else "engineering_availability_regime"
                )
                if calibration_value(
                    realized_calibration, "calibration_quality_regime"
                ) != expected_regime:
                    errors.append("calibration quality regime is inconsistent")

        if data.get("schema_version") == "citlali-raw-timestream-provenance-v4":
            flux_calibration = data["effective"]["config"].get(
                "flux_calibration", {}
            ).get("enabled")
            if type(flux_calibration) is not bool:
                errors.append(
                    "v4 effective flux-calibration state is not boolean"
                )
            lineage = data.get("calibration_lineage")
            lineage_available = bool(
                isinstance(lineage, dict)
                and lineage.get("available") is True
            )
            sibling_path = (
                provenance_path.parent / SELECTED_CALIBRATION_APT_FILENAME
                if provenance_path is not None
                else None
            )
            if flux_calibration is False:
                if lineage_available:
                    errors.append(
                        "uncalibrated v4 unexpectedly publishes calibration lineage"
                    )
                elif not isinstance(lineage, dict) or lineage != {
                    "available": False
                }:
                    errors.append(
                        "uncalibrated v4 unavailable calibration lineage is partial"
                    )
                if sibling_path is not None and sibling_path.exists():
                    errors.append(
                        "uncalibrated v4 unexpectedly publishes a selected APT member"
                    )
            elif not lineage_available:
                errors.append("v4 canonical calibration lineage is unavailable")
            else:
                value = lineage.get("value")
                if not isinstance(value, dict):
                    errors.append("v4 canonical calibration lineage is not a mapping")
                else:
                    if value.get("schema_version") != \
                            CANONICAL_CALIBRATION_LINEAGE_SCHEMA:
                        errors.append(
                            "v4 canonical calibration lineage schema is invalid"
                        )
                    calibration_identity = value.get("calibration_identity")
                    package_identity = value.get("package_identity")
                    for label, identity in (
                        ("calibration", calibration_identity),
                        ("package", package_identity),
                    ):
                        if not canonical_sha256(identity):
                            errors.append(f"v4 {label} identity is not canonical sha256")

                    components = value.get("component_identities")
                    if not isinstance(components, dict) or \
                            set(components) != CANONICAL_CALIBRATION_COMPONENTS:
                        errors.append(
                            "v4 canonical calibration components are incomplete"
                        )
                        components = components if isinstance(components, dict) else {}
                    for name in CANONICAL_CALIBRATION_COMPONENTS - {
                        "tolapt_manifest_association_sha256"
                    }:
                        if not canonical_sha256(components.get(name)):
                            errors.append(
                                f"v4 calibration component {name} is not canonical sha256"
                            )

                    selected = value.get("selected_apt")
                    if not isinstance(selected, dict):
                        errors.append("v4 selected APT lineage is not a mapping")
                        selected = {}
                    if selected.get("package_local_path") != \
                            SELECTED_CALIBRATION_APT_FILENAME:
                        errors.append("v4 selected APT package-local path is not canonical")
                    if selected.get("copy_semantics") != \
                            "exact_byte_copy_digest_verified_required_output":
                        errors.append("v4 selected APT copy semantics are not required/verified")
                    selected_source_digest = selected.get("source_sha256")
                    selected_package_digest = selected.get("package_local_sha256")
                    selected_component_digest = components.get(
                        "selected_apt_sha256"
                    )
                    if not canonical_sha256(selected_source_digest):
                        errors.append("v4 selected APT source digest is invalid")
                    if not canonical_sha256(selected_package_digest):
                        errors.append("v4 selected APT package digest is invalid")
                    if not (
                        selected_source_digest
                        == selected_package_digest
                        == selected_component_digest
                    ):
                        errors.append(
                            "v4 selected APT source/package/component digests differ"
                        )
                    if sibling_path is None:
                        errors.append(
                            "v4 calibrated provenance path is unavailable for sibling validation"
                        )
                    elif not sibling_path.is_file():
                        errors.append("v4 selected APT sibling member is missing")
                    elif selected_package_digest != sha256_file(sibling_path):
                        errors.append("v4 selected APT sibling digest differs")

                    stable_joins = value.get("stable_joins")
                    factor_state = value.get("factor_operator_state")
                    raw_acquisition = value.get("raw_acquisition")
                    try:
                        validate_requested_response_preimage(
                            data, value.get("response_basis")
                        )
                        row_digest = recompute_ordered_row_association(
                            stable_joins, selected_package_digest
                        )
                        if row_digest != components.get(
                            "selected_apt_row_association_sha256"
                        ):
                            raise ValueError(
                                "v4 selected-row component digest differs"
                            )
                        acquisition_digest, _ = \
                            recompute_raw_acquisition_binding(
                                raw_acquisition, stable_joins,
                                selected_package_digest, row_digest,
                            )
                        if acquisition_digest != components.get(
                            "raw_acquisition_binding_sha256"
                        ):
                            raise ValueError(
                                "v4 raw-acquisition component digest differs"
                            )
                        factor_digest = recompute_admitted_factor_state(
                            factor_state
                        )
                        if sibling_path is not None and sibling_path.is_file():
                            validate_selected_apt_factor_binding(
                                sibling_path, stable_joins, factor_state
                            )
                        if factor_digest != components.get(
                            "admitted_factor_state_sha256"
                        ):
                            raise ValueError(
                                "v4 factor-state component digest differs"
                            )
                        manifest_digest = \
                            recompute_tolapt_manifest_association(
                                selected, selected_package_digest
                            )
                        if manifest_digest != components.get(
                            "tolapt_manifest_association_sha256"
                        ):
                            raise ValueError(
                                "v4 TolAPT-manifest component digest differs"
                            )
                        calibration_identity = \
                            recompute_admitted_calibration_identity(
                                value, selected_package_digest, row_digest,
                                acquisition_digest, factor_digest,
                                manifest_digest,
                            )
                        expected_package = \
                            recompute_calibration_package_identity(
                                calibration_identity,
                                selected_package_digest,
                                acquisition_digest,
                            )
                        if package_identity != expected_package:
                            raise ValueError(
                                "v4 package identity does not recompute"
                            )
                        validate_v4_calibration_state_joins(
                            data, value, provenance_path
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        errors.append(str(error))
                    for section_name in ("observation", "realized"):
                        section = data[section_name]
                        if section_name == "observation":
                            section = section.get("value", {})
                        for field, expected in (
                            ("calibration_identity", calibration_identity),
                            ("calibration_package_identity", package_identity),
                        ):
                            record = section.get(field)
                            if not isinstance(record, dict) or \
                                    record.get("available") is not True or \
                                    record.get("value") != expected:
                                errors.append(
                                    f"v4 {section_name} {field} does not join canonical lineage"
                                )

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
        if requested_enabled and requested["n_noise_maps"] <= 0:
            errors.append("enabled noise requested count must be positive")
        if effective["enabled"] and effective["n_noise_maps"] <= 0:
            errors.append("enabled noise effective count must be positive")
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
            "actual_completion_valid",
            "completed_count_matches_effective",
            "uncertainty_use_valid",
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
        expected = data.get("expected")
        if not isinstance(expected, dict) or expected.get("initialized") is not True:
            errors.append("noise expected counts are not initialized")
            return errors
        expected_counts: dict[str, int] = {}
        for name in count_names:
            value = expected.get(name)
            if type(value) is not int or value < 0:
                errors.append(
                    f"noise expected {name} must be a nonnegative integer"
                )
            else:
                expected_counts[name] = value
        realized_counts: dict[str, int] = {}
        for name in count_names:
            try:
                realized_counts[name] = available_count(realized[name], name)
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            return errors

        if expected_counts["noise_maps_per_scientific_map"] != (
            effective["n_noise_maps"]
        ):
            errors.append("noise expected per-map count differs from effective config")
        if expected_counts["observation_noise_realization_count"] != (
            expected_counts["observation_scientific_map_count"]
            * expected_counts["noise_maps_per_scientific_map"]
        ):
            errors.append("noise expected observation realization count is inconsistent")
        if expected_counts["coadd_noise_realization_count"] != (
            expected_counts["coadd_scientific_map_count"]
            * expected_counts["noise_maps_per_scientific_map"]
        ):
            errors.append("noise expected coadd realization count is inconsistent")
        if expected_counts["total_noise_realization_count"] != (
            expected_counts["observation_noise_realization_count"]
            + expected_counts["coadd_noise_realization_count"]
        ):
            errors.append("noise expected total realization count is inconsistent")

        if not effective["enabled"]:
            if realized["generation_executed"]:
                errors.append("disabled noise-products records generation")
            for name in count_names:
                if expected_counts[name] != 0:
                    errors.append(
                        f"disabled noise-products expects nonzero {name}"
                    )
                if realized_counts[name] != 0:
                    errors.append(
                        f"disabled noise-products records nonzero {name}"
                    )
            if realized["actual_completion_valid"] is not True:
                errors.append("disabled noise-products completion is not valid")
            if realized["completed_count_matches_effective"] is not True:
                errors.append("disabled noise-products count does not match effective zero")
            if realized["uncertainty_use_valid"] is not False:
                errors.append("disabled noise-products authorizes uncertainty use")
            if realized.get("completion_basis") != "effective_disabled_zero_work":
                errors.append("disabled noise-products completion basis is inconsistent")
            if realized["outputs_completed"] is not True:
                errors.append("disabled noise-products zero-work completion is incomplete")
            return errors

        for name in count_names:
            if realized_counts[name] != expected_counts[name]:
                errors.append(
                    f"noise observed {name} differs from plan-derived expected count"
                )
        if realized["generation_executed"] is not True:
            errors.append("enabled noise generation was not observed")

        active_product_maps = expected_counts[
            "observation_scientific_map_count"
        ]
        product_count = expected_counts["empirical_product_map_count"]
        if not effective["products"]["enabled"]:
            if product_count != 0:
                errors.append("disabled empirical products have nonzero count")
        elif active_product_maps == 0 and product_count != 0:
            errors.append(
                "empirical products exist without observation maps"
            )
        elif active_product_maps and not (
            active_product_maps <= product_count <= 2 * active_product_maps
        ):
            errors.append("empirical product count has invalid output-stage cardinality")

        output_realizations = expected_counts["total_noise_realization_count"]
        write_count = expected_counts["realization_image_write_count"]
        if not effective["write_realizations"]:
            if write_count != 0:
                errors.append("disabled realization outputs have nonzero count")
        elif output_realizations == 0 and write_count != 0:
            errors.append("realization outputs exist without realizations")
        elif output_realizations and not (
            output_realizations <= write_count <= 2 * output_realizations
        ):
            errors.append("realization write count has invalid output-stage cardinality")
        if realized["actual_completion_valid"] is not True:
            errors.append("enabled noise-products completion is not valid")
        if realized["completed_count_matches_effective"] is not True:
            errors.append("enabled noise-products observed counts do not match expected")
        if realized["uncertainty_use_valid"] != (
            effective["n_noise_maps"] >= 2
        ):
            errors.append("noise uncertainty-use validity is inconsistent")
        if realized.get("completion_basis") != (
            "observed_successful_publication_lifecycle"
        ):
            errors.append("enabled noise-products completion basis is inconsistent")
        if realized["outputs_completed"] is not True:
            errors.append("enabled noise-products outputs are incomplete")
    except (AttributeError, KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate noise-products provenance semantics: {exc}")
    return errors


NOISE_PRODUCT_CONTRACTS = {
    "conditional_finite_stack_scatter": (
        "map_pixel",
        "conditional_completed_stack_descriptive_not_physical_noise_variance_or_covariance",
    ),
    "formal_nonprecision_coefficient_snapshot": (
        "map_pixel",
        "pre_scale_nonprecision_coefficient_not_inverse_variance_or_precision",
    ),
    "global_nonprecision_scaled_coefficient": (
        "map_pixel",
        "existing_use_only_nonprecision_not_inverse_variance_or_precision",
    ),
    "coefficient_standardized_signal": (
        "map_pixel", "engineering_standardization_not_significance",
    ),
    "filtered_pixel_stack_scatter": (
        "filtered_map_pixel",
        "conditional_diagnostic_strict_operator_edge_parity_pending_FLT",
    ),
    "conditional_stack_scatter_ratio": (
        "filtered_map_pixel",
        "descriptive_ratio_not_significance_positive_finite_denominator_required",
    ),
    "source_imprinted_current_realization": (
        "realization_map", "source_imprinted_current_conditional_design_member",
    ),
    "pooled_stack_scale_diagnostic": (
        "map_summary", "engineering_scale_diagnostic_not_significance",
    ),
    "global_nonprecision_scale_diagnostic": (
        "map_summary",
        "engineering_scale_diagnostic_not_precision_or_significance",
    ),
    "source_finder_engineering_score": (
        "source_finder",
        "existing_quicklook_engineering_score_not_significance",
    ),
    "fitted_amplitude_over_full_map_rms_ratio": (
        "source_table", "fitted_amplitude_over_full_map_rms_not_significance",
    ),
    "fixed_projection_stack_scatter": (
        "fixed_linear_projection",
        "conditional_finite_stack_diagnostic_not_aperture_uncertainty",
    ),
}
NOISE_PRODUCT_IDENTITIES = frozenset(NOISE_PRODUCT_CONTRACTS)
NOISE_SEMANTIC_DIGEST_KIND = "semantic_contract_sha256"
NOISE_COMPACT_MISSINGNESS = "nonfinite_unavailable"
NOISE_NETCDF_JOIN_SCHEMA = "citlali_noise_product_join_v1"
NOISE_MEMBER_INVENTORY_DIGEST_KIND = "sha256"
NOISE_MEMBER_INVENTORY_PREIMAGE_ENCODING = (
    "canonical_length_prefixed_member_records_v2"
)


def noise_product_semantic_digest(product_identity: str) -> str:
    canonical = (
        "citlali-noise-products|SCI-NOI-002-v1|" + product_identity
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def noise_realization_scope_is_canonical(scope: str) -> bool:
    prefix = "realization_map_index_"
    if not scope.startswith(prefix):
        return False
    ordinal = scope[len(prefix):]
    return bool(ordinal) and ordinal.isascii() and ordinal.isdigit() and (
        ordinal == "0" or not ordinal.startswith("0")
    )


def noise_fits_join_matches_contract(values: dict[str, Any]) -> bool:
    identity = str(values["NOIPRID"])
    scope = str(values["NOISCOPE"])
    validity = str(values["NOIVALID"])
    restriction = str(values["NOIRESTR"])
    if values["NOIMISS"] != NOISE_COMPACT_MISSINGNESS:
        return False
    map_scope = scope in {"raw_map_pixel", "filtered_map_pixel"}
    if identity == "formal_nonprecision_coefficient_snapshot":
        return (
            map_scope
            and validity == "available"
            and restriction == "nonprecision_snapshot_not_inverse_variance"
        )
    if identity == "global_nonprecision_scaled_coefficient":
        return (
            map_scope
            and validity in {"available", "unavailable"}
            and restriction == "existing_use_only_nonprecision_not_precision"
        )
    if identity == "conditional_finite_stack_scatter":
        if validity not in {"conditional_descriptive", "unavailable"}:
            return False
        return (
            scope == "raw_map_pixel"
            and restriction == (
                "retained_legacy_name_not_physical_noise_variance_or_covariance"
            )
        ) or (
            scope == "filtered_map_pixel"
            and restriction == (
                "retained_legacy_name_not_physical_noise_variance_"
                "strict_parity_pending_FLT"
            )
        )
    if identity == "coefficient_standardized_signal":
        return (
            map_scope
            and validity in {"available_where_finite", "unavailable"}
            and restriction == (
                "retained_legacy_name_engineering_standardization_"
                "not_significance"
            )
        )
    if identity == "filtered_pixel_stack_scatter":
        return (
            scope == "filtered_map_pixel"
            and validity in {
                "available_where_finite_on_valid_support",
                "R_lt_2",
                "scatter_unavailable_or_nonfinite",
                "response_invalid",
                "support_invalid",
            }
            and restriction == (
                "retained_legacy_name_not_aperture_uncertainty_"
                "strict_parity_pending_FLT"
            )
        )
    if identity == "conditional_stack_scatter_ratio":
        return (
            scope == "filtered_map_pixel"
            and validity in {
                "available_where_finite_positive_denominator_on_valid_support",
                "R_lt_2",
                "scatter_unavailable_or_nonfinite",
                "response_invalid",
                "support_invalid",
            }
            and restriction == (
                "retained_legacy_name_conditional_descriptive_ratio_"
                "not_significance"
            )
        )
    if identity == "source_imprinted_current_realization":
        return (
            noise_realization_scope_is_canonical(scope)
            and validity == "conditional_design_member"
            and restriction == (
                "source_imprinted_current_not_physical_noise_repeat"
            )
        )
    return False


def noise_fits_identity_is_empirical_map_product(identity: str) -> bool:
    return identity in {
        "conditional_finite_stack_scatter",
        "formal_nonprecision_coefficient_snapshot",
        "global_nonprecision_scaled_coefficient",
        "coefficient_standardized_signal",
        "filtered_pixel_stack_scatter",
        "conditional_stack_scatter_ratio",
    }


def noise_fits_logical_map_identity(
    identity: str, extname: str, scope: str
) -> str:
    extname = extname.lower()
    prefixes = {
        "formal_nonprecision_coefficient_snapshot": "weight_formal_",
        "global_nonprecision_scaled_coefficient": "weight_",
        "conditional_finite_stack_scatter": "noise_variance_",
        "coefficient_standardized_signal": "sig2noise_",
        "filtered_pixel_stack_scatter": "point_source_uncertainty_",
        "conditional_stack_scatter_ratio": "sig2noise_point_source_",
        "source_imprinted_current_realization": "signal_",
    }
    prefix = prefixes.get(identity)
    if prefix is None:
        raise ValueError("FITS noise-product identity has no EXTNAME binding")
    if not extname.startswith(prefix) or len(extname) == len(prefix):
        raise ValueError(
            "FITS noise-product EXTNAME does not match its identity"
        )
    encoded_map = extname[len(prefix):]
    if identity != "source_imprinted_current_realization":
        return encoded_map

    ordinal = scope[len("realization_map_index_"):]
    leading = ordinal + "_"
    if encoded_map.startswith(leading) and len(encoded_map) > len(leading):
        return encoded_map[len(leading):]
    marker = "_" + ordinal + "_"
    marker_pos = encoded_map.rfind(marker)
    if marker_pos <= 0 or marker_pos + len(marker) >= len(encoded_map):
        raise ValueError(
            "FITS noise-realization EXTNAME does not encode its scope"
        )
    return encoded_map[:marker_pos] + "_" + encoded_map[
        marker_pos + len(marker):
    ]


def fits_noise_member_joins(
    path: Path,
) -> tuple[list[str], int, int, bool]:
    from astropy.io import fits

    identity_counts: Counter[str] = Counter()
    logical_maps: dict[str, dict[str, Any]] = {}
    realization_count = 0
    empirical_map_count = 0
    keys = (
        "NOIPKG", "NOIPROV", "NOIPRID", "NOIPVER", "NOIDGST",
        "NOIDGKND", "NOISCOPE", "NOIVALID", "NOIRESTR", "NOIMISS",
    )
    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus:
            values = {key: hdu.header.get(key) for key in keys}
            if not any(value is not None for value in values.values()):
                continue
            if any(value is None for value in values.values()):
                raise ValueError(f"partial FITS noise-product join in {path}")
            identity = str(values["NOIPRID"])
            if (
                values["NOIPKG"] != "citlali-noise-products"
                or values["NOIPROV"] != "noise_products_provenance.yaml"
                or values["NOIPVER"] != "SCI-NOI-002-v1"
                or values["NOIDGST"] != noise_product_semantic_digest(identity)
                or values["NOIDGKND"] != NOISE_SEMANTIC_DIGEST_KIND
                or identity not in NOISE_PRODUCT_IDENTITIES
                or not noise_fits_join_matches_contract(values)
            ):
                raise ValueError(f"invalid FITS noise-product join in {path}")
            extname = hdu.header.get("EXTNAME")
            if not isinstance(extname, str) or not extname:
                raise ValueError(
                    f"FITS noise-product join lacks an EXTNAME in {path}"
                )
            logical_map_identity = noise_fits_logical_map_identity(
                identity, extname, str(values["NOISCOPE"])
            )
            logical_map = logical_maps.setdefault(
                logical_map_identity,
                {
                    "identity_counts": Counter(),
                    "empirical_scope": None,
                    "realization_scopes": set(),
                },
            )
            identity_counts[identity] += 1
            logical_map["identity_counts"][identity] += 1
            if (
                identity != "source_imprinted_current_realization"
                and logical_map["identity_counts"][identity] > 1
            ):
                raise ValueError(
                    "duplicate non-realization FITS noise-product identity "
                    f"{identity} within logical map {logical_map_identity} "
                    f"in {path}"
                )
            if noise_fits_identity_is_empirical_map_product(identity):
                scope = str(values["NOISCOPE"])
                if logical_map["empirical_scope"] is None:
                    logical_map["empirical_scope"] = scope
                elif logical_map["empirical_scope"] != scope:
                    raise ValueError(
                        "mixed empirical FITS noise-product scopes within "
                        f"logical map {logical_map_identity} in {path}"
                    )
            if identity == "source_imprinted_current_realization":
                scope = str(values["NOISCOPE"])
                realization_scopes = logical_map["realization_scopes"]
                if scope in realization_scopes:
                    raise ValueError(
                        "duplicate FITS noise-realization scope "
                        f"{scope} within logical map {logical_map_identity} "
                        f"in {path}"
                    )
                realization_scopes.add(scope)
                realization_count += 1
    if not identity_counts:
        raise ValueError(f"admitted FITS member has no noise-product join: {path}")
    for logical_map_identity, logical_map in logical_maps.items():
        logical_counts = logical_map["identity_counts"]
        realization_scopes = logical_map["realization_scopes"]
        if realization_scopes != {
            f"realization_map_index_{ordinal}"
            for ordinal in range(len(realization_scopes))
        }:
            raise ValueError(
                "noncanonical FITS noise-realization scope sequence within "
                f"logical map {logical_map_identity} in {path}"
            )
        empirical_identity_present = any(
            noise_fits_identity_is_empirical_map_product(identity)
            for identity in logical_counts
        )
        if not empirical_identity_present:
            continue
        if realization_scopes:
            raise ValueError(
                "mixed realization and empirical FITS noise products within "
                f"logical map {logical_map_identity} in {path}"
            )
        formal_count = logical_counts[
            "formal_nonprecision_coefficient_snapshot"
        ]
        scatter_count = logical_counts["conditional_finite_stack_scatter"]
        scaled_count = logical_counts["global_nonprecision_scaled_coefficient"]
        standardized_count = logical_counts["coefficient_standardized_signal"]
        filtered_scatter_count = logical_counts["filtered_pixel_stack_scatter"]
        ratio_count = logical_counts["conditional_stack_scatter_ratio"]
        full_empirical_bundle = any(
            (
                formal_count,
                scatter_count,
                standardized_count,
                filtered_scatter_count,
                ratio_count,
            )
        )
        if not full_empirical_bundle:
            if scaled_count != 1:
                raise ValueError(
                    "invalid standalone scaled-coefficient FITS noise product "
                    f"within logical map {logical_map_identity} in {path}"
                )
            continue
        if (
            formal_count != 1
            or scatter_count != 1
            or filtered_scatter_count != ratio_count
        ):
            raise ValueError(
                "incomplete or mixed empirical FITS noise-product bundle "
                f"within logical map {logical_map_identity} in {path}"
            )
        empirical_map_count += 1
    return sorted(identity_counts), realization_count, empirical_map_count, True


def ecsv_noise_member_joins(
    path: Path,
) -> tuple[list[str], int, int, bool]:
    from collections.abc import Mapping
    from astropy.table import Table

    header_lines: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            header_lines.append(line)
    missingness_field_count = len(
        re.findall(
            r"(?<![A-Za-z0-9_])missingness\s*:",
            "".join(header_lines),
        )
    )
    if missingness_field_count != 1:
        raise ValueError(f"invalid ECSV noise-product join in {path}")

    table = Table.read(path, format="ascii.ecsv")
    contract = table.meta.get("noise_product_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"admitted ECSV member has no noise-product join: {path}")
    identity = "fitted_amplitude_over_full_map_rms_ratio"
    expected = {
        "package_id": "citlali-noise-products",
        "provenance_id": "noise_products_provenance.yaml",
        "product_identity": identity,
        "product_version": "SCI-NOI-002-v1",
        "semantic_digest": noise_product_semantic_digest(identity),
        "digest_kind": NOISE_SEMANTIC_DIGEST_KIND,
        "missingness": NOISE_COMPACT_MISSINGNESS,
        "column": "sig2noise",
        "scope": "source_table_row",
        "validity": "finite_amplitude_and_finite_positive_full_map_rms",
        "restriction": "legacy_alias_deprecated_not_significance",
    }
    if (
        set(contract) != set(expected)
        or any(contract.get(key) != value for key, value in expected.items())
        or table.colnames.count(expected["column"]) != 1
    ):
        raise ValueError(f"invalid ECSV noise-product join in {path}")
    return [identity], 0, 0, False


NOISE_NETCDF_JOIN_KEYS = (
    "variable",
    "package_id",
    "provenance_id",
    "product_identity",
    "product_version",
    "semantic_digest",
    "digest_kind",
    "missingness",
    "scope",
    "validity",
    "restriction",
)


def noise_netcdf_join_record(
    variable: str,
    identity: str,
    validity: str,
    restriction: str,
) -> str:
    values = {
        "variable": variable,
        "package_id": "citlali-noise-products",
        "provenance_id": "noise_products_provenance.yaml",
        "product_identity": identity,
        "product_version": "SCI-NOI-002-v1",
        "semantic_digest": noise_product_semantic_digest(identity),
        "digest_kind": NOISE_SEMANTIC_DIGEST_KIND,
        "missingness": NOISE_COMPACT_MISSINGNESS,
        "scope": "map_summary",
        "validity": validity,
        "restriction": restriction,
    }
    return NOISE_NETCDF_JOIN_SCHEMA + "|" + "|".join(
        f"{key}={values[key]}" for key in NOISE_NETCDF_JOIN_KEYS
    )


def parse_noise_netcdf_join_comment(comment: str) -> dict[str, str]:
    description, separator, record = comment.rpartition("; ")
    if not separator or not description or record.count(
        NOISE_NETCDF_JOIN_SCHEMA
    ) != 1:
        raise ValueError("missing or duplicate structured NetCDF noise-product join")
    tokens = record.split("|")
    if (
        len(tokens) != len(NOISE_NETCDF_JOIN_KEYS) + 1
        or tokens[0] != NOISE_NETCDF_JOIN_SCHEMA
    ):
        raise ValueError("invalid structured NetCDF noise-product join shape")
    parsed: dict[str, str] = {}
    for expected_key, token in zip(NOISE_NETCDF_JOIN_KEYS, tokens[1:]):
        if token.count("=") != 1:
            raise ValueError(
                f"invalid structured NetCDF noise-product join field {expected_key}"
            )
        key, value = token.split("=", 1)
        if key != expected_key or not value or key in parsed:
            raise ValueError(
                f"invalid structured NetCDF noise-product join field {expected_key}"
            )
        parsed[key] = value
    return parsed


def netcdf_noise_member_joins(
    path: Path,
) -> tuple[list[str], int, int, bool]:
    from netCDF4 import Dataset

    variables = {
        "map_noise_weight_median_ratio": (
            "global_nonprecision_scale_diagnostic",
            "available_when_finite_positive_calibration_support_exists",
            "engineering_scale_diagnostic_not_precision_or_significance",
        ),
        "map_noise_weight_scale": (
            "global_nonprecision_scale_diagnostic",
            "available_when_finite_positive_median_ratio_exists",
            "nonprecision_scale_not_inverse_variance_or_precision",
        ),
        "map_noise_products_s2n_sigma": (
            "pooled_stack_scale_diagnostic",
            "available_when_finite_pooled_stack_scale_exists",
            "engineering_scale_diagnostic_not_calibrated_significance",
        ),
    }
    identities: set[str] = set()
    with Dataset(path, "r") as dataset:
        for name, (identity, validity, restriction) in variables.items():
            if name not in dataset.variables:
                raise ValueError(
                    f"missing NetCDF noise-contract variable {name} in {path}"
                )
            comment = str(getattr(dataset.variables[name], "comment", ""))
            expected_record = noise_netcdf_join_record(
                name, identity, validity, restriction
            )
            try:
                parse_noise_netcdf_join_comment(comment)
            except ValueError as exc:
                raise ValueError(
                    f"invalid NetCDF noise-product join for {name} in {path}: "
                    f"{exc}"
                ) from exc
            if comment.rpartition("; ")[2] != expected_record:
                raise ValueError(
                    f"invalid NetCDF noise-product join for {name} in {path}"
                )
            identities.add(identity)
        for name, variable in dataset.variables.items():
            if name in variables:
                continue
            comment = str(getattr(variable, "comment", ""))
            if NOISE_NETCDF_JOIN_SCHEMA in comment:
                raise ValueError(
                    f"unexpected NetCDF noise-contract variable {name} in {path}"
                )
    return sorted(identities), 0, 0, True


def noise_member_inventory_preimage_v2(members: list[dict[str, Any]]) -> bytes:
    relative_paths = [member["member_product_identity"] for member in members]
    if relative_paths != sorted(relative_paths, key=lambda value: value.encode("utf-8")):
        raise ValueError(
            "noise package member inventory is not in canonical lexical order"
        )
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("noise package member inventory contains duplicate paths")

    preimage = bytearray(b"citlali-noise-member-inventory-v2|")

    def append_field(value: str) -> None:
        encoded = value.encode("utf-8")
        preimage.extend(str(len(encoded)).encode("ascii"))
        preimage.extend(b":")
        preimage.extend(encoded)

    append_field(str(len(members)))
    for member in members:
        append_field(member["member_product_identity"])
        append_field(member["sha256"])
        append_field(str(member["size_bytes"]))
    return bytes(preimage)


def noise_member_inventory_digest_v2(members: list[dict[str, Any]]) -> str:
    return "sha256:" + hashlib.sha256(
        noise_member_inventory_preimage_v2(members)
    ).hexdigest()


def noise_package_integrity_errors(
    data: dict[str, Any], sidecar_path: Path
) -> list[str]:
    errors: list[str] = []
    try:
        package = data["package"]
        if package.get("package_id") != "citlali-noise-products":
            errors.append("noise package identity is inconsistent")
        if package.get("provenance_id") != sidecar_path.name:
            errors.append("noise package provenance join is inconsistent")
        if package.get("product_contract_version") != "SCI-NOI-002-v1":
            errors.append("noise package product version is inconsistent")
        if package.get("authority") != "package_sidecar":
            errors.append("noise package authority is inconsistent")
        if package.get("detached_product_status") != (
            "unverified_out_of_contract"
        ):
            errors.append("noise package detached-product status is inconsistent")
        if package.get("publication_state") != "complete" or (
            package.get("complete") is not True
        ):
            errors.append("noise package publication is not complete")

        contract_inventory = package.get("product_contract_inventory")
        declared_identities: set[str] = set()
        if not isinstance(contract_inventory, list):
            errors.append("noise package product-contract inventory must be a sequence")
        else:
            for index, contract in enumerate(contract_inventory):
                label = f"noise package product contract {index}"
                if not isinstance(contract, dict):
                    errors.append(f"{label} must be a mapping")
                    continue
                identity = contract.get("product_identity")
                if identity not in NOISE_PRODUCT_IDENTITIES:
                    errors.append(f"{label} identity is unknown")
                    continue
                if identity in declared_identities:
                    errors.append(f"duplicate noise product contract: {identity}")
                    continue
                declared_identities.add(identity)
                if contract.get("product_version") != "SCI-NOI-002-v1":
                    errors.append(f"{label} version is inconsistent")
                if contract.get("semantic_digest") != (
                    noise_product_semantic_digest(identity)
                ):
                    errors.append(f"{label} semantic digest is inconsistent")
                if contract.get("digest_kind") != NOISE_SEMANTIC_DIGEST_KIND:
                    errors.append(f"{label} digest kind is inconsistent")
                expected_scope, expected_restriction = (
                    NOISE_PRODUCT_CONTRACTS[identity]
                )
                if contract.get("scope") != expected_scope:
                    errors.append(f"{label} scope is inconsistent")
                if contract.get("restriction") != expected_restriction:
                    errors.append(f"{label} restriction is inconsistent")
            if declared_identities != NOISE_PRODUCT_IDENTITIES:
                errors.append("noise package product-contract inventory is incomplete")

        members = package.get("member_files")
        if not isinstance(members, list):
            return errors + ["noise package member inventory must be a sequence"]
        if package.get("member_count") != len(members):
            errors.append("noise package member count is inconsistent")

        lexical_root = Path(os.path.abspath(sidecar_path.parent))
        root_status = os.lstat(lexical_root)
        if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(
            root_status.st_mode
        ):
            return errors + [
                "noise package reduction root is non-directory or a symlink"
            ]
        root = lexical_root.resolve(strict=True)
        seen: set[str] = set()
        seen_resolved: set[Path] = set()
        verified_inventory_records: list[dict[str, Any]] = []
        previous_relative: bytes | None = None
        realization_count = 0
        empirical_map_product_count = 0
        contains_stack_derived_product = False
        contains_empirical_netcdf = False
        for index, member in enumerate(members):
            label = f"noise package member {index}"
            if not isinstance(member, dict):
                errors.append(f"{label} must be a mapping")
                continue
            relative = member.get("member_product_identity")
            if not isinstance(relative, str) or not relative:
                errors.append(f"{label} has no relative identity")
                continue
            parts = relative.split("/")
            if (
                relative.startswith("/")
                or "\\" in relative
                or any(part in ("", ".", "..") for part in parts)
                or "/".join(parts) != relative
            ):
                errors.append(f"{label} path is not normalized and relative")
                continue
            if relative in seen:
                errors.append(f"duplicate noise package member: {relative}")
                continue
            seen.add(relative)
            relative_bytes = relative.encode("utf-8")
            if previous_relative is not None and (
                previous_relative >= relative_bytes
            ):
                errors.append(
                    "noise package member inventory is not in canonical "
                    "lexical order"
                )
            previous_relative = relative_bytes

            candidate = lexical_root.joinpath(*parts)
            current = lexical_root
            invalid_component = False
            for component_index, part in enumerate(parts):
                current = current / part
                try:
                    component_status = os.lstat(current)
                except FileNotFoundError:
                    errors.append(f"{label} is missing")
                    invalid_component = True
                    break
                is_leaf = component_index == len(parts) - 1
                if stat.S_ISLNK(component_status.st_mode):
                    errors.append(
                        f"{label} is a symlink"
                        if is_leaf
                        else f"{label} has a symlink path component: {current}"
                    )
                    invalid_component = True
                    break
                if is_leaf and not stat.S_ISREG(component_status.st_mode):
                    errors.append(f"{label} is not a regular file")
                    invalid_component = True
                    break
                if not is_leaf and not stat.S_ISDIR(component_status.st_mode):
                    errors.append(f"{label} has a non-directory path component")
                    invalid_component = True
                    break
            if invalid_component:
                continue
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (FileNotFoundError, RuntimeError, ValueError):
                errors.append(f"{label} is missing or outside the reduction root")
                continue
            if not resolved.is_file():
                errors.append(f"{label} is not a regular file")
                continue
            if resolved in seen_resolved:
                errors.append(f"duplicate resolved noise package member: {relative}")
                continue
            seen_resolved.add(resolved)

            kind = member.get("member_kind")
            extensions = {"fits": ".fits", "ecsv": ".ecsv", "netcdf": ".nc"}
            if kind not in extensions or resolved.suffix != extensions[kind]:
                errors.append(f"{label} kind/extension is inconsistent")
                continue
            digest = sha256_file(resolved)
            size = resolved.stat().st_size
            if member.get("sha256") != digest:
                errors.append(f"{label} SHA-256 is inconsistent")
            if member.get("size_bytes") != size:
                errors.append(f"{label} size is inconsistent")
            if member.get("digest_kind") != "file_sha256":
                errors.append(f"{label} digest kind is inconsistent")
            if member.get("detached_status") != (
                "unverified_out_of_contract_without_package"
            ):
                errors.append(f"{label} detached status is inconsistent")

            if kind == "fits":
                (
                    identities,
                    member_realizations,
                    member_empirical_maps,
                    member_is_stack_derived,
                ) = fits_noise_member_joins(resolved)
            elif kind == "ecsv":
                (
                    identities,
                    member_realizations,
                    member_empirical_maps,
                    member_is_stack_derived,
                ) = ecsv_noise_member_joins(resolved)
            else:
                (
                    identities,
                    member_realizations,
                    member_empirical_maps,
                    member_is_stack_derived,
                ) = netcdf_noise_member_joins(resolved)
            listed_identities = member.get("joined_product_identities")
            if listed_identities != identities:
                errors.append(f"{label} joined product identities are inconsistent")
            if not isinstance(listed_identities, list) or any(
                identity not in declared_identities
                for identity in listed_identities
            ):
                errors.append(
                    f"{label} identity is absent from package contract inventory"
                )
            realization_count += member_realizations
            empirical_map_product_count += member_empirical_maps
            contains_stack_derived_product = (
                contains_stack_derived_product or member_is_stack_derived
            )
            contains_empirical_netcdf = contains_empirical_netcdf or (
                kind == "netcdf"
            )
            verified_inventory_records.append(
                {
                    "member_product_identity": relative,
                    "sha256": digest,
                    "size_bytes": size,
                }
            )

        inventory_digest = noise_member_inventory_digest_v2(
            verified_inventory_records
        )
        if package.get("member_inventory_digest") != inventory_digest:
            errors.append("noise package aggregate inventory digest is inconsistent")
        if package.get("member_inventory_digest_kind") != (
            NOISE_MEMBER_INVENTORY_DIGEST_KIND
        ):
            errors.append("noise package aggregate digest kind is inconsistent")
        if package.get("member_inventory_preimage_encoding") != (
            NOISE_MEMBER_INVENTORY_PREIMAGE_ENCODING
        ):
            errors.append("noise package aggregate preimage encoding is inconsistent")

        effective = data["effective"]["config"]
        observed_empirical_maps = available_count(
            data["realized"]["empirical_product_map_count"],
            "empirical_product_map_count",
        )
        if not effective["enabled"]:
            if contains_stack_derived_product:
                errors.append(
                    "disabled noise package contains a stack-derived member"
                )
        elif not effective["products"]["enabled"]:
            if (
                empirical_map_product_count != 0
                or contains_empirical_netcdf
                or observed_empirical_maps != 0
            ):
                errors.append(
                    "noise package contains empirical members while products "
                    "are disabled"
                )
        else:
            if empirical_map_product_count != observed_empirical_maps:
                errors.append(
                    "noise package empirical FITS inventory does not match "
                    "observed empirical product maps"
                )
        observed_writes = available_count(
            data["realized"]["realization_image_write_count"],
            "realization_image_write_count",
        )
        if realization_count != observed_writes:
            errors.append(
                "noise package realization FITS inventory differs from observed writes"
            )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot evaluate noise package integrity: {exc}")
    except Exception as exc:
        errors.append(f"cannot validate noise package member joins: {exc}")
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
        noise_expected = noise["expected"]
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
        expected_observation_maps = observation_map_count
        if noise_expected["observation_scientific_map_count"] != (
            expected_observation_maps
        ):
            errors.append(
                "noise expected observation map count differs from mapmaking provenance"
            )
        if noise_expected["coadd_scientific_map_count"] != coadd_map_count:
            errors.append(
                "noise expected coadd map count differs from mapmaking provenance"
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


def polarimetry_provenance_semantic_errors(
    data: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("polarimetry execution plan is not initialized")
        capability = data["capability"]
        if capability["status"] != "planned-unavailable":
            errors.append("polarimetry capability status is invalid")
        if capability["enabled_supported"] is not False:
            errors.append("enabled polarimetry must remain unavailable")
        for name in ("reason", "exit_condition"):
            if not isinstance(capability[name], str) or not capability[name]:
                errors.append(f"polarimetry capability {name} is empty")

        requested = data["requested"]
        effective = data["effective"]["config"]
        for label, config in (
            ("requested", requested),
            ("effective", effective),
        ):
            if type(config["enabled"]) is not bool:
                errors.append(f"{label} polarimetry enabled must be boolean")
            if config["grouping"] not in {"fg", "loc"}:
                errors.append(f"{label} polarimetry grouping is invalid")
            if config["ignore_hwpr"] not in {"auto", "true", "false"}:
                errors.append(f"{label} polarimetry ignore_hwpr is invalid")

        resolution = data["effective"]["capability_resolution"]
        for name in (
            "enabled_capability_available",
            "requested_enabled",
            "request_accepted",
            "disabled_by_capability",
        ):
            if type(resolution[name]) is not bool:
                errors.append(
                    f"polarimetry capability resolution {name} must be boolean"
                )
        if requested["enabled"] is not False:
            errors.append("successful run requested enabled polarimetry")
        if effective["enabled"] is not False:
            errors.append("successful run has effective enabled polarimetry")
        if resolution != {
            "enabled_capability_available": False,
            "requested_enabled": False,
            "request_accepted": True,
            "disabled_by_capability": False,
        }:
            errors.append(
                "disabled polarimetry capability resolution is inconsistent"
            )

        realized = data["realized"]
        if realized["reduction_completed"] is not True:
            errors.append("polarimetry reduction is not complete")
        if realized["polarimetry_executed"] is not False:
            errors.append("unavailable polarimetry was executed")
        if realized["hwpr_loaded"] is not False:
            errors.append("unavailable polarimetry loaded HWPR data")
    except (KeyError, TypeError) as exc:
        errors.append(
            f"cannot evaluate polarimetry provenance semantics: {exc}"
        )
    return errors


def valid_astrometry_config(config: Any, label: str) -> list[str]:
    errors: list[str] = []
    try:
        offsets = config["pointing_offsets"]
        if offsets["enabled"] is not True:
            errors.append(f"{label} astrometry pointing offsets are not enabled")
        az = offsets["az_arcsec"]
        alt = offsets["alt_arcsec"]
        mjd = offsets["modified_julian_date"]
        if not isinstance(az, list) or len(az) not in (1, 2):
            errors.append(f"{label} astrometry az offsets must have length one or two")
        if not isinstance(alt, list) or len(alt) != len(az):
            errors.append(f"{label} astrometry alt offsets must match az offsets")
        if not isinstance(mjd, list) or len(mjd) != 2:
            errors.append(f"{label} astrometry MJD support must have length two")
        for name, values in (("az", az), ("alt", alt), ("MJD", mjd)):
            if not isinstance(values, list):
                continue
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                errors.append(f"{label} astrometry {name} values must be finite")
    except (KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate {label} astrometry config: {exc}")
    return errors


def astrometry_provenance_semantic_errors(
    data: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        if data["initialized"] is not True:
            errors.append("astrometry execution plan is not initialized")
        if data["reduction_completed"] is not True:
            errors.append("astrometry reduction is not complete")

        authority = data["authority"]
        if authority["calibration_selection"] != "tolteca":
            errors.append("astrometry calibration-selection authority must be tolteca")
        if authority["application"] != "citlali":
            errors.append("astrometry application authority must be citlali")
        if authority["support_origin_metadata_available"] is not False:
            errors.append("astrometry support-origin availability is inconsistent")
        if authority["configured_values_origin"] != "upstream-unspecified":
            errors.append("astrometry configured-values origin is inconsistent")

        identity = data["identity"]
        expected_identity = {
            "axes": ["az", "alt"],
            "offset_unit": "arcsec",
            "time_support": "modified-julian-date",
            "algorithm": "legacy-citlali-constant-or-linear-v1",
        }
        if identity != expected_identity:
            errors.append("astrometry scientific identity is inconsistent")

        contract = data["contract"]
        expected_contract = {
            "upstream_selection_owner": "tolteca",
            "one_configured_value": "constant",
            "two_values_without_positive_mjd_pair": "observation-span-linear",
            "two_values_with_positive_mjd_pair": "explicit-mjd-linear",
            "explicit_mjd_requires_observation_bracketing": True,
            "extrapolation": "forbidden",
        }
        if contract != expected_contract:
            errors.append("astrometry application contract is inconsistent")

        observations = data["observations"]
        expected_count = data["expected_observation_count"]
        if type(expected_count) is not int or expected_count <= 0:
            errors.append("astrometry expected observation count must be positive")
            return errors
        if not isinstance(observations, list):
            errors.append("astrometry observations must be a sequence")
            return errors
        if len(observations) != expected_count:
            errors.append("astrometry observation count differs from expectation")

        for expected_index, observation in enumerate(observations):
            label = f"astrometry observation {expected_index}"
            if not isinstance(observation, dict):
                errors.append(f"{label} is not a mapping")
                continue
            if observation.get("observation_index") != expected_index:
                errors.append(f"{label} has inconsistent index")
            obsnum = observation.get("obsnum")
            if type(obsnum) is not int or obsnum <= 0:
                errors.append(f"{label} has invalid obsnum")

            requested = observation.get("requested")
            effective_record = observation.get("effective")
            if not isinstance(requested, dict) or not isinstance(
                effective_record, dict
            ):
                errors.append(f"{label} config records are malformed")
                continue
            effective = effective_record.get("config")
            resolution = effective_record.get("resolution")
            errors.extend(valid_astrometry_config(requested, f"{label} requested"))
            errors.extend(valid_astrometry_config(effective, f"{label} effective"))
            if requested != effective:
                errors.append(f"{label} effective config differs from request")
            if not isinstance(resolution, dict):
                errors.append(f"{label} resolution is not a mapping")
                continue

            offsets = requested.get("pointing_offsets", {})
            az = offsets.get("az_arcsec", [])
            mjd = offsets.get("modified_julian_date", [])
            explicit_mjd = bool(
                len(az) == 2
                and isinstance(mjd, list)
                and len(mjd) == 2
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value > 0
                    for value in mjd
                )
            )
            expected_mode = (
                "constant"
                if len(az) == 1
                else "explicit-mjd-linear"
                if explicit_mjd
                else "observation-span-linear"
            )
            if resolution.get("application_mode") != expected_mode:
                errors.append(f"{label} application mode is inconsistent")
            if resolution.get("explicit_mjd_support") is not explicit_mjd:
                errors.append(f"{label} MJD resolution is inconsistent")

            realized = observation.get("realized")
            if not isinstance(realized, dict):
                errors.append(f"{label} realized state is not a mapping")
                continue
            installation_count = realized.get("installation_count")
            application_count = realized.get("application_count")
            sample_count = realized.get("telescope_sample_count")
            for name, value in (
                ("installation_count", installation_count),
                ("application_count", application_count),
                ("telescope_sample_count", sample_count),
            ):
                if type(value) is not int or value <= 0:
                    errors.append(f"{label} {name} must be positive")
            if (
                type(installation_count) is int
                and type(application_count) is int
                and installation_count != application_count
            ):
                errors.append(f"{label} installation/application counts differ")
    except (AttributeError, KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate astrometry provenance semantics: {exc}")
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
    require_polarimetry: bool = False,
    require_astrometry: bool = False,
    require_runtime: bool = False,
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
            or (name == "polarimetry" and require_polarimetry)
            or (name == "astrometry" and require_astrometry)
            or (name == "runtime" and require_runtime)
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
                if name == "runtime" and require_runtime:
                    accepted_schema_versions = (spec["schema_version"],)
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
                    if name == "astrometry":
                        semantic_errors = (
                            astrometry_provenance_semantic_errors(data)
                        )
                    elif name == "polarimetry":
                        semantic_errors = (
                            polarimetry_provenance_semantic_errors(data)
                        )
                    elif name == "kids_external":
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
                        semantic_errors = raw_provenance_semantic_errors(
                            data, path
                        )
                    elif name == "mapmaking":
                        semantic_errors = (
                            mapmaking_provenance_semantic_errors(data)
                        )
                    elif name == "coadd":
                        semantic_errors = coadd_provenance_semantic_errors(
                            data
                        )
                    elif name == "noise_products":
                        semantic_errors = noise_provenance_semantic_errors(
                            data
                        ) + noise_package_integrity_errors(data, path)
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
            getattr(args, "require_polarimetry_provenance", False),
            getattr(args, "require_astrometry_provenance", False),
            getattr(args, "require_runtime_provenance", False),
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
        "--require-runtime-provenance",
        action="store_true",
        help=(
            "Fail unless runtime_provenance.yaml records a valid requested, "
            "effective, and realized resource contract."
        ),
    )
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
        "--require-polarimetry-provenance",
        action="store_true",
        help=(
            "Fail unless polarimetry_provenance.yaml records the disabled "
            "planned capability contract."
        ),
    )
    parser.add_argument(
        "--require-astrometry-provenance",
        action="store_true",
        help=(
            "Fail unless astrometry_provenance.yaml records a complete "
            "observation-indexed pointing-offset application contract."
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
