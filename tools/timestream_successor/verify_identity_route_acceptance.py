#!/usr/bin/env python3
"""Validate one authoritative Timestream Successor identity-route record."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA = "citlali-timestream-successor-identity-route-acceptance-v2"
SUBJECT_CANDIDATE_REVISION = "b57d9f606549d524ab6bb61faf0cd3d52ac27db6"
SUBJECT_CANDIDATE_TREE = "32de9791255c6c52032c0f05d64054b83ff44de5"
REPRESENTATIVE_DATASET = "SCI_ALIGN_STAGE7_NGC4449_152390"
TELESCOPE_FILENAME = "tel_toltec_2026-02-19_152390_00_0002.nc"
TELESCOPE_SHA256 = (
    "2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b"
)
TELESCOPE_BYTE_COUNT = 24157872
TELESCOPE_RECORD_COUNT = 62109
OCCURRENCE_SUPPORT_ASSIGNMENT_SCHEMA = (
    "citlali-native-occurrence-support-assignment-v1"
)
OCCURRENCE_SUPPORT_ASSIGNMENT_STATUS = "provisional_calibration_pending"
OCCURRENCE_SUPPORT_ASSIGNMENT_ID = (
    "wp7-provisional-integration-center-152390-v1"
)
OCCURRENCE_SUPPORT_ASSIGNMENT_SHA256 = (
    "6fc4e9009b98190c42cc3f6e7e030fa317e8ae5f9e707cd968110a696fac2b6c"
)
OCCURRENCE_SUPPORT_DURATION_RELATION = (
    "Header.Toltec.AccumLen / Header.Toltec.FpgaFreq"
)
OCCURRENCE_SUPPORT_CALIBRATION_DISPOSITION = (
    "replace_with_calibrated_producer_relation_when_available"
)
PRODUCER_INTERFACE = "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1"
PRODUCER_SHA256 = (
    "f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX_SHORT = re.compile(r"^[0-9a-f]{7,40}$")
UTC_SECONDS = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


class AcceptanceError(ValueError):
    pass


def require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceError(f"{name} must be an object")
    return value


def require_string(record: dict[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise AcceptanceError(f"{name} must be a nonempty string")
    return value


def require_integer(record: dict[str, Any], name: str, minimum: int = 0) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AcceptanceError(f"{name} must be an integer >= {minimum}")
    return value


def require_number(record: dict[str, Any], name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise AcceptanceError(f"{name} must be finite and positive")
    return result


def require_true(record: dict[str, Any], name: str) -> None:
    if record.get(name) is not True:
        raise AcceptanceError(f"{name} must be true")


def require_false(record: dict[str, Any], name: str) -> None:
    if record.get(name) is not False:
        raise AcceptanceError(f"{name} must be false")


def require_zero(record: dict[str, Any], name: str) -> None:
    if require_integer(record, name) != 0:
        raise AcceptanceError(f"{name} must be zero")


def validate(record: dict[str, Any]) -> None:
    if record.get("schema") != SCHEMA:
        raise AcceptanceError(f"schema must be {SCHEMA!r}")
    if record.get("subject_candidate_revision") != SUBJECT_CANDIDATE_REVISION:
        raise AcceptanceError("subject_candidate_revision is not the accepted candidate")
    if record.get("subject_candidate_tree") != SUBJECT_CANDIDATE_TREE:
        raise AcceptanceError("subject_candidate_tree is not the accepted candidate tree")
    source_revision = require_string(record, "source_revision")
    if not HEX40.fullmatch(source_revision):
        raise AcceptanceError("source_revision must be one full lowercase Git SHA")
    executable_revision = require_string(record, "executable_revision")
    if not HEX_SHORT.fullmatch(executable_revision) or not source_revision.startswith(
        executable_revision
    ):
        raise AcceptanceError(
            "executable_revision must be an exact prefix of source_revision"
        )
    if require_string(record, "tooling_revision") != source_revision:
        raise AcceptanceError("tooling_revision must equal source_revision")
    executable_version = require_string(record, "executable_version")
    if "dirty" in executable_version:
        raise AcceptanceError("executable_version must describe a clean build")
    require_true(record, "citlali_source_clean")
    if not HEX64.fullmatch(require_string(record, "executable_sha256")):
        raise AcceptanceError("executable_sha256 must be one lowercase SHA-256")
    if record.get("build_environment") != "spack":
        raise AcceptanceError("build_environment must be 'spack'")
    if record.get("build_profile") != "unity-gcc13":
        raise AcceptanceError("build_profile must be 'unity-gcc13'")
    if not HEX64.fullmatch(require_string(record, "spack_lock_sha256")):
        raise AcceptanceError("spack_lock_sha256 must be one lowercase SHA-256")
    require_integer(record, "spack_lock_byte_count", 1)
    require_true(record, "spack_lock_retained")
    require_string(record, "spack_root_dag")
    require_true(record, "dependency_state_verified")
    for name in ("kidscpp_version", "tula_version"):
        value = require_string(record, name)
        if name.endswith("version") and "dirty" in value:
            raise AcceptanceError(f"{name} must describe a clean dependency")

    for name in (
        "owner_run",
        "real_paired_data",
        "apt_bundle_verified",
        "raw_sources_verified",
        "tune_bindings_verified",
        "tune_accumulation_explicit",
        "product_inspected_in_memory",
        "publication_complete",
    ):
        require_true(record, name)
    if record.get("route_context_state") != "map_facing_context_complete":
        raise AcceptanceError("route_context_state must be map_facing_context_complete")
    for name in (
        "route_activated",
        "ordinary_route_changed",
        "canonical_integration_performed",
        "representative_science_claim",
        "map_action_performed",
    ):
        require_false(record, name)
    if record.get("representative_dataset_id") != REPRESENTATIVE_DATASET:
        raise AcceptanceError("representative_dataset_id is not the approved dataset")
    if require_integer(record, "observation", 1) != 152390:
        raise AcceptanceError("observation must be 152390")
    if require_integer(record, "subobservation") != 0:
        raise AcceptanceError("subobservation must be 0")
    if require_integer(record, "scan", 1) != 2:
        raise AcceptanceError("scan must be 2")
    if require_integer(record, "first_native_row") != 20000:
        raise AcceptanceError("first_native_row must be 20000")
    native_row_count = require_integer(record, "native_row_count", 2048)
    if native_row_count != 2048:
        raise AcceptanceError("native_row_count must be exactly 2048")
    require_string(record, "mapping_instance_id")
    if record.get("telescope_filename") != TELESCOPE_FILENAME:
        raise AcceptanceError("telescope_filename is not approved")
    if record.get("telescope_sha256") != TELESCOPE_SHA256:
        raise AcceptanceError("telescope_sha256 is not approved")
    if require_integer(record, "telescope_byte_count", 1) != TELESCOPE_BYTE_COUNT:
        raise AcceptanceError("telescope_byte_count is not approved")
    if require_integer(record, "telescope_record_count", 1) != TELESCOPE_RECORD_COUNT:
        raise AcceptanceError("telescope_record_count is not approved")
    if record.get("producer_interface_id") != PRODUCER_INTERFACE:
        raise AcceptanceError("producer_interface_id is not the approved interface")
    if record.get("producer_interface_sha256") != PRODUCER_SHA256:
        raise AcceptanceError("producer_interface_sha256 is not approved")

    expected_support = {
        "occurrence_support_assignment_schema": (
            OCCURRENCE_SUPPORT_ASSIGNMENT_SCHEMA
        ),
        "occurrence_support_assignment_id": OCCURRENCE_SUPPORT_ASSIGNMENT_ID,
        "occurrence_support_assignment_sha256": (
            OCCURRENCE_SUPPORT_ASSIGNMENT_SHA256
        ),
        "occurrence_support_assignment_status": (
            OCCURRENCE_SUPPORT_ASSIGNMENT_STATUS
        ),
        "occurrence_support_calibration_disposition": (
            OCCURRENCE_SUPPORT_CALIBRATION_DISPOSITION
        ),
        "occurrence_support_event_time_role": "integration_center",
        "occurrence_support_duration_relation": (
            OCCURRENCE_SUPPORT_DURATION_RELATION
        ),
    }
    for name, expected in expected_support.items():
        if record.get(name) != expected:
            raise AcceptanceError(f"{name} must preserve the approved assignment")
    require_string(record, "occurrence_support_assigned_by")
    if not UTC_SECONDS.fullmatch(
        require_string(record, "occurrence_support_assigned_at_utc")
    ):
        raise AcceptanceError(
            "occurrence_support_assigned_at_utc must use exact UTC seconds"
        )
    require_true(record, "occurrence_support_calibration_pending")
    require_true(record, "ast_present_in_rtc_input_context")
    if record.get("identity_rtc_ast_dependency") != "not_applicable":
        raise AcceptanceError("identity_rtc_ast_dependency must be not_applicable")
    if require_integer(record, "val_initial_generation") != 0:
        raise AcceptanceError("val_initial_generation must be 0")
    if require_integer(record, "val_committed_finding_count") != 0:
        raise AcceptanceError("val_committed_finding_count must be 0")
    require_true(record, "val_exact_snapshot_bound")
    expected_unavailable_states = {
        "calibration_product_state": "unavailable_component_not_admitted",
        "calibration_for_ptc_val_evaluation_state": (
            "unavailable_calibration_product_absent"
        ),
        "ptc_product_state": "unavailable_component_not_admitted",
        "ptc_for_map_val_evaluation_state": "unavailable_ptc_product_absent",
        "map_admission_state": "unavailable_calibration_and_ptc_products",
    }
    for name, expected in expected_unavailable_states.items():
        if record.get(name) != expected:
            raise AcceptanceError(f"{name} must be {expected!r}")
    if record.get("terminal_state") != "complete":
        raise AcceptanceError("terminal_state must be 'complete'")
    if record.get("terminal_failure_cause") != "none":
        raise AcceptanceError("terminal_failure_cause must be 'none'")
    if record.get("terminal_failure_detail") != "":
        raise AcceptanceError("terminal_failure_detail must be empty")

    metrics = require_object(record.get("metrics"), "metrics")
    network_count = require_integer(metrics, "network_count", 1)
    if network_count != 11:
        raise AcceptanceError("network_count must be exactly 11")
    detector_count = require_integer(metrics, "detector_count", 1)
    native_occurrences = require_integer(metrics, "native_occurrence_count", 1)
    detector_occurrences = require_integer(
        metrics, "native_detector_occurrence_count", 1
    )
    if native_occurrences != network_count * native_row_count:
        raise AcceptanceError("native_occurrence_count does not cover the run")
    if detector_occurrences != detector_count * native_row_count:
        raise AcceptanceError(
            "native_detector_occurrence_count does not cover the run"
        )
    paired_numeric_bytes = require_integer(
        metrics, "paired_numeric_payload_bytes", 1
    )
    if paired_numeric_bytes != 2 * detector_occurrences * 8:
        raise AcceptanceError("paired payload must contain exactly two float64 planes")
    storage_components = (
        paired_numeric_bytes,
        require_integer(metrics, "paired_coordinate_state_bytes", 1),
        require_integer(metrics, "paired_occurrence_axis_bytes", 1),
        require_integer(metrics, "paired_detector_axis_bytes", 1),
        require_integer(metrics, "paired_identity_text_bytes", 1),
    )
    if require_integer(metrics, "paired_logical_owned_bytes", 1) != sum(
        storage_components
    ):
        raise AcceptanceError("paired_logical_owned_bytes is not exact")
    if require_integer(metrics, "referenced_native_axis_count", 1) != network_count:
        raise AcceptanceError("referenced_native_axis_count must equal network_count")
    if require_integer(metrics, "rtc_native_occurrence_count", 1) != native_occurrences:
        raise AcceptanceError("RTC occurrence cardinality drifted")
    if (
        require_integer(metrics, "rtc_detector_occurrence_count", 1)
        != detector_occurrences
    ):
        raise AcceptanceError("RTC detector-occurrence cardinality drifted")

    evidence_events = require_integer(metrics, "evidence_event_count")
    direct_x = require_integer(metrics, "direct_x_event_count")
    direct_r = require_integer(metrics, "direct_r_event_count")
    both = require_integer(metrics, "x_and_r_event_count")
    ineligible = require_integer(metrics, "pair_ineligible_cell_count")
    x_available = require_integer(metrics, "x_payload_available_cell_count")
    r_available = require_integer(metrics, "r_payload_available_cell_count")
    x_valid = require_integer(metrics, "x_numerically_valid_cell_count")
    r_valid = require_integer(metrics, "r_numerically_valid_cell_count")
    if evidence_events != ineligible or direct_x + direct_r - both != evidence_events:
        raise AcceptanceError("RTC evidence summary is inconsistent")
    if direct_x != detector_occurrences - x_valid:
        raise AcceptanceError("direct_x_event_count does not bind invalid x cells")
    if direct_r != detector_occurrences - r_valid:
        raise AcceptanceError("direct_r_event_count does not bind invalid r cells")
    if not (
        x_valid <= x_available <= detector_occurrences
        and r_valid <= r_available <= detector_occurrences
        and both <= min(direct_x, direct_r)
        and ineligible <= detector_occurrences
    ):
        raise AcceptanceError("native availability/validity counts are inconsistent")
    if require_integer(metrics, "derived_evidence_bytes") > 16 * evidence_events:
        raise AcceptanceError("derived_evidence_bytes exceeds the compact bound")
    require_zero(metrics, "derived_plan_bytes")
    require_zero(metrics, "rtc_owned_numeric_bytes")

    ast_available = require_integer(metrics, "ast_available_occurrence_count")
    ast_unavailable = require_integer(metrics, "ast_unavailable_occurrence_count")
    if ast_available + ast_unavailable != native_occurrences:
        raise AcceptanceError("AST available/unavailable counts do not cover the run")
    if require_integer(metrics, "ast_support_count") != ast_available:
        raise AcceptanceError("ast_support_count must equal available AST occurrences")
    require_integer(metrics, "ast_raw_owned_bytes", 1)
    require_integer(metrics, "ast_mapped_owned_bytes", 1)
    require_zero(metrics, "align_owned_bytes")
    require_integer(metrics, "rtc_input_owned_bytes", 1)
    require_zero(metrics, "rtc_output_owned_bytes")
    require_zero(metrics, "val_owned_bytes")

    exact_counts = {
        "paired_ingress_value_comparison_count": 2 * detector_occurrences,
        "paired_ingress_identity_comparison_count": detector_count,
        "paired_ingress_member_state_comparison_count": 2 * detector_occurrences,
        "rtc_product_value_comparison_count": 2 * detector_occurrences,
        "identity_comparison_count": detector_occurrences,
        "support_comparison_count": native_occurrences,
        "native_time_comparison_count": native_occurrences,
        "representative_native_comparison_count": native_occurrences,
        "assigned_support_binding_count": native_occurrences,
        "pair_decision_comparison_count": detector_occurrences,
        "pair_causal_evidence_comparison_count": detector_occurrences,
        "chunk_partition_count": 2,
        "chunk_realized_operator_comparison_count": 1,
        "chunk_scientific_comparison_count": detector_occurrences,
        "route_occurrence_binding_count": native_occurrences,
        "ast_mapped_occurrence_count": native_occurrences,
        "val_binding_comparison_count": 8,
        "native_admission_entry_count": 1,
        "learn_entry_count": 1,
        "consider_entry_count": 1,
        "apply_entry_count": 1,
        "finalization_entry_count": 1,
        "publication_entry_count": 1,
    }
    for name, expected in exact_counts.items():
        if require_integer(metrics, name, 1) != expected:
            raise AcceptanceError(f"{name} must be exactly {expected}")
    require_number(metrics, "wall_time_sec")
    require_number(metrics, "cpu_time_sec")
    require_integer(metrics, "process_peak_rss_bytes", 1)

    for name in (
        "x_bitwise_mismatch_count",
        "r_bitwise_mismatch_count",
        "paired_ingress_identity_mismatch_count",
        "paired_ingress_member_state_mismatch_count",
        "identity_mismatch_count",
        "support_mismatch_count",
        "assigned_support_binding_mismatch_count",
        "pair_decision_mismatch_count",
        "pair_causal_evidence_mismatch_count",
        "member_cause_mismatch_count",
        "chunk_realized_operator_mismatch_count",
        "chunk_scientific_mismatch_count",
        "selected_time_mismatch_count",
        "representative_native_mismatch_count",
        "route_occurrence_binding_mismatch_count",
        "ast_identity_mismatch_count",
        "ast_support_mismatch_count",
        "val_binding_mismatch_count",
        "unexpected_error_count",
        "unexpected_critical_count",
    ):
        require_zero(metrics, name)


def load_record(path: str) -> dict[str, Any]:
    if path == "-":
        return require_object(json.load(sys.stdin), "acceptance record")
    with Path(path).open("r", encoding="utf-8") as stream:
        return require_object(json.load(stream), "acceptance record")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", help="acceptance JSON path, or - for stdin")
    args = parser.parse_args()
    try:
        validate(load_record(args.record))
    except (AcceptanceError, json.JSONDecodeError, OSError) as error:
        print(
            f"Timestream Successor identity-route acceptance: FAIL: {error}",
            file=sys.stderr,
        )
        return 1
    print("Timestream Successor identity-route acceptance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
