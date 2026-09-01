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


SCHEMA = "citlali-timestream-successor-identity-route-acceptance-v1"
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


def require_zero(record: dict[str, Any], name: str) -> None:
    if require_integer(record, name) != 0:
        raise AcceptanceError(f"{name} must be zero")


def validate(record: dict[str, Any]) -> None:
    if record.get("schema") != SCHEMA:
        raise AcceptanceError(f"schema must be {SCHEMA!r}")
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
    executable_version = require_string(record, "executable_version")
    if "dirty" in executable_version:
        raise AcceptanceError("executable_version must describe a clean build")
    require_true(record, "citlali_source_clean")
    if not HEX64.fullmatch(require_string(record, "executable_sha256")):
        raise AcceptanceError("executable_sha256 must be one lowercase SHA-256")
    if record.get("build_environment") != "spack":
        raise AcceptanceError("build_environment must be 'spack'")
    require_string(record, "build_profile")
    if not HEX64.fullmatch(require_string(record, "spack_lock_sha256")):
        raise AcceptanceError("spack_lock_sha256 must be one lowercase SHA-256")
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
    require_string(record, "representative_dataset_id")
    if require_integer(record, "observation", 1) != 152390:
        raise AcceptanceError("observation must be 152390")
    require_integer(record, "first_native_row")
    native_row_count = require_integer(record, "native_row_count", 2048)
    require_string(record, "mapping_instance_id")
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
    if record.get("terminal_state") != "complete":
        raise AcceptanceError("terminal_state must be 'complete'")
    if record.get("terminal_failure_cause") != "none":
        raise AcceptanceError("terminal_failure_cause must be 'none'")
    if record.get("terminal_failure_detail") != "":
        raise AcceptanceError("terminal_failure_detail must be empty")

    metrics = require_object(record.get("metrics"), "metrics")
    network_count = require_integer(metrics, "network_count", 1)
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
