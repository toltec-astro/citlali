#!/usr/bin/env python3
"""Validate one owner-run WP-7 identity RTC acceptance record."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA = "citlali-wp7-identity-rtc-acceptance-v2"
OCCURRENCE_SUPPORT_AUTHORITY_SCHEMA = (
    "citlali-native-occurrence-support-authority-v1"
)
OCCURRENCE_SUPPORT_DURATION_RELATION = (
    "Header.Toltec.AccumLen / Header.Toltec.FpgaFreq"
)
PRODUCER_INTERFACE = "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1"
PRODUCER_SHA256 = (
    "f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969"
)
DESIGN_COMMIT = "46824f7de"
ALIGN_REPAIR_COMMIT = "d55deefb3"
KIDSCPP_REVISION = "04088da182622c3e879f04314974a7c0d60ee2d6"
KIDSCPP_PATCH_SHA256 = (
    "98ed435199078e758f1cfe55dceeddbc9d4f623ce6406e84077e6dde04db4d96"
)
TULA_REVISION = "f30f81d97c44bd79618273bb842302ef839c6ab1"
TULA_PATCH_SHA256 = (
    "c331a9aeb61aa3171efb85cc5bc2b50f1a34b243d44c25c5d4a97c2250e70b4a"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
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


def require_number(record: dict[str, Any], name: str, minimum: float = 0.0) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceError(f"{name} must be numeric")
    result = float(value)
    if result <= minimum:
        raise AcceptanceError(f"{name} must be > {minimum}")
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
    if source_revision[:9] not in executable_revision:
        raise AcceptanceError(
            "executable_revision must contain the source revision's first nine hex digits"
        )
    executable_sha256 = require_string(record, "executable_sha256")
    if not HEX64.fullmatch(executable_sha256):
        raise AcceptanceError("executable_sha256 must be one lowercase SHA-256")
    if record.get("kidscpp_revision") != KIDSCPP_REVISION:
        raise AcceptanceError(f"kidscpp_revision must be {KIDSCPP_REVISION}")
    if record.get("kidscpp_build_patch_sha256") != KIDSCPP_PATCH_SHA256:
        raise AcceptanceError(
            f"kidscpp_build_patch_sha256 must be {KIDSCPP_PATCH_SHA256}"
        )
    if record.get("tula_revision") != TULA_REVISION:
        raise AcceptanceError(f"tula_revision must be {TULA_REVISION}")
    if record.get("tula_build_patch_sha256") != TULA_PATCH_SHA256:
        raise AcceptanceError(
            f"tula_build_patch_sha256 must be {TULA_PATCH_SHA256}"
        )
    if record.get("design_commit") != DESIGN_COMMIT:
        raise AcceptanceError(f"design_commit must be {DESIGN_COMMIT}")
    if record.get("align_repair_commit") != ALIGN_REPAIR_COMMIT:
        raise AcceptanceError(f"align_repair_commit must be {ALIGN_REPAIR_COMMIT}")
    require_true(record, "design_is_ancestor")
    require_true(record, "align_repair_is_ancestor")
    require_true(record, "owner_run")
    require_true(record, "real_paired_data")
    require_true(record, "apt_bundle_verified")
    require_true(record, "raw_sources_verified")
    require_true(record, "tune_bindings_verified")
    require_true(record, "tune_accumulation_explicit")
    require_true(record, "product_inspected_in_memory")
    require_true(record, "publication_complete")
    require_string(record, "representative_dataset_id")
    if require_integer(record, "observation", 1) != 152390:
        raise AcceptanceError("observation must be 152390")
    require_integer(record, "first_native_row")
    native_row_count = require_integer(record, "native_row_count", 2048)
    require_string(record, "mapping_instance_id")
    if record.get("producer_interface_id") != PRODUCER_INTERFACE:
        raise AcceptanceError(
            f"producer_interface_id must be {PRODUCER_INTERFACE!r}"
        )
    if record.get("producer_interface_sha256") != PRODUCER_SHA256:
        raise AcceptanceError(
            f"producer_interface_sha256 must be {PRODUCER_SHA256}"
        )
    if (
        record.get("occurrence_support_authority_schema")
        != OCCURRENCE_SUPPORT_AUTHORITY_SCHEMA
    ):
        raise AcceptanceError(
            "occurrence_support_authority_schema must be the approved schema"
        )
    require_string(record, "occurrence_support_authority_id")
    authority_sha256 = require_string(
        record, "occurrence_support_authority_sha256"
    )
    if not HEX64.fullmatch(authority_sha256):
        raise AcceptanceError(
            "occurrence_support_authority_sha256 must be one lowercase SHA-256"
        )
    require_true(record, "occurrence_support_authority_approved")
    require_string(record, "occurrence_support_authority_approved_by")
    approved_at = require_string(
        record, "occurrence_support_authority_approved_at_utc"
    )
    if not UTC_SECONDS.fullmatch(approved_at):
        raise AcceptanceError(
            "occurrence_support_authority_approved_at_utc must be exact UTC seconds"
        )
    if record.get("occurrence_support_event_time_role") not in {
        "integration_start",
        "integration_center",
        "integration_end",
    }:
        raise AcceptanceError(
            "occurrence_support_event_time_role must be an approved role"
        )
    if (
        record.get("occurrence_support_duration_relation")
        != OCCURRENCE_SUPPORT_DURATION_RELATION
    ):
        raise AcceptanceError(
            "occurrence_support_duration_relation must use approved raw facts"
        )
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
    if native_occurrences != network_count * native_row_count:
        raise AcceptanceError(
            "native_occurrence_count must cover every network and native row"
        )
    detector_occurrences = require_integer(
        metrics, "native_detector_occurrence_count", 1
    )
    if detector_occurrences != detector_count * native_row_count:
        raise AcceptanceError(
            "native_detector_occurrence_count must cover every detector and native row"
        )
    paired_numeric_bytes = require_integer(
        metrics, "paired_numeric_payload_bytes", 1
    )
    if paired_numeric_bytes != 2 * detector_occurrences * 8:
        raise AcceptanceError(
            "paired_numeric_payload_bytes must contain exactly two float64 planes"
        )
    member_state_bytes = require_integer(metrics, "paired_member_state_bytes", 1)
    occurrence_interval_bytes = require_integer(
        metrics, "paired_occurrence_interval_bytes", 1
    )
    detector_axis_bytes = require_integer(metrics, "paired_detector_axis_bytes", 1)
    identity_text_bytes = require_integer(metrics, "paired_identity_text_bytes", 1)
    if require_integer(metrics, "paired_logical_owned_bytes", 1) != sum(
        (
            paired_numeric_bytes,
            member_state_bytes,
            occurrence_interval_bytes,
            detector_axis_bytes,
            identity_text_bytes,
        )
    ):
        raise AcceptanceError(
            "paired_logical_owned_bytes must equal its compact storage components"
        )
    if require_integer(metrics, "referenced_native_axis_count", 1) != network_count:
        raise AcceptanceError(
            "referenced_native_axis_count must equal network_count"
        )
    aligned_cells = require_integer(metrics, "aligned_cell_count", 1)
    mapped_cells = require_integer(metrics, "mapped_cell_count", 1)
    if mapped_cells > aligned_cells:
        raise AcceptanceError("mapped_cell_count cannot exceed aligned_cell_count")
    evidence_events = require_integer(metrics, "evidence_event_count")
    direct_x_events = require_integer(metrics, "direct_x_event_count")
    direct_r_events = require_integer(metrics, "direct_r_event_count")
    both_events = require_integer(metrics, "x_and_r_event_count")
    alignment_absences = require_integer(metrics, "alignment_absence_event_count")
    ineligible_cells = require_integer(metrics, "pair_ineligible_cell_count")
    if evidence_events != ineligible_cells:
        raise AcceptanceError(
            "identity RTC requires one causal evidence event per ineligible pair"
        )
    if (
        both_events > direct_x_events
        or both_events > direct_r_events
        or alignment_absences > evidence_events
        or ineligible_cells > aligned_cells
    ):
        raise AcceptanceError("RTC evidence summary counts are inconsistent")
    if require_integer(metrics, "x_numerically_valid_cell_count") > mapped_cells:
        raise AcceptanceError("x valid cell count cannot exceed mapped cells")
    if require_integer(metrics, "r_numerically_valid_cell_count") > mapped_cells:
        raise AcceptanceError("r valid cell count cannot exceed mapped cells")
    derived_evidence_bytes = require_integer(metrics, "derived_evidence_bytes")
    if derived_evidence_bytes > 16 * evidence_events:
        raise AcceptanceError(
            "derived_evidence_bytes exceeds the compact event bound"
        )
    require_integer(metrics, "derived_plan_bytes")
    compared_values = require_integer(metrics, "paired_value_comparison_count", 1)
    if compared_values != 2 * mapped_cells:
        raise AcceptanceError(
            "paired_value_comparison_count must cover x and r for every mapped cell"
        )
    if require_integer(metrics, "identity_comparison_count", 1) != aligned_cells:
        raise AcceptanceError(
            "identity_comparison_count must cover every aligned cell"
        )
    if require_integer(metrics, "support_comparison_count", 1) != aligned_cells:
        raise AcceptanceError(
            "support_comparison_count must cover every aligned cell"
        )
    if (
        require_integer(metrics, "producer_support_binding_count", 1)
        != native_occurrences
    ):
        raise AcceptanceError(
            "producer_support_binding_count must cover every native occurrence"
        )
    if require_integer(metrics, "pair_decision_comparison_count", 1) != aligned_cells:
        raise AcceptanceError(
            "pair_decision_comparison_count must cover every aligned cell"
        )
    if (
        require_integer(metrics, "pair_causal_evidence_comparison_count", 1)
        != aligned_cells
    ):
        raise AcceptanceError(
            "pair_causal_evidence_comparison_count must cover every aligned cell"
        )
    require_integer(metrics, "chunk_partition_count", 2)
    require_number(metrics, "wall_time_sec")
    require_number(metrics, "cpu_time_sec")
    require_integer(metrics, "peak_rss_bytes", 1)
    require_zero(metrics, "rtc_owned_numeric_bytes")

    for name in (
        "x_bitwise_mismatch_count",
        "r_bitwise_mismatch_count",
        "identity_mismatch_count",
        "support_mismatch_count",
        "producer_support_binding_mismatch_count",
        "pair_decision_mismatch_count",
        "pair_causal_evidence_mismatch_count",
        "member_cause_mismatch_count",
        "chunk_scientific_mismatch_count",
        "selected_time_mismatch_count",
        "representative_native_mismatch_count",
        "ast_interpolation_call_count",
        "cal_call_count",
        "val_call_count",
        "ptc_call_count",
        "map_call_count",
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
        print(f"WP-7 identity RTC acceptance: FAIL: {error}", file=sys.stderr)
        return 1
    print("WP-7 identity RTC acceptance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
