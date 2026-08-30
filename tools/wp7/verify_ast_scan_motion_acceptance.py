#!/usr/bin/env python3
"""Validate one exact WP-7 representative AST scan-motion evidence record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA = "citlali-wp7-ast-scan-motion-acceptance-v1"
POLICY_ID = "wp7-ast-scan-motion-v1"
PRODUCT_ROLE = "SCI-AST:scan_motion_planning@1"
DATASET_ID = "SCI_ALIGN_STAGE7_NGC4449_152390"
DESIGN_COMMIT = "46824f7de"
ALIGN_REPAIR_COMMIT = "d55deefb3"
KIDSCPP_REVISION = "04088da182622c3e879f04314974a7c0d60ee2d6"
KIDSCPP_PATCH_SHA256 = (
    "98ed435199078e758f1cfe55dceeddbc9d4f623ce6406e84077e6dde04db4d96"
)
KIDSCPP_TREE = "81569aacea2b6e1831dc5af20d6bf8a4ca78332f"
TULA_REVISION = "f30f81d97c44bd79618273bb842302ef839c6ab1"
TULA_PATCH_SHA256 = (
    "c331a9aeb61aa3171efb85cc5bc2b50f1a34b243d44c25c5d4a97c2250e70b4a"
)
TULA_TREE = "7ae84231a485c67e58134d9aa759b2c5b987c844"
TELESCOPE_FILENAME = "tel_toltec_2026-02-19_152390_00_0002.nc"
TELESCOPE_SHA256 = (
    "2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b"
)
TELESCOPE_BYTE_COUNT = 24157872
TELESCOPE_RECORD_COUNT = 62109
TIME_FIELD = "Data.TelescopeBackend.TelTime"
RA_FIELD = "Data.TelescopeBackend.SourceRaAct"
DEC_FIELD = "Data.TelescopeBackend.SourceDecAct"
EXPECTED_DEFECT_RECORDS = [2504, 12971]
EXPECTED_RAW_VALID_COUNT = 62109
EXPECTED_QUALITY_CLASSIFIED_COUNT = 62099
EXPECTED_REALIZED_VALID_COUNT = 62097
EXPECTED_DERIVATIVE_VALID_COUNT = 62067
EXPECTED_NETWORK_COUNT = 11
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AcceptanceError(ValueError):
    pass


def require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceError(f"{name} must be an object")
    return value


def require_array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AcceptanceError(f"{name} must be an array")
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


def require_finite(record: dict[str, Any], name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AcceptanceError(f"{name} must be finite")
    return result


def require_positive(record: dict[str, Any], name: str) -> float:
    value = require_finite(record, name)
    if value <= 0.0:
        raise AcceptanceError(f"{name} must be positive")
    return value


def require_true(record: dict[str, Any], name: str) -> None:
    if record.get(name) is not True:
        raise AcceptanceError(f"{name} must be true")


def require_false(record: dict[str, Any], name: str) -> None:
    if record.get(name) is not False:
        raise AcceptanceError(f"{name} must be false")


def require_zero(record: dict[str, Any], name: str) -> None:
    if require_integer(record, name) != 0:
        raise AcceptanceError(f"{name} must be zero")


def require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise AcceptanceError(f"{name} must be one lowercase SHA-256")
    return value


def require_sha256_identity(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or not HEX64.fullmatch(value.removeprefix("sha256:"))
    ):
        raise AcceptanceError(f"{name} must be one sha256: identity")
    return value


def validate(record: dict[str, Any]) -> None:
    if record.get("schema") != SCHEMA:
        raise AcceptanceError(f"schema must be {SCHEMA!r}")
    source_revision = require_string(record, "source_revision")
    if not HEX40.fullmatch(source_revision):
        raise AcceptanceError("source_revision must be one full lowercase Git SHA")
    if record.get("executable_revision") != source_revision:
        raise AcceptanceError("executable_revision must equal source_revision exactly")
    require_string(record, "executable_version")
    require_sha256(record.get("executable_sha256"), "executable_sha256")
    require_true(record, "citlali_source_clean")
    require_true(record, "citlali_ignored_source_state_verified")
    require_true(record, "dependency_state_verified")
    expected_dependencies = {
        "kidscpp_revision": KIDSCPP_REVISION,
        "kidscpp_build_patch_sha256": KIDSCPP_PATCH_SHA256,
        "kidscpp_tree": KIDSCPP_TREE,
        "tula_revision": TULA_REVISION,
        "tula_build_patch_sha256": TULA_PATCH_SHA256,
        "tula_tree": TULA_TREE,
    }
    for name, expected in expected_dependencies.items():
        if record.get(name) != expected:
            raise AcceptanceError(f"{name} must be {expected}")
    if record.get("design_commit") != DESIGN_COMMIT:
        raise AcceptanceError(f"design_commit must be {DESIGN_COMMIT}")
    if record.get("align_repair_commit") != ALIGN_REPAIR_COMMIT:
        raise AcceptanceError(f"align_repair_commit must be {ALIGN_REPAIR_COMMIT}")
    require_true(record, "design_is_ancestor")
    require_true(record, "align_repair_is_ancestor")
    require_true(record, "owner_run")
    require_true(record, "representative_data")
    require_true(record, "product_inspected_in_memory")
    require_false(record, "common_analysis_grid_requested")
    require_false(record, "persistent_ast_product_published")
    if record.get("authority_policy_id") != POLICY_ID:
        raise AcceptanceError(f"authority_policy_id must be {POLICY_ID!r}")
    if record.get("product_role") != PRODUCT_ROLE:
        raise AcceptanceError(f"product_role must be {PRODUCT_ROLE!r}")
    if record.get("representative_dataset_id") != DATASET_ID:
        raise AcceptanceError(f"representative_dataset_id must be {DATASET_ID!r}")
    if (
        require_integer(record, "observation", 1),
        require_integer(record, "subobservation"),
        require_integer(record, "scan"),
    ) != (152390, 0, 2):
        raise AcceptanceError("observation scope must be exactly 152390/0/2")

    telescope = require_object(record.get("telescope"), "telescope")
    expected_telescope = {
        "filename": TELESCOPE_FILENAME,
        "sha256": TELESCOPE_SHA256,
        "byte_count": TELESCOPE_BYTE_COUNT,
        "record_count": TELESCOPE_RECORD_COUNT,
        "time_field": TIME_FIELD,
        "ra_field": RA_FIELD,
        "dec_field": DEC_FIELD,
        "observation_goal": "Science",
        "observation_program": "Lissajous",
        "scan_file_valid": 1,
        "source_coordinate_system": 0,
    }
    for name, expected in expected_telescope.items():
        if telescope.get(name) != expected:
            raise AcceptanceError(f"telescope.{name} must be {expected!r}")
    if require_finite(telescope, "source_epoch") != 2000.0:
        raise AcceptanceError("telescope.source_epoch must be 2000.0")
    if require_finite(telescope, "nominal_cadence_hz") != 50.0:
        raise AcceptanceError("telescope.nominal_cadence_hz must be 50.0")
    minimum_interval = require_positive(telescope, "minimum_interval_sec")
    maximum_interval = require_positive(telescope, "maximum_interval_sec")
    if minimum_interval > maximum_interval or maximum_interval > 0.030:
        raise AcceptanceError("telescope intervals must be positive and <= 30 ms")
    if require_positive(
        telescope, "direct_adjacent_maximum_arcsec_per_sec"
    ) <= 1400.0:
        raise AcceptanceError("direct adjacent maximum must retain the raw defect")
    require_integer(telescope, "direct_adjacent_maximizing_record", 1)

    apt_bundle = require_object(record.get("apt_bundle"), "apt_bundle")
    require_sha256(apt_bundle.get("manifest_sha256"), "apt_bundle.manifest_sha256")
    require_sha256_identity(
        apt_bundle.get("semantic_sha256"), "apt_bundle.semantic_sha256"
    )
    require_sha256_identity(
        apt_bundle.get("envelope_sha256"), "apt_bundle.envelope_sha256"
    )
    if require_integer(apt_bundle, "participant_network_count", 1) != EXPECTED_NETWORK_COUNT:
        raise AcceptanceError("APT participant count must be exactly 11")

    binding = require_object(record.get("identity_binding"), "identity_binding")
    expected_binding = {
        "requested": 1523900001,
        "effective": 1523900002,
        "observation_resolved": 1523900003,
        "realized": 1523900004,
    }
    if binding != expected_binding:
        raise AcceptanceError("identity_binding must equal the bounded AST lifecycle binding")

    product = require_object(record.get("raw_product"), "raw_product")
    expected_counts = {
        "raw_direction_valid_count": EXPECTED_RAW_VALID_COUNT,
        "quality_classified_count": EXPECTED_QUALITY_CLASSIFIED_COUNT,
        "telemetry_defect_count": len(EXPECTED_DEFECT_RECORDS),
        "realized_direction_valid_count": EXPECTED_REALIZED_VALID_COUNT,
        "derivative_valid_count": EXPECTED_DERIVATIVE_VALID_COUNT,
        "continuity_run_count": 1,
        "referenced_source_axis_count": 1,
        "referenced_source_direction_plane_count": 2,
    }
    for name, expected in expected_counts.items():
        if require_integer(product, name) != expected:
            raise AcceptanceError(f"raw_product.{name} must be {expected}")
    if product.get("telemetry_defect_records") != EXPECTED_DEFECT_RECORDS:
        raise AcceptanceError(
            f"raw_product.telemetry_defect_records must be {EXPECTED_DEFECT_RECORDS}"
        )
    require_true(product, "maximum_available")
    require_zero(product, "maximum_causes")
    maximum_speed = require_positive(product, "maximum_speed_arcsec_per_sec")
    if not 200.0 < maximum_speed < 230.0:
        raise AcceptanceError("raw_product maximum speed must be in (200, 230) arcsec/s")
    require_integer(product, "maximizing_record", 1)
    require_integer(product, "admitted_candidate_count", 1)
    require_integer(product, "derived_record_bytes", 1)

    chunk = require_object(record.get("chunk_invariance"), "chunk_invariance")
    if require_integer(chunk, "partition_count", 1) != 3:
        raise AcceptanceError("chunk_invariance.partition_count must be 3")
    for name in (
        "record_mismatch_count",
        "telemetry_support_mismatch_count",
        "derivative_support_mismatch_count",
        "summary_mismatch_count",
    ):
        require_zero(chunk, name)

    mapping = require_object(record.get("network_mapping"), "network_mapping")
    if mapping.get("timing_scope") != "network-specific":
        raise AcceptanceError("network_mapping.timing_scope must be network-specific")
    total = require_integer(mapping, "total_occurrence_count", 1_000_001)
    available = require_integer(mapping, "available_count", 1_000_001)
    unavailable = require_integer(mapping, "unavailable_count", 1)
    support = require_integer(mapping, "support_count", 1)
    if available + unavailable != total:
        raise AcceptanceError("network mapping availability must cover every occurrence")
    if support != available:
        raise AcceptanceError("network mapping support count must equal availability")
    for name in (
        "identity_mismatch_count",
        "support_mismatch_count",
        "value_mismatch_count",
        "missing_unavailable_cause_count",
    ):
        require_zero(mapping, name)
    require_integer(mapping, "mapped_owned_bytes", 1)
    nw0_time = require_finite(mapping, "nw0_first_time_unix_sec")
    nw7_time = require_finite(mapping, "nw7_first_time_unix_sec")
    require_true(mapping, "nw0_nw7_times_distinct")
    if nw0_time == nw7_time:
        raise AcceptanceError("nw0 and nw7 must retain distinct network times")

    participants = require_array(mapping.get("participants"), "network_mapping.participants")
    if len(participants) != EXPECTED_NETWORK_COUNT:
        raise AcceptanceError("network mapping must contain exactly 11 participants")
    networks: set[int] = set()
    sum_occurrences = 0
    sum_available = 0
    sum_unavailable = 0
    for index, item in enumerate(participants):
        participant = require_object(item, f"network_mapping.participants[{index}]")
        network = require_integer(participant, "network")
        if network in networks:
            raise AcceptanceError("network mapping participant ids must be unique")
        networks.add(network)
        filename = require_string(participant, "filename")
        if not filename.startswith(f"toltec{network}_152390_000_0002_") or not filename.endswith(".nc"):
            raise AcceptanceError("participant filename must bind its exact network and scope")
        require_sha256(participant.get("sha256"), "participant.sha256")
        require_integer(participant, "byte_count", 1)
        occurrences = require_integer(participant, "occurrence_count", 1)
        participant_available = require_integer(participant, "available_count")
        participant_unavailable = require_integer(participant, "unavailable_count")
        if participant_available + participant_unavailable != occurrences:
            raise AcceptanceError("participant availability must cover every occurrence")
        require_integer(participant, "packet_discontinuity_count")
        require_integer(participant, "telemetry_defect_cause_count")
        first_time = require_finite(participant, "first_time_unix_sec")
        last_time = require_finite(participant, "last_time_unix_sec")
        if not first_time < last_time:
            raise AcceptanceError("participant native time must be strictly increasing")
        sum_occurrences += occurrences
        sum_available += participant_available
        sum_unavailable += participant_unavailable
    if (sum_occurrences, sum_available, sum_unavailable) != (
        total,
        available,
        unavailable,
    ):
        raise AcceptanceError("participant metrics must sum to network mapping totals")

    performance = require_object(record.get("performance"), "performance")
    for name in (
        "raw_product_wall_time_sec",
        "raw_product_cpu_time_sec",
        "network_mapping_wall_time_sec",
        "network_mapping_cpu_time_sec",
    ):
        require_positive(performance, name)
    require_integer(performance, "process_peak_rss_bytes", 1)
    require_zero(record, "unexpected_error_count")


def validate_exact_package(
    record_bytes: bytes,
    *,
    expected_record_sha256: str,
    expected_source_revision: str,
    expected_executable_sha256: str,
    executable: str | None = None,
) -> dict[str, Any]:
    if not isinstance(record_bytes, bytes):
        raise AcceptanceError("record bytes must be supplied as bytes")
    if not HEX64.fullmatch(expected_record_sha256):
        raise AcceptanceError("expected record hash must be one lowercase SHA-256")
    record_digest = hashlib.sha256(record_bytes).hexdigest()
    if record_digest != expected_record_sha256:
        raise AcceptanceError("record bytes disagree with exact package expectation")
    loaded = json.loads(record_bytes)
    record = require_object(loaded, "acceptance record")
    validate(record)
    if not HEX40.fullmatch(expected_source_revision):
        raise AcceptanceError("expected source revision must be one full lowercase SHA")
    if not HEX64.fullmatch(expected_executable_sha256):
        raise AcceptanceError("expected executable hash must be one lowercase SHA-256")
    if record["source_revision"] != expected_source_revision:
        raise AcceptanceError("source revision disagrees with exact package expectation")
    if record["executable_sha256"] != expected_executable_sha256:
        raise AcceptanceError("executable hash disagrees with exact package expectation")
    if executable is not None:
        path = Path(executable)
        if not path.is_file():
            raise AcceptanceError("executable file does not exist")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_executable_sha256:
            raise AcceptanceError("executable file disagrees with exact package expectation")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--expected-record-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--executable")
    arguments = parser.parse_args()
    try:
        validate_exact_package(
            arguments.record.read_bytes(),
            expected_record_sha256=arguments.expected_record_sha256,
            expected_source_revision=arguments.expected_source_revision,
            expected_executable_sha256=arguments.expected_executable_sha256,
            executable=arguments.executable,
        )
    except (AcceptanceError, OSError, json.JSONDecodeError) as error:
        print(f"WP-7 AST scan-motion acceptance validation: FAIL: {error}", file=sys.stderr)
        return 2
    print("WP-7 AST scan-motion acceptance validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
