#!/usr/bin/env python3
"""Validate one owner-run WP-7 identity RTC acceptance record."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA = "citlali-wp7-identity-rtc-acceptance-v1"
PRODUCER_INTERFACE = "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1"
PRODUCER_SHA256 = (
    "f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969"
)
DESIGN_COMMIT = "46824f7de"
ALIGN_REPAIR_COMMIT = "d55deefb3"
HEX40 = re.compile(r"^[0-9a-f]{40}$")


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
    if record.get("design_commit") != DESIGN_COMMIT:
        raise AcceptanceError(f"design_commit must be {DESIGN_COMMIT}")
    if record.get("align_repair_commit") != ALIGN_REPAIR_COMMIT:
        raise AcceptanceError(f"align_repair_commit must be {ALIGN_REPAIR_COMMIT}")
    require_true(record, "design_is_ancestor")
    require_true(record, "align_repair_is_ancestor")
    require_true(record, "owner_run")
    require_true(record, "real_paired_data")
    require_true(record, "product_inspected_in_memory")
    require_true(record, "publication_complete")
    require_string(record, "representative_dataset_id")
    require_integer(record, "observation", 1)
    require_string(record, "mapping_instance_id")
    if record.get("producer_interface_id") != PRODUCER_INTERFACE:
        raise AcceptanceError(
            f"producer_interface_id must be {PRODUCER_INTERFACE!r}"
        )
    if record.get("producer_interface_sha256") != PRODUCER_SHA256:
        raise AcceptanceError(
            f"producer_interface_sha256 must be {PRODUCER_SHA256}"
        )
    if record.get("terminal_state") != "complete":
        raise AcceptanceError("terminal_state must be 'complete'")

    metrics = require_object(record.get("metrics"), "metrics")
    require_integer(metrics, "network_count", 1)
    require_integer(metrics, "detector_count", 1)
    aligned_cells = require_integer(metrics, "aligned_cell_count", 1)
    mapped_cells = require_integer(metrics, "mapped_cell_count", 1)
    if mapped_cells > aligned_cells:
        raise AcceptanceError("mapped_cell_count cannot exceed aligned_cell_count")
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
    if require_integer(metrics, "pair_decision_comparison_count", 1) != aligned_cells:
        raise AcceptanceError(
            "pair_decision_comparison_count must cover every aligned cell"
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
        "pair_decision_mismatch_count",
        "member_cause_mismatch_count",
        "chunk_scientific_mismatch_count",
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
