#!/usr/bin/env python3
"""Validate the checked-in Citlali reduction validation ledger."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
ACCEPTED_STATUSES = {
    "accepted",
    "accepted_with_intended_provenance_change",
    "accepted_with_intended_science_change",
    "accepted_with_scientific_tolerance",
}


class LedgerError(ValueError):
    pass


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise LedgerError(f"{context}: missing required field {key!r}")
    return mapping[key]


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LedgerError(f"{context}: expected object")
    return value


def require_nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{context}: expected non-empty string")
    return value


def validate_record(record: Any, index: int) -> str:
    context = f"records[{index}]"
    item = require_mapping(record, context)
    record_id = require_nonempty_text(require(item, "record_id", context),
                                      f"{context}.record_id")
    status = require_nonempty_text(require(item, "status", context),
                                   f"{context}.status")
    if status not in ACCEPTED_STATUSES:
        raise LedgerError(f"{context}.status: unsupported accepted status {status!r}")
    require_nonempty_text(require(item, "mode", context), f"{context}.mode")

    candidate = require_mapping(require(item, "candidate", context),
                                f"{context}.candidate")
    candidate_sha = require_nonempty_text(
        require(candidate, "citlali_sha", f"{context}.candidate"),
        f"{context}.candidate.citlali_sha")
    if not GIT_SHA_RE.fullmatch(candidate_sha):
        raise LedgerError(f"{context}.candidate.citlali_sha: invalid Git SHA")

    config = require_mapping(require(item, "config", context),
                             f"{context}.config")
    merged_hash = require_nonempty_text(
        require(config, "canonical_merged_sha256", f"{context}.config"),
        f"{context}.config.canonical_merged_sha256")
    if not SHA256_RE.fullmatch(merged_hash):
        raise LedgerError(
            f"{context}.config.canonical_merged_sha256: invalid SHA-256")
    ordered_sources = require(config, "ordered_sources", f"{context}.config")
    if not isinstance(ordered_sources, list) or not ordered_sources:
        raise LedgerError(f"{context}.config.ordered_sources: expected non-empty list")
    for source_index, source in enumerate(ordered_sources):
        source_context = f"{context}.config.ordered_sources[{source_index}]"
        source_item = require_mapping(source, source_context)
        source_hash = require_nonempty_text(
            require(source_item, "sha256", source_context),
            f"{source_context}.sha256")
        if not SHA256_RE.fullmatch(source_hash):
            raise LedgerError(f"{source_context}.sha256: invalid SHA-256")
        require(source_item, "precedence", source_context)
        require_nonempty_text(require(source_item, "role", source_context),
                              f"{source_context}.role")

    audit = require_mapping(require(item, "audit", context), f"{context}.audit")
    if require(audit, "finished", f"{context}.audit") is not True:
        raise LedgerError(f"{context}.audit.finished: accepted run must be complete")
    serious = require_mapping(
        require(audit, "serious_log_counts", f"{context}.audit"),
        f"{context}.audit.serious_log_counts")
    if any(value for value in serious.values()):
        raise LedgerError(f"{context}.audit.serious_log_counts: accepted run has errors")

    comparison = require_mapping(require(item, "comparison", context),
                                 f"{context}.comparison")
    changed = require(comparison, "changed_records", f"{context}.comparison")
    skipped = require(comparison, "skipped_records", f"{context}.comparison")
    if not isinstance(changed, int) or changed < 0:
        raise LedgerError(f"{context}.comparison.changed_records: expected nonnegative int")
    if not isinstance(skipped, int) or skipped < 0:
        raise LedgerError(f"{context}.comparison.skipped_records: expected nonnegative int")
    if skipped != 0:
        raise LedgerError(f"{context}.comparison.skipped_records: accepted run skipped data")

    differences = item.get("accepted_differences", [])
    if not isinstance(differences, list):
        raise LedgerError(f"{context}.accepted_differences: expected list")
    justified_changes = 0
    for difference_index, difference in enumerate(differences):
        difference_context = f"{context}.accepted_differences[{difference_index}]"
        difference_item = require_mapping(difference, difference_context)
        require_nonempty_text(require(difference_item, "kind", difference_context),
                              f"{difference_context}.kind")
        require_nonempty_text(require(difference_item, "item", difference_context),
                              f"{difference_context}.item")
        require_nonempty_text(
            require(difference_item, "rationale", difference_context),
            f"{difference_context}.rationale")
        products = require(difference_item, "products", difference_context)
        if not isinstance(products, list) or not products:
            raise LedgerError(f"{difference_context}.products: expected non-empty list")
        justified_changes += len(products)
    if status != "accepted_with_scientific_tolerance" and changed != justified_changes:
        raise LedgerError(
            f"{context}: changed_records={changed} but justified product changes="
            f"{justified_changes}")

    limitations = require(item, "limitations", context)
    if not isinstance(limitations, list):
        raise LedgerError(f"{context}.limitations: expected list")
    return record_id


def validate_ledger(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        ledger = require_mapping(json.load(stream), str(path))
    if require(ledger, "schema_version", str(path)) != "citlali-validation-ledger-v1":
        raise LedgerError(f"{path}: unsupported schema_version")
    records = require(ledger, "records", str(path))
    if not isinstance(records, list):
        raise LedgerError(f"{path}.records: expected list")

    record_ids = [validate_record(record, index)
                  for index, record in enumerate(records)]
    if len(record_ids) != len(set(record_ids)):
        raise LedgerError(f"{path}: duplicate record_id")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ledger", nargs="?", type=Path,
        default=Path("validation/accepted_runs.json"))
    args = parser.parse_args()
    try:
        count = validate_ledger(args.ledger)
    except (OSError, json.JSONDecodeError, LedgerError) as error:
        print(f"validation ledger invalid: {error}", file=sys.stderr)
        return 1
    print(f"validation ledger valid: records={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
