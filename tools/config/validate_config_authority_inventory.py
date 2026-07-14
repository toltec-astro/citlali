#!/usr/bin/env python3
"""Validate the config authority and provenance migration inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "citlali-config-authority-v2"
EXECUTION_AUTHORITIES = {"typed", "legacy", "mixed", "external"}
ADAPTER_DIRECTIONS = {
    "none",
    "typed-to-legacy",
    "legacy-to-typed",
    "external",
}
MIGRATION_STATUSES = {
    "typed-authoritative",
    "typed-authoritative-with-adapter",
    "mixed-adapter",
    "legacy-authoritative-with-typed-mirror",
    "external-boundary",
}
PROVENANCE_STATUSES = {"missing", "partial", "complete"}
MIGRATION_CONTRACTS = {
    "typed-authoritative": ("typed", "none"),
    "typed-authoritative-with-adapter": ("typed", "typed-to-legacy"),
    "legacy-authoritative-with-typed-mirror": (
        "legacy",
        "legacy-to-typed",
    ),
    "external-boundary": ("external", "external"),
}
REQUIRED_DOMAIN_FIELDS = {
    "id",
    "config_prefixes",
    "typed_owner",
    "loader",
    "execution_authority",
    "legacy_targets",
    "adapter_direction",
    "migration_status",
    "provenance_status",
    "exit_gate",
}


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def string_list(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(nonempty_string(item) for item in value)
    )


def validate(data: Any, repo_root: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["inventory root must be an object"]
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    contract = data.get("contract")
    if not isinstance(contract, dict):
        errors.append("contract must be an object")
    elif contract.get("target_adapter_direction") != (
        "requested_yaml -> typed_config -> legacy_runtime"
    ):
        errors.append(
            "contract.target_adapter_direction must preserve one-way "
            "typed-to-legacy flow"
        )

    domains = data.get("domains")
    if not isinstance(domains, list) or not domains:
        return errors + ["domains must be a non-empty array"]

    ids: set[str] = set()
    prefixes: set[str] = set()
    for index, domain in enumerate(domains):
        label = f"domains[{index}]"
        if not isinstance(domain, dict):
            errors.append(f"{label} must be an object")
            continue
        missing = REQUIRED_DOMAIN_FIELDS - domain.keys()
        if missing:
            errors.append(f"{label} is missing fields: {', '.join(sorted(missing))}")
            continue

        domain_id = domain["id"]
        if not nonempty_string(domain_id):
            errors.append(f"{label}.id must be a non-empty string")
        elif domain_id in ids:
            errors.append(f"duplicate domain id: {domain_id}")
        else:
            ids.add(domain_id)

        if not string_list(domain["config_prefixes"], allow_empty=False):
            errors.append(f"{label}.config_prefixes must be a non-empty string array")
        else:
            for prefix in domain["config_prefixes"]:
                if prefix in prefixes:
                    errors.append(f"duplicate config prefix: {prefix}")
                prefixes.add(prefix)

        for field in ("typed_owner", "loader", "exit_gate"):
            if not nonempty_string(domain[field]):
                errors.append(f"{label}.{field} must be a non-empty string")

        loader = domain["loader"]
        if nonempty_string(loader) and not (repo_root / loader).is_file():
            errors.append(f"{label}.loader does not exist: {loader}")
        if domain["execution_authority"] not in EXECUTION_AUTHORITIES:
            errors.append(f"{label}.execution_authority is invalid")
        if domain["adapter_direction"] not in ADAPTER_DIRECTIONS:
            errors.append(f"{label}.adapter_direction is invalid")
        if domain["migration_status"] not in MIGRATION_STATUSES:
            errors.append(f"{label}.migration_status is invalid")
        if domain["provenance_status"] not in PROVENANCE_STATUSES:
            errors.append(f"{label}.provenance_status is invalid")
        if not string_list(domain["legacy_targets"], allow_empty=True):
            errors.append(f"{label}.legacy_targets must be a string array")

        authority = domain["execution_authority"]
        direction = domain["adapter_direction"]
        targets = domain["legacy_targets"]
        if authority == "external" and direction != "external":
            errors.append(f"{label}: external authority requires an external adapter")
        if authority != "external" and direction == "external":
            errors.append(f"{label}: only external authority may use an external adapter")
        if direction == "typed-to-legacy" and not targets:
            errors.append(f"{label}: typed-to-legacy adapter requires legacy_targets")
        if direction == "legacy-to-typed" and not targets:
            errors.append(f"{label}: legacy-to-typed mirror requires legacy_targets")
        if direction == "none" and targets:
            errors.append(f"{label}: legacy_targets require a declared adapter")
        if authority == "legacy" and direction != "legacy-to-typed":
            errors.append(
                f"{label}: legacy authority requires a legacy-to-typed mirror"
            )
        if direction == "legacy-to-typed" and authority != "legacy":
            errors.append(
                f"{label}: legacy-to-typed mirror requires legacy authority"
            )

        status = domain["migration_status"]
        if status in MIGRATION_CONTRACTS:
            expected_authority, expected_direction = MIGRATION_CONTRACTS[status]
            if authority != expected_authority or direction != expected_direction:
                errors.append(
                    f"{label}: migration status {status!r} requires "
                    f"execution_authority={expected_authority!r} and "
                    f"adapter_direction={expected_direction!r}"
                )

    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inventory",
        nargs="?",
        default="tools/config/config_authority_inventory.json",
    )
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    inventory_path = Path(args.inventory)
    repo_root = Path(args.repo_root).resolve()
    try:
        data = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"config authority inventory: unable to read {inventory_path}: {exc}", file=sys.stderr)
        return 2

    errors = validate(data, repo_root)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    domains = data["domains"]
    counts: dict[str, int] = {}
    for domain in domains:
        status = domain["migration_status"]
        counts[status] = counts.get(status, 0) + 1
    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"config authority inventory: {len(domains)} domains valid ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
