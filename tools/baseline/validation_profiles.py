#!/usr/bin/env python3
"""Validate and inspect the Citlali validation-profile registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .validate_validation_ledger import LedgerError, validate_ledger
except ImportError:
    from validate_validation_ledger import LedgerError, validate_ledger


REGISTRY_SCHEMA_VERSION = "citlali-validation-profile-registry-v1"
SUPPORTED_MODES = {"point", "oof", "beammap", "science"}
SUPPORTED_COMPARATORS = {
    "reduction_products",
    "science_scientific_equivalence",
    "beammap_scientific_equivalence",
}
SUPPORTED_PROVENANCE = {
    "astrometry",
    "beammap",
    "coadd",
    "config_source_manifest",
    "kids_external",
    "mapmaking",
    "noise_products",
    "pointing",
    "polarimetry",
    "post_processing",
    "processed",
    "raw",
}


class RegistryError(ValueError):
    pass


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegistryError(f"{context}: expected object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise RegistryError(f"{context}: expected list")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{context}: expected non-empty string")
    return value


def _unique(items: list[str], context: str) -> None:
    if len(items) != len(set(items)):
        raise RegistryError(f"{context}: duplicate values")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return _mapping(json.load(stream), str(path))


def ledger_records(ledger_path: Path) -> dict[str, dict[str, Any]]:
    try:
        validate_ledger(ledger_path)
    except LedgerError as error:
        raise RegistryError(f"validation ledger is invalid: {error}") from error
    records = load_json(ledger_path)["records"]
    return {record["record_id"]: record for record in records}


def validate_registry(
    registry_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    registry = load_json(registry_path)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise RegistryError(f"{registry_path}: unsupported schema_version")

    active_epoch_id = _text(
        registry.get("active_epoch_id"), f"{registry_path}.active_epoch_id"
    )
    evolution = _mapping(
        registry.get("evolution_policy"), f"{registry_path}.evolution_policy"
    )
    if evolution.get("accepted_snapshots_are_immutable") is not True:
        raise RegistryError(
            f"{registry_path}.evolution_policy: accepted snapshots must be immutable"
        )
    if evolution.get("successor_epoch_required_for_intentional_change") is not True:
        raise RegistryError(
            f"{registry_path}.evolution_policy: intentional changes require a successor epoch"
        )

    epochs = _list(registry.get("epochs"), f"{registry_path}.epochs")
    epoch_ids: list[str] = []
    active_epoch_count = 0
    for index, value in enumerate(epochs):
        context = f"epochs[{index}]"
        epoch = _mapping(value, context)
        epoch_id = _text(epoch.get("epoch_id"), f"{context}.epoch_id")
        status = _text(epoch.get("status"), f"{context}.status")
        if status not in {"active", "superseded"}:
            raise RegistryError(f"{context}.status: unsupported value {status!r}")
        if status == "active":
            active_epoch_count += 1
        _text(epoch.get("established_date"), f"{context}.established_date")
        _text(epoch.get("purpose"), f"{context}.purpose")
        epoch_ids.append(epoch_id)
    _unique(epoch_ids, "epochs.epoch_id")
    if active_epoch_id not in epoch_ids:
        raise RegistryError("active_epoch_id does not name a registered epoch")
    if active_epoch_count != 1:
        raise RegistryError("registry must contain exactly one active epoch")
    active_epoch = next(epoch for epoch in epochs if epoch["epoch_id"] == active_epoch_id)
    if active_epoch["status"] != "active":
        raise RegistryError("active_epoch_id names a non-active epoch")

    records = ledger_records(ledger_path)
    repo_root = registry_path.resolve().parent.parent
    profiles = _list(registry.get("profiles"), f"{registry_path}.profiles")
    profile_ids: list[str] = []
    active_modes: list[str] = []
    for index, value in enumerate(profiles):
        context = f"profiles[{index}]"
        profile = _mapping(value, context)
        profile_id = _text(profile.get("profile_id"), f"{context}.profile_id")
        epoch_id = _text(profile.get("epoch_id"), f"{context}.epoch_id")
        if epoch_id not in epoch_ids:
            raise RegistryError(f"{context}.epoch_id: unknown epoch {epoch_id!r}")
        mode = _text(profile.get("mode"), f"{context}.mode")
        if mode not in SUPPORTED_MODES:
            raise RegistryError(f"{context}.mode: unsupported mode {mode!r}")
        status = _text(profile.get("status"), f"{context}.status")
        if status not in {"active", "superseded"}:
            raise RegistryError(f"{context}.status: unsupported value {status!r}")

        baseline_id = _text(
            profile.get("baseline_record_id"), f"{context}.baseline_record_id"
        )
        if baseline_id not in records:
            raise RegistryError(
                f"{context}.baseline_record_id: unknown ledger record {baseline_id!r}"
            )
        baseline_record = records[baseline_id]
        if baseline_record["mode"] != mode:
            raise RegistryError(
                f"{context}: profile mode {mode!r} differs from baseline mode "
                f"{baseline_record['mode']!r}"
            )
        if baseline_record.get("epoch_id") not in {None, epoch_id}:
            raise RegistryError(
                f"{context}: baseline record names epoch "
                f"{baseline_record['epoch_id']!r}"
            )
        if baseline_record.get("profile_id") not in {None, profile_id}:
            raise RegistryError(
                f"{context}: baseline record names profile "
                f"{baseline_record['profile_id']!r}"
            )

        audit = _mapping(profile.get("audit"), f"{context}.audit")
        _text(audit.get("expected_label"), f"{context}.audit.expected_label")
        required = [
            _text(item, f"{context}.audit.required_provenance")
            for item in _list(
                audit.get("required_provenance"),
                f"{context}.audit.required_provenance",
            )
        ]
        _unique(required, f"{context}.audit.required_provenance")
        unknown_provenance = set(required) - SUPPORTED_PROVENANCE
        if unknown_provenance:
            raise RegistryError(
                f"{context}.audit.required_provenance: unsupported values "
                f"{sorted(unknown_provenance)}"
            )

        config = _mapping(profile.get("config"), f"{context}.config")
        if config.get("policy") != "exact":
            raise RegistryError(f"{context}.config.policy: only 'exact' is supported")
        for item in _list(config.get("ignore_paths", []), f"{context}.config.ignore_paths"):
            _text(item, f"{context}.config.ignore_paths")

        products = _mapping(profile.get("products"), f"{context}.products")
        comparator = _text(products.get("comparator"), f"{context}.products.comparator")
        if comparator not in SUPPORTED_COMPARATORS:
            raise RegistryError(
                f"{context}.products.comparator: unsupported value {comparator!r}"
            )
        if comparator == "reduction_products":
            for number in ("max_array_elements", "atol", "rtol", "frac_floor"):
                if not isinstance(products.get(number), (int, float)):
                    raise RegistryError(f"{context}.products.{number}: expected number")
            for name in ("include", "exclude"):
                for item in _list(products.get(name, []), f"{context}.products.{name}"):
                    _text(item, f"{context}.products.{name}")
            if products.get("strict") is not True:
                raise RegistryError(f"{context}.products.strict: must be true")
        else:
            profile_path = Path(
                _text(products.get("scientific_profile"), f"{context}.products.scientific_profile")
            )
            resolved_profile = (
                profile_path
                if profile_path.is_absolute()
                else repo_root / profile_path
            )
            if not resolved_profile.is_file():
                raise RegistryError(
                    f"{context}.products.scientific_profile: file not found: "
                    f"{resolved_profile}"
                )

        if status == "active" and epoch_id == active_epoch_id:
            active_modes.append(mode)
        profile_ids.append(profile_id)

    _unique(profile_ids, "profiles.profile_id")
    if set(active_modes) != SUPPORTED_MODES or len(active_modes) != len(SUPPORTED_MODES):
        raise RegistryError(
            "active epoch must contain exactly one active profile for each of "
            f"{sorted(SUPPORTED_MODES)}; got {sorted(active_modes)}"
        )
    return registry


def profile_by_id(registry: dict[str, Any], profile_id: str) -> dict[str, Any]:
    matches = [profile for profile in registry["profiles"] if profile["profile_id"] == profile_id]
    if not matches:
        raise RegistryError(f"unknown validation profile {profile_id!r}")
    return matches[0]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("validation/validation_profiles.json"),
    )
    parser.add_argument(
        "--ledger", type=Path, default=Path("validation/accepted_runs.json")
    )
    parser.add_argument("--list", action="store_true", help="List active profiles.")
    args = parser.parse_args(argv)
    try:
        registry = validate_registry(args.registry, args.ledger)
    except (OSError, json.JSONDecodeError, RegistryError) as error:
        print(f"validation profile registry invalid: {error}", file=sys.stderr)
        return 1
    active_epoch = registry["active_epoch_id"]
    active = [
        profile
        for profile in registry["profiles"]
        if profile["status"] == "active" and profile["epoch_id"] == active_epoch
    ]
    print(
        f"validation profile registry valid: epoch={active_epoch} "
        f"profiles={len(active)}"
    )
    if args.list:
        for profile in sorted(active, key=lambda item: item["mode"]):
            print(
                f"{profile['profile_id']}\t{profile['mode']}\t"
                f"{profile['baseline_record_id']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
