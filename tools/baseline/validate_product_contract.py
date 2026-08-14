#!/usr/bin/env python3
"""Validate a Citlali reduction against a versioned product contract."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import re
import struct
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml
import numpy as np

try:
    from astropy.io import fits
    from astropy.table import Table
except Exception:  # pragma: no cover - validation environment dependency
    fits = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

try:
    import netCDF4
except Exception:  # pragma: no cover - validation environment dependency
    netCDF4 = None  # type: ignore[assignment]


SCHEMA_VERSION = "citlali-product-contract-registry-v2"
RESULT_SCHEMA_VERSION = "citlali-product-contract-result-v1"
SCIENCE_MAP_SCHEMA_VERSION = "citlali-science-map-contract-v1"
ARTIFACT_RESULT_SCHEMA_VERSION = "citlali-artifact-contract-result-v1"
CANONICAL_APT_ARTIFACT_CONTRACT_ID = (
    "apt-prod-001-canonical-baseline-apt-v1"
)
CANONICAL_APT_ARTIFACT_CONTRACT_SHA256 = (
    "eb343ced3d4c8f303095b53f3fdca087bb478bd53d675b12958b47df244173b9"
)
CANONICAL_APT_UID_MAX = 9007199254740991
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UINT64_MAX = (1 << 64) - 1
SHA256_REFERENCE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
UTC_TIMESTAMP_RE = re.compile(
    r"^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):"
    r"([0-9]{2})(?:\.([0-9]+))?Z$"
)
SUPPORTED_MODES = {"point", "oof", "science", "beammap"}
SUPPORTED_SCOPES = {
    "reduction",
    "per_array",
    "per_observation",
    "per_observation_array",
}
SUPPORTED_CLASSIFICATIONS = {
    "required",
    "config_conditional",
    "optional_diagnostic",
}
PRODUCT_SUFFIXES = {".fits", ".fit", ".nc", ".nc4", ".cdf", ".csv", ".ecsv"}
OBSERVATION_RE = re.compile(r"^\d+$")
WCS_CARD_RE = re.compile(
    r"^(?:WCSAXES|CRPIX\d+|CRVAL\d+|CDELT\d+|CTYPE\d+|CUNIT\d+|"
    r"PC\d+_\d+|CD\d+_\d+|CROTA\d+|PV\d+_\d+|PS\d+_\d+|"
    r"LONPOLE|LATPOLE|RADESYS|EQUINOX)$"
)
SCIENCE_MAP_PLANES = {
    "geometric_hits_I": ("int64", 64, "count"),
    "contributing_hits_I": ("int64", 64, "count"),
    "coadd_observation_count_I": ("int64", 64, "count"),
    "upstream_eligible_exposure_I": ("float64", -64, "detector s"),
    "retained_exposure_I": ("float64", -64, "detector s"),
    "normalization_support_I": ("uint8", 8, "1"),
    "science_policy_support_I": ("uint8", 8, "1"),
    "science_valid_I": ("uint8", 8, "1"),
}


class ContractError(ValueError):
    pass


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{context}: expected object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{context}: expected list")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{context}: expected non-empty string")
    return value


def _unique(values: list[str], context: str) -> None:
    if len(values) != len(set(values)):
        raise ContractError(f"{context}: duplicate values")


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ContractError(f"non-finite JSON number {value!r} is forbidden")


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_canonical_apt_artifact_contract(
    artifact_id: str, value: Any
) -> dict[str, Any]:
    context = f"artifact_contracts.{artifact_id}"
    artifact = _mapping(value, context)
    if artifact_id != CANONICAL_APT_ARTIFACT_CONTRACT_ID:
        raise ContractError(f"{context}: unsupported artifact contract")
    if artifact.get("artifact_contract_id") != artifact_id:
        raise ContractError(f"{context}.artifact_contract_id: mismatch")
    digest = _canonical_json_sha256(artifact)
    if digest != CANONICAL_APT_ARTIFACT_CONTRACT_SHA256:
        raise ContractError(
            f"{context}: canonical v1 contract/catalog drift "
            f"({digest}; expected {CANONICAL_APT_ARTIFACT_CONTRACT_SHA256})"
        )

    required = _list(artifact.get("required_fields"), f"{context}.required_fields")
    optional = _list(
        artifact.get("optional_extensions"),
        f"{context}.optional_extensions",
    )
    core = _list(artifact.get("core_fields"), f"{context}.core_fields")
    if len(core) != 5 or len(required) != 27 or len(optional) != 20:
        raise ContractError(
            f"{context}: expected exact 5 core, 27 required, and 20 optional fields"
        )
    expected_field_keys = {
        "name",
        "datatype",
        "unit",
        "nullable",
        "authority",
        "authority_reference",
        "nonfinite",
        "registry",
        "description",
        "identity_role",
    }
    names: list[str] = []
    for category, fields in (("required_fields", required),
                             ("optional_extensions", optional)):
        category_names: list[str] = []
        for index, field_value in enumerate(fields):
            field = _mapping(
                field_value, f"{context}.{category}[{index}]"
            )
            if set(field) != expected_field_keys:
                raise ContractError(
                    f"{context}.{category}[{index}]: unexpected field contract keys"
                )
            name = _text(field.get("name"), f"{context}.{category}[{index}].name")
            category_names.append(name)
            names.append(name)
        if category_names != sorted(category_names):
            raise ContractError(f"{context}.{category}: names are not lexical")
        _unique(category_names, f"{context}.{category}.name")
    _unique(names, f"{context} registered field names")
    protected = {
        _text(name, f"{context}.protected_names")
        for name in _list(
            artifact.get("protected_names"), f"{context}.protected_names"
        )
    }
    collision = sorted(protected.intersection(names))
    if collision:
        raise ContractError(
            f"{context}: registered fields collide with protected names {collision}"
        )
    return artifact


def load_registry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        registry = _mapping(
            json.load(
                stream,
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_json_constant,
            ),
            str(path),
        )
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"{path}: unsupported schema_version")

    artifact_contracts = _mapping(
        registry.get("artifact_contracts", {}),
        f"{path}.artifact_contracts",
    )
    for artifact_id, value in artifact_contracts.items():
        _text(artifact_id, "artifact_contracts key")
        _validate_canonical_apt_artifact_contract(artifact_id, value)

    science_map_contracts = _mapping(
        registry.get("science_map_contracts", {}),
        f"{path}.science_map_contracts",
    )
    for contract_id, value in science_map_contracts.items():
        context = f"science_map_contracts.{contract_id}"
        contract = _mapping(value, context)
        if contract.get("schema_version") != SCIENCE_MAP_SCHEMA_VERSION:
            raise ContractError(f"{context}: unsupported schema_version")
        if contract.get("audit_state") != "addressed_pending_reaudit":
            raise ContractError(
                f"{context}.audit_state: must remain addressed_pending_reaudit"
            )
        planes = _list(contract.get("planes"), f"{context}.planes")
        by_name: dict[str, dict[str, Any]] = {}
        for index, plane_value in enumerate(planes):
            plane_context = f"{context}.planes[{index}]"
            plane = _mapping(plane_value, plane_context)
            name = _text(plane.get("name"), f"{plane_context}.name")
            if name in by_name:
                raise ContractError(f"{context}.planes: duplicate name {name!r}")
            by_name[name] = plane
            for key in ("logical_fact", "applicability", "absence_policy"):
                _text(plane.get(key), f"{plane_context}.{key}")
        if set(by_name) != set(SCIENCE_MAP_PLANES):
            raise ContractError(
                f"{context}.planes: expected exactly "
                f"{sorted(SCIENCE_MAP_PLANES)}"
            )
        for name, (dtype, bitpix, unit) in SCIENCE_MAP_PLANES.items():
            plane = by_name[name]
            expected = {"dtype": dtype, "bitpix": bitpix, "unit": unit}
            actual = {key: plane.get(key) for key in expected}
            if actual != expected:
                raise ContractError(
                    f"{context}.planes[{name!r}] type/unit={actual!r}; "
                    f"expected {expected!r}"
                )
        aliases = _mapping(contract.get("aliases"), f"{context}.aliases")
        expected_aliases = {
            "coverage_I": {
                "canonical": "retained_exposure_I",
                "relationship": "bitwise_equal",
                "dtype": "float64",
                "bitpix": -64,
                "unit": "detector s",
                "deprecated": False,
                "validity_authority": False,
            },
            "coverage_bool_I": {
                "canonical": "science_policy_support_I",
                "relationship": "bitwise_equal",
                "dtype": "uint8",
                "bitpix": 8,
                "unit": "1",
                "deprecated": True,
                "validity_authority": False,
            },
        }
        if set(aliases) != set(expected_aliases):
            raise ContractError(
                f"{context}.aliases: expected exactly "
                f"{sorted(expected_aliases)}"
            )
        for alias, expected in expected_aliases.items():
            record = _mapping(aliases.get(alias), f"{context}.aliases.{alias}")
            if any(record.get(key) != value for key, value in expected.items()):
                raise ContractError(
                    f"{context}.aliases.{alias}: does not freeze {expected!r}"
                )
        coefficient = _mapping(
            contract.get("coefficient_contract"),
            f"{context}.coefficient_contract",
        )
        if coefficient.get("product") != "weight_I" or coefficient.get(
            "default_role"
        ) != "nonprecision_normalization_coefficient":
            raise ContractError(
                f"{context}.coefficient_contract: weight_I must default to "
                "a nonprecision normalization coefficient"
            )
        if coefficient.get("precision_status") != "conditional":
            raise ContractError(
                f"{context}.coefficient_contract.precision_status: "
                "must be conditional"
            )
        if coefficient.get("unit") != "1/(signal unit)^2":
            raise ContractError(
                f"{context}.coefficient_contract.unit: unexpected unit"
            )
        precision_conditions = [
            _text(value, f"{context}.coefficient_contract.precision_conditions")
            for value in _list(
                coefficient.get("precision_conditions"),
                f"{context}.coefficient_contract.precision_conditions",
            )
        ]
        if not any("SCI-PTC-001" in value for value in precision_conditions):
            raise ContractError(
                f"{context}.coefficient_contract.precision_conditions: "
                "must retain SCI-PTC-001"
            )
        coadd_precision_status = _text(
            coefficient.get("coadd_precision_status"),
            f"{context}.coefficient_contract.coadd_precision_status",
        )
        if "not established" not in coadd_precision_status:
            raise ContractError(
                f"{context}.coefficient_contract.coadd_precision_status: "
                "must remain not established"
            )
        forbidden_claims = {
            _text(value, f"{context}.coefficient_contract.forbidden_default_claims")
            for value in _list(
                coefficient.get("forbidden_default_claims"),
                f"{context}.coefficient_contract.forbidden_default_claims",
            )
        }
        required_forbidden_claims = {
            "inverse variance",
            "uncertainty",
            "standardized statistical significance",
            "science validity",
        }
        if forbidden_claims != required_forbidden_claims:
            raise ContractError(
                f"{context}.coefficient_contract.forbidden_default_claims: "
                f"expected exactly {sorted(required_forbidden_claims)}"
            )
        for key in (
            "validity_formula",
            "required_companion_policy",
            "jinc_absence_policy",
            "detector_grouping_absence_policy",
            "non_array_grouping_absence_policy",
            "covariance_policy",
        ):
            _text(contract.get(key), f"{context}.{key}")
        parallel_equivalence = _mapping(
            contract.get("parallel_equivalence_policy"),
            f"{context}.parallel_equivalence_policy",
        )
        if parallel_equivalence.get("identity") != (
            "within-scan-exact-scan-farm-2gamma-n-sumabs-v1"
        ):
            raise ContractError(
                f"{context}.parallel_equivalence_policy.identity: "
                "unexpected policy"
            )
        for key in ("within_scan", "scan_farm"):
            _text(
                parallel_equivalence.get(key),
                f"{context}.parallel_equivalence_policy.{key}",
            )

    families = _mapping(registry.get("families"), f"{path}.families")
    check_definitions = _mapping(registry.get("checks", {}), f"{path}.checks")
    for check_id, value in check_definitions.items():
        _text(check_id, "checks key")
        _mapping(value, f"checks.{check_id}")
    required_semantics = {
        "scientific_identity",
        "coordinate_frame",
        "axes",
        "units_policy",
        "indexing_policy",
        "missing_value_policy",
        "failure_policy",
    }
    for family_id, value in families.items():
        _text(family_id, "families key")
        family = _mapping(value, f"families.{family_id}")
        missing = sorted(required_semantics - family.keys())
        if missing:
            raise ContractError(f"families.{family_id}: missing {missing}")
        for key in required_semantics - {"axes"}:
            _text(family[key], f"families.{family_id}.{key}")
        axes = _list(family["axes"], f"families.{family_id}.axes")
        for index, axis in enumerate(axes):
            _text(axis, f"families.{family_id}.axes[{index}]")
        science_map_contract_id = family.get("science_map_contract_id")
        if science_map_contract_id is not None:
            _text(
                science_map_contract_id,
                f"families.{family_id}.science_map_contract_id",
            )
            if science_map_contract_id not in science_map_contracts:
                raise ContractError(
                    f"families.{family_id}.science_map_contract_id: unknown "
                    f"contract {science_map_contract_id!r}"
                )

    contract_sources = _list(registry.get("contracts"), f"{path}.contracts")
    contracts: list[dict[str, Any]] = []
    materialized_contracts: dict[str, dict[str, Any]] = {}
    for source_index, source_value in enumerate(contract_sources):
        source_context = f"contracts[{source_index}]"
        source = _mapping(source_value, source_context)
        extends_contract_id = source.get("extends_contract_id")
        if extends_contract_id is None:
            contract = copy.deepcopy(source)
        else:
            extends_contract_id = _text(
                extends_contract_id, f"{source_context}.extends_contract_id"
            )
            if extends_contract_id not in materialized_contracts:
                raise ContractError(
                    f"{source_context}.extends_contract_id: predecessor "
                    f"{extends_contract_id!r} must appear earlier"
                )
            contract = copy.deepcopy(materialized_contracts[extends_contract_id])
            for key, value in source.items():
                if key not in {"entry_overrides", "extends_contract_id"}:
                    contract[key] = copy.deepcopy(value)
            contract["extends_contract_id"] = extends_contract_id
            overrides = _mapping(
                source.get("entry_overrides", {}),
                f"{source_context}.entry_overrides",
            )
            entries_by_id = {
                entry.get("entry_id"): entry for entry in contract.get("entries", [])
            }
            for entry_id, override_value in overrides.items():
                _text(entry_id, f"{source_context}.entry_overrides key")
                if entry_id not in entries_by_id:
                    raise ContractError(
                        f"{source_context}.entry_overrides: unknown predecessor "
                        f"entry {entry_id!r}"
                    )
                override = _mapping(
                    override_value,
                    f"{source_context}.entry_overrides.{entry_id}",
                )
                entries_by_id[entry_id].update(copy.deepcopy(override))
            additions = _list(
                source.get("entry_additions", []),
                f"{source_context}.entry_additions",
            )
            contract["entries"].extend(copy.deepcopy(additions))
        contract_id = _text(
            contract.get("contract_id"), f"{source_context}.contract_id"
        )
        if contract_id in materialized_contracts:
            raise ContractError(f"contracts.contract_id: duplicate {contract_id!r}")
        materialized_contracts[contract_id] = contract
        contracts.append(contract)
    registry["contracts"] = contracts
    contract_ids: list[str] = []
    profile_ids: list[str] = []
    for contract_index, value in enumerate(contracts):
        context = f"contracts[{contract_index}]"
        contract = _mapping(value, context)
        contract_id = _text(contract.get("contract_id"), f"{context}.contract_id")
        profile_id = _text(contract.get("profile_id"), f"{context}.profile_id")
        mode = _text(contract.get("mode"), f"{context}.mode")
        if mode not in SUPPORTED_MODES:
            raise ContractError(f"{context}.mode: unsupported value {mode!r}")
        arrays = _list(contract.get("arrays"), f"{context}.arrays")
        array_names = [_text(v, f"{context}.arrays") for v in arrays]
        _unique(array_names, f"{context}.arrays")
        entries = _list(contract.get("entries"), f"{context}.entries")
        entry_ids: list[str] = []
        for entry_index, entry_value in enumerate(entries):
            entry_context = f"{context}.entries[{entry_index}]"
            entry = _mapping(entry_value, entry_context)
            entry_id = _text(entry.get("entry_id"), f"{entry_context}.entry_id")
            family_id = _text(entry.get("family_id"), f"{entry_context}.family_id")
            if family_id not in families:
                raise ContractError(
                    f"{entry_context}.family_id: unknown family {family_id!r}"
                )
            scope = _text(entry.get("scope"), f"{entry_context}.scope")
            if scope not in SUPPORTED_SCOPES:
                raise ContractError(f"{entry_context}.scope: unsupported {scope!r}")
            classification = _text(
                entry.get("classification"), f"{entry_context}.classification"
            )
            if classification not in SUPPORTED_CLASSIFICATIONS:
                raise ContractError(
                    f"{entry_context}.classification: unsupported {classification!r}"
                )
            _text(entry.get("condition"), f"{entry_context}.condition")
            required_when = entry.get("required_when")
            if classification == "config_conditional":
                validate_condition_rule(required_when, f"{entry_context}.required_when")
            elif required_when is not None:
                raise ContractError(
                    f"{entry_context}.required_when: only valid for config_conditional entries"
                )
            _text(entry.get("pattern"), f"{entry_context}.pattern")
            check_id = entry.get("check_id")
            if check_id is not None:
                _text(check_id, f"{entry_context}.check_id")
                if check_id not in check_definitions:
                    raise ContractError(
                        f"{entry_context}.check_id: unknown check {check_id!r}"
                    )
            checks = entry.get("checks", {})
            _mapping(checks, f"{entry_context}.checks")
            conditional_checks = _list(
                entry.get("checks_by_config", []),
                f"{entry_context}.checks_by_config",
            )
            for check_index, check_value in enumerate(conditional_checks):
                check_context = (
                    f"{entry_context}.checks_by_config[{check_index}]"
                )
                conditional_check = _mapping(check_value, check_context)
                if set(conditional_check) not in (
                    {"when", "checks"},
                    {"when", "check_id"},
                ):
                    raise ContractError(
                        f"{check_context}: expected when plus exactly one of "
                        "checks or check_id"
                    )
                validate_condition_rule(
                    conditional_check["when"], f"{check_context}.when"
                )
                if "checks" in conditional_check:
                    _mapping(
                        conditional_check["checks"], f"{check_context}.checks"
                    )
                else:
                    conditional_check_id = _text(
                        conditional_check["check_id"],
                        f"{check_context}.check_id",
                    )
                    if conditional_check_id not in check_definitions:
                        raise ContractError(
                            f"{check_context}.check_id: unknown check "
                            f"{conditional_check_id!r}"
                        )
            if "require_matching_config_check" in entry and not isinstance(
                entry["require_matching_config_check"], bool
            ):
                raise ContractError(
                    f"{entry_context}.require_matching_config_check: "
                    "expected boolean"
                )
            required_match_count = entry.get("required_config_check_matches")
            if required_match_count is not None and (
                not isinstance(required_match_count, int)
                or isinstance(required_match_count, bool)
                or required_match_count < 1
            ):
                raise ContractError(
                    f"{entry_context}.required_config_check_matches: "
                    "expected positive integer"
                )
            entry_ids.append(entry_id)
        _unique(entry_ids, f"{context}.entries.entry_id")
        contract_ids.append(contract_id)
        profile_ids.append(profile_id)
    _unique(contract_ids, "contracts.contract_id")
    _unique(profile_ids, "contracts.profile_id")
    if artifact_contracts:
        routed = json.dumps(
            {
                "families": families,
                "checks": check_definitions,
                "contracts": contracts,
            },
            sort_keys=True,
            allow_nan=False,
        )
        if CANONICAL_APT_ARTIFACT_CONTRACT_ID in routed:
            raise ContractError(
                "unactivated canonical APT artifact contract is referenced "
                "by a reduction family/check/contract"
            )
    return registry


def contract_by_id(registry: dict[str, Any], contract_id: str) -> dict[str, Any]:
    matches = [
        contract
        for contract in registry["contracts"]
        if contract["contract_id"] == contract_id
    ]
    if len(matches) != 1:
        raise ContractError(f"unknown product contract {contract_id!r}")
    return matches[0]


def artifact_contract_by_id(
    registry: dict[str, Any], artifact_contract_id: str
) -> dict[str, Any]:
    artifact_contracts = _mapping(
        registry.get("artifact_contracts", {}), "artifact_contracts"
    )
    if artifact_contract_id not in artifact_contracts:
        raise ContractError(
            f"unknown artifact contract {artifact_contract_id!r}"
        )
    return _validate_canonical_apt_artifact_contract(
        artifact_contract_id, artifact_contracts[artifact_contract_id]
    )


def validate_condition_rule(value: Any, context: str) -> None:
    rule = _mapping(value, context)
    if set(rule) == {"path", "equals"}:
        _text(rule["path"], f"{context}.path")
        return
    if set(rule) == {"all"}:
        children = _list(rule["all"], f"{context}.all")
        if not children:
            raise ContractError(f"{context}.all: expected at least one rule")
        for index, child in enumerate(children):
            validate_condition_rule(child, f"{context}.all[{index}]")
        return
    raise ContractError(
        f"{context}: expected {{path, equals}} or {{all}} condition rule"
    )


def config_value(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ContractError(f"low-level config does not contain {path!r}")
        value = value[part]
    return value


def evaluate_condition(rule: dict[str, Any], config: dict[str, Any]) -> bool:
    if "all" in rule:
        return all(evaluate_condition(child, config) for child in rule["all"])
    return config_value(config, rule["path"]) == rule["equals"]


def find_lowlevel_config(reduction: Path) -> Path:
    matches = sorted(reduction.glob("citlali_o*.yaml"))
    if len(matches) != 1:
        raise ContractError(
            f"expected exactly one citlali_o*.yaml in {reduction}; found {len(matches)}"
        )
    return matches[0]


def load_lowlevel_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return _mapping(yaml.safe_load(stream), str(path))


def observation_ids(reduction: Path) -> list[str]:
    return sorted(
        child.name
        for child in reduction.iterdir()
        if child.is_dir() and OBSERVATION_RE.fullmatch(child.name)
    )


def entry_contexts(
    entry: dict[str, Any], observations: list[str], arrays: list[str]
) -> list[dict[str, str]]:
    scope = entry["scope"]
    if scope == "reduction":
        return [{}]
    if scope == "per_array":
        return [{"array": array} for array in arrays]
    if scope == "per_observation":
        return [{"obs": obs} for obs in observations]
    return [
        {"obs": obs, "array": array}
        for obs in observations
        for array in arrays
    ]


def expanded_names(values: list[str], arrays: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if "{array}" in value:
            result.extend(value.format(array=array) for array in arrays)
        else:
            result.append(value)
    return result


def missing_patterns(actual: list[str], required: list[str]) -> list[str]:
    return [
        pattern
        for pattern in required
        if not any(fnmatch.fnmatchcase(value, pattern) for value in actual)
    ]


def merge_checks(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_checks(result[key], value)
        elif isinstance(value, list) and isinstance(result.get(key), list):
            result[key] = list(result[key])
            for item in value:
                if item not in result[key]:
                    result[key].append(copy.deepcopy(item))
        else:
            result[key] = copy.deepcopy(value)
    return result


def _named_hdu(
    hdus: Any, normalized_names: list[str], name: str
) -> Any | None:
    normalized_name = name.casefold()
    if normalized_name not in normalized_names:
        return None
    return hdus[normalized_names.index(normalized_name)]


def _wcs_cards(header: Any) -> dict[str, Any]:
    return {
        key: header[key]
        for key in header
        if WCS_CARD_RE.fullmatch(str(key))
    }


def _bitwise_equal(lhs: Any, rhs: Any) -> bool:
    lhs_array = np.ascontiguousarray(lhs)
    rhs_array = np.ascontiguousarray(rhs)
    return (
        lhs_array.shape == rhs_array.shape
        and lhs_array.dtype == rhs_array.dtype
        and lhs_array.tobytes(order="C") == rhs_array.tobytes(order="C")
    )


def validate_fits(path: Path, checks: dict[str, Any]) -> list[str]:
    if fits is None:
        return ["astropy.io.fits is unavailable"]
    errors: list[str] = []
    try:
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hdus:
            names = [hdu.name for hdu in hdus]
            normalized_names = [name.casefold() for name in names]
            minimum = int(checks.get("min_hdus", 1))
            if len(hdus) < minimum:
                errors.append(f"has {len(hdus)} HDUs; requires at least {minimum}")
            required_names = checks.get("required_extnames", [])
            missing = [
                name
                for name in required_names
                if name.casefold() not in normalized_names
            ]
            if missing:
                errors.append(f"missing FITS extensions {missing}")
            forbidden = [
                name
                for name in checks.get("forbidden_extnames", [])
                if name.casefold() in normalized_names
            ]
            if forbidden:
                errors.append(f"forbidden FITS extensions present {forbidden}")
            forbidden_prefixes = [
                prefix
                for prefix in checks.get("forbidden_extname_prefixes", [])
                if any(
                    name.startswith(prefix.casefold())
                    for name in normalized_names
                )
            ]
            if forbidden_prefixes:
                errors.append(
                    "forbidden FITS extension prefixes present "
                    f"{forbidden_prefixes}"
                )
            missing_prefixes = missing_patterns(
                normalized_names,
                [
                    f"{prefix.casefold()}*"
                    for prefix in checks.get("required_extname_prefixes", [])
                ],
            )
            if missing_prefixes:
                errors.append(f"missing FITS extension prefixes {missing_prefixes}")
            primary_bunit = checks.get("primary_bunit")
            if primary_bunit is not None and hdus[0].header.get("BUNIT") != primary_bunit:
                errors.append(
                    f"primary BUNIT={hdus[0].header.get('BUNIT')!r}; "
                    f"expected {primary_bunit!r}"
                )
            for name, expected in checks.get("ext_bunits", {}).items():
                normalized_name = name.casefold()
                if normalized_name in normalized_names:
                    actual = hdus[normalized_names.index(normalized_name)].header.get("BUNIT")
                    if actual != expected:
                        errors.append(f"{name} BUNIT={actual!r}; expected {expected!r}")
            for name, expected in checks.get("required_ext_bitpix", {}).items():
                hdu = _named_hdu(hdus, normalized_names, name)
                if hdu is None:
                    continue
                actual = hdu.header.get("BITPIX")
                if actual != expected:
                    errors.append(f"{name} BITPIX={actual!r}; expected {expected!r}")
            for name, expected in checks.get("required_ext_dtypes", {}).items():
                hdu = _named_hdu(hdus, normalized_names, name)
                if hdu is None:
                    continue
                if hdu.data is None:
                    errors.append(f"{name} contains no pixel data")
                    continue
                actual = np.asarray(hdu.data).dtype.name
                if actual != expected:
                    errors.append(f"{name} dtype={actual!r}; expected {expected!r}")
            for name, expected_headers in checks.get(
                "required_ext_headers", {}
            ).items():
                hdu = _named_hdu(hdus, normalized_names, name)
                if hdu is None:
                    continue
                for keyword, expected in expected_headers.items():
                    actual = hdu.header.get(keyword)
                    if actual != expected:
                        errors.append(
                            f"{name} {keyword}={actual!r}; expected {expected!r}"
                        )
            for name, required_headers in checks.get(
                "required_ext_headers_present", {}
            ).items():
                hdu = _named_hdu(hdus, normalized_names, name)
                if hdu is None:
                    continue
                missing_headers = [
                    keyword
                    for keyword in required_headers
                    if keyword not in hdu.header
                ]
                if missing_headers:
                    errors.append(
                        f"{name} missing required FITS headers "
                        f"{missing_headers}"
                    )
            for keyword, extnames in checks.get(
                "same_ext_header_values", {}
            ).items():
                values: list[tuple[str, Any]] = []
                for name in extnames:
                    hdu = _named_hdu(hdus, normalized_names, name)
                    if hdu is None:
                        continue
                    if keyword not in hdu.header:
                        errors.append(
                            f"{name} missing {keyword} required for "
                            "cross-HDU header equality"
                        )
                        continue
                    values.append((name, hdu.header[keyword]))
                if values and any(
                    value != values[0][1] for _, value in values[1:]
                ):
                    errors.append(
                        f"cross-HDU {keyword} values differ: {values}"
                    )
            for name, expected_headers in checks.get(
                "required_ext_header_one_of", {}
            ).items():
                hdu = _named_hdu(hdus, normalized_names, name)
                if hdu is None:
                    continue
                for keyword, expected_values in expected_headers.items():
                    actual = hdu.header.get(keyword)
                    if actual not in expected_values:
                        errors.append(
                            f"{name} {keyword}={actual!r}; expected one of "
                            f"{expected_values!r}"
                        )
            for prefix, expected_headers in checks.get(
                "required_prefix_headers", {}
            ).items():
                matching_hdus = [
                    hdu
                    for name, hdu in zip(normalized_names, hdus)
                    if name.startswith(prefix.casefold())
                ]
                for hdu in matching_hdus:
                    for keyword, expected in expected_headers.items():
                        actual = hdu.header.get(keyword)
                        if actual != expected:
                            errors.append(
                                f"{hdu.name} {keyword}={actual!r}; "
                                f"expected {expected!r}"
                            )
            for prefix, expected_headers in checks.get(
                "required_prefix_header_one_of", {}
            ).items():
                matching_hdus = [
                    hdu
                    for name, hdu in zip(normalized_names, hdus)
                    if name.startswith(prefix.casefold())
                ]
                for hdu in matching_hdus:
                    for keyword, expected_values in expected_headers.items():
                        actual = hdu.header.get(keyword)
                        if actual not in expected_values:
                            errors.append(
                                f"{hdu.name} {keyword}={actual!r}; expected "
                                f"one of {expected_values!r}"
                            )
            for name, forbidden_headers in checks.get(
                "forbidden_ext_headers", {}
            ).items():
                hdu = _named_hdu(hdus, normalized_names, name)
                if hdu is None:
                    continue
                present = [
                    keyword
                    for keyword in forbidden_headers
                    if keyword in hdu.header
                ]
                if present:
                    errors.append(
                        f"{name} has forbidden FITS headers {present}"
                    )
            for name in checks.get("binary_extnames", []):
                hdu = _named_hdu(hdus, normalized_names, name)
                if hdu is None:
                    continue
                if hdu.data is None:
                    errors.append(f"{name} contains no pixel data")
                    continue
                values = np.asarray(hdu.data)
                if not np.all(np.isfinite(values)) or not np.all(
                    (values == 0) | (values == 1)
                ):
                    errors.append(f"{name} is not a binary 0/1 mask")
            for canonical, alias in checks.get("exact_aliases", {}).items():
                canonical_hdu = _named_hdu(hdus, normalized_names, canonical)
                alias_hdu = _named_hdu(hdus, normalized_names, alias)
                if canonical_hdu is None or alias_hdu is None:
                    continue
                if canonical_hdu.data is None or alias_hdu.data is None:
                    errors.append(
                        f"{alias} and {canonical} require pixel data for alias check"
                    )
                elif not _bitwise_equal(canonical_hdu.data, alias_hdu.data):
                    errors.append(
                        f"{alias} is not bitwise equal to canonical {canonical}"
                    )
            shape_names = checks.get("same_shape_extnames", [])
            shape_hdus = [
                (name, _named_hdu(hdus, normalized_names, name))
                for name in shape_names
            ]
            available_shapes = [
                (name, tuple(hdu.data.shape) if hdu.data is not None else None)
                for name, hdu in shape_hdus
                if hdu is not None
            ]
            if available_shapes:
                expected_shape = available_shapes[0][1]
                mismatched_shapes = [
                    (name, shape)
                    for name, shape in available_shapes[1:]
                    if shape != expected_shape
                ]
                if mismatched_shapes:
                    errors.append(
                        "cross-HDU shapes differ: "
                        f"reference={available_shapes[0]} mismatches={mismatched_shapes}"
                    )
            wcs_names = checks.get("same_wcs_extnames", [])
            wcs_hdus = [
                (name, _named_hdu(hdus, normalized_names, name))
                for name in wcs_names
            ]
            available_wcs = [
                (name, _wcs_cards(hdu.header))
                for name, hdu in wcs_hdus
                if hdu is not None
            ]
            if available_wcs:
                missing_wcs = [
                    name for name, cards in available_wcs if not cards
                ]
                if missing_wcs:
                    errors.append(
                        f"FITS extensions have no WCS-card inventory: {missing_wcs}"
                    )
                expected_wcs = available_wcs[0][1]
                mismatched_wcs = [
                    name
                    for name, cards in available_wcs[1:]
                    if cards != expected_wcs
                ]
                if mismatched_wcs:
                    errors.append(
                        "cross-HDU WCS cards differ: "
                        f"reference={available_wcs[0][0]!r} "
                        f"mismatches={mismatched_wcs}"
                    )
            data_hdu = next((hdu for hdu in hdus if hdu.header.get("NAXIS", 0) > 0), None)
            if data_hdu is None:
                errors.append("contains no data HDU")
            else:
                expected_axes = checks.get("axis_types", [])
                actual_axes = [
                    data_hdu.header.get(f"CTYPE{index}")
                    for index in range(1, len(expected_axes) + 1)
                ]
                if expected_axes and actual_axes != expected_axes:
                    errors.append(f"CTYPE axes={actual_axes!r}; expected {expected_axes!r}")
                expected_units = checks.get("axis_units", [])
                normalized_expected_units = [
                    "" if value is None else value for value in expected_units
                ]
                actual_units = [
                    data_hdu.header.get(f"CUNIT{index}")
                    for index in range(1, len(expected_units) + 1)
                ]
                if expected_units and actual_units != normalized_expected_units:
                    errors.append(
                        f"CUNIT axes={actual_units!r}; "
                        f"expected {normalized_expected_units!r}"
                    )
    except Exception as error:
        errors.append(f"cannot read FITS: {error}")
    return errors


def validate_netcdf(
    path: Path, checks: dict[str, Any], arrays: list[str]
) -> list[str]:
    if netCDF4 is None:
        return ["netCDF4 is unavailable"]
    errors: list[str] = []
    try:
        with netCDF4.Dataset(path) as dataset:
            dimensions = list(dataset.dimensions)
            variables = list(dataset.variables)
            required_dimensions = expanded_names(
                checks.get("required_dimensions", []), arrays
            )
            missing_dims = missing_patterns(dimensions, required_dimensions)
            if missing_dims:
                errors.append(f"missing NetCDF dimensions {missing_dims}")
            required_variables = expanded_names(
                checks.get("required_variables", []), arrays
            )
            missing_vars = missing_patterns(variables, required_variables)
            if missing_vars:
                errors.append(f"missing NetCDF variables {missing_vars}")
            scalar_values: dict[str, Any] = {}

            def scalar_value(name: str) -> Any | None:
                if name in scalar_values:
                    return scalar_values[name]
                if name not in dataset.variables:
                    errors.append(
                        f"missing NetCDF scalar variable {name!r}"
                    )
                    return None
                values = dataset.variables[name][...]
                if values.size != 1:
                    errors.append(
                        f"NetCDF variable {name!r} has {values.size} values; "
                        "expected one"
                    )
                    return None
                actual = values.reshape(-1)[0]
                if hasattr(actual, "item"):
                    actual = actual.item()
                if isinstance(actual, bytes):
                    actual = actual.decode("utf-8")
                scalar_values[name] = actual
                return actual

            for name, expected in checks.get("scalar_equals", {}).items():
                actual = scalar_value(name)
                if actual is None:
                    continue
                if actual != expected:
                    errors.append(
                        f"NetCDF scalar {name!r}={actual!r}; "
                        f"expected {expected!r}"
                    )
            for name, expected_values in checks.get(
                "scalar_one_of", {}
            ).items():
                actual = scalar_value(name)
                if actual is None:
                    continue
                if actual not in expected_values:
                    errors.append(
                        f"NetCDF scalar {name!r}={actual!r}; "
                        f"expected one of {expected_values!r}"
                    )
            for name, variants in checks.get(
                "required_variables_by_scalar", {}
            ).items():
                actual = scalar_value(name)
                if actual is None:
                    continue
                conditional = expanded_names(
                    variants.get(actual, []), arrays
                )
                missing_conditional = missing_patterns(
                    variables, conditional
                )
                if missing_conditional:
                    errors.append(
                        f"NetCDF scalar {name!r}={actual!r} requires "
                        f"variables {missing_conditional}"
                    )
            for name in checks.get("positive_dimensions", []):
                matches = [
                    value
                    for value in dimensions
                    if fnmatch.fnmatchcase(value, name)
                ]
                if not matches or any(len(dataset.dimensions[value]) <= 0 for value in matches):
                    errors.append(f"NetCDF dimension {name!r} is absent or empty")
    except Exception as error:
        errors.append(f"cannot read NetCDF: {error}")
    return errors


class _CanonicalAptCursor:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.index = 0

    def peek(self) -> str | None:
        if self.index >= len(self.lines):
            return None
        return self.lines[self.index]

    def take(self, context: str) -> str:
        if self.index >= len(self.lines):
            raise ContractError(f"canonical APT ended before {context}")
        value = self.lines[self.index]
        self.index += 1
        return value

    def expect(self, expected: str) -> None:
        actual = self.take(expected)
        if actual != expected:
            raise ContractError(
                f"canonical APT expected {expected!r}; found {actual!r}"
            )


def canonical_frame(label: str, datatype: str, payload: str) -> bytes:
    label_bytes = label.encode("utf-8")
    datatype_bytes = datatype.encode("utf-8")
    payload_bytes = payload.encode("utf-8")
    return (
        b"F"
        + str(len(label_bytes)).encode("ascii")
        + b":"
        + label_bytes
        + b"T"
        + str(len(datatype_bytes)).encode("ascii")
        + b":"
        + datatype_bytes
        + b"V"
        + str(len(payload_bytes)).encode("ascii")
        + b":"
        + payload_bytes
        + b";"
    )


def _canonical_text(value: str) -> bool:
    for character in value:
        code_point = ord(character)
        if (
            code_point == 0
            or code_point == 0x7F
            or 0x80 <= code_point <= 0x9F
            or code_point in {0x85, 0x2028, 0x2029}
            or 0xFDD0 <= code_point <= 0xFDEF
            or code_point & 0xFFFF in {0xFFFE, 0xFFFF}
            or (code_point < 0x20 and code_point != 0x09)
        ):
            return False
    return True


def _require_canonical_text(label: str, value: str, allow_empty: bool = False) -> None:
    if (not allow_empty and not value) or not _canonical_text(value):
        raise ContractError(
            f"canonical APT requires valid single-line UTF-8 text for {label}"
        )


def _yaml_quote(value: str) -> str:
    _require_canonical_text("YAML value", value, allow_empty=True)
    escaped: list[str] = ['"']
    for character in value:
        if character == "\\":
            escaped.append("\\\\")
        elif character == '"':
            escaped.append('\\"')
        elif character == "\n":
            escaped.append("\\n")
        elif character == "\r":
            escaped.append("\\r")
        elif character == "\t":
            escaped.append("\\t")
        elif ord(character) < 0x20:
            escaped.append(f"\\u00{ord(character):02x}")
        else:
            escaped.append(character)
    escaped.append('"')
    return "".join(escaped)


def _yaml_unquote(value: str) -> str:
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        raise ContractError(
            "canonical APT metadata string is not double quoted"
        )
    result: list[str] = []
    index = 1
    while index < len(value) - 1:
        character = value[index]
        if character != "\\":
            result.append(character)
            index += 1
            continue
        index += 1
        if index >= len(value) - 1:
            raise ContractError("truncated canonical APT YAML escape")
        escaped = value[index]
        if escaped == "\\":
            result.append("\\")
        elif escaped == '"':
            result.append('"')
        elif escaped == "n":
            result.append("\n")
        elif escaped == "r":
            result.append("\r")
        elif escaped == "t":
            result.append("\t")
        elif escaped == "u":
            digits = value[index + 1 : index + 5]
            if len(digits) != 4 or not digits.startswith("00") or not re.fullmatch(
                r"[0-9A-Fa-f]{4}", digits
            ):
                raise ContractError(
                    "canonical APT supports only byte-sized YAML escapes"
                )
            result.append(chr(int(digits, 16)))
            index += 4
        else:
            raise ContractError("unsupported canonical APT YAML escape")
        index += 1
    decoded = "".join(result)
    if _yaml_quote(decoded) != value:
        raise ContractError("canonical APT YAML string token is noncanonical")
    return decoded


def _take_yaml(cursor: _CanonicalAptCursor, prefix: str) -> str:
    line = cursor.take(prefix)
    if not line.startswith(prefix):
        raise ContractError(
            f"canonical APT expected metadata prefix {prefix!r}; found {line!r}"
        )
    return _yaml_unquote(line[len(prefix) :])


def _parse_exact_int64(value: str, label: str) -> int:
    if not re.fullmatch(r"0|-?[1-9][0-9]*", value):
        raise ContractError(f"invalid canonical exact int64 {label}: {value!r}")
    result = int(value)
    if result < INT64_MIN or result > INT64_MAX:
        raise ContractError(f"canonical int64 {label} is out of range")
    return result


def _take_int64(cursor: _CanonicalAptCursor, prefix: str, label: str) -> int:
    line = cursor.take(prefix)
    if not line.startswith(prefix):
        raise ContractError(
            f"canonical APT expected metadata prefix {prefix!r}; found {line!r}"
        )
    return _parse_exact_int64(line[len(prefix) :], label)


def _parse_exact_bool(value: str, label: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ContractError(f"invalid canonical boolean {label}: {value!r}")


def _take_bool(cursor: _CanonicalAptCursor, prefix: str, label: str) -> bool:
    line = cursor.take(prefix)
    if not line.startswith(prefix):
        raise ContractError(
            f"canonical APT expected metadata prefix {prefix!r}; found {line!r}"
        )
    return _parse_exact_bool(line[len(prefix) :], label)


def _format_float64(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if math.copysign(1.0, value) < 0 else "inf"
    return format(value, ".17g")


def _parse_float64(value: str, label: str) -> float:
    if value == "nan":
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return -float("inf")
    if not re.fullmatch(
        r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?|0?\.[0-9]+)(?:e[+-]?[0-9]+)?",
        value,
    ):
        raise ContractError(f"invalid canonical float64 {label}: {value!r}")
    result = float(value)
    if _format_float64(result) != value:
        raise ContractError(f"noncanonical float64 token {label}: {value!r}")
    return result


def _csv_quote(value: str) -> str:
    _require_canonical_text("CSV string", value)
    return '"' + value.replace('"', '""') + '"'


def _parse_csv_line(line: str) -> list[tuple[str, bool, str]]:
    cells: list[tuple[str, bool, str]] = []
    index = 0
    while True:
        start = index
        quoted = False
        value: list[str] = []
        if index < len(line) and line[index] == '"':
            quoted = True
            index += 1
            closed = False
            while index < len(line):
                if line[index] != '"':
                    value.append(line[index])
                    index += 1
                elif index + 1 < len(line) and line[index + 1] == '"':
                    value.append('"')
                    index += 2
                else:
                    index += 1
                    closed = True
                    break
            if not closed:
                raise ContractError("unterminated canonical APT CSV cell")
            if index < len(line) and line[index] != ",":
                raise ContractError(
                    "characters follow quoted canonical APT CSV cell"
                )
        else:
            while index < len(line) and line[index] != ",":
                if line[index] == '"':
                    raise ContractError(
                        "quote inside unquoted canonical APT CSV cell"
                    )
                value.append(line[index])
                index += 1
        token = line[start:index]
        decoded = "".join(value)
        if quoted and _csv_quote(decoded) != token:
            raise ContractError("canonical APT CSV string token is noncanonical")
        cells.append((decoded, quoted, token))
        if index == len(line):
            break
        index += 1
        if index == len(line):
            cells.append(("", False, ""))
            break
    return cells


def _valid_utc_timestamp(value: str) -> bool:
    match = UTC_TIMESTAMP_RE.fullmatch(value)
    if match is None:
        return False
    year, month, day_value, hour, minute, second = (
        int(match.group(index)) for index in range(1, 7)
    )
    if year == 0 or hour > 23 or minute > 59 or second > 59:
        return False
    try:
        date(year, month, day_value)
    except ValueError:
        return False
    return True


def _expected_canonical_apt_columns(
    document: dict[str, Any], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    columns = [
        {
            "name": field["name"],
            "datatype": field["datatype"],
            "unit": field["unit"],
            "description": field["description"],
        }
        for field in contract["core_fields"]
    ]
    columns.extend(
        {
            "name": field["name"],
            "datatype": field["datatype"],
            "unit": field["unit"],
            "description": field["description"],
        }
        for field in sorted(document["registered_fields"], key=lambda item: item["name"])
    )
    return columns


def _parse_canonical_apt_v1_bytes(
    artifact_bytes: bytes, contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    if not artifact_bytes or not artifact_bytes.endswith(b"\n"):
        raise ContractError("canonical APT ECSV requires a final LF")
    if b"\r" in artifact_bytes:
        raise ContractError("canonical APT ECSV rejects CR/CRLF")
    try:
        text = artifact_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ContractError(f"canonical APT ECSV is not valid UTF-8: {error}") from error
    lines = text[:-1].split("\n")
    cursor = _CanonicalAptCursor(lines)
    cursor.expect("# %ECSV 1.0")
    cursor.expect("# ---")
    cursor.expect("# datatype:")
    declared_columns: list[dict[str, Any]] = []
    while cursor.peek() is not None and cursor.peek().startswith("# - name: "):
        column = {
            "name": _take_yaml(cursor, "# - name: "),
            "datatype": _take_yaml(cursor, "#   datatype: "),
            "unit": "",
        }
        if cursor.peek() is not None and cursor.peek().startswith("#   unit: "):
            column["unit"] = _take_yaml(cursor, "#   unit: ")
        column["description"] = _take_yaml(cursor, "#   description: ")
        declared_columns.append(column)

    cursor.expect("# meta:")
    cursor.expect("#   canonical_apt_v1:")
    declared_schema = _take_yaml(cursor, "#     schema_version: ")
    profile = _take_yaml(cursor, "#     profile: ")
    field_registry = _take_yaml(cursor, "#     field_registry: ")
    declared_framing = _take_yaml(cursor, "#     framing_encoding: ")
    declared_semantic_scope = _take_yaml(cursor, "#     semantic_scope: ")
    declared_semantic = _take_yaml(cursor, "#     semantic_sha256: ")
    declared_envelope_scope = _take_yaml(cursor, "#     envelope_scope: ")
    declared_envelope = _take_yaml(cursor, "#     envelope_sha256: ")
    declared_transport_scope = _take_yaml(
        cursor, "#     byte_transport_scope: "
    )
    envelope = {
        "occurrence": _take_yaml(cursor, "#     occurrence: "),
        "event_reference": _take_yaml(cursor, "#     event_reference: "),
        "output_role": _take_yaml(cursor, "#     output_role: "),
        "producer": _take_yaml(cursor, "#     producer: "),
        "software_revision": _take_yaml(cursor, "#     software_revision: "),
        "configuration_reference": _take_yaml(
            cursor, "#     configuration_reference: "
        ),
        "event_time_utc": _take_yaml(cursor, "#     event_time_utc: "),
    }
    cursor.expect("#     scientific_context:")
    context = {
        "project_id": _take_yaml(cursor, "#       project_id: "),
        "source_name": _take_yaml(cursor, "#       source_name: "),
        "observation_time_utc": _take_yaml(
            cursor, "#       observation_time_utc: "
        ),
        "coordinate_frame": _take_yaml(cursor, "#       coordinate_frame: "),
    }
    cursor.expect("#     observation:")
    observation = {
        "observation": _take_int64(
            cursor, "#       observation: ", "observation"
        ),
        "subobservation": _take_int64(
            cursor, "#       subobservation: ", "subobservation"
        ),
        "scan": _take_int64(cursor, "#       scan: ", "scan"),
    }
    cursor.expect("#     raw_manifest:")
    raw_inputs: list[dict[str, Any]] = []
    while cursor.peek() is not None and cursor.peek().startswith(
        "#       - network: "
    ):
        raw_inputs.append(
            {
                "network": _take_int64(
                    cursor, "#       - network: ", "raw network"
                ),
                "interface": _take_yaml(cursor, "#         interface: "),
                "channel_count": _take_int64(
                    cursor, "#         channel_count: ", "raw channel_count"
                ),
            }
        )
    cursor.expect("#     registered_fields:")
    registered_fields: list[dict[str, Any]] = []
    while cursor.peek() is not None and cursor.peek().startswith(
        "#       - name: "
    ):
        registered_fields.append(
            {
                "name": _take_yaml(cursor, "#       - name: "),
                "datatype": _take_yaml(cursor, "#         datatype: "),
                "unit": _take_yaml(cursor, "#         unit: "),
                "nullable": _take_bool(
                    cursor, "#         nullable: ", "field nullable"
                ),
                "authority": _take_yaml(cursor, "#         authority: "),
                "authority_reference": _take_yaml(
                    cursor, "#         authority_reference: "
                ),
                "nonfinite": _take_yaml(cursor, "#         nonfinite: "),
                "registry": _take_yaml(cursor, "#         registry: "),
                "description": _take_yaml(cursor, "#         description: "),
                "identity_role": _take_yaml(
                    cursor, "#         identity_role: "
                ),
            }
        )
    cursor.expect("#     null_cell: \"unquoted-empty-v1\"")
    cursor.expect("#     string_cell: \"quoted-utf8-single-line-v1\"")
    cursor.expect("# delimiter: \",\"")
    cursor.expect("# schema: \"astropy-2.0\"")

    document: dict[str, Any] = {
        "profile": profile,
        "field_registry": field_registry,
        "envelope": envelope,
        "context": context,
        "observation": observation,
        "raw_inputs": raw_inputs,
        "registered_fields": registered_fields,
        "rows": [],
    }
    if (
        declared_schema != contract["schema_version"]
        or declared_framing != contract["framing_encoding"]
        or declared_semantic_scope != contract["semantic_scope"]
        or declared_envelope_scope != contract["envelope_scope"]
        or declared_transport_scope != contract["byte_transport_scope"]
    ):
        raise ContractError("canonical APT metadata scope/schema mismatch")

    expected_columns = _expected_canonical_apt_columns(document, contract)
    expected_declared_columns = [
        {
            "name": column["name"],
            "datatype": column["datatype"],
            "unit": "" if column["unit"] == "N/A" else column["unit"],
            "description": column["description"],
        }
        for column in expected_columns
    ]
    if declared_columns != expected_declared_columns:
        raise ContractError("canonical APT ECSV column contract mismatch")

    csv_header = cursor.take("CSV header")
    header_cells = _parse_csv_line(csv_header)
    expected_names = [column["name"] for column in expected_columns]
    if [cell[0] for cell in header_cells] != expected_names or any(
        cell[1] for cell in header_cells
    ):
        raise ContractError("canonical APT CSV header name/order mismatch")

    fields = sorted(registered_fields, key=lambda item: item["name"])
    while cursor.peek() is not None:
        line = cursor.take("canonical APT data row")
        if not line:
            raise ContractError("blank canonical APT ECSV row")
        cells = _parse_csv_line(line)
        if len(cells) != len(expected_names):
            raise ContractError("canonical APT row has wrong field count")
        for index in range(5):
            if cells[index][1]:
                raise ContractError("canonical APT core numeric cell is quoted")
        row: dict[str, Any] = {
            "uid": _parse_exact_int64(cells[0][0], "uid"),
            "tone_freq": _parse_float64(cells[1][0], "tone_freq"),
            "array": _parse_exact_int64(cells[2][0], "array"),
            "nw": _parse_exact_int64(cells[3][0], "nw"),
            "kids_tone": _parse_exact_int64(cells[4][0], "kids_tone"),
            "fields": {},
        }
        for index, field in enumerate(fields, start=5):
            value, quoted, _token = cells[index]
            if not value and not quoted:
                row["fields"][field["name"]] = None
            elif field["datatype"] == "int64":
                if quoted:
                    raise ContractError(
                        f"canonical APT int64 field {field['name']} is quoted"
                    )
                row["fields"][field["name"]] = _parse_exact_int64(
                    value, field["name"]
                )
            elif field["datatype"] == "float64":
                if quoted:
                    raise ContractError(
                        f"canonical APT float64 field {field['name']} is quoted"
                    )
                row["fields"][field["name"]] = _parse_float64(
                    value, field["name"]
                )
            elif field["datatype"] == "bool":
                if quoted or value not in {"True", "False"}:
                    raise ContractError(
                        f"canonical APT bool field {field['name']} is invalid"
                    )
                row["fields"][field["name"]] = value == "True"
            elif field["datatype"] == "string":
                if not quoted or not value:
                    raise ContractError(
                        f"canonical APT string field {field['name']} is invalid"
                    )
                row["fields"][field["name"]] = value
            else:
                raise ContractError(
                    f"unsupported canonical APT field type {field['datatype']!r}"
                )
        document["rows"].append(row)

    declared = {
        "semantic_sha256": declared_semantic,
        "envelope_sha256": declared_envelope,
    }
    _validate_canonical_apt_document(document, contract)
    computed = _canonical_apt_digests(document, contract)
    if (
        not SHA256_REFERENCE_RE.fullmatch(declared_semantic)
        or not SHA256_REFERENCE_RE.fullmatch(declared_envelope)
        or declared != computed
    ):
        raise ContractError(
            "canonical APT embedded semantic/envelope SHA-256 mismatch"
        )
    serialized = _serialize_canonical_apt_document(document, contract, computed)
    if serialized != artifact_bytes:
        raise ContractError(
            "canonical APT ECSV bytes are not exact canonical v1 serialization"
        )
    if Table is None:
        raise ContractError("astropy.table is unavailable for ECSV parity check")
    try:
        Table.read(text.splitlines(keepends=True), format="ascii.ecsv")
    except Exception as error:
        raise ContractError(f"Astropy rejects canonical APT ECSV: {error}") from error
    return document, computed


def _validate_canonical_apt_document(
    document: dict[str, Any], contract: dict[str, Any]
) -> None:
    if (
        document["profile"] != contract["profile"]
        or document["field_registry"] != contract["field_registry"]
    ):
        raise ContractError("canonical APT profile/field registry mismatch")
    envelope = document["envelope"]
    context = document["context"]
    for label, value in (
        ("profile", document["profile"]),
        ("field registry", document["field_registry"]),
        *((f"envelope {key}", value) for key, value in envelope.items()),
        *((f"context {key}", value) for key, value in context.items()),
    ):
        _require_canonical_text(label, value)
    if (
        envelope["output_role"] != contract["output_role"]
        or envelope["producer"] != contract["producer"]
        or not _valid_utc_timestamp(envelope["event_time_utc"])
    ):
        raise ContractError("canonical APT envelope role/producer/time mismatch")
    if (
        context["coordinate_frame"] != contract["coordinate_frame"]
        or not _valid_utc_timestamp(context["observation_time_utc"])
    ):
        raise ContractError("canonical APT scientific context frame/time mismatch")
    if any(value < 0 for value in document["observation"].values()):
        raise ContractError("canonical APT observation tuple must be nonnegative")
    if not document["rows"] or not document["raw_inputs"]:
        raise ContractError("canonical APT requires rows and raw inputs")

    required = {field["name"]: field for field in contract["required_fields"]}
    optional = {
        field["name"]: field for field in contract["optional_extensions"]
    }
    authorized = required | optional
    fields = document["registered_fields"]
    names = [field["name"] for field in fields]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ContractError("canonical APT registered fields are not unique lexical")
    if not set(required).issubset(names):
        raise ContractError("canonical APT is missing required baseline fields")
    if any(name not in authorized for name in names):
        raise ContractError("canonical APT contains an unregistered extension")
    if any(field != authorized[field["name"]] for field in fields):
        raise ContractError("canonical APT registered field contract drift")
    if set(names).intersection(contract["protected_names"]):
        raise ContractError("canonical APT extension collides with protected structure")

    raw_by_network: dict[int, dict[str, Any]] = {}
    interfaces: set[str] = set()
    expected_count = 0
    for raw_input in document["raw_inputs"]:
        network = raw_input["network"]
        interface = raw_input["interface"]
        channel_count = raw_input["channel_count"]
        _require_canonical_text("raw interface", interface)
        if (
            network < 0
            or network > 12
            or interface != f"toltec{network}"
            or channel_count < 1
            or channel_count > CANONICAL_APT_UID_MAX + 1
            or network in raw_by_network
            or interface in interfaces
        ):
            raise ContractError("canonical APT raw manifest input is invalid")
        if expected_count > CANONICAL_APT_UID_MAX + 1 - channel_count:
            raise ContractError("canonical APT raw manifest exceeds v1 capacity")
        expected_count += channel_count
        raw_by_network[network] = raw_input
        interfaces.add(interface)
    if expected_count != len(document["rows"]):
        raise ContractError("canonical APT raw counts do not cover every row")

    uids: set[int] = set()
    relations: set[tuple[int, int]] = set()
    field_by_name = {field["name"]: field for field in fields}
    for row in document["rows"]:
        uid = row["uid"]
        if uid < 0 or uid > CANONICAL_APT_UID_MAX or uid in uids:
            raise ContractError("canonical APT uid is invalid or duplicate")
        uids.add(uid)
        network = row["nw"]
        channel = row["kids_tone"]
        raw_input = raw_by_network.get(network)
        if (
            raw_input is None
            or channel < 0
            or channel >= raw_input["channel_count"]
            or (network, channel) in relations
        ):
            raise ContractError("canonical APT raw row relation is invalid")
        relations.add((network, channel))
        expected_array = 0 if network <= 6 else 1 if network <= 10 else 2
        if row["array"] != expected_array or not math.isfinite(row["tone_freq"]):
            raise ContractError("canonical APT row array/tone contract is invalid")
        if set(row["fields"]) != set(field_by_name):
            raise ContractError("canonical APT row registered-field set mismatch")
        for name, field in field_by_name.items():
            value = row["fields"][name]
            if value is None:
                if not field["nullable"]:
                    raise ContractError(
                        f"canonical APT nonnullable field {name!r} is null"
                    )
                continue
            if field["datatype"] == "int64":
                if type(value) is not int or value < INT64_MIN or value > INT64_MAX:
                    raise ContractError(f"canonical APT field {name!r} is not int64")
            elif field["datatype"] == "float64":
                if type(value) is not float:
                    raise ContractError(f"canonical APT field {name!r} is not float64")
                if math.isinf(value) or (
                    math.isnan(value) and field["nonfinite"] == "reject"
                ):
                    raise ContractError(
                        f"canonical APT field {name!r} violates nonfinite policy"
                    )
            elif field["datatype"] == "bool":
                if type(value) is not bool:
                    raise ContractError(f"canonical APT field {name!r} is not bool")
            elif field["datatype"] == "string":
                if type(value) is not str:
                    raise ContractError(f"canonical APT field {name!r} is not string")
                _require_canonical_text(f"row field {name}", value)
            else:
                raise ContractError(f"unsupported canonical APT datatype {field['datatype']}")
            if field["datatype"] == "int64":
                domains = contract["integer_domains"]
                if (
                    (name == "flag" and value not in domains["flag"])
                    or (
                        name == "flag2"
                        and not domains["flag2_minimum"]
                        <= value
                        <= domains["flag2_maximum"]
                    )
                    or (name in domains["nonnegative"] and value < 0)
                    or (
                        name in {
                            "scan_band_masked_edge",
                            "scan_band_mask_rejected",
                            "cal_amp_method",
                        }
                        and value not in domains[name]
                    )
                ):
                    raise ContractError(
                        f"canonical APT closed integer field {name!r} is invalid"
                    )

    expected_relations = {
        (network, channel)
        for network, raw_input in raw_by_network.items()
        for channel in range(raw_input["channel_count"])
    }
    if relations != expected_relations:
        raise ContractError("canonical APT raw relation is not a complete bijection")


def _float64_semantic_payload(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if math.copysign(1.0, value) < 0 else "+inf"
    return struct.pack(">d", value).hex()


def _canonical_apt_semantic_preimage(
    document: dict[str, Any], contract: dict[str, Any]
) -> bytes:
    frames: list[bytes] = []

    def add(label: str, datatype: str, payload: str) -> None:
        frames.append(canonical_frame(label, datatype, payload))

    def add_string(label: str, value: str) -> None:
        add(label, "utf8", value)

    def add_int(label: str, value: int) -> None:
        add(label, "int64", str(value))

    def add_count(label: str, value: int) -> None:
        add(label, "uint64", str(value))

    def add_bool(label: str, value: bool) -> None:
        add(label, "bool", "true" if value else "false")

    def add_float(label: str, value: float) -> None:
        add(label, "float64-ieee754", _float64_semantic_payload(value))

    add_string("encoding", contract["framing_encoding"])
    add_string("scope", contract["semantic_scope"])
    add_string("schema", contract["schema_version"])
    add_string("profile", document["profile"])
    add_string("field-registry", document["field_registry"])
    add_count("core.count", len(contract["core_fields"]))
    for index, field in enumerate(contract["core_fields"]):
        prefix = f"core.{index}."
        add_string(prefix + "name", field["name"])
        add_string(prefix + "type", field["datatype"])
        add_string(prefix + "unit", field["unit"])
        add_bool(prefix + "nullable", field["nullable"])
        add_string(prefix + "authority", field["authority"])
        add_string(prefix + "identity-role", field["identity_role"])

    fields = sorted(document["registered_fields"], key=lambda item: item["name"])
    add_count("registered.count", len(fields))
    for index, field in enumerate(fields):
        prefix = f"registered.{index}."
        add_string(prefix + "name", field["name"])
        add_string(prefix + "type", field["datatype"])
        add_string(prefix + "unit", field["unit"])
        add_bool(prefix + "nullable", field["nullable"])
        add_string(prefix + "authority", field["authority"])
        add_string(prefix + "authority-reference", field["authority_reference"])
        add_string(prefix + "nonfinite", field["nonfinite"])
        add_string(prefix + "registry", field["registry"])
        add_string(prefix + "description", field["description"])
        add_string(prefix + "identity-role", "nonidentity")

    observation = document["observation"]
    add_int("observation.observation", observation["observation"])
    add_int("observation.subobservation", observation["subobservation"])
    add_int("observation.scan", observation["scan"])
    context = document["context"]
    add_string("context.project-id", context["project_id"])
    add_string("context.source-name", context["source_name"])
    add_string("context.observation-time-utc", context["observation_time_utc"])
    add_string("context.coordinate-frame", context["coordinate_frame"])

    raw_inputs = sorted(
        document["raw_inputs"], key=lambda item: (item["network"], item["interface"])
    )
    add_count("raw-input.count", len(raw_inputs))
    for index, raw_input in enumerate(raw_inputs):
        prefix = f"raw-input.{index}."
        add_int(prefix + "network", raw_input["network"])
        add_string(prefix + "interface", raw_input["interface"])
        add_int(prefix + "channel-count", raw_input["channel_count"])
    add_count("raw-channel.count", len(document["rows"]))
    raw_index = 0
    for raw_input in raw_inputs:
        for channel in range(raw_input["channel_count"]):
            prefix = f"raw-channel.{raw_index}."
            add_int(prefix + "network", raw_input["network"])
            add_int(prefix + "channel", channel)
            add_string(prefix + "interface", raw_input["interface"])
            raw_index += 1

    rows = sorted(document["rows"], key=lambda item: item["uid"])
    add_count("row.count", len(rows))
    for index, row in enumerate(rows):
        prefix = f"row.{index}."
        add_int(prefix + "uid", row["uid"])
        add_float(prefix + "tone_freq", row["tone_freq"])
        add_int(prefix + "array", row["array"])
        add_int(prefix + "nw", row["nw"])
        add_int(prefix + "kids_tone", row["kids_tone"])
        for field in fields:
            label = prefix + "field." + field["name"]
            value = row["fields"][field["name"]]
            if value is None:
                add(label, "null-" + field["datatype"], "null")
            elif field["datatype"] == "int64":
                add_int(label, value)
            elif field["datatype"] == "float64":
                add_float(label, value)
            elif field["datatype"] == "bool":
                add_bool(label, value)
            elif field["datatype"] == "string":
                add_string(label, value)
            else:  # guarded by model validation
                raise ContractError("unsupported canonical APT field datatype")
    return b"".join(frames)


def _canonical_apt_envelope_preimage(
    document: dict[str, Any], contract: dict[str, Any], semantic: str
) -> bytes:
    values = [
        ("encoding", contract["framing_encoding"]),
        ("scope", contract["envelope_scope"]),
        ("schema", contract["schema_version"]),
        ("profile", document["profile"]),
        ("field-registry", document["field_registry"]),
        ("semantic-sha256", semantic),
        ("occurrence", document["envelope"]["occurrence"]),
        ("event-reference", document["envelope"]["event_reference"]),
        ("output-role", document["envelope"]["output_role"]),
        ("producer", document["envelope"]["producer"]),
        ("software-revision", document["envelope"]["software_revision"]),
        (
            "configuration-reference",
            document["envelope"]["configuration_reference"],
        ),
        ("event-time-utc", document["envelope"]["event_time_utc"]),
    ]
    return b"".join(canonical_frame(label, "utf8", value) for label, value in values)


def _canonical_apt_digests(
    document: dict[str, Any], contract: dict[str, Any]
) -> dict[str, str]:
    semantic = "sha256:" + hashlib.sha256(
        _canonical_apt_semantic_preimage(document, contract)
    ).hexdigest()
    envelope = "sha256:" + hashlib.sha256(
        _canonical_apt_envelope_preimage(document, contract, semantic)
    ).hexdigest()
    return {"semantic_sha256": semantic, "envelope_sha256": envelope}


def _serialize_canonical_apt_document(
    document: dict[str, Any],
    contract: dict[str, Any],
    digests: dict[str, str] | None = None,
) -> bytes:
    if digests is None:
        _validate_canonical_apt_document(document, contract)
        digests = _canonical_apt_digests(document, contract)
    fields = sorted(document["registered_fields"], key=lambda item: item["name"])
    raw_inputs = sorted(
        document["raw_inputs"], key=lambda item: (item["network"], item["interface"])
    )
    columns = _expected_canonical_apt_columns(document, contract)
    lines = ["# %ECSV 1.0", "# ---", "# datatype:"]
    for column in columns:
        lines.append("# - name: " + _yaml_quote(column["name"]))
        lines.append("#   datatype: " + _yaml_quote(column["datatype"]))
        if column["unit"] and column["unit"] != "N/A":
            lines.append("#   unit: " + _yaml_quote(column["unit"]))
        lines.append("#   description: " + _yaml_quote(column["description"]))
    lines.extend(
        [
            "# meta:",
            "#   canonical_apt_v1:",
            "#     schema_version: " + _yaml_quote(contract["schema_version"]),
            "#     profile: " + _yaml_quote(document["profile"]),
            "#     field_registry: " + _yaml_quote(document["field_registry"]),
            "#     framing_encoding: " + _yaml_quote(contract["framing_encoding"]),
            "#     semantic_scope: " + _yaml_quote(contract["semantic_scope"]),
            "#     semantic_sha256: " + _yaml_quote(digests["semantic_sha256"]),
            "#     envelope_scope: " + _yaml_quote(contract["envelope_scope"]),
            "#     envelope_sha256: " + _yaml_quote(digests["envelope_sha256"]),
            "#     byte_transport_scope: "
            + _yaml_quote(contract["byte_transport_scope"]),
            "#     occurrence: " + _yaml_quote(document["envelope"]["occurrence"]),
            "#     event_reference: "
            + _yaml_quote(document["envelope"]["event_reference"]),
            "#     output_role: " + _yaml_quote(document["envelope"]["output_role"]),
            "#     producer: " + _yaml_quote(document["envelope"]["producer"]),
            "#     software_revision: "
            + _yaml_quote(document["envelope"]["software_revision"]),
            "#     configuration_reference: "
            + _yaml_quote(document["envelope"]["configuration_reference"]),
            "#     event_time_utc: "
            + _yaml_quote(document["envelope"]["event_time_utc"]),
            "#     scientific_context:",
            "#       project_id: " + _yaml_quote(document["context"]["project_id"]),
            "#       source_name: " + _yaml_quote(document["context"]["source_name"]),
            "#       observation_time_utc: "
            + _yaml_quote(document["context"]["observation_time_utc"]),
            "#       coordinate_frame: "
            + _yaml_quote(document["context"]["coordinate_frame"]),
            "#     observation:",
            "#       observation: " + str(document["observation"]["observation"]),
            "#       subobservation: "
            + str(document["observation"]["subobservation"]),
            "#       scan: " + str(document["observation"]["scan"]),
            "#     raw_manifest:",
        ]
    )
    for raw_input in raw_inputs:
        lines.append("#       - network: " + str(raw_input["network"]))
        lines.append("#         interface: " + _yaml_quote(raw_input["interface"]))
        lines.append("#         channel_count: " + str(raw_input["channel_count"]))
    lines.append("#     registered_fields:")
    for field in fields:
        lines.extend(
            [
                "#       - name: " + _yaml_quote(field["name"]),
                "#         datatype: " + _yaml_quote(field["datatype"]),
                "#         unit: " + _yaml_quote(field["unit"]),
                "#         nullable: " + ("true" if field["nullable"] else "false"),
                "#         authority: " + _yaml_quote(field["authority"]),
                "#         authority_reference: "
                + _yaml_quote(field["authority_reference"]),
                "#         nonfinite: " + _yaml_quote(field["nonfinite"]),
                "#         registry: " + _yaml_quote(field["registry"]),
                "#         description: " + _yaml_quote(field["description"]),
                "#         identity_role: \"nonidentity\"",
            ]
        )
    lines.extend(
        [
            "#     null_cell: \"unquoted-empty-v1\"",
            "#     string_cell: \"quoted-utf8-single-line-v1\"",
            "# delimiter: \",\"",
            "# schema: \"astropy-2.0\"",
            ",".join(column["name"] for column in columns),
        ]
    )
    for row in document["rows"]:
        cells = [
            str(row["uid"]),
            _format_float64(row["tone_freq"]),
            str(row["array"]),
            str(row["nw"]),
            str(row["kids_tone"]),
        ]
        for field in fields:
            value = row["fields"][field["name"]]
            if value is None:
                cells.append("")
            elif field["datatype"] == "int64":
                cells.append(str(value))
            elif field["datatype"] == "float64":
                cells.append(_format_float64(value))
            elif field["datatype"] == "bool":
                cells.append("True" if value else "False")
            elif field["datatype"] == "string":
                cells.append(_csv_quote(value))
            else:
                raise ContractError("unsupported canonical APT field datatype")
        lines.append(",".join(cells))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_canonical_apt_receipt(
    receipt_bytes: bytes, contract: dict[str, Any]
) -> dict[str, Any]:
    if not receipt_bytes or not receipt_bytes.endswith(b"\n") or b"\r" in receipt_bytes:
        raise ContractError("canonical APT receipt requires exact final LF text")
    try:
        text = receipt_bytes.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ContractError("canonical APT receipt is not ASCII") from error
    lines = text[:-1].split("\n")
    if len(lines) != 5 or lines[0] != contract["receipt_schema"]:
        raise ContractError("canonical APT receipt schema/line count mismatch")
    prefixes = ["scope=", "envelope_sha256=", "byte_sha256=", "byte_count="]
    values: list[str] = []
    for line, prefix in zip(lines[1:], prefixes, strict=True):
        if not line.startswith(prefix) or len(line) == len(prefix):
            raise ContractError("canonical APT receipt field order mismatch")
        values.append(line[len(prefix) :])
    scope, envelope, byte_sha, byte_count_token = values
    if (
        scope != contract["byte_transport_scope"]
        or SHA256_REFERENCE_RE.fullmatch(envelope) is None
        or SHA256_REFERENCE_RE.fullmatch(byte_sha) is None
        or re.fullmatch(r"0|[1-9][0-9]*", byte_count_token) is None
    ):
        raise ContractError("canonical APT receipt scope/digest/count is invalid")
    byte_count = int(byte_count_token)
    if byte_count > UINT64_MAX:
        raise ContractError("canonical APT receipt byte count is out of range")
    expected = (
        contract["receipt_schema"]
        + "\nscope="
        + scope
        + "\nenvelope_sha256="
        + envelope
        + "\nbyte_sha256="
        + byte_sha
        + "\nbyte_count="
        + str(byte_count)
        + "\n"
    ).encode("ascii")
    if expected != receipt_bytes:
        raise ContractError("canonical APT receipt text is noncanonical")
    return {
        "scope": scope,
        "envelope_sha256": envelope,
        "byte_sha256": byte_sha,
        "byte_count": byte_count,
    }


def validate_canonical_apt_v1_artifact(
    path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    _validate_canonical_apt_artifact_contract(
        contract.get("artifact_contract_id", ""), contract
    )
    path = path.expanduser().resolve()
    if path.suffix != contract["artifact_suffix"]:
        raise ContractError(
            f"canonical APT artifact requires suffix {contract['artifact_suffix']!r}"
        )
    receipt_path = Path(str(path) + contract["receipt_suffix"])
    result: dict[str, Any] = {
        "schema_version": ARTIFACT_RESULT_SCHEMA_VERSION,
        "artifact_contract_id": contract["artifact_contract_id"],
        "activation_state": contract["activation_state"],
        "artifact": str(path),
        "receipt": str(receipt_path),
        "passed": False,
        "semantic_sha256": None,
        "envelope_sha256": None,
        "byte_sha256": None,
        "byte_count": None,
        "row_count": None,
        "raw_input_count": None,
        "errors": [],
    }
    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = _parse_canonical_apt_receipt(receipt_bytes, contract)
        artifact_bytes = path.read_bytes()
        actual_byte_sha = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
        if (
            receipt["byte_count"] != len(artifact_bytes)
            or receipt["byte_sha256"] != actual_byte_sha
        ):
            raise ContractError(
                "canonical APT receipt byte SHA-256/count mismatch"
            )
        document, digests = _parse_canonical_apt_v1_bytes(
            artifact_bytes, contract
        )
        if receipt["envelope_sha256"] != digests["envelope_sha256"]:
            raise ContractError(
                "canonical APT receipt envelope binding mismatch"
            )
        result.update(
            {
                "passed": True,
                "semantic_sha256": digests["semantic_sha256"],
                "envelope_sha256": digests["envelope_sha256"],
                "byte_sha256": actual_byte_sha,
                "byte_count": len(artifact_bytes),
                "row_count": len(document["rows"]),
                "raw_input_count": len(document["raw_inputs"]),
            }
        )
    except (OSError, UnicodeError, ContractError, ValueError) as error:
        result["errors"].append(str(error))
    return result


def validate_ecsv(
    path: Path, checks: dict[str, Any], arrays: list[str]
) -> list[str]:
    if Table is None:
        return ["astropy.table is unavailable"]
    errors: list[str] = []
    try:
        table = Table.read(path, format="ascii.ecsv")
        missing = [
            name for name in checks.get("required_columns", []) if name not in table.colnames
        ]
        if missing:
            errors.append(f"missing ECSV columns {missing}")
        minimum = int(checks.get("min_rows", 0))
        if len(table) < minimum:
            errors.append(f"has {len(table)} rows; requires at least {minimum}")
        if checks.get("row_count") == "array_count" and len(table) != len(arrays):
            errors.append(f"has {len(table)} rows; expected {len(arrays)} arrays")
    except Exception as error:
        errors.append(f"cannot read ECSV: {error}")
    return errors


def validate_file(
    path: Path, checks: dict[str, Any], arrays: list[str]
) -> list[str]:
    errors: list[str] = []
    if checks.get("nonempty", True) and path.stat().st_size <= 0:
        errors.append("file is empty")
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit"}:
        errors.extend(validate_fits(path, checks))
    elif suffix in {".nc", ".nc4", ".cdf"}:
        errors.extend(validate_netcdf(path, checks, arrays))
    elif suffix == ".ecsv":
        errors.extend(validate_ecsv(path, checks, arrays))
    return errors


def product_files(reduction: Path) -> list[str]:
    return sorted(
        path.relative_to(reduction).as_posix()
        for path in reduction.rglob("*")
        if path.is_file() and path.suffix.lower() in PRODUCT_SUFFIXES
    )


def validate_reduction(
    registry: dict[str, Any],
    contract: dict[str, Any],
    reduction: Path,
    config: dict[str, Any],
    config_path: Path | None = None,
) -> dict[str, Any]:
    if not reduction.is_dir():
        raise ContractError(f"not a reduction directory: {reduction}")
    observations = observation_ids(reduction)
    minimum_observations = int(contract.get("minimum_observations", 1))
    errors: list[str] = []
    if len(observations) < minimum_observations:
        errors.append(
            f"found {len(observations)} observation directories; "
            f"requires at least {minimum_observations}"
        )
    arrays = list(contract["arrays"])
    families = registry["families"]
    check_definitions = registry.get("checks", {})
    matched_by: dict[str, list[str]] = {}
    entry_results: list[dict[str, Any]] = []
    for entry in contract["entries"]:
        classification = entry["classification"]
        if classification == "required":
            requested = True
        elif classification == "optional_diagnostic":
            requested = False
        else:
            requested = evaluate_condition(entry["required_when"], config)
        context_results = []
        for context in entry_contexts(entry, observations, arrays):
            pattern = entry["pattern"].format(**context)
            matches = sorted(
                path
                for path in reduction.glob(pattern)
                if path.is_file()
            )
            minimum = int(entry.get("min_matches", 1 if requested else 0))
            default_maximum = (
                0
                if classification == "config_conditional" and not requested
                else 1
            )
            maximum_value = entry.get("max_matches", default_maximum)
            maximum = None if maximum_value is None else int(maximum_value)
            context_errors: list[str] = []
            if len(matches) < minimum:
                context_errors.append(
                    f"pattern {pattern!r} matched {len(matches)}; requires at least {minimum}"
                )
            if maximum is not None and len(matches) > maximum:
                context_errors.append(
                    f"pattern {pattern!r} matched {len(matches)}; allows at most {maximum}"
                )
            for path in matches:
                relative = path.relative_to(reduction).as_posix()
                matched_by.setdefault(relative, []).append(entry["entry_id"])
                checks = merge_checks(
                    check_definitions.get(entry.get("check_id"), {}),
                    entry.get("checks", {}),
                )
                matched_config_checks = 0
                for conditional_check in entry.get("checks_by_config", []):
                    if evaluate_condition(conditional_check["when"], config):
                        conditional_values = conditional_check.get("checks")
                        if conditional_values is None:
                            conditional_values = check_definitions[
                                conditional_check["check_id"]
                            ]
                        checks = merge_checks(checks, conditional_values)
                        matched_config_checks += 1
                if (
                    entry.get("require_matching_config_check", False)
                    and matched_config_checks == 0
                ):
                    context_errors.append(
                        "no checks_by_config rule matched the realized configuration"
                    )
                required_match_count = entry.get(
                    "required_config_check_matches"
                )
                if (
                    required_match_count is not None
                    and matched_config_checks != required_match_count
                ):
                    context_errors.append(
                        "checks_by_config matched "
                        f"{matched_config_checks} rules; expected exactly "
                        f"{required_match_count}"
                    )
                for message in validate_file(path, checks, arrays):
                    context_errors.append(f"{relative}: {message}")
            errors.extend(f"{entry['entry_id']}: {message}" for message in context_errors)
            context_results.append(
                {
                    "context": context,
                    "pattern": pattern,
                    "matches": [
                        path.relative_to(reduction).as_posix() for path in matches
                    ],
                    "errors": context_errors,
                }
            )
        entry_results.append(
            {
                "entry_id": entry["entry_id"],
                "family_id": entry["family_id"],
                "classification": entry["classification"],
                "condition": entry["condition"],
                "requested_by_config": requested,
                "scientific_identity": families[entry["family_id"]][
                    "scientific_identity"
                ],
                "contexts": context_results,
                "passed": all(not result["errors"] for result in context_results),
            }
        )

    all_products = product_files(reduction)
    unclassified = sorted(set(all_products) - matched_by.keys())
    multiply_classified = sorted(
        {path: entries for path, entries in matched_by.items() if len(entries) > 1}.items()
    )
    if unclassified:
        errors.append(f"unclassified product files: {unclassified}")
    if multiply_classified:
        errors.append(f"multiply classified product files: {multiply_classified}")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "contract_id": contract["contract_id"],
        "profile_id": contract["profile_id"],
        "mode": contract["mode"],
        "reduction": str(reduction.resolve()),
        "low_level_config": str(config_path.resolve()) if config_path else None,
        "passed": not errors,
        "observation_ids": observations,
        "arrays": arrays,
        "product_count": len(all_products),
        "classified_product_count": len(matched_by),
        "unclassified_products": unclassified,
        "multiply_classified_products": [
            {"path": path, "entries": entries}
            for path, entries in multiply_classified
        ],
        "entry_results": entry_results,
        "errors": errors,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Citlali Product Contract",
        "",
        f"- Contract: `{result['contract_id']}`",
        f"- Profile: `{result['profile_id']}`",
        f"- Mode: `{result['mode']}`",
        f"- Reduction: `{result['reduction']}`",
        f"- Low-level config: `{result['low_level_config']}`",
        f"- Verdict: **{'accepted' if result['passed'] else 'rejected'}**",
        f"- Observations: `{len(result['observation_ids'])}`",
        f"- Classified products: `{result['classified_product_count']}/{result['product_count']}`",
        "",
        "## Families",
        "",
    ]
    for entry in result["entry_results"]:
        matches = sum(len(context["matches"]) for context in entry["contexts"])
        state = "pass" if entry["passed"] else "FAIL"
        request_detail = ""
        if entry["classification"] == "config_conditional":
            requested = str(entry["requested_by_config"]).lower()
            request_detail = f" requested={requested}"
        lines.append(
            f"- `{entry['entry_id']}`: **{state}**; "
            f"classification={entry['classification']}{request_detail} "
            f"matches={matches}"
        )
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in result["errors"])
    if not result["errors"]:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def render_artifact_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Citlali Artifact Contract",
        "",
        f"- Artifact contract: `{result['artifact_contract_id']}`",
        f"- Activation: `{result['activation_state']}`",
        f"- Artifact: `{result['artifact']}`",
        f"- Receipt: `{result['receipt']}`",
        f"- Verdict: **{'VALID / conformant' if result['passed'] else 'INVALID'}**",
        "- Production profile: **unactivated / deferred**",
        f"- Semantic SHA-256: `{result['semantic_sha256']}`",
        f"- Envelope SHA-256: `{result['envelope_sha256']}`",
        f"- Byte SHA-256: `{result['byte_sha256']}`",
        f"- Byte count: `{result['byte_count']}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {error}" for error in result["errors"])
    if not result["errors"]:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--contract")
    selector.add_argument("--artifact-contract")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "validation/product_contracts.json",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        registry = load_registry(args.registry.expanduser().resolve())
        if args.artifact_contract:
            if args.json_out or args.report_out:
                raise ContractError(
                    "artifact mode is stdout-only; report outputs are forbidden"
                )
            artifact_contract = artifact_contract_by_id(
                registry, args.artifact_contract
            )
            result = validate_canonical_apt_v1_artifact(
                args.target, artifact_contract
            )
            print(render_artifact_markdown(result), end="")
            return 0 if result["passed"] else 1
        contract = contract_by_id(registry, args.contract)
        reduction = args.target.expanduser().resolve()
        config_path = find_lowlevel_config(reduction)
        result = validate_reduction(
            registry,
            contract,
            reduction,
            load_lowlevel_config(config_path),
            config_path,
        )
    except (
        OSError,
        json.JSONDecodeError,
        yaml.YAMLError,
        ContractError,
        TypeError,
        ValueError,
    ) as error:
        print(f"product contract invalid: {error}", file=sys.stderr)
        return 2
    report = render_markdown(result)
    if args.json_out:
        write_text(
            args.json_out.expanduser(),
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if args.report_out:
        write_text(args.report_out.expanduser(), report)
    print(report, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
