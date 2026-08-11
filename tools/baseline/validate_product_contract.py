#!/usr/bin/env python3
"""Validate a Citlali reduction against a versioned product contract."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import re
import struct
import sys
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


def load_registry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        registry = _mapping(json.load(stream), str(path))
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"{path}: unsupported schema_version")

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
            for rule in checks.get("scalar_tuples_allowed", []):
                names = rule["scalars"]
                actual = tuple(scalar_value(name) for name in names)
                if any(value is None for value in actual):
                    continue
                allowed = {tuple(values) for values in rule["allowed"]}
                if actual not in allowed:
                    errors.append(
                        f"NetCDF scalar tuple {tuple(names)!r}={actual!r}; "
                        f"expected one of {sorted(allowed)!r}"
                    )
            for name, variants in checks.get(
                "required_variables_by_scalar", {}
            ).items():
                actual = scalar_value(name)
                if actual is None:
                    continue
                # JSON object keys are strings while NetCDF integer scalars
                # normalize to Python int. Accept only the exact normalized
                # spelling; do not silently skip integer-conditioned rules.
                variant_key = str(actual)
                conditional = expanded_names(
                    variants.get(variant_key, variants.get(actual, [])), arrays
                )
                missing_conditional = missing_patterns(
                    variables, conditional
                )
                if missing_conditional:
                    errors.append(
                        f"NetCDF scalar {name!r}={actual!r} requires "
                        f"variables {missing_conditional}"
                    )
            for rule_name, inventory, label in (
                ("forbidden_variables_by_scalar", variables, "variables"),
                ("forbidden_dimensions_by_scalar", dimensions, "dimensions"),
            ):
                for name, variants in checks.get(rule_name, {}).items():
                    actual = scalar_value(name)
                    if actual is None:
                        continue
                    forbidden = expanded_names(
                        variants.get(str(actual), variants.get(actual, [])), arrays
                    )
                    present = [
                        value for value in forbidden
                        if any(fnmatch.fnmatchcase(candidate, value)
                               for candidate in inventory)
                    ]
                    if present:
                        errors.append(
                            f"NetCDF scalar {name!r}={actual!r} forbids "
                            f"{label} {present}"
                        )
            for name, variants in checks.get(
                "required_dimensions_by_scalar", {}
            ).items():
                actual = scalar_value(name)
                if actual is None:
                    continue
                conditional = expanded_names(
                    variants.get(str(actual), variants.get(actual, [])), arrays
                )
                missing_conditional = missing_patterns(dimensions, conditional)
                if missing_conditional:
                    errors.append(
                        f"NetCDF scalar {name!r}={actual!r} requires "
                        f"dimensions {missing_conditional}"
                    )
            for dimension_name, scalar_name in checks.get(
                "dimension_size_equals_scalar", {}
            ).items():
                actual = scalar_value(scalar_name)
                if actual is None or dimension_name not in dataset.dimensions:
                    continue
                dimension_size = len(dataset.dimensions[dimension_name])
                if dimension_size != int(actual):
                    errors.append(
                        f"NetCDF dimension {dimension_name!r} has size "
                        f"{dimension_size}; {scalar_name!r}={actual!r}"
                    )
            for condition_name, variants in checks.get(
                "scalar_equals_by_scalar", {}
            ).items():
                actual = scalar_value(condition_name)
                if actual is None:
                    continue
                requirements = variants.get(
                    str(actual), variants.get(actual, {})
                )
                for scalar_name, expected in requirements.items():
                    conditioned = scalar_value(scalar_name)
                    if conditioned is not None and conditioned != expected:
                        errors.append(
                            f"NetCDF scalar {condition_name!r}={actual!r} "
                            f"requires {scalar_name!r}={expected!r}; got "
                            f"{conditioned!r}"
                        )
            for condition_name, variants in checks.get(
                "variable_trailing_dimension_by_scalar", {}
            ).items():
                actual = scalar_value(condition_name)
                if actual is None:
                    continue
                requirements = variants.get(
                    str(actual), variants.get(actual, {})
                )
                for variable_name, dimension_name in requirements.items():
                    if variable_name not in dataset.variables:
                        continue
                    variable_dimensions = dataset.variables[variable_name].dimensions
                    if not variable_dimensions or variable_dimensions[-1] != dimension_name:
                        errors.append(
                            f"NetCDF variable {variable_name!r} trailing "
                            f"dimension={variable_dimensions[-1:]!r}; expected "
                            f"{dimension_name!r}"
                        )
            for condition_name, rule in checks.get(
                "conditional_variable_set_by_scalar", {}
            ).items():
                actual = scalar_value(condition_name)
                if actual is None:
                    continue
                dimension_name = rule["dimension"]
                member_names = expanded_names(rule["variables"], arrays)
                available_value = rule.get("available_value", 1)
                unavailable_value = rule.get("unavailable_value", 0)
                if actual == available_value:
                    if dimension_name not in dataset.dimensions:
                        errors.append(
                            f"NetCDF scalar {condition_name!r}={actual!r} "
                            f"requires dimension {dimension_name!r}"
                        )
                    missing_members = missing_patterns(variables, member_names)
                    if missing_members:
                        errors.append(
                            f"NetCDF scalar {condition_name!r}={actual!r} "
                            f"requires complete conditional variable set "
                            f"{missing_members}"
                        )
                    for variable_name in member_names:
                        if variable_name not in dataset.variables:
                            continue
                        variable_dimensions = dataset.variables[
                            variable_name
                        ].dimensions
                        axis_only = set(rule.get("axis_only_variables", []))
                        context_dimensions = rule.get("context_dimensions")
                        if context_dimensions is not None:
                            expected_dimensions = (
                                [dimension_name]
                                if variable_name in axis_only
                                else list(context_dimensions) + [dimension_name]
                            )
                            dimensions_match = (
                                list(variable_dimensions) == expected_dimensions
                            )
                        else:
                            dimensions_match = bool(variable_dimensions) and \
                                variable_dimensions[-1] == dimension_name
                        if not dimensions_match:
                            if context_dimensions is not None:
                                errors.append(
                                    f"NetCDF conditional variable "
                                    f"{variable_name!r} dimensions="
                                    f"{list(variable_dimensions)!r}; expected "
                                    f"{expected_dimensions!r}"
                                )
                            else:
                                errors.append(
                                    f"NetCDF conditional variable "
                                    f"{variable_name!r} trailing dimension="
                                    f"{variable_dimensions[-1:]!r}; expected "
                                    f"{dimension_name!r}"
                                )
                elif actual == unavailable_value:
                    if dimension_name in dataset.dimensions:
                        errors.append(
                            f"NetCDF scalar {condition_name!r}={actual!r} "
                            f"forbids dimension {dimension_name!r}"
                        )
                    present_members = [
                        name for name in member_names if name in dataset.variables
                    ]
                    if present_members:
                        errors.append(
                            f"NetCDF scalar {condition_name!r}={actual!r} "
                            f"forbids conditional variables {present_members}"
                        )
                else:
                    errors.append(
                        f"NetCDF scalar {condition_name!r}={actual!r}; "
                        f"expected conditional values {available_value!r} or "
                        f"{unavailable_value!r}"
                    )
            for condition_name, variants in checks.get(
                "exact_variable_dimensions_by_scalar", {}
            ).items():
                actual = scalar_value(condition_name)
                if actual is None:
                    continue
                requirements = variants.get(
                    str(actual), variants.get(actual, {})
                )
                for variable_name, expected_dimensions in requirements.items():
                    if variable_name not in dataset.variables:
                        continue
                    actual_dimensions = list(
                        dataset.variables[variable_name].dimensions
                    )
                    if actual_dimensions != expected_dimensions:
                        errors.append(
                            f"NetCDF variable {variable_name!r} dimensions="
                            f"{actual_dimensions!r}; expected exact "
                            f"{expected_dimensions!r}"
                        )
            for name, pattern in checks.get("scalar_regex", {}).items():
                actual = scalar_value(name)
                if actual is not None and re.fullmatch(pattern, str(actual)) is None:
                    errors.append(
                        f"NetCDF scalar {name!r}={actual!r} does not match "
                        f"{pattern!r}"
                    )
            for name in checks.get("nonempty_string_scalars", []):
                actual = scalar_value(name)
                if actual is not None and (
                    not isinstance(actual, str) or not actual
                ):
                    errors.append(f"NetCDF scalar {name!r} is empty")
            for digest_name, bytes_name in checks.get(
                "sha256_string_scalar_relations", {}
            ).items():
                digest = scalar_value(digest_name)
                canonical = scalar_value(bytes_name)
                if digest is None or canonical is None:
                    continue
                if not isinstance(canonical, str):
                    errors.append(
                        f"NetCDF scalar {bytes_name!r} is not a string"
                    )
                    continue
                expected = hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest()
                if digest != expected:
                    errors.append(
                        f"NetCDF scalar {digest_name!r} does not equal "
                        f"SHA-256 of {bytes_name!r}"
                    )
            for rule in checks.get("fir_digest_relations", []):
                digest = scalar_value(rule["digest_scalar"])
                variable_name = rule["coefficient_variable"]
                if digest is None or variable_name not in dataset.variables:
                    continue
                coefficients = np.asarray(
                    dataset.variables[variable_name][...], dtype="<f8"
                ).reshape(-1)
                preimage = struct.pack("<Q", coefficients.size) + \
                    coefficients.tobytes(order="C")
                expected = hashlib.sha256(preimage).hexdigest()
                if digest != expected:
                    errors.append(
                        f"NetCDF scalar {rule['digest_scalar']!r} does not "
                        f"match {variable_name!r} under {rule['convention']!r}"
                    )
            for total_name, part_names in checks.get(
                "sum_variables_equals", {}
            ).items():
                if total_name not in dataset.variables:
                    continue
                missing_parts = [
                    name for name in part_names if name not in dataset.variables
                ]
                if missing_parts:
                    continue
                total_values = np.asarray(dataset.variables[total_name][...])
                part_sum = np.zeros_like(total_values, dtype=np.int64)
                for part_name in part_names:
                    part_sum += np.asarray(
                        dataset.variables[part_name][...], dtype=np.int64
                    )
                if not np.array_equal(total_values, part_sum):
                    errors.append(
                        f"NetCDF variables {part_names!r} do not sum exactly "
                        f"to {total_name!r}"
                    )
            for rule in checks.get("category_fraction_relations", []):
                total_name = rule["total"]
                count_names = rule["counts"]
                fraction_names = rule.get("fractions", [
                    name.removesuffix("_count") + "_fraction"
                    for name in count_names
                ])
                if (total_name not in dataset.variables or
                        len(count_names) != len(fraction_names) or
                        any(name not in dataset.variables
                            for name in count_names + fraction_names)):
                    continue
                totals = np.asarray(
                    dataset.variables[total_name][...], dtype=np.int64
                )
                fraction_sum = np.zeros_like(totals, dtype=np.float64)
                for count_name, fraction_name in zip(
                    count_names, fraction_names
                ):
                    counts = np.asarray(
                        dataset.variables[count_name][...], dtype=np.int64
                    )
                    fractions = np.asarray(
                        dataset.variables[fraction_name][...],
                        dtype=np.float64,
                    )
                    expected = np.divide(
                        counts, totals,
                        out=np.zeros_like(fractions, dtype=np.float64),
                        where=totals != 0,
                    )
                    if (counts.shape != totals.shape or
                            fractions.shape != totals.shape or
                            not np.all(np.isfinite(fractions)) or
                            not np.allclose(fractions, expected, rtol=0.0,
                                            atol=1e-12)):
                        errors.append(
                            f"NetCDF category fraction {fraction_name!r} "
                            f"does not equal {count_name!r}/{total_name!r}"
                        )
                    fraction_sum += fractions
                expected_sum = np.where(totals == 0, 0.0, 1.0)
                if not np.allclose(fraction_sum, expected_sum,
                                   rtol=0.0, atol=1e-12):
                    errors.append(
                        f"NetCDF category fractions for {total_name!r} "
                        "do not sum to the normalized total"
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reduction", type=Path)
    parser.add_argument("--contract", required=True)
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
        contract = contract_by_id(registry, args.contract)
        reduction = args.reduction.expanduser().resolve()
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
