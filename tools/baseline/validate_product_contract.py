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
OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID = (
    "apt-prod-002-observation-target-manifest-v1"
)
MATCH_DISPOSITIONS_ARTIFACT_CONTRACT_ID = (
    "apt-prod-002-match-dispositions-v1"
)
OBSERVATION_MATCHED_APT_ARTIFACT_CONTRACT_ID = (
    "apt-prod-002-observation-matched-apt-v1"
)
OBSERVATION_ARTIFACT_CONTRACT_SHA256 = {
    OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID: (
        "139b76cf556384d34d1b1923694a008dc7b21f1f8022584ec49ff3f8bf2bb72c"
    ),
    MATCH_DISPOSITIONS_ARTIFACT_CONTRACT_ID: (
        "acade470dbbb1ffd9327ada8db8a3df69e26ba02e7393864f1ad90de00d22785"
    ),
    OBSERVATION_MATCHED_APT_ARTIFACT_CONTRACT_ID: (
        "3e51715484a17be7ebc8677fb51d3e2d54cd11602025c8bf6005c3e7f151d286"
    ),
}
OBSERVATION_ARTIFACT_CONTRACT_IDS = frozenset(
    OBSERVATION_ARTIFACT_CONTRACT_SHA256
)
BASELINE_DESCRIPTOR_SCHEMA_V1 = (
    "citlali-verified-beammap-baseline-descriptor-v1"
)
OBSERVATION_TARGET_SCHEMA_V1 = "citlali-observation-target-manifest-v1"
MATCH_DISPOSITIONS_SCHEMA_V1 = "citlali-apt-match-dispositions-v1"
OBSERVATION_MATCHED_APT_SCHEMA_V1 = "citlali-observation-matched-apt-v1"
OBSERVATION_CONTRACT_AUTHORITY_V1 = "citlali"
OBSERVATION_VALUE_ISSUER_V1 = "tolproj"
OBSERVATION_MAPPING_DOMAIN_V1 = (
    "tolproj-observation-tone-to-beammap-seed-v1"
)
OBSERVATION_TRANSFORMATION_REGISTRY_V1 = (
    "citlali-observation-apt-field-transformations-v1"
)
OBSERVATION_TARGET_FIELD_REGISTRY_V1 = (
    "citlali-observation-target-fields-v1"
)
OBSERVATION_MATCHED_OUTPUT_FIELD_REGISTRY_V1 = (
    "citlali-observation-matched-output-fields-v1"
)
OBSERVATION_KMP_SOURCE_FIELD_MAP_PROFILE_V1 = (
    "citlali-kmp-source-field-map-v1"
)
OBSERVATION_UNMATCHED_MISSING_AUTHORITY_V1 = (
    "citlali:typed-missing-unmatched-v1"
)
OBSERVATION_COPY_BASELINE_OR_NULL_OPERATION_V1 = (
    "copy-baseline-when-matched-null-when-unmatched"
)
OBSERVATION_PRESERVE_TARGET_OPERATION_V1 = "preserve-target"
OBSERVATION_KMP_AUTHORIZED_USE_ROLES_V1 = {
    "identity": [],
    "matching": ["kids_Qr", "kids_fr"],
    "application": ["kids_f_out"],
    "transformation": [],
    "output": ["kids_Qr", "kids_f_out", "kids_flag", "kids_fr"],
    "authority": ["kids_Qr", "kids_f_out", "kids_flag", "kids_fr"],
}
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


def _validate_observation_artifact_contract(
    artifact_id: str, value: Any
) -> dict[str, Any]:
    context = f"artifact_contracts.{artifact_id}"
    artifact = _mapping(value, context)
    if artifact_id not in OBSERVATION_ARTIFACT_CONTRACT_SHA256:
        raise ContractError(f"{context}: unsupported observation artifact contract")
    if artifact.get("artifact_contract_id") != artifact_id:
        raise ContractError(f"{context}.artifact_contract_id: mismatch")
    digest = _canonical_json_sha256(artifact)
    expected_digest = OBSERVATION_ARTIFACT_CONTRACT_SHA256[artifact_id]
    if digest != expected_digest:
        raise ContractError(
            f"{context}: canonical observation contract/catalog drift "
            f"({digest}; expected {expected_digest})"
        )
    if (
        artifact.get("contract_schema_version")
        != "citlali-canonical-observation-artifact-contract-v1"
        or artifact.get("activation_state") != "unactivated"
        or artifact.get("contract_authority")
        != OBSERVATION_CONTRACT_AUTHORITY_V1
        or artifact.get("observation_value_issuer")
        != OBSERVATION_VALUE_ISSUER_V1
        or artifact.get("framing_encoding")
        != "citlali-labelled-type-length-v1"
    ):
        raise ContractError(
            f"{context}: successor version/activation/authority/framing mismatch"
        )
    expected_schemas = {
        OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID: OBSERVATION_TARGET_SCHEMA_V1,
        MATCH_DISPOSITIONS_ARTIFACT_CONTRACT_ID: MATCH_DISPOSITIONS_SCHEMA_V1,
        OBSERVATION_MATCHED_APT_ARTIFACT_CONTRACT_ID:
            OBSERVATION_MATCHED_APT_SCHEMA_V1,
    }
    schema = expected_schemas[artifact_id]
    expected_scopes = {
        OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID: (
            "citlali-observation-target-manifest-semantic-sha256-v1",
            "citlali-observation-target-manifest-envelope-sha256-v1",
        ),
        MATCH_DISPOSITIONS_ARTIFACT_CONTRACT_ID: (
            "citlali-apt-match-dispositions-semantic-sha256-v1",
            "citlali-apt-match-dispositions-envelope-sha256-v1",
        ),
        OBSERVATION_MATCHED_APT_ARTIFACT_CONTRACT_ID: (
            "citlali-observation-matched-apt-semantic-sha256-v1",
            "citlali-observation-matched-apt-envelope-sha256-v1",
        ),
    }
    semantic_scope, envelope_scope = expected_scopes[artifact_id]
    if (
        artifact.get("schema_version") != schema
        or artifact.get("validator") != schema
        or artifact.get("semantic_scope") != semantic_scope
        or artifact.get("envelope_scope") != envelope_scope
    ):
        raise ContractError(f"{context}: successor schema/scope/state mismatch")
    if artifact_id in {
        OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID,
        MATCH_DISPOSITIONS_ARTIFACT_CONTRACT_ID,
    }:
        if (
            artifact.get("persistence_state")
            != "embedded-logical-record-v1"
            or any(
                key in artifact
                for key in (
                    "artifact_suffix",
                    "byte_transport_scope",
                    "receipt_schema",
                )
            )
            or "no independent suffix, transport, receipt" not in
                artifact.get("publication_state", "")
        ):
            raise ContractError(
                f"{context}: logical record is falsely independently publishable"
            )
    else:
        if (
            artifact.get("persistence_state") != "persisted-final-artifact-v1"
            or artifact.get("artifact_suffix") != ".apt.ecsv"
            or artifact.get("physical_encoding")
            != "canonical-ecsv-1.0-v1"
            or artifact.get("receipt_schema")
            != "citlali-canonical-apt-publication-receipt-v1"
            or artifact.get("byte_transport_scope")
            != "citlali-observation-matched-apt-byte-transport-sha256-v1"
            or artifact.get("embedded_logical_records")
            != [
                OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID,
                MATCH_DISPOSITIONS_ARTIFACT_CONTRACT_ID,
            ]
            or artifact.get("activation_state") != "unactivated"
        ):
            raise ContractError(
                f"{context}: final matched APT persistence/embedding mismatch"
            )
    if artifact_id == MATCH_DISPOSITIONS_ARTIFACT_CONTRACT_ID:
        if artifact.get("mapping_domain") != OBSERVATION_MAPPING_DOMAIN_V1:
            raise ContractError(f"{context}.mapping_domain: mismatch")
    if artifact_id == OBSERVATION_MATCHED_APT_ARTIFACT_CONTRACT_ID:
        if (
            artifact.get("transformation_registry")
            != OBSERVATION_TRANSFORMATION_REGISTRY_V1
        ):
            raise ContractError(f"{context}.transformation_registry: mismatch")
    record_schemas = _mapping(
        artifact.get("record_schemas"), f"{context}.record_schemas"
    )
    required_record_schemas = {
        OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID: {
            "issuance_envelope",
            "observation_identity",
            "typed_field",
            "source_artifact",
            "target_input",
            "target_row",
            "target_manifest",
        },
        MATCH_DISPOSITIONS_ARTIFACT_CONTRACT_ID: {
            "issuance_envelope",
            "artifact_identity",
            "verified_baseline_reference",
            "row_reference",
            "match_pair",
            "matcher_evidence",
            "network_match_evidence",
            "endpoint_disposition",
            "match_relation",
        },
        OBSERVATION_MATCHED_APT_ARTIFACT_CONTRACT_ID: {
            "issuance_envelope",
            "artifact_identity",
            "verified_baseline_reference",
            "row_reference",
            "typed_field",
            "derived_output_field_contract",
            "field_transformation",
            "matched_output_row",
            "matched_output",
        },
    }
    if set(record_schemas) != required_record_schemas[artifact_id]:
        raise ContractError(f"{context}.record_schemas: incomplete/unknown records")
    member_keys = {
        "name",
        "datatype",
        "unit",
        "nullable",
        "authority",
        "cardinality",
        "identity_role",
    }
    for record_name, record_value in record_schemas.items():
        record_context = f"{context}.record_schemas.{record_name}"
        record = _mapping(record_value, record_context)
        if set(record) != {"cardinality", "unknown_members", "members"}:
            raise ContractError(f"{record_context}: unexpected record keys")
        _text(record.get("cardinality"), f"{record_context}.cardinality")
        if record.get("unknown_members") != "reject":
            raise ContractError(f"{record_context}.unknown_members: must reject")
        members = _list(record.get("members"), f"{record_context}.members")
        if not members:
            raise ContractError(f"{record_context}.members: must be nonempty")
        names: list[str] = []
        for index, member_value in enumerate(members):
            member_context = f"{record_context}.members[{index}]"
            member = _mapping(member_value, member_context)
            if set(member) != member_keys:
                raise ContractError(f"{member_context}: unexpected member keys")
            names.append(_text(member.get("name"), f"{member_context}.name"))
            for key in (
                "datatype",
                "unit",
                "authority",
                "cardinality",
                "identity_role",
            ):
                _text(member.get(key), f"{member_context}.{key}")
            if not isinstance(member.get("nullable"), bool):
                raise ContractError(f"{member_context}.nullable: expected bool")
            datatype = member["datatype"]
            if datatype.startswith("record:") or datatype.startswith("list:"):
                reference = datatype.split(":", 1)[1]
                if reference not in record_schemas:
                    raise ContractError(
                        f"{member_context}.datatype: unresolved record {reference!r}"
                    )
        _unique(names, f"{record_context}.members.name")
    if artifact_id == OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID:
        registry = _mapping(
            artifact.get("field_authorization_registry"),
            f"{context}.field_authorization_registry",
        )
        expected_registry_keys = {
            "registry",
            "source_field_map_profile",
            "source_column_map",
            "required_source_columns",
            "optional_source_columns",
            "required_field_names",
            "optional_field_names",
            "authorized_use_roles",
            "registered_fields",
            "unknown_source_diagnostics",
            "artifact_self_registration",
            "policy",
        }
        expected_fields = [
            _target_field_catalog_v1()[name]
            for name in sorted(_target_field_catalog_v1())
        ]
        if (
            set(registry) != expected_registry_keys
            or registry.get("registry") != OBSERVATION_TARGET_FIELD_REGISTRY_V1
            or registry.get("source_field_map_profile")
            != OBSERVATION_KMP_SOURCE_FIELD_MAP_PROFILE_V1
            or registry.get("source_column_map")
            != OBSERVATION_KMP_SOURCE_COLUMN_MAP_V1
            or registry.get("required_source_columns") != ["fr", "f_out", "Qr"]
            or registry.get("optional_source_columns") != ["flag"]
            or registry.get("required_field_names")
            != sorted(OBSERVATION_REQUIRED_TARGET_FIELDS_V1)
            or registry.get("optional_field_names")
            != sorted(OBSERVATION_OPTIONAL_TARGET_FIELDS_V1)
            or registry.get("authorized_use_roles")
            != OBSERVATION_KMP_AUTHORIZED_USE_ROLES_V1
            or registry.get("registered_fields") != expected_fields
            or registry.get("artifact_self_registration") != "reject"
            or not isinstance(registry.get("unknown_source_diagnostics"), str)
            or not isinstance(registry.get("policy"), str)
        ):
            raise ContractError(
                f"{context}.field_authorization_registry: not the exact closed v1 KMP catalog"
            )
    if artifact_id == OBSERVATION_MATCHED_APT_ARTIFACT_CONTRACT_ID:
        envelope_members = {
            member["name"]: member
            for member in record_schemas["issuance_envelope"]["members"]
        }
        if (
            {name: member["authority"] for name, member in envelope_members.items()}
            != {
                "occurrence": "citlali-canonical-issuer",
                "event_reference": "citlali-canonical-issuer",
                "software_revision": "citlali-build-authority",
                "configuration_reference": "tolproj-request",
                "event_time_utc": "tolproj-request",
            }
            or next(
                member
                for member in record_schemas["matched_output"]["members"]
                if member["name"] == "envelope"
            )["authority"] != "field-specific-citlali-issuance"
            or "occurrence and event issuance" not in artifact.get(
                "authority_contract", ""
            )
        ):
            raise ContractError(
                f"{context}: output issuance envelope authority split mismatch"
            )
        registry = _mapping(
            artifact.get("field_operation_authorization_registry"),
            f"{context}.field_operation_authorization_registry",
        )
        if (
            set(registry) != {
                "registry",
                "target_catalog_source_artifact_contract_id",
                "target_catalog_source_registry",
                "target_catalog_derivation",
                "baseline_catalog_source_artifact_contract_id",
                "baseline_catalog_source_contract_sha256",
                "baseline_catalog_derivation",
                "reserved_target_names",
                "authorized_operations",
                "issuer_declared_fields",
                "artifact_self_registration",
            }
            or registry.get("registry")
            != "citlali-observation-matched-output-fields-v1"
            or registry.get("target_catalog_source_artifact_contract_id")
            != OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID
            or registry.get("target_catalog_source_registry")
            != OBSERVATION_TARGET_FIELD_REGISTRY_V1
            or registry.get("baseline_catalog_source_artifact_contract_id")
            != CANONICAL_APT_ARTIFACT_CONTRACT_ID
            or registry.get("baseline_catalog_source_contract_sha256")
            != CANONICAL_APT_ARTIFACT_CONTRACT_SHA256
            or registry.get("reserved_target_names")
            != sorted(_target_field_catalog_v1())
            or registry.get("authorized_operations")
            != [
                OBSERVATION_PRESERVE_TARGET_OPERATION_V1,
                OBSERVATION_COPY_BASELINE_OR_NULL_OPERATION_V1,
            ]
            or registry.get("issuer_declared_fields") != []
            or registry.get("artifact_self_registration") != "reject"
            or artifact.get("allowed_transformations")
            != [
                OBSERVATION_PRESERVE_TARGET_OPERATION_V1,
                OBSERVATION_COPY_BASELINE_OR_NULL_OPERATION_V1,
            ]
            or not isinstance(registry.get("target_catalog_derivation"), str)
            or not isinstance(registry.get("baseline_catalog_derivation"), str)
        ):
            raise ContractError(
                f"{context}.field_operation_authorization_registry: "
                "not the exact derived closed v1 registry"
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
        if artifact_id == CANONICAL_APT_ARTIFACT_CONTRACT_ID:
            _validate_canonical_apt_artifact_contract(artifact_id, value)
        elif artifact_id in OBSERVATION_ARTIFACT_CONTRACT_IDS:
            _validate_observation_artifact_contract(artifact_id, value)
        else:
            raise ContractError(
                f"artifact_contracts.{artifact_id}: unsupported artifact contract"
            )

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
        routed_artifacts = {
            CANONICAL_APT_ARTIFACT_CONTRACT_ID,
            *OBSERVATION_ARTIFACT_CONTRACT_IDS,
        }
        routed_matches = sorted(
            artifact_id
            for artifact_id in routed_artifacts
            if artifact_id in routed
        )
        if routed_matches:
            raise ContractError(
                "unactivated canonical APT artifact contract is referenced "
                "by a reduction family/check/contract: "
                + ", ".join(routed_matches)
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
    if artifact_contract_id == CANONICAL_APT_ARTIFACT_CONTRACT_ID:
        return _validate_canonical_apt_artifact_contract(
            artifact_contract_id, artifact_contracts[artifact_contract_id]
        )
    return _validate_observation_artifact_contract(
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


def _parse_exact_uint64(value: str, label: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ContractError(f"invalid canonical exact uint64 {label}: {value!r}")
    result = int(value)
    if result > UINT64_MAX:
        raise ContractError(f"canonical uint64 {label} is out of range")
    return result


def canonical_observation_scalar_frame(
    label: str, datatype: str, payload: str
) -> bytes:
    """Return one exact successor-contract scalar frame.

    JSON-facing exact integers remain decimal strings and binary64 values remain
    exact bits.  This deliberately does not accept a Python number and therefore
    cannot round a value before it enters the canonical contract.
    """

    _require_canonical_text("observation frame label", label)
    _require_canonical_text("observation frame datatype", datatype)
    if datatype == "int64":
        _parse_exact_int64(payload, label)
    elif datatype == "uint64":
        _parse_exact_uint64(payload, label)
    elif datatype == "float64-ieee754":
        if re.fullmatch(r"[0-9a-f]{16}", payload) is None:
            raise ContractError(
                f"invalid exact IEEE-754 binary64 token {label}: {payload!r}"
            )
        bits = int(payload, 16)
        exponent = (bits >> 52) & 0x7FF
        significand = bits & ((1 << 52) - 1)
        if exponent == 0x7FF and significand and payload != "7ff8000000000000":
            raise ContractError(
                f"noncanonical IEEE-754 binary64 NaN token {label}: {payload!r}"
            )
    elif datatype in {
        "null-int64",
        "null-uint64",
        "null-float64",
        "null-bool",
        "null-string",
        "null-utf8",
        "null-opaque",
        "null-sha256",
    }:
        if payload != "null":
            raise ContractError(f"invalid typed null token {label}: {payload!r}")
    elif datatype in {"utf8", "opaque", "sha256"}:
        _require_canonical_text(label, payload, allow_empty=datatype == "utf8")
        if datatype == "sha256" and SHA256_REFERENCE_RE.fullmatch(payload) is None:
            raise ContractError(f"invalid SHA-256 reference {label}: {payload!r}")
    elif datatype == "bool":
        _parse_exact_bool(payload, label)
    else:
        raise ContractError(
            f"unsupported observation scalar datatype {datatype!r} for {label}"
        )
    return canonical_frame(label, datatype, payload)


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


def _float64_bits_token(value: float) -> str:
    if math.isnan(value):
        return "7ff8000000000000"
    return struct.pack(">d", value).hex()


def _descriptor_typed_value(datatype: str, value: Any) -> dict[str, Any]:
    normalized_datatype = {
        "float64": "float64-ieee754",
        "int64": "int64",
        "bool": "bool",
        "string": "utf8",
    }.get(datatype)
    if normalized_datatype is None:
        raise ContractError(
            f"verified baseline descriptor: unsupported datatype {datatype!r}"
        )
    if value is None:
        return {"datatype": normalized_datatype, "value": None}
    if datatype == "float64":
        token: Any = _float64_bits_token(value)
    elif datatype == "int64":
        token = str(value)
    elif datatype == "bool":
        token = value
    else:
        token = value
    return {"datatype": normalized_datatype, "value": token}


class VerifiedBaselineDescriptor(dict[str, Any]):
    """Typed view whose trust root remains the exact immutable publication.

    The C++ descriptor retains baseline and receipt bytes behind a private
    constructor.  Python mirrors that property with slots rather than exposing
    caller assertions as an ordinary serializable field.  Every identity or
    canonicalization operation below reconstructs this mapping from these bytes
    and the pinned APT-PROD-001 contract before trusting any typed member.
    """

    __slots__ = ("_artifact_bytes", "_receipt_bytes", "_artifact_contract")

    def __init__(
        self,
        value: dict[str, Any],
        artifact_bytes: bytes,
        receipt_bytes: bytes,
        artifact_contract: dict[str, Any],
    ) -> None:
        super().__init__(value)
        self._artifact_bytes = bytes(artifact_bytes)
        self._receipt_bytes = bytes(receipt_bytes)
        self._artifact_contract = copy.deepcopy(artifact_contract)


def verified_baseline_descriptor_from_bytes(
    artifact_bytes: bytes,
    receipt_bytes: bytes,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Verify immutable baseline bytes and return their complete typed descriptor.

    The descriptor accepts no caller-provided occurrence, row, raw relation, or
    digest assertion.  Every returned fact is reconstructed from the canonical
    ECSV bytes and the exact envelope-bound completion receipt.
    """

    _validate_canonical_apt_artifact_contract(
        contract.get("artifact_contract_id", ""), contract
    )
    receipt = _parse_canonical_apt_receipt(receipt_bytes, contract)
    actual_byte_sha = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
    if (
        receipt["byte_count"] != len(artifact_bytes)
        or receipt["byte_sha256"] != actual_byte_sha
    ):
        raise ContractError(
            "verified baseline descriptor receipt byte SHA-256/count mismatch"
        )
    document, digests = _parse_canonical_apt_v1_bytes(artifact_bytes, contract)
    if receipt["envelope_sha256"] != digests["envelope_sha256"]:
        raise ContractError(
            "verified baseline descriptor receipt envelope binding mismatch"
        )

    registered_fields = copy.deepcopy(document["registered_fields"])
    field_by_name = {field["name"]: field for field in registered_fields}
    rows: list[dict[str, Any]] = []
    for row in document["rows"]:
        fields = {
            name: _descriptor_typed_value(field_by_name[name]["datatype"], value)
            for name, value in sorted(row["fields"].items())
        }
        rows.append(
            {
                "uid": str(row["uid"]),
                "tone_freq": _descriptor_typed_value(
                    "float64", row["tone_freq"]
                ),
                "array": str(row["array"]),
                "network": str(row["nw"]),
                "channel": str(row["kids_tone"]),
                "fields": fields,
            }
        )
    raw_manifest = [
        {
            "network": str(raw_input["network"]),
            "interface": raw_input["interface"],
            "channel_count": str(raw_input["channel_count"]),
        }
        for raw_input in document["raw_inputs"]
    ]
    observation = {
        key: str(value) for key, value in document["observation"].items()
    }
    value = {
        "schema_version": BASELINE_DESCRIPTOR_SCHEMA_V1,
        "contract_authority": OBSERVATION_CONTRACT_AUTHORITY_V1,
        "baseline_value_issuer": OBSERVATION_CONTRACT_AUTHORITY_V1,
        "artifact_contract_id": contract["artifact_contract_id"],
        "artifact_contract_sha256": _canonical_json_sha256(contract),
        "baseline_schema_version": contract["schema_version"],
        "profile": document["profile"],
        "field_registry": document["field_registry"],
        "occurrence": document["envelope"]["occurrence"],
        "event_reference": document["envelope"]["event_reference"],
        "envelope": copy.deepcopy(document["envelope"]),
        "scientific_context": copy.deepcopy(document["context"]),
        "observation": observation,
        "raw_manifest": raw_manifest,
        "registered_fields": registered_fields,
        "rows": rows,
        "wire_presentation_sequence": [str(row["uid"]) for row in document["rows"]],
        "semantic_sha256": digests["semantic_sha256"],
        "envelope_sha256": digests["envelope_sha256"],
        "byte_transport_scope": contract["byte_transport_scope"],
        "byte_sha256": actual_byte_sha,
        "byte_count": str(len(artifact_bytes)),
        "receipt_sha256": "sha256:"
        + hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_byte_count": str(len(receipt_bytes)),
    }
    return VerifiedBaselineDescriptor(
        value, artifact_bytes, receipt_bytes, contract
    )


def _observation_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label}: expected non-empty canonical UTF-8 text")
    _require_canonical_text(label, value)
    return value


def _observation_int64(value: Any, label: str, *, local_key: bool = False) -> int:
    if not isinstance(value, str):
        raise ContractError(f"{label}: exact int64 must be a decimal string")
    result = _parse_exact_int64(value, label)
    if local_key and (result < 0 or result > CANONICAL_APT_UID_MAX):
        raise ContractError(f"{label}: local key is outside [0, 2^53-1]")
    return result


def _observation_uint64(value: Any, label: str) -> int:
    if not isinstance(value, str):
        raise ContractError(f"{label}: exact uint64 must be a decimal string")
    return _parse_exact_uint64(value, label)


def _observation_float64(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{label}: exact binary64 must be a string token")
    canonical_observation_scalar_frame(label, "float64-ieee754", value)
    return value


def _float_token_is_finite(value: str) -> bool:
    return ((int(value, 16) >> 52) & 0x7FF) != 0x7FF


def _float_token_is_negative(value: str) -> bool:
    return bool(int(value, 16) >> 63)


def _require_exact_object(
    value: Any, expected: set[str], context: str
) -> dict[str, Any]:
    result = _mapping(value, context)
    if set(result) != expected:
        raise ContractError(
            f"{context}: expected exact keys {sorted(expected)}; "
            f"found {sorted(result)}"
        )
    return result


def _frame_text(frames: list[bytes], label: str, value: str) -> None:
    frames.append(canonical_observation_scalar_frame(label, "utf8", value))


def _frame_int64(frames: list[bytes], label: str, value: str) -> None:
    frames.append(canonical_observation_scalar_frame(label, "int64", value))


def _frame_uint64(frames: list[bytes], label: str, value: int | str) -> None:
    frames.append(
        canonical_observation_scalar_frame(label, "uint64", str(value))
    )


def _frame_float64(frames: list[bytes], label: str, value: str) -> None:
    frames.append(
        canonical_observation_scalar_frame(label, "float64-ieee754", value)
    )


def _frame_bool(frames: list[bytes], label: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ContractError(f"{label}: expected bool")
    frames.append(
        canonical_observation_scalar_frame(
            label, "bool", "true" if value else "false"
        )
    )


def _validate_observation_identity(value: Any, context: str) -> dict[str, Any]:
    identity = _require_exact_object(
        value, {"observation", "subobservation", "scan"}, context
    )
    for key in ("observation", "subobservation", "scan"):
        if _observation_int64(identity[key], f"{context}.{key}") < 0:
            raise ContractError(f"{context}.{key}: must be nonnegative")
    return identity


def _add_observation_identity(
    frames: list[bytes], prefix: str, value: dict[str, Any]
) -> None:
    for key in ("observation", "subobservation", "scan"):
        _frame_int64(frames, f"{prefix}.{key}", value[key])


def _validate_issuance_envelope(value: Any, context: str) -> dict[str, Any]:
    envelope = _require_exact_object(
        value,
        {
            "occurrence",
            "event_reference",
            "software_revision",
            "configuration_reference",
            "event_time_utc",
        },
        context,
    )
    for key, item in envelope.items():
        _observation_text(item, f"{context}.{key}")
    if not _valid_utc_timestamp(envelope["event_time_utc"]):
        raise ContractError(f"{context}.event_time_utc: not exact UTC")
    return envelope


def _add_envelope(
    frames: list[bytes],
    schema: str,
    scope: str,
    semantic_sha256: str,
    envelope: dict[str, Any],
) -> None:
    values = (
        ("encoding", "citlali-labelled-type-length-v1"),
        ("scope", scope),
        ("schema", schema),
        ("contract-authority", OBSERVATION_CONTRACT_AUTHORITY_V1),
        ("canonical-issuer", OBSERVATION_CONTRACT_AUTHORITY_V1),
        ("observation-value-issuer", OBSERVATION_VALUE_ISSUER_V1),
        ("semantic-sha256", semantic_sha256),
        ("occurrence", envelope["occurrence"]),
        ("event-reference", envelope["event_reference"]),
        ("software-revision", envelope["software_revision"]),
        ("configuration-reference", envelope["configuration_reference"]),
        ("event-time-utc", envelope["event_time_utc"]),
    )
    for label, value in values:
        _frame_text(frames, label, value)


def _decode_descriptor_value(
    cell: Any, field: dict[str, Any], context: str
) -> Any:
    cell = _require_exact_object(cell, {"datatype", "value"}, context)
    expected_datatype = {
        "int64": "int64",
        "float64": "float64-ieee754",
        "bool": "bool",
        "string": "utf8",
    }[field["datatype"]]
    if cell["datatype"] != expected_datatype:
        raise ContractError(f"{context}.datatype: mismatch")
    value = cell["value"]
    if value is None:
        if not field["nullable"]:
            raise ContractError(f"{context}: nonnullable value is null")
        return None
    if field["datatype"] == "int64":
        return _observation_int64(value, context)
    if field["datatype"] == "float64":
        token = _observation_float64(value, context)
        bits = int(token, 16)
        if token == "7ff8000000000000":
            return float("nan")
        return struct.unpack(">d", bits.to_bytes(8, "big"))[0]
    if field["datatype"] == "bool":
        if not isinstance(value, bool):
            raise ContractError(f"{context}: expected bool")
        return value
    return _observation_text(value, context)


def _descriptor_baseline_document(descriptor: dict[str, Any]) -> dict[str, Any]:
    envelope = _require_exact_object(
        descriptor["envelope"],
        {
            "occurrence",
            "event_reference",
            "output_role",
            "producer",
            "software_revision",
            "configuration_reference",
            "event_time_utc",
        },
        "baseline descriptor.envelope",
    )
    if (
        descriptor["occurrence"] != envelope["occurrence"]
        or descriptor["event_reference"] != envelope["event_reference"]
    ):
        raise ContractError(
            "baseline descriptor duplicates disagree with its exact envelope"
        )
    context = _require_exact_object(
        descriptor["scientific_context"],
        {"project_id", "source_name", "observation_time_utc", "coordinate_frame"},
        "baseline descriptor.scientific_context",
    )
    for key, value in (*envelope.items(), *context.items()):
        _observation_text(value, f"baseline descriptor {key}")
    observation_value = _validate_observation_identity(
        descriptor["observation"], "baseline descriptor.observation"
    )
    observation = {
        key: _observation_int64(value, f"baseline descriptor.observation.{key}")
        for key, value in observation_value.items()
    }
    raw_inputs: list[dict[str, Any]] = []
    for index, raw_value in enumerate(_list(
        descriptor["raw_manifest"], "baseline descriptor.raw_manifest"
    )):
        raw = _require_exact_object(
            raw_value,
            {"network", "interface", "channel_count"},
            f"baseline descriptor.raw_manifest[{index}]",
        )
        raw_inputs.append(
            {
                "network": _observation_int64(
                    raw["network"], f"baseline raw[{index}].network"
                ),
                "interface": _observation_text(
                    raw["interface"], f"baseline raw[{index}].interface"
                ),
                "channel_count": _observation_int64(
                    raw["channel_count"], f"baseline raw[{index}].channel_count"
                ),
            }
        )
    registered_fields = _list(
        descriptor["registered_fields"], "baseline descriptor.registered_fields"
    )
    field_keys = {
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
    fields: list[dict[str, Any]] = []
    field_by_name: dict[str, dict[str, Any]] = {}
    for index, field_value in enumerate(registered_fields):
        field = _require_exact_object(
            field_value, field_keys, f"baseline descriptor.registered_fields[{index}]"
        )
        if field["identity_role"] != "nonidentity":
            raise ContractError("baseline descriptor registered field claims identity")
        if field["name"] in field_by_name:
            raise ContractError("baseline descriptor duplicate registered field")
        field_by_name[field["name"]] = field
        fields.append(copy.deepcopy(field))
    rows: list[dict[str, Any]] = []
    wire_sequence: list[str] = []
    for index, row_value in enumerate(_list(
        descriptor["rows"], "baseline descriptor.rows"
    )):
        row = _require_exact_object(
            row_value,
            {"uid", "tone_freq", "array", "network", "channel", "fields"},
            f"baseline descriptor.rows[{index}]",
        )
        uid = _observation_int64(row["uid"], f"baseline row[{index}].uid", local_key=True)
        wire_sequence.append(str(uid))
        row_fields_value = _mapping(row["fields"], f"baseline row[{index}].fields")
        if set(row_fields_value) != set(field_by_name):
            raise ContractError("baseline descriptor row field catalog mismatch")
        row_fields = {
            name: _decode_descriptor_value(
                row_fields_value[name], field_by_name[name],
                f"baseline row[{index}].fields.{name}",
            )
            for name in field_by_name
        }
        tone_field = {"datatype": "float64", "nullable": False}
        rows.append(
            {
                "uid": uid,
                "tone_freq": _decode_descriptor_value(
                    row["tone_freq"], tone_field,
                    f"baseline row[{index}].tone_freq",
                ),
                "array": _observation_int64(row["array"], f"baseline row[{index}].array"),
                "nw": _observation_int64(row["network"], f"baseline row[{index}].network"),
                "kids_tone": _observation_int64(row["channel"], f"baseline row[{index}].channel"),
                "fields": row_fields,
            }
        )
    if descriptor["wire_presentation_sequence"] != wire_sequence:
        raise ContractError(
            "baseline descriptor wire presentation sequence is not its exact row order"
        )
    return {
        "profile": descriptor["profile"],
        "field_registry": descriptor["field_registry"],
        "envelope": copy.deepcopy(envelope),
        "context": copy.deepcopy(context),
        "observation": observation,
        "raw_inputs": raw_inputs,
        "registered_fields": fields,
        "rows": rows,
    }


def baseline_descriptor_preimage(descriptor: dict[str, Any]) -> bytes:
    if not isinstance(descriptor, VerifiedBaselineDescriptor):
        raise ContractError(
            "baseline descriptor must be reconstructed from immutable "
            "artifact and receipt bytes"
        )
    rebuilt = verified_baseline_descriptor_from_bytes(
        descriptor._artifact_bytes,
        descriptor._receipt_bytes,
        descriptor._artifact_contract,
    )
    if dict(rebuilt) != dict(descriptor):
        raise ContractError(
            "baseline descriptor typed content differs from its retained "
            "immutable artifact/receipt reconstruction"
        )
    expected = {
        "schema_version",
        "contract_authority",
        "baseline_value_issuer",
        "artifact_contract_id",
        "artifact_contract_sha256",
        "baseline_schema_version",
        "profile",
        "field_registry",
        "occurrence",
        "event_reference",
        "envelope",
        "scientific_context",
        "observation",
        "raw_manifest",
        "registered_fields",
        "rows",
        "wire_presentation_sequence",
        "semantic_sha256",
        "envelope_sha256",
        "byte_transport_scope",
        "byte_sha256",
        "byte_count",
        "receipt_sha256",
        "receipt_byte_count",
    }
    descriptor = _require_exact_object(descriptor, expected, "baseline descriptor")
    if (
        descriptor["schema_version"] != BASELINE_DESCRIPTOR_SCHEMA_V1
        or descriptor["contract_authority"] != OBSERVATION_CONTRACT_AUTHORITY_V1
        or descriptor["baseline_value_issuer"]
        != OBSERVATION_CONTRACT_AUTHORITY_V1
        or descriptor["artifact_contract_id"]
        != CANONICAL_APT_ARTIFACT_CONTRACT_ID
        or descriptor["baseline_schema_version"] != "citlali-canonical-apt-v1"
        or descriptor["byte_transport_scope"]
        != "citlali-canonical-apt-byte-transport-sha256-v1"
    ):
        raise ContractError("baseline descriptor schema/authority/scope mismatch")
    for key in (
        "artifact_contract_sha256",
        "semantic_sha256",
        "envelope_sha256",
        "byte_sha256",
        "receipt_sha256",
    ):
        if SHA256_REFERENCE_RE.fullmatch(descriptor[key]) is None and key != "artifact_contract_sha256":
            raise ContractError(f"baseline descriptor {key}: invalid digest")
    if re.fullmatch(r"[0-9a-f]{64}", descriptor["artifact_contract_sha256"]) is None:
        raise ContractError("baseline descriptor artifact contract digest is invalid")
    if descriptor["artifact_contract_sha256"] != CANONICAL_APT_ARTIFACT_CONTRACT_SHA256:
        raise ContractError("baseline descriptor artifact contract digest drift")
    _observation_uint64(descriptor["byte_count"], "baseline descriptor byte_count")
    _observation_uint64(
        descriptor["receipt_byte_count"], "baseline descriptor receipt_byte_count"
    )
    document = _descriptor_baseline_document(descriptor)
    digest_contract = {
        "framing_encoding": "citlali-labelled-type-length-v1",
        "semantic_scope": "citlali-canonical-apt-semantic-sha256-v1",
        "envelope_scope": "citlali-canonical-apt-envelope-sha256-v1",
        "schema_version": "citlali-canonical-apt-v1",
        "core_fields": [
            {"name": "uid", "datatype": "int64", "unit": "N/A", "nullable": False, "authority": "canonical-issuer", "identity_role": "artifact-local-row-key"},
            {"name": "tone_freq", "datatype": "float64", "unit": "Hz", "nullable": False, "authority": "raw-readout", "identity_role": "nonidentity-attribute"},
            {"name": "array", "datatype": "int64", "unit": "N/A", "nullable": False, "authority": "network-map", "identity_role": "nonidentity-attribute"},
            {"name": "nw", "datatype": "int64", "unit": "N/A", "nullable": False, "authority": "raw-manifest", "identity_role": "raw-channel-relation"},
            {"name": "kids_tone", "datatype": "int64", "unit": "N/A", "nullable": False, "authority": "raw-manifest", "identity_role": "raw-channel-relation"},
        ],
    }
    recomputed = _canonical_apt_digests(document, digest_contract)
    if (
        recomputed["semantic_sha256"] != descriptor["semantic_sha256"]
        or recomputed["envelope_sha256"] != descriptor["envelope_sha256"]
    ):
        raise ContractError(
            "baseline descriptor typed content does not reproduce its exact baseline identities"
        )
    frames: list[bytes] = []
    for label, value in (
        ("encoding", "citlali-labelled-type-length-v1"),
        ("scope", "citlali-verified-beammap-baseline-descriptor-sha256-v1"),
        ("schema", descriptor["schema_version"]),
        ("contract-authority", descriptor["contract_authority"]),
        ("baseline-value-issuer", descriptor["baseline_value_issuer"]),
        ("baseline-schema", descriptor["baseline_schema_version"]),
        ("baseline-profile", descriptor["profile"]),
        ("baseline-occurrence", descriptor["occurrence"]),
        ("baseline-semantic-sha256", descriptor["semantic_sha256"]),
        ("baseline-envelope-sha256", descriptor["envelope_sha256"]),
        ("baseline-transport-scope", descriptor["byte_transport_scope"]),
        ("baseline-transport-sha256", descriptor["byte_sha256"]),
    ):
        _frame_text(frames, label, value)
    _frame_uint64(frames, "baseline-byte-count", descriptor["byte_count"])
    _frame_text(frames, "receipt-sha256", descriptor["receipt_sha256"])
    _frame_uint64(
        frames, "receipt-byte-count", descriptor["receipt_byte_count"]
    )
    return b"".join(frames)


def baseline_descriptor_sha256(descriptor: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        baseline_descriptor_preimage(descriptor)
    ).hexdigest()


def _baseline_artifact_identity(descriptor: dict[str, Any]) -> dict[str, Any]:
    baseline_descriptor_preimage(descriptor)
    return {
        "schema": descriptor["baseline_schema_version"],
        "occurrence": descriptor["occurrence"],
        "semantic_sha256": descriptor["semantic_sha256"],
        "envelope_sha256": descriptor["envelope_sha256"],
    }


def verified_baseline_reference(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": _baseline_artifact_identity(descriptor),
        "profile": descriptor["profile"],
        "descriptor_sha256": baseline_descriptor_sha256(descriptor),
        "transport_scope": descriptor["byte_transport_scope"],
        "transport_sha256": descriptor["byte_sha256"],
        "byte_count": descriptor["byte_count"],
        "receipt_sha256": descriptor["receipt_sha256"],
        "receipt_byte_count": descriptor["receipt_byte_count"],
    }


def _validate_artifact_identity(value: Any, context: str) -> dict[str, Any]:
    identity = _require_exact_object(
        value,
        {"schema", "occurrence", "semantic_sha256", "envelope_sha256"},
        context,
    )
    _observation_text(identity["schema"], f"{context}.schema")
    _observation_text(identity["occurrence"], f"{context}.occurrence")
    for key in ("semantic_sha256", "envelope_sha256"):
        if SHA256_REFERENCE_RE.fullmatch(identity[key]) is None:
            raise ContractError(f"{context}.{key}: invalid SHA-256 reference")
    return identity


def _add_artifact_identity(
    frames: list[bytes], prefix: str, identity: dict[str, Any]
) -> None:
    _frame_text(frames, f"{prefix}.schema", identity["schema"])
    _frame_text(frames, f"{prefix}.occurrence", identity["occurrence"])
    _frame_text(
        frames, f"{prefix}.semantic-sha256", identity["semantic_sha256"]
    )
    _frame_text(
        frames, f"{prefix}.envelope-sha256", identity["envelope_sha256"]
    )


def _validate_baseline_reference(
    value: Any, descriptor: dict[str, Any], context: str
) -> dict[str, Any]:
    reference = _require_exact_object(
        value,
        {
            "artifact",
            "profile",
            "descriptor_sha256",
            "transport_scope",
            "transport_sha256",
            "byte_count",
            "receipt_sha256",
            "receipt_byte_count",
        },
        context,
    )
    _validate_artifact_identity(reference["artifact"], f"{context}.artifact")
    if reference != verified_baseline_reference(descriptor):
        raise ContractError(f"{context}: does not exactly bind verified baseline")
    return reference


def _add_baseline_reference(
    frames: list[bytes], prefix: str, reference: dict[str, Any]
) -> None:
    _add_artifact_identity(frames, f"{prefix}.artifact", reference["artifact"])
    _frame_text(frames, f"{prefix}.profile", reference["profile"])
    _frame_text(
        frames, f"{prefix}.descriptor-sha256", reference["descriptor_sha256"]
    )
    _frame_text(frames, f"{prefix}.transport-scope", reference["transport_scope"])
    _frame_text(
        frames, f"{prefix}.transport-sha256", reference["transport_sha256"]
    )
    _frame_uint64(frames, f"{prefix}.byte-count", reference["byte_count"])
    _frame_text(frames, f"{prefix}.receipt-sha256", reference["receipt_sha256"])
    _frame_uint64(
        frames, f"{prefix}.receipt-byte-count", reference["receipt_byte_count"]
    )


def _validate_row_reference(value: Any, context: str) -> dict[str, Any]:
    reference = _require_exact_object(
        value,
        {"artifact_schema", "occurrence", "envelope_sha256", "local_key"},
        context,
    )
    _observation_text(reference["artifact_schema"], f"{context}.artifact_schema")
    _observation_text(reference["occurrence"], f"{context}.occurrence")
    if SHA256_REFERENCE_RE.fullmatch(reference["envelope_sha256"]) is None:
        raise ContractError(f"{context}.envelope_sha256: invalid")
    _observation_int64(reference["local_key"], f"{context}.local_key", local_key=True)
    return reference


def _row_reference(identity: dict[str, Any], local_key: str) -> dict[str, Any]:
    _observation_int64(local_key, "row reference local key", local_key=True)
    return {
        "artifact_schema": identity["schema"],
        "occurrence": identity["occurrence"],
        "envelope_sha256": identity["envelope_sha256"],
        "local_key": local_key,
    }


def _add_row_reference(
    frames: list[bytes], prefix: str, reference: dict[str, Any]
) -> None:
    _frame_text(frames, f"{prefix}.artifact-schema", reference["artifact_schema"])
    _frame_text(frames, f"{prefix}.occurrence", reference["occurrence"])
    _frame_text(
        frames, f"{prefix}.envelope-sha256", reference["envelope_sha256"]
    )
    _frame_int64(frames, f"{prefix}.local-key", reference["local_key"])


def _expected_array(network: int) -> int:
    if network < 0 or network > 12:
        raise ContractError("network is outside 0..12")
    return 0 if network <= 6 else 1 if network <= 10 else 2


def _is_finite_binary64_token(token: str) -> bool:
    return ((int(token, 16) >> 52) & 0x7FF) != 0x7FF


def _binary64_value(token: str) -> float:
    return struct.unpack(">d", int(token, 16).to_bytes(8, "big"))[0]


def _target_field_catalog_v1() -> dict[str, dict[str, Any]]:
    specifications = (
        (
            "kids_fr",
            "fr",
            "float64",
            "Hz",
            "kids:model-params-v1",
            "imported KIDs resonant frequency; finite, nonidentity",
        ),
        (
            "kids_f_out",
            "f_out",
            "float64",
            "Hz",
            "kids:model-params-v1",
            "imported KIDs output tone frequency; finite, nonidentity",
        ),
        (
            "kids_Qr",
            "Qr",
            "float64",
            "N/A",
            "kids:model-params-v1",
            "imported KIDs resonator Qr; finite with no positivity rule, nonidentity",
        ),
        (
            "kids_flag",
            "flag",
            "int64",
            "N/A",
            "kids:fit-report-v1",
            "imported KIDs model-fit flag; exact signed integral values, nonidentity",
        ),
    )
    return {
        name: {
            "name": name,
            "source_column": source_column,
            "datatype": datatype,
            "unit": unit,
            "nullable": False,
            "nonfinite": "reject",
            "authority": "copied-declared",
            "authority_reference": authority_reference,
            "registry": OBSERVATION_TARGET_FIELD_REGISTRY_V1,
            "description": description,
            "identity_role": "nonidentity",
        }
        for (
            name,
            source_column,
            datatype,
            unit,
            authority_reference,
            description,
        ) in specifications
    }


OBSERVATION_REQUIRED_TARGET_FIELDS_V1 = frozenset(
    {"kids_fr", "kids_f_out", "kids_Qr"}
)
OBSERVATION_OPTIONAL_TARGET_FIELDS_V1 = frozenset({"kids_flag"})
OBSERVATION_KMP_SOURCE_COLUMN_MAP_V1 = {
    "fr": "kids_fr",
    "f_out": "kids_f_out",
    "Qr": "kids_Qr",
    "flag": "kids_flag",
}


def canonical_target_fields_v1(*, include_kids_flag: bool) -> list[dict[str, Any]]:
    catalog = _target_field_catalog_v1()
    names = ["kids_fr", "kids_f_out", "kids_Qr"]
    if include_kids_flag:
        names.append("kids_flag")
    return [copy.deepcopy(catalog[name]) for name in names]


def _validate_typed_field(
    value: Any, expected_registry: str, context: str
) -> dict[str, Any]:
    field = _require_exact_object(
        value,
        {
            "name",
            "source_column",
            "datatype",
            "unit",
            "nullable",
            "nonfinite",
            "authority",
            "authority_reference",
            "registry",
            "description",
            "identity_role",
        },
        context,
    )
    for key in (
        "name",
        "datatype",
        "unit",
        "nonfinite",
        "authority",
        "authority_reference",
        "registry",
        "description",
        "identity_role",
    ):
        _observation_text(field[key], f"{context}.{key}")
    if field["source_column"] is not None:
        _observation_text(field["source_column"], f"{context}.source_column")
    if (
        field["datatype"] not in {"float64", "int64", "bool", "string"}
        or not isinstance(field["nullable"], bool)
        or field["nonfinite"] not in {"reject", "nan-token", "canonical-token"}
        or field["registry"] != expected_registry
        or field["identity_role"] != "nonidentity"
        or (field["datatype"] != "float64" and field["nonfinite"] != "reject")
    ):
        raise ContractError(f"{context}: invalid typed-field declaration")
    return field


def _validate_successor_value(
    value: Any, field: dict[str, Any], context: str
) -> Any:
    if value is None:
        if not field["nullable"]:
            raise ContractError(f"{context}: nonnullable field is null")
        return None
    datatype = field["datatype"]
    if datatype == "float64":
        token = _observation_float64(value, context)
        finite = _is_finite_binary64_token(token)
        bits = int(token, 16)
        is_infinite = ((bits >> 52) & 0x7FF) == 0x7FF and not (
            bits & ((1 << 52) - 1)
        )
        if (
            (not finite and field["nonfinite"] == "reject")
            or (is_infinite and field["nonfinite"] == "nan-token")
        ):
            raise ContractError(f"{context}: forbidden nonfinite value")
        return token
    if datatype == "int64":
        _observation_int64(value, context)
        return value
    if datatype == "bool":
        if not isinstance(value, bool):
            raise ContractError(f"{context}: expected bool")
        return value
    return _observation_text(value, context)


def _add_typed_field(
    frames: list[bytes], prefix: str, field: dict[str, Any]
) -> None:
    for suffix, value in (
        ("name", field["name"]),
        (
            "type",
            "float64-ieee754"
            if field["datatype"] == "float64"
            else "utf8"
            if field["datatype"] == "string"
            else field["datatype"],
        ),
        ("unit", field["unit"]),
    ):
        _frame_text(frames, f"{prefix}.{suffix}", value)
    _frame_bool(frames, f"{prefix}.nullable", field["nullable"])
    for suffix, value in (
        ("nonfinite", field["nonfinite"]),
        ("authority", field["authority"]),
        ("authority-reference", field["authority_reference"]),
        ("registry", field["registry"]),
        ("description", field["description"]),
    ):
        _frame_text(frames, f"{prefix}.{suffix}", value)
    _frame_bool(
        frames, f"{prefix}.has-source-column",
        field["source_column"] is not None,
    )
    if field["source_column"] is not None:
        _frame_text(
            frames, f"{prefix}.source-column", field["source_column"]
        )
    _frame_text(frames, f"{prefix}.identity-role", field["identity_role"])


def _add_successor_value(
    frames: list[bytes], label: str, value: Any, field: dict[str, Any]
) -> None:
    if value is None:
        frames.append(
            canonical_observation_scalar_frame(
                label, f"null-{field['datatype']}", "null"
            )
        )
    elif field["datatype"] == "float64":
        _frame_float64(frames, label, value)
    elif field["datatype"] == "int64":
        _frame_int64(frames, label, value)
    elif field["datatype"] == "bool":
        _frame_bool(frames, label, value)
    else:
        _frame_text(frames, label, value)


def validate_kmp_source_column_boundary_v1(
    available_columns: Any, requested_uses: Any
) -> tuple[str, ...]:
    """Validate uses without serializing unknown source diagnostics.

    Extra KMP columns may exist in the source report.  They are deliberately
    ignored unless a request tries to use one for a canonical role, at which
    point the closed Citlali registry fails rather than granting authority.
    """

    available = _list(available_columns, "KMP available columns")
    for index, name in enumerate(available):
        _observation_text(name, f"KMP available columns[{index}]")
    if len(available) != len(set(available)):
        raise ContractError("KMP available columns contain a duplicate")
    if not {"fr", "f_out", "Qr"}.issubset(available):
        raise ContractError("KMP source lacks a required model-parameter column")
    canonical_available = {
        canonical
        for source, canonical in OBSERVATION_KMP_SOURCE_COLUMN_MAP_V1.items()
        if source in available
    }
    requests = _require_exact_object(
        requested_uses,
        {
            "identity",
            "matching",
            "application",
            "transformation",
            "output",
            "authority",
        },
        "KMP requested uses",
    )
    allowed = set(_target_field_catalog_v1())
    role_allowed = {
        role: set(names)
        for role, names in OBSERVATION_KMP_AUTHORIZED_USE_ROLES_V1.items()
    }
    for role, names_value in requests.items():
        names = _list(names_value, f"KMP requested uses.{role}")
        for index, name in enumerate(names):
            _observation_text(name, f"KMP requested uses.{role}[{index}]")
        if (
            len(names) != len(set(names))
            or not set(names).issubset(canonical_available)
            or not set(names).issubset(role_allowed[role])
        ):
            raise ContractError(
                f"KMP requested uses.{role}: unknown or unauthorized source field"
            )
    return tuple(sorted(canonical_available.intersection(allowed)))


def _validate_permutation(
    value: Any, expected: set[str], context: str
) -> list[str]:
    sequence = _list(value, context)
    for index, key in enumerate(sequence):
        _observation_int64(key, f"{context}[{index}]", local_key=True)
    if len(sequence) != len(expected) or set(sequence) != expected:
        raise ContractError(f"{context}: not a complete permutation")
    return sequence


def validate_observation_target_manifest_v1(
    document: Any, contract: dict[str, Any]
) -> dict[str, Any]:
    _validate_observation_artifact_contract(
        contract.get("artifact_contract_id", ""), contract
    )
    if contract["artifact_contract_id"] != OBSERVATION_TARGET_ARTIFACT_CONTRACT_ID:
        raise ContractError("target validator received the wrong artifact contract")
    target = _require_exact_object(
        document,
        {
            "schema_version",
            "contract_authority",
            "observation_value_issuer",
            "envelope",
            "observation",
            "inputs",
            "registered_fields",
            "rows",
            "target_source_sequence",
            "target_application_sequence",
        },
        "target manifest",
    )
    if (
        target["schema_version"] != OBSERVATION_TARGET_SCHEMA_V1
        or target["contract_authority"] != OBSERVATION_CONTRACT_AUTHORITY_V1
        or target["observation_value_issuer"] != OBSERVATION_VALUE_ISSUER_V1
    ):
        raise ContractError("target manifest schema/authority/value issuer mismatch")
    _validate_issuance_envelope(target["envelope"], "target manifest.envelope")
    observation = _validate_observation_identity(
        target["observation"], "target manifest.observation"
    )
    registered_values = _list(
        target["registered_fields"], "target manifest.registered_fields"
    )
    registered_fields: dict[str, dict[str, Any]] = {}
    for index, field_value in enumerate(registered_values):
        field = _validate_typed_field(
            field_value,
            OBSERVATION_TARGET_FIELD_REGISTRY_V1,
            f"target manifest.registered_fields[{index}]",
        )
        if field["name"] in registered_fields:
            raise ContractError("target manifest has a duplicate registered field")
        registered_fields[field["name"]] = field
    declared_names = set(registered_fields)
    allowed_names = (
        set(OBSERVATION_REQUIRED_TARGET_FIELDS_V1)
        | set(OBSERVATION_OPTIONAL_TARGET_FIELDS_V1)
    )
    if (
        not set(OBSERVATION_REQUIRED_TARGET_FIELDS_V1).issubset(declared_names)
        or not declared_names.issubset(allowed_names)
        or registered_fields
        != {
            name: copy.deepcopy(_target_field_catalog_v1()[name])
            for name in declared_names
        }
    ):
        raise ContractError(
            "target manifest fields differ from the closed Citlali KMP registry"
        )
    inputs_value = _list(target["inputs"], "target manifest.inputs")
    rows_value = _list(target["rows"], "target manifest.rows")
    if not inputs_value or not rows_value:
        raise ContractError("target manifest requires inputs and rows")
    inputs: dict[str, dict[str, Any]] = {}
    networks: set[str] = set()
    interfaces: set[str] = set()
    source_keys: set[str] = set()
    expected_rows = 0
    source_keys_expected = {
        "source_key",
        "role",
        "diagnostic_locator",
        "content_sha256",
        "byte_count",
        "header_observation",
        "network",
        "interface",
        "channel_count",
    }
    input_keys_expected = {
        "input_key",
        "network",
        "interface",
        "channel_count",
        "raw_source",
        "kmp_source",
    }
    for index, input_value in enumerate(inputs_value):
        context = f"target manifest.inputs[{index}]"
        item = _require_exact_object(input_value, input_keys_expected, context)
        input_key_int = _observation_int64(
            item["input_key"], f"{context}.input_key", local_key=True
        )
        network_int = _observation_int64(item["network"], f"{context}.network")
        channel_count = _observation_int64(
            item["channel_count"], f"{context}.channel_count"
        )
        interface = _observation_text(item["interface"], f"{context}.interface")
        if (
            input_key_int < 0
            or network_int < 0
            or network_int > 12
            or channel_count <= 0
            or channel_count > CANONICAL_APT_UID_MAX + 1
            or interface != f"toltec{network_int}"
            or item["input_key"] in inputs
            or item["network"] in networks
            or interface in interfaces
        ):
            raise ContractError(f"{context}: invalid/duplicate input binding")
        inputs[item["input_key"]] = item
        networks.add(item["network"])
        interfaces.add(interface)
        expected_rows += channel_count
        for role in ("raw", "kmp"):
            source = _require_exact_object(
                item[f"{role}_source"], source_keys_expected,
                f"{context}.{role}_source",
            )
            source_key = source["source_key"]
            _observation_int64(
                source_key, f"{context}.{role}_source.source_key", local_key=True
            )
            if (
                source["role"] != role
                or source_key in source_keys
                or source["network"] != item["network"]
                or source["interface"] != item["interface"]
                or source["channel_count"] != item["channel_count"]
                or SHA256_REFERENCE_RE.fullmatch(source["content_sha256"]) is None
                or _observation_uint64(
                    source["byte_count"], f"{context}.{role}_source.byte_count"
                ) == 0
            ):
                raise ContractError(f"{context}.{role}_source: invalid binding")
            source_keys.add(source_key)
            _observation_text(
                source["diagnostic_locator"],
                f"{context}.{role}_source.diagnostic_locator",
            )
            header = _validate_observation_identity(
                source["header_observation"],
                f"{context}.{role}_source.header_observation",
            )
            if role == "raw" and header != observation:
                raise ContractError("raw source header does not bind target observation")
    if expected_rows != len(rows_value):
        raise ContractError("target channel inventory does not cover every row")
    row_keys: set[str] = set()
    relations: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for index, row_value in enumerate(rows_value):
        context = f"target manifest.rows[{index}]"
        row = _require_exact_object(
            row_value,
            {
                "row_key",
                "input_key",
                "kmp_source_key",
                "kmp_row_index",
                "matching_frequency_hz",
                "output_tone_frequency_hz",
                "array",
                "network",
                "channel",
                "fields",
            },
            context,
        )
        _observation_int64(row["row_key"], f"{context}.row_key", local_key=True)
        if row["row_key"] in row_keys or row["input_key"] not in inputs:
            raise ContractError(f"{context}: duplicate row key or foreign input")
        row_keys.add(row["row_key"])
        item = inputs[row["input_key"]]
        channel = _observation_int64(row["channel"], f"{context}.channel")
        kmp_row_index = _observation_int64(
            row["kmp_row_index"], f"{context}.kmp_row_index"
        )
        _observation_int64(
            row["kmp_source_key"],
            f"{context}.kmp_source_key",
            local_key=True,
        )
        network = _observation_int64(row["network"], f"{context}.network")
        array = _observation_int64(row["array"], f"{context}.array")
        channel_count = _observation_int64(
            item["channel_count"], f"{context}.input.channel_count"
        )
        matching = _observation_float64(
            row["matching_frequency_hz"], f"{context}.matching_frequency_hz"
        )
        output_tone = _observation_float64(
            row["output_tone_frequency_hz"], f"{context}.output_tone_frequency_hz"
        )
        relation = (row["network"], row["channel"])
        if (
            row["network"] != item["network"]
            or row["kmp_source_key"] != item["kmp_source"]["source_key"]
            or kmp_row_index != channel
            or channel < 0
            or channel >= channel_count
            or array != _expected_array(network)
            or not _is_finite_binary64_token(matching)
            or not _is_finite_binary64_token(output_tone)
            or relation in relations
        ):
            raise ContractError(f"{context}: invalid target row/raw relation")
        row_fields = _mapping(row["fields"], f"{context}.fields")
        if set(row_fields) != declared_names:
            raise ContractError(f"{context}: target field catalog is incomplete")
        for name, field in registered_fields.items():
            _validate_successor_value(
                row_fields[name], field, f"{context}.fields.{name}"
            )
        if (
            row_fields["kids_fr"] != matching
            or row_fields["kids_f_out"] != output_tone
        ):
            raise ContractError(
                f"{context}: structural frequency aliases differ from exact KMP fields"
            )
        relations.add(relation)
        counts[row["input_key"]] = counts.get(row["input_key"], 0) + 1
    for key, item in inputs.items():
        if counts.get(key, 0) != _observation_int64(
            item["channel_count"], "target input channel count"
        ):
            raise ContractError("target row/raw/KMP relation is not a complete bijection")
    _validate_permutation(
        target["target_source_sequence"], row_keys, "target source sequence"
    )
    _validate_permutation(
        target["target_application_sequence"],
        row_keys,
        "target application sequence",
    )
    return target


def canonical_observation_target_preimage(
    document: Any, contract: dict[str, Any]
) -> bytes:
    target = validate_observation_target_manifest_v1(document, contract)
    frames: list[bytes] = []
    for label, value in (
        ("encoding", "citlali-labelled-type-length-v1"),
        ("scope", contract["semantic_scope"]),
        ("schema", target["schema_version"]),
        ("contract-authority", target["contract_authority"]),
        ("observation-value-issuer", target["observation_value_issuer"]),
    ):
        _frame_text(frames, label, value)
    _add_observation_identity(frames, "observation", target["observation"])
    _frame_text(
        frames,
        "kmp-source-field-map.profile",
        OBSERVATION_KMP_SOURCE_FIELD_MAP_PROFILE_V1,
    )
    source_field_map = (
        ("fr", "kids_fr", True),
        ("f_out", "kids_f_out", True),
        ("Qr", "kids_Qr", True),
        ("flag", "kids_flag", False),
    )
    _frame_uint64(frames, "kmp-source-field-map.count", len(source_field_map))
    for index, (source_column, canonical_field, required) in enumerate(
        source_field_map
    ):
        prefix = f"kmp-source-field-map.{index}"
        _frame_text(frames, f"{prefix}.source-column", source_column)
        _frame_text(frames, f"{prefix}.canonical-field", canonical_field)
        _frame_bool(frames, f"{prefix}.required", required)
    fields = sorted(target["registered_fields"], key=lambda field: field["name"])
    _frame_uint64(frames, "field.count", len(fields))
    for index, field in enumerate(fields):
        _add_typed_field(frames, f"field.{index}", field)
    inputs = sorted(
        target["inputs"], key=lambda item: _observation_int64(item["input_key"], "input key")
    )
    _frame_uint64(frames, "input.count", len(inputs))
    for index, item in enumerate(inputs):
        prefix = f"input.{index}"
        _frame_int64(frames, f"{prefix}.input-key", item["input_key"])
        _frame_int64(frames, f"{prefix}.network", item["network"])
        _frame_text(frames, f"{prefix}.interface", item["interface"])
        _frame_int64(frames, f"{prefix}.channel-count", item["channel_count"])
        for role in ("raw", "kmp"):
            source = item[f"{role}_source"]
            source_prefix = f"{prefix}.{role}"
            _frame_int64(frames, f"{source_prefix}.source-key", source["source_key"])
            _frame_text(frames, f"{source_prefix}.role", source["role"])
            _frame_text(
                frames, f"{source_prefix}.content-sha256", source["content_sha256"]
            )
            _frame_uint64(frames, f"{source_prefix}.byte-count", source["byte_count"])
            _add_observation_identity(
                frames, f"{source_prefix}.header", source["header_observation"]
            )
            _frame_int64(frames, f"{source_prefix}.network", source["network"])
            _frame_text(frames, f"{source_prefix}.interface", source["interface"])
            _frame_int64(
                frames, f"{source_prefix}.channel-count", source["channel_count"]
            )
    rows = sorted(
        target["rows"], key=lambda row: _observation_int64(row["row_key"], "row key")
    )
    _frame_uint64(frames, "row.count", len(rows))
    for index, row in enumerate(rows):
        prefix = f"row.{index}"
        _frame_int64(frames, f"{prefix}.row-key", row["row_key"])
        _frame_int64(frames, f"{prefix}.input-key", row["input_key"])
        _frame_int64(frames, f"{prefix}.kmp-source-key", row["kmp_source_key"])
        _frame_int64(frames, f"{prefix}.kmp-row-index", row["kmp_row_index"])
        _frame_float64(
            frames, f"{prefix}.matching-frequency-hz", row["matching_frequency_hz"]
        )
        _frame_float64(
            frames,
            f"{prefix}.output-tone-frequency-hz",
            row["output_tone_frequency_hz"],
        )
        _frame_int64(frames, f"{prefix}.array", row["array"])
        _frame_int64(frames, f"{prefix}.network", row["network"])
        _frame_int64(frames, f"{prefix}.channel", row["channel"])
        for field in fields:
            _add_successor_value(
                frames,
                f"{prefix}.field.{field['name']}",
                row["fields"][field["name"]],
                field,
            )
    for name in ("target_source_sequence", "target_application_sequence"):
        label = name.replace("_", "-")
        sequence = target[name]
        _frame_uint64(frames, f"{label}.count", len(sequence))
        for index, key in enumerate(sequence):
            _frame_int64(frames, f"{label}.{index}", key)
    return b"".join(frames)


def observation_target_digests(
    document: Any, contract: dict[str, Any]
) -> dict[str, str]:
    semantic = "sha256:" + hashlib.sha256(
        canonical_observation_target_preimage(document, contract)
    ).hexdigest()
    target = validate_observation_target_manifest_v1(document, contract)
    frames: list[bytes] = []
    _add_envelope(
        frames,
        target["schema_version"],
        contract["envelope_scope"],
        semantic,
        target["envelope"],
    )
    sources = sorted(
        (
            source
            for item in target["inputs"]
            for source in (item["raw_source"], item["kmp_source"])
        ),
        key=lambda source: int(source["source_key"]),
    )
    _frame_uint64(frames, "source-locator.count", len(sources))
    for index, source in enumerate(sources):
        prefix = f"source-locator.{index}"
        _frame_int64(frames, f"{prefix}.source-key", source["source_key"])
        _frame_text(frames, f"{prefix}.role", source["role"])
        _frame_text(
            frames, f"{prefix}.diagnostic-locator",
            source["diagnostic_locator"],
        )
    envelope = "sha256:" + hashlib.sha256(b"".join(frames)).hexdigest()
    return {"semantic_sha256": semantic, "envelope_sha256": envelope}


def observation_target_identity(
    document: Any, contract: dict[str, Any]
) -> dict[str, Any]:
    target = validate_observation_target_manifest_v1(document, contract)
    digests = observation_target_digests(target, contract)
    return {
        "schema": target["schema_version"],
        "occurrence": target["envelope"]["occurrence"],
        **digests,
    }


def validate_match_dispositions_v1(
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: dict[str, Any],
    target_contract: dict[str, Any],
) -> dict[str, Any]:
    _validate_observation_artifact_contract(
        contract.get("artifact_contract_id", ""), contract
    )
    if contract["artifact_contract_id"] != MATCH_DISPOSITIONS_ARTIFACT_CONTRACT_ID:
        raise ContractError("relation validator received the wrong contract")
    target = validate_observation_target_manifest_v1(
        target_document, target_contract
    )
    relation = _require_exact_object(
        document,
        {
            "schema_version",
            "contract_authority",
            "observation_value_issuer",
            "mapping_domain",
            "envelope",
            "baseline_parent",
            "target_parent",
            "matcher",
            "network_evidence",
            "pairs",
            "target_dispositions",
            "seed_dispositions",
            "seed_source_sequence",
        },
        "match relation",
    )
    if (
        relation["schema_version"] != MATCH_DISPOSITIONS_SCHEMA_V1
        or relation["contract_authority"] != OBSERVATION_CONTRACT_AUTHORITY_V1
        or relation["observation_value_issuer"] != OBSERVATION_VALUE_ISSUER_V1
        or relation["mapping_domain"] != OBSERVATION_MAPPING_DOMAIN_V1
    ):
        raise ContractError("relation schema/authority/mapping-domain mismatch")
    envelope = _validate_issuance_envelope(
        relation["envelope"], "match relation.envelope"
    )
    _validate_baseline_reference(
        relation["baseline_parent"], baseline_descriptor,
        "match relation.baseline_parent",
    )
    target_identity = observation_target_identity(target, target_contract)
    if relation["target_parent"] != target_identity:
        raise ContractError("match relation target parent mismatch")
    baseline_identity = _baseline_artifact_identity(baseline_descriptor)
    occurrences = {
        baseline_identity["occurrence"],
        target_identity["occurrence"],
        envelope["occurrence"],
    }
    if len(occurrences) != 3:
        raise ContractError("baseline, target, and relation occurrences must differ")
    matcher = _require_exact_object(
        relation["matcher"],
        {
            "matcher_run_occurrence",
            "implementation_revision",
            "configuration_reference",
            "target_frequency_field",
            "target_quality_factor_field",
            "method",
            "backend",
        },
        "match relation.matcher",
    )
    for key, value in matcher.items():
        _observation_text(value, f"match relation.matcher.{key}")
    if (
        matcher["target_frequency_field"] != "kids_fr"
        or matcher["target_quality_factor_field"] != "kids_Qr"
    ):
        raise ContractError("match relation matcher field registry mismatch")
    target_keys = {row["row_key"] for row in target["rows"]}
    seed_keys = {row["uid"] for row in baseline_descriptor["rows"]}
    _validate_permutation(
        relation["seed_source_sequence"], seed_keys, "seed source sequence"
    )
    pairs_by_key: dict[str, dict[str, Any]] = {}
    target_pairs: dict[str, set[str]] = {}
    seed_pairs: dict[str, set[str]] = {}
    endpoints: set[tuple[str, str]] = set()
    relation_row_keys: set[str] = set()
    pair_keys_exact = {
        "pair_key",
        "target",
        "seed",
        "separation_hz",
        "is_good_match",
    }
    for index, pair_value in enumerate(_list(relation["pairs"], "match relation.pairs")):
        context = f"match relation.pairs[{index}]"
        pair = _require_exact_object(pair_value, pair_keys_exact, context)
        key = pair["pair_key"]
        _observation_int64(key, f"{context}.pair_key", local_key=True)
        target_ref = _validate_row_reference(pair["target"], f"{context}.target")
        seed_ref = _validate_row_reference(pair["seed"], f"{context}.seed")
        expected_target = _row_reference(target_identity, target_ref["local_key"])
        expected_seed = _row_reference(baseline_identity, seed_ref["local_key"])
        separation = _observation_float64(
            pair["separation_hz"], f"{context}.separation_hz"
        )
        endpoint = (target_ref["local_key"], seed_ref["local_key"])
        if (
            key in relation_row_keys
            or target_ref != expected_target
            or seed_ref != expected_seed
            or target_ref["local_key"] not in target_keys
            or seed_ref["local_key"] not in seed_keys
            or not _is_finite_binary64_token(separation)
            or _binary64_value(separation) < 0.0
            or endpoint in endpoints
            or not isinstance(pair["is_good_match"], bool)
        ):
            raise ContractError(f"{context}: invalid pair key/endpoints/evidence")
        relation_row_keys.add(key)
        pairs_by_key[key] = pair
        endpoints.add(endpoint)
        target_pairs.setdefault(endpoint[0], set()).add(key)
        seed_pairs.setdefault(endpoint[1], set()).add(key)
    target_networks = {row["network"] for row in target["rows"]}
    evidence_networks: set[str] = set()
    for index, evidence_value in enumerate(_list(
        relation["network_evidence"], "match relation.network_evidence"
    )):
        context = f"match relation.network_evidence[{index}]"
        evidence = _require_exact_object(
            evidence_value,
            {
                "network",
                "frequency_shift_hz",
                "gate_hz",
                "quality_factor",
                "quality_factor_field",
                "quality_factor_authority_reference",
            },
            context,
        )
        shift = _observation_float64(
            evidence["frequency_shift_hz"], f"{context}.frequency_shift_hz"
        )
        gate = _observation_float64(evidence["gate_hz"], f"{context}.gate_hz")
        quality_factor = _observation_float64(
            evidence["quality_factor"], f"{context}.quality_factor"
        )
        if (
            evidence["network"] not in target_networks
            or evidence["network"] in evidence_networks
            or not _is_finite_binary64_token(shift)
            or not _is_finite_binary64_token(gate)
            or not _is_finite_binary64_token(quality_factor)
            or _binary64_value(gate) < 0.0
            or evidence["quality_factor_field"] != "kids_Qr"
            or evidence["quality_factor_authority_reference"]
            != "kids:model-params-v1"
        ):
            raise ContractError(f"{context}: invalid/incomplete network evidence")
        evidence_networks.add(evidence["network"])
    if evidence_networks != target_networks:
        raise ContractError("relation requires evidence for every target network")
    disposition_keys_exact = {
        "disposition_key",
        "endpoint",
        "state",
        "pair_keys",
        "reason",
    }

    def validate_dispositions(
        values: Any,
        expected_keys: set[str],
        expected_pairs: dict[str, set[str]],
        identity: dict[str, Any],
        target_side: bool,
        context: str,
    ) -> None:
        seen: set[str] = set()
        for index, disposition_value in enumerate(_list(values, context)):
            row_context = f"{context}[{index}]"
            disposition = _require_exact_object(
                disposition_value, disposition_keys_exact, row_context
            )
            disposition_key = disposition["disposition_key"]
            _observation_int64(
                disposition_key, f"{row_context}.disposition_key", local_key=True
            )
            endpoint = _validate_row_reference(
                disposition["endpoint"], f"{row_context}.endpoint"
            )
            endpoint_key = endpoint["local_key"]
            pair_keys = _list(disposition["pair_keys"], f"{row_context}.pair_keys")
            for pair_index, pair_key in enumerate(pair_keys):
                _observation_int64(
                    pair_key,
                    f"{row_context}.pair_keys[{pair_index}]",
                    local_key=True,
                )
            if pair_keys != sorted(pair_keys, key=int) or len(pair_keys) != len(set(pair_keys)):
                raise ContractError(f"{row_context}.pair_keys: not a sorted unique set")
            expected = expected_pairs.get(endpoint_key, set())
            expected_state = "matched" if expected else "unmatched" if target_side else "unused"
            if (
                disposition_key in relation_row_keys
                or endpoint_key not in expected_keys
                or endpoint_key in seen
                or endpoint != _row_reference(identity, endpoint_key)
                or set(pair_keys) != expected
                or disposition["state"] != expected_state
            ):
                raise ContractError(f"{row_context}: incomplete/nonreciprocal disposition")
            _observation_text(disposition["reason"], f"{row_context}.reason")
            relation_row_keys.add(disposition_key)
            seen.add(endpoint_key)
        if seen != expected_keys:
            raise ContractError(f"{context}: does not cover every endpoint")

    validate_dispositions(
        relation["target_dispositions"],
        target_keys,
        target_pairs,
        target_identity,
        True,
        "match relation.target_dispositions",
    )
    validate_dispositions(
        relation["seed_dispositions"],
        seed_keys,
        seed_pairs,
        baseline_identity,
        False,
        "match relation.seed_dispositions",
    )
    return relation


def canonical_match_dispositions_preimage(
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: dict[str, Any],
    target_contract: dict[str, Any],
) -> bytes:
    relation = validate_match_dispositions_v1(
        document, contract, baseline_descriptor, target_document, target_contract
    )
    frames: list[bytes] = []
    for label, value in (
        ("encoding", "citlali-labelled-type-length-v1"),
        ("scope", contract["semantic_scope"]),
        ("schema", relation["schema_version"]),
        ("contract-authority", relation["contract_authority"]),
        ("observation-value-issuer", relation["observation_value_issuer"]),
        ("mapping-domain", relation["mapping_domain"]),
    ):
        _frame_text(frames, label, value)
    _add_baseline_reference(frames, "baseline-parent", relation["baseline_parent"])
    _add_artifact_identity(frames, "target-parent", relation["target_parent"])
    matcher = relation["matcher"]
    for label, key in (
        ("matcher.run-occurrence", "matcher_run_occurrence"),
        ("matcher.implementation-revision", "implementation_revision"),
        ("matcher.configuration-reference", "configuration_reference"),
        ("matcher.target-frequency-field", "target_frequency_field"),
        ("matcher.target-quality-factor-field", "target_quality_factor_field"),
        ("matcher.method", "method"),
        ("matcher.backend", "backend"),
    ):
        _frame_text(frames, label, matcher[key])
    evidence = sorted(relation["network_evidence"], key=lambda item: int(item["network"]))
    _frame_uint64(frames, "network-evidence.count", len(evidence))
    for index, item in enumerate(evidence):
        prefix = f"network-evidence.{index}"
        _frame_int64(frames, f"{prefix}.network", item["network"])
        _frame_float64(frames, f"{prefix}.frequency-shift-hz", item["frequency_shift_hz"])
        _frame_float64(frames, f"{prefix}.gate-hz", item["gate_hz"])
        _frame_float64(frames, f"{prefix}.quality-factor", item["quality_factor"])
        _frame_text(
            frames, f"{prefix}.quality-factor-field",
            item["quality_factor_field"],
        )
        _frame_text(
            frames, f"{prefix}.quality-factor-authority-reference",
            item["quality_factor_authority_reference"],
        )
    pairs = sorted(relation["pairs"], key=lambda item: int(item["pair_key"]))
    _frame_uint64(frames, "pair.count", len(pairs))
    for index, pair in enumerate(pairs):
        prefix = f"pair.{index}"
        _frame_int64(frames, f"{prefix}.pair-key", pair["pair_key"])
        _add_row_reference(frames, f"{prefix}.target", pair["target"])
        _add_row_reference(frames, f"{prefix}.seed", pair["seed"])
        _frame_float64(frames, f"{prefix}.separation-hz", pair["separation_hz"])
        _frame_bool(frames, f"{prefix}.is-good-match", pair["is_good_match"])
    for label, key in (
        ("target-disposition", "target_dispositions"),
        ("seed-disposition", "seed_dispositions"),
    ):
        dispositions = sorted(
            relation[key], key=lambda item: int(item["disposition_key"])
        )
        _frame_uint64(frames, f"{label}.count", len(dispositions))
        for index, disposition in enumerate(dispositions):
            prefix = f"{label}.{index}"
            _frame_int64(
                frames, f"{prefix}.disposition-key", disposition["disposition_key"]
            )
            _add_row_reference(frames, f"{prefix}.endpoint", disposition["endpoint"])
            _frame_text(frames, f"{prefix}.state", disposition["state"])
            _frame_uint64(frames, f"{prefix}.pair-key.count", len(disposition["pair_keys"]))
            for pair_index, pair_key in enumerate(disposition["pair_keys"]):
                _frame_int64(frames, f"{prefix}.pair-key.{pair_index}", pair_key)
            _frame_text(frames, f"{prefix}.reason", disposition["reason"])
    sequence = relation["seed_source_sequence"]
    _frame_uint64(frames, "seed-source-sequence.count", len(sequence))
    for index, key in enumerate(sequence):
        _frame_int64(frames, f"seed-source-sequence.{index}", key)
    return b"".join(frames)


def match_dispositions_digests(
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: dict[str, Any],
    target_contract: dict[str, Any],
) -> dict[str, str]:
    semantic = "sha256:" + hashlib.sha256(
        canonical_match_dispositions_preimage(
            document, contract, baseline_descriptor, target_document, target_contract
        )
    ).hexdigest()
    relation = validate_match_dispositions_v1(
        document, contract, baseline_descriptor, target_document, target_contract
    )
    frames: list[bytes] = []
    _add_envelope(
        frames,
        relation["schema_version"],
        contract["envelope_scope"],
        semantic,
        relation["envelope"],
    )
    return {
        "semantic_sha256": semantic,
        "envelope_sha256": "sha256:" + hashlib.sha256(b"".join(frames)).hexdigest(),
    }


def match_dispositions_identity(
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: dict[str, Any],
    target_contract: dict[str, Any],
) -> dict[str, Any]:
    relation = validate_match_dispositions_v1(
        document, contract, baseline_descriptor, target_document, target_contract
    )
    digests = match_dispositions_digests(
        relation, contract, baseline_descriptor, target_document, target_contract
    )
    return {
        "schema": relation["schema_version"],
        "occurrence": relation["envelope"]["occurrence"],
        **digests,
    }


def canonical_output_field_contracts_v1(
    baseline_descriptor: dict[str, Any],
    target_document: dict[str, Any],
    target_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_descriptor_preimage(baseline_descriptor)
    target = validate_observation_target_manifest_v1(
        target_document, target_contract
    )
    result: dict[str, dict[str, Any]] = {}
    for source in target["registered_fields"]:
        field = copy.deepcopy(source)
        field["registry"] = OBSERVATION_MATCHED_OUTPUT_FIELD_REGISTRY_V1
        result[field["name"]] = {
            "field": field,
            "authorized_operation": OBSERVATION_PRESERVE_TARGET_OPERATION_V1,
            "issuer_authority_reference": "",
        }
    for source in baseline_descriptor["registered_fields"]:
        if source["name"] in _target_field_catalog_v1():
            continue
        field = {
            key: copy.deepcopy(source[key])
            for key in (
                "name",
                "datatype",
                "unit",
                "nullable",
                "nonfinite",
                "authority",
                "authority_reference",
                "registry",
                "description",
                "identity_role",
            )
        }
        field["source_column"] = None
        field["nullable"] = True
        field["registry"] = OBSERVATION_MATCHED_OUTPUT_FIELD_REGISTRY_V1
        if field["name"] in result:
            raise ContractError(
                "verified baseline field collides with a reserved target field"
            )
        result[field["name"]] = {
            "field": field,
            "authorized_operation": OBSERVATION_COPY_BASELINE_OR_NULL_OPERATION_V1,
            "issuer_authority_reference": "",
        }
    return [result[name] for name in sorted(result)]


def _output_contract_map(
    value: Any,
    expected: list[dict[str, Any]],
    context: str,
) -> dict[str, dict[str, Any]]:
    values = _list(value, context)
    result: dict[str, dict[str, Any]] = {}
    for index, contract_value in enumerate(values):
        item_context = f"{context}[{index}]"
        item = _require_exact_object(
            contract_value,
            {"field", "authorized_operation", "issuer_authority_reference"},
            item_context,
        )
        field = _validate_typed_field(
            item["field"], OBSERVATION_MATCHED_OUTPUT_FIELD_REGISTRY_V1,
            f"{item_context}.field",
        )
        if field["name"] in result:
            raise ContractError(f"{context}: duplicate output field")
        if item["authorized_operation"] not in {
            OBSERVATION_PRESERVE_TARGET_OPERATION_V1,
            OBSERVATION_COPY_BASELINE_OR_NULL_OPERATION_V1,
        } or item["issuer_authority_reference"] != "":
            raise ContractError(f"{item_context}: unauthorized operation")
        result[field["name"]] = item
    expected_map = {item["field"]["name"]: item for item in expected}
    if result != expected_map:
        raise ContractError(
            f"{context}: differs from immutable target/baseline-derived registry"
        )
    return result


def _descriptor_row_map(
    descriptor: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {row["uid"]: row for row in descriptor["rows"]}


def validate_observation_matched_apt_v1(
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: dict[str, Any],
    target_contract: dict[str, Any],
    relation_document: dict[str, Any],
    relation_contract: dict[str, Any],
) -> dict[str, Any]:
    _validate_observation_artifact_contract(
        contract.get("artifact_contract_id", ""), contract
    )
    if contract["artifact_contract_id"] != OBSERVATION_MATCHED_APT_ARTIFACT_CONTRACT_ID:
        raise ContractError("matched-output validator received the wrong contract")
    target = validate_observation_target_manifest_v1(
        target_document, target_contract
    )
    relation = validate_match_dispositions_v1(
        relation_document,
        relation_contract,
        baseline_descriptor,
        target,
        target_contract,
    )
    output = _require_exact_object(
        document,
        {
            "schema_version",
            "contract_authority",
            "observation_value_issuer",
            "transformation_registry",
            "envelope",
            "baseline_parent",
            "target_parent",
            "relation_parent",
            "registered_fields",
            "rows",
            "output_presentation_sequence",
        },
        "matched output",
    )
    if (
        output["schema_version"] != OBSERVATION_MATCHED_APT_SCHEMA_V1
        or output["contract_authority"] != OBSERVATION_CONTRACT_AUTHORITY_V1
        or output["observation_value_issuer"] != OBSERVATION_VALUE_ISSUER_V1
        or output["transformation_registry"]
        != OBSERVATION_TRANSFORMATION_REGISTRY_V1
    ):
        raise ContractError("matched output schema/authority/registry mismatch")
    envelope = _validate_issuance_envelope(
        output["envelope"], "matched output.envelope"
    )
    _validate_baseline_reference(
        output["baseline_parent"], baseline_descriptor,
        "matched output.baseline_parent",
    )
    target_identity = observation_target_identity(target, target_contract)
    relation_identity = match_dispositions_identity(
        relation,
        relation_contract,
        baseline_descriptor,
        target,
        target_contract,
    )
    if output["target_parent"] != target_identity:
        raise ContractError("matched output target parent mismatch")
    if output["relation_parent"] != relation_identity:
        raise ContractError("matched output relation parent mismatch")
    baseline_identity = _baseline_artifact_identity(baseline_descriptor)
    if len({
        baseline_identity["occurrence"],
        target_identity["occurrence"],
        relation_identity["occurrence"],
        envelope["occurrence"],
    }) != 4:
        raise ContractError(
            "matched-output occurrence must differ from every parent occurrence"
        )
    expected_contracts = canonical_output_field_contracts_v1(
        baseline_descriptor, target, target_contract
    )
    output_fields = _output_contract_map(
        output["registered_fields"], expected_contracts,
        "matched output.registered_fields",
    )
    target_fields = {
        field["name"]: field for field in target["registered_fields"]
    }
    baseline_fields = {
        field["name"]: field
        for field in baseline_descriptor["registered_fields"]
        if field["name"] not in _target_field_catalog_v1()
    }
    target_rows = {row["row_key"]: row for row in target["rows"]}
    baseline_rows = _descriptor_row_map(baseline_descriptor)
    pairs = {pair["pair_key"]: pair for pair in relation["pairs"]}
    dispositions = {
        item["endpoint"]["local_key"]: item
        for item in relation["target_dispositions"]
    }
    output_keys: set[str] = set()
    covered_targets: set[str] = set()
    transformation_keys = {
        "field_name",
        "operation",
        "before",
        "after",
        "value_source",
        "source_pair_key",
        "source_row",
        "authority_reference",
        "provenance_reference",
    }
    row_keys = {
        "uid",
        "target",
        "target_input_key",
        "tone_frequency_hz",
        "array",
        "network",
        "channel",
        "relation_pair_keys",
        "fields",
        "transformations",
    }
    for index, row_value in enumerate(_list(output["rows"], "matched output.rows")):
        context = f"matched output.rows[{index}]"
        row = _require_exact_object(row_value, row_keys, context)
        _observation_int64(row["uid"], f"{context}.uid", local_key=True)
        target_reference = _validate_row_reference(
            row["target"], f"{context}.target"
        )
        target_key = target_reference["local_key"]
        if (
            row["uid"] in output_keys
            or target_key in covered_targets
            or target_key not in target_rows
            or target_reference != _row_reference(target_identity, target_key)
        ):
            raise ContractError(f"{context}: duplicate/foreign local reference")
        output_keys.add(row["uid"])
        covered_targets.add(target_key)
        target_row = target_rows[target_key]
        tone = _observation_float64(
            row["tone_frequency_hz"], f"{context}.tone_frequency_hz"
        )
        for key in ("target_input_key", "array", "network", "channel"):
            _observation_int64(row[key], f"{context}.{key}")
        if (
            row["target_input_key"] != target_row["input_key"]
            or tone != target_row["output_tone_frequency_hz"]
            or row["array"] != target_row["array"]
            or row["network"] != target_row["network"]
            or row["channel"] != target_row["channel"]
        ):
            raise ContractError(f"{context}: structural/raw target values changed")
        pair_keys = _list(
            row["relation_pair_keys"], f"{context}.relation_pair_keys"
        )
        for pair_index, pair_key in enumerate(pair_keys):
            _observation_int64(
                pair_key,
                f"{context}.relation_pair_keys[{pair_index}]",
                local_key=True,
            )
        if (
            pair_keys != sorted(pair_keys, key=int)
            or len(pair_keys) != len(set(pair_keys))
            or pair_keys != dispositions[target_key]["pair_keys"]
        ):
            raise ContractError(f"{context}: incomplete/reordered relation pair set")
        row_fields = _mapping(row["fields"], f"{context}.fields")
        if set(row_fields) != set(output_fields):
            raise ContractError(f"{context}: incomplete output field map")
        transformations: dict[str, dict[str, Any]] = {}
        for transform_index, transform_value in enumerate(
            _list(row["transformations"], f"{context}.transformations")
        ):
            transform_context = f"{context}.transformations[{transform_index}]"
            transform = _require_exact_object(
                transform_value, transformation_keys, transform_context
            )
            field_name = _observation_text(
                transform["field_name"], f"{transform_context}.field_name"
            )
            if field_name in transformations or field_name not in output_fields:
                raise ContractError(f"{transform_context}: unknown/duplicate field")
            transformations[field_name] = transform
        if set(transformations) != set(output_fields):
            raise ContractError(f"{context}: incomplete transformation set")
        for name, field_contract in output_fields.items():
            field = field_contract["field"]
            value = _validate_successor_value(
                row_fields[name], field, f"{context}.fields.{name}"
            )
            change = transformations[name]
            before = _validate_successor_value(
                change["before"],
                {**field, "nullable": True},
                f"{context}.transformations.{name}.before",
            )
            after = _validate_successor_value(
                change["after"], field,
                f"{context}.transformations.{name}.after",
            )
            if (
                change["operation"] != field_contract["authorized_operation"]
                or after != value
            ):
                raise ContractError(f"{context}: unauthorized transformation for {name}")
            _observation_text(
                change["authority_reference"],
                f"{context}.transformations.{name}.authority_reference",
            )
            provenance = _observation_text(
                change["provenance_reference"],
                f"{context}.transformations.{name}.provenance_reference",
            )
            if name in target_fields:
                expected_provenance = (
                    f"target-kmp-source:{target_row['kmp_source_key']}:"
                    f"row:{target_row['kmp_row_index']}:"
                    f"column:{target_fields[name]['source_column']}"
                )
                if (
                    before != target_row["fields"][name]
                    or after != target_row["fields"][name]
                    or change["value_source"] != "target-row"
                    or change["source_pair_key"] is not None
                    or change["source_row"] != target_reference
                    or change["authority_reference"]
                    != target_fields[name]["authority_reference"]
                    or provenance != expected_provenance
                ):
                    raise ContractError(
                        f"{context}: target field is not exact KMP-preserved {name}"
                    )
                continue
            if before is not None:
                raise ContractError(f"{context}: baseline field before must be typed null")
            if pair_keys:
                if change["source_pair_key"] not in pair_keys:
                    raise ContractError(f"{context}: baseline source pair is absent")
                _observation_int64(
                    change["source_pair_key"],
                    f"{context}.transformations.{name}.source_pair_key",
                    local_key=True,
                )
                pair = pairs[change["source_pair_key"]]
                source_row = _validate_row_reference(
                    change["source_row"],
                    f"{context}.transformations.{name}.source_row",
                )
                seed_key = pair["seed"]["local_key"]
                expected_cell = baseline_rows[seed_key]["fields"][name]["value"]
                if (
                    pair["target"] != target_reference
                    or source_row != pair["seed"]
                    or after != expected_cell
                    or change["value_source"] != "baseline-seed-row"
                    or change["authority_reference"]
                    != baseline_fields[name]["authority_reference"]
                    or provenance != f"relation-pair:{change['source_pair_key']}"
                ):
                    raise ContractError(f"{context}: baseline source mismatch for {name}")
            elif (
                after is not None
                or change["value_source"] != "canonical-null"
                or change["source_pair_key"] is not None
                or change["source_row"] is not None
                or change["authority_reference"]
                != OBSERVATION_UNMATCHED_MISSING_AUTHORITY_V1
                or provenance != "target-unmatched:no-fabricated-seed"
            ):
                raise ContractError(f"{context}: unmatched field is not typed missing")
    if covered_targets != set(target_rows):
        raise ContractError("matched output does not cover every target exactly once")
    _validate_permutation(
        output["output_presentation_sequence"],
        output_keys,
        "matched output presentation sequence",
    )
    return output


def canonical_observation_matched_apt_preimage(
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: dict[str, Any],
    target_contract: dict[str, Any],
    relation_document: dict[str, Any],
    relation_contract: dict[str, Any],
) -> bytes:
    output = validate_observation_matched_apt_v1(
        document, contract, baseline_descriptor, target_document,
        target_contract, relation_document, relation_contract
    )
    frames: list[bytes] = []
    for label, value in (
        ("encoding", "citlali-labelled-type-length-v1"),
        ("scope", contract["semantic_scope"]),
        ("schema", output["schema_version"]),
        ("contract-authority", output["contract_authority"]),
        ("observation-value-issuer", output["observation_value_issuer"]),
        ("transformation-registry", output["transformation_registry"]),
    ):
        _frame_text(frames, label, value)
    _add_baseline_reference(frames, "baseline-parent", output["baseline_parent"])
    _add_artifact_identity(frames, "target-parent", output["target_parent"])
    _add_artifact_identity(frames, "relation-parent", output["relation_parent"])
    fields = sorted(output["registered_fields"], key=lambda item: item["field"]["name"])
    field_map = {item["field"]["name"]: item["field"] for item in fields}
    _frame_uint64(frames, "field.count", len(fields))
    for index, field_contract in enumerate(fields):
        prefix = f"field.{index}"
        _add_typed_field(frames, prefix, field_contract["field"])
        _frame_text(
            frames, f"{prefix}.authorized-operation",
            field_contract["authorized_operation"],
        )
        _frame_text(
            frames, f"{prefix}.issuer-authority-reference",
            field_contract["issuer_authority_reference"],
        )
    rows = sorted(output["rows"], key=lambda row: int(row["uid"]))
    _frame_uint64(frames, "row.count", len(rows))
    for index, row in enumerate(rows):
        prefix = f"row.{index}"
        _frame_int64(frames, f"{prefix}.uid", row["uid"])
        _add_row_reference(frames, f"{prefix}.target", row["target"])
        _frame_int64(frames, f"{prefix}.target-input-key", row["target_input_key"])
        _frame_float64(frames, f"{prefix}.tone-frequency-hz", row["tone_frequency_hz"])
        _frame_int64(frames, f"{prefix}.array", row["array"])
        _frame_int64(frames, f"{prefix}.network", row["network"])
        _frame_int64(frames, f"{prefix}.channel", row["channel"])
        _frame_uint64(
            frames, f"{prefix}.relation-pair-key.count",
            len(row["relation_pair_keys"]),
        )
        for pair_index, pair_key in enumerate(row["relation_pair_keys"]):
            _frame_int64(
                frames, f"{prefix}.relation-pair-key.{pair_index}", pair_key
            )
        for field_contract in fields:
            name = field_contract["field"]["name"]
            _add_successor_value(
                frames, f"{prefix}.field.{name}", row["fields"][name],
                field_contract["field"],
            )
        transformations = sorted(
            row["transformations"], key=lambda item: item["field_name"]
        )
        _frame_uint64(frames, f"{prefix}.transformation.count", len(transformations))
        for transform_index, change in enumerate(transformations):
            transform_prefix = f"{prefix}.transformation.{transform_index}"
            field = field_map[change["field_name"]]
            _frame_text(frames, f"{transform_prefix}.field-name", change["field_name"])
            _frame_text(frames, f"{transform_prefix}.operation", change["operation"])
            _add_successor_value(
                frames, f"{transform_prefix}.before", change["before"], field
            )
            _add_successor_value(
                frames, f"{transform_prefix}.after", change["after"], field
            )
            _frame_text(frames, f"{transform_prefix}.value-source", change["value_source"])
            _frame_bool(
                frames, f"{transform_prefix}.has-source-pair-key",
                change["source_pair_key"] is not None,
            )
            if change["source_pair_key"] is not None:
                _frame_int64(
                    frames, f"{transform_prefix}.source-pair-key",
                    change["source_pair_key"],
                )
            _frame_bool(
                frames, f"{transform_prefix}.has-source-row",
                change["source_row"] is not None,
            )
            if change["source_row"] is not None:
                _add_row_reference(
                    frames, f"{transform_prefix}.source-row", change["source_row"]
                )
            _frame_text(
                frames, f"{transform_prefix}.authority-reference",
                change["authority_reference"],
            )
            _frame_text(
                frames, f"{transform_prefix}.provenance-reference",
                change["provenance_reference"],
            )
    sequence = output["output_presentation_sequence"]
    _frame_uint64(frames, "output-presentation-sequence.count", len(sequence))
    for index, key in enumerate(sequence):
        _frame_int64(frames, f"output-presentation-sequence.{index}", key)
    return b"".join(frames)


def observation_matched_apt_digests(
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: dict[str, Any],
    target_contract: dict[str, Any],
    relation_document: dict[str, Any],
    relation_contract: dict[str, Any],
) -> dict[str, str]:
    preimage = canonical_observation_matched_apt_preimage(
        document, contract, baseline_descriptor, target_document,
        target_contract, relation_document, relation_contract
    )
    semantic = "sha256:" + hashlib.sha256(preimage).hexdigest()
    output = validate_observation_matched_apt_v1(
        document, contract, baseline_descriptor, target_document,
        target_contract, relation_document, relation_contract
    )
    frames: list[bytes] = []
    _add_envelope(
        frames, output["schema_version"], contract["envelope_scope"],
        semantic, output["envelope"],
    )
    return {
        "semantic_sha256": semantic,
        "envelope_sha256": "sha256:" + hashlib.sha256(b"".join(frames)).hexdigest(),
    }


def observation_matched_apt_identity(
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: dict[str, Any],
    target_contract: dict[str, Any],
    relation_document: dict[str, Any],
    relation_contract: dict[str, Any],
) -> dict[str, Any]:
    output = validate_observation_matched_apt_v1(
        document, contract, baseline_descriptor, target_document,
        target_contract, relation_document, relation_contract
    )
    return {
        "schema": output["schema_version"],
        "occurrence": output["envelope"]["occurrence"],
        **observation_matched_apt_digests(
            output, contract, baseline_descriptor, target_document,
            target_contract, relation_document, relation_contract
        ),
    }


def serialize_observation_matched_apt_ecsv_v1(
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: Any,
    target_contract: dict[str, Any],
    relation_document: Any,
    relation_contract: dict[str, Any],
) -> dict[str, Any]:
    """Independently serialize the single persisted APT-PROD-002 ECSV."""
    target = validate_observation_target_manifest_v1(
        target_document, target_contract
    )
    relation = validate_match_dispositions_v1(
        relation_document, relation_contract, baseline_descriptor, target,
        target_contract
    )
    output = validate_observation_matched_apt_v1(
        document, contract, baseline_descriptor, target, target_contract,
        relation, relation_contract
    )
    target_digests = observation_target_digests(target, target_contract)
    relation_digests = match_dispositions_digests(
        relation, relation_contract, baseline_descriptor, target,
        target_contract
    )
    output_digests = observation_matched_apt_digests(
        output, contract, baseline_descriptor, target, target_contract,
        relation, relation_contract
    )
    fields = sorted(output["registered_fields"], key=lambda item: item["field"]["name"])

    columns = [
        ("uid", "int64", "N/A", "exact nonnegative output-artifact-local row key; never persistent detector identity"),
        ("target_row_key", "int64", "N/A", "exact target-parent artifact-local row reference"),
        ("target_input_key", "int64", "N/A", "exact target-parent input binding reference"),
        ("tone_freq", "float64", "Hz", "exact target kids_f_out application value; not identity"),
        ("array", "int64", "N/A", "canonical TolTEC array enum; not row identity"),
        ("nw", "int64", "N/A", "target raw-manifest network key"),
        ("kids_tone", "int64", "N/A", "zero-based target raw channel key within network"),
        ("relation_pair_keys", "string", "N/A", "complete sorted relation-local pair-key set using bracketed-int64-set-v1"),
    ]
    columns.extend(
        (
            item["field"]["name"], item["field"]["datatype"],
            item["field"]["unit"], item["field"]["description"],
        )
        for item in fields
    )
    lines: list[str] = ["# %ECSV 1.0", "# ---", "# datatype:"]

    def quoted(prefix: str, value: Any) -> None:
        lines.append(prefix + _yaml_quote(str(value)))

    def integer(prefix: str, value: Any) -> None:
        lines.append(prefix + str(value))

    def boolean(prefix: str, value: bool) -> None:
        lines.append(prefix + ("true" if value else "false"))

    def emit_envelope(indent: str, value: dict[str, Any]) -> None:
        for key in (
            "occurrence", "event_reference", "software_revision",
            "configuration_reference", "event_time_utc",
        ):
            quoted(f"{indent}{key}: ", value[key])

    def emit_identity(indent: str, value: dict[str, Any]) -> None:
        for key in ("schema", "occurrence", "semantic_sha256", "envelope_sha256"):
            quoted(f"{indent}{key}: ", value[key])

    def emit_baseline(indent: str, value: dict[str, Any]) -> None:
        lines.append(f"{indent}artifact:")
        emit_identity(indent + "  ", value["artifact"])
        for key in ("profile", "descriptor_sha256", "transport_scope", "transport_sha256"):
            quoted(f"{indent}{key}: ", value[key])
        integer(f"{indent}byte_count: ", value["byte_count"])
        quoted(f"{indent}receipt_sha256: ", value["receipt_sha256"])
        integer(f"{indent}receipt_byte_count: ", value["receipt_byte_count"])

    def emit_row_reference(indent: str, value: dict[str, Any]) -> None:
        for key in ("artifact_schema", "occurrence", "envelope_sha256"):
            quoted(f"{indent}{key}: ", value[key])
        integer(f"{indent}local_key: ", value["local_key"])

    def emit_typed_field(first: str, indent: str, field: dict[str, Any]) -> None:
        quoted(first + "- name: ", field["name"])
        quoted(indent + "datatype: ", field["datatype"])
        quoted(indent + "unit: ", field["unit"])
        boolean(indent + "nullable: ", field["nullable"])
        quoted(indent + "nonfinite: ", field["nonfinite"])
        quoted(indent + "authority: ", field["authority"])
        quoted(indent + "authority_reference: ", field["authority_reference"])
        quoted(indent + "registry: ", field["registry"])
        quoted(indent + "description: ", field["description"])
        boolean(indent + "has_source_column: ", field["source_column"] is not None)
        if field["source_column"] is not None:
            quoted(indent + "source_column: ", field["source_column"])
        quoted(indent + "identity_role: ", field["identity_role"])

    def emit_source(heading: str, indent: str, name: str, value: dict[str, Any]) -> None:
        lines.append(f"{heading}{name}:")
        integer(indent + "source_key: ", value["source_key"])
        quoted(indent + "role: ", value["role"])
        quoted(indent + "diagnostic_locator: ", value["diagnostic_locator"])
        quoted(indent + "content_sha256: ", value["content_sha256"])
        integer(indent + "byte_count: ", value["byte_count"])
        lines.append(indent + "header_observation:")
        for key in ("observation", "subobservation", "scan"):
            integer(indent + "  " + key + ": ", value["header_observation"][key])
        integer(indent + "network: ", value["network"])
        quoted(indent + "interface: ", value["interface"])
        integer(indent + "channel_count: ", value["channel_count"])

    def value_token(value: Any, datatype: str) -> str:
        if value is None:
            return "null"
        if datatype == "bool":
            return "true" if value else "false"
        return str(value)

    for name, datatype, unit, description in columns:
        quoted("# - name: ", name)
        quoted("#   datatype: ", datatype)
        if unit not in ("", "N/A"):
            quoted("#   unit: ", unit)
        quoted("#   description: ", description)
    lines.extend(["# meta:", "#   canonical_apt_observation_v1:"])
    for prefix, value in (
        ("schema_version", OBSERVATION_MATCHED_APT_SCHEMA_V1),
        ("artifact_contract_id", OBSERVATION_MATCHED_APT_ARTIFACT_CONTRACT_ID),
        ("contract_authority", OBSERVATION_CONTRACT_AUTHORITY_V1),
        ("observation_value_issuer", OBSERVATION_VALUE_ISSUER_V1),
        ("field_registry", OBSERVATION_MATCHED_OUTPUT_FIELD_REGISTRY_V1),
        ("transformation_registry", OBSERVATION_TRANSFORMATION_REGISTRY_V1),
        ("framing_encoding", "citlali-labelled-type-length-v1"),
        ("semantic_scope", contract["semantic_scope"]),
        ("semantic_sha256", output_digests["semantic_sha256"]),
        ("envelope_scope", contract["envelope_scope"]),
        ("envelope_sha256", output_digests["envelope_sha256"]),
        ("byte_transport_scope", contract["byte_transport_scope"]),
        ("target_semantic_scope", target_contract["semantic_scope"]),
        ("target_semantic_sha256", target_digests["semantic_sha256"]),
        ("target_envelope_scope", target_contract["envelope_scope"]),
        ("target_envelope_sha256", target_digests["envelope_sha256"]),
        ("relation_semantic_scope", relation_contract["semantic_scope"]),
        ("relation_semantic_sha256", relation_digests["semantic_sha256"]),
        ("relation_envelope_scope", relation_contract["envelope_scope"]),
        ("relation_envelope_sha256", relation_digests["envelope_sha256"]),
    ):
        quoted(f"#     {prefix}: ", value)

    lines.append("#     output:")
    for key in ("schema_version", "contract_authority", "observation_value_issuer", "transformation_registry"):
        label = "schema" if key == "schema_version" else key
        quoted(f"#       {label}: ", output[key])
    lines.append("#       envelope:")
    emit_envelope("#         ", output["envelope"])
    lines.append("#       baseline_parent:")
    emit_baseline("#         ", output["baseline_parent"])
    lines.append("#       target_parent:")
    emit_identity("#         ", output["target_parent"])
    lines.append("#       relation_parent:")
    emit_identity("#         ", output["relation_parent"])
    integer("#       registered_field_count: ", len(fields))
    lines.append("#       registered_fields:")
    for item in fields:
        emit_typed_field("#         ", "#           ", item["field"])
        quoted("#           authorized_operation: ", item["authorized_operation"])
        quoted("#           issuer_authority_reference: ", item["issuer_authority_reference"])
    integer("#       output_presentation_sequence_count: ", len(output["output_presentation_sequence"]))
    lines.append("#       output_presentation_sequence:")
    for uid in output["output_presentation_sequence"]:
        integer("#         - ", uid)
    output_rows = sorted(output["rows"], key=lambda row: int(row["uid"]))
    integer("#       transformation_row_count: ", len(output_rows))
    lines.append("#       transformations:")
    field_by_name = {item["field"]["name"]: item["field"] for item in fields}
    for row in output_rows:
        integer("#         - uid: ", row["uid"])
        changes = sorted(row["transformations"], key=lambda item: item["field_name"])
        integer("#           transformation_count: ", len(changes))
        lines.append("#           fields:")
        for change in changes:
            datatype = field_by_name[change["field_name"]]["datatype"]
            quoted("#             - field_name: ", change["field_name"])
            quoted("#               operation: ", change["operation"])
            boolean("#               before_is_null: ", change["before"] is None)
            quoted("#               before: ", value_token(change["before"], datatype))
            boolean("#               after_is_null: ", change["after"] is None)
            quoted("#               after: ", value_token(change["after"], datatype))
            quoted("#               value_source: ", change["value_source"])
            boolean("#               has_source_pair_key: ", change["source_pair_key"] is not None)
            if change["source_pair_key"] is not None:
                integer("#               source_pair_key: ", change["source_pair_key"])
            boolean("#               has_source_row: ", change["source_row"] is not None)
            if change["source_row"] is not None:
                lines.append("#               source_row:")
                emit_row_reference("#                 ", change["source_row"])
            quoted("#               authority_reference: ", change["authority_reference"])
            quoted("#               provenance_reference: ", change["provenance_reference"])

    target_fields = sorted(target["registered_fields"], key=lambda item: item["name"])
    lines.append("#     embedded_target:")
    for key in ("schema_version", "contract_authority", "observation_value_issuer"):
        quoted(f"#       {'schema' if key == 'schema_version' else key}: ", target[key])
    lines.append("#       envelope:")
    emit_envelope("#         ", target["envelope"])
    lines.append("#       observation:")
    for key in ("observation", "subobservation", "scan"):
        integer(f"#         {key}: ", target["observation"][key])
    integer("#       registered_field_count: ", len(target_fields))
    lines.append("#       registered_fields:")
    for field in target_fields:
        emit_typed_field("#         ", "#           ", field)
    target_inputs = sorted(target["inputs"], key=lambda item: int(item["input_key"]))
    integer("#       input_count: ", len(target_inputs))
    lines.append("#       inputs:")
    for item in target_inputs:
        integer("#         - input_key: ", item["input_key"])
        integer("#           network: ", item["network"])
        quoted("#           interface: ", item["interface"])
        integer("#           channel_count: ", item["channel_count"])
        emit_source("#           ", "#             ", "raw_source", item["raw_source"])
        emit_source("#           ", "#             ", "kmp_source", item["kmp_source"])
    target_rows = sorted(target["rows"], key=lambda row: int(row["row_key"]))
    integer("#       row_count: ", len(target_rows))
    lines.append("#       rows:")
    for row in target_rows:
        for label, key in (
            ("row_key", "row_key"), ("input_key", "input_key"),
            ("kmp_source_key", "kmp_source_key"), ("kmp_row_index", "kmp_row_index"),
        ):
            integer(("#         - " if label == "row_key" else "#           ") + label + ": ", row[key])
        quoted("#           matching_frequency_hz: ", row["matching_frequency_hz"])
        quoted("#           output_tone_frequency_hz: ", row["output_tone_frequency_hz"])
        for key in ("array", "network", "channel"):
            integer(f"#           {key}: ", row[key])
        lines.append("#           fields:")
        for field in target_fields:
            quoted(f"#             {field['name']}: ", value_token(row["fields"][field["name"]], field["datatype"]))
    for sequence_name in ("target_source_sequence", "target_application_sequence"):
        integer(f"#       {sequence_name}_count: ", len(target[sequence_name]))
        lines.append(f"#       {sequence_name}:")
        for key in target[sequence_name]:
            integer("#         - ", key)

    lines.append("#     embedded_relation:")
    for key in ("schema_version", "contract_authority", "observation_value_issuer", "mapping_domain"):
        quoted(f"#       {'schema' if key == 'schema_version' else key}: ", relation[key])
    lines.append("#       envelope:")
    emit_envelope("#         ", relation["envelope"])
    lines.append("#       baseline_parent:")
    emit_baseline("#         ", relation["baseline_parent"])
    lines.append("#       target_parent:")
    emit_identity("#         ", relation["target_parent"])
    lines.append("#       matcher:")
    for key in ("matcher_run_occurrence", "implementation_revision", "configuration_reference", "method", "backend", "target_frequency_field", "target_quality_factor_field"):
        quoted(f"#         {key}: ", relation["matcher"][key])
    evidence = sorted(relation["network_evidence"], key=lambda item: int(item["network"]))
    integer("#       network_evidence_count: ", len(evidence))
    lines.append("#       network_evidence:")
    for item in evidence:
        integer("#         - network: ", item["network"])
        for key in ("frequency_shift_hz", "gate_hz", "quality_factor"):
            quoted(f"#           {key}: ", item[key])
        quoted("#           quality_factor_field: ", item["quality_factor_field"])
        quoted("#           quality_factor_authority_reference: ", item["quality_factor_authority_reference"])
    relation_pairs = sorted(relation["pairs"], key=lambda item: int(item["pair_key"]))
    integer("#       pair_count: ", len(relation_pairs))
    lines.append("#       pairs:")
    for pair in relation_pairs:
        integer("#         - pair_key: ", pair["pair_key"])
        lines.append("#           target:")
        emit_row_reference("#             ", pair["target"])
        lines.append("#           seed:")
        emit_row_reference("#             ", pair["seed"])
        quoted("#           separation_hz: ", pair["separation_hz"])
        boolean("#           is_good_match: ", pair["is_good_match"])
    for collection in ("target_dispositions", "seed_dispositions"):
        dispositions = sorted(relation[collection], key=lambda item: int(item["disposition_key"]))
        integer(f"#       {collection}_count: ", len(dispositions))
        lines.append(f"#       {collection}:")
        for item in dispositions:
            integer("#         - disposition_key: ", item["disposition_key"])
            lines.append("#           endpoint:")
            emit_row_reference("#             ", item["endpoint"])
            quoted("#           state: ", item["state"])
            quoted("#           pair_keys: ", "[" + ",".join(item["pair_keys"]) + "]")
            quoted("#           reason: ", item["reason"])
    integer("#       seed_source_sequence_count: ", len(relation["seed_source_sequence"]))
    lines.append("#       seed_source_sequence:")
    for key in relation["seed_source_sequence"]:
        integer("#         - ", key)
    lines.extend([
        '#     null_cell: "unquoted-empty-v1"',
        '#     string_cell: "quoted-utf8-single-line-v1"',
        '#     pair_key_set_cell: "quoted-bracketed-int64-set-v1"',
        '#     metadata_float64: "quoted-ieee754-bits-v1"',
        '# delimiter: ","', '# schema: "astropy-2.0"',
        ",".join(name for name, _, _, _ in columns),
    ])

    def float_from_token(token: str) -> float:
        if token == "nan":
            return float("nan")
        if token == "+inf":
            return float("inf")
        if token == "-inf":
            return -float("inf")
        return struct.unpack(">d", bytes.fromhex(token))[0]

    def csv_value(value: Any, datatype: str) -> str:
        if value is None:
            return ""
        if datatype == "float64":
            return _format_float64(float_from_token(str(value)))
        if datatype == "int64":
            return str(value)
        if datatype == "bool":
            return "True" if value else "False"
        return _csv_quote(str(value))

    for row in output_rows:
        cells = [
            str(row["uid"]), str(row["target"]["local_key"]),
            str(row["target_input_key"]),
            _format_float64(float_from_token(row["tone_frequency_hz"])),
            str(row["array"]), str(row["network"]), str(row["channel"]),
            _csv_quote("[" + ",".join(row["relation_pair_keys"]) + "]"),
        ]
        cells.extend(
            csv_value(row["fields"][item["field"]["name"]], item["field"]["datatype"])
            for item in fields
        )
        lines.append(",".join(cells))
    artifact_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    byte_sha256 = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
    receipt = (
        "citlali-canonical-apt-publication-receipt-v1\n"
        f"scope={contract['byte_transport_scope']}\n"
        f"envelope_sha256={output_digests['envelope_sha256']}\n"
        f"byte_sha256={byte_sha256}\n"
        f"byte_count={len(artifact_bytes)}\n"
    ).encode("ascii")
    return {
        "bytes": artifact_bytes,
        "semantic_sha256": output_digests["semantic_sha256"],
        "envelope_sha256": output_digests["envelope_sha256"],
        "byte_transport_scope": contract["byte_transport_scope"],
        "byte_sha256": byte_sha256,
        "byte_count": len(artifact_bytes),
        "receipt_bytes": receipt,
        "receipt_sha256": "sha256:" + hashlib.sha256(receipt).hexdigest(),
        "receipt_byte_count": len(receipt),
    }


def validate_observation_matched_apt_ecsv_bytes_v1(
    artifact_bytes: bytes,
    receipt_bytes: bytes,
    document: Any,
    contract: dict[str, Any],
    baseline_descriptor: dict[str, Any],
    target_document: Any,
    target_contract: dict[str, Any],
    relation_document: Any,
    relation_contract: dict[str, Any],
) -> dict[str, Any]:
    """Strict independent reserialization and ordinary ECSV readability gate."""
    expected = serialize_observation_matched_apt_ecsv_v1(
        document, contract, baseline_descriptor, target_document,
        target_contract, relation_document, relation_contract
    )
    if artifact_bytes != expected["bytes"]:
        raise ContractError(
            "matched observation ECSV is tampered, stale, reordered, or noncanonical"
        )
    if receipt_bytes != expected["receipt_bytes"]:
        raise ContractError(
            "matched observation ECSV receipt is tampered or binds foreign bytes"
        )
    if Table is None:
        raise ContractError("Astropy is required for matched observation ECSV parity")
    try:
        table = Table.read(artifact_bytes.decode("utf-8").splitlines(), format="ascii.ecsv")
    except Exception as error:
        raise ContractError(f"matched observation ECSV is not Astropy-readable: {error}") from error
    if len(table) != len(document["rows"]):
        raise ContractError("matched observation ECSV row count mismatch")
    root = table.meta.get("canonical_apt_observation_v1")
    if not isinstance(root, dict) or not isinstance(root.get("embedded_target"), dict) or not isinstance(root.get("embedded_relation"), dict):
        raise ContractError("matched observation ECSV lacks complete embedded logical records")
    return expected


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
            if args.artifact_contract in OBSERVATION_ARTIFACT_CONTRACT_IDS:
                raise ContractError(
                    "APT-PROD-002 remains unactivated in product profiles; "
                    "target/relation are embedded logical records and matched "
                    "APT issuance is available only through the versioned "
                    "Citlali machine protocol"
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
