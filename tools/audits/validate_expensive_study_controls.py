#!/usr/bin/env python3
"""Validate costly numerical-study control records.

Schema validation answers whether the three records are well formed.  The
launch gate additionally checks their cross-record semantics, complete
model-free guard coverage, approvals, and on-disk SHA-256 bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker


REGISTER_SCHEMA = "tolerance-and-stop-condition-register-v1.schema.json"
PREFLIGHT_SCHEMA = "expensive-study-preflight-report-v1.schema.json"
CERTIFICATE_SCHEMA = "expensive-execution-readiness-certificate-v1.schema.json"

PATH_COVERAGE_DIMENSIONS = {
    "boundary_neighbors",
    "constant_paths",
    "decimal_binary_conversions",
    "coordinate_transformations",
    "candidate_dispatch",
    "output_formats",
}
NONWARNING_ACTIONS = {"hard_stop", "invalid_evidence", "scientific_failure"}
NUMERICAL_DERIVATIONS = {
    "analytic_bound",
    "conditioning_bound",
    "interval_bound",
    "ulp_bound",
    "propagated_impact",
}
COMPATIBLE_OPERATORS = {
    "exact": {"equals", "not_equals", "diagnostic_compare"},
    "absolute": {
        "less_than",
        "less_than_or_equal",
        "greater_than",
        "greater_than_or_equal",
        "equals",
    },
    "relative": {
        "less_than",
        "less_than_or_equal",
        "greater_than",
        "greater_than_or_equal",
        "equals",
    },
    "ulp": {
        "less_than",
        "less_than_or_equal",
        "greater_than",
        "greater_than_or_equal",
        "equals",
    },
    "interval": {"inside", "inside_inclusive", "outside", "outside_inclusive"},
    "set_identity": {"equals", "subset", "superset"},
    "schema": {"validates", "does_not_validate"},
    "cardinality": {
        "equals",
        "less_than",
        "less_than_or_equal",
        "greater_than",
        "greater_than_or_equal",
    },
}


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses silent duplicate-key replacement."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_document(path: Path) -> dict[str, Any]:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{path}: top-level document must be a mapping")
    return document


def _load_schema(name: str) -> dict[str, Any]:
    path = _repo_root() / "doc" / "audits" / "schemas" / name
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load schema {path}: {exc}") from exc
    Draft202012Validator.check_schema(schema)
    return schema


def _json_pointer(parts: Iterable[Any]) -> str:
    encoded = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(encoded) if encoded else "/"


def validate_schemas(
    register: Mapping[str, Any],
    preflight: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> list[str]:
    """Return all Draft 2020-12 validation errors."""

    documents = (
        ("register", register, REGISTER_SCHEMA),
        ("preflight", preflight, PREFLIGHT_SCHEMA),
        ("certificate", certificate, CERTIFICATE_SCHEMA),
    )
    errors: list[str] = []
    for label, document, schema_name in documents:
        validator = Draft202012Validator(
            _load_schema(schema_name), format_checker=FormatChecker()
        )
        for error in sorted(validator.iter_errors(document), key=lambda item: list(item.path)):
            errors.append(f"{label}{_json_pointer(error.path)}: {error.message}")
    return errors


def _artifact_identity(artifact: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        artifact.get("path"),
        artifact.get("sha256"),
        artifact.get("commit"),
        artifact.get("source_identity"),
    )


def _same_artifact(
    errors: list[str], label: str, actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if _artifact_identity(actual) != _artifact_identity(expected):
        errors.append(
            f"{label}: artifact path/digest/commit/source identity does not match the frozen binding"
        )


def _resolve_artifact_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_artifacts(value: Any, pointer: tuple[Any, ...] = ()):
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield pointer, value
        for key, child in value.items():
            yield from _iter_artifacts(child, (*pointer, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_artifacts(child, (*pointer, index))


def _verify_all_artifact_digests(
    errors: list[str], label: str, document: Mapping[str, Any], root: Path
) -> None:
    checked: set[tuple[str, str]] = set()
    for pointer, artifact in _iter_artifacts(document):
        raw_path = artifact["path"]
        expected = artifact["sha256"]
        key = (raw_path, expected)
        if key in checked:
            continue
        checked.add(key)
        path = _resolve_artifact_path(root, raw_path)
        where = f"{label}{_json_pointer(pointer)}"
        if expected == "0" * 64:
            errors.append(f"{where}: all-zero SHA-256 placeholder is forbidden at launch")
            continue
        if not path.is_file():
            errors.append(f"{where}: bound artifact does not exist: {path}")
            continue
        actual = _sha256(path)
        if actual != expected:
            errors.append(
                f"{where}: SHA-256 mismatch for {path}; expected {expected}, got {actual}"
            )


def _verify_source_site_digests(
    errors: list[str], register: Mapping[str, Any], root: Path
) -> None:
    for condition in register["conditions"]:
        for index, site in enumerate(condition["source_sites"]):
            path = _resolve_artifact_path(root, site["path"])
            label = f"register condition {condition['condition_id']} source site {index}"
            if site["source_sha256"] == "0" * 64:
                errors.append(f"{label}: all-zero source SHA-256 placeholder is forbidden")
                continue
            if not path.is_file():
                errors.append(f"{label}: source file does not exist: {path}")
                continue
            actual = _sha256(path)
            if actual != site["source_sha256"]:
                errors.append(
                    f"{label}: source SHA-256 mismatch; expected {site['source_sha256']}, "
                    f"got {actual}"
                )


def _reject_placeholders(
    errors: list[str], label: str, value: Any, pointer: tuple[Any, ...] = ()
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_placeholders(errors, label, child, (*pointer, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(errors, label, child, (*pointer, index))
    elif isinstance(value, str) and (
        "TEMPLATE" in value.upper() or value.startswith("path/to/")
    ):
        errors.append(f"{label}{_json_pointer(pointer)}: unresolved placeholder {value!r}")


def _verify_bound_input(
    errors: list[str],
    label: str,
    artifact: Mapping[str, Any],
    supplied_path: Path,
    root: Path,
) -> None:
    bound = _resolve_artifact_path(root, artifact["path"])
    if bound != supplied_path.resolve():
        errors.append(f"{label}: binds {bound}, not supplied input {supplied_path.resolve()}")


def _require_approved(errors: list[str], condition: Mapping[str, Any]) -> None:
    approval = condition["approval"]
    condition_id = condition["condition_id"]
    if not approval["required"] or approval["status"] != "approved" or not approval["record"]:
        errors.append(
            f"register condition {condition_id}: non-warning authority requires an "
            "explicit approved record"
        )


def _comparison_fingerprint(comparison: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in comparison.items() if key != "comparison_fingerprint"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_decimal(
    errors: list[str], label: str, canonical_number: Mapping[str, Any]
) -> Decimal | None:
    literal = canonical_number["canonical_literal"]
    try:
        value = Decimal(literal)
    except InvalidOperation:
        errors.append(f"{label}: invalid canonical numeric literal {literal!r}")
        return None
    if not value.is_finite():
        errors.append(f"{label}: numeric literal must be finite")
        return None
    if value < 0:
        errors.append(f"{label}: numeric bound must be nonnegative")
    if "binary64" in canonical_number["representation"].lower():
        try:
            represented = float(literal)
        except ValueError:
            represented = math.nan
        if not math.isfinite(represented):
            errors.append(f"{label}: literal is non-finite in the declared binary64 representation")
    return value


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_approval_coherence(
    errors: list[str], condition: Mapping[str, Any]
) -> None:
    approval = condition["approval"]
    condition_id = condition["condition_id"]
    action = condition["action"]
    condition_class = condition["condition_class"]
    if action == "warning":
        if (
            approval["required"]
            or approval["role"] != "not_required"
            or approval["status"] != "not_required"
            or approval["record"] is not None
        ):
            errors.append(
                f"register condition {condition_id}: warning approval must be not_required with "
                "role not_required and a null record"
            )
        return

    if (
        not approval["required"]
        or approval["status"] != "approved"
        or not approval["record"]
    ):
        errors.append(
            f"register condition {condition_id}: required approval must be approved and "
            "bind a nonempty record before launch"
        )
    allowed_roles = {
        "A_exact_identity_integrity": {"audit_manager", "integrity_owner"},
        "B_derived_numerical_correctness": {
            "audit_manager",
            "numerical_methods_owner",
        },
        "C_scientific_acceptance": {"scientific_owner"},
        "D_engineering_diagnostic": set(),
    }[condition_class]
    if approval["role"] not in allowed_roles:
        expected = " or ".join(sorted(allowed_roles)) or "no non-warning role"
        errors.append(
            f"register condition {condition_id}: {condition_class} non-warning approval "
            f"requires approved role {expected}"
        )


def _validate_register_semantics(register: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if register["status"] != "frozen":
        errors.append("register/status: launch requires status 'frozen'")

    conditions = register["conditions"]
    condition_ids = [condition["condition_id"] for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        errors.append("register/conditions: condition_id values must be unique")

    metric_ids = [metric["metric_id"] for metric in register["final_scientific_metrics"]]
    if len(metric_ids) != len(set(metric_ids)):
        errors.append("register/final_scientific_metrics: metric_id values must be unique")
    metric_id_set = set(metric_ids)

    census = register["guard_site_census"]
    if set(census["condition_ids"]) != set(condition_ids):
        errors.append("register/guard_site_census: condition IDs do not equal registered conditions")
    source_site_count = sum(len(condition["source_sites"]) for condition in conditions)
    if census["site_count"] != source_site_count:
        errors.append(
            "register/guard_site_census/site_count: does not equal the number of "
            f"registered source sites ({source_site_count})"
        )
    legacy_exception = census["legacy_static_exception"]
    if census["harness_kind"] == "new_or_revised":
        if census["routing"] != "condition_id_dispatcher":
            errors.append(
                "register/guard_site_census: a new or revised harness must use the "
                "condition-ID dispatcher"
            )
        if (
            legacy_exception["required"]
            or legacy_exception["status"] != "not_required"
            or legacy_exception["manager_record"] is not None
        ):
            errors.append(
                "register/guard_site_census: new/revised harness cannot claim a legacy exception"
            )
    else:
        if census["routing"] != "condition_id_dispatcher":
            if (
                not legacy_exception["required"]
                or legacy_exception["status"] != "approved"
                or not legacy_exception["manager_record"]
            ):
                errors.append(
                    "register/guard_site_census: legacy static/hybrid routing requires an "
                    "approved manager exception record"
                )
        elif legacy_exception["required"]:
            errors.append(
                "register/guard_site_census: dispatcher-routed legacy harness does not require "
                "a static-census exception"
            )

    study_cost = register["study"]["cost_classification"]["estimate"]["value"]
    if not math.isfinite(study_cost) or study_cost <= 0:
        errors.append(
            "register/study/cost_classification/estimate: a costly study requires a finite, "
            "positive cost estimate"
        )

    for condition in conditions:
        condition_id = condition["condition_id"]
        condition_class = condition["condition_class"]
        action = condition["action"]
        derivation = condition["threshold_derivation"]["kind"]
        effect = condition["maximum_propagated_effect"]
        comparison = condition["comparison"]

        _validate_approval_coherence(errors, condition)
        implemented_actions = {site["implemented_action"] for site in condition["source_sites"]}
        if implemented_actions != {action}:
            errors.append(
                f"register condition {condition_id}: source-site implemented action(s) "
                f"{sorted(implemented_actions)} do not equal registered action {action}"
            )
        if census["harness_kind"] == "new_or_revised" and any(
            not site["embedded_condition_id"] or site["route_kind"] != "condition_id_dispatcher"
            for site in condition["source_sites"]
        ):
            errors.append(
                f"register condition {condition_id}: new/revised harness source sites must route "
                "through embedded condition IDs"
            )

        if derivation == "none_diagnostic" and action != "warning":
            errors.append(
                f"register condition {condition_id}: an unquantified diagnostic may only warn"
            )
        if action in NONWARNING_ACTIONS:
            _require_approved(errors, condition)

        actual_fingerprint = _comparison_fingerprint(comparison)
        if comparison["comparison_fingerprint"] != actual_fingerprint:
            errors.append(
                f"register condition {condition_id}: comparison fingerprint mismatch; "
                f"expected {actual_fingerprint}"
            )
        semantics = comparison["semantics"]
        operator = comparison["operator"]
        if operator not in COMPATIBLE_OPERATORS[semantics]:
            errors.append(
                f"register condition {condition_id}: operator {operator!r} is incompatible with "
                f"{semantics} comparison semantics"
            )

        if semantics in {"absolute", "relative", "ulp", "interval", "cardinality"}:
            if comparison["threshold"] is None:
                errors.append(
                    f"register condition {condition_id}: numerical comparison lacks a threshold"
                )
        if semantics in {"exact", "set_identity", "schema"} and comparison["threshold"] is not None:
            errors.append(
                f"register condition {condition_id}: {semantics} comparison must not carry a "
                "numeric threshold"
            )
        if operator == "diagnostic_compare" and not (
            condition_class == "D_engineering_diagnostic" and action == "warning"
        ):
            errors.append(
                f"register condition {condition_id}: diagnostic_compare is permitted only for a "
                "Class D warning"
            )
        threshold_value: Decimal | None = None
        if comparison["threshold"] is not None:
            threshold_value = _canonical_decimal(
                errors, f"register condition {condition_id} threshold", comparison["threshold"]
            )
        if semantics == "ulp" and threshold_value is not None:
            if threshold_value != threshold_value.to_integral_value():
                errors.append(
                    f"register condition {condition_id}: ULP threshold must be a nonnegative integer"
                )
            if comparison["threshold"]["units"].lower() != "ulp":
                errors.append(
                    f"register condition {condition_id}: ULP threshold units must be ULP"
                )
        if semantics == "cardinality" and threshold_value is not None:
            if threshold_value != threshold_value.to_integral_value():
                errors.append(
                    f"register condition {condition_id}: cardinality threshold must be an integer"
                )

        affected_metrics = effect["affected_metrics"]
        affected_ids = [item["metric_id"] for item in affected_metrics]
        if len(affected_ids) != len(set(affected_ids)):
            errors.append(
                f"register condition {condition_id}: affected metric IDs must be unique"
            )
        for item in affected_metrics:
            if item["metric_id"] not in metric_id_set:
                errors.append(
                    f"register condition {condition_id}: unknown affected metric {item['metric_id']}"
                )
            _canonical_decimal(
                errors,
                f"register condition {condition_id} effect on {item['metric_id']}",
                item["bound"],
            )

        if effect["status"] == "quantified":
            if (
                not affected_metrics
                or effect["exact_integrity_rationale"] is not None
                or effect["derived_correctness_rationale"] is not None
            ):
                errors.append(
                    f"register condition {condition_id}: quantified effect must enumerate every "
                    "affected metric and must not use a non-metric rationale"
                )
        elif effect["status"] == "not_applicable_exact_integrity":
            if (
                condition_class != "A_exact_identity_integrity"
                or affected_metrics
                or not effect["exact_integrity_rationale"]
                or effect["derived_correctness_rationale"] is not None
            ):
                errors.append(
                    f"register condition {condition_id}: exact-integrity effect rationale is "
                    "valid only for Class A with no numeric metric effects"
                )
        elif effect["status"] == "not_applicable_derived_correctness":
            if (
                condition_class != "B_derived_numerical_correctness"
                or affected_metrics
                or effect["exact_integrity_rationale"] is not None
                or not effect["derived_correctness_rationale"]
            ):
                errors.append(
                    f"register condition {condition_id}: derived-correctness rationale is valid "
                    "only for Class B with a proved numerical bound and no claimed metric mapping"
                )
        else:
            if (
                affected_metrics
                or effect["exact_integrity_rationale"] is not None
                or effect["derived_correctness_rationale"] is not None
            ):
                errors.append(
                    f"register condition {condition_id}: unknown diagnostic effect may not "
                    "claim metric bounds or non-metric correctness rationales"
                )
            if action != "warning":
                errors.append(
                    f"register condition {condition_id}: non-warning action lacks a quantified "
                    "or exact-integrity effect disposition"
                )

        if condition_class == "A_exact_identity_integrity":
            if semantics not in {"exact", "interval", "set_identity", "schema", "cardinality"}:
                errors.append(
                    f"register condition {condition_id}: Class A uses incompatible {semantics} semantics"
                )
            if action not in {"hard_stop", "invalid_evidence", "warning"}:
                errors.append(
                    f"register condition {condition_id}: Class A cannot yield a scientific failure"
                )
            if derivation != "exact_identity":
                errors.append(
                    f"register condition {condition_id}: Class A requires exact_identity derivation"
                )

        if condition_class == "B_derived_numerical_correctness":
            if derivation not in NUMERICAL_DERIVATIONS:
                errors.append(
                    f"register condition {condition_id}: Class B lacks an analytic, conditioning, "
                    "interval, ULP, or propagated-impact derivation"
                )
            if effect["status"] not in {
                "quantified",
                "not_applicable_derived_correctness",
            }:
                errors.append(
                    f"register condition {condition_id}: Class B requires either a quantified "
                    "metric mapping or a non-metric derived-correctness rationale"
                )
            if action != "warning" and action not in {"hard_stop", "invalid_evidence"}:
                errors.append(
                    f"register condition {condition_id}: Class B may not masquerade as a "
                    "scientific acceptance result"
                )

        if condition_class == "C_scientific_acceptance":
            if action != "scientific_failure":
                errors.append(
                    f"register condition {condition_id}: Class C must yield scientific_failure, "
                    "not an execution hard stop or evidence invalidation"
                )
            if derivation != "scientific_contract":
                errors.append(
                    f"register condition {condition_id}: Class C threshold must derive from "
                    "the approved scientific contract"
                )

        if condition_class == "D_engineering_diagnostic" and action != "warning":
            errors.append(
                f"register condition {condition_id}: Class D conditions are warning-only; "
                "reclassify proved consequences as Class A, B, or C"
            )

        failure_scope = condition["failure_scope"]
        if action == "warning" and failure_scope["already_written_raw_output"] != "unaffected":
            errors.append(
                f"register condition {condition_id}: warning may not invalidate or place a hold "
                "on already-written raw output"
            )
        if action == "scientific_failure" and (
            failure_scope["validity_layer"] != "scientific_decision"
            or failure_scope["already_written_raw_output"] != "unaffected"
        ):
            errors.append(
                f"register condition {condition_id}: scientific failure must be scoped to the "
                "scientific decision and preserve raw output"
            )
        if action == "invalid_evidence" and failure_scope["validity_layer"] == "scientific_decision":
            errors.append(
                f"register condition {condition_id}: invalid_evidence must name the corrupt "
                "raw, parser/admission, or evaluator layer"
            )

        cost = condition["estimated_cost_exposed_if_fires"]["value"]
        if not math.isfinite(cost) or cost < 0:
            errors.append(
                f"register condition {condition_id}: exposed cost must be finite and nonnegative"
            )

        dependency = condition["data_dependency"]
        if condition["data_dependent"]:
            if (
                not dependency["model_free_inputs_unavailable"]
                or not dependency["independent_review_acknowledged"]
                or dependency["synthetic_or_fault_injection_test_id"] is None
            ):
                errors.append(
                    f"register condition {condition_id}: data-dependent guard requires an input-"
                    "availability basis, independent-review acknowledgement, and synthetic/fault-"
                    "injection test ID"
                )
        elif (
            dependency["model_free_inputs_unavailable"]
            or dependency["synthetic_or_fault_injection_test_id"] is not None
        ):
            errors.append(
                f"register condition {condition_id}: deterministic classification conflicts with "
                "its data-dependency record"
            )

        if not condition["data_dependent"]:
            preflight = condition["preflight"]
            if not preflight["all_frozen_tuples_required"]:
                errors.append(
                    f"register condition {condition_id}: deterministic guard must require all "
                    "frozen tuples in preflight"
                )
            if not preflight["branch_coverage_required"]:
                errors.append(
                    f"register condition {condition_id}: deterministic guard must require branch "
                    "coverage in preflight"
                )
            if condition["earliest_stage"] not in {
                "protocol_construction",
                "model_free_preflight",
                "pre_model_execution",
            }:
                errors.append(
                    f"register condition {condition_id}: deterministic guard first appears after "
                    "the model-free/pre-model stages"
                )

    separation = register["study"]["raw_evaluator_separation"]
    if not separation["raw_model_stage_separable"]:
        errors.append("register/study/raw_evaluator_separation: raw model stage must be separable")
    states = {
        separation["raw_validity_state_artifact"],
        separation["parser_admission_validity_state_artifact"],
        separation["evaluator_validity_state_artifact"],
        separation["scientific_decision_state_artifact"],
    }
    if len(states) != 4:
        errors.append(
            "register/study/raw_evaluator_separation: raw, parser/admission, evaluator, and "
            "scientific decision state artifacts must be distinct"
        )
    return errors


def _validate_preflight_semantics(
    register: Mapping[str, Any], preflight: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    study = register["study"]
    if preflight["status"] != "complete":
        errors.append("preflight/status: launch requires status 'complete'")
    if preflight["study_id"] != study["study_id"]:
        errors.append("preflight/study_id: does not match register")

    bindings = preflight["bindings"]
    _same_artifact(errors, "preflight/bindings/protocol", bindings["protocol"], study["protocol"])
    _same_artifact(errors, "preflight/bindings/runner", bindings["runner"], study["runner"])
    _same_artifact(errors, "preflight/bindings/evaluator", bindings["evaluator"], study["evaluator"])
    case_binding = bindings["frozen_case_set"]
    frozen_cases = study["frozen_case_set"]
    if (
        case_binding["case_set_id"] != frozen_cases["case_set_id"]
        or case_binding["tuple_count"] != frozen_cases["tuple_count"]
    ):
        errors.append("preflight/bindings/frozen_case_set: identity or tuple count mismatch")
    _same_artifact(
        errors,
        "preflight/bindings/frozen_case_set/artifact",
        case_binding["artifact"],
        frozen_cases["artifact"],
    )

    if preflight["execution"]["scientific_model_calls"] != 0:
        errors.append("preflight/execution: dry-run made scientific model calls")

    condition_by_id = {item["condition_id"]: item for item in register["conditions"]}
    condition_ids = set(condition_by_id)
    abort_ids = {
        condition_id
        for condition_id, condition in condition_by_id.items()
        if condition["action"] != "warning"
    }
    inventory = preflight["guard_inventory"]
    if set(inventory["registered_condition_ids"]) != condition_ids:
        errors.append("preflight/guard_inventory: registered IDs do not equal frozen register")
    if set(inventory["discovered_condition_ids"]) != condition_ids:
        errors.append(
            "preflight/guard_inventory: discovered IDs do not account for every registered guard"
        )
    if set(inventory["abort_capable_condition_ids"]) != abort_ids:
        errors.append(
            "preflight/guard_inventory: abort-capable IDs do not equal registered non-warning guards"
        )
    if inventory["unregistered_abort_capable_ids"]:
        errors.append(
            "preflight/guard_inventory: unregistered abort-capable guards present: "
            + ", ".join(inventory["unregistered_abort_capable_ids"])
        )
    if inventory["unknown_condition_ids"]:
        errors.append(
            "preflight/guard_inventory: unknown condition IDs present: "
            + ", ".join(inventory["unknown_condition_ids"])
        )
    discovered = inventory["discovered_conditions"]
    discovered_ids = [item["condition_id"] for item in discovered]
    if len(discovered_ids) != len(set(discovered_ids)):
        errors.append("preflight/guard_inventory/discovered_conditions: duplicate condition IDs")
    if set(discovered_ids) != condition_ids:
        errors.append(
            "preflight/guard_inventory/discovered_conditions: source-action census does not "
            "equal frozen register"
        )
    for item in discovered:
        condition = condition_by_id.get(item["condition_id"])
        if condition is None:
            continue
        if item["source_site_count"] != len(condition["source_sites"]):
            errors.append(
                f"preflight condition {item['condition_id']}: discovered source-site count mismatch"
            )
        if set(item["implemented_actions"]) != {condition["action"]}:
            errors.append(
                f"preflight condition {item['condition_id']}: discovered implemented action(s) "
                f"{item['implemented_actions']} do not equal registered action {condition['action']}"
            )
    census = register["guard_site_census"]
    if inventory["site_count"] != census["site_count"]:
        errors.append("preflight/guard_inventory/site_count: does not match frozen census")
    _same_artifact(
        errors,
        "preflight/guard_inventory/census_artifact",
        inventory["census_artifact"],
        census["artifact"],
    )
    if inventory["method"] != census["routing"]:
        errors.append(
            "preflight/guard_inventory/method: does not match the frozen census routing"
        )
    if census["harness_kind"] == "new_or_revised" and inventory["method"] != "condition_id_dispatcher":
        errors.append(
            "preflight/guard_inventory/method: new/revised harness must use condition-ID dispatcher"
        )
    if inventory["method"] == "condition_id_dispatcher" and any(
        not site["embedded_condition_id"]
        for condition in register["conditions"]
        for site in condition["source_sites"]
    ):
        errors.append(
            "preflight/guard_inventory/method: dispatcher claimed but at least one source site "
            "lacks an embedded condition ID"
        )

    coverage = preflight["coverage"]
    tuple_count = frozen_cases["tuple_count"]
    if coverage["enumerated_tuple_count"] != tuple_count:
        errors.append("preflight/coverage: enumerated tuple count does not match frozen case set")
    if coverage["exercised_tuple_count"] != tuple_count:
        errors.append("preflight/coverage: exercised tuple count is incomplete")
    if not coverage["all_frozen_tuples_complete"]:
        errors.append("preflight/coverage: frozen-tuple coverage is not complete")

    condition_coverage = coverage["condition_coverage"]
    covered_ids = [item["condition_id"] for item in condition_coverage]
    if len(covered_ids) != len(set(covered_ids)):
        errors.append("preflight/coverage/condition_coverage: duplicate condition IDs")
    if set(covered_ids) != condition_ids:
        errors.append("preflight/coverage/condition_coverage: IDs do not equal frozen register")
    for item in condition_coverage:
        condition = condition_by_id.get(item["condition_id"])
        if condition is None:
            continue
        if item["data_dependent"] != condition["data_dependent"]:
            errors.append(
                f"preflight condition {item['condition_id']}: data-dependent classification mismatch"
            )
        required_branches = set(condition["preflight"]["required_branches"])
        if set(item["expected_branches"]) != required_branches:
            errors.append(
                f"preflight condition {item['condition_id']}: expected branches do not equal "
                "the frozen register"
            )
        if not required_branches.issubset(item["observed_branches"]):
            errors.append(
                f"preflight condition {item['condition_id']}: required branch coverage is incomplete"
            )

        observation = item["comparison_observation"]
        comparison = condition["comparison"]
        if observation["comparison_fingerprint"] != comparison["comparison_fingerprint"]:
            errors.append(
                f"preflight condition {item['condition_id']}: comparison fingerprint does not "
                "match frozen register"
            )
        if observation["observed_threshold"] != comparison["threshold"]:
            errors.append(
                f"preflight condition {item['condition_id']}: observed threshold literal, "
                "representation, or units differ from frozen register"
            )

        injected = item["synthetic_or_fault_injection"]
        if condition["data_dependent"]:
            expected_test = condition["data_dependency"][
                "synthetic_or_fault_injection_test_id"
            ]
            if (
                injected is None
                or injected["test_id"] != expected_test
                or not injected["passed"]
                or injected["observed_exercises"] < 1
                or not item["complete"]
            ):
                errors.append(
                    f"preflight condition {item['condition_id']}: data-dependent guard lacks a "
                    "passing, registered synthetic/fault-injection coverage record"
                )
        elif injected is not None:
            errors.append(
                f"preflight condition {item['condition_id']}: deterministic guard has an "
                "unexpected data-dependent fault-injection escape record"
            )
        if not condition["data_dependent"]:
            if item["expected_exercises"] < tuple_count:
                errors.append(
                    f"preflight condition {item['condition_id']}: expected exercise count does "
                    "not cover every frozen tuple"
                )
            if item["observed_exercises"] < item["expected_exercises"]:
                errors.append(
                    f"preflight condition {item['condition_id']}: deterministic exercise count "
                    "is incomplete"
                )
            if not item["all_required_tuples_exercised"] or not item["complete"]:
                errors.append(
                    f"preflight condition {item['condition_id']}: deterministic coverage is incomplete"
                )

    path_coverage = coverage["path_coverage"]
    dimensions = [item["dimension"] for item in path_coverage]
    if len(dimensions) != len(set(dimensions)):
        errors.append("preflight/coverage/path_coverage: duplicate dimensions")
    if set(dimensions) != PATH_COVERAGE_DIMENSIONS:
        errors.append("preflight/coverage/path_coverage: required dimensions are missing")
    for item in path_coverage:
        if not item["required"]:
            errors.append(
                f"preflight path {item['dimension']}: dimension must be explicitly covered; use "
                "a documented not_applicable_verified path when absent"
            )
        if not item["complete"] or not set(item["expected_paths"]).issubset(
            item["observed_paths"]
        ):
            errors.append(f"preflight path {item['dimension']}: coverage is incomplete")

    result = preflight["result"]
    if not result["passed"] or result["failure_ids"]:
        errors.append("preflight/result: preflight did not pass without failures")
    return errors


def _validate_certificate_semantics(
    register: Mapping[str, Any],
    preflight: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    study = register["study"]
    if certificate["status"] != "ready":
        errors.append("certificate/status: launch requires status 'ready'")
    if certificate["study_id"] != study["study_id"]:
        errors.append("certificate/study_id: does not match register")

    bindings = certificate["bindings"]
    _same_artifact(
        errors,
        "certificate/bindings/register",
        bindings["register"],
        preflight["bindings"]["register"],
    )
    _same_artifact(errors, "certificate/bindings/protocol", bindings["protocol"], study["protocol"])
    _same_artifact(errors, "certificate/bindings/runner", bindings["runner"], study["runner"])
    _same_artifact(errors, "certificate/bindings/evaluator", bindings["evaluator"], study["evaluator"])
    _same_artifact(
        errors,
        "certificate/bindings/salvage_policy",
        bindings["salvage_policy"],
        study["salvage_policy"],
    )
    case_binding = bindings["frozen_case_set"]
    frozen_cases = study["frozen_case_set"]
    if (
        case_binding["case_set_id"] != frozen_cases["case_set_id"]
        or case_binding["tuple_count"] != frozen_cases["tuple_count"]
    ):
        errors.append("certificate/bindings/frozen_case_set: identity or tuple count mismatch")
    _same_artifact(
        errors,
        "certificate/bindings/frozen_case_set/artifact",
        case_binding["artifact"],
        frozen_cases["artifact"],
    )

    review = certificate["independent_review"]
    if review["status"] != "approved" or any(
        review[field] != 0
        for field in (
            "unregistered_hard_guard_count",
            "unsupported_threshold_count",
            "unresolved_finding_count",
        )
    ):
        errors.append("certificate/independent_review: independent review is not cleanly approved")

    false_attestations = [
        key for key, value in certificate["attestations"].items() if value is not True
    ]
    if false_attestations:
        errors.append(
            "certificate/attestations: all readiness attestations must be true; false: "
            + ", ".join(sorted(false_attestations))
        )

    scope = certificate["readiness_scope"]
    if not scope["exact_execution_scope_bound"]:
        errors.append(
            "certificate/readiness_scope: exact execution scope is not bound for readiness"
        )
    if scope["launch_authorization"] is not False:
        errors.append(
            "certificate/readiness_scope: readiness certificate must not claim launch authorization"
        )
    forbidden_scope = [
        key
        for key in ("application_repair", "production_disposition_change", "unity_access", "reaudit")
        if scope[key]
    ]
    if forbidden_scope:
        errors.append(
            "certificate/readiness_scope: readiness certificate may not authorize collateral "
            "actions: " + ", ".join(forbidden_scope)
        )

    approval = certificate["manager_approval"]
    if approval["decision"] != "approved" or approval["approved_at_utc"] is None:
        errors.append("certificate/manager_approval: audit manager has not approved readiness")
    elif approval["manager_role"] != study["manager_role"]:
        errors.append("certificate/manager_approval: manager role does not match frozen register")
    if approval["approved_at_utc"] is not None:
        preflight_time = _parse_timestamp(preflight["result"]["completed_at_utc"])
        review_time = _parse_timestamp(review["completed_at_utc"])
        approval_time = _parse_timestamp(approval["approved_at_utc"])
        if review_time < preflight_time:
            errors.append(
                "certificate/independent_review: completion predates the preflight report"
            )
        if approval_time < review_time:
            errors.append(
                "certificate/manager_approval: approval predates independent review completion"
            )

    expected_data_dependent = {
        condition["condition_id"]
        for condition in register["conditions"]
        if condition["data_dependent"]
    }
    remaining = certificate["remaining_data_dependent_conditions"]
    remaining_ids = [item["condition_id"] for item in remaining]
    if len(remaining_ids) != len(set(remaining_ids)):
        errors.append("certificate/remaining_data_dependent_conditions: duplicate condition IDs")
    if set(remaining_ids) != expected_data_dependent:
        errors.append(
            "certificate/remaining_data_dependent_conditions: list does not equal registered "
            "data-dependent guards"
        )
    for item in remaining:
        if not item["cost_exposure_understood"]:
            errors.append(
                f"certificate condition {item['condition_id']}: cost exposure is not understood"
            )
        if not item["independent_review_acknowledged"]:
            errors.append(
                f"certificate condition {item['condition_id']}: independent review did not "
                "acknowledge the data-dependent exception"
            )
    return errors


def validate_launch_gate(
    register: Mapping[str, Any],
    preflight: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    root: Path,
    register_path: Path,
    preflight_path: Path,
    certificate_path: Path,
) -> list[str]:
    """Return all launch-gate failures for schema-valid documents."""

    errors = _validate_register_semantics(register)
    errors.extend(_validate_preflight_semantics(register, preflight))
    errors.extend(_validate_certificate_semantics(register, preflight, certificate))

    _verify_bound_input(
        errors, "preflight/bindings/register", preflight["bindings"]["register"], register_path, root
    )
    _verify_bound_input(
        errors, "certificate/bindings/register", certificate["bindings"]["register"], register_path, root
    )
    _verify_bound_input(
        errors,
        "certificate/bindings/preflight_report",
        certificate["bindings"]["preflight_report"],
        preflight_path,
        root,
    )

    for label, document in (
        ("register", register),
        ("preflight", preflight),
        ("certificate", certificate),
    ):
        _reject_placeholders(errors, label, document)
        _verify_all_artifact_digests(errors, label, document, root)
    _verify_source_site_digests(errors, register, root)

    # The certificate itself is intentionally not self-digest-bound. This
    # check makes the otherwise-unused argument explicit and catches bad CLI
    # routing before execution.
    if not certificate_path.is_file():
        errors.append(f"certificate input does not exist: {certificate_path}")
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="root for relative artifact paths (default: current directory)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--schema-only", action="store_true")
    mode.add_argument("--launch-gate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        register = _load_document(args.register)
        preflight = _load_document(args.preflight)
        certificate = _load_document(args.certificate)
        errors = validate_schemas(register, preflight, certificate)
        if not errors and args.launch_gate:
            errors = validate_launch_gate(
                register,
                preflight,
                certificate,
                root=args.root.resolve(),
                register_path=args.register.resolve(),
                preflight_path=args.preflight.resolve(),
                certificate_path=args.certificate.resolve(),
            )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        errors = [str(exc)]

    if errors:
        print("EXPENSIVE STUDY CONTROL VALIDATION: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    mode = "LAUNCH GATE" if args.launch_gate else "SCHEMA"
    print(f"EXPENSIVE STUDY CONTROL {mode}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
