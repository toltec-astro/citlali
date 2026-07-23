#!/usr/bin/env python3
"""Validate a Citlali reduction against a versioned product contract."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

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


SCHEMA_VERSION = "citlali-product-contract-registry-v1"
RESULT_SCHEMA_VERSION = "citlali-product-contract-result-v1"
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

    contracts = _list(registry.get("contracts"), f"{path}.contracts")
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
            for name, expected in checks.get("scalar_equals", {}).items():
                if name not in dataset.variables:
                    errors.append(f"missing NetCDF scalar variable {name!r}")
                    continue
                values = dataset.variables[name][...]
                if values.size != 1:
                    errors.append(
                        f"NetCDF variable {name!r} has {values.size} values; "
                        "expected one"
                    )
                    continue
                actual = values.reshape(-1)[0]
                if hasattr(actual, "item"):
                    actual = actual.item()
                if isinstance(actual, bytes):
                    actual = actual.decode("utf-8")
                if actual != expected:
                    errors.append(
                        f"NetCDF scalar {name!r}={actual!r}; "
                        f"expected {expected!r}"
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
                checks = dict(check_definitions.get(entry.get("check_id"), {}))
                checks.update(entry.get("checks", {}))
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
