#!/usr/bin/env python3
"""Resolve and audit the Phase 2 low-level configuration leaf contract."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from tools.config import compare_lowlevel_yaml


RULES_SCHEMA = "citlali-config-leaf-contract-rules-v1"
RESOLVED_SCHEMA = "citlali-config-leaf-contract-v1"
DEFAULT_RULES = "tools/config/config_leaf_contract.yaml"
DEFAULT_AUTHORITY = "tools/config/config_authority_inventory.json"
DEFAULT_CASES = "tools/config/compact_compatibility_cases.yaml"
DEFAULT_MANIFEST = "tools/config/config_leaf_contract_resolved.json"


class ContractError(RuntimeError):
    """Raised when the checked leaf contract is incomplete or inconsistent."""


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_path(value: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_authority_owners(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(domain["id"]): str(domain["typed_owner"])
        for domain in data.get("domains", [])
    }


def load_sources(
    repo_root: Path, cases_path: Path, require_all: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    sources = [
        {
            "label": "default",
            "mode": "",
            "path": repo_root / "data/config.yaml",
        }
    ]
    cases = load_yaml(cases_path)
    seen: set[tuple[str, Path]] = set()
    missing: list[str] = []
    for case in cases.get("cases", []):
        mode = str(case.get("intent", ""))
        raw_path = case.get("base_config")
        if not mode or not isinstance(raw_path, str):
            continue
        path = resolve_path(raw_path, cases_path.parent)
        key = (mode, path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            missing.append(str(path))
            continue
        sources.append({"label": mode, "mode": mode, "path": path})
    if missing and require_all:
        raise ContractError("missing config input(s): " + ", ".join(missing))
    return sources, missing


def observed_leaves(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    leaves: dict[str, dict[str, Any]] = {}
    for source in sources:
        data = compare_lowlevel_yaml.extract_low_level(load_yaml(source["path"]))
        for row in compare_lowlevel_yaml.walk_leaves(data):
            path = row["normalized_path"]
            leaf = leaves.setdefault(
                path,
                {
                    "path": path,
                    "value_types": set(),
                    "observed_modes": set(),
                    "sources": set(),
                },
            )
            leaf["value_types"].add(row["value_type"])
            if source["mode"]:
                leaf["observed_modes"].add(source["mode"])
            leaf["sources"].add(source["label"])
    return leaves


def matching_rule(path: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    for rule in rules:
        patterns = rule.get("patterns", [])
        if any(fnmatch.fnmatchcase(path, str(pattern)) for pattern in patterns):
            return rule
    raise ContractError(f"uncovered low-level leaf: {path}")


def resolved_unit(path: str, value_types: set[str], rules: list[dict[str, Any]]) -> str:
    for rule in rules:
        if any(
            fnmatch.fnmatchcase(path, str(pattern))
            for pattern in rule.get("patterns", [])
        ):
            unit = rule.get("unit")
            if isinstance(unit, str) and unit:
                return unit
    if value_types <= {"bool"}:
        return "dimensionless"
    if value_types <= {"str", "NoneType"}:
        return "not-applicable"
    return "dimensionless"


def allowed_domain(value_types: set[str], executable: bool) -> dict[str, Any]:
    if not executable:
        return {"kind": "ignored", "finite_required": False}
    if value_types <= {"bool"}:
        return {"kind": "boolean", "finite_required": False}
    if value_types <= {"int"}:
        return {"kind": "typed-integer", "finite_required": False}
    if value_types <= {"int", "float"}:
        return {"kind": "typed-real", "finite_required": True}
    if value_types <= {"str"}:
        return {"kind": "typed-string-or-enum", "finite_required": False}
    if "NoneType" in value_types:
        return {"kind": "typed-nullable", "finite_required": "float" in value_types}
    return {"kind": "typed-structured", "finite_required": "float" in value_types}


def resolved_allowed_domain(
    rule: dict[str, Any], value_types: set[str], executable: bool
) -> dict[str, Any]:
    override = rule.get("allowed_domain")
    if override is None:
        return allowed_domain(value_types, executable)
    if (
        not isinstance(override, dict)
        or not isinstance(override.get("kind"), str)
        or not isinstance(override.get("finite_required"), bool)
    ):
        raise ContractError(
            f"rule {rule.get('id')!r} has an invalid allowed_domain"
        )
    return dict(override)


def build_contract(
    repo_root: Path,
    rules_path: Path,
    authority_path: Path,
    cases_path: Path,
    require_all: bool,
) -> dict[str, Any]:
    rules = load_yaml(rules_path)
    if not isinstance(rules, dict) or rules.get("schema") != RULES_SCHEMA:
        raise ContractError(f"rules must declare schema: {RULES_SCHEMA}")
    authority_owners = load_authority_owners(authority_path)
    external = rules.get("external_boundaries", {})
    supported_modes = set(rules.get("supported_modes", []))
    sources, missing = load_sources(repo_root, cases_path, require_all)
    leaves = observed_leaves(sources)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)

    for path in sorted(leaves):
        leaf = leaves[path]
        rule = matching_rule(path, rules.get("authority_rules", []))
        domain = rule.get("authority_domain")
        boundary = rule.get("external_boundary")
        if bool(domain) == bool(boundary):
            raise ContractError(
                f"rule {rule.get('id')!r} must declare exactly one authority domain or external boundary"
            )
        if domain:
            if domain not in authority_owners:
                raise ContractError(
                    f"leaf {path} references unknown authority domain {domain!r}"
                )
            owner = authority_owners[domain]
            authority = str(domain)
            validation_source = rule.get("validation_source")
        else:
            boundary_record = external.get(boundary)
            if not isinstance(boundary_record, dict):
                raise ContractError(
                    f"leaf {path} references unknown external boundary {boundary!r}"
                )
            owner = boundary_record.get("typed_owner")
            authority = str(boundary)
            validation_source = boundary_record.get("validation_source")
        if not isinstance(owner, str) or not owner:
            raise ContractError(f"leaf {path} has no typed/external owner")
        if not isinstance(validation_source, str) or not validation_source:
            raise ContractError(f"leaf {path} has no validation source")
        if not (repo_root / validation_source).is_file():
            raise ContractError(
                f"leaf {path} validation source does not exist: {validation_source}"
            )
        applicable_modes = set(rule.get("modes", []))
        if not applicable_modes or not applicable_modes <= supported_modes:
            raise ContractError(f"leaf {path} has invalid mode applicability")
        # TolTECA's merged low-level files retain defaults for inactive modes.
        # Presence is therefore evidence of schema coverage, not execution
        # applicability; the rule records the latter explicitly.
        observed_modes = set(leaf["observed_modes"])
        state_class = rule.get("state_class")
        if state_class not in set(rules.get("state_classes", [])):
            raise ContractError(f"leaf {path} has invalid state classification")
        executable = bool(rule.get("executable", True))
        value_types = set(leaf["value_types"])
        rows.append(
            {
                "path": path,
                "authority": authority,
                "owner": owner,
                "unit": resolved_unit(path, value_types, rules.get("unit_rules", [])),
                "allowed_domain": resolved_allowed_domain(
                    rule, value_types, executable
                ),
                "applicable_modes": sorted(applicable_modes),
                "observed_modes": sorted(observed_modes),
                "state_class": state_class,
                "resolution_stage": str(rule.get("resolution_stage", "")),
                "validation_source": validation_source,
                "executable": executable,
                "rule_id": str(rule.get("id", "")),
                "value_types": sorted(value_types),
                "sources": sorted(leaf["sources"]),
            }
        )
        counts[authority] += 1

    return {
        "schema_version": RESOLVED_SCHEMA,
        "source_boundary": rules.get("source_boundary", {}),
        "summary": {
            "leaf_count": len(rows),
            "executable_leaf_count": sum(1 for row in rows if row["executable"]),
            "non_executable_leaf_count": sum(1 for row in rows if not row["executable"]),
            "authority_counts": dict(sorted(counts.items())),
            "missing_optional_inputs": missing,
        },
        "leaves": rows,
    }


def semantic_view(contract: dict[str, Any]) -> dict[str, Any]:
    copy = json.loads(json.dumps(contract))
    copy.get("summary", {}).pop("missing_optional_inputs", None)
    return copy


def write_compact_manifest(path: Path, contract: dict[str, Any]) -> None:
    """Keep one resolved leaf per line so contract diffs remain reviewable."""
    value = semantic_view(contract)
    leaves = value.pop("leaves")
    lines = ["{"]
    top_items = list(value.items())
    for index, (key, item) in enumerate(top_items):
        rendered = json.dumps(item, sort_keys=True, separators=(",", ":"))
        lines.append(f"  {json.dumps(key)}: {rendered},")
    lines.append('  "leaves": [')
    for index, leaf in enumerate(leaves):
        suffix = "," if index + 1 < len(leaves) else ""
        rendered = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        lines.append(f"    {rendered}{suffix}")
    lines.extend(["  ]", "}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--rules", default=DEFAULT_RULES)
    parser.add_argument("--authority", default=DEFAULT_AUTHORITY)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    try:
        contract = build_contract(
            repo_root,
            (repo_root / args.rules).resolve(),
            (repo_root / args.authority).resolve(),
            (repo_root / args.cases).resolve(),
            args.require_all,
        )
        manifest_path = (repo_root / args.manifest).resolve()
        if args.write_manifest:
            write_compact_manifest(manifest_path, contract)
        elif not manifest_path.is_file():
            raise ContractError(f"resolved manifest is missing: {manifest_path}")
        else:
            checked = json.loads(manifest_path.read_text(encoding="utf-8"))
            if semantic_view(contract) != checked:
                raise ContractError(
                    "resolved leaf contract differs from checked manifest; regenerate with --write-manifest"
                )
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(contract, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (OSError, json.JSONDecodeError, yaml.YAMLError, ContractError) as exc:
        print(f"config leaf contract: {exc}", file=sys.stderr)
        return 1

    summary = contract["summary"]
    print(
        "config leaf contract: "
        f"leaves={summary['leaf_count']} executable={summary['executable_leaf_count']} "
        f"non_executable={summary['non_executable_leaf_count']} "
        f"authorities={len(summary['authority_counts'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
