#!/usr/bin/env python3
"""Merge, inspect, and validate the checked TolTECA mode configuration kits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections.abc import Mapping, MutableMapping, MutableSequence
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "citlali-tolteca-mode-kit-v1"
NUMBERED_YAML_RE = re.compile(r"^(\d+)_.*\.ya?ml$")
LIST_DSL_RE = re.compile(
    r"^\[(?:(?P<index>-?\d+)|(?P<start>-?\d*):(?P<stop>-?\d*)?"
    r"(?::(?P<step>-?\d*))?)?\]$"
)
DEFAULT_REQUIRED_ROLES = {
    "70_pipeline.yaml",
    "71_runtime.yaml",
    "72_observation.yaml",
    "80_products.yaml",
    "90_user_overrides.yaml",
}
MODE_REDUCTION_TYPES = {
    "point": "pointing",
    "oof": "pointing",
    "beammap": "beammap",
    "science": "science",
}


class ModeKitError(RuntimeError):
    """Raised when a mode kit cannot be merged or validated."""


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(value: Any) -> str:
    return yaml.safe_dump(value, sort_keys=False, width=100)


def numbered_yaml_files(directory: Path) -> list[Path]:
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and NUMBERED_YAML_RE.match(path.name)
    ]
    return sorted(files, key=lambda path: (int(path.name.split("_", 1)[0]), path.name))


def _parse_list_key(key: Any) -> tuple[str, int | slice | None]:
    text = str(key)
    match = LIST_DSL_RE.match(text)
    if match is not None:
        groups = match.groupdict()
        if groups["index"] is not None:
            return "index", int(groups["index"])
        if ":" not in text:
            return "slice", None
        parts = [groups[name] or "" for name in ("start", "stop", "step")]
        values = [int(part) if part else None for part in parts]
        return "slice", slice(*values)
    try:
        return "index", int(text)
    except ValueError as error:
        raise ModeKitError(
            f"invalid TolTECA list update key {key!r}; expected an index or list slice"
        ) from error


def recursive_update(target: Any, patch: Mapping[Any, Any]) -> set[str]:
    """Apply Tollan ``rupdate`` semantics without importing TolTECA/Tollan."""
    if not isinstance(target, (MutableMapping, MutableSequence)):
        raise ModeKitError(f"cannot update {type(target).__name__} as a container")

    touched: set[str] = set()
    queue: list[tuple[Any, Mapping[Any, Any], tuple[str, ...]]] = [(target, patch, ())]
    while queue:
        current, updates, prefix = queue.pop(0)
        for raw_key, patch_value in updates.items():
            if isinstance(current, MutableMapping):
                key = raw_key
                child_prefix = prefix + (str(key),)
                if key not in current:
                    current[key] = {}
                current_value = current[key]
            elif isinstance(current, MutableSequence):
                operation, operand = _parse_list_key(raw_key)
                if operation == "slice":
                    slice_value = copy.deepcopy(patch_value)
                    if not isinstance(slice_value, list):
                        slice_value = [slice_value]
                    slice_object = (
                        slice(len(current), len(current)) if operand is None else operand
                    )
                    current[slice_object] = slice_value
                    touched.update(flatten_leaves(current, prefix))
                    continue
                assert isinstance(operand, int)
                try:
                    current_value = current[operand]
                except IndexError as error:
                    raise ModeKitError(
                        f"list index {operand} out of range (length {len(current)})"
                    ) from error
                key = operand
                resolved_index = operand if operand >= 0 else len(current) + operand
                child_prefix = prefix + (f"[{resolved_index}]",)
            else:
                raise ModeKitError(
                    f"cannot update {type(current).__name__} as a container"
                )

            if not isinstance(patch_value, Mapping):
                current[key] = copy.deepcopy(patch_value)
                touched.update(flatten_leaves(current[key], child_prefix))
            elif not isinstance(current_value, (MutableMapping, MutableSequence)):
                current[key] = copy.deepcopy(patch_value)
                touched.update(flatten_leaves(current[key], child_prefix))
            else:
                queue.append((current_value, patch_value, child_prefix))
    return touched


def path_text(parts: tuple[str, ...]) -> str:
    result = ""
    for part in parts:
        if part.startswith("["):
            result += part
        elif result:
            result += "." + part
        else:
            result = part
    return result


def normalized_path(path: str) -> str:
    return re.sub(r"\[-?\d+\]", "[]", path)


def flatten_leaves(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    if isinstance(value, Mapping):
        leaves: dict[str, Any] = {}
        for key, child in value.items():
            leaves.update(flatten_leaves(child, prefix + (str(key),)))
        return leaves
    if isinstance(value, list):
        leaves = {}
        for index, child in enumerate(value):
            leaves.update(flatten_leaves(child, prefix + (f"[{index}]",)))
        return leaves
    return {path_text(prefix): value}


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def merge_files(files: list[Path]) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    changes: list[dict[str, Any]] = []

    for path in files:
        patch = load_yaml(path) or {}
        if not isinstance(patch, Mapping):
            raise ModeKitError(f"numbered config must contain a mapping: {path}")
        before = flatten_leaves(merged)
        touched = recursive_update(merged, patch)
        after = flatten_leaves(merged)

        for leaf in sorted(set(before) | set(after)):
            old_present = leaf in before
            new_present = leaf in after
            old_value = before.get(leaf)
            new_value = after.get(leaf)
            if old_present == new_present and _value_key(old_value) == _value_key(new_value):
                continue
            prior_source = origins.get(leaf)
            if new_present:
                origins[leaf] = path.name
            else:
                origins.pop(leaf, None)
            changes.append(
                {
                    "path": leaf,
                    "source": path.name,
                    "previous_source": prior_source,
                    "kind": (
                        "added" if not old_present else "removed" if not new_present else "overridden"
                    ),
                    "previous": old_value if old_present else None,
                    "value": new_value if new_present else None,
                }
            )
        changed_paths = {
            change["path"] for change in changes if change["source"] == path.name
        }
        for leaf in sorted(touched - changed_paths):
            if leaf not in after:
                continue
            prior_source = origins.get(leaf)
            origins[leaf] = path.name
            changes.append(
                {
                    "path": leaf,
                    "source": path.name,
                    "previous_source": prior_source,
                    "kind": "reasserted",
                    "previous": before.get(leaf),
                    "value": after[leaf],
                }
            )
    return merged, origins, changes


def _step_values(steps: Any) -> list[Any]:
    if isinstance(steps, list):
        return steps
    if isinstance(steps, Mapping):
        def sort_key(key: Any) -> tuple[int, Any]:
            text = str(key)
            return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)

        return [steps[key] for key in sorted(steps, key=sort_key)]
    return []


def extract_low_level(merged: Mapping[str, Any]) -> dict[str, Any]:
    reduce_section = merged.get("reduce", {})
    if not isinstance(reduce_section, Mapping):
        raise ModeKitError("merged config has no reduce mapping")
    for step in _step_values(reduce_section.get("steps", [])):
        if not isinstance(step, Mapping):
            continue
        config = step.get("config", {})
        if isinstance(config, Mapping) and "low_level" in config:
            low_level = config["low_level"] or {}
            if not isinstance(low_level, dict):
                raise ModeKitError("Citlali low_level config must be a mapping")
            return low_level
    raise ModeKitError("merged config has no Citlali low_level section")


def canonical_policy(low_level: Mapping[str, Any]) -> dict[str, Any]:
    policy = copy.deepcopy(dict(low_level))
    policy.pop("inputs", None)
    runtime = policy.get("runtime")
    if isinstance(runtime, dict):
        runtime["output_dir"] = "."
    return policy


def policy_sha256(low_level: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_policy(low_level),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_leaf_contract(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["path"]: row for row in data["leaves"]}


def low_level_origin(
    full_path: str,
    origins: Mapping[str, str],
) -> str | None:
    return origins.get(f"reduce.steps[0].config.low_level.{full_path}")


def build_report(
    mode: str,
    directory: Path,
    manifest_entry: Mapping[str, Any],
    contract: Mapping[str, Mapping[str, Any]],
    allow_policy_drift: bool = False,
) -> dict[str, Any]:
    files = numbered_yaml_files(directory)
    merged, origins, changes = merge_files(files)
    low_level = extract_low_level(merged)
    policy = canonical_policy(low_level)
    leaf_rows = flatten_leaves(policy)
    unknown = sorted(
        path for path in leaf_rows if normalized_path(path) not in contract
    )
    actual_hash = policy_sha256(policy)
    expected_hash = str(manifest_entry.get("policy_sha256", ""))
    expected_type = MODE_REDUCTION_TYPES[mode]
    actual_type = policy.get("runtime", {}).get("reduction_type")
    required_files_value = manifest_entry.get("required_files", DEFAULT_REQUIRED_ROLES)
    if not isinstance(required_files_value, (list, set, tuple)):
        raise ModeKitError("manifest required_files must be a list")
    required_files = {str(value) for value in required_files_value}
    missing_roles = sorted(required_files - {path.name for path in files})
    expert_override_file = str(
        manifest_entry.get("expert_override_file", "90_user_overrides.yaml")
    )

    low_level_prefix = "reduce.steps[0].config.low_level."
    low_level_changes = []
    for change in changes:
        if not change["path"].startswith(low_level_prefix):
            continue
        config_path = change["path"][len(low_level_prefix):]
        contract_row = contract.get(normalized_path(config_path), {})
        low_level_changes.append(
            {
                **change,
                "config_path": config_path,
                "authority": contract_row.get("authority"),
                "owner": contract_row.get("owner"),
            }
        )

    errors: list[str] = []
    if missing_roles:
        errors.append("missing required role files: " + ", ".join(missing_roles))
    if actual_type != expected_type:
        errors.append(
            f"runtime.reduction_type is {actual_type!r}; expected {expected_type!r}"
        )
    if unknown:
        errors.append(f"{len(unknown)} low-level leaves are absent from the leaf contract")
    if not expected_hash:
        errors.append("manifest has no policy_sha256")
    elif actual_hash != expected_hash and not allow_policy_drift:
        errors.append(
            f"policy hash is {actual_hash}; expected {expected_hash}"
        )

    final_leaves = []
    for path, value in sorted(leaf_rows.items()):
        contract_row = contract.get(normalized_path(path), {})
        final_leaves.append(
            {
                "path": path,
                "value": value,
                "source": low_level_origin(path, origins),
                "authority": contract_row.get("authority"),
                "owner": contract_row.get("owner"),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "directory": str(directory),
        "files": [path.name for path in files],
        "record_id": manifest_entry.get("record_id"),
        "baseline_sha256": manifest_entry.get("baseline_sha256"),
        "policy_sha256": actual_hash,
        "expected_policy_sha256": expected_hash,
        "policy_matches_manifest": actual_hash == expected_hash,
        "reduction_type": actual_type,
        "leaf_count": len(leaf_rows),
        "unknown_leaves": unknown,
        "missing_roles": missing_roles,
        "changes": changes,
        "low_level_changes": low_level_changes,
        "expert_override_changes": [
            change
            for change in low_level_changes
            if change["source"] == expert_override_file
        ],
        "final_low_level_leaves": final_leaves,
        "errors": errors,
        "valid": not errors,
        "merged": merged,
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        f"# TolTECA Mode Kit: {report['mode']}",
        "",
        f"- Valid: `{str(report['valid']).lower()}`",
        f"- Reduction type: `{report['reduction_type']}`",
        f"- Low-level leaves: {report['leaf_count']}",
        f"- Policy SHA-256: `{report['policy_sha256']}`",
        f"- Accepted record: `{report.get('record_id')}`",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in report["files"])
    lines.extend(["", "## Errors", ""])
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Effective Low-Level Values",
            "",
            "| Path | Source | Authority | Value |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in report["final_low_level_leaves"]:
        value = json.dumps(row["value"], sort_keys=True, default=str).replace("|", "\\|")
        lines.append(
            f"| `{row['path']}` | `{row.get('source') or ''}` | "
            f"`{row.get('authority') or ''}` | `{value}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_entries(path: Path) -> dict[str, Any]:
    data = load_yaml(path) or {}
    modes = data.get("modes", {}) if isinstance(data, Mapping) else {}
    if not isinstance(modes, Mapping):
        raise ModeKitError(f"manifest has no modes mapping: {path}")
    return dict(modes)


def validate_modes(
    config_root: Path,
    manifest_path: Path,
    contract_path: Path,
    modes: list[str],
    allow_policy_drift: bool = False,
) -> list[dict[str, Any]]:
    manifest = _manifest_entries(manifest_path)
    contract = load_leaf_contract(contract_path)
    reports = []
    for mode in modes:
        if mode not in MODE_REDUCTION_TYPES:
            raise ModeKitError(f"unsupported mode {mode!r}")
        entry = manifest.get(mode)
        if not isinstance(entry, Mapping):
            raise ModeKitError(f"manifest has no entry for mode {mode!r}")
        reports.append(
            build_report(
                mode,
                config_root / mode,
                entry,
                contract,
                allow_policy_drift=allow_policy_drift,
            )
        )
    return reports


def parse_args(argv: list[str]) -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("merge", "validate", "validate-all"),
        help="Operation to perform.",
    )
    parser.add_argument("--mode", choices=tuple(MODE_REDUCTION_TYPES))
    parser.add_argument("--config-root", default=str(root / "config/tolteca"))
    parser.add_argument(
        "--mode-dir",
        help="Inspect one deployed mode directory directly instead of CONFIG_ROOT/MODE.",
    )
    parser.add_argument("--manifest", default=str(root / "config/tolteca/manifest.yaml"))
    parser.add_argument(
        "--leaf-contract",
        default=str(root / "tools/config/config_leaf_contract_resolved.json"),
    )
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    parser.add_argument("--yaml-out")
    parser.add_argument(
        "--allow-policy-drift",
        action="store_true",
        help="Report deliberate policy overrides without failing the baseline hash check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.command in {"merge", "validate"} and not args.mode:
        raise ModeKitError(f"{args.command} requires --mode")
    if args.mode_dir and args.command == "validate-all":
        raise ModeKitError("--mode-dir cannot be used with validate-all")
    modes = list(MODE_REDUCTION_TYPES) if args.command == "validate-all" else [args.mode]
    manifest_path = Path(args.manifest).expanduser().resolve()
    contract_path = Path(args.leaf_contract).expanduser().resolve()
    allow_policy_drift = args.allow_policy_drift or args.command == "merge"
    if args.mode_dir:
        manifest = _manifest_entries(manifest_path)
        contract = load_leaf_contract(contract_path)
        reports = [
            build_report(
                args.mode,
                Path(args.mode_dir).expanduser().resolve(),
                manifest[args.mode],
                contract,
                allow_policy_drift=allow_policy_drift,
            )
        ]
    else:
        reports = validate_modes(
            Path(args.config_root).expanduser().resolve(),
            manifest_path,
            contract_path,
            modes,
            allow_policy_drift=allow_policy_drift,
        )

    payload: Any = reports[0] if len(reports) == 1 else {
        "schema_version": SCHEMA_VERSION,
        "reports": reports,
        "valid": all(report["valid"] for report in reports),
    }
    if args.json_out:
        path = Path(args.json_out).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        if len(reports) != 1:
            raise ModeKitError("--markdown-out requires one --mode")
        write_markdown(reports[0], Path(args.markdown_out).expanduser().resolve())
    if args.yaml_out:
        if len(reports) != 1:
            raise ModeKitError("--yaml-out requires one --mode")
        path = Path(args.yaml_out).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_yaml(reports[0]["merged"]), encoding="utf-8")

    for report in reports:
        state = "PASS" if report["valid"] else "FAIL"
        print(
            f"{state} {report['mode']}: leaves={report['leaf_count']} "
            f"hash={report['policy_sha256']}"
        )
        for error in report["errors"]:
            print(f"  - {error}")
    return 0 if all(report["valid"] for report in reports) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ModeKitError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
