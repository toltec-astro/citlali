#!/usr/bin/env python3
"""Audit compact-config coverage of user-facing low-level Citlali keys."""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

import classify_lowlevel_config
import compare_lowlevel_yaml
import expand_compact_config
import lowlevel_to_compact_config


SCHEMA_VERSION = "citlali-compact-surface-coverage-v1"
SUITE_SCHEMA = "citlali-compact-compatibility-suite-v1"
USER_FACING = "user-facing"
MODES = {"pointing", "oof", "beammap", "science"}


class CoverageError(RuntimeError):
    """Raised for user-correctable audit errors."""


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_baseline_configs(cases_path: Path) -> list[dict[str, str]]:
    suite = load_yaml(cases_path)
    if not isinstance(suite, dict) or suite.get("schema") != SUITE_SCHEMA:
        raise CoverageError(f"cases file must declare schema: {SUITE_SCHEMA}")
    cases = suite.get("cases")
    if not isinstance(cases, list):
        raise CoverageError("cases file must contain a cases list")

    configs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for case in cases:
        if not isinstance(case, dict):
            continue
        base_config = case.get("base_config")
        if not isinstance(base_config, str):
            continue
        path = resolve_path(base_config, cases_path.parent)
        intent = str(case.get("intent", ""))
        key = (str(path), intent)
        if key in seen:
            continue
        seen.add(key)
        label = intent or str(case.get("name", path.stem))
        if any(item["label"] == label for item in configs):
            label = str(case.get("name", path.stem))
        configs.append(
            {
                "label": label,
                "intent": intent,
                "path": str(path),
                "source_case": str(case.get("name", "")),
            }
        )
    return configs


def classify_rows(tree: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in compare_lowlevel_yaml.walk_leaves(tree):
        rule = classify_lowlevel_config.classify_path(row["normalized_path"], rules)
        rows.append(
            {
                **row,
                "classification": rule["classification"],
                "rule_id": rule["id"],
                "rule_pattern": rule["pattern"],
                "owner": rule["owner"],
                "reason": rule["reason"],
            }
        )
    return rows


def compact_mapping_fields(
    low_level_path: str,
    normalized_path: str,
    mappings: list[dict[str, Any]],
) -> list[str]:
    fields: set[str] = set()
    for mapping in mappings:
        source = mapping.get("low_level")
        compact = mapping.get("compact")
        if not isinstance(source, str) or not isinstance(compact, str):
            continue
        if compact == "expert" or source == "*":
            continue
        source_patterns = {source}
        if source.endswith(".*"):
            source_patterns.add(source[:-2])
        matched = False
        for pattern in source_patterns:
            if "*" in pattern:
                matched = fnmatch.fnmatchcase(normalized_path, pattern) or fnmatch.fnmatchcase(low_level_path, pattern)
            else:
                matched = (
                    normalized_path == pattern
                    or low_level_path == pattern
                    or normalized_path.startswith(pattern + ".")
                    or low_level_path.startswith(pattern + ".")
                )
            if matched:
                break
        if matched:
            fields.add(compact)
    return sorted(fields)


def compact_generated_paths(
    compact: dict[str, Any],
    compact_path: Path,
) -> tuple[set[str], list[str]]:
    compact_only = copy.deepcopy(compact)
    compact_only["expert"] = {}
    patch, _applied, warnings = expand_compact_config.build_compact_patch(compact_only, compact_path)
    rows = compare_lowlevel_yaml.walk_leaves(patch)
    return {row["normalized_path"] for row in rows}, warnings


def profile_owned_reason(mode: str, normalized_path: str) -> str:
    if mode != "beammap" and normalized_path.startswith("beammap."):
        return (
            "Beammap defaults are present in the low-level baseline but inactive "
            f"for {mode} reductions; the mode/profile owns them."
        )
    return ""


def audit_config(
    config: dict[str, str],
    *,
    rules: dict[str, Any],
    rules_path: Path,
    profiles_dir: Path,
) -> dict[str, Any]:
    path = Path(config["path"])
    if not path.exists():
        return {
            "label": config["label"],
            "intent": config.get("intent", ""),
            "path": str(path),
            "missing": True,
            "rows": [],
            "summary": {"user_facing": 0, "covered": 0, "profile_owned": 0, "gaps": 0},
        }

    low_level = compare_lowlevel_yaml.extract_low_level(load_yaml(path))
    if not isinstance(low_level, dict):
        raise CoverageError(f"{path} does not contain a low-level mapping")

    intent = config.get("intent") or ""
    mode = intent if intent in MODES else lowlevel_to_compact_config.infer_mode(low_level, None)
    profile = lowlevel_to_compact_config.PASSTHROUGH_PROFILE_BY_MODE[mode]
    expand_compact_config.load_profile(profiles_dir, profile)
    compact, mappings = lowlevel_to_compact_config.build_compact(
        low_level,
        mode=mode,
        profile=profile,
        include_output_dir=True,
        preserve_unmapped=True,
        classification_rules=rules_path,
        compact_path=path.with_suffix(".compact-audit.yaml"),
    )
    covered_paths, warnings = compact_generated_paths(compact, path.with_suffix(".compact-audit.yaml"))

    rows: list[dict[str, Any]] = []
    for row in classify_rows(low_level, rules):
        if row["classification"] != USER_FACING:
            continue
        compact_fields = compact_mapping_fields(row["path"], row["normalized_path"], mappings)
        covered = row["normalized_path"] in covered_paths
        inactive_reason = "" if covered else profile_owned_reason(mode, row["normalized_path"])
        status = "covered" if covered else "profile-owned" if inactive_reason else "gap"
        rows.append(
            {
                "config_label": config["label"],
                "mode": mode,
                "config_path": str(path),
                "path": row["path"],
                "normalized_path": row["normalized_path"],
                "top": row["top"],
                "status": status,
                "compact_fields": compact_fields,
                "rule_id": row["rule_id"],
                "owner": row["owner"],
                "reason": row["reason"],
                "status_reason": inactive_reason,
                "value_preview": row["value_preview"],
            }
        )

    counts = collections.Counter(row["status"] for row in rows)
    return {
        "label": config["label"],
        "intent": intent,
        "mode": mode,
        "path": str(path),
        "missing": False,
        "profile": profile,
        "warnings": warnings,
        "summary": {
            "user_facing": len(rows),
            "covered": counts["covered"],
            "profile_owned": counts["profile-owned"],
            "gaps": counts["gap"],
        },
        "rows": rows,
    }


def build_report(configs: list[dict[str, str]], rules: dict[str, Any], rules_path: Path, require_all: bool) -> dict[str, Any]:
    profiles_dir = Path(__file__).resolve().with_name("profiles")
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for config in configs:
        result = audit_config(config, rules=rules, rules_path=rules_path, profiles_dir=profiles_dir)
        results.append(result)
        if result.get("missing"):
            missing.append(result["path"])
    if missing and require_all:
        raise CoverageError("missing config input(s): " + ", ".join(missing))

    all_rows = [row for result in results for row in result["rows"]]
    unique_by_mode: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in all_rows:
        unique_by_mode[row["mode"]].setdefault(row["normalized_path"], row)

    mode_summary: dict[str, dict[str, Any]] = {}
    for mode, rows_by_path in sorted(unique_by_mode.items()):
        rows = list(rows_by_path.values())
        counts = collections.Counter(row["status"] for row in rows)
        total = len(rows)
        active_total = counts["covered"] + counts["gap"]
        mode_summary[mode] = {
            "user_facing_unique_paths": total,
            "covered": counts["covered"],
            "profile_owned": counts["profile-owned"],
            "gaps": counts["gap"],
            "coverage_fraction": counts["covered"] / active_total if active_total else 1.0,
        }

    unique_rows = {
        f"{row['mode']}:{row['normalized_path']}": row
        for row in all_rows
    }
    counts = collections.Counter(row["status"] for row in unique_rows.values())
    total_unique = len(unique_rows)
    return {
        "schema": SCHEMA_VERSION,
        "rules_file": str(rules_path),
        "config_count": len([result for result in results if not result.get("missing")]),
        "missing_inputs": missing,
        "summary": {
            "user_facing_unique_mode_paths": total_unique,
            "covered": counts["covered"],
            "profile_owned": counts["profile-owned"],
            "gaps": counts["gap"],
            "coverage_fraction": counts["covered"] / (counts["covered"] + counts["gap"])
            if (counts["covered"] + counts["gap"])
            else 1.0,
            "by_mode": mode_summary,
        },
        "configs": [
            {
                key: value
                for key, value in result.items()
                if key != "rows"
            }
            for result in results
        ],
        "rows": all_rows,
    }


def markdown_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Compact Surface Coverage Audit",
        "",
        "This audit checks whether low-level keys classified as `user-facing` are",
        "represented by current compact-config fields for the representative",
        "pointing, OOF, beammap, and science baselines. It is a shadow/config",
        "tooling check only; it does not change Citlali runtime parsing.",
        "",
        "## Summary",
        "",
        f"- Configs checked: {report['config_count']}",
        f"- Unique mode/path user-facing keys: {summary['user_facing_unique_mode_paths']}",
        f"- Covered by compact fields: {summary['covered']}",
        f"- Profile-owned inactive defaults: {summary['profile_owned']}",
        f"- Actionable gaps: {summary['gaps']}",
        f"- Actionable coverage: {format_percent(summary['coverage_fraction'])}",
        "",
        "`Covered` means the current compact fields expand back to that low-level",
        "path without using `expert:`. `Profile-owned` means the key is present",
        "in the low-level baseline but is inactive for that reduction mode and",
        "should be owned by a profile/default rather than normal user authoring.",
        "`Gap` means the value is still preserved only through the expert escape",
        "hatch and needs a compact-field or policy decision.",
        "",
        "| Mode | User-Facing Paths | Covered | Profile-Owned | Gaps | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, mode_summary in summary["by_mode"].items():
        lines.append(
            f"| `{mode}` | {mode_summary['user_facing_unique_paths']} | "
            f"{mode_summary['covered']} | {mode_summary['profile_owned']} | "
            f"{mode_summary['gaps']} | "
            f"{format_percent(mode_summary['coverage_fraction'])} |"
        )

    gaps = [row for row in report["rows"] if row["status"] == "gap"]
    unique_gaps = {
        f"{row['mode']}:{row['normalized_path']}": row
        for row in gaps
    }
    lines.extend(["", "## Gaps", ""])
    if not unique_gaps:
        lines.append("_No user-facing compact coverage gaps were found._")
    else:
        lines.extend([
            "| Mode | Low-Level Path | Rule | Reason |",
            "| --- | --- | --- | --- |",
        ])
        for row in sorted(unique_gaps.values(), key=lambda item: (item["mode"], item["normalized_path"])):
            lines.append(
                f"| `{markdown_cell(row['mode'])}` | `{markdown_cell(row['normalized_path'])}` | "
                f"`{markdown_cell(row['rule_id'])}` | {markdown_cell(row['reason'])} |"
            )

    profile_owned = [row for row in report["rows"] if row["status"] == "profile-owned"]
    unique_profile_owned = {
        f"{row['mode']}:{row['normalized_path']}": row
        for row in profile_owned
    }
    if unique_profile_owned:
        lines.extend(["", "## Profile-Owned Inactive Defaults", ""])
        lines.extend([
            "| Mode | Low-Level Path | Reason |",
            "| --- | --- | --- |",
        ])
        for row in sorted(unique_profile_owned.values(), key=lambda item: (item["mode"], item["normalized_path"])):
            lines.append(
                f"| `{markdown_cell(row['mode'])}` | `{markdown_cell(row['normalized_path'])}` | "
                f"{markdown_cell(row['status_reason'])} |"
            )

    covered = [row for row in report["rows"] if row["status"] == "covered"]
    unique_covered = {
        f"{row['mode']}:{row['normalized_path']}": row
        for row in covered
    }
    lines.extend(["", "## Covered User-Facing Paths", ""])
    lines.extend([
        "| Mode | Low-Level Path | Compact Field(s) |",
        "| --- | --- | --- |",
    ])
    for row in sorted(unique_covered.values(), key=lambda item: (item["mode"], item["normalized_path"])):
        fields = ", ".join(f"`{field}`" for field in row["compact_fields"]) or "`<derived>`"
        lines.append(
            f"| `{markdown_cell(row['mode'])}` | `{markdown_cell(row['normalized_path'])}` | {fields} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_label",
        "mode",
        "status",
        "normalized_path",
        "path",
        "compact_fields",
        "rule_id",
        "owner",
        "reason",
        "status_reason",
        "value_preview",
        "config_path",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["rows"]:
            output = {key: row.get(key, "") for key in fieldnames}
            output["compact_fields"] = ",".join(row.get("compact_fields", []))
            writer.writerow(output)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("compact_compatibility_cases.yaml")),
        help="Compatibility cases file listing representative low-level baselines.",
    )
    parser.add_argument(
        "--classification-rules",
        default=str(Path(__file__).with_name("config_key_classification.yaml")),
        help="Low-level key classification rules.",
    )
    parser.add_argument("--require-all", action="store_true", help="Fail if any baseline input is missing.")
    parser.add_argument("--json-out", default="", help="Optional JSON report output.")
    parser.add_argument("--csv-out", default="", help="Optional CSV row output.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown report output.")
    parser.add_argument("--fail-on-gaps", action="store_true", help="Exit non-zero if user-facing gaps remain.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cases_path = Path(args.cases).expanduser().resolve()
    rules_path = Path(args.classification_rules).expanduser().resolve()
    try:
        rules = classify_lowlevel_config.load_rules(rules_path)
        configs = load_baseline_configs(cases_path)
        report = build_report(configs, rules, rules_path, args.require_all)
    except (
        OSError,
        yaml.YAMLError,
        CoverageError,
        classify_lowlevel_config.ClassificationError,
        expand_compact_config.ConfigError,
        lowlevel_to_compact_config.ConvertError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_out:
        write_csv(report, Path(args.csv_out).expanduser())
    if args.markdown_out:
        write_markdown(report, Path(args.markdown_out).expanduser())

    summary = report["summary"]
    print(
        "compact surface coverage: "
        f"covered={summary['covered']} profile_owned={summary['profile_owned']} gaps={summary['gaps']} "
        f"coverage={format_percent(summary['coverage_fraction'])}"
    )
    if args.fail_on_gaps and summary["gaps"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
