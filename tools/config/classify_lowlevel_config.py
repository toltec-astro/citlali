#!/usr/bin/env python3
"""Classify Citlali low-level YAML keys by intended authoring exposure."""

from __future__ import annotations

import argparse
import collections
import csv
import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

import compare_lowlevel_yaml


SCHEMA_VERSION = "citlali-lowlevel-key-classification-v1"
RULES_SCHEMA = "citlali-config-key-classification-rules-v1"
CLASSIFICATIONS = ("user-facing", "expert", "hidden/internal", "deprecated")


class ClassificationError(RuntimeError):
    """Raised for user-correctable classification errors."""


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def parse_config_arg(value: str, base_dir: Path) -> dict[str, str]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
    else:
        raw_path = value
        label = Path(os.path.expandvars(os.path.expanduser(value))).stem
    if not label:
        raise ClassificationError(f"empty label in --config {value!r}")
    path = resolve_path(raw_path, base_dir)
    return {"label": label, "path": str(path)}


def load_case_configs(cases_path: Path) -> list[dict[str, str]]:
    data = load_yaml(cases_path)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ClassificationError("cases file must contain a cases list")

    configs: list[dict[str, str]] = []
    seen: set[Path] = set()
    for case in data["cases"]:
        if not isinstance(case, dict):
            continue
        base_config = case.get("base_config")
        if not isinstance(base_config, str):
            continue
        path = resolve_path(base_config, cases_path.parent)
        if path in seen:
            continue
        seen.add(path)
        intent = str(case.get("intent", ""))
        name = str(case.get("name", path.stem))
        label = intent or name
        if any(existing["label"] == label for existing in configs):
            label = name
        configs.append({"label": label, "intent": intent, "path": str(path)})
    return configs


def validate_rule(rule: Any, index: int) -> dict[str, str]:
    if not isinstance(rule, dict):
        raise ClassificationError(f"rule {index} must be a mapping")
    pattern = rule.get("pattern")
    classification = rule.get("classification")
    if not isinstance(pattern, str) or not pattern:
        raise ClassificationError(f"rule {index} must define a non-empty pattern")
    if classification not in CLASSIFICATIONS:
        raise ClassificationError(
            f"rule {rule.get('id', index)!r} has invalid classification {classification!r}"
        )
    return {
        "id": str(rule.get("id", pattern)),
        "pattern": pattern,
        "classification": str(classification),
        "owner": str(rule.get("owner", "")),
        "reason": str(rule.get("reason", "")),
    }


def load_rules(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ClassificationError("rules file must be a mapping")
    if data.get("schema") != RULES_SCHEMA:
        raise ClassificationError(f"rules file must declare schema: {RULES_SCHEMA}")

    raw_fallback = data.get("fallback", {})
    if not isinstance(raw_fallback, dict):
        raw_fallback = {}
    fallback_classification = raw_fallback.get("classification", "expert")
    if fallback_classification not in CLASSIFICATIONS:
        raise ClassificationError(f"fallback has invalid classification {fallback_classification!r}")
    fallback = {
        "id": "fallback",
        "pattern": "*",
        "classification": str(fallback_classification),
        "owner": str(raw_fallback.get("owner", "")),
        "reason": str(raw_fallback.get("reason", "")),
    }

    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        raise ClassificationError("rules file must contain a rules list")
    rules = [validate_rule(rule, index) for index, rule in enumerate(raw_rules, start=1)]
    return {
        "path": str(path),
        "description": str(data.get("description", "")),
        "classes": data.get("classes", {}),
        "fallback": fallback,
        "rules": rules,
    }


def classify_path(path: str, rules: dict[str, Any]) -> dict[str, str]:
    for rule in rules["rules"]:
        if fnmatch.fnmatchcase(path, rule["pattern"]):
            return rule
    return rules["fallback"]


def classify_config(config: dict[str, str], rules: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["path"])
    data = compare_lowlevel_yaml.extract_low_level(load_yaml(path))
    rows = compare_lowlevel_yaml.walk_leaves(data)
    classified_rows: list[dict[str, Any]] = []
    for row in rows:
        rule = classify_path(row["normalized_path"], rules)
        classified_rows.append(
            {
                "config_label": config["label"],
                "intent": config.get("intent", ""),
                "config_path": str(path),
                "path": row["path"],
                "normalized_path": row["normalized_path"],
                "top": row["top"],
                "value_type": row["value_type"],
                "value_preview": row["value_preview"],
                "classification": rule["classification"],
                "rule_id": rule["id"],
                "rule_pattern": rule["pattern"],
                "owner": rule["owner"],
                "reason": rule["reason"],
            }
        )

    counts = collections.Counter(row["classification"] for row in classified_rows)
    unique_counts = collections.Counter(
        row["classification"] for row in unique_rows(classified_rows)
    )
    fallback_count = sum(1 for row in classified_rows if row["rule_id"] == "fallback")
    return {
        "label": config["label"],
        "intent": config.get("intent", ""),
        "path": str(path),
        "leaf_count": len(classified_rows),
        "unique_path_count": len({row["normalized_path"] for row in classified_rows}),
        "fallback_leaf_count": fallback_count,
        "leaf_count_by_classification": ordered_class_counts(counts),
        "unique_path_count_by_classification": ordered_class_counts(unique_counts),
        "rows": classified_rows,
    }


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result.setdefault(row["normalized_path"], row)
    return list(result.values())


def ordered_class_counts(counter: collections.Counter[str]) -> dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in CLASSIFICATIONS}


def build_report(configs: list[dict[str, str]], rules: dict[str, Any], require_all: bool) -> dict[str, Any]:
    config_results: list[dict[str, Any]] = []
    missing: list[str] = []
    for config in configs:
        if not Path(config["path"]).exists():
            missing.append(config["path"])
            continue
        config_results.append(classify_config(config, rules))

    if missing and require_all:
        raise ClassificationError("missing config input(s): " + ", ".join(missing))

    all_rows = [row for config in config_results for row in config["rows"]]
    all_unique_rows = unique_rows(all_rows)
    leaf_counts = collections.Counter(row["classification"] for row in all_rows)
    unique_counts = collections.Counter(row["classification"] for row in all_unique_rows)

    top_class_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in all_unique_rows:
        top_class_counts[row["top"]][row["classification"]] += 1

    return {
        "schema": SCHEMA_VERSION,
        "rules_file": rules["path"],
        "rules_description": rules["description"],
        "classes": rules["classes"],
        "missing_inputs": missing,
        "summary": {
            "config_count": len(config_results),
            "leaf_count": len(all_rows),
            "unique_path_count": len({row["normalized_path"] for row in all_rows}),
            "fallback_leaf_count": sum(1 for row in all_rows if row["rule_id"] == "fallback"),
            "leaf_count_by_classification": ordered_class_counts(leaf_counts),
            "unique_path_count_by_classification": ordered_class_counts(unique_counts),
            "unique_path_count_by_top_and_classification": {
                top: ordered_class_counts(counter)
                for top, counter in sorted(top_class_counts.items())
            },
        },
        "configs": [
            {
                key: value
                for key, value in config.items()
                if key != "rows"
            }
            for config in config_results
        ],
        "rows": all_rows,
    }


def markdown_cell(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def class_count_cells(counts: dict[str, int]) -> str:
    return " | ".join(str(counts.get(key, 0)) for key in CLASSIFICATIONS)


def append_path_section(lines: list[str], title: str, rows: list[dict[str, Any]], max_rows: int) -> None:
    lines.extend(["", f"## {title}", ""])
    if not rows:
        lines.append("_none_")
        return
    by_top: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_top[row["top"]].append(row)

    emitted = 0
    for top, top_rows in sorted(by_top.items()):
        if emitted >= max_rows:
            break
        lines.extend([f"### `{top}`", ""])
        for row in sorted(top_rows, key=lambda item: item["normalized_path"]):
            if emitted >= max_rows:
                break
            lines.append(
                f"- `{row['normalized_path']}` ({row['rule_id']}; {row['reason']})"
            )
            emitted += 1
        lines.append("")
    if len(rows) > emitted:
        lines.append(f"_{len(rows) - emitted} more omitted by --max-paths-per-section._")


def write_markdown(report: dict[str, Any], out_path: Path, max_paths_per_section: int) -> None:
    summary = report["summary"]
    lines = [
        "# Citlali Low-Level Config Key Classification",
        "",
        "This report classifies low-level Citlali YAML leaves by intended authoring exposure.",
        "It is documentation metadata only; it does not change runtime parsing.",
        "",
        "## Inputs",
        "",
        f"- Rules: `{report['rules_file']}`",
    ]
    for config in report["configs"]:
        intent = f" ({config['intent']})" if config.get("intent") else ""
        lines.append(f"- `{config['label']}`{intent}: `{config['path']}`")
    for missing in report["missing_inputs"]:
        lines.append(f"- Missing input: `{missing}`")

    lines.extend(["", "## Classification Policy", ""])
    for classification in CLASSIFICATIONS:
        description = report["classes"].get(classification, "")
        lines.append(f"- `{classification}`: {description}")

    lines.extend(
        [
            "",
            "## Overall Summary",
            "",
            f"- Configs classified: {summary['config_count']}",
            f"- Leaf occurrences classified: {summary['leaf_count']}",
            f"- Unique normalized paths classified: {summary['unique_path_count']}",
            f"- Fallback-classified leaf occurrences: {summary['fallback_leaf_count']}",
            "",
            "| Scope | User-facing | Expert | Hidden/internal | Deprecated |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Leaf occurrences | {class_count_cells(summary['leaf_count_by_classification'])} |",
            f"| Unique paths | {class_count_cells(summary['unique_path_count_by_classification'])} |",
            "",
            "## Baselines",
            "",
            "| Label | Intent | Leaves | Unique paths | User-facing | Expert | Hidden/internal | Deprecated | Fallback leaves |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for config in report["configs"]:
        counts = config["leaf_count_by_classification"]
        lines.append(
            f"| `{markdown_cell(config['label'])}` | `{markdown_cell(config.get('intent', ''))}` | "
            f"{config['leaf_count']} | {config['unique_path_count']} | {class_count_cells(counts)} | "
            f"{config['fallback_leaf_count']} |"
        )

    lines.extend(
        [
            "",
            "## Unique Paths By Top-Level Node",
            "",
            "| Node | User-facing | Expert | Hidden/internal | Deprecated | Total |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for top, counts in sorted(
        summary["unique_path_count_by_top_and_classification"].items(),
        key=lambda item: (-sum(item[1].values()), item[0]),
    ):
        total = sum(counts.values())
        lines.append(f"| `{markdown_cell(top)}` | {class_count_cells(counts)} | {total} |")

    unique = unique_rows(report["rows"])
    append_path_section(
        lines,
        "User-Facing Low-Level Paths",
        [row for row in unique if row["classification"] == "user-facing"],
        max_paths_per_section,
    )
    append_path_section(
        lines,
        "Deprecated Low-Level Paths Observed",
        [row for row in unique if row["classification"] == "deprecated"],
        max_paths_per_section,
    )
    append_path_section(
        lines,
        "Fallback-Classified Paths",
        [row for row in unique if row["rule_id"] == "fallback"],
        max_paths_per_section,
    )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- List indexes are normalized to `[]` for unique-path counts.",
            "- Classification is path-based; it does not evaluate whether a key is active for a specific reduction type.",
            "- `runtime.output_dir` is classified as user-facing at the TolTECA authoring layer, even though generated `citlali_*.yaml` files may contain a TolTECA absolute-path rewrite.",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(report: dict[str, Any], out_path: Path) -> None:
    fieldnames = [
        "config_label",
        "intent",
        "config_path",
        "path",
        "normalized_path",
        "top",
        "value_type",
        "value_preview",
        "classification",
        "rule_id",
        "rule_pattern",
        "owner",
        "reason",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules",
        default=str(Path(__file__).with_name("config_key_classification.yaml")),
        help="Classification rules YAML file.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Low-level or TolTECA YAML config to classify. May be repeated.",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Optional compact compatibility cases file; unique base_config paths are classified.",
    )
    parser.add_argument("--require-all", action="store_true", help="Fail if any config path is missing.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown output path.")
    parser.add_argument(
        "--max-paths-per-section",
        type=int,
        default=200,
        help="Maximum paths listed in each Markdown path section.",
    )
    parser.set_defaults(repo_root=repo_root)
    args = parser.parse_args(argv)
    if not args.config and not args.cases:
        parser.error("pass at least one --config or --cases")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    base_dir = Path.cwd()
    try:
        rules = load_rules(resolve_path(args.rules, base_dir))
        configs = [parse_config_arg(value, base_dir) for value in args.config]
        if args.cases:
            configs.extend(load_case_configs(resolve_path(args.cases, base_dir)))
        report = build_report(configs, rules, args.require_all)
    except (OSError, yaml.YAMLError, ClassificationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        json_path = resolve_path(args.json_out, base_dir)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_out:
        write_csv(report, resolve_path(args.csv_out, base_dir))
    if args.markdown_out:
        write_markdown(report, resolve_path(args.markdown_out, base_dir), args.max_paths_per_section)

    summary = report["summary"]
    counts = summary["leaf_count_by_classification"]
    print(
        f"classified_configs={summary['config_count']} "
        f"leaf_count={summary['leaf_count']} "
        f"unique_paths={summary['unique_path_count']}"
    )
    for classification in CLASSIFICATIONS:
        print(f"{classification}={counts[classification]}")
    if report["missing_inputs"]:
        print(f"missing_inputs={len(report['missing_inputs'])}")
    return 0 if not report["missing_inputs"] or not args.require_all else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
