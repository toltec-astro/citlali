#!/usr/bin/env python3
"""Compare TolTECA reduce YAML files with generated Citlali low-level YAML."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "citlali-tolteca-lowlevel-inventory-v1"
TOLTECA_CONFIG_RE = re.compile(r"^(\d+).*\.ya?ml$")


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tolteca_sort_key(path: Path) -> tuple[int, str]:
    match = TOLTECA_CONFIG_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return sys.maxsize, path.name


def collect_authoring_files(dirs: list[str], files: list[str]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()

    for dirname in dirs:
        directory = Path(dirname).expanduser().resolve()
        for path in sorted(directory.iterdir(), key=tolteca_sort_key):
            if not path.is_file() or not TOLTECA_CONFIG_RE.match(path.name):
                continue
            if path not in seen:
                result.append(path)
                seen.add(path)

    for filename in files:
        path = Path(filename).expanduser().resolve()
        if path not in seen:
            result.append(path)
            seen.add(path)

    return result


def value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def value_preview(value: Any, limit: int = 96) -> str:
    text = value_key(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def path_to_string(parts: tuple[str, ...]) -> str:
    result = ""
    for part in parts:
        if part.startswith("["):
            result += part
        elif result:
            result += "." + part
        else:
            result = part
    return result


def normalize_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("[]" if part.startswith("[") else part for part in parts)


def walk_leaves(value: Any, prefix: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key, child in value.items():
            rows.extend(walk_leaves(child, prefix + (str(key),)))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(walk_leaves(child, prefix + (f"[{index}]",)))
        return rows
    return [
        {
            "path": path_to_string(prefix),
            "normalized_path": path_to_string(normalize_parts(prefix)),
            "top": prefix[0] if prefix else "",
            "value_type": type(value).__name__,
            "value": value,
            "value_preview": value_preview(value),
        }
    ]


def step_values(steps: Any) -> list[Any]:
    if isinstance(steps, list):
        return steps
    if isinstance(steps, dict):
        return [steps[key] for key in sorted(steps, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    return []


def get_low_level(authoring_data: Any) -> Any:
    reduce_section = authoring_data.get("reduce", {}) if isinstance(authoring_data, dict) else {}
    for step in step_values(reduce_section.get("steps", [])):
        if not isinstance(step, dict):
            continue
        config = step.get("config", {})
        if isinstance(config, dict) and "low_level" in config:
            return config["low_level"] or {}
    return {}


def rows_by_normalized_path(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        result[row["normalized_path"]].append(row)
    return dict(result)


def classify_generated_row(
    row: dict[str, Any],
    authoring_low_level: list[dict[str, Any]],
    authoring_all: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_path = row["normalized_path"]
    generated_value_key = value_key(row["value"])

    for source in reversed(authoring_low_level):
        matches = source["by_path"].get(normalized_path, [])
        if not matches:
            continue
        values = {value_key(match["value"]) for match in matches}
        if generated_value_key in values:
            source_class = "low_level_same_value"
        else:
            source_class = "low_level_rewritten_value"
        return {
            "source_class": source_class,
            "source_file": source["path"],
            "source_value_preview": value_preview(matches[0]["value"]),
        }

    top = row["top"]
    if top == "inputs":
        return {
            "source_class": "tolteca_generated_inputs",
            "source_file": "",
            "source_value_preview": "",
        }
    if normalized_path in {"runtime.output_dir", "runtime.meta.version"}:
        return {
            "source_class": "tolteca_generated_runtime",
            "source_file": "",
            "source_value_preview": "",
        }

    for source in reversed(authoring_all):
        matches = source["by_path"].get(normalized_path, [])
        if matches:
            return {
                "source_class": "authoring_non_low_level_same_path",
                "source_file": source["path"],
                "source_value_preview": value_preview(matches[0]["value"]),
            }

    return {
        "source_class": "generated_not_found_in_authoring",
        "source_file": "",
        "source_value_preview": "",
    }


def build_inventory(authoring_files: list[Path], generated_file: Path) -> dict[str, Any]:
    authoring_low_level: list[dict[str, Any]] = []
    authoring_all: list[dict[str, Any]] = []

    for path in authoring_files:
        data = load_yaml(path)
        all_rows = walk_leaves(data)
        low_level_rows = walk_leaves(get_low_level(data))
        authoring_all.append(
            {
                "path": str(path),
                "leaf_count": len(all_rows),
                "by_path": rows_by_normalized_path(all_rows),
            }
        )
        authoring_low_level.append(
            {
                "path": str(path),
                "leaf_count": len(low_level_rows),
                "by_path": rows_by_normalized_path(low_level_rows),
            }
        )

    generated_rows = walk_leaves(load_yaml(generated_file))
    rows: list[dict[str, Any]] = []
    for row in generated_rows:
        source = classify_generated_row(row, authoring_low_level, authoring_all)
        rows.append(
            {
                "path": row["path"],
                "normalized_path": row["normalized_path"],
                "top": row["top"],
                "value_type": row["value_type"],
                "generated_value_preview": row["value_preview"],
                **source,
            }
        )

    source_counts = collections.Counter(row["source_class"] for row in rows)
    top_counts = collections.Counter(row["top"] for row in rows)
    top_source_counts: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        top_source_counts[row["top"]][row["source_class"]] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "authoring_files": [str(path) for path in authoring_files],
        "generated_file": str(generated_file),
        "summary": {
            "generated_leaf_count": len(generated_rows),
            "authoring_leaf_counts": {
                source["path"]: source["leaf_count"] for source in authoring_all
            },
            "authoring_low_level_leaf_counts": {
                source["path"]: source["leaf_count"] for source in authoring_low_level
            },
            "generated_leaf_count_by_source_class": dict(sorted(source_counts.items())),
            "generated_leaf_count_by_top": dict(sorted(top_counts.items())),
            "generated_leaf_count_by_top_and_source_class": {
                top: dict(sorted(counter.items())) for top, counter in sorted(top_source_counts.items())
            },
        },
        "generated_rows": rows,
        "notes": [
            "List indexes are normalized to [] for key matching.",
            "Later authoring files take precedence over earlier files.",
            "The comparison is path/value based; it does not run TolTECA or emulate all TolTECA merge rules.",
        ],
    }


def write_csv(inventory: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = inventory["generated_rows"]
    fieldnames = [
        "path",
        "normalized_path",
        "top",
        "source_class",
        "source_file",
        "value_type",
        "generated_value_preview",
        "source_value_preview",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(inventory: dict[str, Any], out_path: Path) -> None:
    summary = inventory["summary"]
    lines = [
        "# TolTECA To Citlali Low-Level Config Inventory",
        "",
        "## Inputs",
        "",
    ]
    for path in inventory["authoring_files"]:
        lines.append(f"- Authoring YAML: `{path}`")
    lines.append(f"- Generated Citlali YAML: `{inventory['generated_file']}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Generated Citlali leaf keys: {summary['generated_leaf_count']}",
            "",
            "Authoring leaf counts:",
            "",
        ]
    )
    for path, count in summary["authoring_leaf_counts"].items():
        lines.append(f"- `{path}`: {count}")
    lines.extend(["", "Authoring low-level leaf counts:", ""])
    for path, count in summary["authoring_low_level_leaf_counts"].items():
        lines.append(f"- `{path}`: {count}")

    lines.extend(
        [
            "",
            "## Generated Keys By Source Class",
            "",
            "| Source class | Leaf keys |",
            "| --- | ---: |",
        ]
    )
    for source_class, count in sorted(
        summary["generated_leaf_count_by_source_class"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"| `{source_class}` | {count} |")

    lines.extend(
        [
            "",
            "## Generated Keys By Top-Level Node",
            "",
            "| Node | Leaf keys | Source classes |",
            "| --- | ---: | --- |",
        ]
    )
    for top, count in sorted(
        summary["generated_leaf_count_by_top"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        classes = ", ".join(
            f"`{key}`: {value}"
            for key, value in summary["generated_leaf_count_by_top_and_source_class"][top].items()
        )
        lines.append(f"| `{top}` | {count} | {classes} |")

    rewritten = [
        row for row in inventory["generated_rows"]
        if row["source_class"] == "low_level_rewritten_value"
    ]
    lines.extend(
        [
            "",
            "## Low-Level Paths Rewritten By TolTECA",
            "",
            "| Path | Source value | Generated value |",
            "| --- | --- | --- |",
        ]
    )
    for row in rewritten:
        lines.append(
            f"| `{row['normalized_path']}` | `{row['source_value_preview']}` | "
            f"`{row['generated_value_preview']}` |"
        )
    if not rewritten:
        lines.append("| _none_ |  |  |")

    not_found = [
        row for row in inventory["generated_rows"]
        if row["source_class"] == "generated_not_found_in_authoring"
    ]
    lines.extend(
        [
            "",
            "## Generated Keys Not Found In Authoring YAML",
            "",
            "| Path | Value |",
            "| --- | --- |",
        ]
    )
    for row in not_found[:100]:
        lines.append(f"| `{row['normalized_path']}` | `{row['generated_value_preview']}` |")
    if not not_found:
        lines.append("| _none_ |  |")
    elif len(not_found) > 100:
        lines.append(f"| _{len(not_found) - 100} more omitted_ |  |")

    lines.extend(["", "## Notes", ""])
    for note in inventory["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authoring-file",
        action="append",
        default=[],
        help="TolTECA reduce YAML file. Pass in merge order, e.g. 70 then 72.",
    )
    parser.add_argument(
        "--authoring-dir",
        action="append",
        default=[],
        help="Directory containing TolTECA NN*.yaml files. Files are sorted by leading number.",
    )
    parser.add_argument("--generated-file", required=True, help="Generated Citlali low-level YAML file.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown output path.")
    args = parser.parse_args(argv)
    if not args.authoring_file and not args.authoring_dir:
        parser.error("pass at least one --authoring-dir or --authoring-file")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    authoring_files = collect_authoring_files(args.authoring_dir, args.authoring_file)
    inventory = build_inventory(
        authoring_files,
        Path(args.generated_file).expanduser().resolve(),
    )
    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_out:
        write_csv(inventory, Path(args.csv_out).expanduser())
    if args.markdown_out:
        write_markdown(inventory, Path(args.markdown_out).expanduser())

    summary = inventory["summary"]
    print(f"generated_leaf_count={summary['generated_leaf_count']}")
    for source_class, count in sorted(summary["generated_leaf_count_by_source_class"].items()):
        print(f"{source_class}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
