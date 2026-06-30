#!/usr/bin/env python3
"""Inventory Citlali config keys and simple source references."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "citlali-config-inventory-v1"
CPP_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
TUPLE_RE = re.compile(r"std::tuple\s*\{([^}]*)\}")
STRING_RE = re.compile(r'"([^"]+)"')
CONFIG_ACCESS_HINTS = (
    "get_config_value",
    ".get_typed",
    ".get_str",
    ".get_node",
    ".has(",
    ".has_typed",
)


def walk_yaml(value: Any, prefix: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key, child in value.items():
            rows.extend(walk_yaml(child, prefix + (str(key),)))
        return rows
    return [
        {
            "path": ".".join(prefix),
            "parts": list(prefix),
            "top": prefix[0] if prefix else "",
            "value_type": type(value).__name__,
            "default": value,
        }
    ]


def iter_source_files(source_root: Path) -> list[Path]:
    roots = [source_root / "include", source_root / "src", source_root / "tests"]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in CPP_SUFFIXES:
                files.append(path)
    return sorted(files)


def extract_references(source_root: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for path in iter_source_files(source_root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        relpath = path.relative_to(source_root).as_posix()
        for lineno, line in enumerate(lines, start=1):
            if not any(hint in line for hint in CONFIG_ACCESS_HINTS):
                continue
            for match in TUPLE_RE.finditer(line):
                parts = STRING_RE.findall(match.group(1))
                if parts:
                    refs.append(
                        {
                            "path": ".".join(parts),
                            "parts": parts,
                            "top": parts[0],
                            "file": relpath,
                            "line": lineno,
                            "source": line.strip(),
                        }
                    )
            if "std::tuple" not in line:
                string_parts = STRING_RE.findall(line)
                if len(string_parts) == 1:
                    refs.append(
                        {
                            "path": string_parts[0],
                            "parts": [string_parts[0]],
                            "top": string_parts[0].split(".", 1)[0],
                            "file": relpath,
                            "line": lineno,
                            "source": line.strip(),
                        }
                    )
    return refs


def build_inventory(config_path: Path, source_root: Path) -> dict[str, Any]:
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    yaml_keys = walk_yaml(config_data)
    refs = extract_references(source_root)

    yaml_by_top = collections.Counter(row["top"] for row in yaml_keys)
    refs_by_top = collections.Counter(row["top"] for row in refs)
    yaml_paths = {row["path"] for row in yaml_keys}
    ref_paths = {row["path"] for row in refs}

    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "source_root": str(source_root),
        "summary": {
            "yaml_leaf_count": len(yaml_keys),
            "source_reference_count": len(refs),
            "yaml_leaf_count_by_top": dict(sorted(yaml_by_top.items())),
            "source_reference_count_by_top": dict(sorted(refs_by_top.items())),
            "referenced_yaml_leaf_count": len(yaml_paths & ref_paths),
            "unreferenced_yaml_leaf_count_simple_scan": len(yaml_paths - ref_paths),
            "reference_paths_not_yaml_leaf_count_simple_scan": len(ref_paths - yaml_paths),
        },
        "yaml_keys": yaml_keys,
        "source_references": refs,
        "notes": [
            "Source references are a simple regex scan. Dynamic accesses and multi-line tuple expressions may be missed.",
            "A key can be valid even if it is reported as unreferenced by this simple scan.",
            "A reference can be valid even if it points at an internal node rather than a YAML leaf.",
        ],
    }


def write_markdown(inventory: dict[str, Any], out_path: Path) -> None:
    summary = inventory["summary"]
    lines: list[str] = [
        "# Citlali Config Inventory",
        "",
        f"Config: `{inventory['config_path']}`",
        f"Source root: `{inventory['source_root']}`",
        "",
        "## Summary",
        "",
        f"- YAML leaf keys: {summary['yaml_leaf_count']}",
        f"- Simple source references: {summary['source_reference_count']}",
        f"- Referenced YAML leaves by exact path: {summary['referenced_yaml_leaf_count']}",
        f"- YAML leaves not found by exact simple scan: {summary['unreferenced_yaml_leaf_count_simple_scan']}",
        f"- Source reference paths that are not YAML leaves: {summary['reference_paths_not_yaml_leaf_count_simple_scan']}",
        "",
        "## YAML Leaf Counts By Top-Level Node",
        "",
        "| Node | Leaf keys |",
        "| --- | ---: |",
    ]
    for key, count in sorted(summary["yaml_leaf_count_by_top"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")

    lines.extend(
        [
            "",
            "## Simple Source Reference Counts By Top-Level Node",
            "",
            "| Node | References |",
            "| --- | ---: |",
        ]
    )
    for key, count in sorted(summary["source_reference_count_by_top"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")

    lines.extend(
        [
            "",
            "## YAML Leaf Keys",
            "",
            "| Path | Type | Default |",
            "| --- | --- | --- |",
        ]
    )
    for row in inventory["yaml_keys"]:
        default = json.dumps(row["default"], sort_keys=True)
        lines.append(f"| `{row['path']}` | `{row['value_type']}` | `{default}` |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in inventory["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data/config.yaml", help="Default config YAML path.")
    parser.add_argument("--source-root", default=".", help="Repository root to scan.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown output path.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    inventory = build_inventory(config_path, source_root)

    if args.json_out:
        json_out = Path(args.json_out).expanduser()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(inventory, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    if args.markdown_out:
        write_markdown(inventory, Path(args.markdown_out).expanduser())

    summary = inventory["summary"]
    print(f"YAML leaf keys: {summary['yaml_leaf_count']}")
    print(f"simple source references: {summary['source_reference_count']}")
    print("top-level YAML leaf counts:")
    for key, count in sorted(summary["yaml_leaf_count_by_top"].items(), key=lambda item: (-item[1], item[0])):
        print(f"  {key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
