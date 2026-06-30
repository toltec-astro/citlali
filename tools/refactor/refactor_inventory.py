#!/usr/bin/env python3
"""Static inventory for the Citlali structural refactor."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "citlali-refactor-inventory-v1"
SOURCE_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
HEADER_SUFFIXES = {".h", ".hpp", ".hh"}
TUPLE_RE = re.compile(r"std::tuple\s*\{([^}]*)\}")
STRING_RE = re.compile(r'"([^"]+)"')
MEMBER_DEF_RE = re.compile(
    r"^\s*(?:[\w:<>,~*&\s]+|auto)\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)::([~A-Za-z_]\w*)\s*\("
)
FREE_FUNC_RE = re.compile(
    r"^\s*(?:void|bool|int|double|float|auto|std::[\w:<>,]+|Eigen::[\w:<>,]+)\s+([A-Za-z_]\w*)\s*\("
)


def iter_source_files(repo: Path) -> list[Path]:
    roots = [repo / "include", repo / "src", repo / "tests"]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in SOURCE_SUFFIXES)
    return sorted(files)


def rel(path: Path, repo: Path) -> str:
    return path.relative_to(repo).as_posix()


def subsystem_for(path: str) -> str:
    if path.startswith("src/citlali/cli/"):
        return "cli"
    if "/timestream/rtc/" in path:
        return "rtc"
    if "/timestream/ptc/" in path:
        return "ptc"
    if "/timestream/" in path:
        return "timestream"
    if "/mapmaking/" in path:
        return "mapmaking"
    if "/engine/" in path:
        return "engine"
    if "/utils/" in path:
        return "utils"
    if path.startswith("tests/"):
        return "tests"
    return "other"


def exit_risk(path: str, line: str) -> str:
    if path.startswith("src/citlali/cli/") or path.endswith("_main.cpp") or path.endswith("main_old.cpp"):
        return "cli_boundary"
    if path.startswith("include/"):
        return "library_header_high"
    if "config" in line.lower() or "input" in line.lower() or "file" in line.lower():
        return "library_preflight_medium"
    return "library_runtime_review"


def scan_exits(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_source_files(repo):
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        relpath = rel(path, repo)
        for lineno, line in enumerate(text, start=1):
            if "std::exit" not in line and "exit(" not in line:
                continue
            rows.append(
                {
                    "file": relpath,
                    "line": lineno,
                    "subsystem": subsystem_for(relpath),
                    "risk": exit_risk(relpath, line),
                    "source": line.strip(),
                }
            )
    return rows


def scan_headers(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((repo / "include").rglob("*")):
        if not path.is_file() or path.suffix not in HEADER_SUFFIXES:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        relpath = rel(path, repo)
        member_defs: list[dict[str, Any]] = []
        free_defs: list[dict[str, Any]] = []
        template_member_defs = 0
        for lineno, line in enumerate(lines, start=1):
            member_match = MEMBER_DEF_RE.match(line)
            if member_match:
                previous = "\n".join(lines[max(0, lineno - 4):lineno - 1])
                is_template = "template" in previous
                if is_template:
                    template_member_defs += 1
                else:
                    member_defs.append(
                        {
                            "line": lineno,
                            "qualifier": member_match.group(1),
                            "name": member_match.group(2),
                            "source": line.strip(),
                        }
                    )
                continue
            free_match = FREE_FUNC_RE.match(line)
            if free_match and not line.strip().endswith(";"):
                previous = "\n".join(lines[max(0, lineno - 4):lineno - 1])
                if "template" not in previous:
                    free_defs.append(
                        {
                            "line": lineno,
                            "name": free_match.group(1),
                            "source": line.strip(),
                        }
                    )
        rows.append(
            {
                "file": relpath,
                "subsystem": subsystem_for(relpath),
                "line_count": len(lines),
                "include_count": sum(1 for line in lines if line.strip().startswith("#include")),
                "non_template_member_def_count": len(member_defs),
                "non_template_free_def_count": len(free_defs),
                "template_member_def_count_simple_scan": template_member_defs,
                "non_template_member_defs": member_defs[:50],
                "non_template_free_defs": free_defs[:50],
            }
        )
    return rows


def scan_cmake(repo: Path) -> list[dict[str, Any]]:
    cmake = repo / "CMakeLists.txt"
    rows: list[dict[str, Any]] = []
    if not cmake.exists():
        return rows
    for lineno, line in enumerate(cmake.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") and ".cpp" in stripped and "src/" in stripped:
            rows.append({"file": "CMakeLists.txt", "line": lineno, "source": stripped})
    return rows


def scan_config_refs(repo: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    hints = ("get_config_value", ".get_typed", ".get_str", ".get_node", ".has(", ".has_typed")
    for path in iter_source_files(repo):
        relpath = rel(path, repo)
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if not any(hint in line for hint in hints):
                continue
            paths: list[str] = []
            for match in TUPLE_RE.finditer(line):
                parts = STRING_RE.findall(match.group(1))
                if parts:
                    paths.append(".".join(parts))
            if not paths:
                strings = STRING_RE.findall(line)
                if len(strings) == 1:
                    paths.append(strings[0])
            for config_path in paths:
                refs.append(
                    {
                        "config_path": config_path,
                        "top": config_path.split(".", 1)[0],
                        "file": relpath,
                        "line": lineno,
                        "subsystem": subsystem_for(relpath),
                    }
                )
    return refs


def build_inventory(repo: Path) -> dict[str, Any]:
    exits = scan_exits(repo)
    headers = scan_headers(repo)
    cmake = scan_cmake(repo)
    config_refs = scan_config_refs(repo)
    exit_by_subsystem = collections.Counter(row["subsystem"] for row in exits)
    exit_by_risk = collections.Counter(row["risk"] for row in exits)
    config_by_subsystem = collections.Counter(row["subsystem"] for row in config_refs)
    config_by_top = collections.Counter(row["top"] for row in config_refs)
    largest_headers = sorted(headers, key=lambda row: row["line_count"], reverse=True)[:20]
    movable_headers = sorted(
        (row for row in headers if row["non_template_member_def_count"] or row["non_template_free_def_count"]),
        key=lambda row: (row["non_template_member_def_count"] + row["non_template_free_def_count"], row["line_count"]),
        reverse=True,
    )[:20]
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": str(repo),
        "summary": {
            "exit_count": len(exits),
            "exit_count_by_subsystem": dict(sorted(exit_by_subsystem.items())),
            "exit_count_by_risk": dict(sorted(exit_by_risk.items())),
            "header_count": len(headers),
            "commented_cmake_source_count": len(cmake),
            "simple_config_reference_count": len(config_refs),
            "simple_config_reference_count_by_subsystem": dict(sorted(config_by_subsystem.items())),
            "simple_config_reference_count_by_top": dict(sorted(config_by_top.items())),
        },
        "exits": exits,
        "headers": headers,
        "largest_headers": largest_headers,
        "headers_with_non_template_defs": movable_headers,
        "commented_cmake_sources": cmake,
        "config_references": config_refs,
        "notes": [
            "Header-definition detection is regex-based and must be manually reviewed before moving code.",
            "Template and inline functions may be intentionally header-only.",
            "Config reference detection misses dynamic and multi-line accesses.",
            "Exit risk is a triage label, not a final migration decision.",
        ],
    }


def write_markdown(inventory: dict[str, Any], out_path: Path) -> None:
    summary = inventory["summary"]
    lines: list[str] = [
        "# Citlali Structural Refactor Inventory",
        "",
        f"Repo: `{inventory['repo']}`",
        "",
        "## Summary",
        "",
        f"- Direct exit calls: {summary['exit_count']}",
        f"- Headers scanned: {summary['header_count']}",
        f"- Commented CMake source entries: {summary['commented_cmake_source_count']}",
        f"- Simple config references: {summary['simple_config_reference_count']}",
        "",
        "## Exit Calls By Subsystem",
        "",
        "| Subsystem | Count |",
        "| --- | ---: |",
    ]
    for key, count in sorted(summary["exit_count_by_subsystem"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## Exit Calls By Risk Label", "", "| Risk | Count |", "| --- | ---: |"])
    for key, count in sorted(summary["exit_count_by_risk"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")

    lines.extend(["", "## Largest Headers", "", "| Header | Lines | Includes |", "| --- | ---: | ---: |"])
    for row in inventory["largest_headers"]:
        lines.append(f"| `{row['file']}` | {row['line_count']} | {row['include_count']} |")

    lines.extend(
        [
            "",
            "## Headers With Simple Non-Template Definitions",
            "",
            "| Header | Non-template member defs | Non-template free defs | Lines |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in inventory["headers_with_non_template_defs"]:
        lines.append(
            f"| `{row['file']}` | {row['non_template_member_def_count']} | "
            f"{row['non_template_free_def_count']} | {row['line_count']} |"
        )

    lines.extend(["", "## Commented CMake Source Entries", ""])
    if inventory["commented_cmake_sources"]:
        for row in inventory["commented_cmake_sources"]:
            lines.append(f"- `{row['file']}:{row['line']}` {row['source']}")
    else:
        lines.append("- None found.")

    lines.extend(["", "## Config References By Subsystem", "", "| Subsystem | References |", "| --- | ---: |"])
    for key, count in sorted(summary["simple_config_reference_count_by_subsystem"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")

    lines.extend(["", "## Config References By Top-Level Node", "", "| Node | References |", "| --- | ---: |"])
    for key, count in sorted(summary["simple_config_reference_count_by_top"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")

    lines.extend(["", "## First Exit Migration Candidates", ""])
    for row in inventory["exits"][:40]:
        lines.append(
            f"- `{row['file']}:{row['line']}` subsystem=`{row['subsystem']}` "
            f"risk=`{row['risk']}` source=`{row['source']}`"
        )

    lines.extend(["", "## Notes", ""])
    for note in inventory["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Repository root.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown output path.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    inventory = build_inventory(repo)
    if args.json_out:
        json_out = Path(args.json_out).expanduser()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        write_markdown(inventory, Path(args.markdown_out).expanduser())
    summary = inventory["summary"]
    print(f"direct exit calls: {summary['exit_count']}")
    print(f"headers scanned: {summary['header_count']}")
    print(f"commented CMake source entries: {summary['commented_cmake_source_count']}")
    print(f"simple config references: {summary['simple_config_reference_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
