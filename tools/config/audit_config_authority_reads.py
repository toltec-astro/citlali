#!/usr/bin/env python3
"""Classify source sites that directly access untyped configuration objects."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "citlali-config-read-census-v1"
SOURCE_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
RAW_ACCESS_RE = re.compile(
    r"\bget_config_value\s*\(|\.get_typed(?:<[^>]+>)?\s*\(|"
    r"\.get_str\s*\(|\.get_node\s*\(|\.has_typed(?:<[^>]+>)?\s*\(|"
    r"\.has\s*\("
)


def category_for(path: str) -> str:
    """Return the architectural role of a file containing direct config reads."""
    if path.startswith("include/citlali/core/cli/"):
        return "cli-boundary"
    if path in {
        "include/citlali/core/engine/io.h",
        "include/citlali/core/engine/kidsproc.h",
        "include/citlali/core/engine/detail/kidsproc_metadata_reduce_impl.h",
        "include/citlali/core/engine/detail/rawobs_collection_impl.h",
        "include/citlali/core/engine/detail/seq_io_coordinator_impl.h",
    }:
        return "external-schema-boundary"
    if path in {
        "include/citlali/core/engine/config.h",
        "include/citlali/core/engine/todproc.h",
        "include/citlali/core/mapmaking/gaussian_filter.h",
        "include/citlali/core/mapmaking/wiener_filter.h",
        "include/citlali/core/mapmaking/wiener_filter_omp.h",
        "src/citlali/core/mapmaking/map.cpp",
    } or path.startswith("include/citlali/core/timestream/rtc/") or path.startswith(
        "include/citlali/core/timestream/ptc/"
    ):
        return "legacy-parser-boundary"
    if path.startswith("include/citlali/core/pipeline/"):
        return "typed-loader-boundary"
    if path.startswith("include/citlali/core/engine/detail/") and "config" in Path(path).name:
        return "typed-loader-boundary"
    if path in {
        "src/citlali/main_old.cpp",
        "src/citlali/mpi_main.cpp",
        "src/citlali/lali_main.cpp",
        "src/citlali/kids_main.cpp",
    }:
        return "legacy-entrypoint"
    return "review-required"


def source_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in ("include", "src"):
        root = repo_root / root_name
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in SOURCE_SUFFIXES
        )
    return sorted(files)


def build_census(repo_root: Path) -> dict[str, Any]:
    sites: list[dict[str, Any]] = []
    for path in source_files(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            matches = list(RAW_ACCESS_RE.finditer(line))
            if not matches:
                continue
            sites.append(
                {
                    "file": relative,
                    "line": line_number,
                    "category": category_for(relative),
                    "access_count": len(matches),
                    "source": line.strip(),
                }
            )

    by_category = collections.Counter(site["category"] for site in sites)
    accesses_by_category = collections.Counter()
    files_by_category: dict[str, set[str]] = collections.defaultdict(set)
    for site in sites:
        category = site["category"]
        accesses_by_category[category] += site["access_count"]
        files_by_category[category].add(site["file"])

    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "site_count": len(sites),
            "access_count": sum(site["access_count"] for site in sites),
            "file_count": len({site["file"] for site in sites}),
            "site_count_by_category": dict(sorted(by_category.items())),
            "access_count_by_category": dict(sorted(accesses_by_category.items())),
            "file_count_by_category": {
                key: len(value) for key, value in sorted(files_by_category.items())
            },
        },
        "sites": sites,
        "notes": [
            "This is a lexical census, not proof that a read occurs during execution.",
            "Legacy parser boundaries are migration inputs; they are not automatically execution-time raw-YAML fallbacks.",
            "Every review-required site must be classified before this census can serve as a migration gate.",
        ],
    }


def write_markdown(census: dict[str, Any], output: Path) -> None:
    summary = census["summary"]
    lines = [
        "# Config Read Census",
        "",
        f"Schema: `{census['schema_version']}`",
        "",
        f"- Direct access sites: {summary['site_count']}",
        f"- Direct access expressions: {summary['access_count']}",
        f"- Files containing direct access: {summary['file_count']}",
        "",
        "| Category | Files | Sites | Accesses |",
        "| --- | ---: | ---: | ---: |",
    ]
    categories = sorted(summary["site_count_by_category"])
    for category in categories:
        lines.append(
            f"| `{category}` | {summary['file_count_by_category'][category]} | "
            f"{summary['site_count_by_category'][category]} | "
            f"{summary['access_count_by_category'][category]} |"
        )
    review_sites = [
        site for site in census["sites"] if site["category"] == "review-required"
    ]
    lines.extend(["", "## Review Required", ""])
    if review_sites:
        for site in review_sites:
            lines.append(f"- `{site['file']}:{site['line']}`: `{site['source']}`")
    else:
        lines.append("No unclassified direct config reads.")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in census["notes"])
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--fail-on-review", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    census = build_census(repo_root)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        write_markdown(census, Path(args.markdown_out))

    summary = census["summary"]
    review_count = summary["site_count_by_category"].get("review-required", 0)
    print(
        "config read census: "
        f"files={summary['file_count']} sites={summary['site_count']} "
        f"accesses={summary['access_count']} review_required={review_count}"
    )
    if args.fail_on_review and review_count:
        for site in census["sites"]:
            if site["category"] == "review-required":
                print(f"review: {site['file']}:{site['line']}: {site['source']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
