#!/usr/bin/env python3
"""Audit the transitional YAML boundary in PTCProc::get_config."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_PATH_COUNT = 171
EXPECTED_PATH_SHA256 = (
    "b2bc50c1cf73064a1279c0335bf03a3033938ec1589095f390fdcac9c0fc67b6"
)

TYPED_READER_SOURCES = {
    "clean": "include/citlali/core/pipeline/processed_clean_config_read.h",
    "fruit_loops": "include/citlali/core/pipeline/fruit_loops_config_read.h",
    "flagging": "include/citlali/core/pipeline/processed_weighting_config_read.h",
    "second_pass_local": "include/citlali/core/pipeline/second_pass_local_config_read.h",
    "weighting": "include/citlali/core/pipeline/processed_weighting_config_read.h",
}

# Map a frozen legacy path to a typed path only when the typed reader
# intentionally accepts it under a different spelling.
COMPATIBILITY_ALIASES: dict[str, str] = {}


def parser_body(path: Path) -> str:
    text = path.read_text()
    start = text.index("void PTCProc::get_config")
    end = text.find("\ntemplate", start + 1)
    if end < 0:
        raise ValueError(f"unable to find end of PTCProc::get_config in {path}")
    return text[start:end]


def literal_paths(body: str) -> list[str]:
    tuple_pattern = re.compile(r"std::tuple\s*\{([^}]*)\}", re.DOTALL)
    string_pattern = re.compile(r'"([^"]+)"')
    paths = {
        ".".join(string_pattern.findall(match.group(1)))
        for match in tuple_pattern.finditer(body)
        if string_pattern.findall(match.group(1))
    }
    return sorted(paths)


def family(path: str) -> str:
    parts = path.split(".")
    if parts[:2] == ["timestream", "fruit_loops"]:
        return "fruit_loops"
    if parts[:3] == ["timestream", "processed_time_chunk", "clean"]:
        return "clean"
    if parts[:3] == ["timestream", "processed_time_chunk", "weighting"]:
        return "weighting"
    if parts[:3] == ["timestream", "processed_time_chunk", "flagging"]:
        return "flagging"
    return "unclassified"


def typed_reader_name(path: str) -> str | None:
    path_family = family(path)
    if path_family == "flagging" and ".second_pass_local." in path:
        return "second_pass_local"
    if path_family in TYPED_READER_SOURCES:
        return path_family
    return None


def typed_reader_coverage(
    paths: list[str], repo_root: Path
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    source_text = {
        name: (repo_root / source).read_text()
        for name, source in TYPED_READER_SOURCES.items()
    }
    records: list[dict[str, str]] = []
    uncovered: list[str] = []
    used_aliases: set[str] = set()
    for legacy_path in paths:
        reader_name = typed_reader_name(legacy_path)
        typed_path = COMPATIBILITY_ALIASES.get(legacy_path, legacy_path)
        if legacy_path in COMPATIBILITY_ALIASES:
            used_aliases.add(legacy_path)
        if reader_name is None:
            uncovered.append(legacy_path)
            continue
        leaf = typed_path.rsplit(".", 1)[-1]
        if f'"{leaf}"' not in source_text[reader_name]:
            uncovered.append(legacy_path)
            continue
        records.append(
            {
                "legacy_path": legacy_path,
                "typed_path": typed_path,
                "reader": TYPED_READER_SOURCES[reader_name],
                "coverage": (
                    "compatibility_alias"
                    if legacy_path in COMPATIBILITY_ALIASES
                    else "direct_typed_reader"
                ),
            }
        )
    stale_aliases = sorted(set(COMPATIBILITY_ALIASES) - used_aliases)
    return records, sorted(uncovered), stale_aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--fail-on-uncovered", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    source = repo_root / "include/citlali/core/timestream/ptc/ptcproc.h"
    body = parser_body(source)
    paths = literal_paths(body)
    digest = hashlib.sha256("\n".join(paths).encode()).hexdigest()
    direct_exit = "std::exit" in body or re.search(r"(?<![\w:])exit\s*\(", body)
    counts = dict(sorted(Counter(map(family, paths)).items()))
    drift = len(paths) != EXPECTED_PATH_COUNT or digest != EXPECTED_PATH_SHA256
    coverage, uncovered, stale_aliases = typed_reader_coverage(paths, repo_root)
    reader_counts = dict(
        sorted(Counter(record["reader"] for record in coverage).items())
    )
    coverage_complete = not uncovered and not stale_aliases
    result = {
        "schema_version": "citlali-processed-config-boundary-audit-v2",
        "source": str(source.relative_to(repo_root)),
        "literal_path_count": len(paths),
        "literal_path_sha256": digest,
        "expected_path_count": EXPECTED_PATH_COUNT,
        "expected_path_sha256": EXPECTED_PATH_SHA256,
        "family_counts": counts,
        "path_drift": drift,
        "direct_process_exit": bool(direct_exit),
        "typed_reader_coverage_complete": coverage_complete,
        "typed_reader_covered_path_count": len(coverage),
        "typed_reader_counts": reader_counts,
        "uncovered_paths": uncovered,
        "stale_compatibility_aliases": stale_aliases,
        "typed_reader_coverage": coverage,
        "paths": paths,
        "note": (
            "Literal tuple paths freeze the legacy boundary; dynamic tuple "
            "components are represented by their literal prefix. Typed "
            "coverage routes each frozen path to its declared reader and "
            "requires the leaf key in that source; spelling differences "
            "require an explicit compatibility alias."
        ),
    }
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        output = Path(args.markdown_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(
            f"| {name} | {count} |" for name, count in counts.items()
        )
        reader_rows = "\n".join(
            f"| `{name}` | {count} |"
            for name, count in reader_counts.items()
        )
        uncovered_rows = "\n".join(f"- `{path}`" for path in uncovered)
        if not uncovered_rows:
            uncovered_rows = "- None"
        output.write_text(
            "# Processed Timestream Boundary Audit\n\n"
            f"- Literal paths: `{len(paths)}`\n"
            f"- Path digest: `{digest}`\n"
            f"- Path drift: `{drift}`\n"
            f"- Direct process exit: `{bool(direct_exit)}`\n"
            f"- Typed reader coverage: `{len(coverage)}/{len(paths)}`\n"
            f"- Typed reader coverage complete: `{coverage_complete}`\n\n"
            "| Family | Paths |\n| --- | ---: |\n"
            f"{rows}\n\n"
            "| Typed reader | Paths |\n| --- | ---: |\n"
            f"{reader_rows}\n\n"
            "## Uncovered Paths\n\n"
            f"{uncovered_rows}\n"
        )
    print(
        "processed config boundary: "
        f"paths={len(paths)} drift={drift} direct_exit={bool(direct_exit)} "
        f"typed_coverage={len(coverage)}/{len(paths)} "
        f"coverage_complete={coverage_complete} families={counts}"
    )
    if args.fail_on_drift and (drift or direct_exit):
        return 1
    if args.fail_on_uncovered and not coverage_complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
